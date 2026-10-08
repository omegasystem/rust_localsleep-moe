//! router.rs - High-Performance Fatigue-Aware MoE Router in Pure Rust
//! Implements entropy-based routing, dynamic sleep refractory masking, in-warp softmax,
//! and Shannon entropy calculation without Python runtime bottlenecks.

use crate::types::{ExpertState, LocalSleepConfig};
use parking_lot::RwLock;
use std::sync::Arc;

/// Dynamic Fatigue & Homeostatic Routing Module
#[derive(Debug, Clone)]
pub struct FatigueMoERouter {
    pub num_experts: usize,
    pub hidden_dim: usize,
    pub top_k: usize,
    pub fatigue_threshold: f32,
    pub wake_threshold: f32,
    pub decay_rate: f32,
    pub sleep_duration: usize,
    pub temperature: f32,

    // Gating Linear Weights: [num_experts, hidden_dim] flattened
    pub gate_weights: Vec<f32>,

    // Dynamic Homeostatic State Buffers (wrapped in RwLock for zero-GIL concurrent updates)
    pub fatigue_scores: Arc<RwLock<Vec<f32>>>,
    pub states: Arc<RwLock<Vec<ExpertState>>>,
    pub sleep_countdown: Arc<RwLock<Vec<usize>>>,
    pub active_token_counts: Arc<RwLock<Vec<usize>>>,
}

impl FatigueMoERouter {
    pub fn new(cfg: &LocalSleepConfig) -> Self {
        // Initialize orthogonal or small random gating projection
        let mut gate_weights = vec![0.0f32; cfg.num_experts * cfg.hidden_dim];
        let scale = (1.0 / (cfg.hidden_dim as f32)).sqrt();
        for i in 0..gate_weights.len() {
            // Deterministic pseudo-random initialization
            let val = ((i * 1103515245 + 12345) % 65536) as f32 / 65536.0;
            gate_weights[i] = (val - 0.5) * scale;
        }

        Self {
            num_experts: cfg.num_experts,
            hidden_dim: cfg.hidden_dim,
            top_k: cfg.top_k,
            fatigue_threshold: cfg.fatigue_threshold,
            wake_threshold: cfg.wake_threshold,
            decay_rate: cfg.decay_rate,
            sleep_duration: cfg.sleep_duration,
            temperature: cfg.router_temperature,
            gate_weights,
            fatigue_scores: Arc::new(RwLock::new(vec![0.0; cfg.num_experts])),
            states: Arc::new(RwLock::new(vec![ExpertState::Active; cfg.num_experts])),
            sleep_countdown: Arc::new(RwLock::new(vec![0; cfg.num_experts])),
            active_token_counts: Arc::new(RwLock::new(vec![0; cfg.num_experts])),
        }
    }

    /// Forward pass: Computes gating logits, masks sleeping experts (-1e9), applies softmax,
    /// selects Top-K experts, and updates cumulative token entropy and fatigue scores.
    pub fn forward(
        &self,
        hidden_states: &[f32], // [batch_size, hidden_dim]
        batch_size: usize,
    ) -> (Vec<f32>, Vec<usize>, f32) {
        assert_eq!(hidden_states.len(), batch_size * self.hidden_dim);

        let mut all_routing_weights = Vec::with_capacity(batch_size * self.top_k);
        let mut all_selected_experts = Vec::with_capacity(batch_size * self.top_k);
        let mut batch_entropy_sum = 0.0f32;

        let states_guard = self.states.read();
        let mut fatigue_guard = self.fatigue_scores.write();
        let mut active_tokens_guard = self.active_token_counts.write();

        for b in 0..batch_size {
            let hidden_slice = &hidden_states[b * self.hidden_dim..(b + 1) * self.hidden_dim];
            let mut logits = vec![0.0f32; self.num_experts];

            // 1. Linear Projection: logits = hidden @ gate_weights.T
            for k in 0..self.num_experts {
                let weight_row = &self.gate_weights[k * self.hidden_dim..(k + 1) * self.hidden_dim];
                let mut dot = 0.0f32;
                for d in 0..self.hidden_dim {
                    dot += hidden_slice[d] * weight_row[d];
                }
                logits[k] = dot;
            }

            // 2. Refractory Sleep Masking & Temperature Scaling
            for k in 0..self.num_experts {
                if states_guard[k] == ExpertState::NremSleep {
                    logits[k] = -1e9; // Hard mask sleeping modules
                } else if self.temperature > 0.0 {
                    logits[k] /= self.temperature;
                }
            }

            // 3. Numerically Stable Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            let mut probs = vec![0.0f32; self.num_experts];
            for k in 0..self.num_experts {
                probs[k] = (logits[k] - max_logit).exp();
                exp_sum += probs[k];
            }
            if exp_sum > 0.0 {
                for k in 0..self.num_experts {
                    probs[k] /= exp_sum;
                }
            }

            // 4. Shannon Entropy Calculation: H = - sum(p * log(p + eps))
            let mut entropy = 0.0f32;
            for k in 0..self.num_experts {
                if probs[k] > 1e-12 {
                    entropy -= probs[k] * probs[k].ln();
                }
            }
            batch_entropy_sum += entropy;

            // 5. Top-K Expert Selection
            let mut expert_prob_pairs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
            expert_prob_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let mut topk_weights = Vec::with_capacity(self.top_k);
            let mut topk_indices = Vec::with_capacity(self.top_k);
            let mut topk_weight_sum = 0.0f32;

            for (idx, p) in expert_prob_pairs.into_iter().take(self.top_k) {
                topk_indices.push(idx);
                topk_weights.push(p);
                topk_weight_sum += p;

                // Update active metrics
                active_tokens_guard[idx] += 1;
                fatigue_guard[idx] += (entropy * 0.1).max(0.01);
            }

            // Re-normalize Top-K routing weights
            if topk_weight_sum > 0.0 {
                for w in &mut topk_weights {
                    *w /= topk_weight_sum;
                }
            }

            all_routing_weights.extend(topk_weights);
            all_selected_experts.extend(topk_indices);
        }

        let mean_entropy = batch_entropy_sum / (batch_size as f32);
        (all_routing_weights, all_selected_experts, mean_entropy)
    }

    /// Checks if any active expert has reached the fatigue threshold
    pub fn check_fatigue_triggers(&self) -> Vec<usize> {
        let mut triggered = Vec::new();
        let fatigue = self.fatigue_scores.read();
        let states = self.states.read();

        for k in 0..self.num_experts {
            if states[k] == ExpertState::Active && fatigue[k] >= self.fatigue_threshold {
                triggered.push(k);
            }
        }
        triggered
    }

    /// Exponential homeostatic fatigue decay step
    pub fn step_decay(&self) {
        let mut fatigue = self.fatigue_scores.write();
        let mut countdown = self.sleep_countdown.write();
        let mut states = self.states.write();

        for k in 0..self.num_experts {
            // Apply passive decay
            fatigue[k] = (fatigue[k] * (1.0 - self.decay_rate)).max(0.0);

            // Decrement sleep countdown if sleeping
            if states[k] == ExpertState::NremSleep {
                if countdown[k] > 0 {
                    countdown[k] -= 1;
                }
                if countdown[k] == 0 && fatigue[k] <= self.wake_threshold {
                    states[k] = ExpertState::Active;
                    fatigue[k] = 0.0;
                }
            }
        }
    }
}
