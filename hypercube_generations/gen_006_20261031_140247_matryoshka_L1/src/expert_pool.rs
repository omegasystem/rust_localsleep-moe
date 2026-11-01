//! expert_pool.rs - Virtual LoRA Expert Pool in Pure Rust
//! Implements low-rank projection delta modulation: delta_h = (x @ A.T) @ B.T * (alpha / rank)

use crate::types::LocalSleepConfig;
use serde::{Deserialize, Serialize};

/// Single Low-Rank Adaptation (LoRA) Expert Module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAExpert {
    pub id: usize,
    pub hidden_dim: usize,
    pub rank: usize,
    pub scaling: f32,

    // Weight A: [rank, hidden_dim] flattened
    pub lora_a: Vec<f32>,
    // Weight B: [hidden_dim, rank] flattened
    pub lora_b: Vec<f32>,
}

impl LoRAExpert {
    pub fn new(id: usize, hidden_dim: usize, rank: usize, alpha: f32) -> Self {
        let scaling = alpha / (rank as f32);
        let mut lora_a = vec![0.0f32; rank * hidden_dim];
        let lora_b = vec![0.0f32; hidden_dim * rank]; // Initialized to zero for identity start

        // Kaiming uniform initialization for LoRA A
        let a_scale = (1.0 / (hidden_dim as f32)).sqrt();
        for i in 0..lora_a.len() {
            let val = ((i * 1664525 + 1013904223 + id * 997) % 65536) as f32 / 65536.0;
            lora_a[i] = (val - 0.5) * 2.0 * a_scale;
        }

        Self {
            id,
            hidden_dim,
            rank,
            scaling,
            lora_a,
            lora_b,
        }
    }

    /// Forward projection: x -> rank -> hidden_dim
    /// input: [hidden_dim], output: [hidden_dim]
    pub fn forward_single(&self, x: &[f32], output: &mut [f32]) {
        assert_eq!(x.len(), self.hidden_dim);
        assert_eq!(output.len(), self.hidden_dim);

        // 1. Bottleneck projection: mid = x @ lora_a.T -> [rank]
        let mut mid = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            let a_row = &self.lora_a[r * self.hidden_dim..(r + 1) * self.hidden_dim];
            let mut dot = 0.0f32;
            for d in 0..self.hidden_dim {
                dot += x[d] * a_row[d];
            }
            mid[r] = dot;
        }

        // 2. Expansion projection: delta = (mid @ lora_b.T) * scaling -> [hidden_dim]
        for d in 0..self.hidden_dim {
            let b_row = &self.lora_b[d * self.rank..(d + 1) * self.rank];
            let mut dot = 0.0f32;
            for r in 0..self.rank {
                dot += mid[r] * b_row[r];
            }
            output[d] += dot * self.scaling;
        }
    }

    /// Calculates L2 norm of LoRA A weights
    pub fn norm_a(&self) -> f32 {
        self.lora_a.iter().map(|w| w * w).sum::<f32>().sqrt()
    }

    /// Calculates percentage of structural sparsity (exact zeros)
    pub fn sparsity_a(&self) -> f32 {
        let zeros = self.lora_a.iter().filter(|&&w| w == 0.0).count();
        (zeros as f32 / self.lora_a.len() as f32) * 100.0
    }
}

/// MoE Pool managing multiple LoRA Experts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAExpertPool {
    pub num_experts: usize,
    pub hidden_dim: usize,
    pub rank: usize,
    pub experts: Vec<LoRAExpert>,
}

impl LoRAExpertPool {
    pub fn new(cfg: &LocalSleepConfig) -> Self {
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for id in 0..cfg.num_experts {
            experts.push(LoRAExpert::new(id, cfg.hidden_dim, cfg.lora_rank, cfg.lora_alpha));
        }

        Self {
            num_experts: cfg.num_experts,
            hidden_dim: cfg.hidden_dim,
            rank: cfg.lora_rank,
            experts,
        }
    }

    /// Batched dynamic modulation pass with Top-K routing weights
    /// input: [batch, hidden_dim]
    /// routing_weights: [batch, top_k]
    /// selected_experts: [batch, top_k]
    /// returns delta: [batch, hidden_dim]
    pub fn forward(
        &self,
        input: &[f32],
        routing_weights: &[f32],
        selected_experts: &[usize],
        top_k: usize,
        batch_size: usize,
    ) -> Vec<f32> {
        let mut final_delta = vec![0.0f32; batch_size * self.hidden_dim];

        for b in 0..batch_size {
            let in_slice = &input[b * self.hidden_dim..(b + 1) * self.hidden_dim];
            let out_slice = &mut final_delta[b * self.hidden_dim..(b + 1) * self.hidden_dim];

            for k in 0..top_k {
                let weight = routing_weights[b * top_k + k];
                let expert_id = selected_experts[b * top_k + k];

                if weight > 0.0 && expert_id < self.num_experts {
                    let mut exp_out = vec![0.0f32; self.hidden_dim];
                    self.experts[expert_id].forward_single(in_slice, &mut exp_out);

                    // Accumulate weighted residual delta
                    for d in 0..self.hidden_dim {
                        out_slice[d] += exp_out[d] * weight;
                    }
                }
            }
        }

        final_delta
    }
}
