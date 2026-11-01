//! lib.rs - Core Library Root for rust_localsleep_moe
//! Exposes all pure Rust modules and optional PyO3 Python bindings with Zero-GIL DualStreamEngine integration.

pub mod cache;
pub mod cuda_bridge;
pub mod downscaler;
pub mod engine;
pub mod expert_pool;
pub mod router;
pub mod telemetry;
pub mod types;

pub use cache::{SinkAwareRollingCache, SinkAwareRollingLayer};
pub use cuda_bridge::CudaEngineBridge;
pub use downscaler::FisherDownscaler;
pub use engine::DualStreamEngine;
pub use expert_pool::{LoRAExpert, LoRAExpertPool};
pub use router::FatigueMoERouter;
pub use telemetry::TelemetryLogger;
pub use types::{ExpertState, ExpertTelemetry, LocalSleepConfig, TelemetryRecord};

#[cfg(feature = "python")]
use parking_lot::RwLock;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::sync::Arc;

#[cfg(feature = "python")]
#[pyclass]
pub struct PyFatigueRouter {
    inner: FatigueMoERouter,
    dual_engine: Arc<DualStreamEngine>,
    downscaler: FisherDownscaler,
    expert_pool: Arc<RwLock<LoRAExpertPool>>,
    cuda_bridge: Arc<CudaEngineBridge>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFatigueRouter {
    #[new]
    #[pyo3(signature = (
        num_experts=8,
        hidden_dim=3840,
        top_k=2,
        fatigue_threshold=2.0,
        temperature=1.0,
        wake_threshold=0.3,
        decay_rate=0.01,
        sleep_duration=40,
        lora_rank=16,
        lora_alpha=32.0,
        shy_lambda=0.08,
        enable_2_4_sparsity=true
    ))]
    fn new(
        num_experts: usize,
        hidden_dim: usize,
        top_k: usize,
        fatigue_threshold: f32,
        temperature: f32,
        wake_threshold: f32,
        decay_rate: f32,
        sleep_duration: usize,
        lora_rank: usize,
        lora_alpha: f32,
        shy_lambda: f32,
        enable_2_4_sparsity: bool,
    ) -> Self {
        let mut cfg = LocalSleepConfig::default();
        cfg.num_experts = num_experts;
        cfg.hidden_dim = hidden_dim;
        cfg.top_k = top_k;
        cfg.fatigue_threshold = fatigue_threshold;
        cfg.router_temperature = temperature;
        cfg.wake_threshold = wake_threshold;
        cfg.decay_rate = decay_rate;
        cfg.sleep_duration = sleep_duration;
        cfg.lora_rank = lora_rank;
        cfg.lora_alpha = lora_alpha;
        cfg.shy_lambda = shy_lambda;
        cfg.enable_2_4_sparsity = enable_2_4_sparsity;

        Self {
            inner: FatigueMoERouter::new(&cfg),
            dual_engine: DualStreamEngine::new(),
            downscaler: FisherDownscaler::new(shy_lambda, enable_2_4_sparsity),
            expert_pool: Arc::new(RwLock::new(LoRAExpertPool::new(&cfg))),
            cuda_bridge: Arc::new(CudaEngineBridge::new()),
        }
    }

    /// Forward routing pass returning (weights, selected_experts, mean_entropy)
    /// Releases Python GIL during computation.
    fn forward(&self, py: Python<'_>, hidden_states: Vec<f32>, batch_size: usize) -> PyResult<(Vec<f32>, Vec<usize>, f32)> {
        let res = py.allow_threads(|| self.inner.forward(&hidden_states, batch_size));
        Ok(res)
    }

    /// Fast deterministic token/logprob-seeded routing on Rust side without transferring a full [hidden_dim] vector from Python!
    /// Ideal for Hybrid Inference Mode (Thor TensorRT-Edge-LLM Server + Local Host Rust Router).
    #[pyo3(signature = (token_id, top_token_ids, top_logprobs))]
    fn forward_from_token_logprobs(
        &self,
        py: Python<'_>,
        token_id: usize,
        top_token_ids: Vec<usize>,
        top_logprobs: Vec<f32>,
    ) -> PyResult<(Vec<f32>, Vec<usize>, f32)> {
        let res = py.allow_threads(|| {
            let h_dim = self.inner.hidden_dim;
            let mut synth_hidden = vec![0.0f32; h_dim];
            let scale = (1.0 / (h_dim as f32)).sqrt();

            // Construct deterministic multi-frequency neural phase representation from token + logprobs
            let base_hash = token_id.wrapping_mul(2654435761);
            for d in 0..h_dim {
                let phase = ((base_hash.wrapping_add(d.wrapping_mul(1103515245))) & 0xFFFF) as f32 / 65536.0;
                synth_hidden[d] = (phase - 0.5) * 2.0 * scale;
            }

            for (idx, (&tid, &lp)) in top_token_ids.iter().zip(top_logprobs.iter()).enumerate() {
                let prob = lp.exp().clamp(1e-5, 1.0);
                let stride = (tid.wrapping_mul(40503) + idx * 97) % h_dim.max(1);
                let span = (h_dim / 8).clamp(16, 512);
                for offset in 0..span {
                    let pos = (stride + offset * 7) % h_dim;
                    let sign = if ((tid + offset) & 1) == 0 { 1.0 } else { -1.0 };
                    synth_hidden[pos] += sign * prob * scale * 2.5;
                }
            }

            self.inner.forward(&synth_hidden, 1)
        });
        Ok(res)
    }

    /// Periodic homeostatic decay step.
    /// Automatically transitions newly fatigued experts into NremSleep and dispatches
    /// Zero-GIL background Fisher 2:4 pruning on DualStreamEngine!
    /// Returns list of expert IDs that entered NREM sleep on this step.
    fn step_decay(&self) -> PyResult<Vec<usize>> {
        let newly_sleeping = self.inner.step_decay();
        for &exp_id in &newly_sleeping {
            let pool_arc = Arc::clone(&self.expert_pool);
            let downscaler = self.downscaler.clone();
            self.dual_engine.dispatch_maintenance(
                exp_id,
                move || {
                    let mut pool = pool_arc.write();
                    if exp_id < pool.experts.len() {
                        downscaler.execute_sleep_maintenance(&mut pool.experts[exp_id].lora_a);
                        downscaler.execute_sleep_maintenance(&mut pool.experts[exp_id].lora_b);
                    }
                },
                move || {},
            );
        }
        Ok(newly_sleeping)
    }

    /// Dispatches GPU 2:4 Fisher pruning directly on a PyTorch/CUDA tensor device pointer (`tensor.data_ptr()`)
    #[pyo3(signature = (d_weights_ptr, rows, cols, shy_lambda=0.08, cuda_stream_ptr=0))]
    fn prune_gpu_tensor_24_ptr(
        &self,
        d_weights_ptr: usize,
        rows: usize,
        cols: usize,
        shy_lambda: f32,
        cuda_stream_ptr: usize,
    ) -> bool {
        self.cuda_bridge.execute_fisher_downscaler_24_ptr(
            d_weights_ptr,
            0,
            shy_lambda,
            rows,
            cols,
            cuda_stream_ptr,
        )
    }

    /// Pulls live real-time homeostatic telemetry across all experts:
    /// Returns a list of tuples: (expert_id, state_int, fatigue_score, active_tokens, sleep_countdown, sparsity_pct, sleep_cycles)
    fn get_expert_states(&self) -> PyResult<Vec<(usize, usize, f32, usize, usize, f32, usize)>> {
        let states = self.inner.states.read();
        let fatigue = self.inner.fatigue_scores.read();
        let tokens = self.inner.active_token_counts.read();
        let countdown = self.inner.sleep_countdown.read();
        let cycles = self.inner.sleep_cycle_counts.read();
        let pool = self.expert_pool.read();

        let mut res = Vec::with_capacity(self.inner.num_experts);
        for i in 0..self.inner.num_experts {
            let sparsity = if i < pool.experts.len() {
                pool.experts[i].sparsity_a()
            } else {
                0.0
            };
            res.push((
                i,
                states[i] as usize,
                fatigue[i],
                tokens[i],
                countdown[i],
                sparsity,
                cycles[i],
            ));
        }
        Ok(res)
    }

    /// Exports persistent neural state for checkpointing:
    /// Returns (gate_weights, fatigue_scores, active_token_counts, states_as_ints)
    fn export_state(&self) -> PyResult<(Vec<f32>, Vec<f32>, Vec<usize>, Vec<usize>)> {
        let gate_weights = self.inner.gate_weights.clone();
        let fatigue_scores = self.inner.fatigue_scores.read().clone();
        let active_tokens = self.inner.active_token_counts.read().clone();
        let states = self.inner.states.read().iter().map(|s| *s as usize).collect();
        Ok((gate_weights, fatigue_scores, active_tokens, states))
    }

    /// Restores persistent neural state from checkpoint
    fn import_state(
        &mut self,
        gate_weights: Vec<f32>,
        fatigue_scores: Vec<f32>,
        active_tokens: Vec<usize>,
        states: Vec<usize>,
    ) -> PyResult<()> {
        if gate_weights.len() == self.inner.gate_weights.len() {
            self.inner.gate_weights = gate_weights;
        }
        if fatigue_scores.len() == self.inner.num_experts {
            *self.inner.fatigue_scores.write() = fatigue_scores;
        }
        if active_tokens.len() == self.inner.num_experts {
            *self.inner.active_token_counts.write() = active_tokens;
        }
        if states.len() == self.inner.num_experts {
            let restored_states: Vec<ExpertState> = states
                .into_iter()
                .map(|s| match s {
                    1 => ExpertState::Fatigued,
                    2 => ExpertState::NremSleep,
                    _ => ExpertState::Active,
                })
                .collect();
            *self.inner.states.write() = restored_states;
        }
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pyclass]
pub struct PyRollingKVCache {
    sink_size: usize,
    window_size: usize,
    num_heads: usize,
    head_dim: usize,
    cuda_bridge: CudaEngineBridge,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyRollingKVCache {
    #[new]
    #[pyo3(signature = (sink_size=4, window_size=2048, num_heads=8, head_dim=128))]
    fn new(sink_size: usize, window_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            sink_size,
            window_size,
            num_heads,
            head_dim,
            cuda_bridge: CudaEngineBridge::new(),
        }
    }

    /// In-place GPU rolling update for FP32 cache tensor ptr
    fn update_f32(&self, cache_ptr: usize, new_token_ptr: usize, current_seq_len: usize) -> bool {
        self.cuda_bridge.execute_rolling_kv_update_f32(
            cache_ptr,
            new_token_ptr,
            current_seq_len,
            self.sink_size,
            self.window_size,
            self.num_heads,
            self.head_dim,
        )
    }

    /// In-place GPU rolling update for BF16 cache tensor ptr
    fn update_bf16(&self, cache_ptr: usize, new_token_ptr: usize, current_seq_len: usize) -> bool {
        self.cuda_bridge.execute_rolling_kv_update_bf16(
            cache_ptr,
            new_token_ptr,
            current_seq_len,
            self.sink_size,
            self.window_size,
            self.num_heads,
            self.head_dim,
        )
    }

    #[getter]
    fn max_capacity(&self) -> usize {
        self.sink_size + self.window_size
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn rust_localsleep_moe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFatigueRouter>()?;
    m.add_class::<PyRollingKVCache>()?;
    Ok(())
}
