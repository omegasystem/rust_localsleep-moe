//! lib.rs - Core Library Root for rust_localsleep_moe
//! Exposes all pure Rust modules and optional PyO3 Python bindings.

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
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pyclass]
pub struct PyFatigueRouter {
    inner: FatigueMoERouter,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyFatigueRouter {
    #[new]
    #[pyo3(signature = (num_experts=8, hidden_dim=3840, top_k=2, fatigue_threshold=2.0, temperature=1.0))]
    fn new(num_experts: usize, hidden_dim: usize, top_k: usize, fatigue_threshold: f32, temperature: f32) -> Self {
        let mut cfg = LocalSleepConfig::default();
        cfg.num_experts = num_experts;
        cfg.hidden_dim = hidden_dim;
        cfg.top_k = top_k;
        cfg.fatigue_threshold = fatigue_threshold;
        cfg.router_temperature = temperature;

        Self {
            inner: FatigueMoERouter::new(&cfg),
        }
    }

    /// Forward routing pass returning (weights, selected_experts, mean_entropy)
    fn forward(&self, hidden_states: Vec<f32>, batch_size: usize) -> PyResult<(Vec<f32>, Vec<usize>, f32)> {
        Ok(self.inner.forward(&hidden_states, batch_size))
    }

    /// Periodic homeostatic decay step
    fn step_decay(&self) {
        self.inner.step_decay();
    }

    /// Pulls live real-time homeostatic telemetry across all experts:
    /// Returns a list of tuples: (expert_id, state_int, fatigue_score, active_tokens, sleep_countdown)
    fn get_expert_states(&self) -> PyResult<Vec<(usize, usize, f32, usize, usize)>> {
        let states = self.inner.states.read();
        let fatigue = self.inner.fatigue_scores.read();
        let tokens = self.inner.active_token_counts.read();
        let countdown = self.inner.sleep_countdown.read();
        let mut res = Vec::with_capacity(self.inner.num_experts);
        for i in 0..self.inner.num_experts {
            res.push((i, states[i] as usize, fatigue[i], tokens[i], countdown[i]));
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
