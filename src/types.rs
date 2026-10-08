//! types.rs - Core Domain Types & Neuro-Dynamic Enums for rust_localsleep_moe

use serde::{Deserialize, Serialize};

/// Homeostatic Neuro-Dynamic State for each MoE Expert
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpertState {
    /// Active: Available for inference routing
    Active = 0,
    /// Fatigued: Exceeded fatigue threshold, scheduled for maintenance
    Fatigued = 1,
    /// NREM_Sleep: Undergoing synaptic downscaling & structural pruning
    NremSleep = 2,
}

impl Default for ExpertState {
    fn default() -> Self {
        ExpertState::Active
    }
}

/// Dynamic Configuration for LocalSleep-MoE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalSleepConfig {
    pub profile_name: String,
    pub model_name_or_path: String,
    pub dtype: String,
    pub hidden_dim: usize,

    // MoE Topology
    pub num_experts: usize,
    pub top_k: usize,
    pub lora_rank: usize,
    pub lora_alpha: f32,

    // Fatigue & Homeostasis
    pub fatigue_threshold: f32,
    pub wake_threshold: f32,
    pub decay_rate: f32,
    pub sleep_duration: usize,
    pub router_temperature: f32,

    // Synaptic Pruning & SHY
    pub shy_lambda: f32,
    pub enable_2_4_sparsity: bool,

    // SWA (Sliding Window Attention)
    pub enable_swa: bool,
    pub swa_window_size: usize,
    pub swa_sink_size: usize,

    // Hardware & Telemetry
    pub device: String,
    pub telemetry_port: u16,
}

impl Default for LocalSleepConfig {
    fn default() -> Self {
        Self {
            profile_name: "thor_12b".to_string(),
            model_name_or_path: "/home/omegacd/local_models/gemma-4-12B-it".to_string(),
            dtype: "bfloat16".to_string(),
            hidden_dim: 3840,
            num_experts: 8,
            top_k: 2,
            lora_rank: 16,
            lora_alpha: 32.0,
            fatigue_threshold: 2.0,
            wake_threshold: 0.3,
            decay_rate: 0.01,
            sleep_duration: 40,
            router_temperature: 1.0,
            shy_lambda: 0.08,
            enable_2_4_sparsity: true,
            enable_swa: false,
            swa_window_size: 2048,
            swa_sink_size: 4,
            device: "cuda".to_string(),
            telemetry_port: 8765,
        }
    }
}

/// Telemetry state snapshot per expert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertTelemetry {
    pub id: usize,
    pub state: usize,
    pub fatigue: f32,
    pub tokens: usize,
    pub countdown: usize,
    pub norm: f32,
    pub sparsity: f32,
}

/// Microsecond-precision Telemetry Record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryRecord {
    pub ts: String,
    pub step: usize,
    pub token_id: u32,
    pub token_str: String,
    pub entropy: f32,
    pub delta_entropy: f32,
    pub top_candidates: Vec<u32>,
    pub top_probs: Vec<f32>,
    pub routed_experts: Vec<usize>,
    pub routing_weights: Vec<f32>,
    pub experts: Vec<ExpertTelemetry>,
}
