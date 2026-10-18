//! cuda_bridge.rs - Safe Rust FFI Bridge to Custom CUDA Kernels
//! Binds to csrc/localsleep_cuda.h with safe fallbacks if running without CUDA hardware.

use std::os::raw::c_void;

#[allow(dead_code)]
extern "C" {
    fn launch_fused_fatigue_topk(
        d_gate_logits: *const f32,
        d_expert_states: *const i32,
        d_fatigue_scores: *mut f32,
        d_routing_weights: *mut f32,
        d_selected_experts: *mut i32,
        d_shannon_entropy: *mut f32,
        batch_size: i32,
        num_experts: i32,
        top_k: i32,
        temperature: f32,
        cuda_stream: *mut c_void,
    );

    fn launch_sparse_expert_gather(
        d_input_hidden: *const f32,
        d_routing_weights: *const f32,
        d_selected_exp: *const i32,
        d_output_delta: *mut f32,
        batch_size: i32,
        hidden_dim: i32,
        top_k: i32,
        cuda_stream: *mut c_void,
    );

    fn launch_sparse_lora_gather_f32(
        d_input_hidden: *const f32,
        d_lora_a: *const f32,
        d_lora_b: *const f32,
        d_routing_weights: *const f32,
        d_selected_exp: *const i32,
        d_output_delta: *mut f32,
        batch_size: i32,
        hidden_dim: i32,
        rank: i32,
        top_k: i32,
        scaling: f32,
        cuda_stream: *mut c_void,
    );

    fn launch_fisher_downscaler_24(
        d_weights: *mut f32,
        d_fisher_diag: *const f32,
        shy_lambda: f32,
        rows: i32,
        cols: i32,
        cuda_stream: *mut c_void,
    );

    fn launch_rolling_kv_update_f32(
        d_cache: *mut f32,
        d_new_token: *const f32,
        current_seq_len: i32,
        sink_size: i32,
        window_size: i32,
        num_heads: i32,
        head_dim: i32,
        cuda_stream: *mut c_void,
    );

    fn launch_rolling_kv_update_bf16(
        d_cache: *mut c_void,
        d_new_token: *const c_void,
        current_seq_len: i32,
        sink_size: i32,
        window_size: i32,
        num_heads: i32,
        head_dim: i32,
        cuda_stream: *mut c_void,
    );
}

/// Safe wrapper for the fused fatigue gating, LoRA gather, and 2:4 pruning kernels
pub struct CudaEngineBridge {
    pub is_available: bool,
}

impl CudaEngineBridge {
    pub fn new() -> Self {
        #[cfg(feature = "cuda")]
        let is_available = true;
        #[cfg(not(feature = "cuda"))]
        let is_available = false;

        Self { is_available }
    }

    pub fn execute_fused_fatigue_topk(
        &self,
        gate_logits: &[f32],
        expert_states: &[i32],
        fatigue_scores: &mut [f32],
        routing_weights: &mut [f32],
        selected_experts: &mut [i32],
        shannon_entropy: &mut [f32],
        batch_size: usize,
        num_experts: usize,
        top_k: usize,
        temperature: f32,
    ) -> bool {
        if !self.is_available {
            return false;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            launch_fused_fatigue_topk(
                gate_logits.as_ptr(),
                expert_states.as_ptr(),
                fatigue_scores.as_mut_ptr(),
                routing_weights.as_mut_ptr(),
                selected_experts.as_mut_ptr(),
                shannon_entropy.as_mut_ptr(),
                batch_size as i32,
                num_experts as i32,
                top_k as i32,
                temperature,
                std::ptr::null_mut(),
            );
            return true;
        }

        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Executes GPU 2:4 structural sparsity pruning + SHY downscaling directly on a device pointer
    pub fn execute_fisher_downscaler_24_ptr(
        &self,
        d_weights_ptr: usize,
        d_fisher_diag_ptr: usize,
        shy_lambda: f32,
        rows: usize,
        cols: usize,
        cuda_stream_ptr: usize,
    ) -> bool {
        if !self.is_available || d_weights_ptr == 0 {
            return false;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            let fisher_ptr = if d_fisher_diag_ptr == 0 {
                std::ptr::null()
            } else {
                d_fisher_diag_ptr as *const f32
            };
            launch_fisher_downscaler_24(
                d_weights_ptr as *mut f32,
                fisher_ptr,
                shy_lambda,
                rows as i32,
                cols as i32,
                cuda_stream_ptr as *mut c_void,
            );
            return true;
        }

        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Executes zero-copy in-place SWA KV cache rolling overwrite on GPU (FP32)
    pub fn execute_rolling_kv_update_f32(
        &self,
        d_cache_ptr: usize,
        d_new_token_ptr: usize,
        current_seq_len: usize,
        sink_size: usize,
        window_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> bool {
        if !self.is_available {
            return false;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            launch_rolling_kv_update_f32(
                d_cache_ptr as *mut f32,
                d_new_token_ptr as *const f32,
                current_seq_len as i32,
                sink_size as i32,
                window_size as i32,
                num_heads as i32,
                head_dim as i32,
                std::ptr::null_mut(),
            );
            return true;
        }

        #[cfg(not(feature = "cuda"))]
        false
    }

    /// Executes zero-copy in-place SWA KV cache rolling overwrite on GPU (BF16)
    pub fn execute_rolling_kv_update_bf16(
        &self,
        d_cache_ptr: usize,
        d_new_token_ptr: usize,
        current_seq_len: usize,
        sink_size: usize,
        window_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> bool {
        if !self.is_available {
            return false;
        }

        #[cfg(feature = "cuda")]
        unsafe {
            launch_rolling_kv_update_bf16(
                d_cache_ptr as *mut c_void,
                d_new_token_ptr as *const c_void,
                current_seq_len as i32,
                sink_size as i32,
                window_size as i32,
                num_heads as i32,
                head_dim as i32,
                std::ptr::null_mut(),
            );
            return true;
        }

        #[cfg(not(feature = "cuda"))]
        false
    }
}
