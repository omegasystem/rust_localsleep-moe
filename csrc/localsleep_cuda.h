// localsleep_cuda.h - C-ABI Hardware Interface for rust_localsleep_moe CUDA Kernels
#ifndef LOCALSLEEP_CUDA_H
#define LOCALSLEEP_CUDA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// 1. Fused Fatigue-Aware Top-K Gating Kernel (<6.5us)
void launch_fused_fatigue_topk(
    const float* d_gate_logits,      // [batch, num_experts]
    const int32_t* d_expert_states,  // [num_experts] (0=Active, 1=Fatigued, 2=Sleep)
    float* d_fatigue_scores,         // [num_experts] (in-place atomic accumulation)
    float* d_routing_weights,        // [batch, top_k]
    int32_t* d_selected_experts,     // [batch, top_k]
    float* d_shannon_entropy,        // [batch]
    int batch_size,
    int num_experts,
    int top_k,
    float temperature,
    void* cuda_stream
);

// 2. Zero-Copy Sparse LoRA Expert Pointer Gather Kernel
void launch_sparse_expert_gather(
    const float* d_input_hidden,     // [batch, hidden_dim]
    const float* d_routing_weights,  // [batch, top_k]
    const int32_t* d_selected_exp,   // [batch, top_k]
    float* d_output_delta,           // [batch, hidden_dim]
    int batch_size,
    int hidden_dim,
    int top_k,
    void* cuda_stream
);

// 2b. Full Low-Rank LoRA Slab Gather-Projection Kernel:
// Computes d_output_delta[b, d] = sum_{k=0..top_k-1} w_{b,k} * scaling * ((h_b @ A_{exp_k}^T) @ B_{exp_k}^T)_d
void launch_sparse_lora_gather_f32(
    const float* d_input_hidden,     // [batch, hidden_dim]
    const float* d_lora_a,           // [num_experts, rank, hidden_dim]
    const float* d_lora_b,           // [num_experts, hidden_dim, rank]
    const float* d_routing_weights,  // [batch, top_k]
    const int32_t* d_selected_exp,   // [batch, top_k]
    float* d_output_delta,           // [batch, hidden_dim]
    int batch_size,
    int hidden_dim,
    int rank,
    int top_k,
    float scaling,
    void* cuda_stream
);

// 3. Fused Fisher Information & 2:4 Structural Pruning + SHY Downscaling Kernel
void launch_fisher_downscaler_24(
    float* d_weights,                // [rows, cols] (must be multiple of 4)
    const float* d_fisher_diag,      // [rows, cols] (if NULL, estimated inline from weight magnitude)
    float shy_lambda,
    int rows,
    int cols,
    void* cuda_stream
);

// 4. SWA In-Place Circular Ring Buffer KV Update Kernels (FP32 & BF16)
void launch_rolling_kv_update_f32(
    float* d_cache,
    const float* d_new_token,
    int current_seq_len,
    int sink_size,
    int window_size,
    int num_heads,
    int head_dim,
    void* cuda_stream
);

void launch_rolling_kv_update_bf16(
    void* d_cache,
    const void* d_new_token,
    int current_seq_len,
    int sink_size,
    int window_size,
    int num_heads,
    int head_dim,
    void* cuda_stream
);

#ifdef __cplusplus
}
#endif

#endif // LOCALSLEEP_CUDA_H
