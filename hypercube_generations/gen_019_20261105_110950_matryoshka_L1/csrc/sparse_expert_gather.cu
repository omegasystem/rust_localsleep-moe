// sparse_expert_gather.cu - Zero-Copy Sparse LoRA Expert Pointer Gather & 2:4 Fisher Pruning Kernels
// Implements:
//   1. launch_sparse_expert_gather (fast residual scaling)
//   2. launch_sparse_lora_gather_f32 (true low-rank LoRA slab projection: delta = sum w_k * scaling * (h @ A_k^T) @ B_k^T)
//   3. launch_fisher_downscaler_24 (in-place GPU 2:4 structural sparsity + SHY homeostatic downscaling)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include "localsleep_cuda.h"

#define BLOCK_DIM 256
#define MAX_LORA_RANK 64
#define MAX_TOPK 8

// 1. Fast Sparse Expert Gather Kernel: Parallelizes over batch_size and hidden_dim
__global__ void sparse_expert_gather_kernel(
    const float* __restrict__ d_input_hidden,     // [batch, hidden_dim]
    const float* __restrict__ d_routing_weights,  // [batch, top_k]
    const int32_t* __restrict__ d_selected_exp,   // [batch, top_k]
    float* __restrict__ d_output_delta,           // [batch, hidden_dim]
    int batch_size,
    int hidden_dim,
    int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * hidden_dim;

    if (idx >= total_elements) return;

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;

    float acc_delta = 0.0f;
    float in_val = d_input_hidden[idx];

    for (int k = 0; k < top_k; ++k) {
        float weight = d_routing_weights[b * top_k + k];
        int exp_id = d_selected_exp[b * top_k + k];

        if (weight > 0.0f && exp_id >= 0) {
            // Deterministic expert-indexed phase modulation fallback when full LoRA slabs are not passed
            float phase = 1.0f + 0.05f * (float)((exp_id + d) % 7 - 3);
            acc_delta += in_val * weight * 0.1f * phase;
        }
    }

    d_output_delta[idx] = acc_delta;
}

// 2. Two-Stage True Low-Rank LoRA Gather-GEMV Kernel
// Stage A: Compute bottleneck activations mid[b, k, r] = dot(h[b, :], A[exp_{b,k}, r, :])
__global__ void lora_down_project_kernel(
    const float* __restrict__ d_input_hidden,     // [batch, hidden_dim]
    const float* __restrict__ d_lora_a,           // [num_experts, rank, hidden_dim]
    const int32_t* __restrict__ d_selected_exp,   // [batch, top_k]
    float* __restrict__ d_mid_buffer,             // [batch, top_k, rank]
    int batch_size,
    int hidden_dim,
    int rank,
    int top_k
) {
    // Grid: (batch_size * top_k, rank), Block: 256 threads reducing over hidden_dim
    int bk = blockIdx.x;
    int r = blockIdx.y;
    if (bk >= batch_size * top_k || r >= rank) return;

    int b = bk / top_k;
    int exp_id = d_selected_exp[bk];
    if (exp_id < 0) {
        if (threadIdx.x == 0) d_mid_buffer[bk * rank + r] = 0.0f;
        return;
    }

    const float* h_row = d_input_hidden + b * hidden_dim;
    const float* a_row = d_lora_a + ((size_t)exp_id * rank + r) * hidden_dim;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
        sum += h_row[d] * a_row[d];
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float s_warp_sums[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        s_warp_sums[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x >> 5)) ? s_warp_sums[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            d_mid_buffer[bk * rank + r] = block_sum;
        }
    }
}

// Stage B: Compute expansion delta[b, d] = sum_{k=0..top_k-1} w_{b,k} * scaling * dot(mid[b, k, :], B[exp_{b,k}, d, :])
__global__ void lora_up_project_kernel(
    const float* __restrict__ d_mid_buffer,       // [batch, top_k, rank]
    const float* __restrict__ d_lora_b,           // [num_experts, hidden_dim, rank]
    const float* __restrict__ d_routing_weights,  // [batch, top_k]
    const int32_t* __restrict__ d_selected_exp,   // [batch, top_k]
    float* __restrict__ d_output_delta,           // [batch, hidden_dim]
    int batch_size,
    int hidden_dim,
    int rank,
    int top_k,
    float scaling
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * hidden_dim;
    if (idx >= total) return;

    int b = idx / hidden_dim;
    int d = idx % hidden_dim;

    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        int bk = b * top_k + k;
        float w = d_routing_weights[bk];
        int exp_id = d_selected_exp[bk];
        if (w > 0.0f && exp_id >= 0) {
            const float* mid_vec = d_mid_buffer + bk * rank;
            const float* b_row = d_lora_b + ((size_t)exp_id * hidden_dim + d) * rank;
            float dot = 0.0f;
            #pragma unroll 4
            for (int r = 0; r < rank; ++r) {
                dot += mid_vec[r] * b_row[r];
            }
            acc += dot * (w * scaling);
        }
    }
    d_output_delta[idx] = acc;
}

// 3. Fused Fisher 2:4 Structural Sparsity Pruning + SHY Homeostatic Downscaling Kernel
// Each thread processes one contiguous 4-element group (16 bytes), zeroes out the 2 smallest
// by Fisher importance |w_i| * sqrt(F_i), and applies w_i *= exp(-shy_lambda / (sqrt(F_i) + 1e-5))
__global__ void fisher_downscaler_24_kernel(
    float* __restrict__ d_weights,
    const float* __restrict__ d_fisher_diag,
    float shy_lambda,
    int num_quads
) {
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_quads) return;

    int base = q * 4;
    float w[4];
    float score[4];
    float sqrt_f[4];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float wi = d_weights[base + i];
        w[i] = wi;
        float fi = (d_fisher_diag != NULL)
            ? d_fisher_diag[base + i]
            : ((wi * 0.05f) * (wi * 0.05f) + 1e-6f);
        float sf = sqrtf(fmaxf(fi, 1e-12f));
        sqrt_f[i] = sf;
        score[i] = fabsf(wi) * sf;
    }

    // Find the 2 smallest elements out of 4 to zero out (2:4 sparsity pattern)
    int idx[4] = {0, 1, 2, 3};
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (score[idx[j]] < score[idx[i]]) {
                int tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }

    // Prune 2 weakest synapses
    w[idx[0]] = 0.0f;
    w[idx[1]] = 0.0f;

    // Apply SHY homeostatic exponential downscaling to the 2 surviving synapses
    // Clamp denominator and enforce a biological survival floor >= 0.985 per sleep cycle
    // (mirrors apply_pytorch_24_fisher_pruning so synapses never underflow to 0.0!)
    #pragma unroll
    for (int k = 2; k < 4; ++k) {
        int s_idx = idx[k];
        if (w[s_idx] != 0.0f) {
            float decay = expf(-shy_lambda / (sqrt_f[s_idx] + 1e-3f));
            if (decay < 0.985f) decay = 0.985f;
            if (decay > 1.0f) decay = 1.0f;
            w[s_idx] *= decay;
        }
    }

    // Write back 4-element group
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        d_weights[base + i] = w[i];
    }
}

// C-ABI Launch Wrapper: Fast Sparse Gather
extern "C" void launch_sparse_expert_gather(
    const float* d_input_hidden,
    const float* d_routing_weights,
    const int32_t* d_selected_exp,
    float* d_output_delta,
    int batch_size,
    int hidden_dim,
    int top_k,
    void* cuda_stream
) {
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    int total_elements = batch_size * hidden_dim;
    int grid_dim = (total_elements + BLOCK_DIM - 1) / BLOCK_DIM;

    sparse_expert_gather_kernel<<<grid_dim, BLOCK_DIM, 0, stream>>>(
        d_input_hidden,
        d_routing_weights,
        d_selected_exp,
        d_output_delta,
        batch_size,
        hidden_dim,
        top_k
    );
}

// C-ABI Launch Wrapper: Full Low-Rank LoRA Slab Gather
extern "C" void launch_sparse_lora_gather_f32(
    const float* d_input_hidden,
    const float* d_lora_a,
    const float* d_lora_b,
    const float* d_routing_weights,
    const int32_t* d_selected_exp,
    float* d_output_delta,
    int batch_size,
    int hidden_dim,
    int rank,
    int top_k,
    float scaling,
    void* cuda_stream
) {
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    float* d_mid_buffer = NULL;
    size_t mid_bytes = (size_t)batch_size * top_k * rank * sizeof(float);
    if (cudaMallocAsync(&d_mid_buffer, mid_bytes, stream) != cudaSuccess) {
        return;
    }

    dim3 grid_down(batch_size * top_k, rank);
    lora_down_project_kernel<<<grid_down, BLOCK_DIM, 0, stream>>>(
        d_input_hidden,
        d_lora_a,
        d_selected_exp,
        d_mid_buffer,
        batch_size,
        hidden_dim,
        rank,
        top_k
    );

    int total_elements = batch_size * hidden_dim;
    int grid_up = (total_elements + BLOCK_DIM - 1) / BLOCK_DIM;
    lora_up_project_kernel<<<grid_up, BLOCK_DIM, 0, stream>>>(
        d_mid_buffer,
        d_lora_b,
        d_routing_weights,
        d_selected_exp,
        d_output_delta,
        batch_size,
        hidden_dim,
        rank,
        top_k,
        scaling
    );

    cudaFreeAsync(d_mid_buffer, stream);
}

// C-ABI Fisher Downscaler & 2:4 Sparsity Kernel Wrapper
extern "C" void launch_fisher_downscaler_24(
    float* d_weights,
    const float* d_fisher_diag,
    float shy_lambda,
    int rows,
    int cols,
    void* cuda_stream
) {
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    int total_elements = rows * cols;
    int num_quads = total_elements / 4;
    if (num_quads <= 0) return;

    int grid_dim = (num_quads + BLOCK_DIM - 1) / BLOCK_DIM;
    fisher_downscaler_24_kernel<<<grid_dim, BLOCK_DIM, 0, stream>>>(
        d_weights,
        d_fisher_diag,
        shy_lambda,
        num_quads
    );
}
