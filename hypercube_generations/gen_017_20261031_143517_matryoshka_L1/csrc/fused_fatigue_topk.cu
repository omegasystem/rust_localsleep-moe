// fused_fatigue_topk.cu - Fused Fatigue-Aware Top-K Gating Kernel
// Collapses logits scaling, sleep masking (-1e9), in-warp softmax, top-k selection,
// and Shannon entropy accumulation into a single kernel launch (<8us target latency).

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>
#include "localsleep_cuda.h"

#define WARP_SIZE 32
#define MAX_EXPERTS 64

// Warp-level reduce sum using shuffle down
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Warp-level reduce max using shuffle down
__device__ inline float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Main Fused Kernel: 1 Warp per batch item
__global__ void fused_fatigue_topk_kernel(
    const float* __restrict__ d_gate_logits,      // [batch, num_experts]
    const int32_t* __restrict__ d_expert_states,  // [num_experts] (0=Active, 1=Fatigued, 2=Sleep)
    float* __restrict__ d_fatigue_scores,         // [num_experts] (atomic add)
    float* __restrict__ d_routing_weights,        // [batch, top_k]
    int32_t* __restrict__ d_selected_experts,     // [batch, top_k]
    float* __restrict__ d_shannon_entropy,        // [batch]
    int batch_size,
    int num_experts,
    int top_k,
    float temperature
) {
    int b = blockIdx.x; // Batch index
    int lane = threadIdx.x; // Lane index inside warp (0..31)

    if (b >= batch_size || lane >= num_experts) return;

    // Shared memory for this batch item
    __shared__ float s_probs[MAX_EXPERTS];
    __shared__ float s_max_val;
    __shared__ float s_sum_exp;
    __shared__ float s_entropy;

    // 1. Read input logit
    float logit = d_gate_logits[b * num_experts + lane];
    int state = d_expert_states[lane];

    // Dynamic refractory sleep mask: sleeping experts set to -1e9
    if (state == 2) {
        logit = -1e9f;
    } else if (temperature > 0.0f) {
        logit /= temperature;
    }

    // 2. Warp-level Max reduction
    float local_max = warp_reduce_max(logit);
    if (lane == 0) {
        s_max_val = local_max;
    }
    __syncwarp();

    // 3. Exponentiate
    float exp_val = expf(logit - s_max_val);
    s_probs[lane] = exp_val;

    // Warp-level Sum reduction
    float local_sum = warp_reduce_sum(exp_val);
    if (lane == 0) {
        s_sum_exp = local_sum > 0.0f ? local_sum : 1.0f;
    }
    __syncwarp();

    // 4. Softmax probability & Shannon Entropy
    float prob = exp_val / s_sum_exp;
    s_probs[lane] = prob;

    float local_ent_term = (prob > 1e-12f) ? (-prob * logf(prob)) : 0.0f;
    float total_entropy = warp_reduce_sum(local_ent_term);
    if (lane == 0) {
        s_entropy = total_entropy;
        d_shannon_entropy[b] = total_entropy;
    }
    __syncwarp();

    // 5. In-warp Top-K selection (performed by lane 0 for Top-2)
    if (lane == 0) {
        float max1 = -1.0f;
        int exp1 = 0;
        float max2 = -1.0f;
        int exp2 = 0;

        for (int i = 0; i < num_experts; ++i) {
            float p = s_probs[i];
            if (p > max1) {
                max2 = max1;
                exp2 = exp1;
                max1 = p;
                exp1 = i;
            } else if (p > max2) {
                max2 = p;
                exp2 = i;
            }
        }

        // Re-normalize top-2
        float top_sum = max1 + max2;
        if (top_sum > 0.0f) {
            max1 /= top_sum;
            max2 /= top_sum;
        }

        d_routing_weights[b * top_k + 0] = max1;
        d_selected_experts[b * top_k + 0] = exp1;

        if (top_k > 1) {
            d_routing_weights[b * top_k + 1] = max2;
            d_selected_experts[b * top_k + 1] = exp2;
        }

        // Update device fatigue scores atomically
        float fatigue_inc = fmaxf(s_entropy * 0.1f, 0.01f);
        atomicAdd(&d_fatigue_scores[exp1], fatigue_inc);
        if (top_k > 1) {
            atomicAdd(&d_fatigue_scores[exp2], fatigue_inc);
        }
    }
}

// C-ABI Launch Wrapper
extern "C" void launch_fused_fatigue_topk(
    const float* d_gate_logits,
    const int32_t* d_expert_states,
    float* d_fatigue_scores,
    float* d_routing_weights,
    int32_t* d_selected_experts,
    float* d_shannon_entropy,
    int batch_size,
    int num_experts,
    int top_k,
    float temperature,
    void* cuda_stream
) {
    cudaStream_t stream = (cudaStream_t)cuda_stream;
    dim3 grid(batch_size);
    dim3 block(WARP_SIZE);

    fused_fatigue_topk_kernel<<<grid, block, 0, stream>>>(
        d_gate_logits,
        d_expert_states,
        d_fatigue_scores,
        d_routing_weights,
        d_selected_experts,
        d_shannon_entropy,
        batch_size,
        num_experts,
        top_k,
        temperature
    );
}
