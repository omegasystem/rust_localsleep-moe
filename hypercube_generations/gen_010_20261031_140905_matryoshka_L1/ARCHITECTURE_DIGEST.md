# 🧬 Parent Architecture Read-Only Digest (Source Cube: `gen_009_20261031_140747_matryoshka_L1`)
- **Source Path (Read-Only)**: `/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/hypercube_generations/gen_009_20261031_140747_matryoshka_L1`
- **Timestamp**: `2026-10-31T14:09:05`

## 📄 `src/router.rs` (234 lines, 9645 bytes)
### 結構簽名 (Signatures):
```text
L011: pub struct FatigueMoERouter {
L032: impl FatigueMoERouter {
L033: pub fn new(cfg: &LocalSleepConfig) -> Self {
L063: pub fn forward(
L161: pub fn check_fatigue_triggers(&self) -> Vec<usize> {
L176: pub fn enter_nrem_sleep_for_fatigued(&self) -> Vec<usize> {
L203: pub fn step_decay(&self) -> Vec<usize> {
```
### 核心摘錄 (Read-Only Excerpt):
```text
//! router.rs - High-Performance Fatigue-Aware MoE Router in Pure Rust
//! Implements entropy-based routing, dynamic sleep refractory masking, in-warp softmax,
//! automatic NREM sleep transition, and Shannon entropy calculation without Python runtime bottlenecks.

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
    pub sleep_duratio
```

## 📄 `src/downscaler.rs` (80 lines, 3041 bytes)
### 結構簽名 (Signatures):
```text
L008: pub struct FisherDownscaler {
L013: impl FisherDownscaler {
L014: pub fn new(shy_lambda: f32, enable_2_4: bool) -> Self {
L023: pub fn estimate_diagonal_fisher(&self, weights: &[f32], grad_scale: f32) -> Vec<f32> {
L035: pub fn prune_2_4_structural(&self, weights: &mut [f32], fim: &[f32]) {
L059: pub fn apply_homeostatic_downscaling(&self, weights: &mut [f32], fim: &[f32]) {
L071: pub fn execute_sleep_maintenance(&self, weights: &mut [f32]) {
```
### 核心摘錄 (Read-Only Excerpt):
```text
//! downscaler.rs - Online Fisher Information Estimator & Synaptic Downscaler
//! Implements Synaptic Homeostasis Hypothesis (SHY) downscaling and 2:4 structural sparsity.

use serde::{Deserialize, Serialize};

/// Synaptic Downscaler implementing Homeostatic Downscaling & Structural Sparsity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherDownscaler {
    pub shy_lambda: f32,
    pub enable_2_4: bool,
}

impl FisherDownscaler {
    pub fn new(shy_lambda: f32, enable_2_4: bool) -> Self {
        Self {
            shy_lambda,
            enable_2_4,
        }
    }

    /// Estimates diagonal Fisher Information Matrix (FIM) f
```

## 📄 `src/engine.rs` (92 lines, 3432 bytes)
### 結構簽名 (Signatures):
```text
L011: pub struct MaintenanceJob {
L018: pub struct DualStreamEngine {
L023: impl DualStreamEngine {
L024: pub fn new() -> Arc<Self> {
L062: pub fn dispatch_maintenance<F, C>(&self, expert_id: usize, job_fn: F, on_complete: C)
L079: pub fn shutdown(&self) {
L088: impl Drop for DualStreamEngine {
L089: fn drop(&mut self) {
```
### 核心摘錄 (Read-Only Excerpt):
```text
//! engine.rs - Zero-GIL Asynchronous Dual-Stream Engine in Pure Rust
//! Employs native OS worker threads and crossbeam channels to completely decouple
//! foreground inference from background NREM sleep maintenance (Fisher SVD / pruning).

use crossbeam_channel::{unbounded, Receiver, Sender};
use parking_lot::Mutex;
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Maintenance job dispatched during expert NREM sleep
pub struct MaintenanceJob {
    pub expert_id: usize,
    pub job_fn: Box<dyn FnOnce() + Send + 'static>,
    pub on_complete_callback: Option<Box<dyn FnOnce() + Send + 'static>>,
}

/// Zero-GIL Asynchronous Dual-St
```

## 📄 `src/expert_pool.rs` (145 lines, 4808 bytes)
### 結構簽名 (Signatures):
```text
L009: pub struct LoRAExpert {
L021: impl LoRAExpert {
L022: pub fn new(id: usize, hidden_dim: usize, rank: usize, alpha: f32) -> Self {
L046: pub fn forward_single(&self, x: &[f32], output: &mut [f32]) {
L073: pub fn norm_a(&self) -> f32 {
L078: pub fn sparsity_a(&self) -> f32 {
L086: pub struct LoRAExpertPool {
L093: impl LoRAExpertPool {
L094: pub fn new(cfg: &LocalSleepConfig) -> Self {
L113: pub fn forward(
```
### 核心摘錄 (Read-Only Excerpt):
```text
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
    pub fn new(id: usize, hidden_dim: usi
```

## 📄 `csrc/sparse_expert_gather.cu` (313 lines, 9939 bytes)
### 結構簽名 (Signatures):
```text
L018: __global__ void sparse_expert_gather_kernel(
L054: __global__ void lora_down_project_kernel(
L109: __global__ void lora_up_project_kernel(
L150: __global__ void fisher_downscaler_24_kernel(
L215: extern "C" void launch_sparse_expert_gather(
L241: extern "C" void launch_sparse_lora_gather_f32(
L293: extern "C" void launch_fisher_downscaler_24(
```
### 核心摘錄 (Read-Only Excerpt):
```text
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

// 1. Fast Sparse Expert Gather Kernel: Paral
```

## 📄 `csrc/fused_fatigue_topk.cu` (171 lines, 5006 bytes)
### 結構簽名 (Signatures):
```text
L033: __global__ void fused_fatigue_topk_kernel(
L142: extern "C" void launch_fused_fatigue_topk(
```
### 核心摘錄 (Read-Only Excerpt):
```text
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
        val += __shfl_down_sync(0xffffffff, val, offse
```

## 📄 `expert_pool_lora.py` (70 lines, 2908 bytes)
### 結構簽名 (Signatures):
```text
L004: class DynamicLoRAExpertPool(nn.Module):
L011: def __init__(self, num_experts=8, in_dim=3840, out_dim=3840, rank=16, alpha=16.0, device="cuda", dty
L031: def forward_delta(self, x, selected_expert_idx, weight=1.0):
L048: def export_tensors(self):
L056: def load_tensors(self, tensor_dict):
```
### 核心摘錄 (Read-Only Excerpt):
```text
import torch
import torch.nn as nn

class DynamicLoRAExpertPool(nn.Module):
    """
    Manages 8 Embodied LoRA Experts with Zero-Copy GPU allocation.
    Each expert maintains:
      - lora_a: [rank, in_dim]
      - lora_b: [out_dim, rank]
    """
    def __init__(self, num_experts=8, in_dim=3840, out_dim=3840, rank=16, alpha=16.0, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scaling = alpha / rank
        self.device = device
        self.dtype = dtype

        self.lora_a = nn.Paramete
```
