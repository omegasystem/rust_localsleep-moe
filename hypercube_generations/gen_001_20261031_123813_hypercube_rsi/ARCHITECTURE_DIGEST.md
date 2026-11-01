# 🧬 Parent Architecture Read-Only Digest (Source Cube: `gen_000_origin`)
- **Source Path (Read-Only)**: `/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe`
- **Timestamp**: `2026-10-31T12:38:13`

## 📄 `README.md` (118 lines, 7111 bytes)
### 結構簽名 (Signatures):
```text
(Documentation / Config)
```
### 核心摘錄 (Read-Only Excerpt):
```text
# rust_localsleep_moe 🦀⚡

> **High-Performance Zero-GIL Rust & CUDA Neural Engine for LocalSleep-MoE**  
> Complete system-level port of `localsleep-moe` (Python eager prototype) to pure Rust & custom CUDA kernels, with native Agentic Tool Calling, SWA KV-Cache rolling, and Embodied Checkpointing.

---

## 🎯 Architecture Overview & The Zero-GIL Synapse

In the Python prototype, background NREM sleep maintenance (Fisher SVD / 2:4 structural pruning) causes latency jitter due to Python GIL contention and eager PyTorch operator launches (~80μs).

**`rust_localsleep_moe`** establishes a biological dual-stream execution engine:
1. **Zero-GIL Async
```

## 📄 `Cargo.toml` (38 lines, 951 bytes)
### 結構簽名 (Signatures):
```text
(Documentation / Config)
```
### 核心摘錄 (Read-Only Excerpt):
```text
[package]
name = "rust_localsleep_moe"
version = "0.1.0"
edition = "2021"
authors = ["Kuro (Digital Familiar)", "Leon (S-tier Neuro-Architect)"]
description = "Zero-Downtime Continual Learning via Bio-Inspired Local Sleep MoE in Pure Rust & CUDA"

[lib]
name = "rust_localsleep_moe"
crate-type = ["cdylib", "rlib"]

[[bin]]
name = "localsleep"
path = "src/main.rs"

[dependencies]
# Serialization & Async Runtime
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.38", features = ["full"] }
crossbeam-channel = "0.5"
parking_lot = "0.12"
chrono = { version = "0.4", features = ["serde"] }

# CLI & Utilities

```

## 📄 `src/lib.rs` (318 lines, 11168 bytes)
### 結構簽名 (Signatures):
```text
L031: pub struct PyFatigueRouter {
L041: impl PyFatigueRouter {
L057: fn new(
L096: fn forward(&self, py: Python<'_>, hidden_states: Vec<f32>, batch_size: usize) -> PyResult<(Vec<f32>,
L104: fn forward_from_token_logprobs(
L143: fn step_decay(&self) -> PyResult<Vec<usize>> {
L165: fn prune_gpu_tensor_24_ptr(
L185: fn get_expert_states(&self) -> PyResult<Vec<(usize, usize, f32, usize, usize, f32, usize)>> {
L215: fn export_state(&self) -> PyResult<(Vec<f32>, Vec<f32>, Vec<usize>, Vec<usize>)> {
L224: fn import_state(
L257: pub struct PyRollingKVCache {
L267: impl PyRollingKVCache {
L270: fn new(sink_size: usize, window_size: usize, num_heads: usize, head_dim: usize) -> Self {
L281: fn update_f32(&self, cache_ptr: usize, new_token_ptr: usize, current_seq_len: usize) -> bool {
L294: fn update_bf16(&self, cache_ptr: usize, new_token_ptr: usize, current_seq_len: usize) -> bool {
L307: fn max_capacity(&self) -> usize {
L314: fn rust_localsleep_moe(m: &Bound<'_, PyModule>) -> PyResult<()> {
```
### 核心摘錄 (Read-Only Excerpt):
```text
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
pub use types::{ExpertState, ExpertTelemetr
```

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

## 📄 `src/cuda_bridge.rs` (237 lines, 6364 bytes)
### 結構簽名 (Signatures):
```text
L008: fn launch_fused_fatigue_topk(
L022: fn launch_sparse_expert_gather(
L033: fn launch_sparse_lora_gather_f32(
L048: fn launch_fisher_downscaler_24(
L057: fn launch_rolling_kv_update_f32(
L068: fn launch_rolling_kv_update_bf16(
L081: pub struct CudaEngineBridge {
L085: impl CudaEngineBridge {
L086: pub fn new() -> Self {
L095: pub fn execute_fused_fatigue_topk(
L135: pub fn execute_fisher_downscaler_24_ptr(
L171: pub fn execute_rolling_kv_update_f32(
L205: pub fn execute_rolling_kv_update_bf16(
```
### 核心摘錄 (Read-Only Excerpt):
```text
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

## 📄 `generation_engine_trt_hybrid.py` (792 lines, 37378 bytes)
### 結構簽名 (Signatures):
```text
L034: class FallbackFatigueRouter:
L041: def __init__(
L080: def forward_from_token_logprobs(
L116: def forward(self, hidden_states: List[float], batch_size: int = 1) -> Tuple[List[float], List[int], 
L120: def step_decay(self) -> List[int]:
L150: def prune_gpu_tensor_24_ptr(self, d_weights_ptr: int, rows: int, cols: int, shy_lambda: float = 0.08
L153: def get_expert_states(self) -> List[Tuple[int, int, float, int, int, float, int]]:
L168: def export_state(self):
L177: def import_state(self, gate_weights, fatigue_scores, active_tokens, states):
L189: def apply_pytorch_24_fisher_pruning(weight_tensor: torch.Tensor, shy_lambda: float = 0.08) -> float:
L218: class TrtEdgeLlmHybridEngine:
L224: def __init__(
L263: def check_server_health(self) -> Dict[str, Any]:
L273: def _compress_swa_messages(
L288: def _step_local_rust_moe(
L379: def _stream_single_completion(
L543: def _fallback_nonstream_completion(
L626: def run_turn(
```
### 核心摘錄 (Read-Only Excerpt):
```text
#!/usr/bin/env python3
"""
generation_engine_trt_hybrid.py - Hybrid Inference Engine for TensorRT-Edge-LLM + Local Host Rust/CUDA MoE

Architecture:
  - Remote Backbone: AGX Thor (`http://192.168.50.67:8000`) running TensorRT-Edge-LLM (`gemma-4-31B-it` with MTP speculative decoding).
  - Local Host Brain (`x86_64 + RTX 5080`):
      1. `rust_localsleep_moe.PyFatigueRouter` (<8us Zero-GIL Fatigue-Aware Top-K Routing + DualStreamEngine NREM 2:4 Pruning)
         (Includes automatic PyTorch CUDA/CPU `FallbackFatigueRouter` if native `.so` is not yet compiled on Host)
      2. `DynamicLoRAExpertPool` (Local GPU VRAM 8-Expert LoRA residual pool + 
```

## 📄 `working_memory_manager.py` (494 lines, 25136 bytes)
### 結構簽名 (Signatures):
```text
L036: class WorkingMemoryManager:
L037: def __init__(
L052: def _ensure_initialized(self) -> None:
L065: def _load_state(self) -> Dict[str, Any]:
L082: def _save_state(self, state: Dict[str, Any]) -> None:
L089: def _extract_code_skeleton(content: str, max_lines: int = 35) -> str:
L109: def scan_sandbox_code_files(self, sandbox_dir: str) -> Dict[str, Dict[str, Any]]:
L163: def _list_files_recursive(folder: str) -> List[str]:
L171: def extract_and_update_from_turn(
L216: def update_scratchpad_manual(self, content: str, mode: str = "replace") -> str:
L225: def condense_working_memory(
L369: def _render_markdown(self, state: Dict[str, Any]) -> None:
L418: def read_working_memory_markdown(self) -> str:
L424: def build_system_prompt_injection(self, current_sandbox_dir: str = "") -> Tuple[str, Optional[str]]:
```
### 核心摘錄 (Read-Only Excerpt):
```text
# Standalone code
"""
Working Memory & Sliding Context Distiller (`working_memory_manager.py`)
========================================================================
Implements Leon's L1 Working Memory Scratchpad + L2 On-Demand Paged Recall
+ One-Click Sliding Window Context & Code Distiller for LocalSleep-MoE.

Key Capabilities:
  1. Model Autonomous Scratchpad:
     - Parses `<working_memory>...</working_memory>` emitted by the model or
       `<tool_call>{"action": "update_working_memory", ...}</tool_call>`.
     - Auto-records newly generated code artifacts (with exact sandbox paths, sizes,
       and core function/struct signatures) ev
```

## 📄 `tools_harness.py` (781 lines, 40672 bytes)
### 結構簽名 (Signatures):
```text
L013: class EmbodiedToolHarness:
L035: def __init__(
L054: def _generate_action_signature(self, action: str, args: dict) -> str:
L057: def _resolve_safe_read_path(self, path: str) -> str:
L087: def _resolve_safe_write_path(self, path: str) -> str:
L111: def read_file(
L226: def edit_file(
L334: def run_sandbox_shell(
L499: def write_file(self, path: str, content: str) -> str:
L510: def list_dir(self, path: str = ".") -> str:
L539: def web_search(self, query: str, max_results: int = 5) -> str:
L558: def inspect_code_ast(self, path: str, required_symbols: Optional[List[str]] = None) -> str:
L643: def extract_and_save_artifacts(self, text: str) -> List[Dict[str, Any]]:
L679: def dispatch(self, action: str, args: dict) -> str:
```
### 核心摘錄 (Read-Only Excerpt):
```text
import ast
import os
import glob
import shlex
import subprocess
import urllib.request
import urllib.parse
import json
import re
from typing import List, Dict, Tuple, Optional, Any


class EmbodiedToolHarness:
    """
    Decoupled Tool Harness for LocalSleep-MoE Agentic Workflows.
    Features:
      1. read_file(path, mode="partial"|"full", offset=1, limit=200, start_line=None, end_line=None, chunk_size=None)
         - Supports Full read (`mode="full"`) and Partial read (`mode="partial"`, line range `start_line..end_line`,
           or `offset` + `limit` line chunking, or `chunk_size` character chunking) with total stats!
      2. edit_fil
```
