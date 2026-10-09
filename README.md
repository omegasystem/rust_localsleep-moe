# rust_localsleep_moe 🦀⚡

> **High-Performance Zero-GIL Rust & CUDA Neural Engine for LocalSleep-MoE**  
> Complete system-level port of `localsleep-moe` (Python eager prototype) to pure Rust & custom CUDA kernels, with native Agentic Tool Calling, SWA KV-Cache rolling, and Embodied Checkpointing.

---

## 🎯 Architecture Overview & The Zero-GIL Synapse

In the Python prototype, background NREM sleep maintenance (Fisher SVD / 2:4 structural pruning) causes latency jitter due to Python GIL contention and eager PyTorch operator launches (~80μs).

**`rust_localsleep_moe`** establishes a biological dual-stream execution engine:
1. **Zero-GIL Asynchronous Worker Thread (`src/engine.rs`)**: Native OS thread with crossbeam MPSC channels running background Fisher downscaling and structural pruning with zero interference to the inference thread.
2. **Fused Fatigue-Aware Top-K Router (`src/router.rs` & `csrc/fused_fatigue_topk.cu`)**: Fuses temperature scaling, dynamic refractory sleep masking (`-1e9`), in-warp Softmax reduction, Top-K selection, and atomic Shannon entropy accumulation down to `<8μs` launch latency.
3. **Sink-Aware Sliding Window Attention (SWA) Cache (`src/cache.rs` / `swa_attention.py`)**: Constant $\mathcal{O}(1)$ VRAM memory footprint by anchoring initial Attention Sinks (`sink_size=4`) and rolling local sliding window (`window_size=2048`). Enables bounded multi-turn generation without KV-cache explosion.
4. **Zero-Copy Sparse LoRA Expert Pointer Gather (`src/expert_pool.rs` & `csrc/sparse_expert_gather.cu`)**: Pointer offset gathering across pre-allocated VRAM slabs, eliminating host-device synchronization and intermediate tensor mallocs.
5. **Agentic Embodied Harness & Multi-Line XML Artifacts (`ls_moe_agentic.py` / `tools_harness.py`)**:
   - Supports tool execution (`read_file`, `list_dir`, `web_search`).
   - Supports native multi-line XML `<file path="...">code</file>` generation, eliminating JSON `\n` escaping failures and writing directly into `./isolated_output/` sandbox.
   - Pre-flight Context Budgeting to compress prompt history before generating long code blocks.
6. **Turn-by-Turn Telemetry & Markdown Archiving (`session_logger.py`)**:
   - Human-readable Markdown logs (`session_{profile}_{ts}.md`) tracking tokens, TPS, duration, and persisted artifacts.
   - Microsecond JSONL event telemetry (`session_{profile}_{ts}.jsonl`).

---

## 🏛️ Project Structure

```
projects/rust_localsleep-moe/
├── Cargo.toml                  # High-performance dependencies & PyO3 feature flags
├── build.rs                    # Auto-detects nvcc and compiles CUDA kernels for sm_120 / sm_89 / sm_87
├── README.md                   # System Architecture Specification & User Guide
├── configs/
│   └── model_profiles.json     # Hyperparameter profiles (thor_12b, thor_31b, thor_26b_a4b)
├── checkpoints/                # SafeTensors continuous synaptic checkpoints
├── isolated_output/            # Sandbox directory for model-generated code and artifacts
├── sessions/                   # Markdown conversation transcripts & JSONL telemetry logs
│
├── ls_moe_agentic.py           # Primary Agentic inference pipeline (SWA + Context Compression + REPL)
├── tools_harness.py            # Decoupled tool execution engine & XML artifact extractor
├── session_logger.py           # Dual Markdown/JSONL turn archiver
├── swa_attention.py            # Sink-Aware Sliding Window Attention KV-cache trimmer
├── expert_pool_lora.py         # Dynamic VRAM LoRA expert synaptic pool
├── checkpoint_manager.py       # SafeTensors checkpoint persistence & restoration
├── cjk_cleaner.py              # CJK streaming token cleaner & punctuation sanitizer
│
├── src/                        # Core Rust Engine (Zero-GIL)
│   ├── lib.rs                  # Crate root & PyO3 Python bindings (rust_localsleep_moe)
│   ├── main.rs                 # Standalone native Rust CLI runner & stress benchmark
│   ├── types.rs                # Core domain types (ExpertState, LocalSleepConfig, Telemetry)
│   ├── router.rs               # Fatigue entropy routing & Top-K gating logic (<8μs)
│   ├── expert_pool.rs          # Virtual LoRA Expert Pool with dynamic scaling
│   ├── downscaler.rs           # Online Fisher information estimator & 2:4 sparsity pruner
│   ├── engine.rs               # DualStreamEngine: Native OS thread worker & task channel
│   ├── cache.rs                # SWA Sink-Aware Rolling KV Cache (Memory bounded)
│   ├── telemetry.rs            # Microsecond-level JSONL telemetry logger & terminal gauge
│   └── cuda_bridge.rs          # Safe Rust FFI bridge to custom CUDA kernels
└── csrc/                       # Hardware Acceleration Kernels
    ├── localsleep_cuda.h       # C-ABI hardware interface header
    ├── fused_fatigue_topk.cu   # Fused Top-K, Softmax, Sleep-Mask & Entropy CUDA kernel
    └── sparse_expert_gather.cu # Zero-copy sparse LoRA pointer gather CUDA kernel
```

---

## 🚀 Building & Running

### 1. Build PyO3 Python Module (Required for `ls_moe_agentic.py`)
Compile the Rust Zero-GIL router into a native Python module:
```bash
maturin develop --release
```

### 2. Run Agentic Pipeline (Interactive REPL Mode)
Launch multi-turn interactive session with SWA rolling cache and tools:
```bash
python3 ls_moe_agentic.py --profile thor_12b --interactive
```

### 3. Run Single-Turn Generation
```bash
python3 ls_moe_agentic.py \
  --profile thor_12b \
  --prompt "請使用 <file path=\"demo.py\"> 寫出一個快速排序法" \
  --max-tokens 2000
```
Generated files are safely extracted to `./isolated_output/demo.py`, and complete turn metrics are archived to `./sessions/`.

### 4. Build Pure Rust Engine & Standalone CLI
```bash
# Build pure Rust CLI
cargo build --release

# Run synthetic stress benchmark (200 tokens)
cargo run --release -- --profile thor_12b benchmark --steps 200

# Test SWA Memory Boundedness (4000 tokens through 4 layers)
cargo run --release -- test-swa --tokens 4000
```

### 5. Build with CUDA Hardware Acceleration
Targeting NVIDIA RTX 5080 (`sm_120`), RTX 4090 (`sm_89`), or Jetson Thor (`sm_87`):
```bash
cargo build --release --features cuda
```

---

## 📊 Performance Benchmarks (Empirical Target)

| Metric | Python Eager Prototype | `rust_localsleep_moe` (Pure Rust) | `rust_localsleep_moe` (Rust + CUDA Fused) |
| :--- | :--- | :--- | :--- |
| **Gating Latency** | ~80 μs | ~12 μs | **< 6.5 μs (12x speedup)** |
| **GIL Contention** | High (locks during SVD) | **Zero (Native OS Thread)** | **Zero (Native OS Thread)** |
| **KV Cache VRAM** | $\mathcal{O}(N)$ Linear Growth | **$\mathcal{O}(1)$ Bounded (SWA)** | **$\mathcal{O}(1)$ Bounded (SWA)** |
| **Throughput Jitter** | 15%~30% P99 variance | **< 2% P99 variance** | **< 1% P99 variance** |
| **Code Generation Stability** | Escaped JSON failures | **Native XML `<file>` Sandbox** | **Native XML `<file>` Sandbox** |
