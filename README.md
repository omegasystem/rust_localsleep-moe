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
├── checkpoint_manager.py       # SafeTensors checkpoint persistence & restoration (rolling max_keep=10)
├── localsleep_telemetry_db.py  # Append-only SQLite WAL blackbox telemetry vault & Grafana-Lite exporter
├── ls_moe_pyqt_gui.py          # Native PyQt6 Tactical GUI (auto-starts :18935 Web UI/API in background)
├── ls_moe_web_ui.py            # Remote Web UI & OpenAI-compatible API Server (:18935)
├── ls_moe_web_ui.html          # Browser-accessible Tactical Console & LoRA Synapse Dashboard
├── run_web_ui.sh               # Standalone launcher for :18935 Web UI / OpenAI API
├── call_localsleep_coder.py    # CLI client for delegating code/UI tasks to :18935
├── pando_lanes.py              # L0 -> L1 -> L2 parallel Git worktree lane manager
├── gits_50k_fractal_forge.py   # 50,000+ char fractal long-form novel forge with KV ceiling guard
├── gits_v3_negative_palindrome_forge.py # v3 Negative-Palindrome anti-echo & epistemic fog novel engine
├── localsleep_fishbowl.py      # SOMA-Fishbowl Linux OS Sandbox (1024x768 FB + 64KB 8-ch DevMem bus)
├── fishbowl_pyqt_gui.py        # Independent PyQt6 Fishbowl Observatory + :18940 Network Control API
├── run_fishbowl.sh             # One-click launcher for SOMA-Fishbowl PyQt6 GUI + :18940 API
├── soma_os_palindrome_loop.py  # 3-Mode SOMA-OS Evolution Engine (v3-control / gpui-forge / v2-ui-forge)
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

---

## 🌐 Remote Web UI & OpenAI-Compatible API (`:18935`)

`ls_moe_web_ui.py` exposes the entire `Gemma-4-31B-IT-NVFP4` (AGX Thor `:8000`) + `16-LoRA Hypercube Core` (`gen_022` on RTX 5080) as a remote Web UI and standard OpenAI-compatible endpoint on **Port `18935`** (automatically started in the background when launching `ls_moe_pyqt_gui.py`, or standalone via `./run_web_ui.sh`).

### 1. Launch Web UI / API Server
```bash
# Option A: Launch PyQt6 Desktop GUI (automatically starts :18935 Web/API server)
./run_gui.sh

# Option B: Launch standalone headless Web UI & OpenAI API on 0.0.0.0:18935
./run_web_ui.sh
```

### 2. HTTP & OpenAI Endpoints (`http://127.0.0.1:18935`)
- **`GET  /`**: Browser-based Tactical Web UI (`ls_moe_web_ui.html`) with live 16-LoRA fatigue/sparsity telemetry, core selector (`gen_022`), LoRA toggle, and streaming chat.
- **`GET  /v1/models`**: Standard OpenAI model discovery (`localsleep-moe`).
- **`POST /v1/chat/completions`**: OpenAI-compatible chat completions (supports both `stream: true` SSE and `stream: false` JSON).
  - **Built-in Extensions**:
    - `"lora_enabled": true | false` — Dynamically enable/bypass 16-LoRA synaptic steering per request.
    - `"reset_kv_cache": true` — Resets session KV state before generation.
    - `"stateless_forge": true` — Prevents long-form batch jobs from accumulating in `conversation_history`, paired with the **Dynamic Context Ceiling Guard** (`safe_ceiling = max(4096, 20000 - est_input_toks)`) to prevent TRT-Edge-LLM 24K context truncation.
- **`GET  /api/state`** & **`POST /api/control`**: Inspect or hot-switch hypercube cores (`gen_022_20261105_114610_matryoshka_L1`), LoRA state, loop modes, and checkpoints.

---

## 🐟 SOMA-Fishbowl Digital Aquarium & AI-Native OS Sandbox (`:18940`)

**`SOMA-Fishbowl`** (`localsleep_fishbowl.py` + `fishbowl_pyqt_gui.py` + `soma_os_palindrome_loop.py`) is an independent Linux OS & kernel-organ sandbox designed for `rust_localsleep-moe` (`:18935`). It provides a `1024×768` virtual Framebuffer, a `64KB` 8-channel shared `/dev/mem` bus, mounted host compilers (`rustc` / `cargo` / `gcc`), a PyQt6 live monitor window, and a network control API on **Port `18940`**.

### 1. Core Substrate Specifications (`fishbowl/rootfs/`)
- **Virtual Framebuffer (`/dev/fb0_rgba.raw`)**: `1024 × 768 × 4` bytes (`RGBA8888`, `3,145,728` bytes), rendered live at 15 FPS in the PyQt6 glass window and exported via `GET /api/screen.png`.
- **64KB 8-Channel Shared Memory Bus (`/dev/mem_channels/devmem_arena.bin`)**:
  - `8` channels × `8,192` bytes (`CH0..CH7`):
    - `0x00..0x04`: Magic `"SOMA"`
    - `0x04..0x08`: `seq: u32` (auto-incrementing write counter; triggers visual activity flashes on the HUD)
    - `0x08..0x10`: `reg_u64: u64` (64-bit register rendered as a 32-cell binary bitplane on the OS desktop)
    - `0x10..0x30`: `tag: [u8; 32]` (channel identifier: `CH0:KERNEL_CTRL`, `CH1:VMMU_SLAB_DMA`, `CH2:LORA_HYPERCUBE`, `CH3:BASH_IPC_RING`, `CH4..CH7:APP_TELEMETRY`)
    - `0x30..0x2000`: `payload: [u8; 8144]` (live telemetry & IPC ring buffer)
  - Includes native Rust CLI utility `/bin/soma_devmem` (`soma_devmem status | peek <ch> | poke <ch> <0xHEX> <tag> <payload>`).
- **Dynamic Hot-Reload Compositor (`soma_compositor_v3`)**:
  - Persistent Rust OS compositor daemon that hot-reloads `/etc/soma_ui_config.json` (`.cfg`) every `80ms` (`12.5 FPS`) with **zero recompilation** and a **monotonic 24-core lock (`cpu_max_cores >= 24`, `8×3` grid)** so UI capabilities never regress.
- **Mounted Toolchain & Docker+FrameBuffer Bridge**:
  - Host `rustc`, `cargo`, and `gcc` are symlinked into `fishbowl/rootfs/toolchain/bin/` so agents inside the OS can compile `/src/*.rs` $\to$ `/bin/*` directly.
  - `fishbowl/run_docker_fishbowl.sh` provides a one-click jump to a full Docker container (`soma-fishbowl-os`) with shared `/dev/fb0_rgba.raw`, `/dev/mem_channels`, and read-only host Rust toolchains.

### 2. Launching SOMA-Fishbowl & Network Control API (`:18940`)
```bash
# Launch the PyQt6 Fishbowl Observatory Window + :18940 HTTP Control Server
bash ./run_fishbowl.sh

# Or run headless HTTP Control Server only on :18940
python3 fishbowl_pyqt_gui.py --headless --port 18940
```

#### `:18940` HTTP Control & Capture Endpoints:
| Method & Endpoint | Description |
| :--- | :--- |
| `GET  /` | Web Observatory HUD mirror (`exports/fishbowl_aquarium_hud.html`) |
| `GET  /api/screen.png` | Raw live PNG image of the `1024×768` `/dev/fb0` Framebuffer |
| `GET  /api/screen_b64` | Base64 PNG + latest observation telemetry JSON |
| `GET  /api/status` | Fishbowl configuration, organs in bay, and Negative-Palindrome immune DB |
| `GET  /api/console` | Captured stdout/stderr, compiler logs, and daemon output |
| `GET  /api/devmem` | Read all 8 `/dev/mem` channels (`CH0..CH7` sequence counters, hex registers, tags, payloads) |
| `POST /api/devmem` | Poke a memory channel: `{"channel": 4, "reg_u64": "0xCAFE1337", "tag": "CH4:AI_CTRL", "payload": "..."}` |
| `GET  /api/ui_config` | Read current hot-reload UI configuration (`/etc/soma_ui_config.json`) |
| `POST /api/ui_config` | Hot-patch UI parameters (`{"osc_gain": 32, "os_title": "..."}`) in 0ms without recompiling Rust |
| `GET  /api/os/apps` | List installed `/bin` apps, `/src` files, mounted `/toolchain/bin` compilers, and daemon PID |
| `POST /api/os/bash` | Execute a command inside the OS environment (`{"cmd": "soma_devmem status"}`) and pipe output to `CH3` |
| `POST /api/start_compositor_v3` | Compile and launch the persistent `1024×768` `soma_compositor_v3` daemon |
| `POST /api/drop_source` | Compile & drop custom Rust/C source (`{"name": "...", "lang": "rust", "code": "...", "daemonize": true}`) |

### 3. Running the 3-Mode SOMA-OS Evolution Loop (`soma_os_palindrome_loop.py`)
```bash
# Mode 1 (Default): v3 AI Internal Environment Control Loop
# Starts persistent 1024x768 compositor, then 31B writes & compiles standalone Rust OS control agents
# inside ./src/ -> ./bin/ across 6 progressive missions to manipulate /dev/mem channels CH0..CH7!
python3 soma_os_palindrome_loop.py --mode v3-control --max-rounds 6

# Mode 2: SOMA-GPUI Declarative Flexbox Forge (https://gpui.rs syntax-compatible kernel SDK)
# 31B writes declarative `use gpui::*; div().flex().flex_row().flex_col().gap_3()...child(...)` layouts
# with built-in RenderCtx::new(tick) and 0.15s single-file rustc compilation!
python3 soma_os_palindrome_loop.py --mode gpui-forge --max-rounds 3

# Mode 3: v2 Visual Self-Critique Loop
# Captures dual-frame /dev/fb0 screenshots + 6-point geometry audits for 31B visual self-scoring.
python3 soma_os_palindrome_loop.py --mode v2-ui-forge --target-score 88 --max-rounds 5
```

