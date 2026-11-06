# 🐟 `SOMA-Fishbowl` 數位魚缸沙盒 (`:18940`) 與遠端 Web 控制台 (`:18935`) 操作手冊

> **最後更新日期：** `2026-11-06`  
> **專案路徑：** `/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe`

---

## 一、遠端 Web 控制台與 OpenAI 相容閘道 (`:18935`)

### 1. 啟動方式
- **隨桌面版 PyQt6 GUI 自動啟動**：
  ```bash
  ./run_gui.sh
  ```
- **獨立無頭 (Headless) 啟動**：
  ```bash
  ./run_web_ui.sh
  # 預設監聽 http://0.0.0.0:18935
  ```

### 2. 核心 API 端點 (`http://127.0.0.1:18935`)
- **`GET /`**：瀏覽器戰術控制台 (`ls_moe_web_ui.html`)，支援即時監控 16 顆 LoRA 專家的疲勞度、2:4 結構化稀疏率、NREM 睡眠計數、切換超立方體核心（如 `gen_022_20261105_114610_matryoshka_L1`）與開關 LoRA。
- **`GET /v1/models`**：回傳 OpenAI 相容模型清單 (`localsleep-moe`)。
- **`POST /v1/chat/completions`**：OpenAI 相容對話與程式碼生成端點（支援 `stream: true` SSE 與 `stream: false` JSON）。
  - **擴充參數**：
    - `"lora_enabled": true | false`：單次請求動態啟用或旁路 16-LoRA。
    - `"reset_kv_cache": true`：生成前重置 KV 快取。
    - `"stateless_forge": true`：長篇批次鍛造專用（不將輸出累積至 `conversation_history`，並搭配 `Dynamic Context Ceiling Guard` 動態預留輸入 Token，防止超過 TRT 24K 上限截斷）。
- **平行車道與 CLI 工具**：
  - `python3 call_localsleep_coder.py`：委派程式碼或戰術 Web UI 任務至 `:18935`。
  - `python3 pando_lanes.py`：管理 L0 $\to$ L1 $\to$ L2 平行 Git Worktree 演化車道。
  - `python3 gits_50k_fractal_forge.py` / `python3 gits_v3_negative_palindrome_forge.py`：5 萬字分形長篇小說與 v3「負面迴文 (`Negative Palindrome`)」防重複鍛造引擎。

---

## 二、`SOMA-Fishbowl` 數位魚缸沙盒與 OS 觀測艙 (`:18940`)

`SOMA-Fishbowl` 提供一個預先配置好 **`1024×768 RGBA8888` 虛擬 Framebuffer (`/dev/fb0`)**、**`64KB` 8 通道共享記憶體匯流排 (`/dev/mem_channels/devmem_arena.bin`)** 與 **掛載外部編譯器 (`/toolchain/bin/rustc`)** 的 Linux 沙盒，讓 `:18935` 智能體能將 Rust/C 程式碼編譯投放進去執行並透過畫面與記憶體通道即時互動。

### 1. 啟動魚缸觀測艙 (`run_fishbowl.sh`)
```bash
# 啟動獨立 PyQt6 魚缸觀測視窗 + :18940 網路控制伺服器
bash ./run_fishbowl.sh

# 若僅需背景啟動 :18940 HTTP 控制與抓圖伺服器（無頭模式）
python3 fishbowl_pyqt_gui.py --headless --port 18940
```

### 2. 魚缸內部硬體與環境規格 (`fishbowl/rootfs/`)
| 子系統 | 路徑 / 介面 | 規格說明 |
| :--- | :--- | :--- |
| **Virtual Framebuffer** | `fishbowl/rootfs/dev/fb0_rgba.raw`<br>(`env: FISHBOWL_FB_PATH`) | `1024 × 768 × 4` Bytes (`RGBA8888`, 共 `3,145,728` Bytes)，PyQt6 視窗以 15 FPS 即時刷新。 |
| **64KB 8-Ch DevMem Bus** | `fishbowl/rootfs/dev/mem_channels/devmem_arena.bin`<br>(`env: SOMA_DEVMEM_PATH`) | `8` 通道 × `8192` Bytes (`CH0..CH7`)：<br>• `0x00..0x04`: Magic `"SOMA"`<br>• `0x04..0x08`: `seq: u32` (寫入計數器)<br>• `0x08..0x10`: `reg_u64: u64` (64-bit 暫存器，渲染為 32 格二進位位元平面)<br>• `0x10..0x30`: `tag: [u8; 32]`<br>• `0x30..0x2000`: `payload: [u8; 8144]` |
| **Dynamic UI Config** | `fishbowl/rootfs/etc/soma_ui_config.json` (`.cfg`)<br>(`env: SOMA_UI_CFG_PATH`) | 由常駐合成器 `soma_compositor_v3` 每 `80ms` 自動熱載入，免重編譯即可改面板參數；內建 `cpu_max_cores >= 24` (`8×3` 核心矩陣) 單調防退化鎖。 |
| **In-OS Binaries & Toolchain** | `fishbowl/rootfs/bin/`<br>`fishbowl/rootfs/src/`<br>`fishbowl/rootfs/toolchain/bin/` | 內建 `/bin/soma_devmem` 與 `/bin/soma_compositor_v3`，並自動連結宿主機 `rustc`、`cargo`、`gcc`，支援於 OS 內直接編譯 `./src/*.rs` $\to$ `./bin/*`。 |
| **Docker + FrameBuffer 跳級腳本** | `fishbowl/run_docker_fishbowl.sh` | 一鍵啟動 `soma-fishbowl-os` Docker 容器，並將 `/dev/fb0_rgba.raw`、`/dev/mem_channels`、`/bin`、`/src` 與外部 Rust 工具鏈直接掛載進容器。 |

### 3. `:18940` 網路控制與畫面/Console 捕獲 API
```bash
# 1. 抓取即時 1024x768 Framebuffer 畫面 PNG
curl -s http://127.0.0.1:18940/api/screen.png -o screen.png

# 2. 查看魚缸內已安裝之 /bin Apps、掛載編譯器與 8 條 /dev/mem 通道狀態
curl -s http://127.0.0.1:18940/api/os/apps | python3 -m json.tool

# 3. 在魚缸 OS 內執行 bash 指令或操作 /bin/soma_devmem（自動同步至 CH3 畫面終端）
curl -s -X POST http://127.0.0.1:18940/api/os/bash \
  -d '{"cmd": "soma_devmem poke 4 0xCAFEBABE1337F00D CH4:AI_CTRL \"DMA_LOCKED=TRUE\""}'

# 4. 零重編譯、動態熱更新 Rust OS 介面配置
curl -s -X POST http://127.0.0.1:18940/api/ui_config \
  -d '{"osc_gain": 30, "os_title": "SOMA-OS V3 AI-CONTROLLED KERNEL"}'

# 5. 一鍵編譯並啟動常駐 1024x768 動態合成器 soma_compositor_v3
curl -s -X POST http://127.0.0.1:18940/api/start_compositor_v3 -d '{}'
```

### 4. 三模式遞歸演化引擎 (`soma_os_palindrome_loop.py`)
1. **`--mode v3-control`（預設：外部 AI 控制內部環境與 8 通道 `/dev/mem`）**：
   ```bash
   python3 soma_os_palindrome_loop.py --mode v3-control --max-rounds 6
   ```
   先啟動 `1024×768` 常駐合成器 `soma_compositor_v3`，接著每一輪由 31B 執行 6 大分階段系統任務（24 核排程、8MB RAM DMA 頻寬實測、16-LoRA 路由遮罩控制、PID/ELF 審計、PageFault 矩陣、64-bit SplitMix64 熵洗滌），在 `./src/` 撰寫並編譯獨立 Rust 程式至 `./bin/`，透過自動連結的 `soma_poke_channel`、`soma_peek_reg`、`soma_set_ui_param` 直接控制內部環境。
2. **`--mode gpui-forge`（`gpui.rs` 宣告式 Flexbox 語法相容內核 `RUST_SOMA_GPUI_SDK`）**：
   ```bash
   python3 soma_os_palindrome_loop.py --mode gpui-forge --max-rounds 3
   ```
   讓 31B 使用 `use gpui::*; div().flex().flex_row().flex_col().flex_grow().gap_3().border_1().child(...)` 宣告式語法與 `RenderCtx::new(tick)` 自動分欄排版，`0.15 秒` 單檔編譯直出 `1024×768` 桌面。
3. **`--mode v2-ui-forge`（雙幀截圖 + 31B 視覺自審評分迴圈）**：
   ```bash
   python3 soma_os_palindrome_loop.py --mode v2-ui-forge --target-score 88 --max-rounds 5
   ```
