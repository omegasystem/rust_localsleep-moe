#!/usr/bin/env python3
"""
LocalSleep-MoE Linux Fishbowl ("數位魚缸" VM / Sandbox Tool & Organ Observatory)
================================================================================
A dedicated Linux sandbox / VM "Fishbowl" for `rust_localsleep-moe` (:18935) where:
1. Default Linux environment is pre-configured with:
   - Standard Linux telemetry interfaces (`/proc/stat`, `/proc/meminfo`, `/proc/loadavg`)
   - Virtual Linux Framebuffer device (`FISHBOWL_FB_PATH` / `/dev/fb0` compatible 640x360x32bpp RGBA mmap file)
   - Organ Drop Bay (`fishbowl/organ_bay/`) & Snapshot LKG (Last-Known-Good) state
   - Negative-Palindrome Immune Memory (`fishbowl/negative_palindrome_db.json`)
2. The AI Agent (`:18935` Gemma-4-31B + gen_022) or user can:
   - Write Rust (`.rs`) or C (`.c`) system/kernel-adjacent code
   - Compile it natively via `rustc` / `gcc` (with automatic Negative-Palindrome compiler error recovery)
   - Drop the compiled binary into the Fishbowl (`drop` / `ai-forge`)
   - Observe its live execution, stdout/stderr, resource timing, AND its live `/dev/fb0` Framebuffer pixels
     rendered inside a Tactical Aquarium Glass HUD (`exports/fishbowl_aquarium_hud.html`)!

CLI Usage:
  python3 localsleep_fishbowl.py init
  python3 localsleep_fishbowl.py status
  python3 localsleep_fishbowl.py drop --source my_organ.rs --name pulse_organ
  python3 localsleep_fishbowl.py ai-forge --name neuro_fb_organ --task "Write a Rust program that reads /proc/meminfo and /proc/loadavg, and draws a cybernetic neural radar + live memory bar graph directly into the 640x360 RGBA framebuffer file at env FISHBOWL_FB_PATH"
  python3 localsleep_fishbowl.py reset
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parent
FISHBOWL_DIR = BASE_DIR / "fishbowl"
ROOTFS_DIR = FISHBOWL_DIR / "rootfs"
ORGAN_BAY_DIR = FISHBOWL_DIR / "organ_bay"
SRC_BAY_DIR = FISHBOWL_DIR / "src_bay"
SNAPSHOT_DIR = FISHBOWL_DIR / "snapshots"
TELEMETRY_LOG = FISHBOWL_DIR / "observation_log.jsonl"
PALINDROME_DB = FISHBOWL_DIR / "negative_palindrome_db.json"
DAEMON_PID_FILE = FISHBOWL_DIR / "active_os_daemon.pid"
DAEMON_LOG_FILE = FISHBOWL_DIR / "active_os_daemon.log"
FB_FILE = ROOTFS_DIR / "dev" / "fb0_rgba.raw"
DEVMEM_DIR = ROOTFS_DIR / "dev" / "mem_channels"
DEVMEM_FILE = DEVMEM_DIR / "devmem_arena.bin"
UI_CONFIG_JSON = ROOTFS_DIR / "etc" / "soma_ui_config.json"
UI_CONFIG_CFG = ROOTFS_DIR / "etc" / "soma_ui_config.cfg"
OS_BIN_DIR = ROOTFS_DIR / "bin"
OS_SRC_DIR = ROOTFS_DIR / "src"
OS_TOOLCHAIN_DIR = ROOTFS_DIR / "toolchain" / "bin"
OS_APPS_STATE_FILE = ROOTFS_DIR / "proc_mirror" / "soma_apps.json"
DOCKER_LAUNCHER_SCRIPT = FISHBOWL_DIR / "run_docker_fishbowl.sh"
HUD_HTML_OUT = BASE_DIR / "exports" / "fishbowl_aquarium_hud.html"

FB_WIDTH = 1024
FB_HEIGHT = 768
FB_BYTES = FB_WIDTH * FB_HEIGHT * 4  # 1024x768 RGBA8888 (3,145,728 bytes)

DEVMEM_CHANNELS = 8
DEVMEM_CH_SIZE = 8192
DEVMEM_TOTAL_BYTES = DEVMEM_CHANNELS * DEVMEM_CH_SIZE  # 65,536 bytes (64 KB shared arena)

DEFAULT_UI_CONFIG: Dict[str, Any] = {
    "version": "3.0-DYNAMIC",
    "os_title": "SOMA-OS v3.0 KERNEL + DEVMEM BUS",
    "cpu_max_cores": 24,       # Monotonically locked >= 24 so it NEVER regresses to 16!
    "cpu_cols": 8,
    "cpu_rows": 3,
    "cpu_col_step": 35,
    "cpu_row_step": 34,
    "cpu_bar_w": 22,
    "cpu_bar_max_h": 18,
    "vmmu_show_devmem": 1,     # 1 = Show 8 live /dev/mem channels (CH0..CH7) + RAM bar
    "lora_box_w": 66,          # Wider box (66px) so "E02 ACTIVE" (60px) never overflows!
    "lora_box_h": 24,
    "lora_col_step": 71,
    "lora_row_step": 29,
    "br_mode": "SPLIT_OSC_BASH",  # SPLIT_OSC_BASH | OSCILLOSCOPE | BASH_CONSOLE
    "osc_gain": 22,
    "dock_text_y_offset": 9,   # Centered baseline inside 26px bottom dock
    "tl_border_r": 56, "tl_border_g": 189, "tl_border_b": 248,
    "tr_border_r": 52, "tr_border_g": 211, "tr_border_b": 153,
    "bl_border_r": 251, "bl_border_g": 191, "bl_border_b": 36,
    "br_border_r": 244, "br_border_g": 114, "br_border_b": 182,
}

DEFAULT_DEVMEM_TAGS = [
    (0, 0x0000_0001_0000_0018, "CH0:KERNEL_CTRL", "STATUS=ONLINE CORES=24 FPS=12.5"),
    (1, 0x0000_0000_0001_F68E, "CH1:VMMU_SLAB_DMA", "ARENA=64KB PAGE_SIZE=4096 MAPPED=OK"),
    (2, 0x0000_FFFF_0003_A10F, "CH2:LORA_HYPERCUBE", "ACTIVE_EXPERTS=16 TOP_K=2 NREM_MASK=0x0300"),
    (3, 0x0000_0000_0000_0001, "CH3:BASH_IPC_RING", "[OS-SHELL READY] /bin/soma_devmem & /toolchain mounted"),
    (4, 0x0000_0000_CAFE_0004, "CH4:APP_TELEMETRY", "SLOT_0=IDLE (Ready for in-OS Rust app telemetry)"),
    (5, 0x0000_0000_CAFE_0005, "CH5:MEM_SCANNER", "SCAN_ADDR=0x8000_0000 ENTROPY=0.942"),
    (6, 0x0000_0000_CAFE_0006, "CH6:USER_RING_BUF", "RING_HEAD=0 RING_TAIL=0 CAP=8144B"),
    (7, 0x0000_0000_DEAD_BEEF, "CH7:WATCHDOG_GUARD", "WDT_TIMEOUT=5000MS PALINDROME_LOCK=ARMED"),
]

DEFAULT_ENDPOINT = "http://127.0.0.1:18935"


def _sync_ui_cfg_file(cfg_dict: Dict[str, Any]) -> None:
    """Writes both JSON and flat key=value .cfg so pure-std Rust compositor hot-reloads in <0.1ms."""
    UI_CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Enforce monotonic lock on cpu_max_cores >= 24
    if int(cfg_dict.get("cpu_max_cores", 24)) < 24:
        cfg_dict["cpu_max_cores"] = 24
    UI_CONFIG_JSON.write_text(json.dumps(cfg_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [f"{k}={v}" for k, v in cfg_dict.items()]
    UI_CONFIG_CFG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_ui_config() -> Dict[str, Any]:
    ensure_fishbowl_initialized()
    if UI_CONFIG_JSON.exists():
        try:
            data = json.loads(UI_CONFIG_JSON.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_UI_CONFIG)
            merged.update(data)
            if int(merged.get("cpu_max_cores", 24)) < 24:
                merged["cpu_max_cores"] = 24
            return merged
        except Exception:
            pass
    _sync_ui_cfg_file(dict(DEFAULT_UI_CONFIG))
    return dict(DEFAULT_UI_CONFIG)


def update_ui_config(patch: Dict[str, Any]) -> Dict[str, Any]:
    """Hot-updates UI configuration without recompiling the Rust compositor. Enforces 24-core monotonic lock."""
    cur = get_ui_config()
    for k, v in patch.items():
        if k == "cpu_max_cores" and int(v) < 24:
            continue  # Never allow regression below 24 cores
        cur[k] = v
    _sync_ui_cfg_file(cur)
    return cur


def init_devmem_arena(force: bool = False) -> None:
    """Initializes the 64KB (/dev/mem_channels/devmem_arena.bin) 8-channel shared memory bus."""
    DEVMEM_DIR.mkdir(parents=True, exist_ok=True)
    if DEVMEM_FILE.exists() and DEVMEM_FILE.stat().st_size == DEVMEM_TOTAL_BYTES and not force:
        return
    arena = bytearray(DEVMEM_TOTAL_BYTES)
    for ch, reg_val, tag, payload in DEFAULT_DEVMEM_TAGS:
        base = ch * DEVMEM_CH_SIZE
        arena[base : base + 4] = b"SOMA"
        arena[base + 4 : base + 8] = (1).to_bytes(4, "little")
        arena[base + 8 : base + 16] = int(reg_val & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
        tag_b = tag.encode("utf-8")[:31].ljust(32, b"\x00")
        arena[base + 16 : base + 48] = tag_b
        pay_b = payload.encode("utf-8")[: DEVMEM_CH_SIZE - 48]
        arena[base + 48 : base + 48 + len(pay_b)] = pay_b
    DEVMEM_FILE.write_bytes(bytes(arena))


def peek_devmem_channel(ch: int) -> Dict[str, Any]:
    """Reads one 8KB channel (0..7) from /dev/mem_channels/devmem_arena.bin."""
    init_devmem_arena()
    ch = max(0, min(DEVMEM_CHANNELS - 1, int(ch)))
    raw = DEVMEM_FILE.read_bytes()
    base = ch * DEVMEM_CH_SIZE
    block = raw[base : base + DEVMEM_CH_SIZE]
    magic = block[0:4].decode("ascii", errors="replace")
    seq = int.from_bytes(block[4:8], "little")
    reg_u64 = int.from_bytes(block[8:16], "little")
    tag = block[16:48].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    payload_raw = block[48:].split(b"\x00", 1)[0]
    payload_str = payload_raw[:512].decode("utf-8", errors="replace")
    hex_head = " ".join(f"{b:02X}" for b in block[:32])
    return {
        "channel": ch,
        "offset_hex": f"0x{base:04X}",
        "magic": magic,
        "seq": seq,
        "reg_u64": reg_u64,
        "reg_hex": f"0x{reg_u64:016X}",
        "tag": tag,
        "payload": payload_str,
        "hex_head_32b": hex_head,
    }


def get_all_devmem_channels() -> List[Dict[str, Any]]:
    init_devmem_arena()
    return [peek_devmem_channel(ch) for ch in range(DEVMEM_CHANNELS)]


def poke_devmem_channel(
    ch: int,
    reg_u64: Optional[int] = None,
    tag: Optional[str] = None,
    payload: Optional[str] = None,
) -> Dict[str, Any]:
    """Writes directly to a 8KB channel in /dev/mem_channels/devmem_arena.bin and increments seq counter."""
    init_devmem_arena()
    ch = max(0, min(DEVMEM_CHANNELS - 1, int(ch)))
    arena = bytearray(DEVMEM_FILE.read_bytes())
    base = ch * DEVMEM_CH_SIZE
    arena[base : base + 4] = b"SOMA"
    cur_seq = int.from_bytes(arena[base + 4 : base + 8], "little") + 1
    arena[base + 4 : base + 8] = (cur_seq & 0xFFFFFFFF).to_bytes(4, "little")
    if reg_u64 is not None:
        arena[base + 8 : base + 16] = int(reg_u64 & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little")
    if tag is not None:
        tag_b = str(tag).encode("utf-8")[:31].ljust(32, b"\x00")
        arena[base + 16 : base + 48] = tag_b
    if payload is not None:
        pay_b = str(payload).encode("utf-8")[: DEVMEM_CH_SIZE - 49]
        arena[base + 48 : base + DEVMEM_CH_SIZE] = b"\x00" * (DEVMEM_CH_SIZE - 48)
        arena[base + 48 : base + 48 + len(pay_b)] = pay_b
    DEVMEM_FILE.write_bytes(bytes(arena))
    return peek_devmem_channel(ch)


def ensure_fishbowl_initialized() -> Dict[str, Any]:
    """Pre-configures the default Linux Fishbowl environment, virtual /dev/fb0, 64KB DevMem bus, and mounted Toolchain."""
    for d in [
        FISHBOWL_DIR,
        ROOTFS_DIR,
        ROOTFS_DIR / "dev",
        DEVMEM_DIR,
        ROOTFS_DIR / "etc",
        OS_BIN_DIR,
        OS_SRC_DIR,
        OS_TOOLCHAIN_DIR,
        ROOTFS_DIR / "proc_mirror",
        ROOTFS_DIR / "sys_mirror",
        ORGAN_BAY_DIR,
        SRC_BAY_DIR,
        SNAPSHOT_DIR,
        BASE_DIR / "exports",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    init_devmem_arena(force=False)
    if not UI_CONFIG_JSON.exists() or not UI_CONFIG_CFG.exists():
        _sync_ui_cfg_file(dict(DEFAULT_UI_CONFIG))

    if not FB_FILE.exists() or FB_FILE.stat().st_size != FB_BYTES:
        # Initialize dark tactical grid background in /dev/fb0_rgba.raw
        buf = bytearray(FB_BYTES)
        for y in range(FB_HEIGHT):
            for x in range(FB_WIDTH):
                idx = (y * FB_WIDTH + x) * 4
                is_grid = (x % 32 == 0) or (y % 32 == 0)
                buf[idx + 0] = 10 if not is_grid else 20   # R
                buf[idx + 1] = 16 if not is_grid else 38   # G
                buf[idx + 2] = 28 if not is_grid else 62   # B
                buf[idx + 3] = 255                         # A
        FB_FILE.write_bytes(bytes(buf))
        # Save LKG clean framebuffer snapshot
        (SNAPSHOT_DIR / "fb0_lkg.raw").write_bytes(bytes(buf))

    if not PALINDROME_DB.exists():
        PALINDROME_DB.write_text(
            json.dumps(
                {
                    "CompileErrors": [],
                    "RuntimeCrashes": [],
                    "ActiveNegativeConstraints": [
                        "Must NOT use external crates (std library only so rustc compiles in <1s without cargo network lock).",
                        "Must check FISHBOWL_FB_PATH environment variable (fallback to ./fishbowl/rootfs/dev/fb0_rgba.raw) and write exactly 640*360*4 RGBA bytes.",
                        "Must terminate cleanly within the observation tick budget (no infinite hang loops without sleep/exit).",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # Detect and mount external compilers into fishbowl/rootfs/toolchain/bin/
    rustc_bin = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
    if not Path(str(rustc_bin)).exists():
        rustc_bin = None
    cargo_bin = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not Path(str(cargo_bin)).exists():
        cargo_bin = None
    gcc_bin = shutil.which("gcc") or shutil.which("cc")
    qemu_aarch64 = shutil.which("qemu-system-aarch64")
    qemu_x86 = shutil.which("qemu-system-x86_64")

    for tool_name, host_p in [("rustc", rustc_bin), ("cargo", cargo_bin), ("gcc", gcc_bin)]:
        if host_p and Path(str(host_p)).exists():
            dst_link = OS_TOOLCHAIN_DIR / tool_name
            if not dst_link.exists():
                try:
                    dst_link.symlink_to(Path(str(host_p)).resolve())
                except Exception:
                    pass

    # Generate Docker + Shared Framebuffer/DevMem + Host Toolchain launcher script
    if not DOCKER_LAUNCHER_SCRIPT.exists():
        cargo_home = os.path.expanduser("~/.cargo")
        rustup_home = os.path.expanduser("~/.rustup")
        DOCKER_LAUNCHER_SCRIPT.write_text(
            f"""#!/usr/bin/env bash
# SOMA-Fishbowl Docker + Shared FrameBuffer & DevMem Channel Bridge
# Mounts /dev/fb0_rgba.raw, /dev/mem_channels/devmem_arena.bin, /bin, /src, and Host Rust Toolchain into Docker!
set -e
CONTAINER_NAME="soma-fishbowl-os"
IMAGE="${{SOMA_DOCKER_IMAGE:-ubuntu:24.04}}"

echo "🐳 [SOMA-Docker-FB] Launching container $CONTAINER_NAME with shared Framebuffer & 64KB DevMem Bus..."
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER_NAME" \\
  --hostname soma-kernel \\
  -v "{ROOTFS_DIR / 'dev'}:/dev/soma_bus" \\
  -v "{OS_BIN_DIR}:/bin/soma" \\
  -v "{OS_SRC_DIR}:/src" \\
  -v "{ROOTFS_DIR / 'etc'}:/etc/soma" \\
  -v "{cargo_home}:/opt/rust/cargo:ro" \\
  -v "{rustup_home}:/opt/rust/rustup:ro" \\
  -e FISHBOWL_FB_PATH="/dev/soma_bus/fb0_rgba.raw" \\
  -e SOMA_DEVMEM_PATH="/dev/soma_bus/mem_channels/devmem_arena.bin" \\
  -e SOMA_UI_CFG_PATH="/etc/soma/soma_ui_config.cfg" \\
  -e PATH="/bin/soma:/opt/rust/cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \\
  -e RUSTUP_HOME="/opt/rust/rustup" \\
  "$IMAGE" tail -f /dev/null

echo "✅ Container $CONTAINER_NAME is UP and bound to Fishbowl Framebuffer: {FB_FILE}"
echo "   Run inside Docker OS: docker exec -it $CONTAINER_NAME bash"
""",
            encoding="utf-8",
        )
        DOCKER_LAUNCHER_SCRIPT.chmod(0o755)

    env_info = {
        "fishbowl_root": str(FISHBOWL_DIR),
        "fb_device_path": str(FB_FILE),
        "fb_resolution": f"{FB_WIDTH}x{FB_HEIGHT}x32bpp (RGBA8888, {FB_BYTES:,} bytes)",
        "devmem_arena_path": str(DEVMEM_FILE),
        "devmem_channels": f"{DEVMEM_CHANNELS} channels x {DEVMEM_CH_SIZE}B ({DEVMEM_TOTAL_BYTES:,} bytes)",
        "ui_config_path": str(UI_CONFIG_JSON),
        "os_bin_dir": str(OS_BIN_DIR),
        "os_src_dir": str(OS_SRC_DIR),
        "os_toolchain_dir": str(OS_TOOLCHAIN_DIR),
        "docker_fb_launcher": str(DOCKER_LAUNCHER_SCRIPT),
        "compilers": {
            "rustc": str(rustc_bin) if rustc_bin else None,
            "cargo": str(cargo_bin) if cargo_bin else None,
            "gcc": str(gcc_bin) if gcc_bin else None,
        },
        "vm_backends": {
            "linux_namespace_sandbox": True,
            "docker_fb_bridge_script": str(DOCKER_LAUNCHER_SCRIPT),
            "qemu_aarch64": str(qemu_aarch64) if qemu_aarch64 else None,
            "qemu_x86_64": str(qemu_x86) if qemu_x86 else None,
        },
    }
    (FISHBOWL_DIR / "fishbowl_config.json").write_text(
        json.dumps(env_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return env_info


def rgba_to_png_b64(rgba_bytes: bytes, width: int = FB_WIDTH, height: int = FB_HEIGHT) -> str:
    """Zero-dependency pure-Python PNG encoder for the Fishbowl /dev/fb0 Framebuffer."""
    if len(rgba_bytes) < width * height * 4:
        rgba_bytes = rgba_bytes.ljust(width * height * 4, b"\x00")

    raw_rows = bytearray()
    stride = width * 4
    for y in range(height):
        raw_rows.append(0)  # Filter type 0 (None)
        row_start = y * stride
        raw_rows.extend(rgba_bytes[row_start : row_start + stride])

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return len(data).to_bytes(4, "big") + chunk_type + data + crc.to_bytes(4, "big")

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes([8, 6, 0, 0, 0])  # 8-bit depth, color type 6 (RGBA)
    )
    idat = zlib.compress(bytes(raw_rows), level=6)
    png = b"\x89PNG\r\n\x1a\n" + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", idat) + make_chunk(b"IEND", b"")
    return base64.b64encode(png).decode("ascii")


def record_negative_palindrome(kind: str, organ_name: str, error_summary: str) -> None:
    """Appends a failure signature to the Negative-Palindrome Immune DB so the SLM never repeats it."""
    ensure_fishbowl_initialized()
    db = json.loads(PALINDROME_DB.read_text(encoding="utf-8"))
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "organ": organ_name,
        "error": error_summary[:400],
    }
    db.setdefault(kind, []).append(entry)
    # Extract concise negative constraint rule
    first_line = error_summary.strip().splitlines()[0][:180] if error_summary.strip() else "unknown error"
    rule = f"[Anti-Echo from {organ_name}]: Avoid `{first_line}`"
    constraints = db.setdefault("ActiveNegativeConstraints", [])
    if rule not in constraints:
        constraints.append(rule)
        if len(constraints) > 12:
            del constraints[:-12]
    PALINDROME_DB.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")


def compile_source_into_organ(source_path: Path, organ_name: str) -> Tuple[bool, Optional[Path], str]:
    """Compiles a .rs (Rust) or .c (C) source file into a native Linux ELF binary inside organ_bay/."""
    ensure_fishbowl_initialized()
    ext = source_path.suffix.lower()
    out_bin = ORGAN_BAY_DIR / organ_name

    rustc_bin = shutil.which("rustc") or os.path.expanduser("~/.cargo/bin/rustc")
    gcc_bin = shutil.which("gcc") or shutil.which("cc")

    if ext == ".rs":
        if not rustc_bin or not Path(str(rustc_bin)).exists():
            return False, None, "rustc compiler not found in PATH or ~/.cargo/bin/rustc"
        cmd = [str(rustc_bin), "-O", str(source_path), "-o", str(out_bin)]
    elif ext == ".c":
        if not gcc_bin:
            return False, None, "gcc/cc compiler not found"
        cmd = [str(gcc_bin), "-O2", "-Wall", str(source_path), "-lm", "-o", str(out_bin)]
    else:
        return False, None, f"Unsupported source extension: {ext} (expected .rs or .c)"

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    dt = (time.time() - t0) * 1000.0
    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout).strip()
        record_negative_palindrome("CompileErrors", organ_name, err_msg)
        return False, None, f"Compilation failed ({dt:.0f}ms):\n{err_msg}"

    out_bin.chmod(0o755)
    # Also deploy into OS /bin directory so in-OS bash and apps can execute it directly
    try:
        os_bin_target = OS_BIN_DIR / organ_name
        shutil.copy2(out_bin, os_bin_target)
        os_bin_target.chmod(0o755)
    except Exception:
        pass
    return True, out_bin, f"Compiled successfully in {dt:.0f}ms -> {out_bin} ({out_bin.stat().st_size:,} bytes)"


def stop_active_os_daemon() -> Optional[int]:
    """Stops any currently running background OS daemon inside the Fishbowl before hot-swapping."""
    if not DAEMON_PID_FILE.exists():
        return None
    try:
        pid = int(DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 15)  # SIGTERM
        for _ in range(10):
            time.sleep(0.05)
            try:
                os.kill(pid, 0)
            except OSError:
                break
        else:
            os.kill(pid, 9)  # SIGKILL
    except Exception:
        pid = None
    try:
        DAEMON_PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return pid


def is_os_daemon_alive() -> Tuple[bool, Optional[int]]:
    if not DAEMON_PID_FILE.exists():
        return False, None
    try:
        pid = int(DAEMON_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        return True, pid
    except Exception:
        return False, None


def run_organ_in_fishbowl(
    bin_path: Path,
    organ_name: str,
    timeout_sec: float = 5.0,
    args_list: Optional[List[str]] = None,
    daemonize: bool = False,
) -> Dict[str, Any]:
    """
    Drops the compiled binary into the Linux Fishbowl.
    If `daemonize=True` (or if the organ stays alive as a persistent OS loop), spawns it as a
    background daemon (hot-swapping any previous daemon), samples 2 frames across 0.7s to verify
    live animation & health, and leaves it running inside the Fishbowl!
    """
    ensure_fishbowl_initialized()
    if daemonize:
        stop_active_os_daemon()

    env = os.environ.copy()
    env["FISHBOWL_FB_PATH"] = str(FB_FILE)
    env["FISHBOWL_WIDTH"] = str(FB_WIDTH)
    env["FISHBOWL_HEIGHT"] = str(FB_HEIGHT)
    env["FISHBOWL_ROOT"] = str(ROOTFS_DIR)
    env["SOMA_DEVMEM_PATH"] = str(DEVMEM_FILE)
    env["SOMA_UI_CFG_PATH"] = str(UI_CONFIG_CFG)
    env["SOMA_OS_BIN_DIR"] = str(OS_BIN_DIR)
    env["SOMA_OS_SRC_DIR"] = str(OS_SRC_DIR)
    env["PATH"] = f"{OS_BIN_DIR}:{OS_TOOLCHAIN_DIR}:{ORGAN_BAY_DIR}:" + env.get("PATH", "/usr/bin:/bin")

    fb_before = FB_FILE.read_bytes()
    (SNAPSHOT_DIR / "fb0_lkg.raw").write_bytes(fb_before)

    cmd = [str(bin_path)] + (args_list or [])
    t0 = time.time()

    if daemonize:
        log_f = open(DAEMON_LOG_FILE, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOTFS_DIR),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        DAEMON_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        # Wait 0.35s for first frames, capture Frame A, wait 0.40s, capture Frame B to measure live FPS/animation
        time.sleep(0.35)
        frame_a = FB_FILE.read_bytes() if FB_FILE.exists() else fb_before
        time.sleep(0.40)
        frame_b = FB_FILE.read_bytes() if FB_FILE.exists() else frame_a
        dt_ms = (time.time() - t0) * 1000.0

        poll_rc = proc.poll()
        alive = poll_rc is None
        rc = 0 if alive else int(poll_rc or 0)
        log_f.flush()
        log_f.close()
        out_str = DAEMON_LOG_FILE.read_text(encoding="utf-8", errors="replace") if DAEMON_LOG_FILE.exists() else ""
        stdout_str = out_str
        stderr_str = "" if alive else f"Daemon exited prematurely with code {rc}"
        anim_diff_pixels = sum(
            1 for i in range(0, FB_BYTES, 4) if frame_a[i : i + 3] != frame_b[i : i + 3]
        )
        timed_out = False
    else:
        anim_diff_pixels = 0
        alive = False
        timed_out = False
        try:
            proc_run = subprocess.run(
                cmd,
                cwd=str(ROOTFS_DIR),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            rc = proc_run.returncode
            stdout_str = proc_run.stdout
            stderr_str = proc_run.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            rc = -9
            stdout_str = (
                e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            )
            stderr_str = f"[WATCHDOG SIGKILL]: Organ exceeded {timeout_sec}s tick budget!"
        dt_ms = (time.time() - t0) * 1000.0

    # Inspect Framebuffer modification
    fb_after = FB_FILE.read_bytes() if FB_FILE.exists() else fb_before
    if len(fb_after) != FB_BYTES:
        fb_after = fb_after[:FB_BYTES].ljust(FB_BYTES, b"\x00")
        FB_FILE.write_bytes(fb_after)

    changed_pixels = sum(
        1
        for i in range(0, FB_BYTES, 4)
        if fb_after[i : i + 3] != fb_before[i : i + 3]
    )
    fb_coverage_pct = round(changed_pixels * 100.0 / (FB_WIDTH * FB_HEIGHT), 2)

    if daemonize:
        status = "DAEMON_RUNNING" if alive else f"DAEMON_EXITED(rc={rc})"
    else:
        status = "HEALTHY" if (rc == 0 and not timed_out) else ("WATCHDOG_TIMEOUT" if timed_out else f"CRASHED(rc={rc})")

    if status not in ("HEALTHY", "DAEMON_RUNNING"):
        record_negative_palindrome("RuntimeCrashes", organ_name, f"{status}: {stderr_str[:300]}")
        if changed_pixels == 0:
            FB_FILE.write_bytes(fb_before)

    loadavg = Path("/proc/loadavg").read_text().strip() if Path("/proc/loadavg").exists() else "N/A"
    meminfo_lines = Path("/proc/meminfo").read_text().splitlines()[:4] if Path("/proc/meminfo").exists() else []
    _, daemon_pid = is_os_daemon_alive()

    obs_record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "organ_name": organ_name,
        "binary_path": str(bin_path),
        "binary_bytes": bin_path.stat().st_size if bin_path.exists() else 0,
        "status": status,
        "daemon_alive": alive,
        "daemon_pid": daemon_pid,
        "anim_diff_pixels": anim_diff_pixels,
        "return_code": rc,
        "execution_ms": round(dt_ms, 2),
        "fb_changed_pixels": changed_pixels,
        "fb_coverage_pct": fb_coverage_pct,
        "linux_loadavg": loadavg,
        "linux_meminfo_head": meminfo_lines,
        "stdout": stdout_str.strip()[:4000],
        "stderr": stderr_str.strip()[:2000],
    }

    with open(TELEMETRY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(obs_record, ensure_ascii=False) + "\n")

    render_aquarium_hud_html(obs_record, fb_after)
    return obs_record


def render_aquarium_hud_html(latest_obs: Dict[str, Any], fb_bytes: bytes) -> Path:
    """Renders the Tactical Aquarium Observatory HUD HTML with live /dev/fb0 PNG & telemetry history."""
    png_b64 = rgba_to_png_b64(fb_bytes, FB_WIDTH, FB_HEIGHT)
    history: List[Dict[str, Any]] = []
    if TELEMETRY_LOG.exists():
        for line in TELEMETRY_LOG.read_text(encoding="utf-8").splitlines()[-15:]:
            if line.strip():
                try:
                    history.append(json.loads(line))
                except Exception:
                    pass

    pal_db = json.loads(PALINDROME_DB.read_text(encoding="utf-8")) if PALINDROME_DB.exists() else {}
    src_code = ""
    organ_name = latest_obs.get("organ_name", "")
    for ext in (".rs", ".c"):
        cand = SRC_BAY_DIR / f"{organ_name}{ext}"
        if cand.exists():
            src_code = cand.read_text(encoding="utf-8")
            break

    payload_json = json.dumps(
        {
            "latest": latest_obs,
            "fb_png_b64": png_b64,
            "fb_width": FB_WIDTH,
            "fb_height": FB_HEIGHT,
            "history": list(reversed(history)),
            "palindrome_db": pal_db,
            "source_code": src_code,
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>🐟 LocalSleep-MoE 數位魚缸觀測艙 (SOMA-Fishbowl Observatory)</title>
<style>
:root{{--bg:#060911;--p:#0c1220;--b:#1c2b48;--c:#38bdf8;--g:#34d399;--y:#fbbf24;--r:#f87171;--t:#e2e8f0;--m:#94a3b8}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--t);font-family:'JetBrains Mono','Inter',monospace;padding:18px;line-height:1.5}}
header{{display:flex;justify-content:space-between;align-items:center;padding:14px 20px;background:var(--p);border:1px solid var(--b);border-radius:10px;margin-bottom:16px}}
h1{{font-size:18px;color:var(--c);letter-spacing:.5px}}
.badge{{padding:4px 10px;border-radius:99px;font-size:12px;font-weight:700;background:rgba(52,211,153,.15);color:var(--g);border:1px solid var(--g)}}
.grid{{display:grid;grid-template-columns:1.25fr 1fr;gap:16px}}
.card{{background:var(--p);border:1px solid var(--b);border-radius:10px;padding:16px;overflow:hidden}}
.card h2{{font-size:13px;color:var(--c);text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;border-bottom:1px solid var(--b);padding-bottom:6px;display:flex;justify-content:space-between}}
.fb-frame{{background:#020408;border:2px solid var(--c);border-radius:8px;padding:6px;box-shadow:0 0 28px rgba(56,189,248,.18);text-align:center}}
.fb-frame img{{width:100%;max-width:640px;height:auto;image-rendering:pixelated;border-radius:4px;display:block;margin:0 auto}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}}
.kpi{{background:#080d18;border:1px solid var(--b);border-radius:8px;padding:10px}}
.kpi .k{{font-size:11px;color:var(--m)}}
.kpi .v{{font-size:16px;font-weight:700;color:var(--t);margin-top:3px}}
pre{{background:#050811;border:1px solid var(--b);border-radius:6px;padding:10px;font-size:12px;color:#cbd5e1;overflow-x:auto;max-height:260px;white-space:pre-wrap}}
.rule{{padding:6px 10px;border-left:3px solid var(--y);background:rgba(251,191,36,.07);font-size:12px;margin-bottom:6px;border-radius:0 6px 6px 0}}
table{{width:100%;border-collapse:collapse;font-size:12px}}
th,td{{padding:7px 8px;border-bottom:1px solid var(--b);text-align:left}}
th{{color:var(--c)}}
</style>
</head>
<body>
<header>
  <div>
    <h1>🐟 SOMA-Fishbowl 數位魚缸觀測艙 — LocalSleep-MoE Linux Organ Sandbox</h1>
    <div style="font-size:12px;color:var(--m);margin-top:3px">Virtual Framebuffer: <code>/dev/fb0 (640×360 RGBA8888)</code> ｜ Kernel Vitals: <code>/proc/stat &amp; /proc/meminfo</code> ｜ Anti-Echo Palindrome Guard</div>
  </div>
  <div id="statusBadge" class="badge">HEALTHY</div>
</header>

<div class="grid">
  <div class="card">
    <h2><span>👁️ 魚缸玻璃：即時 Framebuffer (/dev/fb0) 視覺輸出</span><span id="fbRes">640×360 RGBA</span></h2>
    <div class="fb-frame">
      <img id="fbImg" alt="Fishbowl Framebuffer" />
    </div>
    <div class="kpis">
      <div class="kpi"><div class="k">當前投放器官 (Organ)</div><div class="v" id="kOrgan">-</div></div>
      <div class="kpi"><div class="k">執行耗時 (Tick Latency)</div><div class="v" id="kTime">-</div></div>
      <div class="kpi"><div class="k">Framebuffer 像素渲染率</div><div class="v" id="kPixels" style="color:var(--g)">-</div></div>
      <div class="kpi"><div class="k">魚缸 Linux LoadAvg</div><div class="v" id="kLoad">-</div></div>
    </div>
    <h2 style="margin-top:16px"><span>📟 魚缸內標準輸出 (Organ Stdout / Telemetry)</span></h2>
    <pre id="stdoutBox"></pre>
  </div>

  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="card">
      <h2><span>🪞 負面迴文免疫庫 (Negative-Palindrome Anti-Echo Constraints)</span></h2>
      <div id="rulesBox"></div>
    </div>
    <div class="card">
      <h2><span>🦀 智能體編譯投放之源碼 (Organ Source Code)</span><span id="binSize"></span></h2>
      <pre id="srcBox"></pre>
    </div>
    <div class="card">
      <h2><span>📜 魚缸歷次投放觀測紀錄 (Observation Log)</span></h2>
      <table>
        <thead><tr><th>時間</th><th>器官名稱</th><th>狀態</th><th>耗時</th><th>FB 渲染</th></tr></thead>
        <tbody id="histBody"></tbody>
      </table>
    </div>
  </div>
</div>

<script>
const D = {payload_json};
document.getElementById('fbImg').src = 'data:image/png;base64,' + D.fb_png_b64;
const L = D.latest || {{}};
document.getElementById('kOrgan').textContent = L.organ_name || 'N/A';
document.getElementById('kTime').textContent = (L.execution_ms || 0) + ' ms';
document.getElementById('kPixels').textContent = (L.fb_coverage_pct || 0) + '% (' + (L.fb_changed_pixels || 0).toLocaleString() + ' px)';
document.getElementById('kLoad').textContent = (L.linux_loadavg || '').split(' ').slice(0,3).join(' ');
document.getElementById('stdoutBox').textContent = (L.stdout || '(no stdout)') + (L.stderr ? '\\n[STDERR]:\\n' + L.stderr : '');
document.getElementById('binSize').textContent = L.binary_bytes ? (L.binary_bytes.toLocaleString() + ' bytes ELF') : '';
document.getElementById('srcBox').textContent = D.source_code || '(Binary drop)';
const sb = document.getElementById('statusBadge');
sb.textContent = L.status || 'READY';
if ((L.status || '') !== 'HEALTHY') {{
  sb.style.color = '#f87171';
  sb.style.borderColor = '#f87171';
}}
const rb = document.getElementById('rulesBox');
(D.palindrome_db.ActiveNegativeConstraints || []).forEach(r => {{
  const d = document.createElement('div');
  d.className = 'rule';
  d.textContent = '⛔ ' + r;
  rb.appendChild(d);
}});
const hb = document.getElementById('histBody');
(D.history || []).forEach(h => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${{(h.timestamp||'').split(' ')[1]||''}}</td><td><code>${{h.organ_name}}</code></td><td>${{h.status}}</td><td>${{h.execution_ms}}ms</td><td>${{h.fb_coverage_pct}}%</td>`;
  hb.appendChild(tr);
}});
</script>
</body>
</html>"""
    HUD_HTML_OUT.write_text(html, encoding="utf-8")
    return HUD_HTML_OUT


def extract_code_block(raw: str, lang: str = "rust") -> str:
    """Extracts clean Rust or C source code from LLM markdown output."""
    m = re.search(rf"```(?:{lang}|rs|c|cpp)?\s*\n([\s\S]*?)```", raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: if it starts with use std or #include or fn main
    if "fn main" in raw or "#include" in raw:
        return raw.strip()
    return raw.strip()


def ai_forge_and_drop(
    organ_name: str,
    task_prompt: str,
    lang: str = "rust",
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = "localsleep-moe",
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Has `rust_localsleep-moe` (:18935) write a system organ in Rust/C (guarded by Negative Palindrome
    constraints), compiles it, repairs any compiler error automatically, drops it into the Fishbowl,
    and returns the observation result!
    """
    ensure_fishbowl_initialized()
    ext = ".rs" if lang == "rust" else ".c"
    src_path = SRC_BAY_DIR / f"{organ_name}{ext}"

    for attempt in range(1, max_attempts + 1):
        pal_db = json.loads(PALINDROME_DB.read_text(encoding="utf-8"))
        neg_rules = "\n".join(f"- {r}" for r in pal_db.get("ActiveNegativeConstraints", []))
        recent_errs = "\n".join(
            f"- {e['error']}" for e in pal_db.get("CompileErrors", [])[-2:]
        )

        sys_p = (
            f"You are SOMA-Architect, a system-level {lang.upper()} programmer writing an executable organ for the Linux Fishbowl sandbox.\n"
            "Output ONLY valid, self-contained source code inside a single ```rust ... ``` (or ```c ... ```) code block. Zero prose outside the code block.\n"
            "【NEGATIVE PALINDROME CONSTRAINTS (MUST OBEY)】:\n"
            f"{neg_rules}\n"
        )
        usr_p = (
            f"【NEGATIVE PALINDROME HEAD LOCK [-C]】:\n"
            f"Do NOT use external crates (only Rust `std::*`!). Ensure all integer casts when indexing pixel buffers use `as usize`.\n"
            f"{('Recent compiler errors to avoid:\\n' + recent_errs) if recent_errs else ''}\n"
            f"========================================\n"
            f"【FISHBOWL HARDWARE SPECIFICATION】:\n"
            f"- OS: Linux (`/proc/loadavg`, `/proc/meminfo`, `/proc/stat`, `/proc/uptime` are readable).\n"
            f"- Framebuffer Device: Path is in env `FISHBOWL_FB_PATH` (default `./dev/fb0_rgba.raw`).\n"
            f"- Resolution: `640` width × `360` height × `4` bytes per pixel (RGBA8888, total `921,600` bytes).\n"
            f"- Task for Organ `{organ_name}`:\n{task_prompt}\n"
            f"========================================\n"
            f"【NEGATIVE PALINDROME TAIL LOCK [-C^R]】:\n"
            f"Write complete, compilable `{lang}` code with `fn main()` that reads Linux `/proc`, renders vivid visual graphics into the 640x360 RGBA framebuffer file, prints a structured telemetry report to stdout, and exits cleanly with code 0!"
        )

        print(f"🧠 [Attempt {attempt}/{max_attempts}] Requesting `{organ_name}` ({lang}) from {endpoint}...", flush=True)
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p},
            ],
            "max_tokens": 8192,
            "temperature": 0.35,
            "stream": False,
            "reset_kv_cache": True,
            "stateless_forge": True,
        }
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
        raw_out = resp_data["choices"][0]["message"]["content"]
        code_str = extract_code_block(raw_out, lang=lang)
        src_path.write_text(code_str, encoding="utf-8")

        ok, bin_path, compile_msg = compile_source_into_organ(src_path, organ_name)
        print(f"   🔨 {compile_msg}", flush=True)
        if ok and bin_path:
            obs = run_organ_in_fishbowl(bin_path, organ_name)
            print(
                f"   🐟 [Fishbowl Observation] status={obs['status']} | time={obs['execution_ms']}ms | "
                f"FB rendered={obs['fb_coverage_pct']}% ({obs['fb_changed_pixels']:,} px) -> {HUD_HTML_OUT}",
                flush=True,
            )
            return obs

    raise RuntimeError(f"Failed to compile organ `{organ_name}` after {max_attempts} attempts.")


def create_builtin_seed_organ() -> Path:
    """Creates a default zero-dependency Rust seed organ (`soma_seed_pulse.rs`) to verify the Fishbowl immediately."""
    ensure_fishbowl_initialized()
    seed_rs = SRC_BAY_DIR / "soma_seed_pulse.rs"
    seed_rs.write_text(
        r'''use std::env;
use std::fs;

const W: usize = 640;
const H: usize = 360;

fn set_px(buf: &mut [u8], x: i32, y: i32, r: u8, g: u8, b: u8) {
    if x >= 0 && x < W as i32 && y >= 0 && y < H as i32 {
        let idx = ((y as usize) * W + (x as usize)) * 4;
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
        buf[idx + 3] = 255;
    }
}

fn main() {
    let fb_path = env::var("FISHBOWL_FB_PATH").unwrap_or_else(|_| "./dev/fb0_rgba.raw".to_string());
    let loadavg = fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "0.42 0.35 0.28 1/256 1".to_string());
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();

    let mut mem_total_kb: u64 = 16_000_000;
    let mut mem_avail_kb: u64 = 10_000_000;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            mem_total_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16_000_000);
        } else if line.starts_with("MemAvailable:") {
            mem_avail_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
        }
    }
    let used_ratio = 1.0 - (mem_avail_kb as f64 / mem_total_kb.max(1) as f64);

    let mut buf = vec![0u8; W * H * 4];

    // 1. Tactical Dark Background + Cybernetic Radar Grid
    for y in 0..H {
        for x in 0..W {
            let idx = (y * W + x) * 4;
            let grid = (x % 32 == 0) || (y % 32 == 0);
            buf[idx] = if grid { 18 } else { 7 };
            buf[idx + 1] = if grid { 36 } else { 12 };
            buf[idx + 2] = if grid { 64 } else { 24 };
            buf[idx + 3] = 255;
        }
    }

    // 2. Draw 16-Expert Neural Hypercube Nodes & Rings around center (320, 165)
    let cx = 320.0f64;
    let cy = 165.0f64;
    for r_ring in [55.0f64, 95.0f64, 130.0f64] {
        for deg in 0..720 {
            let rad = (deg as f64) * 0.5 * std::f64::consts::PI / 180.0;
            let px = (cx + r_ring * rad.cos()) as i32;
            let py = (cy + r_ring * rad.sin()) as i32;
            set_px(&mut buf, px, py, 30, 120, 180);
        }
    }

    // 3. Draw Sin/Cos Neural Waveform across the Framebuffer
    for x in 20..(W - 20) {
        let t = (x as f64) * 0.045;
        let wy1 = (cy + (t.sin() * 42.0) + ((t * 2.7).cos() * 18.0)) as i32;
        let wy2 = (cy + ((t * 1.3).cos() * 55.0 * used_ratio.max(0.25))) as i32;
        for dy in -2..=2 {
            set_px(&mut buf, x as i32, wy1 + dy, 56, 189, 248);  // Cyan wave
            set_px(&mut buf, x as i32, wy2 + dy, 52, 211, 153);  // Emerald wave
        }
    }

    // 4. Draw Memory Usage Bar at bottom (y: 315..340)
    let bar_x0 = 40;
    let bar_w = 560;
    let filled_w = ((bar_w as f64) * used_ratio.clamp(0.05, 1.0)) as usize;
    for y in 318..338 {
        for x in 0..bar_w {
            if x < filled_w {
                set_px(&mut buf, (bar_x0 + x) as i32, y, 251, 191, 36);
            } else {
                set_px(&mut buf, (bar_x0 + x) as i32, y, 25, 38, 58);
            }
        }
    }

    fs::write(&fb_path, &buf).expect("Failed to write framebuffer");
    println!("=== [SOMA-SEED-PULSE ORGAN TELEMETRY] ===");
    println!("Linux LoadAvg : {}", loadavg.trim());
    println!("Memory Used   : {:.1}% ({} MB / {} MB)", used_ratio * 100.0, (mem_total_kb - mem_avail_kb) / 1024, mem_total_kb / 1024);
    println!("Framebuffer   : Wrote {}x{} RGBA8888 ({} bytes) -> {}", W, H, buf.len(), fb_path);
}
''',
        encoding="utf-8",
    )
    return seed_rs


def create_and_compile_devmem_cli() -> Tuple[bool, Optional[Path], str]:
    """Creates and compiles `/bin/soma_devmem` inside the OS so bash/apps can peek & poke the 8 RAM channels."""
    ensure_fishbowl_initialized()
    src_p = OS_SRC_DIR / "soma_devmem.rs"
    src_p.write_text(
        r'''#![allow(unused_imports, unused_variables, dead_code, unused_mut)]
use std::env;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

const CHANNELS: usize = 8;
const CH_SIZE: usize = 8192;

fn parse_u64_auto(s: &str) -> u64 {
    let clean = s.trim().replace('_', "");
    if let Some(hex) = clean.strip_prefix("0x").or_else(|| clean.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        clean.parse::<u64>().unwrap_or(0)
    }
}

fn main() {
    let devmem_path = env::var("SOMA_DEVMEM_PATH")
        .unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("status");

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&devmem_path)
        .expect("Failed to open SOMA_DEVMEM_PATH");

    let mut arena = vec![0u8; CHANNELS * CH_SIZE];
    let _ = file.read(&mut arena);

    if cmd == "poke" && args.len() >= 4 {
        let ch = args[2].parse::<usize>().unwrap_or(4).min(CHANNELS - 1);
        let reg_val = parse_u64_auto(&args[3]);
        let tag = args.get(4).map(|s| s.as_str()).unwrap_or("CH_USER_POKE");
        let payload = if args.len() >= 6 { args[5..].join(" ") } else { String::from("OK") };

        let base = ch * CH_SIZE;
        arena[base..base + 4].copy_from_slice(b"SOMA");
        let seq = u32::from_le_bytes([arena[base + 4], arena[base + 5], arena[base + 6], arena[base + 7]]).wrapping_add(1);
        arena[base + 4..base + 8].copy_from_slice(&seq.to_le_bytes());
        arena[base + 8..base + 16].copy_from_slice(&reg_val.to_le_bytes());

        let mut tag_buf = [0u8; 32];
        let tb = tag.as_bytes();
        let tlen = tb.len().min(31);
        tag_buf[..tlen].copy_from_slice(&tb[..tlen]);
        arena[base + 16..base + 48].copy_from_slice(&tag_buf);

        for b in &mut arena[base + 48..base + CH_SIZE] {
            *b = 0;
        }
        let pb = payload.as_bytes();
        let plen = pb.len().min(CH_SIZE - 49);
        arena[base + 48..base + 48 + plen].copy_from_slice(&pb[..plen]);

        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&arena).unwrap();
        println!("[SOMA_DEVMEM POKE] CH{} (0x{:04X}) -> SEQ={} REG=0x{:016X} TAG='{}' PAYLOAD='{}'", ch, base, seq, reg_val, tag, payload);
    } else if cmd == "peek" && args.len() >= 3 {
        let ch = args[2].parse::<usize>().unwrap_or(0).min(CHANNELS - 1);
        let base = ch * CH_SIZE;
        let seq = u32::from_le_bytes([arena[base + 4], arena[base + 5], arena[base + 6], arena[base + 7]]);
        let reg = u64::from_le_bytes([
            arena[base + 8], arena[base + 9], arena[base + 10], arena[base + 11],
            arena[base + 12], arena[base + 13], arena[base + 14], arena[base + 15],
        ]);
        let tag_end = arena[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
        let tag = String::from_utf8_lossy(&arena[base + 16..base + 16 + tag_end]);
        let pay_end = arena[base + 48..base + 304].iter().position(|&b| b == 0).unwrap_or(256);
        let pay = String::from_utf8_lossy(&arena[base + 48..base + 48 + pay_end]);
        println!("CH{} @ 0x{:04X} | SEQ={} | REG=0x{:016X} | TAG={} | PAYLOAD={}", ch, base, seq, reg, tag, pay);
    } else {
        println!("=== [SOMA-OS /dev/mem 8-CHANNEL SHARED ARENA (64KB)] ===");
        for ch in 0..CHANNELS {
            let base = ch * CH_SIZE;
            let seq = u32::from_le_bytes([arena[base + 4], arena[base + 5], arena[base + 6], arena[base + 7]]);
            let reg = u64::from_le_bytes([
                arena[base + 8], arena[base + 9], arena[base + 10], arena[base + 11],
                arena[base + 12], arena[base + 13], arena[base + 14], arena[base + 15],
            ]);
            let tag_end = arena[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
            let tag = String::from_utf8_lossy(&arena[base + 16..base + 16 + tag_end]);
            let pay_end = arena[base + 48..base + 128].iter().position(|&b| b == 0).unwrap_or(80);
            let pay = String::from_utf8_lossy(&arena[base + 48..base + 48 + pay_end]);
            println!("  CH{} [0x{:04X}] SEQ={:<4} REG=0x{:016X} {:<18} | {}", ch, base, seq, reg, tag, pay);
        }
    }
}
''',
        encoding="utf-8",
    )
    return compile_source_into_organ(src_p, "soma_devmem")


def create_and_compile_compositor_v3() -> Tuple[bool, Optional[Path], str]:
    """
    Creates and compiles the Dynamic Config-Driven SOMA-OS v3.0 Compositor Daemon (`soma_compositor_v3`).
    - Hot-reloads `/etc/soma_ui_config.cfg` on every tick without recompiling Rust!
    - Displays all 24 CPU cores (`C00..C23`) in an 8x3 grid (monotonically locked >= 24 cores).
    - Reads & renders the 8 live `/dev/mem` channels (`CH0..CH7`) from `devmem_arena.bin` with activity flashes.
    - Renders the 16-LoRA Hypercube Router with configurable `lora_box_w=66` (zero text overflow).
    - Renders the Split Chronos Oscilloscope + Live OS Bash/App IPC Terminal (`CH3` & `CH4`).
    """
    ensure_fishbowl_initialized()
    create_and_compile_devmem_cli()
    comp_rs = SRC_BAY_DIR / "soma_compositor_v3.rs"
    comp_rs.write_text(
        r'''#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]
use std::env;
use std::fs;
use std::io::Write;
use std::thread;
use std::time::{Duration, Instant};

struct UiConfig {
    os_title: String,
    cpu_max_cores: usize,
    cpu_cols: usize,
    cpu_rows: usize,
    cpu_col_step: usize,
    cpu_row_step: usize,
    cpu_bar_w: usize,
    cpu_bar_max_h: usize,
    vmmu_show_devmem: usize,
    lora_box_w: usize,
    lora_box_h: usize,
    lora_col_step: usize,
    lora_row_step: usize,
    br_mode: String,
    osc_gain: f64,
    dock_text_y_offset: usize,
    tl_rgb: [u8; 3],
    tr_rgb: [u8; 3],
    bl_rgb: [u8; 3],
    br_rgb: [u8; 3],
}

impl UiConfig {
    fn load(path: &str) -> Self {
        let mut cfg = UiConfig {
            os_title: String::from("SOMA-OS V3.0 DYNAMIC KERNEL + DEVMEM BUS"),
            cpu_max_cores: 24,
            cpu_cols: 8,
            cpu_rows: 3,
            cpu_col_step: 35,
            cpu_row_step: 32,
            cpu_bar_w: 22,
            cpu_bar_max_h: 16,
            vmmu_show_devmem: 1,
            lora_box_w: 66,
            lora_box_h: 24,
            lora_col_step: 71,
            lora_row_step: 29,
            br_mode: String::from("SPLIT_OSC_BASH"),
            osc_gain: 22.0,
            dock_text_y_offset: 9,
            tl_rgb: [56, 189, 248],
            tr_rgb: [52, 211, 153],
            bl_rgb: [251, 191, 36],
            br_rgb: [244, 114, 182],
        };
        if let Ok(raw) = fs::read_to_string(path) {
            for line in raw.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    let key = k.trim();
                    let val = v.trim();
                    match key {
                        "os_title" => cfg.os_title = val.to_uppercase(),
                        "cpu_max_cores" => cfg.cpu_max_cores = val.parse::<usize>().unwrap_or(24).max(24),
                        "cpu_cols" => cfg.cpu_cols = val.parse::<usize>().unwrap_or(8).clamp(4, 12),
                        "cpu_rows" => cfg.cpu_rows = val.parse::<usize>().unwrap_or(3).clamp(2, 6),
                        "cpu_col_step" => cfg.cpu_col_step = val.parse::<usize>().unwrap_or(35).clamp(20, 60),
                        "cpu_row_step" => cfg.cpu_row_step = val.parse::<usize>().unwrap_or(32).clamp(20, 50),
                        "cpu_bar_w" => cfg.cpu_bar_w = val.parse::<usize>().unwrap_or(22).clamp(10, 34),
                        "cpu_bar_max_h" => cfg.cpu_bar_max_h = val.parse::<usize>().unwrap_or(16).clamp(8, 28),
                        "vmmu_show_devmem" => cfg.vmmu_show_devmem = val.parse::<usize>().unwrap_or(1),
                        "lora_box_w" => cfg.lora_box_w = val.parse::<usize>().unwrap_or(66).clamp(50, 72),
                        "lora_box_h" => cfg.lora_box_h = val.parse::<usize>().unwrap_or(24).clamp(16, 30),
                        "lora_col_step" => cfg.lora_col_step = val.parse::<usize>().unwrap_or(71).clamp(56, 74),
                        "lora_row_step" => cfg.lora_row_step = val.parse::<usize>().unwrap_or(29).clamp(20, 34),
                        "br_mode" => cfg.br_mode = val.to_uppercase(),
                        "osc_gain" => cfg.osc_gain = val.parse::<f64>().unwrap_or(22.0).clamp(6.0, 38.0),
                        "dock_text_y_offset" => cfg.dock_text_y_offset = val.parse::<usize>().unwrap_or(9).clamp(4, 16),
                        "tl_border_r" => cfg.tl_rgb[0] = val.parse::<u8>().unwrap_or(56),
                        "tl_border_g" => cfg.tl_rgb[1] = val.parse::<u8>().unwrap_or(189),
                        "tl_border_b" => cfg.tl_rgb[2] = val.parse::<u8>().unwrap_or(248),
                        "tr_border_r" => cfg.tr_rgb[0] = val.parse::<u8>().unwrap_or(52),
                        "tr_border_g" => cfg.tr_rgb[1] = val.parse::<u8>().unwrap_or(211),
                        "tr_border_b" => cfg.tr_rgb[2] = val.parse::<u8>().unwrap_or(153),
                        "bl_border_r" => cfg.bl_rgb[0] = val.parse::<u8>().unwrap_or(251),
                        "bl_border_g" => cfg.bl_rgb[1] = val.parse::<u8>().unwrap_or(191),
                        "bl_border_b" => cfg.bl_rgb[2] = val.parse::<u8>().unwrap_or(36),
                        "br_border_r" => cfg.br_rgb[0] = val.parse::<u8>().unwrap_or(244),
                        "br_border_g" => cfg.br_rgb[1] = val.parse::<u8>().unwrap_or(114),
                        "br_border_b" => cfg.br_rgb[2] = val.parse::<u8>().unwrap_or(182),
                        _ => {}
                    }
                }
            }
        }
        cfg
    }
}

#[derive(Clone)]
struct DevMemChannel {
    ch: usize,
    seq: u32,
    reg_u64: u64,
    tag: String,
    payload: String,
}

fn read_devmem_channels(path: &str) -> Vec<DevMemChannel> {
    let mut out = Vec::with_capacity(8);
    let data = fs::read(path).unwrap_or_else(|_| vec![0u8; 65536]);
    for ch in 0..8 {
        let base = ch * 8192;
        if base + 256 <= data.len() {
            let seq = u32::from_le_bytes([data[base + 4], data[base + 5], data[base + 6], data[base + 7]]);
            let reg_u64 = u64::from_le_bytes([
                data[base + 8], data[base + 9], data[base + 10], data[base + 11],
                data[base + 12], data[base + 13], data[base + 14], data[base + 15],
            ]);
            let tag_end = data[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
            let tag = String::from_utf8_lossy(&data[base + 16..base + 16 + tag_end]).to_uppercase();
            let pay_end = data[base + 48..base + 240].iter().position(|&b| b == 0).unwrap_or(192);
            let payload = String::from_utf8_lossy(&data[base + 48..base + 48 + pay_end]).to_uppercase();
            out.push(DevMemChannel { ch, seq, reg_u64, tag, payload });
        } else {
            out.push(DevMemChannel { ch, seq: 0, reg_u64: 0, tag: format!("CH{}", ch), payload: String::new() });
        }
    }
    out
}

struct SysStats {
    load_1m: f64,
    load_5m: f64,
    mem_total_mb: u64,
    mem_pct: f64,
    uptime: u64,
    cpu_pcts: Vec<f64>,
}

fn get_stats(prev_totals: &mut Vec<u64>, prev_idles: &mut Vec<u64>) -> SysStats {
    let load_str = fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let loads: Vec<&str> = load_str.split_whitespace().collect();
    let l1 = loads.first().unwrap_or(&"0.5").parse::<f64>().unwrap_or(0.5);
    let l5 = loads.get(1).unwrap_or(&"0.4").parse::<f64>().unwrap_or(0.4);

    let mem_str = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total: u64 = 0;
    let mut avail: u64 = 0;
    for line in mem_str.lines() {
        if line.starts_with("MemTotal:") {
            total = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        } else if line.starts_with("MemAvailable:") {
            avail = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        }
    }
    let mem_pct = if total > 0 { (1.0 - (avail as f64 / total as f64)) * 100.0 } else { 0.0 };

    let up_str = fs::read_to_string("/proc/uptime").unwrap_or_default();
    let uptime = up_str.split_whitespace().next().unwrap_or("0").parse::<f64>().unwrap_or(0.0) as u64;

    let stat_str = fs::read_to_string("/proc/stat").unwrap_or_default();
    let mut cur_totals = Vec::new();
    let mut cur_idles = Vec::new();
    for line in stat_str.lines() {
        if line.starts_with("cpu") && !line.starts_with("cpu ") {
            let parts: Vec<u64> = line.split_whitespace().skip(1).map(|s| s.parse().unwrap_or(0)).collect();
            let tot: u64 = parts.iter().sum();
            let idle: u64 = parts.get(3).copied().unwrap_or(0) + parts.get(4).copied().unwrap_or(0);
            cur_totals.push(tot);
            cur_idles.push(idle);
        }
    }

    let mut pcts = Vec::new();
    for i in 0..cur_totals.len() {
        if i < prev_totals.len() {
            let dt = cur_totals[i].saturating_sub(prev_totals[i]);
            let di = cur_idles[i].saturating_sub(prev_idles[i]);
            let busy = dt.saturating_sub(di);
            let p = if dt > 0 { (busy as f64 / dt as f64) * 100.0 } else { 8.0 };
            pcts.push(p);
        } else {
            pcts.push(12.0);
        }
    }
    *prev_totals = cur_totals;
    *prev_idles = cur_idles;
    while pcts.len() < 24 {
        pcts.push(10.0);
    }

    SysStats {
        load_1m: l1,
        load_5m: l5,
        mem_total_mb: total / 1024,
        mem_pct,
        uptime,
        cpu_pcts: pcts,
    }
}

fn main() {
    let fb_path = env::var("FISHBOWL_FB_PATH").unwrap_or_else(|_| "./dev/fb0_rgba.raw".to_string());
    let devmem_path = env::var("SOMA_DEVMEM_PATH").unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
    let cfg_path = env::var("SOMA_UI_CFG_PATH").unwrap_or_else(|_| "./etc/soma_ui_config.cfg".to_string());

    let mut buf = vec![0u8; 1024 * 768 * 4];
    let mut prev_totals: Vec<u64> = Vec::new();
    let mut prev_idles: Vec<u64> = Vec::new();
    let mut prev_seqs: [u32; 8] = [0; 8];
    let mut flash_ttl: [u8; 8] = [0; 8];
    let mut tick: u64 = 0;

    println!("SOMA-OS DYNAMIC COMPOSITOR v3.0 (1024x768 AI-NATIVE SUBSTRATE) ONLINE | FB={} | DEVMEM={} | CFG={}", fb_path, devmem_path, cfg_path);

    loop {
        tick += 1;
        let t0 = Instant::now();
        let cfg = UiConfig::load(&cfg_path);
        let stats = get_stats(&mut prev_totals, &mut prev_idles);
        let channels = read_devmem_channels(&devmem_path);

        for ch in 0..8 {
            if channels[ch].seq != prev_seqs[ch] {
                prev_seqs[ch] = channels[ch].seq;
                flash_ttl[ch] = 10;
            } else if flash_ttl[ch] > 0 {
                flash_ttl[ch] -= 1;
            }
        }

        // Clear background (1024x768)
        for i in (0..buf.len()).step_by(4) {
            buf[i] = 5;
            buf[i + 1] = 8;
            buf[i + 2] = 16;
            buf[i + 3] = 255;
        }

        // --- TOP AI-NATIVE MACHINE STATE BARCODE & HEADER (y: 0..42, w: 1024) ---
        soma_fill_rect(&mut buf, 0, 0, 1024, 42, [10, 18, 34]);
        soma_fill_rect(&mut buf, 0, 41, 1024, 1, [40, 80, 135]);
        // Machine-readable spatial state cells across top right (for VLM & AI direct state decoding without OCR)
        for bit in 0..32 {
            let active = ((tick >> (bit % 8)) & 1) == 1 || (bit < 8 && flash_ttl[bit] > 0);
            let c = if active { [52, 211, 153] } else { [22, 38, 62] };
            soma_fill_rect(&mut buf, 620 + bit * 12, 10, 9, 22, c);
        }
        let top_txt = format!("{} | 1024X768 | TICK:{} | UP:{}S", cfg.os_title, tick, stats.uptime);
        soma_draw_text(&mut buf, 14, 14, &top_txt, [215, 235, 255], 1);

        // --- QUADRANT TL (x:12, y:52, w:492, h:332): 24-CORE SPATIAL TENSOR MATRIX (8x3) ---
        soma_draw_window(&mut buf, 12, 52, 492, 332, "TL: 24-CORE TOPOLOGY TENSOR [8X3 LOCKED]", cfg.tl_rgb);
        let total_cores = cfg.cpu_max_cores.min(24);
        for i in 0..total_cores {
            let col = i % 8;
            let row = i / 8;
            let cx = 28 + col * 58;
            let cy = 84 + row * 94;
            let raw_pct = stats.cpu_pcts.get(i).copied().unwrap_or(10.0);
            let wave = (((tick as f64 * 0.25) + (i as f64 * 0.55)).sin() + 1.0) * 16.0;
            let eff_pct = (raw_pct + wave).clamp(8.0, 100.0);
            let bar_h = ((eff_pct / 100.0) * 62.0).round() as usize;
            let bar_h = bar_h.clamp(4, 62);

            // Outer core slot frame
            soma_fill_rect(&mut buf, cx, cy, 46, 74, [16, 26, 44]);
            let bar_rgb = if eff_pct > 70.0 {
                [248, 113, 113]
            } else if eff_pct > 42.0 {
                [251, 191, 36]
            } else {
                cfg.tl_rgb
            };
            soma_fill_rect(&mut buf, cx + 4, cy + 8 + (62 - bar_h), 38, bar_h, bar_rgb);
            soma_draw_text(&mut buf, cx + 10, cy + 2, &format!("C{:02}", i), [180, 200, 225], 1);
        }

        // --- QUADRANT TR (x:520, y:52, w:492, h:332): 64KB /dev/mem 8-CHANNEL CONTROL BUS ---
        soma_draw_window(&mut buf, 520, 52, 492, 332, "TR: DEVMEM 8-CHANNEL BUS [64KB MMAP CONTROL]", cfg.tr_rgb);
        // RAM Arena Bar
        soma_fill_rect(&mut buf, 536, 80, 460, 18, [20, 34, 52]);
        let used_w = ((stats.mem_pct / 100.0) * 460.0).round() as usize;
        soma_fill_rect(&mut buf, 536, 80, used_w.clamp(8, 460), 18, cfg.tr_rgb);

        // 8 Channels x 64-bit Bitplane Spatial Matrix (Pure post-text visual register encoding + hex)
        for ch in 0..8 {
            let ry = 108 + ch * 32;
            let cinfo = &channels[ch];
            let is_flashing = flash_ttl[ch] > 0;
            let row_bg = if is_flashing { [26, 78, 58] } else { [15, 25, 42] };
            soma_fill_rect(&mut buf, 536, ry, 460, 26, row_bg);
            let led_c = if is_flashing { [251, 191, 36] } else { cfg.tr_rgb };
            soma_fill_rect(&mut buf, 540, ry + 4, 8, 18, led_c);
            soma_draw_text(&mut buf, 554, ry + 9, &format!("CH{} S{:03}", ch, cinfo.seq % 1000), [235, 245, 255], 1);

            // Render 32-bit spatial bit-cells of the 64-bit channel register so AI reads binary state directly from pixels!
            for b in 0..32 {
                let bit_on = ((cinfo.reg_u64 >> b) & 1) == 1;
                let cell_c = if bit_on {
                    if is_flashing { [251, 191, 36] } else { [56, 189, 248] }
                } else {
                    [24, 38, 60]
                };
                soma_fill_rect(&mut buf, 620 + b * 8, ry + 5, 6, 16, cell_c);
            }
            let short_tag: String = cinfo.tag.chars().skip(4).take(16).collect();
            soma_draw_text(&mut buf, 885, ry + 9, &short_tag, led_c, 1);
        }

        // --- QUADRANT BL (x:12, y:396, w:492, h:322): 16-LORA HYPERCUBE & SYNAPTIC TOPOLOGY ---
        soma_draw_window(&mut buf, 12, 396, 492, 322, "BL: 16-LORA HYPERCUBE TOPOLOGY [CH2]", cfg.bl_rgb);
        let ch2_mask = channels[2].reg_u64;
        for i in 0..16 {
            let col = i % 4;
            let row = i / 4;
            let x = 28 + col * 116;
            let y = 428 + row * 68;
            let phase = (tick + (i as u64 * 3) + (ch2_mask & 0xF)) % 15;
            let (st_txt, bg_rgb, fg_rgb) = if phase >= 12 {
                ("FIRE", [239, 68, 68], [255, 255, 255])
            } else if phase >= 6 {
                ("ACTV", [245, 158, 11], [10, 15, 25])
            } else {
                ("NREM", [32, 48, 72], [170, 190, 215])
            };
            soma_fill_rect(&mut buf, x, y, 102, 54, bg_rgb);
            // Spatial energy bar inside each expert tile
            let e_w = ((phase as usize + 1) * 6).min(90);
            soma_fill_rect(&mut buf, x + 6, y + 34, e_w, 12, [12, 18, 30]);
            soma_draw_text(&mut buf, x + 10, y + 12, &format!("E{:02} {}", i, st_txt), fg_rgb, 1);
        }

        // --- QUADRANT BR (x:520, y:396, w:492, h:322): AI CONTROL BUS, OS APPS & BASH IPC [CH3/CH4] ---
        soma_draw_window(&mut buf, 520, 396, 492, 322, "BR: AI ENV CONTROL + OS BASH/APPS [CH3/CH4]", cfg.br_rgb);
        // Upper Zone: Real-time Syscall & Channel Phase Field (y: 424..550)
        soma_fill_rect(&mut buf, 536, 424, 460, 126, [8, 13, 24]);
        for x_off in 0..452 {
            let t = (tick as f64 * 0.16) + (x_off as f64 * 0.035);
            let y1 = (487.0 + (t.sin() * (cfg.osc_gain * 1.8))).round() as usize;
            let y2 = (487.0 + ((t * 1.4).cos() * (cfg.osc_gain * 1.3))).round() as usize;
            soma_fill_rect(&mut buf, 540 + x_off, y1.clamp(428, 544), 2, 3, [56, 189, 248]);
            soma_fill_rect(&mut buf, 540 + x_off, y2.clamp(428, 544), 2, 3, [52, 211, 153]);
        }
        // Lower Zone: Internal OS Control State Blocks & IPC Stream (y: 560..704)
        soma_fill_rect(&mut buf, 536, 560, 460, 144, [6, 15, 24]);
        for ch_idx in 3..8 {
            let by = 568 + (ch_idx - 3) * 26;
            let cinfo = &channels[ch_idx];
            let status_c = if flash_ttl[ch_idx] > 0 { [251, 191, 36] } else { [52, 211, 153] };
            soma_fill_rect(&mut buf, 544, by, 14, 18, status_c);
            let pay_snip: String = cinfo.payload.chars().take(66).collect();
            soma_draw_text(&mut buf, 566, by + 5, &format!("CH{}: {}", ch_idx, pay_snip), [220, 238, 255], 1);
        }

        // --- BOTTOM AI CONTROL DOCK (y: 728..768, w: 1024) ---
        soma_fill_rect(&mut buf, 0, 728, 1024, 40, [10, 18, 34]);
        soma_fill_rect(&mut buf, 0, 728, 1024, 1, [40, 80, 135]);
        let frame_us = t0.elapsed().as_micros();
        let dock_str = format!(
            "[SOMA-OS V3 1024X768] [24-CORE TENSOR:LOCKED] [DEVMEM:64KB 8-CH] [TOOLCHAIN:/BIN/RUSTC] [FRAME:{}US]",
            frame_us
        );
        soma_draw_text(&mut buf, 14, 728 + cfg.dock_text_y_offset + 6, &dock_str, [52, 211, 153], 1);

        if let Ok(mut f) = fs::OpenOptions::new().write(true).open(&fb_path) {
            let _ = f.write_all(&buf);
        }
        thread::sleep(Duration::from_millis(80));
    }
}

fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b00110],
        ')' => [0b01100, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01100],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
        _   => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    }
}

pub fn soma_draw_text(buf: &mut [u8], x0: usize, y0: usize, text: &str, rgb: [u8; 3], scale: usize) {
    let sc = scale.max(1);
    for (ci, ch) in text.chars().enumerate() {
        let rows = glyph_5x7(ch);
        let cx = x0 + ci * 6 * sc;
        for (ry, bits) in rows.iter().enumerate() {
            for rx in 0..5 {
                if (bits >> (4 - rx)) & 1 == 1 {
                    for sy in 0..sc {
                        for sx in 0..sc {
                            let px = cx + rx * sc + sx;
                            let py = y0 + ry * sc + sy;
                            if px < 1024 && py < 768 {
                                let idx = (py * 1024 + px) * 4;
                                buf[idx] = rgb[0];
                                buf[idx + 1] = rgb[1];
                                buf[idx + 2] = rgb[2];
                                buf[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn soma_fill_rect(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, rgb: [u8; 3]) {
    let x_end = (x + w).min(1024);
    let y_end = (y + h).min(768);
    for py in y..y_end {
        for px in x..x_end {
            let idx = (py * 1024 + px) * 4;
            buf[idx] = rgb[0];
            buf[idx + 1] = rgb[1];
            buf[idx + 2] = rgb[2];
            buf[idx + 3] = 255;
        }
    }
}

pub fn soma_draw_window(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, title: &str, border_rgb: [u8; 3]) {
    soma_fill_rect(buf, x, y, w, h, [11, 17, 30]);
    soma_fill_rect(buf, x, y, w, 20, [18, 30, 52]);
    soma_fill_rect(buf, x, y, w, 1, border_rgb);
    soma_fill_rect(buf, x, (y + 20).min(767), w, 1, [30, 48, 78]);
    soma_fill_rect(buf, x, (y + h.saturating_sub(1)).min(767), w, 1, border_rgb);
    soma_fill_rect(buf, x, y, 1, h, border_rgb);
    soma_fill_rect(buf, (x + w.saturating_sub(1)).min(1023), y, 1, h, border_rgb);
    soma_draw_text(buf, x + 8, y + 6, title, border_rgb, 1);
}
''',
        encoding="utf-8",
    )
    return compile_source_into_organ(comp_rs, "soma_compositor_v3")


def run_os_bash_command(cmd_str: str, timeout_sec: float = 20.0) -> Dict[str, Any]:
    """
    Executes a bash command inside the Fishbowl OS environment (with `/bin` and `/toolchain/bin` mounted in PATH)
    and automatically writes the output into DevMem Channel 3 (`CH3:BASH_IPC_RING`) so the OS screen displays it live!
    """
    ensure_fishbowl_initialized()
    env = os.environ.copy()
    env["FISHBOWL_FB_PATH"] = str(FB_FILE)
    env["SOMA_DEVMEM_PATH"] = str(DEVMEM_FILE)
    env["SOMA_UI_CFG_PATH"] = str(UI_CONFIG_CFG)
    env["SOMA_OS_BIN_DIR"] = str(OS_BIN_DIR)
    env["SOMA_OS_SRC_DIR"] = str(OS_SRC_DIR)
    env["PATH"] = f"{OS_BIN_DIR}:{OS_TOOLCHAIN_DIR}:{ORGAN_BAY_DIR}:" + env.get("PATH", "/usr/bin:/bin")

    t0 = time.time()
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd_str],
            cwd=str(ROOTFS_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        rc = proc.returncode
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
    except subprocess.TimeoutExpired:
        rc = -9
        out = ""
        err = f"[TIMEOUT after {timeout_sec}s]"
    dt_ms = round((time.time() - t0) * 1000.0, 2)

    combined = out if out else (err if err else "(ok)")
    first_line = combined.splitlines()[0][:64] if combined else "OK"
    summary_line = f"$ {cmd_str[:18]} -> {first_line}"
    poke_devmem_channel(3, reg_u64=(0 if rc == 0 else 0xBAD0_0000 | (rc & 0xFFFF)), tag="CH3:BASH_IPC_RING", payload=summary_line)

    res = {
        "ok": rc == 0,
        "command": cmd_str,
        "return_code": rc,
        "execution_ms": dt_ms,
        "stdout": out[:4000],
        "stderr": err[:2000],
        "ch3_ipc_summary": summary_line,
    }
    return res


def list_os_apps_and_processes() -> Dict[str, Any]:
    """Returns installed OS `/bin` apps, `/src` sources, mounted `/toolchain/bin` compilers, active daemon, and 8 DevMem channels."""
    ensure_fishbowl_initialized()
    alive, d_pid = is_os_daemon_alive()
    bin_apps = []
    if OS_BIN_DIR.exists():
        for p in sorted(OS_BIN_DIR.iterdir()):
            if p.is_file():
                bin_apps.append({"name": p.name, "size_bytes": p.stat().st_size, "path": f"/bin/{p.name}"})
    src_files = []
    if OS_SRC_DIR.exists():
        for p in sorted(OS_SRC_DIR.iterdir()):
            if p.is_file():
                src_files.append({"name": p.name, "size_bytes": p.stat().st_size, "path": f"/src/{p.name}"})
    toolchains = []
    if OS_TOOLCHAIN_DIR.exists():
        for p in sorted(OS_TOOLCHAIN_DIR.iterdir()):
            toolchains.append({"name": p.name, "target": str(p.resolve()) if p.exists() else "broken"})

    return {
        "ok": True,
        "daemon_alive": alive,
        "daemon_pid": d_pid,
        "ui_config": get_ui_config(),
        "os_bin_apps": bin_apps,
        "os_src_files": src_files,
        "mounted_toolchains": toolchains,
        "devmem_channels": get_all_devmem_channels(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalSleep-MoE Linux Fishbowl Sandbox & Organ Observatory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Initialize the Linux Fishbowl environment and run the built-in Rust seed organ")
    sub.add_parser("status", help="Show Fishbowl configuration, Negative-Palindrome DB, and recent observations")
    sub.add_parser("reset", help="Restore Fishbowl Framebuffer to LKG snapshot")

    p_drop = sub.add_parser("drop", help="Compile a .rs or .c file and drop it into the Fishbowl to observe")
    p_drop.add_argument("--source", required=True, help="Path to .rs or .c source file")
    p_drop.add_argument("--name", default="custom_organ")
    p_drop.add_argument("--timeout", type=float, default=5.0)

    p_ai = sub.add_parser("ai-forge", help="Have :18935 AI write, compile, and drop a new organ into the Fishbowl")
    p_ai.add_argument("--name", default="ai_cyber_organ")
    p_ai.add_argument(
        "--task",
        default=(
            "Read `/proc/loadavg` and `/proc/uptime`, then render a multi-layered Ghost-in-the-Shell style "
            "cybernetic telemetry HUD on the 640x360 RGBA framebuffer (draw 16 glowing LoRA expert nodes in a "
            "4x4 matrix, connecting energy lines, and a CPU load waveform), and print the system vitals to stdout."
        ),
    )
    p_ai.add_argument("--lang", choices=["rust", "c"], default="rust")
    p_ai.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    p_ai.add_argument("--model", default="localsleep-moe")

    args = parser.parse_args()

    if args.cmd == "init":
        info = ensure_fishbowl_initialized()
        seed_rs = create_builtin_seed_organ()
        ok, bin_p, msg = compile_source_into_organ(seed_rs, "soma_seed_pulse")
        print(f"🐟 Fishbowl initialized: {json.dumps(info, ensure_ascii=False, indent=2)}")
        print(f"🔨 Seed Organ: {msg}")
        if ok and bin_p:
            obs = run_organ_in_fishbowl(bin_p, "soma_seed_pulse")
            print(f"✅ Seed Observation: {json.dumps(obs, ensure_ascii=False, indent=2)}")
            print(f"🔭 Aquarium Glass HUD ready at: {HUD_HTML_OUT}")

    elif args.cmd == "status":
        info = ensure_fishbowl_initialized()
        pal = json.loads(PALINDROME_DB.read_text(encoding="utf-8"))
        print(json.dumps({"config": info, "negative_palindrome_db": pal}, ensure_ascii=False, indent=2))

    elif args.cmd == "reset":
        ensure_fishbowl_initialized()
        lkg = SNAPSHOT_DIR / "fb0_lkg.raw"
        if lkg.exists():
            FB_FILE.write_bytes(lkg.read_bytes())
        print("♻️ Fishbowl Framebuffer restored to LKG snapshot.")

    elif args.cmd == "drop":
        src = Path(args.source).resolve()
        ok, bin_p, msg = compile_source_into_organ(src, args.name)
        print(msg)
        if not ok or not bin_p:
            sys.exit(1)
        obs = run_organ_in_fishbowl(bin_p, args.name, timeout_sec=args.timeout)
        print(json.dumps(obs, ensure_ascii=False, indent=2))
        print(f"🔭 Aquarium Glass HUD updated: {HUD_HTML_OUT}")

    elif args.cmd == "ai-forge":
        obs = ai_forge_and_drop(
            organ_name=args.name,
            task_prompt=args.task,
            lang=args.lang,
            endpoint=args.endpoint,
            model=args.model,
        )
        print(json.dumps(obs, ensure_ascii=False, indent=2))
        print(f"🔭 Aquarium Glass HUD updated: {HUD_HTML_OUT}")


if __name__ == "__main__":
    main()
