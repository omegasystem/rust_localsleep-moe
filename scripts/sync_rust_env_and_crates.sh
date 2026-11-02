#!/usr/bin/env bash
# ==============================================================================
# sync_rust_env_and_crates.sh
# 同步 WSL Host 端與 Docker 端之 Rust 工具鏈、Maturin 橋接器與常用科學運算 Crate 快取
# ==============================================================================
set -euo pipefail
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1

PROJECT_ROOT="/home/node/.openclaw/workspace/projects/rust_localsleep-moe"
if [ ! -d "$PROJECT_ROOT" ] && [ -d "$HOME/.openclaw/workspace/projects/rust_localsleep-moe" ]; then
    PROJECT_ROOT="$HOME/.openclaw/workspace/projects/rust_localsleep-moe"
fi

echo "=== [1/4] 檢查並同步 Rust Toolchain (rustup / cargo / rustc) ==="
if ! command -v cargo &>/dev/null; then
    echo "⚠️ 未偵測到 cargo，正在透過 rustup 安裝穩定版 Rust 工具鏈..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
else
    echo "✅ 已安裝: $(rustc --version) | $(cargo --version)"
fi

if command -v rustup &>/dev/null; then
    rustup component add rustfmt clippy 2>/dev/null || true
fi

echo "=== [2/4] 檢查 Python <-> Rust PyO3 編譯橋接器 (maturin) ==="
if command -v python3 &>/dev/null; then
    python3 -m pip install --upgrade --quiet maturin setuptools-rust numpy 2>/dev/null || true
    echo "✅ maturin 檢查完成: $(maturin --version 2>/dev/null || echo 'installed in venv')"
fi

echo "=== [3/4] 更新 System Prompt 授權 Cargo.toml 自由擴充與護欄規則 ==="
if [ -d "$PROJECT_ROOT" ]; then
    cd "$PROJECT_ROOT"
    python3 - <<'PY'
import json
from pathlib import Path

CARGO_RULE = (
    "5. 【Cargo.toml 擴充授權與版本護欄】：允許隨時使用 edit_file 修改沙盒內的 Cargo.toml 新增純 Rust 依賴庫"
    "（目前已預載 rand='0.8', rayon='1.8', parking_lot='0.12', smallvec='1.13', rustc-hash='1.1', serde='1.0'）；"
    "但必須保持 pyo3='0.20'、numpy='0.20'、ndarray='0.15' 三件套版本配對，且嚴禁引入需外部 C/Fortran BLAS 的 ndarray-linalg / openblas-src！"
)

p_cfg = Path("configs/model_profiles.json")
if p_cfg.exists():
    cfg = json.loads(p_cfg.read_text(encoding="utf-8"))
    for pname, pdata in cfg.get("profiles", {}).items():
        sp = pdata.get("system_prompt")
        if isinstance(sp, str) and "【Cargo.toml 擴充授權與版本護欄】" not in sp:
            if "若需生成並儲存全新檔案" in sp:
                sp = sp.replace("若需生成並儲存全新檔案", f"{CARGO_RULE}\n若需生成並儲存全新檔案")
            else:
                sp = sp + "\n" + CARGO_RULE
            pdata["system_prompt"] = sp
    p_cfg.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("✅ 已將 Cargo.toml 授權與版本護欄寫入 configs/model_profiles.json")

for py_name in ("ls_moe_pyqt_gui.py", "ls_moe_agentic.py"):
    p_py = Path(py_name)
    if p_py.exists():
        txt = p_py.read_text(encoding="utf-8")
        if "【Cargo.toml 擴充授權與版本護欄】" not in txt and "【檔案覆蓋與局部修補鐵律】" in txt:
            txt = txt.replace(
                "否則容易出錯或遺漏既有方法！",
                f"否則容易出錯或遺漏既有方法！\\n{CARGO_RULE}",
                1,
            )
            p_py.write_text(txt, encoding="utf-8")
            print(f"✅ 已將 Cargo.toml 授權與版本護欄寫入 {py_name}")
PY
fi

echo "=== [4/4] 預熱沙盒常用 Rust Crates 快取 (cargo fetch & cargo check) ==="
SANDBOX_DIR="$PROJECT_ROOT/isolated_output/prj_20261030_220710_3f6f5c57"
if [ -d "$SANDBOX_DIR" ]; then
    cd "$SANDBOX_DIR"
    cargo fetch
    cargo check
    echo "🎉 沙盒 Cargo.toml 常用庫快取與預編譯全部就緒 (EXIT_CODE: 0)！"
fi
