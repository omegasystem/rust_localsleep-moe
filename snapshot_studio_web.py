#!/usr/bin/env python3
"""
snapshot_studio_web.py - Interactive Web UI for SafeTensors Synaptic Anatomy & 3D Attractor Studio
==================================================================================================
Default Port: 18930 (User-launched via `./start_snapshot_studio.sh` or `python3 snapshot_studio_web.py`)

Features:
  1. Single Snapshot Deep Inspector:
     - Select any `.safetensors` checkpoint from `./checkpoints/`.
     - Live 8-Expert Homeostasis Cards (State, Fatigue Gauge, Active Tokens, ||A||_F, ||B||_F, Zero%, 2:4 Valid%).
     - Interactive SVD Singular Value Spectrum (`sigma_1..sigma_32`) & Effective Rank (`eRank`).
     - Interactive 8x8 Pairwise Cosine Orthogonality Matrix (`cos(E_i, E_j)`).
     - Interactive Microscopic 2:4 Synaptic Weight Grid Viewer (select Expert `0..7`, Rank row `0..31`, Col offset `0..5344`).
  2. Dual-Snapshot Mathematical Diff Engine (`T1` vs `T2`):
     - Pick any two `.safetensors` checkpoints (`T1` and `T2`) and click Compare.
     - Computes exact Frobenius Delta `||dA||_F`, Conformal Attenuation `alpha_k`, Estimated SHY Sleep Steps `N_sleep`,
       2:4 Support Mask IoU %, Subspace Angle `theta (deg)`, and Kurtosis.
     - Renders side-by-side T1 vs T2 SVD spectra, Energy bar charts, and 2:4 Quad Heatmaps.
  3. Integrated 3D Phase-Space Attractor & Session Telemetry Player:
     - Directly embeds the classic `localsleep-moe` 3D 8-Wing Lorenz Attractor WebGL visualizer (`/player`)
       connected to `./sessions/*.jsonl`!
"""

import argparse
import glob
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
SESSIONS_DIR = BASE_DIR / "sessions"
LEGACY_PLAYER_HTML = BASE_DIR.parent / "localsleep-moe" / "src" / "telemetry" / "player.html"
UI_HTML_PATH = BASE_DIR / "snapshot_studio.html"


def effective_rank(s: np.ndarray) -> float:
    s_pos = s[s > 1e-12]
    if len(s_pos) == 0:
        return 0.0
    p = s_pos / np.sum(s_pos)
    h = -np.sum(p * np.log(p + 1e-15))
    return float(np.exp(h))


def compute_24_stats(arr: np.ndarray) -> Dict[str, float]:
    flat = arr.reshape(-1)
    numel = flat.size
    if numel == 0:
        return {"zero_pct": 0.0, "valid_24_pct": 0.0}
    zero_pct = float(np.sum(flat == 0.0)) / float(numel) * 100.0
    n_quads = numel // 4
    if n_quads == 0:
        return {"zero_pct": round(zero_pct, 2), "valid_24_pct": 0.0}
    quads = flat[: n_quads * 4].reshape(n_quads, 4)
    zeros_per_q = np.sum(quads == 0.0, axis=1)
    valid_24_pct = float(np.sum(zeros_per_q >= 2)) / float(n_quads) * 100.0
    return {"zero_pct": round(zero_pct, 2), "valid_24_pct": round(valid_24_pct, 2)}


def list_checkpoints() -> List[Dict[str, Any]]:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CKPT_DIR.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        meta = {}
        try:
            with safe_open(str(p), framework="pt") as f:
                meta = f.metadata() or {}
        except Exception:
            pass
        out.append({
            "filename": p.name,
            "size_kb": round(p.stat().st_size / 1024.0, 1),
            "mtime": p.stat().st_mtime,
            "profile": meta.get("profile", "unknown"),
            "session_id": meta.get("session_id", ""),
            "tokens_generated": meta.get("tokens_generated", "0"),
            "timestamp": meta.get("timestamp", ""),
        })
    return out


def analyze_single_checkpoint(filename: str, exp_idx: int = 2, col_offset: int = 0) -> Dict[str, Any]:
    target = (CKPT_DIR / Path(filename).name).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    with safe_open(str(target), framework="pt") as f:
        meta = f.metadata() or {}
    tensors = load_safetensors(str(target))

    gate_w = tensors.get("gate_weights", torch.empty(0)).float().numpy()
    fatigue = tensors.get("fatigue_scores", torch.empty(0)).float().numpy().tolist()
    active_toks = tensors.get("active_tokens", torch.empty(0)).long().numpy().tolist()
    states = tensors.get("states", torch.empty(0)).int().numpy().tolist()
    num_experts = len(fatigue)
    total_toks = sum(active_toks) or 1

    experts = []
    flat_normed = []
    for i in range(num_experts):
        ka = f"expert.{i}.lora_a"
        kb = f"expert.{i}.lora_b"
        a = tensors[ka].float().numpy() if ka in tensors else np.zeros((16, 2560), dtype=np.float32)
        b = tensors[kb].float().numpy() if kb in tensors else np.zeros((2560, 16), dtype=np.float32)

        fro_a = float(np.linalg.norm(a, "fro"))
        fro_b = float(np.linalg.norm(b, "fro"))
        s_vals = np.linalg.svd(a, compute_uv=False)
        erank = effective_rank(s_vals)
        sp_stats = compute_24_stats(a)

        nz = a[a != 0.0]
        kurt = float(np.mean((nz / (np.std(nz) + 1e-12)) ** 4)) if nz.size > 0 else 0.0

        # 64-bin spatial energy profile across hidden_dim
        h_dim = a.shape[1]
        n_bins = 64
        bin_w = max(1, h_dim // n_bins)
        trimmed = a[:, : n_bins * bin_w].reshape(a.shape[0], n_bins, bin_w)
        spatial_profile = [round(float(x), 5) for x in np.sqrt(np.mean(trimmed ** 2, axis=(0, 2)))]

        v = a.reshape(-1)
        v_norm = np.linalg.norm(v)
        flat_normed.append(v / max(1e-12, v_norm))

        experts.append({
            "id": i,
            "state": int(states[i]) if i < len(states) else 0,
            "fatigue": round(float(fatigue[i]), 4),
            "active_tokens": int(active_toks[i]),
            "token_share_pct": round(float(active_toks[i]) / total_toks * 100.0, 2),
            "shape": list(a.shape),
            "fro_a": round(fro_a, 4),
            "fro_b": round(fro_b, 4),
            "sigma_max": round(float(s_vals[0]) if s_vals.size > 0 else 0.0, 4),
            "sigma_min": round(float(s_vals[-1]) if s_vals.size > 0 else 0.0, 4),
            "erank": round(erank, 2),
            "svd_spectrum": [round(float(x), 5) for x in s_vals.tolist()],
            "zero_pct": sp_stats["zero_pct"],
            "valid_24_pct": sp_stats["valid_24_pct"],
            "kurtosis": round(kurt, 3),
            "spatial_profile": spatial_profile,
        })

    # Pairwise Cosine Orthogonality Matrix
    mat = np.stack(flat_normed, axis=0) if flat_normed else np.zeros((1, 1))
    ortho_mat = np.matmul(mat, mat.T)
    ortho_list = [[round(float(ortho_mat[r, c]), 4) for c in range(ortho_mat.shape[1])] for r in range(ortho_mat.shape[0])]

    # Extract microscopic 2:4 patch for selected expert & ALL 8 experts so UI can display all 8 simultaneously!
    safe_exp = max(0, min(num_experts - 1, int(exp_idx)))
    sel_a = tensors[f"expert.{safe_exp}.lora_a"].float().numpy()
    max_col = max(0, sel_a.shape[1] - 32)
    safe_col = max(0, min(max_col, (int(col_offset) // 4) * 4))
    patch = sel_a[: min(16, sel_a.shape[0]), safe_col : safe_col + 32]

    all_micro_patches = []
    for i in range(num_experts):
        ai = tensors[f"expert.{i}.lora_a"].float().numpy()
        pi = ai[: min(8, ai.shape[0]), safe_col : safe_col + 32]
        all_micro_patches.append({
            "expert_id": i,
            "fro_a": experts[i]["fro_a"],
            "fatigue": experts[i]["fatigue"],
            "state": experts[i]["state"],
            "zero_pct": experts[i]["zero_pct"],
            "grid": [[round(float(val), 6) for val in row] for row in pi],
        })

    return {
        "filename": target.name,
        "size_kb": round(target.stat().st_size / 1024.0, 1),
        "metadata": meta,
        "router_gate": {
            "num_experts": num_experts,
            "hidden_dim": int(gate_w.size // max(1, num_experts)),
            "l2_norm": round(float(np.linalg.norm(gate_w)), 4),
            "std": round(float(np.std(gate_w)), 6),
        },
        "experts": experts,
        "orthogonality_matrix": ortho_list,
        "micro_patch": {
            "expert_id": safe_exp,
            "col_offset": safe_col,
            "rows": patch.shape[0],
            "cols": patch.shape[1],
            "grid": [[round(float(val), 6) for val in row] for row in patch],
        },
        "all_micro_patches": all_micro_patches,
    }


def analyze_diff_checkpoints(file_t1: str, file_t2: str, exp_idx: int = 2, col_offset: int = 0) -> Dict[str, Any]:
    p1 = (CKPT_DIR / Path(file_t1).name).resolve()
    p2 = (CKPT_DIR / Path(file_t2).name).resolve()
    if not p1.exists() or not p2.exists():
        raise FileNotFoundError("One or both checkpoint files not found")

    s1 = analyze_single_checkpoint(p1.name, exp_idx=exp_idx, col_offset=col_offset)
    s2 = analyze_single_checkpoint(p2.name, exp_idx=exp_idx, col_offset=col_offset)

    t1 = load_safetensors(str(p1))
    t2 = load_safetensors(str(p2))

    diff_rows = []
    num_exp = min(len(s1["experts"]), len(s2["experts"]))
    for i in range(num_exp):
        ka = f"expert.{i}.lora_a"
        a1 = t1[ka].float().numpy()
        a2 = t2[ka].float().numpy()

        if a1.shape != a2.shape:
            diff_rows.append({
                "id": i,
                "shape_mismatch": True,
                "note": f"Shape mismatch: {list(a1.shape)} vs {list(a2.shape)}",
            })
            continue

        norm1 = float(np.linalg.norm(a1, "fro"))
        norm2 = float(np.linalg.norm(a2, "fro"))
        norm_diff = float(np.linalg.norm(a2 - a1, "fro"))
        cos_sim = float(np.sum(a1 * a2) / max(1e-12, norm1 * norm2))
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))

        nz1 = a1 != 0.0
        nz2 = a2 != 0.0
        inter = int(np.sum(nz1 & nz2))
        union = int(np.sum(nz1 | nz2))
        mask_iou = float(inter / max(1, union)) * 100.0
        newly_pruned = int(np.sum(nz1 & (~nz2)))

        surv = nz1 & nz2
        if np.any(surv):
            ratios = np.abs(a2[surv]) / np.maximum(1e-12, np.abs(a1[surv]))
            atten_mean = float(np.mean(ratios))
            atten_std = float(np.std(ratios))
        else:
            atten_mean = 0.0
            atten_std = 0.0

        est_steps = float(np.log(max(1e-6, atten_mean)) / np.log(0.985)) if atten_mean > 0 else 0.0

        e1 = s1["experts"][i]
        e2 = s2["experts"][i]
        diff_rows.append({
            "id": i,
            "st1": e1["state"],
            "st2": e2["state"],
            "fat1": e1["fatigue"],
            "fat2": e2["fatigue"],
            "dfat": round(e2["fatigue"] - e1["fatigue"], 4),
            "tok1": e1["active_tokens"],
            "tok2": e2["active_tokens"],
            "dtok": e2["active_tokens"] - e1["active_tokens"],
            "fro1": e1["fro_a"],
            "fro2": e2["fro_a"],
            "dfro": round(e2["fro_a"] - e1["fro_a"], 4),
            "norm_diff": round(norm_diff, 4),
            "atten_mean": round(atten_mean, 4),
            "atten_std": round(atten_std, 6),
            "est_shy_steps": round(est_steps, 1),
            "sigma1_t1": e1["sigma_max"],
            "sigma1_t2": e2["sigma_max"],
            "erank_t1": e1["erank"],
            "erank_t2": e2["erank"],
            "zero_t1": e1["zero_pct"],
            "zero_t2": e2["zero_pct"],
            "mask_iou_pct": round(mask_iou, 2),
            "newly_pruned": newly_pruned,
            "cos_sim": round(cos_sim, 6),
            "angle_deg": round(angle_deg, 4),
            "kurt_t1": e1["kurtosis"],
            "kurt_t2": e2["kurtosis"],
        })

    return {
        "t1": s1,
        "t2": s2,
        "diff": diff_rows,
    }


def analyze_profile_series(profile_name: str, exp_idx: int = 2, col_offset: int = 0) -> Dict[str, Any]:
    """
    Loads all chronological `checkpoint_<profile>_*.safetensors` snapshots for a profile
    so the Web UI can dynamically animate/play back the temporal evolution of:
      - 8-Expert Frobenius Norms `||A||_F` & Fatigue Scores
      - SVD Singular Value Spectrum (`sigma_1..sigma_32`) & Effective Rank (`eRank`)
      - 2:4 Structural Sparsity % & Microscopic 16x32 Synaptic Grid frames
      - Step-by-step Conformal Scaling Ratio `alpha_k` vs previous frame!
    """
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    all_ckpts = list_checkpoints()
    avail_profiles = sorted({c["profile"] for c in all_ckpts if c.get("profile") and c["profile"] != "unknown"})

    if not profile_name or profile_name == "auto":
        profile_name = "thor_31b" if "thor_31b" in avail_profiles else (avail_profiles[0] if avail_profiles else "thor_31b")

    matched = sorted(
        [
            p for p in CKPT_DIR.glob(f"checkpoint_{profile_name}_*.safetensors")
        ],
        key=lambda p: p.name,
    )
    if not matched:
        # Fallback: match by metadata profile
        for c in reversed(all_ckpts):
            if c["profile"] == profile_name and not c["filename"].startswith("latest_"):
                matched.append(CKPT_DIR / c["filename"])

    frames = []
    for idx, p in enumerate(matched):
        try:
            snap = analyze_single_checkpoint(p.name, exp_idx=exp_idx, col_offset=col_offset)
            snap["frame_index"] = idx
            frames.append(snap)
        except Exception:
            continue

    return {
        "profile": profile_name,
        "available_profiles": avail_profiles,
        "total_frames": len(frames),
        "frames": frames,
    }



def scan_telemetry_files() -> List[Path]:
    dirs = [
        SESSIONS_DIR,
        BASE_DIR.parent / "localsleep-moe" / "sessions",
        BASE_DIR.parent / "localsleep-moe",
    ]
    out: List[Path] = []
    for d in dirs:
        if d.exists():
            for pat in ("session_real_telemetry_*.jsonl", "session_moe_telemetry_*.jsonl", "session_rust_telemetry_*.jsonl"):
                for p in d.glob(pat):
                    if p.stat().st_size > 10 and p not in out:
                        out.append(p)
    out.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return out


class StudioRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, data: Any, status: int = 200) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path in ("/", "/index.html"):
            html = UI_HTML_PATH.read_text(encoding="utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if path == "/player":
            if LEGACY_PLAYER_HTML.exists():
                html = LEGACY_PLAYER_HTML.read_text(encoding="utf-8")
            else:
                html = "<h1>3D Player HTML not found</h1>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode("utf-8"))
            return

        if path == "/api/checkpoints":
            self._send_json({"checkpoints": list_checkpoints()})
            return

        if path == "/api/inspect":
            fname = (qs.get("file") or [""])[0]
            exp_idx = int((qs.get("expert") or ["2"])[0])
            col_off = int((qs.get("col") or ["0"])[0])
            try:
                if not fname:
                    ckpts = list_checkpoints()
                    fname = ckpts[0]["filename"] if ckpts else ""
                res = analyze_single_checkpoint(fname, exp_idx=exp_idx, col_offset=col_off)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/diff":
            f1 = (qs.get("t1") or [""])[0]
            f2 = (qs.get("t2") or [""])[0]
            exp_idx = int((qs.get("expert") or ["2"])[0])
            col_off = int((qs.get("col") or ["0"])[0])
            try:
                res = analyze_diff_checkpoints(f1, f2, exp_idx=exp_idx, col_offset=col_off)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/series":
            prof = (qs.get("profile") or ["thor_31b"])[0]
            exp_idx = int((qs.get("expert") or ["2"])[0])
            col_off = int((qs.get("col") or ["0"])[0])
            try:
                res = analyze_profile_series(prof, exp_idx=exp_idx, col_offset=col_off)
                self._send_json(res)
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)
            return

        # Endpoints for embedded 3D Attractor Player (`/api/files` & `/api/telemetry`)
        if path == "/api/files":
            tfiles = scan_telemetry_files()
            self._send_json({"files": [p.name for p in tfiles]})
            return

        if path == "/api/telemetry":
            req_name = (qs.get("file") or [""])[0]
            tfiles = scan_telemetry_files()
            target = None
            for p in tfiles:
                if p.name == req_name:
                    target = p
                    break
            if target is None and tfiles:
                target = tfiles[0]
            if target and target.exists():
                records = []
                for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
                    if line.strip():
                        try:
                            rec = json.loads(line)
                            # Normalize both `selected_experts`/`weights` and `routed_experts`/`routing_weights` for 3D Player
                            if "routed_experts" not in rec and "selected_experts" in rec:
                                rec["routed_experts"] = rec["selected_experts"]
                            if "routing_weights" not in rec and "weights" in rec:
                                rec["routing_weights"] = rec["weights"]
                            records.append(rec)
                        except Exception:
                            pass
                self._send_json(records)
            else:
                self._send_json([])
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="LocalSleep-MoE SafeTensors Snapshot Studio & 3D Attractor Web UI")
    parser.add_argument("--port", type=int, default=18930, help="HTTP server port (default: 18930)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    args = parser.parse_args()

    httpd = HTTPServer((args.host, args.port), StudioRequestHandler)
    print("=" * 88)
    print("🔬 \033[1;36mLocalSleep-MoE SafeTensors Synaptic Studio & 3D Attractor Workbench\033[0m")
    print(f"📂 Checkpoints Dir : \033[1;32m{CKPT_DIR}\033[0m ({len(list_checkpoints())} snapshots)")
    print(f"🌐 Studio Web URL  : \033[1;33mhttp://127.0.0.1:{args.port}\033[0m")
    print("=" * 88)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()


if __name__ == "__main__":
    main()
