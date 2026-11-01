#!/usr/bin/env python3
"""
snapshot_inspector.py - SafeTensors Embodied Checkpoint & Neural Homeostasis Inspector
======================================================================================
Parses and validates `rust_localsleep-moe` SafeTensors checkpoints (`./checkpoints/*.safetensors`):
  1. Reads Header Metadata (`session_id`, `profile`, `tokens_generated`, `timestamp`, `format`).
  2. Inspects Rust Zero-GIL Router State (`states`, `fatigue_scores`, `active_tokens`, `gate_weights`).
  3. Inspects 8 Dynamic LoRA Expert Matrices (`expert.0..7.lora_a` & `lora_b`):
     - Exact 2:4 Structural Sparsity % (verifies whether NREM Fisher pruning executed on each expert!)
     - L2 Norm & Frobenius Norm (`||A||_F`, `||B||_F`)
     - Inter-Expert Cosine Orthogonality Matrix (checks the RSI "Orthogonalization" health across experts!)
  4. Optional Snapshot Diff Mode (`--diff <older.safetensors> <newer.safetensors>`) to track synaptic evolution!
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors


STATE_MAP = {
    0: "🟢 ACTIVE     ",
    1: "🟡 FATIGUED   ",
    2: "💤 NREM_SLEEP ",
}


def compute_2_4_group_compliance(tensor: torch.Tensor) -> Tuple[float, float]:
    """
    Returns:
      - `zero_pct`: overall percentage of exact zeros in `tensor`
      - `valid_24_pct`: percentage of 4-element contiguous blocks that have >= 2 zeros (NVIDIA 2:4 hardware rule)
    """
    flat = tensor.detach().float().reshape(-1)
    numel = flat.numel()
    if numel == 0:
        return 0.0, 0.0
    zero_pct = float((flat == 0.0).sum().item()) / float(numel) * 100.0
    n_groups = numel // 4
    if n_groups == 0:
        return zero_pct, 0.0
    groups = flat[: n_groups * 4].view(n_groups, 4)
    zeros_per_group = (groups == 0.0).sum(dim=-1)
    valid_24_pct = float((zeros_per_group >= 2).sum().item()) / float(n_groups) * 100.0
    return zero_pct, valid_24_pct


def compute_expert_orthogonality(tensors: Dict[str, torch.Tensor], num_experts: int = 8) -> List[List[float]]:
    """
    Computes the pairwise Cosine Similarity matrix between flattened `lora_a` matrices of all experts.
    Ideal orthogonal experts have off-diagonal values near 0.0.
    """
    vecs = []
    for i in range(num_experts):
        k = f"expert.{i}.lora_a"
        if k in tensors:
            v = tensors[k].detach().float().reshape(-1)
            norm = torch.norm(v).clamp(min=1e-12)
            vecs.append(v / norm)
    if not vecs:
        return []
    mat = torch.stack(vecs, dim=0)  # [E, N]
    sim = torch.mm(mat, mat.t())
    return [[round(float(sim[i, j].item()), 4) for j in range(len(vecs))] for i in range(len(vecs))]


def inspect_checkpoint(st_path: str, show_orthogonality: bool = True) -> Dict[str, Any]:
    path_obj = Path(st_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {st_path}")

    meta: Dict[str, str] = {}
    with safe_open(str(path_obj), framework="pt") as f:
        meta = f.metadata() or {}

    tensors = load_safetensors(str(path_obj))

    gate_w = tensors.get("gate_weights", torch.empty(0)).float()
    fatigue = tensors.get("fatigue_scores", torch.empty(0)).float().tolist()
    active_toks = tensors.get("active_tokens", torch.empty(0)).long().tolist()
    states = tensors.get("states", torch.empty(0)).int().tolist()
    num_experts = len(fatigue)

    file_kb = path_obj.stat().st_size / 1024.0
    print("\n" + "=" * 92)
    print(f"🔬 \033[1;36m[LocalSleep-MoE SafeTensors Snapshot Inspector]\033[0m")
    print(f"📂 File      : \033[1;33m{path_obj}\033[0m ({file_kb:.1f} KB)")
    print(
        f"🏷️  Metadata  : Profile=\033[1;32m{meta.get('profile', 'unknown')}\033[0m | "
        f"Session=\033[1;35m{meta.get('session_id', 'unknown')}\033[0m | "
        f"Tokens=\033[1;36m{meta.get('tokens_generated', '0')}\033[0m | "
        f"Time={meta.get('timestamp', 'N/A')}"
    )
    if gate_w.numel() > 0 and num_experts > 0:
        inferred_hidden = gate_w.numel() // num_experts
        print(
            f"🧭 Router Gate: shape=[{num_experts}, {inferred_hidden}] "
            f"(L2 Norm={float(torch.norm(gate_w).item()):.4f}, Mean={float(gate_w.mean().item()):+.6f}, Std={float(gate_w.std().item()):.6f})"
        )
    print("=" * 92)

    print(
        f"{'Expert':<8} | {'State':<14} | {'Fatigue (Bar)':<27} | {'Tokens':>7} | "
        f"{'LoRA_A Shape':<13} | {'Zero% (2:4 Valid%)':<20} | {'||A||_F / ||B||_F'}"
    )
    print("-" * 92)

    expert_reports = []
    total_toks_routed = sum(active_toks) or 1
    for i in range(num_experts):
        st_code = states[i] if i < len(states) else 0
        st_str = STATE_MAP.get(st_code, f"UNKNOWN({st_code})")
        fat_val = fatigue[i] if i < len(fatigue) else 0.0
        tok_val = active_toks[i] if i < len(active_toks) else 0
        tok_share = tok_val / total_toks_routed * 100.0

        bar_fill = min(16, int(round(fat_val * 8.0)))
        bar = "█" * bar_fill + "░" * (16 - bar_fill)

        ka = f"expert.{i}.lora_a"
        kb = f"expert.{i}.lora_b"
        if ka in tensors and kb in tensors:
            ta = tensors[ka]
            tb = tensors[kb]
            zero_pct, valid_24 = compute_2_4_group_compliance(ta)
            norm_a = float(torch.norm(ta.float()).item())
            norm_b = float(torch.norm(tb.float()).item())
            shape_str = f"{list(ta.shape)}"
        else:
            zero_pct, valid_24, norm_a, norm_b = 0.0, 0.0, 0.0, 0.0
            shape_str = "N/A"

        print(
            f"Exp #{i:02d}  | {st_str} | {fat_val:5.3f} [{bar}] | "
            f"{tok_val:5d} ({tok_share:4.1f}%) | {shape_str:<13} | "
            f"{zero_pct:5.1f}% ({valid_24:5.1f}% 2:4) | {norm_a:6.3f} / {norm_b:6.3f}"
        )
        expert_reports.append({
            "expert_id": i,
            "state": st_code,
            "fatigue": round(fat_val, 4),
            "active_tokens": tok_val,
            "lora_a_zero_pct": round(zero_pct, 2),
            "lora_a_2_4_compliance_pct": round(valid_24, 2),
            "lora_a_frobenius_norm": round(norm_a, 4),
            "lora_b_frobenius_norm": round(norm_b, 4),
        })

    print("-" * 92)

    ortho = compute_expert_orthogonality(tensors, num_experts=num_experts)
    if show_orthogonality and ortho:
        off_diag = [abs(ortho[r][c]) for r in range(len(ortho)) for c in range(len(ortho)) if r != c]
        max_corr = max(off_diag) if off_diag else 0.0
        mean_corr = sum(off_diag) / len(off_diag) if off_diag else 0.0
        print(
            f"📐 \033[1;35m[RSI Orthogonality Check (LoRA_A Pairwise Cosine Similarity)]\033[0m "
            f"Mean |cos(Ei, Ej)| = {mean_corr:.4f} | Max |cos(Ei, Ej)| = {max_corr:.4f} "
            f"({'✅ High Orthogonality (No Collapse)' if max_corr < 0.25 else '⚠️ Correlated Experts'})"
        )

    return {
        "file": str(path_obj),
        "metadata": meta,
        "experts": expert_reports,
    }


def diff_checkpoints(path_old: str, path_new: str) -> None:
    t_old = load_safetensors(path_old)
    t_new = load_safetensors(path_new)
    with safe_open(path_old, framework="pt") as f1, safe_open(path_new, framework="pt") as f2:
        m_old = f1.metadata() or {}
        m_new = f2.metadata() or {}

    print("\n" + "=" * 92)
    print(f"🔄 \033[1;33m[Snapshot Evolution Diff (Temporal Delta)]\033[0m")
    print(f"  • OLD: {path_old} (Session={m_old.get('session_id')}, Time={m_old.get('timestamp')})")
    print(f"  • NEW: {path_new} (Session={m_new.get('session_id')}, Time={m_new.get('timestamp')})")
    print("-" * 92)

    f_old = t_old.get("fatigue_scores", torch.zeros(8)).float().tolist()
    f_new = t_new.get("fatigue_scores", torch.zeros(8)).float().tolist()
    tok_old = t_old.get("active_tokens", torch.zeros(8)).long().tolist()
    tok_new = t_new.get("active_tokens", torch.zeros(8)).long().tolist()
    st_old = t_old.get("states", torch.zeros(8)).int().tolist()
    st_new = t_new.get("states", torch.zeros(8)).int().tolist()

    for i in range(len(f_new)):
        ka = f"expert.{i}.lora_a"
        da_norm = 0.0
        z_old, z_new = 0.0, 0.0
        if ka in t_old and ka in t_new and t_old[ka].shape == t_new[ka].shape:
            da_norm = float(torch.norm(t_new[ka].float() - t_old[ka].float()).item())
            z_old, _ = compute_2_4_group_compliance(t_old[ka])
            z_new, _ = compute_2_4_group_compliance(t_new[ka])
        print(
            f"  Exp #{i:02d} | State: {st_old[i]} -> {st_new[i]} | "
            f"Fatigue: {f_old[i]:.3f} -> {f_new[i]:.3f} ({f_new[i]-f_old[i]:+.3f}) | "
            f"Tokens: {tok_old[i]} -> {tok_new[i]} ({tok_new[i]-tok_old[i]:+d}) | "
            f"Zero%: {z_old:.1f}% -> {z_new:.1f}% | ||ΔLoRA_A||_F = {da_norm:.4f}"
        )
    print("=" * 92 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and diff LocalSleep-MoE SafeTensors Snapshots")
    parser.add_argument("target", nargs="?", default="all_latest", help="Path to .safetensors file, profile name, or 'all_latest'")
    parser.add_argument("--diff", nargs=2, metavar=("OLD_ST", "NEW_ST"), help="Compare two snapshots to verify synaptic weight evolution")
    parser.add_argument("--history", type=str, default=None, help="Show chronological evolution for a profile (e.g., thor_31b or rtx5080_bolt_e4b_gguf)")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent / "checkpoints"

    if args.diff:
        diff_checkpoints(args.diff[0], args.diff[1])
        return

    if args.history:
        pattern = str(base_dir / f"checkpoint_{args.history}_*.safetensors")
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"[!] No historical checkpoints found for profile '{args.history}'")
            return
        print(f"📜 Found {len(files)} historical snapshots for `{args.history}`:")
        for f in files:
            print(f"   - {os.path.basename(f)}")
        if len(files) >= 2:
            diff_checkpoints(files[0], files[-1])
        inspect_checkpoint(files[-1])
        return

    if args.target == "all_latest":
        latest_files = sorted(glob.glob(str(base_dir / "latest_*.safetensors")))
        for lf in latest_files:
            inspect_checkpoint(lf)
    else:
        cand = Path(args.target)
        if not cand.exists():
            cand = base_dir / f"latest_{args.target}.safetensors"
        inspect_checkpoint(str(cand))


if __name__ == "__main__":
    main()
