#!/usr/bin/env python3
"""
export_recent_48_matrix.py - Export Recent 48 Checkpoints into Analyzable Weight-Change & Re-Ignition Matrices
==============================================================================================================
Loads the most recent N (default 48) `checkpoint_<profile>_*.safetensors` files and exports:
  1. `exports/recent_48_weight_matrices.npz` (NumPy compressed archive for direct Python/NumPy/Pandas analysis):
     - `lora_a_ patch_series`: [T, E, 32, 256] float32 raw weight slice across time
     - `fro_a_matrix`:        [T, E] Frobenius norm ||A_e(t)||_F
     - `delta_fro_matrix`:    [T-1, E] step-to-step weight change ||A_e(t+1) - A_e(t)||_F
     - `reignited_matrix`:    [T-1, E] count of dead synapses (0.0 at t) that re-ignited (!= 0.0 at t+1)
     - `extinguished_matrix`: [T-1, E] count of active synapses (!= 0.0 at t) that turned off (0.0 at t+1)
     - `mask_flip_pct`:       [T-1, E] percentage of 2:4 mask positions that flipped between t and t+1
     - `ever_reignited_per_expert`: [E] total unique synapses per expert that were 0.0 and later re-ignited
     - `conformal_alpha_matrix`:    [T-1, E] mean multiplicative ratio |A(t+1)| / |A(t)| on surviving synapses
     - `fatigue_matrix`:      [T, E] fatigue score per expert
     - `state_matrix`:        [T, E] expert state (0=Active, 1=Fatigued, 2=NREM_Sleep)
  2. `exports/recent_48_reignition_report.json` & `exports/recent_48_reignition_table.csv`:
     - Directly readable summary proving whether dead neurons (0.0) ever re-ignited or were locked in an absorbing state!
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from safetensors import safe_open

BASE_DIR = Path(__file__).resolve().parent
CKPT_DIR = BASE_DIR / "checkpoints"
EXPORT_DIR = BASE_DIR / "exports"


def export_recent_weight_matrices(
    profile: str = "thor_31b_16lora",
    num_frames: int = 48,
    sample_mode: str = "recent",
) -> Dict[str, Any]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    matched = sorted(
        list(CKPT_DIR.glob(f"checkpoint_{profile}_*.safetensors")),
        key=lambda p: p.name,
    )
    if not matched:
        raise FileNotFoundError(f"No checkpoints found for profile={profile} in {CKPT_DIR}")

    total_raw = len(matched)
    if num_frames > 0 and total_raw > num_frames:
        if sample_mode == "stride":
            idx = sorted(set(int(i) for i in np.linspace(0, total_raw - 1, num=num_frames, dtype=int)))
            selected_files = [matched[i] for i in idx]
        else:
            selected_files = matched[-num_frames:]
    else:
        selected_files = matched

    T = len(selected_files)
    # Peek first file to get E, R, C
    with safe_open(str(selected_files[0]), framework="pt") as f0:
        fat0 = f0.get_tensor("fatigue_scores").float().numpy()
        E = int(fat0.size)
        a0 = f0.get_tensor("expert.0.lora_a").float().numpy()
        R, C = a0.shape

    patch_cols = min(256, C)
    lora_a_patch_series = np.zeros((T, E, R, patch_cols), dtype=np.float32)
    fro_a_matrix = np.zeros((T, E), dtype=np.float32)
    fatigue_matrix = np.zeros((T, E), dtype=np.float32)
    state_matrix = np.zeros((T, E), dtype=np.int32)
    tokens_matrix = np.zeros((T, E), dtype=np.int64)
    zero_pct_matrix = np.zeros((T, E), dtype=np.float32)

    delta_fro_matrix = np.zeros((max(0, T - 1), E), dtype=np.float32)
    reignited_matrix = np.zeros((max(0, T - 1), E), dtype=np.int32)
    extinguished_matrix = np.zeros((max(0, T - 1), E), dtype=np.int32)
    mask_flip_pct = np.zeros((max(0, T - 1), E), dtype=np.float32)
    conformal_alpha_matrix = np.zeros((max(0, T - 1), E), dtype=np.float32)

    prev_a_all: List[np.ndarray] = []
    ever_zero_mask = [np.zeros((R, C), dtype=bool) for _ in range(E)]
    ever_reignited_mask = [np.zeros((R, C), dtype=bool) for _ in range(E)]
    filenames: List[str] = []
    timestamps: List[str] = []

    for t_idx, p in enumerate(selected_files):
        filenames.append(p.name)
        with safe_open(str(p), framework="pt") as f:
            meta = f.metadata() or {}
            timestamps.append(meta.get("timestamp", ""))
            fat = f.get_tensor("fatigue_scores").float().numpy()
            tok = f.get_tensor("active_tokens").long().numpy()
            st = f.get_tensor("states").int().numpy()
            fatigue_matrix[t_idx, : E] = fat[:E]
            tokens_matrix[t_idx, : E] = tok[:E]
            state_matrix[t_idx, : E] = st[:E]

            cur_a_all: List[np.ndarray] = []
            for e in range(E):
                a = f.get_tensor(f"expert.{e}.lora_a").float().numpy()
                cur_a_all.append(a)
                lora_a_patch_series[t_idx, e] = a[:, :patch_cols]
                fro_a_matrix[t_idx, e] = float(np.linalg.norm(a, "fro"))
                is_zero = (a == 0.0)
                zero_pct_matrix[t_idx, e] = float(np.mean(is_zero) * 100.0)

                if t_idx > 0:
                    prev_a = prev_a_all[e]
                    prev_zero = (prev_a == 0.0)
                    cur_nonzero = ~is_zero

                    # Dead -> Alive (0.0 -> != 0.0)
                    reignited = int(np.sum(prev_zero & cur_nonzero))
                    # Alive -> Dead (!= 0.0 -> 0.0)
                    extinguished = int(np.sum((~prev_zero) & is_zero))
                    flipped_total = reignited + extinguished

                    reignited_matrix[t_idx - 1, e] = reignited
                    extinguished_matrix[t_idx - 1, e] = extinguished
                    mask_flip_pct[t_idx - 1, e] = float(flipped_total) / float(a.size) * 100.0
                    delta_fro_matrix[t_idx - 1, e] = float(np.linalg.norm(a - prev_a, "fro"))

                    ever_reignited_mask[e] |= (ever_zero_mask[e] & cur_nonzero)

                    surv = (~prev_zero) & cur_nonzero
                    if np.any(surv):
                        ratios = np.abs(a[surv]) / np.maximum(1e-12, np.abs(prev_a[surv]))
                        conformal_alpha_matrix[t_idx - 1, e] = float(np.mean(ratios))
                    else:
                        conformal_alpha_matrix[t_idx - 1, e] = 1.0

                ever_zero_mask[e] |= is_zero
            prev_a_all = cur_a_all

    ever_reignited_counts = [int(np.sum(ever_reignited_mask[e])) for e in range(E)]

    npz_path = EXPORT_DIR / f"recent_{T}_{profile}_weight_matrices.npz"
    np.savez_compressed(
        str(npz_path),
        filenames=np.array(filenames),
        timestamps=np.array(timestamps),
        lora_a_patch_series=lora_a_patch_series,
        fro_a_matrix=fro_a_matrix,
        fatigue_matrix=fatigue_matrix,
        state_matrix=state_matrix,
        tokens_matrix=tokens_matrix,
        zero_pct_matrix=zero_pct_matrix,
        delta_fro_matrix=delta_fro_matrix,
        reignited_matrix=reignited_matrix,
        extinguished_matrix=extinguished_matrix,
        mask_flip_pct=mask_flip_pct,
        conformal_alpha_matrix=conformal_alpha_matrix,
        ever_reignited_counts=np.array(ever_reignited_counts, dtype=np.int32),
    )

    csv_path = EXPORT_DIR / f"recent_{T}_{profile}_reignition_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow([
            "step_transition",
            "file_from",
            "file_to",
            "total_reignited_synapses_all_experts",
            "total_extinguished_synapses_all_experts",
            "mean_mask_flip_pct",
            "mean_delta_fro",
            "mean_conformal_alpha",
            "per_expert_reignited_list",
            "per_expert_alpha_list",
        ])
        for s in range(max(0, T - 1)):
            writer.writerow([
                f"Step_{s:02d}->{s+1:02d}",
                filenames[s],
                filenames[s + 1],
                int(np.sum(reignited_matrix[s])),
                int(np.sum(extinguished_matrix[s])),
                round(float(np.mean(mask_flip_pct[s])), 6),
                round(float(np.mean(delta_fro_matrix[s])), 6),
                round(float(np.mean(conformal_alpha_matrix[s])), 6),
                json.dumps(reignited_matrix[s].tolist()),
                json.dumps([round(float(x), 5) for x in conformal_alpha_matrix[s].tolist()]),
            ])

    expert_summary = []
    for e in range(E):
        expert_summary.append({
            "expert_id": e,
            "fro_start": round(float(fro_a_matrix[0, e]), 4),
            "fro_end": round(float(fro_a_matrix[-1, e]), 4),
            "total_reignited_events": int(np.sum(reignited_matrix[:, e])) if T > 1 else 0,
            "total_extinguished_events": int(np.sum(extinguished_matrix[:, e])) if T > 1 else 0,
            "unique_ever_reignited_synapses": ever_reignited_counts[e],
            "mean_mask_flip_pct": round(float(np.mean(mask_flip_pct[:, e])), 5) if T > 1 else 0.0,
            "mean_conformal_alpha": round(float(np.mean(conformal_alpha_matrix[:, e])), 5) if T > 1 else 1.0,
            "std_conformal_alpha": round(float(np.std(conformal_alpha_matrix[:, e])), 6) if T > 1 else 0.0,
        })

    report = {
        "profile": profile,
        "frames_analyzed": T,
        "sample_mode": sample_mode,
        "num_experts": E,
        "lora_a_shape": [R, C],
        "synapses_per_expert": R * C,
        "npz_file": str(npz_path),
        "csv_file": str(csv_path),
        "global_total_reignited_synapses": int(np.sum(reignited_matrix)) if T > 1 else 0,
        "global_total_extinguished_synapses": int(np.sum(extinguished_matrix)) if T > 1 else 0,
        "global_mean_mask_flip_pct": round(float(np.mean(mask_flip_pct)), 6) if T > 1 else 0.0,
        "expert_summary": expert_summary,
    }
    json_path = EXPORT_DIR / f"recent_{T}_{profile}_reignition_report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["json_file"] = str(json_path)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="thor_31b_16lora")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--mode", default="recent", choices=["recent", "stride"])
    args = ap.parse_args()
    rep = export_recent_weight_matrices(args.profile, args.frames, args.mode)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
