#!/usr/bin/env python3
"""
verify_offspring_cube.py - Self-Verification Gate for Offspring Hypercube (gen_013_20261031_141300_matryoshka_L1)
"""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolved_homeostasis import EvolvedHypercubeHomeostasis


def main():
    ctrl = EvolvedHypercubeHomeostasis(base_shy_lambda=0.08, target_entropy=0.65)
    torch.manual_seed(42)
    experts = [torch.randn(32, 256, dtype=torch.bfloat16) * 0.02 for _ in range(8)]

    for step, ent in enumerate([1.25, 0.85, 0.45, 0.25]):
        for e in experts:
            sp, eff_l, norm_v = ctrl.apply_evolved_24_pruning(e, current_entropy=ent)
            assert abs(sp - 50.0) < 1e-3, f"2:4 Sparsity violated: {sp}%"
            assert norm_v >= 0.249, f"Underflow detected: {norm_v}"

    omega = ctrl.compute_inter_expert_orthogonality(experts)
    report = {
        "cube_id": "gen_013_20261031_141300_matryoshka_L1",
        "status": "PASS",
        "sparsity_pct": 50.0,
        "min_norm": round(min(float(torch.norm(e.float()).item()) for e in experts), 4),
        "orthogonality_omega": omega,
        "lambda_high_ent": ctrl.compute_adaptive_shy_lambda(1.25),
        "lambda_low_ent": ctrl.compute_adaptive_shy_lambda(0.25),
    }
    print("HYPERCUBE_VERIFY_REPORT:" + json.dumps(report))


if __name__ == "__main__":
    main()
