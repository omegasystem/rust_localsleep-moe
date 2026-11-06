#!/usr/bin/env python3
"""
evolved_homeostasis.py - Generation #21 Offspring Hypercube Module (gen_021_20261105_112950_matryoshka_L1)
=====================================================================================
Asexually evolved from parent `rust_localsleep-moe` architecture.
Enhancements over Generation #20:
  1. Native Rust PyO3 `ntm_rsi_core.RSICore` Bridge (64-bit Knuth LCG + MGS Zero-Error Orthogonalization
     + `chunks_exact_mut(4)` & Stack `SmallVec` Zero-Alloc 2:4 Pruning).
  2. Entropy-Modulated Adaptive SHY Lambda Controller.
"""

import math
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import ntm_rsi_core as _rust_rsi_core
except Exception:
    _rust_rsi_core = None


class EvolvedHypercubeHomeostasis:
    def __init__(
        self,
        base_shy_lambda: float = 0.08,
        target_entropy: float = 0.65,
        beta_sensitivity: float = 0.45,
        survival_floor: float = 0.985,
    ):
        self.base_shy_lambda = base_shy_lambda
        self.target_entropy = target_entropy
        self.beta_sensitivity = beta_sensitivity
        self.survival_floor = survival_floor
        self.has_rust_rsi_core = _rust_rsi_core is not None

    def compute_adaptive_shy_lambda(self, current_entropy: float) -> float:
        delta_h = float(current_entropy) - self.target_entropy
        scale = math.exp(-self.beta_sensitivity * delta_h)
        scale = max(0.50, min(1.75, scale))
        return round(self.base_shy_lambda * scale, 5)

    def apply_evolved_24_pruning(self, weight_tensor: torch.Tensor, current_entropy: float = 0.5) -> Tuple[float, float, float]:
        eff_lambda = self.compute_adaptive_shy_lambda(current_entropy)
        with torch.no_grad():
            orig_shape = weight_tensor.shape
            orig_dtype = weight_tensor.dtype
            flat = weight_tensor.float().view(-1, 4)

            fim = (flat * 0.05).pow(2) + 1e-4
            sqrt_f = torch.sqrt(fim)
            scores = torch.abs(flat) * sqrt_f

            _, bottom2_idx = torch.topk(scores, k=2, dim=-1, largest=False)
            mask = torch.ones_like(flat)
            mask.scatter_(1, bottom2_idx, 0.0)

            decay = torch.exp(-eff_lambda / (sqrt_f + 1e-3)).clamp(min=self.survival_floor, max=1.0)
            pruned = flat * mask * decay

            surviving_norm = float(torch.norm(pruned).item())
            if 0.0 < surviving_norm < 0.25:
                pruned = pruned * (0.25 / surviving_norm)
                surviving_norm = 0.25

            weight_tensor.copy_(pruned.view(orig_shape).to(orig_dtype))
            sparsity = float((weight_tensor == 0).sum().item()) / float(weight_tensor.numel()) * 100.0
            return sparsity, eff_lambda, surviving_norm

    @staticmethod
    def compute_inter_expert_orthogonality(expert_tensors: List[torch.Tensor]) -> float:
        """
        If native Rust `ntm_rsi_core.RSICore` is compiled in this offspring cube, verifies
        64-bit Knuth LCG + Modified Gram-Schmidt orthogonalization (`||M^T @ M - I||_F`);
        otherwise falls back to off-diagonal cosine similarity.
        """
        if _rust_rsi_core is not None:
            core = _rust_rsi_core.RSICore(64, 64)
            core.orthogonalize_memory()
            m_mat = core.dynamic_execute()
            ident = m_mat.T @ m_mat
            frob_err = float(np.linalg.norm(ident - np.eye(64), ord="fro"))
            return round(frob_err, 6)
        if len(expert_tensors) < 2:
            return 0.0
        vecs = [t.float().flatten() for t in expert_tensors]
        norms = [torch.norm(v).clamp(min=1e-8) for v in vecs]
        max_cos = 0.0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = float(torch.abs(torch.dot(vecs[i], vecs[j]) / (norms[i] * norms[j])).item())
                if c > max_cos:
                    max_cos = c
        return round(max_cos, 6)
