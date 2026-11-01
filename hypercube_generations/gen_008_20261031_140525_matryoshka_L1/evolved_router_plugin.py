#!/usr/bin/env python3
"""
evolved_router_plugin.py - In-Place Hot-Swappable Core Router Engine (Fallback)
"""
from generation_engine_trt_hybrid import FallbackFatigueRouter
import math


class EvolvedCoreFatigueRouter(FallbackFatigueRouter):
    def forward_from_token_logprobs(self, token_id: int, top_token_ids: list, top_logprobs: list):
        weights, selected, entropy = super().forward_from_token_logprobs(token_id, top_token_ids, top_logprobs)
        mean_f = sum(self.fatigue_scores) / max(1, len(self.fatigue_scores))
        for i in range(self.num_experts):
            if self.fatigue_scores[i] > mean_f:
                self.fatigue_scores[i] = max(0.0, self.fatigue_scores[i] * 0.965)
        return weights, selected, entropy
