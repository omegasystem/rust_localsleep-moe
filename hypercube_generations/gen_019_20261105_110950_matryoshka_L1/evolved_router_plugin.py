from generation_engine_trt_hybrid import FallbackFatigueRouter
import math

class EvolvedCoreFatigueRouter(FallbackFatigueRouter):
    def __init__(self, num_experts=16, **kwargs):
        super().__init__(num_experts=num_experts, **kwargs)

    def forward_from_token_logprobs(self, token_id: int, top_token_ids: list, top_logprobs: list):
        weights, selected, entropy = super().forward_from_token_logprobs(token_id, top_token_ids, top_logprobs)
        
        # Dynamic K-Selection based on Shannon Entropy H(p)
        # High entropy -> expand K to explore; Low entropy -> shrink K to exploit
        k_target = 2 if entropy < 0.5 else (6 if entropy > 1.5 else 4)
        
        # Non-linear Repulsive Potential Field for 16-LoRA Fatigue Balancing
        # Use exp(-fatigue) as a soft-penalty to push routing towards under-utilized experts
        adjusted_weights = []
        for i, exp_id in enumerate(selected):
            # Fatigue-aware weight modulation
            penalty = math.exp(-self.fatigue_scores[exp_id])
            adjusted_weights.append(weights[i] * penalty)
        
        # Re-normalize weights to maintain sum=1.0
        total_w = sum(adjusted_weights) + 1e-9
        weights = [w / total_w for w in adjusted_weights]
        
        # Truncate or pad selected experts to match k_target
        if len(selected) > k_target:
            # Sort by adjusted weights to keep the most 'fresh' and relevant experts
            sorted_indices = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
            top_indices = sorted_indices[:k_target]
            weights = [weights[i] for i in top_indices]
            selected = [selected[i] for i in top_indices]
        
        return weights, selected, entropy
