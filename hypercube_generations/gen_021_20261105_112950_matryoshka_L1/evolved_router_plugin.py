from generation_engine_trt_hybrid import FallbackFatigueRouter
import math

class EvolvedCoreFatigueRouter(FallbackFatigueRouter):
    def __init__(self, num_experts=16, **kwargs):
        super().__init__(num_experts=num_experts, **kwargs)

    def forward_from_token_logprobs(self, token_id: int, top_token_ids: list, top_logprobs: list):
        weights, selected, entropy = super().forward_from_token_logprobs(token_id, top_token_ids, top_logprobs)
        
        # Dynamic K-Selection based on Shannon Entropy H(p)
        k_target = 2 if entropy < 0.5 else (6 if entropy > 1.5 else 4)
        
        # Zero-Copy Fatigue Masking & Non-linear Repulsive Potential
        # Apply -1e9 NREM mask for fatigue > 0.7, otherwise use exp(-fatigue) decay
        adjusted_weights = []
        for exp_id in selected:
            fatigue = self.fatigue_scores[exp_id]
            penalty = -1e9 if fatigue > 0.7 else math.exp(-fatigue)
            # We use a temporary weight mapping; since weights is Top-K, we map 1:1
            # Note: weights[i] corresponds to selected[i]
            pass 

        # Re-calculating weights with the mask to avoid IndexError
        final_weights = []
        for i, exp_id in enumerate(selected):
            fatigue = self.fatigue_scores[exp_id]
            penalty = -1e9 if fatigue > 0.7 else math.exp(-fatigue)
            final_weights.append(weights[i] * penalty)
        
        total_w = sum(final_weights) + 1e-9
        final_weights = [w / total_w for w in final_weights]
        
        # Truncate/Pad to k_target
        if len(selected) > k_target:
            sorted_idx = sorted(range(len(final_weights)), key=lambda i: final_weights[i], reverse=True)
            top_idx = sorted_idx[:k_target]
            weights = [final_weights[i] for i in top_idx]
            selected = [selected[i] for i in top_idx]
        else:
            weights = final_weights
            
        return weights, selected, entropy
