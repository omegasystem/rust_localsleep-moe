from generation_engine_trt_hybrid import FallbackFatigueRouter
import math

class EvolvedCoreFatigueRouter(FallbackFatigueRouter):
    def __init__(self, num_experts=16, **kwargs):
        super().__init__(num_experts=num_experts, **kwargs)

    def forward_from_token_logprobs(self, token_id: int, top_token_ids: list, top_logprobs: list):
        weights, selected, entropy = super().forward_from_token_logprobs(token_id, top_token_ids, top_logprobs)
        
        # Dynamic K-Selection based on Shannon Entropy H(p)
        k_target = 2 if entropy < 0.5 else (6 if entropy > 1.5 else 4)
        
        # Non-linear Repulsive Potential Field for 16-LoRA Fatigue Balancing
        # Use exp(-fatigue) as a soft-penalty to push routing towards under-utilized experts
        adjusted_weights = []
        for exp_id in selected:
            # Fatigue-aware weight modulation: exp(-f) ensures positive decay
            penalty = math.exp(-self.fatigue_scores[exp_id])
            # Apply NREM sleep mask for extreme fatigue (>0.7)
            if self.fatigue_scores[exp_id] > 0.7:
                penalty *= 1e-4
            adjusted_weights.append(penalty)
        
        # Combine original weights with fatigue penalties
        final_weights = [w * p for w, p in zip(weights, adjusted_weights)]
        
        # Re-normalize weights to maintain sum=1.0
        total_w = sum(final_weights) + 1e-9
        final_weights = [w / total_w for w in final_weights]
        
        # Truncate or pad selected experts to match k_target
        if len(selected) > k_target:
            # Sort by adjusted weights to keep the most 'fresh' and relevant experts
            sorted_indices = sorted(range(len(final_weights)), key=lambda i: final_weights[i], reverse=True)
            top_indices = sorted_indices[:k_target]
            final_weights = [final_weights[i] for i in top_indices]
            selected = [selected[i] for i in top_indices]
        
        return final_weights, selected, entropy
