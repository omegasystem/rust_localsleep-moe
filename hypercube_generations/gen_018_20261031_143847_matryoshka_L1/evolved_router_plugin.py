from generation_engine_trt_hybrid import FallbackFatigueRouter
import math

class EvolvedCoreFatigueRouter(FallbackFatigueRouter):
    def __init__(self, num_experts=16, **kwargs):
        super().__init__(num_experts=num_experts, **kwargs)

    def forward_from_token_logprobs(self, token_id: int, top_token_ids: list, top_logprobs: list):
        weights, selected, entropy = super().forward_from_token_logprobs(token_id, top_token_ids, top_logprobs)
        
        # Dynamic Top-K adjustment based on fatigue of selected experts
        selected_fatigue = [self.fatigue_scores[exp_id] for exp_id in selected]
        avg_fatigue = sum(selected_fatigue) / len(selected_fatigue) if selected_fatigue else 0
        
        # If fatigue is high, we potentially shift weight or refine selection
        # Here we implement a soft-penalty on weights based on individual fatigue
        adjusted_weights = []
        for i, exp_id in enumerate(selected):
            penalty = math.exp(-self.fatigue_scores[exp_id])
            adjusted_weights.append(weights[i] * penalty)
        
        # Re-normalize weights
        total_w = sum(adjusted_weights) + 1e-6
        weights = [w / total_w for w in adjusted_weights]
        
        return weights, selected, entropy
