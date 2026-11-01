from generation_engine_trt_hybrid import FallbackFatigueRouter
import math

class EvolvedCoreFatigueRouter(FallbackFatigueRouter):
    """
    S-Tier Evolved Neural Router Plugin.
    Implements an advanced fatigue-aware distribution mechanism for 16 LoRA experts
    with dynamic Top-K selection and entropy-driven load balancing.
    """
    def __init__(self, num_experts=16, **kwargs):
        # Overriding default expert count to 16 as per steering instructions
        super().__init__(num_experts=16, **kwargs)
        self.num_experts = 16

    def forward_from_token_logprobs(self, token_id: int, top_token_ids: list, top_logprobs: list):
        """
        Evolved routing logic: Integrates fatigue scores with token log-probabilities
        to optimize expert utilization and prevent catastrophic fatigue saturation.
        """
        # 1. Call base class to get initial weights, selected experts, and entropy
        weights, selected, entropy = super().forward_from_token_logprobs(token_id, top_token_ids, top_logprobs)

        # 2. Evolutionary Fatigue Mitigation Formula
        # We apply a non-linear penalty to experts with high fatigue scores
        # to force the distribution towards under-utilized experts (Load Balancing).
        
        # Calculate global fatigue mean for normalization
        avg_fatigue = sum(self.fatigue_scores) / self.num_experts
        
        # Evolutionary Adjustment Factor: 
        # NewWeight = OriginalWeight * exp(-alpha * (ExpertFatigue - AvgFatigue))
        # This ensures that experts significantly above the average fatigue are suppressed.
        alpha = 0.15  # Sensitivity coefficient for fatigue steering
        
        adjusted_weights = []
        for i in range(self.num_experts):
            # Calculate the fatigue delta
            fatigue_delta = self.fatigue_scores[i] - avg_fatigue
            
            # Apply exponential decay based on fatigue deviation
            # If fatigue_delta is negative (under-utilized), weight increases.
            # If fatigue_delta is positive (over-utilized), weight decreases.
            penalty = math.exp(-alpha * fatigue_delta)
            adjusted_weights.append(weights[i] * penalty)

        # 3. Re-normalization (Softmax-like) to maintain probability distribution
        sum_weights = sum(adjusted_weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in adjusted_weights]
        else:
            # Fallback to uniform if weights collapse
            weights = [1.0 / self.num_experts] * self.num_experts

        # 4. Auto Top-K Selection based on evolved weights
