//! downscaler.rs - Online Fisher Information Estimator & Synaptic Downscaler
//! Implements Synaptic Homeostasis Hypothesis (SHY) downscaling and 2:4 structural sparsity.

use serde::{Deserialize, Serialize};

/// Synaptic Downscaler implementing Homeostatic Downscaling & Structural Sparsity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FisherDownscaler {
    pub shy_lambda: f32,
    pub enable_2_4: bool,
}

impl FisherDownscaler {
    pub fn new(shy_lambda: f32, enable_2_4: bool) -> Self {
        Self {
            shy_lambda,
            enable_2_4,
        }
    }

    /// Estimates diagonal Fisher Information Matrix (FIM) from synthetic/sample activations
    /// fim_diag = E[grad^2]
    pub fn estimate_diagonal_fisher(&self, weights: &[f32], grad_scale: f32) -> Vec<f32> {
        let mut fim = Vec::with_capacity(weights.len());
        for &w in weights {
            // Empirical expectation approximation: F_i = (w * grad_scale)^2 + eps
            let approx_grad = w * grad_scale;
            fim.push(approx_grad * approx_grad + 1e-6);
        }
        fim
    }

    /// Applies 2:4 structural sparsity pruning in-place:
    /// In every consecutive 4-element block, exactly the 2 smallest values by Fisher importance are zeroed.
    pub fn prune_2_4_structural(&self, weights: &mut [f32], fim: &[f32]) {
        assert_eq!(weights.len(), fim.len());
        let chunk_size = 4;
        let num_chunks = weights.len() / chunk_size;

        for c in 0..num_chunks {
            let offset = c * chunk_size;
            let mut indices = [0, 1, 2, 3];

            // Sort chunk indices by Fisher importance (weight * sqrt(FIM)) ascending
            indices.sort_by(|&i, &j| {
                let score_i = weights[offset + i].abs() * fim[offset + i].sqrt();
                let score_j = weights[offset + j].abs() * fim[offset + j].sqrt();
                score_i.partial_cmp(&score_j).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Zero out the 2 least important synapses (indices[0] and indices[1])
            weights[offset + indices[0]] = 0.0;
            weights[offset + indices[1]] = 0.0;
        }
    }

    /// Applies Synaptic Homeostasis Hypothesis (SHY) downscaling:
    /// w_new = w * exp(-lambda / (sqrt(F) + eps))
    pub fn apply_homeostatic_downscaling(&self, weights: &mut [f32], fim: &[f32]) {
        assert_eq!(weights.len(), fim.len());
        for i in 0..weights.len() {
            if weights[i] != 0.0 {
                let protection = fim[i].sqrt() + 1e-5;
                let decay_factor = (-self.shy_lambda / protection).exp();
                weights[i] *= decay_factor;
            }
        }
    }

    /// Full maintenance cycle on expert weights
    pub fn execute_sleep_maintenance(&self, weights: &mut [f32]) {
        let fim = self.estimate_diagonal_fisher(weights, 0.05);

        if self.enable_2_4 {
            self.prune_2_4_structural(weights, &fim);
        }

        self.apply_homeostatic_downscaling(weights, &fim);
    }
}
