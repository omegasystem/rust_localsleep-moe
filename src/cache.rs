//! cache.rs - Sink-Aware Sliding Window KV Cache (SWA) in Pure Rust
//! Implements constant O(1) VRAM footprint retention via Attention Sinks (first S tokens)
//! and Local Sliding Window (recent W tokens). Eliminates OOM and Attention Collapse.

use serde::{Deserialize, Serialize};

/// Single-layer Sink-Aware Rolling Cache Layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkAwareRollingLayer {
    pub sink_size: usize,
    pub window_size: usize,
    pub head_dim: usize,
    pub num_heads: usize,

    // Flattened buffers: [num_heads, seq_len, head_dim]
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub current_seq_len: usize,
}

impl SinkAwareRollingLayer {
    pub fn new(sink_size: usize, window_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            sink_size,
            window_size,
            head_dim,
            num_heads,
            keys: Vec::new(),
            values: Vec::new(),
            current_seq_len: 0,
        }
    }

    /// Appends new key and value states and enforces the SWA bound: [0:sink] + [-window:]
    pub fn update(
        &mut self,
        new_keys: &[f32],   // [num_heads, new_tokens, head_dim]
        new_values: &[f32], // [num_heads, new_tokens, head_dim]
        new_tokens: usize,
    ) {
        let stride = new_tokens * self.head_dim;
        assert_eq!(new_keys.len(), self.num_heads * stride);
        assert_eq!(new_values.len(), self.num_heads * stride);

        let max_capacity = self.sink_size + self.window_size;

        if self.current_seq_len == 0 {
            self.keys = new_keys.to_vec();
            self.values = new_values.to_vec();
            self.current_seq_len = new_tokens;
        } else {
            // Re-allocate / slice concatenated states
            let total_tokens = self.current_seq_len + new_tokens;
            let mut next_keys = Vec::with_capacity(self.num_heads * total_tokens * self.head_dim);
            let mut next_values = Vec::with_capacity(self.num_heads * total_tokens * self.head_dim);

            for h in 0..self.num_heads {
                // Existing tokens for head h
                let cur_k_offset = h * self.current_seq_len * self.head_dim;
                next_keys.extend_from_slice(&self.keys[cur_k_offset..cur_k_offset + self.current_seq_len * self.head_dim]);
                let cur_v_offset = h * self.current_seq_len * self.head_dim;
                next_values.extend_from_slice(&self.values[cur_v_offset..cur_v_offset + self.current_seq_len * self.head_dim]);

                // Append new tokens for head h
                let new_offset = h * stride;
                next_keys.extend_from_slice(&new_keys[new_offset..new_offset + stride]);
                next_values.extend_from_slice(&new_values[new_offset..new_offset + stride]);
            }

            self.current_seq_len = total_tokens;

            // Enforce rolling eviction if capacity exceeded
            if self.current_seq_len > max_capacity {
                let trimmed_len = max_capacity;
                let mut bounded_keys = Vec::with_capacity(self.num_heads * trimmed_len * self.head_dim);
                let mut bounded_values = Vec::with_capacity(self.num_heads * trimmed_len * self.head_dim);

                for h in 0..self.num_heads {
                    let head_k_start = h * total_tokens * self.head_dim;
                    let head_v_start = h * total_tokens * self.head_dim;

                    // 1. Preserve initial sink tokens [0 : sink_size]
                    let sink_bytes = self.sink_size * self.head_dim;
                    bounded_keys.extend_from_slice(&next_keys[head_k_start..head_k_start + sink_bytes]);
                    bounded_values.extend_from_slice(&next_values[head_v_start..head_v_start + sink_bytes]);

                    // 2. Preserve recent sliding window [-window_size : ]
                    let win_start_token = total_tokens - self.window_size;
                    let win_offset = win_start_token * self.head_dim;
                    let win_bytes = self.window_size * self.head_dim;

                    bounded_keys.extend_from_slice(&next_keys[head_k_start + win_offset..head_k_start + win_offset + win_bytes]);
                    bounded_values.extend_from_slice(&next_values[head_v_start + win_offset..head_v_start + win_offset + win_bytes]);
                }

                self.keys = bounded_keys;
                self.values = bounded_values;
                self.current_seq_len = trimmed_len;
            } else {
                self.keys = next_keys;
                self.values = next_values;
            }
        }
    }

    pub fn seq_len(&self) -> usize {
        self.current_seq_len
    }

    pub fn is_bounded(&self) -> bool {
        self.current_seq_len <= self.sink_size + self.window_size
    }
}

/// Multi-layer SWA Cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkAwareRollingCache {
    pub layers: Vec<SinkAwareRollingLayer>,
    pub num_layers: usize,
}

impl SinkAwareRollingCache {
    pub fn new(num_layers: usize, sink_size: usize, window_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(SinkAwareRollingLayer::new(sink_size, window_size, num_heads, head_dim));
        }

        Self { layers, num_layers }
    }

    pub fn update_layer(&mut self, layer_idx: usize, keys: &[f32], values: &[f32], new_tokens: usize) {
        if layer_idx < self.num_layers {
            self.layers[layer_idx].update(keys, values, new_tokens);
        }
    }
}
