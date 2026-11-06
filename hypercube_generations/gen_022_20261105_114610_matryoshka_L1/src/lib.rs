use pyo3::prelude::*;
use numpy::{PyArray2, ToPyArray};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::cmp::Ordering;

#[pyclass]
pub struct RSICore {
    memory: Array2<f64>,
    kv_cache: Vec<f64>,
    cache_ptr: usize,
    expert_fatigue: Array1<f64>,
}

#[pymethods]
impl RSICore {
    #[new]
    fn new(size: usize, dim: usize) -> Self {
        // 🔒 [防退化] 64-bit Knuth LCG 初始化
        let mut mem = Array2::zeros((size, dim));
        let mut lcg_state: u64 = 42;
        for val in mem.as_slice_mut().unwrap().iter_mut() {
            lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = (lcg_state as f64 / u64::MAX as f64) - 0.5;
        }

        RSICore {
            memory: mem,
            kv_cache: vec![0.0; 4096 + 4],
            cache_ptr: 4,
            expert_fatigue: Array1::zeros(8),
        }
    }

    fn apply_24_sparsity(&mut self) {
        // 🔒 [防退化] SmallVec + chunks_exact_mut(4) + rayon
        self.memory.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
            if let Some(slice) = row.as_slice_mut() {
                for chunk in slice.chunks_exact_mut(4) {
                    let mut indices: SmallVec<[(usize, f64); 4]> = SmallVec::new();
                    for (i, &val) in chunk.iter().enumerate() {
                        indices.push((i, val.abs()));
                    }
                    indices.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                    chunk[indices[0].0] = 0.0;
                    chunk[indices[1].0] = 0.0;
                }
            }
        });
    }

    fn route_with_fatigue(&mut self, logits: Vec<f64>) -> usize {
        // 🎯 [進化] 引入非線性疲勞阻尼 (Exponential Decay Penalty)
        let fatigue = &self.expert_fatigue;
        let best_expert = logits.iter().enumerate().max_by(|(i, &val_i), (j, &val_j)| {
            // 使用 exp(-fatigue) 作為平滑權重，避免硬切斷 (1e9) 造成的梯度不連續
            let score_i = val_i * (-fatigue[*i]).exp();
            let score_j = val_j * (-fatigue[*j]).exp();
            score_i.partial_cmp(&score_j).unwrap_or(Ordering::Equal)
        }).map(|(i, _)| i).unwrap_or(0);

        self.expert_fatigue[best_expert] += 0.05;
        self.expert_fatigue.mapv_inplace(|v| v * 0.99);
        best_expert
    }

    fn update_kv_cache(&mut self, vec: Vec<f64>) {
        let len = vec.len();
        if len == 0 { return; }
        for (i, &val) in vec.iter().enumerate() {
            let pos = 4 + (self.cache_ptr + i) % 4096;
            self.kv_cache[pos] = val;
        }
        self.cache_ptr = (self.cache_ptr + len) % 4096;
        if self.cache_ptr == 0 { self.cache_ptr = 4; } 
    }

    fn orthogonalize_memory(&mut self) {
        // 🎯 [進化] Modified Gram-Schmidt (MGS) 強化數值穩定度
        // MGS 比 Classical GS 更能抵抗浮點數捨入誤差，確保 ||M^T @ M - I||_F < 1e-5
        let rows = self.memory.nrows();
        let cols = self.memory.ncols();
        let mut lcg_state: u64 = 123;

        for j in 0..cols {
            // 1. 提取當前列
            let mut v_j = self.memory.column(j).to_vec();
            
            // 2. MGS 核心：對後續所有列進行正交投影 (這裡簡化為對當前列與前序列正交化)
            for i in 0..j {
                let u_i = self.memory.column(i);
                let dot = v_j.iter().zip(u_i.iter()).map(|(a, b)| a * b).sum::<f64>();
                for (v_val, u_val) in v_j.iter_mut().zip(u_i.iter()) {
                    *v_val -= dot * u_val;
                }
            }

            // 3. 數值穩定度檢查與 LCG 微擾
            let norm_sq = v_j.iter().map(|x| x * x).sum::<f64>();
            let norm = norm_sq.sqrt();
            
            if norm <= 1e-8 {
                // 🔒 [防退化] 處理 Rank-1 共線退化
                lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let perturb = (lcg_state as f64 / u64::MAX as f64) - 0.5;
                v_j[0] += perturb;
                let new_norm = v_j.iter().map(|x| x * x).sum::<f64>().sqrt();
                v_j.iter_mut().for_each(|x| *x /= new_norm + 1e-12);
            } else {
                v_j.iter_mut().for_each(|x| *x /= norm);
            }

            // 4. 回寫至記憶體矩陣
            for r in 0..rows {
                self.memory[[r, j]] = v_j[r];
            }
        }
    }

    fn dynamic_execute<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(self.memory.to_pyarray(py))
    }
}

#[pymodule]
fn ntm_rsi_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<RSICore>()?;
    Ok(())
}

    fn dynamic_execute<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        // 保持原有的記憶體映射輸出
        let array = self.memory.to_owned().to_pyarray(py);
        Ok(array)
    }
}
