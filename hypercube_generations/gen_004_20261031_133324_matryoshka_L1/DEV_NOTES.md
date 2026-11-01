# 📓 子代超立方體開發筆記 (`gen_004_20261031_133324_matryoshka_L1`)
- **母體來源 (Parent Base Cube)**: `gen_003_20261031_132418_matryoshka_L1` (唯讀吸收)
- **子代誕生時間**: `2026-10-31T13:33:24`
- **繼承基石檔案數**: `7` files (`src/router.rs, src/downscaler.rs, src/engine.rs, src/expert_pool.rs, csrc/sparse_expert_gather.cu, csrc/fused_fatigue_topk.cu...`)

---

## 🕒 演化研究日誌 (Autonomous Evolution Log)

### [2026-10-31 13:33:24] 🔹 Stage 1: 唯讀吸收母體架構 (Read-Only Parent Introspection)
- 已以 **100% 唯讀模式 (Read-Only)** 吸收母體 `gen_003_20261031_132418_matryoshka_L1` 共 **7** 個核心模組。
- 關鍵結構映射：
  * `src/router.rs` (234L): `L011: pub struct FatigueMoERouter {, L032: impl FatigueMoERouter {, L033: pub fn new(cfg: &LocalSleepConfig) -> Self {`
  * `src/downscaler.rs` (80L): `L008: pub struct FisherDownscaler {, L013: impl FisherDownscaler {, L014: pub fn new(shy_lambda: f32, enable_2_4: bool) -> Self {`
  * `src/engine.rs` (92L): `L011: pub struct MaintenanceJob {, L018: pub struct DualStreamEngine {, L023: impl DualStreamEngine {`
  * `src/expert_pool.rs` (145L): `L009: pub struct LoRAExpert {, L021: impl LoRAExpert {, L022: pub fn new(id: usize, hidden_dim: usize, rank: usize, alpha: f32) -> Self {`
  * `csrc/sparse_expert_gather.cu` (313L): `L018: __global__ void sparse_expert_gather_kernel(, L054: __global__ void lora_down_project_kernel(, L109: __global__ void lora_up_project_kernel(`
  * `csrc/fused_fatigue_topk.cu` (171L): `L033: __global__ void fused_fatigue_topk_kernel(, L142: extern "C" void launch_fused_fatigue_topk(`
  * `expert_pool_lora.py` (70L): `L004: class DynamicLoRAExpertPool(nn.Module):, L011: def __init__(self, num_experts=8, in_dim=3840, out_dim=3840, rank=16, alpha=16.0, device="cuda", dty, L031: def forward_delta(self, x, selected_expert_idx, weight=1.0):`

---

### [2026-10-31 13:33:47] 🔹 Stage 2: 演化假說推導與目標設定 (Hypothesis & Goals)
### 🧬 子代開發筆記：`evolved_homeostasis.py` (Stage 2)

**1. 演化切入點：從靜態閾值轉向熵感知動態調節**
*   **FatigueMoERouter $\rightarrow$ 熵驅動喚醒**：將 `wake_threshold` 由常數演化為基於路由分佈熵 $H(p)$ 的動態函數。當 $\Delta H$ 劇增時，觸發專家強制喚醒，打破靜態 `decay_rate` 的線性限制。
*   **SHY 穩態降維 $\rightarrow$ 資訊幾何調節**：將 `shy_lambda` 與 Fisher 資訊矩陣 (FIM) 的跡 $\text{Tr}(F)$ 耦合。實作 $\lambda_{dyn} = f(\text{Tr}(F), \text{Fatigue})$，使 2:4 稀疏化在高疲勞度時增加剪枝強度，在低疲勞度時保留權重精度。

**2. 核心實作公式與收斂指標**
*   **動態 $\lambda$ 調節公式**：
    $$\lambda_{t+1} = \lambda_0 \cdot \exp\left( \alpha \cdot \frac{\text{Tr}(F_t)}{\text{Tr}(F_{base})} - \beta \cdot \text{Fatigue}_{score} \right)$$
*   **專家喚醒門檻 $\tau$**：
    $$\tau_{wake} = \tau_{base} \cdot (1 - \text{Softmax}(\text{Entropy}_{route}))$$
*   **收斂指標 (Convergence Metrics)**：
    *   **$\text{Sparsity Stability}$**: $\left| \frac{\partial \text{Sparsity}}{\partial t} \right| \to 0$ (稀疏度在 NREM 週期後趨於穩定)。
    *   **$\text{Routing Efficiency}$**: $\frac{\text{Perplexity}}{\text{Active Experts}} \downarrow$ (在維持性能前提下最小化激活專家數)。

---

### [2026-10-31 13:33:47] 🔹 Stage 3: 輸出子代演化模組 (Offspring Code Synthesis)
- 已於子代專案 `hypercube_generations/gen_004_20261031_133324_matryoshka_L1` 生成：
  1. `evolved_homeostasis.py`（熵調變自適應 SHY 降維與跨專家正交監控器）
  2. `tests/verify_offspring_cube.py`（超立方體維度切換前自動驗證閘門）

---

### [2026-10-31 13:34:22] 🔹 Stage 4: 子代驗證閘門通過 (Verification Gate PASS)
✅ 執行 `tests/verify_offspring_cube.py` 通過！
```json
HYPERCUBE_VERIFY_REPORT:{"cube_id": "gen_004_20261031_133324_matryoshka_L1", "status": "PASS", "sparsity_pct": 50.0, "min_norm": 1.5755, "orthogonality_omega": 0.019852, "lambda_high_ent": 0.06107, "lambda_low_ent": 0.09578}
```
- 狀態已標記為 `VERIFIED_READY_FOR_SWAP`，可隨時執行 **超立方體維度切換 (Hypercube Dimension Swap)**！

---

### [2026-10-31 13:34:22] 🔹 🪆 Stage 4+: Headless 俄羅斯娃娃第 2 層孫代試爆報告 (Matryoshka L2 Benchmark)
```json
{
  "l2_canary_alive": false,
  "error": "\nTraceback (most recent call last):\n  File \"/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/hypercube_generations/gen_004_20261031_133324_matryoshka_L1/evolved_subagent_core.py\", line 194, in <module>\n    rep = core.spawn_and_verify_layer2_canary(l2_target)\n          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/hypercube_generations/gen_004_20261031_133324_matryoshka_L1/evolved_subagent_core.py\", line 125, in"
}
```

---
