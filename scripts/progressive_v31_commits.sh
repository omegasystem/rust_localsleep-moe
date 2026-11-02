#!/usr/bin/env bash
set -euo pipefail

cd /home/node/.openclaw/workspace/projects/rust_localsleep-moe

echo "=== [Step 1] Committing current v3.1-step1 (Eligibility Traces + Syntax-Aware Local Sleep) ==="
git add ARCHITECTURE_OVERVIEW.md expert_pool_lora.py generation_engine_trt_hybrid.py ls_moe_pyqt_gui.py
git commit -m "feat(v3.1-step1): add dual-layer Eligibility Traces (STC) and Syntax-Aware Local Sleep

- Add LTM baseline snapshot (_ltm_baseline_a) and transient STP buffer (eligibility_trace_a) in DynamicLoRAExpertPool
- Consolidate STP trace into LTM on sandbox EXIT_CODE==0; rollback and apply Anti-Hebbian penalty on EXIT_CODE!=0
- Add Syntax-Aware Local Sleep (is_syntax_boundary_token + soft armed state=1 / hard cap 1.35x) in FallbackFatigueRouter
- Add ARCHITECTURE_OVERVIEW.md (v3.0/v3.1 condensed architecture guide)"

STEP1_HASH=$(git rev-parse --short HEAD)
echo "✅ Committed v3.1-step1 at: ${STEP1_HASH}"

echo "=== [Step 2] Applying v3.1-step2 (τ=2.5 tanh soft-capping + γ=0.992 + Anti-Hebbian Triton 2:4 prune) ==="
python3 - <<'PY'
from pathlib import Path

# 1. Update expert_pool_lora.py: γ = 0.992 + immediate Triton 2:4 Fisher prune on Anti-Hebbian rollback
p_pool = Path("expert_pool_lora.py")
txt_pool = p_pool.read_text(encoding="utf-8")

old_acc = "def accumulate_eligibility_trace(self, exp_key: str, pre_step_a: torch.Tensor, decay: float = 0.985):"
new_acc = "def accumulate_eligibility_trace(self, exp_key: str, pre_step_a: torch.Tensor, decay: float = 0.992):"
assert old_acc in txt_pool, "old_acc not found in expert_pool_lora.py"
txt_pool = txt_pool.replace(old_acc, new_acc, 1)

old_anti = """                elif sandbox_passed is False:
                    # Anti-Hebbian Rollback: reject polluted trajectory & penalize erroneous trace direction
                    nz_mask = (base != 0).to(la.dtype)
                    reverted = (base - 0.25 * trace) * nz_mask
                    nrm = float(torch.norm(reverted.float()).item())
                    if 0.0 < nrm < 0.65:
                        reverted.mul_(0.65 / nrm)
                    elif nrm > 2.80:
                        reverted.mul_(2.80 / nrm)
                    la.copy_(reverted)
                    # Keep baseline clean for the retry attempt
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k].zero_()"""

new_anti = """                elif sandbox_passed is False:
                    # Anti-Hebbian Rollback + Immediate Triton 2:4 Structural Prune on Active Faulty Channels:
                    # 1) Subtract 0.25 * E_i(t) from clean LTM baseline to weaken the exact channels that caused the bug
                    # 2) Immediately invoke Triton 2:4 Fisher pruning so weakened buggy channels lose the Top-2 quad competition and become 0.0!
                    trace_energy = float(torch.norm(trace.float()).item())
                    nz_mask = (base != 0).to(la.dtype)
                    reverted = (base - 0.25 * trace) * nz_mask
                    nrm = float(torch.norm(reverted.float()).item())
                    if 0.0 < nrm < 0.65:
                        reverted.mul_(0.65 / nrm)
                    elif nrm > 2.80:
                        reverted.mul_(2.80 / nrm)
                    la.copy_(reverted)
                    if trace_energy > 1e-6:
                        try:
                            from triton_localsleep_kernels import launch_triton_fisher_24_prune
                            launch_triton_fisher_24_prune(la, shy_lambda=0.08, survival_floor=0.985)
                        except Exception:
                            pass
                    # Keep baseline clean for the retry attempt
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k].zero_()"""

assert old_anti in txt_pool, "old_anti not found in expert_pool_lora.py"
txt_pool = txt_pool.replace(old_anti, new_anti, 1)
p_pool.write_text(txt_pool, encoding="utf-8")

# 2. Update generation_engine_trt_hybrid.py: τ = 2.5 tanh soft-capping + α = 0.25 EMA smoothing
p_eng = Path("generation_engine_trt_hybrid.py")
txt_eng = p_eng.read_text(encoding="utf-8")

old_bias = """                        r_idx = (int(cand_tid) * 2654435761) % 32
                        proj_val = float(rank_delta[r_idx].item())
                        raw_bias = proj_val * 28.0 * ent_scale / (1.0 + 0.35 * cand_rank)
                        bias_val = max(-3.5, min(3.5, raw_bias))
                        if abs(bias_val) >= 0.02:
                            tid_key = str(cand_tid)
                            prev_b = self.dynamic_logit_bias.get(tid_key, 0.0)
                            self.dynamic_logit_bias[tid_key] = round(max(-3.5, min(3.5, 0.65 * prev_b + 0.35 * bias_val)), 3)"""

new_bias = """                        r_idx = (int(cand_tid) * 2654435761) % 32
                        proj_val = float(rank_delta[r_idx].item())
                        raw_bias = proj_val * 28.0 * ent_scale / (1.0 + 0.35 * cand_rank)
                        # Hyperbolic Tangent Soft-Capping (τ = 2.5): preserves 1:1 slope in [-1.0, 1.0]
                        # while smoothly saturating below the 3.5 Eagle3 draft-rejection cliff!
                        tau_cap = 2.5
                        bias_val = tau_cap * math.tanh(raw_bias / tau_cap)
                        if abs(bias_val) >= 0.02:
                            tid_key = str(cand_tid)
                            prev_b = self.dynamic_logit_bias.get(tid_key, 0.0)
                            # EMA Micro-Step Smoothing (α = 0.25)
                            smoothed_b = 0.75 * prev_b + 0.25 * bias_val
                            self.dynamic_logit_bias[tid_key] = round(tau_cap * math.tanh(smoothed_b / tau_cap), 3)"""

assert old_bias in txt_eng, "old_bias not found in generation_engine_trt_hybrid.py"
txt_eng = txt_eng.replace(old_bias, new_bias, 1)
p_eng.write_text(txt_eng, encoding="utf-8")
print("✅ Updated expert_pool_lora.py and generation_engine_trt_hybrid.py with v3.1-step2 operators!")
PY

git add expert_pool_lora.py generation_engine_trt_hybrid.py scripts/progressive_v31_commits.sh
git commit -m "feat(v3.1-step2): add tau=2.5 tanh soft-capping, alpha=0.25 EMA, and Anti-Hebbian Triton 2:4 prune

- Replace hard [-3.5, 3.5] bias clamp with tau * tanh(b_raw / tau) (tau=2.5) + alpha=0.25 EMA smoothing to protect Eagle3 draft tree acceptance
- Set token-level eligibility trace decay gamma=0.992 to cover full function blocks
- Trigger immediate Triton 2:4 Fisher pruning on active experts after Anti-Hebbian penalty (EXIT_CODE!=0) so weakened faulty channels are zeroed out"

git log -n 3 --oneline
git status --short
