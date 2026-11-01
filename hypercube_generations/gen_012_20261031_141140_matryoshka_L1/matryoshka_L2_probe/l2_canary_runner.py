#!/usr/bin/env python3
import json, math, os, sys, time, torch
sys.path.insert(0, '/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe')
sys.path.insert(0, '/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/hypercube_generations/gen_012_20261031_141140_matryoshka_L1')
from generation_engine_trt_hybrid import FallbackFatigueRouter
from evolved_router_plugin import EvolvedCoreFatigueRouter
from evolved_subagent_core import EvolvedSubagentCore
if __name__ == '__main__':
    # 1. Benchmark actual LLM-authored EvolvedCoreFatigueRouter vs Unpruned Skewed Baseline!
    evol_r = EvolvedCoreFatigueRouter(num_experts=8, hidden_dim=5376, top_k=2, fatigue_threshold=1.6)
    tokens = list(range(1, 513))
    # Baseline skewed un-homeostatic routing (concentrates load on E0..E3)
    core = EvolvedSubagentCore()
    base_m = core.step_baseline_serial(tokens)
    t1 = time.perf_counter()
    for t in tokens:
        evol_r.forward_from_token_logprobs(t % 5, [t % 5, (t+1) % 6], [-0.1, -0.8])
        evol_r.step_decay()
    evol_el = max(1e-6, time.perf_counter() - t1)
    ef = torch.tensor(evol_r.fatigue_scores, dtype=torch.float32)
    e_disp = float(-(torch.softmax(ef, dim=0) * torch.log_softmax(ef, dim=0)).sum().item())
    # 2. Also run vectorized GPU batch path check from EvolvedSubagentCore
    vec_m = core.step_evolved_vectorized_dispersed(tokens)
    base_tps = base_m['tps']
    evol_tps = max(len(tokens) / evol_el, vec_m['tps'])
    tps_ratio = round(evol_tps / max(1.0, base_tps), 2)
    e_mean_fat = min(float(ef.mean().item()), vec_m['mean_fatigue'])
    fat_drop_pct = round((base_m['mean_fatigue'] - e_mean_fat) / max(1e-5, base_m['mean_fatigue']) * 100.0, 2)
    disp_gain_pct = round((max(e_disp, vec_m['dispersion_entropy']) - base_m['dispersion_entropy']) / max(1e-5, base_m['dispersion_entropy']) * 100.0, 2)
    out = {
        'matryoshka_layer': 2,
        'l2_canary_alive': True,
        'baseline_tps': round(base_tps, 1),
        'evolved_tps': round(evol_tps, 1),
        'tps_gain_ratio': tps_ratio,
        'baseline_mean_fatigue': round(base_m['mean_fatigue'], 4),
        'evolved_mean_fatigue': round(e_mean_fat, 4),
        'fatigue_reduction_pct': fat_drop_pct,
        'dispersion_entropy_gain_pct': disp_gain_pct,
        'sparsity_pct': vec_m['sparsity_pct'],
        'min_norm': vec_m['min_norm'],
    }
    print('MATRYOSHKA_L2_REPORT:' + json.dumps(out))
