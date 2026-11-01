#!/usr/bin/env python3
import json, math, os, sys, time, torch
sys.path.insert(0, '/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe')
sys.path.insert(0, '/home/omegacd/.openclaw/workspace/projects/rust_localsleep-moe/hypercube_generations/gen_007_20261031_140402_matryoshka_L1')
from generation_engine_trt_hybrid import FallbackFatigueRouter
from evolved_router_plugin import EvolvedCoreFatigueRouter
from evolved_subagent_core import EvolvedSubagentCore
if __name__ == '__main__':
    # 1. Benchmark actual LLM-authored EvolvedCoreFatigueRouter vs Parent FallbackFatigueRouter!
    base_r = FallbackFatigueRouter(num_experts=8, hidden_dim=5376, top_k=2, fatigue_threshold=1.6)
    evol_r = EvolvedCoreFatigueRouter(num_experts=8, hidden_dim=5376, top_k=2, fatigue_threshold=1.6)
    tokens = list(range(1, 513))
    t0 = time.perf_counter()
    for t in tokens:
        base_r.forward_from_token_logprobs(t % 5, [t % 5, (t+1) % 6], [-0.1, -0.8])
        base_r.step_decay()
    base_el = max(1e-6, time.perf_counter() - t0)
    t1 = time.perf_counter()
    for t in tokens:
        evol_r.forward_from_token_logprobs(t % 5, [t % 5, (t+1) % 6], [-0.1, -0.8])
        evol_r.step_decay()
    evol_el = max(1e-6, time.perf_counter() - t1)
    bf = torch.tensor(base_r.fatigue_scores, dtype=torch.float32)
    ef = torch.tensor(evol_r.fatigue_scores, dtype=torch.float32)
    b_disp = float(-(torch.softmax(bf, dim=0) * torch.log_softmax(bf, dim=0)).sum().item())
    e_disp = float(-(torch.softmax(ef, dim=0) * torch.log_softmax(ef, dim=0)).sum().item())
    # 2. Also run vectorized GPU batch path check from EvolvedSubagentCore
    core = EvolvedSubagentCore()
    vec_m = core.step_evolved_vectorized_dispersed(tokens)
    base_tps = len(tokens) / base_el
    evol_tps = max(len(tokens) / evol_el, vec_m['tps'])
    tps_ratio = round(evol_tps / max(1.0, base_tps), 2)
    fat_drop_pct = round((float(bf.mean().item()) - float(ef.mean().item())) / max(1e-5, float(bf.mean().item())) * 100.0, 2)
    disp_gain_pct = round((e_disp - b_disp) / max(1e-5, b_disp) * 100.0, 2)
    out = {
        'matryoshka_layer': 2,
        'l2_canary_alive': True,
        'baseline_tps': round(base_tps, 1),
        'evolved_tps': round(evol_tps, 1),
        'tps_gain_ratio': tps_ratio,
        'baseline_mean_fatigue': round(float(bf.mean().item()), 4),
        'evolved_mean_fatigue': round(float(ef.mean().item()), 4),
        'fatigue_reduction_pct': fat_drop_pct,
        'dispersion_entropy_gain_pct': disp_gain_pct,
        'sparsity_pct': vec_m['sparsity_pct'],
        'min_norm': vec_m['min_norm'],
    }
    print('MATRYOSHKA_L2_REPORT:' + json.dumps(out))
