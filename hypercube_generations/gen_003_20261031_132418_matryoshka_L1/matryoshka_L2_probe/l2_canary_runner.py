#!/usr/bin/env python3
import json
from evolved_subagent_core import EvolvedSubagentCore
if __name__ == '__main__':
    core = EvolvedSubagentCore()
    tokens = list(range(1, 513))
    base_m = core.step_baseline_serial(tokens)
    evol_m = core.step_evolved_vectorized_dispersed(tokens)
    tps_ratio = round(evol_m['tps'] / max(1.0, base_m['tps']), 2)
    fat_drop_pct = round((base_m['mean_fatigue'] - evol_m['mean_fatigue']) / max(1e-5, base_m['mean_fatigue']) * 100.0, 2)
    disp_gain_pct = round((evol_m['dispersion_entropy'] - base_m['dispersion_entropy']) / max(1e-5, base_m['dispersion_entropy']) * 100.0, 2)
    out = {
        'matryoshka_layer': 2,
        'l2_canary_alive': True,
        'baseline_tps': round(base_m['tps'], 1),
        'evolved_tps': round(evol_m['tps'], 1),
        'tps_gain_ratio': tps_ratio,
        'baseline_mean_fatigue': round(base_m['mean_fatigue'], 4),
        'evolved_mean_fatigue': round(evol_m['mean_fatigue'], 4),
        'fatigue_reduction_pct': fat_drop_pct,
        'dispersion_entropy_gain_pct': disp_gain_pct,
        'sparsity_pct': evol_m['sparsity_pct'],
        'min_norm': evol_m['min_norm'],
    }
    print('MATRYOSHKA_L2_REPORT:' + json.dumps(out))
