#!/usr/bin/env bash
set -euo pipefail

cd /home/node/.openclaw/workspace/projects/rust_localsleep-moe

# Ensure _on_profile_changed in clean ls_moe_pyqt_gui.py also auto-reloads the LoRA pool when switching profiles
python3 - <<'PY'
from pathlib import Path
p = Path("ls_moe_pyqt_gui.py")
txt = p.read_text(encoding="utf-8")
target = """        else:
            self.inp_url.setText("http://192.168.50.67:8000")
            self.inp_model.setText("Gemma-4-31B-IT-NVFP4")
            self.spn_temp.setValue(0.6)
            self._auto_probe_server_model()"""
replacement = """        else:
            self.inp_url.setText("http://192.168.50.67:8000")
            self.inp_model.setText("Gemma-4-31B-IT-NVFP4")
            self.spn_temp.setValue(0.6)
            self._auto_probe_server_model()
        # Automatically reload the isolated LoRA pool & checkpoint when switching profiles
        self.init_engine_from_ui(reset_ckpt=False, resume_sess=self.chk_resume.isChecked())"""
if target in txt and "Automatically reload the isolated LoRA pool" not in txt:
    p.write_text(txt.replace(target, replacement, 1), encoding="utf-8")
    print("✅ Added automatic LoRA reload on profile switch to clean ls_moe_pyqt_gui.py")
PY

git add \
    ls_moe_pyqt_gui.py \
    ls_moe_pyqt_gui_gpt1900_exp.py \
    generation_engine_trt_hybrid_gpt1900_exp.py \
    configs/model_profiles_gpt1900_exp.json \
    snapshot_studio_gpt1900_exp.html \
    experiments/gpt1900_time_capsule_20261101/ignite_gpt1900_server.py \
    experiments/gpt1900_time_capsule_20261101/ignite_gpt1900_server.sh \
    experiments/gpt1900_time_capsule_20261101/ls_moe_pyqt_gui_gpt1900_exp.py \
    experiments/gpt1900_time_capsule_20261101/generation_engine_trt_hybrid_gpt1900_exp.py \
    experiments/gpt1900_time_capsule_20261101/model_profiles_gpt1900_exp.json \
    experiments/gpt1900_time_capsule_20261101/snapshot_studio_gpt1900_exp.html \
    scripts/rollback_31b_lora.sh \
    scripts/archive_gpt1900_experiment.sh \
    scripts/commit_daily_wrapup_20261102.sh \
    benchmarks/OPTIMIZATION_WITNESS_LOG.md

git commit -m "chore(exp-archive): archive GPT-1900 time-capsule experiment separately and keep main GUI pure Gemma-4 31B

- Keep main ls_moe_pyqt_gui.py, generation_engine_trt_hybrid.py, model_profiles.json, and snapshot_studio.html 100% focused on Gemma-4 31B (thor_31b) flagship research
- Auto-reload isolated LoRA pool & checkpoint when switching profiles in ls_moe_pyqt_gui.py
- Archive GPT-1900 cross-era guidance experiment under *_gpt1900_exp.* and experiments/gpt1900_time_capsule_20261101/
- Add rollback_31b_lora.sh & archive_gpt1900_experiment.sh utility scripts"

git log -n 2 --oneline
git status --short
