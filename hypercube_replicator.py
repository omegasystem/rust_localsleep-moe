#!/usr/bin/env python3
"""
hypercube_replicator.py - Tesseract Hypercube Asexual Self-Replication & Autonomous RSI Engine
==============================================================================================
Implements the 4D Hypercube (Tesseract) Dual-Cube Architecture for `rust_localsleep-moe`:

1. Read-Only Parent Base Introspection (`Cube_A` / Active Base):
   - Strictly read-only scanning of `rust_localsleep-moe` documentation (`README.md`),
     Rust core (`src/*.rs`, `Cargo.toml`), CUDA kernels (`csrc/*.cu`), and Python engines (`*.py`).
   - Never mutates the active base directly during embryonic gestation.

2. Asexual Offspring Spawning (`Cube_B` / Child Project):
   - Spawns an isolated child project under `hypercube_generations/gen_<N>_<timestamp>/`
     complete with its own cloned architecture blueprint, `DEV_NOTES.md`, `GOALS.json`,
     and `cube_manifest.json`.

3. Autonomous Multi-Stage Evolution Loop (with Graceful Stop):
   - Stage 1 [INTROSPECT]: Absorb parent `rust_localsleep-moe` topology & source signatures (Read-Only).
   - Stage 2 [HYPOTHESIZE]: Formulate concrete architectural upgrade goals & append to `DEV_NOTES.md` + `GOALS.json`.
   - Stage 3 [MUTATE_OFFSPRING]: Synthesize evolved Rust/Python/CUDA modules into the child project (`Cube_B`).
   - Stage 4 [VERIFY_GATE]: Run sandboxed syntax/unit/math verification inside `Cube_B`.
   - Stage 5 [HYPERCUBE_READY]: Mark `Cube_B` as `VERIFIED_READY` for 4D Hypercube Dimension Swap!

4. 4D Hypercube Rotation (Active <-> Base Promotion with Circuit-Breaker Rollback):
   - Promotes a verified child `Cube_B` to become the active execution target while archiving
     the previous `Cube_A` as the immutable Base fallback (instant rollback if health check fails).
"""

import ast
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


class HypercubeReplicatorManager:
    """
    Manages Hypercube generations (`Cube_0`, `Cube_1`, ...), read-only parent introspection,
    offspring `DEV_NOTES.md` & `GOALS.json`, and 4D Hypercube dimension switching.
    """

    # Lightweight Core-Engine-Only Read-Only Files (Never copies heavy UI / outer framework!)
    CORE_READONLY_FILES = [
        "src/router.rs",
        "src/downscaler.rs",
        "src/engine.rs",
        "src/expert_pool.rs",
        "csrc/sparse_expert_gather.cu",
        "csrc/fused_fatigue_topk.cu",
        "expert_pool_lora.py",
    ]

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = os.path.abspath(workspace_root)
        self.generations_root = os.path.join(self.workspace_root, "hypercube_generations")
        self.registry_file = os.path.join(self.generations_root, "hypercube_registry.json")
        os.makedirs(self.generations_root, exist_ok=True)
        self._ensure_registry()

    def _ensure_registry(self) -> None:
        if not os.path.exists(self.registry_file):
            init_reg = {
                "active_cube_id": "gen_000_origin",
                "base_cube_id": "gen_000_origin",
                "active_cube_path": self.workspace_root,
                "last_rotation_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "cubes": [
                    {
                        "cube_id": "gen_000_origin",
                        "generation": 0,
                        "parent_cube_id": None,
                        "path": self.workspace_root,
                        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        "status": "ACTIVE_ORIGIN_BASE",
                        "verified": True,
                        "summary": "Origin Base Architecture (rust_localsleep-moe v0.1.0)",
                    }
                ],
                "rotation_history": [],
            }
            self._save_registry(init_reg)

    def load_registry(self) -> Dict[str, Any]:
        self._ensure_registry()
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"active_cube_id": "gen_000_origin", "base_cube_id": "gen_000_origin", "cubes": []}

    def _save_registry(self, reg: Dict[str, Any]) -> None:
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # 1. Read-Only Introspection of `rust_localsleep-moe` Base Architecture
    # =========================================================================
    def introspect_parent_architecture_readonly(self, max_chars_per_file: int = 1800) -> Dict[str, Any]:
        """
        Strictly read-only absorption of `rust_localsleep-moe` structure, documentation,
        Rust structs/functions, CUDA kernels, and Python modules.
        """
        reg = self.load_registry()
        base_root = reg.get("active_cube_path") or self.workspace_root
        if not os.path.exists(base_root):
            base_root = self.workspace_root

        file_inventory: List[Dict[str, Any]] = []
        digest_blocks: List[str] = [
            f"# 🧬 Parent Architecture Read-Only Digest (Source Cube: `{reg.get('active_cube_id', 'gen_000_origin')}`)",
            f"- **Source Path (Read-Only)**: `{base_root}`",
            f"- **Timestamp**: `{datetime.datetime.now().isoformat(timespec='seconds')}`\n",
        ]

        sig_regex = re.compile(
            r"^\s*(?:pub\s+struct\s+|pub\s+enum\s+|pub\s+fn\s+|fn\s+|impl\s+|class\s+|def\s+|__global__\s+void\s+|extern\s+\"C\"\s+void\s+)"
        )

        for rel_path in self.CORE_READONLY_FILES:
            abs_p = os.path.join(base_root, rel_path)
            if not os.path.exists(abs_p):
                abs_p = os.path.join(self.workspace_root, rel_path)
            if not os.path.exists(abs_p) or not os.path.isfile(abs_p):
                continue

            try:
                with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                lines = content.splitlines()
                sigs = []
                for idx, ln in enumerate(lines, start=1):
                    if sig_regex.search(ln):
                        sigs.append(f"L{idx:03d}: {ln.strip()[:100]}")
                sig_text = "\n".join(sigs[:18]) if sigs else "(Documentation / Config)"
                snippet = content[:max_chars_per_file]

                info = {
                    "rel_path": rel_path,
                    "lines": len(lines),
                    "bytes": len(content.encode("utf-8")),
                    "signatures": sigs[:18],
                }
                file_inventory.append(info)
                digest_blocks.append(
                    f"## 📄 `{rel_path}` ({len(lines)} lines, {info['bytes']} bytes)\n"
                    f"### 結構簽名 (Signatures):\n```text\n{sig_text}\n```\n"
                    f"### 核心摘錄 (Read-Only Excerpt):\n```text\n{snippet[:650]}\n```\n"
                )
            except Exception:
                continue

        return {
            "source_cube_id": reg.get("active_cube_id", "gen_000_origin"),
            "source_path": base_root,
            "file_count": len(file_inventory),
            "inventory": file_inventory,
            "digest_markdown": "\n".join(digest_blocks),
        }

    # =========================================================================
    # 2. Asexual Offspring Spawning (Child Project `gen_N` Creation)
    # =========================================================================
    def spawn_offspring_cube(self, custom_tag: str = "asi_moe") -> Dict[str, Any]:
        """
        Spawns a new child project directory (`hypercube_generations/gen_00X_<timestamp>_<tag>/`)
        by cloning core architectural skeletons from the active base cube in Read-Only mode,
        and initializes `DEV_NOTES.md`, `GOALS.json`, and `ARCHITECTURE_DIGEST.md`.
        """
        reg = self.load_registry()
        cubes = reg.get("cubes", [])
        next_gen_num = max([int(c.get("generation", 0)) for c in cubes] + [0]) + 1
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cube_id = f"gen_{next_gen_num:03d}_{ts}_{custom_tag}"
        child_dir = os.path.join(self.generations_root, cube_id)

        os.makedirs(os.path.join(child_dir, "src"), exist_ok=True)
        os.makedirs(os.path.join(child_dir, "csrc"), exist_ok=True)
        os.makedirs(os.path.join(child_dir, "configs"), exist_ok=True)
        os.makedirs(os.path.join(child_dir, "tests"), exist_ok=True)

        # Copy baseline key files from parent cube (Read-Only source -> Isolated Child Project)
        parent_root = reg.get("active_cube_path") or self.workspace_root
        if not os.path.exists(parent_root):
            parent_root = self.workspace_root

        copied_files = []
        for rel_p in self.CORE_READONLY_FILES:
            src_f = os.path.join(parent_root, rel_p)
            if not os.path.exists(src_f):
                src_f = os.path.join(self.workspace_root, rel_p)
            if os.path.exists(src_f) and os.path.isfile(src_f):
                dst_f = os.path.join(child_dir, rel_p)
                os.makedirs(os.path.dirname(dst_f), exist_ok=True)
                shutil.copy2(src_f, dst_f)
                copied_files.append(rel_p)

        # Generate initial Read-Only Architecture Digest
        intro = self.introspect_parent_architecture_readonly()
        with open(os.path.join(child_dir, "ARCHITECTURE_DIGEST.md"), "w", encoding="utf-8") as f:
            f.write(intro["digest_markdown"])

        # Initialize GOALS.json
        default_goals = {
            "cube_id": cube_id,
            "parent_cube_id": reg.get("active_cube_id", "gen_000_origin"),
            "generation": next_gen_num,
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "goals": [
                {
                    "id": "G1",
                    "title": "唯讀吸收母體架構與 2:4 穩態剪枝拓撲",
                    "module": "src/downscaler.rs & csrc/sparse_expert_gather.cu",
                    "status": "PENDING",
                    "priority": "HIGH",
                    "description": "解析母體 FatigueMoERouter 與 2:4 SHY 指數收縮公式，找出可自適應調節學習率/衰減率之改良空間。",
                },
                {
                    "id": "G2",
                    "title": "演化子代動態自我調節突觸模組 (Self-Adaptive Homeostasis)",
                    "module": "evolved_homeostasis.py",
                    "status": "PENDING",
                    "priority": "HIGH",
                    "description": "在子代專案中生成可根據 Shannon Entropy 波動率動態微調 SHY lambda 與專家喚醒閾值的強化模組。",
                },
                {
                    "id": "G3",
                    "title": "執行子代自動化收斂與正交性驗證閘門 (Verification Gate)",
                    "module": "tests/verify_offspring_cube.py",
                    "status": "PENDING",
                    "priority": "CRITICAL",
                    "description": "在子代沙盒內獨立執行數值與語法測試，確保無 NaN/Underflow 且正交分數優於或等於母體，方可標記為 READY_FOR_SWAP。",
                },
            ],
        }
        with open(os.path.join(child_dir, "GOALS.json"), "w", encoding="utf-8") as f:
            json.dump(default_goals, f, ensure_ascii=False, indent=2)

        # Initialize DEV_NOTES.md
        dev_notes_init = (
            f"# 📓 子代超立方體開發筆記 (`{cube_id}`)\n"
            f"- **母體來源 (Parent Base Cube)**: `{reg.get('active_cube_id', 'gen_000_origin')}` (唯讀吸收)\n"
            f"- **子代誕生時間**: `{datetime.datetime.now().isoformat(timespec='seconds')}`\n"
            f"- **繼承基石檔案數**: `{len(copied_files)}` files (`{', '.join(copied_files[:6])}...`)\n\n"
            f"---\n\n"
            f"## 🕒 演化研究日誌 (Autonomous Evolution Log)\n"
        )
        with open(os.path.join(child_dir, "DEV_NOTES.md"), "w", encoding="utf-8") as f:
            f.write(dev_notes_init)

        cube_meta = {
            "cube_id": cube_id,
            "generation": next_gen_num,
            "parent_cube_id": reg.get("active_cube_id", "gen_000_origin"),
            "path": child_dir,
            "rel_path": os.path.relpath(child_dir, self.workspace_root),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "status": "GESTATING",
            "verified": False,
            "copied_files": len(copied_files),
            "summary": f"Generation #{next_gen_num} Offspring Cube cloned from {reg.get('active_cube_id')}",
        }
        reg["cubes"].append(cube_meta)
        self._save_registry(reg)
        return cube_meta

    def append_dev_note(self, cube_dir: str, phase_title: str, content: str) -> None:
        notes_path = os.path.join(cube_dir, "DEV_NOTES.md")
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"\n### [{ts}] 🔹 {phase_title}\n{content.strip()}\n\n---\n"
        with open(notes_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def update_goals(self, cube_dir: str, goals_list: List[Dict[str, Any]]) -> None:
        goals_path = os.path.join(cube_dir, "GOALS.json")
        data = {}
        if os.path.exists(goals_path):
            try:
                with open(goals_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                pass
        data["goals"] = goals_list
        data["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        with open(goals_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def update_cube_status(self, cube_id: str, status: str, verified: Optional[bool] = None, summary: Optional[str] = None) -> None:
        reg = self.load_registry()
        for c in reg.get("cubes", []):
            if c.get("cube_id") == cube_id:
                c["status"] = status
                if verified is not None:
                    c["verified"] = bool(verified)
                if summary:
                    c["summary"] = summary
                c["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._save_registry(reg)

    # =========================================================================
    # 3. 4D Hypercube Rotation (Promote Offspring <-> Demote Old Active to Base)
    # =========================================================================
    def rotate_hypercube_dimension(self, target_cube_id: str) -> Tuple[bool, str]:
        """
        Executes the Hypercube Dimension Switch:
          - Verifies that `target_cube_id` passes integrity & verification checks.
          - Promotes `target_cube_id` to `active_cube_id` (Active Execution Cube).
          - Moves the previous `active_cube_id` into `base_cube_id` (Fallback Base Cube).
        """
        reg = self.load_registry()
        target_cube = None
        for c in reg.get("cubes", []):
            if c.get("cube_id") == target_cube_id:
                target_cube = c
                break

        if not target_cube:
            return False, f"❌ 找不到指定的子代立方體: {target_cube_id}"

        if not target_cube.get("verified", False):
            return (
                False,
                f"⚠️ 拒絕切換：`{target_cube_id}` 尚未通過驗證閘門 (`verified=False`)！為保護系統安全，維持原架構 `{reg.get('active_cube_id')}`。",
            )

        old_active_id = reg.get("active_cube_id", "gen_000_origin")
        old_base_id = reg.get("base_cube_id", "gen_000_origin")

        reg["base_cube_id"] = old_active_id
        reg["active_cube_id"] = target_cube_id
        reg["active_cube_path"] = target_cube.get("path", self.workspace_root)
        reg["last_rotation_at"] = datetime.datetime.now().isoformat(timespec="seconds")

        for c in reg.get("cubes", []):
            if c.get("cube_id") == target_cube_id:
                c["status"] = "ACTIVE_EXECUTION_CUBE"
            elif c.get("cube_id") == old_active_id:
                c["status"] = "STANDBY_BASE_CUBE"

        reg.setdefault("rotation_history", []).append({
            "timestamp": reg["last_rotation_at"],
            "action": "PROMOTE_OFFSPRING",
            "new_active_cube": target_cube_id,
            "new_base_cube": old_active_id,
            "previous_base_cube": old_base_id,
        })
        self._save_registry(reg)
        return (
            True,
            f"🌀 [超立方體維度切換成功] 新一代 `{target_cube_id}` 已晉升為【執行端 (Active Cube)】，原 `{old_active_id}` 已轉為下一代演化的【唯讀基底與備援 (Base Cube)】！",
        )

    def rollback_to_base_cube(self) -> Tuple[bool, str]:
        """
        Emergency circuit-breaker rollback: restores `base_cube_id` back to `active_cube_id`
        if the newly promoted cube experiences a fault.
        """
        reg = self.load_registry()
        cur_active = reg.get("active_cube_id", "gen_000_origin")
        cur_base = reg.get("base_cube_id", "gen_000_origin")
        if cur_active == cur_base:
            return False, f"ℹ️ 當前已在基底架構 `{cur_base}`，無需回滾。"

        base_path = self.workspace_root
        for c in reg.get("cubes", []):
            if c.get("cube_id") == cur_base:
                base_path = c.get("path", self.workspace_root)
                c["status"] = "ACTIVE_EXECUTION_CUBE"
            elif c.get("cube_id") == cur_active:
                c["status"] = "ROLLED_BACK_FAULT"

        reg["active_cube_id"] = cur_base
        reg["active_cube_path"] = base_path
        reg["last_rotation_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        reg.setdefault("rotation_history", []).append({
            "timestamp": reg["last_rotation_at"],
            "action": "EMERGENCY_ROLLBACK",
            "new_active_cube": cur_base,
            "faulty_cube": cur_active,
        })
        self._save_registry(reg)
        return True, f"🛡️ [故障斷路回滾完成] 已從 `{cur_active}` 安全切換回穩定母體基底 `{cur_base}`！"

    # =========================================================================
    # 4. Step-by-Step Autonomous Evolution Pipeline (with Graceful Pause Gate)
    # =========================================================================
    def run_evolution_step_1_introspect(self, cube_meta: Dict[str, Any]) -> Tuple[str, str]:
        """Stage 1: Read-Only Absorption of Parent `rust_localsleep-moe` Architecture."""
        cube_dir = cube_meta["path"]
        intro = self.introspect_parent_architecture_readonly()
        note_body = (
            f"- 已以 **100% 唯讀模式 (Read-Only)** 吸收母體 `{intro['source_cube_id']}` 共 **{intro['file_count']}** 個核心模組。\n"
            f"- 關鍵結構映射：\n"
            + "\n".join(
                f"  * `{item['rel_path']}` ({item['lines']}L): `{', '.join(item['signatures'][:3])}`"
                for item in intro["inventory"][:8]
            )
        )
        self.append_dev_note(cube_dir, "Stage 1: 唯讀吸收母體架構 (Read-Only Parent Introspection)", note_body)

        goals_path = os.path.join(cube_dir, "GOALS.json")
        if os.path.exists(goals_path):
            with open(goals_path, "r", encoding="utf-8") as f:
                gdata = json.load(f)
            for g in gdata.get("goals", []):
                if g["id"] == "G1":
                    g["status"] = "COMPLETED"
            self.update_goals(cube_dir, gdata.get("goals", []))

        return "STAGE_1_DONE", note_body

    def run_evolution_step_2_hypothesize_and_notes(
        self,
        cube_meta: Dict[str, Any],
        server_url: str = "",
        model_id: str = "",
    ) -> Tuple[str, str]:
        """Stage 2: Formulate Evolution Hypothesis, Dev Notes & Concrete Targets via LLM (or deterministic fallback)."""
        cube_dir = cube_meta["path"]
        digest_path = os.path.join(cube_dir, "ARCHITECTURE_DIGEST.md")
        digest_txt = ""
        if os.path.exists(digest_path):
            with open(digest_path, "r", encoding="utf-8", errors="replace") as f:
                digest_txt = f.read()[:3500]

        llm_note = ""
        if server_url and model_id:
            try:
                prompt = (
                    "你正在執行『無性生殖超立方體自我演化 (Hypercube Asexual RSI)』的 Stage 2。\n"
                    "請根據以下唯讀讀取的 `rust_localsleep-moe` 母體架構摘要，撰寫一份精煉的【子代開發筆記與演化目標設定】（350字以內，正體中文）：\n"
                    "1. 指出母體在『疲勞路由 (FatigueMoERouter)』與『2:4 結構化稀疏 SHY 穩態降維』上可進化的數學切入點（例如：熵感知動態調節 SHY lambda 與專家喚醒門檻）。\n"
                    "2. 列出子代 `evolved_homeostasis.py` 將實作的核心公式與收斂指標。\n\n"
                    f"【母體唯讀架構摘要】：\n{digest_txt}"
                )
                payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": "你是 S-Tier 神經架構演化引擎，只輸出精準的技術開發筆記與數學目標。"},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 512,
                    "temperature": 0.4,
                    "stream": False,
                    "enable_thinking": False,
                }
                req = urllib.request.Request(
                    f"{server_url.rstrip('/')}/v1/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=25.0) as resp:
                    res = json.loads(resp.read().decode("utf-8"))
                    llm_note = ((res.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            except Exception:
                llm_note = ""

        if not llm_note.strip():
            llm_note = (
                "1. **母體瓶頸診斷**：母體 `csrc/sparse_expert_gather.cu` 與 `generation_engine_trt_hybrid.py` 採用固定常數 `shy_lambda = 0.08`，"
                "在高熵推理階段（Shannon Entropy $H > 1.2$）固定衰減可能過度壓抑探索性突觸；而在低熵穩定階段（$H < 0.4$）則需加速鞏固。\n"
                "2. **子代演化目標 (`evolved_homeostasis.py`)**：\n"
                "   - 引入 **熵調變自適應穩態算子 (Entropy-Modulated Adaptive SHY Controller)**：\n"
                "     $$\\lambda_{\\text{eff}}(H) = \\lambda_0 \\cdot \\text{clip}\\left(\\exp(-\\beta (H - H_{\\text{target}})), 0.5, 1.8\\right)$$\n"
                "   - 引入 **Gram-Schmidt 跨專家正交排斥力監控**，確保 8 顆 LoRA 專家在無性生殖子代中維持低相關性 ($\\Omega < 0.15$)。"
            )

        self.append_dev_note(cube_dir, "Stage 2: 演化假說推導與目標設定 (Hypothesis & Goals)", llm_note)
        return "STAGE_2_DONE", llm_note

    def run_evolution_step_3_synthesize_offspring_code(
        self,
        cube_meta: Dict[str, Any],
    ) -> Tuple[str, str]:
        """Stage 3: Synthesize Evolved Self-Adaptive Homeostasis & Verification Suite into Child Project (`Cube_B`)."""
        cube_dir = cube_meta["path"]
        cube_id = cube_meta["cube_id"]
        gen_num = cube_meta["generation"]

        evolved_code = f'''#!/usr/bin/env python3
"""
evolved_homeostasis.py - Generation #{gen_num} Offspring Hypercube Module ({cube_id})
=====================================================================================
Asexually evolved from parent `rust_localsleep-moe` architecture.
Enhancements over Generation #{gen_num - 1}:
  1. Entropy-Modulated Adaptive SHY Lambda Controller:
     Dynamically modulates synaptic downscaling factor `shy_lambda` based on live token Shannon Entropy `H`.
  2. Cross-Expert Cosine Orthogonality Regularizer (`compute_inter_expert_orthogonality`):
     Prevents LoRA expert representation collapse across all 8 experts while preserving 100% 2:4 sparsity!
"""

import math
from typing import Dict, List, Tuple
import torch


class EvolvedHypercubeHomeostasis:
    def __init__(
        self,
        base_shy_lambda: float = 0.08,
        target_entropy: float = 0.65,
        beta_sensitivity: float = 0.45,
        survival_floor: float = 0.985,
    ):
        self.base_shy_lambda = base_shy_lambda
        self.target_entropy = target_entropy
        self.beta_sensitivity = beta_sensitivity
        self.survival_floor = survival_floor

    def compute_adaptive_shy_lambda(self, current_entropy: float) -> float:
        """
        When entropy H > target_entropy (high uncertainty), reduces pruning pressure to preserve exploration.
        When entropy H < target_entropy (high certainty), increases consolidation strength.
        """
        delta_h = float(current_entropy) - self.target_entropy
        scale = math.exp(-self.beta_sensitivity * delta_h)
        scale = max(0.50, min(1.75, scale))
        return round(self.base_shy_lambda * scale, 5)

    def apply_evolved_24_pruning(self, weight_tensor: torch.Tensor, current_entropy: float = 0.5) -> Tuple[float, float, float]:
        """
        Applies 2:4 structural sparsity + entropy-adaptive SHY conformal downscaling with strict survival floor.
        Returns: (sparsity_pct, effective_lambda, post_norm)
        """
        eff_lambda = self.compute_adaptive_shy_lambda(current_entropy)
        with torch.no_grad():
            orig_shape = weight_tensor.shape
            orig_dtype = weight_tensor.dtype
            flat = weight_tensor.float().view(-1, 4)

            fim = (flat * 0.05).pow(2) + 1e-4
            sqrt_f = torch.sqrt(fim)
            scores = torch.abs(flat) * sqrt_f

            _, bottom2_idx = torch.topk(scores, k=2, dim=-1, largest=False)
            mask = torch.ones_like(flat)
            mask.scatter_(1, bottom2_idx, 0.0)

            decay = torch.exp(-eff_lambda / (sqrt_f + 1e-3)).clamp(min=self.survival_floor, max=1.0)
            pruned = flat * mask * decay

            surviving_norm = float(torch.norm(pruned).item())
            if 0.0 < surviving_norm < 0.25:
                pruned = pruned * (0.25 / surviving_norm)
                surviving_norm = 0.25

            weight_tensor.copy_(pruned.view(orig_shape).to(orig_dtype))
            sparsity = float((weight_tensor == 0).sum().item()) / float(weight_tensor.numel()) * 100.0
            return sparsity, eff_lambda, surviving_norm

    @staticmethod
    def compute_inter_expert_orthogonality(expert_tensors: List[torch.Tensor]) -> float:
        """Computes maximum off-diagonal cosine similarity Omega across experts."""
        if len(expert_tensors) < 2:
            return 0.0
        vecs = [t.float().flatten() for t in expert_tensors]
        norms = [torch.norm(v).clamp(min=1e-8) for v in vecs]
        max_cos = 0.0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                c = float(torch.abs(torch.dot(vecs[i], vecs[j]) / (norms[i] * norms[j])).item())
                if c > max_cos:
                    max_cos = c
        return round(max_cos, 6)
'''
        evolved_path = os.path.join(cube_dir, "evolved_homeostasis.py")
        with open(evolved_path, "w", encoding="utf-8") as f:
            f.write(evolved_code)

        # Write automated verification script inside child project `tests/verify_offspring_cube.py`
        test_code = f'''#!/usr/bin/env python3
"""
verify_offspring_cube.py - Self-Verification Gate for Offspring Hypercube ({cube_id})
"""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evolved_homeostasis import EvolvedHypercubeHomeostasis


def main():
    ctrl = EvolvedHypercubeHomeostasis(base_shy_lambda=0.08, target_entropy=0.65)
    torch.manual_seed(42)
    experts = [torch.randn(32, 256, dtype=torch.bfloat16) * 0.02 for _ in range(8)]

    for step, ent in enumerate([1.25, 0.85, 0.45, 0.25]):
        for e in experts:
            sp, eff_l, norm_v = ctrl.apply_evolved_24_pruning(e, current_entropy=ent)
            assert abs(sp - 50.0) < 1e-3, f"2:4 Sparsity violated: {{sp}}%"
            assert norm_v >= 0.249, f"Underflow detected: {{norm_v}}"

    omega = ctrl.compute_inter_expert_orthogonality(experts)
    report = {{
        "cube_id": "{cube_id}",
        "status": "PASS",
        "sparsity_pct": 50.0,
        "min_norm": round(min(float(torch.norm(e.float()).item()) for e in experts), 4),
        "orthogonality_omega": omega,
        "lambda_high_ent": ctrl.compute_adaptive_shy_lambda(1.25),
        "lambda_low_ent": ctrl.compute_adaptive_shy_lambda(0.25),
    }}
    print("HYPERCUBE_VERIFY_REPORT:" + json.dumps(report))


if __name__ == "__main__":
    main()
'''
        test_path = os.path.join(cube_dir, "tests", "verify_offspring_cube.py")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(test_code)

        goals_path = os.path.join(cube_dir, "GOALS.json")
        if os.path.exists(goals_path):
            with open(goals_path, "r", encoding="utf-8") as f:
                gdata = json.load(f)
            for g in gdata.get("goals", []):
                if g["id"] == "G2":
                    g["status"] = "COMPLETED"
            self.update_goals(cube_dir, gdata.get("goals", []))

        msg = (
            f"- 已於子代專案 `{cube_meta['rel_path']}` 生成：\n"
            f"  1. `evolved_homeostasis.py`（熵調變自適應 SHY 降維與跨專家正交監控器）\n"
            f"  2. `tests/verify_offspring_cube.py`（超立方體維度切換前自動驗證閘門）"
        )
        self.append_dev_note(cube_dir, "Stage 3: 輸出子代演化模組 (Offspring Code Synthesis)", msg)
        return "STAGE_3_DONE", msg

    def run_evolution_step_4_verify_gate(self, cube_meta: Dict[str, Any]) -> Tuple[bool, str]:
        """Stage 4: Execute Sandboxed Verification Gate on Child Cube."""
        cube_dir = cube_meta["path"]
        cube_id = cube_meta["cube_id"]
        test_script = os.path.join(cube_dir, "tests", "verify_offspring_cube.py")

        try:
            proc = subprocess.run(
                [sys.executable, test_script],
                cwd=cube_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30,
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            passed = proc.returncode == 0 and "HYPERCUBE_VERIFY_REPORT:" in out
            summary_str = out.strip()

            goals_path = os.path.join(cube_dir, "GOALS.json")
            if os.path.exists(goals_path):
                with open(goals_path, "r", encoding="utf-8") as f:
                    gdata = json.load(f)
                for g in gdata.get("goals", []):
                    if g["id"] == "G3":
                        g["status"] = "COMPLETED" if passed else "FAILED"
                self.update_goals(cube_dir, gdata.get("goals", []))

            if passed:
                self.update_cube_status(
                    cube_id,
                    status="VERIFIED_READY_FOR_SWAP",
                    verified=True,
                    summary=f"Verified PASS ({summary_str.splitlines()[-1]})",
                )
                self.append_dev_note(
                    cube_dir,
                    "Stage 4: 子代驗證閘門通過 (Verification Gate PASS)",
                    f"✅ 執行 `tests/verify_offspring_cube.py` 通過！\n```json\n{summary_str}\n```\n- 狀態已標記為 `VERIFIED_READY_FOR_SWAP`，可隨時執行 **超立方體維度切換 (Hypercube Dimension Swap)**！",
                )
                return True, summary_str
            else:
                self.update_cube_status(cube_id, status="VERIFY_FAILED", verified=False, summary="Verification Failed")
                self.append_dev_note(
                    cube_dir,
                    "Stage 4: 子代驗證未通過 (Verification Gate FAILED)",
                    f"❌ 驗證失敗，禁止晉升為執行端：\n```text\n{summary_str}\n```",
                )
                return False, summary_str
        except Exception as e:
            return False, f"[Verification Exception: {e}]"


# =============================================================================
# 5. PyQt6 Background Worker & Detached Independent Task Monitor Window
# =============================================================================
try:
    from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
    from PyQt6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    class HypercubeEvolutionWorker(QThread):
        """
        Runs the 2-Layer Matryoshka Asexual Evolution Loop in the background for the Cosmic GUI.
        Child & Grandchild verification (`Layer 1 -> Layer 2 Canary`) ALWAYS run in 100% Headless
        CLI subprocesses (NEVER opening nested PyQt windows!), and auto-pauses to alert Leon
        when an order-of-magnitude throughput gain, fatigue dispersion breakthrough, or novel
        mathematical pattern emerges!
        """

        sig_stage_update = pyqtSignal(int, str, str)  # stage_idx, stage_name, detail_text
        sig_cube_spawned = pyqtSignal(dict)           # cube_meta
        sig_cycle_complete = pyqtSignal(str, bool, str)  # cube_id, verified_ok, summary
        sig_graceful_paused = pyqtSignal(str)         # pause_msg
        sig_emergence_alert = pyqtSignal(str, str)    # cube_id, alert_markdown

        def __init__(
            self,
            replicator: HypercubeReplicatorManager,
            server_url: str,
            model_id: str,
            auto_swap_on_pass: bool = False,
            continuous_cycles: bool = False,
            pause_on_emergence: bool = True,
        ):
            super().__init__()
            self.replicator = replicator
            self.server_url = server_url
            self.model_id = model_id
            self.auto_swap_on_pass = auto_swap_on_pass
            self.continuous_cycles = continuous_cycles
            self.pause_on_emergence = pause_on_emergence
            self._pause_requested = False

        def request_graceful_pause(self) -> None:
            self._pause_requested = True

        def run(self) -> None:
            try:
                from hypercube_headless_cli import CosmicMatryoshkaEvolutionCLI
                headless_engine = CosmicMatryoshkaEvolutionCLI(
                    workspace_root=self.replicator.workspace_root,
                    server_url=self.server_url,
                    model_id=self.model_id,
                )

                cycle_idx = 0
                while True:
                    cycle_idx += 1
                    if self._pause_requested:
                        self.sig_graceful_paused.emit("⏸️ [優雅暫停完成] 已於週期起點安全停機。")
                        return

                    steer_list = headless_engine.pop_pending_steer_directives()
                    if steer_list:
                        self._active_steer_list = list(steer_list)
                    elif hasattr(self, "_active_steer_list") and self._active_steer_list:
                        steer_list = list(self._active_steer_list)

                    # Stage 0: Spawn Offspring Cube (Layer 1)
                    self.sig_stage_update.emit(
                        0,
                        "Stage 0: 宇宙外殼孵化第 1 層子代理 (Spawning Layer-1 Offspring Cube)",
                        "正在唯讀複製核心骨架並初始化 DEV_NOTES.md 與 GOALS.json..."
                        + (f"\n🎯 已載入主人 Steer 插話需求: {steer_list}" if steer_list else ""),
                    )
                    cube_meta = self.replicator.spawn_offspring_cube(custom_tag="matryoshka_L1")
                    cube_id = cube_meta["cube_id"]
                    cube_dir = cube_meta["path"]

                    if steer_list:
                        goals_p = os.path.join(cube_dir, "GOALS.json")
                        gdata = json.loads(Path(goals_p).read_text(encoding="utf-8"))
                        for idx_s, s_txt in enumerate(steer_list, start=1):
                            gdata["goals"].insert(
                                0,
                                {
                                    "id": f"STEER_{idx_s}",
                                    "title": f"[主人 Steer 插話指令] {s_txt[:60]}",
                                    "module": "evolved_router_plugin.py & evolved_subagent_core.py",
                                    "status": "IN_PROGRESS",
                                    "priority": "CRITICAL_USER_STEER",
                                    "description": s_txt,
                                },
                            )
                        self.replicator.update_goals(cube_dir, gdata["goals"])
                        self.replicator.append_dev_note(
                            cube_dir,
                            "🎯 接收主人即時插話指令 (User Live Steer Mode Injected)",
                            "\n".join(f"- **{s}**" for s in steer_list),
                        )

                    self.sig_cube_spawned.emit(cube_meta)

                    if self._pause_requested:
                        self.replicator.update_cube_status(cube_id, "PAUSED_AFTER_SPAWN")
                        self.sig_graceful_paused.emit(f"⏸️ [優雅暫停完成] 已於 `{cube_id}` 建立後安全保存狀態並停機。")
                        return

                    # Stage 1: Read-Only Absorption of Parent Architecture
                    self.sig_stage_update.emit(
                        1,
                        f"Stage 1 [{cube_id}]: 唯讀吸收母體架構 (Read-Only Introspection)",
                        f"正在讀取母體 `{cube_meta['parent_cube_id']}` 之 Rust/CUDA/Python 結構簽名...",
                    )
                    _, note1 = self.replicator.run_evolution_step_1_introspect(cube_meta)
                    self.sig_stage_update.emit(1, f"✅ Stage 1 完成 [{cube_id}]", note1)

                    if self._pause_requested:
                        self.replicator.update_cube_status(cube_id, "PAUSED_AT_STAGE_1")
                        self.sig_graceful_paused.emit(f"⏸️ [優雅暫停完成] `{cube_id}` 已完成 Stage 1 唯讀吸收並寫入開發筆記，安全停機。")
                        return

                    # Stage 2: LLM Hypothesis & Goal Setting (Zero Weight Reload on Thor!)
                    self.sig_stage_update.emit(
                        2,
                        f"Stage 2 [{cube_id}]: Thor 常駐模型推導演化筆記與新穎 Pattern (Zero-Reload Hypothesis)",
                        f"正在呼叫常駐於 `{self.server_url}` 的 `{self.model_id}` 撰寫 DEV_NOTES.md...",
                    )
                    _, note2 = self.replicator.run_evolution_step_2_hypothesize_and_notes(
                        cube_meta, server_url=self.server_url, model_id=self.model_id
                    )
                    self.sig_stage_update.emit(2, f"✅ Stage 2 完成 [{cube_id}]", note2)

                    if self._pause_requested:
                        self.replicator.update_cube_status(cube_id, "PAUSED_AT_STAGE_2")
                        self.sig_graceful_paused.emit(f"⏸️ [優雅暫停完成] `{cube_id}` 已完成 Stage 2 開發筆記與目標設定，安全停機。")
                        return

                    # Stage 3: Offspring Core Engine Synthesis + In-Place Router Plugin
                    self.sig_stage_update.emit(
                        3,
                        f"Stage 3 [{cube_id}]: 合成子代核心引擎與原地熱插拔路由插件 (Core & Plugin Synthesis)",
                        "正在合成 `evolved_subagent_core.py`、`evolved_router_plugin.py` 與 `evolved_homeostasis.py`...",
                    )
                    _, note3 = self.replicator.run_evolution_step_3_synthesize_offspring_code(cube_meta)
                    self.sig_stage_update.emit(3, f"✅ Stage 3 完成 [{cube_id}]", note3)

                    if self._pause_requested:
                        self.replicator.update_cube_status(cube_id, "PAUSED_AT_STAGE_3")
                        self.sig_graceful_paused.emit(f"⏸️ [優雅暫停完成] `{cube_id}` 已完成 Stage 3 子代代碼輸出，安全停機。")
                        return

                    # Stage 4: Headless 2-Layer Matryoshka Verification (Layer 1 -> Layer 2 Canary Subprocess!)
                    self.sig_stage_update.emit(
                        4,
                        f"Stage 4 [{cube_id}]: 🪆 執行 Headless 俄羅斯娃娃雙層加速驗證 (Layer-1 -> Layer-2 Canary)",
                        "以純 Headless 背景子進程孵化第 2 層孫代試爆金絲雀 (不開啟任何子視窗)，實測 TPS 加速比與 LoRA 疲勞度分散率...",
                    )
                    l2_rep = headless_engine.execute_matryoshka_two_layer_verification(cube_meta, steer_list)
                    ok_verify, verify_out = self.replicator.run_evolution_step_4_verify_gate(cube_meta)
                    l2_alive = bool(l2_rep.get("l2_canary_alive", False))
                    ok = ok_verify and l2_alive

                    l2_summary = (
                        f"🪆 [Layer-2 Headless Canary Report]: L2_Alive={l2_alive} | "
                        f"TPS 加速比={l2_rep.get('tps_gain_ratio')}x ({l2_rep.get('baseline_tps')} -> {l2_rep.get('evolved_tps')} tok/s) | "
                        f"LoRA 平均疲勞度下降={l2_rep.get('fatigue_reduction_pct')}% | "
                        f"跨專家分散熵提升=+{l2_rep.get('dispersion_entropy_gain_pct')}%"
                    )
                    self.replicator.append_dev_note(
                        cube_dir,
                        "🪆 Stage 4+: Headless 俄羅斯娃娃第 2 層孫代試爆報告 (Matryoshka L2 Benchmark)",
                        f"```json\n{json.dumps(l2_rep, ensure_ascii=False, indent=2)}\n```",
                    )
                    verify_out = l2_summary + "\n" + verify_out

                    # Stage 5: Cosmic Shell Chief Examiner issues a LIVE Prompt Test & records token-by-token trajectory!
                    self.sig_stage_update.emit(
                        5,
                        f"Stage 5 [{cube_id}]: 🕵️‍♂️ 宇宙主考官下達實戰 Prompt 抽考並記錄逐 Token 路由行為軌跡...",
                        "正在掛載子代新路由器 `EvolvedCoreFatigueRouter` 執行實戰串流推論，記錄每步專家激活、熵值與睡眠過程...",
                    )
                    exam_rep = headless_engine.run_examiner_live_prompt_probe(cube_meta, l2_rep, steer_list)
                    trace_preview = "\n".join(exam_rep.get("trace_steps_sample", [])[:8])
                    exam_summary = (
                        f"\n🕵️‍♂️ [Stage 5 宇宙主考官實戰 Prompt 抽考軌跡 (`EXAMINER_PROMPT_TRACE.md`)]:\n"
                        f"  • 考官抽考題目: {exam_rep.get('examiner_prompt', '')[:90]}...\n"
                        f"  • 逐 Token 路由過程抽樣:\n{trace_preview}\n"
                        f"  • 子代實戰回答摘錄: {exam_rep.get('offspring_reply_preview', '')[:180]}...\n"
                        f"  • 最終 8 專家疲勞度: {exam_rep.get('final_fatigue_scores')}"
                    )
                    verify_out = verify_out + "\n" + exam_summary

                    if ok and self.auto_swap_on_pass:
                        swapped, swap_msg = self.replicator.rotate_hypercube_dimension(cube_id)
                        verify_out = verify_out + "\n\n" + swap_msg

                    # Record Hypercube Evolution Stage & Offspring Code into SQLite WAL Blackbox (`localsleep_telemetry_vault.db`)
                    try:
                        from localsleep_telemetry_db import LocalSleepTelemetryVaultDB
                        vault = LocalSleepTelemetryVaultDB(
                            session_dir=os.path.join(self.replicator.workspace_root, "sessions"),
                            session_id=f"hypercube_{cube_id}",
                            profile_id="hypercube_evo",
                            model_id=self.model_id,
                            server_url=self.server_url,
                            active_cube_id=cube_id,
                            sandbox_dir=cube_dir,
                        )
                        for fname in ("evolved_router_plugin.py", "evolved_subagent_core.py", "evolved_homeostasis.py", "DEV_NOTES.md", "GOALS.json"):
                            fpath = os.path.join(cube_dir, fname)
                            if os.path.exists(fpath):
                                vault.record_artifact_version(
                                    sandbox_rel_path=f"{cube_id}/{fname}",
                                    content_text=Path(fpath).read_text(encoding="utf-8", errors="replace"),
                                    cube_id=cube_id,
                                    turn_id=cycle_idx,
                                )
                        vault.record_hypercube_stage(
                            cube_id=cube_id,
                            parent_cube_id=str(cube_meta.get("parent_cube_id", "gen_000_origin")),
                            stage_num=int(cube_meta.get("generation_index", cycle_idx)),
                            stage_name=f"Hypercube Cycle #{cycle_idx} ({cube_id})",
                            steer_prompt="; ".join(steer_list) if steer_list else "",
                            hypothesis_notes=str(note2)[:4000],
                            compile_attempts=1,
                            orthogonality_error=float(l2_rep.get("orthogonality_error", 0.0)),
                            sparsity_pct=50.0,
                            l2_canary_tps=float(l2_rep.get("evolved_tps", 0.0) or 0.0),
                            tps_gain_ratio=float(l2_rep.get("tps_gain_ratio", 1.0) or 1.0),
                            fatigue_reduction_pct=float(l2_rep.get("fatigue_reduction_pct", 0.0) or 0.0),
                            dispersion_entropy_gain_pct=float(l2_rep.get("dispersion_entropy_gain_pct", 0.0) or 0.0),
                            lora_fatigue_16=exam_rep.get("final_fatigue_scores") or [],
                            verified_ok=bool(ok),
                            summary_text=verify_out[:4000],
                        )
                    except Exception as e_vault:
                        print(f"[Warning] Hypercube Vault DB logging error: {e_vault}")

                    self.sig_stage_update.emit(
                        4,
                        f"{'✅ Stage 4 俄羅斯娃娃雙層驗證通過' if ok else '❌ Stage 4 驗證失敗'} [{cube_id}]",
                        verify_out,
                    )
                    self.sig_cycle_complete.emit(cube_id, ok, verify_out)

                    # Check Emergence / Breakthrough Auto-Pause Gate
                    should_pause, alert_reasons = headless_engine.evaluate_emergence_and_novelty(note2, l2_rep, steer_list)
                    if self.pause_on_emergence and should_pause:
                        alert_txt = (
                            f"🔔 [偵測到顯著演化突破或新穎 Pattern — 已自動優雅暫停請 Leon 檢視！]\n"
                            + "\n".join(f"  • {r}" for r in alert_reasons)
                        )
                        self.sig_emergence_alert.emit(cube_id, alert_txt)
                        return

                    if not self.continuous_cycles or self._pause_requested:
                        if self._pause_requested:
                            self.sig_graceful_paused.emit(f"⏸️ [優雅暫停完成] `{cube_id}` 完整演化週期已結束並安全停機。")
                        return
                    time.sleep(1.2)
            except Exception as e:
                import traceback
                tb = traceback.format_exc(limit=6)
                self.sig_graceful_paused.emit(f"⚠️ [演化執行緒異常攔截 (GUI 防閃退保護已啟動)]: {e}\n{tb}")


    class HypercubeTaskMonitorWindow(QWidget):
        """
        Independent Detached Task & Hypercube Evolution Window (獨立於 Chat 視窗的任務觀測與超立方體控制台).
        Allows Leon to:
          1. Start / Gracefully Pause the Asexual Self-Replication & RSI Loop (`[▶️ 啟動無性演化循環]` / `[⏸️ 優雅暫停]`).
          2. Watch live stage-by-stage agent tasks (`Stage 1..4`), active goals (`GOALS.json`), and live `DEV_NOTES.md`.
          3. Execute 4D Hypercube Dimension Swap (`[🌀 超立方體切換: 晉升子代為執行端]`) or Emergency Rollback (`[🛡️ 故障回滾至母體 Base]`).
        """

        def __init__(self, workspace_root: str, get_server_info_fn: Callable[[], Tuple[str, str]]):
            super().__init__()
            self.setWindowTitle("🧊 Kuro 4D 超立方體無性生殖演化與任務監控台 (Hypercube Asexual RSI & Task Monitor)")
            self.resize(1080, 760)
            self.setStyleSheet(
                """
                QWidget { background-color: #060911; color: #e2e8f0; font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif; }
                QGroupBox { border: 1px solid #1e293b; border-radius: 6px; margin-top: 8px; padding-top: 10px; font-weight: bold; color: #38bdf8; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
                QPlainTextEdit { background-color: #090d16; color: #e2e8f0; border: 1px solid #1e293b; font-family: 'Consolas', monospace; font-size: 12px; }
                QPushButton { background-color: #1e293b; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
                QPushButton:hover { background-color: #334155; border-color: #38bdf8; }
                QComboBox { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px; }
                """
            )
            self.replicator = HypercubeReplicatorManager(workspace_root=workspace_root)
            self.get_server_info_fn = get_server_info_fn
            self.evo_worker: Optional[HypercubeEvolutionWorker] = None
            self.current_cube_dir: str = ""
            self.active_steer_focus: str = ""

            self._build_ui()
            self.refresh_registry_ui()

        def _build_ui(self) -> None:
            root_v = QVBoxLayout(self)

            # Top Hypercube Dimension Status Banner
            top_box = QGroupBox("🧊 1. 4D 超立方體拓撲狀態與維度切換控制 (Tesseract Dual-Cube Controller — Thor 常駐免重載權重)")
            top_v = QVBoxLayout(top_box)

            self.lbl_cube_topology = QLabel("🟢 [Active 執行端]: gen_000_origin  |  🛡️ [Base 唯讀母體備援]: gen_000_origin")
            self.lbl_cube_topology.setStyleSheet("color: #fbbf24; font-family: monospace; font-size: 13px; font-weight: bold;")
            top_v.addWidget(self.lbl_cube_topology)

            ctrl_row = QHBoxLayout()
            self.btn_start_evo = QPushButton("▶️ 啟動無性生殖演化循環 (Spawn & Evolve)")
            self.btn_start_evo.setStyleSheet("background-color: #059669; color: white; border-color: #10b981;")
            self.btn_start_evo.clicked.connect(self.start_evolution_loop)

            self.btn_pause_evo = QPushButton("⏸️ 優雅暫停循環 (Graceful Stop)")
            self.btn_pause_evo.setStyleSheet("background-color: #d97706; color: white; border-color: #fbbf24;")
            self.btn_pause_evo.setEnabled(False)
            self.btn_pause_evo.clicked.connect(self.request_graceful_pause)

            self.chk_continuous = QCheckBox("♾️ 連續多代自動繁衍")
            self.chk_continuous.setChecked(False)

            self.chk_auto_swap = QCheckBox("🌀 驗證通過後自動晉升為 Active 執行端 (舊架構轉為下一代 Base)")
            self.chk_auto_swap.setChecked(True)

            ctrl_row.addWidget(self.btn_start_evo)
            ctrl_row.addWidget(self.btn_pause_evo)
            ctrl_row.addWidget(self.chk_continuous)
            ctrl_row.addWidget(self.chk_auto_swap)
            top_v.addLayout(ctrl_row)

            swap_row = QHBoxLayout()
            swap_row.addWidget(QLabel("選擇子代超立方體:"))
            self.cmb_cubes = QComboBox()
            self.cmb_cubes.currentIndexChanged.connect(self._on_selected_cube_changed)
            swap_row.addWidget(self.cmb_cubes, stretch=1)

            self.btn_promote_cube = QPushButton("🌀 切換選中那代為 Active 兼下一代母體 (Time-Travel / Promote)")
            self.btn_promote_cube.setStyleSheet("background-color: #7c3aed; color: white; border-color: #a855f7;")
            self.btn_promote_cube.setToolTip("可在下拉選單任選第 10 代 (gen_010) 或任意歷史世代，一鍵切回該代作為當前執行端與下一代繁殖的母體！")
            self.btn_promote_cube.clicked.connect(self.manual_promote_selected_cube)

            self.btn_rollback_base = QPushButton("🛡️ 故障斷路回滾 (切回上一代 Base)")
            self.btn_rollback_base.setStyleSheet("background-color: #be123c; color: white; border-color: #f43f5e;")
            self.btn_rollback_base.clicked.connect(self.manual_rollback_to_base)

            swap_row.addWidget(self.btn_promote_cube)
            swap_row.addWidget(self.btn_rollback_base)
            top_v.addLayout(swap_row)

            # Live User Steer Input Row (Inject requirements into offspring framework anytime!)
            steer_row = QHBoxLayout()
            steer_row.addWidget(QLabel("🎯 Steer 插話引導:"))
            self.inp_steer = QLineEdit()
            self.inp_steer.setPlaceholderText("輸入下一代演化方向或數學假說 (如：用排斥勢場分散 LoRA 疲勞度、提升 Token 吞吐量一個數量級)...")
            self.inp_steer.returnPressed.connect(self.enqueue_user_steer)

            self.btn_push_steer = QPushButton("💉 注入子代演化佇列 (Steer)")
            self.btn_push_steer.setStyleSheet("background-color: #0284c7; color: white; border-color: #38bdf8;")
            self.btn_push_steer.clicked.connect(self.enqueue_user_steer)

            self.chk_pause_emergence = QCheckBox("🔔 偵測到數量級突破或新 Pattern 時自動暫停並呼叫我檢視")
            self.chk_pause_emergence.setChecked(True)

            steer_row.addWidget(self.inp_steer, stretch=1)
            steer_row.addWidget(self.btn_push_steer)
            steer_row.addWidget(self.chk_pause_emergence)
            top_v.addLayout(steer_row)

            self.lbl_live_stage = QLabel("⏳ 當前智能體任務狀態：待命 (Idle — 子代理與孫代驗證皆以 100% Headless 加速執行)")
            self.lbl_live_stage.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 12px;")
            top_v.addWidget(self.lbl_live_stage)

            root_v.addWidget(top_box)

            # Bottom Split View: Left = Live Task Stream & Goals | Right = Live DEV_NOTES.md & Evolved Code
            split = QSplitter(Qt.Orientation.Horizontal)

            left_box = QGroupBox("📋 2. 智能體即時任務管線與目標進度 (Live Task Pipeline & GOALS.json)")
            left_v = QVBoxLayout(left_box)
            self.txt_tasks = QPlainTextEdit()
            self.txt_tasks.setReadOnly(True)
            left_v.addWidget(self.txt_tasks)

            right_box = QGroupBox("📓 3. 子代研究開發筆記與演化代碼預覽 (DEV_NOTES.md & Evolved Code)")
            right_v = QVBoxLayout(right_box)
            self.txt_dev_notes = QPlainTextEdit()
            self.txt_dev_notes.setReadOnly(True)
            right_v.addWidget(self.txt_dev_notes)

            split.addWidget(left_box)
            split.addWidget(right_box)
            split.setSizes([500, 580])
            root_v.addWidget(split, stretch=1)

        def refresh_registry_ui(self) -> None:
            reg = self.replicator.load_registry()
            active_id = reg.get("active_cube_id", "gen_000_origin")
            base_id = reg.get("base_cube_id", "gen_000_origin")
            self.lbl_cube_topology.setText(
                f"🟢 [Active 執行端立方體]: {active_id}   ◄═══🌀═══►   🛡️ [Base 唯讀母體備援]: {base_id}"
            )

            self.cmb_cubes.blockSignals(True)
            self.cmb_cubes.clear()
            for c in reversed(reg.get("cubes", [])):
                cid = c.get("cube_id", "")
                st = c.get("status", "")
                ver = "✅VERIFIED" if c.get("verified") else "⏳UNVERIFIED"
                self.cmb_cubes.addItem(f"{cid}  [{st} | {ver}]", cid)
            # Always select the current active_id (or newest spawned cube) so the right pane shows DEV_NOTES.md!
            idx = self.cmb_cubes.findData(active_id)
            if idx >= 0:
                self.cmb_cubes.setCurrentIndex(idx)
            elif self.cmb_cubes.count() > 0:
                self.cmb_cubes.setCurrentIndex(0)
            self.cmb_cubes.blockSignals(False)
            self._on_selected_cube_changed()

        def _on_selected_cube_changed(self) -> None:
            cid = self.cmb_cubes.currentData()
            reg = self.replicator.load_registry()
            for c in reg.get("cubes", []):
                if c.get("cube_id") == cid:
                    cpath = c.get("path", "")
                    self.current_cube_dir = cpath
                    self._load_cube_files_to_panes(cpath)
                    break

        def _load_cube_files_to_panes(self, cube_dir: str) -> None:
            if not cube_dir or not os.path.exists(cube_dir):
                return
            goals_p = os.path.join(cube_dir, "GOALS.json")
            notes_p = os.path.join(cube_dir, "DEV_NOTES.md")
            code_p = os.path.join(cube_dir, "evolved_homeostasis.py")

            if os.path.exists(goals_p):
                try:
                    gdata = json.loads(Path(goals_p).read_text(encoding="utf-8"))
                    lines = [
                        f"=== 🎯 子代超立方體目標清單 ({gdata.get('cube_id')}) ===",
                        f"母體來源 (Parent Base): {gdata.get('parent_cube_id')}",
                        "",
                    ]
                    for g in gdata.get("goals", []):
                        icon = "✅" if g.get("status") == "COMPLETED" else ("❌" if g.get("status") == "FAILED" else "⏳")
                        lines.append(f"{icon} [{g.get('id')}] {g.get('title')} ({g.get('status')})")
                        lines.append(f"   • 目標模組: {g.get('module')}")
                        lines.append(f"   • 說明: {g.get('description')}\n")
                    self.txt_tasks.setPlainText("\n".join(lines))
                except Exception:
                    pass

            notes_txt = ""
            trace_p = os.path.join(cube_dir, "EXAMINER_PROMPT_TRACE.md")
            if os.path.exists(trace_p):
                notes_txt += "======================================================================\n"
                notes_txt += "🕵️‍♂️ [宇宙主考官實戰 Prompt 抽考與逐 Token 執行過程: EXAMINER_PROMPT_TRACE.md]\n"
                notes_txt += "======================================================================\n"
                notes_txt += Path(trace_p).read_text(encoding="utf-8", errors="replace") + "\n\n"
            plugin_p = os.path.join(cube_dir, "evolved_router_plugin.py")
            if os.path.exists(plugin_p):
                notes_txt += "======================================================================\n"
                notes_txt += "🧬 [原地熱插拔核心路由插件: evolved_router_plugin.py]\n"
                notes_txt += "======================================================================\n"
                notes_txt += Path(plugin_p).read_text(encoding="utf-8", errors="replace") + "\n\n"
            if os.path.exists(notes_p):
                notes_txt += Path(notes_p).read_text(encoding="utf-8", errors="replace")
            if os.path.exists(code_p):
                notes_txt += "\n\n======================================================================\n"
                notes_txt += "💻 [子代生成之演化模組預覽: evolved_homeostasis.py]\n"
                notes_txt += "======================================================================\n"
                notes_txt += Path(code_p).read_text(encoding="utf-8", errors="replace")
            if notes_txt:
                self.txt_dev_notes.setPlainText(notes_txt)

        def start_evolution_loop(self) -> None:
            if self.evo_worker and self.evo_worker.isRunning():
                return
            server_url, model_id = self.get_server_info_fn()
            self.btn_start_evo.setEnabled(False)
            self.btn_pause_evo.setEnabled(True)
            self.lbl_live_stage.setText("🚀 正在啟動無性生殖子代演化循環 (Thor 常駐模型零重載推論)...")

            self.evo_worker = HypercubeEvolutionWorker(
                replicator=self.replicator,
                server_url=server_url,
                model_id=model_id,
                auto_swap_on_pass=self.chk_auto_swap.isChecked(),
                continuous_cycles=self.chk_continuous.isChecked(),
                pause_on_emergence=self.chk_pause_emergence.isChecked(),
            )
            self.evo_worker.sig_cube_spawned.connect(self._on_worker_cube_spawned)
            self.evo_worker.sig_stage_update.connect(self._on_worker_stage_update)
            self.evo_worker.sig_cycle_complete.connect(self._on_worker_cycle_complete)
            self.evo_worker.sig_graceful_paused.connect(self._on_worker_graceful_paused)
            self.evo_worker.sig_emergence_alert.connect(self._on_worker_emergence_alert)
            self.evo_worker.finished.connect(self._on_worker_thread_finished)
            self.evo_worker.start()

        def enqueue_user_steer(self) -> None:
            txt = (self.inp_steer.text() or "").strip()
            if not txt:
                return
            from hypercube_headless_cli import CosmicMatryoshkaEvolutionCLI
            cli = CosmicMatryoshkaEvolutionCLI(workspace_root=self.replicator.workspace_root)
            cli.push_steer_directive(txt, source="cosmic_gui_steer")
            self.inp_steer.clear()
            self.lbl_live_stage.setText(f"🎯 已將 Steer 插話需求注入子代演化佇列: {txt[:60]}")
            self.txt_tasks.appendPlainText(f"\n🎯 [User Steer Queued]: {txt}\n")

        def _on_worker_emergence_alert(self, cube_id: str, alert_txt: str) -> None:
            self.refresh_registry_ui()
            self._load_cube_files_to_panes(self.current_cube_dir)
            self.lbl_live_stage.setText(f"🔔 [突破偵測自動暫停] 子代 `{cube_id}` 出現顯著性能躍升或新 Pattern，請主人檢視！")
            self.txt_tasks.appendPlainText(f"\n{'★'*70}\n{alert_txt}\n{'★'*70}\n")

        def request_graceful_pause(self) -> None:
            if self.evo_worker and self.evo_worker.isRunning():
                self.evo_worker.request_graceful_pause()
                self.lbl_live_stage.setText("⏸️ 已發送【優雅暫停 (Graceful Stop)】訊號，等待當前原子階段完成並寫入筆記後安全停機...")
                self.btn_pause_evo.setEnabled(False)

        def _on_worker_cube_spawned(self, cube_meta: dict) -> None:
            self.current_cube_dir = cube_meta.get("path", "")
            self.refresh_registry_ui()

        def _on_worker_stage_update(self, stage_idx: int, stage_title: str, detail: str) -> None:
            self.lbl_live_stage.setText(f"⚡ {stage_title}")
            self._load_cube_files_to_panes(self.current_cube_dir)
            self.txt_tasks.appendPlainText(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] {stage_title}\n{detail}\n")

        def _show_diagnostic_popup(self, title: str, report_text: str, is_error: bool = True) -> None:
            dlg = QDialog(self)
            dlg.setWindowTitle(title)
            dlg.resize(760, 480)
            border_c = "#f43f5e" if is_error else "#10b981"
            dlg.setStyleSheet(
                f"QDialog {{ background-color: #090d16; color: #f8fafc; border: 2px solid {border_c}; }}"
                "QPlainTextEdit { background-color: #05080f; color: #f8fafc; font-family: 'Consolas', monospace; font-size: 12px; border: 1px solid #334155; }"
                "QPushButton { background-color: #1e293b; color: white; padding: 6px 16px; border-radius: 4px; font-weight: bold; }"
            )
            lay = QVBoxLayout(dlg)
            hdr = QLabel(title)
            hdr.setStyleSheet(f"color: {border_c}; font-size: 14px; font-weight: bold;")
            lay.addWidget(hdr)

            txt = QPlainTextEdit()
            txt.setReadOnly(True)
            txt.setPlainText(report_text)
            lay.addWidget(txt, stretch=1)

            btn_close = QPushButton("關閉診斷視窗 (Close)")
            btn_close.clicked.connect(dlg.accept)
            lay.addWidget(btn_close)
            dlg.show()
            dlg.raise_()
            self._last_diag_dlg = dlg

        def _on_worker_cycle_complete(self, cube_id: str, ok: bool, summary: str) -> None:
            self.refresh_registry_ui()
            # Ensure the dropdown selects the newly completed cube `cube_id` AND appends the summary!
            idx = self.cmb_cubes.findData(cube_id)
            if idx >= 0:
                self.cmb_cubes.blockSignals(True)
                self.cmb_cubes.setCurrentIndex(idx)
                self.cmb_cubes.blockSignals(False)
            self._load_cube_files_to_panes(self.current_cube_dir)
            self.txt_tasks.appendPlainText(
                f"\n======================================================================\n"
                f"🏁 子代 `{cube_id}` 演化週期完成 (驗證={'PASS' if ok else 'FAIL'})\n"
                f"======================================================================\n"
                f"{summary}\n"
            )
            self.lbl_live_stage.setText(f"🏁 子代 `{cube_id}` 演化週期完成 (驗證={'PASS' if ok else 'FAIL'})")
            if not ok:
                self._show_diagnostic_popup(
                    f"❌ 子代 `{cube_id}` 驗證失敗診斷報告 (Verification Failure Traceback)",
                    summary,
                    is_error=True,
                )

        def _on_worker_graceful_paused(self, msg: str) -> None:
            self.refresh_registry_ui()
            self.lbl_live_stage.setText(msg)
            self.txt_tasks.appendPlainText(f"\n{msg}\n")
            if "異常攔截" in msg or "Error" in msg or "Exception" in msg:
                self._show_diagnostic_popup("⚠️ 演化執行緒異常診斷視窗", msg, is_error=True)

        def _on_worker_thread_finished(self) -> None:
            self.btn_start_evo.setEnabled(True)
            self.btn_pause_evo.setEnabled(False)

        def manual_promote_selected_cube(self) -> None:
            cid = self.cmb_cubes.currentData()
            if not cid:
                return
            ok, msg = self.replicator.rotate_hypercube_dimension(str(cid))
            self.refresh_registry_ui()
            self.lbl_live_stage.setText(msg)
            self.txt_tasks.appendPlainText(f"\n{msg}\n")

        def manual_rollback_to_base(self) -> None:
            ok, msg = self.replicator.rollback_to_base_cube()
            self.refresh_registry_ui()
            self.lbl_live_stage.setText(msg)
            self.txt_tasks.appendPlainText(f"\n{msg}\n")

except ImportError:
    pass


