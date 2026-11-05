#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
localsleep_telemetry_db.py — Dual-Track Telemetry & Append-Only SQLite WAL Blackbox Vault
=========================================================================================
Modeled after `rust_dysonsphere_sim/dyson_telemetry_db.py` and tailored for `rust_localsleep-moe`
and the 25-Stage Hypercube Evolution Engine:
  1. Zero Per-Token Disk I/O:
     Buffers per-token 16-LoRA states (`_ram_step_buffer`) in RAM during streaming inference,
     flushing in a single atomic SQLite WAL `executemany` transaction at turn/stage completion.
  2. Append-Only SQLite WAL Blackbox (`sessions/localsleep_telemetry_vault.db`):
     - `sessions`: Session metadata, active Hypercube ID, sandbox directory, file paths.
     - `session_turns`: Full user prompt, full raw assistant output, 16-LoRA weight vector,
       16-LoRA fatigue vector, 16-LoRA norm vector, Shannon entropy, NREM 2:4 prunes, Hebbian state.
     - `lora_step_traces`: Intra-turn per-token 16-LoRA trajectory snapshots.
     - `artifact_versions`: Immutable code version vault (every `<file>` or `write_file`/`edit_file`
       is stored with SHA-256 and full source text — NEVER overwritten!).
     - `sandbox_exec_logs`: 100% unfolded STDOUT & STDERR from every `cargo build` / `python3` run.
     - `hypercube_stage_records`: Dedicated 25-Stage Hypercube Evolution ledger (`stage_num` 1..25,
       `orthogonality_error`, `sparsity_pct`, `l2_canary_tps`, `tps_gain_ratio`, `lora_fatigue_16_json`).
  3. Dyson-Sphere-Style Dual-Track File Outputs:
     - `sessions/telemetry_<session_id>.jsonl`
     - `sessions/telemetry_<session_id>_chart.html` (Zero-dependency 4-Panel Grafana-Lite Canvas HTML)
     - `sessions/localsleep_telemetry_hub.html` (Master Telemetry Index Hub)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class LocalSleepTelemetryVaultDB:
    """
    Append-Only SQLite WAL Blackbox & Grafana-Lite HTML Chart Generator for `rust_localsleep-moe`.
    """

    def __init__(
        self,
        session_dir: str | Path = "./sessions",
        session_id: str = "default_session",
        profile_id: str = "thor_31b",
        model_id: str = "Gemma-4-31B-IT-NVFP4",
        server_url: str = "",
        active_cube_id: str = "gen_018_20261031_143847_matryoshka_L1",
        sandbox_dir: str = "",
        md_path: str = "",
    ) -> None:
        self.session_dir = Path(session_dir).resolve()
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.session_dir / "localsleep_telemetry_vault.db"
        self.session_id = session_id
        self.profile_id = profile_id
        self.model_id = model_id
        self.server_url = server_url
        self.active_cube_id = active_cube_id
        self.sandbox_dir = sandbox_dir
        self.md_path = md_path or str(self.session_dir / f"session_{session_id}.md")
        self.telemetry_jsonl_path = self.session_dir / f"telemetry_{session_id}.jsonl"
        self.chart_html_path = self.session_dir / f"telemetry_{session_id}_chart.html"
        self.hub_html_path = self.session_dir / "localsleep_telemetry_hub.html"

        self._turn_index = 0
        self._last_turn_id = 0
        self._sub_turn_exec_idx = 0
        self._pending_user_prompt = ""
        self._ram_step_buffer: List[Dict[str, Any]] = []
        self._last_expert_states: List[Dict[str, Any]] = []

        self._init_schema()
        self.register_session(
            server_url=server_url,
            active_cube_id=active_cube_id,
            sandbox_dir=sandbox_dir,
        )
        if self.session_id != "thor_31b_20261104_235052":
            try:
                self._auto_backfill_235052_once()
            except Exception:
                pass

    def _auto_backfill_235052_once(self) -> None:
        target_sid = "thor_31b_20261104_235052"
        sess_jsonl = self.session_dir / f"session_{target_sid}.jsonl"
        if not sess_jsonl.exists():
            return
        with self._connect() as conn:
            cnt = conn.execute(
                "SELECT COUNT(*) FROM session_turns WHERE session_id = ?", (target_sid,)
            ).fetchone()[0]
        if cnt == 0:
            sb_dir = self.session_dir.parent / "isolated_output" / "prj_20261104_235052_67bf865c"
            backfill_session_into_vault(self.session_dir, target_sid, sb_dir)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    server_url TEXT,
                    active_cube_id TEXT,
                    sandbox_dir TEXT,
                    md_path TEXT,
                    jsonl_path TEXT,
                    chart_html_path TEXT,
                    started_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    assistant_raw_text TEXT NOT NULL,
                    tokens_generated INTEGER,
                    elapsed_sec REAL,
                    tps REAL,
                    ttft_ms REAL,
                    shannon_entropy REAL,
                    selected_top_k_json TEXT,
                    lora_weights_16_json TEXT,
                    lora_fatigue_16_json TEXT,
                    lora_norms_16_json TEXT,
                    nrem_masked_experts_json TEXT,
                    nrem_24_prune_count INTEGER,
                    hebbian_trace_state TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_turns_session ON session_turns(session_id, turn_index);

                CREATE TABLE IF NOT EXISTS lora_step_traces (
                    trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    token_step INTEGER NOT NULL,
                    token_id INTEGER,
                    token_str TEXT,
                    tps REAL,
                    shannon_entropy REAL,
                    selected_top_k_json TEXT,
                    lora_weights_json TEXT,
                    lora_fatigue_16_json TEXT,
                    lora_norms_16_json TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_traces_session_turn ON lora_step_traces(session_id, turn_id);

                CREATE TABLE IF NOT EXISTS artifact_versions (
                    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    cube_id TEXT,
                    sandbox_rel_path TEXT NOT NULL,
                    sha256_hash TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    content_text TEXT NOT NULL,
                    inline_edits_json TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifact_versions(session_id, sandbox_rel_path);

                CREATE TABLE IF NOT EXISTS sandbox_exec_logs (
                    exec_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    turn_id INTEGER NOT NULL,
                    cube_id TEXT,
                    sub_turn_idx INTEGER NOT NULL,
                    command TEXT NOT NULL,
                    cwd TEXT NOT NULL,
                    exit_code INTEGER NOT NULL,
                    stdout_full TEXT,
                    stderr_full TEXT,
                    hebbian_trace_state TEXT,
                    executed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_exec_session ON sandbox_exec_logs(session_id, turn_id);

                CREATE TABLE IF NOT EXISTS hypercube_stage_records (
                    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    cube_id TEXT NOT NULL,
                    parent_cube_id TEXT,
                    stage_num INTEGER NOT NULL,
                    stage_name TEXT NOT NULL,
                    steer_prompt TEXT,
                    hypothesis_notes TEXT,
                    compile_attempts INTEGER DEFAULT 1,
                    orthogonality_error REAL DEFAULT -1.0,
                    sparsity_pct REAL DEFAULT 50.0,
                    l2_canary_tps REAL DEFAULT 0.0,
                    tps_gain_ratio REAL DEFAULT 1.0,
                    fatigue_reduction_pct REAL DEFAULT 0.0,
                    dispersion_entropy_gain_pct REAL DEFAULT 0.0,
                    lora_fatigue_16_json TEXT,
                    lora_weights_16_json TEXT,
                    verified_ok INTEGER DEFAULT 0,
                    summary_text TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_hypercube_stages ON hypercube_stage_records(cube_id, stage_num);
                """
            )

    def register_session(
        self,
        server_url: str = "",
        active_cube_id: str = "",
        sandbox_dir: str = "",
    ) -> None:
        if server_url:
            self.server_url = server_url
        if active_cube_id:
            self.active_cube_id = active_cube_id
        if sandbox_dir:
            self.sandbox_dir = sandbox_dir

        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT turn_index, turn_id FROM session_turns WHERE session_id = ? ORDER BY turn_index DESC LIMIT 1",
                (self.session_id,),
            )
            row = cur.fetchone()
            if row:
                self._turn_index = int(row[0])
                self._last_turn_id = int(row[1])

            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, profile_id, model_id, server_url, active_cube_id,
                    sandbox_dir, md_path, jsonl_path, chart_html_path, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    server_url = excluded.server_url,
                    active_cube_id = excluded.active_cube_id,
                    sandbox_dir = excluded.sandbox_dir
                """,
                (
                    self.session_id,
                    self.profile_id,
                    self.model_id,
                    self.server_url,
                    self.active_cube_id,
                    self.sandbox_dir,
                    self.md_path,
                    str(self.telemetry_jsonl_path),
                    str(self.chart_html_path),
                    now_iso,
                ),
            )

    # -------------------------------------------------------------------------
    # 1. Per-Token 16-LoRA RAM Buffer (Zero Disk I/O during streaming)
    # -------------------------------------------------------------------------
    def buffer_token_step(
        self,
        token_step: int,
        token_id: int,
        token_str: str,
        tps: float,
        entropy: float,
        selected_experts: List[int],
        weights: List[float],
        expert_states: List[Dict[str, Any]],
    ) -> None:
        self._last_expert_states = expert_states
        fatigues_16 = [round(float(x.get("fatigue", 0.0)), 4) for x in expert_states]
        norms_16 = [round(float(x.get("norm", 2.6)), 3) for x in expert_states]

        # Sample every step up to 500 steps, or every 4th step beyond 500 to keep DB compact & fast
        if len(self._ram_step_buffer) < 500 or (token_step % 4 == 0):
            self._ram_step_buffer.append(
                {
                    "token_step": int(token_step),
                    "token_id": int(token_id),
                    "token_str": str(token_str),
                    "tps": round(float(tps), 2),
                    "shannon_entropy": round(float(entropy), 4),
                    "selected_top_k_json": json.dumps([int(x) for x in selected_experts]),
                    "lora_weights_json": json.dumps([round(float(w), 4) for w in weights]),
                    "lora_fatigue_16_json": json.dumps(fatigues_16),
                    "lora_norms_16_json": json.dumps(norms_16),
                }
            )

    def set_pending_user_prompt(self, user_prompt: str) -> None:
        self._pending_user_prompt = user_prompt or ""
        self._sub_turn_exec_idx = 0

    # -------------------------------------------------------------------------
    # 2. Turn Completion Commit (Batch SQLite WAL Flush + JSONL + Chart HTML)
    # -------------------------------------------------------------------------
    def commit_turn(
        self,
        user_prompt: str,
        assistant_raw_text: str,
        stats: Dict[str, Any],
        artifacts: Optional[List[Any]] = None,
    ) -> int:
        self._turn_index += 1
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        prompt_txt = user_prompt or self._pending_user_prompt or ""
        raw_reply = assistant_raw_text or ""

        tokens = int(stats.get("tokens", 0) or 0)
        elapsed = float(stats.get("elapsed", 0.0) or 0.0)
        tps = float(stats.get("tps", 0.0) or 0.0)
        ttft_ms = float(stats.get("ttft_ms", 0.0) or 0.0)
        mean_entropy = float(stats.get("mean_entropy", 0.0) or 0.0)
        top_k_experts = [int(x) for x in (stats.get("top_k_experts") or [])]
        top_k_weights = [round(float(w), 4) for w in (stats.get("top_k_weights") or [])]
        gpu_24_prunes = int(stats.get("gpu_24_prunes", stats.get("nrem_events", 0)) or 0)

        # Extract 16-LoRA full vectors
        exp_states = stats.get("expert_states_formatted") or self._last_expert_states or []
        if not exp_states and stats.get("expert_states"):
            raw_es = stats["expert_states"]
            for item in raw_es:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    eid, st, fat = int(item[0]), int(item[1]), float(item[2])
                    exp_states.append(
                        {
                            "id": eid,
                            "state": st,
                            "fatigue": round(fat, 4),
                            "norm": round(max(0.25, 2.65 - fat * 0.4), 3),
                        }
                    )

        n_experts = max(16, len(exp_states))
        fatigue_16 = [0.0] * n_experts
        norms_16 = [2.60] * n_experts
        weights_16 = [0.0] * n_experts
        nrem_masked: List[int] = []

        for item in exp_states:
            eid = int(item.get("id", 0))
            if 0 <= eid < n_experts:
                fat_v = round(float(item.get("fatigue", 0.0)), 4)
                fatigue_16[eid] = fat_v
                norms_16[eid] = round(float(item.get("norm", 2.60)), 3)
                if fat_v > 0.70 or int(item.get("state", 0)) != 0:
                    nrem_masked.append(eid)

        for idx_k, eid in enumerate(top_k_experts):
            if 0 <= eid < n_experts and idx_k < len(top_k_weights):
                weights_16[eid] = round(float(top_k_weights[idx_k]), 4)

        if "LTM_CONSOLIDATED" in raw_reply:
            hebbian_state = "LTM_CONSOLIDATED"
        elif "ANTI_HEBBIAN_ROLLBACK" in raw_reply:
            hebbian_state = "ANTI_HEBBIAN_ROLLBACK"
        else:
            hebbian_state = str(stats.get("hebbian_trace_state") or "ACTIVE_STREAM")

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO session_turns (
                    session_id, turn_index, timestamp, user_prompt, assistant_raw_text,
                    tokens_generated, elapsed_sec, tps, ttft_ms, shannon_entropy,
                    selected_top_k_json, lora_weights_16_json, lora_fatigue_16_json,
                    lora_norms_16_json, nrem_masked_experts_json, nrem_24_prune_count,
                    hebbian_trace_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    self._turn_index,
                    now_iso,
                    prompt_txt,
                    raw_reply,
                    tokens,
                    round(elapsed, 3),
                    round(tps, 2),
                    round(ttft_ms, 1),
                    round(mean_entropy, 4),
                    json.dumps(top_k_experts),
                    json.dumps(weights_16),
                    json.dumps(fatigue_16),
                    json.dumps(norms_16),
                    json.dumps(nrem_masked),
                    gpu_24_prunes,
                    hebbian_state,
                ),
            )
            turn_id = int(cur.lastrowid or self._turn_index)
            self._last_turn_id = turn_id

            # Rebind any artifacts or sandbox execs recorded during this turn (where turn_id was 0 / previous)
            conn.execute(
                "UPDATE artifact_versions SET turn_id = ? WHERE session_id = ? AND turn_id = 0",
                (turn_id, self.session_id),
            )
            conn.execute(
                "UPDATE sandbox_exec_logs SET turn_id = ? WHERE session_id = ? AND turn_id = 0",
                (turn_id, self.session_id),
            )

            if self._ram_step_buffer:
                conn.executemany(
                    """
                    INSERT INTO lora_step_traces (
                        session_id, turn_id, token_step, token_id, token_str,
                        tps, shannon_entropy, selected_top_k_json, lora_weights_json,
                        lora_fatigue_16_json, lora_norms_16_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            self.session_id,
                            turn_id,
                            s["token_step"],
                            s["token_id"],
                            s["token_str"],
                            s["tps"],
                            s["shannon_entropy"],
                            s["selected_top_k_json"],
                            s["lora_weights_json"],
                            s["lora_fatigue_16_json"],
                            s["lora_norms_16_json"],
                        )
                        for s in self._ram_step_buffer
                    ],
                )
                self._ram_step_buffer.clear()

        # Also write Dyson-Sphere-style structured telemetry row to `telemetry_<session_id>.jsonl`
        jsonl_row = {
            "session_id": self.session_id,
            "turn_id": turn_id,
            "turn_index": self._turn_index,
            "timestamp": now_iso,
            "profile_id": self.profile_id,
            "model_id": self.model_id,
            "active_cube_id": self.active_cube_id,
            "tokens": tokens,
            "elapsed_sec": round(elapsed, 3),
            "tps": round(tps, 2),
            "ttft_ms": round(ttft_ms, 1),
            "shannon_entropy": round(mean_entropy, 4),
            "selected_top_k": top_k_experts,
            "lora_weights_16": weights_16,
            "lora_fatigue_16": fatigue_16,
            "lora_norms_16": norms_16,
            "nrem_masked_experts": nrem_masked,
            "nrem_24_prune_count": gpu_24_prunes,
            "hebbian_trace_state": hebbian_state,
            "user_prompt_preview": prompt_txt[:160],
            "artifacts_count": len(artifacts or []),
        }
        with open(self.telemetry_jsonl_path, "a", encoding="utf-8") as jf:
            jf.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")

        # Update interactive Grafana-Lite HTML chart & Hub
        try:
            self.export_grafana_lite_html()
            self.update_telemetry_hub_html()
        except Exception:
            pass

        return turn_id

    # -------------------------------------------------------------------------
    # 3. Immutable Code Artifact Version Vault (Never Overwrites!)
    # -------------------------------------------------------------------------
    def record_artifact_version(
        self,
        sandbox_rel_path: str,
        content_text: str,
        cube_id: str = "",
        inline_edits: Optional[List[str]] = None,
        turn_id: Optional[int] = None,
    ) -> int:
        if not content_text:
            return 0
        sha256_hex = hashlib.sha256(content_text.encode("utf-8", errors="replace")).hexdigest()
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        eff_turn_id = turn_id if turn_id is not None else 0
        eff_cube = cube_id or self.active_cube_id

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO artifact_versions (
                    session_id, turn_id, cube_id, sandbox_rel_path,
                    sha256_hash, byte_size, content_text, inline_edits_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    eff_turn_id,
                    eff_cube,
                    sandbox_rel_path,
                    sha256_hex,
                    len(content_text.encode("utf-8", errors="replace")),
                    content_text,
                    json.dumps(inline_edits or [], ensure_ascii=False),
                    now_iso,
                ),
            )
            return int(cur.lastrowid or 0)

    # -------------------------------------------------------------------------
    # 4. 100% Unfolded Sandbox Shell Execution Audit Log
    # -------------------------------------------------------------------------
    def record_sandbox_exec(
        self,
        command: str,
        cwd: str,
        exit_code: int,
        stdout_full: str,
        stderr_full: str,
        cube_id: str = "",
        turn_id: Optional[int] = None,
    ) -> int:
        self._sub_turn_exec_idx += 1
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        eff_turn_id = turn_id if turn_id is not None else 0
        eff_cube = cube_id or self.active_cube_id
        hebbian_state = "LTM_CONSOLIDATED" if exit_code == 0 else "ANTI_HEBBIAN_ROLLBACK"

        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO sandbox_exec_logs (
                    session_id, turn_id, cube_id, sub_turn_idx, command, cwd,
                    exit_code, stdout_full, stderr_full, hebbian_trace_state, executed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    eff_turn_id,
                    eff_cube,
                    self._sub_turn_exec_idx,
                    command,
                    cwd,
                    int(exit_code),
                    stdout_full or "",
                    stderr_full or "",
                    hebbian_state,
                    now_iso,
                ),
            )
            return int(cur.lastrowid or 0)

    # -------------------------------------------------------------------------
    # 5. Dedicated 25-Stage Hypercube Evolution Recorder
    # -------------------------------------------------------------------------
    def record_hypercube_stage(
        self,
        cube_id: str,
        parent_cube_id: str,
        stage_num: int,
        stage_name: str,
        steer_prompt: str = "",
        hypothesis_notes: str = "",
        compile_attempts: int = 1,
        orthogonality_error: float = 0.0,
        sparsity_pct: float = 50.0,
        l2_canary_tps: float = 0.0,
        tps_gain_ratio: float = 1.0,
        fatigue_reduction_pct: float = 0.0,
        dispersion_entropy_gain_pct: float = 0.0,
        lora_fatigue_16: Optional[List[float]] = None,
        lora_weights_16: Optional[List[float]] = None,
        verified_ok: bool = True,
        summary_text: str = "",
    ) -> int:
        now_iso = datetime.datetime.now().isoformat(timespec="seconds")
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO hypercube_stage_records (
                    session_id, cube_id, parent_cube_id, stage_num, stage_name,
                    steer_prompt, hypothesis_notes, compile_attempts,
                    orthogonality_error, sparsity_pct, l2_canary_tps, tps_gain_ratio,
                    fatigue_reduction_pct, dispersion_entropy_gain_pct,
                    lora_fatigue_16_json, lora_weights_16_json,
                    verified_ok, summary_text, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.session_id,
                    cube_id,
                    parent_cube_id,
                    int(stage_num),
                    stage_name,
                    steer_prompt,
                    hypothesis_notes,
                    int(compile_attempts),
                    float(orthogonality_error),
                    float(sparsity_pct),
                    float(l2_canary_tps),
                    float(tps_gain_ratio),
                    float(fatigue_reduction_pct),
                    float(dispersion_entropy_gain_pct),
                    json.dumps(lora_fatigue_16 or []),
                    json.dumps(lora_weights_16 or []),
                    1 if verified_ok else 0,
                    summary_text,
                    now_iso,
                ),
            )
            rec_id = int(cur.lastrowid or 0)

        try:
            self.update_telemetry_hub_html()
        except Exception:
            pass
        return rec_id

    # -------------------------------------------------------------------------
    # 6. Dyson-Sphere-Style Grafana-Lite 4-Panel HTML Analyzer
    # -------------------------------------------------------------------------
    def export_grafana_lite_html(self, out_path: Optional[Path] = None) -> Path:
        target_html = Path(out_path or self.chart_html_path)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            turns = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM session_turns WHERE session_id = ? ORDER BY turn_index ASC",
                    (self.session_id,),
                ).fetchall()
            ]
            traces = [
                dict(r)
                for r in conn.execute(
                    "SELECT token_step, tps, shannon_entropy, selected_top_k_json, lora_fatigue_16_json "
                    "FROM lora_step_traces WHERE session_id = ? ORDER BY trace_id ASC LIMIT 4500",
                    (self.session_id,),
                ).fetchall()
            ]
            artifacts = [
                dict(r)
                for r in conn.execute(
                    "SELECT artifact_id, turn_id, cube_id, sandbox_rel_path, sha256_hash, byte_size, created_at "
                    "FROM artifact_versions WHERE session_id = ? ORDER BY artifact_id ASC",
                    (self.session_id,),
                ).fetchall()
            ]
            execs = [
                dict(r)
                for r in conn.execute(
                    "SELECT exec_id, turn_id, sub_turn_idx, command, exit_code, hebbian_trace_state, executed_at "
                    "FROM sandbox_exec_logs WHERE session_id = ? ORDER BY exec_id ASC",
                    (self.session_id,),
                ).fetchall()
            ]

        payload_json = json.dumps(
            {
                "session_id": self.session_id,
                "profile_id": self.profile_id,
                "model_id": self.model_id,
                "active_cube_id": self.active_cube_id,
                "turns": turns,
                "traces": traces,
                "artifacts": artifacts,
                "execs": execs,
            },
            ensure_ascii=False,
        )

        html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>🐈‍⬛ LocalSleep-MoE 16-LoRA & Hypercube Blackbox Telemetry ({self.session_id})</title>
<style>
  body {{ background: #070d19; color: #e2e8f0; font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif; margin: 0; padding: 14px; }}
  h2 {{ margin: 0 0 10px 0; color: #38bdf8; font-size: 18px; display: flex; justify-content: space-between; align-items: center; }}
  .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 10px 12px; margin-bottom: 12px; font-size: 12px; }}
  .meta-card {{ background: #162032; border-left: 3px solid #38bdf8; padding: 6px 10px; border-radius: 4px; }}
  .meta-card .lbl {{ color: #94a3b8; font-size: 11px; }}
  .meta-card .val {{ color: #f8fafc; font-weight: bold; font-size: 13px; margin-top: 2px; word-break: break-all; }}
  .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; }}
  .chart-box {{ background: #0b1324; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; }}
  .chart-title {{ font-weight: bold; font-size: 13px; color: #38bdf8; margin-bottom: 6px; }}
  canvas {{ width: 100%; height: 220px; display: block; background: #070b14; border-radius: 4px; }}
  table.tbl {{ width: 100%; border-collapse: collapse; font-size: 12px; background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; margin-top: 10px; }}
  table.tbl th, table.tbl td {{ border-bottom: 1px solid #1e293b; padding: 6px 8px; text-align: left; }}
  table.tbl th {{ background: #162032; color: #38bdf8; }}
  .badge-ok {{ background: #065f46; color: #34d399; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
  .badge-rb {{ background: #7f1d1d; color: #fca5a5; padding: 2px 6px; border-radius: 4px; font-weight: bold; }}
</style>
</head>
<body>
<h2>
  <span>🐈‍⬛ LocalSleep-MoE 16-LoRA 神經動力學與沙盒黑盒子分析儀 (Session: {self.session_id})</span>
  <a href="localsleep_telemetry_hub.html" style="color:#38bdf8;font-size:13px;text-decoration:none;">⬅️ 返回遙測總覽 Hub</a>
</h2>
<div class="meta-grid">
  <div class="meta-card"><div class="lbl">🤖 模型 (Model ID)</div><div class="val">{self.model_id}</div></div>
  <div class="meta-card" style="border-left-color:#a855f7;"><div class="lbl">🧊 執行端超立方體 (Active Cube)</div><div class="val">{self.active_cube_id}</div></div>
  <div class="meta-card" style="border-left-color:#34d399;"><div class="lbl">📊 總對話輪數 / 逐 Token 軌跡點</div><div class="val">{len(turns)} Turns / {len(traces)} Token Steps</div></div>
  <div class="meta-card" style="border-left-color:#fbbf24;"><div class="lbl">📦 不可變代碼版本數 / 沙盒執行次數</div><div class="val">{len(artifacts)} Code Versions / {len(execs)} Execs</div></div>
</div>

<div class="chart-grid">
  <div class="chart-box">
    <div class="chart-title">📈 Panel 1: 逐 Token Shannon 熵 H(p) 與多專家共振曲線</div>
    <canvas id="cvEntropy"></canvas>
  </div>
  <div class="chart-box">
    <div class="chart-title">⚡ Panel 2: 逐 Token 推論吞吐率 (TPS tok/s)</div>
    <canvas id="cvTps"></canvas>
  </div>
</div>

<h3 style="color:#38bdf8;font-size:14px;margin:12px 0 6px 0;">🧬 1. 各輪 16-LoRA 路由、疲勞度與赫布資格跡總表 (Session Turns)</h3>
<table class="tbl" id="tblTurns">
  <thead>
    <tr>
      <th>Turn #</th><th>時間</th><th>Prompt 摘要</th><th>Tokens / TPS / TTFT</th>
      <th>Shannon 熵 H</th><th>激活 Top-K 專家</th><th>NREM 2:4 剪枝</th><th>資格跡狀態</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<h3 style="color:#fbbf24;font-size:14px;margin:14px 0 6px 0;">📦 2. 不可變沙盒代碼版本保險箱 (Immutable Artifact Versions in SQLite DB)</h3>
<table class="tbl" id="tblArts">
  <thead>
    <tr>
      <th>Artifact ID</th><th>Turn #</th><th>沙盒相對路徑</th><th>大小 (Bytes)</th><th>SHA-256 雜湊前 16 碼</th><th>寫入時間</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<h3 style="color:#34d399;font-size:14px;margin:14px 0 6px 0;">🛠️ 3. 沙盒編譯與測試執行紀錄 (Sandbox Shell Execution Audit)</h3>
<table class="tbl" id="tblExecs">
  <thead>
    <tr>
      <th>Exec ID</th><th>Turn #</th><th>執行指令</th><th>EXIT_CODE</th><th>赫布資格跡動作</th><th>時間</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>

<script>
const DATA = {payload_json};

function drawLineChart(canvasId, values, colorHex, yLabel) {{
  const cv = document.getElementById(canvasId);
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const w = cv.width = cv.clientWidth || 600;
  const h = cv.height = cv.clientHeight || 220;
  ctx.clearRect(0, 0, w, h);
  if (!values || values.length === 0) {{
    ctx.fillStyle = '#64748b';
    ctx.fillText('尚無逐 Token 軌跡資料', 20, h/2);
    return;
  }}
  let minV = Math.min(...values), maxV = Math.max(...values);
  if (Math.abs(maxV - minV) < 1e-6) {{ maxV += 1.0; minV = Math.max(0, minV - 0.5); }}
  const padL = 42, padR = 14, padT = 16, padB = 24;
  const pw = w - padL - padR, ph = h - padT - padB;

  ctx.strokeStyle = '#1e293b';
  ctx.beginPath();
  for (let i=0; i<=4; i++) {{
    const gy = padT + (ph * i / 4);
    ctx.moveTo(padL, gy); ctx.lineTo(w - padR, gy);
    const val = (maxV - (maxV - minV) * (i / 4)).toFixed(2);
    ctx.fillStyle = '#94a3b8'; ctx.font = '10px monospace';
    ctx.fillText(val, 4, gy + 3);
  }}
  ctx.stroke();

  ctx.strokeStyle = colorHex;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  values.forEach((v, idx) => {{
    const x = padL + (idx / Math.max(1, values.length - 1)) * pw;
    const y = padT + (1.0 - (v - minV) / (maxV - minV)) * ph;
    if (idx === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }});
  ctx.stroke();
}}

drawLineChart('cvEntropy', DATA.traces.map(t => t.shannon_entropy || 0), '#a855f7', 'H(p)');
drawLineChart('cvTps', DATA.traces.map(t => t.tps || 0), '#38bdf8', 'TPS');

const tbTurns = document.querySelector('#tblTurns tbody');
DATA.turns.forEach(t => {{
  const tr = document.createElement('tr');
  const badge = (t.hebbian_trace_state === 'LTM_CONSOLIDATED')
    ? `<span class="badge-ok">🧬 ${{t.hebbian_trace_state}}</span>`
    : (t.hebbian_trace_state === 'ANTI_HEBBIAN_ROLLBACK' ? `<span class="badge-rb">🛡️ ${{t.hebbian_trace_state}}</span>` : t.hebbian_trace_state);
  tr.innerHTML = `
    <td><b>#${{t.turn_index}}</b></td>
    <td>${{t.timestamp}}</td>
    <td>${{(t.user_prompt || '').slice(0, 75)}}...</td>
    <td><code>${{t.tokens_generated}}</code> tok | <b>${{t.tps}}</b> tok/s | TTFT <code>${{t.ttft_ms}}ms</code></td>
    <td><b style="color:#e879f9;">${{t.shannon_entropy}}</b></td>
    <td><code>${{t.selected_top_k_json}}</code></td>
    <td><code>${{t.nrem_24_prune_count}}</code></td>
    <td>${{badge}}</td>
  `;
  tbTurns.appendChild(tr);
}});

const tbArts = document.querySelector('#tblArts tbody');
DATA.artifacts.forEach(a => {{
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><b>v#${{a.artifact_id}}</b></td>
    <td>Turn #${{a.turn_id}}</td>
    <td><code style="color:#38bdf8;">${{a.sandbox_rel_path}}</code></td>
    <td><code>${{a.byte_size}}</code> B</td>
    <td><code>${{(a.sha256_hash || '').slice(0, 16)}}...</code></td>
    <td>${{a.created_at}}</td>
  `;
  tbArts.appendChild(tr);
}});

const tbExecs = document.querySelector('#tblExecs tbody');
DATA.execs.forEach(e => {{
  const tr = document.createElement('tr');
  const badge = (e.exit_code === 0)
    ? `<span class="badge-ok">EXIT 0 (LTM_CONSOLIDATED)</span>`
    : `<span class="badge-rb">EXIT ${{e.exit_code}} (ANTI_HEBBIAN_ROLLBACK)</span>`;
  tr.innerHTML = `
    <td><b>#${{e.exec_id}}</b></td>
    <td>Turn #${{e.turn_id}} (Sub #${{e.sub_turn_idx}})</td>
    <td><code>${{(e.command || '').slice(0, 80)}}</code></td>
    <td><code>${{e.exit_code}}</code></td>
    <td>${{badge}}</td>
    <td>${{e.executed_at}}</td>
  `;
  tbExecs.appendChild(tr);
}});
</script>
</body>
</html>
"""
        target_html.write_text(html, encoding="utf-8")
        return target_html

    def update_telemetry_hub_html(self) -> Path:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            sess_rows = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT s.*,
                           COUNT(DISTINCT t.turn_id) AS turn_cnt,
                           COUNT(DISTINCT a.artifact_id) AS art_cnt,
                           COUNT(DISTINCT e.exec_id) AS exec_cnt
                    FROM sessions s
                    LEFT JOIN session_turns t ON s.session_id = t.session_id
                    LEFT JOIN artifact_versions a ON s.session_id = a.session_id
                    LEFT JOIN sandbox_exec_logs e ON s.session_id = e.session_id
                    GROUP BY s.session_id
                    ORDER BY s.started_at DESC
                    LIMIT 50
                    """
                ).fetchall()
            ]
            stage_rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM hypercube_stage_records ORDER BY record_id DESC LIMIT 50"
                ).fetchall()
            ]

        sess_trs = []
        for s in sess_rows:
            sid = s["session_id"]
            chart_name = f"telemetry_{sid}_chart.html"
            md_name = f"session_{sid}.md"
            sess_trs.append(
                f"<tr>"
                f"<td><b><code>{sid}</code></b></td>"
                f"<td><code>{s.get('model_id','')}</code></td>"
                f"<td><code style='color:#a855f7;'>{s.get('active_cube_id','')}</code></td>"
                f"<td><b>{s.get('turn_cnt',0)}</b> 輪</td>"
                f"<td><b>{s.get('art_cnt',0)}</b> 版代碼 / <b>{s.get('exec_cnt',0)}</b> 次編譯</td>"
                f"<td><a href='{chart_name}' style='color:#38bdf8;'>📊 開啟互動圖表</a> | <a href='{md_name}' style='color:#34d399;'>📄 Markdown</a></td>"
                f"<td>{s.get('started_at','')}</td>"
                f"</tr>"
            )

        stage_trs = []
        for st in stage_rows:
            badge = "✅ PASS" if st.get("verified_ok") else "❌ FAIL"
            stage_trs.append(
                f"<tr>"
                f"<td><b>Stage {st.get('stage_num')}</b> ({st.get('stage_name')})</td>"
                f"<td><code>{st.get('cube_id')}</code> ⬅️ <code>{st.get('parent_cube_id')}</code></td>"
                f"<td><code>{st.get('orthogonality_error'):.6f}</code></td>"
                f"<td><code>{st.get('sparsity_pct'):.1f}%</code></td>"
                f"<td><b>{st.get('l2_canary_tps'):.2f} tok/s</b> ({st.get('tps_gain_ratio')}x)</td>"
                f"<td>{badge}</td>"
                f"<td>{st.get('created_at','')}</td>"
                f"</tr>"
            )

        hub_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8"/>
<title>🧊 LocalSleep-MoE & 25-Stage Hypercube Telemetry Hub</title>
<style>
  body {{ background: #070d19; color: #e2e8f0; font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif; margin: 0; padding: 18px; }}
  h2 {{ color: #38bdf8; margin-top: 0; }}
  h3 {{ color: #fbbf24; margin-top: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; }}
  th, td {{ border-bottom: 1px solid #1e293b; padding: 8px 10px; text-align: left; }}
  th {{ background: #162032; color: #38bdf8; }}
</style>
</head>
<body>
  <h2>🧊 LocalSleep-MoE 雙軌遙測與 SQLite WAL 黑盒子總覽中心 (Telemetry Vault Hub)</h2>
  <p style="color:#94a3b8;font-size:13px;">SQLite 黑盒子路徑：<code>{self.db_path}</code>（Append-Only 永不覆蓋，自動保存 16-LoRA 狀態向量、不可變沙盒代碼版本與 25 階超立方體演化紀錄）</p>

  <h3>🌌 1. 25 階超立方體算子創世演化紀錄 (Hypercube 25-Stage Evolution Ledger)</h3>
  <table>
    <thead>
      <tr><th>階段 (Stage)</th><th>子代 Cube ID ⬅️ 母體</th><th>正交誤差 ||M^T M - I||_F</th><th>2:4 稀疏度</th><th>L2 Canary TPS (加速比)</th><th>驗證狀態</th><th>時間</th></tr>
    </thead>
    <tbody>
      {''.join(stage_trs) if stage_trs else '<tr><td colspan="7" style="color:#64748b;">尚無超立方體 25 階演化紀錄（待啟動演化）</td></tr>'}
    </tbody>
  </table>

  <h3>💬 2. 歷次 Agentic REPL 與沙盒實驗 Session 列表</h3>
  <table>
    <thead>
      <tr><th>Session ID</th><th>模型</th><th>Active Cube</th><th>對話輪數</th><th>保存代碼版本 / 編譯次數</th><th>快速檢視</th><th>啟動時間</th></tr>
    </thead>
    <tbody>
      {''.join(sess_trs)}
    </tbody>
  </table>
</body>
</html>
"""
        self.hub_html_path.write_text(hub_html, encoding="utf-8")
        return self.hub_html_path


def backfill_session_into_vault(
    session_dir: str | Path,
    session_id: str,
    sandbox_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Backfills an existing `.jsonl` / `.md` session (such as `thor_31b_20261104_235052`)
    and its sandbox artifacts into `localsleep_telemetry_vault.db` if not already present.
    """
    sdir = Path(session_dir).resolve()
    vault = LocalSleepTelemetryVaultDB(
        session_dir=sdir,
        session_id=session_id,
        profile_id="thor_31b",
        model_id="Gemma-4-31B-IT-NVFP4",
        sandbox_dir=str(sandbox_dir or ""),
    )

    with vault._connect() as conn:
        existing_turns = conn.execute(
            "SELECT COUNT(*) FROM session_turns WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    if existing_turns > 0:
        vault.export_grafana_lite_html()
        vault.update_telemetry_hub_html()
        return {"status": "already_present", "turns": existing_turns, "db_path": str(vault.db_path)}

    # Load per-token real telemetry if available
    rt_file = sdir / f"session_real_telemetry_{session_id}.jsonl"
    if rt_file.exists():
        with open(rt_file, "r", encoding="utf-8") as rf:
            for line in rf:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    vault.buffer_token_step(
                        token_step=int(row.get("step", 0)),
                        token_id=int(row.get("token_id", 0)),
                        token_str=str(row.get("token_str", "")),
                        tps=float(row.get("tps", 0.0)),
                        entropy=float(row.get("entropy", 0.0)),
                        selected_experts=row.get("selected_experts") or row.get("routed_experts") or [],
                        weights=row.get("weights") or row.get("routing_weights") or [],
                        expert_states=row.get("experts") or [],
                    )
                except Exception:
                    continue

    # Parse `session_<session_id>.jsonl` or `.md`
    sess_jsonl = sdir / f"session_{session_id}.jsonl"
    turns_imported = 0
    if sess_jsonl.exists():
        pending_u = ""
        with open(sess_jsonl, "r", encoding="utf-8") as jf:
            for line in jf:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("type") == "turn_complete":
                    data = rec.get("data") or {}
                    role = data.get("role")
                    if role == "user":
                        pending_u = data.get("content") or ""
                    elif role == "assistant":
                        vault.commit_turn(
                            user_prompt=pending_u,
                            assistant_raw_text=data.get("content") or "",
                            stats=data.get("stats") or {},
                            artifacts=data.get("artifacts") or [],
                        )
                        turns_imported += 1
                        pending_u = ""

    # Also archive all source files inside `sandbox_dir` into `artifact_versions`
    arts_imported = 0
    if sandbox_dir and Path(sandbox_dir).exists():
        sb_path = Path(sandbox_dir).resolve()
        for p in sorted(sb_path.rglob("*")):
            if p.is_file() and p.suffix in (".rs", ".py", ".toml", ".md", ".json") and "target/" not in str(p):
                rel_p = str(p.relative_to(sb_path))
                try:
                    txt = p.read_text(encoding="utf-8", errors="replace")
                    vault.record_artifact_version(
                        sandbox_rel_path=rel_p,
                        content_text=txt,
                        turn_id=vault._last_turn_id or 1,
                    )
                    arts_imported += 1
                except Exception:
                    continue

    chart_p = vault.export_grafana_lite_html()
    hub_p = vault.update_telemetry_hub_html()
    return {
        "status": "backfilled",
        "session_id": session_id,
        "turns_imported": turns_imported,
        "artifacts_imported": arts_imported,
        "db_path": str(vault.db_path),
        "chart_html": str(chart_p),
        "hub_html": str(hub_p),
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    res = backfill_session_into_vault(
        session_dir=base / "sessions",
        session_id="thor_31b_20261104_235052",
        sandbox_dir=base / "isolated_output" / "prj_20261104_235052_67bf865c",
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))

