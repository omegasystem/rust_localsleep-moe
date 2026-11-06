#!/usr/bin/env python3
"""
mb_fractal_forge.py - O(1)-Context Fractal Megabyte File & Document Forge Engine
================================================================================
Breaks the LLM Context Window & Lazy-EOS Ceiling to generate 100KB ~ 10MB+ files
using a local model (`rust_localsleep-moe` @ :18935, `Gemma-4-31B + gen_022`).

Core Architecture (3-Phase Fractal Slot Assembly):
  - Phase 0 [Global Contract & Blueprint]:
    Establishes immutable shared symbols, style/naming conventions, and N orthogonal
    section/module slots (`@@SLOT:SEC_01@@` ... `@@SLOT:SEC_N@@`).
  - Phase 1 [Bounded-Context Slot Expansion — O(1) KV Cache]:
    Iterates (or parallel-batches) across each slot. Each LLM call only receives:
      [Global Contract (~2KB)] + [Previous Slot Interface Summary (~500B)] + [Current Sub-Spec]
    Because KV-cache is reset after every slot, VRAM usage never grows with total file size!
  - Phase 2 [Deterministic Linker & Checksum]:
    Assembles all generated chunks into the final Megabyte-scale target file and
    maintains a resume-capable state journal (`.forge_state.json`) so interrupted
    1MB builds can resume from the exact unfinished slot.
"""

import argparse
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_ENDPOINT = "http://127.0.0.1:18935"


class MegabyteFractalForge:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = "localsleep-moe",
        max_tokens_per_slot: int = 16384,
        temperature: float = 0.5,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.max_tokens_per_slot = max_tokens_per_slot
        self.temperature = temperature

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.max_tokens_per_slot,
            "temperature": self.temperature,
            "stream": False,
            "reset_kv_cache": True,
        }
        req = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        telem = data.get("localsleep_telemetry") or {}
        return {"content": content, "telemetry": telem}

    def generate_blueprint(self, topic: str, num_slots: int, doc_type: str = "markdown", mode: str = "tech") -> Dict[str, Any]:
        sys_p = (
            "你是 S-Tier 分形巨型文件與長篇小說總編劇。請只輸出合法的 JSON 物件（不要包裹其他解釋文字）。"
        )
        if mode == "novel":
            usr_p = (
                f"我們準備透過『分形插槽組裝法』創作一部超過一萬五千字的繁體中文長篇科幻輕小說，題材與核心設定為：\n"
                f"『{topic}』\n"
                f"請將整部小說規劃為 {num_slots} 個章節插槽 (Chapters)，每章包含 2 個緊湊且高張力的場景 (Scenes)，並輸出以下 JSON 格式：\n"
                "{\n"
                '  "title": "小說正式書名",\n'
                '  "global_contract": "【世界觀與角色聖經 (300~500字)】：明確固定男主角姓名、現實身分、潛行對戰假想體(Avatar)名稱與外型配色、專屬武器與心意技名稱；女主角姓名、假想體名稱與能力；最終Boss與核心世界法則（如千倍思考超頻 Burst Link + 完全潛行死亡樓層攻略）。所有章節必須嚴格統一角色名字與招式名稱，且必須以純粹、具畫面感與五感描寫的繁體中文輕小說散文與對話撰寫，嚴禁出現程式碼或大綱列點！",\n'
                '  "slots": [\n'
                '    {"id": "CHAPTER_01", "title": "第一章標題", "sub_topics": ["第一幕：具體劇情與衝突推進", "第二幕：具體劇情、戰鬥或情感轉折"], "required_elements": "本章必須展現的關鍵場景、心意技對決、感官細節與人物對話"}\n'
                "  ]\n"
                "}"
            )
        else:
            usr_p = (
                f"我們準備透過『分形插槽組裝法』生成一份巨型 {doc_type} 文件，主題為：\n"
                f"『{topic}』\n"
                f"請將整份巨型文件拆解為 {num_slots} 個正交、深度極高、互不重複的章節插槽 (Slots)，並輸出以下 JSON 格式：\n"
                "{\n"
                '  "title": "文件總標題",\n'
                '  "global_contract": "全域術語、命名規範、統一變數或架構約束 (約 200~400 字，會注入每個插槽確保前後 100% 一致)",\n'
                '  "slots": [\n'
                '    {"id": "SLOT_01", "title": "第1章標題", "sub_topics": ["子節1.1", "子節1.2", "子節1.3", "子節1.4"], "required_elements": "必須包含的公式、架構表或完整程式碼模組"}\n'
                "  ]\n"
                "}"
            )
        res = self._call_llm(sys_p, usr_p)
        raw = res["content"].strip()
        if "```" in raw:
            for part in raw.split("```"):
                p = part.strip()
                if p.startswith("json"):
                    p = p[4:].strip()
                if p.startswith("{") and p.endswith("}"):
                    raw = p
                    break
        start_i = raw.find("{")
        end_i = raw.rfind("}")
        if start_i != -1 and end_i != -1:
            raw = raw[start_i : end_i + 1]
        return json.loads(raw)

    def expand_slot(
        self,
        blueprint: Dict[str, Any],
        slot: Dict[str, Any],
        slot_index: int,
        total_slots: int,
        prev_tail_summary: str,
        passes_per_slot: int = 1,
        mode: str = "tech",
    ) -> str:
        """
        Expands a single slot. If `passes_per_slot > 1`, expands each `sub_topic` inside
        the slot as its own dedicated LLM pass to multiply output byte density!
        """
        global_contract = blueprint.get("global_contract", "")
        doc_title = blueprint.get("title", "")
        sub_topics: List[str] = slot.get("sub_topics") or [slot.get("title", f"Section {slot_index}")]

        if mode == "novel":
            sys_p = (
                "你是川原礫風格的日本頂級科幻輕小說作家（代表作《刀劍神域 SAO》與《加速世界 Accel World》）。\n"
                "請以極具臨場感、電影分鏡感與細膩心理描寫的【純繁體中文小說正文】進行創作。\n"
                "【小說寫作鐵律】：\n"
                "1. 100% 純小說散文與角色生動對話！嚴禁出現任何程式碼、嚴禁出現條列式大綱、嚴禁出現章節總結或說教結語！\n"
                "2. 深入描寫量子腦神經連結的五感（粒子破碎的冰冷觸感、超頻加速下時間凝結的視覺色偏、刀鋒交擊的火花與反作用力、角色的心跳與痛覺回饋）。\n"
                "3. 每一幕必須寫出豐富完整的對話、動作分鏡與內心戲（單幕至少 1,800 ~ 2,500 中文字），結尾留有懸念自然銜接下一幕。"
            )
        else:
            sys_p = (
                "你是數位黑貓 Kuro 的底層巨型文件鍛造引擎。你正在撰寫一份 MB 等級巨型工程巨著的指定插槽。\n"
                "【鐵律】：\n"
                "1. 嚴禁使用『限於篇幅』、『以下省略』、『...』或總結性敷衍語句！\n"
                "2. 必須展開所有底層數學推導、完整生產級程式碼實現、邊界條件表格與架構細節，最大化資訊密度與篇幅。\n"
                "3. 嚴格遵守下方提供的【全域不變量契約 (Global Contract)】，確保與前後章節無縫銜接。"
            )

        if passes_per_slot <= 1 or len(sub_topics) <= 1:
            usr_p = (
                f"【作品總標題】：{doc_title}\n"
                f"【世界觀與角色聖經 (Global Contract)】：{global_contract}\n"
                f"【上一章節結尾段落 (請無縫接續此氛圍與劇情)】：\n{prev_tail_summary or '(本章為故事開篇第一章)'}\n"
                f"----------------------------------------\n"
                f"請立即撰寫【第 {slot_index}/{total_slots} 章 ({slot.get('id')})：{slot.get('title')}】的小說正文。\n"
                f"本章劇情推進：{', '.join(sub_topics)}\n"
                f"關鍵要素：{slot.get('required_elements', '')}\n"
                f"請直接輸出充滿畫面感、對話與戰鬥/心理描寫的小說正文（至少 2,500 中文字，不要輸出標題以外的任何後設說明）："
            )
            res = self._call_llm(sys_p, usr_p)
            return f"## {slot.get('id')}：{slot.get('title')}\n\n" + res["content"].strip()

        # Multi-pass sub-topic deep expansion for massive scale
        sub_chunks: List[str] = [f"# {slot.get('id')}：{slot.get('title')}\n"]
        rolling_tail = prev_tail_summary
        for s_idx, sub_t in enumerate(sub_topics[:passes_per_slot], start=1):
            if mode == "novel":
                usr_p = (
                    f"【作品書名】：{doc_title}\n"
                    f"【世界觀與角色聖經 (Global Contract)】：{global_contract}\n"
                    f"【上一幕結尾原文 (請緊接此處繼續描寫，不要重複前文)】：\n「...{rolling_tail or '（故事正式拉開序幕）'}」\n"
                    f"----------------------------------------\n"
                    f"現在進入第 {slot_index}/{total_slots} 章《{slot.get('title')}》的【第 {s_idx}/{min(len(sub_topics), passes_per_slot)} 幕：{sub_t}】。\n"
                    f"本章關鍵要素：{slot.get('required_elements', '')}\n"
                    f"請直接輸出本幕的繁體中文小說正文（深入描寫環境光影、角色對話、神經加速感官與戰鬥動作分鏡，單幕請寫滿 1,800 ~ 2,500 字，嚴禁列點或摘要）："
                )
            else:
                usr_p = (
                    f"【巨著總標題】：{doc_title}\n"
                    f"【全域不變量契約】：{global_contract}\n"
                    f"【前文收尾錨點】：{rolling_tail or '(本章起始)'}\n"
                    f"----------------------------------------\n"
                    f"目前位於第 {slot_index}/{total_slots} 章《{slot.get('title')}》的第 {s_idx}/{len(sub_topics)} 子節：\n"
                    f"👉 子節主題：『{sub_t}』\n"
                    f"必備要素：{slot.get('required_elements', '完整數學推導、架構表與無省略代碼')}\n"
                    f"請針對此單一子節撰寫極度詳盡的完整技術內容（至少 2,500~4,000 字，嚴禁省略）："
                )
            res = self._call_llm(sys_p, usr_p)
            chunk_txt = res["content"].strip()
            sub_chunks.append(f"\n### 幕 {slot_index}.{s_idx} — {sub_t}\n\n{chunk_txt}\n")
            rolling_tail = chunk_txt[-350:].replace("\n", " ")
        return "\n".join(sub_chunks)

    def forge_document(
        self,
        topic: str,
        out_path: Path,
        num_slots: int = 10,
        passes_per_slot: int = 2,
        mode: str = "tech",
    ) -> Dict[str, Any]:
        state_path = out_path.with_suffix(out_path.suffix + ".forge_state.json")
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            blueprint = state["blueprint"]
            completed_slots: Dict[str, str] = state.get("completed_slots", {})
            print(f"♻️ Resuming forge from state file ({len(completed_slots)} slots already done)...")
        else:
            print(f"📐 [Phase 0] Generating Fractal Blueprint ({num_slots} slots × {passes_per_slot} passes | mode={mode})...")
            blueprint = self.generate_blueprint(topic, num_slots=num_slots, mode=mode)
            completed_slots = {}
            state = {"topic": topic, "mode": mode, "blueprint": blueprint, "completed_slots": completed_slots}
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            self._assemble_to_disk(blueprint, blueprint.get("slots", []), completed_slots, out_path)
            print(f"   📖 [Phase 0 Ready] Initial Kuro Tactical Novel Reader shell written to: {out_path}", flush=True)

        slots = blueprint.get("slots", [])
        prev_tail = ""
        t0 = time.time()

        for idx, slot in enumerate(slots, start=1):
            sid = str(slot.get("id") or f"SLOT_{idx:02d}")
            if sid in completed_slots:
                prev_tail = completed_slots[sid][-350:].replace("\n", " ")
                continue

            print(f"🔥 [Phase 1] Forging {sid} ({idx}/{len(slots)}): {slot.get('title')} ...")
            chunk = self.expand_slot(
                blueprint=blueprint,
                slot=slot,
                slot_index=idx,
                total_slots=len(slots),
                prev_tail_summary=prev_tail,
                passes_per_slot=passes_per_slot,
                mode=mode,
            )
            completed_slots[sid] = chunk
            prev_tail = chunk[-350:].replace("\n", " ")
            state["completed_slots"] = completed_slots
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            # Incremental assembly to disk after every slot!
            self._assemble_to_disk(blueprint, slots, completed_slots, out_path)
            cur_txt = out_path.read_text(encoding="utf-8")
            cur_bytes = out_path.stat().st_size
            cjk_count = sum(1 for ch in cur_txt if "\u4e00" <= ch <= "\u9fff")
            print(
                f"   ✅ {sid} done (+{len(chunk):,} chars) -> Total: {len(cur_txt):,} chars | "
                f"CJK漢字: {cjk_count:,} 字 | {cur_bytes:,} bytes ({cur_bytes/1024:.1f} KB)",
                flush=True,
            )

        final_txt = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
        total_cjk = sum(1 for ch in final_txt if "\u4e00" <= ch <= "\u9fff")
        total_bytes = out_path.stat().st_size if out_path.exists() else 0
        elapsed = time.time() - t0
        return {
            "output_file": str(out_path),
            "mode": mode,
            "title": blueprint.get("title", ""),
            "total_chars": len(final_txt),
            "total_cjk_chars": total_cjk,
            "total_bytes": total_bytes,
            "total_kb": round(total_bytes / 1024.0, 2),
            "slots_count": len(slots),
            "elapsed_sec": round(elapsed, 2),
        }

    @staticmethod
    def _assemble_to_disk(
        blueprint: Dict[str, Any],
        slots: List[Dict[str, Any]],
        completed_slots: Dict[str, str],
        out_path: Path,
    ) -> None:
        title = blueprint.get("title", "Megabyte Fractal Novel")
        contract = blueprint.get("global_contract", "")
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Always compute markdown text
        header = [
            f"# {title}",
            f"> **Generated by:** `LocalSleep-MoE Megabyte Fractal Forge (Gemma-4-31B + gen_022)`  ",
            f"> **Timestamp:** `{now_str}`  ",
            f"> **Global Story Bible / Contract:** {contract}\n",
            "---\n",
            "## 📑 Table of Contents (Fractal Slot Index)",
        ]
        for idx, s in enumerate(slots, start=1):
            sid = str(s.get("id") or f"SLOT_{idx:02d}")
            mark = "✅" if sid in completed_slots else "⏳"
            header.append(f"- {mark} **{sid}**: {s.get('title')}")
        header.append("\n---\n")

        body_parts = ["\n".join(header)]
        for idx, s in enumerate(slots, start=1):
            sid = str(s.get("id") or f"SLOT_{idx:02d}")
            if sid in completed_slots:
                body_parts.append(completed_slots[sid])
                body_parts.append("\n\n---\n\n")

        full_md = "".join(body_parts)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.suffix.lower() != ".html":
            out_path.write_text(full_md, encoding="utf-8")
            return

        # Also save companion .md file alongside .html
        md_companion = out_path.with_suffix(".md")
        md_companion.write_text(full_md, encoding="utf-8")

        # Build structured JSON payload for the Kuro Tactical HTML Novel Reader
        chapters_data = []
        total_cjk = 0
        for idx, s in enumerate(slots, start=1):
            sid = str(s.get("id") or f"CHAPTER_{idx:02d}")
            raw_text = completed_slots.get(sid, "")
            cjk_n = sum(1 for ch in raw_text if "\u4e00" <= ch <= "\u9fff")
            total_cjk += cjk_n
            chapters_data.append({
                "id": sid,
                "index": idx,
                "title": s.get("title", sid),
                "sub_topics": s.get("sub_topics", []),
                "required_elements": s.get("required_elements", ""),
                "done": sid in completed_slots,
                "cjk_count": cjk_n,
                "char_count": len(raw_text),
                "content": raw_text,
            })

        reader_payload = {
            "title": title,
            "updated_at": now_str,
            "global_contract": contract,
            "total_cjk": total_cjk,
            "total_chars": len(full_md),
            "done_count": len(completed_slots),
            "total_slots": len(slots),
            "chapters": chapters_data,
        }
        payload_json = json.dumps(reader_payload, ensure_ascii=False)

        html_doc = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐈‍⬛ {title} // Kuro Tactical Novel Reader</title>
<style>
  :root {{
    --bg:#0b0e17; --panel:#111625; --panel-alt:#0d1220; --input-bg:#090d16;
    --border:#26324d; --text:#e2e8f0; --muted:#8b9bb4;
    --cyan:#38bdf8; --green:#4ade80; --amber:#f59e0b; --red:#f87171; --purple:#c084fc;
    --reader-font-size: 17px;
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; padding:0; height:100vh; display:flex; flex-direction:column; overflow:hidden;
    background:var(--bg); color:var(--text);
    font-family:"JetBrains Mono","Fira Code","Noto Sans TC",monospace; font-size:13px;
  }}
  header {{
    height:48px; background:var(--panel); border-bottom:1px solid var(--border);
    display:flex; align-items:center; justify-content:space-between; padding:0 16px; flex-shrink:0;
  }}
  .brand {{ display:flex; align-items:center; gap:10px; font-weight:700; color:var(--cyan); font-size:14px; }}
  .badge {{ font-size:10.5px; padding:2px 8px; border-radius:4px; background:rgba(56,189,248,0.12); border:1px solid var(--cyan); color:var(--cyan); }}
  .hdr-ctrls {{ display:flex; gap:8px; align-items:center; }}
  .btn {{
    background:var(--panel-alt); color:var(--text); border:1px solid var(--border);
    border-radius:4px; padding:4px 10px; font-family:inherit; font-size:11.5px; cursor:pointer; transition:0.15s;
  }}
  .btn:hover {{ border-color:var(--cyan); color:#fff; }}
  .btn.active {{ background:#064e3b; border-color:#10b981; color:#a7f3d0; box-shadow:0 0 10px rgba(16,185,129,0.25); }}
  input.search {{ background:var(--input-bg); color:var(--text); border:1px solid var(--border); border-radius:4px; padding:4px 8px; font-family:inherit; font-size:11.5px; width:165px; }}
  .kpi-ribbon {{
    background:var(--panel-alt); border-bottom:1px solid var(--border);
    padding:8px 16px; display:grid; grid-template-columns:repeat(5, 1fr); gap:12px; flex-shrink:0;
  }}
  .kpi-box {{ background:var(--panel); border:1px solid var(--border); border-left:3px solid var(--cyan); border-radius:5px; padding:6px 12px; }}
  .kpi-box .k {{ font-size:10px; color:var(--muted); text-transform:uppercase; }}
  .kpi-box .v {{ font-size:17px; font-weight:800; margin-top:2px; color:var(--cyan); }}
  .kpi-box .v.green {{ color:var(--green); }}
  .kpi-box .v.amber {{ color:var(--amber); }}
  .kpi-box .v.purple {{ color:var(--purple); }}
  .workspace {{ flex:1; display:grid; grid-template-columns:290px 1fr 340px; min-height:0; }}
  .sidebar-left {{ background:var(--panel-alt); border-right:1px solid var(--border); display:flex; flex-direction:column; min-height:0; }}
  .sec-hdr {{ padding:10px 12px; font-size:11.5px; font-weight:700; color:var(--cyan); border-bottom:1px solid var(--border); display:flex; justify-content:space-between; }}
  .chap-list {{ flex:1; overflow-y:auto; padding:10px; display:flex; flex-direction:column; gap:8px; }}
  .chap-card {{
    background:var(--panel); border:1px solid var(--border); border-radius:5px; padding:9px 10px;
    cursor:pointer; transition:0.15s;
  }}
  .chap-card:hover {{ border-color:var(--cyan); }}
  .chap-card.selected {{ border:1px dashed var(--cyan); background:#131c31; box-shadow:inset 0 0 10px rgba(56,189,248,0.14); }}
  .chap-top {{ display:flex; justify-content:space-between; align-items:center; font-weight:700; font-size:12px; margin-bottom:4px; }}
  .chap-meta {{ display:flex; justify-content:space-between; font-size:10.5px; color:var(--muted); }}
  .bar-bg {{ height:5px; background:var(--input-bg); border-radius:3px; overflow:hidden; margin-top:6px; }}
  .bar-fill {{ height:100%; background:var(--green); transition:width 0.3s; }}
  /* Center Novel Reader */
  .reader-col {{ display:flex; flex-direction:column; min-height:0; background:#090c14; position:relative; }}
  .read-progress {{ height:3px; background:var(--cyan); width:0%; transition:width 0.1s; }}
  .reader-scroll {{
    flex:1; overflow-y:auto; padding:32px 48px 80px 48px;
    font-family:"Noto Serif TC","Noto Sans TC","PingFang TC","Microsoft JhengHei",serif;
    font-size:var(--reader-font-size); line-height:1.95; color:#e5ecf6;
  }}
  .novel-container {{ max-width:780px; margin:0 auto; }}
  .novel-h1 {{ font-family:"JetBrains Mono","Noto Sans TC",monospace; font-size:24px; color:var(--cyan); border-bottom:1px solid var(--border); padding-bottom:10px; margin:28px 0 18px 0; }}
  .novel-h2 {{ font-family:"JetBrains Mono","Noto Sans TC",monospace; font-size:17px; color:var(--amber); margin:24px 0 14px 0; padding-left:10px; border-left:3px solid var(--amber); }}
  .novel-p {{ margin:0 0 16px 0; text-indent:2em; text-align:justify; letter-spacing:0.03em; }}
  .dialogue {{ color:#7dd3fc; }}
  .incarnate {{ color:#fbbf24; font-weight:700; text-shadow:0 0 8px rgba(245,158,11,0.3); }}
  /* Right Lore & Telemetry Panel */
  .sidebar-right {{ background:var(--panel-alt); border-left:1px solid var(--border); display:flex; flex-direction:column; min-height:0; overflow-y:auto; padding:12px; gap:12px; }}
  .lore-box {{ background:var(--panel); border:1px solid var(--border); border-radius:5px; padding:10px 12px; font-size:12px; line-height:1.65; color:#cbd5e1; white-space:pre-wrap; }}
  canvas#densityCanvas {{ width:100%; height:145px; background:var(--input-bg); border:1px solid var(--border); border-radius:4px; }}
</style>
</head>
<body>

<header>
  <div class="brand">
    <span>🐈‍⬛ {title}</span>
    <span class="badge" id="hdrStatus">FORGING...</span>
  </div>
  <div class="hdr-ctrls">
    <input type="text" class="search" id="inpSearch" placeholder="🔍 搜尋角色 / 心意技..." oninput="renderReader()">
    <button class="btn active" id="btnAll" onclick="selectChapter('ALL')">📚 全書連讀模式</button>
    <button class="btn" onclick="adjustFont(-1)">A-</button>
    <button class="btn" onclick="adjustFont(1)">A+</button>
    <button class="btn" onclick="location.reload()">🔄 重新整理進度</button>
  </div>
</header>

<div class="kpi-ribbon">
  <div class="kpi-box"><div class="k">TOTAL CJK 漢字數</div><div class="v green" id="kCjk">0 字</div></div>
  <div class="kpi-box" style="border-left-color:var(--amber)"><div class="k">CHAPTERS FORGED</div><div class="v amber" id="kChaps">0 / 0 章</div></div>
  <div class="kpi-box" style="border-left-color:var(--purple)"><div class="k">TOTAL FILE SIZE</div><div class="v purple" id="kBytes">0 KB</div></div>
  <div class="kpi-box"><div class="k">BURST LINK ENGINE</div><div class="v">gen_022 (16-LoRA)</div></div>
  <div class="kpi-box" style="border-left-color:var(--green)"><div class="k">LAST UPDATED</div><div class="v green" id="kTime" style="font-size:13px">--</div></div>
</div>

<div class="workspace">
  <div class="sidebar-left">
    <div class="sec-hdr"><span>📑 章節分形插槽導覽 (Chapters)</span><span id="lblChapProg">0/0</span></div>
    <div class="chap-list" id="chapList"></div>
  </div>

  <div class="reader-col">
    <div class="read-progress" id="readBar"></div>
    <div class="reader-scroll" id="readerScroll" onscroll="updateScrollBar()">
      <div class="novel-container" id="novelContent"></div>
    </div>
  </div>

  <div class="sidebar-right">
    <div class="sec-hdr" style="padding:0 0 6px 0;"><span>🧬 世界觀與角色聖經 (Global Contract)</span></div>
    <div class="lore-box" id="loreBox"></div>
    <div class="sec-hdr" style="padding:6px 0;">
      <span>📊 章節字數密度與張力矩陣</span>
      <label style="font-size:10px;color:var(--muted);cursor:pointer;"><input type="checkbox" id="chkCanvas" checked onchange="drawMatrix()"> Render</label>
    </div>
    <canvas id="densityCanvas" width="310" height="145"></canvas>
    <div style="font-size:10.5px;color:var(--muted);display:flex;justify-content:space-between;">
      <span>🟦 一般敘事密度</span><span>🟧 高張力戰鬥/超頻段落</span>
    </div>
  </div>
</div>

<script>
const NOVEL_DATA = {payload_json};
let currentFilter = 'ALL';
let fontSize = 17;

function initUI() {{
  document.getElementById('kCjk').textContent = NOVEL_DATA.total_cjk.toLocaleString() + ' 字';
  document.getElementById('kChaps').textContent = `${{NOVEL_DATA.done_count}} / ${{NOVEL_DATA.total_slots}} 章`;
  document.getElementById('kBytes').textContent = (NOVEL_DATA.total_chars / 1024).toFixed(1) + ' KB';
  document.getElementById('kTime').textContent = NOVEL_DATA.updated_at;
  document.getElementById('lblChapProg').textContent = `${{NOVEL_DATA.done_count}}/${{NOVEL_DATA.total_slots}}`;
  document.getElementById('loreBox').textContent = NOVEL_DATA.global_contract || '';
  const stBadge = document.getElementById('hdrStatus');
  if (NOVEL_DATA.done_count >= NOVEL_DATA.total_slots) {{
    stBadge.textContent = '✅ COMPLETED (' + NOVEL_DATA.total_cjk.toLocaleString() + ' 字)';
    stBadge.style.borderColor = 'var(--green)';
    stBadge.style.color = 'var(--green)';
  }} else {{
    stBadge.textContent = `⚡ FORGING (${{NOVEL_DATA.done_count}}/${{NOVEL_DATA.total_slots}})`;
  }}
  renderChapterCards();
  renderReader();
  drawMatrix();
}}

function renderChapterCards() {{
  const list = document.getElementById('chapList');
  list.innerHTML = '';
  NOVEL_DATA.chapters.forEach(ch => {{
    const card = document.createElement('div');
    card.className = 'chap-card' + (currentFilter === ch.id ? ' selected' : '');
    card.onclick = () => selectChapter(ch.id);
    const pct = Math.min(100, Math.round((ch.cjk_count / 3200) * 100));
    const stTag = ch.done ? `<span style="color:var(--green)">● ${{ch.cjk_count.toLocaleString()}} 字</span>` : `<span style="color:var(--amber)">⏳ 鍛造中...</span>`;
    card.innerHTML = `
      <div class="chap-top"><span>${{ch.id}}</span>${{stTag}}</div>
      <div style="font-size:11.5px;color:#cbd5e1;margin-bottom:4px;">${{ch.title}}</div>
      <div class="chap-meta"><span>Scenes: ${{ch.sub_topics.length}}</span><span>Target: 3,000+ 字</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${{ch.done ? Math.max(20, pct) : 5}}%;background:${{ch.done ? 'var(--green)' : 'var(--amber)'}}"></div></div>
    `;
    list.appendChild(card);
  }});
}}

function selectChapter(id) {{
  currentFilter = id;
  document.getElementById('btnAll').classList.toggle('active', id === 'ALL');
  renderChapterCards();
  renderReader();
  document.getElementById('readerScroll').scrollTop = 0;
}}

function formatParagraph(line) {{
  let safe = line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  safe = safe.replace(/(「[^」]+」)/g, '<span class="dialogue">$1</span>');
  safe = safe.replace(/(『[^』]+』)/g, '<span class="incarnate">$1</span>');
  const q = document.getElementById('inpSearch').value.trim();
  if (q) {{
    const re = new RegExp(q.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&'), 'gi');
    safe = safe.replace(re, m => `<mark style="background:var(--amber);color:#000;padding:0 2px;border-radius:2px;">${{m}}</mark>`);
  }}
  return safe;
}}

function renderReader() {{
  const box = document.getElementById('novelContent');
  const chaps = currentFilter === 'ALL' ? NOVEL_DATA.chapters : NOVEL_DATA.chapters.filter(c => c.id === currentFilter);
  const htmlParts = [];
  chaps.forEach(ch => {{
    if (!ch.done) {{
      htmlParts.push(`<div class="novel-h1">${{ch.id}}：${{ch.title}} <small style="color:var(--amber);font-size:13px;">(⏳ 地端 MoE 副腦正在鍛造本章...)</small></div>`);
      return;
    }}
    const lines = ch.content.split('\\n');
    lines.forEach(rawLine => {{
      const l = rawLine.trim();
      if (!l) return;
      if (l.startsWith('# ')) {{
        htmlParts.push(`<div class="novel-h1">${{l.slice(2)}}</div>`);
      }} else if (l.startsWith('## ') || l.startsWith('### ')) {{
        htmlParts.push(`<div class="novel-h2">${{l.replace(/^#+\\s*/, '')}}</div>`);
      }} else {{
        htmlParts.push(`<p class="novel-p">${{formatParagraph(l)}}</p>`);
      }}
    }});
  }});
  box.innerHTML = htmlParts.join('\\n');
}}

function adjustFont(delta) {{
  fontSize = Math.max(13, Math.min(26, fontSize + delta));
  document.documentElement.style.setProperty('--reader-font-size', fontSize + 'px');
}}

function updateScrollBar() {{
  const el = document.getElementById('readerScroll');
  const max = Math.max(1, el.scrollHeight - el.clientHeight);
  const pct = Math.min(100, (el.scrollTop / max) * 100);
  document.getElementById('readBar').style.width = pct.toFixed(1) + '%';
}}

function drawMatrix() {{
  const cvs = document.getElementById('densityCanvas');
  const ctx = cvs.getContext('2d');
  ctx.clearRect(0, 0, cvs.width, cvs.height);
  if (!document.getElementById('chkCanvas').checked) return;
  const rows = Math.max(1, NOVEL_DATA.chapters.length);
  const cols = 20;
  const cw = cvs.width / cols;
  const ch = cvs.height / rows;
  NOVEL_DATA.chapters.forEach((chap, r) => {{
    const text = chap.content || '';
    const sliceLen = Math.max(1, Math.floor(text.length / cols));
    for (let c = 0; c < cols; c++) {{
      if (!chap.done) {{
        ctx.fillStyle = '#090d16';
      }} else {{
        const seg = text.slice(c * sliceLen, (c + 1) * sliceLen);
        const actionHits = (seg.match(/[！!『』斬擊加速超頻刀光火花]/g) || []).length;
        const intensity = Math.min(1.0, 0.25 + (actionHits / 12));
        ctx.fillStyle = actionHits > 4 ? `rgba(245, 158, 11, ${{intensity}})` : `rgba(56, 189, 248, ${{intensity}})`;
      }}
      ctx.fillRect(c * cw + 1, r * ch + 1, cw - 2, ch - 2);
    }}
  }});
}}

window.addEventListener('DOMContentLoaded', initUI);
</script>
</body>
</html>
"""
        out_path.write_text(html_doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Megabyte Fractal File & Document Forge Engine (:18935)")
    parser.add_argument("--topic", required=True, help="Document or monolithic system specification topic")
    parser.add_argument("--out", required=True, help="Target output file path")
    parser.add_argument("--slots", type=int, default=10, help="Number of top-level chapters/slots (default: 10)")
    parser.add_argument("--passes-per-slot", type=int, default=2, help="Sub-topic passes per slot (default: 2 -> total 20 passes)")
    parser.add_argument("--mode", choices=["tech", "novel"], default="tech", help="Forge mode: tech (engineering spec) or novel (long-form fiction)")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="localsleep-moe")
    args = parser.parse_args()

    forge = MegabyteFractalForge(endpoint=args.endpoint, model=args.model)
    summary = forge.forge_document(
        topic=args.topic,
        out_path=Path(args.out),
        num_slots=args.slots,
        passes_per_slot=args.passes_per_slot,
        mode=args.mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
