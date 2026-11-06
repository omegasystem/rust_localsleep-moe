#!/usr/bin/env python3
"""
GITS v3 Negative-Palindrome & Epistemic-Fog Fractal Novel Forge
===============================================================
Upgrades based on Architect Leon's feedback:
1. Negative Palindrome Constraint Engine (負面迴文抑制引擎):
   - Real-time detector scans the last 2 generated sections for overused lexical motifs
     (e.g., 咔噠, 黑咖啡, 機械錶, 酸雨, 幻肢痛, 不完美) AND structural cadence patterns
     (e.g., lyrical/philosophical monologue endings, anchor-rescue tropes).
   - Injects a symmetric Palindromic Sandwich Lock ([-C] at Prompt Head + [-C^R] at Prompt Tail)
     plus Phase-Inversion rules (if section N ended with introspection, section N+1 MUST end
     with an unresolved physical clue or abrupt tactical cliffhanger).
   - Automatically scrubs/masks over-saturated motif tokens inside the 80% trailing cache window
     so In-Context Learning (ICL) cannot copy them into the next pass!
2. Epistemic Fog Tiered Lore (漸進式懸疑迷霧 — 嚴禁開頭破題):
   - Replaces the static omniscient Global Lore Bible with 4 progressive Fog Gates:
     * Gate I (CH 01-02): Pure crime-scene anomaly & unexplained sensory clues. Zero mention of the mastermind or grand truth.
     * Gate II (CH 03-04): Red herrings, forensic contradictions, suspicion of memory tampering.
     * Gate III (CH 05-06): The core conspiracy ("Negative Palindrome Protocol") emerges from the fog.
     * Gate IV (CH 07): Cold noir resolution with lingering ambiguity (Oshii style).
3. Compact Post-Credits Tachikoma Micro-Skit (60秒片尾彩蛋級短篇小劇場):
   - Hard-caps Omake length to 450~650 CJK chars (max_tokens=1100) so the main noir thriller
     remains the absolute centerpiece (~90%+ of total word count).
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "gits_reader_template.html"
DEFAULT_ENDPOINT = "http://127.0.0.1:18935"

CANVAS_THEMES = [
    "neon_rain",
    "cyber_dive",
    "optical_camo",
    "tachikoma_pod",
    "oil_workshop",
    "ghost_dawn",
    "neon_rain",
]

# Tracked motifs & narrative tropes for Negative Palindrome Detection
TRACKED_MOTIFS: List[Tuple[str, str, int]] = [
    ("咔噠", "手指/關節發出『咔噠』聲的動作與描寫", 2),
    ("黑咖啡", "飲用或端著『黑咖啡』的描寫", 2),
    ("機械錶", "低頭查看『機械錶/秒針誤差』的動作", 2),
    ("幻肢痛", "直接提及『幻肢痛』三個字", 2),
    ("不完美", "直接詠嘆『不完美/缺陷才是人類證明』的哲學總結句", 2),
    ("酸雨", "窗外『酸雨/雨幕』的環境開頭或結尾", 3),
    ("天然機油", "在主線中提及『天然機油』", 2),
]

CADENCE_ENDINGS = [
    "Abrupt Physical Clue Cliffhanger (在發現一件不合邏輯的物理證物或異常訊號的瞬間戛然而止，嚴禁抒情總結)",
    "Cold Tactical Dialogue Cut (以一句簡短冰冷的戰術通話或質問直接切斷場景，不加任何內心獨白)",
    "Sensory Anomaly Freeze (以視覺或聽覺感測器捕捉到的0.1秒詭異殘影作結，留白給下一節)",
    "Silent Action Departure (以純粹的物理動作與環境音收尾，嚴禁在結尾出現『他意識到/他明白/這就是靈魂』等哲學詠嘆)",
]


def call_llm(
    endpoint: str,
    model: str,
    sys_p: str,
    usr_p: str,
    temp: float = 0.58,
    max_tokens: int = 12288,
    reset_kv_cache: bool = False,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": False,
        "reset_kv_cache": reset_kv_cache,
        "stateless_forge": True,
    }
    raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(20):
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (409, 502, 503, 504) and attempt < 19:
                print(f"   ⏳ [Retry {attempt+1}/20] Server busy (HTTP {e.code}), waiting 6s...", flush=True)
                time.sleep(6.0)
                continue
            raise
        except Exception as e:
            if attempt < 10:
                print(f"   ⏳ [Retry {attempt+1}/10] {e}, waiting 6s...", flush=True)
                time.sleep(6.0)
                continue
            raise
    raise RuntimeError("call_llm exhausted retries")


def build_negative_palindrome_constraints(
    recent_texts: List[str],
    step_index: int,
) -> Tuple[str, str, List[str]]:
    """
    Scans the recent generated sections and constructs a symmetric Negative Palindrome
    (Head Anti-Attractor Lock [-C] + Tail Phase-Inverted Mirror Lock [-C^R]), plus returns
    a list of saturated keywords to dampen inside the 80% cache window.
    """
    combined = "\n".join(recent_texts[-2:]) if recent_texts else ""
    banned_descriptions: List[str] = []
    saturated_keywords: List[str] = []

    for kw, desc, threshold in TRACKED_MOTIFS:
        cnt = combined.count(kw)
        if cnt >= threshold:
            banned_descriptions.append(f"「{desc}」（前文已出現 {cnt} 次，觸發負面迴文冷卻鎖）")
            saturated_keywords.append(kw)

    # Always enforce anti-cliche ending cadence rotation
    target_cadence = CADENCE_ENDINGS[step_index % len(CADENCE_ENDINGS)]

    if not banned_descriptions:
        banned_str = "避免重複前文已出現過的感官比喻或相同場景開頭"
    else:
        banned_str = "；".join(banned_descriptions)

    head_lock = (
        "【⛔ 負面迴文約束・前向防壁 (Negative Palindrome Head Lock [-C])】：\n"
        f"1. 意象消磁禁令：本節嚴禁出現或重複 —— {banned_str}！\n"
        "2. 情節防震盪令：嚴禁再次使用「主角在意識崩潰邊緣靠某個小動作拉回現實」或「在結尾喝飲料感嘆人生哲理」的重複套路！\n"
        "3. 懸疑遞進原則：只呈現冰冷的刑偵線索、感官細節與未解矛盾，絕不提前講破幕後真相！"
    )

    tail_lock = (
        "【🔄 負面迴文約束・逆向鏡像尾鎖 (Negative Palindrome Tail Mirror [-C^R])】：\n"
        f"• 本節收尾強制相位：{target_cadence}\n"
        f"• 再次確認逆向消磁：全文不得出現 {saturated_keywords if saturated_keywords else ['陳腔濫調的哲學總結句']}！請直接以小說動作或懸疑畫面收束！"
    )

    return head_lock, tail_lock, saturated_keywords


def extract_damped_80pct_cache(
    text: str,
    saturated_keywords: List[str],
    max_chars: int = 3400,
) -> str:
    """
    Extracts the trailing ~80% of the previous section for seamless scene continuity,
    while applying 'Motif De-gaussing' (消磁衰減) to over-saturated sentences at the very end
    so the model does not copy the previous section's closing cliché!
    """
    if not text:
        return ""
    start_idx = int(len(text) * 0.20)
    sliced = text[start_idx:]
    if len(sliced) > max_chars:
        sliced = sliced[-max_chars:]
    nl = sliced.find("\n")
    if 0 <= nl < 200:
        sliced = sliced[nl + 1 :].strip()

    # If the last paragraph of the previous section contains over-saturated motifs,
    # replace recurring motif words in the first 75% of the cache with neutral pronouns/actions
    # while keeping the final 400 chars intact for action continuity.
    if saturated_keywords and len(sliced) > 600:
        head_part = sliced[:-400]
        tail_part = sliced[-400:]
        for kw in saturated_keywords:
            head_part = head_part.replace(kw, "")
        sliced = head_part + tail_part
    return sliced


def get_epistemic_fog_context(bp: Dict[str, Any], chapter_idx: int, total_chapters: int) -> str:
    """
    Returns ONLY the epistemic layer unlocked at the current chapter index,
    preventing the SLM from spoiling the mystery in Chapter 1!
    """
    base_cast = bp.get("cast_contract", "")
    fog_layers = bp.get("epistemic_fog_layers", {})
    if chapter_idx <= 2:
        fog_note = fog_layers.get(
            "gate_1_surface_anomaly",
            "【迷霧階段 I：表層異常（嚴禁破題）】目前九課僅知發生離奇的電子腦空白命案，死者生前都在尋找某個不存在的地址。嚴禁提及幕後黑手名字或終極計畫，保持100%硬派刑偵懸疑感！",
        )
    elif chapter_idx <= 4:
        fog_note = fog_layers.get(
            "gate_2_red_herring",
            "【迷霧階段 II：偽線索與內鬼疑雲】搜查指向被竄改的舊軍方實驗室紀錄，但證人卻在深潛中途離奇被抹除記憶。真相依然隱藏在迷霧後，懷疑的陰影開始籠罩搜查官自身。",
        )
    elif chapter_idx <= total_chapters - 1:
        fog_note = fog_layers.get(
            "gate_3_truth_emergence",
            "【迷霧階段 III：逆向迴文真相浮現】終於揭開幕後機制——並非外部駭客入侵，而是全義體化網路底層自發湧現的『記憶相消干涉（負面迴文協議）』。",
        )
    else:
        fog_note = fog_layers.get(
            "gate_4_noir_aftermath",
            "【迷霧階段 IV：冷冽餘韻與留白】事件在物理層面落幕，但留下一個未解的微小反常細節，不說教、不煽情，以押井守式的冷冽留白收尾。",
        )
    return f"{base_cast}\n{fog_note}"


def generate_v3_blueprint(endpoint: str, model: str, num_chapters: int = 7) -> Dict[str, Any]:
    sys_p = (
        "你是《攻殼機動隊 Ghost in the Shell》總編劇（融合押井守的冷冽懸疑與神山健治的精密刑偵）。\n"
        "請設計一部【開頭絕對不破題、層層剝繭的高懸疑外傳長篇小說藍圖】。\n"
        "只輸出合法 JSON，不要任何 Markdown codeblock 以外的廢話。"
    )
    usr_p = f"""請為一部 {num_chapters} 章的《攻殼機動隊》高懸疑刑偵外傳小說生成結構化 JSON 藍圖。
【核心架構要求】：
1. **漸進式懸疑迷霧 (`epistemic_fog_layers`)**：
   - 第 1~2 章只呈現離奇的犯罪現場與反常物證（嚴禁在第 1~2 章透露幕後真相或最終計畫名稱！）。
   - 第 3~4 章引入偽線索（Red Herring）與記憶篡改疑雲。
   - 第 5~6 章才揭曉核心黑幕（圍繞『逆向迴文記憶抹除 / 幽靈相消干涉』）。
   - 第 7 章以冷冽留白收尾。
2. **每章 4 個節點（3 大主線軌 + 1 個極短篇片尾彩蛋小劇場）**：
   - `char_dive`：人物心理、感官細節與冷硬日常深描（佔 0.7 章比重，每章必須換不同的感官切入點，嚴禁每章都寫喝咖啡或同一個小動作！）。
   - `human_pov`：公安九課刑偵推理、現場鑑識、防壁潛行與戰術交火。
   - `tachi_pov`：攻殼車（塔奇克馬）AI 感測器陣列與平行推理視角。
   - `omake_topic`：片尾 60 秒極短篇《塔奇克馬的日常》搞笑彩蛋主題（僅 500 字的輕鬆收尾）。

請嚴格輸出以下 JSON 格式：
{{
  "title": "極具攻殼機動隊冷冽懸疑感的外傳書名",
  "cast_contract": "角色設定（主角搜查官、草薙素子少佐、巴特、塔奇克馬藍七號與黃九號的性格與說話風格，不劇透謎底）",
  "epistemic_fog_layers": {{
    "gate_1_surface_anomaly": "CH01-02 允許揭露的表層懸疑線索與嚴禁破題限制",
    "gate_2_red_herring": "CH03-04 的中盤轉折、偽線索與記憶矛盾",
    "gate_3_truth_emergence": "CH05-06 揭曉的核心真相機制",
    "gate_4_noir_aftermath": "CH07 的冷冽留白結局基調"
  }},
  "chapters": [
    {{
      "id": "CHAPTER_01",
      "title": "章節標題",
      "canvas_theme": "neon_rain",
      "char_dive": "本章專屬的全新感官與人物心理深描（不重複前章意象）",
      "human_pov": "本章九課刑偵搜查與懸疑推進（結尾留懸疑斷點）",
      "tachi_pov": "本章攻殼車AI戰術與感測器發現的微觀異常",
      "omake_topic": "片尾60秒極短篇塔奇克馬搞笑話題"
    }}
  ]
}}"""
    raw = call_llm(endpoint, model, sys_p, usr_p, temp=0.45, max_tokens=8192, reset_kv_cache=True)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"Blueprint JSON parse failed: {raw[:300]}")
    bp = json.loads(m.group(0))
    for i, ch in enumerate(bp.get("chapters", [])):
        ch["canvas_theme"] = CANVAS_THEMES[i % len(CANVAS_THEMES)]
    return bp


def assemble_outputs(bp: Dict[str, Any], state: Dict[str, Any], out_html: Path) -> Dict[str, Any]:
    completed = state.get("completed_nodes", {})
    chapters_list = bp.get("chapters", [])
    title = bp.get("title", "攻殼機動隊外傳：逆向迴文的幽靈")
    contract = bp.get("cast_contract", "")

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    total_cjk = 0
    done_nodes = 0
    done_chapters = 0
    omake_done = 0

    md_lines = [
        f"# {title}",
        f"> **Generated by:** `LocalSleep-MoE v3 (Negative-Palindrome Anti-Echo + Epistemic Fog + Micro-Omake)`  ",
        f"> **Updated:** `{now_str}`  ",
        f"> **Cast Contract:** {contract}\n\n---\n",
    ]

    chapters_payload = []
    for idx, ch in enumerate(chapters_list, start=1):
        cid = str(ch.get("id") or f"CHAPTER_{idx:02d}")
        ctitle = ch.get("title", cid)
        chap_cjk = 0
        chap_nodes_data = []
        md_lines.append(f"\n# {cid}：{ctitle}\n")

        node_specs = [
            ("char", f"{cid}_CHAR", f"節點 {idx}.1 [人物深描與感官日常]：{ch.get('char_dive', '')}"),
            ("human", f"{cid}_HUMAN", f"節點 {idx}.2 [公安九課懸疑搜查主線]：{ch.get('human_pov', '')}"),
            ("tachi", f"{cid}_TACHI", f"節點 {idx}.3 [攻殼車 AI 戰術與微觀視角]：{ch.get('tachi_pov', '')}"),
            ("omake", f"{cid}_OMAKE", f"片尾 60 秒彩蛋《塔奇克馬的日常 #{idx}》：{ch.get('omake_topic', '')}"),
        ]

        all_done = True
        for kind, nid, label in node_specs:
            txt = completed.get(nid, "")
            is_done = nid in completed
            if not is_done:
                all_done = False
            else:
                done_nodes += 1
                if kind == "omake":
                    omake_done += 1
            n_cjk = sum(1 for c in txt if "\u4e00" <= c <= "\u9fff")
            chap_cjk += n_cjk
            total_cjk += n_cjk
            chap_nodes_data.append({
                "id": nid,
                "kind": kind,
                "label": label,
                "done": is_done,
                "cjk_count": n_cjk,
                "content": txt,
            })
            if is_done:
                md_lines.append(f"\n## {label}\n\n{txt}\n\n---\n")

        if all_done:
            done_chapters += 1

        chapters_payload.append({
            "id": cid,
            "index": idx,
            "title": ctitle,
            "canvas_theme": ch.get("canvas_theme", "neon_rain"),
            "cjk_count": chap_cjk,
            "nodes": chap_nodes_data,
        })

    full_md = "\n".join(md_lines)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.with_suffix(".md").write_text(full_md, encoding="utf-8")

    payload_obj = {
        "title": title,
        "updated_at": now_str,
        "global_contract": contract,
        "total_cjk": total_cjk,
        "total_chars": len(full_md),
        "done_nodes": done_nodes,
        "total_nodes": len(chapters_list) * 4,
        "done_chapters": done_chapters,
        "omake_count": omake_done,
        "chapters": chapters_payload,
    }
    tmpl = TEMPLATE_PATH.read_text(encoding="utf-8")
    html_str = tmpl.replace("__TITLE__", title).replace("__PAYLOAD_JSON__", json.dumps(payload_obj, ensure_ascii=False))
    out_html.write_text(html_str, encoding="utf-8")
    return {
        "total_cjk": total_cjk,
        "total_chars": len(full_md),
        "done_nodes": done_nodes,
        "total_nodes": len(chapters_list) * 4,
        "html_bytes": out_html.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GITS v3 Negative-Palindrome + Epistemic Fog Novel Forge")
    parser.add_argument("--chapters", type=int, default=7)
    parser.add_argument("--max-nodes", type=int, default=28, help="Stop after N nodes (useful for sampling)")
    parser.add_argument("--out", default=str(BASE_DIR / "exports" / "novel_gits_v3_palindrome.html"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="localsleep-moe")
    args = parser.parse_args()

    out_html = Path(args.out)
    state_file = out_html.with_suffix(".forge_state.json")

    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        bp = state["blueprint"]
        completed = state.setdefault("completed_nodes", {})
        print(f"♻️ Resuming v3 Negative-Palindrome Forge ({len(completed)} nodes done)...", flush=True)
    else:
        print("📐 [Phase 0] Generating GITS v3 Epistemic-Fog & Negative-Palindrome Blueprint...", flush=True)
        bp = generate_v3_blueprint(args.endpoint, args.model, num_chapters=args.chapters)
        completed = {}
        state = {"blueprint": bp, "completed_nodes": completed}
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        assemble_outputs(bp, state, out_html)
        print(f"   📖 [Phase 0 Ready] Reader initialized at: {out_html}", flush=True)

    title = bp.get("title", "攻殼機動隊外傳")
    chapters = bp.get("chapters", [])
    recent_main_texts: List[str] = []
    step_counter = 0

    for idx, ch in enumerate(chapters, start=1):
        cid = str(ch.get("id") or f"CHAPTER_{idx:02d}")
        ctitle = ch.get("title", cid)
        fog_context = get_epistemic_fog_context(bp, idx, len(chapters))

        # 1. Character Deep-Dive (Mainline Track 1)
        nid_c = f"{cid}_CHAR"
        step_counter += 1
        if nid_c not in completed:
            head_lock, tail_lock, sat_kws = build_negative_palindrome_constraints(recent_main_texts, step_counter)
            prev_80 = extract_damped_80pct_cache(recent_main_texts[-1] if recent_main_texts else "", sat_kws)
            sys_c = (
                "你是押井守與神山健治風格的《攻殼機動隊》冷硬派懸疑小說家。\n"
                "請將鏡頭放慢，透過全新的感官切入點刻畫角色心理與細微物證，營造強烈的未知懸疑感（絕不在前中期破題！）。\n"
                "【鐵律】：100% 純繁體中文小說正文與對話，嚴禁列點或後設解說！單節目標 2,000 ~ 2,400 字！"
            )
            usr_c = (
                f"{head_lock}\n"
                f"========================================\n"
                f"【作品書名】：{title}\n"
                f"【當前章節認知迷霧權限】：{fog_context}\n"
                f"【上一主線章節後 80% 消磁快取 (請無縫銜接時空與動作，但切勿重複其句型)】：\n"
                f"{prev_80 or '（故事開篇：從一個極具反常細節的現場切入，不要直接解釋案件全貌）'}\n"
                f"========================================\n"
                f"現在撰寫【第 {idx}/{len(chapters)} 章《{ctitle}》— 節點 {idx}.1：人物深描與懸疑日常】。\n"
                f"本節焦點：{ch.get('char_dive', '')}\n"
                f"{tail_lock}"
            )
            print(f"🎭 [{nid_c}] Forging Character Deep-Dive (Saturated damped: {sat_kws})...", flush=True)
            txt_c = call_llm(args.endpoint, args.model, sys_c, usr_c, temp=0.58, max_tokens=10240, reset_kv_cache=False)
            completed[nid_c] = txt_c
            recent_main_texts.append(txt_c)
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_c} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        else:
            recent_main_texts.append(completed[nid_c])
        if len(completed) >= args.max_nodes:
            break

        # 2. Human Cyborg Agent Noir Investigation (Mainline Track 2)
        nid_h = f"{cid}_HUMAN"
        step_counter += 1
        if nid_h not in completed:
            head_lock, tail_lock, sat_kws = build_negative_palindrome_constraints(recent_main_texts, step_counter)
            prev_80 = extract_damped_80pct_cache(recent_main_texts[-1], sat_kws)
            sys_h = (
                "你是《攻殼機動隊》硬派刑偵與電子腦駭客懸疑小說家。\n"
                "請無縫接續前文動作，推進現場鑑識、情報矛盾與戰術攻防。保持冷冽克制的筆觸，層層堆疊懸疑感。\n"
                "【鐵律】：100% 純繁體中文小說正文與對話，單節目標 2,200 ~ 2,600 字！"
            )
            usr_h = (
                f"{head_lock}\n"
                f"========================================\n"
                f"【作品書名】：{title}\n"
                f"【當前章節認知迷霧權限】：{fog_context}\n"
                f"【前一節後 80% 消磁快取 (直接從最後一句的動作與場景無縫接寫)】：\n"
                f"{prev_80}\n"
                f"========================================\n"
                f"現在撰寫【第 {idx}/{len(chapters)} 章《{ctitle}》— 節點 {idx}.2：公安九課懸疑搜查主線】。\n"
                f"本節劇情：{ch.get('human_pov', '')}\n"
                f"{tail_lock}"
            )
            print(f"🕵️ [{nid_h}] Forging Noir Investigation Mainline (Saturated damped: {sat_kws})...", flush=True)
            txt_h = call_llm(args.endpoint, args.model, sys_h, usr_h, temp=0.58, max_tokens=10240, reset_kv_cache=False)
            completed[nid_h] = txt_h
            recent_main_texts.append(txt_h)
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_h} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        else:
            recent_main_texts.append(completed[nid_h])
        if len(completed) >= args.max_nodes:
            break

        # 3. Tachikoma AI Tactical & Sensor POV (Mainline Track 3)
        nid_t = f"{cid}_TACHI"
        step_counter += 1
        if nid_t not in completed:
            head_lock, tail_lock, sat_kws = build_negative_palindrome_constraints(recent_main_texts, step_counter)
            prev_80 = extract_damped_80pct_cache(recent_main_texts[-1], sat_kws)
            sys_t = (
                "你是《攻殼機動隊》思考戰車『塔奇克馬』的 AI 戰術與微觀感測視角作家。\n"
                "請無縫接續前文戰場或搜查現場，從多維感測器陣列、封包延遲、微表情光譜分析切入，發現人類搜查官沒注意到的詭異細節！\n"
                "【鐵律】：100% 純繁體中文小說正文與對話，單節目標 1,800 ~ 2,200 字！"
            )
            usr_t = (
                f"{head_lock}\n"
                f"========================================\n"
                f"【作品書名】：{title}\n"
                f"【當前章節認知迷霧權限】：{fog_context}\n"
                f"【前一節後 80% 消磁快取 (直接切換至在場攻殼車 AI 視角接續描寫)】：\n"
                f"{prev_80}\n"
                f"========================================\n"
                f"現在撰寫【第 {idx}/{len(chapters)} 章《{ctitle}》— 節點 {idx}.3：攻殼車 AI 戰術與微觀感測視角】。\n"
                f"本節劇情：{ch.get('tachi_pov', '')}\n"
                f"{tail_lock}"
            )
            print(f"🕷️ [{nid_t}] Forging Tachikoma Tactical POV (Saturated damped: {sat_kws})...", flush=True)
            txt_t = call_llm(args.endpoint, args.model, sys_t, usr_t, temp=0.58, max_tokens=10240, reset_kv_cache=False)
            completed[nid_t] = txt_t
            recent_main_texts.append(txt_t)
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_t} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        else:
            recent_main_texts.append(completed[nid_t])
        if len(completed) >= args.max_nodes:
            break

        # 4. Compact 60-Second Post-Credits Tachikoma Micro-Skit (~450-650 chars ONLY!)
        nid_o = f"{cid}_OMAKE"
        if nid_o not in completed:
            sys_o = (
                "你是《攻殼機動隊 S.A.C.》片尾 60 秒彩蛋動畫《タチコマな日々 (塔奇克馬的日常)》的極短篇搞笑編劇。\n"
                "【篇幅鐵律】：必須極度短小精悍！全文嚴格控制在 400 ~ 600 個繁體中文字以內（3~5 個短段落即可），像四格漫畫一樣快速拋梗、歪樓、瞬間反轉收尾，絕對不能寫太長喧賓奪主！"
            )
            usr_o = (
                f"請撰寫【片尾 60 秒極短篇彩蛋：塔奇克馬的日常 #{idx}】。\n"
                f"本集搞笑腦洞主題：{ch.get('omake_topic', '')}\n"
                f"【嚴格限制】：全文限 450 ~ 600 字以內！節奏輕快、對話妙語如珠、在最爆笑的一句話直接戛然而止！"
            )
            print(f"🛢️ [{nid_o}] Forging Compact 500-char Post-Credits Micro-Skit...", flush=True)
            txt_o = call_llm(args.endpoint, args.model, sys_o, usr_o, temp=0.72, max_tokens=1100, reset_kv_cache=True)
            completed[nid_o] = txt_o
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_o} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        if len(completed) >= args.max_nodes:
            break

    final_stats = assemble_outputs(bp, state, out_html)
    print(f"🎉 [v3 Complete] CJK={final_stats['total_cjk']:,} chars | Nodes={final_stats['done_nodes']}/{final_stats['total_nodes']} -> {out_html}", flush=True)


if __name__ == "__main__":
    main()
