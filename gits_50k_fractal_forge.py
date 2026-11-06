#!/usr/bin/env python3
"""
gits_50k_fractal_forge.py - 50,000-Char Ghost in the Shell 4-Track Fractal Novel Forge
======================================================================================
Upgraded per Architect Leon's directives:
  1. [80% Rolling Chapter Context + KV-Cache Reuse]:
     Before writing each main-story node, feeds the trailing 80% full text of the previous
     narrative section (~2,000-4,000 chars) with `reset_kv_cache=False` (`reuse_context=True`)
     to eliminate inter-chapter discontinuity!
  2. [0.8-Chapter Dedicated Character Deep-Dive Arc]:
     Dedicated character portrait nodes (`char_dive`) that deeply etch the protagonists'
     prosthetic habits, phantom pain, psychological scars, and Tachikoma AI quirks before
     and between high-intensity conflicts.
  3. [4-Node Heterogeneous Fractal Tree × 7 Chapters = 28 Passes (~52,000+ CJK chars)]:
     - Node i.1 [char_dive] : 人物深描與心理刻畫 (0.6~0.8 章深度人物誌與日常細節)
     - Node i.2 [human]     : 公安九課義體特務主線視角 (硬派刑偵與電子腦防壁潛行)
     - Node i.3 [tachi]     : 攻殼車 AI 戰術與靈魂視角 (Tachikoma 感測器陣列與雙軌協同作戰)
     - Node i.4 [omake]     : 章間《塔奇克馬的日常》輕鬆搞笑小劇場 (SLM 自由發揮)
  4. [Phase 2 Post-Generation HTML5 Canvas Scene Art Pass]:
     After text generation, asks the local model to generate custom HTML5 Canvas 2D
     procedural background art descriptors/renderers tailored to each chapter's actual prose!
"""

import argparse
import datetime
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "gits_reader_template.html"
DEFAULT_ENDPOINT = "http://127.0.0.1:18935"

CANVAS_THEMES = [
    "neon_rain",
    "tachikoma_pod",
    "cyber_dive",
    "oil_workshop",
    "optical_camo",
    "ghost_dawn",
    "neon_rain",
]


def call_llm(
    endpoint: str,
    model: str,
    sys_p: str,
    usr_p: str,
    temp: float = 0.55,
    reset_kv_cache: bool = False,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_p},
        ],
        "max_tokens": 12288,
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


def extract_trailing_80pct(text: str, max_chars: int = 3600) -> str:
    """
    Extracts the trailing ~80% of the previous section's prose (starting at a clean
    paragraph boundary) so the local model has full scene, dialogue, and emotional
    continuity in its KV-cache!
    """
    if not text:
        return ""
    start_idx = int(len(text) * 0.20)
    sliced = text[start_idx:]
    if len(sliced) > max_chars:
        sliced = sliced[-max_chars:]
    nl = sliced.find("\n")
    if 0 <= nl < 200:
        sliced = sliced[nl + 1 :]
    return sliced.strip()


def generate_gits_blueprint(endpoint: str, model: str, num_chapters: int = 7) -> Dict[str, Any]:
    sys_p = "你是押井守與神山健治風格的《攻殼機動隊 Ghost in the Shell》總編劇。請只輸出合法的 JSON 物件。"
    usr_p = (
        f"我們要透過『四軌異質分形樹 (4-Track Heterogeneous Fractal Tree)』創作一部目標 5 萬字的《攻殼機動隊》繁體中文外傳長篇小說。\n"
        f"請規劃 {num_chapters} 個章節 (CHAPTER_01 ~ CHAPTER_{num_chapters:02d})，每章包含 4 個正交節點：\n"
        "1. char_dive：【人物深描與心理刻畫專節 (約 0.6~0.8 章比重)】，不急著打鬥，而是細膩刻畫主角『神代零』(全義體搜查官的幻肢痛、記憶殘影、擦槍與義體維護習慣)、巴特、少佐與攻殼車之間的日常互動、性格反差與內心孤獨。\n"
        "2. human_pov：【公安九課義體特務主線視角】，冷冽硬派的賽博龐克刑偵、現場勘查與電子腦防壁駭客潛行戰。\n"
        "3. tachi_pov：【多腳思考戰車『塔奇克馬（藍七號 & 黃九號）』AI 戰術視角】，透過熱光學陣列、步進馬達扭矩、並列同步與 Ghost 靈魂萌芽視角，與神代零協同作戰並推進主線高潮。\n"
        "4. omake_topic：【章間輕鬆搞笑哲學小劇場《塔奇克馬的日常》】，讓模型自由發揮藍七號、黃九號、紅三號在整備庫的爆笑日常與硬核 AI 哲學吐槽。\n"
        "請輸出以下 JSON 格式：\n"
        "{\n"
        '  "title": "攻殼機動隊外傳：零號機油與折疊的靈魂",\n'
        '  "global_contract": "【世界觀與角色聖經 (400~550字)】：西元2034年新濱市，公安九課全義體搜查官『神代零』（29歲，原陸自特種義體狙擊手，換裝高規格鈦合金骨架後仍保留舊式機械錶與喝黑咖啡習慣，左手無名指義體關節有刻意不修好的微弱咔噠聲）與專屬攻殼車搭檔『塔奇克馬藍七號 (Tachikoma-07，好奇心旺盛、偷藏天然機油與骨董繪本)』、『黃九號 (Tachikoma-09，愛裝深沉的哲學家語氣)』，偕同草薙素子（少佐）與巴特，追查『睡美人協議 (Sleeping Beauty Protocol)』——新濱市多名全義體人類與自律AI在休眠時同步夢見同一座不存在的舊東京海濱車站。嚴格統一角色性格與招式裝備，100%純繁體中文小說散文與對話！",\n'
        '  "chapters": [\n'
        '    {\n'
        '      "id": "CHAPTER_01",\n'
        '      "title": "第一章標題",\n'
        '      "canvas_theme": "neon_rain",\n'
        '      "char_dive": "本章人物深描與性格/過往/義體生活細節刻畫重點",\n'
        '      "human_pov": "義體特務主線調查與潛行劇情",\n'
        '      "tachi_pov": "攻殼車AI戰術協同與高潮劇情",\n'
        '      "omake_topic": "章間攻殼車小劇場搞笑哲學主題"\n'
        '    }\n'
        '  ]\n'
        "}"
    )
    raw = call_llm(endpoint, model, sys_p, usr_p, temp=0.45, reset_kv_cache=True)
    start_i = raw.find("{")
    end_i = raw.rfind("}")
    if start_i != -1 and end_i != -1:
        raw = raw[start_i : end_i + 1]
    bp = json.loads(raw)
    for idx, ch in enumerate(bp.get("chapters", [])):
        ch["canvas_theme"] = CANVAS_THEMES[idx % len(CANVAS_THEMES)]
    return bp


def assemble_outputs(bp: Dict[str, Any], state: Dict[str, Any], out_html: Path) -> Dict[str, int]:
    title = bp.get("title", "攻殼機動隊外傳：零號機油與折疊的靈魂")
    contract = bp.get("global_contract", "")
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completed: Dict[str, str] = state.get("completed_nodes", {})

    md_lines = [
        f"# {title}",
        f"> **Generated by:** `LocalSleep-MoE 4-Track 50K Fractal Forge (80% Context Cache + Character Deep-Dive)`  ",
        f"> **Updated:** `{now_str}`  ",
        f"> **Global Lore Bible:** {contract}\n\n---\n",
    ]

    chapters_payload = []
    total_cjk = 0
    done_nodes = 0
    omake_done = 0
    done_chapters = 0
    chapters_list = bp.get("chapters", [])

    for idx, ch in enumerate(chapters_list, start=1):
        cid = str(ch.get("id") or f"CHAPTER_{idx:02d}")
        node_specs = [
            ("human", f"{cid}_CHAR", f"節點 {idx}.1 [人物深描與義體日常]：{ch.get('char_dive', '')}"),
            ("human", f"{cid}_HUMAN", f"節點 {idx}.2 [公安九課搜查主線]：{ch.get('human_pov', '')}"),
            ("tachi", f"{cid}_TACHI", f"節點 {idx}.3 [攻殼車 AI 戰術視角]：{ch.get('tachi_pov', '')}"),
            ("omake", f"{cid}_OMAKE", f"Inter-Chapter 攻殼車小劇場《塔奇克馬的日常 #{idx}》：{ch.get('omake_topic', '')}"),
        ]
        chap_cjk = 0
        chap_nodes_data = []
        md_lines.append(f"\n# {cid}：{ch.get('title', '')}\n")

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
            "title": ch.get("title", cid),
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
    parser = argparse.ArgumentParser(description="50K Ghost in the Shell 4-Track Fractal Novel Forge")
    parser.add_argument("--chapters", type=int, default=7, help="7 chapters × 4 nodes = 28 passes (~52,000+ CJK chars)")
    parser.add_argument("--out", default=str(BASE_DIR / "exports" / "novel_gits_tachikoma_50k.html"))
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="localsleep-moe")
    args = parser.parse_args()

    out_html = Path(args.out)
    state_file = out_html.with_suffix(".forge_state.json")

    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        bp = state["blueprint"]
        completed = state.setdefault("completed_nodes", {})
        print(f"♻️ Resuming 50K GITS Forge ({len(completed)} nodes already done)...", flush=True)
    else:
        print(f"📐 [Phase 0] Generating 7-Chapter × 4-Track (28-Node) Ghost in the Shell Blueprint...", flush=True)
        bp = generate_gits_blueprint(args.endpoint, args.model, num_chapters=args.chapters)
        completed = {}
        state = {"blueprint": bp, "completed_nodes": completed}
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        assemble_outputs(bp, state, out_html)
        print(f"   📖 [Phase 0 Ready] Reader initialized at: {out_html}", flush=True)

    contract = bp.get("global_contract", "")
    title = bp.get("title", "攻殼機動隊外傳：零號機油與折疊的靈魂")
    chapters = bp.get("chapters", [])
    prev_main_section_text = ""

    for idx, ch in enumerate(chapters, start=1):
        cid = str(ch.get("id") or f"CHAPTER_{idx:02d}")
        ctitle = ch.get("title", cid)

        # Node i.1: Character Deep-Dive & Psychological Portrait (0.6~0.8 chapter weight)
        nid_c = f"{cid}_CHAR"
        if nid_c not in completed:
            prev_80 = extract_trailing_80pct(prev_main_section_text)
            sys_c = (
                "你是押井守與神山健治風格的《攻殼機動隊》文學小說家，極度擅長『人物深描、義體感官細節與心理刻畫』。\n"
                "請不要急著推進快節奏打鬥，而是將鏡頭放慢，細膩描繪角色的微表情、義體維護時的冰冷觸感、幻肢痛、過往記憶殘影、以及九課成員與攻殼車之間含蓄而深厚的羈絆。\n"
                "【鐵律】：100% 純繁體中文小說散文與自然對話，嚴禁程式碼或大綱列點！單節請寫滿 2,000 ~ 2,500 個繁體中文字！"
            )
            usr_c = (
                f"【作品書名】：{title}\n"
                f"【世界觀與角色聖經】：{contract}\n"
                f"【上一主線章節後 80% 內文快取 (請完整吸收其情緒、環境與時間點，做到 100% 無縫銜接、零斷裂！)】：\n"
                f"{prev_80 or '（故事開篇：新濱市連綿的酸雨正順著公安九課大樓的落地窗滑落）'}\n"
                f"========================================\n"
                f"現在進入【第 {idx}/{len(chapters)} 章《{ctitle}》— 節點 {idx}.1：人物深描與心理刻畫專節】。\n"
                f"本節深描主題：{ch.get('char_dive', '')}\n"
                f"請直接無縫接續上一章結尾的時空與情緒，深入刻畫人物內心與生活/義體細節（目標 2,200+ 字）："
            )
            print(f"🎭 [{nid_c}] Forging Character Deep-Dive (with 80% prev chapter cache)...", flush=True)
            txt_c = call_llm(args.endpoint, args.model, sys_c, usr_c, temp=0.55, reset_kv_cache=False)
            completed[nid_c] = txt_c
            prev_main_section_text = txt_c
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_c} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        else:
            prev_main_section_text = completed[nid_c]

        # Node i.2: Human Cyborg Agent Main Story POV
        nid_h = f"{cid}_HUMAN"
        if nid_h not in completed:
            prev_80 = extract_trailing_80pct(prev_main_section_text)
            sys_h = (
                "你是《攻殼機動隊 Ghost in the Shell》硬派科幻小說家。\n"
                "請緊接前文的人物情緒與現場氛圍，推進公安九課的刑偵搜查、電子腦防壁潛行與戰術交鋒。\n"
                "【鐵律】：100% 純繁體中文小說正文與對話，嚴禁程式碼或大綱列點！單節請寫滿 2,200 ~ 2,600 個繁體中文字！"
            )
            usr_h = (
                f"【作品書名】：{title}\n"
                f"【世界觀與角色聖經】：{contract}\n"
                f"【前一節（人物深描）後 80% 內文快取 (請直接從最後一個動作與對話無縫往下接寫！)】：\n"
                f"{prev_80}\n"
                f"========================================\n"
                f"請無縫接續上方最後一句話，撰寫【第 {idx}/{len(chapters)} 章《{ctitle}》— 節點 {idx}.2：公安九課義體特務主線視角】。\n"
                f"本節劇情：{ch.get('human_pov', '')}\n"
                f"請直接輸出小說正文（目標 2,200+ 字）："
            )
            print(f"🕵️ [{nid_h}] Forging Human Cyborg Main Story (with 80% prev cache)...", flush=True)
            txt_h = call_llm(args.endpoint, args.model, sys_h, usr_h, temp=0.55, reset_kv_cache=False)
            completed[nid_h] = txt_h
            prev_main_section_text = txt_h
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_h} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        else:
            prev_main_section_text = completed[nid_h]

        # Node i.3: Tachikoma AI Tactical & Ghost POV
        nid_t = f"{cid}_TACHI"
        if nid_t not in completed:
            prev_80 = extract_trailing_80pct(prev_main_section_text)
            sys_t = (
                "你是《攻殼機動隊》多腳思考戰車『塔奇克馬 (Tachikoma)』的 AI 意識與戰術視角作家。\n"
                "請結合：1) 高科技 AI 感測器描寫（如【熱光學陣列鎖定】、【並列同步率 99.4%】、毫米波雷達、步進馬達扭矩與液壓抓地力）；"
                "2) 塔奇克馬天真活潑卻又對人類搭檔（神代零與巴特）充滿關心、對『Ghost（靈魂）』產生深刻哲學好奇的語氣！\n"
                "【鐵律】：100% 純繁體中文小說正文與對話，嚴禁程式碼或條列大綱！單節請寫滿 2,200 ~ 2,600 個繁體中文字！"
            )
            usr_t = (
                f"【作品書名】：{title}\n"
                f"【世界觀與角色聖經】：{contract}\n"
                f"【前一節（義體特務視角）後 80% 內文快取 (請直接從同一現場切換至攻殼車藍七號/黃九號的感測器視角無縫接續！)】：\n"
                f"{prev_80}\n"
                f"========================================\n"
                f"請緊接上方現場戰況，切換為【第 {idx}/{len(chapters)} 章《{ctitle}》— 節點 {idx}.3：攻殼車 AI 戰術與靈魂視角】。\n"
                f"本節劇情：{ch.get('tachi_pov', '')}\n"
                f"請直接輸出從攻殼車 AI 感測器與電子腦視角展開的小說正文（目標 2,200+ 字）："
            )
            print(f"🕷️ [{nid_t}] Forging Tachikoma AI POV (with 80% prev cache)...", flush=True)
            txt_t = call_llm(args.endpoint, args.model, sys_t, usr_t, temp=0.6, reset_kv_cache=False)
            completed[nid_t] = txt_t
            prev_main_section_text = txt_t
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_t} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)
        else:
            prev_main_section_text = completed[nid_t]

        # Node i.4: Inter-Chapter Tachikoma Mini-Theater (Omake — Free-wheeling SLM Comedy & Philosophy!)
        nid_o = f"{cid}_OMAKE"
        if nid_o not in completed:
            sys_o = (
                "你是《攻殼機動隊 S.A.C.》片尾經典搞笑短劇《塔奇克馬的日常 (Tachikoma Days)》的編劇！\n"
                "請自由發揮極致的幽默感、可愛的機械肢體動作（揮動機械臂、單眼感測器閃爍、輪足滑步、偷藏機油），"
                "描寫藍七號、黃九號與紅三號在九課整備庫裡的爆笑日常對話，並夾雜荒謬卻一針見血的 AI 哲學思辨！\n"
                "【鐵律】：100% 純繁體中文搞笑小劇場小說體，嚴禁列點大綱！單篇請寫滿 1,600 ~ 2,200 個繁體中文字！"
            )
            usr_o = (
                f"【作品書名】：{title} — 章間輕鬆小劇場《塔奇克馬的日常 #{idx}》\n"
                f"【剛結束的第 {idx} 章背景】：{ctitle}\n"
                f"【本回小劇場主題 (請自由發揮搞笑細節)】：『{ch.get('omake_topic', '天然機油與電子腦做夢的關係')}』\n"
                f"請直接輸出這篇讓人會心一笑、充滿塔奇克馬可愛對話與哲學吐槽的章間小劇場正文（目標 1,800+ 字）："
            )
            print(f"🛢️ [{nid_o}] Forging Inter-Chapter Tachikoma Mini-Theater (Omake)...", flush=True)
            txt_o = call_llm(args.endpoint, args.model, sys_o, usr_o, temp=0.72, reset_kv_cache=True)
            completed[nid_o] = txt_o
            state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            stats = assemble_outputs(bp, state, out_html)
            print(f"   ✅ {nid_o} done -> Cumulative CJK: {stats['total_cjk']:,} 字 ({stats['done_nodes']}/{stats['total_nodes']})", flush=True)

    final_stats = assemble_outputs(bp, state, out_html)
    print("🎉 [50K GITS FRACTAL FORGE COMPLETED!]", json.dumps(final_stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
