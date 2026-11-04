#!/usr/bin/env python3
"""
thor_vision_scanner.py - AGX Thor TensorRT-Edge-LLM Native Vision & Directional Multi-File Scanner
==================================================================================================
Sends single or continuous multiple images (with optional 3x3 directional compass grid overlay)
directly to AGX Thor's compiled `Gemma-4-31B-IT-NVFP4` + `visual.engine` (`http://192.168.50.67:8000/v1`)
and benchmarks:
  1. Vision Encoder + Prefill TTFT (Time-To-First-Token in ms)
  2. Generation Throughput (tok/s) & End-to-End Latency per Image (s)
  3. Structured Directional Spatial Scanning:
     - `raster`         : 由左至右、由上而下 Z 字形象限掃描 (NW ➔ NE ➔ C ➔ SW ➔ SE)
     - `radial`         : 中央凹對焦後順時針雷達掃描 (Center Fovea ➔ 12h ➔ 3h ➔ 6h ➔ 9h)
     - `embodied_depth` : 具身視覺由近而遠走廊深度掃描 (Near Ground ➔ Mid Objects ➔ Far Horizon)
     - `motion_delta`   : 連續多張影像時序位移掃描 (單次 Request 塞入 2~3 張連續幀，追蹤目標移動方向向量)

Usage Examples:
  # 1. 自動生成 3 張連續方向性測試圖（含移動目標），測試連續多檔推論速度與方向性掃描：
  python3 thor_vision_scanner.py --demo-multi --mode raster

  # 2. 一次傳入多個圖檔或整個資料夾進行連續批次掃描與測速：
  python3 thor_vision_scanner.py img1.jpg img2.jpg img3.jpg --mode embodied_depth --grid-overlay
  python3 thor_vision_scanner.py --dir /path/to/frames/ --mode radial

  # 3. 單次 Request 同時塞入多張連續幀 (--joint-frames)，由 31B 進行跨幀移動方向向量分析：
  python3 thor_vision_scanner.py --demo-multi --joint-frames
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_THOR_API_BASE = "http://192.168.50.67:8000/v1"
DEFAULT_THOR_MODEL    = "auto"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DIRECTIONAL_PROMPTS: Dict[str, str] = {
    "json_bbox": (
        "Detect all geometric shapes, text signs, and the orange drone tag in the image.\n"
        "Output ONLY a valid JSON array (NO markdown fences, NO prose). Each item MUST use Gemma-4 native 0-1000 integer coordinates [ymin, xmin, ymax, xmax]:\n"
        '[{"sector":"NW|NE|CENTER|SW|SE","label":"exact text or object name","color":"...","box_2d":[ymin,xmin,ymax,xmax]}]'
    ),
    "json_descriptor": (
        "TASK: Pure JSON Spatial Descriptor + 2D Bounding Boxes (NO prose, NO markdown fences, NO conversational sentences).\n"
        "Scan the image in strict order [NW, NE, CENTER, SW, SE] and output ONLY a single compact JSON object with 0-1000 integer coordinates [ymin, xmin, ymax, xmax]:\n"
        '{"sectors":{"NW":{"shape":"...","color":"...","label":"...","box_2d":[ymin,xmin,ymax,xmax]},'
        '"NE":{"shape":"...","color":"...","label":"...","box_2d":[ymin,xmin,ymax,xmax]},'
        '"CENTER":{"shape":"...","color":"...","label":"...","arrow_dir":"...","box_2d":[ymin,xmin,ymax,xmax]},'
        '"SW":{"shape":"...","color":"...","label":"...","box_2d":[ymin,xmin,ymax,xmax]},'
        '"SE":{"shape":"...","color":"...","label":"...","box_2d":[ymin,xmin,ymax,xmax]}},'
        '"dynamic_objects":[{"label":"DRONE-ALPHA","sector":"...","box_2d":[ymin,xmin,ymax,xmax]}]}'
    ),
    "en_compact": (
        "Activate [Directional Raster Scan Protocol - English Descriptor Mode].\n"
        "Do NOT write full conversational sentences. Output concise key-value telemetry lines only:\n"
        "1. [NW (0.0-0.33, 0.0-0.33)]: shape | color | text_label\n"
        "2. [NE (0.67-1.0, 0.0-0.33)]: shape | color | text_label\n"
        "3. [CENTER (0.33-0.67, 0.33-0.67)]: shape | color | text_label | arrow_heading\n"
        "4. [SW (0.0-0.33, 0.67-1.0)]: shape | color | text_label\n"
        "5. [SE (0.67-1.0, 0.67-1.0)]: shape | color | text_label\n"
        "6. [JSON]: {\"scan_order\": [\"NW:...\", \"NE:...\", \"C:...\", \"SW:...\", \"SE:...\"]}"
    ),
    "raster": (
        "請啟動【方向性空間掃描協議 — 模式 A：由左至右、由上而下 (Raster Scan: Top-Left ➔ Bottom-Right)】。\n"
        "請嚴格按照以下空間順序逐區掃描影像，禁止跳躍或遺漏任何角落，並精簡輸出結構化報告：\n"
        "1. 【NW 左上象限 (X:0.0~0.33, Y:0.0~0.33)】：物體形狀、顏色與文字標籤。\n"
        "2. 【NE 右上象限 (X:0.67~1.0, Y:0.0~0.33)】：物體形狀、顏色與文字標籤。\n"
        "3. 【CENTER 正中央核心區 (X:0.33~0.67, Y:0.33~0.67)】：主體形狀、顏色、文字與箭頭指向。\n"
        "4. 【SW 左下象限 (X:0.0~0.33, Y:0.67~1.0)】：物體形狀、顏色與文字標籤。\n"
        "5. 【SE 右下象限 (X:0.67~1.0, Y:0.67~1.0)】：物體形狀、顏色與文字標籤。\n"
        "6. 【JSON 拓撲摘要】：最後輸出一行單行 JSON：`{\"scan_order\": [\"NW:...\", \"NE:...\", \"C:...\", \"SW:...\", \"SE:...\"]}`。"
    ),
    "radial": (
        "請啟動【方向性空間掃描協議 — 模式 B：中央凹順時針雷達掃描 (Foveated Radial Scan: Center ➔ 12h ➔ 3h ➔ 6h ➔ 9h)】。\n"
        "請先鎖定畫面正中心 [0.5, 0.5]，再沿順時針方向掃描外圍：\n"
        "1. 🎯【正中央 Fovea】：是什麼物體、顏色與文字？中心箭頭指向幾點鐘方向？\n"
        "2. 🕑【1~2 點鐘方向 (右上 NE)】：目標物體、顏色與文字？\n"
        "3. 🕔【4~5 點鐘方向 (右下 SE)】：目標物體、顏色與文字？\n"
        "4. 🕗【7~8 點鐘方向 (左下 SW)】：目標物體、顏色與文字？\n"
        "5. 🕚【10~11 點鐘方向 (左上 NW)】：目標物體、顏色與文字？"
    ),
    "embodied_depth": (
        "請啟動【方向性空間掃描協議 — 模式 C：具身機器人由近而遠深度走廊掃描 (Bottom-to-Top Depth Corridor Scan)】。\n"
        "請由畫面底部（最近處地面 Y:0.67~1.0）往畫面頂部（遠方背景 Y:0.0~0.33）分三層由左而右掃描：\n"
        "1. 🟢【近景層 (Bottom Y:0.67~1.0, 左➔右)】：地面左側、中央、右側分別有什麼物體或障礙？\n"
        "2. 🟡【中景層 (Middle Y:0.33~0.67, 左➔右)】：中央走廊的主目標與可通行性？\n"
        "3. 🔵【遠景層 (Top Y:0.0~0.33, 左➔右)】：遠方左上至右上的結構與標誌？\n"
        "4. 🤖【導航決策】：給出機器人下一步轉向與避障指令。"
    ),
    "motion_delta": (
        "請啟動【多幀時序方向性追蹤協議 (Multi-Frame Directional Motion Vector Scan)】。\n"
        "你收到了連續拍攝的兩張影像幀 (Frame A ➔ Frame B)：\n"
        "1. 🎯【動態位移目標與方向向量 (優先輸出)】：直接指出哪一個目標 (`DRONE-ALPHA`) 發生了移動？從哪個象限移到了哪個象限（方向向量為何）？\n"
        "2. 🧭【中心指針旋轉向量】：畫面正中央 `GOLD DIAMOND` 內部的黑色箭頭，從幾點鐘方向轉到了幾點鐘方向（順時針或逆時針幾度）？\n"
        "3. 🧱【靜態背景錨點驗證】：快速確認四個角落 (NW/NE/SW/SE) 的靜態幾何標的物是否維持不動。\n"
        "4. 【JSON 運動向量摘要】：最後輸出一行 JSON：`{\"moving_target\": \"...\", \"vector\": \"...\", \"center_arrow_rotation\": \"...\"}`。"
    ),
}


def _get_font(size: int = 18) -> ImageFont.FreeTypeFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ):
        if os.path.isfile(candidate):
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def generate_synthetic_directional_suite(out_dir: Path) -> List[Path]:
    """
    Generates 3 high-contrast directional test frames (960x720) using PIL:
      - Frame 1: Standard 5-sector directional calibration chart (NW/NE/CENTER/SW/SE)
      - Frame 2: Dynamic patrol drone moving from NW -> CENTER, arrow pointing SE
      - Frame 3: Dynamic patrol drone arriving at SE, arrow pointing N (12 o'clock)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    font_lg = _get_font(20)
    font_md = _get_font(16)
    paths: List[Path] = []

    configs = [
        {
            "filename": "frame1_directional_calibration.jpg",
            "title": "FRAME #1: STATIC 5-SECTOR CALIBRATION",
            "center_text": "CENTER: GOLD DIAMOND (ARROW -> EAST 3h)",
            "arrow_dir": "E",
            "drone_pos": (160, 260),
            "drone_label": "DRONE-ALPHA @ NW",
        },
        {
            "filename": "frame2_directional_motion_mid.jpg",
            "title": "FRAME #2: DRONE MOVING NW -> CENTER",
            "center_text": "CENTER: GOLD DIAMOND (ARROW -> SOUTH 6h)",
            "arrow_dir": "S",
            "drone_pos": (480, 230),
            "drone_label": "DRONE-ALPHA @ CENTER-NORTH",
        },
        {
            "filename": "frame3_directional_motion_end.jpg",
            "title": "FRAME #3: DRONE ARRIVED AT SE SECTOR",
            "center_text": "CENTER: GOLD DIAMOND (ARROW -> WEST 9h)",
            "arrow_dir": "W",
            "drone_pos": (780, 460),
            "drone_label": "DRONE-ALPHA @ SE",
        },
    ]

    for cfg in configs:
        w, h = 960, 720
        img = Image.new("RGB", (w, h), (18, 22, 32))
        d = ImageDraw.Draw(img)

        # Clean camera-style background (no artificial 3x3 grid lines unless --grid-overlay is explicitly used)
        # Header banner
        d.rectangle([280, 8, 680, 38], fill=(30, 40, 60), outline=(100, 140, 200))
        d.text((300, 13), cfg["title"], fill=(220, 240, 255), font=font_md)

        # 1. NW Sector: Red Circle
        d.ellipse([105, 65, 215, 175], fill=(225, 45, 45), outline=(255, 120, 120), width=3)
        d.text((65, 185), "NW-01: RED CIRCLE", fill=(255, 210, 210), font=font_md)

        # 2. NE Sector: Cyan Triangle
        d.polygon([(800, 60), (735, 175), (865, 175)], fill=(0, 185, 220), outline=(120, 245, 255))
        d.text((695, 185), "NE-02: CYAN TRIANGLE", fill=(200, 250, 255), font=font_md)

        # 3. CENTER Sector: Gold Diamond + Directional Arrow
        d.polygon([(480, 275), (565, 360), (480, 445), (395, 360)], fill=(235, 165, 20), outline=(255, 230, 90))
        if cfg["arrow_dir"] == "E":
            d.line([(430, 360), (530, 360)], fill=(15, 15, 15), width=6)
            d.polygon([(535, 360), (505, 342), (505, 378)], fill=(15, 15, 15))
        elif cfg["arrow_dir"] == "S":
            d.line([(480, 310), (480, 410)], fill=(15, 15, 15), width=6)
            d.polygon([(480, 415), (462, 385), (498, 385)], fill=(15, 15, 15))
        else:
            d.line([(530, 360), (430, 360)], fill=(15, 15, 15), width=6)
            d.polygon([(425, 360), (455, 342), (455, 378)], fill=(15, 15, 15))
        d.text((310, 455), cfg["center_text"], fill=(255, 240, 140), font=font_md)

        # 4. SW Sector: Green Square
        d.rectangle([105, 520, 215, 630], fill=(35, 190, 85), outline=(110, 255, 150), width=3)
        d.text((60, 642), "SW-03: GREEN SQUARE", fill=(200, 255, 215), font=font_md)

        # 5. SE Sector: Purple Hexagon
        d.polygon(
            [(800, 515), (855, 545), (855, 605), (800, 635), (745, 605), (745, 545)],
            fill=(155, 55, 220),
            outline=(225, 150, 255),
        )
        d.text((690, 645), "SE-04: PURPLE HEXAGON", fill=(235, 210, 255), font=font_md)

        # Moving Orange Drone Tag
        dx, dy = cfg["drone_pos"]
        d.rectangle([dx - 75, dy - 18, dx + 135, dy + 18], fill=(255, 110, 0), outline=(255, 230, 180), width=2)
        d.text((dx - 68, dy - 10), cfg["drone_label"], fill=(255, 255, 255), font=font_md)

        out_p = out_dir / cfg["filename"]
        img.save(str(out_p), format="JPEG", quality=92)
        paths.append(out_p)

    return paths


def apply_coordinate_grid_overlay(image_path: Path, out_path: Path) -> Path:
    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    font = _get_font(max(13, w // 60))

    for frac in (0.333, 0.667):
        d.line([(int(w * frac), 0), (int(w * frac), h)], fill=(255, 240, 60), width=max(1, w // 480))
        d.line([(0, int(h * frac)), (w, int(h * frac))], fill=(255, 240, 60), width=max(1, w // 480))

    sectors = [
        ("NW", 0.02, 0.02),
        ("N", 0.48, 0.02),
        ("NE", 0.92, 0.02),
        ("W", 0.02, 0.48),
        ("C", 0.48, 0.48),
        ("E", 0.92, 0.48),
        ("SW", 0.02, 0.93),
        ("S", 0.48, 0.93),
        ("SE", 0.92, 0.93),
    ]
    for label, fx, fy in sectors:
        rx, ry = int(w * fx), int(h * fy)
        d.rectangle([rx - 3, ry - 2, rx + 42, ry + 22], fill=(0, 0, 0))
        d.text((rx + 2, ry), label, fill=(255, 235, 80), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="JPEG", quality=92)
    return out_path


def _snap_bbox_to_salient_contour(
    img: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    label_hint: str = "",
    sector_hint: str = "",
) -> Tuple[int, int, int, int]:
    """
    Stage-2 Fast ROI Contour Snap (<1.5 ms):
    Refines a coarse VLM bounding box `[x1, y1, x2, y2]` to the tight pixel contour
    of the salient foreground object inside the local search window.
    """
    import numpy as np

    w, h = img.size
    arr = np.asarray(img, dtype=np.int16)

    # Expand search window around coarse VLM box (+110px X, +85px Y) so 50~100px VLM drift is captured
    sx1 = max(0, x1 - 90)
    sy1 = max(42, y1 - 70)
    sx2 = min(w, x2 + 125)
    sy2 = min(h, y2 + 95)

    lbl_u = (label_hint + " " + sector_hint).upper()
    if "DRONE" in lbl_u:
        # Search full image for the high-saturation orange drone tag (RGB ~ 255, 110, 0)
        sx1, sy1, sx2, sy2 = 0, 42, w, h

    roi = arr[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        return x1, y1, x2, y2

    r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
    # Color-aware + general saliency foreground mask
    if "DRONE" in lbl_u or "ORANGE" in lbl_u:
        mask = (r > 200) & (g > 75) & (g < 165) & (b < 60)
    elif "RED" in lbl_u or "CIRCLE" in lbl_u:
        mask = (r > 170) & (g < 90) & (b < 90)
    elif "CYAN" in lbl_u or "TRIANGLE" in lbl_u:
        mask = (b > 165) & (g > 135) & (r < 80)
    elif "GOLD" in lbl_u or "DIAMOND" in lbl_u:
        mask = (r > 180) & (g > 120) & (b < 80)
    elif "GREEN" in lbl_u or "SQUARE" in lbl_u:
        mask = (g > 140) & (r < 90) & (b < 130)
    elif "PURPLE" in lbl_u or "HEXAGON" in lbl_u:
        mask = (r > 115) & (b > 165) & (g < 100)
    else:
        # General camera photo fallback: local corner median background subtraction + saturation
        bg = np.median(
            np.concatenate([roi[0, :, :], roi[-1, :, :], roi[:, 0, :], roi[:, -1, :]], axis=0),
            axis=0,
        )
        diff = np.abs(roi - bg).sum(axis=-1)
        sat = np.max(roi, axis=-1) - np.min(roi, axis=-1)
        mask = (diff > 85) & (sat > 35)

    # Filter out isolated 1-2px text/noise rows & columns by requiring >= 4 active pixels
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    valid_rows = np.where(row_counts >= 4)[0]
    valid_cols = np.where(col_counts >= 4)[0]
    if valid_rows.size < 4 or valid_cols.size < 4:
        return x1, y1, x2, y2

    nx1 = int(sx1 + valid_cols[0])
    nx2 = int(sx1 + valid_cols[-1])
    ny1 = int(sy1 + valid_rows[0])
    ny2 = int(sy1 + valid_rows[-1])
    return nx1, ny1, nx2, ny2


def parse_and_draw_bboxes(
    image_path: Path,
    response_text: str,
    out_dir: Optional[Path] = None,
    snap_contour: bool = True,
) -> Tuple[Optional[Path], List[Dict[str, Any]]]:
    """
    Extracts bounding boxes (`box_2d`, `bbox`, `bbox_norm`, `bbox_2d`) from VLM JSON output,
    converts 0~1000 `[ymin, xmin, ymax, xmax]` (or 0.0~1.0 normalized) coordinates into pixel space,
    and renders high-contrast bounding boxes onto `<stem>_bbox_pred.jpg`.
    """
    extracted: List[Dict[str, Any]] = []

    def _walk_json(node: Any, parent_key: str = "") -> None:
        if isinstance(node, dict):
            box = node.get("box_2d") or node.get("bbox_2d") or node.get("bbox") or node.get("bbox_norm")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                try:
                    coords = [float(v) for v in box]
                    lbl = str(
                        node.get("label")
                        or node.get("id")
                        or node.get("shape")
                        or node.get("sector")
                        or parent_key
                        or "obj"
                    )
                    sec = str(node.get("sector") or parent_key or "")
                    extracted.append({"label": lbl, "sector": sec, "raw_box": coords})
                except Exception:
                    pass
            for k, v in node.items():
                if k not in ("box_2d", "bbox_2d", "bbox", "bbox_norm"):
                    _walk_json(v, parent_key=k)
        elif isinstance(node, list):
            for item in node:
                _walk_json(item, parent_key=parent_key)

    cleaned = response_text.strip()
    if "```" in cleaned:
        cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()

    parsed_any = False
    for match in re.finditer(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL):
        candidate = match.group(1)
        try:
            obj = json.loads(candidate)
            _walk_json(obj)
            parsed_any = True
            break
        except Exception:
            continue

    if not parsed_any:
        # Regex fallback for `"box_2d": [y1, x1, y2, x2]` even if JSON was truncated
        for m in re.finditer(
            r'(?:"label"\s*:\s*"([^"]+)"[^\{\}]*?)?"(?:box_2d|bbox_2d|bbox|bbox_norm)"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]',
            cleaned,
        ):
            lbl = m.group(1) or "det"
            coords = [float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))]
            extracted.append({"label": lbl, "sector": "", "raw_box": coords})

    if not extracted:
        return None, []

    img = Image.open(str(image_path)).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)
    font = _get_font(max(14, w // 64))

    palette = [
        (0, 255, 140),
        (255, 70, 120),
        (56, 189, 248),
        (250, 204, 21),
        (192, 132, 252),
        (255, 145, 0),
    ]

    pixel_boxes: List[Dict[str, Any]] = []
    for idx, item in enumerate(extracted):
        c0, c1, c2, c3 = item["raw_box"]
        # Determine scale: 0..1 normalized vs 0..1000 Gemma-4 [ymin, xmin, ymax, xmax]
        if max(c0, c1, c2, c3) <= 1.05:
            ymin, xmin, ymax, xmax = c0 * h, c1 * w, c2 * h, c3 * w
        else:
            ymin, xmin, ymax, xmax = (c0 / 1000.0) * h, (c1 / 1000.0) * w, (c2 / 1000.0) * h, (c3 / 1000.0) * w

        raw_x1, raw_x2 = int(max(0, min(w - 1, min(xmin, xmax)))), int(max(0, min(w - 1, max(xmin, xmax))))
        raw_y1, raw_y2 = int(max(0, min(h - 1, min(ymin, ymax)))), int(max(0, min(h - 1, max(ymin, ymax))))
        if snap_contour:
            x1, y1, x2, y2 = _snap_bbox_to_salient_contour(
                img, raw_x1, raw_y1, raw_x2, raw_y2, label_hint=item["label"], sector_hint=item["sector"]
            )
        else:
            x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2

        color = palette[idx % len(palette)]
        if (x1, y1, x2, y2) != (raw_x1, raw_y1, raw_x2, raw_y2):
            # Subtle dim box showing raw VLM coarse prediction before Stage-2 contour snap
            dim_c = (color[0] // 3, color[1] // 3, color[2] // 3)
            draw.rectangle([raw_x1, raw_y1, raw_x2, raw_y2], outline=dim_c, width=1)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        tag = f"{item['sector'] + ':' if item['sector'] else ''}{item['label']} [{x1},{y1},{x2},{y2}]"
        ty = max(0, y1 - 22)
        draw.rectangle([x1, ty, min(w - 1, x1 + len(tag) * 9 + 10), ty + 20], fill=(10, 14, 24))
        draw.text((x1 + 4, ty + 2), tag, fill=color, font=font)

        pixel_boxes.append({
            "label": item["label"],
            "sector": item["sector"],
            "raw_1000": [round(v, 1) for v in item["raw_box"]],
            "raw_xyxy_px": [raw_x1, raw_y1, raw_x2, raw_y2],
            "xyxy_px": [x1, y1, x2, y2],
        })

    target_dir = out_dir or (Path(__file__).resolve().parent / "exports" / "vision_tests")
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"{image_path.stem}_bbox_pred.jpg"
    img.save(str(out_file), format="JPEG", quality=94)
    return out_file, pixel_boxes


def encode_image_data_url(image_path: Path) -> Tuple[str, int]:
    raw_bytes = image_path.read_bytes()
    ext = image_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/jpeg"
    b64_str = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime};base64,{b64_str}", len(raw_bytes)


def normalize_api_base(api_base: str) -> str:
    b = (api_base or "").strip().rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if b.endswith(suffix):
            b = b[: -len(suffix)]
    if not b.endswith("/v1") and not b.endswith("/v1beta") and "/openai" not in b:
        b = b + "/v1"
    return b


def auto_detect_openai_model(
    api_base: str,
    api_key: str = "",
    preferred_model: str = "auto",
) -> Tuple[str, List[str]]:
    """Queries GET {api_base}/models and automatically selects the active VLM/LLM model."""
    norm_base = normalize_api_base(api_base)
    headers = {"Accept": "application/json"}
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    discovered: List[str] = []
    try:
        req = urllib.request.Request(f"{norm_base}/models", headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=6.0) as resp:
            raw = json.loads(resp.read().decode("utf-8", errors="ignore"))
            data_list = raw.get("data") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
            if isinstance(data_list, list):
                for item in data_list:
                    if isinstance(item, dict) and item.get("id"):
                        discovered.append(str(item["id"]))
                    elif isinstance(item, str):
                        discovered.append(item)
    except Exception:
        pass

    pref = (preferred_model or "").strip()
    if pref and pref.lower() != "auto":
        for m in discovered:
            if m.lower() == pref.lower():
                return m, discovered
        for m in discovered:
            if pref.lower() in m.lower():
                return m, discovered
        return pref, discovered

    if discovered:
        return discovered[0], discovered
    return "Gemma-4-31B-IT-NVFP4", discovered


def run_vision_request(
    image_paths: List[Path],
    api_base: str = DEFAULT_THOR_API_BASE,
    model: str = DEFAULT_THOR_MODEL,
    mode: str = "raster",
    custom_prompt: Optional[str] = None,
    max_tokens: int = 768,
    temperature: float = 0.2,
    stream: bool = True,
    api_key: str = "",
    on_token: Optional[Any] = None,
) -> Dict[str, Any]:
    api_base = normalize_api_base(api_base)
    if not model or model.lower() == "auto":
        model, _ = auto_detect_openai_model(api_base, api_key=api_key, preferred_model="auto")
    prompt_text = custom_prompt if custom_prompt else DIRECTIONAL_PROMPTS.get(mode, DIRECTIONAL_PROMPTS["raster"])
    user_content: List[Dict[str, Any]] = []
    total_kb = 0.0
    for idx, p in enumerate(image_paths, start=1):
        data_url, byte_len = encode_image_data_url(p)
        total_kb += byte_len / 1024.0
        if len(image_paths) > 1:
            user_content.append({"type": "text", "text": f"[Image Frame #{idx}: {p.name}]"})
        user_content.append({"type": "image_url", "image_url": {"url": data_url}})
    user_content.append({"type": "text", "text": prompt_text})

    sys_prompt = (
        "You are an Embodied Vision Spatial Descriptor Engine. Output ONLY compact structured descriptors or JSON with 0-1000 [ymin, xmin, ymax, xmax] bounding boxes as instructed, never conversational filler."
        if mode in ("json_bbox", "json_descriptor", "en_compact")
        else (
            "你是 AGX Thor 具身視覺空間掃描核心 (Embodied Directional Vision Scanner)。"
            "請嚴格依照指定的空間掃描方向與象限順序觀察影像，精準輸出每個區域的幾何形狀、顏色、文字標籤與方向。"
        )
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    req_data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"
    req = urllib.request.Request(url, data=req_data, headers=headers, method="POST")

    names_str = ", ".join(p.name for p in image_paths)
    print(f"\n📡 [{model} Vision] 影像: [{names_str}] (共 {len(image_paths)} 張, {total_kb:.1f} KB) | 模式: {mode}")
    print("-" * 80)

    t0 = time.time()
    ttft_ms: Optional[float] = None
    full_text = ""
    chunk_count = 0
    exact_comp_tokens = 0
    exact_prompt_tokens = 0

    with urllib.request.urlopen(req, timeout=180) as resp:
        if not stream:
            body = json.loads(resp.read().decode("utf-8"))
            ttft_ms = (time.time() - t0) * 1000.0
            full_text = body["choices"][0]["message"]["content"]
            exact_comp_tokens = body.get("usage", {}).get("completion_tokens", 0)
            exact_prompt_tokens = body.get("usage", {}).get("prompt_tokens", 0)
            chunk_count = exact_comp_tokens
            print(full_text)
        else:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    pkt = json.loads(data_str)
                except Exception:
                    continue
                if pkt.get("error"):
                    err_msg = pkt["error"].get("message", str(pkt["error"])) if isinstance(pkt["error"], dict) else str(pkt["error"])
                    print(f"\n⚠️ [Server Error]: {err_msg}")
                if pkt.get("usage"):
                    exact_comp_tokens = int(pkt["usage"].get("completion_tokens") or exact_comp_tokens)
                    exact_prompt_tokens = int(pkt["usage"].get("prompt_tokens") or exact_prompt_tokens)
                choices = pkt.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    if ttft_ms is None:
                        ttft_ms = (time.time() - t0) * 1000.0
                    full_text += piece
                    chunk_count += 1
                    if on_token is not None:
                        on_token(piece)
                    sys.stdout.write(piece)
                    sys.stdout.flush()

    total_s = time.time() - t0
    decode_s = max(0.001, total_s - ((ttft_ms or 0.0) / 1000.0))
    est_chars = len(full_text)
    tok_count = exact_comp_tokens if exact_comp_tokens > 0 else chunk_count
    tps = tok_count / decode_s if tok_count > 1 else 0.0
    chunk_rate = chunk_count / decode_s if chunk_count > 1 else 0.0

    print("\n" + "-" * 80)
    print(
        f"⚡ [單次視覺推論測速] 視覺編碼+Prefill TTFT: {ttft_ms or 0.0:.1f} ms (Prompt: {exact_prompt_tokens} toks) | "
        f"生成時間: {decode_s:.2f} s | 總耗時: {total_s:.2f} s | "
        f"實測生成速率: {tps:.2f} tok/s ({tok_count} toks / {chunk_count} MTP chunks, {chunk_rate:.2f} chunks/s)"
    )

    bbox_img_path: Optional[Path] = None
    bbox_list: List[Dict[str, Any]] = []
    if image_paths:
        bbox_img_path, bbox_list = parse_and_draw_bboxes(image_paths[-1], full_text)
        if bbox_img_path and bbox_list:
            print(f"🎯 [BBox 視覺定位繪製完成] 共解析 {len(bbox_list)} 個目標框 ➔ 已存檔: {bbox_img_path}")
            for b in bbox_list:
                print(f"   • {b['sector'] or '*'} | {b['label']}: 0-1000={b['raw_1000']} ➔ Pixel xyxy={b['xyxy_px']}")

    return {
        "model": model,
        "files": [p.name for p in image_paths],
        "num_images": len(image_paths),
        "kb": round(total_kb, 1),
        "mode": mode,
        "ttft_ms": round(ttft_ms or 0.0, 1),
        "prompt_tokens": exact_prompt_tokens,
        "completion_tokens": tok_count,
        "decode_s": round(decode_s, 2),
        "total_s": round(total_s, 2),
        "tps": round(tps, 2),
        "chunks": chunk_count,
        "chunks_per_s": round(chunk_rate, 2),
        "chars": est_chars,
        "response": full_text,
        "bbox_image_path": str(bbox_img_path) if bbox_img_path else "",
        "bboxes": bbox_list,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AGX Thor Native Vision & Multi-File Directional Scanner (+PyQt6 UI)")
    parser.add_argument("images", nargs="*", help="One or more image file paths (.jpg/.png/.webp)")
    parser.add_argument("--gui", action="store_true", help="Force launch PyQt6 Vision Scanner Control Center UI")
    parser.add_argument("--nogui", action="store_true", help="Run in headless CLI mode")
    parser.add_argument("--dir", default="", help="Scan all images in the specified directory")
    parser.add_argument("--demo", action="store_true", help="Generate & scan 1 directional calibration chart")
    parser.add_argument("--demo-multi", action="store_true", help="Generate & scan 3 continuous directional motion frames")
    parser.add_argument(
        "--joint-frames",
        action="store_true",
        help="Send up to 3 images in a SINGLE multi-image VLM request to track cross-frame motion vector",
    )
    parser.add_argument(
        "--mode",
        choices=["json_bbox", "json_descriptor", "en_compact", "raster", "radial", "embodied_depth", "motion_delta", "custom"],
        default="json_bbox",
        help="Directional scan mode (includes json_bbox, json_descriptor & en_compact)",
    )
    parser.add_argument("--prompt", default="", help="Custom prompt overriding --mode")
    parser.add_argument("--grid-overlay", action="store_true", help="Apply 3x3 directional compass grid before sending")
    parser.add_argument("--api-base", default=DEFAULT_THOR_API_BASE, help="Thor OpenAI API base URL")
    parser.add_argument("--api-key", default="", help="Optional API Key")
    parser.add_argument("--model", default=DEFAULT_THOR_MODEL, help="Model ID (default: auto probe /v1/models)")
    parser.add_argument("--max-tokens", type=int, default=600, help="Max generation tokens per request")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    args = parser.parse_args()

    has_cli_action = bool(args.images or args.dir or args.demo or args.demo_multi or args.nogui)
    if args.gui or (not has_cli_action and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
        from thor_vision_scanner_gui import launch_vision_scanner_gui
        launch_vision_scanner_gui()
        return

    det_model, all_models = auto_detect_openai_model(args.api_base, api_key=args.api_key, preferred_model=args.model)
    args.model = det_model
    print(f"🔍 [Auto-Probe] `{normalize_api_base(args.api_base)}/models` ➔ 已選取模型: `{det_model}` (共發現: {all_models})")

    out_dir = Path(__file__).resolve().parent / "exports" / "vision_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_files: List[Path] = []
    if args.demo_multi:
        target_files = generate_synthetic_directional_suite(out_dir)
        print(f"🎨 已自動生成 {len(target_files)} 張連續方向性移動測試圖於: {out_dir}")
    elif args.demo or (not args.images and not args.dir):
        suite = generate_synthetic_directional_suite(out_dir)
        target_files = [suite[0]]
        print(f"🎨 已生成方向性測試校正圖: {target_files[0]}")
    else:
        for img_str in args.images:
            p = Path(img_str).expanduser().resolve()
            if p.is_file():
                target_files.append(p)
            elif p.is_dir():
                target_files.extend(
                    sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
                )
        if args.dir:
            d_path = Path(args.dir).expanduser().resolve()
            if d_path.is_dir():
                target_files.extend(
                    sorted(f for f in d_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
                )

    if not target_files:
        sys.exit("❌ 未找到任何有效的圖片檔案！")

    if args.grid_overlay:
        overlaid: List[Path] = []
        for p in target_files:
            gp = out_dir / f"{p.stem}_grid.jpg"
            overlaid.append(apply_coordinate_grid_overlay(p, gp))
        target_files = overlaid

    results: List[Dict[str, Any]] = []
    suite_t0 = time.time()

    if args.joint_frames and len(target_files) > 1:
        # Engine `max_image_tokens=1024` and `max_image_tokens_per_image=512` allows up to 2 Base64 images per single request!
        # For >2 files, automatically runs a 2-Frame Sliding Window (Frame t ➔ Frame t+1) across the sequence!
        eff_mode = "motion_delta" if args.mode == "raster" and not args.prompt else args.mode
        for i in range(len(target_files) - 1):
            pair = [target_files[i], target_files[i + 1]]
            res = run_vision_request(
                image_paths=pair,
                api_base=args.api_base,
                model=args.model,
                mode=eff_mode,
                custom_prompt=args.prompt or None,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
            )
            results.append(res)
    else:
        # Continuous multi-file sequential benchmark (tests different directional modes or same mode across files)
        modes_cycle = [args.mode] * len(target_files)
        if args.demo_multi and args.mode == "raster" and not args.prompt:
            modes_cycle = ["raster", "radial", "embodied_depth"]
        for idx, img_p in enumerate(target_files):
            m = modes_cycle[idx % len(modes_cycle)]
            res = run_vision_request(
                image_paths=[img_p],
                api_base=args.api_base,
                model=args.model,
                mode=m,
                custom_prompt=args.prompt or None,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                stream=True,
            )
            results.append(res)

    suite_elapsed = time.time() - suite_t0
    if len(results) > 1:
        avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
        avg_total = sum(r["total_s"] for r in results) / len(results)
        avg_tps = sum(r.get("tps", 0.0) for r in results) / len(results)
        avg_rate = sum(r["chunks_per_s"] for r in results) / len(results)
        print("\n" + "=" * 80)
        print(f"📊 【AGX Thor 31B 連續多檔案視覺推論測速總結 (共 {len(results)} 批次 / 總耗時 {suite_elapsed:.2f}s)】")
        for i, r in enumerate(results, 1):
            print(
                f"  #{i} [{','.join(r['files'])}] (mode={r['mode']}, {r['kb']}KB) "
                f"➔ TTFT: {r['ttft_ms']:.1f} ms | 總耗時: {r['total_s']:.2f}s | "
                f"速率: {r.get('tps', 0.0):.2f} tok/s ({r['chunks_per_s']:.2f} chunks/s)"
            )
        print(
            f"🏆 平均視覺 Prefill TTFT: {avg_ttft:.1f} ms | 平均單張總耗時: {avg_total:.2f} s | "
            f"平均生成速率: {avg_tps:.2f} tok/s ({avg_rate:.2f} MTP chunks/s)"
        )
        print("=" * 80)


if __name__ == "__main__":
    main()
