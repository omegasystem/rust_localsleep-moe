#!/usr/bin/env python3
"""
thor_vision_scanner_gui.py - PyQt6 Control Center UI for AGX Thor Vision Scanner
Matches the dark-theme layout & automatic `/v1/models` probing of `dyson_universal_openai_bridge.py`.
"""

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from thor_vision_scanner import (
    DIRECTIONAL_PROMPTS,
    SUPPORTED_EXTS,
    apply_coordinate_grid_overlay,
    auto_detect_openai_model,
    generate_synthetic_directional_suite,
    normalize_api_base,
    run_vision_request,
)

BASE_DIR = Path(__file__).resolve().parent
PROFILES_PATH = BASE_DIR / "configs" / "vision_scanner_profiles.json"

DEFAULT_PROFILES: Dict[str, Any] = {
    "active_profile": "thor_port_8000",
    "profiles": {
        "thor_port_8000": {
            "name": "⚡ AGX Thor Primary VLM (:8000 — Bolt-E4B / Gemma-4-31B)",
            "api_base": "http://192.168.50.67:8000/v1",
            "api_key": "",
            "model": "auto",
            "mode": "json_descriptor",
            "max_tokens": 512,
            "temperature": 0.2,
        },
        "thor_port_8004": {
            "name": "🚀 AGX Thor Secondary VLM (:8004 — Parallel Instance)",
            "api_base": "http://192.168.50.67:8004/v1",
            "api_key": "",
            "model": "auto",
            "mode": "json_descriptor",
            "max_tokens": 512,
            "temperature": 0.2,
        },
        "local_5080_port_8001": {
            "name": "🖥️ Local RTX 5080 Server (:8001)",
            "api_base": "http://127.0.0.1:8001/v1",
            "api_key": "",
            "model": "auto",
            "mode": "en_compact",
            "max_tokens": 512,
            "temperature": 0.2,
        },
    },
}


def load_profiles() -> Dict[str, Any]:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROFILES_PATH.exists():
        try:
            data = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "profiles" in data:
                return data
        except Exception:
            pass
    save_profiles(DEFAULT_PROFILES)
    return json.loads(json.dumps(DEFAULT_PROFILES))


def save_profiles(data: Dict[str, Any]) -> None:
    PROFILES_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def launch_vision_scanner_gui() -> None:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QFont, QPixmap, QTextCursor
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QListWidget,
        QMainWindow,
        QPlainTextEdit,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    class ThorVisionScannerGUI(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("👁️ AGX Thor Native Vision Directional Scanner (萬用模型自動探測 & 結構化描述元控制台)")
            self.resize(1080, 820)
            self.setStyleSheet(
                """
                QMainWindow, QWidget { background-color: #0b1120; color: #e2e8f0; font-family: 'Segoe UI', 'Noto Sans CJK TC', sans-serif; }
                QGroupBox { border: 1px solid #1e293b; border-radius: 6px; margin-top: 8px; padding-top: 10px; font-weight: bold; color: #38bdf8; }
                QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {
                    background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 4px 6px;
                }
                QPushButton {
                    background-color: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 5px; padding: 6px 12px; font-weight: bold;
                }
                QPushButton:hover { background-color: #334155; }
                """
            )
            self.profiles_data = load_profiles()
            self.ui_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
            self.image_queue: List[Path] = []
            self.scanning = False

            central = QWidget()
            self.setCentralWidget(central)
            root_v = QVBoxLayout(central)

            # 1. Profile & Auto-Probe Box
            prof_box = QGroupBox("🗂️ 1. 視覺端點 Profile 與 /v1/models 自動探測管理 (configs/vision_scanner_profiles.json)")
            prof_form = QFormLayout(prof_box)

            row_prof = QHBoxLayout()
            self.cmb_profile = QComboBox()
            self.cmb_profile.currentIndexChanged.connect(self._on_profile_selected)
            row_prof.addWidget(self.cmb_profile, stretch=1)

            btn_save = QPushButton("💾 儲存當前 Profile")
            btn_save.setStyleSheet("background-color: #0284c7; color: white;")
            btn_save.clicked.connect(self._save_current_profile)
            row_prof.addWidget(btn_save)

            btn_new = QPushButton("➕ 新增自訂 Profile")
            btn_new.setStyleSheet("background-color: #4f46e5; color: white;")
            btn_new.clicked.connect(self._create_new_profile)
            row_prof.addWidget(btn_new)
            prof_form.addRow("選擇 Profile:", row_prof)

            row_url = QHBoxLayout()
            self.edt_api_base = QLineEdit()
            row_url.addWidget(self.edt_api_base, stretch=2)
            row_url.addWidget(QLabel("API Key:"))
            self.edt_api_key = QLineEdit()
            self.edt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.edt_api_key.setPlaceholderText("選填 (地端免填)")
            row_url.addWidget(self.edt_api_key, stretch=1)
            prof_form.addRow("API Base URL:", row_url)

            row_model = QHBoxLayout()
            self.cmb_model = QComboBox()
            self.cmb_model.setEditable(True)
            self.cmb_model.addItem("auto")
            row_model.addWidget(self.cmb_model, stretch=1)

            btn_probe = QPushButton("🔍 探測 /v1/models 並自動選取")
            btn_probe.setStyleSheet("background-color: #0d9488; color: white;")
            btn_probe.clicked.connect(self._probe_models)
            row_model.addWidget(btn_probe)
            prof_form.addRow("推論 Model ID:", row_model)

            row_mode = QHBoxLayout()
            self.cmb_mode = QComboBox()
            mode_items = [
                ("json_bbox", "🎯 json_bbox — 原生 0-1000 BBox 座標偵測 + 自動畫框存檔 (*_bbox_pred.jpg)"),
                ("json_descriptor", "🧩 json_descriptor — 純 JSON 空間描述元 + BBox (不吐廢話句子)"),
                ("en_compact", "🔤 en_compact — 英文 Key-Value 精簡描述元 (English Telemetry)"),
                ("raster", "📐 raster — 由左至右、由上而下 Z 字象限掃描 (中文)"),
                ("radial", "🎯 radial — 中央凹順時針雷達掃描 (12h➔3h➔6h➔9h)"),
                ("embodied_depth", "🤖 embodied_depth — 具身視覺由近而遠走廊避障掃描"),
                ("motion_delta", "⚡ motion_delta — 雙幀時序移動方向向量追蹤"),
            ]
            for k, label in mode_items:
                self.cmb_mode.addItem(label, userData=k)
            row_mode.addWidget(self.cmb_mode, stretch=2)

            row_mode.addWidget(QLabel("Max Tokens:"))
            self.spn_max_tokens = QSpinBox()
            self.spn_max_tokens.setRange(64, 4096)
            self.spn_max_tokens.setValue(512)
            row_mode.addWidget(self.spn_max_tokens)

            row_mode.addWidget(QLabel("Temp:"))
            self.spn_temp = QDoubleSpinBox()
            self.spn_temp.setRange(0.0, 1.5)
            self.spn_temp.setSingleStep(0.05)
            self.spn_temp.setValue(0.2)
            row_mode.addWidget(self.spn_temp)
            prof_form.addRow("掃描模式與參數:", row_mode)

            row_opts = QHBoxLayout()
            self.edt_custom_prompt = QLineEdit()
            self.edt_custom_prompt.setPlaceholderText("選填：若輸入自訂 Prompt 將覆蓋上方預設模式...")
            row_opts.addWidget(self.edt_custom_prompt, stretch=1)
            self.chk_grid = QCheckBox("疊加 3x3 方位網格 (--grid-overlay)")
            row_opts.addWidget(self.chk_grid)
            self.chk_joint = QCheckBox("雙幀滑動視窗運動追蹤 (--joint-frames)")
            row_opts.addWidget(self.chk_joint)
            prof_form.addRow("自訂與進階選項:", row_opts)
            root_v.addWidget(prof_box)

            # 2. Image Queue & Preview Box
            img_box = QGroupBox("🖼️ 2. 待測影像佇列與即時畫面預覽")
            img_h = QHBoxLayout(img_box)
            left_v = QVBoxLayout()
            btn_img_row = QHBoxLayout()

            btn_add_files = QPushButton("📂 加入圖片檔...")
            btn_add_files.clicked.connect(self._add_files)
            btn_img_row.addWidget(btn_add_files)

            btn_add_dir = QPushButton("📁 載入資料夾...")
            btn_add_dir.clicked.connect(self._add_dir)
            btn_img_row.addWidget(btn_add_dir)

            btn_demo = QPushButton("🎨 自動生成 3 張方向性移動測試圖")
            btn_demo.setStyleSheet("background-color: #7c3aed; color: white;")
            btn_demo.clicked.connect(self._load_demo_suite)
            btn_img_row.addWidget(btn_demo)

            btn_clear_q = QPushButton("🗑️ 清空佇列")
            btn_clear_q.clicked.connect(self._clear_queue)
            btn_img_row.addWidget(btn_clear_q)
            left_v.addLayout(btn_img_row)

            self.lst_images = QListWidget()
            self.lst_images.currentRowChanged.connect(self._on_image_selected)
            left_v.addWidget(self.lst_images)
            img_h.addLayout(left_v, stretch=2)

            self.lbl_preview = QLabel("🖼️ 影像預覽區\n(點擊『🎨 自動生成 3 張測試圖』或加入圖片)")
            self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lbl_preview.setFixedSize(320, 220)
            self.lbl_preview.setStyleSheet("background-color: #020617; border: 1px solid #334155; border-radius: 6px;")
            img_h.addWidget(self.lbl_preview, stretch=1)
            root_v.addWidget(img_box)

            # 3. Action Buttons & Telemetry Status
            ctrl_box = QGroupBox("🚀 3. 批次方向性視覺推論與即時測速狀態")
            ctrl_v = QVBoxLayout(ctrl_box)
            run_row = QHBoxLayout()

            self.btn_run = QPushButton("🚀 開始執行多檔視覺掃描與 TTFT 測速")
            self.btn_run.setStyleSheet("background-color: #059669; color: white; font-size: 13px; padding: 8px 16px;")
            self.btn_run.clicked.connect(self._start_scan)
            run_row.addWidget(self.btn_run, stretch=2)

            self.btn_stop = QPushButton("⏹️ 停止掃描")
            self.btn_stop.setStyleSheet("background-color: #be123c; color: white; font-size: 13px; padding: 8px 16px;")
            self.btn_stop.setEnabled(False)
            self.btn_stop.clicked.connect(self._stop_scan)
            run_row.addWidget(self.btn_stop, stretch=1)

            btn_clear_log = QPushButton("🧹 清空日誌")
            btn_clear_log.clicked.connect(lambda: self.txt_log.clear())
            run_row.addWidget(btn_clear_log)
            ctrl_v.addLayout(run_row)

            self.lbl_status = QLabel("⚪ 狀態: 待命中 — 可點擊『🔍 探測 /v1/models』確認目前 Thor 掛載的是 31B 或 Bolt-E4B")
            self.lbl_status.setStyleSheet("color: #facc15; font-weight: bold;")
            ctrl_v.addWidget(self.lbl_status)
            root_v.addWidget(ctrl_box)

            # 4. Streaming Output Log
            log_box = QGroupBox("📡 4. 即時 VLM 視覺推論串流與 TTFT / Tok/s 統計日誌")
            log_v = QVBoxLayout(log_box)
            self.txt_log = QPlainTextEdit()
            self.txt_log.setReadOnly(True)
            self.txt_log.setFont(QFont("Monospace", 10))
            log_v.addWidget(self.txt_log)
            root_v.addWidget(log_box, stretch=1)

            self._populate_profiles()
            self._load_demo_suite()

            self.timer = QTimer(self)
            self.timer.timeout.connect(self._poll_queue)
            self.timer.start(40)

        def _populate_profiles(self) -> None:
            self.cmb_profile.blockSignals(True)
            self.cmb_profile.clear()
            profs = self.profiles_data.get("profiles", {})
            active = self.profiles_data.get("active_profile", "thor_port_8000")
            sel_idx = 0
            for idx, (k, v) in enumerate(profs.items()):
                self.cmb_profile.addItem(f"{v.get('name', k)} [{k}]", userData=k)
                if k == active:
                    sel_idx = idx
            self.cmb_profile.setCurrentIndex(sel_idx)
            self.cmb_profile.blockSignals(False)
            self._load_profile_fields(active)

        def _on_profile_selected(self) -> None:
            pkey = self.cmb_profile.currentData()
            if pkey:
                self.profiles_data["active_profile"] = pkey
                self._load_profile_fields(str(pkey))

        def _load_profile_fields(self, pkey: str) -> None:
            p = self.profiles_data.get("profiles", {}).get(pkey, {})
            self.edt_api_base.setText(str(p.get("api_base", "http://192.168.50.67:8000/v1")))
            self.edt_api_key.setText(str(p.get("api_key", "")))
            m_val = str(p.get("model", "auto"))
            if self.cmb_model.findText(m_val) < 0:
                self.cmb_model.addItem(m_val)
            self.cmb_model.setCurrentText(m_val)
            mode_val = str(p.get("mode", "json_descriptor"))
            for i in range(self.cmb_mode.count()):
                if self.cmb_mode.itemData(i) == mode_val:
                    self.cmb_mode.setCurrentIndex(i)
                    break
            self.spn_max_tokens.setValue(int(p.get("max_tokens", 512)))
            self.spn_temp.setValue(float(p.get("temperature", 0.2)))

        def _save_current_profile(self) -> None:
            pkey = str(self.cmb_profile.currentData() or "thor_port_8000")
            profs = self.profiles_data.setdefault("profiles", {})
            existing = profs.get(pkey, {})
            profs[pkey] = {
                "name": existing.get("name", pkey),
                "api_base": self.edt_api_base.text().strip(),
                "api_key": self.edt_api_key.text().strip(),
                "model": self.cmb_model.currentText().strip() or "auto",
                "mode": str(self.cmb_mode.currentData() or "json_descriptor"),
                "max_tokens": int(self.spn_max_tokens.value()),
                "temperature": float(self.spn_temp.value()),
            }
            save_profiles(self.profiles_data)
            self._append_log(f"💾 已儲存 Profile `{pkey}` 設定！\n")

        def _create_new_profile(self) -> None:
            pid, ok = QInputDialog.getText(self, "新增 Profile", "輸入新 Profile 代號:")
            if not ok or not pid.strip():
                return
            pid = pid.strip()
            self.profiles_data.setdefault("profiles", {})[pid] = {
                "name": f"🔧 Custom ({pid})",
                "api_base": self.edt_api_base.text().strip(),
                "api_key": self.edt_api_key.text().strip(),
                "model": self.cmb_model.currentText().strip() or "auto",
                "mode": str(self.cmb_mode.currentData() or "json_descriptor"),
                "max_tokens": int(self.spn_max_tokens.value()),
                "temperature": float(self.spn_temp.value()),
            }
            self.profiles_data["active_profile"] = pid
            save_profiles(self.profiles_data)
            self._populate_profiles()

        def _probe_models(self) -> None:
            api_b = self.edt_api_base.text().strip()
            api_k = self.edt_api_key.text().strip()
            det, models = auto_detect_openai_model(api_b, api_key=api_k, preferred_model="auto")
            self.cmb_model.clear()
            self.cmb_model.addItem("auto")
            for m in models:
                self.cmb_model.addItem(m)
            if det:
                if self.cmb_model.findText(det) < 0:
                    self.cmb_model.addItem(det)
                self.cmb_model.setCurrentText(det)
            self.lbl_status.setText(f"🟢 探測成功: 端點目前掛載模型為 `{det}` (共 {len(models)} 個模型)")
            self.lbl_status.setStyleSheet("color: #34d399; font-weight: bold;")
            self._append_log(f"🔍 [Probe `{normalize_api_base(api_b)}/models`] 發現模型: {models} ➔ 自動選取 `{det}`\n")

        def _add_files(self) -> None:
            files, _ = QFileDialog.getOpenFileNames(
                self, "選擇影像檔案", str(BASE_DIR), "Images (*.jpg *.jpeg *.png *.webp *.bmp)"
            )
            for f in files:
                p = Path(f)
                if p.is_file():
                    self.image_queue.append(p)
                    self.lst_images.addItem(str(p))
            if self.lst_images.count() > 0:
                self.lst_images.setCurrentRow(self.lst_images.count() - 1)

        def _add_dir(self) -> None:
            d = QFileDialog.getExistingDirectory(self, "選擇影像資料夾", str(BASE_DIR))
            if not d:
                return
            for p in sorted(Path(d).iterdir()):
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                    self.image_queue.append(p)
                    self.lst_images.addItem(str(p))
            if self.lst_images.count() > 0:
                self.lst_images.setCurrentRow(0)

        def _load_demo_suite(self) -> None:
            out_dir = BASE_DIR / "exports" / "vision_tests"
            paths = generate_synthetic_directional_suite(out_dir)
            self.image_queue = list(paths)
            self.lst_images.clear()
            for p in paths:
                self.lst_images.addItem(str(p))
            if paths:
                self.lst_images.setCurrentRow(0)
            self._append_log(f"🎨 已載入 3 張方向性移動測試圖 ({out_dir})\n")

        def _clear_queue(self) -> None:
            self.image_queue.clear()
            self.lst_images.clear()
            self.lbl_preview.setText("🖼️ 佇列已清空")

        def _on_image_selected(self, row: int) -> None:
            if 0 <= row < len(self.image_queue):
                pix = QPixmap(str(self.image_queue[row]))
                if not pix.isNull():
                    self.lbl_preview.setPixmap(
                        pix.scaled(310, 210, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    )

        def _start_scan(self) -> None:
            if self.scanning or not self.image_queue:
                return
            self._save_current_profile()
            self.scanning = True
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(True)
            threading.Thread(target=self._scan_worker, daemon=True).start()

        def _stop_scan(self) -> None:
            self.scanning = False

        def _scan_worker(self) -> None:
            api_b = self.edt_api_base.text().strip()
            api_k = self.edt_api_key.text().strip()
            pref_m = self.cmb_model.currentText().strip() or "auto"
            mode = str(self.cmb_mode.currentData() or "json_descriptor")
            custom_p = self.edt_custom_prompt.text().strip() or None
            max_toks = int(self.spn_max_tokens.value())
            temp = float(self.spn_temp.value())
            use_grid = self.chk_grid.isChecked()
            use_joint = self.chk_joint.isChecked()

            det_m, all_m = auto_detect_openai_model(api_b, api_key=api_k, preferred_model=pref_m)
            self.ui_queue.put({"type": "status", "text": f"🟢 正在使用模型 `{det_m}` 進行視覺掃描..."})
            self.ui_queue.put({"type": "log", "text": f"\n🔍 自動探測模型: `{det_m}` (伺服器清單: {all_m})\n"})

            out_dir = BASE_DIR / "exports" / "vision_tests"
            files = list(self.image_queue)
            if use_grid:
                files = [apply_coordinate_grid_overlay(p, out_dir / f"{p.stem}_grid.jpg") for p in files]

            results = []
            t_suite = time.time()
            try:
                batches = (
                    [[files[i], files[i + 1]] for i in range(len(files) - 1)]
                    if (use_joint and len(files) > 1)
                    else [[p] for p in files]
                )
                for idx, batch in enumerate(batches, 1):
                    if not self.scanning:
                        break
                    b_names = ", ".join(p.name for p in batch)
                    self.ui_queue.put({
                        "type": "log",
                        "text": f"\n📡 [Batch #{idx}/{len(batches)} | {det_m}] 影像: [{b_names}] | 模式: {mode}\n" + "-" * 76 + "\n",
                    })
                    res = run_vision_request(
                        image_paths=batch,
                        api_base=api_b,
                        model=det_m,
                        mode=mode,
                        custom_prompt=custom_p,
                        max_tokens=max_toks,
                        temperature=temp,
                        stream=True,
                        api_key=api_k,
                        on_token=lambda piece: self.ui_queue.put({"type": "log", "text": piece}),
                    )
                    results.append(res)
                    bbox_info = ""
                    if res.get("bbox_image_path"):
                        bbox_info = f"🎯 已繪製 {len(res.get('bboxes', []))} 個 BBox ➔ `{res['bbox_image_path']}`\n"
                        self.ui_queue.put({"type": "preview", "path": res["bbox_image_path"]})
                    self.ui_queue.put({
                        "type": "log",
                        "text": (
                            f"\n⚡ TTFT: {res['ttft_ms']:.1f} ms (Prompt: {res['prompt_tokens']} toks) | "
                            f"總耗時: {res['total_s']:.2f} s | 速率: {res['tps']:.2f} tok/s ({res['chunks_per_s']:.2f} chunks/s)\n"
                            + bbox_info
                        ),
                    })
                if len(results) > 1:
                    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
                    avg_tot = sum(r["total_s"] for r in results) / len(results)
                    avg_tps = sum(r["tps"] for r in results) / len(results)
                    self.ui_queue.put({
                        "type": "log",
                        "text": (
                            "\n" + "=" * 76 + "\n"
                            f"📊 【{det_m} 連續多檔案視覺推論測速總結 (共 {len(results)} 批次 / 總耗時 {time.time()-t_suite:.2f}s)】\n"
                            f"🏆 平均 TTFT: {avg_ttft:.1f} ms | 平均單批耗時: {avg_tot:.2f} s | 平均速率: {avg_tps:.2f} tok/s\n"
                            + "=" * 76 + "\n"
                        ),
                    })
            except Exception as e:
                self.ui_queue.put({"type": "log", "text": f"\n❌ 視覺推論發生錯誤: {e}\n"})
            finally:
                self.ui_queue.put({"type": "done", "model": det_m})

        def _append_log(self, text: str) -> None:
            c = self.txt_log.textCursor()
            c.movePosition(QTextCursor.MoveOperation.End)
            c.insertText(text)
            self.txt_log.setTextCursor(c)
            self.txt_log.ensureCursorVisible()

        def _poll_queue(self) -> None:
            while not self.ui_queue.empty():
                evt = self.ui_queue.get_nowait()
                t = evt.get("type")
                if t == "log":
                    self._append_log(str(evt.get("text", "")))
                elif t == "status":
                    self.lbl_status.setText(str(evt.get("text", "")))
                elif t == "preview":
                    pix = QPixmap(str(evt.get("path", "")))
                    if not pix.isNull():
                        self.lbl_preview.setPixmap(
                            pix.scaled(310, 210, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        )
                elif t == "done":
                    self.scanning = False
                    self.btn_run.setEnabled(True)
                    self.btn_stop.setEnabled(False)
                    self.lbl_status.setText(f"✅ 掃描完成 (模型: {evt.get('model')})")

    app = QApplication(sys.argv)
    win = ThorVisionScannerGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    launch_vision_scanner_gui()
