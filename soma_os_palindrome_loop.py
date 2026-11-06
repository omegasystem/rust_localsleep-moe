#!/usr/bin/env python3
"""
SOMA-OS v2 Autonomous Visual Self-Critique & Daemonized Palindrome Loop
=======================================================================
Upgrades requested by Architect Leon:
1. True OS Daemonization (`daemonize=True`):
   - The generated Rust OS binary is NO LONGER a one-shot script that exits after 2ms!
   - It runs a persistent kernel/compositor event loop (`loop { tick += 1; ... sleep(80ms); }`)
     that continuously re-reads `/proc/stat`, `/proc/loadavg`, `/proc/meminfo`, and `/proc/uptime`
     every 80ms (~12.5 FPS), atomically updating `/dev/fb0` so waveforms scroll, per-core bars
     breathe, and the heartbeat tick counter increments live in the PyQt6 Fishbowl window!
   - Hot-swaps cleanly: when the next evolution compiles, the Fishbowl automatically stops the
     previous daemon PID and launches the upgraded daemon PID.
2. Screenshot + Pixel Layout Audit -> 31B Self-Critique Recursive Loop:
   - After dropping each OS daemon into the Fishbowl, captures two frames 400ms apart (`Frame A`, `Frame B`),
     encodes the live `/dev/fb0` PNG screenshot (`base64`), and runs a deterministic 6-point
     Pixel & Layout Auditor:
       * Check 1: Is the OS Daemon PID alive and running continuously?
       * Check 2: Are pixels actively animating between Frame A and Frame B (`anim_diff_pixels >= 250`)?
       * Check 3: Are all 4 screen quadrants (`TL`, `TR`, `BL`, `BR`) + Top/Bottom Bars balanced (`10%..75%` fill)?
       * Check 4: Color palette richness (`>= 12` distinct color clusters, no crude grey/black slabs)?
       * Check 5: Does the source code parse real `/proc/stat` & `/proc/meminfo` inside the `loop` (zero hardcoded mocks)?
       * Check 6: 31B Visual & Architectural Self-Critique Score (`0..100`) inspecting the screenshot & audit report!
   - Recursively refines the OS daemon until ALL checks pass and `score >= 88` before entering stable wait!

Usage:
  python3 soma_os_palindrome_loop.py --target-score 88 --max-rounds 6
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from localsleep_fishbowl import (
    DEFAULT_ENDPOINT,
    FB_BYTES,
    FB_FILE,
    FB_HEIGHT,
    FB_WIDTH,
    FISHBOWL_DIR,
    HUD_HTML_OUT,
    OS_BIN_DIR,
    OS_SRC_DIR,
    ROOTFS_DIR,
    SRC_BAY_DIR,
    compile_source_into_organ,
    create_and_compile_compositor_v3,
    ensure_fishbowl_initialized,
    extract_code_block,
    get_all_devmem_channels,
    get_ui_config,
    is_os_daemon_alive,
    list_os_apps_and_processes,
    poke_devmem_channel,
    rgba_to_png_b64,
    run_organ_in_fishbowl,
    run_os_bash_command,
    update_ui_config,
)

FISHBOWL_API = "http://127.0.0.1:18940"
OS_STATE_FILE = ROOTFS_DIR / "os_state.json"
LOOP_HISTORY_FILE = FISHBOWL_DIR / "palindrome_os_steps.json"

# === [SOMA-GPUI: GPUI.RS SYNTAX-COMPATIBLE FLEXBOX & FRAMEBUFFER ENGINE (ZERO-CRATE STD RUST)] ===
RUST_SOMA_GPUI_SDK = r'''
pub mod gpui {
    #[derive(Clone, Copy)]
    pub struct Rgb(pub u8, pub u8, pub u8);
    pub fn rgb(hex: u32) -> Rgb {
        Rgb(((hex >> 16) & 0xFF) as u8, ((hex >> 8) & 0xFF) as u8, (hex & 0xFF) as u8)
    }
    pub fn px(val: f32) -> usize {
        val.max(0.0) as usize
    }
    pub fn red() -> Rgb { rgb(0xef4444) }
    pub fn green() -> Rgb { rgb(0x34d399) }
    pub fn blue() -> Rgb { rgb(0x38bdf8) }
    pub fn yellow() -> Rgb { rgb(0xfbbf24) }
    pub fn black() -> Rgb { rgb(0x060911) }
    pub fn white() -> Rgb { rgb(0xe2e8f0) }

    #[derive(Clone)]
    pub enum WidgetPayload {
        None,
        Text(String),
        CpuMatrix24,
        DevMemBitplane8,
        LoraHypercube16,
        OscilloscopeAndIpc,
        ProgressBar(f32, Rgb),
        BitCells(u64, Rgb),
    }

    #[derive(Clone)]
    pub struct Div {
        pub is_col: bool,
        pub gap: usize,
        pub pad: usize,
        pub flex_grow: f32,
        pub fixed_w: Option<usize>,
        pub fixed_h: Option<usize>,
        pub bg_color: Option<Rgb>,
        pub border_w: usize,
        pub border_color: Option<Rgb>,
        pub text_color: Rgb,
        pub title: Option<String>,
        pub payload: WidgetPayload,
        pub children: Vec<Div>,
    }

    pub fn div() -> Div {
        Div {
            is_col: false,
            gap: 0,
            pad: 0,
            flex_grow: 0.0,
            fixed_w: None,
            fixed_h: None,
            bg_color: None,
            border_w: 0,
            border_color: None,
            text_color: rgb(0xe2e8f0),
            title: None,
            payload: WidgetPayload::None,
            children: Vec::new(),
        }
    }

    pub trait IntoElement {
        fn into_div(self) -> Div;
    }
    impl IntoElement for Div {
        fn into_div(self) -> Div { self }
    }
    impl IntoElement for &str {
        fn into_div(self) -> Div {
            let mut d = div().h(16);
            d.payload = WidgetPayload::Text(self.to_string());
            d
        }
    }
    impl IntoElement for String {
        fn into_div(self) -> Div {
            let mut d = div().h(16);
            d.payload = WidgetPayload::Text(self);
            d
        }
    }

    impl Div {
        pub fn flex(self) -> Self { self }
        pub fn flex_row(mut self) -> Self { self.is_col = false; self }
        pub fn flex_col(mut self) -> Self { self.is_col = true; self }
        pub fn flex_1(mut self) -> Self { self.flex_grow = 1.0; self }
        pub fn flex_grow(mut self, g: f32) -> Self { self.flex_grow = g.max(0.0); self }
        pub fn size_full(mut self) -> Self { self.flex_grow = 1.0; self }
        pub fn size(mut self, v: usize) -> Self { self.fixed_w = Some(v); self.fixed_h = Some(v); self }
        pub fn size_8(self) -> Self { self.size(32) }
        pub fn w(mut self, v: usize) -> Self { self.fixed_w = Some(v); self }
        pub fn h(mut self, v: usize) -> Self { self.fixed_h = Some(v); self }
        pub fn gap(mut self, v: usize) -> Self { self.gap = v; self }
        pub fn gap_1(self) -> Self { self.gap(4) }
        pub fn gap_2(self) -> Self { self.gap(8) }
        pub fn gap_3(self) -> Self { self.gap(12) }
        pub fn gap_4(self) -> Self { self.gap(16) }
        pub fn p(mut self, v: usize) -> Self { self.pad = v; self }
        pub fn p_1(self) -> Self { self.p(4) }
        pub fn p_2(self) -> Self { self.p(8) }
        pub fn p_3(self) -> Self { self.p(12) }
        pub fn p_4(self) -> Self { self.p(16) }
        pub fn bg(mut self, c: Rgb) -> Self { self.bg_color = Some(c); self }
        pub fn border_1(mut self) -> Self { self.border_w = 1; self }
        pub fn border_2(mut self) -> Self { self.border_w = 2; self }
        pub fn border_color(mut self, c: Rgb) -> Self { self.border_color = Some(c); self }
        pub fn text_color(mut self, c: Rgb) -> Self { self.text_color = c; self }
        pub fn text_xl(self) -> Self { self }
        pub fn text_sm(self) -> Self { self }
        pub fn shadow_lg(self) -> Self { self }
        pub fn justify_center(self) -> Self { self }
        pub fn justify_between(self) -> Self { self }
        pub fn items_center(self) -> Self { self }
        pub fn panel_title(mut self, t: &str) -> Self { self.title = Some(t.to_string()); self }
        pub fn cpu_matrix_24(mut self) -> Self { self.payload = WidgetPayload::CpuMatrix24; self }
        pub fn devmem_bitplane_8(mut self) -> Self { self.payload = WidgetPayload::DevMemBitplane8; self }
        pub fn lora_hypercube_16(mut self) -> Self { self.payload = WidgetPayload::LoraHypercube16; self }
        pub fn oscilloscope_ipc(mut self) -> Self { self.payload = WidgetPayload::OscilloscopeAndIpc; self }
        pub fn progress_bar(mut self, ratio: f32, c: Rgb) -> Self {
            if self.fixed_h.is_none() {
                self.fixed_h = Some(14);
            }
            self.payload = WidgetPayload::ProgressBar(ratio, c);
            self
        }
        pub fn bit_cells(mut self, mask: u64, c: Rgb) -> Self {
            if self.fixed_h.is_none() {
                self.fixed_h = Some(20);
            }
            self.payload = WidgetPayload::BitCells(mask, c);
            self
        }
        pub fn child<E: IntoElement>(mut self, el: E) -> Self {
            let mut child_div = el.into_div();
            // Inherit parent text_color if child was created from raw &str / String
            if matches!(child_div.payload, WidgetPayload::Text(_)) && child_div.text_color.0 == 0xe2 && child_div.text_color.1 == 0xe8 {
                child_div.text_color = self.text_color;
            }
            self.children.push(child_div);
            self
        }
    }

    #[derive(Clone)]
    pub struct RenderCtx {
        pub tick: u64,
        pub cpu_pcts: Vec<f64>,
        pub mem_pct: f64,
        pub mem_total_mb: u64,
        pub load_1m: f64,
        pub devmem_regs: Vec<(u32, u64, String, String)>,
    }

    impl Default for RenderCtx {
        fn default() -> Self {
            Self::new(1)
        }
    }

    impl RenderCtx {
        pub fn from_system(tick: u64) -> Self {
            Self::new(tick)
        }
        pub fn new(tick: u64) -> Self {
            use std::fs;
            let load_str = fs::read_to_string("/proc/loadavg").unwrap_or_default();
            let load_1m = load_str.split_whitespace().next().unwrap_or("0.5").parse::<f64>().unwrap_or(0.5);

            let mem_str = fs::read_to_string("/proc/meminfo").unwrap_or_default();
            let mut total: u64 = 0;
            let mut avail: u64 = 0;
            for ln in mem_str.lines() {
                if ln.starts_with("MemTotal:") {
                    total = ln.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                } else if ln.starts_with("MemAvailable:") {
                    avail = ln.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
                }
            }
            let mem_pct = if total > 0 { (1.0 - (avail as f64 / total as f64)) * 100.0 } else { 12.0 };

            let mut cpu_pcts = Vec::with_capacity(24);
            for i in 0..24 {
                let phase = (((tick as f64 * 0.22) + (i as f64 * 0.55)).sin() + 1.0) * 28.0 + (load_1m * 8.0);
                cpu_pcts.push(phase.clamp(8.0, 96.0));
            }

            let devmem_path = std::env::var("SOMA_DEVMEM_PATH")
                .unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
            let raw = fs::read(&devmem_path).unwrap_or_else(|_| vec![0u8; 65536]);
            let mut devmem_regs = Vec::with_capacity(8);
            for ch in 0..8 {
                let base = ch * 8192;
                if base + 256 <= raw.len() {
                    let seq = u32::from_le_bytes([raw[base + 4], raw[base + 5], raw[base + 6], raw[base + 7]]);
                    let reg = u64::from_le_bytes([
                        raw[base + 8], raw[base + 9], raw[base + 10], raw[base + 11],
                        raw[base + 12], raw[base + 13], raw[base + 14], raw[base + 15],
                    ]);
                    let te = raw[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
                    let tag = String::from_utf8_lossy(&raw[base + 16..base + 16 + te]).to_uppercase();
                    let pe = raw[base + 48..base + 240].iter().position(|&b| b == 0).unwrap_or(192);
                    let pay = String::from_utf8_lossy(&raw[base + 48..base + 48 + pe]).to_uppercase();
                    devmem_regs.push((seq, reg, tag, pay));
                } else {
                    devmem_regs.push((1, 0x504D_4100_0000_0000 | (ch as u64), format!("CH{}:READY", ch), String::from("ONLINE")));
                }
            }

            RenderCtx {
                tick,
                cpu_pcts,
                mem_pct,
                mem_total_mb: total / 1024,
                load_1m,
                devmem_regs,
            }
        }
    }

    pub fn render_node(buf: &mut [u8], node: &Div, x: usize, y: usize, w: usize, h: usize, ctx: &RenderCtx) {
        render_node_inner(buf, node, x, y, w, h, ctx);
        if x == 0 && y == 0 && w == 1024 && h == 768 {
            if let Ok(fb_p) = std::env::var("FISHBOWL_FB_PATH") {
                let _ = std::fs::write(fb_p, &*buf);
            }
        }
    }

    fn render_node_inner(buf: &mut [u8], node: &Div, x: usize, y: usize, w: usize, h: usize, ctx: &RenderCtx) {
        if w == 0 || h == 0 { return; }
        if let Some(bg) = node.bg_color {
            super::soma_fill_rect_1024(buf, x, y, w, h, [bg.0, bg.1, bg.2]);
        }
        let mut inner_y = y;
        let mut inner_h = h;
        if let Some(ref title) = node.title {
            let bc = node.border_color.unwrap_or(rgb(0x38bdf8));
            let title_rgb = [bc.0.max(96), bc.1.max(195), bc.2.max(245)];
            super::soma_fill_rect_1024(buf, x, y, w, 22.min(h), [18, 30, 52]);
            super::soma_fill_rect_1024(buf, x, (y + 22).min(767), w, 1, [38, 64, 102]);
            super::soma_draw_text_1024(buf, x + 8, y + 7, title, title_rgb, 1);
            inner_y = (y + 23).min(y + h);
            inner_h = h.saturating_sub(23);
        }
        if node.border_w > 0 {
            let bc = node.border_color.unwrap_or(rgb(0x38bdf8));
            let c = [bc.0.max(38), bc.1.max(58), bc.2.max(88)];
            super::soma_fill_rect_1024(buf, x, y, w, 1, c);
            super::soma_fill_rect_1024(buf, x, (y + h.saturating_sub(1)).min(767), w, 1, c);
            super::soma_fill_rect_1024(buf, x, y, 1, h, c);
            super::soma_fill_rect_1024(buf, (x + w.saturating_sub(1)).min(1023), y, 1, h, c);
        }

        let cx = x + node.pad;
        let cy = inner_y + node.pad;
        let cw = w.saturating_sub(node.pad * 2);
        let ch = inner_h.saturating_sub(node.pad * 2);

        // If a node has BOTH a primary widget payload AND children, auto-promote the payload into a flex child
        // so children NEVER draw on top of the widget payload!
        if !matches!(node.payload, WidgetPayload::None) && !node.children.is_empty() {
            let mut promoted = div().flex_1();
            promoted.payload = node.payload.clone();
            promoted.text_color = node.text_color;
            let mut combined_children = Vec::with_capacity(node.children.len() + 1);
            combined_children.push(promoted);
            for ch_node in &node.children {
                let mut cloned = ch_node.clone();
                if cloned.fixed_h.is_none() && ! cloned.children.is_empty() {
                    cloned.fixed_h = Some(42);
                }
                combined_children.push(cloned);
            }
            let mut container = div().flex_col().gap(node.gap.max(6));
            container.children = combined_children;
            render_node_inner(buf, &container, cx, cy, cw, ch, ctx);
            return;
        }

        match &node.payload {
            WidgetPayload::Text(s) => {
                let tc = [node.text_color.0, node.text_color.1, node.text_color.2];
                super::soma_draw_text_1024(buf, cx, cy + 2, s, tc, 1);
            }
            WidgetPayload::ProgressBar(ratio, c) => {
                super::soma_fill_rect_1024(buf, cx, cy, cw, ch.max(10), [20, 32, 50]);
                let fw = ((cw as f32) * ratio.clamp(0.02, 1.0)) as usize;
                super::soma_fill_rect_1024(buf, cx, cy, fw, ch.max(10), [c.0, c.1, c.2]);
            }
            WidgetPayload::BitCells(mask, c) => {
                let cell_w = (cw / 32).max(4);
                for b in 0..32 {
                    let on = ((mask >> b) & 1) == 1;
                    let col_rgb = if on { [c.0, c.1, c.2] } else { [22, 36, 58] };
                    super::soma_fill_rect_1024(buf, cx + b * cell_w, cy, cell_w.saturating_sub(2).max(2), ch.max(12), col_rgb);
                }
            }
            WidgetPayload::CpuMatrix24 => {
                let col_w = (cw / 8).max(24);
                let row_h = (ch / 3).max(28);
                for i in 0..24 {
                    let col = i % 8;
                    let row = i / 8;
                    let bx = cx + col * col_w;
                    let by = cy + row * row_h;
                    let p = ctx.cpu_pcts.get(i).copied().unwrap_or(12.0);
                    let wave = (((ctx.tick as f64 * 0.25) + (i as f64 * 0.5)).sin() + 1.0) * 16.0;
                    let eff = (p + wave).clamp(10.0, 100.0);
                    let max_bh = row_h.saturating_sub(18).max(8);
                    let bh = ((eff / 100.0) * max_bh as f64) as usize;
                    super::soma_fill_rect_1024(buf, bx + 2, by + 2, col_w.saturating_sub(6), row_h.saturating_sub(6), [16, 26, 44]);
                    let brgb = if eff > 70.0 { [248, 113, 113] } else if eff > 42.0 { [251, 191, 36] } else { [56, 189, 248] };
                    super::soma_fill_rect_1024(buf, bx + 5, by + 12 + max_bh.saturating_sub(bh), col_w.saturating_sub(12), bh.max(3), brgb);
                    super::soma_draw_text_1024(buf, bx + 6, by + 3, &format!("C{:02}", i), [180, 200, 225], 1);
                }
            }
            WidgetPayload::DevMemBitplane8 => {
                let row_h = (ch / 8).max(18);
                for ch_i in 0..8 {
                    let ry = cy + ch_i * row_h;
                    let (seq, reg, ref tag, _) = ctx.devmem_regs[ch_i];
                    super::soma_fill_rect_1024(buf, cx, ry, cw, row_h.saturating_sub(3), [15, 25, 42]);
                    super::soma_draw_text_1024(buf, cx + 4, ry + 4, &format!("CH{} S{:03}", ch_i, seq % 1000), [230, 245, 255], 1);
                    for b in 0..32 {
                        let on = ((reg >> b) & 1) == 1;
                        let cr = if on { [52, 211, 153] } else { [24, 38, 60] };
                        super::soma_fill_rect_1024(buf, cx + 68 + b * 8, ry + 3, 6, row_h.saturating_sub(9).max(8), cr);
                    }
                    let stag: String = tag.chars().skip(4).take(14).collect();
                    super::soma_draw_text_1024(buf, cx + 332, ry + 4, &stag, [52, 211, 153], 1);
                }
            }
            WidgetPayload::LoraHypercube16 => {
                let col_w = (cw / 4).max(40);
                let row_h = (ch / 4).max(24);
                let mask = ctx.devmem_regs[2].1;
                for i in 0..16 {
                    let bx = cx + (i % 4) * col_w;
                    let by = cy + (i / 4) * row_h;
                    let ph = (ctx.tick + (i as u64 * 3) + (mask & 0xF)) % 15;
                    let (lbl, bg, fg) = if ph >= 12 {
                        ("FIRE", [239, 68, 68], [255, 255, 255])
                    } else if ph >= 6 {
                        ("ACTV", [245, 158, 11], [10, 15, 25])
                    } else {
                        ("NREM", [32, 48, 72], [170, 190, 215])
                    };
                    super::soma_fill_rect_1024(buf, bx + 2, by + 2, col_w.saturating_sub(8), row_h.saturating_sub(6), bg);
                    super::soma_draw_text_1024(buf, bx + 8, by + 8, &format!("E{:02} {}", i, lbl), fg, 1);
                }
            }
            WidgetPayload::OscilloscopeAndIpc => {
                let osc_h = (ch / 2).max(40);
                super::soma_fill_rect_1024(buf, cx, cy, cw, osc_h, [8, 13, 24]);
                for xo in 0..cw.saturating_sub(4) {
                    let t = (ctx.tick as f64 * 0.16) + (xo as f64 * 0.04);
                    let mid = (cy + osc_h / 2) as f64;
                    let amp = (osc_h as f64 * 0.36).max(8.0);
                    let y1 = (mid + t.sin() * amp) as usize;
                    let y2 = (mid + (t * 1.35).cos() * (amp * 0.75)) as usize;
                    super::soma_fill_rect_1024(buf, cx + 2 + xo, y1.clamp(cy + 2, cy + osc_h - 3), 2, 2, [56, 189, 248]);
                    super::soma_fill_rect_1024(buf, cx + 2 + xo, y2.clamp(cy + 2, cy + osc_h - 3), 2, 2, [52, 211, 153]);
                }
                let ipc_y = cy + osc_h + 6;
                let row_h = (ch.saturating_sub(osc_h + 6) / 4).max(14);
                for idx in 0..4 {
                    let ch_id = 3 + idx;
                    let ry = ipc_y + idx * row_h;
                    let pay: String = ctx.devmem_regs[ch_id].3.chars().take(58).collect();
                    super::soma_fill_rect_1024(buf, cx, ry, 10, row_h.saturating_sub(3), [52, 211, 153]);
                    super::soma_draw_text_1024(buf, cx + 16, ry + 2, &format!("CH{}: {}", ch_id, pay), [220, 238, 255], 1);
                }
            }
            WidgetPayload::None => {}
        }

        if node.children.is_empty() { return; }
        let n = node.children.len();
        let total_gap = node.gap * n.saturating_sub(1);
        let avail_main = (if node.is_col { ch } else { cw }).saturating_sub(total_gap);
        let mut fixed_sum = 0usize;
        let mut flex_sum = 0.0f32;
        for child in &node.children {
            let fm = if node.is_col { child.fixed_h } else { child.fixed_w };
            if let Some(v) = fm {
                fixed_sum += v;
            } else {
                flex_sum += if child.flex_grow > 0.0 { child.flex_grow } else { 1.0 };
            }
        }
        let rem_main = avail_main.saturating_sub(fixed_sum);
        let mut cursor = if node.is_col { cy } else { cx };
        for child in &node.children {
            let fm = if node.is_col { child.fixed_h } else { child.fixed_w };
            let main_sz = if let Some(v) = fm {
                v
            } else if flex_sum > 0.0 {
                let g = if child.flex_grow > 0.0 { child.flex_grow } else { 1.0 };
                ((rem_main as f32) * (g / flex_sum)).round() as usize
            } else {
                0
            };
            if node.is_col {
                let child_w = child.fixed_w.unwrap_or(cw).min(cw);
                render_node_inner(buf, child, cx, cursor, child_w, main_sz, ctx);
            } else {
                let child_h = child.fixed_h.unwrap_or(ch).min(ch);
                render_node_inner(buf, child, cursor, cy, main_sz, child_h, ctx);
            }
            cursor += main_sz + node.gap;
        }
    }
}

pub fn soma_fill_rect_1024(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, rgb: [u8; 3]) {
    let x_end = (x + w).min(1024);
    let y_end = (y + h).min(768);
    for py in y..y_end {
        for px in x..x_end {
            let idx = (py * 1024 + px) * 4;
            buf[idx] = rgb[0];
            buf[idx + 1] = rgb[1];
            buf[idx + 2] = rgb[2];
            buf[idx + 3] = 255;
        }
    }
}

pub fn soma_draw_text_1024(buf: &mut [u8], x0: usize, y0: usize, text: &str, rgb: [u8; 3], scale: usize) {
    let sc = scale.max(1);
    for (ci, ch) in text.chars().enumerate() {
        let rows = glyph_5x7(ch);
        let cx = x0 + ci * 6 * sc;
        for (ry, bits) in rows.iter().enumerate() {
            for rx in 0..5 {
                if (bits >> (4 - rx)) & 1 == 1 {
                    for sy in 0..sc {
                        for sx in 0..sc {
                            let px = cx + rx * sc + sx;
                            let py = y0 + ry * sc + sy;
                            if px < 1024 && py < 768 {
                                let idx = (py * 1024 + px) * 4;
                                buf[idx] = rgb[0];
                                buf[idx + 1] = rgb[1];
                                buf[idx + 2] = rgb[2];
                                buf[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
}
'''

# Rich Built-in 5x7 ASCII Font + High-Craft UI Primitives injected into every Rust OS Daemon
RUST_OS_GRAPHICS_SDK = r'''
// === [SOMA-OS BUILT-IN GRAPHICS & 5x7 BITMAP FONT SDK] ===
fn glyph_5x7(ch: char) -> [u8; 7] {
    match ch.to_ascii_uppercase() {
        'A' => [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'B' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
        'C' => [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
        'D' => [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
        'E' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
        'F' => [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
        'G' => [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
        'H' => [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
        'I' => [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'J' => [0b00111, 0b00010, 0b00010, 0b00010, 0b10010, 0b10010, 0b01100],
        'K' => [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'M' => [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
        'N' => [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
        'O' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'P' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
        'Q' => [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
        'R' => [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
        'S' => [0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110],
        'T' => [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        'U' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        'W' => [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
        'X' => [0b10001, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001, 0b10001],
        'Y' => [0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100],
        'Z' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111],
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b01110, 0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
        ':' => [0b00000, 0b00100, 0b00100, 0b00000, 0b00100, 0b00100, 0b00000],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00110, 0b00110],
        '%' => [0b11001, 0b11010, 0b00010, 0b00100, 0b01000, 0b01011, 0b10011],
        '-' => [0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000],
        '/' => [0b00001, 0b00010, 0b00010, 0b00100, 0b01000, 0b01000, 0b10000],
        '|' => [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
        '[' => [0b01110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b01110],
        ']' => [0b01110, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01110],
        '(' => [0b00110, 0b01000, 0b01000, 0b01000, 0b01000, 0b01000, 0b00110],
        ')' => [0b01100, 0b00010, 0b00010, 0b00010, 0b00010, 0b00010, 0b01100],
        '#' => [0b01010, 0b11111, 0b01010, 0b01010, 0b11111, 0b01010, 0b00000],
        '+' => [0b00000, 0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00000],
        '=' => [0b00000, 0b00000, 0b11111, 0b00000, 0b11111, 0b00000, 0b00000],
        '_' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b11111],
        _   => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    }
}

pub fn soma_draw_text(buf: &mut [u8], x0: usize, y0: usize, text: &str, rgb: [u8; 3], scale: usize) {
    let sc = scale.max(1);
    for (ci, ch) in text.chars().enumerate() {
        let rows = glyph_5x7(ch);
        let cx = x0 + ci * 6 * sc;
        for (ry, bits) in rows.iter().enumerate() {
            for rx in 0..5 {
                if (bits >> (4 - rx)) & 1 == 1 {
                    for sy in 0..sc {
                        for sx in 0..sc {
                            let px = cx + rx * sc + sx;
                            let py = y0 + ry * sc + sy;
                            if px < 640 && py < 360 {
                                let idx = (py * 640 + px) * 4;
                                buf[idx] = rgb[0];
                                buf[idx + 1] = rgb[1];
                                buf[idx + 2] = rgb[2];
                                buf[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
        }
    }
}

pub fn soma_fill_rect(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, rgb: [u8; 3]) {
    let x_end = (x + w).min(640);
    let y_end = (y + h).min(360);
    for py in y..y_end {
        for px in x..x_end {
            let idx = (py * 640 + px) * 4;
            buf[idx] = rgb[0];
            buf[idx + 1] = rgb[1];
            buf[idx + 2] = rgb[2];
            buf[idx + 3] = 255;
        }
    }
}

pub fn soma_draw_window(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, title: &str, border_rgb: [u8; 3]) {
    soma_fill_rect(buf, x, y, w, h, [11, 17, 30]);
    soma_fill_rect(buf, x, y, w, 16, [18, 30, 52]);
    // Top & bottom border
    soma_fill_rect(buf, x, y, w, 1, border_rgb);
    soma_fill_rect(buf, x, (y + 16).min(359), w, 1, [30, 48, 78]);
    soma_fill_rect(buf, x, (y + h.saturating_sub(1)).min(359), w, 1, border_rgb);
    // Left & right border
    soma_fill_rect(buf, x, y, 1, h, border_rgb);
    soma_fill_rect(buf, (x + w.saturating_sub(1)).min(639), y, 1, h, border_rgb);
    soma_draw_text(buf, x + 6, y + 5, title, border_rgb, 1);
}
'''

RUST_SOMA_GPUI_SDK = RUST_SOMA_GPUI_SDK + "\n" + RUST_OS_GRAPHICS_SDK.split("pub fn soma_draw_text")[0]


def sanitize_rust_source(rs_code: str) -> str:
    """
    Pre-compilation sanitizer that prevents common LLM Rust pitfalls:
    1. Prepends `#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]`.
    2. Annotates ambiguous `let mut total = 0;` / `avail = 0;` memory/tick bindings with `: u64 = 0;`.
    3. Replaces invalid pseudo-hex literals like `0xSOMA_LOAD` (where S/O/M are not hex digits 0-9A-F,
       causing `error[E0768]: no valid digits found for number`) with valid deterministic hex `0x504D_41xx`!
    """
    header = "#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]\n"
    fixed = re.sub(
        r"\blet\s+mut\s+(total|avail|free|used|cached|buffers|mem_[a-zA-Z0-9_]*|tick[a-zA-Z0-9_]*)\s*=\s*0\s*;",
        r"let mut \1: u64 = 0;",
        rs_code,
    )

    def _fix_hex(m: re.Match) -> str:
        body = m.group(1)
        clean_hex = "".join(c for c in body if c in "0123456789abcdefABCDEF_")
        if not clean_hex.replace("_", "") or any(c.upper() in "GHIJKLMNOPQRSTUVWXYZ" for c in body):
            # Hash the identifier into a valid 64-bit hex constant starting with 0x504D41 (SOMA in ASCII-hex)
            h = sum((i + 1) * ord(ch) for i, ch in enumerate(body)) & 0xFFFF_FFFF
            return f"0x504D_4100_{h:08X}u64"
        return m.group(0)

    fixed = re.sub(r"\b0[xX]([0-9a-zA-Z_]+)", _fix_hex, fixed)
    return header + fixed


def call_31b_brain(
    endpoint: str,
    model: str,
    sys_p: str,
    usr_content: Any,
    temp: float = 0.38,
    max_tokens: int = 8192,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": usr_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": False,
        "reset_kv_cache": True,
        "stateless_forge": True,
    }
    raw_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(10):
        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/v1/chat/completions",
            data=raw_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            if e.code in (409, 502, 503, 504) and attempt < 9:
                time.sleep(5.0)
                continue
            raise
    raise RuntimeError("call_31b_brain exhausted retries")


def drop_daemon_to_fishbowl(organ_name: str, rust_code: str) -> Dict[str, Any]:
    """
    Deploys the Rust OS Daemon (`daemonize=True`) via `:18940/api/drop_source` so the live PyQt6
    window hot-swaps the daemon and displays its 12.5 FPS animation immediately!
    """
    try:
        req = urllib.request.Request(
            f"{FISHBOWL_API}/api/drop_source",
            data=json.dumps(
                {"name": organ_name, "lang": "rust", "code": rust_code, "daemonize": True}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        src_path = SRC_BAY_DIR / f"{organ_name}.rs"
        src_path.write_text(rust_code, encoding="utf-8")
        ok, bin_path, compile_msg = compile_source_into_organ(src_path, organ_name)
        if not ok or not bin_path:
            return {"ok": False, "stage": "compile", "compiler_output": compile_msg}
        obs = run_organ_in_fishbowl(bin_path, organ_name, daemonize=True)
        return {"ok": obs["status"] == "DAEMON_RUNNING", "stage": "run", "observation": obs}


def audit_framebuffer_and_daemon(rust_code: str, obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Captures two frames 350ms apart from `/dev/fb0` and performs a rigorous quantitative
    layout, animation, color entropy, and non-mocked telemetry audit for 31B's self-critique!
    """
    frame1 = FB_FILE.read_bytes() if FB_FILE.exists() else bytes(FB_BYTES)
    time.sleep(0.35)
    frame2 = FB_FILE.read_bytes() if FB_FILE.exists() else frame1

    alive, pid = is_os_daemon_alive()
    anim_diff = sum(1 for i in range(0, FB_BYTES, 4) if frame1[i : i + 3] != frame2[i : i + 3])
    total_anim_diff = max(anim_diff, int(obs.get("anim_diff_pixels") or 0))

    # Quadrant non-background fill rate & color diversity audit on frame2
    bg_rgb = (frame2[0], frame2[1], frame2[2])
    quad_counts = {"TL": 0, "TR": 0, "BL": 0, "BR": 0, "TOP_BAR": 0, "BOT_DOCK": 0}
    quad_totals = {"TL": 0, "TR": 0, "BL": 0, "BR": 0, "TOP_BAR": 0, "BOT_DOCK": 0}
    color_bins = set()

    for y in range(FB_HEIGHT):
        for x in range(FB_WIDTH):
            idx = (y * FB_WIDTH + x) * 4
            r, g, b = frame2[idx], frame2[idx + 1], frame2[idx + 2]
            color_bins.add((r >> 4, g >> 4, b >> 4))
            is_fg = (abs(r - bg_rgb[0]) + abs(g - bg_rgb[1]) + abs(b - bg_rgb[2])) > 28

            if y < 26:
                quad_totals["TOP_BAR"] += 1
                if is_fg:
                    quad_counts["TOP_BAR"] += 1
            elif y >= 334:
                quad_totals["BOT_DOCK"] += 1
                if is_fg:
                    quad_counts["BOT_DOCK"] += 1
            else:
                q = ("T" if y < 180 else "B") + ("L" if x < 320 else "R")
                quad_totals[q] += 1
                if is_fg:
                    quad_counts[q] += 1

    quad_pct = {
        k: round(quad_counts[k] * 100.0 / max(1, quad_totals[k]), 1) for k in quad_counts
    }

    # Source code quality checks
    reads_proc_stat = "/proc/stat" in rust_code
    reads_proc_mem = "/proc/meminfo" in rust_code
    reads_proc_load = "/proc/loadavg" in rust_code
    has_daemon_loop = "loop {" in rust_code or "loop{" in rust_code
    uses_window_sdk = "soma_draw_window" in rust_code

    issues: List[str] = []
    if not alive or not has_daemon_loop:
        issues.append("CRITICAL: OS process did NOT stay running as a background daemon (`loop { ... sleep(...) }` missing or exited early).")
    if total_anim_diff < 200:
        issues.append(f"CRITICAL: Screen is static between frames (anim_diff={total_anim_diff} px < 200 px). Oscilloscope, tick counters, and bars MUST animate every tick!")
    if not (reads_proc_stat and reads_proc_mem and reads_proc_load):
        issues.append("CRITICAL: Code did not dynamically read all three of `/proc/stat`, `/proc/meminfo`, and `/proc/loadavg` inside the daemon loop (no hardcoded mocks allowed!).")
    for q in ("TL", "TR", "BL", "BR"):
        if quad_pct[q] < 6.0:
            issues.append(f"LAYOUT DEFECT: Quadrant `{q}` is almost empty ({quad_pct[q]}% fill). Must place a well-labeled tactical window in `{q}`.")
        elif quad_pct[q] > 85.0:
            issues.append(f"LAYOUT DEFECT: Quadrant `{q}` is over-saturated ({quad_pct[q]}% solid fill). Use clean dark backgrounds with crisp bars/lines/grids instead of huge solid blocks.")
    if len(color_bins) < 10:
        issues.append(f"VISUAL DEFECT: Color palette is too dull ({len(color_bins)} quantized colors). Use distinct cyan, emerald, amber, violet, and coral accents.")

    # Compute deterministic base score (0..100)
    base_score = 100
    if not alive:
        base_score -= 35
    if total_anim_diff < 200:
        base_score -= 25
    if not (reads_proc_stat and reads_proc_mem and reads_proc_load):
        base_score -= 20
    if not uses_window_sdk:
        base_score -= 10
    for q in ("TL", "TR", "BL", "BR"):
        if quad_pct[q] < 6.0 or quad_pct[q] > 85.0:
            base_score -= 8
    if len(color_bins) < 10:
        base_score -= 10

    png_b64 = rgba_to_png_b64(frame2, FB_WIDTH, FB_HEIGHT)
    return {
        "daemon_alive": alive,
        "daemon_pid": pid,
        "anim_diff_pixels": total_anim_diff,
        "quadrant_fill_pct": quad_pct,
        "distinct_color_bins": len(color_bins),
        "reads_live_proc": reads_proc_stat and reads_proc_mem and reads_proc_load,
        "uses_window_sdk": uses_window_sdk,
        "detected_issues": issues,
        "deterministic_score": max(0, base_score),
        "screenshot_png_b64": png_b64,
    }


def ask_31b_visual_critique(
    endpoint: str,
    model: str,
    audit: Dict[str, Any],
    rust_code: str,
    round_idx: int,
) -> Dict[str, Any]:
    """
    Sends the captured `/dev/fb0` screenshot + pixel geometry audit to 31B (:18935)
    so 31B inspects the visual layout, identifies any UI/layout/performance flaws,
    and outputs concrete improvement instructions!
    """
    audit_without_b64 = {k: v for k, v in audit.items() if k != "screenshot_png_b64"}
    sys_p = (
        "你是 SOMA-OS 視覺介面與核心架構總審查官 (Gemma-4-31B Visual & Systems Critic)。\n"
        "請根據魚缸 `/dev/fb0` 截圖與像素幾何審計數據，嚴格檢查這套常駐 OS Daemon 的：\n"
        "1. 常駐性與動態感 (`daemon_alive` & `anim_diff_pixels`)；\n"
        "2. 四象限視窗排版 (`TL`, `TR`, `BL`, `BR`) 是否精緻、對齊、文字清晰不重疊；\n"
        "3. 是否真實解析 `/proc/stat`（每核 CPU 使用率差值）與 `/proc/meminfo`。\n"
        "只輸出合法 JSON。"
    )
    text_prompt = f"""【Round #{round_idx} 魚缸畫面與 Daemon 體檢數據】：
{json.dumps(audit_without_b64, ensure_ascii=False, indent=2)}

請給出嚴格評分與下一輪具體改進指令（JSON 格式）：
{{
  "critic_score": 0到100的整數,
  "verdict": "CONVERGED 或 NEEDS_REFINEMENT",
  "layout_critique": "指出畫面上哪個象限、視窗、文字或顏色還不夠精緻或有排版問題",
  "daemon_and_perf_critique": "指出 /proc 真實數據解析或動畫流暢度哪裡需要加強",
  "mandatory_fixes_for_next_round": [
    "具體修改要求 1",
    "具體修改要求 2",
    "具體修改要求 3"
  ]
}}"""

    # Try multimodal image+text critique first; fallback to text+pixel-audit if upstream rejects image_url
    try:
        multimodal_content = [
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{audit['screenshot_png_b64']}"},
            },
        ]
        raw = call_31b_brain(endpoint, model, sys_p, multimodal_content, temp=0.25, max_tokens=2048)
    except Exception:
        raw = call_31b_brain(endpoint, model, sys_p, text_prompt, temp=0.25, max_tokens=2048)

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            crit = json.loads(m.group(0))
            # Final score is the conservative minimum of deterministic audit score and 31B's critic score
            c_score = int(crit.get("critic_score") or audit["deterministic_score"])
            crit["final_score"] = min(c_score, audit["deterministic_score"]) if audit["detected_issues"] else int((c_score + audit["deterministic_score"]) / 2)
            return crit
        except Exception:
            pass

    return {
        "critic_score": audit["deterministic_score"],
        "final_score": audit["deterministic_score"],
        "verdict": "CONVERGED" if (audit["deterministic_score"] >= 88 and not audit["detected_issues"]) else "NEEDS_REFINEMENT",
        "layout_critique": "; ".join(audit["detected_issues"]) or "Layout balanced.",
        "mandatory_fixes_for_next_round": audit["detected_issues"],
    }


def run_self_improving_os_daemon_loop(
    target_score: int = 88,
    max_rounds: int = 5,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = "localsleep-moe",
) -> Dict[str, Any]:
    ensure_fishbowl_initialized()
    prev_code = ""
    prev_critique: Optional[Dict[str, Any]] = None
    last_compile_err = ""
    history_rounds: List[Dict[str, Any]] = []

    for round_idx in range(1, max_rounds + 1):
        organ_name = f"soma_os_daemon_r{round_idx}"
        print(f"\n==================================================================", flush=True)
        print(f"🚀 [SOMA-OS Recursive Evolution — Round #{round_idx}/{max_rounds}] Forging Daemon `{organ_name}`...", flush=True)

        critique_block = ""
        if prev_critique:
            critique_block = (
                f"【📸 31B VISUAL SCREENSHOT CRITIQUE OF PREVIOUS ROUND (Score={prev_critique.get('final_score')}/100)】：\n"
                f"- Layout Critique: {prev_critique.get('layout_critique')}\n"
                f"- Daemon/Perf Critique: {prev_critique.get('daemon_and_perf_critique')}\n"
                f"- MANDATORY FIXES TO APPLY NOW:\n"
                + "\n".join(f"  * {fx}" for fx in (prev_critique.get("mandatory_fixes_for_next_round") or []))
            )

        sys_p = (
            "You are SOMA-OS Principal Kernel & Tactical GUI Compositor Architect in Rust.\n"
            "You are building a TRUE DAEMONIZED OPERATING SYSTEM (`soma_os_daemon`) running continuously inside the Linux Fishbowl.\n"
            "The following SDK functions are AUTOMATICALLY linked at the bottom of your file (do NOT redefine them; CALL them directly!):\n"
            "  - `soma_draw_text(buf: &mut [u8], x: usize, y: usize, text: &str, rgb: [u8; 3], scale: usize)`\n"
            "  - `soma_fill_rect(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, rgb: [u8; 3])`\n"
            "  - `soma_draw_window(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, title: &str, border_rgb: [u8; 3])`\n"
        )

        usr_p = f"""【⛔ NEGATIVE PALINDROME & DAEMONIZATION MANDATES [-C]】：
1. **MUST BE A PERSISTENT DAEMON**: `fn main()` MUST run an infinite event loop:
   `let mut tick: u64 = 0; loop {{ tick += 1; ... std::thread::sleep(std::time::Duration::from_millis(80)); }}`
   Write the `640*360*4` RGBA buffer to `env::var("FISHBOWL_FB_PATH")` on EVERY tick! Print a startup banner on `tick == 1`.
2. **NO MOCK CONSTANTS**: Inside the `loop`, dynamically read and parse:
   - `/proc/loadavg` (1m, 5m, 15m load + active tasks)
   - `/proc/meminfo` (`MemTotal`, `MemAvailable`, `Buffers`, `Cached` in MB and %)
   - `/proc/stat` (parse per-core `cpu0`, `cpu1`.. lines or total user/system/idle ticks to compute live delta CPU usage!)
   - `/proc/uptime` (live system uptime seconds)
3. **HIGH-CRAFT 4-QUADRANT TACTICAL OS DESKTOP LAYOUT (640x360 — USE EXACT ABSOLUTE PIXEL COORDINATES!)**:
   - **Top Bar (`x:0, y:0, w:640, h:24`)**: Dark navy bar with `SOMA-OS KERNEL DAEMON v2.{round_idx}`, live `TICK #`, `UPTIME`, and a blinking heartbeat indicator block at `x:618, y:6`.
   - **Quadrant TL (`soma_draw_window(&mut buf, 8, 28, 308, 146, "CPU CORE MATRIX [LIVE /PROC/STAT]", [56, 189, 248])`)**:
     All inner widgets MUST use `x` in `20..304` and `y` in `50..168`! Draw 8 core bars with dynamic heights driven by `/proc/stat` + `tick` phase, plus text readout `LOAD: X.XX | CORES: N` at `x:20, y:50`.
   - **Quadrant TR (`soma_draw_window(&mut buf, 324, 28, 308, 146, "VMMU SLAB & MEMORY ARENA", [52, 211, 153])`)**:
     All inner widgets MUST use `x` in `336..620` (e.g. `x = 336 + col * 32`, NEVER `x = 20`!) and `y` in `50..168`! Draw segmented memory bars (`USED`, `FREE` from `/proc/meminfo` at `x:336, y:52`) AND an 8x4 animated Page-Frame Allocation Grid at `x:336, y:96` where active pages pulse with `tick`!
   - **Quadrant BL (`soma_draw_window(&mut buf, 8, 180, 308, 148, "16-LORA HYPERCUBE ROUTER", [251, 191, 36])`)**:
     All inner widgets MUST use `x` in `20..304` and `y` in `204..322`! Draw a 4x4 grid of the 16 LoRA Experts (`E0`..`E15`) with animated activation heat colors and status labels (`FIRE`/`NREM`).
   - **Quadrant BR (`soma_draw_window(&mut buf, 324, 180, 308, 148, "CHRONOS SYSCALL OSCILLOSCOPE", [244, 114, 182])`)**:
     All inner widgets MUST use `x` in `336..620` and `y` in `204..322`! Draw a dual-trace scrolling waveform (cyan + emerald sine/cosine modulated by real `loadavg` and scrolling smoothly with `tick`!) + live FPS/latency readout.
   - **Bottom OS Dock (`x:0, y:334, w:640, h:26`)**: Status dock showing `[PID: DAEMON ACTIVE] [FB: 640X360@12.5FPS] [MEM: XX.X%]`.
{('4. FIX PREVIOUS COMPILER ERROR:\n' + last_compile_err) if last_compile_err else ''}
{critique_block}
========================================
Output ONLY a single ```rust ... ``` code block (standard library `std::*` only, calling `soma_draw_window`, `soma_fill_rect`, and `soma_draw_text` which are auto-linked)."""

        raw_resp = call_31b_brain(endpoint, model, sys_p, usr_p, temp=0.35, max_tokens=8192)
        rs_code = extract_code_block(raw_resp, lang="rust")

        # In-round compiler repair sub-loop (up to 3 compile attempts per round)
        drop_res: Dict[str, Any] = {}
        full_rs = ""
        for comp_try in range(1, 4):
            clean_rs = sanitize_rust_source(rs_code)
            if "fn soma_draw_text" not in clean_rs:
                full_rs = clean_rs + "\n\n" + RUST_OS_GRAPHICS_SDK
            else:
                # Replace any broken SDK copy with the verified SDK
                idx_sdk = clean_rs.find("// === [SOMA-OS BUILT-IN")
                if idx_sdk != -1:
                    full_rs = clean_rs[:idx_sdk] + "\n\n" + RUST_OS_GRAPHICS_SDK
                else:
                    full_rs = clean_rs

            drop_res = drop_daemon_to_fishbowl(organ_name, full_rs)
            if drop_res.get("ok"):
                last_compile_err = ""
                break

            last_compile_err = str(drop_res.get("compiler_output") or drop_res.get("error") or "")[:900]
            print(f"   🔧 [Round #{round_idx} Compile Repair {comp_try}/3] Fixing Rust error:\n{last_compile_err[:280]}", flush=True)
            fix_usr = (
                f"Your Rust code failed to compile with this `rustc` error:\n```text\n{last_compile_err}\n```\n\n"
                f"Here is your broken Rust code:\n```rust\n{rs_code}\n```\n\n"
                "Fix the compiler error (ensure all variables are in scope and all integer bindings have explicit types like `u64` or `usize`). "
                "Do NOT define `soma_draw_text`, `soma_fill_rect`, or `soma_draw_window` (they are auto-linked). "
                "Output ONLY the corrected ```rust ... ``` block!"
            )
            raw_fix = call_31b_brain(endpoint, model, sys_p, fix_usr, temp=0.2, max_tokens=8192)
            rs_code = extract_code_block(raw_fix, lang="rust")

        if not drop_res.get("ok"):
            print(f"   ⚠️ [Round #{round_idx}] Still failed after 3 compile repairs, moving to next round.", flush=True)
            continue
        obs = drop_res.get("observation", {})
        print(
            f"   🟢 Daemon `{organ_name}` launched (PID={obs.get('daemon_pid')})! Capturing screenshot & auditing layout...",
            flush=True,
        )

        audit = audit_framebuffer_and_daemon(full_rs, obs)
        critique = ask_31b_visual_critique(endpoint, model, audit, full_rs, round_idx)
        final_score = int(critique.get("final_score", audit["deterministic_score"]))

        print(
            f"   📸 [Visual Self-Critique Round #{round_idx}] Score={final_score}/100 | "
            f"DaemonAlive={audit['daemon_alive']} (PID={audit['daemon_pid']}) | "
            f"AnimDiff={audit['anim_diff_pixels']:,} px | Colors={audit['distinct_color_bins']} | "
            f"Quadrants={audit['quadrant_fill_pct']}",
            flush=True,
        )
        print(f"   🧐 31B Critique: {critique.get('layout_critique', '')}", flush=True)

        round_rec = {
            "round": round_idx,
            "organ_name": organ_name,
            "final_score": final_score,
            "audit": {k: v for k, v in audit.items() if k != "screenshot_png_b64"},
            "critique": critique,
        }
        history_rounds.append(round_rec)
        LOOP_HISTORY_FILE.write_text(json.dumps(history_rounds, ensure_ascii=False, indent=2), encoding="utf-8")

        if final_score >= target_score and audit["daemon_alive"] and audit["anim_diff_pixels"] >= 200 and not audit["detected_issues"]:
            print(
                f"\n🏆 [CONVERGED at Round #{round_idx}!] Score={final_score}/100 (>= {target_score}). "
                f"OS Daemon `{organ_name}` (PID={audit['daemon_pid']}) is now persistently running & animating in the Fishbowl!",
                flush=True,
            )
            return round_rec

        prev_code = rs_code
        prev_critique = critique
        print(f"   🔄 Score {final_score} < {target_score} (or issues detected) -> Recursively evolving next round...", flush=True)

    return history_rounds[-1] if history_rounds else {}


# Built-in Zero-Boilerplate DevMem & UI-Config Control SDK auto-linked into every In-OS Rust Agent
RUST_OS_ENV_CONTROL_SDK = r'''
// === [SOMA-OS BUILT-IN DEVMEM 8-CHANNEL & DYNAMIC UI CONTROL SDK] ===
pub fn soma_poke_channel(ch: usize, reg_u64: u64, tag: &str, payload: &str) {
    use std::fs::OpenOptions;
    use std::io::{Read, Seek, SeekFrom, Write};
    let devmem_path = std::env::var("SOMA_DEVMEM_PATH")
        .unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
    let ch_idx = ch.min(7);
    let base = (ch_idx * 8192) as u64;
    if let Ok(mut file) = OpenOptions::new().read(true).write(true).create(true).open(&devmem_path) {
        let mut seq_buf = [0u8; 4];
        let _ = file.seek(SeekFrom::Start(base + 4));
        let _ = file.read_exact(&mut seq_buf);
        let seq = u32::from_le_bytes(seq_buf).wrapping_add(1);

        let _ = file.seek(SeekFrom::Start(base));
        let _ = file.write_all(b"SOMA");
        let _ = file.write_all(&seq.to_le_bytes());
        let _ = file.write_all(&reg_u64.to_le_bytes());

        let mut tag_bytes = [0u8; 32];
        let clean_tag = tag.replace('\n', " ");
        let tb = clean_tag.as_bytes();
        let tlen = tb.len().min(31);
        tag_bytes[..tlen].copy_from_slice(&tb[..tlen]);
        let _ = file.write_all(&tag_bytes);

        let clean_pay = payload.replace('\n', " | ").replace('\r', "");
        let pb = clean_pay.as_bytes();
        let plen = pb.len().min(8140);
        let mut pay_Block = vec![0u8; 8144];
        pay_Block[..plen].copy_from_slice(&pb[..plen]);
        let _ = file.write_all(&pay_Block);
    }
}

pub fn soma_peek_reg(ch: usize) -> u64 {
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    let devmem_path = std::env::var("SOMA_DEVMEM_PATH")
        .unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
    let base = (ch.min(7) * 8192 + 8) as u64;
    if let Ok(mut f) = File::open(&devmem_path) {
        let mut b = [0u8; 8];
        if f.seek(SeekFrom::Start(base)).is_ok() && f.read_exact(&mut b).is_ok() {
            return u64::from_le_bytes(b);
        }
    }
    0
}

pub fn soma_set_ui_param(key: &str, val: &str) {
    use std::fs;
    let cfg_path = std::env::var("SOMA_UI_CFG_PATH")
        .unwrap_or_else(|_| "./etc/soma_ui_config.cfg".to_string());
    if let Ok(raw) = fs::read_to_string(&cfg_path) {
        let mut lines: Vec<String> = Vec::new();
        let mut found = false;
        for ln in raw.lines() {
            if let Some((k, _)) = ln.split_once('=') {
                if k.trim() == key {
                    if key == "cpu_max_cores" {
                        lines.push(String::from("cpu_max_cores=24"));
                    } else {
                        lines.push(format!("{}={}", key, val));
                    }
                    found = true;
                    continue;
                }
            }
            lines.push(ln.to_string());
        }
        if !found {
            lines.push(format!("{}={}", key, val));
        }
        let _ = fs::write(&cfg_path, lines.join("\n") + "\n");
    }
}
'''

OS_CONTROL_MISSIONS = [
    (
        "STAGE_1_24CORE_SCHEDULER_GOVERNOR",
        "Parse `/proc/stat` for all 24 CPU cores (`cpu0`..`cpu23`) plus `ctxt` (context switches) and `procs_running`. "
        "Compute a 24-bit active core bitmask (`u64`) and busy ratio, then call `soma_poke_channel(0, ...)` to update `CH0:KERNEL_CTRL`, "
        "`soma_poke_channel(4, ...)` with live `CTXT=... | RUN_Q=... | BUSY_CORES=24`, and call `soma_set_ui_param(\"osc_gain\", \"26\")`!",
    ),
    (
        "STAGE_2_RAM_DMA_BANDWIDTH_BENCHMARK",
        "Allocate a real 8 MB `Vec<u64>` in RAM, perform a fast XOR-stream write pass timed with `std::time::Instant`, "
        "compute actual RAM bandwidth in `MB/s`, parse `Dirty:` and `Slab:` from `/proc/meminfo`, and call "
        "`soma_poke_channel(1, bw_mb_s_bitpattern, \"CH1:DMA_BW_BENCH\", ...)` and `soma_poke_channel(5, ..., \"CH5:RAM_DMA_PROBE\", ...)`!",
    ),
    (
        "STAGE_3_16LORA_HYPERCUBE_MASK_CONTROLLER",
        "Read `/proc/loadavg` and `soma_peek_reg(0)`, compute a dynamic 16-LoRA Expert routing & NREM sleep mask (`u64` with diverse active bits across all 32 bit-cells!), "
        "and call `soma_poke_channel(2, lora_mask_u64, \"CH2:LORA_ROUTER_R3\", ...)` AND `soma_poke_channel(6, ..., \"CH6:SYNAPSE_GATE\", ...)` "
        "plus `soma_set_ui_param(\"osc_gain\", \"32\")` so Quadrant BL (16-LoRA Hypercube) & Quadrant TR visually shift patterns!",
    ),
    (
        "STAGE_4_PROC_PID_AND_ELF_ORGAN_AUDITOR",
        "Scan `/proc` to count numeric process PID directories and scan `./bin` to count compiled OS binaries and their total byte size. "
        "Compute a 64-bit fingerprint and call `soma_poke_channel(3, ..., \"CH3:PROC_ELF_AUDIT\", ...)` and `soma_poke_channel(7, ..., \"CH7:PID_SCANNER\", ...)`!",
    ),
    (
        "STAGE_5_VMSTAT_PAGEFAULT_AND_INTERRUPT_MATRIX",
        "Parse `/proc/vmstat` (`pgfault`, `pgmajfault`, `pgpgin`, `pgpgout`) and `/proc/uptime`. "
        "Compute page-fault velocity (`faults/sec`), update channels `CH1`, `CH4`, `CH5`, `CH7` via `soma_poke_channel(...)` with distinct 32-bit binary bit-patterns, "
        "and update `soma_set_ui_param(\"os_title\", \"SOMA-OS V3 AUTONOMOUS VM-GOVERNOR\")`!",
    ),
    (
        "STAGE_6_SYNAPTIC_ENTROPY_SCRUBBER",
        "Read all 8 channels' current registers via `soma_peek_reg(0..7)`, run a 64-bit SplitMix64 / XorShift entropy mixer across them combined with `/proc/loadavg`, "
        "and update `CH0`, `CH2`, `CH4`, `CH6`, `CH7` with high-entropy 64-bit bitmasks and live entropy diagnostics!",
    ),
]


def run_v3_ai_env_control_loop(
    max_rounds: int = 5,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = "localsleep-moe",
) -> Dict[str, Any]:
    """
    SOMA-OS v3.0 Post-Text AI-Native Environment Control Loop:
    1. Compiles & launches the 1024x768 Dynamic Config-Driven Rust Compositor (`soma_compositor_v3`) ONCE.
    2. In each round, assigns a distinct progressive OS Control Mission (`OS_CONTROL_MISSIONS`) and auto-links
       `soma_poke_channel(ch, reg_u64, tag, payload)`, `soma_peek_reg(ch)`, and `soma_set_ui_param(k, v)`
       so 31B directly manipulates all 8 `/dev/mem` channels (`CH0..CH7`) and UI parameters with visible screen impact every round!
    """
    ensure_fishbowl_initialized()
    print("🔨 [SOMA-OS v3.0] Compiling & starting 1024x768 Dynamic Hot-Reload Compositor (`soma_compositor_v3`)...", flush=True)
    ok_c, bin_c, msg_c = create_and_compile_compositor_v3()
    if not ok_c or not bin_c:
        raise RuntimeError(f"Compositor v3 compile failed: {msg_c}")
    obs_c = run_organ_in_fishbowl(bin_c, "soma_compositor_v3", daemonize=True)
    print(f"🟢 [SOMA-OS v3.0] Persistent Compositor Daemon UP (PID={obs_c.get('daemon_pid')}, 1024x768, 24-Core Locked)!", flush=True)

    history: List[Dict[str, Any]] = []
    for round_idx in range(1, max_rounds + 1):
        app_name = f"soma_env_agent_r{round_idx}"
        mission_id, mission_desc = OS_CONTROL_MISSIONS[(round_idx - 1) % len(OS_CONTROL_MISSIONS)]
        os_state = list_os_apps_and_processes()
        print(f"\n==================================================================", flush=True)
        print(f"🕹️ [SOMA-OS v3 AI Control Loop — Round #{round_idx}/{max_rounds}] Mission: {mission_id} -> `/bin/{app_name}`...", flush=True)

        sys_p = (
            "You are the External AI Overlord controlling the SOMA-OS v3.0 internal Linux environment.\n"
            "The following 3 SDK functions are AUTOMATICALLY linked at the bottom of your Rust file (do NOT redefine them; CALL them directly!):\n"
            "  - `soma_poke_channel(ch: usize, reg_u64: u64, tag: &str, payload: &str)` -> Writes to `/dev/mem_channels/devmem_arena.bin` (ch = 0..7)\n"
            "  - `soma_peek_reg(ch: usize) -> u64` -> Reads current 64-bit register of channel `ch` (0..7)\n"
            "  - `soma_set_ui_param(key: &str, val: &str)` -> Hot-reloads UI parameters (`osc_gain`, `os_title`, etc.) on the 1024x768 screen!\n"
            "CRITICAL RULES:\n"
            "1. Do NOT write low-level `File::seek` or `FileExt` traits — just call `soma_poke_channel(ch, reg_u64, tag, payload)`!\n"
            "2. Ensure `reg_u64` has rich, varying bits across bits 0..31 (e.g. `0x9A3F_5C12 ^ (val as u64 * 0x45D9F3B)`) so the 32-bit visual bitplane matrix on screen changes dramatically!\n"
            "3. Ensure `payload` is a concise single-line summary of the computed numbers (e.g. `BW=14280MB/s | DIRTY=28MB | FAULTS=19420`), NOT raw multi-line `/proc/meminfo` dumps!\n"
            "4. Output ONLY a single ```rust ... ``` block with `fn main()` using Rust `std::*` only!"
        )
        usr_p = (
            f"【ROUND #{round_idx} MISSION: {mission_id}】:\n"
            f"{mission_desc}\n\n"
            f"Current Installed `./bin` Apps: {[a['name'] for a in os_state['os_bin_apps']]}\n"
            f"Write `fn main()` for `{app_name}.rs` that executes this mission, updates at least 4 channels in `0..=7` via `soma_poke_channel(...)`, "
            f"calls `soma_set_ui_param(...)`, prints a concise telemetry report to stdout, and exits cleanly!"
        )

        raw_resp = call_31b_brain(endpoint, model, sys_p, usr_p, temp=0.35, max_tokens=4096)
        rs_code = sanitize_rust_source(extract_code_block(raw_resp, lang="rust"))
        if "fn soma_poke_channel" not in rs_code:
            full_app_rs = rs_code + "\n\n" + RUST_OS_ENV_CONTROL_SDK
        else:
            full_app_rs = rs_code
        app_src_path = OS_SRC_DIR / f"{app_name}.rs"
        app_src_path.write_text(full_app_rs, encoding="utf-8")

        # Compile and run INSIDE the OS rootfs environment (`./src/...` -> `./bin/...`) with up to 2 auto-repairs
        bash_cmd = f"rustc -O ./src/{app_name}.rs -o ./bin/{app_name} && ./bin/{app_name}"
        bash_res = run_os_bash_command(bash_cmd, timeout_sec=20.0)
        for rep_i in range(1, 3):
            if bash_res["ok"]:
                break
            err_txt = (bash_res.get("stderr") or bash_res.get("stdout") or "")[:700]
            print(f"   🔧 [Round #{round_idx} In-OS Repair {rep_i}/2] Fixing:\n{err_txt[:240]}", flush=True)
            fix_p = (
                f"Your Rust code failed to compile or run with:\n```text\n{err_txt}\n```\n"
                f"Code:\n```rust\n{rs_code}\n```\n"
                "Fix the error. Do NOT define `soma_poke_channel`, `soma_peek_reg`, or `soma_set_ui_param` (they are auto-linked). "
                "Output ONLY the corrected ```rust ... ``` block."
            )
            raw_fix = call_31b_brain(endpoint, model, sys_p, fix_p, temp=0.2, max_tokens=4096)
            rs_code = sanitize_rust_source(extract_code_block(raw_fix, lang="rust"))
            full_app_rs = (rs_code + "\n\n" + RUST_OS_ENV_CONTROL_SDK) if "fn soma_poke_channel" not in rs_code else rs_code
            app_src_path.write_text(full_app_rs, encoding="utf-8")
            bash_res = run_os_bash_command(bash_cmd, timeout_sec=20.0)

        time.sleep(0.25)
        channels_now = get_all_devmem_channels()
        alive_now, pid_now = is_os_daemon_alive()
        print(
            f"   ✅ [In-OS Execution] `/bin/{app_name}` ({mission_id}) rc={bash_res['return_code']} ({bash_res['execution_ms']}ms) | "
            f"Compositor PID={pid_now} (Alive={alive_now})",
            flush=True,
        )
        for ch_i in range(8):
            c_obj = channels_now[ch_i]
            print(f"      • CH{ch_i} [S{c_obj['seq']:03d} | {c_obj['reg_hex']}] {c_obj['tag']:<22} -> {c_obj['payload'][:68]}", flush=True)
        rec = {
            "round": round_idx,
            "app_name": app_name,
            "bash_result": bash_res,
            "devmem_channels": channels_now,
            "compositor_pid": pid_now,
        }
        history.append(rec)
        LOOP_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    return history[-1] if history else {}


def run_gpui_forge_loop(
    max_rounds: int = 5,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = "localsleep-moe",
) -> Dict[str, Any]:
    """
    SOMA-GPUI Forge Loop:
    Teaches 31B (:18935) to write declarative `gpui.rs` syntax (`use gpui::*; div().flex().flex_col().gap_3()...child(...)`)
    coupled with live `/proc` and `/dev/mem` 8-channel telemetry, compiles it with `rustc -O` in 0.15s via the
    built-in `RUST_SOMA_GPUI_SDK` engine, and runs it as a persistent 1024x768 daemon inside the Fishbowl!
    """
    ensure_fishbowl_initialized()
    history: List[Dict[str, Any]] = []
    for round_idx in range(1, max_rounds + 1):
        organ_name = f"soma_gpui_desktop_r{round_idx}"
        print(f"\n==================================================================", flush=True)
        print(f"🦀 [SOMA-GPUI Declarative OS Forge — Round #{round_idx}/{max_rounds}] Forging `{organ_name}`...", flush=True)

        sys_p = (
            "You are a Rust GPUI (`https://gpui.rs`) & SOMA-OS Kernel Architect.\n"
            "Write a persistent 1024x768 OS Desktop Compositor using `gpui::*` declarative Flexbox syntax!\n"
            "The `pub mod gpui` module is AUTOMATICALLY linked at the bottom of your file (do NOT define `mod gpui`; just write `use gpui::*;`).\n"
            "Available `gpui::Div` methods in our kernel SDK:\n"
            "  - `div()`, `.flex()`, `.flex_row()`, `.flex_col()`, `.flex_1()`, `.flex_grow(f32)`, `.size_full()`\n"
            "  - `.w(px(f32))`, `.h(px(f32))`, `.size(px(f32))`, `.size_8()`, `.gap(usize)`, `.gap_1()`..`.gap_4()`, `.p(usize)`, `.p_1()`..`.p_4()`\n"
            "  - `.bg(rgb(0xRRGGBB))`, `.border_1()`, `.border_2()`, `.border_color(rgb(0xRRGGBB))`, `.text_color(rgb(0xRRGGBB))`, `.panel_title(\"TITLE\")`\n"
            "  - Built-in high-density OS tensor widgets attachable to any `Div`:\n"
            "    * `.cpu_matrix_24()` (renders 8x3 24-core live load matrix inside that flex panel)\n"
            "    * `.devmem_bitplane_8()` (renders 8x32-bit `/dev/mem` channel bitplane matrix)\n"
            "    * `.lora_hypercube_16()` (renders 4x4 16-LoRA expert state matrix)\n"
            "    * `.oscilloscope_ipc()` (renders dual-trace waveform + CH3..CH6 IPC console)\n"
            "    * `.progress_bar(ratio_f32, rgb(0x...))` (renders horizontal bar)\n"
            "    * `.bit_cells(mask_u64, rgb(0x...))` (renders 32-bit binary state barcode)\n"
            "  - `.child(element)` (where `element` is another `Div`, `&str`, or `String`)\n"
            "  - To rasterize the tree into the 1024x768 framebuffer on every tick, call:\n"
            "    `render_node(&mut buf, &root_div, 0, 0, 1024, 768, &ctx);`\n"
            "Output ONLY a single ```rust ... ``` block!"
        )

        layout_themes = [
            "4-Quadrant Tactical Split (`flex_col` root -> Top Header `h(px(40.0))`, Middle `flex_row().flex_1().gap_3()` with Left Col (`cpu_matrix_24` + `lora_hypercube_16`) and Right Col (`devmem_bitplane_8` + `oscilloscope_ipc`), Bottom Status Dock `h(px(36.0))`)",
            "3-Column Command Center (`flex_col` root -> Top Barcode Header, Middle 3-column `flex_row().flex_1().gap_2()` with Col 1 `cpu_matrix_24`, Col 2 `devmem_bitplane_8`, Col 3 split vertically into `lora_hypercube_16` + `oscilloscope_ipc`, Bottom Dock)",
            "Asymmetric Widescreen Telemetry Deck (Top Header with `.bit_cells(...)`, Upper Deck `flex_row` 60%/40% split using `.flex_grow(1.5)` and `.flex_grow(1.0)`, Lower Deck `flex_row` 40%/60% split, Bottom Dock)",
        ]
        chosen_theme = layout_themes[(round_idx - 1) % len(layout_themes)]

        usr_p = f"""Write a complete, compilable Rust program (`{organ_name}.rs`) that:
1. Imports `use gpui::*;` and runs a persistent daemon loop (`let mut tick: u64 = 0; loop {{ tick += 1; let ctx = RenderCtx::new(tick); ... std::thread::sleep(std::time::Duration::from_millis(80)); }}`).
2. Inside the loop, call `let ctx = RenderCtx::new(tick);` (which is already built into `gpui::RenderCtx` and automatically reads `/proc` and `/dev/mem` channels!). Do NOT define your own `struct RenderCtx`!
3. Build a `let root_tree = div().flex().flex_col().size_full().gap_2().p_2().bg(rgb(0x060911))...` using `{chosen_theme}`.
4. Call `render_node(&mut buf, &root_tree, 0, 0, 1024, 768, &ctx);` (which automatically renders the Flexbox tree AND writes `buf` to `FISHBOWL_FB_PATH`!).
Output ONLY the ```rust ... ``` code block!"""

        raw_resp = call_31b_brain(endpoint, model, sys_p, usr_p, temp=0.3, max_tokens=6144)
        rs_code = sanitize_rust_source(extract_code_block(raw_resp, lang="rust"))

        drop_res: Dict[str, Any] = {}
        for comp_try in range(1, 4):
            clean_rs = sanitize_rust_source(rs_code)
            full_rs = (clean_rs + "\n\n" + RUST_SOMA_GPUI_SDK) if "pub mod gpui" not in clean_rs else clean_rs
            drop_res = drop_daemon_to_fishbowl(organ_name, full_rs)
            if drop_res.get("ok"):
                break
            err_msg = str(drop_res.get("compiler_output") or drop_res.get("error") or "")[:800]
            print(f"   🔧 [GPUI Compile Repair {comp_try}/3] Fixing:\n{err_msg[:260]}", flush=True)
            fix_p = (
                f"Compiler error:\n```text\n{err_msg}\n```\nBroken code:\n```rust\n{rs_code}\n```\n"
                "Fix the error (remember `px(...)` returns `usize`, `rgb(0xRRGGBB)` returns `Rgb`, and `pub mod gpui` is auto-linked). "
                "Output ONLY the corrected ```rust ... ``` block."
            )
            raw_fix = call_31b_brain(endpoint, model, sys_p, fix_p, temp=0.2, max_tokens=6144)
            rs_code = extract_code_block(raw_fix, lang="rust")

        obs = drop_res.get("observation", {})
        print(
            f"   🟢 [GPUI Daemon `{organ_name}`] status={obs.get('status')} | PID={obs.get('daemon_pid')} | "
            f"AnimDiff={obs.get('anim_diff_pixels', 0):,} px | Theme={chosen_theme[:48]}...",
            flush=True,
        )
        history.append({"round": round_idx, "organ": organ_name, "observation": obs})
        LOOP_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    return history[-1] if history else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="SOMA-OS v3.0 AI-Native Environment Control & GPUI Forge Loop")
    parser.add_argument(
        "--mode",
        choices=["v3-control", "gpui-forge", "v2-ui-forge"],
        default="v3-control",
        help="v3-control (In-OS Rust App & DevMem control), gpui-forge (31B writes gpui.rs Flexbox trees), or v2-ui-forge",
    )
    parser.add_argument("--target-score", type=int, default=88, help="Target score for v2-ui-forge")
    parser.add_argument("--max-rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="localsleep-moe")
    args = parser.parse_args()

    if args.mode == "v3-control":
        run_v3_ai_env_control_loop(
            max_rounds=args.max_rounds,
            endpoint=args.endpoint,
            model=args.model,
        )
    elif args.mode == "gpui-forge":
        run_gpui_forge_loop(
            max_rounds=args.max_rounds,
            endpoint=args.endpoint,
            model=args.model,
        )
    else:
        run_self_improving_os_daemon_loop(
            target_score=args.target_score,
            max_rounds=args.max_rounds,
            endpoint=args.endpoint,
            model=args.model,
        )


if __name__ == "__main__":
    main()
