#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]
#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]
use gpui::*;
use std::thread;
use std::time::Duration;

fn main() {
    // Initialize the framebuffer buffer for the SOMA-OS kernel
    let mut buf = vec![0u8; 1024 * 768 * 4];
    let mut tick: u64 = 0;

    loop {
        tick += 1;
        
        // RenderCtx automatically polls /proc and /dev/mem for the tensor widgets
        let ctx = RenderCtx::new(tick);

        // 4-Quadrant Tactical Split Layout
        let root_tree = div()
            .flex()
            .flex_col()
            .size_full()
            .gap_2()
            .p_2()
            .bg(rgb(0x060911))
            .child(
                // Top Header: System Identity & Global State
                div()
                    .flex()
                    .flex_row()
                    .h(px(40.0))
                    .p_2()
                    .bg(rgb(0x1A1F2E))
                    .border_1()
                    .border_color(rgb(0x3F4A6B))
                    .text_color(rgb(0x00FFCC))
                    .panel_title("SOMA-OS KERNEL COMPOSITOR v1.0")
                    .child(
                        div()
                            .flex_1()
                            .flex_row()
                            .child("SYS_TICK: ")
                            .child(tick.to_string())
                    )
            )
            .child(
                // Middle Section: Tactical Data Grid
                div()
                    .flex()
                    .flex_row()
                    .flex_1()
                    .gap_3()
                    .child(
                        // Left Column: Compute & Neural State
                        div()
                            .flex_1()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .bg(rgb(0x0D1117))
                                    .border_1()
                                    .border_color(rgb(0x222B3C))
                                    .panel_title("CORE_LOAD_MATRIX")
                                    .cpu_matrix_24()
                            )
                            .child(
                                div()
                                    .bg(rgb(0x0D1117))
                                    .border_1()
                                    .border_color(rgb(0x222B3C))
                                    .panel_title("LoRA_EXPERT_STATE")
                                    .lora_hypercube_16()
                            )
                    )
                    .child(
                        // Right Column: Memory & Signal Analysis
                        div()
                            .flex_1()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .bg(rgb(0x0D1117))
                                    .border_1()
                                    .border_color(rgb(0x222B3C))
                                    .panel_title("DEV_MEM_BITPLANE")
                                    .devmem_bitplane_8()
                            )
                            .child(
                                div()
                                    .bg(rgb(0x0D1117))
                                    .border_1()
                                    .border_color(rgb(0x222B3C))
                                    .panel_title("IPC_OSCILLOSCOPE")
                                    .oscilloscope_ipc()
                            )
                    )
            )
            .child(
                // Bottom Status Dock: Hardware Telemetry
                div()
                    .flex()
                    .flex_row()
                    .h(px(36.0))
                    .p_1()
                    .bg(rgb(0x0A0E17))
                    .border_2()
                    .border_color(rgb(0x3F4A6B))
                    .child(
                        div()
                            .flex_1()
                            .flex_row()
                            .gap_4()
                            .child(
                                div()
                                    .progress_bar(0.75, rgb(0x00FF41))
                                    .child("MEM_UTIL")
                            )
                            .child(
                                div()
                                    .bit_cells(0xDEADBEEFCAFEBABE, rgb(0x00BFFF))
                                    .child("REG_SOMA_S0")
                            )
                    )
            );

        // Rasterize the declarative tree into the 1024x768 framebuffer
        render_node(&mut buf, &root_tree, 0, 0, 1024, 768, &ctx);

        // Maintain 12.5Hz refresh rate for the kernel compositor
        thread::sleep(Duration::from_millis(80));
    }
}


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
        pub fn progress_bar(mut self, ratio: f32, c: Rgb) -> Self { self.payload = WidgetPayload::ProgressBar(ratio, c); self }
        pub fn bit_cells(mut self, mask: u64, c: Rgb) -> Self { self.payload = WidgetPayload::BitCells(mask, c); self }
        pub fn child<E: IntoElement>(mut self, el: E) -> Self {
            self.children.push(el.into_div());
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
            super::soma_fill_rect_1024(buf, x, y, w, 22.min(h), [18, 30, 52]);
            super::soma_fill_rect_1024(buf, x, (y + 22).min(767), w, 1, [30, 48, 78]);
            super::soma_draw_text_1024(buf, x + 8, y + 7, title, [bc.0, bc.1, bc.2], 1);
            inner_y = (y + 23).min(y + h);
            inner_h = h.saturating_sub(23);
        }
        if node.border_w > 0 {
            let bc = node.border_color.unwrap_or(rgb(0x38bdf8));
            let c = [bc.0, bc.1, bc.2];
            super::soma_fill_rect_1024(buf, x, y, w, 1, c);
            super::soma_fill_rect_1024(buf, x, (y + h.saturating_sub(1)).min(767), w, 1, c);
            super::soma_fill_rect_1024(buf, x, y, 1, h, c);
            super::soma_fill_rect_1024(buf, (x + w.saturating_sub(1)).min(1023), y, 1, h, c);
        }

        let cx = x + node.pad;
        let cy = inner_y + node.pad;
        let cw = w.saturating_sub(node.pad * 2);
        let ch = inner_h.saturating_sub(node.pad * 2);

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

