#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]
use std::env;
use std::fs;
use std::io::Write;
use std::thread;
use std::time::{Duration, Instant};

struct UiConfig {
    os_title: String,
    cpu_max_cores: usize,
    cpu_cols: usize,
    cpu_rows: usize,
    cpu_col_step: usize,
    cpu_row_step: usize,
    cpu_bar_w: usize,
    cpu_bar_max_h: usize,
    vmmu_show_devmem: usize,
    lora_box_w: usize,
    lora_box_h: usize,
    lora_col_step: usize,
    lora_row_step: usize,
    br_mode: String,
    osc_gain: f64,
    dock_text_y_offset: usize,
    tl_rgb: [u8; 3],
    tr_rgb: [u8; 3],
    bl_rgb: [u8; 3],
    br_rgb: [u8; 3],
}

impl UiConfig {
    fn load(path: &str) -> Self {
        let mut cfg = UiConfig {
            os_title: String::from("SOMA-OS V3.0 DYNAMIC KERNEL + DEVMEM BUS"),
            cpu_max_cores: 24,
            cpu_cols: 8,
            cpu_rows: 3,
            cpu_col_step: 35,
            cpu_row_step: 32,
            cpu_bar_w: 22,
            cpu_bar_max_h: 16,
            vmmu_show_devmem: 1,
            lora_box_w: 66,
            lora_box_h: 24,
            lora_col_step: 71,
            lora_row_step: 29,
            br_mode: String::from("SPLIT_OSC_BASH"),
            osc_gain: 22.0,
            dock_text_y_offset: 9,
            tl_rgb: [56, 189, 248],
            tr_rgb: [52, 211, 153],
            bl_rgb: [251, 191, 36],
            br_rgb: [244, 114, 182],
        };
        if let Ok(raw) = fs::read_to_string(path) {
            for line in raw.lines() {
                if let Some((k, v)) = line.split_once('=') {
                    let key = k.trim();
                    let val = v.trim();
                    match key {
                        "os_title" => cfg.os_title = val.to_uppercase(),
                        "cpu_max_cores" => cfg.cpu_max_cores = val.parse::<usize>().unwrap_or(24).max(24),
                        "cpu_cols" => cfg.cpu_cols = val.parse::<usize>().unwrap_or(8).clamp(4, 12),
                        "cpu_rows" => cfg.cpu_rows = val.parse::<usize>().unwrap_or(3).clamp(2, 6),
                        "cpu_col_step" => cfg.cpu_col_step = val.parse::<usize>().unwrap_or(35).clamp(20, 60),
                        "cpu_row_step" => cfg.cpu_row_step = val.parse::<usize>().unwrap_or(32).clamp(20, 50),
                        "cpu_bar_w" => cfg.cpu_bar_w = val.parse::<usize>().unwrap_or(22).clamp(10, 34),
                        "cpu_bar_max_h" => cfg.cpu_bar_max_h = val.parse::<usize>().unwrap_or(16).clamp(8, 28),
                        "vmmu_show_devmem" => cfg.vmmu_show_devmem = val.parse::<usize>().unwrap_or(1),
                        "lora_box_w" => cfg.lora_box_w = val.parse::<usize>().unwrap_or(66).clamp(50, 72),
                        "lora_box_h" => cfg.lora_box_h = val.parse::<usize>().unwrap_or(24).clamp(16, 30),
                        "lora_col_step" => cfg.lora_col_step = val.parse::<usize>().unwrap_or(71).clamp(56, 74),
                        "lora_row_step" => cfg.lora_row_step = val.parse::<usize>().unwrap_or(29).clamp(20, 34),
                        "br_mode" => cfg.br_mode = val.to_uppercase(),
                        "osc_gain" => cfg.osc_gain = val.parse::<f64>().unwrap_or(22.0).clamp(6.0, 38.0),
                        "dock_text_y_offset" => cfg.dock_text_y_offset = val.parse::<usize>().unwrap_or(9).clamp(4, 16),
                        "tl_border_r" => cfg.tl_rgb[0] = val.parse::<u8>().unwrap_or(56),
                        "tl_border_g" => cfg.tl_rgb[1] = val.parse::<u8>().unwrap_or(189),
                        "tl_border_b" => cfg.tl_rgb[2] = val.parse::<u8>().unwrap_or(248),
                        "tr_border_r" => cfg.tr_rgb[0] = val.parse::<u8>().unwrap_or(52),
                        "tr_border_g" => cfg.tr_rgb[1] = val.parse::<u8>().unwrap_or(211),
                        "tr_border_b" => cfg.tr_rgb[2] = val.parse::<u8>().unwrap_or(153),
                        "bl_border_r" => cfg.bl_rgb[0] = val.parse::<u8>().unwrap_or(251),
                        "bl_border_g" => cfg.bl_rgb[1] = val.parse::<u8>().unwrap_or(191),
                        "bl_border_b" => cfg.bl_rgb[2] = val.parse::<u8>().unwrap_or(36),
                        "br_border_r" => cfg.br_rgb[0] = val.parse::<u8>().unwrap_or(244),
                        "br_border_g" => cfg.br_rgb[1] = val.parse::<u8>().unwrap_or(114),
                        "br_border_b" => cfg.br_rgb[2] = val.parse::<u8>().unwrap_or(182),
                        _ => {}
                    }
                }
            }
        }
        cfg
    }
}

#[derive(Clone)]
struct DevMemChannel {
    ch: usize,
    seq: u32,
    reg_u64: u64,
    tag: String,
    payload: String,
}

fn read_devmem_channels(path: &str) -> Vec<DevMemChannel> {
    let mut out = Vec::with_capacity(8);
    let data = fs::read(path).unwrap_or_else(|_| vec![0u8; 65536]);
    for ch in 0..8 {
        let base = ch * 8192;
        if base + 256 <= data.len() {
            let seq = u32::from_le_bytes([data[base + 4], data[base + 5], data[base + 6], data[base + 7]]);
            let reg_u64 = u64::from_le_bytes([
                data[base + 8], data[base + 9], data[base + 10], data[base + 11],
                data[base + 12], data[base + 13], data[base + 14], data[base + 15],
            ]);
            let tag_end = data[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
            let tag = String::from_utf8_lossy(&data[base + 16..base + 16 + tag_end]).to_uppercase();
            let pay_end = data[base + 48..base + 240].iter().position(|&b| b == 0).unwrap_or(192);
            let payload = String::from_utf8_lossy(&data[base + 48..base + 48 + pay_end]).to_uppercase();
            out.push(DevMemChannel { ch, seq, reg_u64, tag, payload });
        } else {
            out.push(DevMemChannel { ch, seq: 0, reg_u64: 0, tag: format!("CH{}", ch), payload: String::new() });
        }
    }
    out
}

struct SysStats {
    load_1m: f64,
    load_5m: f64,
    mem_total_mb: u64,
    mem_pct: f64,
    uptime: u64,
    cpu_pcts: Vec<f64>,
}

fn get_stats(prev_totals: &mut Vec<u64>, prev_idles: &mut Vec<u64>) -> SysStats {
    let load_str = fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let loads: Vec<&str> = load_str.split_whitespace().collect();
    let l1 = loads.first().unwrap_or(&"0.5").parse::<f64>().unwrap_or(0.5);
    let l5 = loads.get(1).unwrap_or(&"0.4").parse::<f64>().unwrap_or(0.4);

    let mem_str = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total: u64 = 0;
    let mut avail: u64 = 0;
    for line in mem_str.lines() {
        if line.starts_with("MemTotal:") {
            total = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        } else if line.starts_with("MemAvailable:") {
            avail = line.split_whitespace().nth(1).unwrap_or("0").parse().unwrap_or(0);
        }
    }
    let mem_pct = if total > 0 { (1.0 - (avail as f64 / total as f64)) * 100.0 } else { 0.0 };

    let up_str = fs::read_to_string("/proc/uptime").unwrap_or_default();
    let uptime = up_str.split_whitespace().next().unwrap_or("0").parse::<f64>().unwrap_or(0.0) as u64;

    let stat_str = fs::read_to_string("/proc/stat").unwrap_or_default();
    let mut cur_totals = Vec::new();
    let mut cur_idles = Vec::new();
    for line in stat_str.lines() {
        if line.starts_with("cpu") && !line.starts_with("cpu ") {
            let parts: Vec<u64> = line.split_whitespace().skip(1).map(|s| s.parse().unwrap_or(0)).collect();
            let tot: u64 = parts.iter().sum();
            let idle: u64 = parts.get(3).copied().unwrap_or(0) + parts.get(4).copied().unwrap_or(0);
            cur_totals.push(tot);
            cur_idles.push(idle);
        }
    }

    let mut pcts = Vec::new();
    for i in 0..cur_totals.len() {
        if i < prev_totals.len() {
            let dt = cur_totals[i].saturating_sub(prev_totals[i]);
            let di = cur_idles[i].saturating_sub(prev_idles[i]);
            let busy = dt.saturating_sub(di);
            let p = if dt > 0 { (busy as f64 / dt as f64) * 100.0 } else { 8.0 };
            pcts.push(p);
        } else {
            pcts.push(12.0);
        }
    }
    *prev_totals = cur_totals;
    *prev_idles = cur_idles;
    while pcts.len() < 24 {
        pcts.push(10.0);
    }

    SysStats {
        load_1m: l1,
        load_5m: l5,
        mem_total_mb: total / 1024,
        mem_pct,
        uptime,
        cpu_pcts: pcts,
    }
}

fn main() {
    let fb_path = env::var("FISHBOWL_FB_PATH").unwrap_or_else(|_| "./dev/fb0_rgba.raw".to_string());
    let devmem_path = env::var("SOMA_DEVMEM_PATH").unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
    let cfg_path = env::var("SOMA_UI_CFG_PATH").unwrap_or_else(|_| "./etc/soma_ui_config.cfg".to_string());

    let mut buf = vec![0u8; 1024 * 768 * 4];
    let mut prev_totals: Vec<u64> = Vec::new();
    let mut prev_idles: Vec<u64> = Vec::new();
    let mut prev_seqs: [u32; 8] = [0; 8];
    let mut flash_ttl: [u8; 8] = [0; 8];
    let mut tick: u64 = 0;

    println!("SOMA-OS DYNAMIC COMPOSITOR v3.0 (1024x768 AI-NATIVE SUBSTRATE) ONLINE | FB={} | DEVMEM={} | CFG={}", fb_path, devmem_path, cfg_path);

    loop {
        tick += 1;
        let t0 = Instant::now();
        let cfg = UiConfig::load(&cfg_path);
        let stats = get_stats(&mut prev_totals, &mut prev_idles);
        let channels = read_devmem_channels(&devmem_path);

        for ch in 0..8 {
            if channels[ch].seq != prev_seqs[ch] {
                prev_seqs[ch] = channels[ch].seq;
                flash_ttl[ch] = 10;
            } else if flash_ttl[ch] > 0 {
                flash_ttl[ch] -= 1;
            }
        }

        // Clear background (1024x768)
        for i in (0..buf.len()).step_by(4) {
            buf[i] = 5;
            buf[i + 1] = 8;
            buf[i + 2] = 16;
            buf[i + 3] = 255;
        }

        // --- TOP AI-NATIVE MACHINE STATE BARCODE & HEADER (y: 0..42, w: 1024) ---
        soma_fill_rect(&mut buf, 0, 0, 1024, 42, [10, 18, 34]);
        soma_fill_rect(&mut buf, 0, 41, 1024, 1, [40, 80, 135]);
        // Machine-readable spatial state cells across top right (for VLM & AI direct state decoding without OCR)
        for bit in 0..32 {
            let active = ((tick >> (bit % 8)) & 1) == 1 || (bit < 8 && flash_ttl[bit] > 0);
            let c = if active { [52, 211, 153] } else { [22, 38, 62] };
            soma_fill_rect(&mut buf, 620 + bit * 12, 10, 9, 22, c);
        }
        let top_txt = format!("{} | 1024X768 | TICK:{} | UP:{}S", cfg.os_title, tick, stats.uptime);
        soma_draw_text(&mut buf, 14, 14, &top_txt, [215, 235, 255], 1);

        // --- QUADRANT TL (x:12, y:52, w:492, h:332): 24-CORE SPATIAL TENSOR MATRIX (8x3) ---
        soma_draw_window(&mut buf, 12, 52, 492, 332, "TL: 24-CORE TOPOLOGY TENSOR [8X3 LOCKED]", cfg.tl_rgb);
        let total_cores = cfg.cpu_max_cores.min(24);
        for i in 0..total_cores {
            let col = i % 8;
            let row = i / 8;
            let cx = 28 + col * 58;
            let cy = 84 + row * 94;
            let raw_pct = stats.cpu_pcts.get(i).copied().unwrap_or(10.0);
            let wave = (((tick as f64 * 0.25) + (i as f64 * 0.55)).sin() + 1.0) * 16.0;
            let eff_pct = (raw_pct + wave).clamp(8.0, 100.0);
            let bar_h = ((eff_pct / 100.0) * 62.0).round() as usize;
            let bar_h = bar_h.clamp(4, 62);

            // Outer core slot frame
            soma_fill_rect(&mut buf, cx, cy, 46, 74, [16, 26, 44]);
            let bar_rgb = if eff_pct > 70.0 {
                [248, 113, 113]
            } else if eff_pct > 42.0 {
                [251, 191, 36]
            } else {
                cfg.tl_rgb
            };
            soma_fill_rect(&mut buf, cx + 4, cy + 8 + (62 - bar_h), 38, bar_h, bar_rgb);
            soma_draw_text(&mut buf, cx + 10, cy + 2, &format!("C{:02}", i), [180, 200, 225], 1);
        }

        // --- QUADRANT TR (x:520, y:52, w:492, h:332): 64KB /dev/mem 8-CHANNEL CONTROL BUS ---
        soma_draw_window(&mut buf, 520, 52, 492, 332, "TR: DEVMEM 8-CHANNEL BUS [64KB MMAP CONTROL]", cfg.tr_rgb);
        // RAM Arena Bar
        soma_fill_rect(&mut buf, 536, 80, 460, 18, [20, 34, 52]);
        let used_w = ((stats.mem_pct / 100.0) * 460.0).round() as usize;
        soma_fill_rect(&mut buf, 536, 80, used_w.clamp(8, 460), 18, cfg.tr_rgb);

        // 8 Channels x 64-bit Bitplane Spatial Matrix (Pure post-text visual register encoding + hex)
        for ch in 0..8 {
            let ry = 108 + ch * 32;
            let cinfo = &channels[ch];
            let is_flashing = flash_ttl[ch] > 0;
            let row_bg = if is_flashing { [26, 78, 58] } else { [15, 25, 42] };
            soma_fill_rect(&mut buf, 536, ry, 460, 26, row_bg);
            let led_c = if is_flashing { [251, 191, 36] } else { cfg.tr_rgb };
            soma_fill_rect(&mut buf, 540, ry + 4, 8, 18, led_c);
            soma_draw_text(&mut buf, 554, ry + 9, &format!("CH{} S{:03}", ch, cinfo.seq % 1000), [235, 245, 255], 1);

            // Render 32-bit spatial bit-cells of the 64-bit channel register so AI reads binary state directly from pixels!
            for b in 0..32 {
                let bit_on = ((cinfo.reg_u64 >> b) & 1) == 1;
                let cell_c = if bit_on {
                    if is_flashing { [251, 191, 36] } else { [56, 189, 248] }
                } else {
                    [24, 38, 60]
                };
                soma_fill_rect(&mut buf, 620 + b * 8, ry + 5, 6, 16, cell_c);
            }
            let short_tag: String = cinfo.tag.chars().skip(4).take(16).collect();
            soma_draw_text(&mut buf, 885, ry + 9, &short_tag, led_c, 1);
        }

        // --- QUADRANT BL (x:12, y:396, w:492, h:322): 16-LORA HYPERCUBE & SYNAPTIC TOPOLOGY ---
        soma_draw_window(&mut buf, 12, 396, 492, 322, "BL: 16-LORA HYPERCUBE TOPOLOGY [CH2]", cfg.bl_rgb);
        let ch2_mask = channels[2].reg_u64;
        for i in 0..16 {
            let col = i % 4;
            let row = i / 4;
            let x = 28 + col * 116;
            let y = 428 + row * 68;
            let phase = (tick + (i as u64 * 3) + (ch2_mask & 0xF)) % 15;
            let (st_txt, bg_rgb, fg_rgb) = if phase >= 12 {
                ("FIRE", [239, 68, 68], [255, 255, 255])
            } else if phase >= 6 {
                ("ACTV", [245, 158, 11], [10, 15, 25])
            } else {
                ("NREM", [32, 48, 72], [170, 190, 215])
            };
            soma_fill_rect(&mut buf, x, y, 102, 54, bg_rgb);
            // Spatial energy bar inside each expert tile
            let e_w = ((phase as usize + 1) * 6).min(90);
            soma_fill_rect(&mut buf, x + 6, y + 34, e_w, 12, [12, 18, 30]);
            soma_draw_text(&mut buf, x + 10, y + 12, &format!("E{:02} {}", i, st_txt), fg_rgb, 1);
        }

        // --- QUADRANT BR (x:520, y:396, w:492, h:322): AI CONTROL BUS, OS APPS & BASH IPC [CH3/CH4] ---
        soma_draw_window(&mut buf, 520, 396, 492, 322, "BR: AI ENV CONTROL + OS BASH/APPS [CH3/CH4]", cfg.br_rgb);
        // Upper Zone: Real-time Syscall & Channel Phase Field (y: 424..550)
        soma_fill_rect(&mut buf, 536, 424, 460, 126, [8, 13, 24]);
        for x_off in 0..452 {
            let t = (tick as f64 * 0.16) + (x_off as f64 * 0.035);
            let y1 = (487.0 + (t.sin() * (cfg.osc_gain * 1.8))).round() as usize;
            let y2 = (487.0 + ((t * 1.4).cos() * (cfg.osc_gain * 1.3))).round() as usize;
            soma_fill_rect(&mut buf, 540 + x_off, y1.clamp(428, 544), 2, 3, [56, 189, 248]);
            soma_fill_rect(&mut buf, 540 + x_off, y2.clamp(428, 544), 2, 3, [52, 211, 153]);
        }
        // Lower Zone: Internal OS Control State Blocks & IPC Stream (y: 560..704)
        soma_fill_rect(&mut buf, 536, 560, 460, 144, [6, 15, 24]);
        for ch_idx in 3..8 {
            let by = 568 + (ch_idx - 3) * 26;
            let cinfo = &channels[ch_idx];
            let status_c = if flash_ttl[ch_idx] > 0 { [251, 191, 36] } else { [52, 211, 153] };
            soma_fill_rect(&mut buf, 544, by, 14, 18, status_c);
            let pay_snip: String = cinfo.payload.chars().take(66).collect();
            soma_draw_text(&mut buf, 566, by + 5, &format!("CH{}: {}", ch_idx, pay_snip), [220, 238, 255], 1);
        }

        // --- BOTTOM AI CONTROL DOCK (y: 728..768, w: 1024) ---
        soma_fill_rect(&mut buf, 0, 728, 1024, 40, [10, 18, 34]);
        soma_fill_rect(&mut buf, 0, 728, 1024, 1, [40, 80, 135]);
        let frame_us = t0.elapsed().as_micros();
        let dock_str = format!(
            "[SOMA-OS V3 1024X768] [24-CORE TENSOR:LOCKED] [DEVMEM:64KB 8-CH] [TOOLCHAIN:/BIN/RUSTC] [FRAME:{}US]",
            frame_us
        );
        soma_draw_text(&mut buf, 14, 728 + cfg.dock_text_y_offset + 6, &dock_str, [52, 211, 153], 1);

        if let Ok(mut f) = fs::OpenOptions::new().write(true).open(&fb_path) {
            let _ = f.write_all(&buf);
        }
        thread::sleep(Duration::from_millis(80));
    }
}

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
        '>' => [0b10000, 0b01000, 0b00100, 0b00010, 0b00100, 0b01000, 0b10000],
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

pub fn soma_fill_rect(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, rgb: [u8; 3]) {
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

pub fn soma_draw_window(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, title: &str, border_rgb: [u8; 3]) {
    soma_fill_rect(buf, x, y, w, h, [11, 17, 30]);
    soma_fill_rect(buf, x, y, w, 20, [18, 30, 52]);
    soma_fill_rect(buf, x, y, w, 1, border_rgb);
    soma_fill_rect(buf, x, (y + 20).min(767), w, 1, [30, 48, 78]);
    soma_fill_rect(buf, x, (y + h.saturating_sub(1)).min(767), w, 1, border_rgb);
    soma_fill_rect(buf, x, y, 1, h, border_rgb);
    soma_fill_rect(buf, (x + w.saturating_sub(1)).min(1023), y, 1, h, border_rgb);
    soma_draw_text(buf, x + 8, y + 6, title, border_rgb, 1);
}
