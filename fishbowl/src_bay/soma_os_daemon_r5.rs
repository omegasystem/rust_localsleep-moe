#![allow(unused_imports, unused_variables, dead_code, unused_mut, unused_parens)]
use std::env;
use std::fs;
use std::io::{Read, Write};
use std::thread;
use std::time::{Duration, Instant};

struct SysStats {
    load_1m: f64,
    load_5m: f64,
    load_15m: f64,
    mem_total: u64,
    mem_avail: u64,
    mem_pct: f64,
    uptime: u64,
    cpu_deltas: Vec<f64>,
}

fn get_stats(prev_cpu: &mut Vec<u64>) -> SysStats {
    // LoadAvg
    let load_str = fs::read_to_string("/proc/loadavg").unwrap_or_default();
    let loads: Vec<&str> = load_str.split_whitespace().collect();
    let l1 = loads.get(0).unwrap_or(&"0.0").parse().unwrap_or(0.0);
    let l5 = loads.get(1).unwrap_or(&"0.0").parse().unwrap_or(0.0);
    let l15 = loads.get(2).unwrap_or(&"0.0").parse().unwrap_or(0.0);

    // MemInfo
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

    // Uptime
    let up_str = fs::read_to_string("/proc/uptime").unwrap_or_default();
    let uptime = up_str.split_whitespace().next().unwrap_or("0").parse::<f64>().unwrap_or(0.0) as u64;

    // CPU Stat (Delta calculation)
    let stat_str = fs::read_to_string("/proc/stat").unwrap_or_default();
    let mut current_cpu = Vec::new();
    for line in stat_str.lines() {
        if line.starts_with("cpu") && !line.starts_with("cpu ") { // Match cpu0, cpu1...
            let parts: Vec<u64> = line.split_whitespace().skip(1).map(|s| s.parse().unwrap_or(0)).collect();
            let total: u64 = parts.iter().sum();
            current_cpu.push(total);
        }
    }
    
    let mut deltas = Vec::new();
    for i in 0..current_cpu.len() {
        if i < prev_cpu.len() {
            let diff = current_cpu[i].saturating_sub(prev_cpu[i]);
            deltas.push(diff as f64);
        } else {
            deltas.push(0.0);
        }
    }
    *prev_cpu = current_cpu;

    SysStats { load_1m: l1, load_5m: l5, load_15m: l15, mem_total: total / 1024, mem_avail: avail / 1024, mem_pct, uptime, cpu_deltas: deltas }
}

fn main() {
    let fb_path = env::var("FISHBOWL_FB_PATH").expect("FISHBOWL_FB_PATH not set");
    let mut buf = vec![0u8; 640 * 360 * 4];
    let mut prev_cpu_ticks = Vec::new();
    let mut tick: u64 = 0;
    let start_time = Instant::now();

    println!("SOMA-OS KERNEL DAEMON v2.5 INITIALIZING...");
    println!("TARGET: {} | RESOLUTION: 640x360 | MODE: PERSISTENT_DAEMON", fb_path);

    loop {
        tick += 1;
        let stats = get_stats(&mut prev_cpu_ticks);
        let frame_start = Instant::now();

        // Clear Buffer (Black)
        buf.fill(0);

        // --- TOP BAR ---
        soma_fill_rect(&mut buf, 0, 0, 640, 24, [10, 20, 40]);
        soma_draw_text(&mut buf, 10, 6, &format!("SOMA-OS KERNEL DAEMON v2.5 | TICK: {} | UPTIME: {}s", tick, stats.uptime), [200, 200, 200], 1);
        let heartbeat_color = if tick % 10 < 5 { [0, 255, 0] } else { [0, 100, 0] };
        soma_fill_rect(&mut buf, 618, 6, 16, 12, heartbeat_color);

        // --- QUADRANT TL: CPU CORE MATRIX ---
        soma_draw_window(&mut buf, 8, 28, 308, 146, "CPU CORE MATRIX [LIVE /PROC/STAT]", [56, 189, 248]);
        soma_draw_text(&mut buf, 20, 50, &format!("LOAD: {:.2} | CORES: {}", stats.load_1m, stats.cpu_deltas.len()), [255, 255, 255], 1);
        for (i, &delta) in stats.cpu_deltas.iter().enumerate().take(16) {
            let col = i % 8;
            let row = i / 8;
            let x = 20 + col * 35;
            let y = 70 + row * 40;
            let height = (delta.min(30.0) as usize).max(2);
            let color = if delta > 20.0 { [255, 50, 50] } else if delta > 10.0 { [255, 200, 50] } else { [50, 200, 255] };
            soma_fill_rect(&mut buf, x, y + 30 - height, 20, height, color);
            soma_draw_text(&mut buf, x, y, &format!("C{:02}", i), [180, 180, 180], 1);
        }

        // --- QUADRANT TR: VMMU SLAB & MEMORY ARENA ---
        soma_draw_window(&mut buf, 324, 28, 308, 146, "VMMU SLAB & MEMORY ARENA", [52, 211, 153]);
        // Memory Bar
        soma_fill_rect(&mut buf, 336, 52, 270, 15, [40, 40, 40]);
        let used_w = (270.0 * (stats.mem_pct / 100.0)) as usize;
        soma_fill_rect(&mut buf, 336, 52, used_w, 15, [52, 211, 153]);
        soma_draw_text(&mut buf, 336, 70, &format!("MEM: {:.1}% | TOTAL: {}MB", stats.mem_pct, stats.mem_total), [255, 255, 255], 1);
        // Page-Frame Grid
        for row in 0..4 {
            for col in 0..8 {
                let x = 336 + col * 32;
                let y = 96 + row * 15;
                let phase = (tick + (row * 8 + col) as u64) % 20;
                let color = if phase < 5 { [100, 255, 100] } else if phase < 12 { [40, 180, 40] } else { [20, 100, 20] };
                soma_fill_rect(&mut buf, x, y, 28, 12, color);
            }
        }

        // --- QUADRANT BL: 16-LORA HYPERCUBE ROUTER ---
        soma_draw_window(&mut buf, 8, 180, 308, 148, "16-LORA HYPERCUBE ROUTER", [251, 191, 36]);
        for i in 0..16 {
            let col = i % 4;
            let row = i / 4;
            let x = 20 + col * 70;
            let y = 204 + row * 32;
            let state_val = (tick + i as u64) % 13;
            let (label, color) = if state_val > 10 { ("FIRE", [255, 69, 0]) } else if state_val > 4 { ("ACTIVE", [255, 215, 0]) } else { ("NREM", [100, 100, 100]) };
            soma_fill_rect(&mut buf, x, y, 60, 25, color);
            soma_draw_text(&mut buf, x + 5, y + 5, &format!("E{:02} {}", i, label), [0, 0, 0], 1);
        }

        // --- QUADRANT BR: CHRONOS SYSCALL OSCILLOSCOPE ---
        soma_draw_window(&mut buf, 324, 180, 308, 148, "CHRONOS SYSCALL OSCILLOSCOPE", [244, 114, 182]);
        let pad_x = 346; // 10px padding
        let pad_y = 214;
        let scope_w = 250;
        let scope_h = 90;
        soma_fill_rect(&mut buf, pad_x, pad_y, scope_w, scope_h, [20, 10, 20]);
        for x_off in 0..scope_w {
            let t = (tick as f64 * 0.1) + (x_off as f64 * 0.05);
            let y1 = (t.sin() * 20.0 + (stats.load_1m * 5.0)) as i32;
            let y2 = (t.cos() * 20.0 - (stats.load_5m * 5.0)) as i32;
            soma_fill_rect(&mut buf, pad_x + x_off, (pad_y + scope_h/2) as usize + y1 as usize, 1, 1, [0, 255, 255]);
            soma_fill_rect(&mut buf, pad_x + x_off, (pad_y + scope_h/2) as usize + y2 as usize, 1, 1, [50, 255, 150]);
        }
        let latency = frame_start.elapsed().as_micros();
        soma_draw_text(&mut buf, pad_x, pad_y + scope_h + 5, &format!("LATENCY: {}us | FPS: {:.1}", latency, 1000000.0 / latency as f64), [200, 200, 200], 1);

        // --- BOTTOM DOCK ---
        soma_fill_rect(&mut buf, 0, 334, 640, 26, [20, 20, 30]);
        soma_draw_text(&mut buf, 10, 342, &format!("[PID: DAEMON ACTIVE] [FB: 640X360@12.5FPS] [MEM: {:.1}%]", stats.mem_pct), [150, 150, 150], 1);

        // Write to Framebuffer
        if let Ok(mut file) = fs::OpenOptions::new().write(true).open(&fb_path) {
            let _ = file.write_all(&buf);
        }

        thread::sleep(Duration::from_millis(80));
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
