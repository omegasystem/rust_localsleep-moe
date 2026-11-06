use std::env;
use std::fs;

const W: usize = 640;
const H: usize = 360;

fn set_px(buf: &mut [u8], x: i32, y: i32, r: u8, g: u8, b: u8) {
    if x >= 0 && x < W as i32 && y >= 0 && y < H as i32 {
        let idx = ((y as usize) * W + (x as usize)) * 4;
        buf[idx] = r;
        buf[idx + 1] = g;
        buf[idx + 2] = b;
        buf[idx + 3] = 255;
    }
}

fn main() {
    let fb_path = env::var("FISHBOWL_FB_PATH").unwrap_or_else(|_| "./dev/fb0_rgba.raw".to_string());
    let loadavg = fs::read_to_string("/proc/loadavg").unwrap_or_else(|_| "0.42 0.35 0.28 1/256 1".to_string());
    let meminfo = fs::read_to_string("/proc/meminfo").unwrap_or_default();

    let mut mem_total_kb: u64 = 16_000_000;
    let mut mem_avail_kb: u64 = 10_000_000;
    for line in meminfo.lines() {
        if line.starts_with("MemTotal:") {
            mem_total_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(16_000_000);
        } else if line.starts_with("MemAvailable:") {
            mem_avail_kb = line.split_whitespace().nth(1).and_then(|s| s.parse().ok()).unwrap_or(10_000_000);
        }
    }
    let used_ratio = 1.0 - (mem_avail_kb as f64 / mem_total_kb.max(1) as f64);

    let mut buf = vec![0u8; W * H * 4];

    // 1. Tactical Dark Background + Cybernetic Radar Grid
    for y in 0..H {
        for x in 0..W {
            let idx = (y * W + x) * 4;
            let grid = (x % 32 == 0) || (y % 32 == 0);
            buf[idx] = if grid { 18 } else { 7 };
            buf[idx + 1] = if grid { 36 } else { 12 };
            buf[idx + 2] = if grid { 64 } else { 24 };
            buf[idx + 3] = 255;
        }
    }

    // 2. Draw 16-Expert Neural Hypercube Nodes & Rings around center (320, 165)
    let cx = 320.0f64;
    let cy = 165.0f64;
    for r_ring in [55.0f64, 95.0f64, 130.0f64] {
        for deg in 0..720 {
            let rad = (deg as f64) * 0.5 * std::f64::consts::PI / 180.0;
            let px = (cx + r_ring * rad.cos()) as i32;
            let py = (cy + r_ring * rad.sin()) as i32;
            set_px(&mut buf, px, py, 30, 120, 180);
        }
    }

    // 3. Draw Sin/Cos Neural Waveform across the Framebuffer
    for x in 20..(W - 20) {
        let t = (x as f64) * 0.045;
        let wy1 = (cy + (t.sin() * 42.0) + ((t * 2.7).cos() * 18.0)) as i32;
        let wy2 = (cy + ((t * 1.3).cos() * 55.0 * used_ratio.max(0.25))) as i32;
        for dy in -2..=2 {
            set_px(&mut buf, x as i32, wy1 + dy, 56, 189, 248);  // Cyan wave
            set_px(&mut buf, x as i32, wy2 + dy, 52, 211, 153);  // Emerald wave
        }
    }

    // 4. Draw Memory Usage Bar at bottom (y: 315..340)
    let bar_x0 = 40;
    let bar_w = 560;
    let filled_w = ((bar_w as f64) * used_ratio.clamp(0.05, 1.0)) as usize;
    for y in 318..338 {
        for x in 0..bar_w {
            if x < filled_w {
                set_px(&mut buf, (bar_x0 + x) as i32, y, 251, 191, 36);
            } else {
                set_px(&mut buf, (bar_x0 + x) as i32, y, 25, 38, 58);
            }
        }
    }

    fs::write(&fb_path, &buf).expect("Failed to write framebuffer");
    println!("=== [SOMA-SEED-PULSE ORGAN TELEMETRY] ===");
    println!("Linux LoadAvg : {}", loadavg.trim());
    println!("Memory Used   : {:.1}% ({} MB / {} MB)", used_ratio * 100.0, (mem_total_kb - mem_avail_kb) / 1024, mem_total_kb / 1024);
    println!("Framebuffer   : Wrote {}x{} RGBA8888 ({} bytes) -> {}", W, H, buf.len(), fb_path);
}
