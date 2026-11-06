use std::env;
use std::fs::{File, read_to_string};
use std::io::{Write, Read};
use std::time::{SystemTime, UNIX_EPOCH};

const WIDTH: usize = 640;
const HEIGHT: usize = 360;
const CHANNELS: usize = 4;
const FB_SIZE: usize = WIDTH * HEIGHT * CHANNELS;

struct Telemetry {
    load_1: f32,
    mem_total: u64,
    mem_free: u64,
    mem_used_pct: f32,
}

fn main() {
    let telemetry = collect_telemetry();
    let mut buffer = vec![0u8; FB_SIZE];

    // Background: Deep Void
    for i in (0..FB_SIZE).step_by(4) {
        buffer[i] = 10;     // R
        buffer[i+1] = 12;   // G
        buffer[i+2] = 20;   // B
        buffer[i+3] = 255;  // A
    }

    // Draw CPU Bar (Top)
    draw_bar(&mut buffer, 50, 40, 540, 20, telemetry.load_1 / 4.0, [0, 255, 150, 255]);
    
    // Draw RAM Bar (Below CPU)
    draw_bar(&mut buffer, 50, 70, 540, 20, telemetry.mem_used_pct / 100.0, [255, 50, 100, 255]);

    // Draw Neural Hypercube Radar (Center)
    let cx = WIDTH / 2;
    let cy = HEIGHT / 2 + 40;
    let radius = 80;
    let points = 50;

    for i in 0..points {
        let angle = (i as f32) * (2.0 * std::f32::consts::PI / points as f32);
        let next_angle = ((i + 1) as f32) * (2.0 * std::f32::consts::PI / points as f32);
        
        // Dynamic scaling based on load and memory
        let scale = 0.5 + (telemetry.load_1 * 0.1) + (telemetry.mem_used_pct * 0.002);
        let r_val = (radius as f32 * scale).clamp(20.0, radius as f32);
        
        let x1 = cx as f32 + r_val * angle.cos();
        let y1 = cy as f32 + r_val * angle.sin();
        let x2 = cx as f32 + r_val * next_angle.cos();
        let y2 = cy as f32 + r_val * next_angle.sin();

        draw_line(&mut buffer, x1, y1, x2, y2, [0, 200, 255, 255]);
    }

    // Draw Radar Spokes
    for i in 0..points {
        let angle = (i as f32) * (2.0 * std::f32::consts::PI / points as f32);
        let x = cx as f32 + (radius as f32) * angle.cos();
        let y = cy as f32 + (radius as f32) * angle.sin();
        draw_line(&mut buffer, cx as f32, cy as f32, x, y, [50, 50, 80, 255]);
    }

    // Write to Framebuffer
    let fb_path = env::var("FISHBOWL_FB_PATH")
        .unwrap_or_else(|_| "./fishbowl/rootfs/dev/fb0_rgba.raw".to_string());
    
    if let Ok(mut file) = File::create(fb_path) {
        let _ = file.write_all(&buffer);
    }

    // Telemetry Report
    println!("--- CYBER_PULSE_ORGAN TELEMETRY ---");
    println!("Timestamp: {:?}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs());
    println!("CPU Load (1m): {:.2}", telemetry.load_1);
    println!("RAM Total: {} KB", telemetry.mem_total);
    println!("RAM Used: {:.2}%", telemetry.mem_used_pct);
    println!("Status: OPERATIONAL");
    println!("----------------------------------");
}

fn collect_telemetry() -> Telemetry {
    let load_str = read_to_string("/proc/loadavg").unwrap_or_else(|_| "0.00 0.00 0.00 1/1".to_string());
    let load_1 = load_str.split_whitespace().next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(0.0);

    let mem_str = read_to_string("/proc/meminfo").unwrap_or_else(|_| "".to_string());
    let mut total = 0u64;
    let mut free = 0u64;
    for line in mem_str.lines() {
        if line.starts_with("MemTotal:") {
            total = parse_mem_line(line);
        } else if line.starts_with("MemAvailable:") {
            free = parse_mem_line(line);
        }
    }
    
    let used_pct = if total > 0 {
        ((total - free) as f32 / total as f32) * 100.0
    } else { 0.0 };

    Telemetry { load_1, mem_total: total, mem_free: free, mem_used_pct: used_pct }
}

fn parse_mem_line(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

fn draw_bar(buf: &mut [u8], x: usize, y: usize, w: usize, h: usize, pct: f32, color: [u8; 4]) {
    let fill_w = (w as f32 * pct.clamp(0.0, 1.0)) as usize;
    for row in 0..h {
        for col in 0..w {
            let px = x + col;
            let py = y + row;
            if px < WIDTH && py < HEIGHT {
                let idx = (py * WIDTH + px) * CHANNELS;
                if col < fill_w {
                    buf[idx] = color[0];
                    buf[idx+1] = color[1];
                    buf[idx+2] = color[2];
                    buf[idx+3] = color[3];
                } else {
                    buf[idx] = 40;
                    buf[idx+1] = 40;
                    buf[idx+2] = 40;
                    buf[idx+3] = 255;
                }
            }
        }
    }
}

fn draw_line(buf: &mut [u8], x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 4]) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()) as usize;
    if steps == 0 { return; }
    
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let px = (x0 + t * dx) as usize;
        let py = (y0 + t * dy) as usize;
        if px < WIDTH && py < HEIGHT {
            let idx = (py * WIDTH + px) * CHANNELS;
            buf[idx] = color[0];
            buf[idx+1] = color[1];
            buf[idx+2] = color[2];
            buf[idx+3] = color[3];
        }
    }
}