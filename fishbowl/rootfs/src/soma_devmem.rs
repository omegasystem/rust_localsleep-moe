#![allow(unused_imports, unused_variables, dead_code, unused_mut)]
use std::env;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};

const CHANNELS: usize = 8;
const CH_SIZE: usize = 8192;

fn parse_u64_auto(s: &str) -> u64 {
    let clean = s.trim().replace('_', "");
    if let Some(hex) = clean.strip_prefix("0x").or_else(|| clean.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        clean.parse::<u64>().unwrap_or(0)
    }
}

fn main() {
    let devmem_path = env::var("SOMA_DEVMEM_PATH")
        .unwrap_or_else(|_| "./dev/mem_channels/devmem_arena.bin".to_string());
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("status");

    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&devmem_path)
        .expect("Failed to open SOMA_DEVMEM_PATH");

    let mut arena = vec![0u8; CHANNELS * CH_SIZE];
    let _ = file.read(&mut arena);

    if cmd == "poke" && args.len() >= 4 {
        let ch = args[2].parse::<usize>().unwrap_or(4).min(CHANNELS - 1);
        let reg_val = parse_u64_auto(&args[3]);
        let tag = args.get(4).map(|s| s.as_str()).unwrap_or("CH_USER_POKE");
        let payload = if args.len() >= 6 { args[5..].join(" ") } else { String::from("OK") };

        let base = ch * CH_SIZE;
        arena[base..base + 4].copy_from_slice(b"SOMA");
        let seq = u32::from_le_bytes([arena[base + 4], arena[base + 5], arena[base + 6], arena[base + 7]]).wrapping_add(1);
        arena[base + 4..base + 8].copy_from_slice(&seq.to_le_bytes());
        arena[base + 8..base + 16].copy_from_slice(&reg_val.to_le_bytes());

        let mut tag_buf = [0u8; 32];
        let tb = tag.as_bytes();
        let tlen = tb.len().min(31);
        tag_buf[..tlen].copy_from_slice(&tb[..tlen]);
        arena[base + 16..base + 48].copy_from_slice(&tag_buf);

        for b in &mut arena[base + 48..base + CH_SIZE] {
            *b = 0;
        }
        let pb = payload.as_bytes();
        let plen = pb.len().min(CH_SIZE - 49);
        arena[base + 48..base + 48 + plen].copy_from_slice(&pb[..plen]);

        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&arena).unwrap();
        println!("[SOMA_DEVMEM POKE] CH{} (0x{:04X}) -> SEQ={} REG=0x{:016X} TAG='{}' PAYLOAD='{}'", ch, base, seq, reg_val, tag, payload);
    } else if cmd == "peek" && args.len() >= 3 {
        let ch = args[2].parse::<usize>().unwrap_or(0).min(CHANNELS - 1);
        let base = ch * CH_SIZE;
        let seq = u32::from_le_bytes([arena[base + 4], arena[base + 5], arena[base + 6], arena[base + 7]]);
        let reg = u64::from_le_bytes([
            arena[base + 8], arena[base + 9], arena[base + 10], arena[base + 11],
            arena[base + 12], arena[base + 13], arena[base + 14], arena[base + 15],
        ]);
        let tag_end = arena[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
        let tag = String::from_utf8_lossy(&arena[base + 16..base + 16 + tag_end]);
        let pay_end = arena[base + 48..base + 304].iter().position(|&b| b == 0).unwrap_or(256);
        let pay = String::from_utf8_lossy(&arena[base + 48..base + 48 + pay_end]);
        println!("CH{} @ 0x{:04X} | SEQ={} | REG=0x{:016X} | TAG={} | PAYLOAD={}", ch, base, seq, reg, tag, pay);
    } else {
        println!("=== [SOMA-OS /dev/mem 8-CHANNEL SHARED ARENA (64KB)] ===");
        for ch in 0..CHANNELS {
            let base = ch * CH_SIZE;
            let seq = u32::from_le_bytes([arena[base + 4], arena[base + 5], arena[base + 6], arena[base + 7]]);
            let reg = u64::from_le_bytes([
                arena[base + 8], arena[base + 9], arena[base + 10], arena[base + 11],
                arena[base + 12], arena[base + 13], arena[base + 14], arena[base + 15],
            ]);
            let tag_end = arena[base + 16..base + 48].iter().position(|&b| b == 0).unwrap_or(32);
            let tag = String::from_utf8_lossy(&arena[base + 16..base + 16 + tag_end]);
            let pay_end = arena[base + 48..base + 128].iter().position(|&b| b == 0).unwrap_or(80);
            let pay = String::from_utf8_lossy(&arena[base + 48..base + 48 + pay_end]);
            println!("  CH{} [0x{:04X}] SEQ={:<4} REG=0x{:016X} {:<18} | {}", ch, base, seq, reg, tag, pay);
        }
    }
}
