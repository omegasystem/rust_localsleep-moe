# 🐈‍⬛ LocalSleep-MoE × TensorRT-Edge-LLM 推論優化見證總表 (Optimization Witness Log)
> 自動記錄每一次架構重構、CUDA/Rust 算子優化與推論壓測的完整遙測數據 (`🏁 [Hybrid Turn Complete]` & `🦀 [Rust MoE Telemetry]`)。
## 📊 歷次推論效能演進總覽
| 時間戳 | Session ID | 階段說明 | Tokens (Chars) | 耗時 | 速度 (TPS) | TTFT | Top-2 專家 (熵 $H$) | NREM 2:4 剪枝 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `2026-10-19T02:08:40` | `thor_31b_20261019_020825` | Stage 1 - Initial Handshake Probe | `9` (`9`) | `13.83s` | **`0.65 tps`** | `845.8ms` | `[0, 1]` (`H=0.693`) | `0` |
| `2026-10-19T02:10:26` | `thor_31b_20261019_021009` | Stage 2 - Pre-Fix C++ StreamChunk::text UTF-8 Split Abort | `20` (`22`) | `16.45s` | **`1.22 tps`** | `1227.0ms` | `[1, 0]` (`H=1.210`) | `3` |
| `2026-10-19T02:35:40` | `thor_31b_20261019_023324` | Stage 3 - First UTF-8 Bypass (errors='ignore') + Full Rust Zero-GIL MoE 2:4 Pruning | `1500` (`3343`) | `135.43s` | **`11.08 tps`** | `1243.5ms` | `[5, 4]` (`H=1.250`) | `220` |
| `2026-10-19T02:49:40` | `thor_31b_20261019_024708` | Stage 4 - Lossless Incremental UTF-8 Decoder (codecs.getincrementaldecoder) Run #1 | `1755` (`4070`) | `151.30s` | **`11.60 tps`** | `1228.3ms` | `[7, 0]` (`H=1.245`) | `262` |
| `2026-10-19T02:53:15` | `thor_31b_20261019_025043` | Stage 5 - Lossless Incremental UTF-8 Decoder + Rust Zero-GIL MoE Witness Run #2 | `1755` (`4070`) | `151.30s` | **`11.60 tps`** | `1227.2ms` | `[7, 0]` (`H=1.245`) | `256` |

---

## 📜 歷次推論終端機完整遙測紀錄 (Terminal Telemetry Banners)

### 🔹 Stage 1 - Initial Handshake Probe (`thor_31b_20261019_020825`)
- **備註**: Initial connectivity check between Host RTX 5080 and AGX Thor :8000 (gemma-4-31B-it MTP).
```text
🏁 [Hybrid Turn Complete] Tokens: 9 | Elapsed: 13.83s (0.65 tps) | TTFT: 845.8ms | Top-2: [0, 1] (H=0.693)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.14,Tok=9,Sp=0%,Slp=0) | E1:🟢Active(F=0.05,Tok=9,Sp=0%,Slp=0)
```

### 🔹 Stage 2 - Pre-Fix C++ StreamChunk::text UTF-8 Split Abort (`thor_31b_20261019_021009`)
- **備註**: Diagnosed NVIDIA TensorRT-Edge-LLM v0.11.0 bug in edgellm_pybind.cpp:995 & engine.py:1746.
```text
🏁 [Hybrid Turn Complete] Tokens: 20 | Elapsed: 16.45s (1.22 tps) | TTFT: 1227.0ms | Top-2: [1, 0] (H=1.210)
🦀 [Rust MoE Telemetry (E0..E3)]: Aborted at token 20 due to pybind11 StreamChunk::text UTF-8 split across MTP steps (pos 6-7).
```

### 🔹 Stage 3 - First UTF-8 Bypass (errors='ignore') + Full Rust Zero-GIL MoE 2:4 Pruning (`thor_31b_20261019_023324`)
- **備註**: Throughput jumped 9.1x to 11.08 tps; NREM sleep + 2:4 Fisher pruning verified (Sp=50%), but errors='ignore' dropped split CJK chars at MTP boundaries.
```text
🏁 [Hybrid Turn Complete] Tokens: 1500 | Elapsed: 135.43s (11.08 tps) | TTFT: 1243.5ms | Top-2: [5, 4] (H=1.250)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=1.30,Tok=409,Sp=50%,Slp=30) | E1:💤NREM(F=0.88,Tok=402,Sp=50%,Slp=30) | E2:🟡Fatigued(F=1.38,Tok=365,Sp=50%,Slp=25) | E3:🟢Active(F=0.43,Tok=376,Sp=50%,Slp=26)
```

### 🔹 Stage 4 - Lossless Incremental UTF-8 Decoder (codecs.getincrementaldecoder) Run #1 (`thor_31b_20261019_024708`)
- **備註**: Replaced errors='ignore' with stateful codecs.getincrementaldecoder('utf-8'); 0 dropped CJK characters, speed increased to 11.60 tps.
```text
🏁 [Hybrid Turn Complete] Tokens: 1755 | Elapsed: 151.30s (11.60 tps) | TTFT: 1228.3ms | Top-2: [7, 0] (H=1.245)
🦀 [Rust MoE Telemetry (E0..E3)]: 100% lossless UTF-8 reassembly across MTP chunks (4070 chars vs 3343 chars); 262 NREM 2:4 pruning cycles.
```

### 🔹 Stage 5 - Lossless Incremental UTF-8 Decoder + Rust Zero-GIL MoE Witness Run #2 (`thor_31b_20261019_025043`)
- **備註**: Confirmed reproducible 11.60 tps & 1227.2ms TTFT with complete mathematical & architectural output.
```text
🏁 [Hybrid Turn Complete] Tokens: 1755 | Elapsed: 151.30s (11.60 tps) | TTFT: 1227.2ms | Top-2: [7, 0] (H=1.245)
🦀 [Rust MoE Telemetry (E0..E3)]: E0..E7 Active/NREM homeostasis with Sp=50% 2:4 structured sparsity (256 NREM sleep cycles).
```


### 🔹 Run `2026-10-19 03:35:07` (`thor_31b_20261019_033235` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1755 | Elapsed: 150.94s (11.63 tps) | TTFT: 725.9ms | Top-2: [1, 6] (H=1.251)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=0.76,Tok=1794,Sp=50%,Slp=34) | E1:🟢Active(F=0.67,Tok=1779,Sp=50%,Slp=34) | E2:💤NREM(F=0.68,Tok=1668,Sp=50%,Slp=30) | E3:💤NREM(F=0.75,Tok=1675,Sp=50%,Slp=31)
```

### 🔹 Run `2026-10-19 13:12:24` (`thor_31bnvfp4_20261019_131102` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1650 | Elapsed: 81.47s (20.25 tps) | TTFT: 342.0ms | Top-2: [7, 1] (H=1.250)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=1.03,Tok=409,Sp=50%,Slp=32) | E1:💤NREM(F=1.59,Tok=438,Sp=50%,Slp=33) | E2:💤NREM(F=0.67,Tok=406,Sp=50%,Slp=29) | E3:🟢Active(F=0.74,Tok=411,Sp=50%,Slp=29)
```

### 🔹 Run `2026-10-19 13:23:41` (`thor_31bnvfp4_20261019_132220` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1650 | Elapsed: 81.47s (20.25 tps) | TTFT: 307.4ms | Top-2: [7, 0] (H=1.252)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.15,Tok=843,Sp=50%,Slp=33) | E1:💤NREM(F=1.41,Tok=860,Sp=50%,Slp=33) | E2:💤NREM(F=0.79,Tok=804,Sp=50%,Slp=29) | E3:🟢Active(F=1.16,Tok=829,Sp=50%,Slp=30)
```

### 🔹 Run `2026-10-19 20:37:17` (`thor_31bnvfp4_20261019_203555` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1650 | Elapsed: 81.51s (20.24 tps) | TTFT: 309.5ms | Top-2: [7, 0] (H=1.247)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.35,Tok=1261,Sp=50%,Slp=32) | E1:💤NREM(F=1.42,Tok=1284,Sp=50%,Slp=32) | E2:🟡Fatigued(F=1.23,Tok=1220,Sp=50%,Slp=29) | E3:🟢Active(F=0.82,Tok=1235,Sp=50%,Slp=30)
```

### 🔹 Run `2026-10-30 21:46:36` (`rtx5080_bolt_e4b_gguf_20261030_214623` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 168 | Elapsed: 3.16s (53.20 tps) | TTFT: 1777.8ms | Top-2: [0, 1] (H=1.765)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟡Fatigued(F=1.36,Tok=7,Sp=0%,Slp=0) | E1:💤NREM(F=1.54,Tok=8,Sp=50%,Slp=1) | E2:🟢Active(F=0.22,Tok=1,Sp=0%,Slp=0) | E3:🟢Active(F=0.00,Tok=0,Sp=0%,Slp=0)
```

### 🔹 Run `2026-10-30 21:47:35` (`rtx5080_bolt_e4b_gguf_20261030_214623` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1024 | Elapsed: 9.84s (104.11 tps) | TTFT: 2836.7ms | Top-2: [0, 1] (H=1.212)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.24,Tok=260,Sp=50%,Slp=19) | E1:🟢Active(F=0.24,Tok=267,Sp=50%,Slp=19) | E2:💤NREM(F=0.80,Tok=227,Sp=50%,Slp=19) | E3:💤NREM(F=0.80,Tok=227,Sp=50%,Slp=19)
```

### 🔹 Run `2026-10-30 21:49:45` (`rtx5080_bolt_e4b_gguf_20261030_214838` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 806 | Elapsed: 7.77s (103.74 tps) | TTFT: 2305.8ms | Top-2: [4, 5] (H=1.222)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=0.62,Tok=461,Sp=50%,Slp=15) | E1:💤NREM(F=0.63,Tok=472,Sp=50%,Slp=15) | E2:💤NREM(F=1.04,Tok=408,Sp=50%,Slp=15) | E3:💤NREM(F=0.97,Tok=407,Sp=50%,Slp=15)
```

### 🔹 Run `2026-10-30 21:50:14` (`rtx5080_bolt_e4b_gguf_20261030_214838` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1024 | Elapsed: 10.03s (102.13 tps) | TTFT: 2499.0ms | Top-2: [4, 7] (H=1.190)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=0.82,Tok=747,Sp=50%,Slp=35) | E1:💤NREM(F=0.82,Tok=761,Sp=50%,Slp=35) | E2:💤NREM(F=1.49,Tok=655,Sp=50%,Slp=35) | E3:💤NREM(F=1.49,Tok=654,Sp=50%,Slp=35)
```

### 🔹 Run `2026-10-30 21:54:48` (`rtx5080_bolt_e4b_gguf_20261030_215403` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1875 | Elapsed: 18.26s (102.70 tps) | TTFT: 3448.2ms | Top-2: [0, 1] (H=1.194)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.11,Tok=1310,Sp=50%,Slp=39) | E1:🟢Active(F=0.11,Tok=1324,Sp=50%,Slp=39) | E2:💤NREM(F=0.92,Tok=1149,Sp=50%,Slp=39) | E3:💤NREM(F=0.92,Tok=1148,Sp=50%,Slp=39)
```

### 🔹 Run `2026-10-30 22:08:56` (`thor_31b_20261030_220710` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2207 | Elapsed: 90.97s (24.26 tps) | TTFT: 208.1ms | Top-2: [0, 1] (H=1.246)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.64,Tok=2378,Sp=50%,Slp=44) | E1:🟢Active(F=1.24,Tok=2329,Sp=50%,Slp=43) | E2:🟢Active(F=0.98,Tok=2216,Sp=50%,Slp=38) | E3:💤NREM(F=1.09,Tok=2225,Sp=50%,Slp=37)
```

### 🔹 Run `2026-10-30 22:24:26` (`thor_31b_20261030_220710` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 938 | Elapsed: 61.54s (15.24 tps) | TTFT: 1069.6ms | Top-2: [0, 1] (H=1.246)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.62,Tok=2585,Sp=50%,Slp=60) | E1:🟢Active(F=0.31,Tok=2558,Sp=50%,Slp=61) | E2:🟡Fatigued(F=1.25,Tok=2443,Sp=50%,Slp=54) | E3:💤NREM(F=1.11,Tok=2448,Sp=50%,Slp=52)
```

### 🔹 Run `2026-10-30 22:30:30` (`thor_31b_20261030_220710` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 989 | Elapsed: 56.00s (17.66 tps) | TTFT: 505.7ms | Top-2: [4, 6] (H=1.234)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=0.87,Tok=2826,Sp=50%,Slp=78) | E1:🟢Active(F=0.00,Tok=2814,Sp=50%,Slp=80) | E2:🟢Active(F=0.00,Tok=2659,Sp=50%,Slp=70) | E3:🟢Active(F=0.00,Tok=2701,Sp=50%,Slp=69)
```

### 🔹 Run `2026-10-30 22:43:01` (`thor_31b_20261030_220710` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1251 | Elapsed: 66.92s (18.69 tps) | TTFT: 621.2ms | Top-2: [7, 3] (H=1.235)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:💤NREM(F=0.65,Tok=3148,Sp=50%,Slp=101) | E1:💤NREM(F=0.56,Tok=3141,Sp=50%,Slp=105) | E2:🟡Fatigued(F=1.35,Tok=2960,Sp=50%,Slp=90) | E3:🟡Fatigued(F=1.36,Tok=3030,Sp=50%,Slp=91)
```

### 🔹 Run `2026-10-30 23:05:29` (`thor_31b_20261030_230442` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 425 | Elapsed: 21.62s (19.66 tps) | TTFT: 2256.3ms | Top-2: [0, 3] (H=1.259)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟡Fatigued(F=1.34,Tok=111,Sp=50%,Slp=7) | E1:💤NREM(F=0.75,Tok=116,Sp=50%,Slp=9) | E2:🟢Active(F=1.08,Tok=101,Sp=50%,Slp=6) | E3:🟢Active(F=0.85,Tok=114,Sp=50%,Slp=8)
```

### 🔹 Run `2026-10-30 23:10:37` (`thor_31b_20261030_230442` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 989 | Elapsed: 60.67s (16.30 tps) | TTFT: 285.9ms | Top-2: [0, 3] (H=1.252)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟡Fatigued(F=1.44,Tok=362,Sp=50%,Slp=24) | E1:💤NREM(F=0.80,Tok=367,Sp=50%,Slp=29) | E2:💤NREM(F=1.46,Tok=350,Sp=50%,Slp=24) | E3:💤NREM(F=1.54,Tok=356,Sp=50%,Slp=25)
```

### 🔹 Run `2026-10-30 23:15:00` (`thor_31b_20261030_230442` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2959 | Elapsed: 116.00s (25.51 tps) | TTFT: 698.1ms | Top-2: [7, 0] (H=1.248)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.46,Tok=1132,Sp=50%,Slp=85) | E1:🟢Active(F=0.65,Tok=1149,Sp=50%,Slp=86) | E2:🟡Fatigued(F=1.34,Tok=1048,Sp=50%,Slp=73) | E3:🟢Active(F=1.04,Tok=1072,Sp=50%,Slp=75)
```

### 🔹 Run `2026-10-30 23:51:32` (`thor_31b_20261030_230442` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1741 | Elapsed: 82.52s (21.10 tps) | TTFT: 2386.7ms | Top-2: [0, 1] (H=1.248)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=1.07,Tok=1584,Sp=50%,Slp=119) | E1:🟢Active(F=0.39,Tok=1617,Sp=50%,Slp=122) | E2:💤NREM(F=0.85,Tok=1453,Sp=50%,Slp=103) | E3:🟢Active(F=0.57,Tok=1478,Sp=50%,Slp=104)
```

### 🔹 Run `2026-10-30 23:55:23` (`thor_31b_20261030_235400` | `trt-edgellm-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 560 | Elapsed: 30.01s (18.66 tps) | TTFT: 1022.7ms | Top-2: [7, 1] (H=1.250)
🦀 [Rust MoE Telemetry (E0..E3)]: E0:🟢Active(F=0.00,Tok=141,Sp=50%,Slp=11) | E1:🟢Active(F=0.39,Tok=145,Sp=50%,Slp=11) | E2:💤NREM(F=0.59,Tok=134,Sp=50%,Slp=10) | E3:💤NREM(F=0.87,Tok=143,Sp=50%,Slp=11)
```

### 🔹 Run `2026-10-31 00:33:18` (`thor_31b_20261031_003124` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 191 | Elapsed: 13.05s (14.64 tps) | TTFT: 326.2ms | Top-2: [1, 7] (H=1.310)

```

### 🔹 Run `2026-10-31 00:34:51` (`thor_31b_20261031_003437` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 17 | Elapsed: 1.52s (11.15 tps) | TTFT: 189.9ms | Top-2: [7, 0] (H=1.768)

```

### 🔹 Run `2026-10-31 00:46:42` (`thor_31b_20261031_004458` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 494 | Elapsed: 24.39s (20.26 tps) | TTFT: 951.9ms | Top-2: [7, 0] (H=1.238)

```

### 🔹 Run `2026-10-31 00:48:00` (`thor_31b_20261031_004458` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 711 | Elapsed: 32.59s (21.82 tps) | TTFT: 353.5ms | Top-2: [7, 0] (H=1.246)

```

### 🔹 Run `2026-10-31 00:50:08` (`thor_31b_20261031_004458` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1337 | Elapsed: 70.64s (18.93 tps) | TTFT: 420.1ms | Top-2: [6, 3] (H=1.230)

```

### 🔹 Run `2026-10-31 01:03:21` (`thor_31b_20261031_010154` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1 | Elapsed: 26.72s (0.04 tps) | TTFT: 0.0ms | Top-2: [0, 1] (H=0.000)

```

### 🔹 Run `2026-10-31 01:05:53` (`thor_31b_20261031_010505` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 378 | Elapsed: 19.38s (19.51 tps) | TTFT: 2346.8ms | Top-2: [0, 1] (H=1.267)

```

### 🔹 Run `2026-10-31 01:21:59` (`thor_31b_20261031_011950` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 188 | Elapsed: 13.29s (14.14 tps) | TTFT: 2691.8ms | Top-2: [6, 2] (H=1.297)

```

### 🔹 Run `2026-10-31 01:34:02` (`thor_31b_20261031_013216` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 546 | Elapsed: 35.21s (15.51 tps) | TTFT: 231.7ms | Top-2: [0, 2] (H=1.486)

```

### 🔹 Run `2026-10-31 01:48:32` (`thor_31b_20261031_014659` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 852 | Elapsed: 49.20s (17.32 tps) | TTFT: 226.2ms | Top-2: [0, 1] (H=1.481)

```

### 🔹 Run `2026-10-31 01:52:15` (`thor_31b_20261031_014659` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 977 | Elapsed: 57.57s (16.97 tps) | TTFT: 541.9ms | Top-2: [2, 3] (H=1.475)

```

### 🔹 Run `2026-10-31 14:59:11` (`thor_31b_20261031_145611` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1235 | Elapsed: 77.96s (15.84 tps) | TTFT: 2709.4ms | Top-2: [7, 0, 14, 1, 10] (H=2.078)

```

### 🔹 Run `2026-10-31 15:42:19` (`thor_31b_20261031_153923` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1393 | Elapsed: 72.37s (19.25 tps) | TTFT: 209.9ms | Top-2: [7, 0, 1, 10] (H=2.090)

```

### 🔹 Run `2026-10-31 15:54:05` (`thor_31b_20261031_153923` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 643 | Elapsed: 51.00s (12.61 tps) | TTFT: 812.3ms | Top-2: [7, 0, 14, 1, 10] (H=2.080)

```

### 🔹 Run `2026-10-31 22:37:21` (`thor_31b_20261031_153923` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3343 | Elapsed: 150.11s (22.27 tps) | TTFT: 549.4ms | Top-2: [7, 0, 14, 1, 10] (H=2.082)

```

### 🔹 Run `2026-10-31 23:10:58` (`thor_31b_20261031_230927` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1024 | Elapsed: 72.48s (14.13 tps) | TTFT: 2802.3ms | Top-2: [7, 1, 10, 8, 13] (H=2.084)

```

### 🔹 Run `2026-10-31 23:21:00` (`thor_31b_20261031_231845` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1324 | Elapsed: 90.78s (14.58 tps) | TTFT: 2788.9ms | Top-2: [7, 14, 0, 1, 8] (H=2.079)

```

### 🔹 Run `2026-10-31 23:29:34` (`thor_31b_20261031_231845` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3730 | Elapsed: 142.78s (26.12 tps) | TTFT: 1220.1ms | Top-2: [6, 9, 8, 4, 1] (H=2.092)

```

### 🔹 Run `2026-11-01 14:31:40` (`thor_31b_20261101_143037` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 259 | Elapsed: 15.20s (17.04 tps) | TTFT: 2341.3ms | Top-2: [8, 15, 3, 9] (H=2.087)

```

### 🔹 Run `2026-11-01 14:32:42` (`thor_31b_20261101_143210` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 411 | Elapsed: 21.26s (19.33 tps) | TTFT: 213.7ms | Top-2: [10, 8, 13, 3, 6] (H=2.095)

```

### 🔹 Run `2026-11-01 14:37:05` (`thor_31b_20261101_143553` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1055 | Elapsed: 66.99s (15.75 tps) | TTFT: 231.5ms | Top-2: [7, 0, 10, 8, 13] (H=2.068)

```

### 🔹 Run `2026-11-01 15:03:46` (`thor_31b_20261101_150141` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1953 | Elapsed: 107.23s (18.21 tps) | TTFT: 2797.0ms | Top-2: [7, 14, 0, 1] (H=2.083)

```

### 🔹 Run `2026-11-01 15:08:44` (`thor_31b_20261101_150719` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1243 | Elapsed: 81.16s (15.32 tps) | TTFT: 1268.3ms | Top-2: [7, 0, 14, 10] (H=2.070)

```

### 🔹 Run `2026-11-01 15:28:22` (`thor_31b_20261101_152205` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4054 | Elapsed: 251.99s (16.09 tps) | TTFT: 394.0ms | Top-2: [4, 8, 10, 0, 12] (H=2.104)

```

### 🔹 Run `2026-11-01 15:32:20` (`thor_31b_20261101_152855` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 391 | Elapsed: 129.92s (3.01 tps) | TTFT: 1381.1ms | Top-2: [7, 0, 14, 1, 13] (H=2.091)

```

### 🔹 Run `2026-11-01 15:43:43` (`thor_31b_20261101_153918` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4042 | Elapsed: 259.17s (15.60 tps) | TTFT: 1380.9ms | Top-2: [7, 0, 1, 10, 13] (H=2.100)

```

### 🔹 Run `2026-11-01 15:54:18` (`thor_31b_20261101_154952` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4084 | Elapsed: 257.75s (15.84 tps) | TTFT: 1382.7ms | Top-2: [7, 0, 14, 1, 10] (H=2.099)

```
