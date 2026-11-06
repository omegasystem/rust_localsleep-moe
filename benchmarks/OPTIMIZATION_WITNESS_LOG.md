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

### 🔹 Run `2026-11-01 22:36:41` (`thor_31b_20261101_223332` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 656 | Elapsed: 72.22s (9.08 tps) | TTFT: 938.3ms | Top-2: [4, 3, 1, 15, 0] (H=2.153)

```

### 🔹 Run `2026-11-01 22:50:10` (`thor_31b_20261101_224612` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1440 | Elapsed: 153.84s (9.36 tps) | TTFT: 859.1ms | Top-2: [0, 6, 12, 3, 15] (H=2.124)

```

### 🔹 Run `2026-11-01 22:52:22` (`thor_31b_20261101_224612` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 336 | Elapsed: 19.14s (17.56 tps) | TTFT: 194.2ms | Top-2: [5, 8, 7] (H=2.068)

```

### 🔹 Run `2026-11-01 22:54:27` (`gpt1900_3b_20261101_225408` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 40 | Elapsed: 4.99s (8.02 tps) | TTFT: 44.6ms | Top-2: [5, 8, 15, 2, 12, 9] (H=2.601)

```

### 🔹 Run `2026-11-01 22:56:49` (`gpt1900_3b_20261101_225408` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1440 | Elapsed: 136.61s (10.54 tps) | TTFT: 58.5ms | Top-2: [7, 12, 13, 2, 8] (H=2.082)

```

### 🔹 Run `2026-11-01 23:01:37` (`thor_31b_20261101_225900` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1440 | Elapsed: 138.41s (10.40 tps) | TTFT: 545.2ms | Top-2: [12, 8, 5, 11, 2] (H=2.123)

```

### 🔹 Run `2026-11-01 23:35:53` (`gpt1900_3b_20261101_233433` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 875 | Elapsed: 70.15s (12.47 tps) | TTFT: 42.2ms | Top-2: [2, 4, 6, 1, 14] (H=2.141)

```

### 🔹 Run `2026-11-01 23:44:30` (`thor_31b_20261101_234130` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2534 | Elapsed: 168.32s (15.05 tps) | TTFT: 2280.6ms | Top-2: [14, 7, 0, 8, 13] (H=2.090)

```

### 🔹 Run `2026-11-02 11:45:01` (`thor_31b_20261102_113936` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4246 | Elapsed: 308.77s (13.75 tps) | TTFT: 1376.4ms | Top-2: [0, 1, 10, 8, 13] (H=2.135)

```

### 🔹 Run `2026-11-02 11:53:42` (`thor_31b_20261102_115054` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2134 | Elapsed: 150.74s (14.16 tps) | TTFT: 323.2ms | Top-2: [14, 7, 0, 1, 8] (H=2.130)

```

### 🔹 Run `2026-11-02 12:05:02` (`thor_31b_20261102_120306` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1383 | Elapsed: 107.29s (12.89 tps) | TTFT: 180.1ms | Top-2: [13, 7, 4, 15, 5] (H=2.138)

```

### 🔹 Run `2026-11-02 12:08:57` (`thor_31b_20261102_120714` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1195 | Elapsed: 92.08s (12.98 tps) | TTFT: 1374.8ms | Top-2: [13, 7, 15, 12, 5] (H=2.132)

```

### 🔹 Run `2026-11-02 22:43:42` (`thor_31b_20261102_224146` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1175 | Elapsed: 112.30s (10.46 tps) | TTFT: 212.9ms | Top-2: [7, 0, 14, 1, 8] (H=2.140)

```

### 🔹 Run `2026-11-02 22:56:21` (`thor_31b_20261102_224146` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1880 | Elapsed: 172.06s (10.93 tps) | TTFT: 1131.6ms | Top-2: [7, 0, 14, 1, 8] (H=2.118)

```

### 🔹 Run `2026-11-02 23:19:02` (`thor_31b_20261102_230116` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 6267 | Elapsed: 584.10s (10.73 tps) | TTFT: 2314.4ms | Top-2: [14, 7, 0, 1, 8] (H=2.141)

```

### 🔹 Run `2026-11-02 23:28:59` (`thor_31b_20261102_232558` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 993 | Elapsed: 106.89s (9.29 tps) | TTFT: 2433.4ms | Top-2: [14, 1, 8, 10, 13] (H=2.182)

```

### 🔹 Run `2026-11-02 23:34:47` (`thor_31b_20261102_233100` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2241 | Elapsed: 214.17s (10.46 tps) | TTFT: 217.9ms | Top-2: [0, 14, 1, 13] (H=2.134)

```

### 🔹 Run `2026-11-02 23:40:22` (`thor_31b_20261102_233100` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1011 | Elapsed: 103.31s (9.79 tps) | TTFT: 4241.2ms | Top-2: [3, 5, 2, 9, 8] (H=2.167)

```

### 🔹 Run `2026-11-02 23:46:56` (`thor_31b_20261102_233100` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1237 | Elapsed: 126.42s (9.78 tps) | TTFT: 1840.3ms | Top-2: [7, 14, 1, 8, 13] (H=2.151)

```

### 🔹 Run `2026-11-03 01:02:15` (`thor_31b_20261103_005351` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2755 | Elapsed: 273.62s (10.07 tps) | TTFT: 2451.5ms | Top-2: [7, 0, 14, 8] (H=2.148)

```

### 🔹 Run `2026-11-03 01:14:20` (`thor_31b_20261103_005351` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4350 | Elapsed: 418.41s (10.40 tps) | TTFT: 2316.2ms | Top-2: [0, 14, 10, 8] (H=2.154)

```

### 🔹 Run `2026-11-03 01:23:04` (`thor_31b_20261103_005351` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2228 | Elapsed: 200.32s (11.12 tps) | TTFT: 5047.8ms | Top-2: [8, 9] (H=2.136)

```

### 🔹 Run `2026-11-03 01:29:56` (`thor_31b_20261103_005351` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1497 | Elapsed: 166.32s (9.00 tps) | TTFT: 2922.9ms | Top-2: [7, 14, 1, 10] (H=2.143)

```

### 🔹 Run `2026-11-03 20:53:20` (`thor_31b_20261103_205053` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1268 | Elapsed: 140.13s (9.05 tps) | TTFT: 1951.3ms | Top-2: [13, 4, 7, 15, 12] (H=2.133)

```

### 🔹 Run `2026-11-03 20:58:00` (`thor_31b_20261103_205053` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2150 | Elapsed: 247.88s (8.67 tps) | TTFT: 724.3ms | Top-2: [7, 14, 0, 11, 4] (H=2.121)

```

### 🔹 Run `2026-11-03 21:06:02` (`thor_31b_20261103_210516` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 755 | Elapsed: 36.61s (20.62 tps) | TTFT: 228.7ms | Top-2: [14, 7, 0, 8] (H=2.148)

```

### 🔹 Run `2026-11-03 21:07:12` (`thor_31b_20261103_210516` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1187 | Elapsed: 67.80s (17.51 tps) | TTFT: 449.1ms | Top-2: [0, 14, 1, 13] (H=2.125)

```

### 🔹 Run `2026-11-03 21:09:00` (`thor_31b_20261103_210516` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2187 | Elapsed: 99.70s (21.94 tps) | TTFT: 712.9ms | Top-2: [1, 8, 13, 3] (H=2.130)

```

### 🔹 Run `2026-11-03 21:12:11` (`thor_31b_20261103_210516` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3256 | Elapsed: 151.01s (21.56 tps) | TTFT: 1506.3ms | Top-2: [7, 0, 14, 1, 13] (H=2.115)

```

### 🔹 Run `2026-11-03 21:33:13` (`thor_31b_20261103_213142` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1467 | Elapsed: 63.29s (23.18 tps) | TTFT: 2822.0ms | Top-2: [7, 14, 1, 10, 8] (H=2.128)

```

### 🔹 Run `2026-11-03 21:36:06` (`thor_31b_20261103_213142` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2411 | Elapsed: 106.22s (22.70 tps) | TTFT: 1196.3ms | Top-2: [0, 14, 1, 10, 8] (H=2.134)

```

### 🔹 Run `2026-11-04 01:44:13` (`thor_bolt_e4b_20261104_014030` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4984 | Elapsed: 205.61s (24.24 tps) | TTFT: 382.4ms | Top-2: [13, 4, 15, 5] (H=2.128)

```

### 🔹 Run `2026-11-04 01:47:59` (`thor_bolt_e4b_20261104_014030` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 231 | Elapsed: 10.08s (22.92 tps) | TTFT: 419.9ms | Top-2: [5, 13, 6, 12, 2] (H=2.142)

```

### 🔹 Run `2026-11-04 01:56:57` (`thor_31b_20261104_015616` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 249 | Elapsed: 11.32s (22.01 tps) | TTFT: 1021.1ms | Top-2: [5, 13, 6, 10, 12] (H=2.173)

```

### 🔹 Run `2026-11-04 02:01:30` (`thor_bolt_e4b_20261104_020104` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 290 | Elapsed: 12.10s (23.98 tps) | TTFT: 253.6ms | Top-2: [3, 0, 4] (H=2.170)

```

### 🔹 Run `2026-11-04 02:11:39` (`thor_bolt_e4b_20261104_020829` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4176 | Elapsed: 176.68s (23.64 tps) | TTFT: 182.0ms | Top-2: [3, 5, 9, 8] (H=2.154)

```

### 🔹 Run `2026-11-04 23:33:11` (`thor_31b_20261104_233057` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 751 | Elapsed: 43.65s (17.20 tps) | TTFT: 2456.4ms | Top-2: [7, 14, 1, 8, 13] (H=2.126)

```

### 🔹 Run `2026-11-04 23:37:18` (`thor_31b_20261104_233057` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2433 | Elapsed: 101.07s (24.07 tps) | TTFT: 466.9ms | Top-2: [14, 1, 0, 8] (H=2.120)

```

### 🔹 Run `2026-11-04 23:52:40` (`thor_31b_20261104_235052` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1699 | Elapsed: 75.68s (22.45 tps) | TTFT: 175.1ms | Top-2: [0, 14, 1, 8, 13] (H=2.116)

```

### 🔹 Run `2026-11-05 00:02:48` (`thor_31b_20261104_235052` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3375 | Elapsed: 136.10s (24.80 tps) | TTFT: 1158.7ms | Top-2: [7, 1, 10, 8, 13] (H=2.140)

```

### 🔹 Run `2026-11-05 00:11:45` (`thor_31b_20261104_235052` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3488 | Elapsed: 143.97s (24.23 tps) | TTFT: 2597.8ms | Top-2: [7, 0, 14, 8, 15] (H=2.143)

```

### 🔹 Run `2026-11-05 09:40:28` (`thor_31b_20261105_093715` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4523 | Elapsed: 184.73s (24.48 tps) | TTFT: 4174.9ms | Top-2: [7, 14, 10, 3] (H=2.140)

```

### 🔹 Run `2026-11-05 11:22:12` (`thor_31b_20261105_112019` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2775 | Elapsed: 109.95s (25.24 tps) | TTFT: 679.4ms | Top-2: [14, 1, 10, 13] (H=2.135)

```

### 🔹 Run `2026-11-05 11:40:38` (`thor_31b_20261105_113945` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 863 | Elapsed: 45.21s (19.09 tps) | TTFT: 708.0ms | Top-2: [14, 10, 8, 3] (H=2.158)

```

### 🔹 Run `2026-11-05 11:41:54` (`thor_31b_20261105_113945` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 401 | Elapsed: 21.79s (18.41 tps) | TTFT: 526.9ms | Top-2: [8, 0, 11, 3, 5] (H=2.102)

```

### 🔹 Run `2026-11-05 11:42:58` (`thor_31b_20261105_114212` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 809 | Elapsed: 43.02s (18.81 tps) | TTFT: 696.4ms | Top-2: [7, 0, 14, 10] (H=2.156)

```

### 🔹 Run `2026-11-05 11:45:39` (`thor_31b_20261105_114212` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3128 | Elapsed: 127.57s (24.52 tps) | TTFT: 778.9ms | Top-2: [0, 14, 1, 10] (H=2.137)

```

### 🔹 Run `2026-11-05 13:44:42` (`thor_31b_20261105_134353` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 815 | Elapsed: 43.58s (18.70 tps) | TTFT: 699.6ms | Top-2: [7, 0, 1, 10] (H=2.152)

```

### 🔹 Run `2026-11-05 13:52:32` (`thor_31b_20261105_134353` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3591 | Elapsed: 141.48s (25.38 tps) | TTFT: 785.4ms | Top-2: [7, 14, 1, 10, 13] (H=2.129)

```

### 🔹 Run `2026-11-05 18:27:50` (`thor_31b_20261105_182109` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2560 | Elapsed: 88.51s (28.92 tps) | TTFT: 313.0ms | Top-2: [7, 1, 8, 13] (H=2.124)

```

### 🔹 Run `2026-11-05 20:26:36` (`thor_31b_20261105_182109` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4096 | Elapsed: 132.41s (30.93 tps) | TTFT: 195.7ms | Top-2: [4, 13, 14, 3, 6] (H=2.158)

```

### 🔹 Run `2026-11-05 20:28:48` (`thor_31b_20261105_182109` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 4096 | Elapsed: 130.65s (31.35 tps) | TTFT: 822.9ms | Top-2: [8, 12, 2, 13] (H=2.120)

```

### 🔹 Run `2026-11-05 20:39:01` (`thor_31b_20261105_182109` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3531 | Elapsed: 122.97s (28.71 tps) | TTFT: 792.4ms | Top-2: [7, 1, 10, 8, 13] (H=2.121)

```

### 🔹 Run `2026-11-05 20:45:14` (`thor_31b_20261105_182109` | `web-ui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2142 | Elapsed: 80.97s (26.45 tps) | TTFT: 7997.1ms | Top-2: [7, 0, 10] (H=2.160)

```

### 🔹 Run `2026-11-05 20:50:10` (`thor_31b_20261105_182109` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2059 | Elapsed: 63.81s (32.27 tps) | TTFT: 5826.3ms | Top-2: [7, 14, 1, 8] (H=2.186)

```

### 🔹 Run `2026-11-05 20:55:50` (`thor_31b_20261105_182109` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2494 | Elapsed: 87.98s (28.35 tps) | TTFT: 1492.5ms | Top-2: [7, 1, 4, 15] (H=2.163)

```

### 🔹 Run `2026-11-05 20:58:34` (`thor_31b_20261105_182109` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2513 | Elapsed: 73.78s (34.06 tps) | TTFT: 1846.1ms | Top-2: [7, 0, 14, 13] (H=2.187)

```

### 🔹 Run `2026-11-05 21:02:43` (`thor_31b_20261105_182109` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 535 | Elapsed: 38.68s (13.83 tps) | TTFT: 7247.3ms | Top-2: [0, 14, 1, 3] (H=2.150)

```

### 🔹 Run `2026-11-05 21:06:26` (`thor_31b_20261105_182109` | `pyqt6-gui-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2568 | Elapsed: 107.87s (23.81 tps) | TTFT: 647.6ms | Top-2: [0, 14, 13, 4] (H=2.120)

```

### 🔹 Run `2026-11-05 21:20:29` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1426 | Elapsed: 51.59s (27.64 tps) | TTFT: 256.9ms | Top-2: [11, 9, 13, 14] (H=2.124)

```

### 🔹 Run `2026-11-05 21:21:39` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2064 | Elapsed: 68.88s (29.97 tps) | TTFT: 1018.5ms | Top-2: [7, 0, 14, 1, 10] (H=2.121)

```

### 🔹 Run `2026-11-05 21:40:27` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1327 | Elapsed: 79.43s (16.71 tps) | TTFT: 338.1ms | Top-2: [7, 0, 1, 14, 8] (H=2.155)

```

### 🔹 Run `2026-11-05 21:43:22` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2254 | Elapsed: 174.28s (12.93 tps) | TTFT: 395.9ms | Top-2: [0, 14, 1, 8] (H=2.167)

```

### 🔹 Run `2026-11-05 21:45:56` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2059 | Elapsed: 152.31s (13.52 tps) | TTFT: 265.0ms | Top-2: [14, 1, 8, 13] (H=2.175)

```

### 🔹 Run `2026-11-05 21:48:48` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2222 | Elapsed: 171.02s (12.99 tps) | TTFT: 252.8ms | Top-2: [1, 0, 3, 13] (H=2.159)

```

### 🔹 Run `2026-11-05 21:50:57` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1715 | Elapsed: 127.99s (13.40 tps) | TTFT: 274.9ms | Top-2: [14, 0, 8, 10] (H=2.161)

```

### 🔹 Run `2026-11-05 21:53:36` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2090 | Elapsed: 157.07s (13.31 tps) | TTFT: 247.0ms | Top-2: [1, 10, 8, 13, 3] (H=2.168)

```

### 🔹 Run `2026-11-05 21:56:14` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2101 | Elapsed: 156.29s (13.44 tps) | TTFT: 250.6ms | Top-2: [0, 1, 10, 8] (H=2.162)

```

### 🔹 Run `2026-11-05 21:58:35` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1893 | Elapsed: 139.96s (13.53 tps) | TTFT: 246.2ms | Top-2: [8, 3, 4, 6] (H=2.169)

```

### 🔹 Run `2026-11-05 22:01:11` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2095 | Elapsed: 155.26s (13.49 tps) | TTFT: 246.3ms | Top-2: [14, 7, 0, 1, 8] (H=2.155)

```

### 🔹 Run `2026-11-05 22:03:33` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1852 | Elapsed: 140.71s (13.16 tps) | TTFT: 248.1ms | Top-2: [4, 7, 13, 12, 15] (H=2.168)

```

### 🔹 Run `2026-11-05 22:06:15` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2195 | Elapsed: 160.55s (13.67 tps) | TTFT: 253.1ms | Top-2: [7, 0, 14, 13] (H=2.159)

```

### 🔹 Run `2026-11-05 23:10:10` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2353 | Elapsed: 143.66s (16.38 tps) | TTFT: 415.2ms | Top-2: [0, 14, 1, 10, 8] (H=2.155)

```

### 🔹 Run `2026-11-05 23:12:46` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1924 | Elapsed: 155.10s (12.40 tps) | TTFT: 359.8ms | Top-2: [13, 4, 15, 6, 12] (H=2.163)

```

### 🔹 Run `2026-11-05 23:16:01` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2412 | Elapsed: 193.61s (12.46 tps) | TTFT: 937.9ms | Top-2: [7, 0, 1, 10] (H=2.171)

```

### 🔹 Run `2026-11-05 23:19:03` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2260 | Elapsed: 180.43s (12.53 tps) | TTFT: 1200.3ms | Top-2: [1, 13, 15, 4] (H=2.176)

```

### 🔹 Run `2026-11-05 23:21:51` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2169 | Elapsed: 165.99s (13.07 tps) | TTFT: 207.8ms | Top-2: [7, 0, 1, 10] (H=2.196)

```

### 🔹 Run `2026-11-05 23:24:59` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2399 | Elapsed: 186.99s (12.83 tps) | TTFT: 967.9ms | Top-2: [0, 14, 1, 10] (H=2.155)

```

### 🔹 Run `2026-11-05 23:28:31` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2798 | Elapsed: 210.45s (13.30 tps) | TTFT: 976.8ms | Top-2: [0, 14, 1, 8, 13] (H=2.154)

```

### 🔹 Run `2026-11-05 23:31:30` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2373 | Elapsed: 178.03s (13.33 tps) | TTFT: 1100.2ms | Top-2: [7, 1, 10, 8] (H=2.181)

```

### 🔹 Run `2026-11-05 23:34:14` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2149 | Elapsed: 162.51s (13.22 tps) | TTFT: 211.8ms | Top-2: [7, 0, 14, 3] (H=2.193)

```

### 🔹 Run `2026-11-05 23:37:07` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2127 | Elapsed: 171.42s (12.41 tps) | TTFT: 1057.0ms | Top-2: [7, 0, 14, 8, 13] (H=2.161)

```

### 🔹 Run `2026-11-05 23:39:41` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2026 | Elapsed: 152.94s (13.25 tps) | TTFT: 961.0ms | Top-2: [7, 0, 14, 1] (H=2.159)

```

### 🔹 Run `2026-11-05 23:45:23` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2471 | Elapsed: 189.98s (13.01 tps) | TTFT: 182.9ms | Top-2: [7, 0, 14, 8] (H=2.150)

```

### 🔹 Run `2026-11-05 23:48:01` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2078 | Elapsed: 156.99s (13.24 tps) | TTFT: 1000.6ms | Top-2: [0, 14, 10, 3] (H=2.163)

```

### 🔹 Run `2026-11-05 23:51:04` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2353 | Elapsed: 181.71s (12.95 tps) | TTFT: 192.2ms | Top-2: [0, 13, 3, 4] (H=2.180)

```

### 🔹 Run `2026-11-05 23:53:34` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1901 | Elapsed: 147.92s (12.85 tps) | TTFT: 915.0ms | Top-2: [7, 0, 1, 13] (H=2.170)

```

### 🔹 Run `2026-11-05 23:56:28` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2349 | Elapsed: 173.35s (13.55 tps) | TTFT: 907.3ms | Top-2: [14, 1, 7, 0] (H=2.156)

```

### 🔹 Run `2026-11-05 23:59:05` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2046 | Elapsed: 155.81s (13.13 tps) | TTFT: 1072.0ms | Top-2: [7, 0, 14, 10, 8] (H=2.174)

```

### 🔹 Run `2026-11-06 00:02:20` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2456 | Elapsed: 193.76s (12.68 tps) | TTFT: 208.5ms | Top-2: [13, 4, 7, 15, 6] (H=2.185)

```

### 🔹 Run `2026-11-06 00:04:46` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1875 | Elapsed: 144.14s (13.01 tps) | TTFT: 899.6ms | Top-2: [0, 10, 13, 15, 3] (H=2.167)

```

### 🔹 Run `2026-11-06 00:08:06` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2603 | Elapsed: 197.98s (13.15 tps) | TTFT: 824.5ms | Top-2: [7, 0, 10, 13] (H=2.162)

```

### 🔹 Run `2026-11-06 00:10:46` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2209 | Elapsed: 159.17s (13.88 tps) | TTFT: 1034.0ms | Top-2: [14, 10, 8, 3] (H=2.168)

```

### 🔹 Run `2026-11-06 00:13:39` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2329 | Elapsed: 172.02s (13.54 tps) | TTFT: 196.4ms | Top-2: [7, 14, 0, 8, 13] (H=2.176)

```

### 🔹 Run `2026-11-06 00:16:40` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2326 | Elapsed: 179.12s (12.99 tps) | TTFT: 943.9ms | Top-2: [0, 14, 10, 8] (H=2.154)

```

### 🔹 Run `2026-11-06 00:19:48` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2487 | Elapsed: 187.41s (13.27 tps) | TTFT: 981.1ms | Top-2: [13, 4, 7, 15, 6] (H=2.164)

```

### 🔹 Run `2026-11-06 00:22:38` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2141 | Elapsed: 168.20s (12.73 tps) | TTFT: 990.9ms | Top-2: [7, 0, 14, 1, 10] (H=2.165)

```

### 🔹 Run `2026-11-06 00:25:22` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2070 | Elapsed: 163.29s (12.68 tps) | TTFT: 197.2ms | Top-2: [7, 0, 1, 10, 8] (H=2.190)

```

### 🔹 Run `2026-11-06 00:28:10` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2163 | Elapsed: 166.68s (12.98 tps) | TTFT: 911.3ms | Top-2: [7, 0, 14, 1, 10] (H=2.168)

```

### 🔹 Run `2026-11-06 00:31:02` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2197 | Elapsed: 170.32s (12.90 tps) | TTFT: 1069.9ms | Top-2: [7, 14, 1, 8] (H=2.177)

```

### 🔹 Run `2026-11-06 00:44:14` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2171 | Elapsed: 166.55s (13.04 tps) | TTFT: 187.8ms | Top-2: [7, 0, 14, 8] (H=2.171)

```

### 🔹 Run `2026-11-06 00:47:12` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2370 | Elapsed: 177.41s (13.36 tps) | TTFT: 1014.7ms | Top-2: [7, 0, 1, 8, 13] (H=2.175)

```

### 🔹 Run `2026-11-06 00:49:54` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2037 | Elapsed: 160.02s (12.73 tps) | TTFT: 190.7ms | Top-2: [7, 0, 14, 10] (H=2.187)

```

### 🔹 Run `2026-11-06 01:15:21` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2187 | Elapsed: 143.94s (15.19 tps) | TTFT: 400.8ms | Top-2: [1, 8, 13, 3] (H=2.151)

```

### 🔹 Run `2026-11-06 01:17:53` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1947 | Elapsed: 151.06s (12.89 tps) | TTFT: 388.4ms | Top-2: [7, 0, 14, 8, 13] (H=2.163)

```

### 🔹 Run `2026-11-06 01:20:09` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1666 | Elapsed: 134.15s (12.42 tps) | TTFT: 1056.5ms | Top-2: [0, 14, 13, 3] (H=2.156)

```

### 🔹 Run `2026-11-06 01:22:44` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2106 | Elapsed: 154.24s (13.65 tps) | TTFT: 949.1ms | Top-2: [14, 1, 7, 0] (H=2.157)

```

### 🔹 Run `2026-11-06 01:23:19` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 439 | Elapsed: 34.14s (12.86 tps) | TTFT: 195.6ms | Top-2: [7, 0, 14, 1, 10] (H=2.168)

```

### 🔹 Run `2026-11-06 01:30:27` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1506 | Elapsed: 118.73s (12.68 tps) | TTFT: 1055.8ms | Top-2: [13, 7, 12, 5] (H=2.167)

```

### 🔹 Run `2026-11-06 01:33:14` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2284 | Elapsed: 166.05s (13.75 tps) | TTFT: 735.4ms | Top-2: [7, 14, 1, 0] (H=2.168)

```

### 🔹 Run `2026-11-06 01:35:21` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1775 | Elapsed: 126.28s (14.06 tps) | TTFT: 1070.2ms | Top-2: [7, 0, 14, 10, 13] (H=2.178)

```

### 🔹 Run `2026-11-06 01:35:58` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 456 | Elapsed: 35.35s (12.90 tps) | TTFT: 175.7ms | Top-2: [0, 6, 12, 1, 11, 7] (H=2.148)

```

### 🔹 Run `2026-11-06 01:38:30` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1966 | Elapsed: 150.72s (13.04 tps) | TTFT: 889.8ms | Top-2: [8, 10, 9, 6] (H=2.165)

```

### 🔹 Run `2026-11-06 01:40:35` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1547 | Elapsed: 124.26s (12.45 tps) | TTFT: 1041.1ms | Top-2: [1, 0, 3] (H=2.160)

```

### 🔹 Run `2026-11-06 01:42:51` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1807 | Elapsed: 134.60s (13.43 tps) | TTFT: 767.1ms | Top-2: [7, 14, 10, 13, 3] (H=2.161)

```

### 🔹 Run `2026-11-06 01:43:30` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 446 | Elapsed: 37.15s (12.01 tps) | TTFT: 182.3ms | Top-2: [7, 14, 1, 8, 13] (H=2.187)

```

### 🔹 Run `2026-11-06 01:45:39` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1638 | Elapsed: 128.58s (12.74 tps) | TTFT: 958.8ms | Top-2: [7, 0, 14, 13] (H=2.163)

```

### 🔹 Run `2026-11-06 01:48:20` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2041 | Elapsed: 159.44s (12.80 tps) | TTFT: 1025.9ms | Top-2: [7, 0, 14, 1] (H=2.165)

```

### 🔹 Run `2026-11-06 01:50:04` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1392 | Elapsed: 103.12s (13.50 tps) | TTFT: 997.1ms | Top-2: [0, 1, 10, 8] (H=2.165)

```

### 🔹 Run `2026-11-06 01:50:43` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 468 | Elapsed: 36.95s (12.67 tps) | TTFT: 180.8ms | Top-2: [3, 2, 10, 11, 12, 8] (H=2.172)

```

### 🔹 Run `2026-11-06 01:52:47` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1592 | Elapsed: 122.90s (12.95 tps) | TTFT: 681.9ms | Top-2: [0, 14, 1, 10, 8] (H=2.162)

```

### 🔹 Run `2026-11-06 01:55:04` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1770 | Elapsed: 136.24s (12.99 tps) | TTFT: 824.2ms | Top-2: [14, 7, 1, 0, 8] (H=2.162)

```

### 🔹 Run `2026-11-06 01:57:20` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1855 | Elapsed: 134.43s (13.80 tps) | TTFT: 870.8ms | Top-2: [13, 4, 7, 15, 6] (H=2.167)

```

### 🔹 Run `2026-11-06 01:58:02` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 531 | Elapsed: 40.80s (13.01 tps) | TTFT: 174.8ms | Top-2: [7, 14, 0, 1, 8] (H=2.166)

```

### 🔹 Run `2026-11-06 02:00:07` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1587 | Elapsed: 124.11s (12.79 tps) | TTFT: 876.3ms | Top-2: [14, 1, 8, 13] (H=2.169)

```

### 🔹 Run `2026-11-06 02:02:37` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1956 | Elapsed: 149.21s (13.11 tps) | TTFT: 735.2ms | Top-2: [0, 1, 8] (H=2.161)

```

### 🔹 Run `2026-11-06 02:04:50` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1779 | Elapsed: 131.75s (13.50 tps) | TTFT: 948.3ms | Top-2: [14, 10, 8, 13, 3] (H=2.158)

```

### 🔹 Run `2026-11-06 02:05:28` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 493 | Elapsed: 37.64s (13.10 tps) | TTFT: 174.1ms | Top-2: [7, 0, 8, 3] (H=2.197)

```

### 🔹 Run `2026-11-06 02:07:42` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1642 | Elapsed: 132.05s (12.43 tps) | TTFT: 890.0ms | Top-2: [13, 4, 15, 6, 12] (H=2.158)

```

### 🔹 Run `2026-11-06 02:10:33` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2164 | Elapsed: 169.62s (12.76 tps) | TTFT: 871.4ms | Top-2: [7, 1, 8, 3] (H=2.175)

```

### 🔹 Run `2026-11-06 02:13:08` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2027 | Elapsed: 154.47s (13.12 tps) | TTFT: 1035.9ms | Top-2: [7, 10, 13, 3] (H=2.158)

```

### 🔹 Run `2026-11-06 02:13:45` (`thor_31b_20261105_211233` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 436 | Elapsed: 35.64s (12.23 tps) | TTFT: 197.4ms | Top-2: [7, 1, 10, 8] (H=2.171)

```

### 🔹 Run `2026-11-06 10:00:21` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2007 | Elapsed: 68.50s (29.30 tps) | TTFT: 263.5ms | Top-2: [7, 0, 14, 1] (H=2.135)

```

### 🔹 Run `2026-11-06 10:22:34` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 267 | Elapsed: 14.06s (18.99 tps) | TTFT: 335.3ms | Top-2: [14, 8, 10, 6] (H=2.139)

```

### 🔹 Run `2026-11-06 10:23:44` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1988 | Elapsed: 68.75s (28.92 tps) | TTFT: 436.0ms | Top-2: [11, 2, 0, 13, 7] (H=2.147)

```

### 🔹 Run `2026-11-06 10:24:03` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 366 | Elapsed: 18.03s (20.29 tps) | TTFT: 338.9ms | Top-2: [1, 14, 8, 9, 10] (H=2.152)

```

### 🔹 Run `2026-11-06 10:25:22` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2181 | Elapsed: 77.07s (28.30 tps) | TTFT: 456.9ms | Top-2: [0, 14, 1, 10] (H=2.156)

```

### 🔹 Run `2026-11-06 10:25:42` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 392 | Elapsed: 17.88s (21.92 tps) | TTFT: 507.3ms | Top-2: [7, 0, 14, 8] (H=2.150)

```

### 🔹 Run `2026-11-06 10:27:09` (`thor_31b_20261106_095246` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2556 | Elapsed: 86.69s (29.49 tps) | TTFT: 472.3ms | Top-2: [11, 2, 9, 0, 14] (H=2.143)

```

### 🔹 Run `2026-11-06 10:49:31` (`thor_31b_20261106_104633` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2935 | Elapsed: 98.12s (29.91 tps) | TTFT: 632.6ms | Top-2: [7, 0, 14, 10] (H=2.145)

```

### 🔹 Run `2026-11-06 10:51:04` (`thor_31b_20261106_104633` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2797 | Elapsed: 92.10s (30.37 tps) | TTFT: 735.4ms | Top-2: [7, 0, 1, 10, 8] (H=2.152)

```

### 🔹 Run `2026-11-06 10:52:45` (`thor_31b_20261106_104633` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2990 | Elapsed: 99.68s (29.99 tps) | TTFT: 731.5ms | Top-2: [7, 10, 13, 3] (H=2.145)

```

### 🔹 Run `2026-11-06 10:54:29` (`thor_31b_20261106_104633` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3144 | Elapsed: 102.76s (30.60 tps) | TTFT: 688.4ms | Top-2: [7, 0, 14, 1, 10] (H=2.145)

```

### 🔹 Run `2026-11-06 10:56:10` (`thor_31b_20261106_104633` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3001 | Elapsed: 100.32s (29.91 tps) | TTFT: 717.1ms | Top-2: [7, 0, 10, 8] (H=2.150)

```

### 🔹 Run `2026-11-06 10:58:28` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2935 | Elapsed: 99.23s (29.58 tps) | TTFT: 625.2ms | Top-2: [7, 0, 14, 10] (H=2.145)

```

### 🔹 Run `2026-11-06 11:00:08` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2952 | Elapsed: 98.27s (30.04 tps) | TTFT: 725.3ms | Top-2: [7, 0, 10, 8] (H=2.151)

```

### 🔹 Run `2026-11-06 11:01:49` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2985 | Elapsed: 99.57s (29.98 tps) | TTFT: 716.9ms | Top-2: [10, 8, 15, 6] (H=2.151)

```

### 🔹 Run `2026-11-06 11:03:28` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2944 | Elapsed: 97.87s (30.08 tps) | TTFT: 710.8ms | Top-2: [10, 8, 13, 15] (H=2.147)

```

### 🔹 Run `2026-11-06 11:05:09` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3044 | Elapsed: 99.24s (30.67 tps) | TTFT: 716.3ms | Top-2: [0, 10, 8, 13, 3] (H=2.152)

```

### 🔹 Run `2026-11-06 11:07:45` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2799 | Elapsed: 93.44s (29.95 tps) | TTFT: 671.1ms | Top-2: [7, 0, 14, 1, 10] (H=2.149)

```

### 🔹 Run `2026-11-06 11:08:59` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2672 | Elapsed: 73.02s (36.59 tps) | TTFT: 1863.3ms | Top-2: [11, 2, 9, 0, 13] (H=2.170)

```

### 🔹 Run `2026-11-06 11:09:28` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 439 | Elapsed: 25.61s (17.14 tps) | TTFT: 714.2ms | Top-2: [0, 8, 15, 13] (H=2.166)

```

### 🔹 Run `2026-11-06 11:11:54` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2799 | Elapsed: 93.34s (29.99 tps) | TTFT: 661.8ms | Top-2: [7, 0, 1, 10, 13] (H=2.142)

```

### 🔹 Run `2026-11-06 11:13:09` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2668 | Elapsed: 73.94s (36.08 tps) | TTFT: 1878.0ms | Top-2: [11, 2, 9, 0, 13, 14] (H=2.171)

```

### 🔹 Run `2026-11-06 11:13:43` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 471 | Elapsed: 29.60s (15.91 tps) | TTFT: 707.8ms | Top-2: [7, 0, 14, 10, 3] (H=2.153)

```

### 🔹 Run `2026-11-06 11:17:38` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2791 | Elapsed: 92.94s (30.03 tps) | TTFT: 652.2ms | Top-2: [7, 0, 1, 10, 8] (H=2.149)

```

### 🔹 Run `2026-11-06 11:18:53` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2664 | Elapsed: 73.56s (36.22 tps) | TTFT: 1846.5ms | Top-2: [11, 2, 9, 13, 14] (H=2.166)

```

### 🔹 Run `2026-11-06 11:19:24` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 441 | Elapsed: 26.14s (16.87 tps) | TTFT: 704.8ms | Top-2: [7, 0, 1, 14, 8] (H=2.156)

```

### 🔹 Run `2026-11-06 11:21:10` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3162 | Elapsed: 105.30s (30.03 tps) | TTFT: 885.8ms | Top-2: [0, 14, 1, 10, 8] (H=2.145)

```

### 🔹 Run `2026-11-06 11:21:43` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 464 | Elapsed: 28.70s (16.17 tps) | TTFT: 698.8ms | Top-2: [14, 13, 9] (H=2.160)

```

### 🔹 Run `2026-11-06 11:23:27` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3073 | Elapsed: 102.85s (29.88 tps) | TTFT: 902.5ms | Top-2: [7, 0, 1, 10] (H=2.145)

```

### 🔹 Run `2026-11-06 11:23:56` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 396 | Elapsed: 24.64s (16.07 tps) | TTFT: 701.6ms | Top-2: [0, 1, 10, 13] (H=2.159)

```

### 🔹 Run `2026-11-06 11:25:50` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3361 | Elapsed: 112.45s (29.89 tps) | TTFT: 869.3ms | Top-2: [0, 1, 10, 8] (H=2.145)

```

### 🔹 Run `2026-11-06 11:26:21` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 446 | Elapsed: 26.81s (16.64 tps) | TTFT: 609.6ms | Top-2: [1, 14, 15, 9] (H=2.184)

```

### 🔹 Run `2026-11-06 11:28:10` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 3154 | Elapsed: 108.49s (29.07 tps) | TTFT: 908.2ms | Top-2: [7, 0, 1, 10, 8] (H=2.138)

```

### 🔹 Run `2026-11-06 11:28:38` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 360 | Elapsed: 23.96s (15.02 tps) | TTFT: 685.0ms | Top-2: [7, 0, 1, 14, 15] (H=2.148)

```

### 🔹 Run `2026-11-06 11:55:59` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1174 | Elapsed: 42.72s (27.48 tps) | TTFT: 1017.2ms | Top-2: [0, 1, 14, 15, 9] (H=2.120)

```

### 🔹 Run `2026-11-06 11:56:46` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1122 | Elapsed: 45.32s (24.76 tps) | TTFT: 1008.5ms | Top-2: [7, 0, 14, 9, 13] (H=2.115)

```

### 🔹 Run `2026-11-06 11:57:29` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1071 | Elapsed: 40.62s (26.37 tps) | TTFT: 1041.9ms | Top-2: [7, 1, 8, 15, 10] (H=2.127)

```

### 🔹 Run `2026-11-06 11:58:12` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1128 | Elapsed: 42.10s (26.79 tps) | TTFT: 1026.2ms | Top-2: [7, 0, 15, 9] (H=2.121)

```

### 🔹 Run `2026-11-06 11:59:00` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1236 | Elapsed: 46.43s (26.62 tps) | TTFT: 1020.3ms | Top-2: [7, 0, 1, 8, 15] (H=2.128)

```

### 🔹 Run `2026-11-06 12:00:48` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1108 | Elapsed: 39.32s (28.18 tps) | TTFT: 1033.6ms | Top-2: [7, 14, 8, 15, 9] (H=2.122)

```

### 🔹 Run `2026-11-06 12:01:40` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1354 | Elapsed: 50.18s (26.98 tps) | TTFT: 1061.7ms | Top-2: [13, 11, 15, 9] (H=2.137)

```

### 🔹 Run `2026-11-06 12:03:10` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2489 | Elapsed: 89.44s (27.83 tps) | TTFT: 1082.5ms | Top-2: [7, 1, 10, 13, 3] (H=2.127)

```

### 🔹 Run `2026-11-06 12:04:32` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 2248 | Elapsed: 79.87s (28.15 tps) | TTFT: 1056.7ms | Top-2: [7, 0, 14, 1, 10] (H=2.124)

```

### 🔹 Run `2026-11-06 12:05:26` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1295 | Elapsed: 53.08s (24.40 tps) | TTFT: 1054.2ms | Top-2: [7, 0, 14, 1, 10] (H=2.121)

```

### 🔹 Run `2026-11-06 12:06:13` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1162 | Elapsed: 44.47s (26.13 tps) | TTFT: 1076.0ms | Top-2: [0, 1, 14, 8, 15] (H=2.114)

```

### 🔹 Run `2026-11-06 12:07:04` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1256 | Elapsed: 49.19s (25.53 tps) | TTFT: 1213.3ms | Top-2: [7, 10, 8, 13] (H=2.130)

```

### 🔹 Run `2026-11-06 12:07:41` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 970 | Elapsed: 36.17s (26.82 tps) | TTFT: 1226.9ms | Top-2: [0, 14, 8, 15, 13] (H=2.129)

```

### 🔹 Run `2026-11-06 12:08:24` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1104 | Elapsed: 40.14s (27.51 tps) | TTFT: 1850.0ms | Top-2: [7, 0, 15, 9, 13] (H=2.126)

```

### 🔹 Run `2026-11-06 12:09:04` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1082 | Elapsed: 38.72s (27.95 tps) | TTFT: 1874.0ms | Top-2: [7, 14, 8, 15] (H=2.131)

```

### 🔹 Run `2026-11-06 12:09:43` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1014 | Elapsed: 36.94s (27.45 tps) | TTFT: 1915.9ms | Top-2: [14, 13, 0, 2] (H=2.120)

```

### 🔹 Run `2026-11-06 12:10:23` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1056 | Elapsed: 38.70s (27.29 tps) | TTFT: 1926.4ms | Top-2: [7, 0, 1, 8, 15] (H=2.144)

```

### 🔹 Run `2026-11-06 12:11:06` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1118 | Elapsed: 41.07s (27.22 tps) | TTFT: 1948.7ms | Top-2: [7, 1, 8, 10] (H=2.127)

```

### 🔹 Run `2026-11-06 12:11:50` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1124 | Elapsed: 42.67s (26.34 tps) | TTFT: 1913.6ms | Top-2: [7, 0, 8, 9, 13] (H=2.125)

```

### 🔹 Run `2026-11-06 12:12:38` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1159 | Elapsed: 46.18s (25.10 tps) | TTFT: 1913.3ms | Top-2: [0, 1, 14, 8, 15] (H=2.127)

```

### 🔹 Run `2026-11-06 12:13:23` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1148 | Elapsed: 42.72s (26.87 tps) | TTFT: 1932.3ms | Top-2: [7, 0, 1, 14, 8] (H=2.134)

```

### 🔹 Run `2026-11-06 12:14:01` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1050 | Elapsed: 36.60s (28.69 tps) | TTFT: 1936.4ms | Top-2: [7, 8, 15, 9, 13] (H=2.131)

```

### 🔹 Run `2026-11-06 12:14:42` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1072 | Elapsed: 39.93s (26.85 tps) | TTFT: 1912.0ms | Top-2: [7, 8, 15, 9, 13] (H=2.121)

```

### 🔹 Run `2026-11-06 12:15:22` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1006 | Elapsed: 37.69s (26.69 tps) | TTFT: 1947.9ms | Top-2: [14, 8, 15, 10] (H=2.133)

```

### 🔹 Run `2026-11-06 12:16:04` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1078 | Elapsed: 40.70s (26.49 tps) | TTFT: 1943.6ms | Top-2: [7, 1, 14, 8, 15] (H=2.127)

```

### 🔹 Run `2026-11-06 12:16:43` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1001 | Elapsed: 36.82s (27.19 tps) | TTFT: 1951.8ms | Top-2: [13, 15, 9, 12] (H=2.128)

```

### 🔹 Run `2026-11-06 12:17:21` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 988 | Elapsed: 36.76s (26.88 tps) | TTFT: 1928.9ms | Top-2: [7, 0, 1, 14, 8] (H=2.137)

```

### 🔹 Run `2026-11-06 12:18:03` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1107 | Elapsed: 40.37s (27.42 tps) | TTFT: 1959.0ms | Top-2: [7, 0, 10, 3] (H=2.128)

```

### 🔹 Run `2026-11-06 12:18:43` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1026 | Elapsed: 38.67s (26.53 tps) | TTFT: 1959.2ms | Top-2: [7, 1, 15, 13, 10] (H=2.120)

```

### 🔹 Run `2026-11-06 12:19:33` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1237 | Elapsed: 48.31s (25.60 tps) | TTFT: 1964.3ms | Top-2: [7, 0, 1, 10] (H=2.126)

```

### 🔹 Run `2026-11-06 12:20:22` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1197 | Elapsed: 46.90s (25.52 tps) | TTFT: 1939.7ms | Top-2: [7, 0, 1, 10, 13] (H=2.116)

```

### 🔹 Run `2026-11-06 12:21:22` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1154 | Elapsed: 43.82s (26.33 tps) | TTFT: 1925.3ms | Top-2: [1, 8, 15, 9] (H=2.125)

```

### 🔹 Run `2026-11-06 12:22:02` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1039 | Elapsed: 38.65s (26.88 tps) | TTFT: 1952.6ms | Top-2: [7, 8, 15, 13, 10] (H=2.117)

```

### 🔹 Run `2026-11-06 12:22:41` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1017 | Elapsed: 36.94s (27.53 tps) | TTFT: 1966.3ms | Top-2: [0, 14, 10, 8] (H=2.133)

```

### 🔹 Run `2026-11-06 12:23:23` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1083 | Elapsed: 40.39s (26.82 tps) | TTFT: 1952.6ms | Top-2: [14, 15, 13, 10] (H=2.134)

```

### 🔹 Run `2026-11-06 12:40:58` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1368 | Elapsed: 51.32s (26.66 tps) | TTFT: 2044.3ms | Top-2: [7, 14, 1, 3] (H=2.134)

```

### 🔹 Run `2026-11-06 12:41:32` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1165 | Elapsed: 33.05s (35.25 tps) | TTFT: 996.2ms | Top-2: [11, 13, 14, 4] (H=2.156)

```

### 🔹 Run `2026-11-06 12:42:24` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1244 | Elapsed: 50.38s (24.69 tps) | TTFT: 2045.9ms | Top-2: [13, 11, 2, 15, 9] (H=2.125)

```

### 🔹 Run `2026-11-06 12:43:14` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1251 | Elapsed: 48.09s (26.02 tps) | TTFT: 2059.4ms | Top-2: [7, 0, 1, 15] (H=2.134)

```

### 🔹 Run `2026-11-06 12:43:48` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1124 | Elapsed: 32.64s (34.43 tps) | TTFT: 1080.9ms | Top-2: [7, 0, 8] (H=2.151)

```

### 🔹 Run `2026-11-06 12:44:37` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1246 | Elapsed: 47.66s (26.15 tps) | TTFT: 2081.9ms | Top-2: [9, 10, 2, 4] (H=2.127)

```

### 🔹 Run `2026-11-06 12:45:25` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1228 | Elapsed: 46.65s (26.32 tps) | TTFT: 2056.3ms | Top-2: [1, 14, 8, 15] (H=2.125)

```

### 🔹 Run `2026-11-06 12:46:12` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1139 | Elapsed: 44.33s (25.69 tps) | TTFT: 2078.9ms | Top-2: [15, 13, 10, 6, 3] (H=2.121)

```

### 🔹 Run `2026-11-06 12:47:06` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1357 | Elapsed: 52.37s (25.91 tps) | TTFT: 2048.9ms | Top-2: [7, 0, 14, 1, 10] (H=2.131)

```

### 🔹 Run `2026-11-06 12:54:35` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1110 | Elapsed: 41.25s (26.91 tps) | TTFT: 408.7ms | Top-2: [7, 0, 1, 10, 13] (H=2.137)

```

### 🔹 Run `2026-11-06 12:55:17` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1077 | Elapsed: 39.69s (27.13 tps) | TTFT: 394.9ms | Top-2: [7, 14, 10, 8] (H=2.142)

```

### 🔹 Run `2026-11-06 12:55:51` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 820 | Elapsed: 32.15s (25.50 tps) | TTFT: 440.8ms | Top-2: [7, 14, 1, 10] (H=2.136)

```

### 🔹 Run `2026-11-06 12:56:27` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 872 | Elapsed: 34.46s (25.30 tps) | TTFT: 398.2ms | Top-2: [7, 0, 13, 3] (H=2.155)

```

### 🔹 Run `2026-11-06 12:57:04` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 906 | Elapsed: 34.44s (26.31 tps) | TTFT: 418.6ms | Top-2: [7, 0, 14, 1, 13] (H=2.134)

```

### 🔹 Run `2026-11-06 12:57:36` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 772 | Elapsed: 30.96s (24.94 tps) | TTFT: 396.7ms | Top-2: [7, 1, 14, 8, 15] (H=2.135)

```

### 🔹 Run `2026-11-06 12:58:03` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 807 | Elapsed: 25.20s (32.02 tps) | TTFT: 810.3ms | Top-2: [14, 10, 8, 15] (H=2.167)

```

### 🔹 Run `2026-11-06 12:58:48` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1180 | Elapsed: 43.20s (27.32 tps) | TTFT: 395.4ms | Top-2: [7, 0, 1, 8, 15] (H=2.133)

```

### 🔹 Run `2026-11-06 12:59:32` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1107 | Elapsed: 42.40s (26.11 tps) | TTFT: 397.7ms | Top-2: [7, 1, 8, 13] (H=2.144)

```

### 🔹 Run `2026-11-06 13:00:10` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 915 | Elapsed: 35.47s (25.79 tps) | TTFT: 462.0ms | Top-2: [11, 9, 0, 13, 4] (H=2.149)

```

### 🔹 Run `2026-11-06 13:00:45` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 943 | Elapsed: 33.54s (28.12 tps) | TTFT: 425.7ms | Top-2: [7, 0, 1, 8, 13] (H=2.137)

```

### 🔹 Run `2026-11-06 13:12:21` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1422 | Elapsed: 51.88s (27.41 tps) | TTFT: 450.8ms | Top-2: [13, 15, 9, 12] (H=2.124)

```

### 🔹 Run `2026-11-06 13:13:09` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1482 | Elapsed: 46.82s (31.65 tps) | TTFT: 1094.5ms | Top-2: [7, 14, 1, 10, 13] (H=2.151)

```

### 🔹 Run `2026-11-06 13:13:49` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1337 | Elapsed: 38.76s (34.50 tps) | TTFT: 1091.1ms | Top-2: [7, 14, 1, 10, 8] (H=2.143)

```

### 🔹 Run `2026-11-06 13:14:30` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1351 | Elapsed: 39.29s (34.38 tps) | TTFT: 1046.2ms | Top-2: [11, 2, 9, 0, 13] (H=2.154)

```

### 🔹 Run `2026-11-06 13:15:22` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1471 | Elapsed: 50.57s (29.09 tps) | TTFT: 449.8ms | Top-2: [7, 0, 14, 1, 10] (H=2.116)

```

### 🔹 Run `2026-11-06 13:16:10` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1449 | Elapsed: 46.29s (31.30 tps) | TTFT: 1133.3ms | Top-2: [0, 10, 8, 13, 3] (H=2.146)

```

### 🔹 Run `2026-11-06 13:16:56` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1461 | Elapsed: 45.19s (32.33 tps) | TTFT: 1103.3ms | Top-2: [7, 0, 14, 1, 8] (H=2.152)

```

### 🔹 Run `2026-11-06 13:17:42` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1480 | Elapsed: 43.94s (33.68 tps) | TTFT: 1091.3ms | Top-2: [7, 0, 14, 1, 8] (H=2.146)

```

### 🔹 Run `2026-11-06 13:18:36` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1464 | Elapsed: 52.18s (28.05 tps) | TTFT: 443.4ms | Top-2: [7, 0, 14, 10] (H=2.116)

```

### 🔹 Run `2026-11-06 13:19:27` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1537 | Elapsed: 49.97s (30.76 tps) | TTFT: 1119.5ms | Top-2: [0, 14, 1, 10, 8] (H=2.149)

```

### 🔹 Run `2026-11-06 13:20:23` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1153 | Elapsed: 41.84s (27.55 tps) | TTFT: 450.4ms | Top-2: [7, 0, 1, 10, 8] (H=2.127)

```

### 🔹 Run `2026-11-06 13:21:05` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1066 | Elapsed: 38.65s (27.58 tps) | TTFT: 1500.5ms | Top-2: [14, 10, 6] (H=2.127)

```

### 🔹 Run `2026-11-06 13:21:51` (`thor_31b_20261106_105639` | `openai-api-hybrid`)
```text
🏁 [Hybrid Turn Complete] Tokens: 1191 | Elapsed: 42.93s (27.74 tps) | TTFT: 459.8ms | Top-2: [10, 8, 13, 3] (H=2.125)

```
