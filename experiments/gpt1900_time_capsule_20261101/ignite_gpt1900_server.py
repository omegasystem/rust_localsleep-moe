#!/usr/bin/env python3
"""
ignite_gpt1900_server.py
==============================================================================
Option A: OpenAI & TensorRT-Edge-LLM Compatible Server for GPT-1900 (3.29B)
with Hot-Swappable In-VRAM Live MoE Mailbox (/v1/moe/live_gate)
==============================================================================
Loads `mhla/gpt1900-instruct-v3-sft` (`model_000075.pt` + `meta_000075.json` +
`tokenizer/tokenizer.pkl`) natively on RTX 5080 in BF16 with PyTorch SDPA
KV-Cache acceleration, and exposes:
  - GET  /health
  - GET  /v1/moe/live_gate
  - POST /v1/moe/live_gate       (Zero-restart in-VRAM logit_bias injection!)
  - POST /v1/chat/completions    (OpenAI SSE streaming & non-streaming)
"""

import argparse
import asyncio
import json
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn


# ==============================================================================
# 1. NanoChat / GPT-1900 Model Architecture (Self-Contained with SDPA KV-Cache)
# ==============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 34
    n_head: int = 17
    n_kv_head: int = 17
    n_embd: int = 2176
    window_pattern: str = "SSSL"


def norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class LayerKVCache:
    """Fast GPU VRAM KV-Cache for 34-Layer GPT-1900."""
    def __init__(self, n_layer: int):
        self.k_cache: List[Optional[torch.Tensor]] = [None] * n_layer
        self.v_cache: List[Optional[torch.Tensor]] = [None] * n_layer
        self.pos: int = 0

    def get_pos(self) -> int:
        return self.pos

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, max_window: int):
        # k, v shape: (B, T, H_kv, D)
        if self.k_cache[layer_idx] is None:
            self.k_cache[layer_idx] = k
            self.v_cache[layer_idx] = v
        else:
            self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=1)
            self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=1)
        if max_window > 0 and self.k_cache[layer_idx].size(1) > max_window:
            self.k_cache[layer_idx] = self.k_cache[layer_idx][:, -max_window:, :, :]
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, -max_window:, :, :]
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, n_tokens: int):
        self.pos += n_tokens


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor], cos_sin, window_size, kv_cache: Optional[LayerKVCache]):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2.0 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q = norm(apply_rotary_emb(q, cos, sin))
        k = norm(apply_rotary_emb(k, cos, sin))

        win_left = window_size[0] if isinstance(window_size, tuple) else -1
        if kv_cache is not None:
            k_full, v_full = kv_cache.append(self.layer_idx, k, v, max_window=win_left)
        else:
            k_full, v_full = k, v

        # Transpose to (B, H, T, D) for PyTorch Flash/SDPA
        q_t = q.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v_full.transpose(1, 2)

        is_causal = (T > 1) and (kv_cache is None or kv_cache.get_pos() == 0)
        y = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            is_causal=is_causal,
            enable_gqa=(self.n_head != self.n_kv_head),
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, ve: Optional[torch.Tensor], cos_sin, window_size, kv_cache: Optional[LayerKVCache]):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT1900Model(nn.Module):
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = max(config.sequence_len * 10, 16384)
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len: int, head_dim: int, base: float = 10000.0, device=None):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def refresh_rotary_on_device(self, device):
        head_dim = self.config.n_embd // self.config.n_head
        self.cos, self.sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, device=device)

    def _compute_window_sizes(self, config: GPTConfig):
        pattern = config.window_pattern.upper()
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = [char_to_window[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def forward(self, idx: torch.Tensor, kv_cache: Optional[LayerKVCache] = None) -> torch.Tensor:
        B, T = idx.size()
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]

        x = norm(self.transformer.wte(idx))
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        if kv_cache is not None:
            kv_cache.advance(T)
        x = norm(x)

        softcap = 15.0
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = softcap * torch.tanh(logits / softcap)
        return logits


# ==============================================================================
# 2. RustBPE / Tiktoken Tokenizer Loader
# ==============================================================================

class GPT1900Tokenizer:
    def __init__(self, tokenizer_dir: str):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            self.enc = pickle.load(f)
        self.vocab_size = self.enc.n_vocab
        self.bos_id = self.encode_special("<|bos|>")
        self.user_start_id = self.encode_special("<|user_start|>")
        self.user_end_id = self.encode_special("<|user_end|>")
        self.assistant_start_id = self.encode_special("<|assistant_start|>")
        self.assistant_end_id = self.encode_special("<|assistant_end|>")

    def encode_special(self, token_str: str) -> int:
        return self.enc.encode_single_token(token_str)

    def encode(self, text: str) -> List[int]:
        return self.enc.encode_ordinary(text)

    def decode(self, token_ids: List[int]) -> str:
        valid_ids = [t for t in token_ids if 0 <= int(t) < self.vocab_size]
        try:
            return self.enc.decode(valid_ids)
        except Exception:
            return ""

    def format_messages(self, messages: List[Dict[str, Any]], max_context_tokens: int = 1700) -> List[int]:
        # Merge leading system message into first user message (exact nanochat convention)
        msgs = [dict(m) for m in messages if m.get("content")]
        if len(msgs) >= 2 and msgs[0].get("role") == "system":
            sys_txt = str(msgs[0]["content"]).strip()
            msgs = msgs[1:]
            if msgs and msgs[0].get("role") == "user":
                msgs[0]["content"] = f"{sys_txt}\n\n{msgs[0]['content']}"

        ids: List[int] = [self.bos_id]
        for m in msgs:
            role = m.get("role", "user")
            content = str(m.get("content", ""))
            content_ids = self.encode(content)
            if role == "user":
                ids.extend([self.user_start_id] + content_ids + [self.user_end_id])
            elif role == "assistant":
                ids.extend([self.assistant_start_id] + content_ids + [self.assistant_end_id])

        # Sliding window crop if conversation exceeds context window
        if len(ids) > max_context_tokens:
            ids = [self.bos_id] + ids[-(max_context_tokens - 1):]

        ids.append(self.assistant_start_id)
        return ids


# ==============================================================================
# 3. Live VRAM MoE Gate State (100% Compatible with TensorRT C++ /v1/moe/live_gate)
# ==============================================================================

class LiveMoEVramMailbox:
    def __init__(self, vocab_size: int, device: torch.device):
        self.vocab_size = vocab_size
        self.device = device
        self.lock = threading.Lock()
        self.bias_gpu = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
        self.oracle_bias_gpu = torch.zeros((1, vocab_size), dtype=torch.float32, device=device)
        self.alpha: float = 0.0
        self.active_count: int = 0
        self.oracle_active_count: int = 0
        self.update_seq: int = 0

    def reset(self, clear_oracle: bool = False):
        with self.lock:
            self.bias_gpu.zero_()
            self.alpha = 0.0
            self.active_count = 0
            if clear_oracle:
                self.oracle_bias_gpu.zero_()
                self.oracle_active_count = 0
            self.update_seq += 1

    def update(
        self,
        logit_bias: Dict[str, float],
        alpha: float = 1.0,
        reset_first: bool = True,
        token_strings: Optional[Dict[str, float]] = None,
        tokenizer: Optional[GPT1900Tokenizer] = None,
        clear_oracle: bool = False,
    ):
        with self.lock:
            if clear_oracle:
                self.oracle_bias_gpu.zero_()
                self.oracle_active_count = 0
            if reset_first and logit_bias:
                self.bias_gpu.zero_()
                self.active_count = 0
            count = 0
            for k_str, val in logit_bias.items():
                try:
                    tid = int(k_str)
                except ValueError:
                    continue
                if tid < 0:
                    continue
                if tid >= self.vocab_size:
                    tid = tid % self.vocab_size
                self.bias_gpu[0, tid] = float(val)
                count += 1
            if logit_bias:
                self.active_count = count if reset_first else (self.active_count + count)

            if token_strings and tokenizer is not None:
                self.oracle_bias_gpu.zero_()
                o_cnt = 0
                for word_str, val in token_strings.items():
                    for cand in (" " + word_str.strip(), word_str.strip()):
                        if not cand.strip():
                            continue
                        tids = tokenizer.encode(cand)
                        # Apply bias to the leading subword token of the concept word
                        if tids:
                            tid0 = tids[0]
                            if 0 <= tid0 < self.vocab_size:
                                self.oracle_bias_gpu[0, tid0] = float(val)
                                o_cnt += 1
                self.oracle_active_count = o_cnt

            self.alpha = float(alpha)
            self.update_seq += 1
            return self.get_status_unlocked()

    def apply_to_logits(self, logits: torch.Tensor):
        with self.lock:
            if self.alpha != 0.0 and self.active_count > 0:
                logits.add_(self.bias_gpu, alpha=self.alpha)
            if self.oracle_active_count > 0:
                logits.add_(self.oracle_bias_gpu)

    def get_status_unlocked(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "active_bias_tokens": self.active_count,
            "oracle_bias_tokens": self.oracle_active_count,
            "alpha": self.alpha,
            "update_seq": self.update_seq,
            "vocab_size": self.vocab_size,
            "engine": "gpt1900_pytorch_sdpa_vram_mailbox",
        }

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return self.get_status_unlocked()


# ==============================================================================
# 4. FastAPI Server Application
# ==============================================================================

app = FastAPI(title="GPT-1900 (Machina Mirabilis) TensorRT-Compatible Server")
GLOBAL_STATE: Dict[str, Any] = {}


def resolve_model_dir(user_path: str) -> str:
    candidates = [
        user_path,
        user_path.replace("/home/omegacd/", "/home/node/"),
        user_path.replace("/home/node/", "/home/omegacd/"),
        "/home/omegacd/.openclaw/workspace/models/gpt1900-instruct-v3-sft",
        "/home/node/.openclaw/workspace/models/gpt1900-instruct-v3-sft",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return user_path


def load_gpt1900_engine(model_dir: str, device_str: str = "cuda"):
    resolved_dir = resolve_model_dir(model_dir)
    print(f"🕰️ [GPT-1900 Server] Loading from directory: {resolved_dir}")

    meta_files = [f for f in os.listdir(resolved_dir) if f.startswith("meta_") and f.endswith(".json")]
    pt_files = [f for f in os.listdir(resolved_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not meta_files or not pt_files:
        raise FileNotFoundError(f"Could not find meta_*.json or model_*.pt in {resolved_dir}")

    meta_path = os.path.join(resolved_dir, sorted(meta_files)[-1])
    pt_path = os.path.join(resolved_dir, sorted(pt_files)[-1])
    tok_dir = os.path.join(resolved_dir, "tokenizer")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = GPTConfig(**meta["model_config"])
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    tokenizer = GPT1900Tokenizer(tok_dir)
    print(f"🕰️ [GPT-1900 Server] Loading & casting checkpoint {pt_path} to BF16 on CPU first (prevents 21GB FP32 double-allocation spike!)...")
    raw_sd = torch.load(pt_path, map_location="cpu", weights_only=True)
    state_dict = {}
    for k, v in raw_sd.items():
        clean_k = k.removeprefix("_orig_mod.")
        state_dict[clean_k] = v.to(dtype=torch.bfloat16) if ( isinstance(v, torch.Tensor) and v.is_floating_point() ) else v
    del raw_sd

    with torch.device("meta"):
        model = GPT1900Model(cfg)
        model.to(dtype=torch.bfloat16)
    model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict
    model.to(device=device, dtype=torch.bfloat16)
    model.refresh_rotary_on_device(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mailbox = LiveMoEVramMailbox(vocab_size=cfg.vocab_size, device=device)
    GLOBAL_STATE["model"] = model
    GLOBAL_STATE["tokenizer"] = tokenizer
    GLOBAL_STATE["config"] = cfg
    GLOBAL_STATE["device"] = device
    GLOBAL_STATE["mailbox"] = mailbox
    GLOBAL_STATE["model_dir"] = resolved_dir
    print(
        f"✅ [GPT-1900 Ready] Params: 3.29B | Layers: {cfg.n_layer} | Hidden: {cfg.n_embd} | "
        f"Vocab: {cfg.vocab_size} | Live VRAM Mailbox Active!"
    )


@app.get("/health")
async def health():
    mailbox: Optional[LiveMoEVramMailbox] = GLOBAL_STATE.get("mailbox")
    return {
        "status": "ok",
        "model": "gpt1900-instruct-v3-sft",
        "live_gate": mailbox.get_status() if mailbox else {},
    }


@app.get("/v1/moe/live_gate")
async def get_moe_live_gate():
    mailbox: LiveMoEVramMailbox = GLOBAL_STATE["mailbox"]
    return mailbox.get_status()


@app.post("/v1/moe/live_gate")
async def post_moe_live_gate(req: Request):
    body = await req.json()
    mailbox: LiveMoEVramMailbox = GLOBAL_STATE["mailbox"]
    tokenizer: GPT1900Tokenizer = GLOBAL_STATE["tokenizer"]
    if body.get("reset", False) and not body.get("logit_bias") and not body.get("token_strings"):
        mailbox.reset()
        return mailbox.get_status()

    if "enabled" in body and not bool(body["enabled"]):
        mailbox.reset()
        return mailbox.get_status()

    logit_bias = body.get("logit_bias", {}) or {}
    token_strings = body.get("token_strings", {}) or {}
    alpha = float(body.get("alpha", 1.0))
    reset_first = bool(body.get("reset_first", True))
    status = mailbox.update(
        logit_bias=logit_bias,
        alpha=alpha,
        reset_first=reset_first,
        token_strings=token_strings,
        tokenizer=tokenizer,
    )
    return status


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    body = await req.json()
    messages = body.get("messages", [])
    max_tokens = min(int(body.get("max_tokens", 512)), 1536)
    temperature = float(body.get("temperature", 0.6))
    top_k = int(body.get("top_k", 50))
    stream = bool(body.get("stream", False))
    req_logit_bias = body.get("logit_bias", None)

    model: GPT1900Model = GLOBAL_STATE["model"]
    tokenizer: GPT1900Tokenizer = GLOBAL_STATE["tokenizer"]
    mailbox: LiveMoEVramMailbox = GLOBAL_STATE["mailbox"]
    device: torch.device = GLOBAL_STATE["device"]

    # If this is the first turn of a request (only 1 user message, no assistant reply yet),
    # clear any previous turn's Oracle word biases so each new experiment starts fresh.
    if sum(1 for m in messages if m.get("role") == "assistant") == 0:
        mailbox.reset(clear_oracle=True)

    prompt_ids = tokenizer.format_messages(messages, max_context_tokens=max(512, 2048 - max_tokens - 16))
    stop_token_ids = {
        tokenizer.assistant_end_id,
        tokenizer.user_start_id,
        tokenizer.bos_id,
    }

    req_id = f"chatcmpl-gpt1900-{uuid.uuid4().hex[:10]}"
    created_ts = int(time.time())

    if not stream:
        out_tokens: List[int] = []
        kv_cache = LayerKVCache(model.config.n_layer)
        curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for _ in range(max_tokens):
                logits = model(curr_ids, kv_cache=kv_cache)[:, -1, :]
                mailbox.apply_to_logits(logits)
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                if temperature > 1e-4:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)
                tid = int(next_id.item())
                if tid in stop_token_ids:
                    break
                out_tokens.append(tid)
                curr_ids = next_id
        text_out = tokenizer.decode(out_tokens)
        return JSONResponse({
            "id": req_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": "gpt1900-instruct-v3-sft",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text_out},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_tokens),
                "total_tokens": len(prompt_ids) + len(out_tokens),
            },
        })

    async def sse_generator():
        kv_cache = LayerKVCache(model.config.n_layer)
        curr_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        comp_tokens = 0
        pending_ids: List[int] = []

        for step in range(max_tokens):
            with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(curr_ids, kv_cache=kv_cache)[:, -1, :]
                mailbox.apply_to_logits(logits)
                log_probs_full = F.log_softmax(logits, dim=-1)
                top4_vals, top4_ids = torch.topk(log_probs_full, k=4, dim=-1)
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                if temperature > 1e-4:
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = torch.argmax(logits, dim=-1, keepdim=True)

            tid = int(next_id.item())
            if tid in stop_token_ids:
                break

            chosen_lp = float(log_probs_full[0, tid].item())
            top_lps_list = [
                {
                    "token_id": int(top4_ids[0, r].item()),
                    "logprob": float(top4_vals[0, r].item()),
                }
                for r in range(4)
            ]

            comp_tokens += 1
            pending_ids.append(tid)
            piece = tokenizer.decode(pending_ids)
            if piece and not piece.endswith("\ufffd"):
                pending_ids.clear()
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": "gpt1900-instruct-v3-sft",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "logprobs": {
                            "content": [{
                                "token": piece,
                                "token_id": tid,
                                "logprob": chosen_lp,
                                "top_logprobs": top_lps_list,
                            }]
                        },
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            curr_ids = next_id
            if step % 4 == 0:
                await asyncio.sleep(0)

        if pending_ids:
            piece = tokenizer.decode(pending_ids)
            if piece:
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": "gpt1900-instruct-v3-sft",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": "gpt1900-instruct-v3-sft",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": comp_tokens,
                "total_tokens": len(prompt_ids) + comp_tokens,
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-1900 TensorRT-Compatible Live Gate Server")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/home/omegacd/.openclaw/workspace/models/gpt1900-instruct-v3-sft",
        help="Path to gpt1900-instruct-v3-sft directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    load_gpt1900_engine(args.model_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
