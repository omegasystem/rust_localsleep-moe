import os
import json
from datetime import datetime

try:
    from safetensors.torch import save_file as save_safetensors, load_file as load_safetensors
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

class EmbodiedCheckpointManager:
    """
    Manages Zero-Copy SafeTensors persistence for:
      1. DynamicLoRAExpertPool (8 Experts' LoRA A and B weights)
      2. Rust Router Homeostasis State (fatigue_scores, active_tokens, states, gate_weights)
      3. Session Metadata Header (tokens_generated, profile, session_id, timestamp)
    """
    def __init__(self, checkpoint_dir="./checkpoints", profile="thor_12b"):
        self.checkpoint_dir = checkpoint_dir
        self.profile = profile
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.latest_st_path = os.path.join(self.checkpoint_dir, f"latest_{self.profile}.safetensors")
        self.latest_json_path = os.path.join(self.checkpoint_dir, f"latest_{self.profile}.json")

    def load_state(self, rust_router, expert_pool=None, reset=False):
        """
        Restores state into Rust router and Expert LoRA pool.
        Prefers SafeTensors (mmap Zero-Copy), falls back to Legacy JSON.
        """
        if reset:
            print(f"🔄 \033[1;33m[Virgin Reset]\033[0m Starting with virgin state (--reset supplied).")
            return 0

        # 1. Try SafeTensors
        if SAFETENSORS_AVAILABLE and os.path.exists(self.latest_st_path):
            try:
                tensors = load_safetensors(self.latest_st_path)
                if hasattr(rust_router, "import_state"):
                    rust_router.import_state(
                        tensors["gate_weights"].tolist(),
                        tensors["fatigue_scores"].tolist(),
                        tensors["active_tokens"].tolist(),
                        tensors["states"].tolist()
                    )

                # Restore Expert LoRA tensors to GPU if expert_pool provided
                if expert_pool is not None:
                    expert_pool.load_tensors(tensors)

                meta_tokens = 0
                try:
                    with safe_open(self.latest_st_path, framework="pt") as f:
                        meta = f.metadata()
                        if meta and "tokens_generated" in meta:
                            meta_tokens = int(meta["tokens_generated"])
                except Exception:
                    pass

                print(f"🔄 \033[1;32m[SafeTensors Auto-Resume Active]\033[0m Restored neural state & 8 LoRA Experts from \033[1;36m{self.latest_st_path}\033[0m (Zero-Copy mmap)!")
                return meta_tokens
            except Exception as e:
                print(f"[!] Warning reading SafeTensors checkpoint: {e}")

        # 2. Fallback to Legacy JSON
        if os.path.exists(self.latest_json_path):
            try:
                with open(self.latest_json_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                if hasattr(rust_router, "import_state"):
                    rust_router.import_state(
                        ckpt.get("gate_weights", []),
                        ckpt.get("fatigue_scores", []),
                        ckpt.get("active_tokens", []),
                        ckpt.get("states", [])
                    )
                print(f"🔄 \033[1;32m[Legacy JSON Auto-Resume]\033[0m Restored state from \033[1;36m{self.latest_json_path}\033[0m (Will promote to SafeTensors on save)!")
                return ckpt.get("tokens_generated", 0)
            except Exception as e:
                print(f"[!] Warning reading legacy JSON checkpoint: {e}")

        print("ℹ️  No existing checkpoint found. Initializing fresh router state.")
        return 0

    def save_state(self, rust_router, expert_pool=None, tokens_generated=0):
        """
        Persists both router state and 8 LoRA expert matrices to SafeTensors.
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{timestamp_str}_{os.getpid()}"
        save_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.profile}_{session_id}.safetensors")

        # 1. Export Rust state (PyO3 returns a tuple: (gate_weights, fatigue_scores, active_tokens, states))
        state_raw = rust_router.export_state() if hasattr(rust_router, "export_state") else ()
        if isinstance(state_raw, tuple) and len(state_raw) == 4:
            gw, fs, at, st = state_raw
        elif isinstance(state_raw, dict):
            gw = state_raw.get("gate_weights", [])
            fs = state_raw.get("fatigue_scores", [])
            at = state_raw.get("active_tokens", [])
            st = state_raw.get("states", [])
        else:
            gw, fs, at, st = [], [], [], []

        import torch
        tensors_dict = {
            "gate_weights": torch.tensor(gw, dtype=torch.float32),
            "fatigue_scores": torch.tensor(fs, dtype=torch.float32),
            "active_tokens": torch.tensor(at, dtype=torch.int64),
            "states": torch.tensor(st, dtype=torch.int32),
        }

        # 2. Export 8 Expert LoRA matrices
        if expert_pool is not None:
            lora_tensors = expert_pool.export_tensors()
            tensors_dict.update(lora_tensors)

        metadata = {
            "session_id": str(session_id),
            "profile": str(self.profile),
            "tokens_generated": str(tokens_generated),
            "timestamp": datetime.now().isoformat(),
            "format": "safetensors_embodied_v2"
        }

        if SAFETENSORS_AVAILABLE:
            save_safetensors(tensors_dict, save_path, metadata=metadata)
            save_safetensors(tensors_dict, self.latest_st_path, metadata=metadata)
            print(f"💾 \033[1;32m[SafeTensors Embodied Checkpoint Saved]\033[0m -> \033[1;36m{self.latest_st_path}\033[0m (8 Experts + State)")
        else:
            # Fallback json
            legacy_data = {
                "gate_weights": list(gw),
                "fatigue_scores": list(fs),
                "active_tokens": list(at),
                "states": list(st),
                **metadata
            }
            with open(self.latest_json_path, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f, indent=2)
            print(f"💾 \033[1;33m[Legacy JSON Saved]\033[0m -> {self.latest_json_path}")
