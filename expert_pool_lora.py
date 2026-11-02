import torch
import torch.nn as nn

class DynamicLoRAExpertPool(nn.Module):
    """
    Manages 8 Embodied LoRA Experts with Zero-Copy GPU allocation.
    Each expert maintains:
      - lora_a: [rank, in_dim]
      - lora_b: [out_dim, rank]
    """
    def __init__(self, num_experts=8, in_dim=3840, out_dim=3840, rank=16, alpha=16.0, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.num_experts = num_experts
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scaling = alpha / rank
        self.device = device
        self.dtype = dtype

        self.lora_a = nn.ParameterDict()
        self.lora_b = nn.ParameterDict()
        self._ltm_baseline_a = {}
        self.eligibility_trace_a = {}

        for i in range(num_experts):
            la = nn.Parameter(torch.empty(rank, in_dim, device=device, dtype=dtype))
            lb = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(la, a=5**0.5)
            self.lora_a[f"exp_{i}"] = la
            self.lora_b[f"exp_{i}"] = lb
            self._ltm_baseline_a[f"exp_{i}"] = la.data.detach().clone()
            self.eligibility_trace_a[f"exp_{i}"] = torch.zeros_like(la.data)

    def begin_eligibility_episode(self):
        """
        [Synaptic Tagging & Capture - Episode Start]
        Snapshots current Long-Term Memory (LTM) weights `lora_a` and resets the transient
        Short-Term Plasticity (STP) eligibility trace buffer before token generation begins.
        """
        with torch.no_grad():
            for i in range(self.num_experts):
                k = f"exp_{i}"
                if k in self.lora_a:
                    self._ltm_baseline_a[k] = self.lora_a[k].data.detach().clone()
                    self.eligibility_trace_a[k] = torch.zeros_like(self.lora_a[k].data)

    def accumulate_eligibility_trace(self, exp_key: str, pre_step_a: torch.Tensor, decay: float = 0.992):
        """
        Records single-step Wake Hebbian LTP change ΔA_t into the decaying eligibility trace e_t.
        """
        with torch.no_grad():
            if exp_key in self.lora_a:
                if exp_key not in self.eligibility_trace_a:
                    self.eligibility_trace_a[exp_key] = torch.zeros_like(self.lora_a[exp_key].data)
                delta = self.lora_a[exp_key].data - pre_step_a
                self.eligibility_trace_a[exp_key].mul_(decay).add_(delta)

    def settle_eligibility_trace(self, sandbox_passed: bool | None = None) -> str:
        """
        [Dual-Layer Plasticity Settlement]
          - sandbox_passed is True  (EXIT_CODE == 0): Dopaminergic consolidation -> locks STP trace into LTM!
          - sandbox_passed is False (EXIT_CODE != 0): Anti-Hebbian reversal -> rolls back to LTM baseline
            and applies negative trace penalty so failed code attempts never pollute checkpoints!
          - sandbox_passed is None  (Conversational): Smoothly commits current wake homeostasis to LTM.
        """
        with torch.no_grad():
            for i in range(self.num_experts):
                k = f"exp_{i}"
                if k not in self.lora_a:
                    continue
                la = self.lora_a[k].data
                base = self._ltm_baseline_a.get(k)
                trace = self.eligibility_trace_a.get(k)
                if base is None or trace is None:
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k] = torch.zeros_like(la)
                    continue

                if sandbox_passed is True:
                    # Dopaminergic Consolidation: reinforce causal trace & lock into LTM
                    nz_mask = (la != 0).to(la.dtype)
                    la.add_((0.15 * trace) * nz_mask)
                    nrm = float(torch.norm(la.float()).item())
                    if nrm > 2.80:
                        la.mul_(2.80 / nrm)
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k].zero_()
                elif sandbox_passed is False:
                    # Anti-Hebbian Rollback + Immediate Triton 2:4 Structural Prune on Active Faulty Channels:
                    # 1) Subtract 0.25 * E_i(t) from clean LTM baseline to weaken the exact channels that caused the bug
                    # 2) Immediately invoke Triton 2:4 Fisher pruning so weakened buggy channels lose the Top-2 quad competition and become 0.0!
                    trace_energy = float(torch.norm(trace.float()).item())
                    nz_mask = (base != 0).to(la.dtype)
                    reverted = (base - 0.25 * trace) * nz_mask
                    nrm = float(torch.norm(reverted.float()).item())
                    if 0.0 < nrm < 0.65:
                        reverted.mul_(0.65 / nrm)
                    elif nrm > 2.80:
                        reverted.mul_(2.80 / nrm)
                    la.copy_(reverted)
                    if trace_energy > 1e-6:
                        try:
                            from triton_localsleep_kernels import launch_triton_fisher_24_prune
                            launch_triton_fisher_24_prune(la, shy_lambda=0.08, survival_floor=0.985)
                        except Exception:
                            pass
                    # Keep baseline clean for the retry attempt
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k].zero_()
                else:
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k].zero_()

        if sandbox_passed is True:
            return "LTM_CONSOLIDATED (EXIT_CODE=0)"
        if sandbox_passed is False:
            return "ANTI_HEBBIAN_ROLLBACK (EXIT_CODE!=0)"
        return "LTM_COMMITTED"

    def forward_delta(self, x, selected_expert_idx, weight=1.0):
        """
        Computes residual LoRA delta: delta = weight * scaling * (x @ A^T) @ B^T
        Shape: [batch, seq, hidden_dim]
        """
        idx_key = f"exp_{selected_expert_idx}"
        if idx_key not in self.lora_a:
            return torch.zeros_like(x)

        A = self.lora_a[idx_key]
        B = self.lora_b[idx_key]

        # x: [B, H] -> [B, R] -> [B, H]
        h = torch.matmul(x, A.t())
        delta = torch.matmul(h, B.t()) * (self.scaling * weight)
        return delta

    def export_tensors(self):
        """Returns tensor dictionary ready for SafeTensors serialization."""
        tensors = {}
        for i in range(self.num_experts):
            tensors[f"expert.{i}.lora_a"] = self.lora_a[f"exp_{i}"].data.detach().cpu().float()
            tensors[f"expert.{i}.lora_b"] = self.lora_b[f"exp_{i}"].data.detach().cpu().float()
        return tensors

    def load_tensors(self, tensor_dict):
        """Zero-copy / direct loading of tensors back into GPU memory."""
        for i in range(self.num_experts):
            ka = f"expert.{i}.lora_a"
            kb = f"expert.{i}.lora_b"
            if ka in tensor_dict:
                loaded_a = tensor_dict[ka].to(self.device, dtype=self.dtype)
                # Self-Healing Synaptic Floor: if a saved checkpoint was previously collapsed to 0.0,
                # automatically re-seed Kaiming uniform weights so the expert never stays dead!
                if float(torch.norm(loaded_a.float()).item()) < 1e-4:
                    nn.init.kaiming_uniform_(self.lora_a[f"exp_{i}"], a=5**0.5)
                else:
                    self.lora_a[f"exp_{i}"].data.copy_(loaded_a)
                self._ltm_baseline_a[f"exp_{i}"] = self.lora_a[f"exp_{i}"].data.detach().clone()
                self.eligibility_trace_a[f"exp_{i}"] = torch.zeros_like(self.lora_a[f"exp_{i}"].data)
            if kb in tensor_dict:
                self.lora_b[f"exp_{i}"].data.copy_(tensor_dict[kb].to(self.device, dtype=self.dtype))
