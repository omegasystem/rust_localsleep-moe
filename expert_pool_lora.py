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
            with torch.no_grad():
                # Immediately enforce initial 2:4 structural sparsity (50% zeros) and homeostatic norm μ0=1.32
                # on Virgin Reset so fresh weights never spike live_gate biases or trigger EAGLE3 draft rejections!
                if in_dim % 4 == 0:
                    q = la.data.float().view(rank, in_dim // 4, 4)
                    _, top2 = torch.topk(torch.abs(q), k=2, dim=-1)
                    mask = torch.zeros_like(q, dtype=torch.bool).scatter_(-1, top2, True)
                    la.data.copy_(torch.where(mask, q, torch.zeros_like(q)).reshape(rank, in_dim).to(dtype))
                init_norm = float(torch.norm(la.data.float()).item())
                if init_norm > 1e-6:
                    target_init = 1.28 + 0.02 * (i % 5)
                    la.data.mul_(target_init / init_norm)
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

    @staticmethod
    def apply_incoherent_hadamard_dispersion(la: torch.Tensor, mix: float = 0.06) -> None:
        """
        [In-Place Fast Walsh-Hadamard Orthonormal Quad Dispersion (FWHT)]
        Blends a 4x4 orthonormal Walsh-Hadamard transform (H_4^T H_4 = I_4) across each
        4-element quad prior to Triton 2:4 Fisher pruning. Prevents localized channel polarization
        and 4-hot-channel quad collisions while preserving Frobenius norm and eRank.
        """
        with torch.no_grad():
            if la.ndim != 2 or (la.shape[1] % 4) != 0:
                return
            r, c = la.shape
            quads = la.float().view(r, c // 4, 4)
            x0, x1, x2, x3 = quads[..., 0], quads[..., 1], quads[..., 2], quads[..., 3]
            h0 = 0.5 * (x0 + x1 + x2 + x3)
            h1 = 0.5 * (x0 - x1 + x2 - x3)
            h2 = 0.5 * (x0 + x1 - x2 - x3)
            h3 = 0.5 * (x0 - x1 - x2 + x3)
            had = torch.stack([h0, h1, h2, h3], dim=-1)
            blended = ((1.0 - mix) * quads + mix * had).reshape(r, c)
            la.copy_(blended.to(la.dtype))

    def accumulate_eligibility_trace(
        self,
        exp_key: str,
        pre_step_a: torch.Tensor,
        decay: float = 0.9985,
        is_syntax_boundary: bool = False,
        gamma_syntax: float = 0.935,
    ):
        """
        [Hybrid Dual-Scale Eligibility Trace Decay]
          E_i(t) = (γ_token * γ_syntax^I(is_boundary)) * E_i(t-1) + ΔA_t
          - γ_token  = 0.9985 (eliminates token-length verbosity bias within a single statement)
          - γ_syntax = 0.9350 (step-decays on \\n, }, ; boundaries; half-life ~10 statements)
        """
        with torch.no_grad():
            if exp_key in self.lora_a:
                if exp_key not in self.eligibility_trace_a:
                    self.eligibility_trace_a[exp_key] = torch.zeros_like(self.lora_a[exp_key].data)
                eff_decay = float(decay) * (float(gamma_syntax) if is_syntax_boundary else 1.0)
                delta = self.lora_a[exp_key].data - pre_step_a
                self.eligibility_trace_a[exp_key].mul_(eff_decay).add_(delta)

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
                    # Dopaminergic Consolidation: reinforce causal trace on CURRENT 2:4 mask (preserving newly re-ignited synapses!)
                    nz_mask = (la != 0).to(la.dtype)
                    la.add_((0.15 * trace) * nz_mask)
                    nrm = float(torch.norm(la.float()).item())
                    if 0.0 < nrm < 1.05:
                        la.mul_(1.15 / nrm)
                    elif nrm > 2.80:
                        la.mul_(2.80 / nrm)
                    self._ltm_baseline_a[k] = la.detach().clone()
                    self.eligibility_trace_a[k].zero_()
                elif sandbox_passed is False:
                    # Anti-Hebbian Rollback + Active Mask Escape:
                    # 1) Weaken channels that fired during the failed attempt while preserving current `la` mask flips
                    # 2) Apply stronger Hadamard dispersion (mix=0.36) on active fault channels before Triton 2:4 pruning
                    #    so dead synapses in faulty quads overtake weakened survivors and flip the 2:4 mask!
                    trace_energy = float(torch.norm(trace.float()).item())
                    cur_nz_mask = (la != 0).to(la.dtype)
                    reverted = (la - 0.30 * trace) * cur_nz_mask
                    nrm = float(torch.norm(reverted.float()).item())
                    if 0.0 < nrm < 1.05:
                        reverted.mul_(1.18 / nrm)
                    elif nrm > 2.80:
                        reverted.mul_(2.80 / nrm)
                    la.copy_(reverted)
                    if trace_energy > 1e-6:
                        try:
                            from triton_localsleep_kernels import launch_triton_fisher_24_prune
                            self.apply_incoherent_hadamard_dispersion(la, mix=0.36)
                            launch_triton_fisher_24_prune(la, shy_lambda=0.05, survival_floor=0.992)
                        except Exception:
                            pass
                    # Keep baseline updated for the retry attempt
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
