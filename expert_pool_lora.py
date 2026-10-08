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

        for i in range(num_experts):
            la = nn.Parameter(torch.empty(rank, in_dim, device=device, dtype=dtype))
            lb = nn.Parameter(torch.zeros(out_dim, rank, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(la, a=5**0.5)
            self.lora_a[f"exp_{i}"] = la
            self.lora_b[f"exp_{i}"] = lb

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
                self.lora_a[f"exp_{i}"].data.copy_(tensor_dict[ka].to(self.device, dtype=self.dtype))
            if kb in tensor_dict:
                self.lora_b[f"exp_{i}"].data.copy_(tensor_dict[kb].to(self.device, dtype=self.dtype))
