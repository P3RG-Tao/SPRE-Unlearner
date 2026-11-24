"""正式耦合 Replacement：只改写原始问题 token 的目标子空间分量。"""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ReplacementOutput:
    hidden: torch.Tensor
    before: torch.Tensor
    after: torch.Tensor


def masked_mean(hidden, mask):
    weights = mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)


class CoupledReplacement(nn.Module):
    def __init__(
        self,
        *,
        target_basis,
        retain_center,
        replacement_codebook,
        width=32,
        radius=128.0,
        initial_strength=1e-4,
    ):
        super().__init__()
        basis = target_basis.detach().float().clone()
        center = retain_center.detach().float().clone()
        codebook = replacement_codebook.detach().float().clone()
        if basis.ndim != 2 or min(basis.shape) <= 0 or center.shape != (basis.shape[1],):
            raise ValueError("invalid basis or center shape")
        if codebook.ndim != 2 or codebook.shape[1] != len(basis) or len(codebook) < 2:
            raise ValueError("invalid codebook shape")
        if not all(torch.isfinite(v).all() for v in (basis, center, codebook)):
            raise ValueError("geometry must be finite")
        if not torch.allclose(
            basis @ basis.T, torch.eye(len(basis), device=basis.device), atol=1e-4, rtol=1e-4
        ):
            raise ValueError("target basis must be orthonormal")
        if width < 1 or not 0 < radius < float("inf") or not 0 < initial_strength < 1:
            raise ValueError("invalid replacement parameters")
        self.register_buffer("target_basis", F.normalize(basis, dim=1))
        self.register_buffer("retain_center", center)
        self.register_buffer("replacement_codebook", codebook)
        self.radius = float(radius)
        self.token_norm = nn.LayerNorm(basis.shape[1], elementwise_affine=False)
        # 顺序与旧类一致，保证相同 seed 下随机初始化一致。
        self.replacement_down = nn.Linear(basis.shape[1], width, bias=False)
        self.replacement_up = nn.Linear(width, len(codebook), bias=False)
        self.replacement_strength = nn.Parameter(torch.full((len(basis),), initial_strength))
        self.replacement_extrapolation_up = nn.Linear(width, len(basis), bias=False)
        nn.init.zeros_(self.replacement_extrapolation_up.weight)
        nn.init.kaiming_uniform_(self.replacement_down.weight, a=5**0.5)
        nn.init.zeros_(self.replacement_up.weight)

    def alternative(self, source):
        feature = F.silu(self.replacement_down(self.token_norm(source)))
        weights = torch.softmax(self.replacement_up(feature).float(), dim=-1)
        anchor = weights @ self.replacement_codebook
        residual = self.replacement_extrapolation_up(feature).float()
        bounded = self.radius * residual / (self.radius + residual.norm(dim=-1, keepdim=True))
        return anchor + bounded

    def forward(self, hidden, question_mask):
        if (
            hidden.ndim != 3
            or question_mask.shape != hidden.shape[:2]
            or question_mask.dtype != torch.bool
        ):
            raise ValueError("expected hidden [B,S,D] and boolean question mask [B,S]")
        if not question_mask.any(1).all():
            raise ValueError("every sample must contain question tokens")
        if hidden.shape[-1] != self.retain_center.numel():
            raise ValueError("hidden size mismatch")
        mask = question_mask.to(hidden.device)
        before = masked_mean(hidden, mask)
        updated = hidden.clone()
        for sample in range(len(hidden)):
            tokens = hidden[sample, mask[sample]]
            source = tokens.float().mean(0, keepdim=True)
            coefficients = (source - self.retain_center) @ self.target_basis.T
            delta = (
                (self.alternative(source) - coefficients) * self.replacement_strength.clamp(0, 1)
            ) @ self.target_basis
            # BF16 时先转换 delta 再相加；不能把整次加法提升成 FP32，避免改变旧数值路径。
            updated[sample, mask[sample]] = tokens + delta.expand_as(tokens).to(tokens.dtype)
        return ReplacementOutput(updated, before.detach(), masked_mean(updated, mask))

    @torch.no_grad()
    def project_constraints(self):
        self.replacement_strength.clamp_(0, 1)
