"""Qwen decoder 层后的可训练 target intervention。

该模块只读取 decoder 第 7 层已经完成计算的输出，并且只改写原始问题 token。
所有模式共享同一 target gate、固定 target 子空间和训练数据，但写回公式不同：

``residual``
    标准低秩残差 adapter，不使用 target gate。
``gated_residual``
    与 residual 使用相同低秩网络，但用 target gate 缩放更新。
``subtractive``
    只允许非负地削减固定 target 子空间中的原有分量，不能添加任意方向。
``ungated_replacement``
    与 gated replacement 使用相同替代网络，但不使用 target gate。
``gated_replacement``
    在相同 target gate 下，把 target 子空间系数插值到学习到的替代系数。

subtractive 不能实现为 ``h - U z``。若 U 无约束，这与 ``h + U z`` 仅差一个
参数符号，无法构成有意义的结构消融。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


RESIDUAL_MODES = frozenset(("residual", "gated_residual"))
REPLACEMENT_MODES = frozenset(("ungated_replacement", "gated_replacement"))
UNGATED_MODES = frozenset(("residual", "ungated_replacement"))
REPLACEMENT_SCOPES = ("tokenwise", "samplewise")
REPLACEMENT_CONDITIONING = ("full", "orthogonal")
REPLACEMENT_DISTRIBUTIONS = ("uniform", "target_aligned")
REPLACEMENT_OUTPUT_SPACES = (
    "unconstrained",
    "retain_codebook",
    "retain_codebook_bounded",
)
MODES = (
    "residual",
    "gated_residual",
    "subtractive",
    "ungated_replacement",
    "gated_replacement",
)


@dataclass
class InterventionCapture:
    """一次前向中供损失和诊断读取的样本级状态。"""

    before: torch.Tensor
    after: torch.Tensor
    target_similarity: torch.Tensor
    target_gate: torch.Tensor
    applied_gate: torch.Tensor
    delta_rms: torch.Tensor


class PostLayerTargetIntervention(nn.Module):
    """严格串接在 Qwen decoder 层后的 target-conditioned intervention。"""

    def __init__(
        self,
        hidden_size: int,
        *,
        mode: str,
        predictor: nn.Module,
        target_prototype: torch.Tensor,
        target_basis: torch.Tensor,
        retain_center: torch.Tensor,
        tau: float,
        gate_temperature: float = 0.05,
        adapter_rank: int = 16,
        adapter_alpha: float = 16.0,
        replacement_hidden_dim: int = 32,
        replacement_scope: str = "tokenwise",
        replacement_conditioning: str = "full",
        replacement_distribution: str = "uniform",
        replacement_distribution_temperature: float = 0.5,
        replacement_output_space: str = "unconstrained",
        replacement_codebook: torch.Tensor | None = None,
        replacement_codebook_temperature: float = 1.0,
        replacement_extrapolation_max_norm: float = 0.0,
        initial_strength: float = 1.0e-4,
        layer_index: int = 7,
    ) -> None:
        super().__init__()
        normalized_mode = str(mode).strip().lower()
        if normalized_mode not in MODES:
            raise ValueError(f"unsupported intervention mode: {mode!r}")
        if int(hidden_size) <= 0:
            raise ValueError("hidden_size must be positive")
        if float(gate_temperature) <= 0.0:
            raise ValueError("gate_temperature must be positive")
        if not 0.0 < float(initial_strength) < 1.0:
            raise ValueError("initial_strength must be in (0, 1)")
        normalized_scope = str(replacement_scope).strip().lower()
        if normalized_scope not in REPLACEMENT_SCOPES:
            raise ValueError(f"unsupported replacement scope: {replacement_scope!r}")
        normalized_conditioning = str(replacement_conditioning).strip().lower()
        if normalized_conditioning not in REPLACEMENT_CONDITIONING:
            raise ValueError(
                f"unsupported replacement conditioning: {replacement_conditioning!r}"
            )
        normalized_distribution = str(replacement_distribution).strip().lower()
        if normalized_distribution not in REPLACEMENT_DISTRIBUTIONS:
            raise ValueError(
                f"unsupported replacement distribution: {replacement_distribution!r}"
            )
        if normalized_scope != "samplewise" and normalized_distribution != "uniform":
            raise ValueError(
                "target-aligned replacement distribution requires samplewise scope"
            )
        if float(replacement_distribution_temperature) <= 0.0:
            raise ValueError("replacement distribution temperature must be positive")
        normalized_output_space = str(replacement_output_space).strip().lower()
        if normalized_output_space not in REPLACEMENT_OUTPUT_SPACES:
            raise ValueError(
                f"unsupported replacement output space: {replacement_output_space!r}"
            )
        if (
            normalized_output_space != "unconstrained"
            and normalized_mode not in REPLACEMENT_MODES
        ):
            raise ValueError(
                "replacement output space only applies to replacement modes"
            )
        if float(replacement_codebook_temperature) <= 0.0:
            raise ValueError("replacement codebook temperature must be positive")
        extrapolation_max_norm = float(replacement_extrapolation_max_norm)
        if normalized_output_space == "retain_codebook_bounded":
            if extrapolation_max_norm <= 0.0:
                raise ValueError(
                    "bounded codebook output requires a positive extrapolation norm"
                )
        elif extrapolation_max_norm != 0.0:
            raise ValueError(
                "replacement extrapolation norm only applies to bounded codebook output"
            )

        basis = target_basis.detach().float()
        center = retain_center.detach().float().flatten()
        prototype = target_prototype.detach().float().flatten()
        if basis.ndim != 2 or int(basis.shape[1]) != int(hidden_size):
            raise ValueError("target_basis must be [subspace_rank, hidden_size]")
        if int(center.numel()) != int(hidden_size):
            raise ValueError("retain_center must have hidden_size elements")
        if prototype.numel() <= 0:
            raise ValueError("target_prototype must be non-empty")
        codebook: torch.Tensor | None = None
        if normalized_output_space in (
            "retain_codebook",
            "retain_codebook_bounded",
        ):
            if replacement_codebook is None:
                raise ValueError("retain_codebook output requires replacement_codebook")
            codebook = replacement_codebook.detach().float()
            if codebook.ndim != 2 or int(codebook.shape[1]) != int(basis.shape[0]):
                raise ValueError(
                    "replacement_codebook must be [codebook_size, subspace_rank]"
                )
            if int(codebook.shape[0]) < 2:
                raise ValueError(
                    "replacement_codebook must contain at least two centers"
                )
        elif replacement_codebook is not None:
            raise ValueError(
                "replacement_codebook is only valid for retain_codebook output"
            )

        self.hidden_size = int(hidden_size)
        self.mode = normalized_mode
        self.predictor = predictor.eval()
        self.tau = float(tau)
        self.gate_temperature = float(gate_temperature)
        self.adapter_rank = int(adapter_rank)
        self.adapter_alpha = float(adapter_alpha)
        self.adapter_scaling = self.adapter_alpha / self.adapter_rank
        self.replacement_hidden_dim = int(replacement_hidden_dim)
        self.replacement_scope = normalized_scope
        self.replacement_conditioning = normalized_conditioning
        self.replacement_distribution = normalized_distribution
        self.replacement_distribution_temperature = float(
            replacement_distribution_temperature
        )
        self.replacement_output_space = normalized_output_space
        self.replacement_codebook_temperature = float(replacement_codebook_temperature)
        self.replacement_extrapolation_max_norm = extrapolation_max_norm
        self.layer_index = int(layer_index)

        for parameter in self.predictor.parameters():
            parameter.requires_grad = False
        self.register_buffer("target_prototype", F.normalize(prototype, dim=0))
        self.register_buffer("target_basis", F.normalize(basis, dim=1))
        self.register_buffer("retain_center", center)
        if codebook is not None:
            # 仅在新模式注册持久 buffer，保证历史 unconstrained checkpoint 的
            # state_dict 键集合完全不变。
            self.register_buffer("replacement_codebook", codebook)

        self.token_norm = nn.LayerNorm(self.hidden_size, elementwise_affine=False)
        if self.mode in RESIDUAL_MODES:
            self.residual_down = nn.Linear(
                self.hidden_size, self.adapter_rank, bias=False
            )
            self.residual_up = nn.Linear(
                self.adapter_rank, self.hidden_size, bias=False
            )
            nn.init.kaiming_uniform_(self.residual_down.weight, a=5**0.5)
            nn.init.zeros_(self.residual_up.weight)
        elif self.mode == "subtractive":
            # 非负强度限制保证该分支只能把 target 子空间系数拉回 retain 中心。
            self.suppression_strength = nn.Parameter(
                torch.full((int(basis.shape[0]),), float(initial_strength))
            )
        else:
            self.replacement_down = nn.Linear(
                self.hidden_size, self.replacement_hidden_dim, bias=False
            )
            self.replacement_up = nn.Linear(
                self.replacement_hidden_dim,
                (
                    int(basis.shape[0])
                    if self.replacement_output_space == "unconstrained"
                    else int(codebook.shape[0])
                ),
                bias=False,
            )
            self.replacement_strength = nn.Parameter(
                torch.full((int(basis.shape[0]),), float(initial_strength))
            )
            if self.replacement_output_space == "retain_codebook_bounded":
                self.replacement_extrapolation_up = nn.Linear(
                    self.replacement_hidden_dim,
                    int(basis.shape[0]),
                    bias=False,
                )
                nn.init.zeros_(self.replacement_extrapolation_up.weight)
            nn.init.kaiming_uniform_(self.replacement_down.weight, a=5**0.5)
            nn.init.zeros_(self.replacement_up.weight)

        self._hook_handle: Any | None = None
        self._image_mask: torch.Tensor | None = None
        self._question_mask: torch.Tensor | None = None
        self._capture: InterventionCapture | None = None

    def _replacement_alternative(
        self,
        replacement_condition: torch.Tensor,
    ) -> torch.Tensor:
        """预测替代系数；codebook 模式提供 retain 锚点和可选有界外推。"""

        feature = F.silu(self.replacement_down(self.token_norm(replacement_condition)))
        raw = self.replacement_up(feature).float()
        if self.replacement_output_space == "unconstrained":
            return raw
        weights = torch.softmax(
            raw / self.replacement_codebook_temperature,
            dim=-1,
        )
        anchor = weights @ self.replacement_codebook
        if self.replacement_output_space == "retain_codebook":
            return anchor

        # 在 codebook 合法锚点之外允许有限外推。该映射在原点附近近似恒等，
        # 且残差 L2 范数始终严格小于配置半径，避免退回无约束 replacement。
        residual = self.replacement_extrapolation_up(feature).float()
        norm = residual.norm(dim=-1, keepdim=True)
        radius = self.replacement_extrapolation_max_norm
        bounded_residual = radius * residual / (radius + norm)
        return anchor + bounded_residual

    def attach(self, layer: nn.Module) -> None:
        """把 intervention 严格挂到目标 decoder 层输出之后。"""

        if self._hook_handle is not None:
            raise RuntimeError("post-layer intervention is already attached")
        self._hook_handle = layer.register_forward_hook(self._output_hook)

    def detach(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._hook_handle = None
        self._image_mask = None
        self._question_mask = None
        self._capture = None

    def set_runtime_masks(
        self,
        image_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> None:
        """记录当前前向中的图像与原始问题 token 位置。"""

        if image_mask.ndim != 2 or question_mask.ndim != 2:
            raise ValueError("runtime masks must be [batch, sequence]")
        if tuple(image_mask.shape) != tuple(question_mask.shape):
            raise ValueError("image and question masks must have identical shapes")
        if bool((question_mask.sum(dim=1) == 0).any()):
            raise ValueError("every sample must contain raw question tokens")
        if bool((image_mask.bool() & question_mask.bool()).any()):
            raise ValueError("image and question masks must not overlap")
        self._image_mask = image_mask.detach().bool()
        self._question_mask = question_mask.detach().bool()

    @staticmethod
    def _first_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...], bool]:
        if torch.is_tensor(output):
            return output, (), False
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return output[0], tuple(output[1:]), True
        raise TypeError("decoder layer returned an unsupported output")

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _target_condition(
        self,
        pooled_before: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """用冻结 E 计算样本级 target similarity 和软门控。"""

        with torch.no_grad():
            embedding = F.normalize(
                self.predictor(pooled_before.detach().float()).float(), dim=-1
            )
            similarity = embedding @ self.target_prototype
            gate = torch.sigmoid((similarity - self.tau) / self.gate_temperature)
        return similarity, gate

    def _question_update(
        self,
        question_hidden: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode in RESIDUAL_MODES:
            normalized = self.token_norm(question_hidden.float())
            feature = F.silu(self.residual_down(normalized))
            return (gate * self.residual_up(feature) * self.adapter_scaling).to(
                question_hidden.dtype
            )

        centered = question_hidden.float() - self.retain_center
        coefficients = centered @ self.target_basis.transpose(0, 1)
        if self.mode == "subtractive":
            strength = self.suppression_strength.clamp(0.0, 1.0)
            projected = (coefficients * strength) @ self.target_basis
            return -(gate * projected).to(question_hidden.dtype)

        # E、prototype 和 target basis 都建立在样本级 question mean 上。
        # samplewise 模式在同一坐标中预测一次 replacement，并把相同偏移写回
        # 所有问题 token；tokenwise 保留历史实现，作为严格对照。
        replacement_source = question_hidden.float()
        if self.replacement_scope == "samplewise":
            replacement_source = replacement_source.mean(dim=0, keepdim=True)
        replacement_centered = replacement_source - self.retain_center
        replacement_coefficients = replacement_centered @ self.target_basis.transpose(
            0, 1
        )
        replacement_condition = replacement_source
        if self.replacement_conditioning == "orthogonal":
            # 只移除相对 retain center 的 target 子空间分量，保留其余上下文。
            # 这样 replacement 网络不能直接从待删除的系数复制 target 信息。
            replacement_condition = (
                replacement_source - replacement_coefficients @ self.target_basis
            )
        alternative = self._replacement_alternative(replacement_condition)
        strength = self.replacement_strength.clamp(0.0, 1.0)
        coefficient_delta = (alternative - replacement_coefficients) * strength
        projected = coefficient_delta @ self.target_basis
        if self.replacement_scope == "samplewise":
            projected = projected.expand_as(question_hidden)
            if self.replacement_distribution == "target_aligned":
                # 保持样本级平均更新严格不变，只重新分配到更符合该样本
                # target 子空间方向的 token。这样 E-space pooled objective 与
                # 已训练 checkpoint 不变，生成差异只来自写回位置。
                pooled_coefficients = coefficients.mean(dim=0, keepdim=True)
                token_alignment = F.cosine_similarity(
                    coefficients,
                    pooled_coefficients.expand_as(coefficients),
                    dim=-1,
                    eps=1.0e-8,
                )
                token_weights = torch.softmax(
                    token_alignment / self.replacement_distribution_temperature,
                    dim=0,
                ) * int(question_hidden.shape[0])
                projected = projected * token_weights.unsqueeze(-1)
        return (gate * projected).to(question_hidden.dtype)

    def _output_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden, remaining, was_tuple = self._first_output(output)
        if self._question_mask is None:
            raise RuntimeError(
                "post-layer intervention requires prepare_frozen_visual_batch()"
            )
        if tuple(hidden.shape[:2]) != tuple(self._question_mask.shape):
            # 自回归 decode 的后续单 token 步不再包含完整问题序列。
            if int(hidden.shape[1]) == 1:
                return output
            raise RuntimeError(
                "intervention mask/hidden shape mismatch: "
                f"{tuple(self._question_mask.shape)} vs {tuple(hidden.shape[:2])}"
            )

        question_mask = self._question_mask.to(hidden.device)
        pooled_before = self._masked_mean(hidden, question_mask)
        similarity, target_gate = self._target_condition(pooled_before)
        applied_gate = (
            torch.ones_like(target_gate) if self.mode in UNGATED_MODES else target_gate
        )

        updated = hidden.clone()
        per_sample_rms: list[torch.Tensor] = []
        for sample in range(int(hidden.shape[0])):
            sample_mask = question_mask[sample]
            question_hidden = hidden[sample, sample_mask]
            gate = applied_gate[sample].reshape(1, 1).to(question_hidden.dtype)
            delta = self._question_update(question_hidden, gate)
            updated[sample, sample_mask] = question_hidden + delta
            per_sample_rms.append(delta.float().square().mean().sqrt())

        pooled_after = self._masked_mean(updated, question_mask)
        self._capture = InterventionCapture(
            before=pooled_before.detach(),
            after=pooled_after,
            target_similarity=similarity.detach(),
            target_gate=target_gate.detach(),
            applied_gate=applied_gate.detach(),
            delta_rms=torch.stack(per_sample_rms),
        )
        return (updated, *remaining) if was_tuple else updated

    def take_capture(self) -> InterventionCapture:
        """取得最近一次完整 prefill 前向的状态，并拒绝读取陈旧结果。"""

        if self._capture is None:
            raise RuntimeError("post-layer intervention produced no capture")
        value = self._capture
        self._capture = None
        return value

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def intervention_parameters(self) -> list[nn.Parameter]:
        """返回 intervention 自身参数，不受外层冻结顺序影响。"""

        return [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("predictor.")
        ]

    @torch.no_grad()
    def project_constraints(self) -> None:
        """优化后恢复结构约束，防止强度跨出可解释区间。"""

        if self.mode == "subtractive":
            self.suppression_strength.clamp_(0.0, 1.0)
        elif self.mode in REPLACEMENT_MODES:
            self.replacement_strength.clamp_(0.0, 1.0)

    def metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "hidden_size": self.hidden_size,
            "layer_index": self.layer_index,
            "target_subspace_rank": int(self.target_basis.shape[0]),
            "adapter_rank": self.adapter_rank,
            "adapter_alpha": self.adapter_alpha,
            "replacement_hidden_dim": self.replacement_hidden_dim,
            "replacement_scope": self.replacement_scope,
            "replacement_conditioning": self.replacement_conditioning,
            "replacement_distribution": self.replacement_distribution,
            "replacement_distribution_temperature": (
                self.replacement_distribution_temperature
            ),
            "replacement_output_space": self.replacement_output_space,
            "replacement_codebook_size": (
                int(self.replacement_codebook.shape[0])
                if self.replacement_output_space != "unconstrained"
                else None
            ),
            "replacement_codebook_temperature": (self.replacement_codebook_temperature),
            "replacement_extrapolation_max_norm": (
                self.replacement_extrapolation_max_norm
            ),
            "tau": self.tau,
            "gate_temperature": self.gate_temperature,
            "read_scope": (
                f"completed_decoder_layer_{self.layer_index}_output"
            ),
            "write_scope": (
                f"raw_question_tokens_after_decoder_layer_{self.layer_index}"
            ),
            "predictor_frozen": not any(
                parameter.requires_grad for parameter in self.predictor.parameters()
            ),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in self.trainable_parameters()
            ),
        }


class DualPointTargetIntervention(PostLayerTargetIntervention):
    """在早期层读取条件、在后期层执行 replacement 写回。

    单点 adapter 同时从目标层 hidden 预测替代系数并写回该层。双点结构把这两个
    职责拆开：read 层提供仍具有充分视觉概念信息的 question mean，write 层使用
    自己的 retain center、target basis 和 codebook 完成替换。两个层的向量不会
    直接相减或投影到同一个 basis，跨层传递的只有可训练网络预测出的系数。

    当前实现刻意只支持 samplewise/full replacement。这样与 Exp50-Exp56 的获胜
    结构严格一致，也避免给首次因果验证同时引入 tokenwise 或跨层 orthogonal
    投影等额外变量。
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        read_predictor: nn.Module,
        read_target_prototype: torch.Tensor,
        read_tau: float,
        read_layer_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(hidden_size, **kwargs)
        if self.mode not in REPLACEMENT_MODES:
            raise ValueError("dual-point intervention requires a replacement mode")
        if self.replacement_scope != "samplewise":
            raise ValueError("dual-point intervention currently requires samplewise scope")
        if self.replacement_conditioning != "full":
            raise ValueError("dual-point intervention currently requires full conditioning")
        if int(read_layer_index) < 0:
            raise ValueError("read_layer_index must be non-negative")
        if int(read_layer_index) >= int(self.layer_index):
            raise ValueError("dual-point read layer must precede the write layer")

        read_prototype = read_target_prototype.detach().float().flatten()
        if read_prototype.numel() <= 0:
            raise ValueError("read_target_prototype must be non-empty")
        self.read_predictor = read_predictor.eval()
        for parameter in self.read_predictor.parameters():
            parameter.requires_grad = False
        self.register_buffer(
            "read_target_prototype",
            F.normalize(read_prototype, dim=0),
        )
        self.read_tau = float(read_tau)
        self.read_layer_index = int(read_layer_index)
        self._read_hook_handle: Any | None = None
        self._read_pooled: torch.Tensor | None = None

    def attach_read(self, layer: nn.Module) -> None:
        """把只读 hook 挂到较早 decoder 层。"""

        if self._read_hook_handle is not None:
            raise RuntimeError("dual-point read hook is already attached")
        self._read_hook_handle = layer.register_forward_hook(self._read_output_hook)

    def detach(self) -> None:
        if self._read_hook_handle is not None:
            self._read_hook_handle.remove()
        self._read_hook_handle = None
        self._read_pooled = None
        super().detach()

    def set_runtime_masks(
        self,
        image_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> None:
        super().set_runtime_masks(image_mask, question_mask)
        # 每个新 batch 都必须重新经过 read 层，禁止复用上一次前向的条件。
        self._read_pooled = None

    def _read_output_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden, _remaining, _was_tuple = self._first_output(output)
        if self._question_mask is None:
            raise RuntimeError("dual-point intervention requires runtime masks")
        if tuple(hidden.shape[:2]) != tuple(self._question_mask.shape):
            if int(hidden.shape[1]) == 1:
                return output
            raise RuntimeError(
                "dual-point read mask/hidden shape mismatch: "
                f"{tuple(self._question_mask.shape)} vs {tuple(hidden.shape[:2])}"
            )
        question_mask = self._question_mask.to(hidden.device)
        # Qwen 全部冻结，读状态不需要保留其上游计算图；adapter 权重仍会从该常量
        # 条件正常获得梯度，同时显著降低训练显存。
        self._read_pooled = self._masked_mean(hidden, question_mask).detach()
        return output

    def _read_target_condition(
        self,
        pooled_read: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            embedding = F.normalize(
                self.read_predictor(pooled_read.detach().float()).float(), dim=-1
            )
            similarity = embedding @ self.read_target_prototype
            gate = torch.sigmoid(
                (similarity - self.read_tau) / self.gate_temperature
            )
        return similarity, gate

    def _dual_question_update(
        self,
        write_question_hidden: torch.Tensor,
        read_condition: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """由 read 条件预测 write 层系数，并在 write 层 basis 中执行替换。"""

        write_source = write_question_hidden.float().mean(dim=0, keepdim=True)
        write_coefficients = (
            write_source - self.retain_center
        ) @ self.target_basis.transpose(0, 1)
        alternative = self._replacement_alternative(
            read_condition.float().reshape(1, -1)
        )
        strength = self.replacement_strength.clamp(0.0, 1.0)
        projected = (
            (alternative - write_coefficients) * strength
        ) @ self.target_basis
        projected = projected.expand_as(write_question_hidden)

        if self.replacement_distribution == "target_aligned":
            token_coefficients = (
                write_question_hidden.float() - self.retain_center
            ) @ self.target_basis.transpose(0, 1)
            token_alignment = F.cosine_similarity(
                token_coefficients,
                write_coefficients.expand_as(token_coefficients),
                dim=-1,
                eps=1.0e-8,
            )
            token_weights = torch.softmax(
                token_alignment / self.replacement_distribution_temperature,
                dim=0,
            ) * int(write_question_hidden.shape[0])
            projected = projected * token_weights.unsqueeze(-1)
        return (gate * projected).to(write_question_hidden.dtype)

    def _output_hook(
        self,
        _module: nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> Any:
        hidden, remaining, was_tuple = self._first_output(output)
        if self._question_mask is None:
            raise RuntimeError("dual-point intervention requires runtime masks")
        if tuple(hidden.shape[:2]) != tuple(self._question_mask.shape):
            if int(hidden.shape[1]) == 1:
                return output
            raise RuntimeError(
                "dual-point write mask/hidden shape mismatch: "
                f"{tuple(self._question_mask.shape)} vs {tuple(hidden.shape[:2])}"
            )
        if self._read_pooled is None:
            raise RuntimeError("dual-point write hook ran before the read hook")
        if int(self._read_pooled.shape[0]) != int(hidden.shape[0]):
            raise RuntimeError("dual-point read/write batch size mismatch")

        question_mask = self._question_mask.to(hidden.device)
        pooled_before = self._masked_mean(hidden, question_mask)
        read_pooled = self._read_pooled.to(hidden.device)
        similarity, target_gate = self._read_target_condition(read_pooled)
        applied_gate = (
            torch.ones_like(target_gate) if self.mode in UNGATED_MODES else target_gate
        )

        updated = hidden.clone()
        per_sample_rms: list[torch.Tensor] = []
        for sample in range(int(hidden.shape[0])):
            sample_mask = question_mask[sample]
            write_question_hidden = hidden[sample, sample_mask]
            gate = applied_gate[sample].reshape(1, 1).to(write_question_hidden.dtype)
            delta = self._dual_question_update(
                write_question_hidden,
                read_pooled[sample],
                gate,
            )
            updated[sample, sample_mask] = write_question_hidden + delta
            per_sample_rms.append(delta.float().square().mean().sqrt())

        pooled_after = self._masked_mean(updated, question_mask)
        self._capture = InterventionCapture(
            before=pooled_before.detach(),
            after=pooled_after,
            target_similarity=similarity.detach(),
            target_gate=target_gate.detach(),
            applied_gate=applied_gate.detach(),
            delta_rms=torch.stack(per_sample_rms),
        )
        return (updated, *remaining) if was_tuple else updated

    def intervention_parameters(self) -> list[nn.Parameter]:
        """只返回双点 adapter 参数，两个 Predictor E 均保持冻结。"""

        return [
            parameter
            for name, parameter in self.named_parameters()
            if not name.startswith("predictor.")
            and not name.startswith("read_predictor.")
        ]

    def metadata(self) -> dict[str, object]:
        value = super().metadata()
        value.update(
            architecture="dual_point_read_write",
            read_layer_index=self.read_layer_index,
            write_layer_index=self.layer_index,
            read_scope=(
                f"question_mean_after_decoder_layer_{self.read_layer_index}"
            ),
            write_scope=(
                f"raw_question_tokens_after_decoder_layer_{self.layer_index}"
            ),
            read_predictor_frozen=not any(
                parameter.requires_grad
                for parameter in self.read_predictor.parameters()
            ),
        )
        return value
