from __future__ import annotations
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
TRAIN_SPLITS = ("prototype", "neighbor", "utility")

class GatedBottleneckE(nn.Module):
    """阶段 B 固定使用的轻量 gated MLP，避免同时搜索模型结构。"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.value = nn.Linear(int(input_dim), int(hidden_dim))
        self.gate = nn.Linear(int(input_dim), int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        value = F.gelu(self.value(hidden))
        gate = torch.sigmoid(self.gate(hidden))
        return self.output(value * gate)

def structure_loss(raw_hidden: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    raw_similarity = F.normalize(raw_hidden.detach().float(), dim=-1)
    raw_similarity = raw_similarity @ raw_similarity.T
    predicted_similarity = F.normalize(predicted.float(), dim=-1)
    predicted_similarity = predicted_similarity @ predicted_similarity.T
    return F.mse_loss(predicted_similarity, raw_similarity)

def pairwise_contrastive_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """持续惩罚违反正负 margin 的样本对，不使用会提前归零的均值 ReLU。"""
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    center = F.normalize(positive.mean(dim=0, keepdim=True), dim=-1)
    positive_similarity = positive @ center.T
    negative_similarity = negative @ center.T
    differences = positive_similarity.T - negative_similarity
    temperature = max(float(temperature), 1.0e-5)
    return (
        F.softplus((float(margin) - differences) / temperature) * temperature
    ).mean()

def sample_indices(size: int, count: int, *, generator: torch.Generator) -> torch.Tensor:
    return torch.randint(0, int(size), (int(count),), generator=generator)

def train_predictor(
    train_raw: dict[str, torch.Tensor],
    train_input: dict[str, torch.Tensor],
    *,
    objective: str,
    output_dim: int,
    hidden_dim: int,
    steps: int,
    per_class_batch_size: int,
    lr: float,
    structure_weight: float,
    contrastive_weight: float,
    neighbor_weight: float,
    utility_weight: float,
    contrastive_margin: float,
    contrastive_temperature: float,
    seed: int,
    device: torch.device,
) -> tuple[GatedBottleneckE, list[dict[str, float | int]]]:
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed) + 17)
    predictor = GatedBottleneckE(
        input_dim=int(train_input["prototype"].shape[1]),
        hidden_dim=int(hidden_dim),
        output_dim=int(output_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=float(lr))
    raw = {key: value.float().to(device) for key, value in train_raw.items()}
    inputs = {key: value.float().to(device) for key, value in train_input.items()}
    history: list[dict[str, float | int]] = []

    for step in range(1, int(steps) + 1):
        indices = {
            split: sample_indices(
                len(inputs[split]),
                int(per_class_batch_size),
                generator=generator,
            ).to(device)
            for split in TRAIN_SPLITS
        }
        raw_batch = torch.cat([
            raw[split][indices[split]] for split in TRAIN_SPLITS
        ], dim=0)
        predicted = {
            split: predictor(inputs[split][indices[split]]) for split in TRAIN_SPLITS
        }
        predicted_batch = torch.cat([predicted[split] for split in TRAIN_SPLITS], dim=0)
        current_structure = structure_loss(raw_batch, predicted_batch)
        neighbor_contrast = predicted_batch.new_zeros(())
        utility_contrast = predicted_batch.new_zeros(())
        if objective == "contrastive_structure":
            neighbor_contrast = pairwise_contrastive_loss(
                predicted["prototype"],
                predicted["neighbor"],
                margin=float(contrastive_margin),
                temperature=float(contrastive_temperature),
            )
            utility_contrast = pairwise_contrastive_loss(
                predicted["prototype"],
                predicted["utility"],
                margin=float(contrastive_margin),
                temperature=float(contrastive_temperature),
            )
        elif objective != "structure_only":
            raise ValueError(f"unsupported objective {objective!r}")
        contrast = (
            float(neighbor_weight) * neighbor_contrast
            + float(utility_weight) * utility_contrast
        )
        loss = (
            float(structure_weight) * current_structure
            + float(contrastive_weight) * contrast
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == int(steps):
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "structure_loss": float(current_structure.detach().cpu()),
                "neighbor_contrastive_loss": float(neighbor_contrast.detach().cpu()),
                "utility_contrastive_loss": float(utility_contrast.detach().cpu()),
            }
            history.append(row)
            print(f"[E train] objective={objective} output={output_dim} {row}", flush=True)
    predictor.eval()
    return predictor, history
