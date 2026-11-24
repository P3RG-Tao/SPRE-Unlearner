"""一次性训练 E：保持 hidden 余弦结构，并分离 prototype 与保留类。"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import AlignConfig

TRAIN_SPLITS = ("prototype", "neighbor", "utility")


class GatedBottleneckE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.value = nn.Linear(input_dim, hidden_dim)
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden):
        return self.output(F.gelu(self.value(hidden)) * torch.sigmoid(self.gate(hidden)))


def structure_loss(hidden, predicted):
    raw = F.normalize(hidden.detach().float(), dim=-1)
    encoded = F.normalize(predicted.float(), dim=-1)
    return F.mse_loss(encoded @ encoded.T, raw @ raw.T)


def pairwise_contrastive_loss(positive, negative, *, margin, temperature):
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    center = F.normalize(positive.mean(dim=0, keepdim=True), dim=-1)
    differences = (positive @ center.T).T - negative @ center.T
    temperature = max(float(temperature), 1e-5)
    return (F.softplus((margin - differences) / temperature) * temperature).mean()


def validate_hidden_splits(hidden):
    if set(hidden) != set(TRAIN_SPLITS):
        raise ValueError("geometry requires exactly train prototype/neighbor/utility")
    dims = set()
    for name, values in hidden.items():
        if values.ndim != 2 or min(values.shape) <= 0 or not torch.isfinite(values).all():
            raise ValueError(f"invalid hidden matrix: {name}")
        dims.add(values.shape[1])
    if len(dims) != 1:
        raise ValueError("hidden dimensions differ between splits")


def align_predictor(hidden, config: AlignConfig, *, seed, device="cpu"):
    """输入必须来自 train；不用答案 embedding，不读取 dev/test。

    seed 是实验种子；正式实现内部使用 seed+100，再以 +17 采样。
    AdamW 的 weight_decay=0.01 与旧实现默认值一致，不能改成遗忘阶段的0。
    """
    validate_hidden_splits(hidden)
    torch.manual_seed(seed + 100)
    generator = torch.Generator().manual_seed(seed + 117)
    predictor = GatedBottleneckE(
        hidden["prototype"].shape[1], config.hidden_dim, config.output_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(), lr=config.learning_rate, weight_decay=0.01
    )
    inputs = {key: value.detach().float().to(device) for key, value in hidden.items()}
    history = []
    for step in range(1, config.steps + 1):
        batch = {
            s: inputs[s][
                torch.randint(len(inputs[s]), (config.batch_size,), generator=generator).to(device)
            ]
            for s in TRAIN_SPLITS
        }
        predicted = {s: predictor(batch[s]) for s in TRAIN_SPLITS}
        structure = structure_loss(
            torch.cat([batch[s] for s in TRAIN_SPLITS]),
            torch.cat([predicted[s] for s in TRAIN_SPLITS]),
        )
        neighbor = pairwise_contrastive_loss(
            predicted["prototype"],
            predicted["neighbor"],
            margin=config.margin,
            temperature=config.temperature,
        )
        utility = pairwise_contrastive_loss(
            predicted["prototype"],
            predicted["utility"],
            margin=config.margin,
            temperature=config.temperature,
        )
        loss = config.structure_weight * structure + config.contrastive_weight * (
            config.neighbor_weight * neighbor + config.utility_weight * utility
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite alignment loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == config.steps:
            history.append(
                dict(
                    step=step,
                    loss=loss.item(),
                    structure_loss=structure.item(),
                    neighbor_contrastive_loss=neighbor.item(),
                    utility_contrastive_loss=utility.item(),
                )
            )
    predictor.eval().requires_grad_(False)
    # 清掉对齐阶段残留的梯度，后续梯度隔离检查只检查新产生的梯度。
    predictor.zero_grad(set_to_none=True)
    return predictor, history
