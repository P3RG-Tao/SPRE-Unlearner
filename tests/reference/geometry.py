from __future__ import annotations
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F
TRAIN_SPLITS = ("prototype", "neighbor", "utility")

def tensor_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().float().cpu().flatten()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }

def build_target_basis(
    train_hidden: dict[str, torch.Tensor],
    *,
    rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """从 train-only target/retain 差分构造固定正交子空间。"""

    retain = torch.cat(
        [train_hidden["neighbor"], train_hidden["utility"]], dim=0
    ).float()
    retain_center = retain.mean(dim=0)
    target = train_hidden["prototype"].float()
    count = min(
        len(target), len(train_hidden["neighbor"]), len(train_hidden["utility"])
    )
    difference_matrix = torch.cat(
        [
            target - retain_center,
            target[:count] - train_hidden["neighbor"][:count].float(),
            target[:count] - train_hidden["utility"][:count].float(),
        ],
        dim=0,
    )
    matrix = difference_matrix.to(device)
    _, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    basis = vh[: int(rank)].detach().float().cpu()
    explained = singular_values.square()
    summary = {
        "construction": "svd([prototype-retain_center, prototype-neighbor, prototype-utility])",
        "row_count": int(matrix.shape[0]),
        "hidden_size": int(matrix.shape[1]),
        "rank": int(rank),
        "top_rank_energy_ratio": float(
            explained[: int(rank)].sum() / explained.sum().clamp_min(1.0e-12)
        ),
        "orthogonality_max_error": float(
            (basis @ basis.T - torch.eye(int(rank))).abs().max()
        ),
    }
    return basis, retain_center, summary

def build_retain_coefficient_codebook(
    train_hidden: dict[str, torch.Tensor],
    *,
    target_basis: torch.Tensor,
    retain_center: torch.Tensor,
    codebook_size: int,
    max_iterations: int = 50,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """对 retain 样本的 target 子空间系数执行确定性 k-means。

    初始化使用离总体均值最远的样本，再逐次选择离已有中心最远的样本；
    因此不依赖随机数和 sklearn，同一共享产物必然生成同一 codebook。
    """

    retain = torch.cat(
        [train_hidden["neighbor"], train_hidden["utility"]], dim=0
    ).float()
    basis = target_basis.detach().float()
    center = retain_center.detach().float().flatten()
    coefficients = (retain - center) @ basis.transpose(0, 1)
    size = int(codebook_size)
    if size < 2:
        raise ValueError("replacement codebook size must be at least 2")
    if size > int(coefficients.shape[0]):
        raise ValueError("replacement codebook size exceeds retain sample count")

    global_mean = coefficients.mean(dim=0, keepdim=True)
    first_index = int((coefficients - global_mean).square().sum(dim=1).argmax())
    selected = [first_index]
    minimum_distance = torch.full(
        (int(coefficients.shape[0]),), float("inf"), dtype=torch.float32
    )
    for _ in range(1, size):
        latest = coefficients[selected[-1]]
        distance = (coefficients - latest).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
        selected.append(int(minimum_distance.argmax()))
    codebook = coefficients[selected].clone()

    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        distance = torch.cdist(coefficients, codebook).square()
        assignment = distance.argmin(dim=1)
        updated = codebook.clone()
        for cluster_index in range(size):
            members = coefficients[assignment == cluster_index]
            if int(members.shape[0]) > 0:
                updated[cluster_index] = members.mean(dim=0)
        if torch.allclose(updated, codebook, atol=1.0e-7, rtol=0.0):
            codebook = updated
            break
        codebook = updated

    final_distance = torch.cdist(coefficients, codebook).square()
    final_assignment = final_distance.argmin(dim=1)
    occupancy = torch.bincount(final_assignment, minlength=size)
    quantization_mse = final_distance.gather(1, final_assignment.unsqueeze(1)).mean()
    summary = {
        "construction": "deterministic_farthest_first_kmeans",
        "source_splits": ["neighbor", "utility"],
        "sample_count": int(coefficients.shape[0]),
        "coefficient_dim": int(coefficients.shape[1]),
        "codebook_size": size,
        "iterations": int(iterations),
        "quantization_mse": float(quantization_mse),
        "coefficient_norm": tensor_summary(coefficients.norm(dim=1)),
        "occupancy_min": int(occupancy.min()),
        "occupancy_max": int(occupancy.max()),
        "centroid_norm": tensor_summary(codebook.norm(dim=1)),
    }
    return codebook, summary
