"""用 train hidden 构造固定参照；E 原型与 hidden 子空间是不同对象。"""

import torch
from torch.nn import functional as F

from .alignment import validate_hidden_splits


@torch.no_grad()
def build_target_basis(hidden, rank, device="cpu"):
    validate_hidden_splits(hidden)
    h = {s: v.detach().float().cpu() for s, v in hidden.items()}
    center = torch.cat([h["neighbor"], h["utility"]]).mean(0)
    target = h["prototype"]
    count = min(map(len, h.values()))
    # 继承正式实现的顺序差分，不声称这些行是语义匹配的样本对。
    difference = torch.cat(
        [
            target - center,
            target[:count] - h["neighbor"][:count],
            target[:count] - h["utility"][:count],
        ]
    )
    if not 0 < rank <= min(difference.shape):
        raise ValueError("subspace rank exceeds matrix dimensions")
    _, _, vh = torch.linalg.svd(difference.to(device), full_matrices=False)
    return vh[:rank].float().cpu(), center


@torch.no_grad()
def build_codebook(hidden, basis, center, size, max_iterations=50):
    retain = torch.cat([hidden["neighbor"], hidden["utility"]]).detach().float().cpu()
    coefficients = (retain - center.cpu()) @ basis.cpu().T
    if not 2 <= size <= len(coefficients) or max_iterations < 1:
        raise ValueError("invalid codebook size or iteration count")
    first = int((coefficients - coefficients.mean(0, keepdim=True)).square().sum(1).argmax())
    selected = [first]
    distance = torch.full((len(coefficients),), float("inf"))
    for _ in range(1, size):
        distance = torch.minimum(
            distance, (coefficients - coefficients[selected[-1]]).square().sum(1)
        )
        distance[selected] = -1
        selected.append(int(distance.argmax()))
    centers = coefficients[selected].clone()
    for _ in range(max_iterations):
        assignment = torch.cdist(coefficients, centers).square().argmin(1)
        updated = centers.clone()
        for k in range(size):
            members = coefficients[assignment == k]
            if len(members):
                updated[k] = members.mean(0)
        converged = torch.allclose(updated, centers, atol=1e-7, rtol=0)
        centers = updated
        if converged:
            break
    return centers


@torch.no_grad()
def build_geometry(hidden, predictor, config, *, device="cpu"):
    basis, center = build_target_basis(hidden, config.rank, device)
    codebook = build_codebook(hidden, basis, center, config.codebook_size)
    embeddings = {
        s: F.normalize(predictor(v.detach().float().to(device)).float(), dim=-1).cpu()
        for s, v in hidden.items()
    }
    prototype = F.normalize(embeddings["prototype"].mean(0), dim=0)
    # tau 仅作为历史诊断保存；正式 ungated 写回及 cosine loss 不使用 tau。
    tau = torch.maximum(
        torch.quantile(embeddings["neighbor"] @ prototype, 0.85),
        torch.quantile(embeddings["utility"] @ prototype, 0.95),
    ).item()
    return dict(
        target_basis=basis,
        retain_center=center,
        replacement_codebook=codebook,
        target_prototype=prototype,
        diagnostic_tau=tau,
    )
