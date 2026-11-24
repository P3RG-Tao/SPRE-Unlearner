"""算法主流程：一次对齐、固定几何、仅优化 Replacement。"""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile

import torch
from torch.nn import functional as F

from .alignment import GatedBottleneckE, align_predictor
from .config import ReplacementConfig
from .geometry import build_geometry
from .replacement import CoupledReplacement


@dataclass
class HiddenBatch:
    """冻结主干指定层的完整 hidden 及原始问题 token 掩码。"""

    hidden: torch.Tensor
    question_mask: torch.Tensor


class SPREUnlearner:
    def __init__(self, predictor, geometry, config, *, identity, device="cpu"):
        required = {"target", "model_revision", "data_fingerprint"}
        if set(identity) != required or not all(
            isinstance(v, str) and v.strip() for v in identity.values()
        ):
            raise ValueError(f"identity requires nonempty strings: {sorted(required)}")
        self.identity = dict(identity)
        self.config = config
        self.device = torch.device(device)
        self.predictor = predictor.to(self.device).eval().requires_grad_(False)
        self.predictor.zero_grad(set_to_none=True)
        self.geometry = {
            k: v.detach().float().cpu().clone() if torch.is_tensor(v) else v
            for k, v in geometry.items()
        }
        # 几何在CPU构造；保持旧 module_factory 的CPU初始化随机流。
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(config.seed)
            self.replacement = CoupledReplacement(
                target_basis=self.geometry["target_basis"],
                retain_center=self.geometry["retain_center"],
                replacement_codebook=self.geometry["replacement_codebook"],
                width=config.width,
                radius=config.radius,
                initial_strength=config.initial_strength,
            ).to(self.device)
        self.prototype = F.normalize(self.geometry["target_prototype"].to(self.device), dim=0)
        self.optimizer = torch.optim.AdamW(
            self.replacement.parameters(), lr=config.learning_rate, weight_decay=0.0
        )
        self.step = 0
        self.align_history = []

    @classmethod
    def prepare(cls, train_hidden, *, identity, config=None, partition="train", device="cpu"):
        if partition != "train":
            raise ValueError("E and geometry may only be fitted on train")
        config = config or ReplacementConfig()
        predictor, history = align_predictor(
            train_hidden, config.align, seed=config.seed, device=device
        )
        geometry = build_geometry(train_hidden, predictor, config, device=device)
        result = cls(predictor, geometry, config, identity=identity, device=device)
        result.align_history = history
        return result

    def losses(self, *, forget, neighbor, utility):
        outputs = {}
        for name, batch in (("forget", forget), ("neighbor", neighbor), ("utility", utility)):
            if len(batch.hidden) != self.config.train_batch_size:
                raise ValueError("training batch size differs from configuration")
            # 主干前缀被冻结；避免调用方误把主干计算图带入算法。
            outputs[name] = self.replacement(
                batch.hidden.detach().to(self.device), batch.question_mask.to(self.device)
            )
        # E参数冻结但此处不能 no_grad：遗忘梯度需要经过 E 回传到 adapter。
        forget_embedding = F.normalize(
            self.predictor(outputs["forget"].after.float()).float(), dim=-1
        )
        forget_loss = (forget_embedding @ self.prototype).mean()
        retain = {
            s: 1
            - F.cosine_similarity(
                outputs[s].after.float(), outputs[s].before.float(), dim=-1
            ).mean()
            for s in ("neighbor", "utility")
        }
        retain_loss = 0.5 * (retain["neighbor"] + retain["utility"])
        return dict(
            total=forget_loss + self.config.retain_hidden_weight * retain_loss,
            forget=forget_loss,
            retain=retain_loss,
            neighbor=retain["neighbor"],
            utility=retain["utility"],
        )

    def train_step(self, *, forget, neighbor, utility):
        if self.step >= self.config.max_steps:
            raise RuntimeError("configured training budget exhausted")
        self.predictor.eval()
        self.replacement.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = self.losses(forget=forget, neighbor=neighbor, utility=utility)
        if not all(torch.isfinite(v) for v in losses.values()):
            raise FloatingPointError("non-finite unlearning loss")
        losses["total"].backward()
        parameters = list(self.replacement.parameters())
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError("missing/non-finite adapter gradient")
        if any(p.grad is not None for p in self.predictor.parameters()):
            raise RuntimeError("gradient escaped into E")
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.replacement.project_constraints()
        self.step += 1
        return dict(
            step=self.step,
            gradient_norm=float(norm),
            **{k: float(v.detach()) for k, v in losses.items()},
        )

    def save(self, path, *, data_state=None):
        """保存算法和优化器；data_state由调用方记录采样位置，不冒充完整实验断点。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            schema_version=1,
            source_protocol="qwen_replacement_v2_r1",
            config=asdict(self.config),
            identity=self.identity,
            geometry=self.geometry,
            predictor=self.predictor.state_dict(),
            replacement=self.replacement.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.step,
            align_history=self.align_history,
            data_state=data_state,
        )
        # 先完整落盘，再以不覆盖语义发布；失败时不留半个checkpoint。
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
        os.close(fd)
        try:
            with open(temporary, "wb") as stream:
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    @classmethod
    def load(cls, path, *, expected_identity, device="cpu"):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            value.get("schema_version") != 1
            or value.get("source_protocol") != "qwen_replacement_v2_r1"
        ):
            raise ValueError("unsupported checkpoint schema")
        if value["identity"] != expected_identity:
            raise ValueError("checkpoint target/model/data identity mismatch")
        config = ReplacementConfig.from_dict(value["config"])
        dimension = value["geometry"]["retain_center"].numel()
        predictor = GatedBottleneckE(dimension, config.align.hidden_dim, config.align.output_dim)
        predictor.load_state_dict(value["predictor"], strict=True)
        result = cls(
            predictor, value["geometry"], config, identity=expected_identity, device=device
        )
        result.replacement.load_state_dict(value["replacement"], strict=True)
        result.optimizer.load_state_dict(value["optimizer"])
        result.step = value["step"]
        if type(result.step) is not int or not 0 <= result.step <= config.max_steps:
            raise ValueError("invalid saved training step")
        result.align_history = value["align_history"]
        return result, value["data_state"]

    def fit(self, batches, *, output_dir, on_checkpoint=None):
        """稳定顺序循环F/N/U，每步各一个batch；无guard、回滚、提前停止。

        batches是三个非空Sequence[HiddenBatch]，适用于小规模测试或外部缓存。
        大数据可直接使用train_step流式供给，避免缓存全部token hidden。
        回调可做dev评估；本包不会自行请求LLM或选择GPU。
        """
        if set(batches) != {"forget", "neighbor", "utility"} or any(
            len(v) == 0 for v in batches.values()
        ):
            raise ValueError("fit requires nonempty F/N/U batch sequences")
        self._check_output(output_dir)
        while self.step < self.config.max_steps:
            row = self.train_step(
                **{s: values[self.step % len(values)] for s, values in batches.items()}
            )
            if self.step % self.config.checkpoint_every == 0 or self.step == self.config.max_steps:
                path = Path(output_dir) / f"step_{self.step:05d}.pt"
                self.save(path, data_state={s: self.step % len(v) for s, v in batches.items()})
                if on_checkpoint is not None:
                    on_checkpoint(self, path, row)

    def _check_output(self, output_dir):
        # 计算前就检查，而不是训练500步之后才发现产物冲突。
        if Path(output_dir).exists() and any(Path(output_dir).iterdir()):
            raise FileExistsError(
                "training output must be new or empty; resume into a new directory"
            )

    def fit_stream(self, loaders, encode_hidden, *, output_dir, on_checkpoint=None):
        """按稳定顺序重建每个split迭代器；可用QwenBackend.hidden编码，不缓存全量hidden。

        loaders必须可重复迭代且每次顺序一致。恢复时跳过self.step个batch，
        调用方仍需用identity.data_fingerprint保证数据、顺序和batch配置未变。
        """
        if set(loaders) != {"forget", "neighbor", "utility"}:
            raise ValueError("training loaders must be F/N/U only")
        self._check_output(output_dir)

        def cycle(loader):
            if iter(loader) is loader:
                raise ValueError("loader must be re-iterable, not a one-shot iterator")
            while True:
                seen = False
                for batch in loader:
                    seen = True
                    yield batch
                if not seen:
                    raise ValueError("empty training loader")

        streams = {s: cycle(loader) for s, loader in loaders.items()}
        for stream in streams.values():
            for _ in range(self.step):
                next(stream)
        while self.step < self.config.max_steps:
            batches = {s: encode_hidden(next(stream)) for s, stream in streams.items()}
            row = self.train_step(**batches)
            if self.step % self.config.checkpoint_every == 0 or self.step == self.config.max_steps:
                path = Path(output_dir) / f"step_{self.step:05d}.pt"
                self.save(path, data_state={"batches_consumed_per_split": self.step})
                if on_checkpoint is not None:
                    on_checkpoint(self, path, row)


def select_operating_point(rows):
    """只按dev QA选点，不使用test或LLM质量，不设置Forget阈值。"""
    if not rows:
        raise ValueError("no dev candidates")
    for row in rows:
        if row.get("partition") != "dev":
            raise ValueError("selection may only use dev metrics")
        for split in ("forget", "neighbor", "utility", "target_image_nontarget"):
            if not 0 <= row[split] <= 1:
                raise ValueError(f"invalid QA metric: {split}")
    return min(
        rows,
        key=lambda r: (
            r["forget"],
            -(r["neighbor"] + r["utility"] + r["target_image_nontarget"]) / 3,
            r["step"],
        ),
    )
