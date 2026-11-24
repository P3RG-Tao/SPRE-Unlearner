"""只暴露正式路径使用的参数，不混入历史消融开关。"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class AlignConfig:
    steps: int = 1000
    hidden_dim: int = 768
    output_dim: int = 512
    batch_size: int = 64
    learning_rate: float = 5e-4
    structure_weight: float = 300.0
    contrastive_weight: float = 1.0
    neighbor_weight: float = 1.0
    utility_weight: float = 0.5
    margin: float = 0.15
    temperature: float = 0.05

    def __post_init__(self):
        for name in ("steps", "hidden_dim", "output_dim", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "temperature"):
            _positive(name, getattr(self, name))
        for name in (
            "structure_weight",
            "contrastive_weight",
            "neighbor_weight",
            "utility_weight",
            "margin",
        ):
            _positive(name, getattr(self, name), allow_zero=True)


def _positive(name, value, *, allow_zero=False):
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"invalid {name}: {value}")


@dataclass(frozen=True)
class ReplacementConfig:
    seed: int = 2027
    layer_index: int = 7  # 零基索引：第八个 decoder block 的输出。
    rank: int = 16
    width: int = 32
    codebook_size: int = 16
    radius: float = 128.0
    initial_strength: float = 1e-4
    learning_rate: float = 5e-4
    retain_hidden_weight: float = 1.0
    max_steps: int = 8000
    checkpoint_every: int = 500
    train_batch_size: int = 1
    align: AlignConfig = field(default_factory=AlignConfig)

    def __post_init__(self):
        for name in (
            "rank",
            "width",
            "codebook_size",
            "max_steps",
            "checkpoint_every",
            "train_batch_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.codebook_size < 2 or self.layer_index < 0 or self.seed < 0:
            raise ValueError("invalid codebook size, layer or seed")
        for name in ("radius", "learning_rate"):
            _positive(name, getattr(self, name))
        _positive("retain_hidden_weight", self.retain_hidden_weight, allow_zero=True)
        if not 0 < self.initial_strength < 1:
            raise ValueError("initial_strength must be in (0, 1)")

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["align"] = AlignConfig(**value.get("align", {}))
        return cls(**value)
