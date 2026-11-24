from types import SimpleNamespace
import copy

import pytest
import torch
from torch import nn

from spre_unlearner import SPREUnlearner, ReplacementConfig, AlignConfig
from spre_unlearner.qwen import QwenBackend, PreparedBatch, question_mask_from_prompts


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(12, 12)
        self.calls = 0

    def forward(self, hidden):
        self.calls += 1
        return (hidden + self.linear(hidden).tanh(), "preserved_tail")


class ToyQwen(nn.Module):
    """只验证适配器控制流，不作为真实Qwen精度/速度证据。"""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen2_5_vl", image_token_id=19)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([Block() for _ in range(9)])
        self.embedding = nn.Embedding(20, 12)
        self.saved = []

    def forward(self, input_ids, **kwargs):
        hidden = self.embedding(input_ids)
        for block in self.model.layers:
            hidden, tail = block(hidden)
            assert tail == "preserved_tail"
        self.saved.append(hidden.detach().clone())
        return hidden

    def generate(self, input_ids, **kwargs):
        self.forward(input_ids)
        one = torch.ones(len(input_ids), 1, dtype=torch.long)
        self.forward(one)
        return torch.cat([input_ids, one], 1)


def fixture():
    model = ToyQwen()
    backend = QwenBackend(model)
    batch = PreparedBatch(
        dict(input_ids=torch.tensor([[19, 2, 3, 4, 5]]), attention_mask=torch.ones(1, 5)),
        torch.tensor([[False, True, True, False, False]]),
    )
    geometry = {s: torch.randn(8, 12) for s in ("prototype", "neighbor", "utility")}
    config = ReplacementConfig(
        rank=3,
        width=5,
        codebook_size=4,
        max_steps=2,
        checkpoint_every=1,
        align=AlignConfig(steps=2, hidden_dim=8, output_dim=6, batch_size=4),
    )
    algorithm = SPREUnlearner.prepare(
        geometry,
        config=config,
        identity=dict(target="toy", model_revision="toy", data_fingerprint="toy"),
    )
    return model, backend, batch, algorithm


def test_early_exit_and_hook_cleanup():
    model, backend, batch, algorithm = fixture()
    state = copy.deepcopy(model.state_dict())
    hidden = backend.hidden(batch)
    assert model.model.layers[7].calls == 1
    assert model.model.layers[8].calls == 0
    assert not backend.layer._forward_hooks
    algorithm.train_step(forget=hidden, neighbor=hidden, utility=hidden)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert all(p.grad is None for p in model.parameters())


def test_generation_prefill_only_and_cleanup():
    model, backend, batch, algorithm = fixture()
    calls = []
    handle = algorithm.replacement.register_forward_hook(lambda *args: calls.append(1))
    output = backend.generate(batch, algorithm)
    handle.remove()
    assert output.shape == (1, 1)
    assert calls == [1]
    assert not backend.layer._forward_hooks


def test_failure_cleans_hook():
    model, backend, batch, algorithm = fixture()

    def broken(*args, **kwargs):
        raise RuntimeError("intentional")

    model.generate = broken
    with pytest.raises(RuntimeError, match="intentional"):
        backend.generate(batch, algorithm)
    assert not backend.layer._forward_hooks
    assert not backend._active
    assert backend.hidden(batch).hidden.shape == (1, 5, 12)


def test_mask_validation():
    model, backend, batch, algorithm = fixture()
    batch.question_mask[0, 0] = True
    with pytest.raises(ValueError, match="image"):
        backend.hidden(batch)
    batch.question_mask[0, 0] = False
    batch.inputs["attention_mask"][0, 1] = 0
    with pytest.raises(ValueError, match="padding"):
        backend.hidden(batch)


def test_mask_offsets_and_ambiguous_question():
    def tokenizer(text, **kwargs):
        return dict(
            input_ids=[ord(c) for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))]
        )

    prompt, question = "[image]what?\nshort answer", "what?"
    # processor将一个图像片段扩展成多token；匹配问题不依赖展开前的位置。
    ids = torch.tensor([[999, 999, 999] + [ord(c) for c in prompt[7:]]])
    mask = question_mask_from_prompts(tokenizer, ids, torch.ones_like(ids), [prompt], [question])
    assert mask.sum() == 5
    assert mask[0, 3:8].all()
    with pytest.raises(ValueError, match="exactly once"):
        question_mask_from_prompts(
            tokenizer, ids, torch.ones_like(ids), ["what? what?"], [question]
        )


def test_streaming_full_flow(tmp_path):
    model, backend, batch, algorithm = fixture()
    loaders = {s: [batch] for s in ("forget", "neighbor", "utility")}
    algorithm.fit_stream(loaders, backend.hidden, output_dir=tmp_path / "run")
    assert algorithm.step == 2
    restored, cursor = SPREUnlearner.load(
        tmp_path / "run/step_00002.pt", expected_identity=algorithm.identity
    )
    assert cursor["batches_consumed_per_split"] == 2
    assert backend.generate(batch, restored).shape == (1, 1)
    with pytest.raises(FileExistsError):
        algorithm.fit_stream(loaders, backend.hidden, output_dir=tmp_path / "run")
