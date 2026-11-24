import copy
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from spre_unlearner import SPREUnlearner, HiddenBatch, AlignConfig, ReplacementConfig
from spre_unlearner.alignment import align_predictor
from spre_unlearner.algorithm import select_operating_point
from spre_unlearner.geometry import build_target_basis, build_codebook

sys.path.insert(0, str(Path(__file__).parent))
from reference import alignment as old_align
from reference import geometry as old_geometry
from reference.replacement import PostLayerTargetIntervention

torch.set_num_threads(1)
IDENTITY = dict(target="synthetic", model_revision="toy-v1", data_fingerprint="synthetic-seed-8")


@pytest.fixture
def config():
    return ReplacementConfig(
        rank=3,
        width=5,
        codebook_size=4,
        max_steps=4,
        checkpoint_every=2,
        train_batch_size=2,
        align=AlignConfig(steps=3, hidden_dim=8, output_dim=6, batch_size=4),
    )


@pytest.fixture
def hidden():
    generator = torch.Generator().manual_seed(8)
    return {
        s: torch.randn(n, 12, generator=generator)
        for s, n in (("prototype", 8), ("neighbor", 9), ("utility", 10))
    }


@pytest.fixture
def algorithm(config, hidden):
    return SPREUnlearner.prepare(hidden, config=config, identity=IDENTITY)


def batches(dtype=torch.float32):
    generator = torch.Generator().manual_seed(22)
    mask = torch.tensor([[False, True, True, True, False], [False, False, True, True, False]])
    return {
        s: HiddenBatch(torch.randn(2, 5, 12, generator=generator).to(dtype), mask)
        for s in ("forget", "neighbor", "utility")
    }


def reference(algorithm):
    g, c = algorithm.geometry, algorithm.config
    predictor = copy.deepcopy(algorithm.predictor)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c.seed)
        result = PostLayerTargetIntervention(
            12,
            mode="ungated_replacement",
            predictor=predictor,
            target_prototype=g["target_prototype"],
            target_basis=g["target_basis"],
            retain_center=g["retain_center"],
            tau=g["diagnostic_tau"],
            replacement_hidden_dim=c.width,
            replacement_scope="samplewise",
            replacement_conditioning="full",
            replacement_distribution="uniform",
            replacement_output_space="retain_codebook_bounded",
            replacement_codebook=g["replacement_codebook"],
            replacement_extrapolation_max_norm=c.radius,
            initial_strength=c.initial_strength,
        )
    return result


def test_alignment_matches_frozen(hidden, config):
    c = config.align
    new, history = align_predictor(hidden, c, seed=config.seed)
    old, old_history = old_align.train_predictor(
        hidden,
        hidden,
        objective="contrastive_structure",
        output_dim=c.output_dim,
        hidden_dim=c.hidden_dim,
        steps=c.steps,
        per_class_batch_size=c.batch_size,
        lr=c.learning_rate,
        structure_weight=c.structure_weight,
        contrastive_weight=c.contrastive_weight,
        neighbor_weight=c.neighbor_weight,
        utility_weight=c.utility_weight,
        contrastive_margin=c.margin,
        contrastive_temperature=c.temperature,
        seed=config.seed + 100,
        device=torch.device("cpu"),
    )
    assert history == old_history
    for name, value in new.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)


def test_geometry_matches_frozen(hidden, config):
    basis, center = build_target_basis(hidden, config.rank)
    old_basis, old_center, _ = old_geometry.build_target_basis(
        hidden, rank=config.rank, device=torch.device("cpu")
    )
    torch.testing.assert_close(basis, old_basis, rtol=0, atol=0)
    torch.testing.assert_close(center, old_center, rtol=0, atol=0)
    codebook = build_codebook(hidden, basis, center, config.codebook_size)
    expected, _ = old_geometry.build_retain_coefficient_codebook(
        hidden, target_basis=basis, retain_center=center, codebook_size=config.codebook_size
    )
    torch.testing.assert_close(codebook, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_gradient_and_adam_match_frozen(algorithm, dtype):
    old = reference(algorithm)
    for name, value in algorithm.replacement.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[name], rtol=0, atol=0)
    # 非零的头和较强强度可以检查真实写回，避免初始近恒等映射掩盖错误。
    with torch.no_grad():
        algorithm.replacement.replacement_strength.fill_(0.3)
        algorithm.replacement.replacement_up.weight.fill_(0.03)
        algorithm.replacement.replacement_extrapolation_up.weight.fill_(0.02)
    old.load_state_dict(algorithm.replacement.state_dict(), strict=False)
    data = batches(dtype)
    captures = {}
    for name, batch in data.items():
        old.set_runtime_masks(torch.zeros_like(batch.question_mask), batch.question_mask)
        expected = old._output_hook(None, (), batch.hidden)
        captures[name] = old.take_capture()
        actual = algorithm.replacement(batch.hidden, batch.question_mask)
        torch.testing.assert_close(actual.hidden, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual.before, captures[name].before, rtol=0, atol=0)
        torch.testing.assert_close(actual.after, captures[name].after, rtol=0, atol=0)
        assert torch.equal(actual.hidden[~batch.question_mask], batch.hidden[~batch.question_mask])
    forget = (
        F.normalize(old.predictor(captures["forget"].after.float()).float(), dim=-1)
        @ old.target_prototype
    ).mean()
    retain = 0.5 * sum(
        1
        - F.cosine_similarity(captures[s].after.float(), captures[s].before.float(), dim=-1).mean()
        for s in ("neighbor", "utility")
    )
    old_loss = forget + algorithm.config.retain_hidden_weight * retain
    losses = algorithm.losses(**data)
    torch.testing.assert_close(losses["total"], old_loss, rtol=0, atol=0)
    losses["total"].backward()
    old_loss.backward()
    for name, parameter in algorithm.replacement.named_parameters():
        torch.testing.assert_close(
            parameter.grad, dict(old.named_parameters())[name].grad, rtol=0, atol=0
        )
    optimizer = torch.optim.AdamW(
        old.intervention_parameters(), lr=algorithm.config.learning_rate, weight_decay=0.0
    )
    torch.nn.utils.clip_grad_norm_(old.intervention_parameters(), 1.0)
    optimizer.step()
    old.project_constraints()
    algorithm.train_step(**data)
    for name, parameter in algorithm.replacement.named_parameters():
        torch.testing.assert_close(parameter, dict(old.named_parameters())[name], rtol=0, atol=0)


def test_checkpoint_restores_next_update(algorithm, tmp_path):
    algorithm.train_step(**batches())
    path = tmp_path / "checkpoint.pt"
    algorithm.save(path, data_state={"batch_cursor": 1})
    restored, state = SPREUnlearner.load(path, expected_identity=IDENTITY)
    assert state == {"batch_cursor": 1}
    assert restored.step == 1
    assert algorithm.train_step(**batches()) == restored.train_step(**batches())
    for name, value in algorithm.replacement.state_dict().items():
        torch.testing.assert_close(value, restored.replacement.state_dict()[name], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        algorithm.save(path)
    with pytest.raises(ValueError, match="identity"):
        SPREUnlearner.load(path, expected_identity=dict(IDENTITY, target="other"))


def test_fit_budget_callbacks_and_freeze(algorithm, tmp_path):
    original = copy.deepcopy(algorithm.predictor.state_dict())
    seen = []
    algorithm.fit(
        {s: [v] for s, v in batches().items()},
        output_dir=tmp_path,
        on_checkpoint=lambda a, p, row: seen.append(a.step),
    )
    assert seen == [2, 4]
    assert len(list(tmp_path.glob("*.pt"))) == 2
    for key, value in original.items():
        assert torch.equal(value, algorithm.predictor.state_dict()[key])
    assert all(p.grad is None for p in algorithm.predictor.parameters())
    with pytest.raises(RuntimeError, match="budget"):
        algorithm.train_step(**batches())


def test_reject_bad_geometry_and_masks(algorithm, hidden, config):
    with pytest.raises(ValueError, match="train"):
        SPREUnlearner.prepare(hidden, config=config, identity=IDENTITY, partition="test")
    with pytest.raises(ValueError, match="rank"):
        build_target_basis(hidden, 99)
    batch = batches()["forget"]
    with pytest.raises(ValueError, match="question"):
        algorithm.replacement(batch.hidden, torch.zeros_like(batch.question_mask))
    with pytest.raises(ValueError):
        ReplacementConfig(radius=float("nan"))
    with pytest.raises(ValueError):
        AlignConfig(temperature=0)


def test_select_no_threshold_and_no_test_leak():
    def row(step, forget, retain):
        return dict(
            step=step,
            partition="dev",
            forget=forget,
            neighbor=retain,
            utility=retain,
            target_image_nontarget=retain,
        )

    rows = [row(500, 0.2, 0.9), row(1000, 0.01, 0.3), row(1500, 0.01, 0.4), row(2000, 0.01, 0.4)]
    assert select_operating_point(rows)["step"] == 1500
    with pytest.raises(ValueError, match="dev"):
        select_operating_point([dict(rows[0], partition="test")])


def test_bounded_extrapolation_and_strength(algorithm):
    module = algorithm.replacement
    with torch.no_grad():
        module.replacement_extrapolation_up.weight.fill_(1000)
        module.replacement_strength.fill_(3)
        module.project_constraints()
    assert torch.equal(module.replacement_strength, torch.ones_like(module.replacement_strength))
    source = torch.randn(5, 12)
    feature = F.silu(module.replacement_down(module.token_norm(source)))
    anchor = torch.softmax(module.replacement_up(feature), -1) @ module.replacement_codebook
    assert ((module.alternative(source) - anchor).norm(dim=-1) < module.radius).all()


def test_batch_size_guard_and_zero_strength(algorithm):
    data = batches()
    with torch.no_grad():
        algorithm.replacement.replacement_strength.zero_()
    sample = data["forget"]
    output = algorithm.replacement(sample.hidden, sample.question_mask)
    assert torch.equal(output.hidden, sample.hidden)
    data["forget"] = HiddenBatch(sample.hidden[:1], sample.question_mask[:1])
    with pytest.raises(ValueError, match="batch size"):
        algorithm.train_step(**data)
