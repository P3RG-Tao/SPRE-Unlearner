"""默认仅输出配置；smoke显式执行CPU合成数据测试，不加载真实模型。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch

from . import SPREUnlearner, AlignConfig, HiddenBatch, ReplacementConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("config", "smoke"), nargs="?", default="config")
    args = parser.parse_args()
    if args.action == "config":
        print(json.dumps(asdict(ReplacementConfig()), indent=2))
        return
    torch.set_num_threads(1)
    torch.manual_seed(1)
    config = ReplacementConfig(
        rank=3,
        width=8,
        codebook_size=4,
        max_steps=4,
        checkpoint_every=2,
        align=AlignConfig(steps=3, hidden_dim=16, output_dim=8, batch_size=4),
    )
    hidden = {s: torch.randn(12, 16) for s in ("prototype", "neighbor", "utility")}
    identity = dict(
        target="synthetic", model_revision="no-model", data_fingerprint="synthetic-seed-1"
    )
    algorithm = SPREUnlearner.prepare(hidden, config=config, identity=identity)
    mask = torch.tensor([[False, True, True, False]])
    batches = {
        s: [HiddenBatch(torch.randn(1, 4, 16), mask)] for s in ("forget", "neighbor", "utility")
    }
    with tempfile.TemporaryDirectory() as directory:
        algorithm.fit(batches, output_dir=directory)
        restored, _ = SPREUnlearner.load(
            Path(directory) / "step_00004.pt", expected_identity=identity
        )
        print(
            json.dumps(
                dict(
                    passed=restored.step == 4,
                    align_steps=3,
                    train_steps=4,
                    checkpoints=2,
                    device="cpu",
                    real_model_loaded=False,
                )
            )
        )


if __name__ == "__main__":
    main()
