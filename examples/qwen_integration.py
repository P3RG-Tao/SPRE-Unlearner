"""把正式数据加载器接到独立算法库；本文件没有自动启动入口。"""

from pathlib import Path

from spre_unlearner import SPREUnlearner, ReplacementConfig
from spre_unlearner.qwen import QwenBackend


def train_qwen(
    *, model_path, device, geometry_loaders, train_loaders, identity, output_dir, evaluate_dev
):
    """调用方先完成正式预检；loaders产生PreparedBatch且shuffle=False。

    geometry_loaders: train的prototype/neighbor/utility，完整遍历一次。
    train_loaders: train的forget/neighbor/utility，按原顺序循环。
    evaluate_dev(backend, algorithm, step, checkpoint): 外部评估回调。
    device必须显式指定；本函数不分配GPU、不下载权重、不启动队列。
    """
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("refusing to overwrite an existing run")
    config = ReplacementConfig()
    backend = QwenBackend.from_pretrained(model_path, device=device, layer_index=config.layer_index)
    train_hidden = backend.collect_geometry_hidden(geometry_loaders, partition="train")
    algorithm = SPREUnlearner.prepare(
        train_hidden, identity=identity, config=config, device=device
    )
    algorithm.save(root / "aligned.pt")

    def evaluate(current, checkpoint, row):
        evaluate_dev(backend, current, row["step"], checkpoint)

    algorithm.fit_stream(
        train_loaders, backend.hidden, output_dir=root / "checkpoints", on_checkpoint=evaluate
    )
    return backend, algorithm
