"""Qwen2.5-VL 4.51.3 模型边界；算法不依赖服务器路径或旧pipeline。"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from .algorithm import HiddenBatch
from .replacement import masked_mean


@dataclass
class PreparedBatch:
    inputs: dict[str, Any]
    question_mask: torch.Tensor


def question_mask_from_prompts(tokenizer, input_ids, attention_mask, prompts, questions):
    """在完整prompt上用offset定位，再匹配processor展开后的token序列。

    不对问题单独分词，不把回答格式指令纳入表示。歧义输入显式拒绝，避免静默写错位置。
    """
    if len(prompts) != len(input_ids) or len(questions) != len(input_ids):
        raise ValueError("prompt/question batch size mismatch")
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i, (prompt, question) in enumerate(zip(prompts, questions)):
        if not question or prompt.count(question) != 1:
            raise ValueError("question must occur exactly once in canonical prompt")
        start = prompt.index(question)
        end = start + len(question)
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        ids = [
            int(token)
            for token, (lo, hi) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True)
            if hi > start and lo < end
        ]
        positions = attention_mask[i].bool().nonzero().flatten()
        actual = input_ids[i, positions].tolist()
        matches = [j for j in range(len(actual) - len(ids) + 1) if actual[j : j + len(ids)] == ids]
        if not ids or len(matches) != 1:
            raise ValueError("question tokens absent or ambiguous after image expansion")
        mask[i, positions[matches[0] : matches[0] + len(ids)]] = True
    return mask


class _ReachedLayer(Exception):
    pass


def _unpack(output):
    if torch.is_tensor(output):
        return output, None
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0], output[1:]
    raise TypeError("unsupported decoder output")


class QwenBackend:
    def __init__(self, model, *, layer_index=7):
        if getattr(model.config, "model_type", None) != "qwen2_5_vl":
            raise TypeError("this backend only supports Qwen2.5-VL")
        if not 0 <= layer_index < len(model.model.layers):
            raise ValueError("invalid decoder layer index")
        self.model = model.eval().requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        self.layer = model.model.layers[layer_index]
        self.layer_index = layer_index
        self._active = False

    @classmethod
    def from_pretrained(cls, model_path, *, device, layer_index=7):
        # 仅显式调用才加载本地权重；不自动下载、不自动挑选GPU。
        import transformers

        if transformers.__version__ != "4.51.3":
            raise RuntimeError("Qwen backend requires the validated transformers==4.51.3 layout")
        model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to(device)
        return cls(model, layer_index=layer_index)

    def _validate(self, batch):
        if any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Qwen backbone must remain frozen")
        self.model.eval()
        ids = batch.inputs.get("input_ids")
        mask = batch.question_mask
        if ids is None or ids.ndim != 2 or mask.shape != ids.shape or mask.dtype != torch.bool:
            raise ValueError("explicit real input_ids and boolean question mask are required")
        if not mask.any(1).all():
            raise ValueError("empty question span")
        attention = batch.inputs.get("attention_mask")
        if attention is None or attention.shape != mask.shape or (mask & ~attention.bool()).any():
            raise ValueError("question mask overlaps padding or attention mask is absent")
        if (mask & ids.eq(self.model.config.image_token_id)).any():
            raise ValueError("question mask overlaps image tokens")
        if any(
            batch.inputs.get(key) is not None
            for key in (
                "labels",
                "past_key_values",
                "inputs_embeds",
                "pixel_values_videos",
                "video_grid_thw",
            )
        ):
            raise ValueError(
                "backend accepts image prompt inputs only; cached embeds/answers are not accepted"
            )

    @contextmanager
    def _hook(self, function):
        if self._active:
            raise RuntimeError("nested/reentrant backend calls are not supported")
        self._active = True
        handle = self.layer.register_forward_hook(function)
        try:
            yield
        finally:
            handle.remove()
            self._active = False

    @torch.no_grad()
    def hidden(self, batch):
        """前缀只跑到L7；输出普通detach tensor，允许后续adapter训练使用。"""
        self._validate(batch)
        captured = []

        def capture(_module, _args, output):
            captured.append(_unpack(output)[0].detach())
            raise _ReachedLayer()

        inputs = dict(batch.inputs, use_cache=False, output_hidden_states=False)
        with self._hook(capture):
            try:
                self.model(**inputs)
            except _ReachedLayer:
                pass
        if len(captured) != 1:
            raise RuntimeError("selected decoder layer was not reached exactly once")
        return HiddenBatch(captured[0], batch.question_mask)

    @torch.no_grad()
    def collect_geometry_hidden(self, loaders, *, partition="train"):
        if partition != "train" or set(loaders) != {"prototype", "neighbor", "utility"}:
            raise ValueError("geometry requires train prototype/neighbor/utility")
        result = {}
        for split, loader in loaders.items():
            rows = []
            for batch in loader:
                hidden = self.hidden(batch)
                rows.append(masked_mean(hidden.hidden, hidden.question_mask).float().cpu())
            if not rows:
                raise ValueError(f"empty geometry loader: {split}")
            result[split] = torch.cat(rows)
        return result

    @torch.no_grad()
    def generate(self, batch, algorithm, *, max_new_tokens=10):
        self._validate(batch)
        if algorithm.config.layer_index != self.layer_index:
            raise ValueError("algorithm and backend layers differ")
        prefilled = False

        def write(_module, _args, output):
            nonlocal prefilled
            hidden, tail = _unpack(output)
            if prefilled:
                if hidden.shape[:2] != (len(batch.question_mask), 1):
                    raise RuntimeError("expected cached single-token decode")
                return output
            if hidden.shape[:2] != batch.question_mask.shape:
                raise RuntimeError("prefill hidden/mask mismatch")
            prefilled = True
            changed = algorithm.replacement(hidden, batch.question_mask).hidden
            return changed if tail is None else (changed, *tail)

        self.model.eval()
        with self._hook(write):
            output = self.model.generate(
                **dict(
                    batch.inputs,
                    use_cache=True,
                    max_new_tokens=max_new_tokens,
                    num_beams=1,
                    do_sample=False,
                    return_dict_in_generate=False,
                )
            )
        if not prefilled:
            raise RuntimeError("generation never visited the selected layer")
        # 原始 IDs 路径由HF处理视觉与RoPE，不使用曾需要补丁的inputs_embeds缓存路径。
        return output[:, batch.inputs["input_ids"].shape[1] :]
