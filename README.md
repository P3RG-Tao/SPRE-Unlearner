# SPRE-Unlearner

**Structure-Preserving Metric-Guided Representation Editing for Multimodal Concept Unlearning**

SPRE-Unlearner is a representation-editing approach to targeted concept unlearning in multimodal language models. It learns a structure-preserving metric space, identifies a target-related subspace, and trains a lightweight replacement module to modify question-token representations while preserving representations of neighboring and unrelated concepts. The pretrained backbone remains frozen.

This repository provides the algorithm implementation, a **Qwen2.5-VL backend**, configuration defaults, and regression tests.

## Method Overview

SPRE-Unlearner consists of three stages:

1. **Learn the metric space.** Train a gated projection network, `E`, on frozen multimodal representations. A structure-preservation objective maintains pairwise similarities, while contrastive objectives distinguish target-related prototype samples from neighbor and utility samples. After this stage, `E` is frozen.
2. **Build fixed geometric references.** Construct a target prototype in the learned metric space, a low-rank target basis in the backbone hidden space, and a retain codebook from neighbor and utility representations. All references are fitted on training data only.
3. **Optimize representation replacement.** Train a lightweight module that replaces target-subspace coefficients of question-token representations. Minimize target similarity in the learned metric space while penalizing changes to neighbor and utility representations. Only the replacement module is updated.

```text
Image + question
       |
Frozen decoder prefix
       |
Question-token hidden states
       |
Trainable subspace replacement
       |
Frozen decoder suffix
       |
Generated answer
```

The replacement module predicts one offset per sample and applies it to the original question tokens. Image tokens, prompt-template tokens, and generated answer tokens are not directly edited. The subsequent decoder layers process the modified question states.

### Training Objective

Let `q` and `q'` denote the mean question-token representation before and after replacement, and let `p` be the fixed target prototype in the metric space. The replacement objective is:

```text
L_forget = mean cos(E(q_forget'), p)

L_retain = 0.5 * [mean(1 - cos(q_neighbor', q_neighbor))
                + mean(1 - cos(q_utility', q_utility))]

L = L_forget + lambda_retain * L_retain
```

Gradients pass through the frozen `E` to the replacement module. The backbone, metric network, and geometric references remain fixed during unlearning. This implementation does not use answer-level supervision or suppression, a similarity-threshold gate, or loss truncation.

## Installation

Python **3.10 or newer** and PyTorch **2.4 or newer, below 3.0** are required. The Qwen backend requires **Transformers 4.51.3**. Real-model use requires a device supporting the backend's BF16 model loading.

```bash
git clone https://github.com/P3RG-Tao/SPRE-Unlearner.git
cd SPRE-Unlearner

# Algorithm core and test dependencies
python -m pip install -e '.[test]'

# Optional: Qwen model integration
python -m pip install -e '.[qwen]'
```

Model weights and datasets are not included. `QwenBackend.from_pretrained` loads an existing local model directory; it does not download weights automatically.

## Quick Start

Inspect the default configuration and run the synthetic CPU smoke test:

```bash
python -m spre_unlearner config
python -m spre_unlearner smoke
python -m pytest -q
```

The smoke test exercises alignment, replacement training, checkpoint saving, and restoration using synthetic tensors. It does not load a language model, use a GPU, or contact an external service.

### Qwen Integration

The [integration example](examples/qwen_integration.py) demonstrates the training lifecycle. Supply a local model, prepared data loaders, an explicit device, and a development-set evaluation callback:

```python
from examples.qwen_integration import train_qwen

# geometry_loaders, train_loaders, and evaluate_dev are supplied by your pipeline.
backend, unlearner = train_qwen(
    model_path="/path/to/Qwen2.5-VL-7B-Instruct",
    device="cuda:0",
    geometry_loaders=geometry_loaders,
    train_loaders=train_loaders,
    identity={
        "target": "cat",
        "model_revision": "<immutable model revision or weight fingerprint>",
        "data_fingerprint": "<data, order, batching, and preprocessing fingerprint>",
    },
    output_dir="outputs/cat",
    evaluate_dev=evaluate_dev,
)
```

The output directory must not already exist. The example is an integration API, not a standalone dataset-training command.

**Data contract**

| Input | Required content |
| --- | --- |
| `geometry_loaders` | Training-only `prototype`, `neighbor`, and `utility` loaders |
| `train_loaders` | Nonempty, re-iterable `forget`, `neighbor`, and `utility` loaders in a stable order |
| Each loader batch | `PreparedBatch(inputs=..., question_mask=...)` |
| `inputs` | Processor-produced model inputs, including real `input_ids`, `attention_mask`, and image inputs |
| `question_mask` | Boolean mask matching `input_ids`, covering only the original question tokens |
| `evaluate_dev` | Callback with signature `(backend, unlearner, step, checkpoint)` |

Use `question_mask_from_prompts` to locate question tokens in the complete prompt after image-token expansion. Move input tensors to the model device before use, and use the model-compatible dtype for floating-point image inputs. Geometry fitting and training must not consume development or test samples.

Generate answers with the trained replacement module attached:

```python
answer_ids = backend.generate(prepared_batch, unlearner, max_new_tokens=10)
```

Generation is greedy and uses the model's KV cache. The current backend supports image prompts, not video inputs, beam search, or concurrent calls on the same backend instance.

## Default Configuration

The complete configuration is available in [configs/qwen_formal.json](configs/qwen_formal.json) and exposed through `ReplacementConfig`.

| Component | Default |
| --- | --- |
| Intervention point | Output of decoder block index 7, the eighth block |
| Metric network `E` | Gated MLP, hidden width 768, output dimension 512 |
| Alignment budget | 1,000 steps, performed once |
| Alignment batch size | 64 |
| Alignment weights | Structure: 300; prototype-neighbor: 1; prototype-utility: 0.5 |
| Contrastive margin / temperature | 0.15 / 0.05 |
| Target-subspace rank | 16 |
| Replacement bottleneck width | 32 |
| Retain codebook size | 16 |
| Residual coefficient radius | 128 |
| Replacement training budget | 8,000 steps |
| Replacement batch size | One sample per split per step |
| Learning rate | `5e-4` for alignment and replacement |
| Retain loss weight | 1.0 |
| Checkpoint interval | 500 steps |

Replacement uses coupled deletion and insertion strengths, without a target gate. The target basis is constructed by SVD, and the retain codebook uses deterministic k-means. The difference vectors used to build the basis are not semantically matched sample pairs.

## Evaluation and Checkpoints

Training exposes an `on_checkpoint` callback for external evaluation. `select_operating_point` selects the checkpoint with the **lowest development-set Forget QA**. Ties are resolved by the mean QA of neighbor, utility, and target-image/non-target samples, then by the earlier training step. Test QA and LLM quality scores are not used for checkpoint selection.

Checkpoints store the configuration, metric network, geometric references, replacement parameters, optimizer state, training step, and target/model/data identity. Loading requires a matching identity. Resuming a data stream requires the same stable sample order and batching; arbitrary shuffled-loader state is not restored. Existing checkpoints are never overwritten.

Dataset preparation, VQA answer normalization, benchmark manifests, and LLM judging are external to this package.

## Repository Structure

```text
configs/                     Default algorithm configuration
examples/                    Qwen integration example
src/spre_unlearner/
    alignment.py             Metric-network alignment
    geometry.py              Prototype, target basis, and retain codebook
    replacement.py           Coupled subspace replacement
    algorithm.py             Training, checkpoints, and dev selection
    qwen.py                  Qwen hidden extraction and generation hooks
    config.py                Typed configuration and validation
tests/                       Algorithm and backend regression tests
tools/verify.py              CPU validation and checksum generation
```

## Validation and Release Scope

The current release passes **18 CPU regression tests**, including comparisons against frozen reference implementations, forward and gradient checks, checkpoint restoration, and synthetic backend tests. See [TEST_RESULTS.json](TEST_RESULTS.json) for the recorded environment and results, [PROVENANCE.json](PROVENANCE.json) for source provenance, and [SHA256SUMS](SHA256SUMS) for file checksums.

**End-to-end regression with a real Qwen model has not yet been completed for this refactored release.** The repository currently includes the algorithm core and Qwen backend, not the complete multi-model benchmark pipeline. Model weights, datasets, and experiment outputs are not distributed here. CPU test results should not be interpreted as full reproduction of the paper's experiments.

## License

A license has not yet been specified. Public availability does not grant additional permission to use, modify, or redistribute the code beyond applicable rights.
