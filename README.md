# SPRE-Unlearner

**Structure-Preserving Metric-Guided Representation Editing for Multimodal Concept Unlearning**

Repository: <https://github.com/P3RG-Tao/SPRE-Unlearner>

Python package: `spre_unlearner`; main algorithm class: `SPREUnlearner`.

This release contains the extracted algorithm and a Qwen2.5-VL backend. It is not yet a
standalone reproduction package for every model and experiment in the paper. CPU
regression tests pass; real-model regression of this refactored package remains pending.

这是SPRE-Unlearner的独立算法实现，从历史冻结协议 `qwen_replacement_v2_r1` 提取。历史协议名仅用于追溯，不是论文方法名；对外名称统一为SPRE-Unlearner。
不再依赖旧 `training_core.py`、诊断脚本、服务器绝对路径或运行时 monkey patch。

**当前状态：算法级CPU回归通过，真实Qwen回归尚未执行。** 本目录不替换服务器冻结版本，不自动恢复任何实验。
这是算法代码库，不是包含数据、模型权重、任务队列和LLM服务的整套实验平台。

## 目录

| 文件 | 职责 |
| --- | --- |
| `src/spre_unlearner/config.py` | 类型化配置，正式默认参数与合法性检查 |
| `src/spre_unlearner/alignment.py` | GatedBottleneckE及一次性对齐训练 |
| `src/spre_unlearner/geometry.py` | 固定E原型、hidden目标子空间、retain码本 |
| `src/spre_unlearner/replacement.py` | 耦合、无门控、样本级Replacement写回 |
| `src/spre_unlearner/algorithm.py` | 遗忘损失、优化、保存/恢复、dev选点 |
| `src/spre_unlearner/qwen.py` | 问题token定位、冻结前缀提取、生成时挂接 |
| `examples/qwen_integration.py` | 接入现有处理器与数据加载器的完整算法调用 |
| `tests/` | 公式/梯度/优化器/恢复/模型边界回归 |
| `tests/reference/` | 从冻结源码提取的只读测试参照，不被运行时代码导入 |
| `PROVENANCE.json` | 历史来源SHA256、提取符号和参照文件SHA256 |
| `configs/qwen_formal.json` | 由配置类生成的正式算法默认参数 |

## 算法步骤

### 1. 对齐E

Qwen主干冻结。提取零基索引L7，即第8个decoder block输出的**原始问题token均值**。
只用train中的prototype、neighbor、utility，不使用答案embedding，不读取dev/test拟合几何。

E为gated MLP：`E(h) = W_out(GELU(W_value h) * sigmoid(W_gate h))`。
训练目标为：

```text
L_align = 300 * L_structure + L_prototype_neighbor + 0.5 * L_prototype_utility
```

结构损失保持batch内hidden两两余弦矩阵；类别项使用softplus比较prototype样本与负类相对于prototype中心的相似度差。
E仅训练一次1000步，随后冻结。原实现对齐使用AdamW默认weight_decay=0.01，本库明确保留。

### 2. 构造三个固定参照

- E空间原型 `p`：归一化每个prototype的E输出，求均值，再归一化；供遗忘损失使用。
- hidden目标基底 `B`：对prototype相对retain中心、neighbor、utility的差分矩阵做SVD，取16个正交方向。
- retain码本 `C`：将neighbor和utility映射到目标子空间，用确定性k-means得到16个中心。

其中retain中心 `mu` 是neighbor和utility全部样本的合并均值。顺序差分继承原实现，**不是经过语义匹配的样本对**。
历史tau只保存在诊断元数据里，当前无门控算法不使用tau控制写回或截断遗忘loss。

### 3. Replacement

对每个样本只预测一次替代系数，向该样本全部原始问题token写入同一偏移：

```text
q = mean(question tokens)
c = (q - mu) @ B.T
z = SiLU(W_down LayerNorm(q))
a = softmax(W_codebook z) @ C
r = R * (W_extra z) / (R + ||W_extra z||), R=128
delta = ((a + r - c) * clamp(strength, 0, 1)) @ B
h_question' = h_question + delta
```

删除和写入共享同一逐维strength；没有独立写入强度，没有target gate，也没有第二个干预层。
图像、模板、回答格式指令等token不直接修改。后续decoder层继续使用已修改的问题状态。

### 4. 遗忘训练

```text
L_forget = mean(cos(E(q_forget'), p))
L_retain = 0.5 * (mean(1-cos(q_neighbor', q_neighbor))
                + mean(1-cos(q_utility', q_utility)))
L_total = L_forget + 1.0 * L_retain
```

只更新Replacement，E参数冻结但保留其对输入的梯度。主干和几何参照不更新。
每步各取一个F/N/U batch，正式每个batch为1个样本，配置会检查该大小。AdamW lr=5e-4、weight_decay=0、梯度裁剪1，更新后把strength投影到[0,1]。
连续8000步，每500步保存。**没有ReLU/tau遗忘截断、答案抑制、QA/KL、TIN训练、guard、rollback或early stopping。**

### 5. 评估边界

`on_checkpoint`回调用于接入原有dev评估；`select_operating_point`只选择最低dev Forget QA，同分按保留类宏平均、较早step排序。
test与LLM仅用于报告，不参与选点。VQA答案标准化、正式题单、judge配置和队列仍由实验系统管理，不在本库重新定义。

## 使用

推荐使用Python3.10和原正式环境PyTorch2.4.0。Qwen后端固定Transformers4.51.3；CPU算法测试也可在其它支持版本执行。

```bash
python -m pip install -e '.[test,qwen]'
python -m spre_unlearner config
python -m spre_unlearner smoke
python -m pytest -q
```

默认命令仅打印配置；`smoke`只使用CPU随机张量，在临时目录验证对齐、4步训练、保存和恢复，不加载真实模型、不请求API、不占GPU。
实际接入见`examples/qwen_integration.py`。输入使用现有正式数据加载器和generation profile，不随意重写prompt。
`PreparedBatch`必须包含真实input_ids、attention_mask和精确问题mask；可用`question_mask_from_prompts`按完整prompt的offset构造。
输入tensor须提前移到模型设备，pixel_values等浮点输入按原processor/模型精度处理；mask为bool。不支持多视频、beam search或同一backend并发调用。
数据身份应包括划分清单哈希、题目顺序、batch配置、prompt/processor版本；模型身份使用明确revision或权重指纹，不能只写一个可变目录名。

## 与历史源码的边界

算法公式、默认参数、初始化顺序、E空间原型归一化、BF16写回顺序与正式路径保持一致。
重构仅保留coupled/ungated/samplewise/full/uniform/bounded-codebook分支。

工程层有以下明确变化，不能把CPU一致性冒充真实模型端到端复现：

- Qwen适配器走原始input_ids和pixel_values，不走旧pipeline的缓存inputs_embeds生成路径，因此不需要该路径的RoPE补丁；**不宣称速度与原缓存实现相同**。
- 训练前缀到L7即退出，再对冻结hidden优化adapter。支持流式输入，也支持外部缓存的完整token hidden；不要只缓存池化均值来替代BF16逐token写回。
- 新增数据/模型/target绑定，保存优化器；拒绝覆盖已有checkpoint，生成异常后自动移除hook。
- 本地checkpoint是新格式，不能直接当旧服务器checkpoint加载。保存采样位置不等于任意数据加载器的精确续训；只对相同稳定顺序和相同数据身份的加载器提供续接流程。
- 缺少原始数据、权重、显卡真实生成回归和正式预检。正式使用前须建立独立修订、验证模型scope/环境/数据/输出隔离，再做固定样本Qwen生成与loss回归；不得覆盖旧协议。

本仓库不包含模型权重、数据集、实验预测结果或密钥。发布版测试证据见`TEST_RESULTS.json`。

## 名称与来源

面向用户的入口统一为`SPRE-Unlearner`、`spre_unlearner`和`SPREUnlearner`。测试参照、来源清单与checkpoint的`source_protocol`保留历史标识，不能通过改写历史代码制造新的验证证据。

本次发布不擅自指定开源许可证；代码公开不等于授予任意使用或再分发许可。许可证和正式论文引用信息待作者确认。
