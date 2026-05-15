# Prefix Sharing 详细设计规格书

> 本文档是当前开发的主规格文档。目标读者包括后续接手开发的工程师，以及刚进入 prefix sharing / verl / Megatron 训练链路的开发者。历史讨论、被推翻方案、备选方案和阶段性分析记录见 `doc-design-history.md`。

---

## 1. 背景、目标与边界

### 1.1 背景

在 RLHF / RL 训练中，actor logprob 与 actor update 经常在同一个 micro-batch 内处理共享 prompt 或长前缀的多条样本。普通 causal LM forward 会对每条样本重复计算相同 prefix 的 attention、KV 和后续 logits。这部分重复计算在长 prompt、组内多 response、DPO/GRPO/RL rollout 场景中会成为明显开销。

本项目的目标是在 `rllm + verl + Megatron` 链路中实现 prefix sharing：同一 micro-batch 内，某些样本只计算 suffix query，并在 attention 内复用其他样本已经计算出的 prefix K/V，从而减少重复计算，同时保持 logprob、loss 和梯度语义一致。

阶段一选择非侵入式 patch / wrapper / monkey-patch 方案。这样可以快速验证语义和链路正确性，并为后续 verl / Megatron 正式 hook 或上游 PR 准备清晰的核心模块边界。

### 1.2 总目标

实现一个可迁移、可测试、可逐步产品化的 prefix sharing 方案：

- `core/` 保持框架无关，表达 prefix sharing 的核心语义、计划和坐标映射。
- `backends/` 消费统一 metadata，处理不同硬件和 attention kernel 的执行细节。
- `integrations/` 只负责接入 verl / Megatron，不沉淀核心算法。
- patch 层可启停、可回滚、可校验，不污染非 prefix sharing 路径。
- 正确性优先于性能；性能优化进入后续阶段。

### 1.3 Phase 1 范围

阶段一聚焦 `verl + Megatron` 的 RL MVP：

- 支持 actor logprob / update 主路径。
- 使用 `TriePrefixDetector` 自动识别 micro-batch 内可复用 prefix。
- 支持 per-sample reuse relation：同一个 provider 可向不同 reuser 提供不同长度的 prefix slice。
- reuse 样本裁剪 prefix，仅计算 suffix query。
- attention 内通过 KV injection 复用 provider prefix K/V。
- logprob 对齐使用 Prefix-Last Restore，保证第一个 suffix token 的 next-token 语义正确。
- 对 verl 和 Megatron 的源码改动保持极小：只允许 import / 单个 helper 调用 / context manager 这类特性使能入口；核心逻辑必须沉淀在 `prefix_sharing.integrations`。
- 本地 CPU 侧开发者测试必须通过；CUDA / CANN / verl 等依赖缺失的测试可作为 optional 测试保留但不强制运行。

阶段一启动边界：

- `pipeline_model_parallel_size == 1`
- `context_parallel_size == 1`
- `apply_rope_fusion == False`
- `fused_single_qkv_rope == False`
- 普通 text causal LM，不覆盖 VLM、多模态 position、mRoPE、MLA 等结构。

### 1.4 Phase 1 非目标

阶段一不交付以下能力：

- PP / CP / EP 的完整业务落地能力。
- 高性能 fused attention、TE fused RoPE、CANN 专用融合算子。
- VLM、多模态位置编码、mRoPE、MLA。
- standalone Megatron / FSDP 的 SFT 或 pretrain。
- verl 或 Megatron 上游 PR。
- 跨 micro-batch 或跨 forward 的持久 KV cache。

这些能力进入后续 roadmap。阶段一必须把语义和模块边界设计成可以向这些能力演进，而不是写死在 MVP 假设里。

### 1.5 与 survey / dependency 中参考项目的关系

`prefix-0501` 暂时以整个项目为版本管理颗粒度，`survey/`、`dependency/`、`refactor/prefix-sharing/` 一起管理。`refactor/prefix-sharing/` 是当前实际开发目录；后续可以再独立拆分为开源仓库。

参考项目的定位如下：

- `dependency/PrefixTrain_dev`：最重要的设计参考。其在线 Trie 思路支持 provider 面向不同 reuser 提供不同长度的子序列复用。Phase 1 采用同类 per-sample reuse relation，而不是“一个 group 只有一个统一 prefix_len”的简化设计。
- `survey/flash-preference`：参考 shared prefix 在 preference training / DPO 类任务中的收益形态、接口风格和测试场景，但不直接继承其实现边界。
- `survey/dpo-prefix-sharing`：参考 prompt / response 边界明确时的 prefix sharing 形态。Phase 1 暂不实现 `PromptPrefixDetector`，但保留未来扩展空间。
- `dependency/verl_v070`：当前集成目标框架。Phase 1 通过 patch 接入 actor logprob / update 主路径。
- `dependency/megatron_v0150`：当前 attention / packed sequence / RoPE 接入目标。Phase 1 禁用不支持的 fused 路径。

与 PrefixTrain_dev 的关键异同：

- 相同点：采用在线 Trie，从历史样本中寻找当前样本的可复用 prefix；允许一个 provider 服务多个不同 prefix length；关系粒度是 sample-level relation。
- 不同点：本项目将 detector、planner、metadata、backend、integration 明确分层；Phase 1 还必须处理 verl actor logprob 对齐、Prefix-Last Restore、patch 生命周期和测试分层。

---

## 2. 核心结论

### 2.1 最终精度方案

阶段一正式方案为：

```text
One-Forward + KV Injection + Prefix-Last Restore
```

对存在 prefix reuse 的样本：

- provider 样本保留完整 `[prefix | suffix]` 计算。
- reuser 样本只保留 suffix query。
- 每层 attention 注入 provider 的 prefix K/V，使 reuser suffix 能 attend 到完整历史。
- logprob 对齐阶段从 provider 恢复 prefix-last 位置的输出/logits/logprob，用于计算 reuser 第一个 suffix token 的 logprob。

### 2.2 为什么必须做 Prefix-Last Restore

causal LM 的训练和打分语义是 next-token prediction：

```text
output(P_last) -> predict S0
output(S0)     -> predict S1
```

如果 reuser 只计算 suffix query：

```text
Q  = [S0, S1, S2]
KV = [P0, P1, P2, S0, S1, S2]
```

attention 输出只有：

```text
output(S0), output(S1), output(S2)
```

这可以计算 `S1`、`S2` 等后续 suffix token 的 logprob，但缺少：

```text
output(P2) -> predict S0
```

Prefix-Last Restore 的作用就是把 provider 已经计算过的 `output(P_last)` 显式恢复给 reuser，用于第一个 suffix token 的 logprob。

### 2.3 等价性假设

Prefix-Last Restore 的理论前提是：复用关系中 provider 与 reuser 的被复用 prefix token 完全一致，并且 prefix 的 position、mask、RoPE、模型参数和随机性条件一致。因此 provider 样本计算得到的 prefix-last hidden/logits/logprob，与 reuser 独立完整计算时对应位置的结果等价。

精度对齐必须在以下条件下验证：

- eval mode；或 train mode 下关闭 dropout 并固定 RNG。
- 相同模型参数、相同 dtype、相同 attention backend 配置。
- prefix token、position、mask 完全一致。
- cache 不 `detach`，梯度能从多个 reuser 的 restore logprob 回流并累积到共享 prefix 计算图。
- 只比较 prefix sharing 语义声明覆盖的 logprob / loss 区间。

---

## 3. 总体架构

项目拆为三层：

```text
prefix_sharing/
├── core/                         # 通用语义层
│   ├── config.py                 # PrefixSharingConfig + 约束校验
│   ├── metadata.py               # PrefixSharingBatchMeta / PrefixLastRestoreSpec
│   ├── prefix_detector.py        # PrefixDetector / TriePrefixDetector / PrefixReuseSpec
│   ├── planner.py                # 裁剪、offset、restore 计划
│   ├── batch_trim.py             # input / label / mask 裁剪与 packed 视图
│   ├── cache.py                  # PrefixKVCache 生命周期管理
│   └── logprob.py                # Prefix-Last Restore logprob 工具
├── integrations/                 # 框架集成层
│   ├── patch_manager.py          # patch 安装、卸载、幂等、回滚
│   ├── context.py                # forward / micro-batch 上下文
│   ├── verl_mcore.py             # verl Megatron actor 接入
│   ├── megatron_attention.py     # Megatron SelfAttention patch
│   └── megatron_rope.py          # 非 fused RoPE offset 适配
└── backends/                     # 后端适配层
    ├── base.py                   # PrefixAttentionBackend 接口
    ├── torch_ref.py              # 正确性参考实现
    ├── cuda_ref.py               # CUDA 可运行参考后端
    └── cann_ref.py               # CANN NPU 可运行参考后端
```

核心原则：

- `core/` 不依赖 verl、Megatron、CUDA TE、flash-attn 或 CANN。
- `integrations/` 只做接入和数据桥接，不沉淀核心算法。
- `backends/` 不重新解释 prefix sharing 语义，只消费 `PrefixSharingBatchMeta`。
- patch 是阶段性接入方式，正式逻辑必须尽量放在可迁移模块中。

---

## 4. 核心数据结构

### 4.1 PrefixSharingConfig

配置对象描述是否启用 prefix sharing、使用哪个 detector/backend，以及 Phase 1 的硬约束。

```python
@dataclass
class PrefixSharingConfig:
    enabled: bool = False
    detector: str = "trie"
    backend: str = "torch_ref"
    min_prefix_len: int = 1
    min_group_size: int = 2
    boundary_strategy: str = "prefix_last_restore"

    require_pp_size: int = 1
    require_cp_size: int = 1
    require_rope_fusion_disabled: bool = True
    require_fused_qkv_rope_disabled: bool = True

    validate_precision: bool = False
```

`validate()` 的职责：

- 校验配置字段合法性，例如 `min_prefix_len >= 1`、`min_group_size >= 2`。
- 校验 `boundary_strategy` 当前只能是 `prefix_last_restore`。
- 校验模型/并行/融合路径满足 Phase 1 边界。
- 不满足约束时直接报错，不能静默 fallback。

### 4.2 PrefixReuseSpec

`PrefixReuseSpec` 是 Phase 1 的核心语义单位：

```python
@dataclass(frozen=True)
class PrefixReuseSpec:
    reuse_idx_in_batch: int
    sample_idx_in_batch: int
    prefix_len: int
```

含义：

- `reuse_idx_in_batch`：当前裁剪 prefix、复用别人 KV 的样本。
- `sample_idx_in_batch`：提供 prefix KV 的历史样本。
- `prefix_len`：reuser 从 provider 复用的 token 数量。

关键规则：

- 每个 reuser 在 Phase 1 中最多一条 reuse relation。
- 一个 provider 可以对应多条 reuse relation。
- 同一个 provider 可以为不同 reuser 提供不同 `prefix_len`。
- provider 本身也可能是之前某条 relation 的 reuser，只要 backend 已经为它构造并缓存完整 logical KV。

### 4.3 PrefixDetectionResult

Detector 输出结构：

```python
@dataclass(frozen=True)
class PrefixDetectionResult:
    batch_size: int
    reuse_specs: tuple[PrefixReuseSpec, ...]
    groups: tuple[PrefixGroup, ...]
    group_ids: tuple[int, ...]
    provider_index: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    is_provider: tuple[bool, ...]
```

语义来源：

- `reuse_specs` 是 source of truth。
- `provider_index[i]` 和 `prefix_lens[i]` 是为了 planner 快速访问的按 batch 展开视图。
- `groups` / `group_ids` 只用于调试、统计或后续执行优化。由于一个 provider 可服务多个 prefix length，单个 `group_id` 不能表达 provider 的全部复用关系。

### 4.4 PrefixSharingBatchMeta

`PrefixSharingBatchMeta` 是一次 micro-batch 的完整执行计划：

```python
@dataclass(frozen=True)
class PrefixSharingBatchMeta:
    forward_id: int
    micro_batch_id: int
    batch_size: int
    original_lengths: list[int]

    reuse_specs: list[PrefixReuseSpec]
    group_ids: list[int]
    is_provider: list[bool]
    provider_index: list[int]
    prefix_lens: list[int]
    suffix_lens: list[int]

    kept_lengths_q: list[int]
    expanded_lengths_kv: list[int]
    cu_seqlens_q: list[int]
    cu_seqlens_kv: list[int]
    max_seqlen_q: int
    max_seqlen_kv: int

    q_position_offsets: list[int]
    kv_position_offsets: list[int]

    input_keep_ranges: list[tuple[int, int]]
    label_keep_ranges: list[tuple[int, int]]
    loss_mask_keep_ranges: list[tuple[int, int]]

    prefix_last_restore: list[PrefixLastRestoreSpec]
```

字段不变量：

- 所有 per-batch list 的长度必须等于 `batch_size`。
- `cu_seqlens_q` 和 `cu_seqlens_kv` 长度必须等于 `batch_size + 1`。
- 对 reuser：`kept_lengths_q[i] == original_lengths[i] - prefix_lens[i]`。
- 对非 reuser：`kept_lengths_q[i] == original_lengths[i]`。
- `expanded_lengths_kv[i] == original_lengths[i]`，因为 attention 逻辑上仍看到完整序列。
- reuser 的 `q_position_offsets[i] == prefix_lens[i]`。
- `kv_position_offsets[i] == 0`。

### 4.5 PrefixLastRestoreSpec

```python
@dataclass(frozen=True)
class PrefixLastRestoreSpec:
    reuse_idx_in_batch: int
    sample_idx_in_batch: int
    provider_prefix_last_pos: int
    reuse_first_suffix_label_pos: int
    output_slot: int
    group_id: int
```

含义：

- `provider_prefix_last_pos == prefix_len - 1`。
- `reuse_first_suffix_label_pos == prefix_len`。
- `output_slot` 是恢复结果在 reuser suffix 输出中的插入位置；Phase 1 为 `0`。
- 该结构只描述语义，不绑定 logits 是 dense tensor、packed tensor，还是 backend 内部结构。

### 4.6 PrefixKVSlotId / CachedPrefixKV

```python
@dataclass(frozen=True)
class PrefixKVSlotId:
    forward_id: int
    micro_batch_id: int
    layer_id: int
    sample_idx_in_batch: int
    tp_rank: int = 0

@dataclass(frozen=True)
class CachedPrefixKV:
    key_tensor: Any
    value_tensor: Any
    prefix_len: int
```

cache key 不包含 `group_id`。原因是 provider 可能面向多个 reuser 提供不同 prefix length，cache 中应保存 provider 的完整 logical K/V，再由 reuser 按自己的 `prefix_len` 切片。

---

## 5. 模块详细设计

本章是开发交接的核心。任何工程师接手时，应优先读本章来判断代码应该写在哪里、接口应该如何保持稳定、哪些不变量不能破坏。

### 5.1 `core/config.py`

**定位**：统一配置入口和 Phase 1 约束守门模块。

核心类：

- `PrefixSharingConfig`
- `PrefixSharingConfigError`

主要接口：

```python
config = PrefixSharingConfig(enabled=True, min_prefix_len=8)
config.validate(model_config=model_config)
```

职责：

- 保存 detector/backend/boundary strategy 等开关。
- 校验基础参数类型和取值。
- 校验 Phase 1 不支持的并行和 fused path。
- 给 integration 层提供统一的启动失败原因。

实现要求：

- 配置错误必须 fail fast。
- 不允许因为配置不支持而静默关闭 prefix sharing。
- 后续新增 `boundary_token`、`strict_suffix` 等策略时，只扩展 `boundary_strategy` 枚举，不把字段名改回 `restore_mode`。

### 5.2 `core/prefix_detector.py`

**定位**：从 token 序列中发现可复用 prefix，输出 sample-level reuse relation。

核心类/函数：

- `PrefixDetector`
- `TriePrefixDetector`
- `PrefixReuseSpec`
- `PrefixGroup`
- `PrefixDetectionResult`
- `common_prefix_len()`

主要接口：

```python
detector = TriePrefixDetector(min_prefix_len=2, min_group_size=2)
result = detector.detect(input_ids)
```

输入：

- `input_ids: Sequence[Sequence[int]]`
- batch 内每条样本是一条 token id 序列。

输出：

- `PrefixDetectionResult`
- 其中 `reuse_specs` 是语义主结果。

算法：

1. 初始化空 Trie。
2. 按 batch 顺序处理样本。
3. 当前样本只与已经插入 Trie 的历史样本匹配。
4. 沿 token 路径寻找最长命中 prefix。
5. 如果命中长度满足 `min_prefix_len`，且该 Trie 节点覆盖的历史样本数加当前样本达到 `min_group_size`，生成一条 `PrefixReuseSpec`。
6. 将当前样本插入 Trie，使它可作为后续样本的 provider。

关键设计：

- provider 选择来自匹配节点记录的历史样本。
- 不采用“最长前缀优先、非重叠 group”的贪心分组。
- 同一个 provider 可以服务多个 reuser，且 prefix length 可不同。
- reuser 也可以成为后续 provider；backend 必须为它缓存完整 logical K/V。

不负责：

- 不裁剪 input / label / mask。
- 不生成 `cu_seqlens`。
- 不处理 RoPE。
- 不操作 KV cache。
- 不决定 logprob restore 细节。

测试重点：

- 多个 reuser 复用同一 provider 的不同 prefix length。
- reuser 继续作为后续 provider。
- `min_prefix_len` 和 `min_group_size` 阈值。
- 空 batch、无共享、完全共享、部分共享。

### 5.3 `core/planner.py`

**定位**：把 detector 的复用关系转成可执行的 micro-batch 计划。

核心类/函数：

- `PrefixSharingPlanner`
- `_cumsum()`

主要接口：

```python
planner = PrefixSharingPlanner(config)
meta = planner.plan(input_ids, forward_id=1, micro_batch_id=1)
```

输入：

- 原始 `input_ids`
- `PrefixSharingConfig`
- 可选 `forward_id` / `micro_batch_id`
- 可选外部传入的 `PrefixDetectionResult`

输出：

- `PrefixSharingBatchMeta`

职责：

- 调用 detector，或消费已经生成的 detection result。
- 展开 `provider_index`、`prefix_lens`、`suffix_lens`。
- 计算 provider 和 reuser 的 input / label / loss mask 保留范围。
- 计算 packed sequence 所需的 `kept_lengths_q`、`expanded_lengths_kv`、`cu_seqlens_q`、`cu_seqlens_kv`。
- 计算 RoPE 所需的 `q_position_offsets`、`kv_position_offsets`。
- 为每个有 suffix 的 reuser 生成 `PrefixLastRestoreSpec`。

核心规则：

- 非 reuser：保留完整 `[0, original_len)`。
- reuser：保留 `[prefix_len, original_len)`。
- reuser 的 query 逻辑位置从 `prefix_len` 开始。
- reuser 的 expanded KV 长度仍是 `original_len`。
- 如果 `prefix_len > original_len`，必须报错。

不负责：

- 不执行 tensor 裁剪。
- 不直接调用 backend。
- 不访问模型或框架对象。
- 不做 provider K/V 存取。

测试重点：

- `PrefixSharingBatchMeta` 所有长度和 range 一致。
- Prefix-Last Restore 指向正确 provider 和 `prefix_len - 1`。
- 无 sharing 时 metadata 等价于普通 packed batch。
- per-sample relation 下不同 reuser 的 restore spec 各自独立。

### 5.4 `core/batch_trim.py`

**定位**：把 planner 产生的 keep range 应用到 input、label、mask 上，并生成 packed 视图。

核心类/函数：

- `TrimmedBatch`
- `trim_inputs()`
- `trim_labels()`
- `trim_loss_masks()`

主要接口：

```python
trimmed = trim_inputs(input_ids, meta)
labels = trim_labels(labels, meta)
loss_masks = trim_loss_masks(loss_masks, meta)
```

职责：

- 根据 `input_keep_ranges` 裁剪 input。
- 根据 `label_keep_ranges` 裁剪 labels。
- 根据 `loss_mask_keep_ranges` 裁剪 loss mask / response mask。
- 保持 batch 顺序不变。
- 输出 `rows`、`flattened`、`cu_seqlens`，供 integration 层映射到框架 tensor。

核心规则：

- provider 输出完整保留。
- reuser 输出只来自 suffix query。
- reuser 第一个 suffix token 的 logprob 不由 suffix output 直接产生，具体补齐由 `core/logprob.py` 负责。
- batch trim 层只处理裁剪和 packed 视图，不处理 logits 数值计算。

不负责：

- 不检测 prefix。
- 不生成 metadata。
- 不操作 KV cache。
- 不执行 attention。

测试重点：

- input / label / loss mask 裁剪区间一致。
- packed 后长度等于 `sum(meta.kept_lengths_q)`。
- 空 suffix、无 sharing、全 sharing 场景。

### 5.5 `core/cache.py`

**定位**：管理当前 forward/backward 生命周期内的 logical K/V。

核心类：

- `PrefixKVSlotId`
- `CachedPrefixKV`
- `PrefixKVCache`

主要接口：

```python
cache.store(slot_id, key_tensor=key_tensor, value_tensor=value_tensor, prefix_len=n, overwrite=True)
entry = cache.load(slot_id)
cache.clear()
cache.close()
```

职责：

- 按 `forward_id`、`micro_batch_id`、`layer_id`、`sample_idx_in_batch`、`tp_rank` 隔离 K/V。
- 保存 provider 或 reuser 的完整 logical K/V。
- 为后续 reuser 提供可切片的 K/V。
- 管理生命周期，防止 close 后继续读写。

核心规则：

- cache tensor 不允许 `detach`。
- reuser 构造出的 expanded K/V 必须写回 cache，使其可作为后续 provider。
- `group_id` 不进入 cache key。
- 默认不允许覆盖；backend 明确传 `overwrite=True` 时可覆盖同一 key。

不负责：

- 不判断一个样本是否应该 reuse。
- 不计算 prefix length。
- 不做 attention mask。
- 不跨 micro-batch 持久化。

测试重点：

- store/load/contains/clear/close。
- duplicate key 行为。
- close 后读写报错。
- key 中不含 `group_id`。

### 5.6 `core/logprob.py`

**定位**：处理 Prefix-Last Restore 的 logits/logprob 组装。

核心函数：

- `restore_prefix_last_logprobs()`
- `build_provider_prefix_last_values()`
- `compute_token_logprobs_from_logits()`
- `gather_provider_prefix_last_logits()`
- `restore_prefix_last_logprobs_tensor()`

主要接口：

```python
provider_logits = gather_provider_prefix_last_logits(logits_by_batch, meta)
token_logprobs = compute_token_logprobs_from_logits(logits, labels)
restored = restore_prefix_last_logprobs_tensor(suffix_logprobs, first_suffix_logprobs, meta)
```

职责：

- 提供 Python list 版本的 Prefix-Last Restore，供 CPU 单元测试和框架无关 adapter 使用。
- 从 provider 的 prefix-last 输出位置取 logits。
- 根据 reuser 第一个 suffix label 计算对应 logprob。
- 把该 logprob 放回 reuser 的 `output_slot=0`。
- 保持 autograd 图连通，使 provider prefix 相关计算接收来自 reuser 的梯度贡献。

核心规则：

- 不 detach logits 或 logprob。
- 恢复只覆盖 `prefix_last_restore` 描述的位置。
- 后续 suffix token 的 logprob 使用 reuser suffix query 输出。

不负责：

- 不决定哪些样本需要 restore。
- 不裁剪输入。
- 不执行模型 forward。

测试重点：

- restore 后第一个 suffix token logprob 正确。
- autograd 能从 restored loss 回到 provider logits。
- 无 restore specs 时保持输入等价。

### 5.7 `backends/base.py`

**定位**：定义 backend 能力和统一执行接口。

核心类：

- `BackendCapabilities`
- `PrefixAttentionBackend`

建议接口：

```python
class PrefixAttentionBackend:
    capabilities: BackendCapabilities

    def validate(self, config, model_config=None) -> None: ...
    def apply_rope(self, query, key, meta, **kwargs): ...
    def build_kv(self, key, value, cache, meta, *, layer_id, tp_rank=0): ...
    def attention(self, query, key, value, meta, **kwargs): ...
```

职责：

- 声明 backend 支持的硬件、dtype、q/k/v 长度差异、restore 能力。
- 隔离 PyTorch / CUDA / CANN / kernel 细节。
- 让 integration 层不直接依赖具体 backend 实现。

设计要求：

- backend 必须消费 `PrefixSharingBatchMeta`，不能重新做 prefix detection。
- backend validate 必须显式拒绝不支持的配置。
- 新 backend 必须先通过 reference backend 的语义测试，再接硬件特化。

### 5.8 `backends/torch_ref.py`

**定位**：CPU/PyTorch 正确性参考实现，不追求性能。

核心类：

- `TorchReferenceBackend`

主要接口：

```python
expanded_key, expanded_value = backend.build_kv(key, value, cache, meta, layer_id=0)
output = backend.attention(query, expanded_key, expanded_value, meta)
```

职责：

- 支持 q_len 与 kv_len 不同。
- 对非 reuser：缓存完整 key/value，并直接进入输出。
- 对 reuser：从 provider cache 读取完整 logical K/V，按自己的 `prefix_len` 切片，与本地 suffix K/V 拼接。
- 将 reuser 拼接出的完整 logical K/V 再写回 cache。
- 用纯 PyTorch causal mask 验证 attention 语义。

核心规则：

- `_split_packed()` 必须严格校验 packed tensor 第一维与 lengths 总和一致。
- reuser 的 expanded K/V 长度必须等于原始序列长度。
- attention mask 由 `q_position_offsets` 和 expanded KV 长度共同决定。

测试重点：

- q_len != kv_len。
- transitive reuse：reuser 作为后续 provider。
- autograd 不断。
- CUDA 可用时 optional smoke test。

### 5.9 `backends/cuda_ref.py` 与 `backends/cann_ref.py`

**定位**：硬件可运行参考后端，Phase 1 只要求能跑通，不要求最终性能。

职责：

- 复用 `TorchReferenceBackend` 的语义路径。
- 显式声明 CUDA / CANN capability。
- 在硬件存在时通过 optional smoke test。

限制：

- 不绑定 TE fused attention。
- 不把 CUDA/CANN 特有参数泄漏进 `core/`。
- 不在 Phase 1 承诺高性能。

### 5.10 `integrations/context.py`

**定位**：在一次 forward / micro-batch 执行期间保存当前 prefix sharing runtime state。

核心对象：

- `PrefixSharingRuntimeContext`
- `prefix_sharing_context()`
- `get_current_prefix_sharing_context()`

职责：

- 绑定当前 `PrefixSharingBatchMeta`。
- 绑定当前 `PrefixKVCache`。
- 让 attention patch 在不改变大量函数签名的情况下读取当前 meta/cache。
- 保证上下文退出后恢复旧状态。

设计要求：

- 必须支持嵌套上下文的正确恢复。
- 无上下文时 patch 应走普通路径或显式报错，具体取决于入口配置。
- 不把全局状态暴露给 `core/`。

### 5.11 `integrations/patch_manager.py`

**定位**：统一管理 monkey patch 生命周期。

核心类：

- `PatchHandle`
- `PatchManager`

主要接口：

```python
handle = manager.install_patch(target, attr_name, replacement)
handle.disable()
manager.disable_all()
```

职责：

- 安装 patch。
- 幂等处理重复安装。
- 保存原始函数。
- 卸载 patch 并恢复原函数。
- 安装失败时回滚。

设计要求：

- patch 必须可逆。
- patch 前应检查目标函数是否存在、签名是否符合预期。
- patch 层保持薄，只调用 `core/`、`backend`、`context`。
- 未来上游 PR 时，优先把 patch 点替换成正式 hook，避免重写核心逻辑。

### 5.12 `integrations/megatron_attention.py` / `integrations/megatron_runtime.py`

**定位**：Megatron SelfAttention 的执行接入点。

职责：

- 在 QKV projection 后、Megatron 标准 RoPE 前接入 prefix sharing backend。
- 从当前 runtime context 读取 `meta` 和 `cache`。
- 按 verl micro-batch 传入的原始 `position_ids` 为压缩后的 packed THD query/key 重新选择 RoPE 频率，避免 reuser suffix 被错误当作 position 0 开始。
- 调用 backend 的 `build_kv()`、`attention()` 和 Megatron 原始 output projection。
- 在未启用 prefix sharing 或无共享关系时保持原路径行为。

关键流程：

```text
hidden_states
  -> QKV projection
  -> positioned RoPE for kept tokens
  -> backend.build_kv(key, value, cache, meta)
  -> backend.attention(query, expanded_key, expanded_value, meta)
  -> output projection
```

设计要求：

- 必须显式禁用 Phase 1 不支持的 fused path。
- 必须处理 packed THD / no-padding 参数的语义对齐。
- 必须保证 layer_id、tp_rank 正确进入 cache key。
- 当前实现状态：`dependency/megatron_v0150/megatron/core/transformer/attention.py` 在 packed THD squeeze 后、标准 RoPE 前新增一个极薄 hook，调用 `maybe_run_prefix_sharing_attention()`；真实 RoPE 位置修正、KV expansion、GQA reference attention 和 output projection 都在 `prefix_sharing.integrations.megatron_runtime` / backend 中完成。无 runtime context 时返回原 Megatron 路径。

### 5.13 `integrations/megatron_rope.py`

**定位**：处理非 fused RoPE 下 suffix query 的 position offset。

职责：

- 根据 `meta.q_position_offsets` 调整 query 的 position。
- 保持 key/value 的 logical position 从 0 开始。
- 在 fused RoPE 启用时拒绝 Phase 1 路径。

设计要求：

- RoPE offset 必须与 planner 生成的保留区间一致。
- 不能通过临时平移 token id 或 mask 规避 position 问题。

### 5.14 `integrations/verl_mcore.py`

**定位**：verl Megatron actor 主路径接入。

核心类/函数：

- `VerlMCoreBatchAdapter`
- `VerlMCorePreparedBatch`
- `prepare_megatron_actor_micro_batch()`
- `megatron_actor_prefix_sharing_context()`
- `restore_megatron_actor_log_probs()`
- `VerlMCoreIntegration`
- `enable_prefix_sharing()`
- `prefix_sharing_enabled()`

职责：

- 在 micro-batch preprocess 阶段生成 `PrefixSharingBatchMeta`。
- 调用 `batch_trim` 裁剪 input、labels、loss mask、response mask。
- 进入 `prefix_sharing_context()`，让 Megatron attention patch 可读取 meta/cache。
- postprocess 阶段组装 logprob，执行 Prefix-Last Restore。
- 保持 verl actor 原始 batch 顺序和返回格式。
- 在 `dependency/verl_v070/verl/workers/actor/megatron_actor.py` 中仅保留最小使能入口：micro-batch 准备、forward context 和 logprob restore 调用。

设计要求：

- patch 启停不改变非 prefix sharing 路径行为。
- 无可复用 relation 时输出应等价于 baseline。
- 多 micro-batch 之间 cache 不得污染。
- 报错信息应能定位是 preprocess、attention、restore 还是 patch 约束失败。
- 当前实现状态：框架无关 adapter 与真实 verl actor helper 均已落地。启用 `config.prefix_sharing.enabled=True` 且 `config.megatron.use_remove_padding=True` 时，micro-batch 会按 attention mask 生成 plan，reuser prefix 位会从 attention mask 中移除，packed THD 位置表和 Prefix-Last Restore dense slot 会进入 runtime context。fused actor kernel、multi-modal batch、非 THD 路径仍按 Phase 1 约束显式拒绝。

---

## 6. Phase 1 数据流

### 6.1 Preprocess

输入来自 verl Megatron actor micro-batch。

流程：

1. `TriePrefixDetector` 识别 per-sample reuse relation。
2. `PrefixSharingPlanner` 生成 `PrefixSharingBatchMeta`。
3. `batch_trim` 同步裁剪 input、label、loss mask、response mask。
4. provider 样本保留完整 `[prefix | suffix]`。
5. reuser 样本裁剪 prefix，仅保留 suffix。
6. 构造裁剪后的 packed input 和 `cu_seqlens_q`。
7. 记录 logical expanded KV 长度和 `cu_seqlens_kv`。
8. 把 meta/cache 写入当前 `prefix_sharing_context`。

### 6.2 Attention

每层 attention 的语义流程：

```text
trimmed hidden states
  -> QKV projection
  -> RoPE with q_position_offsets
  -> store/load/build expanded K/V
  -> causal attention with q_len != kv_len
  -> output projection
```

注意：

- reuser query 长度和 expanded KV 长度可以不同。
- suffix query 的 RoPE 逻辑位置从 `prefix_len` 开始。
- expanded KV 的逻辑位置从 0 连续到原始序列长度。
- attention mask 必须保证因果性等价于原始完整序列。

### 6.3 Postprocess / Logprob

verl actor 当前 logprob 语义中，response 第一个 token 的 label 位于 prompt 最后一个输出位置。因此 postprocess 必须：

- 保留 provider 的完整 logits/logprob。
- 对 reuser 组装 suffix logits/logprob。
- 对 reuser 第一个 suffix token，从 provider prefix-last 输出恢复对应 logits/logprob。
- 对 reuser 后续 suffix token，使用 suffix query 输出计算 logits/logprob。
- 按原始 batch 顺序返回给 verl actor loss 逻辑。

---

## 7. Patch 集成策略

阶段一必须对 verl 和 Megatron 保持非侵入式：

- 不直接修改 site-packages / submodule 源码。
- 不 fork verl / Megatron 主代码。
- 通过 monkey patch、wrapper、context manager 接入。
- patch 层保持薄，核心逻辑放在 `core/` 和 `backends/`。

推荐入口：

```python
handle = enable_prefix_sharing(config)
try:
    train()
finally:
    handle.disable()
```

或：

```python
with prefix_sharing_enabled(config):
    train()
```

`patch_manager.py` 必须具备：

- 幂等安装。
- 完整卸载。
- 版本和函数签名检查。
- 安装失败回滚。
- 约束校验失败时拒绝启动。

未来向 verl / Megatron PR 时，优先把 patch 点迁移成正式 hook 或 config branch，`core/`、`metadata`、`planner`、`mapping`、`backend` 接口尽量保持不变。

---

## 8. 测试与验收规格

### 8.1 单元测试

目录：`tests/unit_test/`

覆盖：

- config 参数校验和 Phase 1 约束。
- detector 的 relation 输出、阈值、多长度 provider、transitive reuse。
- planner 的 lengths、ranges、offsets、restore specs。
- batch trim 的 input / label / mask 裁剪。
- cache 生命周期和 key 语义。
- logprob restore 的张量语义。

要求：

- 本地 CPU 必须通过。
- 不依赖 torch / verl / torch_npu 的测试应始终运行。

### 8.2 集成测试

目录：`tests/integrated_test/`

覆盖：

- patch manager 安装、卸载、幂等和回滚。
- runtime context 与 backend 的协作。
- reference backend 在 q_len != kv_len 下的 attention 语义。
- reference backend 在 packed THD + grouped-query attention 下的 attention 语义。
- verl Megatron actor runtime helper 的 mask 裁剪、position 传递、Prefix-Last Restore autograd 路径。
- Prefix-Last Restore 的 autograd 路径。

要求：

- 纯 CPU / 无外部框架依赖部分必须通过。
- torch、CUDA、CANN、verl 缺失时相关测试使用 `pytest.importorskip()` 或 skip marker。

### 8.3 系统测试

目录：`tests/system_test/`

覆盖：

- Phase 1 core pipeline：detect -> plan -> trim -> build metadata -> restore。
- 无 sharing 和有 sharing 的端到端语义。
- 多 micro-batch cache 隔离。

后续扩展：

- 小型 causal LM 前向对齐。
- Megatron 单层/小层数模型前向/反向对齐。
- verl actor logprob/update smoke test。

当前本地无 torch / verl / GPU / NPU，因此真实框架 smoke test 和加速器闭环以 optional 测试或待运行项保留；除这些环境依赖项外，Phase 1 代码层面的主链路入口、runtime context、Megatron attention hook 和 Prefix-Last Restore 已补齐。

### 8.4 精度验收

对比对象：

- baseline 完整序列 forward。
- prefix sharing patched forward。

比较范围：

- provider 样本 logits/logprob。
- reuser 第一个 suffix token logprob。
- reuser 后续 suffix token logprob。
- loss。
- 关键参数梯度。

验收条件：

- eval mode 或 train mode 关闭 dropout。
- 固定 RNG。
- fp32 使用严格阈值。
- bf16/fp16 使用符合 backend 数值误差的阈值。

---

## 9. Roadmap

### 9.1 阶段一：verl + Megatron RL MVP

目标：打通当前最关键业务链路的最小正确闭环。

成功标准：

- 正确性测试通过。
- small-scale RL actor 链路可运行。
- 在合理精度假设下 logprob/loss/grad 对齐。
- patch 可启停，不污染非 prefix sharing 路径。

当前实现已补齐真实 Megatron SelfAttention hook 和真实 verl actor preprocess/postprocess 入口。剩余闭环缺口集中在真实 GPU / NPU / verl 环境下运行 small-scale actor logprob/update smoke test，并据实际 kernel 行为修正可能暴露的设备侧兼容问题。

### 9.2 阶段二：业务落地能力

重点：

- TP 完整验证。
- CP / PP 支持。
- EP 兼容性分析和必要适配。
- DP / micro-batch / gradient accumulation 稳定性。
- CUDA TE / flash-attn 后端优化。
- CANN NPU 专用后端和算子适配。
- backend capability 机制。
- 性能策略：`min_prefix_len`、`min_group_size`、收益估计、自动 fallback。

探索特性：Prefix Activation Reuse。

Phase 1 采用 KV injection，因为 suffix token 在 attention 中会反复消费 prefix K/V，而 reuser prefix Q/output 本身不需要重新产出。这是最小正确闭环。Phase 2 应探索更充分的 prefix activation reuse：当 provider 与 reuser 共享相同 prefix 时，除 prefix K/V 外，进一步复用 provider prefix 的 layer hidden states、attention/MLP 中间激活、prefix-last logits 或其他可安全共享的 prefix 输出。

该方向的目标：

- 减少 reuser prefix 的整条 forward 重复计算，而不仅是 attention 中的 K/V 重复。
- 在训练中减少重复保存的 prefix activations，降低 backward activation 显存压力。
- 让多个 reuser 的 loss / logprob 梯度正确累积回 provider prefix 计算图。
- 支持 per-sample reuse relation，即同一个 provider 可向不同 reuser 暴露不同长度的 prefix activation slice。

设计约束：

- 不允许通过 `detach` 切断 provider prefix activation 的梯度路径。
- 必须兼容 activation checkpointing / recompute；共享 activation 在重算时不能产生语义漂移。
- 必须明确 TP / PP / SP / CP 下 activation 分片、跨 stage 传递和生命周期边界。
- 必须与 Prefix-Last Restore、full-prefix restore、logprob/loss mask 对齐。
- 先以实验后端或显式 feature flag 形式进入，不作为 Phase 2 默认路径。

参考关系：

- `PrefixTrain_dev` 更接近 activation reuse 方向，证明了共享 prefix 激活是可探索路线，但其直接魔改 Megatron、跨 stage activation 传递和梯度处理方式不能直接照搬。
- `flash-preference`、`dpo-prefix-sharing` 说明 prompt/prefix 相同的 preference/DPO 场景存在进一步复用 prefix 计算的需求；Phase 2 可借鉴其 prompt/response 边界和 pairwise 场景建模。

### 9.3 阶段三：训练范式扩展

范围：

- `verl + FSDP` RL 路径。
- standalone Megatron SFT。
- standalone Megatron pretrain。
- standalone FSDP SFT / pretrain。
- full-prefix restore。
- PromptPrefixDetector。
- 面向 next-token prediction 的通用 API。

### 9.4 阶段四：上游 PR 与产品化

方向：

- verl PR：prefix sharing extension hook 或可选 config path。
- Megatron PR：attention / packed sequence / RoPE / output restore 正式扩展点。
- 独立插件或 pip package。
- 稳定 API 文档。
- 兼容矩阵。
- benchmark。

---

## 10. 版本管理策略

当前项目版本管理以 `prefix-0501` 为颗粒度。`survey/`、`dependency/`、`refactor/prefix-sharing/` 暂时一起提交和推送。

`refactor/prefix-sharing/` 曾经作为独立 Git 仓库开发。该独立历史已保留在：

- `https://github.com/Jackie2049/prefix-sharing.git`
- 最后独立提交：`af21cd0 [feat] 支持按样本前缀复用关系`

后续开源 `prefix-sharing` 时，可以基于当前 `prefix-0501` 的 `refactor/prefix-sharing/` 路径拆分，也可以基于保留的独立仓库历史继续整理。

---

## 11. 当前实现审计结论

截至当前版本，Phase 1 代码主链路已经补齐；GPU / NPU / 真实 verl 框架 smoke test 是剩余闭环项。准确状态如下：

已完成：

- `core/config.py`：Phase 1 配置校验。
- `core/prefix_detector.py`：per-sample reuse relation detector。
- `core/planner.py`：metadata、裁剪范围、position offset、Prefix-Last Restore spec。
- `core/batch_trim.py`：input / label / loss mask 裁剪与 packed 视图。
- `core/logprob.py`：Python list 与 torch tensor 级 Prefix-Last Restore helper。
- `core/cache.py`：forward/micro-batch/layer/provider/tp 维度的 K/V cache。
- `backends/torch_ref.py`：reference attention backend，支持 q_len != kv_len 和 transitive reuse cache。
- `integrations/patch_manager.py` 与 `integrations/context.py`：patch 生命周期与 runtime context。
- `integrations/verl_mcore.py`：框架无关 batch adapter 和真实 verl actor helper，已实际调用 `batch_trim` 与 `logprob` 完成 preprocess/postprocess。

剩余待真实环境验证：

- Megatron 单层/小层数模型前向/反向对齐。
- verl actor logprob/update smoke test。
- CUDA / CANN / NPU backend 与设备侧 kernel 行为验证。
- 非 fused RoPE 在真实 Megatron packed 参数下的实际接线。
- verl actor 真实 preprocess/postprocess 函数 patch。
- 从 provider prefix-last logits 到 reuser first suffix label 的真实 verl tensor 路径接线。
- 真实 verl + Megatron actor logprob/update smoke test。
- GPU / NPU 环境下的 CUDA / CANN optional 测试。

因此，当前代码可以支撑 Phase 1 的 core semantic 开发与本地 CPU 自测，但不能直接交付真实业务链路。后续开发优先级应是：

1. 接入真实 verl actor micro-batch preprocess/postprocess，把 `VerlMCoreBatchAdapter` 返回的 trimmed payload 转换为 verl/Megatron 实际输入格式。
2. 完成 Megatron attention patch 的 QKV rewiring，并在上下文存在时调用 backend，而不是抛出 `NotImplementedError`。
3. 在安装了 torch、verl、Megatron 的环境中补真实 smoke test。
4. 再进入 CUDA / CANN reference backend 和性能后端验证。
