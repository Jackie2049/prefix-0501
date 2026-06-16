# Prefix-Sharing 术语表 (Glossary)

本文档记录 prefix-sharing 项目中使用的专业术语和概念，用于团队内部对齐和代码/文档的一致性。

---

## 核心概念

### Reuse Relation（复用关系）

**定义**：一条按样本粒度描述的前缀复用关系，表示某个 reuser 从哪个 provider 复用多长的 prefix。

**说明**：
- 这是当前 Phase 1 的核心语义单位
- 一个 provider 可以对应多条 reuse relation
- 不同 reuser 可以从同一个 provider 复用不同长度的 prefix
- Prefix Group 只作为调试、统计或后续执行优化视图，不是语义核心

**相关代码**：
- `PrefixReuseSpec.reuse_idx_in_batch`
- `PrefixReuseSpec.provider_idx_in_batch`
- `PrefixReuseSpec.prefix_len`
- `PrefixDetectionResult.reuse_specs`

---

### Provider

**定义**：在一条或多条复用关系中，负责计算可复用 prefix KV 表示的序列。

**说明**：
- Provider 的前缀 token 会正常经过 attention 计算，产生 KV cache
- 其他序列（reuser）会复用 provider 计算好的 prefix KV，避免重复计算
- 同一个 provider 可以向不同 reuser 提供不同长度的 prefix slice
- 当前 `TriePrefixDetector` 按 batch 顺序处理，历史样本可作为后续样本的 provider
- 若历史样本本身也是 reuser，reference backend 会先构造并缓存它的完整 logical KV，使其仍可作为 provider

**相关代码**：
- `PrefixReuseSpec.provider_idx_in_batch`
- `PrefixDetectionResult.provider_index`
- `PrefixDetectionResult.is_provider`

---

### Reuser

**定义**：在一条复用关系中，复用 provider 的 prefix KV cache 的序列。

**说明**：
- Reuser 的 prefix token 不会经过实际的 attention 计算
- Reuser 会将 provider 的 prefix KV 与自己的 query 进行 attention 计算
- 每个 reuser 在 Phase 1 中最多对应一条 reuse relation
- 从 provider 的角度看，reuser 是消费其 prefix KV 的序列

**相关代码**：
- 判断方式：`provider_index[index] != index and prefix_lens[index] > 0`
- `PrefixReuseSpec.reuse_idx_in_batch`

---

### Prefix Group（前缀共享组）

**定义**：一组具有相同 provider 和相同 prefix_len 的复用关系形成的调试/统计视图。

**说明**：
- Prefix Group 不再是 Phase 1 的核心语义
- 同一个 provider 可出现在多个 Prefix Group 中，分别对应不同 `prefix_len`
- 单个 `group_id` 不能完整表达 provider 的所有复用关系
- 执行计划以 `PrefixReuseSpec`、`provider_index`、`prefix_lens` 为准

**相关代码**：
- `PrefixGroup` 数据类
- `PrefixDetectionResult.groups`

---

### Prefix Detection（前缀检测）

**定义**：分析一批序列，识别其中可复用前缀并生成 per-sample reuse relation 的过程。

**说明**：
- 输入：一批 token 序列
- 输出：`PrefixDetectionResult`，包含 `reuse_specs` 和按 batch 展开的 provider/reuser 分配
- 可配置参数：
  - `min_prefix_len`: 最小前缀长度，小于此值的前缀不形成复用关系
  - `min_group_size`: 最小共享样本数；当前前缀覆盖的历史样本数加当前样本小于该值时，不形成复用关系

**相关代码**：
- `TriePrefixDetector.detect()`

---

### Activation Reuse（前缀复用）

**定义**：在 micro-batch 内，provider 序列计算并存储 prefix 的中间状态，reuser 序列复用这些状态以避免重复计算的通用机制。

**说明**：
- Prefix reuse 是核心优化思想，具体实现方式取决于 mixer 类型
- 所有复用机制都必须保留完整 autograd 计算图（不能 `detach`），确保梯度正确回传
- Reuser 的 prefix tokens 不参与实际计算，但仍需参与 logprob 计算（通过 Prefix-Last Restore）

**两种实现方式**：

1. **KV Cache Reuse（KV 缓存复用）**
   - **适用场景**：Softmax Attention（如 Llama、Qwen2 等标准 Transformer）
   - **机制**：provider 计算并缓存 prefix 的 key/value 张量；reuser 将 provider KV 与自身 suffix KV 拼接后计算 attention
   - **存储结构**：`PrefixAttentionStore`，存储 `StoredAttentionKV` 条目
   - **关键约束**：KV 张量必须保留梯度，拼接操作需维持计算图连续性

2. **State Reuse（激活状态复用）**
   - **适用场景**：当前仅面向 Qwen3.5/Qwen3.6 的 GatedDeltaNet
   - **机制**：provider 计算 prefix 后发布其内部状态（recurrent state、conv state 等）；reuser 在 prefix 边界读取状态并继续计算 suffix
   - **存储结构**：`PrefixDeltanetStore`，存储 `StoredDeltanetState`
   - **关键约束**：状态张量必须保留梯度；gate 等当前 token 计算的值不属于可复用 prefix state

**相关代码**：
- Attention 后端接口：`PrefixAttentionBackend`
- GatedDeltaNet 后端接口：`PrefixDeltanetBackend`
- KV 复用实现：`TorchReferenceBackend.build_kv()`
- State 复用实现：`TorchReferenceBackend.build_deltanet_states()`
- 存储基类：`PrefixActivationStore`
- KV 专用存储：`PrefixAttentionStore`
- State 专用存储：`PrefixDeltanetStore`
- 存储条目类型：`StoredAttentionKV`、`StoredDeltanetState`
- Slot ID 类型：`PrefixActivationSlotId`

---

## 数据结构术语

### Token Sequence

**定义**：一个整数序列，每个整数代表一个 token ID。

**类型别名**：`Sequence[int]`

---

### Batch

**定义**：一次处理的多个 token sequence 的集合。

**说明**：
- 在 prefix-sharing 中，batch 内的序列会被分析共享前缀
- `PrefixDetectionResult.batch_size` 记录批次大小

---

### PrefixSharingRuntimeState（前缀共享运行时状态）

**定义**：连接框架层（verl）与算子层（Megatron attention）的运行时状态载体，封装 prefix-sharing 执行所需的全部元数据。

**说明**：
- 由 `build_prefix_sharing_micro_batch()` 在 forward 前构建，携带 micro-batch 的前缀分析结果
- 包含三个核心字段：
  - `prefix_sharing_plan`: 前缀共享计划（哪些序列共享、保留范围、恢复点映射）
  - `backend`: 后端执行引擎（如 `TorchReferenceBackend`），负责 KV 缓存的存储与检索
  - `batch_runtime_layout`: 真实 packed batch 布局，包含 valid/padded lengths、packed position ids 和 valid token mask
- 通过 `prefix_sharing_runtime_context` context manager 注入执行上下文，使 attention 层无需修改函数签名即可获取状态
- 采用 `contextvars` 机制实现跨层隐式传递，避免在 verl actor 和 Megatron attention 之间显式传递参数

**生命周期**：
1. **构建阶段**（verl actor）：分析 micro-batch 内序列的共同前缀，裁剪 batch，生成 runtime state
2. **传递阶段**（context manager）：通过 `with prefix_sharing_runtime_context(state)` 将状态绑定到当前执行上下文
3. **消费阶段**（Megatron attention）：`maybe_run_prefix_sharing_attention()` 从 context 读取 plan，执行 KV 缓存注入和恢复

**设计目的**：
- 解决框架层与算子层之间缺乏直接参数传递通道的问题
- 确保 KV 缓存的读写位置与原始序列位置在 THD（packed）格式下精确对齐
- 保持计算图完整性，prefix KV store 只保存有效 token KV，不保存 packed padding

**相关代码**：
- `PrefixSharingRuntimeState` 数据类定义：`prefix_sharing/integrations/verl_mcore.py`
- 构建逻辑：`build_prefix_sharing_micro_batch()` in `prefix_sharing/integrations/verl_mcore.py:194-340`
- Runtime layout：`prefix_sharing/backends/batch_layout.py`，包含 `ThdBatchLayout` / `BshdBatchLayout`
- 上下文管理器：`prefix_sharing_runtime_context()` in `prefix_sharing/integrations/context.py`

---

## 实现相关术语

### Trie（前缀树）

**定义**：当前 `TriePrefixDetector` 使用的数据结构，用于高效发现共享前缀。

**说明**：
- Trie 只是实现方式之一，不是唯一选择
- 内部节点记录经过该节点的序列索引
- 节点的深度等于前缀长度

---

## 命名规范

### 代码命名

| 概念 | 类/变量名 | 字段名 | 布尔标记 |
|------|-----------|--------|----------|
| Provider | `provider` | `provider_index`, `provider_idx` | `is_provider` |
| Reuser | `reuser` | `reuse_idx_in_batch` | `is_reuser` (如有需要) |
| 复用关系 | `PrefixReuseSpec` | `reuse_specs` | - |
| 前缀组 | `PrefixGroup` | `groups`, `member_indices` | - |
| 前缀长度 | `prefix_len` | `prefix_lens` | - |
| THD batch 布局 | `ThdBatchLayout` | `valid_lengths`, `padded_lengths`, `position_ids`, `valid_token_mask` | - |
| 前缀共享运行时状态 | `PrefixSharingRuntimeState` | `prefix_sharing_plan`, `backend`, `batch_runtime_layout` | - |

### 文档命名

- 正文使用首字母大写的 **Provider** 和 **Reuser**
- 代码引用使用反引号包裹，如 `` `provider_index` ``
- 中文文档中可直接使用英文术语，或加注中文说明

---

## 术语对照表

| 英文 | 中文（参考） | 说明 |
|------|--------------|------|
| Provider | 提供者 / 宿主序列 | 计算 prefix KV 的序列 |
| Reuser | 复用者 / 复用序列 | 复用 prefix KV 的序列 |
| Reuse Relation | 复用关系 | reuser 从 provider 复用指定长度 prefix |
| Prefix Group | 前缀共享组 | 调试/统计视图，不是核心语义 |
| Prefix Detection | 前缀检测 | 识别 per-sample 复用关系的过程 |
| Prefix Reuse | 前缀复用 | 复用 provider prefix 中间状态的通用机制 |
| KV Cache Reuse | KV 缓存复用 | Softmax Attention 的前缀复用实现 |
| Activation State Reuse | 激活状态复用 | 当前用于 GatedDeltaNet 的前缀状态复用 |
| Token Sequence | Token 序列 | 整数序列 |
| Batch | 批次 | 一次处理的多个序列 |
| PrefixSharingRuntimeState | 前缀共享运行时状态 | 连接框架层与算子层的状态载体 |

---

*文档更新遵循时间倒序原则，最新修改在最上方。*
