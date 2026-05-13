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
- `PrefixReuseSpec.reuse_batch_index`
- `PrefixReuseSpec.provider_batch_index`
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
- `PrefixReuseSpec.provider_batch_index`
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
- `PrefixReuseSpec.reuse_batch_index`

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

### KV Cache Reuse（KV 缓存复用）

**定义**：在 attention 计算中，复用已计算好的 key/value 缓存，避免重复计算的技术。

**说明**：
- 在 prefix-sharing 场景下，provider 计算 prefix 的 KV，reuser 直接复用
- 复用时需要确保梯度正确传播（不能 detach）
- 技术实现上涉及 attention 层的修改，将 provider 的 KV 拼接到 reuser 的 KV 中

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
| Reuser | `reuser` | `reuse_batch_index` | `is_reuser` (如有需要) |
| 复用关系 | `PrefixReuseSpec` | `reuse_specs` | - |
| 前缀组 | `PrefixGroup` | `groups`, `member_indices` | - |
| 前缀长度 | `prefix_len` | `prefix_lens` | - |

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
| KV Cache Reuse | KV 缓存复用 | 复用已计算的 KV |
| Token Sequence | Token 序列 | 整数序列 |
| Batch | 批次 | 一次处理的多个序列 |

---

*文档更新遵循时间倒序原则，最新修改在最上方。*
