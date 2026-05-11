# 方案设计记录

> **规则**: 所有方案设计按时间倒序记录，最新在前。

---

## 2026-05-11 20:49 详细设计：三层架构 + Metadata + 后端适配方案

> 本设计是在 One-Forward + KV Injection 核心思路上做结构化重构。
> 旧方案的核心方向保留：同一次 forward 内缓存组内首条序列的 prefix KV，并注入给后续 suffix-only 序列。
> 新方案修正旧设计中过度依赖 `position_ids`、绑定 CUDA TransformerEngine、postprocess/loss 对齐不明确等问题。

### 一、设计目标

**项目目标**：在 rllm + verl + Megatron 的 RL 训练 pipeline 中，以插件化方式实现 prefix sharing，减少共享 prefix 的重复计算，同时保持 logits/logprob/loss/梯度语义正确。

**核心原则**：
- **语义层与后端解耦**：prefix sharing 的分组、裁剪、位置、输出映射必须独立于 CUDA TE / flash-attn / CANN NPU 等具体 kernel。
- **MVP 先打通最简单路径**：先限制到普通 text GPT、PP=1、CP=1、非 fused RoPE、非 fused QKV，再逐步扩展。
- **保留未来扩展性**：CP、PP、NPU、fused kernel、block/page KV 共享都通过后端适配层扩展，不污染上层语义。
- **显式 metadata 驱动**：所有 token 裁剪、KV 注入、RoPE offset、label/logprob 对齐、output restore 都由统一 metadata 描述，避免隐含状态散落在 `input_ids` / `cu_seqlens` / cache 中。
- **精度验证有明确假设**：只在 eval mode，或 train mode 关闭 dropout、固定 RNG、固定后端配置时做 logits/grad 容差验证。

### 二、总体三层架构

```
prefix_sharing/
├── core/                         # 通用语义层：不依赖 CUDA/TE/NPU
│   ├── config.py                 # PrefixSharingConfig + 约束校验
│   ├── metadata.py               # PrefixSharingBatchMeta
│   ├── detector.py               # TriePrefixDetector / PromptPrefixDetector
│   ├── planner.py                # 根据 batch 生成裁剪、分组、offset、映射计划
│   ├── cache.py                  # PrefixKVCache + 生命周期管理
│   └── mapping.py                # input / label / output restore 映射
├── integrations/                 # 模型与训练框架集成层
│   ├── verl_mcore.py             # 替代 preprocess/postprocess，接入 verl logprob
│   ├── megatron_attention.py     # SelfAttention hook，调用 backend
│   ├── megatron_rope.py          # 非 fused RoPE offset 适配
│   └── context.py                # prefix_sharing_context 生命周期
└── backends/                     # 加速器与 kernel 后端层
    ├── base.py                   # PrefixAttentionBackend 接口
    ├── torch_ref.py              # 正确性参考实现，CUDA/NPU 都可跑，慢
    ├── cuda_te.py                # CUDA TransformerEngine 后端（MVP 可选）
    ├── cuda_flash.py             # CUDA flash-attn 后端（后续）
    └── cann_npu.py               # CANN NPU 后端（后续）
```

#### 2.1 通用语义层

负责回答“哪些 token 可以共享，删哪些 token，逻辑位置是什么，输出如何对齐”。

这层只处理 PyTorch tensor 和 Python metadata，不依赖具体 attention kernel。CUDA GPU、CANN NPU、fallback torch 后端都消费同一份语义计划。

#### 2.2 模型集成层

负责把通用语义落到 verl/Megatron 的真实调用链：
- 替代或包裹 `preprocess_thd_no_padding()`，生成裁剪后的 packed input 和 `PackedSeqParams`
- patch Megatron `SelfAttention.forward()`，在 QKV 投影后调用后端完成 RoPE、KV 注入、attention
- 替代或包裹 `postprocess_thd_no_padding()`，按 metadata 还原或裁剪 logits/logprobs
- 同步处理 label、loss mask、response mask，避免 token 裁剪后 logprob 错位

#### 2.3 后端适配层

负责回答“在当前硬件和 kernel 上如何执行 attention”。

后端统一实现以下接口：

```python
class PrefixAttentionBackend:
    def validate(self, config, model_config) -> None: ...
    def apply_rope(self, query, key, meta, packed_seq_params, rope_args): ...
    def build_kv(self, key, value, cache, meta): ...
    def attention(self, query, key, value, attention_mask, packed_seq_params, meta): ...
```

MVP 不应被 CUDA TE 绑死。建议先实现：
- `TorchReferenceBackend`：用于正确性验证，CUDA/NPU 都能跑，性能不是目标
- `CudaTEBackend`：用于 CUDA 最小性能路径，必须显式关闭 unsupported fused 配置

后续通过新增 backend 支持 CANN NPU、CP、PP、fused kernel。

### 三、PrefixSharingBatchMeta 设计

`PrefixSharingBatchMeta` 是新方案的核心。它显式记录一次 micro-batch 中 prefix sharing 所需的全部语义信息。

```python
@dataclass
class PrefixSharingBatchMeta:
    # 原始 batch 结构
    batch_size: int
    original_lengths: list[int]

    # 分组与共享关系
    group_ids: list[int]
    is_provider: list[bool]
    provider_index: list[int]
    prefix_lens: list[int]
    suffix_lens: list[int]

    # 裁剪后的 query 序列，以及注入后的 KV 序列
    kept_lengths_q: list[int]
    expanded_lengths_kv: list[int]
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    max_seqlen_q: int
    max_seqlen_kv: int

    # RoPE 逻辑位置。suffix-only query 从 prefix_len 开始；KV 注入后从 0 开始连续。
    q_position_offsets: list[int]
    kv_position_offsets: list[int]

    # 输入、label、loss、输出映射
    input_keep_ranges: list[tuple[int, int]]
    label_keep_ranges: list[tuple[int, int]]
    loss_mask_keep_ranges: list[tuple[int, int]]
    output_restore_ranges: list[tuple[int, int]]
```

**示例**：

```
seq_0 = [P0 | S0]  provider
seq_1 = [P0 | S1]  reuse seq_0
seq_2 = [P0 | S2]  reuse seq_0
```

裁剪后：

```
query input:
  seq_0: [P0 | S0]  q_offset = 0
  seq_1: [S1]       q_offset = len(P0)
  seq_2: [S2]       q_offset = len(P0)

expanded KV:
  seq_0: [P0 | S0]  kv_offset = 0
  seq_1: [P0 | S1]  kv_offset = 0
  seq_2: [P0 | S2]  kv_offset = 0
```

**为什么不用单纯 `position_ids`**：
verl mcore THD 路径调用 Megatron 时普通 RoPE 模型传入的是 `position_ids=None`。Megatron 根据 `packed_seq_params` 生成 RoPE freqs，并在 `_apply_rotary_pos_emb_thd()` 或 fused THD kernel 中按 `cu_seqlens` 应用。因此 suffix-only 序列的位置修正必须由 metadata 驱动 RoPE offset，而不是只生成 `position_ids`。

### 四、数据流设计

#### 4.1 preprocess

输入为 verl micro-batch 中的 nested `input_ids`、label/logprob 所需参数和 mask。

流程：
1. detector 识别共享 prefix group
2. planner 选择每组 provider，计算每条样本的 `prefix_len` / `suffix_len`
3. 对非 provider 样本裁剪掉 prefix，只保留 suffix token
4. 构造裁剪后 packed input
5. 构造 `cu_seqlens_q`
6. 预先计算逻辑 `cu_seqlens_kv`
7. 生成 `PrefixSharingBatchMeta`
8. 对 label / loss mask / response mask 生成对应 keep ranges

#### 4.2 attention

每层 `SelfAttention.forward()`：

```
hidden_states
  → QKV projection
  → backend.apply_rope(query, key, meta)
  → cache.store(provider prefix K/V, layer_number, group_id)
  → backend.build_kv(key, value, cache, meta)
  → backend.attention(query, expanded_key, expanded_value, packed_seq_params, meta)
  → output projection
```

缓存 prefix KV 时绝不 `detach`。如需避免 view 被后续原地修改，可以 `clone()`，但必须保留 autograd 图。

#### 4.3 postprocess / loss 对齐

prefix sharing 后，非 provider 样本的输出长度从 `P+S` 变成 `S`。因此不能直接沿用原始 `input_ids.offsets()` 做 postprocess。

MVP 策略：
- provider 样本可保留完整 `[P|S]` 输出
- reuse 样本只返回 suffix 区间输出
- label、loss mask、response mask 同步裁剪到 suffix 区间
- logprob 只在保留区间计算，不尝试恢复被裁掉 prefix 的 logits

后续如果某些训练目标需要完整序列 logits，可以基于 `output_restore_ranges` 做稀疏还原，但这不是 MVP 必需。

### 五、MVP 约束

MVP 阶段只支持最小可验证路径：

| 维度 | MVP 约束 | 原因 |
|------|----------|------|
| 模型 | 普通 text GPT / causal LM | 避免 VLM、mRoPE、MLA 位置语义复杂化 |
| PP | `pipeline_model_parallel_size == 1` | patch 方案暂不跨 pipeline stage 传 KV |
| CP | `context_parallel_size == 1` | CP 下 RoPE、cu_seqlens、KV 分片语义需专项设计 |
| RoPE fusion | `apply_rope_fusion == False` | Megatron fused THD 路径绕过 `_apply_rotary_pos_emb_thd()` |
| QKV RoPE fusion | `fused_single_qkv_rope == False` | MVP 在 Q/K/V 分离后注入 KV |
| 后端 | Torch reference + CUDA 非 fused backend | 先保证正确性，再优化性能 |
| dropout | 精度测试关闭 dropout | train mode dropout 下无法做逐 token 精度对齐 |

这些约束必须在 `PrefixSharingConfig.validate()` 中硬校验。配置不满足时直接报错，不能静默 fallback 到错误语义。

### 六、未来扩展路径

#### 6.1 CANN NPU

NPU 不应复用 CUDA TE 细节。应新增 `CannNpuBackend`，消费同一份 `PrefixSharingBatchMeta`：
- 如果 CANN attention kernel 支持变长 Q/KV，则直接映射 `cu_seqlens_q` / `cu_seqlens_kv`
- 如果不支持，则用后端内部 fallback：分组 dense attention 或分段 attention
- 正确性先对齐 `TorchReferenceBackend`，再做性能优化

#### 6.2 Context Parallel

CP 需要专项处理：
- packed sequence 在 CP rank 间拆分后，prefix provider 和 reuse suffix 可能落在不同 rank
- RoPE offset 需要结合 CP rank 的局部 position
- prefix KV cache 可能需要跨 CP 通信或重新定义共享粒度

扩展方式：新增 `ContextParallelBackend` 或在 backend 中增加 CP-aware plan，不改变 core metadata 的语义字段。

#### 6.3 Pipeline Parallel

PP 下每个 stage 只持有部分层，不能依赖单进程全局 cache。

可选方向：
- 将 `PrefixSharingBatchMeta` 随 micro-batch 传递到所有 stage
- 每个 stage 在自己的层内独立缓存/注入 prefix KV
- 如需跨 stage 传额外 tensors，应做专项设计，不能混入 MVP patch

#### 6.4 Fused Kernel

未来支持 fused RoPE / fused QKV / TE fused THD 时，不应改 core 语义层，只新增或增强 backend：
- fused backend 必须显式声明支持哪些 metadata 字段
- 若 kernel 不支持 per-sequence RoPE offset，则不能启用 prefix sharing
- 所有 fused backend 都必须通过 TorchReferenceBackend 的数值/梯度对齐测试

### 七、验证计划

```
Step 0: Config validate + backend 接口
  ├─ PrefixSharingConfig.validate()
  ├─ PrefixAttentionBackend 抽象
  └─ TorchReferenceBackend 最小实现

Step 1: Metadata / planner 单元测试
  ├─ prefix group 检测
  ├─ input_keep_ranges / label_keep_ranges / output_restore_ranges
  ├─ cu_seqlens_q / cu_seqlens_kv
  └─ q_position_offsets / kv_position_offsets

Step 2: Reference attention 正确性
  ├─ 手写 dense causal attention 对齐
  ├─ q_len != kv_len 场景
  └─ suffix-only RoPE offset 场景

Step 3: Megatron 单层集成
  ├─ 关闭 fused RoPE / fused QKV
  ├─ patch SelfAttention.forward()
  ├─ logits 对齐：eval mode
  └─ grad 对齐：train mode 关闭 dropout + 固定 RNG

Step 4: verl logprob 集成
  ├─ preprocess/postprocess 替换
  ├─ label/logprob/loss mask 对齐
  └─ RL batch shape 验证

Step 5: 端到端小规模训练
  ├─ 收敛曲线对齐
  ├─ CUDA backend 性能测试
  └─ NPU reference/backend 可运行性验证
```

### 八、技术风险与应对

| 风险 | 严重程度 | 应对 |
|------|----------|------|
| RoPE fused 路径绕过 patch | 高 | MVP 强制 `apply_rope_fusion=False`；未来 fused backend 专项支持 |
| output / label / loss mask 错位 | 高 | `PrefixSharingBatchMeta` 显式记录 keep/restore ranges，postprocess 不再依赖原始 offsets |
| 后端绑定 CUDA TE | 高 | backend adapter 解耦；TorchReferenceBackend 作为共同正确性基线；CANN NPU 单独 backend |
| `cu_seqlens_q != cu_seqlens_kv` kernel 语义不一致 | 高 | 每个 backend 必须先通过 q_len/kv_len 不等的最小 attention 测试 |
| PP/CP 扩展复杂 | 高 | MVP 硬限制 PP=1/CP=1；metadata 保留扩展字段，PP/CP 走专项 backend |
| 物理 `cat(prefix_kv, suffix_kv)` 收益为负 | 中 | MVP 先保证正确性；后续增加 `min_prefix_len` / `min_group_size` / saved-token 阈值 |
| Trie 共享过短 prefix 无收益 | 中 | 默认配置增加收益阈值；通用 Trie 保留，但 planner 可按场景增加 group scope |
| cache 污染 | 中 | 使用 `prefix_sharing_context` 管理 `forward_id` / `micro_batch_id` / `layer_number` / `group_id` |

### 九、与旧 One-Forward 方案的关系

旧方案保留的部分：
- One-Forward + KV Injection 的核心策略
- prefix KV 不 detach，保留 autograd
- PP=1 的 MVP 限制
- 参考 PrefixTrain_dev 主流程中已跑通的 Trie 检测和 KV 注入思路

旧方案需要修正的部分：
- 不再依赖单纯生成 `position_ids` 修正 THD RoPE
- 不再默认 monkey-patch `_apply_rotary_pos_emb_thd()` 能覆盖 fused RoPE
- 不再假设 `PackedSeqParams` 有独立字段就等于底层 kernel 语义已验证
- 不再把 CUDA TE 作为唯一实现中心
- 明确补上 label/logprob/loss mask/output mapping 设计

---

## 2026-05-10 详细设计：One-Forward + KV Injection 方案

> 三个参考项目（PrefixTrain_dev、flash-preference、dpo-prefix-sharing）全部采用一次 forward 实现 prefix sharing。
> 本方案采用相同策略，在同一次 model forward 中完成 prefix KV 的缓存与注入。
> 旧方案 Two-Phase Forward（拆分为 prefix forward + suffix forward）已废弃。

### 一、核心思路

**问题本质**：同一组序列共享相同 prefix，传统方式重复计算 prefix 部分。

**解决方式**：在一次 model forward 中：
1. 组内第一条序列包含完整 prefix+suffix，其 prefix 部分的 KV 在每层 attention 中被缓存
2. 组内后续序列只包含 suffix（prefix 已从输入 token 中去除），通过注入缓存的 prefix KV，与自己的 suffix KV 拼接后做 attention
3. 整个过程在同一次 forward、同一个 attention layer 中完成

**关键约束**：
- 缓存的 prefix KV 不 detach，保留 autograd 计算图，确保梯度正确性
- 多个 suffix 共享同一 prefix KV 时，梯度通过 autograd 自然累积
- 必须保证与无 prefix sharing 时的前向结果数值完全一致
- **RoPE 位置编码必须连续**：suffix-only 序列的 RoPE 位置必须从 prefix_len 开始，不能从 0 重新开始
- **PP 限制**：MVP 阶段只支持 PP=1（无 pipeline parallel），PP 场景后续通过专项方案支持

**方案对比**：

| 维度 | Two-Phase Forward (废弃) | One-Forward + KV Injection |
|------|--------------------------|----------------------------|
| forward 次数 | 2 | 1 |
| 梯度正确性 | 需跨 forward 保持 autograd 图 | 单次 forward 内自然保持 |
| PP 兼容性 | 均不支持（MVP 阶段 PP=1） | 均不支持（MVP 阶段 PP=1） |
| 显存 | 高（prefix 全部中间激活保留到 backward） | 低（只缓存 KV，不缓存全部激活） |
| 性能 | 差（两次 forward 开销） | 好（一次 forward） |
| 与参考项目一致性 | 不一致 | 一致（三个项目均用 One-Forward） |

### 二、verl 数据流的改造点

#### 2.1 现状

```
forward_step(data_iterator, model)
  → forward_fn(model, input_ids, ...)
    → preprocess_thd_no_padding()     # 拼接序列、计算 cu_seqlens
    → model(input_ids, ...)           # Megatron GPTModel.forward()
    → postprocess_thd_no_padding()    # 提取 logits
  → loss_func(output, batch)
```

#### 2.2 改造方案

通过两个 patch 实现（均在 prefix-sharing 项目中开发，不修改 verl/megatron 源码）：

1. **数据预处理 patch**：替代 `preprocess_thd_no_padding()`，去除非首条序列的 prefix tokens，构造不对称的 `cu_seqlens_q` / `cu_seqlens_kv`，同时生成正确的 `position_ids`
2. **Attention hook patch**：monkey-patch `SelfAttention.forward()`，在 QKV 投影后注入 prefix KV 缓存与拼接逻辑
3. **RoPE patch**：monkey-patch `_apply_rotary_pos_emb_thd()`，支持 per-sequence 的位置偏移（suffix-only 序列的 RoPE 从 prefix_len 开始）

### 三、详细数据流设计

#### 3.1 数据组织

不同 RL 场景的 prefix 模式：

```
GRPO/DPO:      prefix = prompt
Tree-mode RL:  prefix = prompt + 分支前的共同推理
Step-mode RL:  prefix = prompt + 完整历史 action/observation
Token-mode RL: prefix = prompt + 共同 token 前缀
```

prefix 长度由 Trie 树从 input_ids 中自动检测，不假设 prefix=prompt，不依赖 response_mask。

**通用示例（8 条序列，3 个前缀组）**:

```
输入:
  seq_0: [prefix_0 | suffix_0]   prefix_len = L0  ← 组内第一条
  seq_1: [prefix_0 | suffix_1]   prefix_len = L0
  seq_2: [prefix_0 | suffix_2]   prefix_len = L0
  seq_3: [prefix_1 | suffix_3]   prefix_len = L1  ← 新组第一条
  seq_4: [prefix_1 | suffix_4]   prefix_len = L1
  seq_5: [prefix_1 | suffix_5]   prefix_len = L1
  seq_6: [prefix_2 | suffix_6]   prefix_len = L2  ← 新组第一条
  seq_7: [prefix_2 | suffix_7]   prefix_len = L2

预处理后（去除非首条序列的 prefix tokens）:
  seq_0: [prefix_0 | suffix_0]   → P_0+S_0 tokens
  seq_1: [suffix_1]              → S_1 tokens  (prefix_0 去除)
  seq_2: [suffix_2]              → S_2 tokens
  seq_3: [prefix_1 | suffix_3]   → P_1+S_3 tokens
  seq_4: [suffix_4]              → S_4 tokens  (prefix_1 去除)
  seq_5: [suffix_5]              → S_5 tokens
  seq_6: [prefix_2 | suffix_6]   → P_2+S_6 tokens
  seq_7: [suffix_7]              → S_7 tokens  (prefix_2 去除)
```

#### 3.2 Packing 与 cu_seqlens

**Packed input_ids**:
```
[prefix_0 | suffix_0 | suffix_1 | suffix_2 | prefix_1 | suffix_3 | suffix_4 | suffix_5 | prefix_2 | suffix_6 | suffix_7]
```

**cu_seqlens_q**（query 长度 = 实际输入 token 数）:
```python
[0, P_0+S_0, P_0+S_0+S_1, P_0+S_0+S_1+S_2, ..., total_nnz]
```

**cu_seqlens_kv**（KV 长度 = 注入的 prefix KV + suffix KV）:
```python
[0,
 P_0+S_0,                           # seq_0: 自身全部 KV
 (P_0+S_0)+(P_0+S_1),               # seq_1: 注入 prefix_0 KV + suffix_1 KV
 ...+(P_0+S_2),                      # seq_2: 注入 prefix_0 KV + suffix_2 KV
 ...+(P_1+S_3),                      # seq_3: 自身全部 KV（新组首条）
 ...+(P_1+S_4),                      # seq_4: 注入 prefix_1 KV + suffix_4 KV
 ...,
 total_kv_nnz]                       # > total_nnz（prefix KV 被重复注入）
```

**position_ids（关键修正）**:
```python
# 首条序列: position 与原始一致
seq_0: [0, 1, ..., P_0-1, P_0, ..., P_0+S_0-1]
seq_3: [0, 1, ..., P_1-1, P_1, ..., P_1+S_3-1]
seq_6: [0, 1, ..., P_2-1, P_2, ..., P_2+S_6-1]

# suffix-only 序列: position 必须从 prefix_len 开始，不能从 0 重新开始
seq_1: [P_0, P_0+1, ..., P_0+S_1-1]
seq_2: [P_0, P_0+1, ..., P_0+S_2-1]
seq_4: [P_1, P_1+1, ..., P_1+S_4-1]
# ...
```
> **为什么重要**：Megatron THD 模式下 RoPE 默认从位置 0 开始应用。如果 suffix-only 序列的 position_ids 从 0 开始，RoPE 编码会错误，导致 attention 计算的位置信息不准确。因此 suffix-only 序列必须使用原始完整序列中 suffix 对应的位置编号。

**关键**: `cu_seqlens_q ≠ cu_seqlens_kv`。Megatron `PackedSeqParams` 已支持两者独立设置（已验证 `packed_seq_params.py` 中 `cu_seqlens_q` 和 `cu_seqlens_kv` 是独立字段）。

#### 3.3 每层 Attention Forward

```
SelfAttention.forward(hidden_states, ..., packed_seq_params):

  1. QKV 线性投影 → Q, K, V [total_nnz, ng, hn]
     - seq_0 的 KV: prefix_0 + suffix_0（输入包含完整 tokens）
     - seq_1 的 KV: 仅 suffix_1（输入只有 suffix tokens）

  2. Prefix KV 缓存（从首条序列提取）
     - 从 seq_0 的 KV 中提取 prefix_0 部分（前 P_0 个 token）
     - 存入 PrefixKVCache（保留 autograd，不 detach）

  3. Prefix KV 注入（到后续序列）
     - 对 seq_1: K = cat(prefix_0_K, seq_1_K)  # [P_0+S_1, ng, hn]
     -             V = cat(prefix_0_V, seq_1_V)
     - 对 seq_0（首条）: 不注入，KV 已包含 prefix

  4. 调整 PackedSeqParams:
     - cu_seqlens_q 不变
     - cu_seqlens_kv 扩展为包含 prefix KV 的长度

  5. RoPE 应用（关键修正）
     → Megatron THD 模式下 `_apply_rotary_pos_emb_thd` 默认 CASE 2：
       每个序列的 RoPE 从 freqs[0] 开始，不支持 per-sequence 偏移
     → **需要 monkey-patch**：传入 `position_offsets` 参数
       - seq_0 offset = 0
       - seq_1 offset = P_0（suffix-only 序列从 prefix_len 开始）
       - seq_2 offset = P_0
       - seq_3 offset = 0（新组首条）
       - ...
     → patch 后的 `_apply_rotary_pos_emb_thd` 对每个序列使用正确的偏移

  6. FlashAttention varlen(Q, K_expanded, V_expanded, packed_seq_params)
     → Q 长度 = 实际输入长度
     → KV 长度 = prefix + suffix
     → causal mask 保证正确 attention

  7. 输出 → output projection → MLP → 下一层
```

### 四、代码组织：Patch 形式

**不修改 verl 和 megatron 源码**，所有逻辑在 prefix-sharing 项目中实现。

```
prefix_sharing/
├── __init__.py
├── config.py                     # PrefixSharingConfig
├── detector.py                   # TriePrefixDetector, PromptPrefixDetector
├── kv_cache.py                   # PrefixKVCache（micro-batch 隔离）
├── patches/
│   ├── __init__.py
│   ├── megatron_attention.py     # monkey-patch SelfAttention.forward()
│   ├── megatron_rope.py          # monkey-patch _apply_rotary_pos_emb_thd()
│   └── verl_preprocess.py        # 替代 preprocess_thd_no_padding()
├── integration/
│   ├── __init__.py
│   └── verl_megatron.py          # enable_prefix_sharing(actor)
└── utils.py
```

**Patch 方式**：
- `patches/verl_preprocess.py`: 运行时替换 forward 函数中的 preprocess 步骤
- `patches/megatron_attention.py`: `SelfAttention.forward = patched_forward`
- `integration/verl_megatron.py`: `enable_prefix_sharing(actor)` 一行启用，`disable_prefix_sharing()` 卸载

### 五、复用 PrefixTrain_dev 已调测代码

PrefixTrain_dev 是经过调测的技术原型，大部分代码可用（仅需修复 detach bug）。新方案尽量复用其已验证代码，减少调测成本。

#### 5.1 可直接复用

| PrefixTrain_dev 代码 | 功能 | 复用方式 |
|---------------------|------|----------|
| `prefix_match.py` → `get_store_shared_tensor()` | Trie 树前缀检测 | 迁移到 `detector.py`，仅改接口 |
| `prefix_match.py` → `process_in_order()` | 统计总 token 数 | 直接迁移 |
| `prefix_match.py` → `partition_micro_batch_token_level()`, `kk_partition()` | 缓存感知微批次划分 | 迁移到 `scheduler.py` |
| `memory_manager/memory.py` → `MemoryManager` | KV 缓存管理 | 迁移并修复 `clone().detach()` → `clone()` |
| `model/flex_ops.py` → `FlexFlashAttentionOp.forward()` | attention 层 KV 注入拼接 | 参考 KV 注入流程，适配到 monkey-patch |

#### 5.2 需新开发

| 模块 | 原因 |
|------|------|
| `patches/verl_preprocess.py` | PrefixTrain_dev 未与 verl 集成 |
| `patches/megatron_attention.py` | PrefixTrain_dev 直接魔改 Megatron，需改为 monkey-patch |
| `integration/verl_megatron.py` | PrefixTrain_dev 未与 verl 集成 |

#### 5.3 PrefixTrain_dev detach Bug（已确认）

`memory_manager/memory.py:42` 对缓存的 KV 执行 `clone().detach()`，切断了 prefix KV 的 autograd 图。后果：suffix 序列的梯度无法通过 prefix KV 回传到模型参数，QKV 权重只收到 suffix 部分的梯度。PoC 未发现原因：仅在模拟数据上跑了一个 iteration，未验证训练收敛性。

**我们的修复**：使用 `clone()` 不做 `detach()`，保留梯度流。代价是显存占用更高，但梯度正确性是硬性要求。

### 六、梯度正确性分析

#### 6.1 前向计算图（单次 forward 内）

```
packed_input_ids (含首条完整 tokens 和后续 suffix tokens)
    ↓ (embedding, 带梯度)
hidden_states [total_nnz, b, h]
    ↓ (每层 TransformerLayer)

每层 attention:
  QKV 投影 → Q, K, V

  首条序列: prefix KV → 存入 PrefixKVCache（保留 autograd）
  后续序列: prefix KV 从 cache 取出（带 autograd）+ suffix KV → 拼接

  attention(Q, K, V) → output → MLP → ... → logits → loss
```

#### 6.2 反向传播

```
loss.backward():
  逐层 backward:
    dL/dK 拆分为 d_prefix_K + d_suffix_K
    dL/dV 拆分为 d_prefix_V + d_suffix_V
    d_prefix_K, d_prefix_V → 通过 autograd 回传到首条序列的对应层

  多个 suffix 共享同一 prefix KV:
    梯度在 autograd 图中自然累积
    等价于无 prefix sharing 时 N 条完整序列的 prefix 部分梯度之和

  最终: 模型参数梯度与无 prefix sharing 时完全一致 ✓
```

#### 6.3 数值等价性

**无 prefix sharing**: 每个 S_i = [P | R_i] 独立 forward，P 部分的 KV 值完全相同（因为输入 tokens 相同）。

**有 prefix sharing**: seq_0 的 prefix KV 缓存后在 seq_1..N 的 attention 中注入。同一 forward 内、同一组参数下的 prefix KV 值与独立 forward 时一致。FlashAttention 输入 (Q, K, V) 完全相同，输出数值一致。

**结论**: One-Forward + KV Injection 的结果与逐条独立 forward 数值完全一致，且梯度正确。

### 七、实现步骤

```
Step 1: 前缀检测
  ├─ 从 PrefixTrain_dev 迁移 get_store_shared_tensor() → TriePrefixDetector
  ├─ 实现 PromptPrefixDetector（GRPO/DPO 场景特化）
  └─ 单元测试 + verl batch 验证

Step 2: 数据预处理 Patch + position_ids
  ├─ 实现 preprocess_with_prefix_sharing()
  ├─ 去除非首条序列的 prefix tokens, 构造不对称 cu_seqlens_q / cu_seqlens_kv
  ├─ 生成正确的 position_ids（suffix-only 序列从 prefix_len 开始）
  └─ 验证 cu_seqlens 和 position_ids 正确性

Step 3: 单层 KV 缓存 + Hook + RoPE 验证
  ├─ 迁移 MemoryManager → PrefixKVCache（修复 detach bug，micro-batch 隔离）
  ├─ 单层 patched_self_attention_forward()
  ├─ monkey-patch `_apply_rotary_pos_emb_thd` 支持 per-sequence 偏移
  ├─ 数值验证: logits 差异 < 1e-5
  ├─ 梯度验证: 参数梯度差异 < 1e-5
  └─ RoPE 验证: 比较 patch 前后 suffix-only 序列的 RoPE 输出

Step 4: 多层完整 Hook
  ├─ 完整多层 PrefixKVCache + monkey-patch SelfAttention.forward()
  ├─ 完整模型数值/梯度一致性验证
  └─ 性能测试: prefix sharing 加速比

Step 5: verl MegatronPPOActor 集成
  ├─ 实现 enable_prefix_sharing() / disable_prefix_sharing()
  ├─ verl PPO 训练循环端到端验证
  ├─ 训练收敛曲线一致性验证
  └─ 端到端加速比测量

Step 6: 微批次优化
  ├─ 迁移 partition_micro_batch_token_level() 和 kk_partition()
  ├─ 缓存感知的 micro-batch 排序
  └─ 优化后加速比测量

Step 7: 分布式验证（MVP 限制 PP=1）
  ├─ TP: prefix KV 在 tensor parallel 下的正确性
  ├─ DP: 数据并行下的梯度一致性
  └─ 多卡性能测试

  > 注：PP（pipeline parallel）在 MVP 阶段不支持。PrefixTrain_dev 通过魔改 Megatron
  > 跨 stage 传递 activation 实现 PP 下的 prefix sharing，我们的 patch 方案暂无法做到。
  > PP 支持作为后续专项任务。
```

### 八、技术风险

| 风险 | 严重程度 | 应对 |
|------|----------|------|
| **RoPE 位置编码不连续** | 高 | monkey-patch `_apply_rotary_pos_emb_thd` 支持 per-sequence 偏移；Step 3 验证 RoPE 输出与原始一致 |
| **PP 不支持** | 高 | MVP 阶段限制 PP=1；PP 支持需专项设计（PrefixTrain_dev 通过魔改 Megatron 跨 stage 传递 activation 实现） |
| **cu_seqlens_q ≠ cu_seqlens_kv 需验证** | 中 | `PackedSeqParams` 已支持独立设置，Step 3 小规模验证 |
| **Prefix KV 物理拷贝的显存开销** | 中 | 限制同组 suffix 数量 / KK 负载均衡 / FlashAttention block table 共享 |
| **Micro-batch cache 污染** | 中 | PrefixKVCache 使用 micro-batch ID 隔离；PP 虚拟流水线场景禁用 prefix sharing |
| **TP 下 KV 格式一致性** | 低 | 每个 GPU 只缓存自己的 heads 的 KV，无需跨 GPU 通信 |

---

## 2026-05-10 补充：Megatron-LM v0.15.0 Forward 流程与注入点分析

### 一、GPTModel Forward 完整调用链

```
GPTModel.forward(input_ids, position_ids, attention_mask, packed_seq_params, ...)
  │
  ├─ _preprocess()
  │    ├─ embedding(input_ids, position_ids)    # [b,s] → [s,b,h]
  │    └─ rotary_pos_emb 计算                   # RoPE 位置编码
  │
  ├─ TransformerBlock.forward(hidden_states, attention_mask, ...)
  │    │  输入: [s, b, h]
  │    │
  │    ├─ TransformerLayer[0].forward(...)
  │    │    ├─ _forward_attention()
  │    │    │    ├─ SelfAttention.forward(hidden_states, attention_mask, ...)
  │    │    │    │    ├─ get_query_key_value_tensors()     # QKV 线性投影 [s,b,h] → Q,K,V [s,b,np,hn]
  │    │    │    │    ├─ _adjust_key_value_for_inference() # 推理时 KV cache 管理
  │    │    │    │    ├─ rotary_pos_emb 应用               # RoPE 作用于 Q, K
  │    │    │    │    └─ DotProductAttention.forward(Q, K, V, attention_mask, ...)
  │    │    │    │         ├─ Q·K^T → attention_scores    # [b,np,sq,sk]
  │    │    │    │         ├─ softmax + dropout
  │    │    │    │         └─ attention_probs · V → output  # [s,b,h]
  │    │    │    └─ output_projection
  │    │    └─ _forward_mlp()
  │    │         └─ MLP / MoE
  │    │
  │    ├─ TransformerLayer[1].forward(...)
  │    ├─ ...
  │    └─ TransformerLayer[N-1].forward(...)
  │
  └─ _postprocess()
       ├─ final_layernorm
       └─ output_layer → logits 或 loss
```

### 二、张量形状变化

| 阶段 | 张量 | 形状 |
|------|------|------|
| 输入 | `input_ids` | `[b, s]` |
| Embedding 后 | `hidden_states` | `[s, b, h]` |
| QKV 投影后 | `Q, K, V` | `[s, b, np, hn]`（np=attention heads per TP rank） |
| Attention 得分 | `scores` | `[b, np, sq, sk]` |
| Attention 输出 | `context` | `[s, b, h]` |
| 最终输出 | `logits` | `[b, s, vocab]` |

### 三、Prefix Sharing 精确注入位置

#### 注入层 1: verl 数据预处理层（前缀检测与分组）

**位置**: `verl/models/mcore/util.py` → `preprocess_thd_no_padding()`

```
职责: 在数据进入 Megatron 之前，识别共享前缀并构建分组信息
注入点:
  - cu_seqlens 计算处 (约 line 295-303): 插入前缀分组信息
  - 序列重打包处 (约 line 319-339): 按前缀分组重新组织序列
需要新增:
  - PrefixInfo 数据结构（prefix_length, group_id, cache_sample_idx）
  - 前缀检测算法调用
```

#### 注入层 2: verl forward 入口层（编排共享 forward）

**位置**: `verl/models/mcore/model_forward.py` → `gptmodel_forward_no_padding()`

```
职责: 在调用 Megatron model() 之前/之后，编排 prefix sharing 的 forward
注入点:
  - model() 调用前 (约 line 178-190): 前缀部分先做一次 forward
  - model() 调用处 (约 line 190-196): 非前缀部分使用缓存的 KV 继续计算
需要新增:
  - 前缀 embedding 缓存机制
  - 前缀 KV cache 注入逻辑
```

#### 注入层 3: Megatron Attention 层（KV cache 共享核心）

**位置**: `megatron/core/transformer/attention.py` → `SelfAttention.forward()`

```
职责: 在 attention 层实现 KV cache 的复用
注入点:
  - get_query_key_value_tensors() 之后 (约 line 728-737): 替换/扩展 K, V
  - _adjust_key_value_for_inference() 处 (约 line 783-795): 注入缓存的 prefix KV
需要新增:
  - 从 PrefixSharingManager 获取缓存的 prefix KV
  - 将 prefix KV 拼接到当前序列的 KV 前面
  - 正确处理 attention_mask 以包含 prefix 部分
```

#### 注入层 4: DotProductAttention 层（注意力计算适配）

**位置**: `megatron/core/transformer/dot_product_attention.py` → `forward()`

```
职责: 适配 attention 计算以支持扩展的 KV（prefix + suffix）
注入点:
  - KV 扩展处 (约 line 150-193): 处理拼接了 prefix KV 的 K, V
  - attention_mask 处理: 确保 prefix KV 可被 suffix 正确 attend
注意:
  - 如果在注入层 3 已处理好 KV 拼接，此层可能无需修改
  - 但需要验证 FlashAttention 对超长 KV 序列的处理
```

#### 注入层 5: 微批次调度层（缓存感知调度）

**位置**: `megatron/core/pipeline_parallel/schedules.py`

```
职责: 确保同一前缀组的 micro-batch 连续调度，最大化缓存复用
注入点:
  - set_current_microbatch() (约 line 181-197): 注入缓存感知的排序
需要新增:
  - 缓存感知的 micro-batch 排序策略
  - 前缀缓存的生命周期管理（何时分配、何时清理）
```

### 四、推荐注入策略

```
优先级 1（MVP）: 层 1 + 层 2 + 层 3
  - 在 verl 预处理层做前缀检测
  - 在 verl forward 入口编排共享 forward
  - 在 Megatron attention 层注入 KV cache 复用
  - 跳过微批次优化，暂用默认调度

优先级 2（优化）: 层 5
  - 加入缓存感知的微批次调度
  - 减少缓存失效带来的开销

优先级 3（验证）: 层 4
  - 验证 FlashAttention 对拼接 KV 的正确性
  - 必要时做针对性适配
```

### 五、mbridge 桥接层分析

#### 5.1 两种桥接模式

| 模式 | 导入路径 | 特点 |
|------|----------|------|
| VANILLA_MBRIDGE | `from mbridge import AutoBridge` | 独立包，基础功能 |
| MEGATRON-BRIDGE | `from megatron.bridge import AutoBridge` | 深度集成，支持 PEFT |

#### 5.2 核心接口

```
AutoBridge
├── from_config(hf_config) → 创建桥接器
├── from_hf_pretrained(path) → 从 HF 模型创建
├── load_weights() → 加载 Megatron 格式权重
├── load_hf_weights() → 加载 HF 格式权重
├── save_weights() / save_hf_weights() → 保存权重
├── export_weights() → 导出权重
└── to_megatron_provider() → 创建 Megatron Provider（仅 MEGATRON-BRIDGE）
```

#### 5.3 verl 调用 Megatron 的实际流程

```
verl MegatronEngine._build_megatron_module()
  → AutoBridge.from_config(hf_config) 或 from_hf_pretrained()
  → make_megatron_module(bridge=bridge, ...)
  → 得到 Megatron 模块

verl MegatronEngine.forward_step()
  → get_mcore_forward_no_padding_fn(hf_config)
  → forward_fn(model, input_ids, ...)
      → preprocess_thd_no_padding()   # 数据预处理
      → model(input_ids, ...)          # 调用 Megatron GPTModel.forward()
      → postprocess_thd_no_padding()   # 后处理
```

#### 5.4 对 prefix sharing 的启示

- mbridge 负责**配置转换和权重加载**，不参与 forward 流程，所以 prefix sharing 不需要修改 mbridge
- prefix sharing 的注入点在 mbridge 之后的 `forward_fn` 和 Megatron attention 层
- mbridge 的 PEFT 扩展机制（LoRA/DoRA）可以作为参考，但 prefix sharing 本质上是 forward 流程优化，不适合作为 PEFT 类型

### 六、Megatron Sequence Packing 机制

#### 6.1 PackedSeqParams 数据结构

```python
@dataclass
class PackedSeqParams:
    qkv_format: str = None        # "thd" 或 "bshd"
    cu_seqlens_q: Tensor = None   # [num_seqs+1] 累计 query 序列长度
    cu_seqlens_kv: Tensor = None   # [num_seqs+1] 累计 KV 序列长度
    max_seqlen_q: int = None       # 最大 query 序列长度
    max_seqlen_kv: int = None      # 最大 KV 序列长度
```

#### 6.2 Packing 对 prefix sharing 的意义

- **现有基础设施可复用**: cu_seqlens 机制天然支持多个序列在同一个 batch 中
- **前缀共享可视为特殊的 packing**: 将 prefix 和 suffix 分别视为不同的 "段"
- **可能的扩展**: 为每个序列添加 `cu_prefix_lens`（每个序列的前缀长度），在 attention 中据此区分 prefix KV 和 suffix KV

---

## 2026-05-10 初始方案设计与代码分析

### 一、代码分析

#### 1.1 PrefixTrain_dev (PoC) 分析

**架构**:
```
PrefixTrain_dev/
├── runtime/megatron/          # 魔改版 Megatron
│   ├── model/
│   │   ├── flex_gpt.py       # 灵活 GPT 模型（动态并行划分）
│   │   ├── flex_model.py     # 灵活模型框架（PP 阶段管理）
│   │   └── flex_ops.py       # 操作定义
│   ├── training.py           # 训练循环（集成前缀共享）
│   ├── prefix_match.py       # 前缀匹配算法核心
│   └── memory.py             # 内存管理（RingMemBuffer）
├── hetero_search/            # 异构并行搜索
└── our_scripts/              # 实验脚本
```

**主流程实际使用的核心机制**（经代码验证，仅包含 training.py 主流程中真正被调用的部分）:
1. **前缀检测**: 仅使用 **Trie 树**算法
   - `get_store_shared_tensor(data)`: 核心函数，按顺序遍历序列，Trie 树匹配已有前缀，返回每个样本可复用的前缀长度。training.py:164,184,323 处调用
   - `process_in_order(token_lists)`: 辅助函数，统计总需计算 token 数，用于负载均衡。training.py:235,238,272 处调用
   - 注意：`compute_longest_shared_prefixes` 和 `compute_longest_shared_prefixes_tokens`（排序后相邻比较）是死代码，仅在 `__main__` 注释中出现，主流程不调用
2. **前缀共享**: 每个样本记录 `shared_prefix_len`（可复用长度）和 `store_for_sample_idx`（`get_store_shared_tensor` 的输出）
3. **Forward 优化**: `effective_len -= shared_prefix_len[idx]`，跳过已缓存的前缀计算
4. **微批次划分**: 缓存感知的负载均衡
   - `partition_micro_batch_token_level`: token 级别划分（首选）
   - `partition_micro_batch`: 基本划分（fallback）
   - `kk_partition`: Karmarkar-Karp 算法做跨 pipeline stage 的负载均衡
5. **权重共享**: 跨 pipeline stage 的权重共享和梯度同步

**关键数据结构**:
```python
shared_prefix_len = []       # 每个样本可复用的前缀长度
store_for_sample_idx = []    # 提供缓存的样本索引
shared_for_sample_idx = []   # 从哪个样本复用缓存
```

**PoC 局限性**:
- 仅基于模拟 rollout 数据运行，非真实 RL 场景
- 深度耦合魔改版 Megatron，无法独立使用
- 仅支持 GPT 模型
- 仅跑通一个 iteration 的 forward/backward/update

#### 1.2 verl v0.7.0 分析

**架构**:
```
verl/
├── protocol.py              # DataProto 统一数据协议（基于 TensorDict）
├── trainer/
│   ├── ppo/                 # PPO 训练器（RayPPOTrainer）
│   └── config/              # 配置管理
├── workers/
│   ├── actor/               # Actor（含 FSDP/Megatron 后端）
│   ├── critic/              # Critic
│   ├── rollout/             # 推理引擎（vLLM/SGLang/Naive）
│   ├── engine/              # 计算引擎抽象层
│   └── reward_manager/      # 奖励模型管理
└── utils/
```

**RL 训练流程**:
```
Prompt → Rollout(生成response) → Reward(计算奖励) → Advantage(计算优势)
    → Actor Update + Critic Update → 下一个 epoch
```

**与 Megatron 集成**: 通过 `mbridge` 桥接层，`MegatronEngine` 封装 TP/PP/DP 并行

**关键集成切入点**:
1. `workers/actor/megatron_actor.py` - Actor 的 log_prob 计算
2. `workers/rollout/` - Rollout 生成阶段
3. `protocol.py` - DataProto 数据协议扩展
4. `trainer/ppo/` - 训练循环
5. `workers/engine/` - Megatron 引擎层

**插件化机制**: 注册机制（`@register_adv_est`, `@register_policy_loss`）、引擎/Rollout 可插拔、dataclass + Hydra 配置

#### 1.3 flash-preference 分析

**核心 API**:
```python
with shared_prefix(model, input_ids=..., attention_mask=...):
    output = model(**inputs)
```

**实现机制**:
1. 上下文管理器管理 prefix sharing 生命周期
2. Monkey Patch 修改 FlashAttention、RotaryEmbedding、LayerAttention
3. `get_prefix_lens` 通过比较相邻 token ID 检测共享前缀
4. `to_shared` 合并相同前缀，`to_unshared` 恢复原始格式
5. **首层共享、末层恢复**: 第一个 layer 共享前缀，最后一个 layer 恢复

**设计亮点**: 极简 API、零侵入性、2-3x 加速、30-50% 显存节省
**局限**: 基于 HF Transformers，非 Megatron；不涉及分布式优化

#### 1.4 dpo-prefix-sharing 分析

**核心**: TRL DPOTrainer 扩展 + FlexAttention 自定义 mask
**序列结构**: `[prompt + chosen + prompt_last_token + rejected]`
**设计亮点**: 数值等价性保证、99.5% packing 效率
**局限**: 仅 DPO、依赖 PyTorch 2.5+ FlexAttention、非分布式

#### 1.5 技术对比

| 维度 | PrefixTrain_dev | flash-preference | dpo-prefix-sharing |
|------|-----------------|------------------|---------------------|
| 目标场景 | 通用 RL 训练 | DPO/RM/GRPO | DPO |
| 底层框架 | 魔改 Megatron | HF + FlashAttention | HF + FlexAttention |
| 前缀检测 | Trie 树 | token ID 比较 | 假定 prompt 已知 |
| 共享机制 | activation 复用 | monkey patch 层级共享 | FlexAttention mask |
| 分布式支持 | TP/PP/DP | 无 | 无 |
| 精度保证 | 未验证 | 梯度正确传播 | 数值等价 |

**关键借鉴点**:
1. flash-preference 的上下文管理器 API → 目标 API 设计灵感
2. PrefixTrain_dev 的分布式训练集成 → Megatron/verl 集成参考
3. PrefixTrain_dev 的 Trie 树前缀检测（`get_store_shared_tensor`）→ 高效前缀检测算法
4. dpo-prefix-sharing 的数值等价性保证 → 精度验证方法

---

### 二、系统架构设计

#### 2.1 设计原则

1. **插件化**: 通过配置开关启用，不修改 verl/megatron 核心代码
2. **模块化**: 前缀检测、前缀共享、训练集成各层解耦
3. **零精度损失**: prefix sharing 前后训练结果数值一致
4. **通用性**: 支持 PPO/GRPO/DPO 等多种 RL 算法和 tree/step-mode 场景
5. **高性能**: 充分利用 Megatron TP/PP/DP 并行

#### 2.2 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                    prefix-sharing 包                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ prefix_detect │  │ prefix_share │  │  integration  │  │
│  │ • Trie 树检测 │  │ • KV Cache   │  │ • verl actor │  │
│  │ • 自定义策略  │  │   共享/复用   │  │ • verl critic│  │
│  │              │  │ • Activation │  │ • megatron   │  │
│  │              │  │   共享/复用    │  │   engine     │  │
│  │              │  │ • 微批次优化   │  │ • rollout    │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│  ┌──────────────┐  ┌──────────────┐                     │
│  │   config     │  │   utils      │                     │
│  │ • 参数配置    │  │ • 内存管理    │                     │
│  │ • 场景策略    │  │ • 性能分析    │                     │
│  └──────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
   ┌──────────┐      ┌──────────────┐    ┌───────────┐
   │   verl   │      │  Megatron-LM │    │  mbridge  │
   └──────────┘      └──────────────┘    └───────────┘
```

#### 2.3 核心模块接口

**prefix_detect（前缀检测）**:
```python
class PrefixDetector:
    def detect(self, input_ids: torch.Tensor, group_size: int = 2,
               strategy: str = "trie") -> PrefixInfo:
        """检测序列 batch 中的共享前缀"""
```

**prefix_share（前缀共享）**:
```python
class PrefixSharingManager:
    def __init__(self, config: PrefixSharingConfig): ...
    def share_forward(self, model, input_ids, prefix_info): ...
    def cache_prefix(self, layer_idx, prefix_key, activation): ...
    def reuse_prefix(self, layer_idx, prefix_key): ...
```

**integration（verl 集成）**:
```python
class VerlPrefixSharing:
    def patch_actor(self, actor): ...
    def patch_critic(self, critic): ...
    def patch_rollout(self, rollout): ...

class MegatronPrefixSharing:
    """Megatron 引擎层的 prefix-sharing 实现"""
```

#### 2.4 用户 API（目标）

```yaml
# 方式 1: 配置化启用（推荐）
prefix_sharing:
  enabled: true
  strategy: "trie"
  cache_type: "kv"          # "kv" | "activation"
  group_size: 8
  scene: "tree_mode"        # "tree_mode" | "step_mode" | "dpo"
```

```python
# 方式 2: 代码 API
from prefix_sharing import enable_prefix_sharing
enable_prefix_sharing(model_or_actor, strategy="trie", scene="tree_mode", group_size=8)
```

#### 2.5 数据流设计

**Tree-mode RL 场景**:
```
原始 rollout（同一个 prompt 生成多条 response）:
  prompt_1 + response_1a ──┐
  prompt_1 + response_1b ──┤ prompt_1 完全相同，可共享
  ...                      │
  prompt_1 + response_1h ──┘

前缀共享处理:
  1. 前缀检测: 识别每组共享 prompt 的长度
  2. KV Cache: prompt 只做一次 forward
  3. 续接计算: 各 response 续接在共享 KV cache 后
  4. 梯度处理: 正确处理共享部分的梯度累积
```

---

### 三、施工计划

| Phase | 内容 | 关键交付 |
|-------|------|----------|
| **1. 基础设施** | 包结构、前缀检测模块、精度验证框架 | 可运行的检测算法 + 测试 |
| **2. 核心实现** | 单卡 PoC、KV Cache 管理、梯度正确性 | 单卡 Megatron 上跑通 prefix sharing |
| **3. verl 集成** | Actor/Critic 集成、配置系统、端到端测试 | verl PPO/GRPO pipeline 可用 |
| **4. 分布式优化** | TP/PP 支持、通信优化、内存优化 | 多卡分布式训练可用 |
| **5. 工程化** | API 完善、测试覆盖、文档、性能报告 | 可发布状态 |

### 四、风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Megatron KV cache 拦截困难 | 高 | 先在简化模型验证可行性 |
| 分布式梯度同步正确性 | 高 | 逐步验证 TP → PP → DP |
| 精度等价性难以保证 | 高 | 建立严格数值验证框架 |
| verl 接口不稳定 | 中 | 基于 v0.7.0，保持兼容层 |
| 性能收益不及预期 | 中 | 先做单卡 benchmark |
