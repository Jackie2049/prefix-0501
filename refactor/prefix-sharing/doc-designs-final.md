# Prefix Sharing 最终设计方案

> 本文档维护当前用于开发的最终方案。历史讨论、被推翻方案、备选方案和阶段性分析记录见 `doc-design-history.md`。

---

## 1. 项目目标

### 1.1 总目标

在 rllm + verl + Megatron 的 RL 训练链路中，以非侵入式 patch / wrapper / monkey-patch 方式实现 prefix sharing，减少同一 micro-batch 内共享 prefix 的重复计算，同时在明确假设下保持 logits、logprob、loss 和梯度语义一致。

### 1.2 阶段一目标

阶段一只聚焦 `verl + Megatron` 的 RL MVP：

- 支持 actor logprob / update 主路径
- 使用 Trie 自动识别 batch 内共同 prefix
- reuse 样本裁剪 prefix，仅计算 suffix query
- attention 内通过 KV injection 复用 provider prefix K/V
- logprob 对齐使用 Prefix-Last Restore，保证第一个 response token 的 next-token 语义正确
- 对 verl 和 Megatron 不做源码侵入式修改

### 1.3 非目标

阶段一不追求以下能力：

- 不支持 PP / CP / EP 的完整业务落地能力
- 不支持 fused RoPE / fused QKV RoPE / 高性能 fused attention 后端
- 不支持 VLM、多模态位置编码、mRoPE、MLA 等复杂模型结构
- 不支持独立 Megatron / FSDP 的 SFT 或 pretrain
- 不做 verl 或 Megatron 上游 PR

这些能力纳入后续 roadmap。

---

## 2. 核心结论

### 2.1 最终精度方案

阶段一正式方案为：

```
One-Forward + KV Injection + Prefix-Last Restore
```

对共享 prefix 的一组样本：

- provider 样本保留完整 `[prefix | suffix]` 计算
- reuse 样本只保留 suffix query
- 每层 attention 通过注入 provider 的 prefix K/V，使 reuse suffix 能 attend 到完整历史
- 在 logprob 对齐阶段，从 provider 恢复 prefix-last 位置的输出/logits/logprob，用于计算 reuse 样本第一个 suffix token 的 logprob

### 2.2 为什么必须做 Prefix-Last Restore

causal LM 的训练和打分语义是 next-token prediction：

```
output(P_last) -> predict S0
output(S0)     -> predict S1
```

如果 reuse 样本只计算：

```
Q  = [S0, S1, S2]
KV = [P0, P1, P2, S0, S1, S2]
```

则 attention 输出只有：

```
output(S0), output(S1), output(S2)
```

这可以计算 `S1`、`S2` 等后续 suffix token 的 logprob，但缺少：

```
output(P2) -> predict S0
```

Prefix-Last Restore 的作用就是把 provider 已经计算过的 `output(P_last)` 显式恢复给 reuse 样本，用于第一个 response token 的 logprob。

### 2.3 等价性假设

Prefix-Last Restore 的理论前提是：同一 prefix group 内，各样本的 prefix token 完全一致，并且 prefix 的位置、mask、RoPE、模型参数和随机性条件一致。因此 provider 样本计算得到的 prefix-last hidden/logits/logprob，与每个 reuse 样本独立完整计算时对应位置的结果等价。

精度对齐必须在以下条件下验证：

- eval mode；或 train mode 下关闭 dropout 并固定 RNG
- 相同模型参数、相同 dtype、相同 attention backend 配置
- prefix token、position、mask 完全一致
- cache 不 `detach`，梯度能从多个 reuse 样本的 restore logprob 回流并累积到共享 prefix 计算图
- 对齐范围只比较 prefix sharing 语义声明覆盖的 logprob / loss 区间

---

## 3. 总体架构

项目拆为三层：

```
prefix_sharing/
├── core/                         # 通用语义层
│   ├── config.py                 # PrefixSharingConfig + 约束校验
│   ├── metadata.py               # PrefixSharingBatchMeta
│   ├── detector.py               # PrefixDetector / TriePrefixDetector
│   ├── planner.py                # 分组、裁剪、offset、restore 计划
│   ├── mapping.py                # input / label / mask / output 坐标映射
│   └── cache.py                  # PrefixKVCache 生命周期管理
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

阶段一的核心原则：

- `core/` 不依赖 verl、Megatron、CUDA TE、flash-attn 或 CANN 细节
- `integrations/` 只做接入和数据桥接，不沉淀核心算法
- `backends/` 消费统一 metadata，负责不同硬件和 kernel 的执行细节
- patch 层设计为未来上游 PR 的临时形态，正式逻辑尽量放在可迁移模块中

---

## 4. 核心数据模型

### 4.1 PrefixSharingConfig

阶段一配置至少包含：

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

`validate()` 必须硬校验阶段一约束：

- `pipeline_model_parallel_size == 1`
- `context_parallel_size == 1`
- `apply_rope_fusion == False`
- `fused_single_qkv_rope == False`
- 当前模型为普通 text causal LM
- 当前 integration path 为 verl + Megatron actor

不满足约束时直接报错，不能静默 fallback 到可能错误的语义。

### 4.2 PrefixSharingBatchMeta

`PrefixSharingBatchMeta` 是一次 micro-batch 的完整语义计划：

```python
@dataclass
class PrefixReuseSpec:
    reuse_batch_index: int
    provider_batch_index: int
    prefix_len: int

@dataclass
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
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    max_seqlen_q: int
    max_seqlen_kv: int

    q_position_offsets: list[int]
    kv_position_offsets: list[int]

    input_keep_ranges: list[tuple[int, int]]
    label_keep_ranges: list[tuple[int, int]]
    loss_mask_keep_ranges: list[tuple[int, int]]

    prefix_last_restore: list[PrefixLastRestoreSpec]
```

`prefix_last_restore` 描述每个 reuse 样本的第一个 suffix token logprob 应从哪个 provider 的哪个 prefix-last 输出位置恢复。

阶段一的核心语义是 **per-sample reuse relation**，而不是“一个 group 只能有一个统一 prefix_len”。同一个 provider 可以为不同 reuser 提供不同长度的 prefix slice，例如：

```python
PrefixReuseSpec(reuse_batch_index=1, provider_batch_index=0, prefix_len=5)
PrefixReuseSpec(reuse_batch_index=2, provider_batch_index=0, prefix_len=10)
```

`provider_index[i]` 和 `prefix_lens[i]` 是按样本索引展开后的快速访问字段。`group_ids` 只作为调试、统计或后续执行优化信息，不作为 prefix sharing 语义的唯一来源；provider 可能出现在多个不同长度的 group 中，因此不应依赖单个 `group_id` 表达 provider 的完整复用关系。

### 4.3 PrefixLastRestoreSpec

```python
@dataclass
class PrefixLastRestoreSpec:
    reuse_batch_index: int
    provider_batch_index: int
    provider_prefix_last_pos: int
    reuse_first_suffix_label_pos: int
    output_slot: int
```

该结构只描述语义，不绑定 logits 是 dense tensor、packed tensor，还是后端内部结构。

---

## 5. 模块职责

### 5.1 detector.py

阶段一只实现 `TriePrefixDetector`。

职责：

- 从 batch 的 `input_ids` 中自动识别共同 prefix
- 输出 per-sample reuse relation：`reuse_batch_index -> provider_batch_index + prefix_len`
- 不处理 token 裁剪、RoPE、KV cache、label/logprob 对齐

阶段一采用 PrefixTrain_dev 风格的在线 Trie 思路：

1. 按 batch 顺序处理样本。
2. 当前样本只在已经插入 Trie 的历史样本中寻找最长可复用前缀。
3. 如果命中长度满足 `min_prefix_len`，且该前缀覆盖的历史样本数加当前样本达到 `min_group_size`，则记录一条 `PrefixReuseSpec`。
4. 当前样本随后插入 Trie，成为后续样本的潜在 provider。

该设计允许同一个 provider 面向不同 reuser 提供不同子序列复用，不再采用“最长前缀优先、非重叠 group”的简化方案。若某个样本本身是 reuser，它在后端构造出完整 logical KV 后仍可作为后续样本的 provider。

`PromptPrefixDetector` 后移到后续阶段。它适合 prompt 边界已知的 DPO/GRPO/SFT 场景，但不是阶段一 MVP 的关键路径。

### 5.2 planner.py

职责：

- 根据 detector 输出的 `PrefixReuseSpec` 生成每个样本的执行计划
- 决定 provider 样本和 reuse 样本的 query 保留范围
- 计算 `cu_seqlens_q`、`cu_seqlens_kv`、position offset
- 生成 `PrefixSharingBatchMeta`
- 生成 Prefix-Last Restore 计划

planner 是把“识别出的共同 prefix”转化为“可执行计划”的中心模块。

### 5.3 mapping.py

职责：

- 维护原始序列和裁剪后 packed 序列之间的坐标映射
- 处理 input、label、loss mask、response mask 的同步裁剪
- 处理 verl actor logprob 切片和 Prefix-Last Restore 的对齐

阶段一的关键规则：

- provider 样本保留完整输出
- reuse 样本保留 suffix 输出
- reuse 样本第一个 suffix token 的 logprob 由 `prefix_last_restore` 提供
- reuse 样本后续 suffix token 的 logprob 由 suffix query 输出提供

### 5.4 cache.py

职责：

- 管理当前 forward/backward 生命周期内的 prefix K/V
- 按 layer、provider、tp rank 隔离缓存
- 为 reuse 样本按自己的 `prefix_len` 从 provider K/V 中切片并构造 expanded KV

cache key 至少包含：

```python
(
    forward_id,
    micro_batch_id,
    layer_id,
    provider_batch_index,
    tp_rank,
)
```

cache 不负责：

- prefix 检测
- RoPE offset
- label/logprob 映射
- backend 选择
- 跨 micro-batch 持久化

cache tensor 不允许 `detach`。如需复制，只能在保持 autograd 图的前提下 `clone()`。

### 5.5 backends

阶段一 backend 目标是“CUDA 和 CANN 都能跑通”，不是最终性能。

统一接口：

```python
class PrefixAttentionBackend:
    def validate(self, config, model_config) -> None: ...
    def apply_rope(self, query, key, meta, packed_seq_params, rope_args): ...
    def build_kv(self, key, value, cache, meta): ...
    def attention(self, query, key, value, attention_mask, packed_seq_params, meta): ...
```

阶段一优先实现：

- `TorchReferenceBackend`：纯 PyTorch 正确性参考路径
- CUDA 可运行路径：先保证在 CUDA 上可用，不绑定 TE fused 能力
- CANN 可运行路径：先保证在 NPU 上可用，不把 CUDA TE 细节泄漏到语义层

高性能 TE、flash-attn、CANN 专用算子进入阶段二。

---

## 6. 阶段一数据流

### 6.1 preprocess

输入来自 verl Megatron actor micro-batch。

流程：

1. `TriePrefixDetector` 识别 prefix group
2. `planner` 选择 provider 并生成 `PrefixSharingBatchMeta`
3. `mapping` 同步处理 input、label、loss mask、response mask
4. provider 样本保留完整 `[prefix | suffix]`
5. reuse 样本裁剪 prefix，仅保留 suffix
6. 构造裁剪后的 packed input 和 `cu_seqlens_q`
7. 记录逻辑 expanded KV 长度和 `cu_seqlens_kv`
8. 把 meta 写入当前 `prefix_sharing_context`

### 6.2 attention

每层 Megatron `SelfAttention.forward()` 通过 patch 接入：

```
hidden_states
  -> QKV projection
  -> backend.apply_rope(query, key, meta)
  -> cache.store(provider prefix K/V)
  -> backend.build_kv(key, value, cache, meta)
  -> backend.attention(query, expanded_key, expanded_value, meta)
  -> output projection
```

注意：

- reuse query 长度和 expanded KV 长度可以不同
- suffix query 的 RoPE 逻辑位置从 `prefix_len` 开始
- expanded KV 的逻辑位置从 0 连续到原始序列长度
- attention mask 必须保证因果性等价于原始完整序列

### 6.3 postprocess / logprob

verl Megatron actor 当前 logprob 语义中，response 第一个 token 的 label 位于 prompt 最后一个输出位置。因此阶段一 postprocess 必须：

- 保留 provider 的完整 logits/log_probs
- 对 reuse 样本组装 suffix logits/log_probs
- 对 reuse 第一个 response token，从 provider prefix-last 输出恢复对应 logits/logprob
- 对 reuse 后续 response token，使用 suffix query 输出计算 logits/logprob
- 按原始 batch 顺序返回给 verl actor loss 逻辑

---

## 7. Patch 集成策略

阶段一必须对 verl 和 Megatron 保持非侵入式：

- 不直接修改 site-packages / submodule 源码
- 不 fork verl / Megatron 主代码
- 通过 monkey patch、wrapper、context manager 接入
- patch 层保持薄，核心逻辑放在 `core/` 和 `backends/`

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

- 幂等安装
- 完整卸载
- 版本和函数签名检查
- 安装失败回滚
- 约束校验失败时拒绝启动

未来向 verl / Megatron PR 时，优先把 patch 点迁移成正式 hook 或 config branch，`core/`、`metadata`、`planner`、`mapping`、`backend` 接口尽量保持不变。

---

## 8. 精度验收

阶段一采用三层精度验收。

### 8.1 语义层验收

验证 `core/` 在小 batch 上生成正确 metadata：

- group / provider 选择正确
- query 裁剪范围正确
- expanded KV 长度正确
- RoPE offset 正确
- label / loss mask / response mask 映射正确
- Prefix-Last Restore spec 指向正确 provider 和 prefix-last 位置

### 8.2 单模型前向验收

在小型 causal LM 或 Megatron 单层/小层数模型上比较：

- baseline 完整序列 forward
- prefix sharing patched forward

比较对象：

- provider 样本 logits/logprob
- reuse 样本第一个 response token logprob
- reuse 样本后续 response token logprob
- loss
- 关键参数梯度

验收条件：

- eval mode 或 train mode 关闭 dropout
- 固定 RNG
- fp32 使用较严格阈值
- bf16/fp16 使用符合 backend 数值误差的阈值

### 8.3 verl actor 链路验收

在 verl + Megatron actor logprob/update 主流程上验证：

- patch 启停不改变非 prefix sharing 路径行为
- 无 prefix group 时输出等价于 baseline
- 有 prefix group 时 logprob/loss 在目标区间内对齐
- backward 可完成，provider prefix 相关参数梯度能接收来自多个 reuse 样本的贡献
- 多 micro-batch 不发生 cache 污染

---

## 9. 阶段一实施计划

### 9.1 基础模块

1. 建立包结构 `prefix_sharing/`
2. 实现 `PrefixSharingConfig.validate()`
3. 实现 `PrefixSharingBatchMeta` 和 `PrefixLastRestoreSpec`
4. 实现 `TriePrefixDetector`
5. 实现 `planner.py`
6. 实现 `mapping.py`
7. 实现 `PrefixKVCache`

### 9.2 Reference Backend

1. 实现 `PrefixAttentionBackend` 接口
2. 实现 `TorchReferenceBackend`
3. 支持 q_len 与 kv_len 不同的 causal attention
4. 支持 suffix query RoPE offset
5. 支持 provider prefix K/V 注入

### 9.3 Megatron Patch

1. patch Megatron SelfAttention forward
2. 在 QKV projection 后接管 RoPE / KV injection / attention
3. 处理 packed THD no-padding 参数
4. 禁用阶段一不支持的 fused 路径
5. 确保 cache 生命周期按 layer 和 micro-batch 隔离

### 9.4 verl Patch

1. patch verl Megatron actor preprocess / postprocess / logprob 相关路径
2. 在 micro-batch 上生成和传递 `PrefixSharingBatchMeta`
3. 对齐 response label 和 logprob 切片
4. 实现 Prefix-Last Restore 的 logprob 组装
5. 保持 patch 可启停、可回滚

### 9.5 测试

1. detector / planner / mapping 单测
2. cache 生命周期单测
3. reference attention 数值单测
4. Prefix-Last Restore logprob 单测
5. Megatron 小模型前向/反向对齐测试
6. verl actor logprob/update smoke test

---

## 10. Roadmap

### 10.1 阶段一：verl + Megatron RL MVP

目标：打通当前最关键业务链路的最小正确闭环。

范围：

- `verl + Megatron` actor logprob / update
- Trie 自动 prefix 检测
- Prefix-Last Restore 精度方案
- 非侵入式 patch 集成
- PP=1、CP=1、EP 相关能力关闭或不启用
- CUDA / CANN 都能通过 reference 路径跑通

成功标准：

- 正确性测试通过
- small-scale RL actor 链路可运行
- 在合理精度假设下 logprob/loss/grad 对齐
- patch 可启停，不污染非 prefix sharing 路径

### 10.2 阶段二：业务落地能力

目标：补齐真实业务训练需要的并行策略、硬件解耦和性能后端，使方案具备业务落地条件。

重点：

- TP 完整验证
- CP 支持
- PP 支持
- EP 兼容性分析和必要适配
- DP / micro-batch / gradient accumulation 场景稳定性
- CUDA TE / flash-attn 后端优化
- CANN NPU 专用后端和算子适配
- backend capability 机制：每个后端显式声明支持的并行、fused、dtype、mask、packed 格式
- 性能策略：`min_prefix_len`、`min_group_size`、收益估计、自动 fallback

PromptPrefixDetector 可在本阶段或更后阶段加入，但它不是阶段二的核心目标。阶段二优先保证复杂并行和硬件后端能够服务真实业务。

### 10.3 阶段三：训练范式扩展

目标：把 prefix sharing 从当前 RL MVP 扩展成可独立用于并行训练的通用能力。

范围：

- `verl + FSDP` RL 路径
- standalone Megatron SFT
- standalone Megatron pretrain
- standalone FSDP SFT / pretrain
- full-prefix restore，用于完整序列 loss 场景
- PromptPrefixDetector，用于 prompt 边界明确的 SFT / DPO / GRPO 数据
- 面向 next-token prediction 的通用 API，而不是只服务 actor logprob

设计要求：

- Megatron 和 FSDP 的接入不能相互耦合
- RL、SFT、pretrain 共享 `core/` 语义层
- integration 层按训练框架拆分
- full-prefix restore 与 prefix-last restore 共享 metadata 体系

### 10.4 阶段四：上游 PR 与产品化

目标：将能力从外挂 patch 演进为可维护、可上游、可独立分发的正式方案。

方向：

- verl PR：提供 prefix sharing extension hook 或可选 config path
- Megatron PR：提供 attention / packed sequence / RoPE / output restore 的正式扩展点
- 独立插件：面向 Megatron 或 FSDP 的单独 package
- 稳定 API：config、metadata、backend、integration hook 文档化
- 兼容矩阵：模型、硬件、并行策略、dtype、kernel 后端
- benchmark：不同 prefix 重复率、序列长度、并行策略下的收益和开销

---

## 11. 设计边界

### 11.1 阶段一边界不是长期边界

阶段一的 PP=1、CP=1、非 fused、reference backend 等限制，只是 MVP 为了保证正确性和可调试性而设置的启动边界。长期方案必须支持真实业务中的 TP、CP、PP、EP、DP 组合，并支持 CUDA GPU 与 CANN NPU。

### 11.2 patch 是阶段性接入方式

阶段一必须通过 patch 保持对 verl 和 Megatron 非侵入。但 patch 不是长期架构目标。所有核心逻辑应沉淀到可迁移模块，未来可以迁移为：

- verl 正式 extension hook
- Megatron config branch
- Megatron/FSDP 独立插件
- 单独 pip package

### 11.3 最终实现不引入 boundary mode

最终开发方案只采用 Prefix-Last Restore 作为精度路径。历史上讨论过的其他方案保留在 `doc-design-history.md`，不进入阶段一或阶段二的实现目标。
