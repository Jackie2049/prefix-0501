# Phase 2 并行策略支持方案

> 本文档聚焦阶段二第一步：让 prefix sharing 从 Phase 1 的 `PP=1 / CP=1 / 非 fused` MVP，演进为可在真实 `verl + Megatron` 训练并行配置中稳定启用的能力。高性能 backend、flash-attn / TE / CANN 优化，以及 KV 以外的全量 activation reuse 属于阶段二后续专项，本文只定义它们与并行策略相关的接口预留。

---

## 1. 阶段目标

阶段二总目标包括四条主线：

1. 并行策略支持：TP、DP、micro-batch、gradient accumulation、PP、CP、SP、EP。
2. backend 后端解耦：以 capability 明确声明每个 backend 支持的设备、layout、并行、dtype 和 fused path。
3. 高性能实现：flash-attn、Transformer Engine、CANN NPU 专用后端。
4. KV 以外的全量 activation reuse：prefix hidden states、attention/MLP 中间激活、prefix-last logits 等。

阶段二第一步只做并行策略支持。成功标准不是“所有并行一次性全开”，而是建立清晰的能力矩阵、失败边界、测试阶梯和最小可用路径。

---

## 2. 当前 Phase 1 基线

现有代码的关键入口：

- `core/planner.py`：生成 `PrefixSharingBatchMeta`，描述每个样本的 reuse relation、Q/KV 长度、packed seqlens、position offset 和 Prefix-Last Restore。
- `core/prefix_store.py`：按 `forward_id / micro_batch_id / layer_id / sample_idx_in_batch / tp_rank` 保存当前 forward 内 logical K/V。
- `integrations/verl_mcore.py`：在 verl Megatron actor micro-batch 侧生成 plan，裁剪 reuser attention mask，传入 `kept_position_ids` 和 restore dense slot。
- `integrations/megatron_runtime.py`：在 Megatron packed THD QKV 后、标准 RoPE 前，按原始 position id 重新 apply RoPE，随后 build expanded KV、执行 backend attention、接原 output projection。
- `backends/torch_ref.py`：reference backend，支持 packed THD、`q_len != kv_len`、GQA repeat 和 transitive reuse store。

当前显式限制：

- `pipeline_model_parallel_size == 1`
- `context_parallel_size == 1`
- `apply_rope_fusion == False`
- `fused_single_qkv_rope == False`
- text-only causal LM
- verl Megatron actor path，且要求 `megatron.use_remove_padding=True`

阶段二第一步的核心任务就是逐步拆掉这些限制，但每拆一个限制都必须有 capability、fallback、测试和精度验收。

---

## 3. 总体原则

### 3.1 并行能力必须显式声明

`PrefixSharingConfig.validate()` 不能继续只写死 Phase 1 约束。阶段二应引入 `ParallelCapabilities` / 扩展 `BackendCapabilities`，由 integration + backend 联合决定是否允许启用：

- `supports_tensor_parallel`
- `supports_sequence_parallel`
- `supports_data_parallel`
- `supports_pipeline_parallel`
- `supports_virtual_pipeline_parallel`
- `supports_context_parallel`
- `supports_expert_parallel`
- `supports_remove_padding_thd`
- `supports_gqa`
- `supports_fused_rope`
- `supports_flash_attention`

不满足时必须 fail fast 或显式 fallback，不能静默走错语义。

### 3.2 reuse relation 仍以 local micro-batch 为边界

阶段二第一步不做跨 DP rank、跨 pipeline micro-batch、跨 forward 的 prefix sharing。所有 reuse relation 仍只在当前 rank 的当前 actor micro-batch 内检测和执行。

理由：

- DP rank 之间没有相同 batch 语义，不应通信寻找 prefix。
- PP stage 之间只传递 activation，不应跨 micro-batch 复用 store。
- 跨 forward 持久 KV cache 会引入生命周期、梯度和参数版本问题，属于后续专项。

### 3.3 correctness 优先于性能

每支持一种并行，必须先用 reference backend 对齐：

- forward logits / logprob
- loss
- provider prefix-last restore
- reuser suffix logprob
- 关键参数梯度
- prefix provider 梯度是否累积来自多个 reuser

只有 reference path 对齐后，才接 flash-attn / TE / CANN 等高性能 backend。

---

## 4. 并行策略分解

### 4.1 DP / micro-batch / gradient accumulation

优先级：P0

结论：DP 本身不改变单 rank 内 attention 语义，是最先应该稳定支持的并行维度。

需要做：

1. 明确 `forward_id` / `micro_batch_id` 的生成来源，不能只依赖本地 itertools 默认值。
2. 在 verl actor 每个 micro-batch 进入 context 时绑定可追踪 id。
3. store 生命周期严格限定在一个 forward context 内，退出 context 后必须 close。
4. gradient accumulation 下多个 micro-batch 不能共享 store，但梯度正常由 autograd 累积。
5. 日志中输出 DP rank、micro-batch id、sharing stats。

测试：

- 单进程模拟多个 micro-batch，验证 store 隔离。
- 多次 gradient accumulation，验证无跨 micro-batch 读写。
- DP=2 small actor smoke，比较 prefix sharing on/off 的 loss/grad。

验收：

- DP rank 内结果对齐 baseline。
- 不发生跨 micro-batch provider lookup。
- 关闭 prefix sharing 后完全回到原路径。

### 4.2 TP

优先级：P0

结论：TP 是最重要的模型并行支持。当前 store key 已包含 `tp_rank`，这为 TP 支持提供了基础，但仍需验证 QKV projection、GQA、output projection 和 vocab-parallel restore。

需要做：

1. 从 Megatron `parallel_state.get_tensor_model_parallel_rank()`、`get_tensor_model_parallel_world_size()` 读取 TP 信息并写入 runtime context。
2. 保持每个 TP rank 只保存本 rank 的 local K/V shard，不做跨 TP gather。
3. `backend.build_kv()` 以 local head shard 为单位拼 prefix K/V。
4. GQA/MQA 下确认 `num_query_groups / tensor_model_parallel_size` 后的 local KV head 数与 local Q head 数关系。
5. Prefix-Last Restore 在 vocab-parallel logits 上调用 verl / Megatron 原生 gather logprob 函数，不能先 all-gather logits。
6. capability 中声明 backend 是否支持 TP local shard。

测试：

- TP=1 当前测试保持通过。
- TP=2 tiny Megatron attention forward 对齐。
- TP=2 backward 对齐，检查 provider prefix K/V 路径梯度。
- GQA 配置下 packed THD reference 对齐。
- vocab-parallel restore 测试：provider prefix-last logits 来自 provider row，label 来自 reuser first suffix。

验收：

- 每个 TP rank store slot 独立。
- 不引入额外全量 K/V all-gather。
- loss / grad 与 baseline 在允许误差内对齐。

### 4.3 SP

优先级：P1

结论：Megatron sequence parallel 通常与 TP 配套，主要影响 layernorm / dropout / linear 通信边界，不应改变 attention 内 packed THD 的 logical token 坐标；但需要验证 hidden_states 进入 attention 前后的 scatter/gather 时机。

需要做：

1. 明确 prefix sharing hook 所在位置看到的是完整 packed token 流，还是 SP 切分后的 token shard。
2. 如果 attention 前是 local sequence shard，则 `PrefixSharingBatchMeta` 必须下沉为 rank-local meta，不能使用全局 cu_seqlens。
3. 如果 attention 前 Megatron 已经聚合为 local rank 可完整处理的 packed THD，则沿用 TP 方案。
4. capability 中区分 `supports_sequence_parallel_with_global_packed_thd` 和 `supports_sequence_parallel_local_seq_shard`。

测试：

- TP=2 + SP=True tiny attention 对齐。
- 检查 packed first dimension 与 `sum(meta.kept_lengths_q)` 是否一致。
- 梯度对齐。

验收：

- 不假设 SP 总是无影响。
- 对实际 Megatron layout 有显式断言和错误信息。

### 4.4 PP / VPP

优先级：P1

结论：PP 不改变单层 attention 的局部语义，但改变 layer 分布、micro-batch 调度和 activation 传递。KV store 必须在每个 pipeline stage 本地维护，不能跨 stage 共享 tensor。

需要做：

1. `PrefixKVSlotId` 增加或派生 `pp_rank` / `virtual_pp_rank` 维度，避免不同 stage layer id 冲突。
2. runtime context 随 pipeline micro-batch 在每个 stage 正确进入和退出。
3. 每个 stage 只为本 stage 的 attention layers 存 K/V。
4. Prefix-Last Restore 只发生在最后 pipeline stage，因为 logits/logprobs 只在最后 stage 可见。
5. 中间 stage 不做 restore，但必须保持 reuser suffix activation 的 packed layout 与 downstream stage 一致。
6. VPP 下 layer id 与 virtual stage id 必须进入 tracing 日志。

测试：

- PP=2 tiny model forward 对齐。
- PP=2 backward 对齐。
- PP=2 + gradient accumulation，验证多个 in-flight micro-batch 不串 store。
- VPP 最小配置 smoke。

验收：

- 每个 PP stage store 独立。
- 最后 stage restore 正确。
- 多个 pipeline micro-batch 并发时 context 不串线。

### 4.5 CP

优先级：P2

结论：CP 是高风险项。Context parallel 将 sequence 维度切到多个 rank，attention 通常需要 ring / all-to-all 交换 KV。Prefix sharing 的 “provider prefix K/V + reuser suffix K/V” 会直接影响 CP 通信协议和 causal mask。

需要做：

1. 先只做 CP capability 拒绝和错误信息精化。
2. 阅读 Megatron CP attention 通信路径，确认 CP rank 持有的 token shard 与 packed THD cu_seqlens 的关系。
3. 设计 CP-local `PrefixSharingBatchMeta`：每个 CP rank 只描述本 rank 的 local Q/KV shard，同时保留 global logical position。
4. build_kv 不能简单 `torch.cat(provider_prefix, suffix)`，需要与 CP KV exchange 协议一致。
5. RoPE position 必须使用 global position，而不是 CP local offset。
6. Prefix-Last Restore 如果 provider prefix-last 不在本 CP rank，需要定义跨 CP rank gather 方案。

测试：

- CP=2 no sharing baseline smoke。
- CP=2 sharing forward 对齐。
- provider prefix-last 位于同 rank / 异 rank 两种 restore 场景。
- backward 对齐。

验收：

- CP 支持前默认拒绝启用。
- 支持后必须覆盖异 rank restore。
- 不破坏 Megatron 原 CP 通信调度。

### 4.6 EP / MoE

优先级：P2

结论：EP 对 attention KV sharing 影响间接，对 MLP/MoE 层的 activation reuse 影响直接。阶段二第一步只做兼容性分析和 gate，不做 activation reuse 下的 MoE 跨 expert 共享。

需要做：

1. attention-only KV injection 在 MoE 模型中原则上可工作，因为 attention 层位于 MoE MLP 前。
2. 检查 EP + TP group 下 attention TP group 是否仍按普通 TP 读取。
3. 对 activation reuse 明确禁止跨 expert 复用 MLP 中间激活。
4. capability 中加入 `supports_expert_parallel_attention_only` 与 `supports_expert_parallel_activation_reuse`。

测试：

- EP=2 MoE tiny model attention-only smoke。
- 确认 router / token dispatch 不读取 prefix store。

验收：

- KV sharing 不影响 MoE router 语义。
- activation reuse 在 EP 下默认关闭。

---

## 5. 代码改造计划

### 5.1 新增并行环境描述

新增模块：

```text
prefix_sharing/integrations/parallel_env.py
```

建议结构：

```python
@dataclass(frozen=True)
class ParallelEnv:
    dp_rank: int
    dp_world_size: int
    tp_rank: int
    tp_world_size: int
    pp_rank: int
    pp_world_size: int
    virtual_pp_rank: int | None
    cp_rank: int
    cp_world_size: int
    ep_rank: int
    ep_world_size: int
    sequence_parallel: bool
```

职责：

- 从 Megatron `parallel_state` 和 model config 读取并行信息。
- 在无 Megatron 环境的本地测试中提供默认单进程 env。
- 作为 config validation、backend capability 和 store key 的共同输入。

### 5.2 扩展 runtime context

`PrefixSharingRuntimeContext` 增加：

- `parallel_env`
- `forward_id`
- `micro_batch_id`
- `stage_id`
- `stats`

context 必须由 verl actor micro-batch helper 创建，不能让 attention hook 自己猜测 micro-batch 生命周期。

### 5.3 扩展 store key

`PrefixKVSlotId` 当前已有 `tp_rank`，阶段二建议扩展为：

```python
PrefixKVSlotId(
    forward_id,
    micro_batch_id,
    layer_id,
    sample_idx_in_batch,
    tp_rank,
    pp_rank=0,
    virtual_pp_rank=None,
    cp_rank=0,
)
```

TP 下保存 local head shard；PP 下按 stage 隔离；CP 支持前 `cp_rank` 保持 0 并由 config 拒绝 CP>1。

### 5.4 capability-driven validation

当前 `PrefixSharingConfig.validate()` 应拆成两层：

1. config 自身合法性：字段、阈值、boundary strategy。
2. runtime capability 合法性：并行、backend、layout、dtype、fused path。

建议新增：

```python
validate_prefix_sharing_runtime(config, model_config, backend, parallel_env)
```

它返回明确结论：

- supported
- fallback_to_baseline
- fail_fast

阶段二开发期默认 fail fast，业务稳定后再允许配置化 fallback。

### 5.5 测试矩阵

本地无分布式依赖时：

- unit：ParallelEnv 默认值、capability validation、store key 隔离。
- integrated optional：torch distributed tiny TP/PP/CP smoke。
- system optional：verl Megatron actor logprob/update smoke。

真实环境矩阵：

| 阶段 | 配置 | 目标 |
| --- | --- | --- |
| P0-A | DP=1 TP=1 PP=1 CP=1 | 保持 Phase 1 基线 |
| P0-B | DP=2 TP=1 PP=1 CP=1 | DP/micro-batch 稳定性 |
| P0-C | DP=1 TP=2 PP=1 CP=1 | TP local K/V shard |
| P1-A | DP=1 TP=2 SP=True | SP layout 验证 |
| P1-B | DP=1 TP=1 PP=2 CP=1 | PP stage store 隔离 |
| P1-C | PP=2 VPP=2 | VPP micro-batch/context 隔离 |
| P2-A | CP=2 | CP 设计验证 |
| P2-B | EP=2 MoE | attention-only 兼容 |

---

## 6. 开发优先级

### P0：必须先完成

1. 新增 `ParallelEnv` 和 capability validation。
2. DP / micro-batch / gradient accumulation 生命周期测试。
3. TP local K/V shard 支持与测试。
4. vocab-parallel Prefix-Last Restore 真实路径验证。
5. 日志和 stats：reuse count、saved tokens、fallback reason、parallel env。

### P1：第二批

1. SP 真实 layout 审计和支持。
2. PP stage-local store 与最后 stage restore。
3. PP 多 micro-batch in-flight context 隔离。
4. VPP smoke。

### P2：第三批

1. CP 默认拒绝改为设计实现。
2. CP local/global position 与异 rank restore。
3. EP attention-only 兼容性。

### P3：进入后端和 activation reuse

1. flash-attn / TE backend capability。
2. CANN backend capability。
3. activation reuse feature flag 与 PP/SP/CP 生命周期设计。

---

## 7. 建议的第一批提交拆分

1. `[feat] 增加并行环境描述与 capability 校验`
2. `[test] 覆盖 DP micro-batch store 生命周期`
3. `[feat] 支持 TP local prefix KV store`
4. `[test] 增加 TP tiny attention 对齐测试`
5. `[doc] 记录阶段二并行策略支持矩阵`

每个提交都必须能独立运行本地非分布式测试；依赖 torch / verl / GPU 的测试放到 optional，并在提交说明中明确本地 skip 原因。
