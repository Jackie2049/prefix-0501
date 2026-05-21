# Phase 2 方案设计历史记录

> **规则**: 所有 Phase 2 方案设计按时间倒序记录，最新在前。本文档记录尚未完全收敛的方案、讨论、备选路径和阶段性结论。待方案磋商出最终版本后，再抽取到 `design-final.md`。

---

## 2026-05-21 14:10 DP 并行策略详细设计初稿

### 一、定位

DP 是 Phase 2 并行策略支持的第一步。它本身不切分模型参数、attention head、sequence 或 pipeline stage，因此不应改变单 rank 内 prefix sharing 的数学语义。

DP 支持的核心不是改 attention 算法，而是把 Phase 1 的 runtime 生命周期从“单进程本地测试默认成立”提升为“真实多 DP rank、多 micro-batch、gradient accumulation 下仍然可追踪、可隔离、可回滚”。

### 二、当前结论

DP 下 prefix sharing 的 scope 定义为：

```text
scope = 当前 DP rank 内的当前 actor micro-batch
```

明确不做：

- 不跨 DP rank 检测 shared prefix。
- 不跨 DP rank 交换 provider prefix K/V。
- 不跨 forward / optimizer step 持久保存 K/V。
- 不把 DP group 纳入 `PrefixKVSlotId`，除非后续日志和调试证明需要。

理由：

- DP rank 之间处理的是不同数据 shard，跨 rank 找 prefix 会引入额外通信和训练语义不确定性。
- 当前收益目标来自同一 micro-batch 内 prompt / prefix 重复，local rank 内已经能捕获最常见场景。
- 跨 rank sharing 会让 provider 的 autograd graph 分布到另一个 DP rank，和 DDP / Megatron DP 梯度同步模型冲突，暂不进入 Phase 2.1。

### 三、需要支持的 DP 场景

Phase 2.1 先覆盖以下场景：

1. `dp_world_size=1`：保持 Phase 1 行为不变。
2. `dp_world_size>1`：每个 DP rank 独立运行 prefix detector、planner、store、restore。
3. gradient accumulation：同一 DP rank 上连续多个 micro-batch 累积梯度，store 必须按 micro-batch 隔离。
4. verl actor logprob / update：prefix sharing on/off 在每个 DP rank 内分别对齐 baseline。

暂不覆盖：

- 跨 DP rank prefix 发现。
- DP rank 间共享 provider activation。
- 分布式 sample reorder 后的跨 rank restore。

### 四、设计不变量

1. 每个 DP rank 的 `PrefixSharingRuntimeContext` 只服务当前 rank 当前 micro-batch。
2. `PrefixKVStore` 在 context 退出后必须 close，不允许跨 micro-batch 读取。
3. `forward_id` 和 `micro_batch_id` 必须足以区分 gradient accumulation 期间的多个 forward。
4. DP rank 不参与 provider lookup。`provider_idx_in_batch` 和 `reuse_idx_in_batch` 都是 local batch index。
5. Prefix-Last Restore 只从当前 rank 当前 micro-batch 的 provider logits 中取值。
6. 关闭 prefix sharing 时，verl / Megatron 原路径不应看到任何 runtime context 或 altered mask。

### 五、ParallelEnv 设计

DP 支持应以统一的 `ParallelEnv` 起步，即使 DP 本身不改变 attention 语义，也要把 rank 信息纳入日志和 validation。

建议新增：

```python
@dataclass(frozen=True)
class ParallelEnv:
    dp_rank: int = 0
    dp_world_size: int = 1
    tp_rank: int = 0
    tp_world_size: int = 1
    pp_rank: int = 0
    pp_world_size: int = 1
    virtual_pp_rank: int | None = None
    cp_rank: int = 0
    cp_world_size: int = 1
    ep_rank: int = 0
    ep_world_size: int = 1
    sequence_parallel: bool = False
```

DP 第一阶段只使用：

- `dp_rank`
- `dp_world_size`
- `tp_rank=0`
- `pp_rank=0`
- `cp_rank=0`

读取来源：

- 优先从 Megatron `parallel_state.get_data_parallel_rank()` / `get_data_parallel_world_size()` 读取。
- 如果 Megatron parallel state 未初始化，fallback 为单进程默认值。
- 不应直接依赖 torch distributed default group，因为 verl / Megatron 的 DP group 可能不是默认 group。

### 六、Runtime Context 设计

`PrefixSharingRuntimeContext` 建议扩展：

```python
@dataclass
class PrefixSharingRuntimeContext:
    meta: PrefixSharingBatchMeta
    store: PrefixKVStore
    backend: Any | None = None
    kept_position_ids: Any | None = None
    restore_positions: list[Any] = field(default_factory=list)
    parallel_env: ParallelEnv = field(default_factory=ParallelEnv)
    forward_id: int | None = None
    micro_batch_id: int | None = None
    stats: PrefixSharingStats | None = None
```

DP 阶段要求：

- `parallel_env.dp_rank` 进入 stats / debug log。
- `forward_id`、`micro_batch_id` 与 `meta.forward_id`、`meta.micro_batch_id` 对齐。
- context manager 退出时 close store。
- nested context 只允许在同线程 / 同任务内按 stack 恢复，不能泄漏到下一个 micro-batch。

### 七、forward_id / micro_batch_id 来源

Phase 1 中 planner 可自动生成 id。DP 阶段需要更明确：

1. 如果 verl actor helper 能拿到外部 batch id / micro-batch id，优先传入。
2. 如果拿不到，则由 prefix-sharing integration 生成 rank-local monotonic id。
3. id 只要求 rank-local 唯一，不要求跨 DP rank 全局唯一。
4. 日志展示时使用 `(dp_rank, forward_id, micro_batch_id)` 作为 trace key。

建议：

```text
trace_key = dp{dp_rank}/fw{forward_id}/mb{micro_batch_id}
```

### 八、Store Key 是否需要 dp_rank

初稿结论：`PrefixKVSlotId` 暂不需要加入 `dp_rank`。

理由：

- 每个 DP rank 是独立进程，`PrefixKVStore` 不跨进程共享。
- store 生命周期被 context 限定在单 micro-batch 内。
- 当前 key 已包含 `forward_id / micro_batch_id / layer_id / sample_idx_in_batch / tp_rank`。

保留条件：

- 如果未来有同进程多 DP rank 或 worker multiplexing，再把 `dp_rank` 纳入 slot id。
- `dp_rank` 先进入 `ParallelEnv` 和 `stats`，用于日志和 debugging。

### 九、DP 下的 Prefix-Last Restore

DP 不改变 restore 语义：

```text
provider logits: 当前 DP rank 当前 micro-batch 的 provider row
reuse label:     当前 DP rank 当前 micro-batch 的 reuser first suffix label
output slot:     当前 DP rank 当前 micro-batch 的 reuser dense/logprob slot
```

关键要求：

- restore 不做 DP all-gather。
- restore 不依赖全局 batch index。
- restore 后 loss 在本 rank 参与正常 DP 梯度同步。
- provider prefix 的梯度来自本 rank 内多个 reuser，随后由 DP 做参数梯度 all-reduce。

### 十、DP 下的 Autograd 与梯度同步

prefix sharing 增加的是 rank-local autograd graph 内的多 consumer 路径：

```text
provider prefix output
  -> provider own loss
  -> reuser first suffix restore loss
  -> reuser suffix attention path consuming provider K/V
```

DP 梯度同步仍由 Megatron / DDP 在 backward 后处理。prefix-sharing 不应自行调用 DP all-reduce。

需要验证：

- provider prefix 相关参数在本 rank 内收到 reuser loss 贡献。
- DP all-reduce 后 on/off loss 和 grad 与 baseline 对齐。
- gradient accumulation 多个 micro-batch 之间 graph 不串联。

### 十一、Capability 与配置

新增或扩展 capability：

```python
supports_data_parallel: bool = True
requires_rank_local_reuse: bool = True
supports_cross_dp_reuse: bool = False
```

DP 阶段 validation：

- `dp_world_size >= 1` 均允许。
- `tp_world_size != 1` 仍按后续 TP 阶段规则处理。
- `pp_world_size != 1`、`cp_world_size != 1` 仍 fail fast。
- 如果用户配置要求 `cross_dp_reuse=True`，直接报错。

### 十二、Stats 与日志

DP 支持必须补可观测性，否则真实多 rank 环境很难定位问题。

建议 stats：

```python
@dataclass
class PrefixSharingStats:
    trace_key: str
    dp_rank: int
    dp_world_size: int
    batch_size: int
    reuse_count: int
    provider_count: int
    saved_tokens_q: int
    expanded_tokens_kv: int
    fallback_reason: str | None = None
```

日志策略：

- 默认只在 debug 下输出 per micro-batch stats。
- fail-fast 报错必须包含 `trace_key` 和 parallel env。
- 后续性能评估可聚合各 DP rank 的 stats，但不作为 DP 支持第一步的 blocker。

### 十三、测试计划

#### 单元测试

1. `ParallelEnv` 无 Megatron 环境时返回单进程默认值。
2. `ParallelEnv` mock Megatron parallel state，返回 DP rank/world size。
3. `PrefixSharingRuntimeContext` 携带 `parallel_env` 并在退出后 close store。
4. 多 micro-batch 创建多个 context，验证 store 不共享。
5. `PrefixKVSlotId` 不含 dp_rank 的情况下，不同 context 仍隔离。

#### 集成测试

1. 模拟 `dp_rank=0/1` 两个 rank-local micro-batch，分别 plan / trim / restore，验证 local index 不串。
2. gradient accumulation 模拟：连续 N 个 micro-batch，每个 micro-batch 的 restore 和 store 独立。
3. prefix sharing disabled 时 context 不创建，batch 不改变。

#### 真实环境 optional 测试

1. `DP=2, TP=1, PP=1, CP=1` actor logprob smoke。
2. `DP=2` actor update smoke。
3. prefix sharing on/off loss 对齐。
4. provider prefix 参数梯度对齐。

### 十四、开发拆分

建议提交顺序：

1. `[feat] 增加 parallel env 描述`
2. `[feat] runtime context 携带 parallel env 和 trace id`
3. `[test] 覆盖 DP micro-batch store 隔离`
4. `[test] 增加 DP rank-local restore 语义测试`
5. `[doc] 收敛 DP 并行支持设计`

### 十五、待确认问题

1. verl Megatron actor 当前是否能直接提供 micro-batch id，还是需要 prefix-sharing 自己生成。
2. DP rank 信息应从 Megatron `parallel_state` 读取，还是 verl worker 已有更稳定的 rank source。
3. stats 是否需要在第一阶段暴露给用户配置，还是只作为 internal debug。
4. prefix sharing disabled / no-sharing fallback 是否需要记录 rank-local stats。

### 当前建议

DP 第一版只做 rank-local 支持，不做跨 rank 优化。实现重点放在 `ParallelEnv`、context 生命周期、micro-batch id、stats 和测试上。这样可以用最小代码改动建立 Phase 2 并行能力的基础框架，为 TP / PP / CP 后续接入复用同一套 runtime abstraction。

### 本轮确认

已确认 DP 第一版需要做工程适配，但不需要改 prefix sharing 核心算法：

- 增加 DP runtime 信息，用于 debug、日志和后续 validation。
- 补充 DP rank-local 模拟测试。
- 补充 micro-batch store 生命周期隔离测试。
- 暂不做跨 DP rank prefix 发现、K/V 通信或 activation 共享。

---

## 2026-05-21 13:55 Phase 2 总体设计初稿

## 1. 阶段定位

Phase 2 的目标是让 prefix sharing 从 Phase 1 的语义 MVP 进入真实业务训练可落地状态。

Phase 1 已经补齐：

- per-sample reuse relation。
- One-Forward + KV Injection + Prefix-Last Restore。
- `core / integrations / backends` 三层边界。
- verl Megatron actor helper。
- Megatron attention runtime hook。
- torch reference backend。

Phase 2 不再重新定义核心语义，而是在真实训练约束下扩展能力边界：

1. 并行策略支持。
2. backend 后端解耦。
3. flash-attn / Transformer Engine / CANN 等高性能实现。
4. KV 以外的全量 prefix activation reuse。

---

## 2. 阶段原则

### 2.1 Correctness first

任何新增并行能力或 backend，都必须先与 baseline full forward 对齐：

- logits / logprob。
- Prefix-Last Restore 的第一个 suffix token logprob。
- loss。
- provider prefix 相关参数梯度。
- 多个 reuser 的梯度累积。

### 2.2 Capability-driven runtime

Phase 1 的 config validation 主要是硬编码限制。Phase 2 需要改为 capability-driven：

- `PrefixSharingConfig` 只负责用户意图和基础字段合法性。
- backend 声明自身支持的 device、dtype、layout、parallel、fused path。
- integration 读取 Megatron / verl runtime parallel env。
- runtime validation 联合判断是否启用、fail fast 或 fallback。

### 2.3 先 reference，后优化

高性能 backend 不应先行。所有 flash-attn / TE / CANN 实现都必须以 torch reference path 为正确性基线。

### 2.4 不跨 forward 持久化

Phase 2 默认仍只在当前 forward / micro-batch 生命周期内复用 prefix，不做跨 forward 或跨参数版本的持久 KV / activation cache。

---

## 3. 工作主线

### 3.1 并行策略支持

详细方案见 `parallel-plan.md`。

范围：

- DP / micro-batch / gradient accumulation。
- TP local K/V shard。
- SP layout 验证。
- PP / VPP stage-local store。
- CP local/global position 与 KV exchange。
- EP / MoE attention-only 兼容。

第一批优先级：

1. 新增 parallel env 描述。
2. 扩展 runtime context 和 store key。
3. 支持 DP / micro-batch 生命周期隔离。
4. 支持 TP local prefix K/V shard。
5. 补 vocab-parallel Prefix-Last Restore 真实路径验证。

### 3.2 Backend 后端解耦

目标：integration 层不直接绑定具体 backend 行为。

需要扩展 `BackendCapabilities`：

- device：CPU / CUDA / CANN。
- dtype：fp32 / fp16 / bf16。
- layout：dense / packed THD。
- attention shape：`q_len == kv_len` / `q_len != kv_len`。
- head relation：MHA / GQA / MQA。
- parallel：TP / SP / PP / CP / EP。
- fused path：fused RoPE、fused QKV RoPE、flash-attn、TE attention。
- restore：Prefix-Last Restore、后续 full-prefix restore。

建议新增 runtime validation：

```python
validate_prefix_sharing_runtime(config, model_config, backend, parallel_env)
```

返回结果应能区分：

- supported：可启用。
- fail_fast：用户显式要求 prefix sharing，但当前组合不安全。
- fallback_to_baseline：业务稳定期可配置启用，开发期默认不用静默 fallback。

### 3.3 高性能实现

目标：在 reference path 正确后减少 prefix sharing 自身额外开销。

候选后端：

- CUDA torch reference path：先验证设备侧 tensor / autograd 行为。
- flash-attn backend：重点验证 packed THD、`q_len != kv_len`、causal offset mask、GQA。
- Transformer Engine backend：重点验证 TE fused path 与 RoPE / projection 接线。
- CANN NPU backend：独立 capability，不复用 CUDA 假设。

性能策略：

- `min_prefix_len`。
- `min_group_size`。
- saved-token 阈值。
- build_kv / cat 开销估计。
- batch sharing ratio。
- 自动 fallback reason。
- profiling stats。

### 3.4 Prefix Activation Reuse

目标：从 prefix K/V sharing 扩展到更广义的 prefix activation sharing。

探索对象：

- provider prefix layer hidden states。
- attention 中间激活。
- MLP 中间激活。
- prefix-last logits。
- 后续 full-prefix restore 所需的 prefix outputs。

硬约束：

- 不允许 `detach` 切断 provider prefix activation 的梯度路径。
- 必须兼容 activation checkpointing / recompute。
- 必须明确 TP / SP / PP / CP 下 activation 分片、跨 stage 传递和生命周期。
- EP / MoE 下默认禁止跨 expert 复用 MLP activation。
- 先以显式 feature flag 进入，不作为 Phase 2 默认路径。

Phase 2 对 activation reuse 的建议范围：

1. 先完成设计文档和最小 PoC。
2. 只在 TP=1 / PP=1 / CP=1 的小模型上验证 autograd。
3. 不和 flash-attn / CANN 同时推进。
4. 不进入默认业务路径。

---

## 4. 里程碑

### Milestone 1：并行 runtime 基础

- `ParallelEnv`。
- capability-driven validation。
- runtime context 扩展。
- store key 扩展。
- DP / micro-batch 生命周期测试。

### Milestone 2：TP 可用

- TP local K/V shard。
- TP=2 tiny attention forward / backward 对齐。
- GQA local head shard 验证。
- vocab-parallel restore 验证。

### Milestone 3：SP / PP 可用

- SP packed layout 审计。
- PP stage-local store。
- 最后 stage restore。
- VPP context 隔离。

### Milestone 4：CP / EP 兼容

- CP fail-fast 到 CP-aware plan。
- CP 异 rank prefix-last restore。
- EP attention-only smoke。

### Milestone 5：Backend 与 activation reuse 专项

- flash-attn / TE / CANN capability。
- performance stats 和 fallback。
- activation reuse PoC。

---

## 5. 文档关系

- `../roadmap.md`：全局阶段顺序和摘要。
- `design-history.md`：Phase 2 方案设计历史。
- `design-final.md`：Phase 2 最终方案，待方案磋商收敛后创建。
- `parallel-plan.md`：Phase 2.1 并行策略详细方案。
