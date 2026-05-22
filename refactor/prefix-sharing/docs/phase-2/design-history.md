# Phase 2 方案设计历史记录

> **规则**: 所有 Phase 2 方案设计按时间倒序记录，最新在前。本文档记录尚未完全收敛的方案、讨论、备选路径和阶段性结论。待方案磋商出最终版本后，再抽取到 `design-final.md`。

---

## 2026-05-21 16:05 基于 uid 的 Prefix Group DP 负载均衡设计

### 一、决策摘要

Phase 2 的 DP 负载均衡采用 **prefix group locality first** 策略：

```text
先保证同一 prefix group 的样本被分配到同一个 DP rank；
再在 group 粒度上平衡各 DP rank 的 workload。
```

默认使用 verl `DataProto.non_tensor_batch["uid"]` 作为 `prefix_group_id`。

原因：

- 当前 verl PPO / GRPO 已经为每个原始 prompt 生成 `uid`。
- `DataProto.repeat(..., interleave=True)` 会把同一个 prompt 的 `uid` 复制到 n 条 rollout response 上。
- GRPO advantage 本身也用 `uid` 做同 prompt 多 response 分组。
- 对 prefix sharing 来说，同一个 prompt 的多条 response 正是最稳定的共享前缀来源。

因此当前普通 RL 场景下：

```text
uid == prefix_group_id == prompt group id
```

未来接入 step / tree 模式时，`uid` 的语义可以扩展为公共祖先 id：

```text
step mode: uid = step_root_id / shared_state_id
tree mode: uid = tree_root_id / common_ancestor_id
```

verl 和 prefix-sharing 不需要理解 tree 内部结构，只需要知道同一个 `uid` 的样本属于同一个前缀复用语义组。

### 二、目标与非目标

#### 目标

1. 避免 verl 原有 sample-level token balance 把同一 prompt / tree / step group 拆散到不同 DP rank。
2. 保持同 prefix group 样本的 DP rank locality，提高 rank-local micro-batch 内 prefix reuse ratio。
3. 在 group locality 前提下，尽量平衡各 DP rank 的训练 workload。
4. 只做样本重排，不改变 tensor 内容、label、loss mask、reward、advantage、old logprob。
5. 控制 verl / Megatron 修改面：核心 partitioner 放在 `prefix_sharing` 包，verl 只保留薄调用点。

#### 非目标

1. 不做跨 DP rank K/V 或 activation 共享。
2. 不做跨 micro-batch activation 共享。
3. 不拆单条 sequence。
4. 第一版不强依赖 PrefixTrain_dev 的跨 micro-batch memory manager。
5. 第一版不要求 verl 原生支持 tree 数据结构，只要求 per-sample metadata 能携带 group id。

### 三、为什么 verl 原 token balance 不够

verl 当前 `_balance_batch()` 的逻辑是 sample-level：

```text
sample -> sequence length workload -> Karmarkar-Karp -> DP partitions -> batch.reorder()
```

它优化的是 dense forward 下每个 DP rank 的 token workload，但它不知道哪些样本共享前缀。

在 prefix sharing 场景中，这会带来风险：

```text
同一个 prompt 的 n 条 response
  -> 被 sample-level balance 分散到多个 DP rank
  -> 每个 DP rank 内只剩少量同前缀样本
  -> rank-local prefix sharing reuse ratio 下降
```

所以 prefix sharing 下 DP balance 的基本单位不应是 sample，而应是 prefix group。

### 四、数据模型

新增 prefix-sharing 内部抽象：

```python
@dataclass(frozen=True)
class PrefixGroup:
    group_id: str
    sample_indices: tuple[int, ...]
    original_tokens: int
    estimated_compute_tokens: int
    reusable_prefix_tokens: int
```

```python
@dataclass(frozen=True)
class PrefixGroupPartition:
    dp_rank_to_indices: tuple[tuple[int, ...], ...]
    dp_rank_to_group_ids: tuple[tuple[str, ...], ...]
    group_workloads: dict[str, int]
    fallback_reason: str | None = None
```

默认字段来源：

```text
group_key = "uid"
group_ids = data.non_tensor_batch["uid"]
```

后续可配置：

```yaml
prefix_sharing:
  dp_balance:
    enabled: true
    group_key: uid
    workload: prefix_estimate
    fallback_to_seqlen_balance: true
```

### 五、Workload 估算

DP group balance 直接使用 prefix-aware workload，不再引入 dense workload 作为中间阶段。

对每个 group 内的 token 序列估算 prefix sharing 后真实 compute tokens：

```text
compute_tokens = estimate_incremental_prefix_compute_tokens(group_token_ids)
reusable_prefix_tokens = original_tokens - compute_tokens
group_workload = f(compute_tokens)
```

`estimate_incremental_prefix_compute_tokens()` 的语义：

```text
只建模当前 DP 调度组内、rank-local、按现有样本顺序发生的增量前缀复用。
每条样本找到和此前样本的最长已存在 prefix。
该样本新增计算量 = len(sample) - matched_prefix_len。
group workload = 所有样本新增计算量之和。
```

这里只保留 prefix-aware workload 的工程语义，不继承 PrefixTrain_dev / Aceso runtime 的函数命名、接口或实现细节。

输入 token 序列优先来源：

```text
batch["input_ids"] + batch["attention_mask"]
```

估算时只使用 attention mask 标记的有效 token，避免 padding 影响 prefix match。

对于同一个 `uid` 内的样本顺序，第一版使用 batch 当前顺序，保证实现简单且和实际训练 batch order 一致。后续如果要进一步提高 prefix reuse，可单独讨论 group 内重排策略；当前 DP balance 不改变 group 内样本相对顺序。

dense sequence-length workload 只作为 fallback：

```text
prefix-aware estimate 不可用
  -> 如果 fallback_to_seqlen_balance=True，使用 verl 原 seqlen balance
  -> 否则不重排并记录 fallback_reason
```

它不再作为 prefix-sharing DP balance 的阶段性目标。

### 六、DP 分配算法

输入：

```text
DataProto batch
dp_size
group_key = "uid"
```

流程：

1. 读取 `group_ids = batch.non_tensor_batch[group_key]`。
2. 按 `group_id` 收集 sample indices。
3. 为每个 group 计算 workload。
4. 使用 Karmarkar-Karp / LDM 将 group 分配给 `dp_size` 个分区。
5. 每个分区内按 workload 做稳定排序，减少 PP bubble。
6. 展开为全局 sample index：

```text
global_idx = concat(dp0_sample_indices, dp1_sample_indices, ...)
```

7. 调用 `batch.reorder(global_idx)`。
8. verl 原有 dispatch 继续按重排后的连续区间切给各 DP rank。

关键不变量：

```text
同一个 group_id 的 sample indices 只出现在一个 DP partition 内。
```

如果某个 group 大到超过单个 DP rank 理想 workload，第一版仍保持不拆 group。后续再引入显式配置：

```yaml
allow_split_oversized_group: false
```

默认不拆，避免破坏 prefix reuse locality。

### 七、verl 接入点

最合适的接入点是 `RayPPOTrainer._balance_batch()` 附近。

原因：

- 这里仍能看到完整 global train batch。
- 这里已经有 dp_size 查询和 `batch.reorder()`。
- 重排后 verl worker group dispatch 会自动把连续区间分给 DP ranks。
- 到 `MegatronPPOActor` 时每个 rank 只看到 local batch，已经无法做真正跨 DP rank placement。

建议 verl 侧只加薄分支：

```python
if prefix_sharing_dp_balance_enabled(batch, config):
    metrics.update(
        reorder_dataproto_for_prefix_group_dp_balance(
            batch=batch,
            dp_size=dp_size,
            group_key=config.prefix_sharing.dp_balance.group_key,
        )
    )
else:
    existing_seqlen_balance(batch, metrics)
```

核心实现放在：

```text
prefix_sharing/integrations/verl_dp_balance.py
prefix_sharing/core/group_partition.py
```

Megatron 不需要修改。

### 八、和 micro-batch 的关系

当前设计只解决 DP rank 间 placement。

rank-local micro-batch 切分仍按现有 verl 路径运行：

```text
MegatronPPOActor.forward_backward_batch()
  -> rearrange_micro_batches()
```

短期风险：

- 同一个 DP rank 内，`rearrange_micro_batches()` 仍可能把同一个 `uid` 拆到多个 micro-batch。

当前决策：

- 先不做跨 micro-batch activation 共享。
- 先完成 DP group locality，确保同组样本至少在同一 DP rank。
- 下一步再单独设计 rank-local prefix-group-aware micro-batch 切分。

如果短期必须避免 micro-batch 内拆组，可以在后续加第二个薄 hook：

```text
rearrange_micro_batches()
  -> prefix_group_aware_rearrange_micro_batches()
```

但这属于下一件事，不混入当前 DP 负载均衡设计。

### 九、step / tree 模式接入

verl 当前普通 PPO / GRPO 场景已天然支持：

```text
uid = 原始 prompt id
```

step / tree 模式接入要求数据生产方或 rollout manager 显式设置：

```text
data.non_tensor_batch["uid"] = common_ancestor_id
```

或使用独立字段：

```text
data.non_tensor_batch["prefix_group_id"] = common_ancestor_id
```

若使用独立字段，配置：

```yaml
prefix_sharing:
  dp_balance:
    group_key: prefix_group_id
```

建议语义：

- `uid`: 默认 prefix group id，兼容当前 verl advantage grouping。
- `prefix_group_id`: 可选覆盖字段，当业务希望将 advantage grouping 和 prefix grouping 解耦时使用。
- `trajectory_id`: 可选，仅用于调试和日志。
- `node_id` / `parent_id`: 可选，仅 tree-aware 分析需要，DP balance 第一版不读取。

第一版不要求 verl 原生理解 step/tree；只要 `DataProto.non_tensor_batch` 里携带 per-sample group id 即可。

### 十、正确性边界

DP group balance 只做 sample reorder，理论上不改变每条样本的训练目标。

必须保证：

1. `batch.batch` 和 `batch.non_tensor_batch` 同步 reorder。
2. `uid` 与 reward / advantage / old logprob / response / label 对齐。
3. GRPO advantage 在 reorder 后仍按 `uid` 分组。
4. rollout 输出与训练 batch union 后，`uid` 被正确 repeat。
5. padding sample 若存在，不参与真实 workload 或用 fallback 处理。

可能变化：

- 浮点归约顺序。
- dropout / RNG 消耗顺序。
- optimizer 更新轨迹。

这些属于样本调度变化带来的常规非 bitwise 差异，不是 prefix sharing 语义错误。

### 十一、Fallback 策略

触发 fallback 的情况：

1. `group_key` 不在 `batch.non_tensor_batch`。
2. `len(group_ids) != batch_size`。
3. `dp_size <= 1`。
4. group 数量少于 `dp_size` 且不允许空 rank。
5. partition 后无法满足 verl dispatch 的 equal-size 要求。

fallback 行为：

```text
如果 fallback_to_seqlen_balance=True:
  使用 verl 原 seqlen balance
否则:
  不重排并记录 fallback_reason
```

日志 / metrics 至少记录：

```text
prefix_dp_balance/enabled
prefix_dp_balance/fallback_reason
prefix_dp_balance/group_count
prefix_dp_balance/group_min_size
prefix_dp_balance/group_max_size
prefix_dp_balance/dp_min_workload
prefix_dp_balance/dp_max_workload
prefix_dp_balance/dp_imbalance_ratio
```

### 十二、测试计划

#### prefix-sharing 单元测试

1. 同一 `uid` 的样本分配到同一个 DP partition。
2. 多个 `uid` 的 group workload 能被均衡分配。
3. 缺失 `uid` 时返回 fallback reason。
4. group 数小于 dp_size 时按配置 fallback。
5. reorder 后 sample index 覆盖完整且无重复。

#### verl 集成测试

1. 构造 `DataProto`，`uid=[a,a,b,b,c,c]`，验证 `_balance_batch()` 后同 uid 不跨 DP partition。
2. 验证 `batch.batch` 与 `non_tensor_batch` reorder 同步。
3. GRPO advantage 使用 reorder 后 `uid` 仍正确分组。
4. prefix-sharing disabled 时保持 verl 原逻辑。

#### 性能观测

1. 记录 sample-level seqlen balance 与 group-level balance 的 reuse ratio 对比。
2. 记录各 DP rank original tokens / estimated compute tokens / group count。
3. 对比 step wall time 和 prefix sharing saved token ratio。

### 十三、开发拆分

建议提交顺序：

1. `[feat] 新增 prefix group partitioner`
2. `[test] 覆盖 group-level DP partition`
3. `[feat] verl _balance_batch 接入 prefix group balance 薄 hook`
4. `[test] 覆盖 DataProto reorder 与 uid locality`
5. `[doc] 收敛 DP group balance 设计`

### 当前结论

Phase 2 的 DP 负载均衡第一版应以 `uid` 为默认 `prefix_group_id`，在 verl controller 侧做 group-level reorder。它具备稳定性能收益预期，且不引入跨 rank / 跨 micro-batch activation 共享的精度风险。

## 2026-05-21 15:20 DP workload balance 与 rank-local DP 的关系澄清

### 一、问题背景

阅读 `dependency/PrefixTrain_dev` 后确认，它确实实现了 DP 之间的 mini-batch / workload balance：

- 参数入口：`--dp-workload-balance`
- 主要实现：`runtime/megatron/training.py::balance_initialize_data`
- 估算函数：`runtime/megatron/prefix_match.py` 内的在线 Trie 负载估算逻辑
- 分区函数：`runtime/megatron/prefix_match.py::kk_partition`
- micro-batch 划分：`partition_micro_batch` / `partition_micro_batch_token_level`

这部分逻辑容易和我们当前讨论的“DP 适配”混淆，需要拆成两个概念。

### 二、两个不同问题

#### 1. DP correctness / rank-local DP

这是我们 Phase 2.1 已经确定并完成第一步实现的内容。

定义：

```text
每个 DP rank 对自己拿到的 local micro-batch 独立执行 prefix detect / plan / store / restore。
```

它解决的是正确性和运行时隔离问题：

- 不同 DP rank 的 runtime context 不串。
- 同一 rank 的多个 micro-batch / gradient accumulation 不串。
- restore 使用 local batch index。
- store 生命周期限制在当前 micro-batch context。
- 日志和 stats 能看到 `dp_rank / dp_world_size`。

这部分不要求改变样本分发方式，也不要求跨 DP rank 通信。

#### 2. DP workload balance

这是 PrefixTrain_dev 额外做的性能优化，不是 DP correctness 的前提。

它解决的是性能负载不均问题：

```text
普通 DP 通常按样本数切 batch。
prefix sharing 后，不同样本真实 compute tokens = 原始 tokens - 可复用 prefix tokens。
如果仍按样本数平均分，每个 DP rank 的实际计算量可能严重不均。
step wall time 会被最慢的 DP rank 决定。
```

PrefixTrain_dev 的做法是：

1. 对每个 uid / trajectory 内的 token 序列做在线 Trie 负载估算。
2. 用 Trie 模拟按顺序 prefix reuse 后还需要计算的 token 数。
3. 将每个 uid 的 compute-token count 作为负载权重。
4. 用 `kk_partition()` 把 uid 分到多个 DP / pipeline 分区，使每个分区 compute tokens 尽量接近。
5. 再在每个分区内用 `partition_micro_batch*()` 切 micro-batch。
6. 生成 `shared_prefix_len / store_for_sample_idx / shared_for_sample_idx / batch_idx_mapping_sample_idx`，供后续 activation memory manager 使用。

所以它不是“DP rank 之间共享 K/V”，而是“prefix-aware 的数据分发和 micro-batch 切分”。

### 三、和我们当前 DP 方案的关系

当前 rank-local DP 已经能保证：

- DP world size 大于 1 时，每个 rank 的 prefix sharing 语义正确。
- 不跨 rank 复用，也不会破坏 DDP / Megatron DP 的梯度同步。
- 不需要引入跨 rank activation 传输。

但当前方案还不能保证：

- 不同 DP rank 的 prefix-sharing 后 compute tokens 均衡。
- micro-batch 内每次 forward 的真实 token workload 均衡。
- 高 cache ratio 场景下 DP step time 最优。

因此结论是：

```text
基础 DP 适配：不需要 DP workload balance。
性能可用性 / 对齐 PrefixTrain_dev：后续需要 DP workload balance。
```

### 四、是否需要适配

分阶段判断：

1. Phase 2.1 DP correctness smoke：不适配，继续保持 rank-local DP。
2. DP 性能评估：必须记录每个 rank 的 original tokens、saved tokens、compute tokens、cache ratio。
3. 如果各 rank compute-token imbalance 明显，再引入 DP workload balance。
4. 若目标是复现 PrefixTrain_dev 的大规模训练收益，DP workload balance 应进入 Phase 2.1 后半段或 Phase 2.2。

不建议现在直接把 PrefixTrain_dev 的实现搬过来，原因：

- 它绑定离线 `input_token_path` 和 Megatron 初始化流程。
- 它按 uid / trajectory 粒度重排数据，和 verl / Ray actor 的 batch dispatch 需要重新设计边界。
- 它的 memory manager 依赖全局 sample index 与跨 micro-batch activation store，而我们当前 Phase 2.1 明确先限制在 rank-local micro-batch context。
- token-level balance 甚至会拆 sequence，需要额外验证 label、position、loss mask、restore mapping。

### 五、建议适配路径

先把 workload balance 作为独立能力，不和 DP correctness 混在一起。

建议新增概念：

```text
PrefixSharingWorkloadEstimate:
  original_tokens
  reusable_prefix_tokens
  compute_tokens
  provider_count
  reuse_count

PrefixSharingBatchPartition:
  rank_assignments
  micro_batch_assignments
  restore_order_mapping
```

实施顺序：

1. **只观测，不重排**：在当前 rank-local DP stats 中增加 per-rank `compute_tokens = original_tokens - saved_tokens`。
2. **离线模拟器**：给定一批 token ids 和 dp_world_size，用 prefix-sharing 自己的 workload estimator 和 partitioner 输出 imbalance before / after。
3. **rank-local micro-batch balance**：先只在单 rank 内按 estimated compute tokens 切 micro-batch，不改变跨 rank 数据分发。
4. **verl dispatch 适配**：在 actor group dispatch 前做 prefix-aware rank assignment，并保留 restore order mapping。
5. **可选 token-level balance**：最后再评估是否支持切 sequence；默认不做，因为风险最高。

### 六、当前决策

DP workload balance 是 Phase 2 并行策略里的重要性能子任务，但不是当前 DP correctness 的 blocker。

当前阶段保持：

- rank-local prefix sharing。
- 不跨 DP rank 共享 activation。
- 不改变 verl / Megatron 的数据分发。
- 先通过 stats 暴露 rank 间 compute-token imbalance。

后续当 DP=2/4 smoke 跑通后，再以 benchmark 数据决定是否进入 workload balance 实现。

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
