# Phase 2 最终方案

> 本文档承载 Phase 2 已收敛方案。尚未收敛的讨论、备选方案和阶段性记录继续写入 `design-history.md`。

---

## 2026-05-22 DP 与 Prefix Group DP 负载均衡

### 一、范围

本节是 Phase 2 并行策略的第一项最终方案，覆盖：

1. DP correctness / rank-local DP。
2. DP rank 间 prefix group 负载均衡。

暂不覆盖：

- 跨 DP rank K/V 或 activation 共享。
- 跨 micro-batch activation 共享。
- TP / PP / CP / EP 的最终方案。
- step / tree 数据结构本身，只定义它们如何传递 prefix group id。

### 二、DP Correctness

DP 下 prefix sharing 的执行 scope 是：

```text
当前 DP rank 内的当前 actor micro-batch
```

每个 DP rank 独立执行：

```text
prefix detect -> plan -> trim -> attention store/restore -> logprob restore
```

不做：

- 不跨 DP rank 检测 shared prefix。
- 不跨 DP rank 交换 provider prefix K/V。
- 不跨 forward / optimizer step 持久保存 K/V。
- 不在 prefix-sharing 内自行做 DP all-reduce。

必须保证：

1. `PrefixSharingRuntimeContext` 按 micro-batch 创建和销毁。
2. `PrefixKVStore` 生命周期限制在当前 context。
3. `forward_id / micro_batch_id` 足以区分 gradient accumulation 期间的多个 forward。
4. `provider_idx_in_batch / reuse_idx_in_batch` 均为 local batch index。
5. Prefix-Last Restore 只使用当前 DP rank 当前 micro-batch 的 provider logits。
6. stats / debug log 包含 `dp_rank / dp_world_size / trace_key`。

### 三、ParallelEnv

DP rank 信息由 `ParallelEnv` 承载：

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

读取来源：

- 优先从 Megatron `parallel_state` 获取 DP rank/world size。
- Megatron 未初始化时 fallback 到单进程默认值。
- 不直接依赖 torch distributed default group，因为 verl / Megatron 的 DP group 未必等于默认 group。

### 四、DP 负载均衡目标

prefix sharing 的 DP 负载均衡采用：

```text
prefix group locality first
prefix-aware workload balance second
```

目标：

1. 同一 prefix group 的样本尽量进入同一个 DP rank。
2. DP rank 间按 prefix reuse 后的 compute workload 尽量均衡。
3. 只做样本重排，不改变样本内容、label、loss mask、reward、advantage、old logprob。
4. 核心逻辑放在 `prefix_sharing` 包，verl 只保留薄调用点。

### 五、Prefix Group

默认使用 verl `DataProto.non_tensor_batch["uid"]` 作为 prefix group id。

当前普通 PPO / GRPO 场景：

```text
uid == prefix_group_id == prompt group id
```

原因：

- verl 为每个原始 prompt 生成 `uid`。
- `DataProto.repeat(..., interleave=True)` 会把同一个 prompt 的 `uid` 复制到 n 条 rollout response。
- GRPO advantage 已经用 `uid` 做同 prompt 多 response 分组。
- 同 prompt 多 response 是最稳定的前缀复用来源。

step / tree 模式接入时，数据生产方应把公共祖先写入 `uid`：

```text
step mode: uid = step_root_id / shared_state_id
tree mode: uid = tree_root_id / common_ancestor_id
```

如需解耦 advantage grouping 与 prefix grouping，可使用独立字段：

```text
data.non_tensor_batch["prefix_group_id"] = common_ancestor_id
```

并通过配置指定：

```yaml
prefix_sharing:
  dp_balance:
    group_key: prefix_group_id
```

### 六、Prefix-Aware Workload

DP group balance 直接使用前缀复用后的 workload 估算。

对每个 group 内 token 序列执行：

```text
compute_tokens = estimate_incremental_prefix_compute_tokens(group_token_ids)
reusable_prefix_tokens = original_tokens - compute_tokens
group_workload = compute_tokens
```

`estimate_incremental_prefix_compute_tokens()` 语义：

1. 只建模当前 DP 调度组内、rank-local、按样本现有顺序发生的增量前缀复用。
2. 每条样本找到和此前样本的最长已存在 prefix。
3. 当前样本新增计算量 = `len(sample) - matched_prefix_len`。
4. group workload = 所有样本新增计算量之和。
5. 该函数是 prefix-sharing 自己的 workload 估算器，不承诺兼容 PrefixTrain_dev 的函数签名或实现细节。

输入 token 序列优先从：

```text
batch["input_ids"] + batch["attention_mask"]
```

估算时只使用 attention mask 标记的有效 token，padding 不参与 prefix match。

dense seqlen balance 只作为 fallback，不作为 prefix-sharing DP balance 的阶段目标。

### 七、DP Partition

输入：

```text
DataProto batch
dp_size
group_key = "uid"
```

流程：

1. 读取 `group_ids = batch.non_tensor_batch[group_key]`。
2. 按 `group_id` 聚合 sample indices。
3. 对每个 group 计算 prefix-aware workload。
4. 使用 Karmarkar-Karp / greedy LDM 将 group 分配给 `dp_size` 个 partition。
5. 每个 partition 内按 workload 稳定排序。
6. 展开为全局 sample index：

```text
global_idx = concat(dp0_sample_indices, dp1_sample_indices, ...)
```

7. 调用 `batch.reorder(global_idx)`。
8. verl 原有 dispatch 按重排后的连续区间切给 DP ranks。

不变量：

```text
同一个 group_id 的样本只出现在一个 DP partition 内。
```

默认不拆 group。若 group 大到无法满足 equal-size dispatch 约束，则 fallback。

### 八、verl 接入

接入点：`RayPPOTrainer._balance_batch()`。

原因：

- 这里能看到完整 global train batch。
- 这里已有 dp_size 查询与 `batch.reorder()`。
- 重排后 worker group dispatch 会自动把连续区间分给 DP ranks。
- 到 `MegatronPPOActor` 时每个 rank 只看到 local batch，已经无法做跨 DP rank placement。

verl 侧只保留薄分支：

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
prefix_sharing/core/group_partition.py
prefix_sharing/integrations/verl_dp_balance.py
```

Megatron 不需要修改。

### 九、Fallback

触发 fallback：

1. `group_key` 不在 `batch.non_tensor_batch`。
2. `len(group_ids) != batch_size`。
3. `dp_size <= 1`。
4. group 数量少于 `dp_size` 且不允许空 rank。
5. group 不可拆导致无法满足 equal-size dispatch。
6. 缺失 `input_ids` 或 `attention_mask`，无法计算 prefix-aware workload。

fallback 行为：

```text
fallback_to_seqlen_balance=True:
  使用 verl 原 seqlen balance
fallback_to_seqlen_balance=False:
  不重排并记录 fallback_reason
```

### 十、Metrics

至少记录：

```text
prefix_dp_balance/enabled
prefix_dp_balance/fallback_reason
prefix_dp_balance/group_count
prefix_dp_balance/group_min_size
prefix_dp_balance/group_max_size
prefix_dp_balance/dp_min_workload
prefix_dp_balance/dp_max_workload
prefix_dp_balance/dp_imbalance_ratio
prefix_dp_balance/reusable_prefix_tokens
prefix_dp_balance/original_tokens
prefix_dp_balance/compute_tokens
```

### 十一、测试要求

单元测试：

1. `estimate_incremental_prefix_compute_tokens()` 正确计算 prefix reuse 后 compute tokens。
2. 同一 `uid` 的样本分配到同一个 DP partition。
3. partition workload 使用 prefix-aware compute tokens。
4. 缺失 `uid` / token 字段时返回 fallback reason。
5. 不可满足 equal-size dispatch 时 fallback。

集成测试：

1. 构造 DataProto-like batch，验证 reorder 后同 `uid` 不跨 DP partition。
2. 验证 tensor batch 与 non_tensor_batch 同步 reorder。
3. prefix-sharing disabled 时保持 verl 原逻辑。
4. verl `_balance_batch()` 薄 hook 只在配置开启时调用 prefix-sharing helper。

### 十二、正确性边界

DP group balance 理论上不改变每条样本训练目标，因为它只改变样本顺序。

可能变化：

- 浮点归约顺序。
- dropout / RNG 消耗顺序。
- optimizer 更新轨迹。

这些属于样本调度变化导致的常规非 bitwise 差异，不是 prefix sharing 语义错误。
