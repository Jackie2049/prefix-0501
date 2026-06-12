---
name: fix-prefix-sharing-logprob-entropy
overview: 修复 prefix-sharing 的 3 个 bug：① interior response logprob restore 用错 label 位置（off-by-one）；② entropy 没有恢复 trimmed prefix 位置的值，导致 PPO loss 熵奖励错误；③ dump 位置在 packed 中间态，无法正确对比 baseline。
todos:
  - id: context-add-entropy-cache
    content: 在 context.py 的 PrefixSharingRuntimeContext 新增 entropy_restore_cache 字段，修正 interior 注释 off-by-one
    status: completed
  - id: verl_mcore-fix-interior-label
    content: 修复 verl_mcore.py 中 restore_suffix_first_log_probs_from_prefix 的 interior label 索引（provider_1d_pos+1 → provider_1d_pos）
    status: completed
    dependencies:
      - context-add-entropy-cache
  - id: verl_mcore-add-entropy-restore
    content: 在 verl_mcore.py 的 restore 函数中新增 entropy 缓存逻辑，新增 write_restored_entropy_to_2d 函数
    status: completed
    dependencies:
      - verl_mcore-fix-interior-label
  - id: megatron_actor-integrate
    content: 在 megatron_actor.py 中 import 新函数、串联调用 write_restored_entropy_to_2d、将 dump 移到 2D 空间
    status: completed
    dependencies:
      - verl_mcore-add-entropy-restore
  - id: run-tests
    content: 运行 unit_test 和 optional 集成测试确认修复通过
    status: completed
    dependencies:
      - megatron_actor-integrate
---

## 产品概述

修复 prefix-sharing 模式下 forward 链路中 3 个独立 bug，确保 logprob / entropy / logits 的语义与 baseline 完全一致。

## 核心修复

### BUG #1: Interior Response logprob 恢复的 label 索引 off-by-one

- **位置**: `verl_mcore.py` line 430-433 + `context.py` line 74-75 注释
- **错误**: 使用 `labels[provider_1d_pos + 1]`（取到 `token at interior_pos+1`）
- **正确**: 使用 `labels[provider_1d_pos]`（取到 `token at interior_pos`）
- **根因**: `provider_1d_pos` 对应原位置 `interior_pos-1`，其 label 按 verl 约定是 next-token = `token at interior_pos`。`+1` 偏移后变成了下下个 token

### BUG #2: Entropy 在 trimmed prefix 位置未恢复

- **影响**: `postprocess_packed_seqs` scatter 后 trimmed prefix 位置 entropy=0，导致 PPO loss 的 entropy bonus 错误（共享 prefix 内 response token 的 entropy bonus 缺失）
- **修复**: 新增 `entropy_restore_cache` 到 `PrefixSharingRuntimeContext`，在 restore 阶段同时缓存 entropy，新增 `write_restored_entropy_to_2d` drain 到 2D output

### BUG #3: Dump 位置在 packed 中间态

- **当前**: dump 在 `logits_processor` 内部（packed 格式），shape 与 baseline 不同无法对比
- **修复**: dump 移到 `write_restored_logprobs_to_2d` + `write_restored_entropy_to_2d` 之后（2D 空间）

## 技术栈

- Python + PyTorch（现有项目不变）
- 修改范围：`prefix-sharing/` 2 个文件 + `dependency/verl_v070/` 1 个文件

## 实现方案

### 整体策略

按依赖顺序修复，先修 context 结构（新增 cache 字段），再修 restore 计算逻辑（off-by-one + 新增 entropy 缓存），然后修 2D 注入（新增 `write_restored_entropy_to_2d`），最后在 actor 层串联调用 + 移动 dump 位置。

### 修复 1: `context.py` — 新增 `entropy_restore_cache` 字段

**文件**: `prefix-sharing/prefix_sharing/integrations/context.py`

在 `PrefixSharingRuntimeContext` dataclass 新增字段，紧邻 `logprob_restore_cache`：

```python
entropy_restore_cache: dict[tuple[int, int], Any] = field(default_factory=dict)
```

同时修正 line 74-75 注释中的 `provider_1d_pos + 1` 为 `provider_1d_pos`。

### 修复 2: `verl_mcore.py` — 修正 interior label + 新增 entropy 缓存

**文件**: `prefix-sharing/prefix_sharing/integrations/verl_mcore.py`

#### 2a. 修正 `restore_suffix_first_log_probs_from_prefix` (line 386-449)

**off-by-one fix (line 430-433)**:

```python
# 修改前
provider_label = labels[0:1, index.provider_1d_pos + 1 : index.provider_1d_pos + 2]

# 修改后
provider_label = labels[0:1, index.provider_1d_pos : index.provider_1d_pos + 1]
```

**新增 entropy 缓存 (在 line 445 restored_logprob 计算之后)**:

```python
# Compute entropy from the same provider logits
provider_logits_2d = provider_logits.squeeze(0)  # [1, 1, V//tp] -> [1, V//tp]
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy
restored_entropy = vocab_parallel_entropy(provider_logits_2d).reshape(())
ctx.entropy_restore_cache[(index.reuse_idx_in_batch, index.target_2d_pos)] = restored_entropy
```

注意：`provider_logits` 在上面的 `if/else` 之前已经 `.clone()` 过，是带温度除过的 logits，entropy 应直接基于此计算。`provider_logits_2d = provider_logits.squeeze(0)` 将 `[1, 1, V//tp]` 压缩为 `[1, V//tp]` 以匹配 `vocab_parallel_entropy` 的输入要求。

#### 2b. 新增 `write_restored_entropy_to_2d` 函数 (紧接在 line 480 之后)

```python
def write_restored_entropy_to_2d(output: dict[str, Any]) -> dict[str, Any]:
    """Drain the entropy restore cache into the 2D output tensor."""
    ctx = current_prefix_sharing_context()
    if ctx is None or not ctx.entropy_restore_cache:
        return output
    entropy = output.get("entropy")
    if entropy is None:
        return output
    for (batch_idx, target_2d_pos), ent_val in ctx.entropy_restore_cache.items():
        entropy[batch_idx, target_2d_pos] = ent_val
    return output
```

### 修复 3: `megatron_actor.py` — 串联调用 + 移动 dump

**文件**: `dependency/verl_v070/verl/workers/actor/megatron_actor.py`

#### 3a. 新增 import（line 67-69 附近）

```python
write_restored_entropy_to_2d,
```

需要添加到 `try/except` 块中，失败时设为 `None`。

#### 3b. forward_step 中调用（line 740-742）

在 `write_restored_logprobs_to_2d` 之后添加：

```python
if write_restored_entropy_to_2d is not None:
    output = write_restored_entropy_to_2d(output)
```

#### 3c. 移动 dump 到 restore 之后（替换 line 690-699）

**删除** `logits_processor` 内部的 dump 代码块（line 690-699，即 `########## prefix-sharing logits/entropy dump ##########` 包围的全部代码）。

**新增** dump 到 `write_restored_entropy_to_2d` 之后（line 740-742 之后）：

```python
########## prefix-sharing dump (2D space after all restores) ##########
_ps_dump_dir = os.environ.get("PREFIX_SHARING_DUMP_DIR")
if _ps_dump_dir and forward_only:
    os.makedirs(_ps_dump_dir, exist_ok=True)
    if torch.distributed.get_rank() == 0:
        tag = "old" if forward_only else "train"
        torch.save(output["log_probs"].detach().clone(), os.path.join(_ps_dump_dir, f"logprobs_{tag}.pt"))
        if "entropy" in output:
            torch.save(output["entropy"].detach().clone(), os.path.join(_ps_dump_dir, f"entropy_{tag}.pt"))
        torch.save(label.detach().cpu().clone(), os.path.join(_ps_dump_dir, "label.pt"))
########## prefix-sharing dump ##########
```

关键设计：仅 `forward_only=True`（compute_log_prob 路径）时 dump，避免训练路径频繁 I/O 影响性能。logits 不再 dump（2D 完整 logits 太大，且不直接进 loss）。文件名改为 `logprobs_{tag}.pt` 和 `entropy_{tag}.pt` 更准确。

## 实现注意事项

1. **熵计算在温度除法之后**：`provider_logits` 已通过 `logits.div_(temperature)` 处理，entropy 基于此计算与 baseline 一致
2. **TP all-reduce**：`vocab_parallel_entropy` 内部含 all-reduce，在 restore 的 for 循环中逐条调用会有额外通信开销，但 restore 条目数量有限（~微批数量 × (interior_count + 1)），可接受
3. **向后兼容**：`write_restored_entropy_to_2d` 检查 `output.get("entropy")` 是否为 None，未启用 `calculate_entropy` 时自动跳过
4. **dump 仅 forward_only**：避免训练前向在大量 micro-batch 上频繁写文件

## Agent Extensions

### SubAgent

- **code-explorer**
- 用途：在 plan 执行前对 `context.py`、`verl_mcore.py`、`megatron_actor.py` 做最终精确行号确认，确保修改位置与当前代码一致
- 预期结果：确认所有修改点的精确行号和上下文，避免 plan 执行时出现行号偏差