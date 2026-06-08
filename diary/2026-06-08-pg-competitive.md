# 2026-06-08: PR #4368 (PrefixGrouper) 竞品分析 — v4 完整 Benchmark + GRPO E2E

## 背景

PR #4368 (https://github.com/verl-project/verl/pull/4368) 将 PrefixGrouper (https://github.com/johncaged/PrefixGrouper) 集成到 verl 的 FSDP worker，加速 GRPO 训练。

核心思想：GRPO 中每个 prompt rollout.n 次复制，PrefixGrouper 将冗余注意力分解为：
1. **Prefix self-attention**: 每个 unique prompt 只计算一次
2. **Suffix concat-attention**: 每个 response 注意共享的 prefix output

## v4 Benchmark 方法

- **模型**: Qwen2.5-0.5B-Instruct (494M, 24层, 896 hidden, 14 heads, 2 kv_heads)
- **GPU**: RTX 4090 单卡 (24GB)
- **方法**: transformers + PrefixGrouper 包，使用 `attn_implementation="flash_attention_2"` (PG 的 sdpa 路径会报错因为 mask 格式不匹配)
- **指标**: 3 次重复取 median，有 warmup; forward-only + forward+backward
- **PG 集成**: 通过 verl monkey_patch (`apply_prefix_grouper_patch`) 正确注入 prefix_grouper 参数

## 核心结果

### Forward Only

| Config | NoPG(ms) | PG(ms) | Speedup | NoPG GiB | PG GiB | MemSave% | cos_sim |
|--------|---------|--------|---------|---------|--------|----------|---------|
| p64_s64_n4 | 28.5 | 78.6 | **0.36x** | 1.22 | 1.41 | -15.4% | 0.999994 |
| p128_s128_n4 | 29.4 | 79.9 | **0.37x** | 1.52 | 1.88 | -23.8% | 0.999985 |
| p256_s64_n4 | 31.5 | 80.6 | **0.39x** | 1.67 | 1.95 | -17.3% | 0.999987 |
| p256_s256_n4 | 31.3 | 82.2 | **0.38x** | 2.10 | 2.83 | -34.4% | 0.999990 |
| p256_s256_n8 | 43.0 | 87.2 | **0.49x** | 3.27 | 4.57 | -39.9% | 0.999995 |
| p512_s128_n4 | 32.3 | 82.3 | **0.39x** | 2.39 | 2.97 | -24.1% | 0.999992 |
| p512_s256_n8 | 63.2 | 85.5 | **0.74x** | 4.43 | 5.87 | -32.6% | 0.999991 |
| p512_s256_n4_np2 | 63.6 | 90.1 | **0.71x** | 4.43 | 6.16 | -39.2% | 0.999993 |

**平均 forward 加速**: **0.48x** (PG 更慢 2倍!)

### Forward + Backward

| Config | NoPG(ms) | PG(ms) | Speedup | NoPG GiB | PG GiB | MemSave% |
|--------|---------|--------|---------|---------|--------|----------|
| p64_s64_n4 | 83.1 | 173.0 | **0.48x** | 2.67 | 2.85 | -6.6% |
| p128_s128_n4 | 77.6 | 176.1 | **0.44x** | 3.75 | 3.33 | +11.2% |
| p256_s64_n4 | 80.5 | 185.7 | **0.43x** | 4.41 | 3.40 | +23.0% |
| p256_s256_n4 | 97.3 | 181.6 | **0.54x** | 6.56 | 5.64 | +13.9% |
| p256_s256_n8 | 150.6 | 193.7 | **0.78x** | 12.02 | 9.69 | +19.4% |
| p512_s128_n4 | 108.2 | 181.3 | **0.60x** | 7.88 | 5.26 | +33.2% |
| p512_s256_n8 | 212.2 | 194.2 | **1.09x** | 17.71 | 11.58 | +34.6% |
| p512_s256_n4_np2 | 211.7 | 199.3 | **1.06x** | 17.71 | 12.92 | +27.1% |

**平均 fwd+bwd 加速**: **0.68x** (PG 更慢!)
**唯一加速场景**: p512_s256_n8 = **1.09x** (非常长 prefix + 大 n)

## 关键发现

### 1. PG 在 forward-only 上**更慢** (0.36-0.74x，平均 0.48x)

**根因**：PG 的 monkey-patch 将每个 attention layer 的 forward 变成两步调用 (prefix_attn + suffix_attn)，导致：
- 2x kernel launch overhead (每层 2 次 flash_attention 调用 vs 1 次)
- Concat_input 的 grouping/ungrouping Python overhead
- 0.5B 小模型 forward 本身就 ~30ms，双步 overhead 占 ~50ms

对比 No-PG: 单次 flash_attn 覆盖整个序列 → 1 次 kernel launch

### 2. PG 在 fwd+bwd 上接近**无加速** (0.43-1.09x，平均 0.68x)

在 backward 中，PG 的内存节省开始显现 (activation 不需要重复存储 prefix 部分):
- **内存节省**: 小 batch 无 (-6.6%) 到大 batch (+34.6%)
- **速度**: 仅在极端场景 (p512_n8) 才有 1.09x 加速

### 3. PG forward 内存**增加**而非减少 (-15% 到 -40%)

原因：PG concat_input 创建的超长序列 = prefix + n * suffix，activation 需要整个序列的 KV cache 和 intermediates，而 No-PG 的 batch 中每个序列是独立的 (prefix_len + suffix_len)。

### 4. 精度优秀 (cos_sim = 0.999985-0.999995)

使用 flash_attention_2 + 正确的 monkey_patch，PG 精度很高！cos_sim 全部 > 0.99998，远好于之前 v3 测试 (0.987)。之前精度差的原因是使用了 sdpa (mask 格式不匹配导致错误)。

max_diff = 0.16-0.35，来自 bf16 累积误差，可以接受。

### 5. PG 只优化 attention，不优化 MLP

| Component | No-PG compute | PG compute | Savings |
|-----------|--------------|------------|---------|
| **Prefix attention** | n × prefix_len² | 1 × prefix_len² | (n-1)/n |
| **Suffix attention** | n × suffix_len × total_len | n × suffix_len × (prefix + suffix) | 0% |
| **MLP** | n × total_len | n × total_len | 0% |
| **Overall** | varies | varies | only ~8-15% |

**理论 attn-only 加速**: 1.60-2.50x
**理论 overall 加速**: 0.3 × attn_speedup + 0.7 × 1.0 = 1.18-1.45x
**实际 forward**: 0.36-0.74x (kernel overhead 完全抵消!)

## 与我们的 Prefix-Sharing 对比

| 维度 | PrefixGrouper (PR #4368) | 我们的 PS (prefix-0501) |
|------|--------------------------|-----------|
| **优化范围** | 仅 attention (~8-15%) | attention + MLP (全模型) |
| **方法** | Concat grouped batch (单次 forward) | 两-pass (prefix hidden state 复用) |
| **精度** | cos_sim ≈ 0.99999 (flash_attn path) | cos_sim ≈ 0.99999 (两-pass math等价) |
| **Forward 加速** | 0.36-0.74x (更慢!) | 需要在 verl 中集成验证 |
| **Fwd+Bwd 加速** | 0.43-1.09x (仅极端场景加速) | 需要在 verl 中集成验证 |
| **Forward 内存** | 增加 15-40% | 减少 40-61% |
| **Fwd+Bwd 内存** | 减少 7-35% | 减少 40-61% |
| **Worker 支持** | 仅 FSDP | FSDP + Megatron |
| **模型修改** | monkey_patch ALL_ATTENTION_FUNCTIONS | Megatron SelfAttention.forward hook |
| **Qwen3.6 支持** | 无 (不支持 DeltaNet) | 有 (SelfAttention + GatedDeltaNet) |

### 我们 PS 的核心优势

1. **全模型复用**: MLP 也被复用 → 优化 68-80% 的计算量 (vs PG 仅 8-15%)
2. **不增加 forward 内存**: prefix hidden state 只需一次 forward → 减少 40-61%
3. **Qwen3.6 DeltaNet**: PG 不支持 DeltaNet 层; 我们的 PS 同时支持 SelfAttention 和 DeltaNet carry state 复用
4. **Megatron 支持**: 可以在 TP/PP/FSDP 下工作

## 为什么 PG 的 E2E 加速有限

PR 作者在 Qwen3-8B + 4xH800 上报告 1.14-1.70x step 加速。差异原因：
1. **大模型 (8B vs 0.5B)**: attention 计算量更大 → kernel overhead 占比更低
2. **H800 GPU**: 更多 SMs → flash_attn launch overhead 更低
3. **长 context**: prefix 占比更高 → 复用收益更大
4. **但他们也只报告 step 级别**: generation 占 ~97% 总时间 → E2E 加速受限

## GRPO 端到端训练实测

使用简化 GRPO 训练循环（HF transformers rollout + PrefixGrouper monkey_patch training），在 Qwen2.5-0.5B-Instruct + RTX 4090 上实测完整 GRPO step：

### 配置

- **模型**: Qwen2.5-0.5B-Instruct (494M)
- **GPU**: RTX 4090 单卡
- **数据**: GSM8K (4 prompts, n=4 responses each)
- **Rollout**: HF transformers model.generate() (no vLLM dependency)
- **训练**: bf16, AdamW, lr=1e-6, clip_ratio=0.2, kl_coef=0.001
- **对比**: 3 training steps, PG ON vs OFF

### 结果

| Metric | No-PG | PG (PrefixGrouper) | Speedup |
|--------|-------|---------------------|---------|
| Training fwd+bwd | 220.8ms | 226.6ms | **0.97x** (PG slower!) |
| Peak memory | 17.22 GiB | 18.21 GiB | **-5.9%** (PG uses MORE!) |
| Full step | 10.248s | 10.254s | **1.00x** (no speedup) |
| Rollout fraction | 97.8% | 97.8% | — |
| Reward fraction | 0.0% | 0.0% | — |

### GRPO E2E 关键发现

1. **Rollout 占 97.8% 的 step 时间**: generation 是绝对瓶颈，训练仅占 2.2%
2. **PG training speedup = 0.97x**: 在 n=4 + 0.5B 模型上 PG 反而更慢
3. **PG step-level speedup = 1.00x**: 即使 PG 能加速训练 2x，step 也仅加速 1.04x
4. **PG 内存反而增加**: concat_input 创建更长序列 → 18.21 GiB vs 17.22 GiB

### 为什么 PG 的 GRPO E2E 加速几乎为零

**数学分析**:
- step_speedup = 1 / (rollout_fraction + (1 - rollout_fraction) / training_speedup)
- 当 rollout_fraction = 97.8%:
  - training_speedup = 1.0x → step_speedup = 1.0x
  - training_speedup = 2.0x → step_speedup = 1.04x
  - training_speedup = 10x → step_speedup = 1.20x
- **即使 PG 将训练加速到不可能的 10x，step 也仅加速 1.20x!**

**结论**: 在 GRPO 训练中，rollout generation 是决定性瓶颈。Prefix-sharing/PrefixGrouper 只优化训练部分，对整体 step speedup 的贡献被 rollout 的 97%+ 时间占比完全稀释。

**PR 作者在 8B+H800 上的加速原因**: 他们可能使用了 async rollout（vLLM rollout 在独立 GPU 上运行，训练 GPU 不等 rollout），这时 rollout_fraction ≈ 0%，PG 的训练加速才能充分体现。

## 下一步

1. 在我们的 PS v3 fair benchmark 中验证 speedup (需要 verl monkey-patch 集成)
2. 用真实 Qwen3.6 权重验证 sigmoid gate 精度 (已确认：真实模型用 sigmoid，config "swish" 是误导)
3. 对比 PG vs PS 在相同条件下的 performance
4. **重要**: PS 的加速也受 rollout 瓶颈限制 → 需要 async rollout 才能发挥训练加速