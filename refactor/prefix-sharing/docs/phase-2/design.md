# Phase 2 总体设计

> 本文档承载阶段二的总体设计和工作主线。阶段二第一步“并行策略支持”的详细执行方案见 `parallel-plan.md`；全局阶段顺序见 `../roadmap.md`。

---

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
- `design.md`：Phase 2 总体设计和主线。
- `parallel-plan.md`：Phase 2.1 并行策略详细方案。
