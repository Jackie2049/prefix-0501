# Prefix Sharing Roadmap

> 本文档是项目全局 roadmap。Phase 1 详细规格见 `phase-1/design-final.md`，Phase 2 总体设计见 `phase-2/design-history.md`，并行策略方案见 `phase-2/parallel-plan.md`。

---

## 当前状态

当前代码处于 Phase 2 分支。Phase 1 的 core 语义和本地 CPU 开发者测试已经完成，真实 GPU / NPU / verl + Megatron 环境 smoke test 仍是业务落地前的必要闭环。

阶段二的详细设计已经抽离到 `phase-2/design-history.md`。阶段二第一步先聚焦并行策略支持，不直接进入高性能 backend 或全量 activation reuse。

---

## Phase 1：verl + Megatron RL MVP

目标：打通当前最关键业务链路的最小正确闭环。

成功标准：

- 正确性测试通过。
- small-scale RL actor 链路可运行。
- 在合理精度假设下 logprob、loss、grad 对齐。
- patch 可启停，不污染非 prefix sharing 路径。

当前状态：

- `core/config.py`、`prefix_detector.py`、`planner.py`、`batch_trim.py`、`logprob.py`、`prefix_store.py` 已落地。
- `backends/torch_ref.py` reference backend 已支持 packed THD、`q_len != kv_len`、GQA repeat 和 transitive reuse store。
- `integrations/verl_mcore.py`、`megatron_runtime.py` 已补齐真实 verl actor helper 和 Megatron attention runtime hook。
- 本地测试通过：`28 passed, 5 skipped`。skip 来自本地缺少 `torch`、`torch_npu`、`verl`。

剩余闭环：

- 真实 GPU / NPU / verl 环境下 small-scale actor logprob smoke test。
- 真实 actor update smoke test。
- Megatron 单层/小层数模型 forward / backward 对齐。
- 真实 packed THD、RoPE、vocab-parallel restore 路径验证。

---

## Phase 2：业务落地能力

目标：让 prefix sharing 从 Phase 1 的单机语义 MVP 进入真实训练配置可用状态。

详细设计见 `phase-2/design-history.md`。

### Phase 2.1 并行策略支持

详细方案见 `phase-2/parallel-plan.md`。

优先级：

- P0：DP / micro-batch / gradient accumulation 生命周期隔离。
- P0：TP local K/V shard 支持。
- P0：vocab-parallel Prefix-Last Restore 验证。
- P1：SP layout 验证。
- P1：PP stage-local store、最后 stage restore、VPP context 隔离。
- P2：CP local/global position、KV exchange 和异 rank restore。
- P2：EP / MoE attention-only 兼容。

阶段验收：

- 每一种并行能力都有 capability 声明。
- 不支持的组合明确 fail fast 或显式 fallback。
- reference backend 先完成 forward / loss / grad 对齐。
- 分布式 optional tests 覆盖关键组合。

### Phase 2.2 Backend 解耦

摘要：扩展 `BackendCapabilities`，新增 runtime validation，明确 device、dtype、layout、parallel、fused path 和 fallback reason。详细设计见 `phase-2/design-history.md`。

### Phase 2.3 高性能 Backend

摘要：在 reference path 正确后接入 CUDA torch reference、flash-attn、Transformer Engine 和 CANN NPU backend。详细设计见 `phase-2/design-history.md`。

### Phase 2.4 Prefix Activation Reuse

摘要：探索 provider prefix hidden states、attention/MLP 中间激活、prefix-last logits 等 KV 以外的 activation reuse。详细设计见 `phase-2/design-history.md`。

---

## Phase 3：训练范式扩展

目标：把 prefix sharing 从 RL MVP 扩展为通用训练能力。

范围：

- `verl + FSDP` RL 路径。
- standalone Megatron SFT。
- standalone Megatron pretrain。
- standalone FSDP SFT / pretrain。
- full-prefix restore，用于完整序列 loss。
- PromptPrefixDetector。
- 面向 next-token prediction 的通用 API。

---

## Phase 4：上游 PR 与产品化

目标：把验证后的能力沉淀为稳定 API 和上游扩展点。

方向：

- verl PR：prefix sharing extension hook 或可选 config path。
- Megatron PR：attention / packed sequence / RoPE / output restore 正式扩展点。
- 独立插件或 pip package。
- 稳定 API 文档。
- 兼容矩阵。
- benchmark。
