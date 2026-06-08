# 遗留事项

本文档统一归档项目开发中的明确遗留事项。后续开发如果发现暂不处理但需要追踪的技术债、兼容性缺口或待验证场景，应按时间倒序追加到这里。

## 当前事项

### 2026-06-08：评估 PrefixTrain_dev 式多流异步预取与异步释放

**问题**：`survey/PrefixTrain_dev` 中的 `MemoryManager` 使用 CUDA stream / event 做 prefix activation 预取和异步释放，思路上可能降低 KV 拼接、跨 micro-batch 复用或 activation checkpointing recompute 场景中的等待时间。但该实现直接依赖 `torch.cuda.Stream/Event`，且缓存路径中存在 `clone().detach()`，不能直接迁移到 prefix-sharing；当前 prefix-0501 主路径仍是同一 micro-batch 内 `build_kv -> attention` 的强依赖链，异步预取很可能最终仍要 `wait_stream`，收益不稳定。

**方案**：后续作为独立性能实验推进，不进入当前 TorchRef P0 主线。先抽象 GPU/NPU 统一的 device stream runtime（CUDA 对接 `torch.cuda.*`，NPU 对接 `torch.npu.*`），再在 `PREFIX_SHARING_ASYNC_PREFETCH=1` 之类实验开关下验证：inter micro-batch prefix activation prefetch、activation checkpointing recompute 前的 prefix state prefetch、以及 event-based delayed free。只有当 GPU/NPU profile 都显示稳定 micro-batch / mini-batch e2e 收益，并且不破坏 autograd 路径时，才考虑进入默认路径。

### 2026-06-08：评估 PrefixGrouper 式 two-stage attention 路线

**问题**：verl main 的 PrefixGrouper 通过 group / ungroup 和 two-stage attention 思路减少重复计算，性能收益值得参考。但 prefix-0501 当前业务目标依赖 verl + Megatron actor 训练、TP/PP/SP、prefix 可覆盖 prompt 以外内容、链式复用和 Prefix-Last Restore；直接切换到 PrefixGrouper 路线会改变 attention 组织方式和 logprob restore 边界，集成风险高于当前 TorchRef `build_kv` 热路径优化。

**方案**：短期只吸收其确定性的工程思想：批量化 tensor transform、减少 per-row Python cat/load、保持 autograd-safe layout。two-stage attention 作为下一阶段候选 backend 单独验证，必须覆盖链式复用、TP/PP/SP、suffix-first logprob restore、梯度一致性和 NPU 可运行性后再决定是否合入。

### 2026-06-08：评估 fully tensorized scatter / gather 的 build_kv 实现

**问题**：当前性能分支已将 `build_kv` 从 per-row `store.load + torch.cat` 改为预分配 expanded KV buffer，但仍保留按 batch row 顺序的 Python 循环，以维护 provider-before-reuser 和链式复用语义。进一步完全张量化 scatter/gather 理论上能减少 Python 开销，但链式复用下 reuser 可能依赖前一个 reuser 的 expanded prefix，不能简单并行拷贝所有行。

**方案**：后续若 profile 显示 Python loop 仍是主要瓶颈，再设计 topology-aware build phase：先把 reuse DAG 分层，能够同层并行的 row 用 src/dst index scatter，跨层保持依赖顺序。实现前必须先补链式复用、多级 reuser、TP padding、autograd 梯度路径测试，避免为了张量化破坏复用语义。

### 2026-06-08：评估 NPU/GPU FlashAttention kernel 路线

**问题**：TorchRef 当前是性能优化主线，因为 NPU 上 flash-attention 路径尚未稳定打通。FlashAttention / varlen kernel 理论上会比 TorchRef attention 更快，但 prefix-sharing 下存在 Q 长度、expanded KV 长度、TP padding、RoPE position、Prefix-Last Restore 等额外约束，且 GPU/NPU kernel 行为和可用 API 不完全一致。

**方案**：当前阶段不把 flash-attention 作为性能收益承诺项。等 TorchRef profile 把 `build_kv`、`attention`、`restore` 等阶段拆清楚后，再单独推进 GPU / NPU FlashAttention backend：先做 correctness parity，再做 NPU kernel 可运行性和性能验证；无法稳定通过精度一致性测试前，不替换默认路径。

### 2026-06-02：设计 inter micro-batch sharing 的 store 生命周期与 PP 隔离

**问题**：当前 PP 支持只覆盖单 micro-batch 内 prefix-sharing。`PrefixAttentionStore` / `PrefixDeltanetStore` 仍绑定在单次 `PrefixSharingRuntimeContext` 生命周期内，PP 下也保持 stage-local，不跨 micro-batch、不跨 PP stage 传递 prefix activation。后续若支持 inter micro-batch sharing，需要重新定义缓存生命周期、复用边界和清理策略。

**方案**：后续单独设计 inter micro-batch sharing：引入跨 micro-batch 的稳定 key，明确 forward/backward 生命周期与梯度保留策略，补充 PP rank/stage 隔离和过期清理机制；如果需要跨进程或跨 stage 共享，再单独设计通信/分布式 store，不能复用当前 context-local store 假设。

### 2026-05-30：接入支持 Qwen3.5/Qwen3.6 HybridAttention 的训练引擎

**问题**：当前 `dependency/` 中的 verl / Megatron 快照尚未支持 Qwen3.5/Qwen3.6 的 HybridAttention 训练主路径。prefix-sharing 本轮只在 `prefix-sharing` 侧准备 full gated attention KV 复用和 gated delta net activation state 复用的框架无关抽象，暂不做训练引擎侵入式接入。

**方案**：后续待 verl + MindSpeed + MindSpeed-MM 的 Qwen3.5/Qwen3.6 模型仓和 RL 适配代码进入 `dependency/` 后，补充 thin integration / patch：full attention 接 `PrefixAttentionStore` 的 KV injection，gated delta net 接 `PrefixDeltanetStore` 或等价 cache-param 复用入口，并补齐真实引擎下的精度一致性测试。

当前 `PrefixSharingRuntimeContext` 仍只持有 `PrefixAttentionStore`，这是因为现有 runtime context 只服务 Megatron attention hook。后续接入真实 HybridAttention 训练引擎时，需要扩展 runtime context / runtime state，使其能同时承载 attention KV store 和 GatedDeltaNet state store，并明确两类 store 的生命周期、layer 维度隔离、TP rank 隔离和 context 清理逻辑。

### 2026-05-30：补充 `use_fp8_padding=True` 的 packed layout 对齐与测试

**问题**：TP v1 只支持 CP=1、非 FP8 padding 主路径。verl 在 `use_fp8_padding=True` 时会改变 packed padding 规则：先使用 `lcm(16, align_size)` 对每个序列长度做 padding，再对最后一行追加额外 padding 以满足 Transformer Engine 的总长度对齐要求。当前 `PackedBatchLayout` 尚未实现这套 FP8 padding 规则。

**方案**：后续启用 FP8 前，扩展 `PackedBatchLayout` builder，使其接收 `use_fp8_padding` 或等价配置，并严格对齐 `dependency/verl_v070/verl/models/mcore/util.py::preprocess_packed_seqs()` 的 FP8 padding 逻辑。同时补充 TP=2/4/8 下 `use_fp8_padding=True` 的 layout、restore index 和 backend 测试。
