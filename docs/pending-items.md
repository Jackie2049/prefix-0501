# 遗留事项

本文档统一归档项目开发中的明确遗留事项。后续开发如果发现暂不处理但需要追踪的技术债、兼容性缺口或待验证场景，应按时间倒序追加到这里。

## 当前事项

### 2026-06-16：Qwen3/Qwen3.5 NPU BSHD 主路径真机长跑验证

**问题**：当前 prefix-sharing 已在代码层支持 verl080 monkey patch 下的 BSHD/THD runtime layout，BSHD 由 `use_remove_padding=False` 选择，THD 由 `use_remove_padding=True` 选择。本地测试可覆盖 layout、restore、torch reference backend 和 v080 batch 构建，但无法验证真实 NPU + MindSpeed + Qwen3/Qwen3.5 RL 长跑中的 fused attention、梯度、显存和吞吐表现。

**方案**：后续在真实 NPU 环境中以 Qwen3/Qwen3.5 RL 配置执行长跑验证：先用 `torch_ref` backend 验证精度一致性与稳定性，再验证 `flash_atten_npu`。重点关注 BSHD logits/log_probs restore 坐标、PP last-stage restore、TP/SP 下每个 rank 的 layout 日志，以及 activation checkpointing 重算路径是否稳定进入 prefix-sharing。

### 2026-06-02：设计 inter micro-batch sharing 的 store 生命周期与 PP 隔离

**问题**：当前 PP 支持只覆盖单 micro-batch 内 prefix-sharing。`PrefixAttentionStore` / `PrefixDeltanetStore` 仍绑定在单次 `PrefixSharingRuntimeContext` 生命周期内，PP 下也保持 stage-local，不跨 micro-batch、不跨 PP stage 传递 prefix activation。后续若支持 inter micro-batch sharing，需要重新定义缓存生命周期、复用边界和清理策略。

**方案**：后续单独设计 inter micro-batch sharing：引入跨 micro-batch 的稳定 key，明确 forward/backward 生命周期与梯度保留策略，补充 PP rank/stage 隔离和过期清理机制；如果需要跨进程或跨 stage 共享，再单独设计通信/分布式 store，不能复用当前 context-local store 假设。

### 2026-05-30：接入支持 Qwen3.5/Qwen3.6 HybridAttention 的训练引擎

**问题**：当前 `dependency/` 中的 verl / Megatron 快照尚未支持 Qwen3.5/Qwen3.6 的 HybridAttention 训练主路径。prefix-sharing 本轮只在 `prefix-sharing` 侧准备 full gated attention KV 复用和 gated delta net activation state 复用的框架无关抽象，暂不做训练引擎侵入式接入。

**方案**：后续待 verl + MindSpeed + MindSpeed-MM 的 Qwen3.5/Qwen3.6 模型仓和 RL 适配代码进入 `dependency/` 后，补充 thin integration / patch：full attention 接 `PrefixAttentionStore` 的 KV injection，gated delta net 接 `PrefixDeltanetStore` 或等价 cache-param 复用入口，并补齐真实引擎下的精度一致性测试。

当前 `PrefixSharingRuntimeContext` 仍只持有 `PrefixAttentionStore`，这是因为现有 runtime context 只服务 Megatron attention hook。后续接入真实 HybridAttention 训练引擎时，需要扩展 runtime context / runtime state，使其能同时承载 attention KV store 和 GatedDeltaNet state store，并明确两类 store 的生命周期、layer 维度隔离、TP rank 隔离和 context 清理逻辑。

### 2026-05-30：补充 `use_fp8_padding=True` 的 THD layout 对齐与测试

**问题**：TP v1 只支持 CP=1、非 FP8 padding 主路径。verl 在 `use_fp8_padding=True` 时会改变 packed padding 规则：先使用 `lcm(16, align_size)` 对每个序列长度做 padding，再对最后一行追加额外 padding 以满足 Transformer Engine 的总长度对齐要求。当前 `ThdBatchLayout` 尚未实现这套 FP8 padding 规则。

**方案**：后续启用 FP8 前，扩展 `ThdBatchLayout` builder，使其接收 `use_fp8_padding` 或等价配置，并严格对齐 `dependency/verl_v070/verl/models/mcore/util.py::preprocess_packed_seqs()` 的 FP8 padding 逻辑。同时补充 TP=2/4/8 下 `use_fp8_padding=True` 的 layout、restore index 和 backend 测试。
