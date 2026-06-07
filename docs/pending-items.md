# 遗留事项

本文档统一归档项目开发中的明确遗留事项。后续开发如果发现暂不处理但需要追踪的技术债、兼容性缺口或待验证场景，应按时间倒序追加到这里。

## 当前事项

### 2026-06-07：对齐官方 verl + Megatron 的 Qwen3.5/Qwen3.6-27B 依赖版本

**问题**：当前 `dependency/` 里的 verl / Megatron 快照还停留在旧版本，不能作为 Qwen3.5/Qwen3.6-27B RL 训练的可靠基线。后续如果要基于官方 `verl + Megatron + vLLM` 技术路线接入 prefix-sharing，需要先升级并锁定一套能跑通 Qwen3.5/Qwen3.6-27B baseline 的配套依赖，包括 verl、Megatron-LM / Megatron-Core、Megatron-Bridge 或 mbridge、MindSpeed、MindSpeed-MM、vLLM / vLLM-Ascend、CANN / CUDA 等。

**方案**：后续先以官方 Qwen3.5 NPU Megatron recipe 和官方 Qwen3.5/Qwen3.6 Megatron-Bridge 支持为基准，建立无 prefix-sharing baseline；确认 Qwen3.5/Qwen3.6-27B 在 CUDA 和 CANN 环境下的可运行依赖矩阵后，再迁移 prefix-sharing 的 thin integration / patch。升级过程中必须保持 `prefix-sharing` 核心逻辑在本模块内沉淀，`dependency/` 只保留必要使能入口和训练引擎 hook。

### 2026-06-06：补充 Qwen3.5 NPU 非 packed 路径的 DenseBatchLayout 支持

**问题**：当前 prefix-sharing 主路径假设 Megatron actor 使用 packed / THD token layout：`PrefixSharingPlan`、`PackedBatchLayout`、RoPE、KV split / store / load、prefix-last restore 都围绕 packed 1D 坐标设计。但官方 Qwen3.5 NPU recipe 中 Gated DeltaNet 当前不支持 packed sequence，训练配置需要保持 `use_remove_padding=False` / `use_dynamic_bsz=False`。这意味着 Qwen3.5 NPU 落地时可能走 dense BSHD layout，现有 packed-only runtime 坐标无法直接复用。

**方案**：后续接入 Qwen3.5/Qwen3.6 NPU 训练引擎时，先明确检测 `use_remove_padding=False` 场景并避免静默错跑；再设计 dense runtime layout 支持。建议引入 `DenseBatchLayout`，并在更上层以 `BatchRuntimeLayout` 或等价协议统一 packed / dense 两类真实 tensor 坐标：`PackedBatchLayout` 继续服务 THD packed 路径，`DenseBatchLayout` 表达 dense BSHD 下的 batch / seq 坐标、valid token mask、position ids 和 prefix-last restore dense index。`PrefixSharingPlan` 仍只保存 prefix-sharing 逻辑语义，不混入具体 packed / dense runtime layout 字段。

第一版 dense path 应优先验证精度一致性，覆盖 full gated attention KV 复用和 GatedDeltaNet state 复用在 `use_remove_padding=False` 下的 logprob / loss / gradient 对齐；性能优化和重新启用 packed GatedDeltaNet 可以作为后续事项。

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
