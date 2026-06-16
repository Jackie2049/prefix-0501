# prefix-0501 项目总览

> 本文档面向项目团队成员，提供项目整体视角的快速入门指南。

---

## 1. 背景与目标

### 1.1 问题背景

在 `rllm + verl + Megatron` 的 RL 训练 pipeline 中，rollout 阶段需要大量前向计算以得到 logprob / loss。无论是 **step 模式**（按步展开多条轨迹）还是 **tree 模式**（按树分支展开），同一 batch 内往往存在大量**相同前缀**——来自共享的 prompt、历史上下文，或分支前的公共路径。

当前实现通常对每条轨迹独立做完整前向，公共前缀的激活值会被反复计算，形成显著冗余。本项目的目标是在保证训练语义不变的前提下，在 micro-batch 粒度识别并复用这些前缀。

### 1.2 项目目标

在 Agentic RL 场景中（比如 step 模式、tree 模式），实现训练阶段激活值的前缀共享，在真实业务场景中提高训练性能，支持插件化使用并落入社区：

| 目标 | 说明 |
|------|------|
| **激活值前缀复用** | 训练阶段复用公共前缀的 KV 及扩展的层激活值，消除冗余计算 |
| **Agentic RL 场景** | 支持 step 模式、tree 模式等多轨迹展开场景中的前缀共享 |
| **精度与性能** | 精度与原始独立前向**完全一致**，性能显著提升 |
| **易用性** | 少量代码即可在 verl、Megatron 等框架中启用，未来插件化落入社区 |
| **分层设计** | core（语义层）→ backends（执行层）→ integrations（适配层）|

> **Prefix Sharing vs. Prefix Caching**
>
> 推理引擎的 **Prefix Caching**（如 vLLM）只缓存 KV，本质是**单条轨迹内的时序复用**——把历史 KV 存起来供后续 token attend，不同请求之间独立无共享。
>
> 本项目的 **Prefix Sharing** 面向训练场景，是**同一 batch 内多条轨迹间的空间复用**——共同前缀的完整计算可以被多条轨迹共享。Phase 1 先实现 KV 复用，但理论上所有层激活值（hidden states、MLP 中间结果等）都可复用，因为共同前缀在数学上是完全冗余的。
>
> **关键差异**：推理是**自回归串行**生成（token-by-token），只需存储 KV 供后续 token attend；训练是**多条轨迹并行**前向，共同前缀的所有层激活在数学上等价，均可共享复用。"Caching" 强调存储供后续使用，"Sharing" 强调多条轨迹真正共享共同前缀的完整计算结果。

### 1.3 核心挑战

1. **正确性优先于性能**：任何性能优化不得牺牲训练精度
2. **梯度完整性**：prefix KV 缓存必须保留 autograd 计算图，禁止 `detach()`
3. **语义一致性**：采用 **One-Forward + KV Injection + Prefix-Last Restore** 方案确保结果等价
4. **框架边界**：核心算法与框架无关，通过 thin adapter 接入 verl/Megatron

---

## 2. 关键概念

### 2.1 核心概念

| 术语 | 定义 |
|------|------|
| **Reuse Relation** | 描述 reuser 从哪个 provider 复用多长 prefix 的**核心语义单位** |
| **Provider** | 前缀提供方，实际计算并缓存 prefix KV 的样本 |
| **Reuser** | 前缀复用方，复用 provider 的 prefix KV 后继续计算自身的 suffix |
| **Prefix Detection** | 检测 batch 内样本间最长公共前缀的算法过程 |
| **Prefix Group** | 一组序列，他们从同一个 provider 共享长度为prefix_len的前缀 |

> 更多概念和术语见 [`concepts.md`](concepts.md)

### 2.2 数据结构

**PrefixSharingPlan** — 一个 micro-batch 内部单词前向的前缀复用执行计划
```
forward_id: int               # 前向传播标识
micro_batch_id: int            # micro-batch 序号
batch_size: int               # batch 大小
original_lengths: list[int]    # 原始序列长度
reuse_specs: list[PrefixReuseSpec]  # 复用关表
is_provider: list[bool]       # 标记哪些是 provider
group_ids: list[int]          # 样本所属分组 ID
```

**PrefixReuseSpec** — 单条复用关系
```
reuser_idx_in_batch: int      # 复用方在 batch 中的索引
provider_idx_in_batch: int     # 提供方在 batch 中的索引  
prefix_len: int               # 共享前缀长度
```

### 2.3 关键环节

| 环节 | 方案 | 理由 |
|------|------|------|
| 前缀检测 | 在线 Trie | O(total_len) 复杂度，支持流式检测 |
| 精度一致性 | One-Forward + Restore | 确保梯度计算与独立前向完全一致 |
| 激活值复用策略 | 不 detach，保留计算图 | 支持端到端梯度回传 |
| 运行时后端 | `backends/` 加速器解耦 | 统一接口屏蔽 CUDA / CANN 等差异，上层不感知具体加速库 |

---

## 3. 架构与流程

### 3.1 架构全景

#### 3.1.1 模块分层

```
┌────────────────────────────────────────────────────────────────┐
│                    dependency/  依赖层（外部框架）                │
│  ┌──────────────────────┐  ┌────────────────────────────────┐  │
│  │      verl_v070       │  │         megatron_v0150         │  │
│  │  - 上游框架代码        │  │      - 上游框架代码              │  │
│  │  - 少量侵入式修改      │  │      - 少量侵入式修改             │  │
│  │  - 未来：patch或PR合入 │  ｜     - 未来：patch或PR合入       │  │
│  └──────────────────────┘  └────────────────────────────────┘  │
├────────────────────────────────────────────────────────────────┤
│                    integrations/  框架适配层（薄）                │
│  ┌─────────────────┐  ┌─────────────────────────────────────┐ │
│  │ verl_integration│  │      megatron_integration           │ │
│  │ - patch 模块     │  │  - helper 函数                       │ │
│  │ - context 管理器 │  │  - 使能入口                          │ │
│  └─────────────────┘  └─────────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────┤
│                    core/  核心语义层（框架无关）                │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐ │
│  │   config   │ │  detector  │ │  planner   │ │batch_meta │ │
│  │ 配置验证    │ │ Trie 检测   │ │ 执行计划     │ │ 计划数据   │ │
│  └────────────┘ └────────────┘ └────────────┘ └───────────┘ │
├────────────────────────────────────────────────────────────────┤
│                    backends/  硬件执行层                        │
│  ┌──────────────────────┐  ┌─────────────────────────────┐   │
│  │   torch_reference/   │  │      cuda_kernels/           │   │
│  │  - PyTorch 参考实现   │  │  - CUDA 高性能算子（可选）    │   │
│  │  - 功能完整，易调试   │  │  - 生产环境高性能执行        │   │
│  └──────────────────────┘  └─────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 类关系概览

```
PrefixSharingConfig           PrefixDetector (abstract)
       │                              │
       │                    ┌──────────┘
       │                    │
       ▼                    ▼
       └──────────►  TriePrefixDetector
                              │
                              │ detect()
                              ▼
                    PrefixDetectionResult ──────► PrefixSharingPlanner
                                                          │
                                                          │ plan()
                                                          ▼
                                              PrefixSharingPlan
```

#### 3.1.3 目录边界

| 目录 | 职责 | 改动策略 |
|------|------|----------|
| `core/` | 框架无关的核心语义：配置、检测、规划 | **主要开发区域** |
| `backends/` | 硬件执行：PyTorch 参考实现、CUDA 优化 | 按需扩展 |
| `integrations/` | 框架适配：verl/Megatron 接入辅助 | 保持 thin |
| `dependency/` | verl/Megatron 源码快照 | 谨慎改动，优先插件化 |

### 3.2 Prefix-Sharing 文件组织

以下是 `prefix-sharing/` 模块内部的完整文件结构，按分层架构组织：

```
prefix-sharing/
├── prefix_sharing/
│   ├── __init__.py                      # 包入口，导出核心类
│   │
│   ├── core/                            # 核心语义层（框架无关）
│   │   ├── __init__.py
│   │   ├── config.py                    # PrefixSharingConfig 配置类
│   │   ├── detector.py                  # 前缀检测抽象基类
│   │   ├── trie_detector.py             # TriePrefixDetector 实现
│   │   ├── planner.py                   # PrefixSharingPlanner / PrefixSharingPlan 执行规划
│   │   ├── batch_trim.py                # Batch 裁剪工具
│   │   └── logprob.py                   # Prefix-Last Restore 逻辑
│   │
│   ├── backends/                        # 硬件执行层
│   │   ├── __init__.py
│   │   ├── torch_ref/                   # PyTorch 参考实现
│   │   │   ├── __init__.py
│   │   │   └── attention.py             # TorchReferenceBackend
│   │   └── cuda_kernels/                # CUDA 高性能算子（可选）
│   │       └── (预留)
│   │
│   └── integrations/                    # 框架适配层
│       ├── __init__.py
│       ├── context.py                   # PrefixSharingRuntimeContext
│       ├── verl_mcore.py                # verl + Megatron 集成适配
│       ├── megatron_runtime.py          # Megatron attention hook
│       └── megatron_attention.py        # Megatron 注意力补丁管理
│
└── tests/                               # 测试套件
    ├── unit/                            # 单元测试
    │   ├── test_config.py
    │   ├── test_trie_detector.py
    │   ├── test_planner.py
    │   └── test_torch_backend.py
    └── integration/                     # 集成测试
        └── (预留)
```

**各目录职责与核心文件说明**：

| 路径 | 核心类/函数 | 职责 |
|------|------------|------|
| `core/config.py` | `PrefixSharingConfig` | 配置验证与默认值管理 |
| `core/detector.py` | `PrefixDetector` (ABC) | 前缀检测抽象接口 |
| `core/trie_detector.py` | `TriePrefixDetector` | 基于 Trie 的前缀检测实现 |
| `core/planner.py` | `PrefixSharingPlanner` / `PrefixSharingPlan` | 将检测结果转为 Batch 级执行计划 |
| `core/batch_trim.py` | `trim_inputs/labels/masks` | 裁剪输入序列的 prefix |
| `core/logprob.py` | `restore_prefix_last_logprobs` | Prefix-Last 位置的 logprob 恢复 |
| `backends/torch_ref/attention.py` | `TorchReferenceBackend` | PyTorch 参考实现，支持 autograd |
| `backends/batch_layout.py` | `ThdBatchLayout` / `BshdBatchLayout` | 描述 attention backend 看到的 THD/BSHD runtime 坐标 |
| `integrations/context.py` | `PrefixSharingRuntimeContext` | 运行时上下文管理（thread-local） |
| `integrations/context.py` | `prefix_sharing_runtime_context` | 绑定当前 prefix sharing runtime context |
| `integrations/verl_mcore.py` | `build_prefix_sharing_micro_batch` | prefix sharing micro-batch 构建 |
| `integrations/verl_mcore.py` | `restore_suffix_first_log_probs_from_prefix` | 从 prefix-last logits 恢复 suffix-first logprob |
| `integrations/megatron_runtime.py` | `maybe_run_prefix_sharing_attention` | Megatron attention hook 入口 |
| `integrations/megatron_attention.py` | `MegatronAttentionIntegration` | Megatron attention 补丁安装/卸载 |

---

### 3.3 整体流程

#### 3.3.1 数据流图

```
输入 Micro-batch
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Prefix Detection                               │
│  ─────────────────                                      │
│  TriePrefixDetector.detect(input_ids)                   │
│  └── 找出 batch 内样本间的复用关系                        │
│  └── 输出: PrefixDetectionResult                        │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Step 2: Planning                                       │
│  ────────────                                           │
│  PrefixSharingPlanner.plan()                              │
│  └── 将检测结果转换为完整执行计划                         │
│  └── 输出: PrefixSharingPlan                              │
│       ├── provider/reuser 映射                          │
│       ├── prefix 裁剪策略                               │
│       └── restore 位置计算                              │
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Step 3: Prefix Sharing Forward                         │
│  ────────────────────────                               │
│  backend.build_kv() + backend.attention()                 │
│  ├── 裁剪：移除 reuser 的 prefix tokens                  │
│  ├── 存储：provider KV 存入 PrefixAttentionStore (按 layer_id/tp_rank)│
│  ├── 注入：从 PrefixAttentionStore 加载 prefix KV 到 reuser    │
│  └── 执行：单次前向传播（One-Forward）                    │
│                                                         │
│  HybridAttention 预留路径：                              │
│  ├── full gated attention：KV injection 后应用当前 token gate│
│  └── gated deltanet：PrefixDeltanetStore 复用 prefix state│
└─────────────────────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│  Step 4: Prefix-Last Restore                             │
│  ───────────────────────                                │
│  在 prefix 最后一个 token 位置恢复独立计算                 │
│  ├── 确保每个样本 suffix 独立计算                        │
│  └── 保证输出与独立前向传播一致                          │
└─────────────────────────────────────────────────────────┘
     │
     ▼
输出: logits / loss / gradients（与独立前向等价）
```

#### 3.3.2 组件调用时序

以下为 integration 层与 dependency 层的完整调用时序，展示了从 batch 准备到 attention 计算的全链路交互：

```
┌─────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│  Verl   │     │   verl_mcore     │     │      context        │     │ megatron_runtime │
│  Actor  │     │   (integrations) │     │   (integrations)    │     │  (integrations)  │
└────┬────┘     └────────┬─────────┘     └──────────┬──────────┘     └────────┬────────┘
     │                   │                        │                         │
     │  1. build_prefix_sharing_micro_batch()     │                         │
     │──────────────────────────────────────────>│                         │
     │                   │                        │                         │
     │                   │  2. PrefixSharingPlanner.plan()                      │
     │                   │  3. 裁剪 batch                                        │
     │                   │  4. 构建 PrefixSharingRuntimeState                    │
     │                   │                        │                         │
     │  返回 (trimmed_batch, runtime_state)       │                         │
     │<──────────────────────────────────────────│                         │
     │                   │                        │                         │
     │  5. with prefix_sharing_runtime_context(runtime_state):               │
     │────────────────────────────────────────────────────────────────────>│
     │                   │                        │  6. 创建 PrefixSharingRuntimeContext
     │                   │                        │  7. ContextVar.set(ctx)   │
     │                   │                        │     初始化 PrefixAttentionStore  │
     │                   │                        │                         │
     │                   │                        │  8. yield ctx            │
     │<────────────────────────────────────────────────────────────────────│
     │                   │                        │                         │
     │  9. forward_fn(model, ...)  ────────>  进入 Megatron SelfAttention     │
     │                   │                        │                         │
     │                   │                        │  10. maybe_run_prefix_sharing_attention()
     │                   │                        │<────────────────────────│
     │                   │                        │  11. ctx = current_prefix_sharing_context()
     │                   │                        │  12. backend.build_kv(ctx.store, ...)
     │                   │                        │  13. backend.attention(...)
     │                   │                        │                         │
     │                   │                        │  14. 返回 (output, bias) │
     │<────────────────────────────────────────────────────────────────────│
     │                   │                        │  15. Context 退出        │
     │                   │                        │     store.close()        │
     │                   │                        │                         │
     │  16. restore_suffix_first_log_probs_from_prefix()  │                         │
     │──────────────────────────────────────────>│                         │
     │                   │  17. 从 ctx 读取 restore_slots                    │
     │                   │  18. 复制并重新计算 logprob                        │
     │  返回 restored log_probs                   │                         │
     │<──────────────────────────────────────────│                         │
```

**关键调用说明：**

| 步骤 | 调用方 | 被调用方 | 职责 |
|:----:|:-------|:---------|:-----|
| 1-4 | `MegatronPPOActor` | `build_prefix_sharing_micro_batch` | 前缀检测、batch 裁剪、构建 RuntimeState |
| 5-8 | `verl_mcore` | `prefix_sharing_runtime_context` (ContextManager) | 初始化线程安全的运行时上下文，包括 `PrefixAttentionStore` |
| 9 | `verl actor` | `Megatron SelfAttention.forward` | 进入 Megatron attention 计算 |
| 10-14 | `SelfAttention` | `maybe_run_prefix_sharing_attention` | 拦截 attention，执行 RoPE + KV 扩展 + attention |
| 15 | ContextManager | `ctx.store.close()` | 清理 KV cache，释放资源 |
| 16-18 | `MegatronPPOActor` | `restore_suffix_first_log_probs_from_prefix` | Prefix-Last Restore，恢复 reuser 的 logprob |

#### 3.3.3 阶段说明

一次完整的 Prefix Sharing 前向传播包含以下阶段：

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|----------|
| **Detect** | `input_ids: [B, L]` | `groups`, `provider_index`, `prefix_lens` | Trie 构建与匹配 |
| **Plan** | `detection_result` | `prefix_sharing_plan` | 计算裁剪长度、restore 位置 |
| **Crop** | `input_ids`, `plan` | `cropped_ids` | 移除 reuser 的 prefix |
| **Forward** | `cropped_ids` | `hidden_states`, `K_cache`, `V_cache` | 单次前向传播 |
| **Inject** | `K/V_cache`, `plan` | `injected_cache` | 将 prefix KV 复制到 reuser 槽位 |
| **Restore** | `hidden_states`, `plan` | `restored_states` | prefix-last 位置恢复独立计算 |
| **Head** | `restored_states` | `logits` | 输出头计算最终结果 |

#### 3.3.4 HybridAttention 预适配

Qwen3.5/Qwen3.6 的 HybridAttention 同时包含 full gated attention 与 GatedDeltaNet。prefix-sharing 侧保持框架无关，只表达当前明确需要支持的两类可复用状态：

| 路径 | 可复用状态 | 当前实现 |
|------|------------|----------|
| full gated attention | prefix K/V | 使用 `PrefixAttentionStore` 存储 `StoredAttentionKV`；attention 输出后应用当前 kept token 的 gate，gate 不缓存 |
| GatedDeltaNet | prefix recurrent state / conv state / cache params | 使用 `PrefixDeltanetStore` 存储 `StoredDeltanetState`；reference backend 用 recurrent trajectory 验证 provider prefix state → reuser suffix initial state |

后续当支持 Qwen3.5/Qwen3.6 的 verl + MindSpeed + MindSpeed-MM 训练引擎进入 `dependency/` 后，只在 integration 层补 thin patch：full attention 接 KV injection，GatedDeltaNet 接 activation/cache-param 复用入口。

#### 3.3.5 与训练框架的集成点

```
┌──────────────────────────────────────────┐
│           RL Training Loop               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │  Rollout │  │  Advantage│  │ Update  │  │
│  │  Phase   │  │  Compute │  │  Phase  │  │
│  └────┬────┘  └────┬────┘  └────┬────┘  │
│       │            │            │       │
│       └────────────┴────────────┘       │
│                    │                     │
│                    ▼                     │
│  ┌─────────────────────────────────────┐ │
│  │   verl worker / Megatron engine     │ │
│  │  ┌───────────────────────────────┐  │ │
│  │  │  prefix-sharing integration   │  │ │
│  │  │  - 拦截 forward 调用           │  │ │
│  │  │  - 启用 PrefixSharingPlanner  │  │ │
│  │  │  - 传递 batch_meta 到 backend  │  │ │
│  │  └───────────────────────────────┘  │ │
│  └─────────────────────────────────────┘ │
└──────────────────────────────────────────┘
```

---

## 4. 模块设计

### 4.1 Dependency 层（侵入式修改）

Dependency 层对外部框架（verl、Megatron）源码进行**最小化、可恢复**的修改，通过 `try/except` 实现可选加载，确保 prefix-sharing 模块不可用时自动回退到原始行为。

#### 4.1.1 verl_v070 - Megatron Actor 集成点

**文件**: `dependency/verl_v070/verl/workers/actor/megatron_actor.py`

**导入与回退机制（line 61-70）**:

```python
try:
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_mcore import (
        build_prefix_sharing_micro_batch,
        restore_suffix_first_log_probs_from_prefix,
    )
except ModuleNotFoundError:
    prefix_sharing_runtime_context = None
    build_prefix_sharing_micro_batch = None
    restore_suffix_first_log_probs_from_prefix = None
```

用 `try/except` 实现可选依赖，模块缺失时设为 `None`，自动回退到原始框架行为。

**Micro-batch 准备（line 585-593）**:

```python
prefix_sharing_runtime_state = None
if build_prefix_sharing_micro_batch is not None:
    batch, prefix_sharing_runtime_state = build_prefix_sharing_micro_batch(
        batch,
        self.config,
        self.tf_config,
    )
```

在每个 micro-batch 进入模型前调用，进行前缀检测和 batch 裁剪。返回 `(trimmed_batch, runtime_state)`，后者包含执行计划和恢复位置信息。

**Context 包装（line 680-681）**:

```python
prefix_context = prefix_sharing_runtime_context or nullcontext
with prefix_context(prefix_sharing_runtime_state):
    output = forward_fn(...)
```

使用 `ContextVar` 在 thread-local 存储运行时上下文，Megatron attention 层通过 `current_prefix_sharing_context()` 访问状态。

**Logprob 恢复（line 669-675）**:

```python
if restore_suffix_first_log_probs_from_prefix is not None:
    log_probs = restore_suffix_first_log_probs_from_prefix(
        logits_bak,
        label,
        log_probs,
        vocab_parallel_log_probs_from_logits,
    )
```

在计算 log_probs 后，恢复 reuser 第一个 suffix token 的 logprob，确保与独立前向传播结果一致。

**兼容性检查（line 631-632）**:

```python
if prefix_sharing_runtime_state is not None:
    raise RuntimeError("prefix sharing phase 1 requires actor fused kernels to be disabled")
```

Phase 1 实现与 fused kernels 不兼容，若同时启用则报错。

#### 4.1.2 megatron_v0150 - Attention Hook

**文件**: `dependency/megatron_v0150/megatron/core/transformer/attention.py`

**Hook 插入点（line 803-833）**:

```python
try:
    from prefix_sharing.integrations.megatron_runtime import maybe_run_prefix_sharing_attention
    
    prefix_sharing_output = maybe_run_prefix_sharing_attention(
        self, query, key, value,
        attention_mask, rotary_pos_emb, packed_seq_params,
        mscale=...,
    )
except ModuleNotFoundError:
    prefix_sharing_output = None

if prefix_sharing_output is not None:
    return prefix_sharing_output  # 拦截成功

# 继续原始 attention 逻辑...
```

在 QKV 计算完成后、RoPE 应用之前插入 hook。拦截成功则直接返回 `(output, bias)`，否则继续原始 `RoPE → core_attention → linear_proj` 流程。

---

### 4.2 Integrations 层（框架适配）

Integrations 层实现具体的框架适配逻辑，可被独立测试和迭代，与 Dependency 层通过清晰的接口契约交互。

#### 4.2.1 verl_mcore.py - Batch 准备与恢复

**文件**: `prefix-sharing/prefix_sharing/integrations/verl_mcore.py`

**build_prefix_sharing_micro_batch 核心流程**:

```
1. 配置检查
   └── PrefixSharingConfig.from_raw(actor_config.prefix_sharing_config)
   
2. 前置条件检查（6条路径）
   ├── Path 1: config.enable_prefix_sharing=False → 返回 (batch, None)
   ├── Layout: use_remove_padding=True → THD；use_remove_padding=False → BSHD
   ├── Path 3: multi_modal_inputs 非空 → RuntimeError
   ├── Path 4: 非 2D 张量 → RuntimeError
   ├── Path 5: 无共享检测到 → 返回 (batch, None)
   └── Path 6: 有共享 → 继续处理
   
3. 前缀检测与规划
   └── PrefixSharingPlanner(config).plan(sequences)
   
4. Batch 裁剪
   └── 根据 input_keep_ranges 保留各 row 的 suffix
   
5. 构建 ThdBatchLayout
   └── valid_lengths / padded_lengths / cu_seqlens / position_ids / valid_token_mask
   
6. 组装 RuntimeState
   └── PrefixSharingRuntimeState(
         prefix_sharing_plan=plan,
         backend=TorchReferenceBackend(),
         batch_runtime_layout=...,  # 真实 THD runtime 坐标
         parallel_info=...,  # 当前并行信息；Megatron 场景中为 MegatronParallelInfo
       )
   
7. 进入 runtime context
   └── 根据 plan.prefix_last_restore + ThdBatchLayout 派生 PackedPrefixLastRestoreIndex
```

**关键数据结构**:

| 类/结构 | 职责 |
|---------|------|
| `PrefixSharingPlan` | 框架无关的执行计划，含裁剪范围、cumsum 长度、restore 规格 |
| `ThdBatchLayout` | packed batch 的运行时坐标，包含 valid/padded lengths、position ids 和 valid mask |
| `MegatronParallelInfo` | Megatron `parallel_state` 的 TP/CP/PP rank-size 快照 |
| `PrefixSharingRuntimeState` | 跨层传递的运行时状态，包含 plan、backend、batch runtime layout 和 Megatron 并行信息 |
| `PrefixLastRestoreIndex` | runtime context 中由 plan + layout 派生出的 restore 读写索引；THD 为 1D packed index，BSHD 为 `(batch, seq)` token index |

**restore_suffix_first_log_probs_from_prefix 逻辑**:

```python
def restore_suffix_first_log_probs_from_prefix(logits, labels, log_probs, vocab_fn):
    ctx = current_prefix_sharing_context()
    if ctx is None or not ctx.prefix_last_restore_indices:
        return log_probs
    if not ctx.parallel_info.is_pipeline_last_stage:
        return log_probs
    
    restored = log_probs.clone()
    for index in ctx.prefix_last_restore_indices:
        # 从 provider 的 prefix-last 位置取 logits
        provider_logits = logits_at(index.provider_token_index)
        # 用 reuser 的第一个 suffix label 计算 logprob
        reuse_label = labels_at(index.reuse_token_index)
        write_logprob(restored, index.reuse_token_index, vocab_fn(provider_logits, reuse_label))
    return restored
```

#### 4.2.2 context.py - 运行时上下文管理

**文件**: `prefix-sharing/prefix_sharing/integrations/context.py`

```python
_current_context: ContextVar[PrefixSharingRuntimeContext | None] = ContextVar(
    "prefix_sharing_context", default=None
)

@dataclass
class PrefixSharingRuntimeContext:
    prefix_sharing_plan: PrefixSharingPlan   # 执行计划
    batch_runtime_layout: ThdBatchLayout    # THD runtime 坐标
    parallel_info: MegatronParallelInfo
    store: PrefixAttentionStore              # 每层 attention KV 缓存
    backend: Any | None = None                # 后端实现
    prefix_last_restore_indices: list[PackedPrefixLastRestoreIndex] = field(default_factory=list)

@contextmanager
def prefix_sharing_runtime_context(runtime_state):
    store = PrefixAttentionStore()
    ctx = PrefixSharingRuntimeContext(runtime_state, store)
    token = _current_context.set(ctx)  # 设置 thread-local
    try:
        yield ctx
    finally:
        _current_context.reset(token)
        ctx.store.close()  # 清理 KV cache

def current_prefix_sharing_context() -> PrefixSharingRuntimeContext | None:
    return _current_context.get()  # 线程安全读取
```

**设计要点**:
- 使用 Python `ContextVar` 实现线程安全，支持异步/并发
- `PrefixAttentionStore` 按 `(forward_id, micro_batch_id, layer_id, batch_idx, prefix_state_type, tp_rank)` 索引 KV
- 物理 PP 下 store 保持 stage-local；Prefix-Last Restore 只在 last PP stage 执行
- Context 退出时自动调用 `store.close()` 清理资源

#### 4.2.3 megatron_runtime.py - Attention 拦截执行

**文件**: `prefix-sharing/prefix_sharing/integrations/megatron_runtime.py`

**maybe_run_prefix_sharing_attention 执行流程**:

```python
def maybe_run_prefix_sharing_attention(
    attention_module, query, key, value,
    attention_mask, rotary_pos_emb, packed_seq_params, *, mscale=1.0
):
    # 1. 检查上下文
    ctx = current_prefix_sharing_context()
    if ctx is None:
        return None  # 无激活 context，走原始路径
    
    # 2. 验证前置条件
    assert packed_seq_params.qkv_format == "thd"
    assert rotary_pos_emb is not None
    assert ctx.batch_runtime_layout.position_ids is not None
    
    # 3. 应用 RoPE（使用 position_ids 对齐真实 packed tensor）
    query, key = _apply_positioned_rope(
        attention_module, query, key,
        q_pos_emb, k_pos_emb,
        ctx.batch_runtime_layout.position_ids,
        mscale=mscale
    )
    
    # 4. 构建 KV Cache（核心）
    tp_rank = _tensor_parallel_rank()
    layer_id = attention_module.layer_number
    expanded_key, expanded_value = backend.build_kv(
        key, value,
        ctx.store,              # KV 存储
        ctx.prefix_sharing_plan,
        batch_runtime_layout=ctx.batch_runtime_layout,
        layer_id=layer_id,
        tp_rank=tp_rank,
    )
    
    # 5. 执行 Attention
    core_attn_out = backend.attention(
        query, expanded_key, expanded_value,
        ctx.prefix_sharing_plan,
        batch_runtime_layout=ctx.batch_runtime_layout,
        attention_mask=attention_mask,
    )
    
    # 6. 输出投影
    return attention_module.linear_proj(core_attn_out.reshape(...))
```

**_apply_positioned_rope 关键逻辑**:

```python
def _apply_positioned_rope(..., position_ids, ...):
    positions = position_ids.to(device=query.device)
    max_needed = positions.max().item() + 1
    
    # 扩展 pos_emb：原始 pos_emb 只覆盖 [0, max_seqlen_q)
    # 但 prefix sharing 保留原始位置（如 suffix 从位置 75 开始）
    if max_needed > q_pos_emb.shape[0]:
        # 利用 RoPE 线性特性：freq[p] = p * inv_freq
        # 从 pos_emb[1] - pos_emb[0] 恢复 step
        step = q_pos_emb[1:2] - q_pos_emb[0:1]
        # 计算扩展位置的编码
        extra_emb = extrapolate_positions(step, n_extra)
        q_pos_emb = torch.cat([q_pos_emb, extra_emb])
    
    # 按 position_ids 选择对应频率
    q_freqs = q_pos_emb.index_select(0, positions)
    query = apply_rotary_pos_emb(query.unsqueeze(1), q_freqs, ...).squeeze(1)
    return query, key
```

#### 4.2.4 两层交互总结

```
┌─────────────────────────────────────────────────────────────────┐
│  DEPENDENCY 层（最小侵入）                                       │
│  ─────────────────────                                           │
│  megatron_actor.py:                                              │
│    1. 尝试导入 integrations 函数（失败则设为 None）              │
│    2. 调用 build_prefix_sharing_micro_batch()                    │
│       → 获得裁剪 batch + PrefixSharingRuntimeState               │
│    3. with prefix_sharing_runtime_context(state):                 │
│         通过 ContextVar 激活运行时上下文                          │
│         进入 Megatron forward                                     │
│    4. Megatron SelfAttention.forward() 中被拦截                  │
│    5. 调用 restore_suffix_first_log_probs_from_prefix() 恢复 logprob         │
│                                                                  │
│  attention.py:                                                   │
│    1. QKV 计算完成后，调用 maybe_run_prefix_sharing_attention()   │
│    2. 若返回非 None，直接返回结果                                │
│    3. 若返回 None，继续原始 RoPE → core_attention → linear_proj   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  INTEGRATIONS 层（具体实现）                                     │
│  ─────────────────────                                           │
│  verl_mcore.py:                                                  │
│    - prepare: 前缀检测 → 裁剪 batch → 构建 RuntimeState           │
│    - restore: 仅 last PP stage 从 context 读取 slots → 复制 provider logits │
│                                                                  │
│  context.py:                                                     │
│    - ContextVar 管理，thread-local 存储                           │
│    - MegatronParallelInfo: 记录 TP/CP/PP rank-size 和 PP stage    │
│    - PrefixAttentionStore: 按 (layer, tp_rank, batch_idx) 索引 KV │
│                                                                  │
│  megatron_runtime.py:                                            │
│    - 拦截 attention，执行 RoPE + build_kv + attention             │
│    - _apply_positioned_rope: 处理非连续位置编码                     │
│                                                                  │
│  backends/torch_ref.py:                                          │
│    - build_kv: provider 存储 KV，reuser 加载并拼接                │
│    - attention: flash-attn 风格计算                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 延伸阅读

## 延伸阅读

| 文档 | 内容 | 适合场景 |
|------|------|----------|
| [`AGENTS.md`](AGENTS.md) | 仓库级开发规范 | 所有开发工作 |
| [`prefix-sharing/AGENTS.md`](prefix-sharing/AGENTS.md) | 模块级详细规范 | core/integration 开发 |
| [`docs/concepts.md`](docs/concepts.md) | 当前概念、术语和核心语义约定 | 语义对齐、命名分歧、术语查询 |
| [`docs/overview.md`](docs/overview.md) | 当前架构说明 | 理解模块关系和数据流 |
| [`docs/overview.puml`](docs/overview.puml) | 架构类图（PlantUML）| 详细架构设计 |
| [`docs/pending-items.md`](docs/pending-items.md) | 当前明确遗留事项 | 查兼容性缺口和后续待验证场景 |
| [`docs/legacy/`](docs/legacy/) | 历史 `doc-*` 文档归档 | 理解历史方案和被推翻方案 |
