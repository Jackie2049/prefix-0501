# 工作进展记录

> **规则**: 所有工作进展按时间倒序记录，最新在前。

---

## 2026-05-11 20:49 补充 5：方案升级为三层架构

### 背景

对当前 One-Forward + KV Injection 设计进行二次审视后，确认核心方向可行，但旧方案存在工程边界不清的问题：
- 过度依赖 `position_ids`，但 verl mcore THD 普通 RoPE 路径实际传入 `position_ids=None`
- RoPE patch 只覆盖 `_apply_rotary_pos_emb_thd()`，无法覆盖 `apply_rope_fusion=True` 时的 fused THD 路径
- logits / label / loss mask / output restore 在 prefix token 被裁剪后缺少明确映射
- 方案容易绑定 CUDA TransformerEngine，不利于同时支持 CUDA GPU 和 CANN NPU
- cache 生命周期、backend 能力边界、MVP 约束需要显式化

### 完成事项

1. 更新 `doc-designs.md`，新增最新设计章节：`2026-05-11 20:49 详细设计：三层架构 + Metadata + 后端适配方案`

2. 新设计将 prefix sharing 拆为三层：
   - **通用语义层**：`PrefixSharingBatchMeta`、detector、planner、cache、mapping，不绑定具体硬件后端
   - **模型集成层**：verl mcore preprocess/postprocess、Megatron attention hook、RoPE offset、logprob/loss 对齐
   - **后端适配层**：`TorchReferenceBackend`、CUDA backend、CANN NPU backend 等，通过统一接口消费 metadata

3. 明确 MVP 约束：
   - 普通 text GPT / causal LM
   - PP=1
   - CP=1
   - `apply_rope_fusion=False`
   - `fused_single_qkv_rope=False`
   - 精度测试在 eval mode，或 train mode 关闭 dropout 且固定 RNG

4. 明确未来扩展路径：
   - CANN NPU 通过独立 backend 适配，不复用 CUDA TE 细节
   - CP / PP 作为专项 backend 或专项 plan 扩展
   - fused kernel 必须声明能力并对齐 `TorchReferenceBackend`

### 关键设计决策

- 保留 One-Forward + KV Injection 核心策略，不推翻旧方案的主方向
- 新增 `PrefixSharingBatchMeta` 作为所有模块共享的唯一语义来源
- 不再把 `position_ids` 作为 THD RoPE 修正的主要机制，改为 metadata 驱动 per-sequence RoPE offset
- 不再假设 `PackedSeqParams` 支持独立字段就代表底层 kernel 语义已验证，每个 backend 必须单独验证 `q_len != kv_len`
- PrefixTrain_dev 主流程代码只作为参考迁移对象，不再描述为可直接复用的可靠实现

### 下一步

- 按新设计先实现 `PrefixSharingConfig.validate()`、`PrefixSharingBatchMeta`、planner 和 `TorchReferenceBackend`
- 用 reference backend 验证裁剪、RoPE offset、KV 注入、label/logprob/output mapping
- 再进入 Megatron 单层 hook 和 verl logprob 集成

---

## 2026-05-10 补充 4：设计方案审视 — 发现 RoPE 与 PP 关键问题

### 审视范围

深入阅读 PrefixTrain_dev `flex_ops.py` / `training.py`、Megatron v0.15 `attention.py` / `rope_utils.py` / `gpt_model.py`、verl v0.7 `model_forward.py` / `util.py` / `megatron_actor.py`，对当前 One-Forward + KV Injection 方案进行系统性审视。

### 发现的严重问题

**1. RoPE 位置编码不连续（严重）**
- 去除 prefix tokens 后，suffix-only 序列的 RoPE 默认从位置 0 开始
- Megatron v0.15 THD 模式下 `_apply_rotary_pos_emb_thd` CASE 2 不支持 per-sequence 偏移
- 会导致 suffix token 的 attention score 位置信息错误，破坏因果性和语义
- **PrefixTrain_dev 未暴露此问题**——其 position_ids 是随机生成的模拟数据，不依赖真实位置

**2. PP 不支持（严重）**
- 原方案对比表声称 One-Forward "PP 兼容简单"，这是错误的
- PP 下同一序列的不同层在不同 GPU stage 上，stage 0 缓存的 prefix KV 无法在 stage 1 注入
- PrefixTrain_dev 通过**直接魔改 Megatron 核心代码**（flex_model.py 跨 stage activation 共享）实现 PP 支持
- 我们的 patch 方案无法做到跨 stage KV 传递

### 修复方案

| 问题 | 修复措施 |
|------|----------|
| RoPE 不连续 | monkey-patch `_apply_rotary_pos_emb_thd`，传入 `position_offsets` 参数，为每个序列使用正确的 RoPE 偏移 |
| PP 不支持 | MVP 阶段明确限制 PP=1；PP 支持作为后续专项任务 |
| Cache 污染 | PrefixKVCache 增加 micro-batch ID 隔离 |

### 文档更新

1. 更新 `doc-designs.md`：
   - 修正 PP 兼容性描述（"简单"→"均不支持，MVP 阶段 PP=1"）
   - 添加 position_ids 处理说明（suffix-only 序列从 prefix_len 开始）
   - 添加 RoPE patch 模块（`patches/megatron_rope.py`）
   - 更新 attention forward 流程（步骤 5 添加 RoPE 修正）
   - 更新技术风险表（添加 RoPE、PP、cache 污染）
   - 更新实现步骤（Step 2 添加 position_ids，Step 3 添加 RoPE 验证，Step 7 标注 PP 限制）

2. 更新 `doc-progress.md` — 本记录

---

## 2026-05-10 补充 3：方案从 Two-Phase Forward 改为 One-Forward + KV Injection

### 背景

设计文档中原来的 Two-Phase Forward 方案（将 forward 拆为 prefix forward + suffix forward 两个阶段）是错误的。经用户指出后，调研确认三个参考项目（PrefixTrain_dev、flash-preference、dpo-prefix-sharing）**全部采用一次 forward** 实现 prefix sharing。

### 调研结论

- **PrefixTrain_dev**: 在 `FlexFlashAttentionOp.forward()` 中，一次 forward 内完成 KV 缓存和注入
- **flash-preference**: 一次 forward + monkey-patch attention，`to_shared`/`to_unshared` 管理共享
- **dpo-prefix-sharing**: 一次 forward + 自定义 FlexAttention mask

### 完成事项

1. **方案变更** — 将 doc-designs.md 中的 Two-Phase Forward 设计替换为 One-Forward + KV Injection 设计
   - 一次 model forward，在每层 attention 中缓存/注入 prefix KV
   - 代码全部以 patch 形式在 prefix-sharing 项目中开发（不修改 verl/megatron）
   - 明确标注复用 PrefixTrain_dev 的已调测代码（迁移 + 修复 detach bug）

2. **更新 doc-designs.md** — 替换主设计章节，保留 Megatron 分析和初始架构作为历史参考

### 关键设计差异

| 维度 | Two-Phase Forward (废弃) | One-Forward + KV Injection |
|------|--------------------------|----------------------------|
| forward 次数 | 2 | 1 |
| PP 兼容性 | 复杂 | 简单 |
| 显存 | 高 | 低 |
| 与参考项目一致性 | 不一致 | 一致 |

### 下一步

- 按 Step 1-7 的实现顺序开始编码（优先 Step 1-3：前缀检测 + 数据拆分 + 单层验证）

---

## 2026-05-10 补充 2：Two-Phase Forward 详细设计

### 完成事项

1. **verl MegatronActor 完整数据流分析** — 逐函数跟踪了 Actor 从收到 rollout 数据到完成 loss 计算的调用链
   - 关键路径: `RayPPOTrainer._update_actor()` → `MegatronPPOActor.forward_backward_batch()` → `forward_step()` → `get_mcore_forward_fn()` → `preprocess_thd_no_padding()` → `model()` → `postprocess_thd_no_padding()` → `loss_func()`
   - 确认 micro-batch 切分在 `forward_backward_batch()` 中完成
   - 确认 `preprocess_thd_no_padding` 将变长序列 flatten 为 `[total_nnz, hidden_size]`，用 `PackedSeqParams` 携带 cu_seqlens

2. **PrefixTrain_dev training.py 主流程验证** — 阅读了 PoC 的完整训练流程
   - 确认 PoC 使用 `FlexFlashAttentionOp` 中的 `memory_manager` 管理 KV 缓存
   - 关键: `effective_len -= shared_prefix_len[idx]` 减去可复用前缀长度，cu_seqlens 反映的是有效长度而非原始长度
   - Pipeline parallel 下通过 `extra_tensors` 在 stage 间传递 KV 信息

3. **Megatron attention KV 生成和传递机制** — 逐行阅读了 SelfAttention.forward()
   - QKV 由 `linear_qkv` 一次性投影，再 split 为 Q、K、V
   - KV 形状: `[seq_len, batch, num_query_groups, head_dim]`
   - FlashAttention varlen 通过 `cu_seqlens` 管理变长序列
   - 确认 Megatron 没有内置的 prefix KV 共享机制

4. **产出详细设计** — 完成 Two-Phase Forward 方案的详细设计，详见 `doc-designs.md`

### 关键设计决策

- **注入层级**: 选择在 verl 的 `forward_step` 函数中注入（verl → Megatron 的桥梁层），而非修改 Megatron 核心代码
- **前缀检测需 Trie 树**：不同 RL 场景中 prefix ≠ prompt。GRPO/DPO 中 prefix 等于 prompt，但 tree-mode 中 prefix = prompt + 共同推理步骤，step-mode 中 prefix = prompt + 全部历史 action/observation。因此 Trie 树前缀检测是必要的通用方案，GRPO/DPO 场景可特化优化
- **梯度正确性**: prefix KV 保留 autograd 计算图（不 detach），多个 suffix 的梯度通过 prefix KV 自然累积
- **Hook 安装**: MVP 阶段用 monkey-patch SelfAttention.forward()，稳定后迁移到 ModuleSpec

### 发现的 PoC Bug

- PrefixTrain_dev 的 `memory_manager/memory.py:42` 对缓存的 prefix KV 执行 `clone().detach()`，切断了梯度流
- 后果：suffix 序列的梯度无法通过 prefix KV 回传到模型参数，QKV 权重只收到 suffix 部分的梯度
- PoC 未发现原因：仅在模拟数据上跑了一个 iteration，未验证收敛性
- 已在设计文档中记录此 bug，我们的方案绝不使用 detach

### 下一步

- 按 Step 1-7 的实现顺序开始编码（优先 Step 1-3：前缀检测 + 数据拆分 + 单层验证）

---

## 2026-05-10 补充：Megatron-LM v0.15.0 与 mbridge 深入分析

### 完成事项

1. **Megatron-LM v0.15.0 深入阅读** — 完整跟踪了 GPTModel forward 调用链和 attention 实现细节
   - 完整调用链: `GPTModel.forward()` → `_preprocess()` (embedding + RoPE) → `TransformerBlock.forward()` → `TransformerLayer.forward()` → `_forward_attention()` → `SelfAttention.forward()` → `DotProductAttention.forward()`
   - 张量形状: input `[b, s]` → embedding `[s, b, h]` → QKV `[s, b, np, hn]` → attention output `[s, b, h]`
   - 找到了 prefix sharing 的 5 个精确注入位置（详见 `doc-designs.md`）

2. **mbridge 桥接层分析** — 理解了 verl ↔ mbridge ↔ Megatron 的数据流转
   - 两种桥接: VANILLA_MBRIDGE（`mbridge` 包）和 MEGATRON-BRIDGE（`megatron.bridge`）
   - 核心功能: HF 配置转换、权重加载/导出、PEFT 支持
   - verl 通过 `get_mcore_forward_no_padding_fn` 调用 Megatron forward

3. **Sequence Packing 分析** — 理解了 `PackedSeqParams` 和 `cu_seqlens` 机制
   - `preprocess_thd_no_padding()` 处理序列拼接和对齐
   - prefix sharing 可复用现有的 packing 基础设施

### 关键发现

- **Megatron 的 forward 分层清晰**：GPTModel → TransformerBlock → TransformerLayer → SelfAttention → DotProductAttention，每层都是可拦截的
- **KV cache 管理在 attention 层**：`SelfAttention.forward()` 中 `get_query_key_value_tensors()` 和 `_adjust_key_value_for_inference()` 是关键拦截点
- **verl 的数据预处理是重要入口**：`preprocess_thd_no_padding()` 负责 cu_seqlens 计算，prefix sharing 可在此注入前缀分组信息
- **mbridge 已有 PEFT 扩展机制**（LoRA/DoRA），但 prefix sharing 不适合作为 PEFT 类型，更适合作为 forward 流程优化
- **Pipeline Parallel 的微批次调度**在 `schedules.py` 中，prefix sharing 需要在 micro-batch 层面考虑缓存复用

### 下一步

- 基于 Megatron forward 细节，精化方案设计中的注入层级
- 搭建 Python 包工程脚手架

---

## 2026-05-10 项目启动 - 环境搭建与预备知识准备

### 完成事项

1. **仓库结构搭建** — 完成项目工作区初始化
   ```
   prefix-0501/
   ├── survey/                  # 调研参考项目
   │   ├── dpo-prefix-sharing/  # DPO 前缀共享（TRL + FlexAttention）
   │   └── flash-preference/    # 通用前缀共享（HF + monkey patch）
   ├── dependency/              # 当前 RL pipeline 依赖
   │   ├── verl_v070/           # verl v0.7.0（shallow clone）
   │   ├── megatron_v0150/      # Megatron-LM core_v0.15.0（shallow clone）
   │   └── PrefixTrain_dev/     # 团队 PoC 代码
   └── refactor/
       └── prefix-sharing/      # 正式开发仓库（Jackie2049/prefix-sharing）
   ```

2. **代码分析** — 深入阅读了 4 个关键仓库
   - **PrefixTrain_dev**: Trie 树前缀检测 + activation 复用 + 魔改 Megatron 的分布式训练集成。PoC 级别，仅基于模拟数据跑通一个 iteration。详见 `doc-designs.md`
   - **verl v0.7.0**: 高度模块化的 RL 框架，Actor/Critic/Rollout 均可插拔，有 Megatron 引擎层。详见 `doc-designs.md`
   - **flash-preference**: 上下文管理器 API（一行代码启用），monkey patch 方式实现，2-3x 加速
   - **dpo-prefix-sharing**: FlexAttention + 自定义 mask，数值等价性保证，仅支持 DPO

3. **初步方案设计** — 产出系统架构和 5 阶段施工计划，详见 `doc-designs.md`

4. **工程管理规范** — 建立文档管理规定
   - `doc-progress.md`: 所有工作进展记录
   - `doc-designs.md`: 所有方案设计记录
   - 严格按时间倒序排列

### 关键发现

- verl v0.7.0 的 Megatron 依赖版本是 core_v0.15.0（通过 Dockerfile 确定，setup.py 中未直接标注）
- PrefixTrain_dev 的前缀检测**仅使用 Trie 树算法**（`get_store_shared_tensor`）。排序相邻比较版本（`compute_longest_shared_prefixes_tokens`）是死代码，主流程未调用（已验证 training.py 的 import 和调用点）
- flash-preference 的 "首层共享、末层恢复" 策略是关键设计，值得借鉴
- 所有参考项目都不涉及 Megatron 分布式场景下的 prefix sharing，这正是我们的差异化价值

### 下一步

- 搭建 Python 包工程脚手架
- 实现前缀检测模块原型
- 深入理解 Megatron forward 流程中的 KV cache 机制
