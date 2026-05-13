# 工作进展记录

> **规则**: 所有工作进展按时间倒序记录，最新在前。

---

## 2026-05-14 补充 14：补齐 verl + Megatron Phase 1 真实代码入口

### 背景

用户要求在代码层面补齐 Phase 1 目标缺口，使 GPU / NPU / 真实框架测试成为唯一遗留项。为此重新阅读了 `dependency/verl_v070/verl/workers/actor/megatron_actor.py`、`dependency/megatron_v0150/megatron/core/transformer/attention.py`、Megatron RoPE / packed sequence 逻辑，并对照 PrefixTrain_dev 的 per-sample reuse relation 与 flash-preference / dpo-prefix-sharing 的 prefix sharing 形态。

### 完成事项

1. 补齐 verl actor 主路径入口：
   - 在 `megatron_actor.py` 中新增极薄 import / helper 调用 / context manager。
   - `prepare_megatron_actor_micro_batch()` 在 micro-batch 内生成 prefix-sharing plan，按 attention mask 裁掉 reuser prefix。
   - `megatron_actor_prefix_sharing_context()` 把 meta、cache、backend、原始 position_ids、restore slot 传给 Megatron attention。
   - `restore_megatron_actor_log_probs()` 从 provider prefix-last logits 恢复 reuser 第一个 suffix token logprob，保持 autograd 路径。

2. 补齐 Megatron attention 真实 hook：
   - 在 `Attention.forward()` packed THD squeeze 后、标准 RoPE 前新增 `maybe_run_prefix_sharing_attention()` 调用。
   - `integrations/megatron_runtime.py` 负责按原始 position_ids 选择 RoPE 频率、构建 expanded KV、执行 reference attention、调用原始 output projection。
   - 无 runtime context 时完全返回 Megatron 原路径。

3. 补齐 backend 能力：
   - `TorchReferenceBackend.attention()` 支持 packed THD `[T, H, D]`。
   - 支持 Megatron GQA：当 query heads 是 KV heads 的整数倍时对 key/value 执行 repeat_interleave。

4. 补充测试：
   - 新增 packed THD + GQA reference backend optional 测试。
   - 新增 verl Megatron runtime helper optional 测试，覆盖 mask 裁剪、kept position_ids、context restore slot、Prefix-Last Restore autograd。
   - 本地执行：`29 passed, 4 skipped`。skip 原因均为当前环境缺少 `torch`、`torch_npu`、`verl`。

### 当前结论

代码层面 Phase 1 主链路缺口已补齐：core detector/planner/mapping、verl actor entry、runtime context、Megatron attention hook、RoPE 位置修正、KV injection、logprob restore 均已存在。当前唯一未闭环项是需要在真实 GPU / NPU / verl 环境中运行 small-scale actor logprob/update smoke test 与加速器测试。

---

## 2026-05-14 补充 13：审计 Phase 1 完成度并补齐 mapping integration adapter

### 背景

用户指出 `mapping.py` 虽然在详细设计中是 preprocess/postprocess 核心模块，但当前代码没有在 runtime / integration 主路径中使用。重新审计后确认：当前版本不能被表述为已经完成 `verl + Megatron` Phase 1 RL MVP；它完成的是 core semantic、reference backend、patch/context 骨架和本地 CPU 测试，真实 verl actor patch 与 Megatron QKV rewiring 仍未完成。

### 完成事项

1. 补齐 `integrations/verl_mcore.py` 的框架无关 batch adapter：
   - 新增 `VerlMCoreBatchAdapter`
   - 新增 `VerlMCorePreparedBatch`
   - `prepare_micro_batch()` 调用 planner + mapping，实际裁剪 input / labels / loss masks
   - `prepared_context()` 建立 runtime context，供 patched attention 消费
   - `restore_logprobs()` 调用 mapping 的 Prefix-Last Restore

2. 补齐公开入口：
   - 新增 `enable_prefix_sharing()`
   - 新增 `prefix_sharing_enabled()`
   - 在 `integrations/__init__.py` 导出 adapter 和入口函数

3. 修正配置语义不一致：
   - `boundary_strategy` 当前唯一允许值统一为 `prefix_last_restore`
   - `restore_last_prefix_token` 作为旧值不再通过校验

4. 更新详细设计：
   - 明确 `VerlMCoreBatchAdapter` 已经接入 mapping
   - 明确 Megatron attention patch 仍未完成真实 QKV rewiring
   - 新增“当前实现审计结论”，区分已完成和未完成项

5. 补充测试：
   - 集成测试覆盖 `VerlMCoreBatchAdapter` preprocess、context、restore
   - 测试 `prefix_sharing_enabled()` 在 install 失败时传播错误

### 当前结论

当前代码可以支撑 Phase 1 的 core semantic 开发与本地 CPU 自测，但不能直接交付真实 `verl + Megatron` 业务链路。下一步必须补真实 verl actor preprocess/postprocess patch 和 Megatron SelfAttention QKV rewiring。

---

## 2026-05-13 18:00 补充 12：Phase 1 改为支持 per-sample reuse relation

### 背景

用户明确要求 Phase 1 必须支持 PrefixTrain_dev 风格的 per-sample reuse relation：同一个 provider 可以面向不同 reuser 提供不同长度的 prefix 子序列复用，例如 `seq1` 复用 `seq0` 的 `0..5`，`seq2` 复用 `seq0` 的 `0..10`。这属于核心语义，不应采用“最长前缀优先、非重叠 group”的妥协方案。

### 完成事项

1. 改造 `TriePrefixDetector`：
   - 新增 `PrefixReuseSpec`
   - 检测结果以 `reuse_specs` 作为语义核心
   - 采用在线 Trie：当前样本从历史样本中匹配最长可复用 prefix，再插入 Trie 作为后续 provider
   - 支持同一 provider 服务多个不同 `prefix_len`
   - `min_group_size` 在 relation 语义下仍生效：历史命中样本数加当前样本需达到阈值

2. 改造 metadata / planner：
   - `PrefixSharingBatchMeta` 新增 `reuse_specs`
   - planner 基于 `provider_index[i] != i and prefix_lens[i] > 0` 判断 reuser
   - `prefix_last_restore` 从 per-sample reuse relation 派生

3. 改造 cache / reference backend：
   - cache key 去掉 `group_id`
   - provider 缓存完整 K/V
   - reuser 按自己的 `prefix_len` 从 provider K/V 中切片构造 expanded KV
   - reuser 构造出的完整 logical KV 会再次写入 cache，因此 reuser 也可以作为后续样本的 provider

4. 更新文档：
   - `doc-designs-final.md` 明确 per-sample reuse relation 是 Phase 1 核心语义
   - `doc-glossary.md` 新增 Reuse Relation，并将 Prefix Group 降级为调试/统计视图

5. 更新测试：
   - 增加同一 provider 多长度复用场景
   - 增加 reuser 继续作为后续 provider 的 transitive reuse 场景
   - 增加 `min_group_size` relation 阈值场景
   - 校验不同 reuser 的 `provider_prefix_last_pos` 分别指向不同 prefix-last 位置

### 自测结果

已运行：

```bash
python3 -m pytest tests/unit_test
python3 -m pytest tests/system_test
python3 -m pytest
```

结果：

- 单元测试：`21 passed`
- 系统测试：`1 passed`
- 全量测试：`26 passed, 3 skipped`

---

## 2026-05-13 12:08 补充 11：将边界处理配置改为通用策略开关

### 背景

用户指出不应通过 `restore_mode` 来表达 prefix/suffix 边界处的 last-token 处理方式。该配置未来需要能够扩展到 `boundary_token`、`strict_suffix` 等策略；当前阶段只实现 Prefix-Last Restore，但开关语义不应绑定到 restore。

### 完成事项

1. 将 `PrefixSharingConfig.restore_mode` 改为更通用的 `boundary_strategy`。
2. 当前阶段唯一允许值为 `prefix_last_restore`。
3. 错误信息中明确后续可扩展策略包括 `boundary_token` 和 `strict_suffix`。
4. 同步更新 `doc-designs-final.md` 和配置单元测试。
5. 修正 `test_detector.py` 中嵌套列表场景的断言：当前 detector 以一层 token 序列为语义边界，不对嵌套列表做 flatten。

### 自测结果

已运行：

```bash
python3 -m pytest tests/unit_test
python3 -m pytest
```

结果：

- 单元测试：`17 passed`
- 全量测试：`22 passed, 3 skipped`

---

## 2026-05-13 10:06 补充 10：按测试层级拆分目录

### 背景

用户要求将单元测试、集成测试、系统测试拆分到 `tests/` 下的不同目录，分别命名为 `unit_test`、`integrated_test`、`system_test`，便于后续按测试层级独立运行和维护。

### 完成事项

1. 调整测试目录：
   - `tests/unit_test/`：配置、detector、planner、mapping、cache、context 等纯模块测试
   - `tests/integrated_test/`：patch manager、integration unavailable、torch reference backend、加速器/框架 optional 测试
   - `tests/system_test/`：阶段一 core 端到端语义流测试

2. 更新 `pyproject.toml`：
   - `testpaths = ["tests/unit_test", "tests/integrated_test", "tests/system_test"]`

3. 清理旧 `tests/` 目录和测试生成缓存。

### 自测结果

已分别运行：

```bash
python3 -m pytest
python3 -m pytest tests/unit_test
python3 -m pytest tests/integrated_test
python3 -m pytest tests/system_test
```

结果：

- 全量：`22 passed, 3 skipped`
- 单元测试：`17 passed`
- 集成测试：`4 passed, 3 skipped`
- 系统测试：`1 passed`

3 个 skip 仍然来自本地缺少 `torch`、`torch_npu`、`verl`，符合当前无 GPU/NPU/真实框架环境的预期。

---

## 2026-05-13 01:19 补充 9：完成阶段一核心代码与开发者测试

### 背景

基于 `doc-designs-final.md` 和 Terminal Codex 主会话中的补充讨论，开始阶段一代码开发。阶段一目标是先完成 `verl + Megatron` RL MVP 所需的可迁移核心模块、reference backend、patch 框架和开发者自测。当前本地无 GPU / NPU，且未安装 PyTorch、verl、Megatron，因此加速器和真实框架测试先以 optional skip 形式落地。

### 完成事项

1. 新增 `prefix_sharing/` 包结构：
   - `core/config.py`：`PrefixSharingConfig` 和阶段一硬约束校验
   - `core/metadata.py`：`PrefixSharingBatchMeta`、`PrefixLastRestoreSpec`
   - `core/detector.py`：`TriePrefixDetector`
   - `core/planner.py`：prefix group 到可执行 metadata 的计划生成
   - `core/mapping.py`：input / label / loss / Prefix-Last Restore 坐标映射
   - `core/cache.py`：forward/backward 生命周期内的 `PrefixKVCache`
   - `core/logprob.py`：Prefix-Last Restore 的 logits/logprob tensor 辅助函数，保持 provider prefix-last 梯度路径

2. 新增 backend 层：
   - `backends/base.py`：统一 backend capability 与接口
   - `backends/torch_ref.py`：PyTorch reference attention / KV build 骨架
   - `backends/cuda_ref.py`、`backends/cann_ref.py`：阶段一可运行参考后端占位，避免语义层绑定 CUDA TE

3. 新增 integration 层：
   - `integrations/context.py`：runtime context 与 cache 生命周期
   - `integrations/patch_manager.py`：幂等 patch handle、卸载和回滚
   - `integrations/megatron_attention.py`、`integrations/megatron_rope.py`、`integrations/verl_mcore.py`：非侵入式 patch 接入骨架和依赖缺失时的清晰报错

4. 新增开发者测试：
   - 配置约束测试
   - Trie detector / planner / metadata 测试
   - mapping / Prefix-Last Restore / cache / context 测试
   - patch manager 和 integration unavailable 测试
   - 纯 Python 阶段一系统流测试
   - PyTorch reference backend、CUDA、CANN、真实 verl/Megatron optional 测试

### 自测结果

已运行：

```bash
python3 -m pytest
python3 -m compileall prefix_sharing
```

结果：

- `22 passed`
- `3 skipped`
- skip 原因：本地未安装 `torch`、`torch_npu`、`verl`
- `compileall` 通过

### 当前结论

阶段一核心语义层、reference backend 接口、patch 框架和 CPU 可运行开发者测试已经完成。后续如进入真实 GPU / NPU 或 verl/Megatron 环境，需要继续启用 optional 测试并补齐真实 QKV rewiring 的集成细节。

---

## 2026-05-13 00:41 补充 8：新增提交后必须 push 的协作规则

### 背景

用户要求本项目所有修改在 commit 完成后都必须 push 到远程仓库。

### 完成事项

1. 更新 `CLAUDE.md` 和 `agents-readme.md`：
   - 新增规则：每次 commit 完成后必须 push 到远程仓库。
   - 同步修正文档清单，明确 `doc-design-history.md` 和 `doc-designs-final.md` 的职责。

2. 本次规则变更完成后，将按新规则执行 commit 并 push。

---

## 2026-05-13 00:41 补充 7：拆分最终设计文档并收敛 Roadmap

### 背景

在继续讨论阶段规划后，确认最终 roadmap 需要按业务目标重新组织：

- 阶段一聚焦 `verl + Megatron` RL MVP。
- 阶段二聚焦业务落地所需的并行策略、硬件解耦和加速后端。
- 阶段三扩展到 standalone Megatron / FSDP 的 SFT 和 pretrain。
- 阶段四面向 verl / Megatron 上游 PR 与产品化。

同时确认最终实现不再考虑 boundary mode；正式精度路径统一为 Prefix-Last Restore。

### 完成事项

1. 将 `doc-designs.md` 重命名为 `doc-design-history.md`。

2. 新增 `doc-designs-final.md`，作为后续开发使用的最终设计文档，包含：
   - 项目目标与阶段一非目标
   - `One-Forward + KV Injection + Prefix-Last Restore` 核心方案
   - 三层架构：`core/`、`integrations/`、`backends/`
   - `PrefixSharingConfig`、`PrefixSharingBatchMeta`、`PrefixLastRestoreSpec`
   - detector、planner、mapping、cache、backend 职责
   - 阶段一 preprocess / attention / postprocess 数据流
   - 非侵入式 patch 集成策略
   - 三层精度验收方案
   - 阶段一实施计划
   - 四阶段 roadmap

3. 更新 `doc-design-history.md` 顶部，记录最终方案收敛：
   - 阶段一：`verl + Megatron` RL MVP
   - 阶段二：并行策略、硬件解耦、加速后端，目标是业务可落地
   - 阶段三：扩展到 FSDP / Megatron 的 SFT 和 pretrain
   - 阶段四：上游 PR 与产品化

### 当前结论

当前已经完成需求、约束、总体架构、精度路径和 roadmap 的设计收敛。下一步可以基于 `doc-designs-final.md` 继续细化阶段一的接口签名、测试用例和开发任务拆分。

---

## 2026-05-12 19:24 补充 6：明确 next-token 边界问题与 Prefix-Last Restore 主线

### 背景

继续审视 prefix sharing 的 logprob 精度一致性时，发现一个关键边界问题：reuse 样本如果只保留 suffix query，虽然 suffix token 可以通过注入的 prefix KV attend 到 prefix，但模型不会产生被裁剪掉的 prefix-last query/output。

对 causal LM：

```
output(P_last) -> predict S0
output(S0)     -> predict S1
```

因此第一个 suffix token `S0` 的 logprob 依赖 prefix 最后一个 token 的输出。如果只计算：

```
Q = [S0, S1, S2]
```

则缺失：

```
output(P_last)
```

### 参考项目核对

1. **dpo-prefix-sharing**
   - 构造输入时使用 `prompt + chosen + prompt[-1:] + rejected`
   - `prompt[-1:]` 本质是 boundary token，用来产生 `output(prompt_last) -> rejected_first`

2. **flash-preference**
   - 在第一层前 `to_shared()` 合并重复 prefix
   - 在最后一层后 `to_unshared()` 把 prefix hidden states 复制回每条原始序列
   - 本质是 Prefix Output Restore 方案，用恢复的 prefix-last output 计算第一个 suffix token logprob

3. **PrefixTrain_dev**
   - 主流程证明了 Trie 检测、prefix cache、KV/activation 复用可以跑通
   - 但 loss 侧存在随机 label / PoC 痕迹，且有已知 detach bug
   - 不能证明其已解决真实 causal LM next-token logprob 边界问题

### 设计结论

1. **Strict suffix mode 不作为最终精度一致方案**
   - 只作为内部调试模式
   - 可用于验证 KV injection、RoPE offset、metadata、backend 主链路
   - 不能用于最终 actor logprob/update 精度验收

2. **MVP 第一阶段主线采用 Prefix-Last Restore Mode**
   - reuse 样本只计算 suffix query
   - 从 provider 恢复 prefix-last 位置的 output/logit/logprob
   - 用恢复的 prefix-last 输出计算第一个 response token logprob
   - 与 verl Megatron actor 当前 `[-response_length - 1 : -1]` logprob 切片语义兼容

3. **Boundary Token Mode 作为 fallback / 对照方案**
   - 对齐 dpo-prefix-sharing 的 `prompt[-1:]` 思路
   - 当某些后端难以做 output restore 时，可考虑额外保留 prefix-last query

4. **Full Prefix Restore 作为后续扩展**
   - 面向单独 Megatron SFT / pretrain 或完整序列 loss 场景
   - 对齐 flash-preference 的完整 prefix hidden restore 思路

### 文档更新

更新 `doc-designs.md`：
- 新增 `2026-05-12 19:24 关键分析：Next-Token 边界问题与 Restore Mode 决策`
- 记录 strict suffix、boundary token、prefix output restore 的差异
- 明确 MVP 主线为 `One-Forward + KV Injection + Prefix-Last Restore`

---

## 2026-05-11 21:31 补充 5：方案升级为三层架构

### 背景

当前工作树回到 `origin/main` 的 `3d8a657` 后，此前本地写入的三层架构设计记录不在当前文档中。重新审视后，确认需要将三层架构方案作为最新设计记录恢复到 `doc-designs.md` 顶部。

旧 One-Forward + KV Injection 方案核心方向可行，但存在工程边界不清的问题：
- 过度依赖 `position_ids`，但 verl mcore THD 普通 RoPE 路径实际传入 `position_ids=None`
- RoPE patch 只覆盖 `_apply_rotary_pos_emb_thd()`，无法覆盖 `apply_rope_fusion=True` 时的 fused THD 路径
- logits / label / loss mask / output restore 在 prefix token 被裁剪后缺少明确映射
- 方案容易绑定 CUDA TransformerEngine，不利于同时支持 CUDA GPU 和 CANN NPU
- cache 生命周期、backend 能力边界、MVP 约束、patch 接入边界需要显式化

### 完成事项

1. 更新 `doc-designs.md`，新增最新设计章节：`2026-05-11 21:31 详细设计：三层架构 + Metadata + 后端适配方案`

2. 新设计将 prefix sharing 拆为三层：
   - **通用语义层**：`PrefixSharingBatchMeta`、detector、planner、mapping、cache，不绑定具体硬件后端
   - **模型集成层**：MVP 完全通过 patch 接入 verl mcore preprocess/postprocess、Megatron attention hook、RoPE offset、logprob/loss 对齐
   - **后端适配层**：`TorchReferenceBackend`、CUDA backend、CANN NPU backend 等，通过统一接口消费 metadata

3. 明确 detector 策略：
   - `TriePrefixDetector` 用于全自动识别共同前缀
   - `PromptPrefixDetector` 用于预先知道共同前缀边界的场景

4. 明确核心模块职责：
   - `mapping.py` 负责原始序列与裁剪后序列的 input / label / loss / output 坐标映射
   - `cache.py` 负责当前 forward/backward 生命周期内每层 prefix K/V 暂存和读取，不做检测、映射或后端选择

5. 明确 MVP 约束：
   - 普通 text GPT / causal LM
   - PP=1
   - CP=1
   - `apply_rope_fusion=False`
   - `fused_single_qkv_rope=False`
   - 精度测试在 eval mode，或 train mode 关闭 dropout 且固定 RNG

6. 明确未来扩展路径：
   - CANN NPU 通过独立 backend 适配，不复用 CUDA TE 细节
   - CP / PP 作为专项 backend 或专项 plan 扩展
   - fused kernel 必须声明能力并对齐 `TorchReferenceBackend`
   - 当前 patch 接入未来迁移为 verl/Megatron 正式 extension hook 或 config branch

### 下一步

- 按新设计先实现 `PrefixSharingConfig.validate()`、`PrefixSharingBatchMeta`、planner、mapping 和 `TorchReferenceBackend`
- 用 reference backend 验证裁剪、RoPE offset、KV 注入、label/logprob/output mapping
- 再进入 Megatron 单层 hook 和 verl logprob patch 集成

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
