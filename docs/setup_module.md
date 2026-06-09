# `prefix_sharing/setup/` 模块设计文档

> 本文档描述 `setup` 模块的架构、职责边界、与其他模块的关系、文件清单、兼容矩阵、使用方式。

---

## 1. 模块定位

### 1.1 四层分工

```
┌───────────────────────────────────────────────────────────────┐
│  setup/          引导层：版本门卫 → patch 选择 → 注入 → 日志    │
├───────────────────────────────────────────────────────────────┤
│  integrations/   业务逻辑层：context、runtime hook、batch 适配   │ ← 不改
├───────────────────────────────────────────────────────────────┤
│  backends/       执行层：RoPE、KV build、attention 计算          │ ← 不改
├───────────────────────────────────────────────────────────────┤
│  core/           语义层：plan、detector、logprob、store          │ ← 不改
└───────────────────────────────────────────────────────────────┘
```

### 1.2 setup 的职责

setup 模块**只做三件事**：

| 职责 | 说明 |
|------|------|
| **版本校验** | 探测运行环境中的 verl、Megatron Core、MindSpeed 版本，对照兼容矩阵决定能否 patch |
| **patch 注入** | 为校验通过的版本组合，选择对应的 patch wrapper，通过 LoggedPatchManager 运行时替换目标方法 |
| **patch 管理** | 提供 describe() 查看 patch 详情、disable() 回滚并打印恢复日志 |

setup **不做的事**：

- 不做 prefix detection、KV expansion、logprob restore、attention 计算——这些在 core/、backends/、integrations/
- 不沉淀任何 prefix-sharing 语义逻辑
- 不修改 dependency/ 中任何源码文件

### 1.3 与 integrations/ 的关系

```
integrations/ 的旧代码继续存在，服务旧版场景（verl 0.7.x + mcore 0.15.x）
setup/ 的 patch wrapper 直接调用 core/ 和 backends/，
        新版 API 适配由 setup/runtime_adapters.py 完成
        integrations/ 中的 context.py（ContextVar 机制）被 setup/ 的 patch 共用
```

关键区别：setup 的 patch wrapper（如 `attention.py`）**不再调用** `integrations/megatron_runtime.maybe_run_prefix_sharing_attention` 或 `integrations/verl_mcore.build_prefix_sharing_micro_batch`——这些函数内部可能依赖旧版 API。setup 自己封装新版适配逻辑，确保 integrations/ 一行不改。

**唯一共用点**：`integrations/context.py`（ContextVar + prefix_sharing_runtime_context）——这是跨版本的 runtime 机制，setup 和 integrations 共用。

---

## 2. 完整架构关系图

```
                    ┌─────────────── 用户 ───────────────┐
                    │                                     │
                    │  prefix_sharing.setup.install()      │
                    │  prefix_sharing.setup.check()        │
                    │  handle.describe() / handle.disable()│
                    └─────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────┐
│  setup/  (引导层)                                                   │
│                                                                     │
│  ┌─── __init__.py ──────────────────────────────────────────────┐  │
│  │  install()  check()  IncompatibleEnvironment                 │  │
│  └──────────┬───────────────────────────────────────────────────┘  │
│             │                                                       │
│  ┌─── version_guard.py ──► detect_versions() ───────────────────┐ │
│  │  探测 verl.__version__, megatron.core.__version__,            │ │
│  │  importlib.metadata("mindspeed")                              │ │
│  └─────────┬────────────────────────────────────────────────────┘ │
│             │ DetectedVersions                                      │
│  ┌─── compat_matrix.py ──► COMPAT_MATRIX ───────────────────────┐ │
│  │  CompatEntry(verl SpecifierSet, mcore SpecifierSet,           │ │
│  │              mindspeed SpecifierSet | None, patch_set_id)     │ │
│  │  第一条匹配的规则生效 → 决定用哪个 patches/*/ 下的 patch_set   │ │
│  └─────────┬────────────────────────────────────────────────────┘ │
│             │ patch_set_id                                          │
│  ┌─── registry.py ──► PatchSpec 注册 + LoggedPatchManager ──────┐ │
│  │  已加载模块 → 立即 patch_attr() + 日志                        │ │
│  │  未加载模块 → 注册 import hook，加载时自动 patch               │ │
│  │  import hook 全部完成后 → 自动恢复 builtins.__import__        │ │
│  └─────────┬────────────────────────────────────────────────────┘ │
│             │                                                       │
│  ┌─── logged_patch.py ──► LoggedPatchManager + PatchHandle ─────┐ │
│  │  patch_attr(): 替换属性 + 写日志 "[PS] Patched X.Y: ..."     │ │
│  │  disable():   恢复属性 + 写日志 "[PS] Restored X.Y → ..."    │ │
│  │  describe():  返回 patch 清单 + ACTIVE/INACTIVE 状态         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─── runtime_adapters.py ──► 新版 API 适配 ─────────────────────┐ │
│  │  apply_positioned_rope_v016():  mcore 0.16.x cu_seqlens 参数  │ │
│  │  extract_sequences_from_batch(): NestedTensor + plain tensor   │ │
│  │  trim_batch():                 NestedTensor + plain tensor 裁剪│ │
│  │  compute_packed_cu_seqlens():  THD packed cu_seqlens 计算     │ │
│  │  restore_logprobs_v016():      THD 1D packed logprob restore  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─── patches/ ──────────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  ┌─ verl080_mcore016_ms0153/ ─── verl + mcore + MindSpeed ─┐  │ │
│  │  │  attention.py      → Attention.forward patch (mcore 0.16) │  │ │
│  │  │    无 context → 调用 original_forward                     │  │ │
│  │  │    有 context → QKV + squeeze + runtime_adapters.RoPE     │  │ │
│  │  │                  + backends.build_kv/attention            │  │ │
│  │  │                  + self.linear_proj                       │  │ │
│  │  │  vocab_logprobs.py → vocab_parallel patch (verl 0.8)     │  │ │
│  │  │    无 context → 调用 original_fn                          │  │ │
│  │  │    有 context → original_fn + runtime_adapters.restore    │  │ │
│  │  │  forward_step.py  → MegatronEngineWithLMHead.forward_step │  │ │
│  │  │    消费 batch → runtime_adapters 适配 → core.Planner      │  │ │
│  │  │    → runtime_adapters.trim → context wrap → 原始方法      │  │ │
│  │  └────────────────────────────────────────────────────────┘  │ │
│  │                                                                │ │ │
│  │  ┌─ mcore012_ms012/ ──── 纯 Megatron + MindSpeed（无 verl）─┐  │ │
│  │  │  attention.py      → SelfAttention.forward (mcore 0.12)    │  │ │
│  │  │  无 verl → 无 forward_step / vocab_logprobs patch          │ │ │
│  │  │  仅 patch 注意力层，训练循环由 MindSpeed 管理               │  │ │
│  │  └────────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

          setup/ 的 patch 调用路径：

          patches/*/attention.py
            ├── core/config.py          PrefixSharingConfig.from_raw()
            ├── core/planner.py         PrefixSharingPlanner.plan()
            ├── backends/torch_ref.py   TorchReferenceBackend.build_kv(), .attention()
            ├── integrations/context.py current_prefix_sharing_context()
            │                          prefix_sharing_runtime_context()
            │                          PrefixSharingRuntimeState
            └── setup/runtime_adapters  apply_positioned_rope_v016()

          patches/*/vocab_logprobs.py
            ├── integrations/context.py current_prefix_sharing_context()
            └── setup/runtime_adapters  restore_logprobs_v016()

          patches/*/forward_step.py
            ├── core/config.py          PrefixSharingConfig.from_raw()
            ├── core/planner.py         PrefixSharingPlanner.plan()
            ├── integrations/context.py prefix_sharing_runtime_context()
            │                          PrefixSharingRuntimeState
            └── setup/runtime_adapters  extract_sequences_from_batch()
                                       trim_batch()
                                       compute_packed_cu_seqlens()
                                       _collect_kept_positions()
```

---

## 3. 版本兼容矩阵

只支持以下两个版本组合。**精确版本号匹配**，不支持版本范围模糊匹配。

### 组合一：verl + Megatron Core + Megatron Bridge + MindSpeed

| 库 | 版本 | patch_set_id |
|----|------|-------------|
| verl | **0.8.0.dev** | `verl080_mcore016_ms0153` |
| megatron-core | **0.16.0** | |
| megatron-bridge | **0.4.0** | |
| mindspeed | **0.15.3** | |

适用场景：verl PPO pipeline + MindSpeed NPU 加速。

Patch 目标：
- `Attention.forward`（Megatron Core 0.16.0）
- `vocab_parallel_log_probs_from_logits`（verl 0.8.x）
- `MegatronEngineWithLMHead.forward_step`（verl 0.8.x）

### 组合二：Megatron Core + MindSpeed（无 verl）

| 库 | 版本 | patch_set_id |
|----|------|-------------|
| megatron-core | **0.12.x** | `mcore012_ms012` |
| mindspeed | **0.12.x** | |
| verl | 不需要 | |

适用场景：纯 Megatron + MindSpeed 训练，不使用 verl PPO pipeline。

Patch 目标：
- 仅 `SelfAttention.forward`（Megatron Core 0.12.x）
- 无 forward_step / vocab_logprobs patch（verl 不在场）

> **注意**：组合二的 dependency 快照尚未在 `dependency/` 目录中，patch 目录为占位骨架。
> 待 mindspeed 0.12 + megatron-core 0.12 快照就位后，再填充 patch 实现。

匹配规则：从上到下遍历，**第一条匹配的规则生效**。不匹配则抛出 `IncompatibleEnvironment`。

版本探测来源：

| 库 | 运行时版本标识 |
|----|-------------|
| verl | `verl.__version__`（从 `verl/version/version` 文件读取） |
| megatron-core | `megatron.core.__version__`（从 `megatron/core/package_info.py` 读取） |
| megatron-bridge | `megatron.bridge.__version__`（从 `megatron/bridge/package_info.py` 读取） |
| mindspeed | `importlib.metadata.version("mindspeed")`（MindSpeed 不暴露 `__version__`） |

---

## 4. Patch 注入目标

### 4.1 三个核心 Patch

每个版本组合的 patch_set 均包含 3 个 PatchSpec，目标方法相同（但实现不同）：

| Patch | 目标方法 | 机制 |
|-------|---------|------|
| A | `Attention.forward`（或 `SelfAttention.forward`） | 前置检查 context → prefix-sharing 路径（QKV + RoPE + KV expansion + attention + projection） / 原始路径 |
| B | `vocab_parallel_log_probs_from_logits` | 前置检查 context → 调用原始 + restore / 直接调用原始 |
| C | `MegatronEngineWithLMHead.forward_step`（或 `MegatronActor.forward_step`） | 消费 batch → prefix-sharing reorg → 构建 context → 包装回原始方法 |

### 4.2 Patch 联动数据流

```
forward_step patch (C)
  │ 消费 batch_iter → 提取 batch
  │ build_prefix_sharing_micro_batch（在 C 内直接调用 core + runtime_adapters）
  │ 构建 prefix_sharing_runtime_state
  │ 包装修改后的 batch 为新 iterator
  │ with prefix_sharing_runtime_context(state):
  │     │
  │     ▼ 调用原始 forward_step(modified_iter, ...)
  │         │ 原始方法内部调用 forward_fn → model forward
  │         │   │ 每个 Attention 层 → Attention.forward (patch A)
  │         │   │   │ ctx is not None → QKV + RoPE + KV expand + attn + proj
  │         │   │   │ 调用 runtime_adapters.apply_positioned_rope_v016
  │         │   │   │ 调用 backends.TorchReferenceBackend.build_kv / attention
  │         │   │
  │         │ logits_processor closure → vocab_parallel_log_probs_from_logits (patch B)
  │         │   │ ctx is not None → original_fn + runtime_adapters.restore_logprobs_v016
  │         │   │ ctx is None     → original_fn 直接
  │         │
  │         ▼ 返回 output
  │
  ▼ 返回 output + 原始 postprocess_micro_batch_func(data=batch)
```

---

## 5. Import Hook 机制

### 5.1 触发时机

当 `install()` 被调用时，如果目标模块尚未加载（不在 `sys.modules` 中），registry.py 会临时替换 `builtins.__import__`，在该模块被 import 时自动应用 patch。

### 5.2 生命周期

```
install() → 对已加载模块：立即 patch + 日志
          → 对未加载模块：注册 import hook
          → 返回 PatchHandle

import hook 激活时：
  builtins.__import__ 被替换
  目标模块被 import → hook 检测到 → 自动 patch + 日志
  所有目标模块都已 patch → hook 自动移除 → builtins.__import__ 恢复

PatchHandle.disable() 时：
  所有 patch 恢复为原始方法 + 日志
  import hook 此时已移除（不需要额外操作）
```

### 5.3 安全性

- import hook 仅在目标模块未加载时激活
- 所有目标模块加载完毕后**立即移除** hook，恢复原始 `builtins.__import__`
- 不影响非目标模块的导入行为
- 如果用户在 hook 激活期间 import 了非目标模块，hook 仅做版本名匹配检查（dict lookup），不干预其他模块

---

## 6. 日志输出

### 6.1 安装时

```
[PS] Detected: verl=0.8.0.dev, megatron_core=0.16.0, megatron_bridge=0.4.0, mindspeed=0.15.3
[PS] Version check → compatible (patch_set=verl080_mcore016_ms0153)
[PS] Patched megatron.core.transformer.attention.Attention.forward: Attention.forward → patched_forward
[PS] Patched verl.utils.megatron.tensor_parallel.vocab_parallel_log_probs_from_logits: vocab_parallel_log_probs_from_logits → patched_fn
[PS] Patched verl.workers.engine.megatron.transformer_impl.MegatronEngineWithLMHead.forward_step: MegatronEngineWithLMHead.forward_step → patched_forward_step
[PS] install() complete. 3 patches active. patch_set=verl080_mcore016_ms0153
```

### 6.2 查看状态

```
>>> handle.describe()
PatchHandle (ACTIVE, 3 patches):
  1. megatron.core.transformer.attention.Attention.forward: Attention.forward → patched_forward
  2. verl.utils.megatron.tensor_parallel.vocab_parallel_log_probs_from_logits: vocab_parallel_log_probs_from_logits → patched_fn
  3. verl.workers.engine.megatron.transformer_impl.MegatronEngineWithLMHead.forward_step: MegatronEngineWithLMHead.forward_step → patched_forward_step
```

### 6.3 回滚时

```
>>> handle.disable()
[PS] Restored verl.workers.engine.megatron.transformer_impl.MegatronEngineWithLMHead.forward_step → MegatronEngineWithLMHead.forward_step
[PS] Restored verl.utils.megatron.tensor_parallel.vocab_parallel_log_probs_from_logits → vocab_parallel_log_probs_from_logits
[PS] Restored megatron.core.transformer.attention.Attention.forward → Attention.forward
[PS] All 3 patches reverted.
```

### 6.4 回滚后状态

```
>>> handle.describe()
PatchHandle (INACTIVE (rolled back), 3 patches):
  1. megatron.core.transformer.attention.Attention.forward: Attention.forward → patched_forward
  2. verl.utils.megatron.tensor_parallel.vocab_parallel_log_probs_from_logits: vocab_parallel_log_probs_from_logits → patched_fn
  3. verl.workers.engine.megatron.transformer_impl.MegatronEngineWithLMHead.forward_step: MegatronEngineWithLMHead.forward_step → patched_forward_step
```

### 6.5 版本不兼容时

```
>>> prefix_sharing.install()
IncompatibleEnvironment: 不兼容的版本组合: verl=0.9.0, megatron_core=0.17.0, mindspeed=None
  支持的组合：
  组合一: verl=0.8.0.dev + megatron-core=0.16.0 + megatron-bridge=0.4.0 + mindspeed=0.15.3
  组合二: megatron-core=0.12.x + mindspeed=0.12.x (无 verl)
```

---

## 7. 使用方式

```python
import prefix_sharing

# ── 一键安装 ──
handle = prefix_sharing.install()

# ── 仅校验版本 ──
versions = prefix_sharing.check()
print(f"verl={versions.verl}, mcore={versions.megatron_core}, ms={versions.mindspeed}")

# ── 查看 patch 详情 ──
print(handle.describe())

# ── 回滚 ──
handle.disable()
```

YAML 配置示例（verl YAML 配置文件，不改 verl 源码）：

```yaml
actor:
  strategy: megatron
  megatron:
    override_transformer_config:
      prefix_sharing_config:
        enable_prefix_sharing: true
        min_prefix_len: 16
        min_group_size: 2
```

---

## 8. 扩展新版本组合

当 verl/Megatron 发布新版本时：

1. `setup/compat_matrix.py` — 新增一条 `CompatEntry`
2. `setup/patches/` — 新建目录（如 `verl090_mcore017/`），实现 3 个 patch wrapper
3. `setup/runtime_adapters.py` — 如有 API 变化，新增适配函数
4. 写测试验证

**不变的文件**：`dependency/` 全部、`core/` 全部、`backends/` 全部、`integrations/` 全部。

---

## 9. Phase 1 约束

所有版本组合的 patch 均遵循 Phase 1 约束：

| 约束 | 原因 |
|------|------|
| PP = 1 | prefix KV injection 跨 PP stage 语义不清 |
| CP = 1 | THD packed + prefix KV 的 cu_seqlens 在 CP 下需特殊处理 |
| 无 fused kernels | fused_forward_fn 路径完全绕过 logits_processor，无法做 logprob restore |
| 无 fused QKV+RoPE | 自定义 position-aware RoPE 与 fused QKV+RoPE 不兼容 |
| 无 multi-modal | Phase 1 仅支持纯文本因果 LM |
| 无 output gate | Phase 1 禁止；后续可扩展 |
| 无 MTP training | MTP loss_mask 与 prefix trim 交互复杂 |

---

## 10. 文件清单

| 文件 | 状态 | 职责 |
|------|------|------|
| `setup/__init__.py` | 新增 | install(), check(), IncompatibleEnvironment |
| `setup/version_guard.py` | 新增 | 版本探测 |
| `setup/compat_matrix.py` | 新增 | 兼容矩阵 |
| `setup/registry.py` | 新增 | PatchSpec + LoggedPatchManager 调度 + import hook |
| `setup/logged_patch.py` | 新增 | 带日志的 PatchManager/PatchHandle |
| `setup/runtime_adapters.py` | 新增 | 新版 API 适配（RoPE、batch、logprob） |
| `setup/patches/__init__.py` | 新增 | 空 |
| `setup/patches/verl080_mcore016_ms0153/__init__.py` | 新增 | PATCH_SET 导出 |
| `setup/patches/verl080_mcore016_ms0153/attention.py` | 新增 | Attention.forward patch (mcore 0.16 + MindSpeed) |
| `setup/patches/verl080_mcore016_ms0153/vocab_logprobs.py` | 新增 | vocab_parallel patch (verl 0.8) |
| `setup/patches/verl080_mcore016_ms0153/forward_step.py` | 新增 | forward_step patch (verl 0.8 + MindSpeed) |
| `setup/patches/mcore012_ms012/__init__.py` | 新增 | PATCH_SET（占位骨架，待版本快照就位） |
| `setup/patches/mcore012_ms012/attention.py` | 新增 | SelfAttention.forward patch (mcore 0.12 + MindSpeed) |
| `prefix_sharing/__init__.py` | 微调（+2 行） | 导出 install, check, IncompatibleEnvironment |

**不变的文件**：`core/` 全部、`backends/` 全部、`integrations/` 全部、`dependency/` 全部。

新增文件总计：17 个（含 3 个占位骨架）。

---

## 11. 与现有架构图的衔接

现有 `docs/overview.puml` 描述了 dependency/integrations/core/backends 四层关系。
加入 setup 层后，架构关系变为：

```
dependency/   ← 不再修改（旧版 inline hook 保留但不再使用）
setup/        ← 新增引导层，取代 dependency 中的 inline hook
integrations/ ← 不变，继续服务旧版场景；context.py 被 setup 共用
core/         ← 不变
backends/     ← 不变
```

依赖方向：

```
setup/patches/verl080_mcore016_ms0153/attention.py
  → core/config, core/planner
  → backends/torch_ref
  → integrations/context (ContextVar, runtime context, RuntimeState)
  → setup/runtime_adapters (新版 RoPE、batch、logprob 适配)

setup/patches/verl080_mcore016_ms0153/forward_step.py
  → core/config, core/planner
  → integrations/context (runtime context, RuntimeState)
  → setup/runtime_adapters (batch 提取/裁剪/cu_seqlens)

setup/patches/verl080_mcore016_ms0153/vocab_logprobs.py
  → integrations/context (current_prefix_sharing_context)
  → setup/runtime_adapters (logprob restore)

setup/patches/mcore012_ms012/attention.py（占位骨架）
  → 仅 attention patch，无 verl 相关依赖
  → core/config, core/planner
  → backends/torch_ref
  → integrations/context

setup/__init__.py → setup/version_guard → setup/compat_matrix → setup/registry → setup/logged_patch
```

setup 不依赖 integrations 的 verl_mcore.py、megatron_runtime.py、megatron_attention.py、megatron_rope.py——新版适配逻辑全在 setup/runtime_adapters.py 中。