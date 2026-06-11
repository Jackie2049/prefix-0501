# prefix-sharing — Agent 开发规范

> **本文件仅约定 prefix-sharing 模块的专属开发规范（模块边界、技术约束、注入方式等）。**
> **所有开发者必须同时遵守仓库完整开发规范：[`../AGENTS.md`](../AGENTS.md)**，包括但不限于：
> - Git 提交规范（**强制 `[type] <中文简要说明>` 格式，必须有中文简要说明**）
> - PR 规范（测试结果小节、title 格式）
> - Cursor Agent 特殊规则（commit 须获用户明确同意）
> - 不可违反的技术原则、禁止事项、目录边界
> - 测试规范、文档入口
>
> 两份规范同时生效；本文件是对完整规范的**补充**，不是替代。

## 项目背景

RL 训练中前缀复用（prefix sharing），目标是插件化集成到 rllm+verl+megatron 的 RL pipeline。`prefix-sharing/` 是 `prefix-0501` 仓库中的正式开发目录；因上下游（verl、Megatron、调研 PoC）存在依赖，与 `survey/`、`dependency/`、`docs/` 放在同一仓库内联开发，便于快速迭代与集成验证。

## 工作区结构

```
prefix-0501/
├── docs/                    # 项目文档
├── survey/                  # 调研项目（flash-preference, dpo-prefix-sharing, PrefixTrain_dev）
├── dependency/              # verl_v070, megatron_v0150
└── prefix-sharing/          # 正式开发目录（prefix sharing 核心代码与测试）
```

当前开发默认先读本文件和相关代码；涉及 prefix sharing 语义、术语命名、分层边界或设计取舍时，再查阅 [`docs/concepts.md`](../docs/concepts.md)。架构关系参考 [`docs/overview.md`](../docs/overview.md) 和 [`docs/overview.puml`](../docs/overview.puml)。[`docs/legacy/`](../docs/legacy/) 仅作历史背景参考，不作为当前实现规范。

## 代码分析规范

- 分析 PoC（PrefixTrain_dev）等参考代码时，只关注**主流程中真正被调用的代码**，忽略死代码和未使用的函数
- 对 agent 返回的分析结果，如果涉及关键设计决策，需到源码中验证函数是否在主流程中被实际调用

## 技术约束

- **TDD 优先**：新增功能、bug fix、精度语义调整前，优先先写测试表达目标行为；纯重构、命名整理、小范围机械调整可视风险决定是否补测，但不得降低核心语义覆盖率。
- **精度一致性是红线**：任何 prefix sharing 优化都必须先保证 logprob / loss / 梯度与 baseline 语义一致，再考虑性能。涉及 suffix-first logprob、prefix-last restore、KV 注入、position ids、packed layout、autograd 路径等改动时，要显式思考精度影响。
- 缓存 prefix KV 时**绝不 detach**，必须保留 autograd 计算图，确保梯度正确性
- PrefixTrain_dev 的 `memory_manager/memory.py:42` 使用了 `clone().detach()` 是一个已知 bug，不可复现此错误

## 模块边界

- `core/`：框架无关语义层，只表达 prefix sharing 计划、裁剪、logprob 语义和 store 生命周期；不要 import verl / Megatron integration，也不要沉淀硬件或框架特定 packed layout。
- `backends/`：执行后端，消费 `PrefixSharingPlan` 和 runtime context 提供的信息；不要重新做 prefix detection 或重新解释复用语义。
- `integrations/`：verl / Megatron 薄适配层，负责 patch、helper、context、packed layout 和框架调用点接入；不要把 core 算法散落到 integration 层。
- `dependency/`：只允许最小必要侵入式修改，例如 import、helper 调用、context manager 包裹和 hook；优先把可测试逻辑放回 `prefix-sharing/`。

## 注入方式决策

- **当前阶段**：采用 monkey-patch 方式注入 Megatron attention 的 prefix KV 拼接逻辑
  - 理由：作为仓库内的独立模块，patch 方式简单、易管理，可尽量少改 `dependency/` 中的 verl/Megatron 快照
- **后续计划**：项目充分验证后，向 verl/megatron 社区提 PR 时，改为对对应类进行继承和扩展的方式
- 两种方式的核心逻辑一致，只是代码组织形式不同，迁移成本可控
