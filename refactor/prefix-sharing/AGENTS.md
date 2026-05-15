# prefix-sharing 项目 — Claude Code 工作规范

## 项目背景

RL 训练中前缀复用（prefix sharing），目标是插件化集成到 rllm+verl+megatron 的 RL pipeline。本仓库（Jackie2049/prefix-sharing）是正式开发仓库。

## 工作区结构

```
prefix-0501/
├── survey/                  # 调研项目（flash-preference, dpo-prefix-sharing）
├── dependency/              # verl_v070, megatron_v0150, PrefixTrain_dev
└── refactor/prefix-sharing/ # 本仓库
```

## 文档管理规定

- 所有文档撰写均严格按**时间倒序**排列（最新在前）
- 每次记录的抬头格式：`yyyy-mm-dd hh:mm <title>`
- `doc-progress.md` — 记录所有工作进展
- `doc-design-history.md` — 记录历史方案设计、阶段性分析、被推翻方案和备选方案
- `doc-designs-final.md` — 记录当前最终方案，作为后续开发依据

## 提交规范

- 每一次修改（文档或代码），在修改完结后必须 commit 提交
- 每次 commit 完成后必须 push 到远程仓库
- 如果修改未完成或只做了一半，**不要 commit**
- commit message 格式：`[type] <中文简要说明>`
- type 取值：`feat`(特性)、`fix`(修复)、`chore`(琐事)、`test`(测试)、`doc`(文档)

### ⚠️ Cursor Agent 特殊规则（重要）

**如果当前 agent 来自 Cursor，则 commit 操作必须获得用户明确同意。** 这与其它 agent 的行为有本质不同：

- **必须获得用户明确同意**后才能执行 `git commit`
- 一旦 commit 完成，**必须立即执行 `git push`**，将修改推送到远程仓库
- 或者等待**用户手动完成提交和推送**

在任何情况下，Cursor agent 都不应在未获得用户明确指令的情况下自动执行 commit 操作。如果用户没有明确说"提交"，agent 应该只完成代码/文档修改并等待用户确认。但一旦用户同意提交，agent 应当完整执行 commit + push 流程。

## 代码分析规范

- 分析 PoC（PrefixTrain_dev）等参考代码时，只关注**主流程中真正被调用的代码**，忽略死代码和未使用的函数
- 对 agent 返回的分析结果，如果涉及关键设计决策，需到源码中验证函数是否在主流程中被实际调用

## 技术约束

- 缓存 prefix KV 时**绝不 detach**，必须保留 autograd 计算图，确保梯度正确性
- PrefixTrain_dev 的 `memory_manager/memory.py:42` 使用了 `clone().detach()` 是一个已知 bug，不可复现此错误

## 注入方式决策

- **当前阶段**：采用 monkey-patch 方式注入 Megatron attention 的 prefix KV 拼接逻辑
  - 理由：prefix-sharing 是独立项目，patch 方式简单、易管理、不依赖上游代码修改
- **后续计划**：项目充分验证后，向 verl/megatron 社区提 PR 时，改为对对应类进行继承和扩展的方式
- 两种方式的核心逻辑一致，只是代码组织形式不同，迁移成本可控
