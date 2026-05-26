# prefix-sharing — Agent 开发规范

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

文档、提交、测试等规范统一见仓库根目录 [`AGENTS.md`](../AGENTS.md)。本文件仅保留 prefix-sharing 模块专属的开发和约束规范：

- 文档撰写与管理：见 `AGENTS.md` 第 3 节
- 提交规范 / Cursor Agent 特殊规则：见 `AGENTS.md` 第 7 节
- 测试规范：见 `AGENTS.md` 第 6 节

## 代码分析规范

- 分析 PoC（PrefixTrain_dev）等参考代码时，只关注**主流程中真正被调用的代码**，忽略死代码和未使用的函数
- 对 agent 返回的分析结果，如果涉及关键设计决策，需到源码中验证函数是否在主流程中被实际调用

## 技术约束

- 缓存 prefix KV 时**绝不 detach**，必须保留 autograd 计算图，确保梯度正确性
- PrefixTrain_dev 的 `memory_manager/memory.py:42` 使用了 `clone().detach()` 是一个已知 bug，不可复现此错误

## 注入方式决策

- **当前阶段**：采用 monkey-patch 方式注入 Megatron attention 的 prefix KV 拼接逻辑
  - 理由：作为仓库内的独立模块，patch 方式简单、易管理，可尽量少改 `dependency/` 中的 verl/Megatron 快照
- **后续计划**：项目充分验证后，向 verl/megatron 社区提 PR 时，改为对对应类进行继承和扩展的方式
- 两种方式的核心逻辑一致，只是代码组织形式不同，迁移成本可控
