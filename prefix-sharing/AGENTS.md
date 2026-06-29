# prefix-sharing — Agent 开发规范

> 本文件仅约定 prefix-sharing 模块的专属开发规范（模块边界、技术约束、注入方式等）。
> 所有开发者必须同时遵守仓库完整开发规范：[`../AGENTS.md`](../AGENTS.md)。

## 项目背景

RL 训练中前缀复用（prefix sharing），目标是插件化集成到 verl+megatron 的 RL pipeline。`prefix-sharing/` 是本仓库中的正式开发目录。

## 工作区结构

```
PrefixSharing/
├── dependency/       # verl, megatron, mindspeed 依赖快照
└── prefix-sharing/   # prefix sharing 核心代码与测试
```

## 技术约束

- **TDD 优先**：新增功能、bug fix、精度语义调整前，优先先写测试表达目标行为
- **精度一致性是红线**：任何 prefix sharing 优化都必须先保证 logprob / loss / 梯度与 baseline 语义一致
- 缓存 prefix KV 时**绝不 detach**，必须保留 autograd 计算图
- PrefixTrain_dev 的 `memory_manager/memory.py:42` 使用了 `clone().detach()` 是一个已知 bug，不可复现此错误

## 模块边界

- `core/`：框架无关语义层，只表达 prefix sharing 计划、裁剪、logprob 语义和 store 生命周期
- `backends/`：执行后端，消费 `PrefixSharingPlan` 和 runtime context 提供的信息
- `integrations/`：verl / Megatron 薄适配层，负责 patch、helper、context、packed layout 和框架调用点接入

## 注入方式决策

- **当前阶段**：采用 monkey-patch 方式注入 Megatron attention 的 prefix KV 拼接逻辑
- **后续计划**：项目充分验证后，向 verl/megatron 社区提 PR 时，改为对对应类进行继承和扩展的方式
