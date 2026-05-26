# prefix-0501 — Agent 开发规范

> 本文档是 **prefix-0501 整个仓库**的顶层规范，说明目录结构、边界和任务路由。
> prefix-sharing 的详细开发规范见 [`prefix-sharing/AGENTS.md`](prefix-sharing/AGENTS.md)。

---

## 1. 仓库定位

`prefix-0501` 以**整个仓库**为版本管理单元（`docs/`、`survey/`、`dependency/`、`prefix-sharing/` 统一管理），目标是在 `rllm + verl + Megatron` RL pipeline 中实现 **prefix sharing**（同一 micro-batch 内复用共享前缀的 KV，减少重复计算，同时保持 logprob / loss / 梯度语义一致）。

```
prefix-0501/
├── docs/             # 项目文档（概念、设计、进展等）
├── survey/           # 调研与 PoC（只读参考）
├── dependency/       # verl / Megatron 依赖快照（必要时少量改动，优先插件化接入）
└── prefix-sharing/   # 正式开发目录
```

---

## 2. 目录边界

| 目录 | 默认行为 | 说明与约束 |
|------|----------|------------|
| `prefix-sharing/` | **可改** | 所有功能开发、测试必须落在 `prefix-sharing/prefix_sharing/`（`core/` → `backends/` → `integrations/` 分层，不要把 core 逻辑写进 integration 层） |
| `docs/` | **可改** | 仓库级项目文档（设计、进展、术语等）；撰写与管理规范见下文「文档管理规定」 |
| `dependency/` | **谨慎改动** | `verl_v070`、`megatron_v0150` 快照；允许少量必要改动，但**优先**通过 `prefix-sharing` 插件化接入（patch / helper / context）；verl / Megatron 内只保留特性使能入口（import、helper 调用、context manager），**不在其中沉淀核心逻辑** |
| `survey/` | **只读** | `PrefixTrain_dev`、`flash-preference`、`dpo-prefix-sharing` 等重要参考；默认只读，除非用户明确要求；**不可**把 PoC 的侵入式 Megatron 魔改直接照搬 |
| 仓库根目录 | **按需** | 维护顶层说明（如本文件）、`.gitignore` 等基础设施 |

---

## 3. 文档管理规定

项目文档统一放在仓库根目录 `docs/`，由 **`prefix-0501` 仓库**维护，不属于 `prefix-sharing/` 模块职责。

| 文件 | 用途 |
|------|------|
| `docs/doc-progress.md` | 记录所有工作进展 |
| `docs/doc-design-history.md` | 记录历史方案设计、阶段性分析、被推翻方案和备选方案 |
| `docs/doc-designs-final.md` | 记录当前最终方案，作为后续开发依据 |
| `docs/doc-glossary.md` | 全局术语表 |

撰写规范：

- 所有文档均严格按**时间倒序**排列（最新在前）
- 每次记录的抬头格式：`yyyy-mm-dd hh:mm <title>`
- 关键设计决策若来自参考代码分析，需回源码验证是否在主流程中被调用

遇到不同任务，先读对应文档：

| 任务 | 阅读顺序 |
|------|----------|
| 写代码 / 修 bug | `docs/doc-designs-final.md` → `prefix-sharing/AGENTS.md` |
| 理解历史方案 | `docs/doc-design-history.md` |
| 记录工作进展 | `docs/doc-progress.md` |
| 查术语 | `docs/doc-glossary.md` |
| 了解模块分层 | `prefix-sharing/prefix_sharing/` 下 `core/`、`backends/`、`integrations/` |

---

## 4. 不可违反的技术原则

以下原则在整个仓库内全局有效，细节见 `prefix-sharing/AGENTS.md` 和 `docs/doc-designs-final.md`：

1. **正确性优先于性能**
2. 精度方案：**One-Forward + KV Injection + Prefix-Last Restore**
3. 缓存 prefix KV 时**绝不 detach**，必须保留 autograd 计算图
4. 分层：`core/`（框架无关语义）→ `backends/`（硬件执行）→ `integrations/`（verl/Megatron 薄适配）
5. **TDD**：先写表达目标行为的失败测试，再实现；修 bug 先写复现测试

---

## 5. 常见任务指引

### 新增 core 能力

1. 读 `docs/doc-designs-final.md` 确认是否在 Phase 范围内
2. 在 `prefix-sharing/tests/` 写失败测试
3. 改 `prefix-sharing/prefix_sharing/core/`
4. 跑测试（见下文命令）

### 接入 verl / Megatron

1. **首选**：在 `prefix-sharing/prefix_sharing/integrations/` 实现 patch / helper / context，插件化接入
2. 到 `dependency/` 查调用点，确认最小必要的使能入口
3. 若必须在 verl / Megatron 内改动，只做**最小、可追踪**的修改，不在其中沉淀 prefix-sharing 核心逻辑
4. 集成测试放 `prefix-sharing/tests/integrated_test/`

### 分析 survey / dependency 代码

1. `survey/` 以只读分析为主，输出结论到 doc 或回复
2. `dependency/` 可查阅也可按需少量改动，但改动前先评估能否通过 `prefix-sharing` 插件化实现
3. 只关注主流程真正被调用的代码，忽略死代码
4. 不把 PoC 代码直接 copy 进 prefix-sharing

### 改文档

1. 设计变更：更新 `docs/doc-designs-final.md` 或 `docs/doc-design-history.md`
2. 每次完成的工作：在 `docs/doc-progress.md` 顶部追加记录
3. 路径 / 结构变更：同步更新本文件和 `prefix-sharing/AGENTS.md`

---

## 6. 测试

标准本地测试命令（在仓库根目录执行）：

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test \
  prefix-sharing/tests/integrated_test \
  prefix-sharing/tests/system_test
```

- 改动 core / backend / integration 代码后，必须跑与范围匹配的测试
- 缺少 torch / verl / GPU 的 optional 测试可 skip，但核心语义测试必须通过
- 仅文档整理类修改可例外，需在 commit 说明或回复中注明未跑测试的原因

---

## 7. Git 提交规范

- 每一次修改（文档或代码），在修改完结后必须 commit 提交
- 每次 commit 完成后必须 push 到远程仓库
- 如果修改未完成或只做了一半，**不要 commit**
- commit message 格式：`[type] <中文简要说明>`
- type 取值：`feat`(特性)、`fix`(修复)、`chore`(琐事)、`test`(测试)、`doc`(文档)
- 每次提交前必须运行与改动范围匹配的必要测试并确认通过；仅知识整理类文档修改可例外，但必须在提交说明或回复中明确说明未运行测试的原因

### Cursor Agent 特殊规则

**如果当前 agent 来自 Cursor，则 commit 操作必须获得用户明确同意。** 这与其它 agent 的行为有本质不同：

- **必须获得用户明确同意**后才能执行 `git commit`
- 一旦 commit 完成，**必须立即执行 `git push`**，将修改推送到远程仓库
- 或者等待**用户手动完成提交和推送**

在任何情况下，Cursor agent 都不应在未获得用户明确指令的情况下自动执行 commit 操作。如果用户没有明确说"提交"，agent 应该只完成代码/文档修改并等待用户确认。但一旦用户同意提交，agent 应当完整执行 commit + push 流程

---

## 8. 禁止事项

- 在 `dependency/` 中做超出必要的 verl / Megatron 改动，或把 core 算法散落到 dependency 中
- prefix KV 缓存使用 `detach()` 或切断 autograd 图
- 未经要求修改 `survey/` 源码
- 未经同意自动 commit / push
- 做与任务无关的大范围重构
- 把 PrefixTrain_dev 的 Megatron 魔改方式直接当作最终集成方案

---

## 9. 进一步阅读

- 专项开发规范：[`prefix-sharing/AGENTS.md`](prefix-sharing/AGENTS.md)
- 当前主规格：[`docs/doc-designs-final.md`](docs/doc-designs-final.md)
- 工作进展：[`docs/doc-progress.md`](docs/doc-progress.md)
