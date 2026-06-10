# prefix-0501 — Agent 开发规范

> 本文档是 **prefix-0501 整个仓库**的顶层规范，说明目录结构、边界和任务路由。
> prefix-sharing 的详细开发规范见 [`prefix-sharing/AGENTS.md`](prefix-sharing/AGENTS.md)。

---

## 1. 仓库定位

`prefix-0501` 以**整个仓库**为版本管理单元（`docs/`、`survey/`、`dependency/`、`prefix-sharing/` 统一管理），目标是在 `rllm + verl + Megatron` RL pipeline 中实现 **prefix sharing**（同一 micro-batch 内复用共享前缀的 KV，减少重复计算，同时保持 logprob / loss / 梯度语义一致）。

```
prefix-0501/
├── docs/             # 项目文档（概念、架构、历史归档等）
├── survey/           # 调研与 PoC（只读参考）
├── dependency/       # verl / Megatron 依赖快照（必要时少量改动，优先插件化接入）
└── prefix-sharing/   # 正式开发目录
```

---

## 2. 目录边界

| 目录 | 默认行为 | 说明与约束 |
|------|----------|------------|
| `prefix-sharing/` | **可改** | 所有功能开发、测试必须落在 `prefix-sharing/prefix_sharing/`（`core/` → `backends/` → `integrations/` 分层，不要把 core 逻辑写进 integration 层） |
| `docs/` | **可改** | 仓库级项目文档（概念、架构、历史归档等）；撰写与管理规范见下文「文档入口」 |
| `dependency/` | **谨慎改动** | `verl_v070`、`megatron_v0150` 快照；允许少量必要改动，但**优先**通过 `prefix-sharing` 插件化接入（patch / helper / context）；verl / Megatron 内只保留特性使能入口（import、helper 调用、context manager），**不在其中沉淀核心逻辑** |
| `survey/` | **只读** | `PrefixTrain_dev`、`flash-preference`、`dpo-prefix-sharing` 等重要参考；默认只读，除非用户明确要求；**不可**把 PoC 的侵入式 Megatron 魔改直接照搬 |
| 仓库根目录 | **按需** | 维护顶层说明（如本文件）、`.gitignore` 等基础设施 |

---

## 3. 文档入口

项目文档统一放在仓库根目录 `docs/`，由 **`prefix-0501` 仓库**维护，不属于 `prefix-sharing/` 模块职责。

| 文件 | 用途 |
|------|------|
| `docs/concepts.md` | 当前概念、术语、核心语义和设计约定；用于语义争议或命名分歧时对齐 |
| `docs/overview.md` | 当前项目架构、模块关系和主要数据流 |
| `docs/overview.puml` | 当前架构图（PlantUML） |
| `docs/pending-items.md` | 当前明确遗留事项、兼容性缺口和后续待验证场景 |
| `docs/legacy/` | 历史 `doc-*` 文档归档；只作背景参考，不作为当前实现规范 |

文档维护原则：

- 后续开发**不要求**每次记录工作进展日志。
- 概念、术语、语义约定变化时，按需更新 `docs/concepts.md`。
- 架构、模块边界、数据流变化时，按需更新 `docs/overview.md` 和 `docs/overview.puml`。
- 明确遗留事项、暂不处理的兼容性缺口或待验证场景，统一记录到 `docs/pending-items.md`。
- `docs/legacy/` 默认只读，除非用户明确要求整理历史材料。
- 关键设计决策若来自参考代码分析，需回源码验证是否在主流程中被调用。

遇到不同任务，先读对应文档：

| 任务 | 阅读建议 |
|------|----------|
| 写代码 / 修 bug | 先读 `prefix-sharing/AGENTS.md` 和相关代码；涉及语义、术语、分层争议时再查 `docs/concepts.md` |
| 理解架构 / 模块关系 | `docs/overview.md` / `docs/overview.puml` |
| 查概念 / 术语 | `docs/concepts.md` |
| 理解历史方案 | `docs/legacy/` |
| 查遗留事项 | `docs/pending-items.md` |
| 了解模块分层 | `prefix-sharing/prefix_sharing/` 下 `core/`、`backends/`、`integrations/`，并参考 `docs/overview.md` |

---

## 4. 不可违反的技术原则

以下原则在整个仓库内全局有效，细节见 `prefix-sharing/AGENTS.md` 和 `docs/concepts.md`：

1. **TDD 优先** —— 代码开发前优先设计测试用例，并先写能表达目标行为的测试代码，再实现功能或修复问题。该原则不是机械绝对：纯重构、小幅命名调整、文档整理等低风险改动不强制额外补测试，但必须维护较高测试覆盖率，不能让关键语义缺少保护。
2. **精度一致性大于性能** —— prefix sharing 的红线是 logprob / loss / 梯度语义与 baseline 一致。所有设计必须先回答是否影响精度、如何保证精度一致，再考虑性能收益；last token restore、prefix KV 不 detach 等问题都属于精度红线。
3. 精度方案：**One-Forward + KV Injection + Prefix-Last Restore**。
4. **KV 缓存绝不 `detach()`** —— 缓存 prefix KV 时必须保留完整 autograd 计算图，禁止切断梯度。
5. **分层清晰** —— `core/`（框架无关语义）→ `backends/`（硬件执行）→ `integrations/`（verl/Megatron 薄适配）。优先通过 `prefix-sharing` 插件化接入，严格控制对 `dependency/verl_v070`、`dependency/megatron_v0150` 的侵入式修改；verl / Megatron 内只保留必要入口，不沉淀 core 逻辑。

---

## 5. 常见任务指引

### 新增 core 能力

1. 先读 `prefix-sharing/AGENTS.md` 和相关 core 代码。
2. 涉及概念、术语、语义边界时，查 `docs/concepts.md`。
3. 优先在 `prefix-sharing/tests/` 写失败测试，用测试表达目标行为和精度边界。
4. 改 `prefix-sharing/prefix_sharing/core/`。
5. 跑测试（见下文命令）。

### 接入 verl / Megatron

1. **首选**：在 `prefix-sharing/prefix_sharing/integrations/` 实现 patch / helper / context，插件化接入
2. 到 `dependency/` 查调用点，确认最小必要的使能入口
3. 若必须在 verl / Megatron 内改动，只做**最小、可追踪**的修改，不在其中沉淀 prefix-sharing 核心逻辑
4. 先设计集成行为和精度风险，再在 `prefix-sharing/tests/integrated_test/` 补充必要测试

### 分析 survey / dependency 代码

1. `survey/` 以只读分析为主，输出结论到 doc 或回复
2. `dependency/` 可查阅也可按需少量改动，但改动前先评估能否通过 `prefix-sharing` 插件化实现
3. 只关注主流程真正被调用的代码，忽略死代码
4. 不把 PoC 代码直接 copy 进 prefix-sharing

### 改文档

1. 概念 / 术语 / 语义约定变更：更新 `docs/concepts.md`。
2. 架构 / 模块关系 / 数据流变更：更新 `docs/overview.md` 和 `docs/overview.puml`。
3. 路径 / 结构变更：同步更新本文件和 `prefix-sharing/AGENTS.md`。
4. 历史归档 `docs/legacy/` 默认不改，除非用户明确要求。

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

- 修改完成后，默认等待用户明确要求提交；用户要求“提交”时再执行 commit。
- 用户要求提交时，必须先运行与改动范围匹配的必要测试并确认通过；仅知识整理类文档修改可例外，但必须在回复中明确说明未运行测试的原因。
- commit 完成后按用户要求或仓库协作约定尝试 push；若网络或权限策略阻止 push，需报告本地分支 ahead 状态。
- 如果修改未完成或只做了一半，**不要 commit**
- commit message 格式：`[type] <中文简要说明>`
- type 取值：
  - `feat`：新特性或能力
  - `fix`：修复 bug 或错误行为
  - `refactor`：不改变外部行为的代码整理；包括重命名、结构调整，以及在源码中补充/改写**语义说明性注释**（帮助理解实现约束与不变量，但不改运行逻辑）
  - `test`：仅测试相关改动
  - `doc`：仅 `docs/` 等文档内容改动
  - `chore`：与业务代码语义无关的仓库维护；如依赖快照、CI、脚本、配置、ignore 规则等
- 选型提示：改源码注释以澄清实现语义时用 `refactor`，不要误标为 `chore`；改 `docs/` 下文档时用 `doc`

### Cursor Agent 特殊规则

**如果当前 agent 来自 Cursor，则 commit 操作必须获得用户明确同意。** 这与其它 agent 的行为有本质不同：

- **必须获得用户明确同意**后才能执行 `git commit`
- 一旦 commit 完成，**必须立即执行 `git push`**，将修改推送到远程仓库
- 或者等待**用户手动完成提交和推送**

在任何情况下，Cursor agent 都不应在未获得用户明确指令的情况下自动执行 commit 操作。如果用户没有明确说"提交"，agent 应该只完成代码/文档修改并等待用户确认。但一旦用户同意提交，agent 应当完整执行 commit + push 流程

---

## 8. PR 规范

- 每个 PR 的 body 中**必须**包含 `## 测试结果` 小节
- 测试结果小节中需提供：
  - 运行的测试命令
  - 测试通过/失败/跳过的数量摘要
  - 如果有 skip，需简要说明 skip 原因（如缺少 GPU、verl、torch_npu 等本地不可用的 optional 依赖）
  - 如果未跑测试（仅文档/依赖快照等改动），必须在测试结果小节中**明确说明未跑测试的原因**
- PR 提交前，开发者必须确认本地测试已通过；测试结果小节中的数据应与本地实际结果一致
- PR title 格式参照 commit message 格式：`[type] 中文简要说明`

---

## 9. 禁止事项

- 在 `dependency/` 中做超出必要的 verl / Megatron 改动，或把 core 算法散落到 dependency 中
- prefix KV 缓存使用 `detach()` 或切断 autograd 图
- 未经要求修改 `survey/` 源码
- 未经同意自动 commit / push
- 做与任务无关的大范围重构
- 把 PrefixTrain_dev 的 Megatron 魔改方式直接当作最终集成方案

---

## 10. 进一步阅读

- 专项开发规范：[`prefix-sharing/AGENTS.md`](prefix-sharing/AGENTS.md)
- 概念与语义约定：[`docs/concepts.md`](docs/concepts.md)
- 架构说明：[`docs/overview.md`](docs/overview.md)
- 架构图：[`docs/overview.puml`](docs/overview.puml)
- 遗留事项：[`docs/pending-items.md`](docs/pending-items.md)
- 历史归档：[`docs/legacy/`](docs/legacy/)
