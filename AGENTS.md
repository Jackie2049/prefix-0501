# PrefixAttention — 开发者规范

> 本文档是本仓库的顶层规范，说明目录结构、边界和开发约定。

## 1. 仓库定位

本仓库目标是在 `verl + Megatron` RL pipeline 中实现 **prefix sharing**（同一 micro-batch 内复用共享前缀的 KV，减少重复计算，保持 logprob / loss / 梯度语义一致）。

## 2. 目录结构

```
PrefixAttention/
├── dependency/       # verl / Megatron / MindSpeed 依赖快照
└── prefix-sharing/   # prefix sharing 核心实现
```

## 3. 技术约束

1. **精度一致性大于性能** — prefix sharing 的红线是 logprob / loss / 梯度语义与 baseline 一致
2. **精度方案**：One-Forward + KV Injection + Prefix-Last Restore
3. **KV 缓存绝不 `detach()`** — 必须保留完整 autograd 计算图
4. **分层清晰** — `core/`（框架无关语义）→ `backends/`（硬件执行）→ `integrations/`（框架适配）

## 4. 测试

```bash
PYTHONPATH=prefix-sharing pytest -q \
  prefix-sharing/tests/unit_test \
  prefix-sharing/tests/integrated_test \
  prefix-sharing/tests/system_test
```

## 5. 贡献

见 [CONTRIBUTING.md](CONTRIBUTING.md)。
