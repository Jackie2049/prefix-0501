# PrefixSharing

在 `verl + Megatron` RL 训练 pipeline 中复用共享前缀的 KV cache，减少 step/tree 模式下多条轨迹间的冗余前向计算。精度与独立前向完全一致。

## 1. Installation

安装依赖快照（Qwen3.5 配套）：

```bash
cd dependency/Megatron-Bridge_de93536e   && pip install --no-deps -v -e .
cd dependency/Megatron-LM-core_v0.16.1   && pip install --no-deps -v -e .
cd dependency/MindSpeed_core_r0.16.0     && pip install --no-deps -v -e .
cd dependency/verl_cdd9014f              && pip install --no-deps -v -e .
```

安装本模块：

```bash
cd prefix-sharing && pip install -e .
```

> 各依赖版本：verl_cdd9014f + Megatron-LM core_v0.16.1 + MindSpeed core_r0.16.0 + Megatron-Bridge de93536e。

## 2. Quick Start

准备数据：从 HuggingFace 下载 [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)，按 [verl 数据准备指引](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html) 转为 parquet 格式。

下载模型权重：https://huggingface.co/Qwen/Qwen2.5-0.5B

```bash
# 开启 Prefix Sharing
ENABLE_PREFIX_SHARING=1 bash examples/run_prefix_sharing.sh

# 关闭（基线对比）
ENABLE_PREFIX_SHARING=0 bash examples/run_prefix_sharing.sh
```

> 首次跑先设置 `trainer.total_training_steps=1`、`trainer.total_epochs=1` 验证环境可用。

## 3. Architecture

```
Core（前缀检测/复用计划/Store）→ Backends（attention 执行）→ Integrations（verl/Megatron 适配）
```

模块化分层，核心语义与框架无关。

## 4. Dependencies

| 依赖 | 版本 |
|------|------|
| verl | cdd9014f |
| Megatron-LM core | v0.16.1 |
| MindSpeed core | r0.16.0 |
| Megatron-Bridge | de93536e |

## 5. Citation

```bibtex
@misc{prefixsharing2026,
  title={PrefixSharing: Prefix KV Sharing for verl + Megatron RL Training},
  year={2026},
}
```

## License

MIT
