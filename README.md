# prefix-0501 运行指南

在 verl + Megatron RL pipeline 中实现 prefix sharing，复用共享前缀的 KV 减少重复计算。

## 1. 安装依赖包

在已有前置环境（Python、PyTorch、CUDA/Ascend 等）的基础上，进入各依赖目录执行安装：

```bash
# 按顺序安装
cd dependency/mbridge-main && pip install --no-deps -v -e . && cd ../..
cd dependency/Megatron-LM-core_v0.12.1 && pip install --no-deps -v -e . && cd ../..
cd dependency/MindSpeed-v2.2.0_core_r0.12.1 && pip install --no-deps -v -e . && cd ../..
cd dependency/verl_v070 && pip install --no-deps -v -e . && cd ../..
```

使用 `--no-deps` 避免依赖冲突，各包的运行时依赖需由前置环境提供。

## 2. 启动训练

启动脚本分为 NPU 和 GPU 两个版本。

### NPU 环境

```bash
export PYTHONPATH="/path/to/prefix-0501/prefix-sharing:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0       # NPU 专用：禁用 NZ 格式
export ENABLE_PREFIX_SHARING=1        # 启用 prefix sharing

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer' \
    data.train_files=/path/to/data/512_gsm8k/train.parquet \
    data.val_files=/path/to/data/512_gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=/path/to/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=/path/to/Qwen2.5-0.5B \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    2>&1 | tee log/verl_prefix_demo.log
```

### GPU 环境

```bash
export PYTHONPATH="/path/to/prefix-0501/prefix-sharing:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export ENABLE_PREFIX_SHARING=1          # 启用 prefix sharing

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer' \
    data.train_files=/path/to/data/512_gsm8k/train.parquet \
    data.val_files=/path/to/data/512_gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=/path/to/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    critic.optim.lr=1e-5 \
    critic.model.path=/path/to/Qwen2.5-0.5B \
    critic.ppo_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    2>&1 | tee log/verl_prefix_demo.log
```

### Phase 1 约束

- 仅支持 TP（tensor parallel），不支持 PP（pipeline parallel）和 CP（context parallel）
- 必须开启 `use_remove_padding=True`
- 必须关闭 `use_fused_kernels=False`
- NPU 需设置 `VLLM_ASCEND_ENABLE_NZ=0` 并通过 `override_transformer_config.use_flash_attn=True` 启用 flash attention

## 3. 进阶场景

### 开启 Flash Attention

Flash Attention 在两个层面起作用：**训练引擎**（Megatron/MindSpeed 的 attention 计算）和 **prefix-sharing 后端**（共享前缀的 attention 调度）。两者的配置方式不同，需分别设置。

#### 训练引擎层面

训练引擎的 flash attention 通过 Megatron 的 `use_flash_attn` 参数控制：

**NPU 环境**（必须开启）

NPU 不支持 `flash-attn` pip 包，而是通过 MindSpeed/CANN 的 `npu_fusion_attention` 融合算子实现 flash attention。需在启动命令中通过 `override_transformer_config` 为 actor 和 ref 分别开启：

```bash
export VLLM_ASCEND_ENABLE_NZ=0

# 在启动命令中追加：
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
```

> 注意：使用 MindSpeed 作为训练后端时，flash attention **必须开启**。NPU 的 flash attention 为非确定性计算，与 `--make-vocab-size-divisible-by` 的确定性模式不兼容。

**GPU 环境**（默认已支持）

GPU 环境通过 verl 前置安装脚本安装 `flash-attn` 包。Megatron core 会自动检测并使用 `flash-attn`，通常无需额外配置。如需显式开启：

```bash
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
+actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
```

#### Prefix-Sharing 后端层面

prefix-sharing 的 attention 计算通过 `backend` 参数选择后端实现，当前支持：

| 后端 | 硬件 | 说明 |
|------|------|------|
| `torch_ref`（默认） | GPU + NPU | 纯 PyTorch 实现，兼容性最好 |
| `flash_atten_gpu` | 仅 GPU | 基于 `flash-attn` 包的 `flash_attn_varlen_func`，性能更优 |
| `flash_atten_npu` | NPU | 占位符，暂未实现 |

如需在 GPU 上使用 flash attention 加速 prefix-sharing 路径，通过环境变量或配置设置：

```bash
export PREFIX_SHARING_BACKEND=flash_atten_gpu
```

> 注意：`flash_atten_gpu` 后端要求输入为 THD（varlen）格式，需确保 `use_remove_padding=True`。NPU 环境目前只能使用 `torch_ref` 后端，`flash_atten_npu` 尚在开发中。

### 并行策略

在第 2 节的 NPU 或 GPU 启动命令基础上，追加或覆盖以下增量配置：

```bash
trainer.n_gpus_per_node=2 \
trainer.nnodes=1 \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.use_fused_kernels=False \
actor_rollout_ref.actor.megatron.use_remove_padding=True \
actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2 \
actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1 \
actor_rollout_ref.actor.megatron.context_parallel_size=1 \
actor_rollout_ref.ref.megatron.use_remove_padding=True \
actor_rollout_ref.ref.megatron.tensor_model_parallel_size=2 \
actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
actor_rollout_ref.ref.megatron.context_parallel_size=1 \
actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

第一轮建议先保持 `actor_rollout_ref.rollout.tensor_model_parallel_size=1`，只验证 Megatron actor/ref 的 TP=2 prefix-sharing 路径。若需要同时验证 rollout TP，再单独改为 `2`。

建议测试顺序：

1. `ENABLE_PREFIX_SHARING=0` 跑 TP=2，确认原始 Megatron TP 环境可用。
2. `ENABLE_PREFIX_SHARING=1` 跑 TP=2，确认 prefix-sharing 路径可用。
3. 首轮保持 `trainer.total_training_steps=1`、`trainer.total_epochs=1`、`trainer.val_before_train=False`、`trainer.save_freq=-1`、`trainer.test_freq=-1`。

日志中可关注以下信号，每个 Megatron TP rank 都应分别打印自己的 `tp_rank` 和 `tp_size`：

```text
[PS][prepare] config.enable_prefix_sharing=True
[PS][prepare] PATH 6: sharing detected
[PS][prepare][global_rank=0 tp_rank=0/tp_size=2 cp_rank=0/cp_size=1] packed_batch_layout: valid_lengths=..., padded_lengths=..., cu_seqlens=...
[PS][prepare][global_rank=1 tp_rank=1/tp_size=2 cp_rank=0/cp_size=1] packed_batch_layout: valid_lengths=..., padded_lengths=..., cu_seqlens=...
[PS][attention][global_rank=0 tp_rank=0/tp_size=2 layer=...] enter prefix-sharing path: query_shape=..., key_shape=..., value_shape=...
[PS][attention][global_rank=1 tp_rank=1/tp_size=2 layer=...] enter prefix-sharing path: query_shape=..., key_shape=..., value_shape=...
[PS][attention][global_rank=0 tp_rank=0/tp_size=2 layer=...] built expanded kv: expanded_key_shape=..., expanded_value_shape=...
[PS][attention][global_rank=1 tp_rank=1/tp_size=2 layer=...] built expanded kv: expanded_key_shape=..., expanded_value_shape=...
```

如果只看到 `tp_rank=0/tp_size=2` 而没有 `tp_rank=1/tp_size=2`，优先检查日志聚合方式是否只收集了 rank0；若确认 rank1 日志也被收集但没有 prefix-sharing attention 日志，再排查 TP rank1 是否进入了 Megatron actor/ref forward。`global_rank` 是 torch distributed global rank，不等价于 TP rank；判断 TP 路径是否覆盖完整应看 `tp_rank`。`padded_lengths` 应体现 TP padding，例如有效长度 `[5, 2]` 在 TP=2 下会变成 `[6, 2]`。

如果出现 `PATH 5: no sharing detected`，说明当前 micro-batch 没检测到共享前缀，不代表 TP 失败。可使用 `actor_rollout_ref.rollout.n=8` 等配置增加同 prompt 多 response 的概率。

## 4. 前置环境安装

### verl 依赖

参照 [verl 官方安装文档](https://verl.org.cn/en/latest/start/install.html#install-from-custom-environment)，使用 verl 提供的脚本安装推理框架和基础依赖：

```bash
cd dependency/verl_v070
bash scripts/install_vllm_sglang_mcore.sh
```

该脚本会安装 vLLM、SGLang、FlashAttention、TransformerEngine 等依赖。注意脚本中默认安装的 Megatron-LM 版本与本项目不同，本项目使用 `dependency/Megatron-LM-core_v0.12.1` 的快照版本，需在第 1 步单独安装覆盖。

### MindSpeed（NPU）

MindSpeed 是华为 Ascend NPU 适配层，需要根据 NPU 环境单独安装，请参考 MindSpeed 官方文档和团队经验。

## 5. 常见问题

### numpy 版本

安装完成后可能遇到 numpy 版本过高的问题，根据报错提示降级安装：

```bash
pip install "numpy<2.0.0"
```

### transformer_engine

GPU 环境中可能会报 `transformer_engine` 等包未安装，根据报错安装对应版本：

```bash
pip install transformer-engine
```

### 切换到 Megatron 配置

使用 Megatron 作为训练后端时，需要通过 `--config-name='ppo_megatron_trainer'` 指定 Megatron 配置文件，使 actor、critic、ref 均使用 Megatron 后端：

| 配置文件 | actor | critic | ref |
|----------|-------|--------|-----|
| `ppo_trainer` | dp_actor | dp_critic | dp_ref |
| `ppo_megatron_trainer` | megatron_actor | megatron_critic | megatron_ref |

### mbridge 与 megatron-bridge

verl 通过桥接层实现 HuggingFace 权重与 Megatron 权重的在线转换，目前有两种方式：

- **mbridge**（默认）：社区维护，`megatron.use_mbridge=True` + `megatron.vanilla_mbridge=True`
- **megatron-bridge**：NVIDIA 官方，`megatron.vanilla_mbridge=False`，需要额外安装 [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)

如需使用 megatron-bridge 替代 mbridge，安装后在启动参数中设置：

```bash
actor_rollout_ref.model.megatron.vanilla_mbridge=False
```
