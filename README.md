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

## 3. 前置环境安装

### verl 依赖

参照 [verl 官方安装文档](https://verl.org.cn/en/latest/start/install.html#install-from-custom-environment)，使用 verl 提供的脚本安装推理框架和基础依赖：

```bash
cd dependency/verl_v070
bash scripts/install_vllm_sglang_mcore.sh
```

该脚本会安装 vLLM、SGLang、FlashAttention、TransformerEngine 等依赖。注意脚本中默认安装的 Megatron-LM 版本与本项目不同，本项目使用 `dependency/Megatron-LM-core_v0.12.1` 的快照版本，需在第 1 步单独安装覆盖。

### MindSpeed（NPU）

MindSpeed 是华为 Ascend NPU 适配层，需要根据 NPU 环境单独安装，请参考 MindSpeed 官方文档和团队经验。

## 4. 常见问题

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
