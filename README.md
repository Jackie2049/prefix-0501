# prefix-0501 运行指南

在 verl + Megatron RL pipeline 中实现 prefix sharing，复用共享前缀的 KV 减少重复计算。

## 1. 安装依赖包

在已有前置环境（Python、PyTorch、CUDA/Ascend 等）的基础上，进入各依赖目录执行安装：

```bash
# Qwen3.5 新配套（v0.16.1 系列）
cd dependency/Megatron-Bridge_de93536e && pip install --no-deps -v -e . && cd ../..
cd dependency/Megatron-LM-core_v0.16.1 && pip install --no-deps -v -e . && cd ../..
cd dependency/MindSpeed_core_r0.16.0 && pip install --no-deps -v -e . && cd ../..
cd dependency/verl_cdd9014f && pip install --no-deps -v -e . && cd ../..
```

使用 `--no-deps` 避免依赖冲突，各包的运行时依赖需由前置环境提供。

> **版本配套说明**
>
> - **Qwen3.5 系列**：verl_cdd9014f + Megatron-LM core_v0.16.1 + MindSpeed core_r0.16.0 + Megatron-Bridge de93536e
>   - verl、mindspeed、megatron、megatron-bridge四大依赖的配套版本选用参考自 [verl Qwen3.5 NPU 教程](https://verl.readthedocs.io/en/latest/ascend_tutorial/model_support/examples/qwen3_5_122b_npu.html) 
>   - 其他运行时依赖推荐使用以下版本（出自 [官方 Dockerfile](dependency/verl_cdd9014f/docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5) ），或直接使用云道上的zzf-verl080-qwen35镜像：
>
>     | 组件 | 版本 | 说明 |
>     |------|------|------|
>     | CANN | 8.5.2 | toolkit + 芯片 ops + nnal/ATB |
>     | Python | 3.11 | |
>     | torch / torch_npu | 2.9.0 / 2.9.0 | 版本必须匹配 |
>     | torchvision | 0.24.0 | |
>     | vLLM | 0.18.0 | rollout 推理后端 |
>     | vllm-ascend | commit `54879467` | 与 vLLM 0.18.0 配套，NPU 专用 |
>     | transformers | 较新版本（推荐 5.10.2） | 需支持 `qwen3_5` / `qwen3_5_moe`；见第 4 节 |
>     | triton-ascend | 3.2.0 | verl `requirements-npu.txt` 要求 |
>     | numpy | < 2.0.0 | verl `requirements-npu.txt` 要求 |

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
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/data/512_gsm8k/train.parquet \
    data.val_files=/path/to/data/512_gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=/path/to/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.ref.megatron.override_transformer_config.use_flash_attn=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
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
    algorithm.adv_estimator=grpo \
    data.train_files=/path/to/data/512_gsm8k/train.parquet \
    data.val_files=/path/to/data/512_gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=/path/to/Qwen2.5-0.5B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
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

### 当前功能约束

- 支持 TP、物理 PP、Megatron TP-bound SP；不支持 CP、virtual PP
- Qwen2.5 必须开启 `use_remove_padding=True`；Qwen3.5 GDN linear attention 当前不支持 packed THD 格式，需设置 `use_remove_padding=False`（bshd 格式）
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

如需在 GPU 上使用 flash attention 加速 prefix-sharing 路径，在启动命令中追加：

```bash
+actor_rollout_ref.actor.prefix_sharing_config.backend=flash_atten_gpu \
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

当前已支持 Megatron actor/ref 的物理 PP 和 Megatron TP-bound SP。启用 PP 时可按资源覆盖 `pipeline_model_parallel_size`；启用 SP 时在 TP 配置基础上追加：

```bash
actor_rollout_ref.actor.megatron.sequence_parallel=True \
actor_rollout_ref.ref.megatron.sequence_parallel=True
```

SP 支持范围为 Megatron `sequence_parallel=True` 且 prefix-sharing attention hook / prefix-last restore 仍使用 global packed THD token 坐标；若未来训练引擎把 hook 输入改为 SP-local shard，运行时 guard 会显式报错，需要单独适配 local/global 坐标映射。

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

### Qwen3.5 配套（推荐）

参照 [verl Qwen3.5 NPU 教程](https://verl.readthedocs.io/en/latest/ascend_tutorial/model_support/examples/qwen3_5_122b_npu.html)。NPU 环境推荐使用 verl 官方预构建镜像，或基于本仓库 Dockerfile 自行构建。

#### NPU 运行 Qwen3.5 注意事项

在 NPU 上运行 Qwen3.5 时，需要注意以下配置：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| 模型权重 | `Qwen3.5-0.8B` | 推荐使用 Qwen3.5-0.8B 进行测试 |
| vanilla_mbridge | `False` | NPU 必须使用 megatron-bridge 而非 mbridge |
| nvidia-modelopt | 推荐 0.44.0 | NPU 运行依赖包，需确保已安装 |
| total_training_steps | `1` | 建议只跑 1 个 step 进行验证 |

**关键配置说明**：

1. **必须使用 megatron-bridge**：NPU 环境需要将 `vanilla_mbridge` 设为 `False`，让 NPU 走 megatron-bridge 而非 mbridge。在启动命令中添加：
   ```bash
   actor_rollout_ref.model.megatron.vanilla_mbridge=False
   ```

2. **必须安装 nvidia-modelopt**（推荐 0.44.0）：
   ```bash
   pip install nvidia-modelopt==0.44.0
   ```

3. **推荐模型和训练步数**：使用 Qwen3.5-0.8B 权重，并设置 `trainer.total_training_steps=1` 进行快速验证。

#### 官方 Docker 镜像

```bash
# A3（Atlas 800T A3，推荐）
docker pull quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5

# A2（Atlas 200T A2 Box16 / Atlas 900 A2 PODc，910B）
docker pull quay.io/ascend/verl:verl-8.5.2-910b-ubuntu22.04-py3.11-qwen3-5
```

镜像由 `dependency/verl_cdd9014f/docker/ascend/Dockerfile.ascend_8.5.2_a{2,3}_qwen3-5` 构建，职责是提供 **CANN + vLLM rollout + verl + Qwen3.5 transformers** 的运行底座；**不包含** Megatron-LM、MindSpeed、Megatron-Bridge 和 prefix-sharing，这些需在第 1 节另行安装。

镜像内已对齐的版本：

| 组件 | 版本 |
|------|------|
| 操作系统 | Ubuntu 22.04 |
| Python | 3.11 |
| CANN | 8.5.2（toolkit + 芯片 ops + nnal） |
| vLLM | 0.18.0 |
| vllm-ascend | commit `54879467` |
| transformers | commit `cc7ab9be`（源码安装，含 Qwen3.5 模型定义） |
| torch / torch_npu | 2.9.0 / 2.9.0 |
| torchvision | 0.24.0 |
| accelerate | 1.13.0 |
| verl（镜像内） | commit `4045d670` |

> GPU 场景 Megatron + Qwen3.5 可参考 `dependency/verl_cdd9014f/examples/grpo_trainer/run_qwen3_5_122b_a10b_megatron.sh`，使用 `verlai/verl:vllm017.latest` 镜像（vLLM 0.17.x）并额外 `pip install --upgrade transformers`。

#### 从 Dockerfile 自行构建

```bash
cd dependency/verl_cdd9014f/docker/ascend

# A2（910B）
docker build -f Dockerfile.ascend_8.5.2_a2_qwen3-5 -t verl-ascend:8.5.2-a2-qwen3-5 .

# A3
docker build -f Dockerfile.ascend_8.5.2_a3_qwen3-5 -t verl-ascend:8.5.2-a3-qwen3-5 .
```

构建会编译 vllm-ascend custom kernel，耗时较长；A2/A3 仅底层芯片 ops 包不同，上层软件版本一致。

#### 启动容器

```bash
docker run -dit \
    --ipc=host \
    --network host \
    --name prefix-qwen35-npu \
    --privileged \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /usr/sbin:/usr/sbin \
    -v /home:/home \
    -v /data:/data \
    -v /path/to/prefix-0501:/workspace/prefix-0501 \
    quay.io/ascend/verl:verl-8.5.2-a3-ubuntu22.04-py3.11-qwen3-5 \
    /bin/bash

docker exec -it prefix-qwen35-npu bash
```

进入容器后加载昇腾环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

#### 容器内补充安装

镜像可能缺少 Megatron 训练相关依赖，在容器内执行：

```bash
# Megatron 训练栈 + 本项目 verl 快照（覆盖镜像内 verl）
cd /workspace/prefix-0501
cd dependency/Megatron-Bridge_de93536e && pip install --no-deps -v -e . && cd ../..
cd dependency/Megatron-LM-core_v0.16.1 && pip install --no-deps -v -e . && cd ../..
cd dependency/MindSpeed_core_r0.16.0 && pip install --no-deps -v -e . && cd ../..
cd dependency/verl_cdd9014f && pip install --no-deps -v -e . && cd ../..

# 其他可能缺失的依赖（viztracer 推荐 1.1.1，nvidia-modelopt 推荐 0.44.0）
pip install viztracer==1.1.1 flash-linear-attention nvidia-modelopt==0.44.0 nvidia-ml-py nvidia-resiliency-ext megatron-energon
```

**transformers**：Qwen3.5 需要较新版本（推荐 5.10.2）以支持 `qwen3_5` / `qwen3_5_moe` 模型类型：

```bash
pip install --upgrade transformers
# 或指定版本
pip install transformers==5.10.2
```

若未使用官方镜像、需手动安装 vLLM rollout 栈：

```bash
pip install vllm==0.18.0
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && git checkout 54879467c41784a446aa5b486a391d9bfbf488fa
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1 && pip install -v -e . --no-build-isolation
```

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
