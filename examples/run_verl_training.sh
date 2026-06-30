#!/usr/bin/env bash
# PrefixSharing demo — Qwen2.5-0.5B, GSM8K, 1 GPU
#
# Usage:
#   bash examples/run_prefix_sharing.sh       # with prefix sharing (default)
#   ENABLE_PREFIX_SHARING=1 bash examples/run_prefix_sharing.sh   # explicit on
#   ENABLE_PREFIX_SHARING=0 bash examples/run_prefix_sharing.sh   # baseline对比

set -euo pipefail

# ── 前置条件 ────────────────────────────────────────────────
# 1. 安装 dependency 配套：
#    cd dependency/Megatron-Bridge_de93536e   && pip install --no-deps -v -e .
#    cd dependency/Megatron-LM-core_v0.16.1   && pip install --no-deps -v -e .
#    cd dependency/MindSpeed_core_r0.16.0     && pip install --no-deps -v -e .
#    cd dependency/verl_cdd9014f              && pip install --no-deps -v -e .
# 2. 安装本模块：
#    cd prefix-sharing && pip install -e .
# 3. 准备 GSM8K 数据（见 README Quick Start）
# 4. 下载 Qwen2.5-0.5B 权重
#    https://huggingface.co/Qwen/Qwen2.5-0.5B

# ── 用户可配置参数 ──────────────────────────────────────────
MODEL_PATH=${MODEL_PATH:-/path/to/Qwen2.5-0.5B}
TRAIN_FILE=${TRAIN_FILE:-/path/to/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-/path/to/gsm8k/test.parquet}
ENABLE_PREFIX_SHARING=${ENABLE_PREFIX_SHARING:-1}

# ── 路径设置（假设脚本从仓库根目录执行）────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_DIR}/prefix-sharing:${PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export ENABLE_PREFIX_SHARING=${ENABLE_PREFIX_SHARING}

echo "=============================="
echo " PrefixSharing Demo"
echo " GPU count      : 1"
echo " Model          : ${MODEL_PATH}"
echo " PS enabled     : ${ENABLE_PREFIX_SHARING}"
echo "=============================="

# ── 启动训练 ────────────────────────────────────────────────
python3 -m verl.trainer.main_ppo \
    --config-name='ppo_megatron_trainer' \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=128 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
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
    trainer.total_epochs=1
