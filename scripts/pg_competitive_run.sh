#!/bin/bash
# PR #4368 (PrefixGrouper) Competitive Analysis - Qwen2.5-0.5B GRPO Training
# This script runs verl GRPO training with and without PrefixGrouper on single GPU
# DO NOT use any prefix-0501 code!

set -e

WORK_DIR="$HOME/rollout-prefix/pg-competitive"
VERL_DIR="$HOME/rollout-prefix/verl-pg"
MODEL_PATH="$HOME/rollout-prefix/models/Qwen2.5-0.5B-Instruct"
DATA_DIR="$HOME/rollout-prefix/pg-competitive/data"

# Choose a free GPU
GPU_ID=${1:-0}
echo "Using GPU $GPU_ID"

export CUDA_VISIBLE_DEVICES=$GPU_ID

cd $VERL_DIR

# Common config
COMMON_ARGS="
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/gsm8k_train.parquet \
    data.val_files=$DATA_DIR/gsm8k_test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=128 \
    data.max_response_length=256 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.response_length=256 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    reward_model.enable=False \
    custom_reward_function.path=$VERL_DIR/verl/utils/reward_score/gsm8k.py \
    custom_reward_function.name=compute_score \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.project_name=pg_competitive_qwen25_05b \
    trainer.logger=[console] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    trainer.save_freq=-1 \
    trainer.val_before_train=False \
    trainer.balance_batch=False \
"

echo "=============================================="
echo "Running WITHOUT PrefixGrouper (baseline)"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    $COMMON_ARGS \
    actor_rollout_ref.actor.use_prefix_grouper=False \
    trainer.experiment_name=qwen25_05b_grpo_no_pg \
    2>&1 | tee $WORK_DIR/results_no_pg.log

echo ""
echo "=============================================="
echo "Running WITH PrefixGrouper"
echo "=============================================="

python3 -m verl.trainer.main_ppo \
    $COMMON_ARGS \
    actor_rollout_ref.actor.use_prefix_grouper=True \
    trainer.experiment_name=qwen25_05b_grpo_pg \
    2>&1 | tee $WORK_DIR/results_pg.log

echo ""
echo "Done! Results saved to $WORK_DIR/results_no_pg.log and $WORK_DIR/results_pg.log"