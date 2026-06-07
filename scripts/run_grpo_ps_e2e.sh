#!/bin/bash
# verl GRPO + PS training test on 8×RTX 4090
# Uses 16-layer reduced Qwen3.6-27B model with prefix-sharing enabled
# Settings: n=4, TP=4, disable mbridge (no transformer_engine on RTX 4090)

set -e

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Activate conda environment
source ~/anaconda3/bin/activate llm

# Paths
MODEL_PATH=$HOME/rollout-prefix/models/Qwen3-27B-text-only-16layers
TRAIN_DATA=$HOME/rollout-prefix/data/synthetic_grpo/train.parquet
TEST_DATA=$HOME/rollout-prefix/data/synthetic_grpo/train.parquet

# Add PS module paths
export PYTHONPATH=$HOME/rollout-prefix/prefix-0501/prefix-sharing:$HOME/rollout-prefix/prefix-0501:$HOME/rollout-prefix/prefix-0501/dependency/verl_v070:$PYTHONPATH

# Ensure Ray is running
ray start --head --port=6379 --num-gpus=8 2>/dev/null || true

# Run verl GRPO with PS
# Key config notes:
# - use_mbridge=False: skip mbridge (no transformer_engine on RTX 4090)
# - TP=4 for both actor.megatron and rollout
# - train_batch_size=8, n=4 → real_batch=32, minimal_bsz=2*4=8 (divisible)
# - ppo_micro_batch_size_per_gpu=2 to keep memory safe
python3 -m verl.trainer.main_ppo --config-path=config \
    --config-name='ppo_megatron_trainer.yaml' \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TEST_DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=64 \
    data.max_response_length=128 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=megatron \
    ++actor_rollout_ref.actor.megatron.use_mbridge=False \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.sequence_parallel=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    +actor_rollout_ref.actor.prefix_sharing_config.enable_prefix_sharing=True \
    +actor_rollout_ref.actor.prefix_sharing_config.min_prefix_len=64 \
    +actor_rollout_ref.actor.prefix_sharing_config.min_group_size=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.ref.megatron.use_mbridge=False \
    ++actor_rollout_ref.ref.megatron.tensor_model_parallel_size=4 \
    ++actor_rollout_ref.ref.megatron.sequence_parallel=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.project_name=prefix_sharing_qwen3_6 \
    trainer.experiment_name=grpo_ps_16layers_e2e \
    trainer.logger=console \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=-1 \
    trainer.val_before_train=False