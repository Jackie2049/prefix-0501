#!/bin/bash
# PG E2E GRPO — verl pipeline, PG ON (PrefixGrouper enabled)
# Qwen2.5-0.5B-Instruct, 1 GPU, HF rollout, FSDP worker
# Key: use_remove_padding=False + use_prefix_grouper=True + balance_batch=True

set -e

source ~/anaconda3/bin/activate llm
export PYTHONPATH=/home/zxw/rollout-prefix/verl-pg:${PYTHONPATH:-}
export VERL_DATAPROTO_SERIALIZATION_METHOD=numpy

MODEL_PATH=/home/zxw/rollout-prefix/models/Qwen2.5-0.5B-Instruct
DATA_PATH=/home/zxw/rollout-prefix/data/grpo_math/train.parquet
OUTPUT_DIR=/home/zxw/rollout-prefix/pg_e2e_output/pg_on

echo "========================================"
echo "Running PG ON (PrefixGrouper enabled) — verl pipeline"
echo "========================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_PATH \
    data.val_files=$DATA_PATH \
    data.train_batch_size=4 \
    data.max_prompt_length=64 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=999 \
    trainer.test_freq=999 \
    trainer.val_before_train=False \
    trainer.project_name='pg_competitive' \
    trainer.experiment_name='qwen0.5b_pg_on' \
    actor_rollout_ref.actor.use_prefix_grouper=True \
    trainer.balance_batch=True \
    +ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH=/home/zxw/rollout-prefix/verl-pg \
    hydra.run.dir=$OUTPUT_DIR

echo "========================================"
echo "PG ON run complete!"
echo "========================================"