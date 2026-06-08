#!/bin/bash
# PG E2E GRPO Competitive Analysis — Qwen2.5-0.5B-Instruct on RTX 4090
# Uses verl's RayPPOTrainer with FSDP worker + HF rollout + PrefixGrouper
# No vllm/SGLang required — HF rollout for minimal dependency setup

set -e

export PYTHONPATH=~/rollout-prefix/verl-pg:${PYTHONPATH:-}
source ~/anaconda3/bin/activate llm

MODEL_PATH=~/rollout-prefix/models/Qwen2.5-0.5B-Instruct
DATA_PATH=~/rollout-prefix/data/grpo_math/train.parquet
OUTPUT_DIR_PG_ON=~/rollout-prefix/pg_e2e_output/pg_on
OUTPUT_DIR_PG_OFF=~/rollout-prefix/pg_e2e_output/pg_off

# Shared config for both runs
# - GRPO (adv_estimator=grpo, no critic needed)
# - FSDP worker (required by PG)
# - HF rollout (no vllm dependency)
# - use_remove_padding=False (required by PG)
# - n=4 for GRPO sampling
# - 1 GPU (Qwen2.5-0.5B is tiny, fits on 1 GPU)
# - train_batch_size=32 (small for quick test)
# - 5 steps only for timing comparison

COMMON_ARGS="
algorithm.adv_estimator=grpo \
data.train_files=$DATA_PATH \
data.train_batch_size=32 \
data.max_prompt_length=128 \
data.max_response_length=256 \
data.filter_overlong_prompts=True \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.model.use_remove_padding=False \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=32 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.entropy_coeff=0 \
actor_rollout_ref.model.enable_gradient_checkpointing=False \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.rollout.name=hf \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
actor_rollout_ref.rollout.temperature=1.0 \
actor_rollout_ref.rollout.do_sample=True \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.use_kl_in_reward=False \
trainer.critic_warmup=0 \
trainer.logger=['console'] \
trainer.total_epochs=1 \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.save_freq=999 \
trainer.test_freq=999 \
trainer.balance_batch=True \
"

echo "========================================"
echo "Running PG OFF (baseline)"
echo "========================================"

python3 -m verl.trainer.main_ppo \
    $COMMON_ARGS \
    actor_rollout_ref.actor.use_prefix_grouper=False \
    trainer.project_name='pg_competitive' \
    trainer.experiment_name='qwen0.5b_pg_off' \
    hydra.run.dir=$OUTPUT_DIR_PG_OFF

echo "========================================"
echo "Running PG ON (PrefixGrouper enabled)"
echo "========================================"

python3 -m verl.trainer.main_ppo \
    $COMMON_ARGS \
    actor_rollout_ref.actor.use_prefix_grouper=True \
    trainer.project_name='pg_competitive' \
    trainer.experiment_name='qwen0.5b_pg_on' \
    hydra.run.dir=$OUTPUT_DIR_PG_ON

echo "========================================"
echo "Both runs complete! Check logs for timing comparison."
echo "========================================"