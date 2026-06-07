#!/bin/bash
# Run verl GRPO E2E training with prefix-sharing for Qwen3.6-27B (16-layer)
# Usage: bash scripts/run_grpo_e2e.sh [PS_ON|PS_OFF]

set -e

MODE=${1:-PS_OFF}
WORK_DIR=~/rollout-prefix/prefix-0501

cd $WORK_DIR

# Activate conda
source ~/anaconda3/bin/activate llm

# Set environment
export PYTHONPATH="$WORK_DIR/dependency/verl_v070:$WORK_DIR/prefix-sharing:$PYTHONPATH"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Stop any existing Ray
ray stop 2>/dev/null || true

# Start Ray with 4 GPUs
ray start --head --num-gpus=4 --port=6379

# Wait for Ray to be ready
sleep 3

# Set prefix-sharing mode
if [ "$MODE" = "PS_ON" ]; then
    echo "Running with Prefix-Sharing ON"
    export PREFIX_SHARING_MODE=ON
else
    echo "Running with Prefix-Sharing OFF (default)"
fi

# Run GRPO training
cd $WORK_DIR/dependency/verl_v070
python -m verl.trainer.main_ppo \
    config_name=ppo_trainer \
    +configs@=../../$WORK_DIR/configs/qwen3_6_grpo_ps_16layers.yaml

# Stop Ray after training
ray stop