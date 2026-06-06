#!/bin/bash
# Run TP4 model instantiation test - captures output to log
set -e
source ~/anaconda3/bin/activate llm
cd ~/rollout-prefix
rm -rf ~/rollout-prefix/prefix-0501/dependency/verl_v070/verl/models/*/__pycache__ ~/rollout-prefix/prefix-0501/dependency/verl_v070/verl/models/qwen3_6/megatron/layers/__pycache__ 2>/dev/null
rm -f test_model_tp4.log

torchrun --nproc_per_node=4 --nnodes=1 --master_port=29502 test_model_tp8.py > test_model_tp4.log 2>&1
echo "EXIT CODE: $?"