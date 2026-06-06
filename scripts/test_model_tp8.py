#!/usr/bin/env python3
"""Test Qwen3-27B model instantiation with TP=8 across all 8 GPUs.
Uses bf16 dtype and Megatron tensor parallelism.
"""
import os
import sys

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_path)
os.environ["PYTHONPATH"] = prefix_path

import torch
from transformers import AutoConfig

# Single-process TP=8 test using torchrun-like manual init
# Each GPU will hold 1/8 of the model parameters
TP_SIZE = 4
WORLD_SIZE = 4

# Initialize distributed - use env:// for torchrun compatibility
torch.distributed.init_process_group(
    backend="nccl",
    init_method="env://",
)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

print(f"[Rank {local_rank}] PyTorch {torch.__version__}, GPU {torch.cuda.get_device_name(local_rank)}")

# Initialize RNG tracker
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    get_cuda_rng_tracker,
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=1234)

# Initialize Megatron parallel state
from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)
print(f"[Rank {local_rank}] Megatron initialized: TP={TP_SIZE}, PP=1")

# Load config
model_path = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
config = AutoConfig.from_pretrained(model_path)
if local_rank == 0:
    print(f"Config: {type(config).__name__}, hidden={config.hidden_size}, layers={config.num_hidden_layers}")
    full_count = sum(1 for t in config.layer_types if t == "full_attention")
    linear_count = sum(1 for t in config.layer_types if t == "linear_attention")
    print(f"  Full attention: {full_count}, Linear: {linear_count}")

# Instantiate model
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

device = torch.device(f"cuda:{local_rank}")
print(f"[Rank {local_rank}] Instantiating model on {device} with bf16...")

try:
    model = ParallelQwen3_6ForCausalLMRmPad(config=config, megatron_config=megatron_config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    param_mem_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
    if local_rank == 0:
        print(f"[Rank 0] Model instantiated!")
        print(f"  Total params per rank: {total_params:,}")
        print(f"  Memory per rank: {param_mem_gb:.2f} GB")
        print(f"  dtype: {next(model.parameters()).dtype}")

    # Forward pass test
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    print(f"[Rank {local_rank}] Forward pass: batch={batch_size}, seq={seq_len}")
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    if local_rank == 0:
        print(f"[Rank 0] Output logits shape: {output.logits.shape}")
        print(f"[Rank 0] Logits dtype: {output.logits.dtype}")
        print(f"[Rank 0] FORWARD PASS OK!")

except Exception as e:
    print(f"[Rank {local_rank}] FAILED: {e}")
    import traceback
    traceback.print_exc()

# GPU memory
alloc_gb = torch.cuda.memory_allocated() / 1024**3
peak_gb = torch.cuda.max_memory_allocated() / 1024**3
print(f"[Rank {local_rank}] GPU memory: {alloc_gb:.2f} GB allocated, {peak_gb:.2f} GB peak")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()