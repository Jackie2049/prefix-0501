#!/usr/bin/env python3
"""Test Qwen3-27B model instantiation with verl's Megatron implementation.
Runs on a single GPU to verify the model can be constructed and do a forward pass.
Must initialize Megatron parallel state first.
"""
import os
import sys

# Setup paths
verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_path)
os.environ["PYTHONPATH"] = prefix_path

import torch
from transformers import AutoConfig

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name}, {mem_gb:.1f} GB")

# Initialize Megatron parallel state for single GPU (TP=1, PP=1)
from megatron.core import parallel_state

# Must initialize torch.distributed first
torch.distributed.init_process_group(
    backend="nccl",
    init_method="tcp://127.0.0.1:29500",
    world_size=1,
    rank=0,
)
print("torch.distributed initialized (world_size=1)")

# Initialize Megatron parallel state
from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    get_cuda_rng_tracker,
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
# Must add model-parallel RNG state explicitly
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=1234)
print("RNG tracker initialized with model-parallel-rng state")

parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)
print("Megatron parallel state initialized (TP=1, PP=1)")

# Load config
model_path = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
config = AutoConfig.from_pretrained(model_path)
print(f"\nConfig: {type(config).__name__}")
print(f"  hidden_size={config.hidden_size}, layers={config.num_hidden_layers}")
print(f"  heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}")
print(f"  head_dim={config.head_dim}, vocab={config.vocab_size}")
print(f"  full_attention_interval={config.full_attention_interval}")
full_count = sum(1 for t in config.layer_types if t == "full_attention")
linear_count = sum(1 for t in config.layer_types if t == "linear_attention")
print(f"  Full attention: {full_count}, Linear: {linear_count}")

# Import and instantiate model
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,  # Use bf16 to fit in 24GB GPU memory
)

device = torch.device("cuda:0")
print(f"\nInstantiating model on {device}...")

try:
    model = ParallelQwen3_6ForCausalLMRmPad(config=config, megatron_config=megatron_config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated! Total params: {total_params:,}")
    print(f"Model device: {next(model.parameters()).device}")

    # Forward pass test
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

    print(f"\nForward pass: batch={batch_size}, seq={seq_len}")
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"  Logits shape: {output.logits.shape}")
    print(f"  Logits dtype: {output.logits.dtype}")
    logits_min = output.logits.min().item()
    logits_max = output.logits.max().item()
    print(f"  Logits range: [{logits_min:.4f}, {logits_max:.4f}]")
    print("FORWARD PASS OK!")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

# GPU memory
if torch.cuda.is_available():
    alloc_gb = torch.cuda.memory_allocated() / 1024**3
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nGPU memory: {alloc_gb:.2f} GB allocated, {peak_gb:.2f} GB peak")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()