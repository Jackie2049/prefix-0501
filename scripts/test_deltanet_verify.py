#!/usr/bin/env python3
"""Verify DeltaNet forward computation on single GPU."""
import os, sys
verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_path)

import torch
from transformers import AutoConfig
from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel
from megatron.core.tensor_parallel.random import initialize_rng_tracker, get_cuda_rng_tracker, _MODEL_PARALLEL_RNG_TRACKER_NAME

torch.cuda.set_device(0)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=42)

torch.distributed.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:29597", world_size=1, rank=0)
parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

model_path = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
config = AutoConfig.from_pretrained(model_path)
megatron_config = ModelParallelConfig(tensor_model_parallel_size=1, pipeline_model_parallel_size=1, sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16)

from verl.models.qwen3_6.megatron.layers.parallel_deltanet import ParallelQwen3_6GatedDeltaNet
deltanet = ParallelQwen3_6GatedDeltaNet(config=config, megatron_config=megatron_config).cuda()

x = torch.randn(2, 32, config.hidden_size, device="cuda:0", dtype=torch.bfloat16)
with torch.no_grad():
    out = deltanet(x)

cos_sim = torch.nn.functional.cosine_similarity(x.flatten(), out.flatten(), dim=0)
max_diff = (x - out).abs().max().item()

print(f"DeltaNet forward: input {x.shape} -> output {out.shape}")
print(f"Output dtype: {out.dtype}")
print(f"Cosine similarity (input vs output): {cos_sim.item():.6f}")
print(f"Max abs difference: {max_diff:.6f}")
print(f"DELTANET NOT IDENTITY: {cos_sim.item() < 0.99}")

print(f"Has in_proj_q: {hasattr(deltanet, 'in_proj_q')}")
print(f"Has conv1d: {hasattr(deltanet, 'conv1d')}")
print(f"Has A_log: {hasattr(deltanet, 'A_log')}")
print(f"Has norm (RMSNormGated): {hasattr(deltanet, 'norm')}")
print(f"conv1d weight shape: {deltanet.conv1d.weight.shape}")
print(f"A_log shape: {deltanet.A_log.shape}")
print(f"out_proj input_size: {deltanet.out_proj.weight.shape[1]}")

# Verify no old attributes exist
print(f"Has beta_proj (OLD): {hasattr(deltanet, 'beta_proj')}")
print(f"Has decay_proj (OLD): {hasattr(deltanet, 'decay_proj')}")
print(f"Has gate_proj (OLD): {hasattr(deltanet, 'gate_proj')}")

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()