#!/usr/bin/env python3
"""Measure prefix pass cost at different prefix lengths."""
import os, sys, time, torch, torch.distributed as dist

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
sys.path.insert(0, verl_path)

dist.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker, get_cuda_rng_tracker, _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=42)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=4, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)

from transformers import AutoConfig
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
config = AutoConfig.from_pretrained(HF_MODEL_PATH, local_files_only=True)
vocab_size = config.vocab_size

from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLM
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=4, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
device = torch.device(f"cuda:{local_rank}")
model = ParallelQwen3_6ForCausalLM(config=config, megatron_config=megatron_config).to(device)

# Warmup with long sequence (sets rotary cache to 512)
ids_w = torch.randint(0, vocab_size, (4, 512), device=device)
mask_w = torch.ones(4, 512, dtype=torch.long, device=device)
pos_w = torch.arange(512, dtype=torch.long, device=device).unsqueeze(0).expand(4, -1)
with torch.no_grad():
    model(ids_w, attention_mask=mask_w, position_ids=pos_w)
torch.cuda.synchronize()

# Measure prefix pass at different lengths
if local_rank == 0:
    print("Prefix pass cost vs sequence length:")
for prefix_len in [64, 128, 256, 512]:
    ids_p = torch.randint(0, vocab_size, (1, prefix_len), device=device)
    mask_p = torch.ones(1, prefix_len, dtype=torch.long, device=device)
    pos_p = torch.arange(prefix_len, dtype=torch.long, device=device).unsqueeze(0)
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(ids_p, attention_mask=mask_p, position_ids=pos_p)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    avg = sum(times) / len(times)
    if local_rank == 0:
        per_tok = avg / prefix_len * 1000
        print(f"  Prefix (1x{prefix_len}): {avg*1000:.1f}ms ({per_tok:.2f}ms/tok)")

parallel_state.destroy_model_parallel()
dist.destroy_process_group()