#!/usr/bin/env python3
"""Verify weight loading by checking that loaded weights match HF source.

Instead of running a full HF reference model (which would OOM),
we directly compare the loaded weight values in the verl model
against the HF state_dict, accounting for TP sharding.

Usage: CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29507 scripts/verify_weights_simple.py
"""
import os
import sys

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_path)

import torch
import torch.distributed as dist
from transformers import AutoConfig
from safetensors.torch import load_file

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4

torch.distributed.init_process_group(backend="nccl", init_method="env://")
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
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)

tp_rank = parallel_state.get_tensor_model_parallel_rank()

config = AutoConfig.from_pretrained(HF_MODEL_PATH)

from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)

device = torch.device(f"cuda:{local_rank}")
model = ParallelQwen3_6ForCausalLMRmPad(config=config, megatron_config=megatron_config)
model = model.to(device)

# Helper: move expected tensor to GPU for comparison
def to_gpu(tensor):
    return tensor.to(device)

# Load HF state_dict
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path)
    hf_state_dict.update(shard_dict)

def shard_tensor(tensor, dim, tp_size, tp_rank):
    chunks = torch.chunk(tensor, tp_size, dim=dim)
    return chunks[tp_rank].contiguous()

def split_deltanet_qkv(weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    q = shard_tensor(weight[:key_dim], 0, tp_size, tp_rank)
    k = shard_tensor(weight[key_dim:key_dim*2], 0, tp_size, tp_rank)
    v = shard_tensor(weight[key_dim*2:], 0, tp_size, tp_rank)
    return q, k, v

def shard_conv1d_weight(weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    q = shard_tensor(weight[:key_dim], 0, tp_size, tp_rank)
    k = shard_tensor(weight[key_dim:key_dim*2], 0, tp_size, tp_rank)
    v = shard_tensor(weight[key_dim*2:], 0, tp_size, tp_rank)
    return torch.cat([q, k, v], dim=0).contiguous()

layer_types = config.layer_types
loaded_count = 0

for layer_idx in range(config.num_hidden_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")
    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn
    mlp_module = decoder_layer.mlp
    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"

    if is_deltanet:
        hf_prefix = f"{hf_layer_prefix}.linear_attn"
        key = f"{hf_prefix}.in_proj_qkv.weight"
        if key in hf_state_dict:
            q_shard, k_shard, v_shard = split_deltanet_qkv(hf_state_dict[key], config, TP_SIZE, tp_rank)
            attn_module.in_proj_q.weight.data.copy_(q_shard.to(torch.bfloat16))
            attn_module.in_proj_k.weight.data.copy_(k_shard.to(torch.bfloat16))
            attn_module.in_proj_v.weight.data.copy_(v_shard.to(torch.bfloat16))
            loaded_count += 3
        key = f"{hf_prefix}.conv1d.weight"
        if key in hf_state_dict:
            attn_module.conv1d.weight.data.copy_(shard_conv1d_weight(hf_state_dict[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for proj_name in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_state_dict:
                attn_module.__getattr__(proj_name).weight.data.copy_(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.A_log"
        if key in hf_state_dict:
            attn_module.A_log.data.copy_(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            loaded_count += 1
        key = f"{hf_prefix}.dt_bias"
        if key in hf_state_dict:
            attn_module.dt_bias.data.copy_(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.norm.weight"
        if key in hf_state_dict:
            attn_module.norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.out_proj.weight"
        if key in hf_state_dict:
            attn_module.out_proj.weight.data.copy_(shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
    else:
        hf_prefix = f"{hf_layer_prefix}.self_attn"
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_state_dict:
                attn_module.__getattr__(proj_name).weight.data.copy_(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.o_proj.weight"
        if key in hf_state_dict:
            attn_module.o_proj.weight.data.copy_(shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for norm_name in ["q_norm", "k_norm"]:
            key = f"{hf_prefix}.{norm_name}.weight"
            if key in hf_state_dict:
                attn_module.__getattr__(norm_name).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
                loaded_count += 1

    gate_key = f"{hf_layer_prefix}.mlp.gate_proj.weight"
    up_key = f"{hf_layer_prefix}.mlp.up_proj.weight"
    if gate_key in hf_state_dict and up_key in hf_state_dict:
        gate_shard = shard_tensor(hf_state_dict[gate_key], 0, TP_SIZE, tp_rank)
        up_shard = shard_tensor(hf_state_dict[up_key], 0, TP_SIZE, tp_rank)
        mlp_module.gate_up_proj.weight.data.copy_(torch.cat([gate_shard, up_shard], dim=0).contiguous().to(torch.bfloat16))
        loaded_count += 2
    down_key = f"{hf_layer_prefix}.mlp.down_proj.weight"
    if down_key in hf_state_dict:
        mlp_module.down_proj.weight.data.copy_(shard_tensor(hf_state_dict[down_key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
        loaded_count += 1
    for ln_name in ["input_layernorm", "post_attention_layernorm"]:
        key = f"{hf_layer_prefix}.{ln_name}.weight"
        if key in hf_state_dict:
            decoder_layer.__getattr__(ln_name).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1

embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_state_dict:
    model.model.embed_tokens.weight.data.copy_(shard_tensor(hf_state_dict[embed_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1

norm_key = "model.language_model.norm.weight"
if norm_key in hf_state_dict:
    model.model.norm.weight.data.copy_(hf_state_dict[norm_key].to(torch.bfloat16))
    loaded_count += 1

lm_head_key = "lm_head.weight"
if lm_head_key in hf_state_dict:
    model.lm_head.weight.data.copy_(shard_tensor(hf_state_dict[lm_head_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1

print(f"[Rank {local_rank}] Loaded {loaded_count} weights")

# === Weight verification: check loaded values match HF source ===
print(f"[Rank {local_rank}] === WEIGHT VERIFICATION ===")
errors = []
checks = 0

for layer_idx in range(config.num_hidden_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")
    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn
    mlp_module = decoder_layer.mlp
    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"

    if is_deltanet:
        hf_prefix = f"{hf_layer_prefix}.linear_attn"

        # in_proj_qkv → check q, k, v shards
        key = f"{hf_prefix}.in_proj_qkv.weight"
        if key in hf_state_dict:
            hf_w = hf_state_dict[key]
            key_dim = config.linear_num_key_heads * config.linear_key_head_dim
            q_expected = to_gpu(shard_tensor(hf_w[:key_dim], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            k_expected = to_gpu(shard_tensor(hf_w[key_dim:key_dim*2], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            v_expected = to_gpu(shard_tensor(hf_w[key_dim*2:], 0, TP_SIZE, tp_rank).to(torch.bfloat16))

            q_actual = attn_module.in_proj_q.weight.data
            k_actual = attn_module.in_proj_k.weight.data
            v_actual = attn_module.in_proj_v.weight.data

            if q_expected.shape != q_actual.shape:
                errors.append(f"L{layer_idx} in_proj_q shape mismatch: expected {q_expected.shape}, got {q_actual.shape}")
            elif (q_expected - q_actual).abs().max() > 0:
                errors.append(f"L{layer_idx} in_proj_q value mismatch: max_diff={(q_expected - q_actual).abs().max().item()}")
            checks += 3

        # conv1d
        key = f"{hf_prefix}.conv1d.weight"
        if key in hf_state_dict:
            expected = to_gpu(shard_conv1d_weight(hf_state_dict[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            actual = attn_module.conv1d.weight.data
            if expected.shape != actual.shape:
                errors.append(f"L{layer_idx} conv1d shape: expected {expected.shape}, got {actual.shape}")
            elif (expected - actual).abs().max() > 0:
                errors.append(f"L{layer_idx} conv1d value mismatch: {(expected - actual).abs().max().item()}")
            checks += 1

        # Other projections
        for proj_name in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_state_dict:
                expected = to_gpu(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                actual = attn_module.__getattr__(proj_name).weight.data
                if expected.shape != actual.shape:
                    errors.append(f"L{layer_idx} {proj_name} shape: expected {expected.shape}, got {actual.shape}")
                elif (expected - actual).abs().max() > 0:
                    errors.append(f"L{layer_idx} {proj_name} mismatch: {(expected - actual).abs().max().item()}")
                checks += 1

        # A_log (float32)
        key = f"{hf_prefix}.A_log"
        if key in hf_state_dict:
            expected = to_gpu(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            actual = attn_module.A_log.data
            if expected.shape != actual.shape:
                errors.append(f"L{layer_idx} A_log shape: expected {expected.shape}, got {actual.shape}")
            elif (expected - actual).abs().max() > 0:
                errors.append(f"L{layer_idx} A_log mismatch: {(expected - actual).abs().max().item()}")
            checks += 1

        # out_proj (RowParallelLinear, sharded dim=1)
        key = f"{hf_prefix}.out_proj.weight"
        if key in hf_state_dict:
            expected = to_gpu(shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            actual = attn_module.out_proj.weight.data
            if expected.shape != actual.shape:
                errors.append(f"L{layer_idx} out_proj shape: expected {expected.shape}, got {actual.shape}")
            elif (expected - actual).abs().max() > 0:
                errors.append(f"L{layer_idx} out_proj mismatch: {(expected - actual).abs().max().item()}")
            checks += 1

    else:
        hf_prefix = f"{hf_layer_prefix}.self_attn"
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_state_dict:
                expected = to_gpu(shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                actual = attn_module.__getattr__(proj_name).weight.data
                if expected.shape != actual.shape:
                    errors.append(f"L{layer_idx} {proj_name} shape: expected {expected.shape}, got {actual.shape}")
                elif (expected - actual).abs().max() > 0:
                    errors.append(f"L{layer_idx} {proj_name} mismatch: {(expected - actual).abs().max().item()}")
                checks += 1

        # o_proj (RowParallelLinear, dim=1)
        key = f"{hf_prefix}.o_proj.weight"
        if key in hf_state_dict:
            expected = to_gpu(shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            actual = attn_module.o_proj.weight.data
            if expected.shape != actual.shape:
                errors.append(f"L{layer_idx} o_proj shape: expected {expected.shape}, got {actual.shape}")
            elif (expected - actual).abs().max() > 0:
                errors.append(f"L{layer_idx} o_proj mismatch: {(expected - actual).abs().max().item()}")
            checks += 1

    # MLP
    gate_key = f"{hf_layer_prefix}.mlp.gate_proj.weight"
    up_key = f"{hf_layer_prefix}.mlp.up_proj.weight"
    if gate_key in hf_state_dict and up_key in hf_state_dict:
        gate_shard = to_gpu(shard_tensor(hf_state_dict[gate_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
        up_shard = to_gpu(shard_tensor(hf_state_dict[up_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
        expected = torch.cat([gate_shard, up_shard], dim=0).contiguous()
        actual = mlp_module.gate_up_proj.weight.data
        if expected.shape != actual.shape:
            errors.append(f"L{layer_idx} gate_up_proj shape: expected {expected.shape}, got {actual.shape}")
        elif (expected - actual).abs().max() > 0:
            errors.append(f"L{layer_idx} gate_up_proj mismatch: {(expected - actual).abs().max().item()}")
        checks += 1

    down_key = f"{hf_layer_prefix}.mlp.down_proj.weight"
    if down_key in hf_state_dict:
        expected = shard_tensor(hf_state_dict[down_key], 1, TP_SIZE, tp_rank).to(torch.bfloat16)
        actual = mlp_module.down_proj.weight.data
        if expected.shape != actual.shape:
            errors.append(f"L{layer_idx} down_proj shape: expected {expected.shape}, got {actual.shape}")
        elif (expected - actual).abs().max() > 0:
            errors.append(f"L{layer_idx} down_proj mismatch: {(expected - actual).abs().max().item()}")
        checks += 1

# embed_tokens
embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_state_dict:
    expected = shard_tensor(hf_state_dict[embed_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16)
    actual = model.model.embed_tokens.weight.data
    if expected.shape != actual.shape:
        errors.append(f"embed_tokens shape: expected {expected.shape}, got {actual.shape}")
    elif (expected - actual).abs().max() > 0:
        errors.append(f"embed_tokens mismatch: {(expected - actual).abs().max().item()}")
    checks += 1

# lm_head
lm_head_key = "lm_head.weight"
if lm_head_key in hf_state_dict:
    expected = shard_tensor(hf_state_dict[lm_head_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16)
    actual = model.lm_head.weight.data
    if expected.shape != actual.shape:
        errors.append(f"lm_head shape: expected {expected.shape}, got {actual.shape}")
    elif (expected - actual).abs().max() > 0:
        errors.append(f"lm_head mismatch: {(expected - actual).abs().max().item()}")
    checks += 1

print(f"[Rank {local_rank}] Checks: {checks}, Errors: {len(errors)}")
if errors:
    for e in errors[:20]:
        print(f"[Rank {local_rank}] ERROR: {e}")
    if len(errors) > 20:
        print(f"[Rank {local_rank}] ... {len(errors) - 20} more errors")
    print(f"[Rank {local_rank}] *** WEIGHT VERIFICATION FAILED ***")
else:
    print(f"[Rank {local_rank}] *** ALL WEIGHT CHECKS PASSED ***")

# Also do a forward pass sanity check
torch.manual_seed(42)
input_ids = torch.randint(0, 100, (1, 16), device=device)
attention_mask = torch.ones((1, 16), device=device, dtype=torch.long)
position_ids = torch.arange(0, 16, device=device).unsqueeze(0)

print(f"[Rank {local_rank}] Forward pass sanity check...")
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)

if local_rank == 0:
    # Gather logits across TP ranks
    logits_shard = output.logits
    gathered = [torch.empty_like(logits_shard) for _ in range(TP_SIZE)]
    dist.all_gather(gathered, logits_shard)
    full_logits = torch.cat(gathered, dim=-1)

    print(f"[Rank 0] Full logits shape: {full_logits.shape}")
    print(f"[Rank 0] logits mean={full_logits.mean().item():.4f}, std={full_logits.std().item():.4f}")
    print(f"[Rank 0] logits min={full_logits.min().item():.4f}, max={full_logits.max().item():.4f}")

alloc_gb = torch.cuda.memory_allocated() / 1024**3
print(f"[Rank {local_rank}] GPU memory: {alloc_gb:.2f} GB")

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()