#!/usr/bin/env python3
"""Load Qwen3.5-27B pretrained weights into verl Megatron model.

Handles:
1. Name mapping: HF `model.language_model.layers.X.linear_attn.Y` → verl `model.layers.X.self_attn.Y`
   (DeltaNet layers use `linear_attn` in HF but `self_attn` in verl decoder)
2. Name mapping: HF `model.language_model.layers.X.self_attn.Y` → verl `model.layers.X.self_attn.Y`
3. DeltaNet in_proj_qkv splitting: (10240, 5120) → q (2048) + k (2048) + v (6144), TP shard each
4. DeltaNet conv1d TP sharding: (10240, 1, 4) → shard by QKV channel portions → (2560, 1, 4) per TP
5. DeltaNet A_log/dt_bias TP sharding: (48,) → (12,) per TP shard
6. DeltaNet norm.weight replication: (128,) → no sharding needed
7. DeltaNet out_proj RowParallelLinear: (5120, 6144) → shard dim=1 → (5120, 1536) per TP
8. Full attn q_proj *2: (12288, 5120) → ColumnParallelLinear shard dim=0
9. Full attn k/v_proj: (1024, 5120) → ColumnParallelLinear shard dim=0
10. Full attn o_proj: (5120, 6144) → RowParallelLinear shard dim=1
11. Full attn q_norm/k_norm: (256,) → replicated (no sharding)
12. MLP gate_proj+up_proj → fused gate_up_proj: concat + TP shard dim=0
13. MLP down_proj: (5120, 17408) → RowParallelLinear shard dim=1
14. embed_tokens: VocabParallelEmbedding (shard vocab dim)
15. lm_head: ColumnParallelLinear (shard output dim)
16. norm.weight: replicated

Usage: torchrun --nproc_per_node=4 scripts/load_weights_qwen36.py
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

# Model paths
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4

# Initialize distributed
torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker,
    get_cuda_rng_tracker,
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=42)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)

tp_rank = parallel_state.get_tensor_model_parallel_rank()
print(f"[Rank {local_rank}] TP rank {tp_rank}")

# Load config
config = AutoConfig.from_pretrained(HF_MODEL_PATH)
print(f"[Rank {local_rank}] Config: {type(config).__name__}")
print(f"[Rank {local_rank}] hidden_size={config.hidden_size}, intermediate_size={config.intermediate_size}")
print(f"[Rank {local_rank}] num_attention_heads={config.num_attention_heads}, num_key_value_heads={config.num_key_value_heads}")
print(f"[Rank {local_rank}] head_dim={getattr(config, 'head_dim', 'N/A')}")
print(f"[Rank {local_rank}] num_hidden_layers={config.num_hidden_layers}")
print(f"[Rank {local_rank}] layer_types count: {len(config.layer_types)}")

# Determine layer types from config
layer_types = config.layer_types
full_attn_interval = config.full_attention_interval

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
model = ParallelQwen3_6ForCausalLMRmPad(config=config, megatron_config=megatron_config)
model = model.to(device)

# Print model parameter names for verification
model_param_names = set(n for n, _ in model.named_parameters())
print(f"[Rank {local_rank}] Model has {len(model_param_names)} parameters")

# Load HF state_dict (all shards)
print(f"[Rank {local_rank}] Loading HF state_dict...")
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path)
    hf_state_dict.update(shard_dict)

# Filter: only keep model.language_model and lm_head keys (skip mtp, vision, etc.)
hf_keys_filtered = {}
for k, v in hf_state_dict.items():
    if k.startswith("model.language_model.") or k.startswith("lm_head."):
        hf_keys_filtered[k] = v

total_params_hf = sum(v.numel() for v in hf_keys_filtered.values())
print(f"[Rank {local_rank}] HF state_dict (filtered): {len(hf_keys_filtered)} tensors, {total_params_hf:,} params")

# Helper: shard a weight tensor for TP
def shard_tensor(tensor, dim, tp_size, tp_rank):
    """Split tensor along dim into tp_size chunks, return chunk for tp_rank."""
    chunks = torch.chunk(tensor, tp_size, dim=dim)
    return chunks[tp_rank].contiguous()

# Helper: split fused QKV into separate q, k, v with TP sharding
def split_deltanet_qkv(in_proj_qkv_weight, config, tp_size, tp_rank):
    """Split in_proj_qkv (10240, 5120) into per-shard q, k, v weights.

    Layout: [key_dim (2048), key_dim (2048), value_dim (6144)]
    For TP: shard each portion independently by its head count.
    """
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim  # 2048
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim  # 6144

    q_portion = in_proj_qkv_weight[:key_dim]           # (2048, 5120) - for query
    k_portion = in_proj_qkv_weight[key_dim:key_dim*2]  # (2048, 5120) - for key
    v_portion = in_proj_qkv_weight[key_dim*2:]         # (6144, 5120) - for value

    # TP shard each portion along output dim (dim=0)
    q_shard = shard_tensor(q_portion, 0, tp_size, tp_rank)
    k_shard = shard_tensor(k_portion, 0, tp_size, tp_rank)
    v_shard = shard_tensor(v_portion, 0, tp_size, tp_rank)

    return q_shard, k_shard, v_shard

# Helper: shard conv1d weight by QKV channel portions
def shard_conv1d_weight(conv1d_weight, config, tp_size, tp_rank):
    """Shard conv1d weight (10240, 1, 4) along channel dim.

    Layout matches the fused QKV: [q_channels, k_channels, v_channels]
    Each portion sharded by its head count, then concatenated back.
    """
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim  # 2048
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim  # 6144

    q_portion = conv1d_weight[:key_dim]
    k_portion = conv1d_weight[key_dim:key_dim*2]
    v_portion = conv1d_weight[key_dim*2:]

    q_shard = shard_tensor(q_portion, 0, tp_size, tp_rank)
    k_shard = shard_tensor(k_portion, 0, tp_size, tp_rank)
    v_shard = shard_tensor(v_portion, 0, tp_size, tp_rank)

    return torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()

# Helper: load MLP weights (gate_proj + up_proj → fused gate_up_proj)
def load_mlp_weights(hf_state_dict, mlp_module, hf_layer_prefix, tp_size, tp_rank, intermediate_size):
    """Load MLP gate_proj + up_proj → fused gate_up_proj, and down_proj."""
    loaded = 0

    # gate_proj and up_proj → fused gate_up_proj
    gate_proj_key = f"{hf_layer_prefix}.mlp.gate_proj.weight"
    up_proj_key = f"{hf_layer_prefix}.mlp.up_proj.weight"

    if gate_proj_key in hf_state_dict and up_proj_key in hf_state_dict:
        gate_weight = hf_state_dict[gate_proj_key]  # (intermediate_size, hidden_size)
        up_weight = hf_state_dict[up_proj_key]       # (intermediate_size, hidden_size)

        # Shard gate and up separately, then concatenate
        # MergedColumnParallelLinear layout: [gate_shard, up_shard]
        gate_shard = shard_tensor(gate_weight, 0, tp_size, tp_rank)
        up_shard = shard_tensor(up_weight, 0, tp_size, tp_rank)
        gate_up_shard = torch.cat([gate_shard, up_shard], dim=0).contiguous()

        mlp_module.gate_up_proj.weight.data.copy_(gate_up_shard.to(torch.bfloat16))
        loaded += 2

    # down_proj → RowParallelLinear, shard along input (dim=1)
    down_proj_key = f"{hf_layer_prefix}.mlp.down_proj.weight"
    if down_proj_key in hf_state_dict:
        shard = shard_tensor(hf_state_dict[down_proj_key], 1, tp_size, tp_rank)
        mlp_module.down_proj.weight.data.copy_(shard.to(torch.bfloat16))
        loaded += 1

    return loaded

# Weight loading loop
loaded_count = 0
skipped_count = 0
missing_keys = []

for layer_idx in range(config.num_hidden_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")

    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn
    mlp_module = decoder_layer.mlp

    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"

    if is_deltanet:
        # === DeltaNet layer weights ===
        hf_prefix = f"{hf_layer_prefix}.linear_attn"

        # in_proj_qkv → split into in_proj_q, in_proj_k, in_proj_v
        key = f"{hf_prefix}.in_proj_qkv.weight"
        if key in hf_state_dict:
            q_shard, k_shard, v_shard = split_deltanet_qkv(
                hf_state_dict[key], config, TP_SIZE, tp_rank
            )
            attn_module.in_proj_q.weight.data.copy_(q_shard.to(torch.bfloat16))
            attn_module.in_proj_k.weight.data.copy_(k_shard.to(torch.bfloat16))
            attn_module.in_proj_v.weight.data.copy_(v_shard.to(torch.bfloat16))
            loaded_count += 3
        else:
            missing_keys.append(key)

        # conv1d weight → shard by QKV channels
        key = f"{hf_prefix}.conv1d.weight"
        if key in hf_state_dict:
            conv1d_shard = shard_conv1d_weight(
                hf_state_dict[key], config, TP_SIZE, tp_rank
            )
            attn_module.conv1d.weight.data.copy_(conv1d_shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # in_proj_z → ColumnParallelLinear, shard along output (dim=0)
        key = f"{hf_prefix}.in_proj_z.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.in_proj_z.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # in_proj_b → ColumnParallelLinear, shard along output (dim=0)
        key = f"{hf_prefix}.in_proj_b.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.in_proj_b.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # in_proj_a → ColumnParallelLinear, shard along output (dim=0)
        key = f"{hf_prefix}.in_proj_a.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.in_proj_a.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # A_log → per v_head, shard along dim=0
        key = f"{hf_prefix}.A_log"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.A_log.data.copy_(shard.to(torch.float32))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # dt_bias → per v_head, shard along dim=0
        key = f"{hf_prefix}.dt_bias"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.dt_bias.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # norm.weight → replicated (head_v_dim=128)
        key = f"{hf_prefix}.norm.weight"
        if key in hf_state_dict:
            attn_module.norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # out_proj → RowParallelLinear, shard along input (dim=1)
        key = f"{hf_prefix}.out_proj.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank)
            attn_module.out_proj.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

    else:
        # === Full attention layer weights ===
        hf_prefix = f"{hf_layer_prefix}.self_attn"

        # q_proj → ColumnParallelLinear, shard along output (dim=0)
        # q_proj has *2 output size (12288, 5120) = fused query + gate
        key = f"{hf_prefix}.q_proj.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.q_proj.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # k_proj → ColumnParallelLinear, shard along output (dim=0)
        key = f"{hf_prefix}.k_proj.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.k_proj.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # v_proj → ColumnParallelLinear, shard along output (dim=0)
        key = f"{hf_prefix}.v_proj.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank)
            attn_module.v_proj.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # o_proj → RowParallelLinear, shard along input (dim=1)
        key = f"{hf_prefix}.o_proj.weight"
        if key in hf_state_dict:
            shard = shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank)
            attn_module.o_proj.weight.data.copy_(shard.to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # q_norm → replicated (head_dim=256 per head)
        key = f"{hf_prefix}.q_norm.weight"
        if key in hf_state_dict:
            attn_module.q_norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

        # k_norm → replicated (head_dim=256 per head)
        key = f"{hf_prefix}.k_norm.weight"
        if key in hf_state_dict:
            attn_module.k_norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        else:
            missing_keys.append(key)

    # === MLP weights (shared for both DeltaNet and full attention) ===
    loaded_mlp = load_mlp_weights(
        hf_state_dict, mlp_module, hf_layer_prefix, TP_SIZE, tp_rank,
        config.intermediate_size
    )
    loaded_count += loaded_mlp

    # input_layernorm → replicated
    key = f"{hf_layer_prefix}.input_layernorm.weight"
    if key in hf_state_dict:
        decoder_layer.input_layernorm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
        loaded_count += 1
    else:
        missing_keys.append(key)

    # post_attention_layernorm → replicated
    key = f"{hf_layer_prefix}.post_attention_layernorm.weight"
    if key in hf_state_dict:
        decoder_layer.post_attention_layernorm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
        loaded_count += 1
    else:
        missing_keys.append(key)

# === Non-layer weights ===
# embed_tokens → VocabParallelEmbedding
# VocabParallelEmbedding shards along vocab dimension (dim=0)
embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_state_dict:
    vocab_size = hf_state_dict[embed_key].shape[0]
    hidden_size = hf_state_dict[embed_key].shape[1]
    # Shard vocab rows per TP rank
    shard = shard_tensor(hf_state_dict[embed_key], 0, TP_SIZE, tp_rank)
    model.model.embed_tokens.weight.data.copy_(shard.to(torch.bfloat16))
    loaded_count += 1
else:
    missing_keys.append(embed_key)

# Final norm → replicated
norm_key = "model.language_model.norm.weight"
if norm_key in hf_state_dict:
    model.model.norm.weight.data.copy_(hf_state_dict[norm_key].to(torch.bfloat16))
    loaded_count += 1
else:
    missing_keys.append(norm_key)

# lm_head → ColumnParallelLinear, shard along output (dim=0)
lm_head_key = "lm_head.weight"
if lm_head_key in hf_state_dict:
    shard = shard_tensor(hf_state_dict[lm_head_key], 0, TP_SIZE, tp_rank)
    model.lm_head.weight.data.copy_(shard.to(torch.bfloat16))
    loaded_count += 1
else:
    missing_keys.append(lm_head_key)

# Report
print(f"[Rank {local_rank}] Loaded {loaded_count} weight tensors")
if missing_keys:
    print(f"[Rank {local_rank}] WARNING: {len(missing_keys)} missing keys:")
    for k in missing_keys[:20]:
        print(f"  - {k}")
    if len(missing_keys) > 20:
        print(f"  ... and {len(missing_keys) - 20} more")

# Verify parameter count
total_model_params = sum(p.numel() for p in model.parameters())
total_loaded_params = sum(p.numel() for n, p in model.named_parameters() if n in model_param_names)
print(f"[Rank {local_rank}] Model total params: {total_model_params:,}")
print(f"[Rank {local_rank}] Per-rank params: {total_loaded_params:,}")

# Test forward pass with loaded weights
batch_size = 2
seq_len = 32
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

print(f"[Rank {local_rank}] Forward pass with loaded weights...")
with torch.no_grad():
    output = model(input_ids=input_ids, attention_mask=attention_mask)

if local_rank == 0:
    print(f"[Rank 0] Output logits shape: {output.logits.shape}")
    print(f"[Rank 0] logits mean: {output.logits.mean().item():.6f}, std: {output.logits.std().item():.6f}")
    print(f"[Rank 0] WEIGHTS LOADED OK!")

alloc_gb = torch.cuda.memory_allocated() / 1024**3
print(f"[Rank {local_rank}] GPU memory: {alloc_gb:.2f} GB")

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()