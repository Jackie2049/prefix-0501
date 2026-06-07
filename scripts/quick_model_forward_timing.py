#!/usr/bin/env python3
"""Quick timing of model.forward() for prefix and suffix passes.

Measures inference-only model.forward() timings to estimate the monkey-patch
PS ON timing. Both PS OFF and PS ON use model.forward() (fused kernels),
so this gives the upper bound for PS speedup.

Usage: torchrun --nproc_per_node=4 scripts/quick_model_forward_timing.py
"""

import os, sys, time

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_sharing_path)
sys.path.insert(0, prefix_path)

import torch
import torch.distributed as dist
from transformers import AutoConfig
from safetensors.torch import load_file

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4
PREFIX_LEN = 64
SUFFIX_LEN = 64

# ===== Initialize distributed =====
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
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)
tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

# ===== Load config and model =====
config = AutoConfig.from_pretrained(HF_MODEL_PATH, local_files_only=True)
vocab_size = config.vocab_size

from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLM
from megatron.core import ModelParallelConfig

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLM(config=config, megatron_config=megatron_config).to(device)

# ===== Load weights (same as other benchmarks) =====
def shard_tensor(tensor, dim, tp_size, tp_rank):
    return torch.chunk(tensor, tp_size, dim=dim)[tp_rank].contiguous()

hf_state_dict = {}
for fname in os.listdir(HF_MODEL_PATH):
    if fname.startswith("model.safetensors") and fname.endswith(".safetensors"):
        hf_state_dict.update(load_file(os.path.join(HF_MODEL_PATH, fname)))

layer_types = config.layer_types
num_layers = config.num_hidden_layers

def split_deltanet_qkv(w, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    return (shard_tensor(w[:key_dim], 0, tp_size, tp_rank),
            shard_tensor(w[key_dim:key_dim*2], 0, tp_size, tp_rank),
            shard_tensor(w[key_dim*2:], 0, tp_size, tp_rank))

def shard_conv1d_weight(w, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    return torch.cat([
        shard_tensor(w[:key_dim], 0, tp_size, tp_rank),
        shard_tensor(w[key_dim:key_dim*2], 0, tp_size, tp_rank),
        shard_tensor(w[key_dim*2:], 0, tp_size, tp_rank),
    ], dim=0).contiguous()

def load_mlp_weights(hf_sd, mlp_module, prefix, tp_size, tp_rank):
    loaded = 0
    gk = f"{prefix}.mlp.gate_proj.weight"
    uk = f"{prefix}.mlp.up_proj.weight"
    if gk in hf_sd and uk in hf_sd:
        g_s = shard_tensor(hf_sd[gk], 0, tp_size, tp_rank)
        u_s = shard_tensor(hf_sd[uk], 0, tp_size, tp_rank)
        mlp_module.gate_up_proj.weight.data.copy_(
            torch.cat([g_s, u_s], dim=0).contiguous().to(torch.bfloat16))
        loaded += 2
    dk = f"{prefix}.mlp.down_proj.weight"
    if dk in hf_sd:
        mlp_module.down_proj.weight.data.copy_(
            shard_tensor(hf_sd[dk], 1, tp_size, tp_rank).to(torch.bfloat16))
        loaded += 1
    return loaded

loaded_count = 0
for layer_idx in range(num_layers):
    is_dn = (layer_types[layer_idx] == "linear_attention")
    dec = model.model.layers[layer_idx]
    attn = dec.self_attn
    mlp = dec.mlp
    pfx = f"model.language_model.layers.{layer_idx}"

    if is_dn:
        dn_pfx = f"{pfx}.linear_attn"
        key = f"{dn_pfx}.in_proj_qkv.weight"
        if key in hf_state_dict:
            q_s, k_s, v_s = split_deltanet_qkv(hf_state_dict[key], config, TP_SIZE, tp_rank)
            attn.in_proj_q.weight.data.copy_(q_s.to(torch.bfloat16))
            attn.in_proj_k.weight.data.copy_(k_s.to(torch.bfloat16))
            attn.in_proj_v.weight.data.copy_(v_s.to(torch.bfloat16))
            loaded_count += 3
        key = f"{dn_pfx}.conv1d.weight"
        if key in hf_state_dict:
            attn.conv1d.weight.data.copy_(
                shard_conv1d_weight(hf_state_dict[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for pn in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{dn_pfx}.{pn}.weight"
            if key in hf_state_dict:
                getattr(attn, pn).weight.data.copy_(
                    shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                loaded_count += 1
        key = f"{dn_pfx}.A_log"
        if key in hf_state_dict:
            attn.A_log.data.copy_(
                shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            loaded_count += 1
        key = f"{dn_pfx}.dt_bias"
        if key in hf_state_dict:
            attn.dt_bias.data.copy_(
                shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        key = f"{dn_pfx}.norm.weight"
        if key in hf_state_dict:
            attn.norm.weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1
        key = f"{dn_pfx}.out_proj.weight"
        if key in hf_state_dict:
            attn.out_proj.weight.data.copy_(
                shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
    else:
        fa_pfx = f"{pfx}.self_attn"
        for pn in ["q_proj", "k_proj", "v_proj"]:
            key = f"{fa_pfx}.{pn}.weight"
            if key in hf_state_dict:
                getattr(attn, pn).weight.data.copy_(
                    shard_tensor(hf_state_dict[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
                loaded_count += 1
        key = f"{fa_pfx}.o_proj.weight"
        if key in hf_state_dict:
            attn.o_proj.weight.data.copy_(
                shard_tensor(hf_state_dict[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for nn in ["q_norm", "k_norm"]:
            key = f"{fa_pfx}.{nn}.weight"
            if key in hf_state_dict:
                getattr(attn, nn).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
                loaded_count += 1

    loaded_count += load_mlp_weights(hf_state_dict, mlp, pfx, TP_SIZE, tp_rank)
    for ln in ["input_layernorm", "post_attention_layernorm"]:
        key = f"{pfx}.{ln}.weight"
        if key in hf_state_dict:
            getattr(dec, ln).weight.data.copy_(hf_state_dict[key].to(torch.bfloat16))
            loaded_count += 1

embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_state_dict:
    model.model.embed_tokens.weight.data.copy_(
        shard_tensor(hf_state_dict[embed_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1
norm_key = "model.language_model.norm.weight"
if norm_key in hf_state_dict:
    model.model.norm.weight.data.copy_(hf_state_dict[norm_key].to(torch.bfloat16))
    loaded_count += 1
lm_key = "lm_head.weight"
if lm_key in hf_state_dict:
    model.lm_head.weight.data.copy_(
        shard_tensor(hf_state_dict[lm_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1

hf_state_dict.clear()
torch.cuda.empty_cache()
if local_rank == 0:
    print(f"Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ===== Timing measurements =====
model.eval()
total_len = PREFIX_LEN + SUFFIX_LEN

if local_rank == 0:
    print(f"\n{'='*60}")
    print("model.forward() timing (fused, no Python loop overhead)")
    print(f"{'='*60}")

torch.manual_seed(42)

# CRITICAL: Run longest sequence first to set rotary_emb cos/sin cache
# (max_seq_len_cached must be >= PREFIX_LEN + SUFFIX_LEN = 128)
# This warmup call sets the cache.
ids_4_128 = torch.randint(0, vocab_size, (4, total_len), device=device)
mask_4_128 = torch.ones(4, total_len, dtype=torch.long, device=device)
pos_4_128 = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(4, -1)

with torch.no_grad():
    model(ids_4_128, attention_mask=mask_4_128, position_ids=pos_4_128)
torch.cuda.synchronize()

def measure(fn, n_runs=10):
    """Measure average time over n_runs."""
    times = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    return sum(times) / len(times)

# --- A: PS OFF n=4 (4 seqs, 128 tokens) ---
avg_off_4 = measure(lambda: model(ids_4_128, attention_mask=mask_4_128, position_ids=pos_4_128))

# --- B: PS OFF n=2 (2 seqs, 128 tokens) ---
ids_2_128 = torch.randint(0, vocab_size, (2, total_len), device=device)
mask_2_128 = torch.ones(2, total_len, dtype=torch.long, device=device)
pos_2_128 = torch.arange(total_len, dtype=torch.long, device=device).unsqueeze(0).expand(2, -1)
avg_off_2 = measure(lambda: model(ids_2_128, attention_mask=mask_2_128, position_ids=pos_2_128))

# --- C: Prefix pass (1 seq, 64 tokens) ---
ids_1_64 = torch.randint(0, vocab_size, (1, PREFIX_LEN), device=device)
mask_1_64 = torch.ones(1, PREFIX_LEN, dtype=torch.long, device=device)
pos_1_64 = torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0)
avg_prefix = measure(lambda: model(ids_1_64, attention_mask=mask_1_64, position_ids=pos_1_64))

# --- D: Suffix pass (4 seqs, 64 tokens) ---
# NOTE: For suffix-only model.forward(), position_ids start from 0.
# In the real PS monkey-patch, the suffix tokens are at global positions
# 64..127, but the suffix-only forward treats them as positions 0..63
# and the monkey-patch adjusts the RoPE positions inside attention.
# For this timing estimate, we use pos 0..63 (same seq_len, different positions).
ids_4_64 = torch.randint(0, vocab_size, (4, SUFFIX_LEN), device=device)
mask_4_64 = torch.ones(4, SUFFIX_LEN, dtype=torch.long, device=device)
pos_4_64 = torch.arange(SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(4, -1)
avg_suffix_4 = measure(lambda: model(ids_4_64, attention_mask=mask_4_64, position_ids=pos_4_64))

# --- E: Suffix pass n=2 (2 seqs, 64 tokens) ---
ids_2_64 = torch.randint(0, vocab_size, (2, SUFFIX_LEN), device=device)
mask_2_64 = torch.ones(2, SUFFIX_LEN, dtype=torch.long, device=device)
pos_2_64 = torch.arange(SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(2, -1)
avg_suffix_2 = measure(lambda: model(ids_2_64, attention_mask=mask_2_64, position_ids=pos_2_64))

# ===== Results =====
if local_rank == 0:
    print(f"\n{'='*60}")
    print("RESULTS: model.forward() timing (no_grad, inference)")
    print(f"{'='*60}")
    print(f"PS OFF n=2 (2x128):  {avg_off_2*1000:.1f}ms")
    print(f"PS OFF n=4 (4x128):  {avg_off_4*1000:.1f}ms")
    print(f"Prefix pass (1x64):  {avg_prefix*1000:.1f}ms")
    print(f"Suffix pass n=2 (2x64): {avg_suffix_2*1000:.1f}ms")
    print(f"Suffix pass n=4 (4x64): {avg_suffix_4*1000:.1f}ms")
    print(f"{'='*60}")
    ps_on_2 = avg_prefix + avg_suffix_2
    ps_on_4 = avg_prefix + avg_suffix_4
    print(f"Estimated monkey-patch PS ON (n=2): {ps_on_2*1000:.0f}ms")
    print(f"Estimated monkey-patch PS ON (n=4): {ps_on_4*1000:.0f}ms")
    print(f"{'='*60}")
    speedup_2 = avg_off_2 / ps_on_2
    speedup_4 = avg_off_4 / ps_on_4
    print(f"Estimated speedup n=2: {speedup_2:.2f}x (theoretical: 1.33x)")
    print(f"Estimated speedup n=4: {speedup_4:.2f}x (theoretical: 1.60x)")
    print(f"{'='*60}")
    print(f"NOTE: These are forward-only (inference) timings.")
    print(f"Training includes backward + optimizer (~2-3x forward cost),")
    print(f"but forward speedup translates to overall training speedup")

parallel_state.destroy_model_parallel()
dist.destroy_process_group()