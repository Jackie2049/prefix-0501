#!/usr/bin/env python3
"""Benchmark prefix-sharing vs normal forward throughput.

Simulates GRPO n=8 rollout scenario:
- 8 sequences sharing a common prefix (e.g., same prompt)
- Prefix pass: 1 forward with provider's prefix tokens (1×prefix_len)
- Suffix pass: 1 forward with all sequences' suffix tokens (8×suffix_len)
- Normal forward: 8×(prefix_len + suffix_len) tokens

Measures:
1. Throughput: tokens processed per second
2. Time comparison: normal vs PS
3. Memory: peak GPU memory
4. Precision: cos_sim between PS and normal logits

Usage: torchrun --nproc_per_node=4 scripts/benchmark_ps_throughput.py
"""

import os
import sys
import time

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path)
sys.path.insert(0, prefix_sharing_path)
sys.path.insert(0, prefix_path)

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from safetensors.torch import load_file

# ===== Configuration =====
HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only")
TP_SIZE = 4
PREFIX_LEN = 256   # Longer prefix (common prompt length)
SUFFIX_LEN = 64    # Short response (typical GRPO)
N_SEQUENCES = 8    # GRPO n=8
SEED = 42
N_WARMUP = 2       # Warmup iterations
N_MEASURE = 5      # Measurement iterations

# ===== Initialize distributed =====
torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker, get_cuda_rng_tracker, _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker()
rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=SEED)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    expert_model_parallel_size=1,
)

tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

# ===== Load config =====
config = AutoConfig.from_pretrained(HF_MODEL_PATH)

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,
    bf16=True,
    params_dtype=torch.bfloat16,
)

# Instantiate RmPad model
model = ParallelQwen3_6ForCausalLMRmPad(config, megatron_config)
model = model.to(device)

if local_rank == 0:
    print(f"RmPad model instantiated, params_dtype={megatron_config.params_dtype}")

# ===== Load pretrained weights =====
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path)
    hf_state_dict.update(shard_dict)

hf_keys_filtered = {}
for k, v in hf_state_dict.items():
    if k.startswith("model.language_model.") or k.startswith("lm_head."):
        hf_keys_filtered[k] = v

layer_types = config.layer_types
num_layers = config.num_hidden_layers


def shard_tensor(tensor, dim, tp_size, tp_rank):
    chunks = torch.chunk(tensor, tp_size, dim=dim)
    return chunks[tp_rank].contiguous()


def split_deltanet_qkv(in_proj_qkv_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    q_portion = in_proj_qkv_weight[:key_dim]
    k_portion = in_proj_qkv_weight[key_dim:key_dim*2]
    v_portion = in_proj_qkv_weight[key_dim*2:]
    return (shard_tensor(q_portion, 0, tp_size, tp_rank),
            shard_tensor(k_portion, 0, tp_size, tp_rank),
            shard_tensor(v_portion, 0, tp_size, tp_rank))


def shard_conv1d_weight(conv1d_weight, config, tp_size, tp_rank):
    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    q_portion = conv1d_weight[:key_dim]
    k_portion = conv1d_weight[key_dim:key_dim*2]
    v_portion = conv1d_weight[key_dim*2:]
    return torch.cat([
        shard_tensor(q_portion, 0, tp_size, tp_rank),
        shard_tensor(k_portion, 0, tp_size, tp_rank),
        shard_tensor(v_portion, 0, tp_size, tp_rank),
    ], dim=0).contiguous()


loaded_count = 0
for layer_idx in range(num_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")
    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn
    mlp_module = decoder_layer.mlp
    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"

    # MLP (same for both)
    gate_key = f"{hf_layer_prefix}.mlp.gate_proj.weight"
    up_key = f"{hf_layer_prefix}.mlp.up_proj.weight"
    if gate_key in hf_keys_filtered and up_key in hf_keys_filtered:
        gate_shard = shard_tensor(hf_keys_filtered[gate_key], 0, TP_SIZE, tp_rank)
        up_shard = shard_tensor(hf_keys_filtered[up_key], 0, TP_SIZE, tp_rank)
        gate_up_shard = torch.cat([gate_shard, up_shard], dim=0).contiguous()
        mlp_module.gate_up_proj.weight.data.copy_(gate_up_shard.to(torch.bfloat16))
        loaded_count += 2
    down_key = f"{hf_layer_prefix}.mlp.down_proj.weight"
    if down_key in hf_keys_filtered:
        shard = shard_tensor(hf_keys_filtered[down_key], 1, TP_SIZE, tp_rank)
        mlp_module.down_proj.weight.data.copy_(shard.to(torch.bfloat16))
        loaded_count += 1

    if is_deltanet:
        hf_prefix = f"{hf_layer_prefix}.linear_attn"
        key = f"{hf_prefix}.in_proj_qkv.weight"
        if key in hf_keys_filtered:
            q_s, k_s, v_s = split_deltanet_qkv(hf_keys_filtered[key], config, TP_SIZE, tp_rank)
            attn_module.in_proj_q.weight.data.copy_(q_s.to(torch.bfloat16))
            attn_module.in_proj_k.weight.data.copy_(k_s.to(torch.bfloat16))
            attn_module.in_proj_v.weight.data.copy_(v_s.to(torch.bfloat16))
            loaded_count += 3
        key = f"{hf_prefix}.conv1d.weight"
        if key in hf_keys_filtered:
            attn_module.conv1d.weight.data.copy_(
                shard_conv1d_weight(hf_keys_filtered[key], config, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for proj_name in ["in_proj_z", "in_proj_b", "in_proj_a"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_keys_filtered:
                shard = shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank)
                getattr(attn_module, proj_name).weight.data.copy_(shard.to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.A_log"
        if key in hf_keys_filtered:
            attn_module.A_log.data.copy_(
                shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.float32))
            loaded_count += 1
        key = f"{hf_prefix}.dt_bias"
        if key in hf_keys_filtered:
            attn_module.dt_bias.data.copy_(
                shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.norm.weight"
        if key in hf_keys_filtered:
            attn_module.norm.weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
            loaded_count += 1
        key = f"{hf_prefix}.out_proj.weight"
        if key in hf_keys_filtered:
            attn_module.out_proj.weight.data.copy_(
                shard_tensor(hf_keys_filtered[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
    else:
        hf_prefix = f"{hf_layer_prefix}.self_attn"
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            key = f"{hf_prefix}.{proj_name}.weight"
            if key in hf_keys_filtered:
                shard = shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank)
                getattr(attn_module, proj_name).weight.data.copy_(shard.to(torch.bfloat16))
                loaded_count += 1
        key = f"{hf_prefix}.o_proj.weight"
        if key in hf_keys_filtered:
            attn_module.o_proj.weight.data.copy_(
                shard_tensor(hf_keys_filtered[key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
            loaded_count += 1
        for norm_name in ["q_norm", "k_norm"]:
            key = f"{hf_prefix}.{norm_name}.weight"
            if key in hf_keys_filtered:
                getattr(attn_module, norm_name).weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
                loaded_count += 1

    for ln_name in ["input_layernorm", "post_attention_layernorm"]:
        key = f"{hf_layer_prefix}.{ln_name}.weight"
        if key in hf_keys_filtered:
            getattr(decoder_layer, ln_name).weight.data.copy_(hf_keys_filtered[key].to(torch.bfloat16))
            loaded_count += 1

embed_key = "model.language_model.embed_tokens.weight"
if embed_key in hf_keys_filtered:
    model.model.embed_tokens.weight.data.copy_(
        shard_tensor(hf_keys_filtered[embed_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1
norm_key = "model.language_model.norm.weight"
if norm_key in hf_keys_filtered:
    model.model.norm.weight.data.copy_(hf_keys_filtered[norm_key].to(torch.bfloat16))
    loaded_count += 1
lm_key = "lm_head.weight"
if lm_key in hf_keys_filtered:
    model.lm_head.weight.data.copy_(
        shard_tensor(hf_keys_filtered[lm_key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
    loaded_count += 1

hf_state_dict.clear()
hf_keys_filtered.clear()
torch.cuda.empty_cache()

mem_after_load = torch.cuda.memory_allocated() / 1024**3
print(f"[Rank {local_rank}] Loaded {loaded_count} weights, GPU {mem_after_load:.2f} GB")

# ===== Create test sequences (GRPO n=8: all share same prefix) =====
vocab_size = config.vocab_size
torch.manual_seed(SEED + 500)

prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                      for _ in range(N_SEQUENCES)]

full_sequences = []
for suffix in suffix_tokens_list:
    full_seq = torch.cat([prefix_tokens, suffix])
    full_sequences.append(full_seq)

full_input_ids = torch.stack(full_sequences)
suffix_input_ids = torch.stack(suffix_tokens_list)
prefix_input_ids = prefix_tokens.unsqueeze(0)

# ===== Benchmark: Normal forward =====
full_attention_mask = torch.ones_like(full_input_ids, dtype=torch.bool, device=device)
full_position_ids = torch.arange(PREFIX_LEN + SUFFIX_LEN, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

if local_rank == 0:
    print(f"=== Benchmark: Normal forward (N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}) ===")
    print(f"Total tokens per batch: {N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN)}")

torch.cuda.synchronize()
# Warmup
for _ in range(N_WARMUP):
    with torch.no_grad():
        output = model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                       position_ids=full_position_ids)
    del output

torch.cuda.synchronize()
t_normal_total = 0
for i in range(N_MEASURE):
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output_normal = model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                              position_ids=full_position_ids)
    torch.cuda.synchronize()
    t_normal_total += time.time() - t0
    if i == 0:
        logits_normal = output_normal.logits

t_normal = t_normal_total / N_MEASURE
normal_tokens = N_SEQUENCES * (PREFIX_LEN + SUFFIX_LEN)
normal_throughput = normal_tokens / t_normal

if local_rank == 0:
    print(f"Normal: {t_normal:.3f}s/batch, {normal_throughput:.0f} tok/s")
    mem_normal = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory (normal): {mem_normal:.2f} GB")

# ===== Benchmark: Two-pass PS forward =====
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner

ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=PREFIX_LEN, min_group_size=N_SEQUENCES)
integration = VerlQwen3_6Integration(config=ps_config)
patch_handle = integration.install(model_config=config)

if local_rank == 0:
    print(f"=== Benchmark: Two-pass PS forward ===")

from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.backends.packed_layout import PackedBatchLayout

sequences_list = [seq.tolist() for seq in full_input_ids]
ps_plan = PrefixSharingPlanner(ps_config).plan(sequences_list)

if local_rank == 0:
    print(f"PS plan: has_sharing={ps_plan.has_sharing}")

prefix_attention_mask = torch.ones(1, PREFIX_LEN, dtype=torch.bool, device=device)
prefix_position_ids = torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0)
suffix_attention_mask = torch.ones_like(suffix_input_ids, dtype=torch.bool, device=device)
suffix_position_ids = torch.arange(PREFIX_LEN, PREFIX_LEN + SUFFIX_LEN, dtype=torch.long, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

suffix_lengths = [SUFFIX_LEN] * N_SEQUENCES
suffix_packed_layout = PackedBatchLayout.from_valid_lengths(suffix_lengths)

ps_runtime_state = PrefixSharingRuntimeState(
    prefix_sharing_plan=ps_plan,
    backend=None,
    packed_batch_layout=suffix_packed_layout,
    model_spec=ModelSpec.from_hf_config(config),
    prefix_input_ids=prefix_input_ids,
    prefix_attention_mask=prefix_attention_mask,
    prefix_position_ids=prefix_position_ids,
)

# Warmup
for _ in range(N_WARMUP):
    with prefix_sharing_runtime_context(ps_runtime_state) as ctx_state:
        with torch.no_grad():
            prefix_output = model(
                input_ids=ps_runtime_state.prefix_input_ids,
                attention_mask=ps_runtime_state.prefix_attention_mask,
                position_ids=ps_runtime_state.prefix_position_ids,
            )
        del prefix_output
        with torch.no_grad():
            suffix_output = model(
                input_ids=suffix_input_ids,
                attention_mask=suffix_attention_mask,
                position_ids=suffix_position_ids,
            )
        del suffix_output

torch.cuda.synchronize()
t_ps_total = 0
for i in range(N_MEASURE):
    torch.cuda.synchronize()
    t0 = time.time()
    with prefix_sharing_runtime_context(ps_runtime_state) as ctx_state:
        with torch.no_grad():
            prefix_output = model(
                input_ids=ps_runtime_state.prefix_input_ids,
                attention_mask=ps_runtime_state.prefix_attention_mask,
                position_ids=ps_runtime_state.prefix_position_ids,
            )
        del prefix_output
        with torch.no_grad():
            suffix_output = model(
                input_ids=suffix_input_ids,
                attention_mask=suffix_attention_mask,
                position_ids=suffix_position_ids,
            )
        logits_ps = suffix_output.logits
    torch.cuda.synchronize()
    t_ps_total += time.time() - t0

t_ps = t_ps_total / N_MEASURE
ps_tokens = PREFIX_LEN + N_SEQUENCES * SUFFIX_LEN  # Prefix (1×) + Suffix (N×)
ps_throughput = ps_tokens / t_ps

if local_rank == 0:
    print(f"PS: {t_ps:.3f}s/batch, {ps_throughput:.0f} tok/s (effective)")
    mem_ps = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory (PS): {mem_ps:.2f} GB")

# ===== Precision check =====
if local_rank == 0:
    print("=== Precision alignment ===")

normal_suffix_logits = logits_normal[:, PREFIX_LEN:, :]
ps_suffix_logits = logits_ps[:, :, :]

all_cos_sims = []
for i in range(N_SEQUENCES):
    cos_sim = F.cosine_similarity(
        ps_suffix_logits[i].float().flatten(),
        normal_suffix_logits[i].float().flatten(),
        dim=0,
    ).item()
    all_cos_sims.append(cos_sim)
    if local_rank == 0:
        print(f"Seq {i}: cos_sim={cos_sim:.6f}")

overall_cos = sum(all_cos_sims) / len(all_cos_sims)

# ===== Uninstall patches =====
patch_handle.disable()

# ===== Summary =====
compute_savings = (normal_tokens - ps_tokens) / normal_tokens * 100
speedup_ratio = t_normal / t_ps

if local_rank == 0:
    print(f"\n{'='*70}")
    print(f"PREFIX-SHARING THROUGHPUT BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: Qwen3.6-27B (TP=4, bf16, RmPad, pretrained)")
    print(f"Config: N={N_SEQUENCES}, prefix={PREFIX_LEN}, suffix={SUFFIX_LEN}")
    print(f"{'='*70}")
    print(f"Normal forward:")
    print(f"  Tokens per batch: {normal_tokens}")
    print(f"  Time: {t_normal:.3f}s/batch ({N_MEASURE} iterations avg)")
    print(f"  Throughput: {normal_throughput:.0f} tok/s")
    print(f"  Peak memory: {mem_normal:.2f} GB")
    print(f"{'='*70}")
    print(f"PS two-pass forward:")
    print(f"  Tokens per batch: {ps_tokens} (1×prefix + {N_SEQUENCES}×suffix)")
    print(f"  Time: {t_ps:.3f}s/batch ({N_MEASURE} iterations avg)")
    print(f"  Throughput: {ps_throughput:.0f} tok/s (effective)")
    print(f"  Peak memory: {mem_ps:.2f} GB")
    print(f"{'='*70}")
    print(f"Compute savings: {compute_savings:.1f}%")
    print(f"Speedup ratio: {speedup_ratio:.2f}x")
    print(f"Overall cos_sim: {overall_cos:.6f}")
    print(f"{'='*70}")
    if overall_cos >= 0.999:
        print("PASS: PS precision alignment + throughput benchmark!")
    else:
        print(f"FAIL: cos_sim {overall_cos:.6f} < 0.999")
    print(f"{'='*70}")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()