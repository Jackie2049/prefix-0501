#!/usr/bin/env python3
"""PS scaling with longer suffixes: realistic GRPO suffix_len=128,256,512.

Usage: torchrun --nproc_per_node=4 scripts/benchmark_ps_suffix_scaling.py
"""
import os, sys, time
verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path); sys.path.insert(0, prefix_sharing_path); sys.path.insert(0, prefix_path)

import torch
from transformers import AutoConfig
from safetensors.torch import load_file

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4; PREFIX_LEN = 64; SEED = 42; N_WARMUP = 2; N_MEASURE = 5

torch.distributed.init_process_group(backend="nccl", init_method="env://")
local_rank = int(os.environ.get("LOCAL_RANK", 0)); torch.cuda.set_device(local_rank)

from megatron.core.tensor_parallel.random import (
    initialize_rng_tracker, get_cuda_rng_tracker, _MODEL_PARALLEL_RNG_TRACKER_NAME,
)
initialize_rng_tracker(use_te_rng_tracker=False, inference_rng_tracker=False)
rng_tracker = get_cuda_rng_tracker(); rng_tracker.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, seed=SEED)

from megatron.core import parallel_state
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)
tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

config = AutoConfig.from_pretrained(HF_MODEL_PATH)

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLMRmPad(config, megatron_config).to(device)

# Load pretrained weights (compact inline)
hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path); hf_state_dict.update(shard_dict)

hf_keys_filtered = {k: v for k, v in hf_state_dict.items()
                    if k.startswith("model.language_model.") or k.startswith("lm_head.")}

layer_types = config.layer_types; num_layers = config.num_hidden_layers

def shard_tensor(t, dim, tp_size, tp_rank):
    return torch.chunk(t, tp_size, dim=dim)[tp_rank].contiguous()

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

loaded_count = 0
for layer_idx in range(num_layers):
    is_deltanet = (layer_types[layer_idx] == "linear_attention")
    decoder_layer = model.model.layers[layer_idx]
    attn_module = decoder_layer.self_attn; mlp_module = decoder_layer.mlp
    hf_layer_prefix = f"model.language_model.layers.{layer_idx}"
    gate_key = f"{hf_layer_prefix}.mlp.gate_proj.weight"
    up_key = f"{hf_layer_prefix}.mlp.up_proj.weight"
    if gate_key in hf_keys_filtered and up_key in hf_keys_filtered:
        gate_up_shard = torch.cat([
            shard_tensor(hf_keys_filtered[gate_key], 0, TP_SIZE, tp_rank),
            shard_tensor(hf_keys_filtered[up_key], 0, TP_SIZE, tp_rank),
        ], dim=0).contiguous()
        mlp_module.gate_up_proj.weight.data.copy_(gate_up_shard.to(torch.bfloat16))
        loaded_count += 2
    down_key = f"{hf_layer_prefix}.mlp.down_proj.weight"
    if down_key in hf_keys_filtered:
        mlp_module.down_proj.weight.data.copy_(
            shard_tensor(hf_keys_filtered[down_key], 1, TP_SIZE, tp_rank).to(torch.bfloat16))
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
                getattr(attn_module, proj_name).weight.data.copy_(
                    shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
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
                getattr(attn_module, proj_name).weight.data.copy_(
                    shard_tensor(hf_keys_filtered[key], 0, TP_SIZE, tp_rank).to(torch.bfloat16))
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

hf_state_dict.clear(); hf_keys_filtered.clear(); torch.cuda.empty_cache()
if local_rank == 0:
    print(f"Loaded {loaded_count} weights, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Install PS patches
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration
from prefix_sharing.core.config import PrefixSharingConfig

ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=PREFIX_LEN, min_group_size=2)
integration = VerlQwen3_6Integration(config=ps_config)
patch_handle = integration.install(model_config=config)

from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.backends.packed_layout import PackedBatchLayout

vocab_size = config.vocab_size

if local_rank == 0:
    print(f"\n{'='*80}")
    print(f"PS SUFFIX SCALING: 16-layer ~7B, P={PREFIX_LEN}, N=8")
    print(f"Testing suffix_len=32,64,128,256 to find realistic GRPO performance")
    print(f"{'='*80}")
    print(f"{'S':>4} {'Normal':>8} {'PS':>8} {'Ratio':>7} {'Save%':>7} {'Pref_t':>7} {'Suf_t':>7}")
    print(f"{'-'*4} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

N = 8
results = []
for SUFFIX_LEN in [32, 64, 128, 256]:
    torch.manual_seed(SEED + 500 + SUFFIX_LEN)
    total_len = PREFIX_LEN + SUFFIX_LEN

    prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
    suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                          for _ in range(N)]
    full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])
    suffix_input_ids = torch.stack(suffix_tokens_list)
    prefix_input_ids = prefix_tokens.unsqueeze(0)

    # Normal forward timing
    full_attention_mask = torch.ones(N, total_len, dtype=torch.bool, device=device)
    full_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N, -1)

    # Warmup
    for _ in range(N_WARMUP):
        with torch.no_grad():
            model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                  position_ids=full_position_ids)

    torch.cuda.synchronize()
    t_normal_total = 0
    for _ in range(N_MEASURE):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=full_input_ids, attention_mask=full_attention_mask,
                        position_ids=full_position_ids)
        torch.cuda.synchronize(); t_normal_total += time.time() - t0
        del out
    t_normal = t_normal_total / N_MEASURE

    # PS two-pass timing with breakdown
    sequences_list = [seq.tolist() for seq in full_input_ids]
    ps_plan = PrefixSharingPlanner(ps_config).plan(sequences_list)

    prefix_attention_mask = torch.ones(1, PREFIX_LEN, dtype=torch.bool, device=device)
    prefix_position_ids = torch.arange(PREFIX_LEN, dtype=torch.long, device=device).unsqueeze(0)
    suffix_attention_mask = torch.ones(N, SUFFIX_LEN, dtype=torch.bool, device=device)
    suffix_position_ids = torch.arange(PREFIX_LEN, total_len, dtype=torch.long, device=device).unsqueeze(0).expand(N, -1)

    suffix_lengths = [SUFFIX_LEN] * N
    suffix_packed_layout = PackedBatchLayout.from_valid_lengths(suffix_lengths)

    ps_runtime_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=ps_plan, backend=None,
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
                out1 = model(input_ids=ps_runtime_state.prefix_input_ids,
                            attention_mask=ps_runtime_state.prefix_attention_mask,
                            position_ids=ps_runtime_state.prefix_position_ids)
            del out1
            with torch.no_grad():
                out2 = model(input_ids=suffix_input_ids,
                            attention_mask=suffix_attention_mask,
                            position_ids=suffix_position_ids)
            del out2

    torch.cuda.synchronize()
    t_prefix_total = 0; t_suffix_total = 0; t_ps_total = 0
    for _ in range(N_MEASURE):
        torch.cuda.synchronize(); t0 = time.time()
        with prefix_sharing_runtime_context(ps_runtime_state) as ctx_state:
            torch.cuda.synchronize(); t1 = time.time()
            with torch.no_grad():
                prefix_output = model(input_ids=ps_runtime_state.prefix_input_ids,
                                     attention_mask=ps_runtime_state.prefix_attention_mask,
                                     position_ids=ps_runtime_state.prefix_position_ids)
            del prefix_output
            torch.cuda.synchronize(); t_prefix = time.time()
            with torch.no_grad():
                suffix_output = model(input_ids=suffix_input_ids,
                                     attention_mask=suffix_attention_mask,
                                     position_ids=suffix_position_ids)
            del suffix_output
            torch.cuda.synchronize(); t_end = time.time()
        t_prefix_total += t_prefix - t1
        t_suffix_total += t_end - t_prefix
        t_ps_total += t_end - t0

    t_prefix = t_prefix_total / N_MEASURE
    t_suffix = t_suffix_total / N_MEASURE
    t_ps = t_ps_total / N_MEASURE

    savings_pct = (N * total_len - PREFIX_LEN - N * SUFFIX_LEN) / (N * total_len) * 100

    if local_rank == 0:
        print(f"{SUFFIX_LEN:>4} {t_normal:>8.3f}s {t_ps:>8.3f}s {t_normal/t_ps:>7.2f}x {savings_pct:>7.1f}% {t_prefix:>7.3f}s {t_suffix:>7.3f}s")

    results.append((SUFFIX_LEN, t_normal, t_ps, t_normal/t_ps, savings_pct))

    del full_input_ids, suffix_input_ids, prefix_input_ids
    torch.cuda.empty_cache()

patch_handle.disable()

if local_rank == 0:
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for s, tn, tp, ratio, sav in results:
        faster = "PS faster" if ratio > 1.0 else "Normal faster"
        print(f"  S={s}: {faster} ({ratio:.2f}x), savings={sav:.1f}%")

parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()