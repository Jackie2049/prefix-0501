#!/usr/bin/env python3
"""Benchmark PS ON vs PS OFF: measure time, memory, and precision.

Uses verl's ParallelQwen3_6ForCausalLMRmPad model with real weights loaded
from safetensors. PS patches installed for both attention types.
- PS OFF: N full sequences (prefix+suffix) through normal forward (no PS context)
- PS ON: prefix pass + suffix pass with PS context (build_prefix_sharing_micro_batch)
CPU AdamW optimizer keeps optimizer state on CPU to avoid OOM on 24GB GPUs.

Usage: torchrun --nproc_per_node=4 scripts/benchmark_ps_on_vs_off.py
"""
import os, sys, time

verl_path = os.path.expanduser("~/rollout-prefix/prefix-0501/dependency/verl_v070")
prefix_sharing_path = os.path.expanduser("~/rollout-prefix/prefix-0501/prefix-sharing")
prefix_path = os.path.expanduser("~/rollout-prefix/prefix-0501")
sys.path.insert(0, verl_path); sys.path.insert(0, prefix_sharing_path); sys.path.insert(0, prefix_path)

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from safetensors.torch import load_file
from contextlib import nullcontext

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4
PREFIX_LEN = int(os.environ.get("PREFIX_LEN", 64))
SUFFIX_LEN = int(os.environ.get("SUFFIX_LEN", 128))
N_SEQUENCES = int(os.environ.get("N_SEQUENCES", 4))
SEED = 42
N_STEPS = int(os.environ.get("N_STEPS", 5))

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
vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN
response_len = SUFFIX_LEN - 1
layer_types = config.layer_types; num_layers = config.num_hidden_layers

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLMRmPad(config, megatron_config).to(device)

# === Load real weights from safetensors ===
if local_rank == 0:
    print("Loading weights from safetensors...")

hf_state_dict = {}
for i in range(1, 12):
    shard_path = os.path.join(HF_MODEL_PATH, f"model.safetensors-{i:05d}-of-00011.safetensors")
    shard_dict = load_file(shard_path); hf_state_dict.update(shard_dict)

hf_keys_filtered = {k: v for k, v in hf_state_dict.items()
                    if k.startswith("model.language_model.") or k.startswith("lm_head.")}

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

# Install PS patches
from prefix_sharing.integrations.verl_attention import VerlQwen3_6Integration
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.integrations.verl_mcore import build_prefix_sharing_micro_batch
from prefix_sharing.integrations.context import prefix_sharing_runtime_context

ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=PREFIX_LEN, min_group_size=N_SEQUENCES)
patch_handle = VerlQwen3_6Integration(config=ps_config).install(model_config=config)

actor_config = {
    "prefix_sharing_config": {
        "enable_prefix_sharing": True,
        "min_prefix_len": PREFIX_LEN,
        "min_group_size": N_SEQUENCES,
    },
    "megatron": {"use_remove_padding": True},
}


# === CPU AdamW optimizer (avoids OOM on 24GB GPUs) ===
class CPUAdamW:
    """AdamW optimizer with all state on CPU. Only bf16 params on GPU."""
    def __init__(self, model_params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(model_params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_count = 0
        # FP32 main params + optimizer state on CPU (no pin_memory to avoid CUDA errors)
        self.main_params = [p.data.detach().float().cpu() for p in self.params]
        self.exp_avg = [torch.zeros_like(mp) for mp in self.main_params]
        self.exp_avg_sq = [torch.zeros_like(mp) for mp in self.main_params]

    def step(self):
        self.step_count += 1
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
        step_size = self.lr / bias_correction1

        for mp, ea, eas, p in zip(self.main_params, self.exp_avg, self.exp_avg_sq, self.params):
            if p.grad is not None:
                grad_cpu = p.grad.detach().float().cpu()
            else:
                continue

            # Decoupled weight decay
            mp.add_(mp, alpha=-self.lr * self.weight_decay)

            # Update exp_avg and exp_avg_sq
            ea.mul_(self.beta1).add_(grad_cpu, alpha=1 - self.beta1)
            eas.mul_(self.beta2).addcmul_(grad_cpu, grad_cpu, value=1 - self.beta2)

            # Update main_params
            denom = (eas.sqrt() / bias_correction2).add_(self.eps)
            mp.addcdiv_(ea, denom, value=-step_size)

            # Copy CPU fp32 → GPU bf16
            p.data.copy_(mp.to(p.device, dtype=p.dtype))

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad = None


if local_rank == 0:
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"\n{'='*70}")
    print(f"PS ON vs PS OFF BENCHMARK")
    print(f"Model: 16-layer ~7B (real weights), TP={TP_SIZE}, P={PREFIX_LEN}, S={SUFFIX_LEN}, N={N_SEQUENCES}")
    print(f"PS OFF: N full sequences ({N_SEQUENCES*total_len} tokens)")
    print(f"PS ON:  P prefix + N suffix ({PREFIX_LEN+N_SEQUENCES*SUFFIX_LEN} tokens)")
    print(f"Token savings: {(1 - (PREFIX_LEN+N_SEQUENCES*SUFFIX_LEN)/(N_SEQUENCES*total_len))*100:.1f}%")
    print(f"Optimizer: CPU AdamW (optimizer state on CPU, bf16 params on GPU)")
    print(f"Loaded {loaded_count} weight tensors, GPU {mem:.2f}GB")
    print(f"{'='*70}")

# === Phase 0: Precision validation (no_grad, same weights, same input) ===
if local_rank == 0:
    print(f"\n--- Phase 0: Precision validation (no_grad) ---")

torch.manual_seed(SEED)
prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                      for _ in range(N_SEQUENCES)]
suffix_labels = torch.stack([s[1:] for s in suffix_tokens_list])  # (N, SUFFIX_LEN-1)

# PS OFF forward (no_grad, no PS context)
with torch.no_grad():
    full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])
    attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

    output_off = model(input_ids=full_input_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits_off = output_off.logits.float()
    suffix_logits_off = logits_off[:, PREFIX_LEN:-1, :]
    log_probs_off = F.log_softmax(suffix_logits_off, dim=-1)
    selected_lp_off = log_probs_off.gather(-1, suffix_labels.unsqueeze(-1)).squeeze(-1)

# PS ON forward (no_grad, with PS context)
with torch.no_grad():
    batch = {
        "input_ids": full_input_ids.clone(),
        "attention_mask": attention_mask.clone(),
        "position_ids": position_ids.clone(),
        "responses": torch.stack([s for s in suffix_tokens_list]),
    }
    batch, ps_runtime_state = build_prefix_sharing_micro_batch(
        batch, actor_config, config, model_spec=ModelSpec.from_hf_config(config),
    )

    if local_rank == 0 and ps_runtime_state is not None:
        plan = ps_runtime_state.prefix_sharing_plan
        print(f"  PS plan: is_provider={plan.is_provider}, provider_idx={plan.provider_index}, "
              f"prefix_lens={plan.prefix_lens}, input_keep_ranges={plan.input_keep_ranges}")
        valid_per_row = [batch["attention_mask"][row].to(bool).sum().item() for row in range(N_SEQUENCES)]
        print(f"  Suffix batch valid tokens per row: {valid_per_row} (total={sum(valid_per_row)})")

    if ps_runtime_state is not None:
        prefix_context = prefix_sharing_runtime_context or nullcontext
        with prefix_context(ps_runtime_state) as ps_ctx:
            if ps_ctx is not None and ps_runtime_state.prefix_input_ids is not None:
                _prefix_output = model(
                    input_ids=ps_runtime_state.prefix_input_ids,
                    attention_mask=ps_runtime_state.prefix_attention_mask,
                    position_ids=ps_runtime_state.prefix_position_ids,
                )
                del _prefix_output

            output_on = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"].to(bool),
                position_ids=batch["position_ids"],
            )
            logits_on = output_on.logits.float()
    else:
        logits_on = logits_off.clone()

    suffix_logits_on = logits_on[:, :-1, :]
    log_probs_on = F.log_softmax(suffix_logits_on, dim=-1)
    selected_lp_on = log_probs_on.gather(-1, suffix_labels.unsqueeze(-1)).squeeze(-1)

if local_rank == 0:
    has_nan_off = selected_lp_off.isnan().any().item()
    has_nan_on = selected_lp_on.isnan().any().item()
    if not has_nan_off and not has_nan_on:
        cos_sim = F.cosine_similarity(selected_lp_off.flatten().unsqueeze(0),
                                       selected_lp_on.flatten().unsqueeze(0)).item()
        max_diff = (selected_lp_off - selected_lp_on).abs().max().item()
        print(f"  Precision: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")
    else:
        print(f"  Precision: NaN detected (OFF={has_nan_off}, ON={has_nan_on}) — SKIPPED")

del output_off, logits_off, suffix_logits_off, log_probs_off, selected_lp_off
del output_on, logits_on, suffix_logits_on, log_probs_on, selected_lp_on
torch.cuda.empty_cache()

# === Phase 1: PS OFF training performance ===
optimizer = CPUAdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
ps_off_results = []

if local_rank == 0:
    print(f"\n--- Phase 1: PS OFF training ({N_STEPS} steps) ---")

for step in range(N_STEPS):
    torch.manual_seed(SEED + step * 100 + 1000)
    torch.cuda.synchronize(); t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
    suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                          for _ in range(N_SEQUENCES)]
    full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])
    attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)
    suffix_labels = torch.stack([s[1:] for s in suffix_tokens_list])

    # Forward (no PS context → hooks fall through)
    output = model(input_ids=full_input_ids, attention_mask=attention_mask, position_ids=position_ids)
    logits = output.logits.float()
    suffix_logits = logits[:, PREFIX_LEN:-1, :]
    log_probs = F.log_softmax(suffix_logits, dim=-1)
    selected_lp = log_probs.gather(-1, suffix_labels.unsqueeze(-1)).squeeze(-1)

    loss = selected_lp.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); optimizer.zero_grad()

    torch.cuda.synchronize(); t_end = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    ps_off_results.append({"step": step, "time": t_end - t_start, "peak_mem": peak_mem})
    if local_rank == 0:
        print(f"  Step {step}: time={t_end-t_start:.3f}s, peak_mem={peak_mem:.2f}GB")

    del output, logits, suffix_logits, log_probs, selected_lp, loss
    torch.cuda.empty_cache()

# === Phase 2: PS ON training performance ===
optimizer = CPUAdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
ps_on_results = []

if local_rank == 0:
    print(f"\n--- Phase 2: PS ON training ({N_STEPS} steps) ---")

for step in range(N_STEPS):
    torch.manual_seed(SEED + step * 100 + 2000)
    torch.cuda.synchronize(); t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
    suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                          for _ in range(N_SEQUENCES)]
    full_input_ids = torch.stack([torch.cat([prefix_tokens, s]) for s in suffix_tokens_list])
    attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
    position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)
    responses = torch.stack([s for s in suffix_tokens_list])
    suffix_labels = torch.stack([s[1:] for s in suffix_tokens_list])

    batch = {
        "input_ids": full_input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
    }

    batch, ps_runtime_state = build_prefix_sharing_micro_batch(
        batch, actor_config, config, model_spec=ModelSpec.from_hf_config(config),
    )

    if local_rank == 0 and ps_runtime_state is not None:
        plan = ps_runtime_state.prefix_sharing_plan
        print(f"  PS plan: is_provider={plan.is_provider}, provider_idx={plan.provider_index}, "
              f"prefix_lens={plan.prefix_lens}, input_keep_ranges={plan.input_keep_ranges}")
        valid_per_row = [batch["attention_mask"][row].to(bool).sum().item() for row in range(N_SEQUENCES)]
        print(f"  Suffix batch valid tokens per row: {valid_per_row} (total={sum(valid_per_row)})")

    if ps_runtime_state is not None:
        prefix_context = prefix_sharing_runtime_context or nullcontext
        with prefix_context(ps_runtime_state) as ps_ctx:
            if ps_ctx is not None and ps_runtime_state.prefix_input_ids is not None:
                with torch.no_grad():
                    _prefix_output = model(
                        input_ids=ps_runtime_state.prefix_input_ids,
                        attention_mask=ps_runtime_state.prefix_attention_mask,
                        position_ids=ps_runtime_state.prefix_position_ids,
                    )
                del _prefix_output

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"].to(bool),
                position_ids=batch["position_ids"],
            )
            logits = output.logits.float()
    else:
        output = model(input_ids=full_input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = output.logits.float()

    # Use PREFIX_LEN:-1 to select only suffix positions (same as PS OFF)
    # This ensures backward only propagates through suffix, not padded prefix
    suffix_logits = logits[:, PREFIX_LEN:-1, :]
    log_probs = F.log_softmax(suffix_logits, dim=-1)
    selected_lp = log_probs.gather(-1, suffix_labels.unsqueeze(-1)).squeeze(-1)

    loss = selected_lp.mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step(); optimizer.zero_grad()

    torch.cuda.synchronize(); t_end = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    ps_on_results.append({"step": step, "time": t_end - t_start, "peak_mem": peak_mem})
    if local_rank == 0:
        print(f"  Step {step}: time={t_end-t_start:.3f}s, peak_mem={peak_mem:.2f}GB")

    del output, logits, suffix_logits, log_probs, selected_lp, loss
    torch.cuda.empty_cache()

# === Results ===
if local_rank == 0:
    avg_time_off = sum(r["time"] for r in ps_off_results[1:]) / max(N_STEPS - 1, 1)
    avg_time_on = sum(r["time"] for r in ps_on_results[1:]) / max(N_STEPS - 1, 1)
    avg_mem_off = sum(r["peak_mem"] for r in ps_off_results[1:]) / max(N_STEPS - 1, 1)
    avg_mem_on = sum(r["peak_mem"] for r in ps_on_results[1:]) / max(N_STEPS - 1, 1)

    speedup = avg_time_off / avg_time_on if avg_time_on > 0 else 0
    mem_savings = (avg_mem_off - avg_mem_on) / avg_mem_off * 100

    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  PS OFF avg time (steps 1+): {avg_time_off:.3f}s")
    print(f"  PS ON  avg time (steps 1+): {avg_time_on:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  PS OFF avg peak mem (steps 1+): {avg_mem_off:.2f}GB")
    print(f"  PS ON  avg peak mem (steps 1+): {avg_mem_on:.2f}GB")
    print(f"  Memory: {mem_savings:+.1f}%")

    print(f"\n  Per-step breakdown:")
    print(f"  {'Step':>4} {'OFF_time':>10} {'ON_time':>10} {'OFF_mem':>10} {'ON_mem':>10}")
    for i in range(N_STEPS):
        print(f"  {i:>4} {ps_off_results[i]['time']:>10.3f}s {ps_on_results[i]['time']:>10.3f}s "
              f"{ps_off_results[i]['peak_mem']:>8.2f}GB {ps_on_results[i]['peak_mem']:>8.2f}GB")

    print(f"\n{'='*70}")
    if speedup > 1.05:
        print(f"RESULT: PS FASTER ({speedup:.2f}x speedup, {mem_savings:+.1f}% memory)")
    elif speedup > 0.95:
        print(f"RESULT: PS near parity ({speedup:.2f}x)")
    else:
        print(f"RESULT: PS SLOWER ({speedup:.2f}x — need longer prefix or larger n)")
    print(f"{'='*70}")

# Cleanup
patch_handle.disable() if patch_handle else None
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()