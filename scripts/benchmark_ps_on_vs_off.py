#!/usr/bin/env python3
"""Benchmark PS ON vs PS OFF: measure time, memory, and precision.

Uses verl's ParallelQwen3_6ForCausalLMRmPad model with PS patches installed.
- PS OFF: N full sequences (prefix+suffix) through normal forward (no PS context)
- PS ON: prefix pass + suffix pass with PS context (build_prefix_sharing_micro_batch)

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
from contextlib import nullcontext

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4
PREFIX_LEN = 64
SUFFIX_LEN = 128
N_SEQUENCES = 4
SEED = 42
N_STEPS = 3

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

from megatron.core import ModelParallelConfig
from verl.models.qwen3_6.megatron.modeling_qwen3_6_megatron import ParallelQwen3_6ForCausalLMRmPad

megatron_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    sequence_parallel=False, bf16=True, params_dtype=torch.bfloat16,
)
model = ParallelQwen3_6ForCausalLMRmPad(config, megatron_config).to(device)

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

if local_rank == 0:
    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"\n{'='*70}")
    print(f"PS ON vs PS OFF BENCHMARK")
    print(f"Model: 16-layer ~7B, TP={TP_SIZE}, P={PREFIX_LEN}, S={SUFFIX_LEN}, N={N_SEQUENCES}")
    print(f"PS OFF: N full sequences ({N_SEQUENCES*total_len} tokens)")
    print(f"PS ON:  P prefix + N suffix ({PREFIX_LEN+N_SEQUENCES*SUFFIX_LEN} tokens)")
    print(f"Token savings: {(1 - (PREFIX_LEN+N_SEQUENCES*SUFFIX_LEN)/(N_SEQUENCES*total_len))*100:.1f}%")
    print(f"Model memory: {mem:.2f}GB")
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
    suffix_logits_off = logits_off[:, PREFIX_LEN:-1, :]  # (N, response_len, vocab)
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

    suffix_logits_on = logits_on[:, :-1, :]  # (N, response_len, vocab)
    log_probs_on = F.log_softmax(suffix_logits_on, dim=-1)
    selected_lp_on = log_probs_on.gather(-1, suffix_labels.unsqueeze(-1)).squeeze(-1)

if local_rank == 0:
    cos_sim = F.cosine_similarity(selected_lp_off.flatten().unsqueeze(0),
                                   selected_lp_on.flatten().unsqueeze(0)).item()
    max_diff = (selected_lp_off - selected_lp_on).abs().max().item()
    print(f"  Precision: cos_sim={cos_sim:.6f}, max_diff={max_diff:.6f}")

del output_off, logits_off, suffix_logits_off, log_probs_off, selected_lp_off
del output_on, logits_on, suffix_logits_on, log_probs_on, selected_lp_on
torch.cuda.empty_cache()

# === Phase 1: PS OFF training performance ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
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
    optimizer.step(); optimizer.zero_grad()

    torch.cuda.synchronize(); t_end = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    ps_off_results.append({"step": step, "time": t_end - t_start, "peak_mem": peak_mem})
    if local_rank == 0:
        print(f"  Step {step}: time={t_end-t_start:.3f}s, peak_mem={peak_mem:.2f}GB")

# === Phase 2: PS ON training performance ===
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
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

    suffix_logits = logits[:, :-1, :]
    log_probs = F.log_softmax(suffix_logits, dim=-1)
    selected_lp = log_probs.gather(-1, suffix_labels.unsqueeze(-1)).squeeze(-1)

    loss = selected_lp.mean()
    loss.backward()
    optimizer.step(); optimizer.zero_grad()

    torch.cuda.synchronize(); t_end = time.time()
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    ps_on_results.append({"step": step, "time": t_end - t_start, "peak_mem": peak_mem})
    if local_rank == 0:
        print(f"  Step {step}: time={t_end-t_start:.3f}s, peak_mem={peak_mem:.2f}GB")

# === Results ===
if local_rank == 0:
    avg_time_off = sum(r["time"] for r in ps_off_results[1:]) / max(N_STEPS - 1, 1)
    avg_time_on = sum(r["time"] for r in ps_on_results[1:]) / max(N_STEPS - 1, 1)
    avg_mem_off = sum(r["peak_mem"] for r in ps_off_results) / N_STEPS
    avg_mem_on = sum(r["peak_mem"] for r in ps_on_results) / N_STEPS

    speedup = avg_time_off / avg_time_on if avg_time_on > 0 else 0
    mem_savings = (avg_mem_off - avg_mem_on) / avg_mem_off * 100

    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"  PS OFF avg time (steps 1+): {avg_time_off:.3f}s")
    print(f"  PS ON  avg time (steps 1+): {avg_time_on:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  PS OFF avg peak mem: {avg_mem_off:.2f}GB")
    print(f"  PS ON  avg peak mem: {avg_mem_on:.2f}GB")
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