#!/usr/bin/env python3
"""v17: BSHD-mode Prefix-Sharing E2E test for Qwen3-27B (16-layer).

Uses the ACTUAL model pipeline: Qwen3_6HybridModel → GPTModel with
SelfAttention for full-attention layers and GatedDeltaNetAttention for
linear-attention layers.

BSHD mode = use_remove_padding=False (packed_seq_params=None).
Q/K/V are 4D (sq, b, h, hn) format.

PS flow:
- Prefix pass: SelfAttention/GatedDeltaNetAttention forward → store KV/carry
- Suffix pass: SelfAttention loads stored KV, expands + concatenates → flash_attn_varlen_func
                GatedDeltaNet loads stored carry state → injects as initial state

Usage: torchrun --nproc_per_node=4 scripts/test_bshd_ps_e2e.py
"""
import os
import sys
import time

WORK_DIR = os.path.expanduser("~/rollout-prefix/prefix-0501")
VERL_DIR = os.path.join(WORK_DIR, "dependency/verl_v070")
PS_DIR = os.path.join(WORK_DIR, "prefix-sharing")
MEGATRON_DIR = os.path.join(WORK_DIR, "dependency/Megatron-LM-core_v0.12.1")

sys.path.insert(0, VERL_DIR)
sys.path.insert(0, PS_DIR)
sys.path.insert(0, MEGATRON_DIR)

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

import torch
import torch.nn.functional as F
from transformers import AutoConfig

HF_MODEL_PATH = os.path.expanduser("~/rollout-prefix/models/Qwen3-27B-text-only-16layers")
TP_SIZE = 4
PREFIX_LEN = 64
SUFFIX_LEN = 32
N_SEQUENCES = 8
SEED = 42

# Initialize distributed
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
    tensor_model_parallel_size=TP_SIZE, pipeline_model_parallel_size=1,
    context_parallel_size=1, expert_model_parallel_size=1,
)
tp_rank = parallel_state.get_tensor_model_parallel_rank()
device = torch.device(f"cuda:{local_rank}")

# Load HF config
config = AutoConfig.from_pretrained(HF_MODEL_PATH)
if local_rank == 0:
    print(f"Model config: num_hidden_layers={config.num_hidden_layers}, "
          f"full_attention_interval={config.full_attention_interval}, "
          f"partial_rotary_factor={config.partial_rotary_factor}, "
          f"attn_output_gate={config.attn_output_gate}")

# Initialize model via Qwen3_6HybridModel → GPTModel (same as verl training)
from megatron.core import ModelParallelConfig
from verl.models.mcore.registry import init_mcore_model, hf_to_mcore_config

mcore_config = ModelParallelConfig(
    tensor_model_parallel_size=TP_SIZE,
    pipeline_model_parallel_size=1,
    sequence_parallel=False,  # BSHD mode requires no SP
    bf16=True,
    params_dtype=torch.bfloat16,
)
tfconfig = hf_to_mcore_config(config, torch.bfloat16)
model = init_mcore_model(tfconfig, config, pre_process=True, post_process=True)
model = model.to(device)

# Verify model architecture
num_layers = len(model.decoder.layers)
full_attn_count = sum(1 for i in range(num_layers) if i % config.full_attention_interval == 0)
deltanet_count = num_layers - full_attn_count
if local_rank == 0:
    attn_types = []
    for i in range(num_layers):
        attn = model.decoder.layers[i].self_attention
        type_name = attn.__class__.__name__
        attn_types.append(f"L{i}:{type_name}")
    print(f"Model layers: {attn_types}")
    print(f"Full attention: {full_attn_count}, DeltaNet: {deltanet_count}")

mem = torch.cuda.memory_allocated() / 1024**3
if local_rank == 0:
    print(f"Model loaded, GPU {mem:.2f} GiB")

# ===== Step 1: Normal forward (reference) =====
vocab_size = config.vocab_size
total_len = PREFIX_LEN + SUFFIX_LEN

torch.manual_seed(SEED + 500)
prefix_tokens = torch.randint(0, vocab_size, (PREFIX_LEN,), dtype=torch.long, device=device)
suffix_tokens_list = [torch.randint(0, vocab_size, (SUFFIX_LEN,), dtype=torch.long, device=device)
                      for _ in range(N_SEQUENCES)]
full_sequences = [torch.cat([prefix_tokens, suffix]) for suffix in suffix_tokens_list]
full_input_ids = torch.stack(full_sequences)  # (N, total_len)

full_attention_mask = torch.ones(N_SEQUENCES, total_len, dtype=torch.bool, device=device)
full_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

# BSHD-mode forward: preprocess_bshd → model(input_ids, position_ids, attention_mask)
from verl.models.mcore.util import preprocess_bshd

processed_ids, processed_pos, processed_mask = preprocess_bshd(
    full_input_ids, full_position_ids, full_attention_mask, config
)

torch.cuda.synchronize()
t_normal_start = time.time()

with torch.no_grad():
    normal_output = model(
        input_ids=processed_ids,
        position_ids=processed_pos,
        attention_mask=processed_mask,
    )

torch.cuda.synchronize()
t_normal = time.time() - t_normal_start

# Get logits - model returns (hidden_states, _) from GPTModel
if isinstance(normal_output, tuple):
    hidden_states = normal_output[0]
else:
    hidden_states = normal_output

# Apply lm_head manually
logits_normal = model.lm_head(hidden_states)

# Extract suffix log_probs
log_probs_normal = F.log_softmax(logits_normal.float(), dim=-1)
suffix_labels = full_input_ids[:, PREFIX_LEN:]  # (N, SUFFIX_LEN)
# Select log_prob for each token in suffix
suffix_log_probs_normal = log_probs_normal[:, PREFIX_LEN-1:-1, :]  # shifted by 1 for next-token prediction
selected_normal = suffix_log_probs_normal.gather(
    dim=-1, index=suffix_labels.unsqueeze(-1)
).squeeze(-1).to(device='cpu', dtype=torch.float32)

del normal_output, hidden_states, logits_normal, log_probs_normal, suffix_log_probs_normal
torch.cuda.empty_cache()

if local_rank == 0:
    print(f"Step 1 (normal forward): {t_normal:.3f}s, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

# ===== Step 2: Two-pass PS forward =====
suffix_input_ids = torch.stack(suffix_tokens_list)  # (N, SUFFIX_LEN)
prefix_input_ids = prefix_tokens.unsqueeze(0)  # (1, PREFIX_LEN)

suffix_attention_mask = torch.ones(N_SEQUENCES, SUFFIX_LEN, dtype=torch.bool, device=device)
suffix_position_ids = torch.arange(PREFIX_LEN, total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)
prefix_attention_mask_p = torch.ones(1, PREFIX_LEN, dtype=torch.bool, device=device)
prefix_position_ids_p = torch.arange(PREFIX_LEN, device=device).unsqueeze(0)

# Preprocess both passes
prefix_ids_proc, prefix_pos_proc, prefix_mask_proc = preprocess_bshd(
    prefix_input_ids, prefix_position_ids_p, prefix_attention_mask_p, config
)
suffix_ids_proc, suffix_pos_proc, suffix_mask_proc = preprocess_bshd(
    suffix_input_ids, suffix_position_ids, suffix_attention_mask, config
)

# Build PS context
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.integrations.context import prefix_sharing_runtime_context, PrefixSharingRuntimeContext
from prefix_sharing.backends.packed_layout import PackedBatchLayout

ps_config = PrefixSharingConfig(
    enable_prefix_sharing=True,
    min_prefix_len=PREFIX_LEN,
    min_group_size=N_SEQUENCES,
)

sequences_list = [seq.tolist() for seq in full_input_ids]
ps_plan = PrefixSharingPlanner(ps_config).plan(sequences_list)

suffix_lengths = [SUFFIX_LEN] * N_SEQUENCES
suffix_layout = PackedBatchLayout.from_valid_lengths(suffix_lengths)

model_spec = ModelSpec.from_hf_config(config)

ps_runtime_state = PrefixSharingRuntimeContext(
    prefix_sharing_plan=ps_plan,
    packed_batch_layout=suffix_layout,
    store=None,  # will be created by context manager
    deltanet_store=None,
    backend=None,
    model_spec=model_spec,
)

torch.cuda.synchronize()
t_ps_start = time.time()

with prefix_sharing_runtime_context(ps_runtime_state) as ctx:
    # Prefix pass (no grad)
    with torch.no_grad():
        prefix_out = model(
            input_ids=prefix_ids_proc,
            position_ids=prefix_pos_proc,
            attention_mask=prefix_mask_proc,
        )
    del prefix_out
    torch.cuda.empty_cache()

    if local_rank == 0:
        print(f"  Prefix pass done, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
        # Check store contents
        from prefix_sharing.core.prefix_store import PREFIX_STATE_TYPE_ATTENTION_KV, PREFIX_STATE_TYPE_DELTANET_STATE
        kv_count = sum(1 for s in ctx.store._slots if s.prefix_state_type == PREFIX_STATE_TYPE_ATTENTION_KV)
        dn_count = sum(1 for s in ctx.deltanet_store._slots if s.prefix_state_type == PREFIX_STATE_TYPE_DELTANET_STATE)
        print(f"  Store: KV slots={kv_count}, DeltaNet slots={dn_count}")

    # Suffix pass (with gradients)
    suffix_out = model(
        input_ids=suffix_ids_proc,
        position_ids=suffix_pos_proc,
        attention_mask=suffix_mask_proc,
    )

torch.cuda.synchronize()
t_ps = time.time() - t_ps_start

# Get logits from suffix pass
if isinstance(suffix_out, tuple):
    suffix_hidden = suffix_out[0]
else:
    suffix_hidden = suffix_out

logits_ps = model.lm_head(suffix_hidden)

log_probs_ps = F.log_softmax(logits_ps.float(), dim=-1)
# suffix logits start at position 0 (suffix only), need next-token prediction
suffix_labels_ps = suffix_input_ids[:, 1:]  # (N, SUFFIX_LEN-1)
selected_ps = log_probs_ps[:, :-1, :].gather(
    dim=-1, index=suffix_labels_ps.unsqueeze(-1)
).squeeze(-1)

del suffix_out, suffix_hidden, logits_ps, log_probs_ps
torch.cuda.empty_cache()

if local_rank == 0:
    print(f"Step 2 (PS forward): {t_ps:.3f}s, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

# ===== Step 3: Precision check =====
# Compare suffix log_probs from normal vs PS forward
# Normal: selected_normal covers positions PREFIX_LEN-1 to total_len-2 (shifted next-token)
# PS: selected_ps covers positions 0 to SUFFIX_LEN-2 (suffix-only, shifted next-token)
# Both should predict the same suffix tokens

# Align the log_probs: normal covers PREFIX_LEN+1 positions of suffix
# PS covers SUFFIX_LEN-1 positions of suffix
# For comparison, take the suffix portion from both

old_lp = selected_normal.to(device=device)  # (N, SUFFIX_LEN) from normal
new_lp = selected_ps.to(device=device)      # (N, SUFFIX_LEN-1) from PS

# The normal forward predicts tokens at positions [PREFIX_LEN, total_len-1]
# The PS forward predicts tokens at positions [1, SUFFIX_LEN-1] (within suffix)
# We need to align: normal's suffix predictions start at index PREFIX_LEN-1 (predicting PREFIX_LEN token)
# and PS's predictions start at index 0 (predicting suffix token 1)
# They should match for the overlapping suffix positions

# Normal: selected_normal[i] = log_prob of suffix_tokens[i] at position PREFIX_LEN+i-1
# PS: selected_ps[i] = log_prob of suffix_tokens[i+1] at position i (in suffix)
# Wait, let me be more careful...

# In normal forward: logits at position p predict token at position p+1
# For suffix tokens at positions PREFIX_LEN to total_len-1:
#   log_prob of suffix_token[j] (at absolute position PREFIX_LEN+j) is from logits at position PREFIX_LEN+j-1
# selected_normal = log_probs_normal[:, PREFIX_LEN-1:-1, :].gather(dim=-1, index=suffix_labels.unsqueeze(-1)).squeeze(-1)
# This gives (N, SUFFIX_LEN) — log_probs for each suffix token

# In PS forward: suffix-only sequence, logits at position p (within suffix) predict token at position p+1
# selected_ps = log_probs_ps[:, :-1, :].gather(dim=-1, index=suffix_labels_ps.unsqueeze(-1)).squeeze(-1)
# suffix_labels_ps = suffix_input_ids[:, 1:]  — shifted next-token
# This gives (N, SUFFIX_LEN-1)

# The PS suffix-only positions are 0..SUFFIX_LEN-1, but in absolute coordinates they're PREFIX_LEN..PREFIX_LEN+SUFFIX_LEN-1
# The first suffix token (at position PREFIX_LEN) is predicted from PS logits at position 0
# But PS position 0 has absolute position PREFIX_LEN, so it should predict token at absolute position PREFIX_LEN+1
# This is suffix_token[1]

# So alignment:
# normal: selected_normal[:, j] = log_prob of suffix_tokens[:, j] (N, SUFFIX_LEN)
# PS:     selected_ps[:, j] = log_prob of suffix_tokens[:, j+1] (N, SUFFIX_LEN-1)

# We can compare selected_normal[:, 1:] with selected_ps (both predict suffix tokens 1..SUFFIX_LEN-1)
old_aligned = selected_normal[:, 1:].to(device=device, dtype=torch.float32)
new_aligned = selected_ps.to(device=device, dtype=torch.float32)

all_cos_sims = []
for i in range(N_SEQUENCES):
    cos_sim = F.cosine_similarity(
        new_aligned[i].flatten(),
        old_aligned[i].flatten(),
        dim=0,
    ).item()
    all_cos_sims.append(cos_sim)
overall_cos = sum(all_cos_sims) / len(all_cos_sims)

max_diff = (new_aligned - old_aligned).abs().max().item()

if local_rank == 0:
    print(f"\n{'='*60}")
    print(f"BSHD PS E2E Test Results (Qwen3-27B 16-layer)")
    print(f"{'='*60}")
    print(f"Precision: log_prob cos_sim = {overall_cos:.6f}")
    print(f"           max_diff = {max_diff:.6f}")
    print(f"           per-sequence: {all_cos_sims}")
    status = "PASS" if overall_cos >= 0.99 else "FAIL"
    print(f"Status: {status}")
    print(f"Normal forward time: {t_normal:.3f}s")
    print(f"PS forward time: {t_ps:.3f}s")
    speedup = t_normal / t_ps if t_ps > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    print(f"{'='*60}")

# ===== Step 4: Backward pass (gradient flow check) =====
# Re-run PS forward with gradients for backward
suffix_ids_proc2, suffix_pos_proc2, suffix_mask_proc2 = preprocess_bshd(
    suffix_input_ids, suffix_position_ids, suffix_attention_mask, config
)

ps_runtime_state2 = PrefixSharingRuntimeContext(
    prefix_sharing_plan=ps_plan,
    packed_batch_layout=suffix_layout,
    store=None,
    deltanet_store=None,
    backend=None,
    model_spec=model_spec,
)

with prefix_sharing_runtime_context(ps_runtime_state2) as ctx2:
    # Prefix pass
    with torch.no_grad():
        prefix_out2 = model(
            input_ids=prefix_ids_proc,
            position_ids=prefix_pos_proc,
            attention_mask=prefix_mask_proc,
        )
    del prefix_out2
    torch.cuda.empty_cache()

    # Suffix pass with gradients
    suffix_out2 = model(
        input_ids=suffix_ids_proc2,
        position_ids=suffix_pos_proc2,
        attention_mask=suffix_mask_proc2,
    )

if isinstance(suffix_out2, tuple):
    suffix_hidden2 = suffix_out2[0]
else:
    suffix_hidden2 = suffix_out2

logits2 = model.lm_head(suffix_hidden2)
loss = logits2.float().mean()

model.zero_grad()
try:
    loss.backward()
    total_grad_norm = 0.0
    params_with_grad = 0
    layers_with_grad = set()
    for name, param in model.named_parameters():
        if param.grad is not None:
            total_grad_norm += param.grad.norm().item() ** 2
            params_with_grad += 1
            for i in range(num_layers):
                if f"layers.{i}" in name or f"decoder.layers.{i}" in name:
                    layers_with_grad.add(i)
    total_grad_norm = total_grad_norm ** 0.5
    n_layers_grad = len(layers_with_grad)
    grad_status = "PASS" if n_layers_grad == num_layers else "PARTIAL"
    if local_rank == 0:
        print(f"Backward: {grad_status}, grad_norm={total_grad_norm:.4f}, "
              f"params_with_grad={params_with_grad}, layers={n_layers_grad}/{num_layers}")
except torch.cuda.OutOfMemoryError:
    if local_rank == 0:
        print("Backward OOM!")
    grad_status = "OOM"
    model.zero_grad()
    torch.cuda.empty_cache()

peak_mem = torch.cuda.max_memory_allocated() / 1024**3
if local_rank == 0:
    print(f"Peak GPU memory: {peak_mem:.2f} GiB")
    if overall_cos >= 0.99 and grad_status == "PASS":
        print("\n*** BSHD PS E2E: ALL PASS ***")
    elif overall_cos >= 0.99:
        print(f"\n*** BSHD PS E2E: PARTIAL (forward PASS, backward {grad_status}) ***")
    else:
        print(f"\n*** BSHD PS E2E: FAIL (cos_sim={overall_cos:.6f}) ***")

# Cleanup
parallel_state.destroy_model_parallel()
torch.distributed.destroy_process_group()