#!/usr/bin/env python3
"""v18: BSHD-mode Prefix-Sharing E2E test for Qwen3-27B (16-layer).

Key fixes from v17:
1. Override sequence_parallel=False (critical for BSHD PS, TP>1 defaults to True)
2. Pass attention_mask=None (DotProductAttention expects 4D mask, causal masking
   auto-generates it when mask=None — see FusedScaleMaskSoftmax line 199)
3. Skip preprocess_bshd (no left padding needed for same-length sequences,
   avoids 2D→4D mask format mismatch)

PS flow (same as v17):
- Prefix pass: SelfAttention/GatedDeltaNetAttention forward → store KV/carry
- Suffix pass: SelfAttention loads stored KV, expands + concatenates → flash_attn_varlen_func
                GatedDeltaNet loads stored carry state → injects as initial state

Note: PS hook intercepts BEFORE RoPE (attention.py lines 582-606 before 608-645),
so the hook can apply RoPE at correct absolute positions.

Usage: torchrun --nproc_per_node=4 scripts/test_bshd_ps_e2e.py
"""
import os
import sys
import time
from dataclasses import dataclass

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
# CRITICAL: override sequence_parallel=False for BSHD PS
from verl.models.mcore.registry import init_mcore_model, hf_to_mcore_config

tfconfig = hf_to_mcore_config(config, torch.bfloat16, sequence_parallel=False)
if local_rank == 0:
    print(f"TransformerConfig: sequence_parallel={tfconfig.sequence_parallel}")
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
full_position_ids = torch.arange(total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)

torch.cuda.synchronize()
t_normal_start = time.time()

with torch.no_grad():
    # attention_mask=None: causal masking auto-generates the mask
    # (FusedScaleMaskSoftmax line 199: causal mask generated when mask=None)
    logits_normal = model(
        input_ids=full_input_ids,
        position_ids=full_position_ids,
        attention_mask=None,
    )

torch.cuda.synchronize()
t_normal = time.time() - t_normal_start

# Extract suffix log_probs
log_probs_normal = F.log_softmax(logits_normal.float(), dim=-1)
suffix_labels = full_input_ids[:, PREFIX_LEN:]  # (N, SUFFIX_LEN)
# logits at position p predict token at position p+1
# For suffix starting at PREFIX_LEN: logits[PREFIX_LEN-1] predicts token at PREFIX_LEN
selected_normal = log_probs_normal[:, PREFIX_LEN-1:-1, :].gather(
    dim=-1, index=suffix_labels.unsqueeze(-1)
).squeeze(-1).to(device='cpu', dtype=torch.float32)

del logits_normal, log_probs_normal
torch.cuda.empty_cache()

if local_rank == 0:
    print(f"Step 1 (normal forward): {t_normal:.3f}s, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

# ===== Step 2: Two-pass PS forward =====
suffix_input_ids = torch.stack(suffix_tokens_list)  # (N, SUFFIX_LEN)
prefix_input_ids = prefix_tokens.unsqueeze(0)  # (1, PREFIX_LEN)

# Suffix position_ids start at PREFIX_LEN (absolute positions)
suffix_position_ids = torch.arange(PREFIX_LEN, total_len, device=device).unsqueeze(0).expand(N_SEQUENCES, -1)
prefix_position_ids = torch.arange(PREFIX_LEN, device=device).unsqueeze(0)

# Build PS context
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
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

# Simple state object with the required attributes (context manager creates stores)
@dataclass
class PsState:
    prefix_sharing_plan: object
    packed_batch_layout: object
    backend: object
    model_spec: object

ps_runtime_state = PsState(
    prefix_sharing_plan=ps_plan,
    packed_batch_layout=suffix_layout,
    backend=None,
    model_spec=model_spec,
)

torch.cuda.synchronize()
t_ps_start = time.time()

with prefix_sharing_runtime_context(ps_runtime_state) as ctx:
    # Prefix pass (no grad)
    with torch.no_grad():
        prefix_logits = model(
            input_ids=prefix_input_ids,
            position_ids=prefix_position_ids,
            attention_mask=None,
        )
    del prefix_logits
    torch.cuda.empty_cache()

    if local_rank == 0:
        print(f"  Prefix pass done, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
        kv_count = len(ctx.store._slots)
        dn_count = len(ctx.deltanet_store._slots)
        print(f"  Store: KV slots={kv_count}, DeltaNet slots={dn_count}")

    # Suffix pass (with gradients)
    suffix_logits = model(
        input_ids=suffix_input_ids,
        position_ids=suffix_position_ids,
        attention_mask=None,
    )

torch.cuda.synchronize()
t_ps = time.time() - t_ps_start

# Extract suffix log_probs from PS forward
log_probs_ps = F.log_softmax(suffix_logits.float(), dim=-1)
# PS suffix-only: logits at position p (absolute PREFIX_LEN+p) predict token at p+1 (absolute PREFIX_LEN+p+1)
# We want log_probs for suffix tokens 1..SUFFIX_LEN-1 (absolute positions PREFIX_LEN+1..total_len-1)
suffix_labels_ps = suffix_input_ids[:, 1:]  # (N, SUFFIX_LEN-1) — tokens at suffix positions 1..SUFFIX_LEN-1
selected_ps = log_probs_ps[:, :-1, :].gather(
    dim=-1, index=suffix_labels_ps.unsqueeze(-1)
).squeeze(-1)

del suffix_logits, log_probs_ps
torch.cuda.empty_cache()

if local_rank == 0:
    print(f"Step 2 (PS forward): {t_ps:.3f}s, GPU {torch.cuda.memory_allocated()/1024**3:.2f} GiB")

# ===== Step 3: Precision check =====
# Normal: selected_normal[:, j] = log_prob of suffix token j (at absolute pos PREFIX_LEN+j)
# PS:     selected_ps[:, j] = log_prob of suffix token j+1 (at absolute pos PREFIX_LEN+j+1)
# Align: selected_normal[:, 1:] vs selected_ps (both predict suffix tokens 1..SUFFIX_LEN-1)
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
ps_runtime_state2 = PsState(
    prefix_sharing_plan=ps_plan,
    packed_batch_layout=suffix_layout,
    backend=None,
    model_spec=model_spec,
)

grad_status = "SKIP"
with prefix_sharing_runtime_context(ps_runtime_state2) as ctx2:
    # Prefix pass
    with torch.no_grad():
        prefix_logits2 = model(
            input_ids=prefix_input_ids,
            position_ids=prefix_position_ids,
            attention_mask=None,
        )
    del prefix_logits2
    torch.cuda.empty_cache()

    # Suffix pass with gradients
    suffix_logits2 = model(
        input_ids=suffix_input_ids,
        position_ids=suffix_position_ids,
        attention_mask=None,
    )

loss = suffix_logits2.float().mean()

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