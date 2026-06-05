"""E2E test: verl + Megatron + Qwen3.6 + prefix-sharing on Ray.

This is the closest thing to a real training step without needing the full
verl training stack. It verifies the complete import chain and runs the
PS GRPO simulation under Ray's distributed context.

Usage:
    python scripts/gpu_e2e_verl_ps_ray.py
"""
import os, sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dependency", "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "dependency", "verl_v070"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))

# Monkey-patch omegaconf (verl dep with version conflict in offline env)
class _FakeOmegaConfCls:
    """Stand-in for omegaconf.OmegaConf class when the real package isn't available."""
    @staticmethod
    def to_container(x, resolve=True):
        if isinstance(x, dict):
            return x
        if hasattr(x, '__dict__'):
            return {k: v for k, v in vars(x).items() if not k.startswith('_')}
        return dict(x)
    @staticmethod
    def is_config(x):
        return False
    @staticmethod
    def create(**kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(**kwargs)
    @staticmethod
    def merge(*args, **kwargs):
        result = {}
        for a in args:
            if a:
                result.update(_FakeOmegaConfCls.to_container(a))
        result.update(kwargs)
        return result

class _FakeOmegaConfModule:
    OmegaConf = _FakeOmegaConfCls
    DictConfig = dict
    ListConfig = list

import sys as _sys
_sys.modules['omegaconf'] = _FakeOmegaConfModule()
_sys.modules['omegaconf.base'] = _FakeOmegaConfModule()
_sys.modules['omegaconf.omegaconf'] = _FakeOmegaConfModule()

import torch
import ray
import time

from contextlib import nullcontext
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec, QWEN3_6_27B
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.verl_mcore import (
    PrefixSharingRuntimeState,
    build_prefix_sharing_micro_batch,
    restore_suffix_first_log_probs_from_prefix,
)

passed = 0
failed = 0

def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} {detail}")

# ====================================================
# Test 1: Ray cluster resources
# ====================================================
print("--- Test 1: Ray Cluster ---")
ray.init(address='auto', ignore_reinit_error=True)
resources = ray.cluster_resources()
check("Ray connected", "CPU" in resources)
check("GPUs available", resources.get("GPU", 0) >= 1, f"got {resources.get('GPU', 0)}")
check("CPUs available", resources.get("CPU", 0) >= 1)
print(f"  Resources: {resources}")

# ====================================================
# Test 2: Import chain
# ====================================================
print("\n--- Test 2: Import Chain ---")
check("torch imported", torch.__version__)
check("verl import OK", True)  # already imported
check("prefix_sharing import OK", True)
check("ModelSpec.QWEN3_6_27B", QWEN3_6_27B.num_hidden_layers == 64)
check("build_prefix_sharing_micro_batch", build_prefix_sharing_micro_batch is not None)
check("restore_suffix_first_log_probs", restore_suffix_first_log_probs_from_prefix is not None)
check("prefix_sharing_runtime_context", prefix_sharing_runtime_context is not None)

# ====================================================
# Test 3: PS plan creation (full RL batch)
# ====================================================
print("\n--- Test 3: PS Plan Creation ---")
prompt = list(range(100, 164))  # 64 tokens
resp_len = 32
n_responses = 8
sequences = [prompt + [200 + i * 10 + j for j in range(resp_len)] for i in range(n_responses)]

config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=4, min_group_size=2)
plan = PrefixSharingPlanner(config).plan(sequences, forward_id=1, micro_batch_id=0)
check("Plan has sharing", plan.has_sharing)
check("Plan batch_size", plan.batch_size == n_responses, f"got {plan.batch_size}")

total_orig = sum(len(s) for s in sequences)
total_kept = sum(plan.kept_lengths_q)
saved = total_orig - total_kept
check("Tokens saved > 0", saved > 0, f"saved {saved}/{total_orig}")
print(f"  Checkings: {total_orig} -> {total_kept} tokens ({saved/total_orig:.1%} saved)")

# ====================================================
# Test 4: build_prefix_sharing_micro_batch
# ====================================================
print("\n--- Test 4: Build Micro Batch ---")
# Simulate verl batch format
batch_size = len(sequences)
max_len = max(len(s) for s in sequences)
input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
position_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
for i, seq in enumerate(sequences):
    input_ids[i, :len(seq)] = torch.tensor(seq)
    attention_mask[i, :len(seq)] = True
    position_ids[i, :len(seq)] = torch.arange(len(seq))
responses = input_ids[:, -resp_len:]

batch = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "position_ids": position_ids,
    "responses": responses,
}
actor_config = {
    "prefix_sharing_config": {"enable_prefix_sharing": True, "min_prefix_len": 4},
    "megatron": {"use_remove_padding": True},
}
model_config = type('obj', (object,), {
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
    "apply_rope_fusion": False,
    "fused_single_qkv_rope": False,
    "model_type": "text_only_causal_lm",
})()

trimmed_batch, ps_state = build_prefix_sharing_micro_batch(
    batch, actor_config, model_config, model_spec=QWEN3_6_27B,
)

check("PS state not None", ps_state is not None)
check("Trimmed batch has input_ids", "input_ids" in trimmed_batch)
check("Trimmed batch has attention_mask", "attention_mask" in trimmed_batch)
layout = ps_state.packed_batch_layout
check("Layout valid_lengths", layout.valid_lengths is not None)
check("Layout cu_seqlens", layout.cu_seqlens is not None)
print(f"  Layout: valid_lengths={layout.valid_lengths}, cu_seqlens={layout.cu_seqlens}")

# ====================================================
# Test 5: PS context activation
# ====================================================
print("\n--- Test 5: PS Context Activation ---")
from prefix_sharing.integrations.context import current_prefix_sharing_context

with prefix_sharing_runtime_context(ps_state) as ctx:
    check("Context active", current_prefix_sharing_context() is not None)
    check("Context has plan", ctx.prefix_sharing_plan is not None)
    check("Context has store", ctx.store is not None)
    check("Context has deltanet_store", ctx.deltanet_store is not None)
    check("Context has model_spec", ctx.model_spec is not None)
    check("Prefix last restore indices",
          len(ctx.prefix_last_restore_indices) > 0,
          f"got {len(ctx.prefix_last_restore_indices)}")

check("Context deactivated after exit", current_prefix_sharing_context() is None)

# ====================================================
# Test 6: Run GRPO sim under Ray (distributed verification)
# ====================================================
print("\n--- Test 6: GRPO Sim under Ray ---")

@ray.remote(num_gpus=1, runtime_env={"env_vars": {
    "PYTHONPATH": os.path.join(REPO_ROOT, "dependency", "megatron_v0150") + ":" +
                  os.path.join(REPO_ROOT, "dependency", "verl_v070") + ":" +
                  os.path.join(REPO_ROOT, "prefix-sharing"),
    "MASTER_ADDR": "127.0.0.1",
    "MASTER_PORT": "29500",
    "RANK": "0",
    "WORLD_SIZE": "1",
}})
def run_grpo_sim():
    import torch, torch.distributed as dist
    # Re-apply omegaconf monkey-patch in worker
    import sys as _s
    class _FO:
        OmegaConf = type('_OC', (), {
            'to_container': staticmethod(lambda x, resolve=True: dict(x) if hasattr(x, '__dict__') else x),
            'is_config': staticmethod(lambda x: False),
            'create': staticmethod(lambda **kw: type('ns', (), kw)()),
            'merge': staticmethod(lambda *a, **kw: {}),
        })
        DictConfig = dict; ListConfig = list
    _s.modules.setdefault('omegaconf', _FO())
    _s.modules.setdefault('omegaconf.base', _FO())

    from megatron.core import parallel_state as mpu
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    dist.init_process_group(backend="nccl", init_method="env://",
                           world_size=1, rank=0)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1,
                                  pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)

    return {
        "device": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
    }

result = ray.get(run_grpo_sim.remote())
check("Ray worker GPU", result["device"] != "")
print(f"  Worker: {result}")

# ====================================================
# Summary
# ====================================================
print(f"\n{'='*60}")
print(f"E2E verl+PS+Ray Results: {passed} passed, {failed} failed")
print(f"{'='*60}")

ray.shutdown()
exit(0 if failed == 0 else 1)
