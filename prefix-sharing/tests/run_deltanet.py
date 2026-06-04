#!/usr/bin/env python3
"""DeltaNet recurrent state reuse precision test.

Validates that prefix-sharing's build_deltanet_states produces identical
results compared to independent forward passes.

The DeltaNet state is modeled as a cumulative trajectory (cumsum), which
is a simplified version of the GatedDeltaNet recurrent update. The key
invariant: a reuser's suffix trajectory, starting from the provider's
prefix boundary state, must match the corresponding portion of an
independent full-sequence trajectory.

Usage:
    PYTHONPATH=prefix-sharing python tests/run_deltanet.py
    PYTHONPATH=prefix-sharing python tests/run_deltanet.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
except ImportError:
    print("PyTorch required")
    sys.exit(1)

from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.prefix_store import PrefixDeltanetStore


class Result:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self):
        return self.failed == 0

    def check_close(self, name, actual, expected, atol=1e-6, rtol=1e-5):
        if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
            max_diff = (actual - expected).abs().max().item()
            self.failed += 1
            msg = f"FAIL {name}: max_diff={max_diff:.2e}"
            self.errors.append(msg)
            print(f"  {msg}")
            return False
        self.passed += 1
        return True

    def check_true(self, name, cond, msg=""):
        if not cond:
            self.failed += 1
            full = f"FAIL {name}: {msg}" if msg else f"FAIL {name}"
            self.errors.append(full)
            print(f"  {full}")
            return False
        self.passed += 1
        return True

    def summary(self):
        total = self.passed + self.failed
        status = "PASS" if self.ok() else "FAIL"
        print(f"\n{'='*60}")
        print(f"DeltaNet Results: {self.passed}/{total} [{status}]")
        if self.errors:
            for e in self.errors:
                print(f"  - {e}")
        return self.ok()


def run_test(sequences, state_dim, device, result, name):
    """Run DeltaNet state reuse test.

    Model: state_trajectory[t] = cumsum(state_update[:t+1])
    For reuser: suffix_trajectory = provider_state[prefix_len-1] + cumsum(suffix_update)
    """
    print(f"\n[{name}] batch={len(sequences)}, state_dim={state_dim}")

    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()

    result.check_true(f"{name}/has_sharing", plan.has_sharing)
    if not plan.has_sharing:
        return

    seq_lens = [len(s) for s in sequences]

    # Token-ID-based state updates (same token → same update, like KV embeddings)
    all_token_ids = set()
    for seq in sequences:
        all_token_ids.update(seq)
    max_tid = max(all_token_ids) + 1
    update_emb = torch.randn(max_tid, state_dim, device=device)

    # Per-sequence updates: indexed by token ID
    all_updates = [update_emb[seq] for seq in sequences]

    # Independent: full cumulative trajectory for each sequence
    ind_trajectories = [u.cumsum(dim=0) for u in all_updates]

    # Prefix-sharing: trim updates to kept ranges, then build states
    trimmed = []
    for i, upd in enumerate(all_updates):
        s, e = plan.input_keep_ranges[i]
        trimmed.append(upd[s:e])

    packed = torch.cat(trimmed, dim=0)
    store = PrefixDeltanetStore()
    ps_output = backend.build_deltanet_states(packed, store, plan, layer_id=0)

    ps_rows = list(torch.split(ps_output, plan.kept_lengths_q))

    # Verify: PS output for each sequence
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        kept_len = e - s
        if not plan.is_reuser(i):
            # Provider: output is the full cumulative trajectory (same positions as independent)
            result.check_close(f"{name}/provider_seq{i}", ps_rows[i][:kept_len],
                             ind_trajectories[i][:kept_len])
        else:
            # Reuser: output is the suffix trajectory starting from provider's prefix boundary.
            # The backend produces: provider_state[prefix_len-1] + cumsum(suffix_updates)
            # This should equal the independent trajectory at suffix positions.
            #
            # Independent trajectory at position t: cumsum(updates[:t+1])
            # PS suffix trajectory at relative position j: provider_state[p-1] + cumsum(suffix_updates[:j+1])
            #   where provider_state[p-1] = cumsum(provider_updates[:p])[-1]
            #   and suffix_updates = updates[p:]
            #
            # So PS suffix[j] = cumsum(updates[:p])[-1] + cumsum(updates[p:p+j+1])
            #                 = cumsum(updates[:p+j+1])[-1]
            #                 = ind_trajectory[p+j]
            # This is exactly the independent trajectory from position prefix_len onwards.
            result.check_close(f"{name}/reuser_seq{i}", ps_rows[i][:kept_len],
                             ind_trajectories[i][s:e], atol=1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print(f"DeltaNet State Reuse Tests | device={args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    result = Result()
    t0 = time.time()

    # Test 1: Basic 2-sequence
    run_test([[1,2,3,10,20], [1,2,3,30,40,50]], 8, args.device, result, "basic_2seq")

    # Test 2: Cascading 3-sequence
    run_test([[1,2,3], [1,2,3,4], [1,2,3,4,5]], 8, args.device, result, "cascading_3seq")

    # Test 3: RL-like batch
    prompt = list(range(100, 164))
    seqs = [prompt + [200+i*10+j for j in range(16)] for i in range(8)]
    run_test(seqs, 16, args.device, result, "rl_8x64x16")

    # Test 4: Mixed (shared + unshared)
    run_test([[1,2,3,4,10], [1,2,3,4,20,21], [1,2,3,4,30], [99,98]], 8, args.device, result, "mixed_4seq")

    # Test 5: Gradient flow through DeltaNet states
    print("\n[test_grad] Gradient flow through DeltaNet build")
    torch.manual_seed(42)
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    seqs = [[1,2,3,10], [1,2,3,20,21]]
    plan = PrefixSharingPlanner(config).plan(seqs, forward_id=0, micro_batch_id=0)
    backend = TorchReferenceBackend()

    seq_lens = [len(s) for s in seqs]
    all_upd = torch.randn(sum(seq_lens), 8, requires_grad=True)
    upd_rows = list(torch.split(all_upd, seq_lens))

    trimmed = []
    for i, upd in enumerate(upd_rows):
        s, e = plan.input_keep_ranges[i]
        trimmed.append(upd[s:e])
    packed = torch.cat(trimmed, dim=0)

    store = PrefixDeltanetStore()
    out = backend.build_deltanet_states(packed, store, plan, layer_id=0)
    loss = out.sum()
    loss.backward()

    result.check_true("grad_exists", all_upd.grad is not None, "Gradient is None")
    if all_upd.grad is not None:
        provider_grad = all_upd.grad[:seq_lens[0]].abs().sum().item()
        result.check_true("provider_grad_nonzero", provider_grad > 0,
                         f"Provider grad sum={provider_grad:.6f}")

    elapsed = time.time() - t0
    ok = result.summary()
    print(f"Time: {elapsed:.2f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
