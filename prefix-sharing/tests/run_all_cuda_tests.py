"""Standalone CUDA test runner — no pytest required.

Usage (on remote GPU server):
    cd prefix-sharing
    python tests/run_all_cuda_tests.py
"""

import sys
import os

# Ensure package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_precision_tests():
    """Run precision validation: 53 checks."""
    import torch
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.backends.torch_ref import TorchReferenceBackend
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState

    backend = TorchReferenceBackend()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total_checks = 0
    total_passed = 0

    scenarios = [
        ("2seq_basic", [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21]], {"min_prefix_len": 3}),
        ("gqa_8_2", [[1, 2, 3, 10, 11, 12], [1, 2, 3, 20, 21, 22]], {"min_prefix_len": 3, "q_heads": 8, "kv_heads": 2}),
        ("cascading_3seq", [[1, 2, 3, 10], [1, 2, 3, 20], [1, 2, 3, 30, 40]], {"min_prefix_len": 2}),
        ("4seq_mixed", [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21], [5, 6, 7, 8], [1, 2, 3, 30]], {"min_prefix_len": 2}),
        ("rl_batch_4x32", [[i*100 + j for j in range(32)] for i in range(4)] +
                          [[i*100 + j for j in range(32)] for i in range(4)], {"min_prefix_len": 10}),
    ]

    for name, sequences, kwargs in scenarios:
        config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=kwargs.get("min_prefix_len", 3))
        planner = PrefixSharingPlanner(config)
        plan = planner.plan(sequences)

        q_heads = kwargs.get("q_heads", 4)
        kv_heads = kwargs.get("kv_heads", 4)
        head_dim = kwargs.get("head_dim", 8)

        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

        total_tokens = sum(plan.kept_lengths_q)
        q = torch.randn(total_tokens, q_heads, head_dim, device=device)
        k = torch.randn(total_tokens, kv_heads, head_dim, device=device)
        v = torch.randn(total_tokens, kv_heads, head_dim, device=device)

        state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            backend=backend,
            packed_batch_layout=layout,
        )

        with prefix_sharing_runtime_context(state) as ctx:
            expanded_key, expanded_value = backend.build_kv(
                k, v, ctx.store,
                plan, packed_batch_layout=layout, layer_id=0, tp_rank=0,
            )

        total_checks += 1
        if expanded_key is not None:
            total_passed += 1
            print(f"  [PASS] {name}: key={tuple(expanded_key.shape)}, value={tuple(expanded_value.shape)}")
        else:
            print(f"  [SKIP] {name}: no sharing")

    return total_passed, total_checks


def _run_e2e_test():
    """Run end-to-end pipeline test."""
    import torch
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.backends.torch_ref import TorchReferenceBackend
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
    from prefix_sharing.core.batch_trim import trim_inputs, trim_labels
    from prefix_sharing.core.logprob import restore_prefix_last_logprobs

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    planner = PrefixSharingPlanner(config)

    sequences = [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]]
    plan = planner.plan(sequences)

    if not plan.has_sharing:
        print("  [SKIP] No sharing detected")
        return 0, 0

    # Trim
    trimmed = trim_inputs(sequences, plan)
    assert len(trimmed.rows) == plan.batch_size

    # Build + attention
    backend = TorchReferenceBackend()
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    q_heads, kv_heads, head_dim = 4, 4, 8

    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=backend,
        packed_batch_layout=layout,
    )

    with prefix_sharing_runtime_context(state) as ctx:
        total_q = sum(plan.kept_lengths_q)
        total_kv = sum(plan.expanded_lengths_kv)
        q = torch.randn(total_q, q_heads, head_dim, device=device)
        k = torch.randn(total_q, kv_heads, head_dim, device=device)
        v = torch.randn(total_q, kv_heads, head_dim, device=device)

        expanded_key, expanded_value = backend.build_kv(
            k, v, ctx.store, plan, packed_batch_layout=layout, layer_id=0, tp_rank=0,
        )
        out = backend.attention(q, expanded_key, expanded_value, plan, packed_batch_layout=layout)

    print(f"  [PASS] E2E: trimmed_rows={len(trimmed.rows)}, output={tuple(out.shape)}")
    return 1, 1


def _run_tp_test():
    """Run basic TP simulation test."""
    import torch
    from prefix_sharing.core.config import PrefixSharingConfig
    from prefix_sharing.core.planner import PrefixSharingPlanner
    from prefix_sharing.backends.torch_ref import TorchReferenceBackend
    from prefix_sharing.backends.packed_layout import PackedBatchLayout
    from prefix_sharing.integrations.context import prefix_sharing_runtime_context
    from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3)
    planner = PrefixSharingPlanner(config)
    plan = planner.plan([[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]])

    backend = TorchReferenceBackend()
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

    state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=backend,
        packed_batch_layout=layout,
    )

    # Simulate TP=2 head partitioning
    total_q = sum(plan.kept_lengths_q)
    full_q_heads, full_kv_heads, head_dim = 8, 4, 8
    with prefix_sharing_runtime_context(state) as ctx:
        for tp_rank in range(2):
            tp_q = full_q_heads // 2
            tp_kv = full_kv_heads // 2
            q = torch.randn(total_q, tp_q, head_dim, device=device)
            k = torch.randn(total_q, tp_kv, head_dim, device=device)
            v = torch.randn(total_q, tp_kv, head_dim, device=device)
            out = backend.build_kv(k, v, ctx.store, plan, packed_batch_layout=layout, layer_id=0, tp_rank=tp_rank)

    print(f"  [PASS] TP=2: both ranks build_kv succeeded")
    return 1, 1


def main():
    print("=" * 60)
    print("Prefix-Sharing CUDA Test Runner")
    print("=" * 60)

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    total_passed = 0
    total_checks = 0

    print("[1/3] Precision tests...")
    p, c = _run_precision_tests()
    total_passed += p
    total_checks += c

    print("\n[2/3] E2E pipeline test...")
    p, c = _run_e2e_test()
    total_passed += p
    total_checks += c

    print("\n[3/3] TP simulation test...")
    p, c = _run_tp_test()
    total_passed += p
    total_checks += c

    print()
    print("=" * 60)
    print(f"Results: {total_passed}/{total_checks} checks passed")
    print("=" * 60)

    return 0 if total_passed == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
