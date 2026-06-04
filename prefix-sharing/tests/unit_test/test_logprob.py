import pytest

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.logprob import (
    build_provider_prefix_last_values,
    restore_prefix_last_logprobs,
)
from prefix_sharing.core.planner import PrefixSharingPlanner


def _prefix_sharing_plan():
    planner = PrefixSharingPlanner(PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3))
    return planner.plan(
        [[1, 2, 3, 10, 11], [1, 2, 3, 20, 21, 22]],
        forward_id=10,
        micro_batch_id=20,
    )


def test_prefix_last_restore_inserts_reuse_first_suffix_logprob():
    # _meta: provider [1,2,3,10,11]; reuser [1,2,3|20,21,22] (prefix_len=3).
    # Suffix forward only yields logprobs for 21 and 22; token 20 is restored from provider pos 2.
    prefix_sharing_plan = _prefix_sharing_plan()
    suffix_logprobs = [
        [0.10, 0.11, 0.12, 0.13, 0.14],  # provider row, already complete
        [0.21, 0.22],  # reuser: slot 0 is the second suffix token (21), not 20
    ]
    first_suffix_logprobs_by_batch = [0.0, 0.20]  # precomputed from provider logits + label 20

    restored = restore_prefix_last_logprobs(
        suffix_logprobs,
        first_suffix_logprobs_by_batch,
        prefix_sharing_plan,
    )

    assert restored == [
        [0.10, 0.11, 0.12, 0.13, 0.14],
        [0.20, 0.21, 0.22],  # 0.20 prepended to match full-sequence causal LM logprobs
    ]


def test_build_provider_prefix_last_values_uses_restore_specs():
    prefix_sharing_plan = _prefix_sharing_plan()
    values = build_provider_prefix_last_values(
        [["p0", "p1", "p_last", "s0"], ["unused"]],
        prefix_sharing_plan,
    )
    # Gather p_last at provider_prefix_last_pos=2 for scoring reuser's first suffix label 20
    assert values == [None, "p_last"]


# --- Tensor-based logprob tests (require torch) ---

torch = pytest.importorskip("torch")


def test_compute_token_logprobs_from_logits():
    from prefix_sharing.core.logprob import compute_token_logprobs_from_logits

    # 2 tokens, vocab size 4
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5], [0.1, 0.2, 0.3, 0.4]])
    labels = torch.tensor([2, 3])

    logprobs = compute_token_logprobs_from_logits(logits, labels)
    assert logprobs.shape == (2,)

    # Verify manually: log_softmax(logits)[i, labels[i]]
    expected = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(logprobs, expected, atol=1e-6)


def test_gather_provider_prefix_last_logits():
    from prefix_sharing.core.logprob import gather_provider_prefix_last_logits

    prefix_sharing_plan = _prefix_sharing_plan()
    # Batch=2, seq=5, vocab=4
    logits = torch.randn(2, 6, 4)
    result = gather_provider_prefix_last_logits(logits, prefix_sharing_plan)

    assert result.shape == (2, 4)
    # Reuser (index 1) should have provider's prefix-last logits (pos=2)
    assert torch.equal(result[1], logits[0, 2])
    # Non-reuse row should be zeros
    assert torch.all(result[0] == 0)


def test_restore_prefix_last_logprobs_tensor():
    from prefix_sharing.core.logprob import restore_prefix_last_logprobs_tensor

    prefix_sharing_plan = _prefix_sharing_plan()
    # Provider (index 0): kept_length=5 tokens
    # Reuser (index 1): kept_length=3 tokens (suffix only, no first-suffix)
    # Padded to max kept_length (5) since torch.tensor requires uniform dimensions
    suffix_logprobs = torch.zeros(2, 5)
    suffix_logprobs[0, :5] = torch.tensor([0.10, 0.11, 0.12, 0.13, 0.14])
    suffix_logprobs[1, :3] = torch.tensor([0.21, 0.22, 0.23])
    first_suffix_logprobs = torch.tensor([0.0, 0.20])  # restored value for reuser

    restored = restore_prefix_last_logprobs_tensor(
        suffix_logprobs, first_suffix_logprobs, prefix_sharing_plan,
    )

    assert restored.shape[0] == 2
    # Provider row stays the same
    assert torch.allclose(restored[0, :5], suffix_logprobs[0])
    # Reuser row has first_suffix prepended
    assert torch.allclose(restored[1, 0], first_suffix_logprobs[1])
    assert torch.allclose(restored[1, 1:4], suffix_logprobs[1, :3])
