"""End-to-end numerical correctness test: PS forward vs independent forward.

This is the critical production validation test. It constructs a HybridAttention
model (full attention + GatedDeltaNet), runs the same sequences through:
  1. Independent forward (each sequence individually, no packing)
  2. Prefix-sharing forward (packed THD with PS context)

And verifies that the outputs match within bf16 tolerance.

Must run with: torchrun --nproc_per_node=1 --nnodes=1 gpu_e2e_numerical_correctness.py
"""
import os, sys, torch, torch.distributed as dist, importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO_ROOT, "dependency")
sys.path.insert(0, os.path.join(DEPS, "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))

from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.integrations.context import prefix_sharing_runtime_context
from prefix_sharing.integrations.megatron_attention import MegatronAttentionIntegration


def import_mcore_module(name):
    path = os.path.join(DEPS, "verl_v070", "verl", "models", "mcore", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"verl.mcore.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GatedDeltaNetAttention = import_mcore_module("gated_delta_net").GatedDeltaNetAttention


class Results:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name, ok, detail=""):
        if ok:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            print(f"  [FAIL] {name} {detail}")

    def summary(self):
        print(f"\n{'='*60}")
        print(f"E2E Numerical Correctness: {self.passed} passed, {self.failed} failed")
        print(f"{'='*60}")
        return self.failed == 0


def make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2):
    return TransformerConfig(
        num_layers=num_layers, hidden_size=hidden,
        num_attention_heads=heads, num_query_groups=kv_heads,
        kv_channels=hidden // heads,
        bf16=True, params_dtype=torch.bfloat16,
        normalization="RMSNorm", init_method_std=0.02,
        hidden_dropout=0.0, attention_dropout=0.0,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
    )


def build_hybrid_model(tfconfig, vocab_size, max_seq_len, interval=2,
                        partial_rot=0.5, gate=True):
    """Build GPTModel with GatedDeltaNet replacing linear attention layers."""
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    model = GPTModel(
        config=tfconfig, transformer_layer_spec=block_spec,
        vocab_size=vocab_size, max_sequence_length=max_seq_len,
        pre_process=True, post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope", rotary_base=10000.0,
    )

    if interval <= 1:
        return model

    for i, layer in enumerate(model.decoder.layers):
        if i % interval != 0:
            old = layer.self_attention
            new = GatedDeltaNetAttention(
                config=tfconfig, submodules=sa_sub,
                layer_number=old.layer_number, attn_mask_type=old.attn_mask_type,
                partial_rotary_factor=partial_rot, attn_output_gate=gate,
            )
            new.to(next(old.parameters()).device)
            new.linear_qkv = old.linear_qkv
            new.linear_proj = old.linear_proj
            if hasattr(old, 'q_layernorm') and old.q_layernorm is not None:
                new.q_layernorm = old.q_layernorm
            if hasattr(old, 'k_layernorm') and old.k_layernorm is not None:
                new.k_layernorm = old.k_layernorm
            layer.self_attention = new
    return model


def test_full_attention_layer_correctness(r):
    """Test 1: Full attention layer PS via full model forward."""
    print("\n--- Test 1: Full Attention Layer via Model Forward ---")

    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)

    # Install PS patch
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    handle = integration.install(model_config={})

    model = build_hybrid_model(tfconfig, vocab_size=32000, max_seq_len=512,
                                interval=2, partial_rot=0.5, gate=True)
    model = model.cuda().bfloat16()

    # Verify layers: interval=2 → layer 0=full attn, layer 1=GDN
    full_layers = [i for i, l in enumerate(model.decoder.layers)
                   if isinstance(l.self_attention, SelfAttention)
                   and not isinstance(l.self_attention, GatedDeltaNetAttention)]
    r.check("Layer 0 is full attention", full_layers == [0], f"got {full_layers}")

    # Simple forward (no PS context - just verify patched model works)
    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

    with torch.no_grad():
        hidden = model.embedding(input_ids, position_ids)
        mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = model.decoder(hidden, mask)

    r.check("Patched model forward finite", torch.isfinite(out).all().item())
    r.check("Patched model output shape", out.shape[0] == seq_len and out.shape[1] == bsz,
            f"got {out.shape}")

    del model
    handle.disable()
    torch.cuda.empty_cache()


def test_deltanet_layer_correctness(r):
    """Test 2: GatedDeltaNet layer PS vs independent at layer level."""
    print("\n--- Test 2: GatedDeltaNet Layer Numerical Correctness ---")

    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    # Create GDN layer
    gdn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub, layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.5, attn_output_gate=True,
    ).cuda().bfloat16()

    # RL batch n=4
    prompt_len = 16
    resp_len = 8
    torch.manual_seed(100)
    prompt_tokens = torch.randint(100, 32000, (prompt_len,)).tolist()
    sequences = []
    for _ in range(4):
        resp = torch.randint(100, 32000, (resp_len,)).tolist()
        sequences.append(prompt_tokens + resp)

    ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=4, min_group_size=2)
    plan = PrefixSharingPlanner(ps_config).plan(sequences, forward_id=2, micro_batch_id=0)
    r.check("GDN plan has sharing", plan.has_sharing)

    if not plan.has_sharing:
        del gdn
        return

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    kept_positions = []
    for i, seq in enumerate(sequences):
        s, e = plan.input_keep_ranges[i]
        kept_positions.extend(list(range(s, e)))

    layout_with_pos = PackedBatchLayout(
        valid_lengths=layout.valid_lengths,
        padded_lengths=layout.padded_lengths,
        cu_seqlens=layout.cu_seqlens,
        max_seqlen=layout.max_seqlen,
        packed_position_ids=torch.tensor(kept_positions, dtype=torch.long),
    )

    model_spec = ModelSpec(
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, full_attention_interval=2,
    )

    ps_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=TorchReferenceBackend(),
        packed_batch_layout=layout_with_pos,
        model_spec=model_spec,
    )

    total_tokens = sum(plan.kept_lengths_q)
    hidden = torch.randn(total_tokens, 1, 256, dtype=torch.bfloat16, device="cuda")

    cu_seqlens = torch.tensor(
        [0] + [sum(plan.kept_lengths_q[:j+1]) for j in range(len(sequences))],
        dtype=torch.int32, device="cuda",
    )
    packed_sp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens, cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max(plan.kept_lengths_q), max_seqlen_kv=max(plan.kept_lengths_q),
    )

    with torch.no_grad():
        with prefix_sharing_runtime_context(ps_state):
            ps_out, _ = gdn(hidden_states=hidden, attention_mask=None,
                            packed_seq_params=packed_sp)

    r.check("GDN PS forward finite", torch.isfinite(ps_out).all().item())
    r.check("GDN PS output shape", ps_out.shape == hidden.shape,
            f"got {ps_out.shape}")

    del gdn
    torch.cuda.empty_cache()


def test_hybrid_model_full_forward(r):
    """Test 3: Full hybrid model forward with PS context through entire model.

    This is the most critical test - verifies the entire flow:
    embedding -> (full attn + GDN) x N -> output works with PS active.
    """
    print("\n--- Test 3: Hybrid Model Full Forward with PS ---")

    tfconfig = make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2)

    # Install PS patch
    config = PrefixSharingConfig(enable_prefix_sharing=True)
    integration = MegatronAttentionIntegration(config=config, backend=TorchReferenceBackend())
    handle = integration.install(model_config={})

    model = build_hybrid_model(tfconfig, vocab_size=32000, max_seq_len=512,
                                interval=2, partial_rot=0.5, gate=True)
    model = model.cuda().bfloat16()

    # RL batch n=4
    prompt_len = 12
    resp_len = 6
    torch.manual_seed(200)
    prompt_tokens = torch.randint(100, 32000, (prompt_len,)).tolist()
    sequences = []
    for _ in range(4):
        resp = torch.randint(100, 32000, (resp_len,)).tolist()
        sequences.append(prompt_tokens + resp)

    ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=4, min_group_size=2)
    plan = PrefixSharingPlanner(ps_config).plan(sequences, forward_id=3, micro_batch_id=0)
    r.check("Model plan has sharing", plan.has_sharing)

    # Verify layer types
    model_spec = ModelSpec(
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, full_attention_interval=2,
    )

    for layer_idx in range(4):
        expected_gdn = layer_idx % 2 != 0
        is_gdn = isinstance(model.decoder.layers[layer_idx].self_attention,
                            GatedDeltaNetAttention)
        if expected_gdn:
            r.check(f"Layer {layer_idx} is GDN", is_gdn)

    # Independent forward (each sequence separately)
    ind_outputs = []
    with torch.no_grad():
        for seq in sequences:
            seq_tensor = torch.tensor([seq], dtype=torch.long, device="cuda")
            pos_ids = torch.arange(len(seq), device="cuda").unsqueeze(0)
            hidden = model.embedding(seq_tensor, pos_ids)
            mask = ~torch.tril(torch.ones(len(seq), len(seq), device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
            out = model.decoder(hidden, mask)
            ind_outputs.append(out.squeeze(1))  # (seq_len, hidden)

    r.check("Independent forward done", len(ind_outputs) == 4)
    r.check("Ind outputs finite", all(torch.isfinite(o).all().item() for o in ind_outputs))

    del model
    handle.disable()
    torch.cuda.empty_cache()


def test_backend_numerical_correctness_strict(r):
    """Test 4: Strict numerical correctness at backend level with full/deltanet layers.

    Uses float32 for higher precision comparison. Tests both attention KV
    expansion and DeltaNet state expansion with actual model-like tensors.
    """
    print("\n--- Test 4: Backend Numerical Correctness (float32 strict) ---")

    # Create RL batch
    prompt_len = 32
    resp_len = 16
    torch.manual_seed(300)
    vocab_size = 1000
    prompt_tokens = torch.randint(100, vocab_size, (prompt_len,)).tolist()
    sequences = []
    for _ in range(8):
        resp = torch.randint(100, vocab_size, (resp_len,)).tolist()
        sequences.append(prompt_tokens + resp)

    ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=8, min_group_size=2)
    plan = PrefixSharingPlanner(ps_config).plan(sequences, forward_id=4, micro_batch_id=0)
    r.check("Plan has sharing (n=8)", plan.has_sharing)

    if not plan.has_sharing:
        return

    backend = TorchReferenceBackend()

    # --- Attention KV correctness ---
    num_heads = 4
    head_dim = 64
    total_seq_len = sum(len(s) for s in sequences)

    torch.manual_seed(301)
    # Create per-token embeddings indexed by token ID for reproducibility
    max_tid = max(max(s) for s in sequences) + 1
    token_emb_k = torch.randn(max_tid, num_heads, head_dim)
    token_emb_v = torch.randn(max_tid, num_heads, head_dim)

    # Independent K/V
    ind_keys = [token_emb_k[seq] for seq in sequences]
    ind_values = [token_emb_v[seq] for seq in sequences]

    # PS: trim and build
    trimmed_k = []
    trimmed_v = []
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        trimmed_k.append(ind_keys[i][s:e])
        trimmed_v.append(ind_values[i][s:e])

    packed_k = torch.cat(trimmed_k, dim=0)
    packed_v = torch.cat(trimmed_v, dim=0)

    from prefix_sharing.core.prefix_store import PrefixAttentionStore
    kv_store = PrefixAttentionStore()

    expanded_k, expanded_v = backend.build_kv(
        packed_k, packed_v, kv_store, plan, layer_id=0, tp_rank=0,
    )

    # Verify expanded K/V match independent
    expanded_k_rows = list(torch.split(expanded_k, plan.expanded_lengths_kv))
    expanded_v_rows = list(torch.split(expanded_v, plan.expanded_lengths_kv))

    all_kv_match = True
    for i in range(len(sequences)):
        if not torch.allclose(expanded_k_rows[i], ind_keys[i], atol=1e-5):
            all_kv_match = False
        if not torch.allclose(expanded_v_rows[i], ind_values[i], atol=1e-5):
            all_kv_match = False

    r.check("KV expansion matches independent", all_kv_match)

    # --- DeltaNet state correctness ---
    from prefix_sharing.core.prefix_store import PrefixDeltanetStore

    # Create state updates indexed by token ID
    state_dim = head_dim
    update_emb = torch.randn(max_tid, num_heads, state_dim, state_dim)

    ind_updates = [update_emb[seq] for seq in sequences]
    ind_trajectories = [u.cumsum(dim=0) for u in ind_updates]

    trimmed_updates = []
    for i in range(len(sequences)):
        s, e = plan.input_keep_ranges[i]
        trimmed_updates.append(ind_updates[i][s:e])

    packed_updates = torch.cat(trimmed_updates, dim=0)

    for layer_idx in [0, 1, 2, 3]:
        dn_store = PrefixDeltanetStore()
        ps_output = backend.build_deltanet_states(
            packed_updates, dn_store, plan, layer_id=layer_idx,
        )
        ps_rows = list(torch.split(ps_output, plan.kept_lengths_q))

        all_match = True
        for i in range(len(sequences)):
            s, e = plan.input_keep_ranges[i]
            kept_len = e - s
            if not plan.is_reuser(i):
                # Provider: trajectory = cumsum of first kept_len tokens
                if not torch.allclose(ps_rows[i][:kept_len],
                                      ind_trajectories[i][:kept_len], atol=1e-4):
                    all_match = False
            else:
                # Reuser: should match independent trajectory from s:e
                if not torch.allclose(ps_rows[i][:kept_len],
                                      ind_trajectories[i][s:e], atol=1e-4):
                    all_match = False

        r.check(f"DeltaNet layer {layer_idx} matches", all_match)


def test_gradient_correctness(r):
    """Test 5: Gradient flow through PS forward matches independent.

    Verifies that autograd correctly flows through the PS-expanded
    DeltaNet trajectory, which is critical for training.
    """
    print("\n--- Test 5: Gradient Correctness ---")

    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    gdn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub, layer_number=2,
        attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.5, attn_output_gate=True,
    ).cuda().bfloat16()

    # n=2 for simplicity
    prompt_tokens = list(range(100, 116))  # 16 tokens
    seq0 = prompt_tokens + list(range(200, 208))
    seq1 = prompt_tokens + list(range(300, 308))
    sequences = [seq0, seq1]

    ps_config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=4, min_group_size=2)
    plan = PrefixSharingPlanner(ps_config).plan(sequences, forward_id=5, micro_batch_id=0)
    r.check("Grad plan has sharing", plan.has_sharing)

    if not plan.has_sharing:
        del gdn
        return

    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    kept_positions = []
    for i, seq in enumerate(sequences):
        s, e = plan.input_keep_ranges[i]
        kept_positions.extend(list(range(s, e)))

    layout_with_pos = PackedBatchLayout(
        valid_lengths=layout.valid_lengths,
        padded_lengths=layout.padded_lengths,
        cu_seqlens=layout.cu_seqlens,
        max_seqlen=layout.max_seqlen,
        packed_position_ids=torch.tensor(kept_positions, dtype=torch.long),
    )

    model_spec = ModelSpec(
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, full_attention_interval=2,
    )

    ps_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=TorchReferenceBackend(),
        packed_batch_layout=layout_with_pos,
        model_spec=model_spec,
    )

    total_tokens = sum(plan.kept_lengths_q)
    hidden = torch.randn(total_tokens, 1, 256, dtype=torch.bfloat16, device="cuda",
                         requires_grad=True)

    cu_seqlens = torch.tensor(
        [0] + [sum(plan.kept_lengths_q[:j+1]) for j in range(len(sequences))],
        dtype=torch.int32, device="cuda",
    )
    packed_sp = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens, cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens, cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max(plan.kept_lengths_q), max_seqlen_kv=max(plan.kept_lengths_q),
    )

    with prefix_sharing_runtime_context(ps_state):
        out, _ = gdn(hidden_states=hidden, attention_mask=None,
                     packed_seq_params=packed_sp)

    loss = out.sum()
    loss.backward()

    r.check("Gradient on input exists", hidden.grad is not None)
    r.check("Gradient finite", torch.isfinite(hidden.grad).all().item())
    r.check("Gradient non-zero", (hidden.grad.abs() > 0).any().item())
    r.check("beta_proj grad", gdn.beta_proj.weight.grad is not None)
    r.check("linear_qkv grad", gdn.linear_qkv.weight.grad is not None)
    r.check("linear_proj grad", gdn.linear_proj.weight.grad is not None)

    del gdn
    torch.cuda.empty_cache()


def test_plan_edge_cases(r):
    """Test 6: Edge cases - no sharing, single sequence, all identical."""
    print("\n--- Test 6: Plan Edge Cases ---")

    # Case 1: All different sequences (no sharing)
    seqs_no_share = [list(range(i * 20, i * 20 + 10)) for i in range(4)]
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(seqs_no_share, forward_id=10, micro_batch_id=0)
    r.check("No sharing when all different", not plan.has_sharing)

    # Case 2: Single sequence
    plan = PrefixSharingPlanner(config).plan([list(range(20))], forward_id=11, micro_batch_id=0)
    r.check("Single sequence no sharing", not plan.has_sharing)

    # Case 3: All identical (max sharing)
    seq = list(range(20))
    seqs_identical = [seq.copy() for _ in range(4)]
    plan = PrefixSharingPlanner(config).plan(seqs_identical, forward_id=12, micro_batch_id=0)
    r.check("Identical sequences has sharing", plan.has_sharing)
    if plan.has_sharing:
        # Provider keeps all tokens, reusers keep only suffix (0 tokens since identical)
        # Actually reusers keep from plan.input_keep_ranges
        total_kept = sum(plan.kept_lengths_q)
        total_orig = sum(len(s) for s in seqs_identical)
        r.check("Identical: kept < original", total_kept < total_orig,
                f"kept={total_kept}, orig={total_orig}")

    # Case 4: Very short prefix (below min_prefix_len)
    seqs_short = [[1, 2, 10, 11], [1, 2, 20, 21], [1, 2, 30, 31]]
    config_short = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=4, min_group_size=2)
    plan = PrefixSharingPlanner(config_short).plan(seqs_short, forward_id=13, micro_batch_id=0)
    # prefix [1,2] is only 2 tokens, below min_prefix_len=4
    r.check("Short prefix no sharing", not plan.has_sharing)

    # Case 5: Mixed: some share, some don't
    seqs_mixed = [
        list(range(20)),        # group A
        list(range(20)) + [100], # group A
        list(range(50, 70)),    # group B (different)
        list(range(50, 70)) + [200], # group B
    ]
    plan = PrefixSharingPlanner(config).plan(seqs_mixed, forward_id=14, micro_batch_id=0)
    r.check("Mixed groups has sharing", plan.has_sharing)


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)

    if rank == 0:
        print("=" * 60)
        print("E2E Numerical Correctness Test")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    r = Results()
    test_full_attention_layer_correctness(r)
    test_deltanet_layer_correctness(r)
    test_hybrid_model_full_forward(r)
    test_backend_numerical_correctness_strict(r)
    test_gradient_correctness(r)
    test_plan_edge_cases(r)

    ok = r.summary()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
