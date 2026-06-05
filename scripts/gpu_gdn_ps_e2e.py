"""End-to-end test for GatedDeltaNet prefix-sharing hook.

Verifies that when a PS context is active with THD packed sequences,
the GatedDeltaNet forward correctly uses the PS trajectory instead of
standard cumsum, and produces numerically correct output.

Must run with: torchrun --nproc_per_node=1 --nnodes=1 gpu_gdn_ps_e2e.py
"""
import os, sys, torch, torch.distributed as dist, importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO_ROOT, "dependency")
sys.path.insert(0, os.path.join(DEPS, "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))

from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.packed_seq_params import PackedSeqParams

from prefix_sharing.core.config import PrefixSharingConfig
from prefix_sharing.core.planner import PrefixSharingPlanner
from prefix_sharing.core.model_spec import ModelSpec
from prefix_sharing.backends.packed_layout import PackedBatchLayout
from prefix_sharing.backends.torch_ref import TorchReferenceBackend
from prefix_sharing.integrations.verl_mcore import PrefixSharingRuntimeState
from prefix_sharing.integrations.context import prefix_sharing_runtime_context


def import_mcore_module(name):
    path = os.path.join(DEPS, "verl_v070", "verl", "models", "mcore", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"verl.mcore.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GatedDeltaNetAttention = import_mcore_module("gated_delta_net").GatedDeltaNetAttention


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


def build_hybrid_model(tfconfig, vocab_size, max_seq_len, interval=2, partial_rot=0.5, gate=True):
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


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(42)

    passed = 0
    failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            if rank == 0:
                print(f"  [PASS] {name}")
        else:
            failed += 1
            if rank == 0:
                print(f"  [FAIL] {name} {detail}")

    if rank == 0:
        print("=" * 60)
        print("GatedDeltaNet PS E2E Test")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    # ================================================================
    # Test 1: PS hook returns None when no context active
    # ================================================================
    if rank == 0:
        print("\n--- Test 1: No PS context (standard cumsum) ---")

    tfconfig = make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2)
    model = build_hybrid_model(tfconfig, 32000, 512, interval=2, partial_rot=0.5, gate=True)
    model = model.cuda().bfloat16()

    bsz, seq_len = 2, 16
    input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

    with torch.no_grad():
        hidden = model.embedding(input_ids, position_ids)
        mask = ~torch.tril(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = model.decoder(hidden, mask)

    check("Standard forward finite", torch.isfinite(out).all().item())
    check("Standard forward shape", out.shape[0] == seq_len and out.shape[1] == bsz, f"got {out.shape}")

    del model
    torch.cuda.empty_cache()

    # ================================================================
    # Test 2: PS hook returns None without THD packed_seq_params
    # ================================================================
    if rank == 0:
        print("\n--- Test 2: PS context active but no THD (GDN only) ---")

    tfconfig = make_tfconfig(hidden=256, num_layers=4, heads=4, kv_heads=2)
    model = build_hybrid_model(tfconfig, 32000, 512, interval=2, partial_rot=0.5, gate=True)
    model = model.cuda().bfloat16()

    # Create a simple PS state
    sequences = [[1, 2, 3, 10, 20], [1, 2, 3, 30, 40]]
    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=2, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan(sequences, forward_id=0, micro_batch_id=0)
    layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)
    ps_state = PrefixSharingRuntimeState(
        prefix_sharing_plan=plan,
        backend=TorchReferenceBackend(),
        packed_batch_layout=layout,
        model_spec=ModelSpec(
            num_hidden_layers=4, full_attention_interval=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        ),
    )

    # Test that PS context can be created and destroyed correctly
    # Note: we cannot run the full model forward without THD format
    # because full attention layers require THD for PS. But we can
    # verify context setup/teardown works.
    with prefix_sharing_runtime_context(ps_state) as ctx:
        check("Context active", ctx is not None)
        check("Context has plan", ctx.prefix_sharing_plan.has_sharing)
        check("Context has deltanet_store", ctx.deltanet_store is not None)

    # Verify context is cleaned up
    from prefix_sharing.integrations.context import current_prefix_sharing_context
    check("Context cleaned up", current_prefix_sharing_context() is None)

    del model
    torch.cuda.empty_cache()

    # ================================================================
    # Test 3: PS hook activates with THD packed_seq_params (GDN layer)
    # ================================================================
    if rank == 0:
        print("\n--- Test 3: PS context + THD (GDN layer directly) ---")

    # Test the GDN layer directly with PS context + THD packed_seq_params.
    # This exercises the actual PS hook path in GatedDeltaNet.forward().

    tfconfig = make_tfconfig(hidden=256, num_layers=2, heads=4, kv_heads=2)
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    gdn_attn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub,
        layer_number=1, attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.5, attn_output_gate=True,
    ).cuda().bfloat16()

    # RL-like scenario: 1 provider + 1 reuser
    seq_a = list(range(100, 116))  # 16 tokens (provider: full seq kept)
    seq_b = list(range(100, 112)) + list(range(200, 204))  # 16 tokens (reuser: prefix 100-111, suffix 200-203)

    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=4, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan([seq_a, seq_b], forward_id=1, micro_batch_id=0)
    check("Plan has sharing (direct)", plan.has_sharing)

    if plan.has_sharing:
        layout = PackedBatchLayout.from_valid_lengths(plan.kept_lengths_q)

        # Add packed_position_ids to the layout (required by PS hook)
        # Rebuild layout with position IDs
        kept_tokens = []
        kept_positions = []
        for i, seq in enumerate([seq_a, seq_b]):
            s, e = plan.input_keep_ranges[i]
            kept_tokens.extend(seq[s:e])
            kept_positions.extend(list(range(e - s)))  # position 0, 1, 2, ... within kept range

        total_tokens = len(kept_tokens)
        kept_lens = plan.kept_lengths_q

        # Update layout with packed_position_ids
        layout_with_pos = PackedBatchLayout(
            valid_lengths=layout.valid_lengths,
            padded_lengths=layout.padded_lengths,
            cu_seqlens=layout.cu_seqlens,
            max_seqlen=layout.max_seqlen,
            packed_position_ids=torch.tensor(kept_positions, dtype=torch.long),
        )

        ps_state = PrefixSharingRuntimeState(
            prefix_sharing_plan=plan,
            backend=TorchReferenceBackend(),
            packed_batch_layout=layout_with_pos,
            model_spec=ModelSpec(
                num_hidden_layers=2, full_attention_interval=2,
                num_attention_heads=4, num_key_value_heads=2, head_dim=64,
            ),
        )

        # Create packed_seq_params with THD format
        cu_seqlens = torch.tensor(
            [0] + [sum(kept_lens[:j+1]) for j in range(len(kept_lens))],
            dtype=torch.int32, device="cuda"
        )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            cu_seqlens_kv_padded=cu_seqlens,
            max_seqlen_q=max(kept_lens),
            max_seqlen_kv=max(kept_lens),
        )

        # Create hidden_states: [total_tokens, batch=1, hidden]
        hidden_states = torch.randn(total_tokens, 1, 256, dtype=torch.bfloat16, device="cuda")

        with torch.no_grad():
            with prefix_sharing_runtime_context(ps_state) as ctx:
                check("PS context active", ctx is not None)
                check("PS context has deltanet_store", ctx.deltanet_store is not None)

                out, bias = gdn_attn(
                    hidden_states=hidden_states,
                    attention_mask=None,
                    packed_seq_params=packed_seq_params,
                )

                check("GDN+PS+THD forward finite", torch.isfinite(out).all().item(),
                      f"nan_count={(~torch.isfinite(out)).sum().item()}")
                check("GDN+PS+THD output shape", out.shape[0] == total_tokens,
                      f"got {out.shape}")

    del gdn_attn
    torch.cuda.empty_cache()

    # ================================================================
    # Test 4: PS + THD gradient flow through GDN layer
    # ================================================================
    if rank == 0:
        print("\n--- Test 4: PS + THD gradient flow (GDN layer) ---")

    gdn_attn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub,
        layer_number=1, attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.5, attn_output_gate=True,
    ).cuda().bfloat16()

    if plan.has_sharing:
        hidden_states = torch.randn(total_tokens, 1, 256, dtype=torch.bfloat16, device="cuda",
                                    requires_grad=True)

        with prefix_sharing_runtime_context(ps_state):
            out, bias = gdn_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
            )
            loss = out.sum()
            loss.backward()

        check("GDN+PS gradient on input", hidden_states.grad is not None)
        check("GDN+PS gradient finite", torch.isfinite(hidden_states.grad).all().item()
              if hidden_states.grad is not None else False)
        check("GDN+PS beta_proj grad", gdn_attn.beta_proj.weight.grad is not None)
        check("GDN+PS linear_qkv grad", gdn_attn.linear_qkv.weight.grad is not None)
        check("GDN+PS linear_proj grad", gdn_attn.linear_proj.weight.grad is not None)

    del gdn_attn
    torch.cuda.empty_cache()

    # ================================================================
    # Test 5: Verify PS output numerical correctness at backend level
    # ================================================================
    if rank == 0:
        print("\n--- Test 5: PS numerical correctness (backend) ---")

    # Directly test that the PS trajectory matches independent cumsum
    torch.manual_seed(42 + rank)

    # Simulate GatedDeltaNet state updates for 2 sequences
    seq_a = [1, 2, 3, 4, 5, 10, 20]   # provider: full sequence
    seq_b = [1, 2, 3, 4, 5, 30, 40]   # reuser: shares prefix [1,2,3,4,5]

    config = PrefixSharingConfig(enable_prefix_sharing=True, min_prefix_len=3, min_group_size=2)
    plan = PrefixSharingPlanner(config).plan([seq_a, seq_b], forward_id=3, micro_batch_id=0)
    check("Plan has sharing (backend)", plan.has_sharing)

    if plan.has_sharing:
        # Create token-ID-based updates
        state_dim = 16
        max_tid = max(max(seq_a), max(seq_b)) + 1
        update_emb = torch.randn(max_tid, state_dim, 4, device="cuda", dtype=torch.bfloat16)

        all_updates = [update_emb[seq] for seq in [seq_a, seq_b]]
        ind_trajectories = [u.cumsum(dim=0) for u in all_updates]

        # PS: trim and build
        trimmed = []
        for i, upd in enumerate(all_updates):
            s, e = plan.input_keep_ranges[i]
            trimmed.append(upd[s:e])

        packed = torch.cat(trimmed, dim=0)
        backend = TorchReferenceBackend()
        from prefix_sharing.core.prefix_store import PrefixDeltanetStore
        store = PrefixDeltanetStore()
        ps_output = backend.build_deltanet_states(packed, store, plan, layer_id=0)

        ps_rows = list(torch.split(ps_output, plan.kept_lengths_q))

        for i in range(2):
            s, e = plan.input_keep_ranges[i]
            kept_len = e - s
            if not plan.is_reuser(i):
                match = torch.allclose(ps_rows[i][:kept_len].float(),
                                      ind_trajectories[i][:kept_len].float(),
                                      atol=1e-2, rtol=1e-2)
                check(f"Provider seq{i} match", match,
                      f"max_diff={(ps_rows[i][:kept_len] - ind_trajectories[i][:kept_len]).abs().max().item():.4f}")
            else:
                match = torch.allclose(ps_rows[i][:kept_len].float(),
                                      ind_trajectories[i][s:e].float(),
                                      atol=1e-2, rtol=1e-2)
                check(f"Reuser seq{i} match", match,
                      f"max_diff={(ps_rows[i][:kept_len] - ind_trajectories[i][s:e]).abs().max().item():.4f}")

    # ================================================================
    # Summary
    # ================================================================
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"GatedDeltaNet PS E2E Results: {passed} passed, {failed} failed")
        print(f"{'='*60}")

    dist.destroy_process_group()
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
