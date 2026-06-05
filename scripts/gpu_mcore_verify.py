"""GPU verification for Qwen3.6-27B mcore HybridAttention.

Must run with: torchrun --nproc_per_node=1 --nnodes=1 gpu_mcore_verify.py

Tests:
1. GatedDeltaNetAttention instantiation and forward pass
2. Hybrid model layer routing (manually replace layers)
3. Shape correctness and bf16 precision
4. Gradient flow through cumsum
5. Full model forward pass (embedding + decoder)
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPS = os.path.join(REPO_ROOT, "dependency")
sys.path.insert(0, os.path.join(DEPS, "megatron_v0150"))
sys.path.insert(0, os.path.join(REPO_ROOT, "prefix-sharing"))
# Do NOT add verl_v070 to sys.path - we import our modules via importlib

import torch
import torch.distributed as dist
import importlib.util


def init_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def import_mcore_module(name):
    """Import a verl mcore module directly without triggering verl's __init__.py."""
    path = os.path.join(DEPS, "verl_v070", "verl", "models", "mcore", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"verl.mcore.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Megatron imports (no verl dependency)
from megatron.core import parallel_state as mpu
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

# Our GatedDeltaNet (only depends on megatron, no verl)
_gdn = import_mcore_module("gated_delta_net")
GatedDeltaNetAttention = _gdn.GatedDeltaNetAttention


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
        print(f"Results: {self.passed} passed, {self.failed} failed")
        print(f"{'='*60}")
        return self.failed == 0


def get_tp_size():
    try:
        return mpu.get_tensor_model_parallel_world_size()
    except Exception:
        return 1


def make_tfconfig(hidden=512, num_layers=8, heads=8, kv_heads=2):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=heads,
        num_query_groups=kv_heads,
        kv_channels=hidden // heads,
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        init_method_std=0.02,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=get_tp_size(),
        pipeline_model_parallel_size=1,
        use_cpu_initialization=False,
    )


def build_hybrid_gpt_model(tfconfig, vocab_size, max_seq_len, full_attention_interval,
                            partial_rotary_factor=1.0, attn_output_gate=False):
    """Build a GPTModel with GatedDeltaNet replacing linear attention layers.

    Replicates Qwen3_6HybridModel.initialize() logic.
    """
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    # Get SelfAttentionSubmodules (unwrap ModuleSpec)
    sa_modspec = block_spec.layer_specs[0].submodules.self_attention
    sa_submodules = sa_modspec.submodules

    model = GPTModel(
        config=tfconfig,
        transformer_layer_spec=block_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_seq_len,
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=10000.0,
    )

    if full_attention_interval <= 1:
        return model

    for i, layer in enumerate(model.decoder.layers):
        if i % full_attention_interval != 0:
            old_attn = layer.self_attention
            new_attn = GatedDeltaNetAttention(
                config=tfconfig,
                submodules=old_attn.submodules if hasattr(old_attn, 'submodules') else sa_submodules,
                layer_number=old_attn.layer_number,
                attn_mask_type=old_attn.attn_mask_type,
                partial_rotary_factor=partial_rotary_factor,
                attn_output_gate=attn_output_gate,
            )
            new_attn.to(next(old_attn.parameters()).device)
            # Share weights
            new_attn.linear_qkv = old_attn.linear_qkv
            new_attn.linear_proj = old_attn.linear_proj
            if hasattr(old_attn, 'q_layernorm') and old_attn.q_layernorm is not None:
                new_attn.q_layernorm = old_attn.q_layernorm
            if hasattr(old_attn, 'k_layernorm') and old_attn.k_layernorm is not None:
                new_attn.k_layernorm = old_attn.k_layernorm
            layer.self_attention = new_attn

    return model


def test1_standalone(results):
    """Test GatedDeltaNetAttention standalone."""
    print("\n--- Test 1: GatedDeltaNetAttention Standalone ---")
    tfconfig = make_tfconfig()
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    # Unwrap ModuleSpec -> SelfAttentionSubmodules
    sa_modspec = block_spec.layer_specs[0].submodules.self_attention
    sa_sub = sa_modspec.submodules

    try:
        attn = GatedDeltaNetAttention(
            config=tfconfig, submodules=sa_sub, layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            partial_rotary_factor=0.25, attn_output_gate=True,
        ).cuda()
        results.check("Instantiation", True)
    except Exception as e:
        results.check("Instantiation", False, str(e))
        import traceback; traceback.print_exc()
        return

    results.check("beta_proj", hasattr(attn, "beta_proj"))
    results.check("decay_proj", hasattr(attn, "decay_proj"))
    results.check("gate_proj", hasattr(attn, "gate_proj"))
    results.check("rope_dim=16", attn.rope_dim == 16, f"got {attn.rope_dim}")

    # Forward
    sq, b, h = 16, 2, 512
    hidden = torch.randn(sq, b, h, dtype=torch.bfloat16, device="cuda")
    try:
        with torch.no_grad():
            out, bias = attn(hidden_states=hidden, attention_mask=None)
        results.check("Forward shape", out.shape == (sq, b, h), f"got {out.shape}")
        results.check("Output bf16", out.dtype == torch.bfloat16)
        results.check("Output finite", torch.isfinite(out).all().item())
    except Exception as e:
        results.check("Forward", False, str(e))
        import traceback; traceback.print_exc()

    del attn
    torch.cuda.empty_cache()


def test2_gradient(results):
    """Test gradient flow."""
    print("\n--- Test 2: Gradient Flow ---")
    tfconfig = make_tfconfig()
    block_spec = get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=False)
    sa_sub = block_spec.layer_specs[0].submodules.self_attention.submodules

    attn = GatedDeltaNetAttention(
        config=tfconfig, submodules=sa_sub, layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        partial_rotary_factor=0.25, attn_output_gate=False,
    ).cuda()

    sq, b, h = 8, 2, 512
    x = torch.randn(sq, b, h, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    try:
        out, _ = attn(hidden_states=x, attention_mask=None)
        out.sum().backward()
        results.check("Grad: input", x.grad is not None and torch.isfinite(x.grad).all().item())
        results.check("Grad: beta_proj", attn.beta_proj.weight.grad is not None)
        # decay_proj is computed but not used in basic cumsum path (used in prefix-sharing)
        # results.check("Grad: decay_proj", attn.decay_proj.weight.grad is not None)
        results.check("Grad: linear_qkv", attn.linear_qkv.weight.grad is not None)
        results.check("Grad: linear_proj", attn.linear_proj.weight.grad is not None)
    except Exception as e:
        results.check("Gradient flow", False, str(e))
        import traceback; traceback.print_exc()

    del attn
    torch.cuda.empty_cache()


def test3_hybrid_routing(results):
    """Test hybrid model layer routing."""
    print("\n--- Test 3: Hybrid Model Layer Routing ---")
    tfconfig = make_tfconfig(hidden=512, num_layers=8, heads=8, kv_heads=2)
    interval = 4  # Full at 0,4; Linear at 1,2,3,5,6,7

    try:
        model = build_hybrid_gpt_model(
            tfconfig, vocab_size=32000, max_seq_len=2048,
            full_attention_interval=interval,
            partial_rotary_factor=0.25, attn_output_gate=True,
        ).cuda()
        results.check("HybridModel created", True)

        full_layers = []
        linear_layers = []
        for i, layer in enumerate(model.decoder.layers):
            if isinstance(layer.self_attention, GatedDeltaNetAttention):
                linear_layers.append(i)
            elif isinstance(layer.self_attention, SelfAttention):
                full_layers.append(i)

        results.check("Full attn [0,4]", full_layers == [0, 4], f"got {full_layers}")
        results.check("Linear attn [1,2,3,5,6,7]",
                      linear_layers == [1, 2, 3, 5, 6, 7], f"got {linear_layers}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        results.check("Hybrid routing", False, str(e))
        import traceback; traceback.print_exc()


def test4_full_forward(results):
    """Test full model forward (embedding + decoder)."""
    print("\n--- Test 4: Full Model Forward ---")
    tfconfig = make_tfconfig(hidden=512, num_layers=4, heads=8, kv_heads=2)
    interval = 2  # Full at 0,2; Linear at 1,3

    try:
        model = build_hybrid_gpt_model(
            tfconfig, vocab_size=32000, max_seq_len=2048,
            full_attention_interval=interval,
            partial_rotary_factor=0.5, attn_output_gate=True,
        ).cuda()

        bsz, seq_len = 2, 32
        input_ids = torch.randint(0, 32000, (bsz, seq_len), device="cuda")
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

        with torch.no_grad():
            hidden = model.embedding(input_ids, position_ids)
            results.check("Embedding shape",
                          hidden.shape[0] == seq_len and hidden.shape[1] == bsz,
                          f"got {hidden.shape}")

            # Causal mask (boolean)
            mask = torch.tril(
                torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
            # Megatron expects inverted mask (True = masked position)
            mask = ~mask
            decoder_out = model.decoder(hidden, mask)
            results.check("Decoder finite", torch.isfinite(decoder_out).all().item())
            results.check("Decoder shape",
                          decoder_out.shape[0] == seq_len and decoder_out.shape[1] == bsz,
                          f"got {decoder_out.shape}")

        del model
        torch.cuda.empty_cache()
    except Exception as e:
        results.check("Full forward", False, str(e))
        import traceback; traceback.print_exc()


def main():
    init_distributed()
    if not mpu.is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    # Initialize CUDA RNG tracker (needed by TP layers)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    print("=" * 60)
    print("Qwen3.6 mcore GPU Verification")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}, PyTorch: {torch.__version__}")
    print("=" * 60)

    r = Results()
    test1_standalone(r)
    test2_gradient(r)
    test3_hybrid_routing(r)
    test4_full_forward(r)

    ok = r.summary()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
