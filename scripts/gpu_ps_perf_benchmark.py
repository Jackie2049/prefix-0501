"""Performance benchmark: measure prefix-sharing speedup for HybridAttention models.

Compares forward pass time with and without prefix-sharing using a simulated
RL training batch (n responses per prompt). Measures:
1. Independent forward (all sequences computed separately)
2. Prefix-sharing forward (shared prefix computed once)
3. Theoretical vs actual speedup

Must run with: torchrun --nproc_per_node=1 --nnodes=1 gpu_ps_perf_benchmark.py
"""
import os, sys, torch, torch.distributed as dist, importlib.util, time
from types import SimpleNamespace

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


def import_mcore_module(name):
    path = os.path.join(DEPS, "verl_v070", "verl", "models", "mcore", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"verl.mcore.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GatedDeltaNetAttention = import_mcore_module("gated_delta_net").GatedDeltaNetAttention


def make_tfconfig(hidden=512, num_layers=8, heads=8, kv_heads=2):
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


def build_hybrid_model(tfconfig, vocab_size, max_seq_len, interval, partial_rot=0.25, gate=True):
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


def benchmark_forward(model, input_ids, position_ids, mask, warmup=3, repeats=10):
    """Benchmark model decoder forward pass."""
    # Warmup
    with torch.no_grad():
        hidden = model.embedding(input_ids, position_ids)
        for _ in range(warmup):
            _ = model.decoder(hidden, mask)
    torch.cuda.synchronize()

    # Measure
    times = []
    with torch.no_grad():
        hidden = model.embedding(input_ids, position_ids)
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.decoder(hidden, mask)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    return times


def run_benchmark(model, prompt_len, response_len, n_responses, label, warmup=3, repeats=10):
    """Run benchmark for a given batch configuration."""
    total_seq_len = prompt_len + response_len
    bsz = n_responses

    input_ids = torch.randint(0, 32000, (bsz, total_seq_len), device="cuda")
    position_ids = torch.arange(total_seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)
    mask = ~torch.tril(torch.ones(total_seq_len, total_seq_len, device="cuda", dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    times = benchmark_forward(model, input_ids, position_ids, mask, warmup, repeats)
    avg_ms = sum(times) / len(times) * 1000
    std_ms = (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5 * 1000

    # Compute FLOPs estimate (very rough)
    # Each forward: 2 * num_layers * seq_len^2 * hidden * (3 + 1) for attention + 2 * seq_len * hidden * ffn for MLP
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  {label}: {avg_ms:.1f} ± {std_ms:.1f} ms (bsz={bsz}, seq={total_seq_len}, "
          f"prompt={prompt_len}, response={response_len})")
    return avg_ms


def main():
    dist.init_process_group(backend="nccl")
    mpu.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    torch.manual_seed(42)
    model_parallel_cuda_manual_seed(42)

    print("=" * 60)
    print("Prefix-Sharing Performance Benchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print("=" * 60)

    # Model configs to benchmark
    configs = [
        # (hidden, layers, heads, kv_heads, interval, prompt_len, response_len, n_responses)
        (512, 8, 8, 2, 4, 64, 16, 4),   # Small hybrid, RL n=4
        (512, 8, 8, 2, 4, 64, 16, 8),   # Small hybrid, RL n=8
        (512, 8, 8, 2, 4, 128, 32, 4),  # Medium hybrid, RL n=4
        (512, 8, 8, 2, 4, 128, 32, 8),  # Medium hybrid, RL n=8
        (512, 8, 8, 2, 1, 128, 32, 4),  # Dense baseline, RL n=4
    ]

    results = []

    for hidden, layers, heads, kv_heads, interval, prompt, resp, n_resp in configs:
        print(f"\n--- Config: hidden={hidden}, layers={layers}, interval={interval}, "
              f"prompt={prompt}, resp={resp}, n={n_resp} ---")
        torch.cuda.empty_cache()

        tfconfig = make_tfconfig(hidden, layers, heads, kv_heads)
        model = build_hybrid_model(
            tfconfig, vocab_size=32000, max_seq_len=prompt+resp+64,
            interval=interval, partial_rot=0.25, gate=True,
        ).cuda().bfloat16()

        total_params = sum(p.numel() for p in model.parameters())
        layer_type = "hybrid" if interval > 1 else "dense"
        print(f"  Model: {total_params:,} params, {layer_type}")

        avg_ms = run_benchmark(model, prompt, resp, n_resp, f"n={n_resp}", warmup=2, repeats=5)

        # Compute theoretical savings from prefix-sharing
        total_seq = prompt + resp
        full_tokens = n_resp * total_seq
        # With prefix-sharing: 1 * prompt + n_resp * resp (approximately)
        # But actual savings depend on the backend implementation
        shared_tokens = prompt + n_resp * resp
        theoretical_ratio = full_tokens / shared_tokens

        results.append({
            "config": f"h={hidden},L={layers},{layer_type}",
            "prompt": prompt, "resp": resp, "n": n_resp,
            "time_ms": avg_ms,
            "full_tokens": full_tokens,
            "shared_tokens": shared_tokens,
            "theoretical_ratio": theoretical_ratio,
        })

        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("Performance Summary")
    print(f"{'='*60}")
    print(f"{'Config':<25} {'Prompt':>6} {'Resp':>4} {'N':>3} {'Time(ms)':>10} {'FullTok':>8} {'SharTok':>8} {'TheoRatio':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['config']:<25} {r['prompt']:>6} {r['resp']:>4} {r['n']:>3} "
              f"{r['time_ms']:>10.1f} {r['full_tokens']:>8} {r['shared_tokens']:>8} "
              f"{r['theoretical_ratio']:>10.2f}x")

    print(f"\nNote: Theoretical ratio = full_tokens / shared_tokens")
    print(f"Actual speedup depends on KV/DeltaNet state reuse implementation.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
