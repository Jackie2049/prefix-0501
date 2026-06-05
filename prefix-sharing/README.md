# Prefix Sharing: RL Training Acceleration via Batch-Internal KV Reuse

Prefix sharing accelerates RL training (GRPO/PPO) by reusing common prefix KV cache across sequences within a micro-batch in Megatron's attention layers.

## How It Works

In RL training with GRPO (n=8), multiple responses share the same prompt. Instead of computing attention independently for each sequence, prefix sharing:

1. **Detects** shared prefixes within the batch using a Trie-based prefix detector
2. **Trims** reuser sequences to keep only their unique suffix
3. **Injects** the provider's prefix KV cache during attention computation
4. **Restores** logprobs at the prefix-suffix boundary for training correctness

```
Before:  [prompt | resp_1] [prompt | resp_2] ... [prompt | resp_8]  → 8× full attention
After:   [prompt | resp_1] [resp_2] [resp_3] ... [resp_8]           → 1× full + 7× suffix-only
```

## Performance

Measured on RTX 4090 (bf16, GQA 24:4):

| Config | Tokens Saved | SDPA Speedup | Flash Attention Speedup |
|---|---|---|---|
| GRPO-8x2048x128-Qwen36 | 82.4% | **11.85x** | **6.91x** |
| GRPO-16x1024x128-Qwen36 | 83.3% | **6.48x** | **3.60x** |
| GRPO-8x2048x256-Qwen36 | 77.8% | **5.08x** | **7.31x** |
| GRPO-8x1024x128-Qwen36 | 77.8% | **3.96x** | **1.81x** |

## Supported Models

- **Qwen3.6-27B**: GQA 24:4, head_dim=256, HybridAttention (64 layers: 16 full + 48 linear attention)
- Any model with shared prefixes in the batch (common in RL workloads)

## Supported Parallel Strategies

- **TP=1, 2, 4, 8**: All validated on RTX 4090
- **PP=1**: Pipeline parallelism not supported in phase 1
- **CP=1**: Context parallelism not supported in phase 1

## Installation

```bash
pip install -e .
```

Optional backends:
- **torch_ref**: Uses PyTorch SDPA (default, works on CPU and CUDA)
- **flash_atten_gpu**: Uses Flash Attention 2 (requires `flash-attn`)

## Quick Start with verl + Megatron

### 1. Enable in verl config

```yaml
# actor config
actor:
  prefix_sharing_config:
    enable_prefix_sharing: true
    min_prefix_len: 2
    min_group_size: 2
  megatron:
    use_remove_padding: true  # Required for prefix sharing
```

### 2. Or enable via environment variable

```bash
export ENABLE_PREFIX_SHARING=true
```

### 3. Verify setup

```python
import prefix_sharing
status = prefix_sharing.diagnose()
for k, v in status.items():
    print(f"  {k}: {v}")
```

### 4. Programmatic API

```python
from prefix_sharing import enable_prefix_sharing, PrefixSharingConfig

config = PrefixSharingConfig(enable_prefix_sharing=True)
handle = enable_prefix_sharing(config, model_config=model_config)
# ... training ...
handle.disable()  # or use prefix_sharing_enabled() context manager
```

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `enable_prefix_sharing` | `false` | Enable prefix sharing |
| `min_prefix_len` | `2` | Minimum prefix length to consider for sharing |
| `min_group_size` | `2` | Minimum group size for sharing |
| `backend` | `"auto"` | Backend: `"auto"`, `"torch_ref"`, `"flash_atten_gpu"` |
| `detector` | `"trie"` | Prefix detector algorithm (only `"trie"` in phase 1) |

Environment variable `ENABLE_PREFIX_SHARING` can override the config (accepts: `1`, `true`, `yes`, `on`, `y`).

## Phase 1 Constraints

- Text-only actor (no multi-modal)
- `use_remove_padding=True` (THD format)
- No pipeline parallelism (`pipeline_model_parallel_size=1`)
- No context parallelism (`context_parallel_size=1`)
- No RoPE fusion (`apply_rope_fusion=False`)
- No fused QKV RoPE (`fused_single_qkv_rope=False`)
- No fused kernels (`use_fused_kernels=False`)

## Architecture

```
prefix_sharing/
├── core/                    # Framework-independent
│   ├── config.py            # Config validation
│   ├── prefix_detector.py   # Trie-based prefix detection
│   ├── planner.py           # Plan generation (THD layout)
│   ├── prefix_store.py      # KV/DeltaNet state store
│   ├── model_spec.py        # Model architecture spec (HybridAttention)
│   ├── logprob.py           # Logprob restore
│   └── batch_trim.py        # Input/label/mask trimming
├── backends/                # Attention execution
│   ├── torch_ref.py         # PyTorch SDPA reference
│   ├── flash_atten_gpu.py   # Flash Attention 2
│   ├── flash_atten_base.py  # FlashAttention mixin
│   └── packed_layout.py     # THD packed layout
└── integrations/            # Framework adapters
    ├── verl_mcore.py        # verl+Megatron batch integration
    ├── megatron_attention.py # Megatron attention monkey-patch
    ├── megatron_runtime.py  # Runtime hook (RoPE, KV, attention)
    ├── context.py           # ContextVar runtime state
    └── patch_manager.py     # Monkey-patch lifecycle
```

## Testing

```bash
# Unit tests (no GPU required)
python -m pytest tests/unit_test/ -v

# Full test suite (requires CUDA + flash-attn for some tests)
python -m pytest tests/ -v

# CUDA test runner (standalone, no pytest)
python tests/run_all_cuda_tests.py

# Benchmarks
PYTHONPATH=. python tests/benchmark/bench_rl_workload.py --device cuda
PYTHONPATH=. python tests/benchmark/bench_qwen36_realistic.py
```

## Backend Selection

- **SDPA (torch_ref)**: Better for small-medium sequences (lower overhead). Recommended for sequences < 2048.
- **Flash Attention 2 (flash_atten_gpu)**: Better for large sequences with head_dim=256. Recommended for Qwen3.6-27B with long prompts.
- **auto**: Uses flash-attn if available, falls back to SDPA.
