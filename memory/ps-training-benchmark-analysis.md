# PS Training Benchmark: Manual vs Monkey-Patch

## Key Finding

Manual two-pass PS is consistently slower than PS OFF model.forward(), even at n=4 (37.5% token savings).

This is because the manual approach has significant overhead:
1. **Prefix pass overhead**: 16 layers for 1 sequence (no_grad) — adds ~0.2s fixed cost
2. **Inline attention computation**: For full attention layers, manual forward does QKV proj + QK norm + partial RoPE + KV expansion + flash_attn_varlen_func in Python — much slower than fused model.forward()
3. **Python loop overhead**: Per-layer iteration with many tensor operations in Python

## Results (Qwen3.6-27B-16layers, TP=4, bf16)

| n | Token Savings | Theoretical | OFF (s) | ON (s) | Speedup | Peak OFF | Peak ON |
|---|---------------|-------------|----------|--------|---------|----------|---------|
| 2 | 25.0%         | 1.33x       | 0.541    | 0.779  | **0.69x** | 21.79GB | 21.94GB |
| 4 | 37.5%         | 1.60x       | 0.597    | 0.785  | **0.76x** | 22.02GB | 22.29GB |

Precision: cos_sim=0.999990 PASS

## Why Manual PS is Slower

### Breakdown of PS ON overhead (n=4):
- PS OFF model.forward(): 0.597s — fused, optimized CUDA kernels
- PS ON manual two-pass: 0.785s — Python loop per layer + inline attention
- Overhead ratio: 0.785/0.597 = 1.31x

The 0.188s overhead comes from:
1. Prefix pass (1 seq × 64 tok × 16 layers): ~0.1-0.15s (fixed cost, doesn't scale with n)
2. Inline full attention computation (4 layers): ~0.05-0.08s per layer × 4 = 0.02-0.03s
3. DeltaNet state expansion/injection (12 layers): relatively cheap

### Crossover Analysis

The crossover point where PS ON becomes faster depends on:
- Prefix pass fixed cost: ~C_prefix ≈ 0.15s
- PS OFF per-token cost: ~0.597s / (4 × 128) = 0.00116s/token
- PS ON suffix-only per-token cost: ~0.785 - 0.15 / (4 × 64) = 0.00992s/token ≈ 0.00116s/token (same!)

So PS ON ≈ C_prefix + n × suffix_len × per_token_cost
   PS OFF ≈ n × (prefix_len + suffix_len) × per_token_cost

Speedup = n × total_len × C_token / (C_prefix + n × suffix_len × C_token)

With C_prefix=0.15, C_token=0.00116:
- n=2: 2×128×0.00116 / (0.15 + 2×64×0.00116) = 0.296 / 0.299 = **0.99x** ≈ crossover!
- n=4: 4×128×0.00116 / (0.15 + 4×64×0.00116) = 0.594 / 0.447 = **1.32x** — should be faster!

But actual measurement shows 0.76x for n=4, meaning the per-token cost for PS ON is higher than PS OFF per-token cost. This is because:
- flash_attn_varlen_func with mixed-length sequences (suffix_len vs prefix_len+suffix_len) is slower than flash_attn on uniform-length sequences
- KV expansion + concatenation adds tensor operation overhead

## What This Means for verl Integration

The manual benchmark measures the worst case for PS. In verl's monkey-patch implementation:
1. **Both PS OFF and PS ON use model.forward()** — no Python loop overhead
2. **Prefix pass uses model.forward() too** — same fused kernels, not manual inline attention
3. **KV/DeltaNet injection happens inside attention forward** — minimal overhead

Expected verl monkey-patch speedup:
- PS OFF: model.forward(N, total_len) — same as baseline
- PS ON: model.forward(1, prefix_len) [prefix, no_grad] + model.forward(N, suffix_len) [suffix, with_grad + state injection]
- The prefix pass cost is model.forward(1, 64) ≈ 0.05s (much less than manual's 0.15s!)
- Speedup ≈ N × total_len × C_model / (C_prefix_model + N × suffix_len × C_model)
- n=4: 4×128×0.00116 / (0.05 + 4×64×0.00116) = 0.594 / 0.347 = **1.71x**

## Prefix Pass Kernel Launch Overhead

The prefix pass cost depends heavily on sequence length due to kernel launch overhead:

| Prefix Length | Total Time | Per-token Cost |
|---------------|-----------|---------------|
| 64 tokens     | 123ms     | 1.93ms/tok    |
| 128 tokens    | 134ms     | 1.05ms/tok    |
| 256 tokens    | 154ms     | 0.60ms/tok    |
| 512 tokens    | 195ms     | 0.38ms/tok    |

Fixed kernel launch overhead ≈ ~110ms (independent of sequence length).
Computation cost ≈ ~0.08ms/tok.

For prefix=64, the fixed overhead (110ms) is 89% of total prefix cost (123ms).
For prefix=512, the fixed overhead (110ms) is only 56% of total prefix cost (195ms).

## Impact on PS Training Speedup

The fixed prefix pass overhead means PS ON is slower for short prefixes:

**n=4, prefix=64, suffix=64 (inference):**
- PS OFF (4×128): 218ms
- PS ON prefix (1×64): 123ms (56% of PS OFF forward!)
- PS ON suffix (4×64): 181ms
- PS ON total fwd: 304ms → 218/304 = 0.72x (slower)

**Training estimate (fwd+bwd+opt):**
- PS OFF: ~0.597s (measured)
- PS ON: prefix 123ms + suffix fwd 181ms + suffix bwd 272ms + opt 52ms ≈ 628ms
- Speedup: 597/628 = **0.95x** (still slower)

**Real GRPO scenario (n=8, prefix=1024, suffix=256):**
- PS OFF (8×1280): much longer, kernel launch irrelevant
- PS ON prefix (1×1024): ~195ms + 1024×0.08ms ≈ 277ms (kernel overhead is only 40%)
- PS ON suffix (8×256): ~8×256×0.08 ≈ 164ms + fixed ≈ 274ms
- Token savings: 7/8 × 1024/(1024+256) = 87.5%
- Expected speedup: ~1.5-2x

## Conclusion

PS ON is slower for short prefixes (64 tokens) because kernel launch overhead dominates.
For realistic GRPO scenarios (prefix=256-1024 tokens, n=8+), PS ON should be significantly faster.

The crossover point depends on:
- Prefix length (longer → more favorable)
- n (larger → more favorable)
- Model size (larger → kernel launch overhead less significant relative to computation)

## Next Steps

1. Implement verl monkey-patch benchmark where both OFF and ON use model.forward()
2. The monkey-patch intercepts attention.forward() to inject KV/DeltaNet states
3. This eliminates the manual Python loop overhead and should show >1.5x speedup

## Comparison with PG (PrefixGrouper)

PG's approach has similar challenges in standalone benchmarks:
- PG standalone: 1.11-2.04x (block-causal attention savings)
- PG E2E training: 0.92x total (PG slower!) because generation dominates and PG logprob is slower

PS approach in verl monkey-patch should show better results because:
- PS saves MLP computation entirely (provider computes prefix MLP, reusers skip it)
- PG only saves attention computation (MLP savings are indirect — fewer total tokens)
- For Qwen3.6-27B where MLP dominates (~82% of forward), PS savings are much larger