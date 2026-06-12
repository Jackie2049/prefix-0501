"""Compare first-layer attention outputs between ON and OFF dumps.

Usage:
    python cmp_first_attn.py --dir-on ./attn_dump_on --dir-off ./attn_dump_off

Diagnostics:
    1. Cosine similarity:  ~1.0 = precision noise,  << 1.0 = computation error
    2. Relative error:      abs_diff / abs(b) per hidden dim
    3. Dim bucket counts:   how many hidden dims exceed 1e-3, 1e-2, etc.
"""

from __future__ import annotations

import argparse
import math
import os

import torch


def load_dump(dir_path: str) -> dict:
    return {
        "output": torch.load(
            os.path.join(dir_path, "first_attn_output.pt"), weights_only=True
        ),
        "cu_seqlens": torch.load(
            os.path.join(dir_path, "cu_seqlens_q.pt"), weights_only=True
        ),
        "prefix_lens": torch.load(
            os.path.join(dir_path, "prefix_lens.pt"), weights_only=True
        ),
    }


def _diagnose(a: torch.Tensor, b: torch.Tensor, prefix_len: int,
              max_tokens: int = 5) -> None:
    """Print per-token diagnostics: cosine, relative error, dim buckets."""
    n_tokens, hidden = a.shape
    n_show = min(n_tokens, max_tokens)

    # --- per-token diagnostics for top-N worst tokens ---
    diff_abs = (a - b).abs()                                   # [N, H]
    token_max_diff, _ = diff_abs.max(dim=-1)                    # [N]
    worst_indices = token_max_diff.topk(min(n_show, n_tokens)).indices

    eps = 1e-8
    a_norm = a.norm(dim=-1) + eps                              # [N]
    b_norm = b.norm(dim=-1) + eps                              # [N]
    dot = (a * b).sum(dim=-1)                                  # [N]
    cos_sim = dot / (a_norm * b_norm)                           # [N]

    # Relative error per dim: |a-b| / max(|b|, eps)
    rel_err = diff_abs / (b.abs() + eps)                        # [N, H]

    # Dim bucket counts: fraction of hidden dims exceeding thresholds
    thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    dim_buckets = {}
    for t in thresholds:
        dim_buckets[t] = (diff_abs > t).sum(dim=-1).float()    # [N]

    # Aggregate over all tokens
    all_cos = cos_sim.tolist()
    avg_cos = sum(all_cos) / len(all_cos)
    min_cos = min(all_cos)

    all_max_diff = token_max_diff.tolist()
    avg_max_diff = sum(all_max_diff) / len(all_max_diff)

    # Mean relative error per token
    all_mean_rel = rel_err.mean(dim=-1).tolist()
    avg_mean_rel = sum(all_mean_rel) / len(all_mean_rel)

    # Overall dim bucket fractions
    total_dims = n_tokens * hidden
    print(f"  n_tokens={n_tokens} hidden={hidden}")
    print(f"  cosine_sim:    avg={avg_cos:.6f}  min={min_cos:.6f}")
    print(f"  max_abs_diff:   avg={avg_max_diff:.6e}  max={max(token_max_diff.tolist()):.6e}")
    print(f"  mean_rel_err:   avg={avg_mean_rel:.6e}")
    print(f"  dim_frac >1e-5: {(dim_buckets[1e-5].sum().item())/total_dims:.4f}",
          f">1e-4: {(dim_buckets[1e-4].sum().item())/total_dims:.4f}",
          f">1e-3: {(dim_buckets[1e-3].sum().item())/total_dims:.4f}",
          f">1e-2: {(dim_buckets[1e-2].sum().item())/total_dims:.4f}",
          f">1e-1: {(dim_buckets[1e-1].sum().item())/total_dims:.4f}")

    # Show worst token details
    for rank, idx in enumerate(worst_indices.tolist()):
        abs_pos = prefix_len + idx
        cos = cos_sim[idx].item()
        max_d = token_max_diff[idx].item()
        mean_rel = rel_err[idx].mean().item()
        # dim bucket for this token
        bucket_str = " ".join(
            f">{t}={dim_buckets[t][idx].item():.0f}/{hidden}"
            for t in thresholds[:4]
        )
        print(f"  worst#{rank+1} abs_pos={abs_pos} cos={cos:.6f} "
              f"max_diff={max_d:.4e} mean_rel={mean_rel:.4e}  [{bucket_str}]")

    # Verdict
    if avg_cos > 0.9999 and min_cos > 0.999:
        print(f"  VERDICT: PRECISION NOISE (cosine ~1, likely bf16/fp32 rounding)")
    elif avg_cos > 0.99:
        print(f"  VERDICT: SUSPICIOUS (small systematic deviation)")
    else:
        print(f"  VERDICT: COMPUTATION ERROR (fundamentally wrong output)")


def main():
    ap = argparse.ArgumentParser(
        description="Compare first-layer attention outputs ON vs OFF"
    )
    ap.add_argument("--dir-on", required=True, help="ON dump directory")
    ap.add_argument("--dir-off", required=True, help="OFF dump directory")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-5)
    args = ap.parse_args()

    on = load_dump(args.dir_on)
    off = load_dump(args.dir_off)

    on_out = on["output"].float()    # [packed_on, 1, hidden]
    off_out = off["output"].float()  # [packed_off, 1, hidden]
    cu_on = on["cu_seqlens"]         # [batch+1]
    cu_off = off["cu_seqlens"]       # [batch+1]
    pl = on["prefix_lens"]           # [batch]

    hidden = on_out.shape[-1]
    batch = cu_on.shape[0] - 1

    total_tokens = 0
    global_max_diff = 0.0
    ok_rows = 0
    fail_rows = 0

    for i in range(batch):
        pf = int(pl[i])
        if pf <= 0:
            continue  # provider or standalone, skip

        # ON: suffix only, starts at cu_on[i]
        on_start = int(cu_on[i])
        on_len = int(cu_on[i + 1] - cu_on[i])
        # OFF: full sequence, suffix starts at cu_off[i] + pf
        off_start = int(cu_off[i]) + pf
        off_len_on = on_len  # same number of suffix tokens

        if off_start + off_len_on > off_out.shape[0]:
            print(f"[WARN] row[{i}] prefix_len={pf}: OFF out of bounds, skip")
            continue
        if off_len_on == 0:
            print(f"[WARN] row[{i}] prefix_len={pf}: zero suffix tokens, skip")
            continue

        a = on_out[on_start : on_start + off_len_on, 0, :]
        b = off_out[off_start : off_start + off_len_on, 0, :]

        diff_abs = (a - b).abs()
        row_max = diff_abs.max().item()
        tokens_max_diff, _ = diff_abs.max(dim=-1)
        row_mismatch = (~torch.isclose(
            a, b, rtol=args.rtol, atol=args.atol
        )).any(dim=-1)
        n_mismatch = row_mismatch.sum().item()

        if n_mismatch > 0:
            fail_rows += 1
            global_max_diff = max(global_max_diff, row_max)
            print(
                f"row[{i}] prefix_len={pf}: "
                f"{n_mismatch}/{off_len_on} tokens exceed atol={args.atol}"
                f" (max_abs_diff={row_max:.6e})"
            )
            _diagnose(a, b, pf, max_tokens=3)
        else:
            ok_rows += 1
            print(
                f"row[{i}] prefix_len={pf}: "
                f"ALL MATCH ({off_len_on} tokens)"
            )

        total_tokens += off_len_on

    print()
    print("=" * 60)
    print(f"ROWS: {ok_rows} OK, {fail_rows} FAIL")
    print(f"TOTAL_SUFFIX_TOKENS: {total_tokens}")
    print(f"MAX_ABS_DIFF: {global_max_diff:.6e}")
    if fail_rows == 0:
        print("PASS: first attention layer output is IDENTICAL")
        print("  => bug is after attention (logprob/restore/postprocess)")
    else:
        print("FAIL: first attention layer output DIFFERS")
        print("  => bug is inside attention (KV injection / RoPE / layout)")


if __name__ == "__main__":
    main()
