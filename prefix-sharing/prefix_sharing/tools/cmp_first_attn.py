"""Compare first-layer attention outputs between ON and OFF dumps.

Usage:
    python cmp_first_attn.py --dir-on ./attn_dump_on --dir-off ./attn_dump_off
"""

from __future__ import annotations

import argparse
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

    total_diff = 0
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

        a = on_out[on_start : on_start + off_len_on, 0, :]
        b = off_out[off_start : off_start + off_len_on, 0, :]

        diff = (a - b).abs()
        row_max = diff.max().item()
        tokens_max_diff, _ = diff.max(dim=-1)
        row_mismatch = (~torch.isclose(
            a, b, rtol=args.rtol, atol=args.atol
        )).any(dim=-1)
        n_mismatch = row_mismatch.sum().item()

        if n_mismatch > 0:
            fail_rows += 1
            global_max_diff = max(global_max_diff, row_max)
            print(
                f"row[{i}] prefix_len={pf}: "
                f"{n_mismatch}/{off_len_on} tokens MISMATCH "
                f"(max_diff={row_max:.6e})"
            )
            topk = tokens_max_diff.topk(min(5, n_mismatch))
            for k, idx in enumerate(topk.indices.tolist()):
                abs_pos = pf + idx  # absolute position in original sequence
                print(
                    f"  #{k+1} abs_pos={abs_pos} "
                    f"(suffix_idx={idx}): diff={topk.values[k].item():.6e}"
                )
        else:
            ok_rows += 1
            print(
                f"row[{i}] prefix_len={pf}: "
                f"ALL MATCH ({off_len_on} tokens)"
            )

        total_diff += n_mismatch
        total_tokens += off_len_on

    print()
    print("=" * 60)
    print(f"ROWS: {ok_rows} OK, {fail_rows} FAIL")
    print(f"TOKENS: {total_diff}/{total_tokens} differ")
    print(f"MAX_ABS_DIFF: {global_max_diff:.6e}")
    if total_diff == 0:
        print("PASS: first attention layer output is IDENTICAL")
        print("  => bug is after attention (logprob/restore/postprocess)")
    else:
        print("FAIL: first attention layer output DIFFERS")
        print("  => bug is inside attention (KV injection / RoPE / layout)")


if __name__ == "__main__":
    main()
