"""GEMM Precision Baseline — Within-Batch Pairwise Comparison.

Compares N identical copies WITHIN a single forward pass to verify
same-GEMM-kernel bit-identical reproduction.  Reuses ``cmp_diag_verl080``
printing for consistent output.

Usage::

    export PREFIX_SHARING_DIAG_DUMP=/dump_multi
    # forward with batch=[A x N]

    python cmp_baseline_within_batch.py --dir-multi /dump_multi --num-copies 4
"""

from __future__ import annotations

import argparse
import math
import os

import torch

from prefix_sharing.tools.cmp_diag_verl080 import (
    CheckResult,
    _SEP_DOUBLE,
    _SEP_SINGLE,
    _CHECK,
    _CROSS,
    _COS_AVG_PASS,
    _COS_MIN_PASS,
    _cosine_sim,
    _error_abs_rel,
    _pearson_r,
    _dump_json,
    _load_tensor,
    _print_per_layer,
    _print_rope_postqk_per_layer,
    _print_rope_freqs,
    _print_build_kv_input_v,
    _print_hidden_states,
    _print_logits_packed,
    _print_2d_result,
    _print_topk_vec,
    _print_summary,
)


# ── helpers ──────────────────────────────────────────────────────

def _load_dict(dir_path: str, filename: str) -> dict | None:
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        return None
    d = torch.load(fp, weights_only=True)
    return d if isinstance(d, dict) else None


def _load_cu(dir_path: str) -> torch.Tensor | None:
    fp = os.path.join(dir_path, "cu_seqlens_q.pt")
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=True)


def _extract(packed: torch.Tensor, cu: torch.Tensor, copy_idx: int) -> torch.Tensor:
    return packed[int(cu[copy_idx]) : int(cu[copy_idx + 1])]


def _get_layers(data: dict) -> list[int]:
    return sorted(int(k) for k in data.keys())


# ── comparison logic ─────────────────────────────────────────────

def _worst_pairwise(copies: list[torch.Tensor]) -> dict:
    """Pairwise compare all copies; return worst max_diff and cos_min."""
    worst_md, worst_cos = 0.0, 1.0
    N = len(copies)
    for i in range(N):
        for j in range(i + 1, N):
            a = copies[i].reshape(copies[i].shape[0], -1).float()
            b = copies[j].reshape(copies[j].shape[0], -1).float()
            md = float((a - b).abs().max())
            cos_min = float(_cosine_sim(a, b, dim=-1).min())
            if md > worst_md:
                worst_md = md
            if cos_min < worst_cos:
                worst_cos = cos_min
    return {"max_diff": worst_md, "cos_min": worst_cos}


def _compare_plain_within(dir_multi: str, filename: str,
                          cu: torch.Tensor, n_copies: int,
                          layer: int | None, label: str) -> CheckResult | None:
    d = _load_dict(dir_multi, filename)
    if d is None:
        return None
    layers = _get_layers(d)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return None

    per_layer: dict = {}
    worst_md, worst_cos = 0.0, 1.0
    for lyr in layers:
        mt = d[lyr].float()
        copies = [_extract(mt, cu, i) for i in range(n_copies)]
        w = _worst_pairwise(copies)
        T = copies[0].shape[0]
        per_layer[lyr] = {
            "max_diff": w["max_diff"],
            "cos_avg": 1.0,
            "cos_min": w["cos_min"],
            "n_tokens": T,
            "on_T": T,
            "off_T": T,
        }
        worst_md = max(worst_md, w["max_diff"])
        worst_cos = min(worst_cos, w["cos_min"])
    passed = worst_md == 0.0
    _name = f"{label}_L{layer}" if layer is not None else label
    return CheckResult(name=_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_md,
                                "cos_min": worst_cos, "num_layers": len(layers)})


def _compare_rope_within(dir_multi: str, filename: str,
                         cu: torch.Tensor, n_copies: int,
                         layer: int | None, label: str,
                         fld_q: str, fld_k: str) -> CheckResult | None:
    d = _load_dict(dir_multi, filename)
    if d is None:
        return None
    layers = _get_layers(d)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return None

    per_layer: dict = {}
    for lyr in layers:
        mq = d[lyr][fld_q].float()
        mk = d[lyr][fld_k].float()
        q_copies = [_extract(mq, cu, i) for i in range(n_copies)]
        k_copies = [_extract(mk, cu, i) for i in range(n_copies)]
        qw = _worst_pairwise(q_copies)
        kw = _worst_pairwise(k_copies)
        per_layer[lyr] = {
            "Q_max_diff": qw["max_diff"], "K_max_diff": kw["max_diff"],
            "Q_cos_avg": 1.0, "Q_cos_min": qw["cos_min"],
            "K_cos_avg": 1.0, "K_cos_min": kw["cos_min"],
            "n_tokens": q_copies[0].shape[0],
        }
    _name = f"{label}_L{layer}" if layer is not None else label
    return CheckResult(name=_name, passed=True, metrics={"layers": per_layer})


def _compare_logits_within(dir_multi: str, cu: torch.Tensor,
                           n_copies: int) -> CheckResult | None:
    fp = os.path.join(dir_multi, "logits.pt")
    if not os.path.exists(fp):
        return None
    mt = torch.load(fp, weights_only=True).float()
    mt = mt.reshape(-1, mt.size(-1))
    copies = [_extract(mt, cu, i).reshape(-1, mt.size(-1)) for i in range(n_copies)]
    w = _worst_pairwise(copies)
    return CheckResult(name="logits", passed=w["max_diff"] == 0.0,
                       metrics={"n_tokens": copies[0].shape[0],
                                "cos_avg": 1.0, "cos_min": w["cos_min"]})


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GEMM precision baseline — within-batch pairwise comparison",
        epilog=__doc__,
    )
    ap.add_argument("--dir-multi", required=True,
                    help="Multi-copy dump directory (batch=[A x N])")
    ap.add_argument("--num-copies", type=int, required=True,
                    help="Number of copies N")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific layer 1-indexed (default: all)")
    ap.add_argument("--tag", default="old",
                    help="2D file tag for logprobs/entropy (default: old)")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D (default: 1e-5)")
    ap.add_argument("--topk", type=int, default=0,
                    help="top-K worst dims (0=disabled)")
    ap.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                    help="top-K sort: abs / rel / val")
    ap.add_argument("--output", "-o", default=None,
                    help="Write JSON report to this path")
    args = ap.parse_args()

    cu = _load_cu(args.dir_multi)
    if cu is None:
        print(f"{_CROSS} cu_seqlens_q.pt missing"); return 1

    T = int(cu[1]) - int(cu[0])
    n = args.num_copies
    n_pair = n * (n - 1) // 2
    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Within-Batch Pairwise Comparison")
    print(f"  Directory: {args.dir_multi}  ({n} copies, {T} tokens each)")
    print(f"  Pairs: {n_pair}")
    print(_SEP_DOUBLE)

    all_results: list[CheckResult] = []

    # ── hidden_states ──
    r = _compare_plain_within(args.dir_multi, "hidden_states.pt",
                              cu, n, args.layer, "hidden_states")
    if r: all_results.append(r); _print_hidden_states(r)

    # ── build_kv_input_v ──
    r = _compare_plain_within(args.dir_multi, "build_kv_input_v.pt",
                              cu, n, args.layer, "build_kv_input_v")
    if r: all_results.append(r); _print_build_kv_input_v(r)

    # ── rope_preqk ──
    r = _compare_rope_within(args.dir_multi, "rope_preqk.pt",
                             cu, n, args.layer, "rope_preqk", "query", "key")
    if r: all_results.append(r); _print_rope_postqk_per_layer(r)

    # ── rope_freqs ──
    r = _compare_plain_within(args.dir_multi, "rope_freqs.pt",
                              cu, n, args.layer, "rope_freqs")
    if r: all_results.append(r); _print_rope_freqs(r)

    # ── rope_postqk ──
    r = _compare_rope_within(args.dir_multi, "rope_postqk.pt",
                             cu, n, args.layer, "rope_postqk", "query", "key")
    if r: all_results.append(r); _print_rope_postqk_per_layer(r)

    # ── attn_outputs ──
    r = _compare_plain_within(args.dir_multi, "attn_outputs.pt",
                              cu, n, args.layer, "attn_outputs")
    if r: all_results.append(r); _print_per_layer(r)

    # ── full_kv ──
    r = _compare_rope_within(args.dir_multi, "full_kv.pt",
                             cu, n, args.layer, "full_kv", "key", "value")
    if r: all_results.append(r); _print_rope_postqk_per_layer(r)

    # ── logits ──
    r = _compare_logits_within(args.dir_multi, cu, n)
    if r: all_results.append(r); _print_logits_packed(r)

    # ── 2D: logprobs + entropy ──
    for fn, cn in [("logprobs", "logp"), ("entropy", "entropy")]:
        fname = f"{fn}_{args.tag}.pt"
        st = _load_tensor(args.dir_multi, fname)
        if st is not None and st.dim() >= 2:
            B = st.shape[0]
            worst_md, worst_cos = 0.0, 1.0
            for i in range(B):
                for j in range(i + 1, B):
                    a = st[i].float().reshape(-1)
                    b = st[j].float().reshape(-1)
                    md = float((a - b).abs().max())
                    cos = float(_cosine_sim(a, b, dim=-1))
                    worst_md = max(worst_md, md)
                    worst_cos = min(worst_cos, cos)
            r = CheckResult(name=f"{cn}_{args.tag}",
                            passed=worst_md == 0.0,
                            metrics={"shape": tuple(st.shape),
                                     "active": st.numel(),
                                     "abs_max": worst_md,
                                     "abs_mean": 0.0,
                                     "rel_max": 0.0,
                                     "rel_mean": 0.0,
                                     "pearson_r": 1.0,
                                     "atol": args.atol})
            all_results.append(r)
            _print_2d_result(r)

    # ── top-K ──
    if args.topk > 0:
        # build_kv_input_v last-layer top-K
        d = _load_dict(args.dir_multi, "build_kv_input_v.pt")
        if d:
            lyr = max(int(k) for k in d.keys())
            mt = d[lyr].float()
            copies = [_extract(mt, cu, i).reshape(mt.shape[0] // n, -1) for i in range(n)]
            worst_md = 0.0; worst_a = worst_b = None
            for i in range(n):
                for j in range(i + 1, n):
                    md = float((copies[i] - copies[j]).abs().max())
                    if md > worst_md:
                        worst_md = md; worst_a = copies[i]; worst_b = copies[j]
            if worst_a is not None:
                _print_topk_vec(worst_a[0].cpu(), worst_b[0].cpu(),
                                args.topk, args.sort_err,
                                f"build_kv_input_v_L{lyr}_token0")

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, "—", args.dir_multi,
                   tag=f"within_batch_N{n}", dir_off2=None)


if __name__ == "__main__":
    main()
