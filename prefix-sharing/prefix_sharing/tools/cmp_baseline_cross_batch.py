"""GEMM Precision Baseline — Cross-Batch-Size Comparison.

Compares the SAME data processed at DIFFERENT batch sizes to quantify
GEMM floating-point noise.  Reuses comparison metrics and printing
functions from ``cmp_diag_verl080`` for consistent output.

Usage::

    # Run 1: single copy
    export PREFIX_SHARING_DIAG_DUMP=/dump_single
    # forward with batch=[A]

    # Run 2: stacked copies
    export PREFIX_SHARING_DIAG_DUMP=/dump_stacked
    # forward with batch=[A x N]

    python cmp_baseline_cross_batch.py \
        --dir-single /dump_single --dir-stacked /dump_stacked --num-copies 4
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
    _SEP_THIN,
    _CHECK,
    _CROSS,
    _COS_AVG_PASS,
    _COS_MIN_PASS,
    _cosine_sim,
    _error_abs_rel,
    _pearson_r,
    _dump_json,
    _load_tensor,
    _print_header,
    _print_per_layer,
    _print_rope_postqk_per_layer,
    _print_rope_freqs,
    _print_build_kv_input_v,
    _print_hidden_states,
    _print_logits_packed,
    _print_2d_result,
    _print_topk_vec,
    _print_topk_2d,
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

def _compare_plain(dir_single: str, dir_stacked: str, filename: str,
                   cu_s: torch.Tensor, cu_m: torch.Tensor,
                   n_copies: int, layer: int | None,
                   label: str) -> CheckResult | None:
    """Compare ``{layer: [T, ...]}`` per-layer dicts across copies.

    Handles variable-length sequences by matching copies by sequence length.
    """
    sd = _load_dict(dir_single, filename)
    md = _load_dict(dir_stacked, filename)
    if sd is None or md is None:
        return None
    layers = _get_layers(sd)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return None

    # single: B seqs, stacked: B*stack seqs
    B = cu_s.numel() - 1
    per_layer: dict = {}
    worst_md = 0.0
    worst_cos = 1.0
    for lyr in layers:
        st = sd[lyr].float()
        mt = md[lyr].float()
        layer_md, layer_cos_min = 0.0, 1.0
        all_cos: list[float] = []
        for i in range(B):
            s_seq = st[int(cu_s[i]):int(cu_s[i + 1])]
            T_i = s_seq.shape[0]
            if T_i == 0:
                continue
            on_f = s_seq.reshape(T_i, -1)
            for k in range(n_copies):
                j = i + k * B
                c_seq = mt[int(cu_m[j]):int(cu_m[j + 1])]
                if c_seq.shape[0] != T_i:
                    continue
                off_f = c_seq.reshape(T_i, -1)
                cos = _cosine_sim(on_f, off_f, dim=-1)
                all_cos.extend(cos.tolist())
                md_i = float((on_f - off_f).abs().max())
                cos_i = float(cos.min())
                layer_md = max(layer_md, md_i)
                layer_cos_min = min(layer_cos_min, cos_i)
        per_layer[lyr] = {
            "max_diff": layer_md,
            "cos_avg": sum(all_cos) / len(all_cos) if all_cos else 0.0,
            "cos_min": layer_cos_min,
            "n_tokens": int(sum(c.shape[0] for c in [st[int(cu_s[i]):int(cu_s[i+1])] for i in range(B)])) if B > 0 else 0,
            "on_T": st.shape[0], "off_T": mt.shape[0],
        }
        worst_md = max(worst_md, layer_md)
        worst_cos = min(worst_cos, layer_cos_min)

    passed = worst_md < 1e-5
    _name = f"{label}_L{layer}" if layer is not None else label
    return CheckResult(name=_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_md,
                                "cos_min": worst_cos, "num_layers": len(layers)})


def _compare_rope(dir_single: str, dir_stacked: str, filename: str,
                  cu_s: torch.Tensor, cu_m: torch.Tensor,
                  n_copies: int, layer: int | None,
                  label: str, fld_q: str, fld_k: str
                  ) -> CheckResult | None:
    """Compare ``{layer: {fld_q, fld_k}}`` dicts across copies."""
    sd = _load_dict(dir_single, filename)
    md = _load_dict(dir_stacked, filename)
    if sd is None or md is None:
        return None
    layers = _get_layers(sd)
    if layer is not None:
        layers = [l for l in layers if l == layer]
    if not layers:
        return None

    B = cu_s.numel() - 1
    per_layer: dict = {}
    for lyr in layers:
        sq = sd[lyr][fld_q].float(); mq = md[lyr][fld_q].float()
        sk = sd[lyr][fld_k].float(); mk = md[lyr][fld_k].float()
        q_max, k_max = 0.0, 0.0
        q_cos_min, k_cos_min = 1.0, 1.0
        q_all, k_all = [], []
        for i in range(B):
            s_q = sq[int(cu_s[i]):int(cu_s[i + 1])]
            s_k = sk[int(cu_s[i]):int(cu_s[i + 1])]
            T_i = s_q.shape[0]
            if T_i == 0: continue
            qf = s_q.reshape(T_i, -1); kf = s_k.reshape(T_i, -1)
            for h in range(n_copies):
                j = i + h * B
                cq = mq[int(cu_m[j]):int(cu_m[j + 1])]
                ck = mk[int(cu_m[j]):int(cu_m[j + 1])]
                if cq.shape[0] != T_i: continue
                cfq = cq.reshape(T_i, -1); cfk = ck.reshape(T_i, -1)
                q_cos = _cosine_sim(qf, cfq, dim=-1)
                k_cos = _cosine_sim(kf, cfk, dim=-1)
                q_all.extend(q_cos.tolist()); k_all.extend(k_cos.tolist())
                q_max = max(q_max, float((qf - cfq).abs().max()))
                k_max = max(k_max, float((kf - cfk).abs().max()))
                q_cos_min = min(q_cos_min, float(q_cos.min()))
                k_cos_min = min(k_cos_min, float(k_cos.min()))
        per_layer[lyr] = {
            "Q_max_diff": q_max, "K_max_diff": k_max,
            "Q_cos_avg": sum(q_all)/len(q_all) if q_all else 0.0,
            "Q_cos_min": q_cos_min,
            "K_cos_avg": sum(k_all)/len(k_all) if k_all else 0.0,
            "K_cos_min": k_cos_min,
            "n_tokens": len(q_all),
        }
    _name = f"{label}_L{layer}" if layer is not None else label
    return CheckResult(name=_name, passed=True, metrics={"layers": per_layer})


def _compare_logits(dir_single: str, dir_stacked: str,
                    cu_s: torch.Tensor, cu_m: torch.Tensor,
                    n_copies: int) -> CheckResult | None:
    """Compare packed logits across copies."""
    fp_s = os.path.join(dir_single, "logits.pt")
    fp_m = os.path.join(dir_stacked, "logits.pt")
    if not os.path.exists(fp_s) or not os.path.exists(fp_m):
        return None
    st = torch.load(fp_s, weights_only=True).float()
    mt = torch.load(fp_m, weights_only=True).float()
    st = st.reshape(-1, st.size(-1))
    mt = mt.reshape(-1, mt.size(-1))
    B = cu_s.numel() - 1
    worst_md, worst_cos_min = 0.0, 1.0
    all_cos, total_tokens = [], 0
    for i in range(B):
        s_seq = st[int(cu_s[i]):int(cu_s[i + 1])]
        if s_seq.shape[0] == 0: continue
        total_tokens += s_seq.shape[0]
        for k in range(n_copies):
            j = i + k * B
            c_seq = mt[int(cu_m[j]):int(cu_m[j + 1])]
            if c_seq.shape[0] != s_seq.shape[0]: continue
            cos = _cosine_sim(s_seq, c_seq, dim=-1)
            all_cos.extend(cos.tolist())
            worst_md = max(worst_md, float((s_seq - c_seq).abs().max()))
            worst_cos_min = min(worst_cos_min, float(cos.min()))
    return CheckResult(name="logits", passed=worst_md < 1e-5,
                       metrics={"n_tokens": total_tokens,
                                "cos_avg": sum(all_cos)/len(all_cos) if all_cos else 0.0,
                                "cos_min": worst_cos_min})


def _compare_2d_file(dir_single: str, dir_stacked: str, filename: str,
                     name: str, n_copies: int, num_seq: int,
                     atol: float = 1e-5
                     ) -> tuple[CheckResult | None, torch.Tensor | None, torch.Tensor | None]:
    """Compare 2D [B, L_max] tensors — row i from single vs rows i+k*num_seq from stacked."""
    st = _load_tensor(dir_single, filename)
    mt = _load_tensor(dir_stacked, filename)
    if st is None or mt is None:
        return None, st, mt
    st = st.float(); mt = mt.float()
    if st.dim() < 2 or mt.dim() < 2:
        return None, st, mt
    # only compare valid (non-padded) rows
    worst_md, worst_cos = 0.0, 1.0
    for i in range(num_seq):
        if i >= st.shape[0]:
            break
        a = st[i].reshape(-1)
        for k in range(n_copies):
            j = i + k * num_seq
            if j >= mt.shape[0]:
                continue
            b = mt[j].reshape(-1)
            md = float((a - b).abs().max())
            cos = float(_cosine_sim(a, b, dim=-1))
            worst_md = max(worst_md, md)
            worst_cos = min(worst_cos, cos)
    pr = _pearson_r(st[:num_seq].reshape(-1), mt[:num_seq * n_copies].reshape(-1))
    return (CheckResult(name=name,
                        passed=worst_md <= atol,
                        metrics={"shape": tuple(st.shape),
                                 "active": st[:num_seq].numel(),
                                 "abs_max": worst_md,
                                 "abs_mean": 0.0,
                                 "rel_max": 0.0,
                                 "rel_mean": 0.0,
                                 "pearson_r": pr,
                                 "atol": atol}), st, mt)


# ── top-K helpers (reuse cmp_diag_verl080 top-K printers) ────────

def _topk_per_layer(dir_single: str, dir_stacked: str,
                    cu_s: torch.Tensor, cu_m: torch.Tensor, n_copies: int,
                    filename: str, label: str,
                    topk: int, sort_err: str):
    """Print top-K worst dims for a per-layer plain file."""
    sd = _load_dict(dir_single, filename)
    md = _load_dict(dir_stacked, filename)
    if sd is None or md is None:
        return
    lyr = max(int(k) for k in sd.keys())
    st = sd[lyr].float(); mt = md[lyr].float()
    B = cu_s.numel() - 1
    worst_md = 0.0; worst_on = worst_off = None
    for i in range(B):
        s_seq = st[int(cu_s[i]):int(cu_s[i + 1])]
        if s_seq.shape[0] == 0: continue
        on_f = s_seq.reshape(s_seq.shape[0], -1)
        for k in range(n_copies):
            j = i + k * B
            ct = mt[int(cu_m[j]):int(cu_m[j + 1])]
            if ct.shape[0] != s_seq.shape[0]: continue
            off_f = ct.reshape(ct.shape[0], -1)
            md = float((on_f - off_f).abs().max())
            if md > worst_md: worst_md = md; worst_on = on_f; worst_off = off_f
    if worst_on is not None:
        _print_topk_vec(worst_on[0].cpu(), worst_off[0].cpu(),
                        topk, sort_err, f"{label}_L{lyr}_token0")


def _topk_rope(dir_single: str, dir_stacked: str,
               cu_s: torch.Tensor, cu_m: torch.Tensor, n_copies: int,
               filename: str, fld_q: str, fld_k: str,
               label: str, topk: int, sort_err: str):
    """Print top-K worst dims for a rope file."""
    sd = _load_dict(dir_single, filename)
    md = _load_dict(dir_stacked, filename)
    if sd is None or md is None: return
    lyr = max(int(k) for k in sd.keys())
    B = cu_s.numel() - 1
    for fld, tag in [(fld_q, f"{label}_Q"), (fld_k, f"{label}_K")]:
        sq = sd[lyr][fld].float(); mq = md[lyr][fld].float()
        worst_md = 0.0; worst_on = worst_off = None
        for i in range(B):
            s_seq = sq[int(cu_s[i]):int(cu_s[i + 1])]
            if s_seq.shape[0] == 0: continue
            on_f = s_seq.reshape(s_seq.shape[0], -1)
            for k in range(n_copies):
                j = i + k * B
                ct = mq[int(cu_m[j]):int(cu_m[j + 1])]
                if ct.shape[0] != s_seq.shape[0]: continue
                off_f = ct.reshape(ct.shape[0], -1)
                md = float((on_f - off_f).abs().max())
                if md > worst_md: worst_md = md; worst_on = on_f; worst_off = off_f
        if worst_on is not None:
            _print_topk_vec(worst_on[0].cpu(), worst_off[0].cpu(),
                            topk, sort_err, f"{tag}_L{lyr}_token0")


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GEMM precision baseline — cross-batch-size (single vs N copies)",
        epilog=__doc__,
    )
    ap.add_argument("--dir-single", required=True,
                    help="Single-copy dump directory (batch=[A])")
    ap.add_argument("--dir-stacked", required=True,
                    help="Stacked-copies dump directory (batch=[A x N])")
    ap.add_argument("--num-copies", type=int, required=True,
                    help="Number of stacked copies N")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific layer 1-indexed (default: all)")
    ap.add_argument("--tag", default="old",
                    help="2D file tag for logprobs/entropy (default: old)")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D (default: 1e-5)")
    ap.add_argument("--topk", type=int, default=0,
                    help="top-K worst dims for packed-token (0=disabled)")
    ap.add_argument("--sort-err", choices=["abs", "rel", "val"], default="abs",
                    help="top-K sort: abs / rel / val")
    ap.add_argument("--output", "-o", default=None,
                    help="Write JSON report to this path")
    args = ap.parse_args()

    cu_s  = _load_cu(args.dir_single)
    cu_m  = _load_cu(args.dir_stacked)
    if cu_s is None or cu_m is None:
        print(f"{_CROSS} cu_seqlens_q.pt missing"); return 1

    T = int(cu_s[-1])
    n = args.num_copies
    num_seq = cu_s.numel() - 1
    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Cross-Batch-Size Comparison")
    print(f"  Single:  {args.dir_single}  (1 seq, {T} tokens)")
    print(f"  Stacked: {args.dir_stacked}  ({n} copies, {n * T} tokens)")
    print(_SEP_DOUBLE)

    all_results: list[CheckResult] = []

    # ── hidden_states ──
    r = _compare_plain(args.dir_single, args.dir_stacked,
                       "hidden_states.pt", cu_s, cu_m, n, args.layer, "hidden_states")
    if r: all_results.append(r); _print_hidden_states(r)

    # ── build_kv_input_v ──
    r = _compare_plain(args.dir_single, args.dir_stacked,
                       "build_kv_input_v.pt", cu_s, cu_m, n, args.layer, "build_kv_input_v")
    if r: all_results.append(r); _print_build_kv_input_v(r)

    # ── rope_preqk ──
    r = _compare_rope(args.dir_single, args.dir_stacked, "rope_preqk.pt",
                      cu_s, cu_m, n, args.layer, "rope_preqk", "query", "key")
    if r: all_results.append(r); _print_rope_postqk_per_layer(r)

    # ── rope_freqs ──
    r = _compare_plain(args.dir_single, args.dir_stacked,
                       "rope_freqs.pt", cu_s, cu_m, n, args.layer, "rope_freqs")
    if r: all_results.append(r); _print_rope_freqs(r)

    # ── rope_postqk ──
    r = _compare_rope(args.dir_single, args.dir_stacked, "rope_postqk.pt",
                      cu_s, cu_m, n, args.layer, "rope_postqk", "query", "key")
    if r: all_results.append(r); _print_rope_postqk_per_layer(r)

    # ── attn_outputs ──
    r = _compare_plain(args.dir_single, args.dir_stacked,
                       "attn_outputs.pt", cu_s, cu_m, n, args.layer, "attn_outputs")
    if r: all_results.append(r); _print_per_layer(r)

    # ── full_kv ──
    r = _compare_rope(args.dir_single, args.dir_stacked, "full_kv.pt",
                      cu_s, cu_m, n, args.layer, "full_kv", "key", "value")
    if r: all_results.append(r); _print_rope_postqk_per_layer(r)

    # ── logits ──
    r = _compare_logits(args.dir_single, args.dir_stacked, cu_s, cu_m, n)
    if r: all_results.append(r); _print_logits_packed(r)

    # ── 2D: logprobs + entropy ──
    for fn, cn in [("logprobs", "logp"), ("entropy", "entropy")]:
        fname = f"{fn}_{args.tag}.pt"
        r, t1, t2 = _compare_2d_file(args.dir_single, args.dir_stacked,
                                      fname, f"{cn}_{args.tag}", n, num_seq, args.atol)
        if r: all_results.append(r); _print_2d_result(r)

    # ── top-K ──
    if args.topk > 0:
        _topk_per_layer(args.dir_single, args.dir_stacked, cu_s, cu_m, n,
                        "build_kv_input_v.pt", "build_kv_input_v",
                        args.topk, args.sort_err)
        _topk_rope(args.dir_single, args.dir_stacked, cu_s, cu_m, n,
                   "rope_preqk.pt", "query", "key", "rope_preqk",
                   args.topk, args.sort_err)
        _topk_rope(args.dir_single, args.dir_stacked, cu_s, cu_m, n,
                   "rope_postqk.pt", "query", "key", "rope_postqk",
                   args.topk, args.sort_err)

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_single, args.dir_stacked,
                   tag=f"cross_batch_N{n}", dir_off2=None)


if __name__ == "__main__":
    main()
