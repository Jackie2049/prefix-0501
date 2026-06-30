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
    _print_logits_packed,
    _print_2d_result,
    _print_topk_vec,
    _print_topk_2d,
    _print_summary,
    _print_shapes,
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
    """Pairwise compare all copies of the SAME length; return worst + avg cos."""
    worst_md, worst_cos_min = 0.0, 1.0
    all_cos: list[float] = []
    for i in range(len(copies)):
        for j in range(i + 1, len(copies)):
            a = copies[i].reshape(copies[i].shape[0], -1).float()
            b = copies[j].reshape(copies[j].shape[0], -1).float()
            cos = _cosine_sim(a, b, dim=-1)
            all_cos.extend(cos.tolist())
            md = float((a - b).abs().max())
            worst_md = max(worst_md, md)
            worst_cos_min = min(worst_cos_min, float(cos.min()))
    return {"max_diff": worst_md, "cos_avg": sum(all_cos)/len(all_cos) if all_cos else 0.0,
            "cos_min": worst_cos_min}


def _group_by_len(copies: list[torch.Tensor]) -> dict[int, list[torch.Tensor]]:
    """Group copies by sequence length (tokens)."""
    groups: dict[int, list[torch.Tensor]] = {}
    for c in copies:
        groups.setdefault(c.shape[0], []).append(c)
    return groups


def _compare_plain_within(dir_multi: str, filename: str,
                          cu: torch.Tensor, total_copies: int,
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
        copies = [mt[int(cu[i]):int(cu[i + 1])] for i in range(total_copies)]
        groups = _group_by_len(copies)
        layer_md, layer_cos_min = 0.0, 1.0
        all_cos_avg: list[float] = []
        for group_copies in groups.values():
            if len(group_copies) < 2:
                continue
            w = _worst_pairwise(group_copies)
            layer_md = max(layer_md, w["max_diff"])
            layer_cos_min = min(layer_cos_min, w["cos_min"])
            all_cos_avg.append(w["cos_avg"])
        per_layer[lyr] = {
            "max_diff": layer_md,
            "cos_avg": sum(all_cos_avg)/len(all_cos_avg) if all_cos_avg else 0.0,
            "cos_min": layer_cos_min,
            "n_tokens": mt.shape[0], "on_T": mt.shape[0], "off_T": mt.shape[0],
        }
        worst_md = max(worst_md, layer_md)
        worst_cos = min(worst_cos, layer_cos_min)
    passed = worst_md == 0.0
    _name = f"{label}_L{layer}" if layer is not None else label
    return CheckResult(name=_name, passed=passed,
                       metrics={"layers": per_layer, "max_diff": worst_md,
                                "cos_min": worst_cos, "num_layers": len(layers)})


def _compare_rope_within(dir_multi: str, filename: str,
                         cu: torch.Tensor, total_copies: int,
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
        mq = d[lyr][fld_q].float(); mk = d[lyr][fld_k].float()
        q_copies = [mq[int(cu[i]):int(cu[i + 1])] for i in range(total_copies)]
        k_copies = [mk[int(cu[i]):int(cu[i + 1])] for i in range(total_copies)]
        q_groups = _group_by_len(q_copies)
        k_groups = _group_by_len(k_copies)
        qw = {"max_diff": 0.0, "cos_min": 1.0, "cos_avg": 0.0}
        kw = {"max_diff": 0.0, "cos_min": 1.0, "cos_avg": 0.0}
        q_avgs, k_avgs = [], []
        for g in q_groups.values():
            if len(g) >= 2:
                w = _worst_pairwise(g)
                qw["max_diff"] = max(qw["max_diff"], w["max_diff"])
                qw["cos_min"] = min(qw["cos_min"], w["cos_min"])
                q_avgs.append(w["cos_avg"])
        for g in k_groups.values():
            if len(g) >= 2:
                w = _worst_pairwise(g)
                kw["max_diff"] = max(kw["max_diff"], w["max_diff"])
                kw["cos_min"] = min(kw["cos_min"], w["cos_min"])
                k_avgs.append(w["cos_avg"])
        per_layer[lyr] = {
            "Q_max_diff": qw["max_diff"], "K_max_diff": kw["max_diff"],
            "Q_cos_avg": sum(q_avgs)/len(q_avgs) if q_avgs else 0.0,
            "Q_cos_min": qw["cos_min"],
            "K_cos_avg": sum(k_avgs)/len(k_avgs) if k_avgs else 0.0,
            "K_cos_min": kw["cos_min"],
            "n_tokens": sum(g[0].shape[0] * len(g) for g in q_groups.values()),
        }
    _name = f"{label}_L{layer}" if layer is not None else label
    return CheckResult(name=_name, passed=True, metrics={"layers": per_layer})


def _compare_logits_within(dir_multi: str, cu: torch.Tensor,
                           total_copies: int) -> CheckResult | None:
    fp = os.path.join(dir_multi, "logits.pt")
    if not os.path.exists(fp):
        return None
    mt = torch.load(fp, weights_only=True).float()
    mt = mt.reshape(-1, mt.size(-1))
    copies = [mt[int(cu[i]):int(cu[i + 1])] for i in range(total_copies)]
    groups = _group_by_len(copies)
    worst_md, worst_cos_min = 0.0, 1.0
    all_cos_avg: list[float] = []
    for g in groups.values():
        if len(g) >= 2:
            w = _worst_pairwise(g)
            worst_md = max(worst_md, w["max_diff"])
            worst_cos_min = min(worst_cos_min, w["cos_min"])
            all_cos_avg.append(w["cos_avg"])
    return CheckResult(name="logits", passed=worst_md == 0.0,
                       metrics={"n_tokens": copies[0].shape[0] if copies else 0,
                                "cos_avg": sum(all_cos_avg)/len(all_cos_avg) if all_cos_avg else 0.0,
                                "cos_min": worst_cos_min})


# ── print wrappers ─────────────────────────────────────────────

def _print_plain_baseline(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [{r.name}]  within-batch pairwise")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {_CROSS} {m['error']}\n"); return
    layers = m.get("layers", {})
    print(f"  {'LAYER':>6s}  {'MAXDIFF':>12s} {'COS_AVG':>10s} {'COS_MIN':>10s}  "
          f"{'TOKENS':>8s}  {'STATUS':>8s}")
    print(f"  {'─' * 6}  {'─' * 12} {'─' * 10} {'─' * 10}  {'─' * 8}  {'─' * 8}")
    for lyr in sorted(layers):
        d = layers[lyr]
        if "max_diff" not in d:
            print(f"  {lyr:>6d}  {d.get('error','')}"); continue
        md = d["max_diff"]; ca = d.get("cos_avg", 0); cm = d.get("cos_min", 1)
        ok = md == 0.0
        print(f"  {lyr:>6d}  {md:>12.3e} {ca:>10.6f} {cm:>10.6f}  "
              f"{d.get('n_tokens','—'):>8}  "
              f"{'PASS' if ok else 'DIFF':>8s}")
    print(f"\n  max_diff={m.get('max_diff')}  cos_min={m.get('cos_min')}  "
          f"{_CHECK if r.passed else _CROSS}")
    print()


def _print_rope_baseline(r: CheckResult, label: str):
    print(_SEP_SINGLE + f"\n  [{r.name}]  {label}  within-batch pairwise")
    print(_SEP_SINGLE)
    layers = r.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'Q_MAXDIFF':>12s} {'Q_COS_AVG':>12s} {'Q_COS_MIN':>12s}  "
              f"{'K_MAXDIFF':>12s} {'K_COS_AVG':>12s} {'K_COS_MIN':>12s}  "
              f"{'TOKENS':>8s}")
        print(f"  {'─' * 6}  {'─' * 12} {'─' * 12} {'─' * 12}  "
              f"{'─' * 12} {'─' * 12} {'─' * 12}  {'─' * 8}")
        for lyr in sorted(layers.keys()):
            d = layers[lyr]
            if "error" in d:
                print(f"  {lyr:>6d}  {d['error']}"); continue
            print(f"  {lyr:>6d}  {d.get('Q_max_diff',0.0):>12.3e} {d.get('Q_cos_avg',0.0):>12.6e} "
                  f"{d.get('Q_cos_min',0.0):>12.6e}  "
                  f"{d.get('K_max_diff',0.0):>12.3e} {d.get('K_cos_avg',0.0):>12.6e} "
                  f"{d.get('K_cos_min',0.0):>12.6e}  {d.get('n_tokens','—'):>8}")
    print()


# ── main ─────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GEMM precision baseline — within-batch pairwise comparison",
        epilog=__doc__,
    )
    ap.add_argument("--dir-multi", required=True,
                    help="Multi-copy dump directory (batch=[A x N])")
    ap.add_argument("--num-copies", type=int, default=None,
                    help="Total sequences in batch (auto-detected from cu_seqlens)")
    ap.add_argument("--num-seq", type=int, default=None,
                    help="Distinct sequences before stacking (required for 2D within-batch)")
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

    total_copies = cu.numel() - 1
    if args.num_copies is not None and args.num_copies != total_copies:
        print(f"[warn] --num-copies={args.num_copies} but cu_seqlens has {total_copies} copies; using {total_copies}")
    lengths = [int(cu[i + 1]) - int(cu[i]) for i in range(total_copies)]
    print(_SEP_DOUBLE)
    print("  GEMM Precision Baseline — Within-Batch Pairwise Comparison")
    print(f"  Directory: {args.dir_multi}  ({total_copies} sequences)")
    print(f"  Lengths: {lengths}")
    print(_SEP_DOUBLE)

    _print_shapes(args.dir_multi, args.dir_multi, args.tag)

    all_results: list[CheckResult] = []

    # ── hidden_states ──
    r = _compare_plain_within(args.dir_multi, "hidden_states.pt",
                              cu, total_copies, args.layer, "hidden_states")
    if r: all_results.append(r); _print_plain_baseline(r)

    # ── build_kv_input_v ──
    r = _compare_plain_within(args.dir_multi, "build_kv_input_v.pt",
                              cu, total_copies, args.layer, "build_kv_input_v")
    if r: all_results.append(r); _print_plain_baseline(r)

    # ── rope_preqk ──
    r = _compare_rope_within(args.dir_multi, "rope_preqk.pt",
                             cu, total_copies, args.layer, "rope_preqk", "query", "key")
    if r: all_results.append(r); _print_rope_baseline(r, r.name)

    # ── rope_freqs ──
    r = _compare_plain_within(args.dir_multi, "rope_freqs.pt",
                              cu, total_copies, args.layer, "rope_freqs")
    if r: all_results.append(r); _print_plain_baseline(r)

    # ── rope_postqk ──
    r = _compare_rope_within(args.dir_multi, "rope_postqk.pt",
                             cu, total_copies, args.layer, "rope_postqk", "query", "key")
    if r: all_results.append(r); _print_rope_baseline(r, r.name)

    # ── attn_outputs ──
    r = _compare_plain_within(args.dir_multi, "attn_outputs.pt",
                              cu, total_copies, args.layer, "attn_outputs")
    if r: all_results.append(r); _print_plain_baseline(r)

    # ── full_kv ──
    r = _compare_rope_within(args.dir_multi, "full_kv.pt",
                             cu, total_copies, args.layer, "full_kv", "key", "value")
    if r: all_results.append(r); _print_rope_baseline(r, r.name)

    # ── logits ──
    r = _compare_logits_within(args.dir_multi, cu, total_copies)
    if r: all_results.append(r); _print_logits_packed(r)

    # ── 2D: logprobs + entropy ──
    _2d_within: list[tuple[str, torch.Tensor]] = []
    _num_seq = args.num_seq if args.num_seq else (total_copies // (args.num_copies or total_copies) if args.num_copies else total_copies)
    for fn, cn in [("logprobs", "logp"), ("entropy", "entropy")]:
        fname = f"{fn}_{args.tag}.pt"
        st = _load_tensor(args.dir_multi, fname)
        if st is not None and st.dim() >= 2:
            B = st.shape[0]
            stack = B // _num_seq if _num_seq > 0 and B % _num_seq == 0 else 1
            all_abs, all_rel = [], []
            worst_md, worst_cos, worst_rel = 0.0, 1.0, 0.0
            # only compare stack copies of the SAME sequence (i vs i+k*num_seq)
            for i in range(_num_seq):
                if i >= B: break
                a = st[i].float().reshape(-1)
                for k in range(1, stack):
                    j = i + k * _num_seq
                    if j >= B: continue
                    b = st[j].float().reshape(-1)
                    diff = (a - b).abs()
                    rel = diff / a.abs().clamp(min=1e-8)
                    all_abs.extend(diff.tolist()); all_rel.extend(rel.tolist())
                    md = float(diff.max()); rm = float(rel.max())
                    worst_md = max(worst_md, md)
                    worst_rel = max(worst_rel, rm)
                    worst_cos = min(worst_cos, float(_cosine_sim(a, b, dim=-1)))
            r = CheckResult(name=f"{cn}_{args.tag}",
                            passed=worst_md == 0.0,
                            metrics={"shape": tuple(st.shape),
                                     "active": _num_seq * st.shape[1] if st.dim() >= 2 else st.numel(),
                                     "abs_max": worst_md,
                                     "abs_mean": sum(all_abs)/len(all_abs) if all_abs else 0.0,
                                     "rel_max": worst_rel,
                                     "rel_mean": sum(all_rel)/len(all_rel) if all_rel else 0.0,
                                     "pearson_r": _pearson_r(st[:_num_seq].float().reshape(-1),
                                                             st[_num_seq:2*_num_seq].float().reshape(-1))
                                     if B >= 2*_num_seq else 1.0,
                                     "atol": args.atol})
            all_results.append(r)
            _print_2d_result(r)
            _2d_within.append((f"{cn}_{args.tag}", st))

    # ── top-K ──
    if args.topk > 0:
        # 2D top-K: compare row 0 vs row _num_seq (stack copies of same seq)
        for label, st2d in _2d_within:
            ns = _num_seq if _num_seq > 0 and (st2d.shape[0] % _num_seq == 0) else st2d.shape[0]
            if st2d.shape[0] >= ns + 1:
                _print_topk_2d(st2d[:1].cpu(), st2d[ns:ns+1].cpu(), None,
                               args.topk, args.sort_err, label)
        # build_kv_input_v top-K
        d = _load_dict(args.dir_multi, "build_kv_input_v.pt")
        if d:
            lyr = max(int(k) for k in d.keys())
            mt = d[lyr].float()
            copies = [mt[int(cu[i]):int(cu[i+1])].reshape(-1) for i in range(total_copies)]
            groups = _group_by_len(copies)
            worst_md = 0.0; worst_a = worst_b = None
            for g in groups.values():
                if len(g) < 2: continue
                for i in range(len(g)):
                    for j in range(i + 1, len(g)):
                        md = float((g[i] - g[j]).abs().max())
                        if md > worst_md:
                            worst_md = md; worst_a = g[i]; worst_b = g[j]
            if worst_a is not None:
                _print_topk_vec(worst_a.cpu(), worst_b.cpu(),
                                args.topk, args.sort_err,
                                f"build_kv_input_v_L{lyr}_token0")

    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, "—", args.dir_multi,
                   tag=f"within_batch_N{total_copies}", dir_off2=None)


if __name__ == "__main__":
    main()
