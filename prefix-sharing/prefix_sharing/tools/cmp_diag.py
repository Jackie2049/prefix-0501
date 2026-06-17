"""Precision comparison — ON vs OFF (and OFF vs OFF) diagnostic dumps.

Covers the precision verification spec (see plan.md):

  1. position_ids      — absolute equality (STOP if fail)
  2. RoPE encoding     — absolute equality (STOP if fail)
  3. attn_output per-layer cos  — avg / min per layer
  4. First-token suite — mean_abs / max_abs / rel(max,mean) / cos / pearson
  5. entropy / logp    — error_abs(max,mean) / error_rel(max,mean) / pearson
  6. OFF vs OFF baseline — noise floor for all metrics

Usage:
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --tag old
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --tag old \
        --mask-file ./my_mask.pt --layer 12
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off \
        --dir-off2 ./dump_off2 --tag old -o report.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

# ── Constants ──
_SEP_DOUBLE = "=" * 70
_SEP_SINGLE = "-" * 70
_SEP_THIN   = "─" * 70
_CHECK      = "\u2713"
_CROSS      = "\u2717"


def _fmt_shape(s: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in s) + ")"


# ══════════════════════════════════════════════════════════════════
#  Dataclasses
# ══════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool = True
    metrics: dict = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════
#  Mask helpers
# ══════════════════════════════════════════════════════════════════

def _trim_last_true_per_row(mask: torch.Tensor) -> torch.Tensor:
    """In each row of a 2D bool mask, flip the last True to False."""
    if mask.dim() != 2:
        return mask
    result = mask.clone()
    for row in range(mask.shape[0]):
        pos = mask[row].nonzero(as_tuple=True)[0]
        if len(pos) > 0:
            result[row, pos[-1]] = False
    return result


def _load_label_mask(dir_path: str) -> torch.Tensor | None:
    path = os.path.join(dir_path, "label_mask.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).to(torch.bool)


def _resolve_label_mask(mask_file: str | None, dir_off: str,
                        trim_last: bool = True) -> torch.Tensor | None:
    """Resolve effective label_mask: external file > auto-load from OFF dump.

    Always applies last-True→False trim when ``trim_last=True``.
    """
    if mask_file:
        mask = torch.load(mask_file, weights_only=True).to(torch.bool)
        return _trim_last_true_per_row(mask) if trim_last else mask
    mask = _load_label_mask(dir_off)
    if mask is not None and trim_last:
        mask = _trim_last_true_per_row(mask)
    return mask


# ══════════════════════════════════════════════════════════════════
#  Metric computation helpers
# ══════════════════════════════════════════════════════════════════

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1) -> torch.Tensor:
    eps = 1e-8
    return (a * b).sum(dim=dim) / (a.norm(dim=dim) + eps) / (b.norm(dim=dim) + eps)


def _pearson_r(t1: torch.Tensor, t2: torch.Tensor,
               mask: torch.Tensor | None = None) -> float:
    x = (t1[mask] if mask is not None else t1.flatten()).to(torch.float64)
    y = (t2[mask] if mask is not None else t2.flatten()).to(torch.float64)
    if x.numel() < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx, sy = ((x - mx) ** 2).sum().sqrt(), ((y - my) ** 2).sum().sqrt()
    return float("nan") if sx == 0 or sy == 0 else float(cov / (sx * sy))


def _error_abs_rel(a: torch.Tensor, b: torch.Tensor,
                   mask: torch.Tensor | None = None) -> dict:
    diff = (a - b).abs()
    rel = diff / torch.maximum(a.abs(), b.abs()).clamp(min=1e-8)
    if mask is not None:
        diff, rel = diff[mask], rel[mask]
    if diff.numel() == 0:
        return {"abs_max": 0.0, "abs_mean": 0.0, "rel_max": 0.0, "rel_mean": 0.0}
    return {"abs_max": float(diff.max()), "abs_mean": float(diff.mean()),
            "rel_max": float(rel.max()), "rel_mean": float(rel.mean())}


def _first_token_metrics(a_vec: torch.Tensor, b_vec: torch.Tensor) -> dict:
    err = _error_abs_rel(a_vec, b_vec)
    cos = float(_cosine_sim(a_vec, b_vec, dim=-1))
    pr  = _pearson_r(a_vec, b_vec)
    return {"mean_abs": err["abs_mean"], "max_abs": err["abs_max"],
            "rel_max": err["rel_max"], "rel_mean": err["rel_mean"],
            "cos": cos, "pearson": pr}


# ══════════════════════════════════════════════════════════════════
#  Loading helpers
# ══════════════════════════════════════════════════════════════════

def _load_tensor(dir_path: str, filename: str) -> torch.Tensor | None:
    fp = os.path.join(dir_path, filename)
    return torch.load(fp, weights_only=True).float() if os.path.exists(fp) else None


def _load_packed_meta(dir_path: str,
                      cu_fname: str = "cu_seqlens_q.pt") -> dict | None:
    fp = os.path.join(dir_path, cu_fname)
    if not os.path.exists(fp):
        fp = os.path.join(dir_path, "cu_seqlens_q.pt")
        if not os.path.exists(fp):
            return None
    pl_fp = os.path.join(dir_path, "prefix_lens.pt")
    if not os.path.exists(pl_fp):
        return None
    return {"cu_seqlens": torch.load(fp, weights_only=True),
            "prefix_lens": torch.load(pl_fp, weights_only=True)}


def _load_attn_output(dir_path: str, layer: int) -> torch.Tensor | None:
    """Load a single layer's attn_output from attn_outputs.pt dict."""
    fp = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(fp):
        return None
    d = torch.load(fp, weights_only=True)
    return d.get(layer) if isinstance(d, dict) else None


def _load_attention_mask_2d(dir_path: str) -> torch.Tensor | None:
    """Load 2D [B, L_max] attention_mask.pt dumped by ``dump_attention_mask_2d``.

    Semantics (see diagnostic_dump.dump_attention_mask_2d):
      - ON: True only at suffix columns per row (rewritten by verl_mcore)
      - OFF: True at all valid token columns per row

    Both share the same [B, L_max] shape and per-row column semantics
    (column ``c`` corresponds to the same original token position in both
    runs).  ON mask is a strict subset of OFF mask per row, which
    ``_build_alignment_mask_from_2d`` relies on.
    """
    fp = os.path.join(dir_path, "attention_mask.pt")
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=True).to(torch.bool)


def _get_num_layers(dir_path: str) -> int:
    fp = os.path.join(dir_path, "attn_outputs.pt")
    if not os.path.exists(fp):
        return 0
    d = torch.load(fp, weights_only=True)
    return max(d.keys()) if isinstance(d, dict) and d else 0


# ══════════════════════════════════════════════════════════════════
#  Packed alignment — build 1D suffix mask, align ON/OFF tensors
# ══════════════════════════════════════════════════════════════════

def _build_alignment_mask(cu_seqlens: torch.Tensor,
                          prefix_lens: torch.Tensor,
                          total_tokens: int) -> torch.Tensor:
    """Build 1D bool mask [total_tokens]: True iff position is a suffix token.

    Built from ON's metadata and applied to OFF's full packed data to extract
    the same suffix region, yielding equal-shape tensors for comparison.

    Example:
        cu_seqlens=[0,7,13], prefix_lens=[3,4], total_tokens=13
        → [0,0,0, 1,1,1,1,  0,0,0,0, 1,1]
    """
    mask = torch.zeros(total_tokens, dtype=torch.bool)
    for i in range(cu_seqlens.shape[0] - 1):
        pf = int(prefix_lens[i])
        start = int(cu_seqlens[i]) + pf
        end = int(cu_seqlens[i + 1])
        mask[start:end] = True
    return mask


def _align_packed(on_tensor: torch.Tensor,
                  off_tensor: torch.Tensor,
                  alignment_mask: torch.Tensor,
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Align ON-only-suffix with OFF-full-packed via 1D alignment_mask.

    Returns (on, off_suffix) where off_suffix has same shape[0] as on_tensor.

    Double-pointer semantics:
      for each i in [0, OFF_total_tokens):
        if alignment_mask[i]: compare ON[on_ptr++] vs OFF[i]
        else:                skip (prefix position in OFF)
    """
    T = off_tensor.shape[0]
    n_suffix = int(alignment_mask.sum())
    if alignment_mask.shape[0] != T:
        raise ValueError(
            f"alignment_mask len {alignment_mask.shape[0]} != OFF tokens {T}")
    if on_tensor.shape[0] != n_suffix:
        raise ValueError(
            f"ON tokens {on_tensor.shape[0]} != alignment True count {n_suffix}")
    return on_tensor, off_tensor[alignment_mask]


def _build_alignment_mask_from_2d(
    mask_on_2d: torch.Tensor,
    mask_off_2d: torch.Tensor,
    cu_seqlens_off: torch.Tensor,
    total_off: int,
) -> torch.Tensor:
    """Build 1D alignment mask [total_off] from ON/OFF 2D [B, L_max] masks.

    ON mask is a strict subset of OFF mask per row (ON ⊂ OFF).
    For each sequence row r:
      - OFF packed row r = cu_seqlens_off[r] : cu_seqlens_off[r+1],
        whose tokens correspond to mask_off_2d[r] True columns in order.
      - ON wants the suffix tokens → columns where mask_on_2d[r] is True.
      - Since mask_on[r] ⊂ mask_off[r], we intersect to find which OFF
        packed positions correspond to suffix tokens in ON.
    """
    align = torch.zeros(total_off, dtype=torch.bool)
    for r in range(mask_on_2d.shape[0]):
        row_start = int(cu_seqlens_off[r])
        row_end = int(cu_seqlens_off[r + 1])
        if row_start >= row_end:
            continue
        off_len = row_end - row_start
        # Columns in original L_max that correspond to OFF packed row tokens
        off_cols = mask_off_2d[r].nonzero(as_tuple=True)[0]
        if off_cols.numel() < off_len:
            continue
        off_cols = off_cols[:off_len]
        # Mark positions where ON mask also has True (these are suffix token columns)
        is_on = mask_on_2d[r, off_cols].to(torch.bool)
        align[row_start:row_end] = is_on
    return align


def _extract_suffix_rows(packed: torch.Tensor, cu_seqlens: torch.Tensor,
                         prefix_lens: torch.Tensor) -> list[torch.Tensor]:
    """Extract per-sequence suffix rows from packed tensor.

    Works for both ON (real prefix_lens) and OFF (prefix_lens=all-0,
    i.e. entire sequence is suffix).  Sequences with no suffix are skipped.
    """
    rows = []
    for i in range(cu_seqlens.shape[0] - 1):
        pf = int(prefix_lens[i])
        total_len = int(cu_seqlens[i + 1] - cu_seqlens[i])
        suffix_len = total_len - pf
        if suffix_len <= 0:
            continue
        start = int(cu_seqlens[i]) + pf
        t = packed[start:start + suffix_len]
        rows.append(t.squeeze(1) if t.dim() == 3 else t)
    return rows


# ══════════════════════════════════════════════════════════════════
#  1. position_ids — absolute equality
# ══════════════════════════════════════════════════════════════════

def cmp_position_ids(dir_a: str, dir_b: str) -> CheckResult | None:
    a = _load_tensor(dir_a, "position_ids.pt")
    b = _load_tensor(dir_b, "position_ids.pt")
    if a is None or b is None:
        return None
    ok = bool((a.to(torch.long) == b.to(torch.long)).all())
    return CheckResult(name="position_ids", passed=ok,
                       metrics={"shape": _fmt_shape(tuple(a.shape))})


# ══════════════════════════════════════════════════════════════════
#  2. RoPE encoding — absolute equality
# ══════════════════════════════════════════════════════════════════

def cmp_rope_emb(dir_a: str, dir_b: str) -> CheckResult | None:
    fa = os.path.join(dir_a, "rope_emb.pt")
    fb = os.path.join(dir_b, "rope_emb.pt")
    if not os.path.exists(fa) or not os.path.exists(fb):
        return None
    a = torch.load(fa, weights_only=True)
    b = torch.load(fb, weights_only=True)
    if not isinstance(a, dict) or not isinstance(b, dict):
        return None
    la, lb = set(a.keys()), set(b.keys())
    if la != lb:
        return CheckResult(name="rope_emb", passed=False,
                           metrics={"error": "layer set mismatch"})
    md = 0.0
    for lyr in sorted(la):
        for k in ("query", "key"):
            md = max(md, float((a[lyr][k] - b[lyr][k]).abs().max()))
    return CheckResult(name="rope_emb", passed=md == 0.0,
                       metrics={"max_diff": md, "num_layers": len(la)})


# ══════════════════════════════════════════════════════════════════
#  3. attention_output per-layer cos
# ══════════════════════════════════════════════════════════════════

def _per_layer_cos(dir_on: str, dir_off: str, layer: int | None) -> dict | None:
    """Per-layer cosine similarity with packed alignment.

    ON attn_output may have fewer tokens (suffix only) than OFF (all tokens).
    Builds alignment_mask from ON's cu_seqlens/prefix_lens and uses it to
    extract the matching suffix region from OFF before comparison.
    """

    def _cos_for_layer(a, b, lyr, need_align, align_mask):
        if a.dim() == 3:
            a, b = a.squeeze(1), b.squeeze(1)
        if need_align and a.shape[0] != b.shape[0]:
            try:
                a, b = _align_packed(a, b, align_mask)
            except ValueError as e:
                return {"cos_avg": 0.0, "cos_min": 0.0,
                        "n_tokens": a.shape[0], "error": str(e)}
        cos = _cosine_sim(a, b, dim=-1)
        return {"cos_avg": float(cos.mean()), "cos_min": float(cos.min()),
                "n_tokens": a.shape[0]}

    # Single-layer mode
    if layer is not None:
        a = _load_attn_output(dir_on, layer)
        b = _load_attn_output(dir_off, layer)
        if a is None or b is None:
            return None
        ma = _load_packed_meta(dir_on)
        align_mask = None
        need_align = (ma is not None and a.shape[0] != b.shape[0])
        if need_align:
            mb = _load_packed_meta(dir_off)
            T = int(mb["cu_seqlens"][-1]) if mb and mb["cu_seqlens"].numel() > 0 else b.shape[0]
            # Prefer 2D attention_mask.pt for exact suffix alignment (dp aligned)
            mask_on_2d = _load_attention_mask_2d(dir_on)
            mask_off_2d = _load_attention_mask_2d(dir_off)
            if mask_on_2d is not None and mask_off_2d is not None and mb is not None:
                align_mask = _build_alignment_mask_from_2d(
                    mask_on_2d, mask_off_2d, mb["cu_seqlens"], T)
            else:
                align_mask = _build_alignment_mask(
                    mb["cu_seqlens"], ma["prefix_lens"], T)
        d = _cos_for_layer(a, b, layer, need_align, align_mask)
        d["layer"] = layer
        return d

    # All-layers mode
    fa = os.path.join(dir_on, "attn_outputs.pt")
    fb = os.path.join(dir_off, "attn_outputs.pt")
    if not os.path.exists(fa) or not os.path.exists(fb):
        return None
    da = torch.load(fa, weights_only=True)
    db = torch.load(fb, weights_only=True)
    if not isinstance(da, dict) or not isinstance(db, dict):
        return None

    # Build alignment mask once from ON metadata
    ma = _load_packed_meta(dir_on)
    align_mask = None
    need_align = False
    if ma is not None:
        mb = _load_packed_meta(dir_off)
        T = int(mb["cu_seqlens"][-1]) if mb and mb["cu_seqlens"].numel() > 0 else 0
        if T > 0:
            # Check if any layer has shape mismatch
            for lyr in da:
                if lyr in db and da[lyr].shape != db[lyr].shape:
                    need_align = True
                    break
            if need_align:
                # Prefer 2D attention_mask.pt for exact suffix alignment
                mask_on_2d = _load_attention_mask_2d(dir_on)
                mask_off_2d = _load_attention_mask_2d(dir_off)
                if mask_on_2d is not None and mask_off_2d is not None:
                    align_mask = _build_alignment_mask_from_2d(
                        mask_on_2d, mask_off_2d, mb["cu_seqlens"], T)
                else:
                    align_mask = _build_alignment_mask(
                        mb["cu_seqlens"], ma["prefix_lens"], T)

    results = {}
    for lyr in sorted(set(da.keys()) & set(db.keys())):
        results[lyr] = _cos_for_layer(da[lyr], db[lyr], lyr, need_align, align_mask)
    return results


def cmp_attn_layer(dir_on: str, dir_off: str,
                   layer: int | None) -> CheckResult | None:
    r = _per_layer_cos(dir_on, dir_off, layer)
    if r is None:
        return None
    if "cos_avg" in r:
        lyr = r["layer"]
        return CheckResult(name=f"attn_L{lyr}",
                           passed=r["cos_avg"] > 0.9999 and r["cos_min"] > 0.999,
                           metrics=r)
    return CheckResult(name="attn_per_layer", passed=True,
                       metrics={"layers": r})


# ══════════════════════════════════════════════════════════════════
#  4. First-token metrics (last-layer attn + logits)
# ══════════════════════════════════════════════════════════════════

def _first_token_from_rows(rows_on: list, rows_off: list) -> dict | None:
    """Compare first token of each suffix row across ON/OFF.

    Both row lists must have the same number of sequences.  Per-sequence
    suffix lengths may differ — we use the shorter length.
    """
    if not rows_on or not rows_off:
        return None
    if len(rows_on) != len(rows_off):
        return None
    agg = {}
    for seq_idx, (ro, r_off) in enumerate(zip(rows_on, rows_off)):
        # Take the shorter suffix if lengths differ (safety belt)
        n = min(ro.shape[0], r_off.shape[0])
        if n == 0:
            continue
        ro, r_off = ro[:n], r_off[:n]
        fm = _first_token_metrics(ro[0], r_off[0])
        for k, v in fm.items():
            agg.setdefault(k, []).append(v)
    if not agg:
        return None
    return {k: {"mean": sum(v) / len(v), "min": min(v), "max": max(v)}
            for k, v in agg.items()}


def cmp_first_token(dir_on: str, dir_off: str) -> list[CheckResult]:
    results: list[CheckResult] = []
    last = _get_num_layers(dir_on) or _get_num_layers(dir_off)

    # attn_output (last layer only)
    if last:
        ma = _load_packed_meta(dir_on)
        mb = _load_packed_meta(dir_off)
        if ma and mb:
            a = _load_attn_output(dir_on, last)
            b = _load_attn_output(dir_off, last)
            if a is not None and b is not None:
                ft = _first_token_from_rows(
                    _extract_suffix_rows(a, ma["cu_seqlens"], ma["prefix_lens"]),
                    _extract_suffix_rows(b, mb["cu_seqlens"], mb["prefix_lens"]))
                if ft:
                    results.append(CheckResult(name="first_token_attn", metrics=ft))

    # logits
    ml_on  = _load_packed_meta(dir_on,  "cu_seqlens_q_logits.pt")
    ml_off = _load_packed_meta(dir_off, "cu_seqlens_q_logits.pt")
    lo = _load_tensor(dir_on,  "logits.pt")
    lf = _load_tensor(dir_off, "logits.pt")
    if ml_on and ml_off and lo is not None and lf is not None:
        ft = _first_token_from_rows(
            _extract_suffix_rows(lo, ml_on["cu_seqlens"], ml_on["prefix_lens"]),
            _extract_suffix_rows(lf, ml_off["cu_seqlens"], ml_off["prefix_lens"]))
        if ft:
            results.append(CheckResult(name="first_token_logits", metrics=ft))
    return results


# ══════════════════════════════════════════════════════════════════
#  5. 2D comparison (entropy / logp) — error_abs + error_rel + pearson
# ══════════════════════════════════════════════════════════════════

def cmp_2d(dir_on: str, dir_off: str, filename: str, name: str,
           mask: torch.Tensor | None, atol: float,
           label: torch.Tensor | None = None) -> CheckResult:
    t1 = _load_tensor(dir_on,  filename)
    t2 = _load_tensor(dir_off, filename)
    if t1 is None or t2 is None:
        return CheckResult(name=name, passed=False,
                           metrics={"error": "file missing"})
    if t1.shape != t2.shape:
        return CheckResult(name=name, passed=False,
                           metrics={"error": "shape mismatch",
                                    "s_on": _fmt_shape(tuple(t1.shape)),
                                    "s_off": _fmt_shape(tuple(t2.shape))})
    if mask is None:
        mask = torch.ones(t1.shape, dtype=torch.bool, device=t1.device)
    else:
        mask = mask.to(t1.device)
    mask = mask & ~torch.isnan(t1) & ~torch.isnan(t2)
    err = _error_abs_rel(t1, t2, mask)
    n_act = int(mask.sum())
    return CheckResult(name=name,
                       passed=n_act == 0 or err["abs_max"] <= atol,
                       metrics={"shape": _fmt_shape(tuple(t1.shape)),
                                "active": n_act,
                                "abs_max": err["abs_max"],
                                "abs_mean": err["abs_mean"],
                                "rel_max": err["rel_max"],
                                "rel_mean": err["rel_mean"],
                                "pearson_r": _pearson_r(t1, t2, mask),
                                "atol": atol})


# ══════════════════════════════════════════════════════════════════
#  Output helpers
# ══════════════════════════════════════════════════════════════════

def _print_header(dir_on, dir_off, dir_off2, tag, mask_file, layer):
    print(_SEP_DOUBLE)
    print(f"  Prefix-Sharing Diag Report")
    print(f"  ON :  {dir_on}\n  OFF:  {dir_off}")
    if dir_off2:
        print(f"  OFF2: {dir_off2}")
    if tag:
        print(f"  TAG: {tag}")
    if mask_file:
        print(f"  MASK: {mask_file} (external, last-token trimmed)")
    if layer is not None:
        print(f"  LAYER: {layer}")
    print(_SEP_DOUBLE + "\n")


def _print_pos_ids(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [position_ids]  {_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print(_SEP_SINGLE + f"\n  shape: {r.metrics.get('shape','—')}")
    if not r.passed:
        print(f"  {_CROSS} CRITICAL — STOP.\n")
    print()


def _print_rope(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [rope_emb]  {_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
    print(_SEP_SINGLE)
    m = r.metrics
    if "error" in m:
        print(f"  {m['error']}")
    else:
        print(f"  layers: {m.get('num_layers','—')}  max_diff: {m.get('max_diff','—')}")
    if not r.passed:
        print(f"  {_CROSS} CRITICAL — STOP.\n")
    print()


def _print_per_layer(r: CheckResult):
    print(_SEP_SINGLE + "\n  [attn_output]  Per-Layer Cosine Similarity")
    print(_SEP_SINGLE)
    layers = r.metrics.get("layers")
    if isinstance(layers, dict):
        print(f"  {'LAYER':>6s}  {'COS_AVG':>10s}  {'COS_MIN':>10s}  {'TOKENS':>8s}  {'STATUS':>8s}")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*8}")
        bad = []
        for lyr in sorted(layers.keys()):
            d = layers[lyr]
            ok = d["cos_avg"] > 0.9999 and d["cos_min"] > 0.999
            print(f"  {lyr:>6d}  {d['cos_avg']:>10.6f}  {d['cos_min']:>10.6f}  "
                  f"{d['n_tokens']:>8d}  {'PASS' if ok else 'WARN':>8s}")
            if not ok:
                bad.append(lyr)
        if bad:
            print(f"\n  ⚠ First deviating layer: {bad[0]}")
    elif "cos_avg" in r.metrics:
        d = r.metrics
        print(f"  L{d['layer']}  cos_avg={d['cos_avg']:.6f}  cos_min={d['cos_min']:.6f}")
    print()


def _print_first_token(r: CheckResult):
    print(_SEP_SINGLE + f"\n  [first_token]  {r.name}")
    print(_SEP_SINGLE)
    m = r.metrics
    print(f"  {'METRIC':>12s}  {'MEAN':>14s}  {'MIN':>14s}  {'MAX':>14s}")
    print(f"  {'─'*12}  {'─'*14}  {'─'*14}  {'─'*14}")
    for k in ["mean_abs", "max_abs", "rel_max", "rel_mean", "cos", "pearson"]:
        v = m.get(k)
        if isinstance(v, dict):
            print(f"  {k:>12s}  {v['mean']:>14.6e}  {v['min']:>14.6e}  {v['max']:>14.6e}")
        elif v is not None:
            print(f"  {k:>12s}  {v:>14.6e}")
    print()


def _print_summary(all_results: list[CheckResult]):
    print(_SEP_DOUBLE + "\n  SUMMARY")
    print(_SEP_DOUBLE)
    hdr = (f"  {'NAME':<28s} {'SHAPE':<18s} {'ABS_MAX':>12s}  {'REL_MAX':>10s}  "
           f"{'PEARSON_R':>10s}  {'STATUS':>8s}")
    print(hdr + "\n  " + "─" * (len(hdr) - 2))
    for r in all_results:
        m = r.metrics
        shape = m.get("shape", m.get("error", "—"))
        am = f"{m.get('abs_max', float('nan')):.6e}" if "abs_max" in m else "—"
        rm = f"{m.get('rel_max', float('nan')):.6e}" if "rel_max" in m else "—"
        pr = f"{m.get('pearson_r', float('nan')):.6f}" if m.get('pearson_r') is not None else "—"
        s  = f"  {_CHECK} PASS" if r.passed else f"  {_CROSS} FAIL"
        print(f"  {r.name:<28s} {shape:<18s} {am:>12s}  {rm:>10s}  {pr:>10s}  {s}")
    print(_SEP_DOUBLE + "\n")


def _dump_json(all_results: list[CheckResult], path: str,
               dir_on: str, dir_off: str, tag: str, dir_off2: str | None):
    record = {"dir_on": dir_on, "dir_off": dir_off}
    if dir_off2:
        record["dir_off2"] = dir_off2
    if tag:
        record["tag"] = tag
    record["results"] = [{k: v for k, v in r.metrics.items()} | {"name": r.name, "passed": r.passed}
                         for r in all_results]
    record["all_passed"] = all(r.passed for r in all_results)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  Report saved to: {path}\n")


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Precision comparison — ON vs OFF diagnostic dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir-on", required=True, help="ON dump directory")
    ap.add_argument("--dir-off", required=True, help="OFF dump directory")
    ap.add_argument("--dir-off2", default=None,
                    help="Second OFF directory for OFF-vs-OFF baseline")
    ap.add_argument("--tag", default=None,
                    help="Dump tag for 2D files (e.g. 'old', 'train')")
    ap.add_argument("--tag-on", default=None, help="Override tag for ON")
    ap.add_argument("--tag-off", default=None, help="Override tag for OFF")
    ap.add_argument("--atol", type=float, default=1e-5,
                    help="Absolute tolerance for 2D comparison (default: 1e-5)")
    ap.add_argument("--layer", type=int, default=None,
                    help="Compare specific attn layer (default: all layers)")
    ap.add_argument("--mask-file", default=None,
                    help="External label_mask.pt file (auto-trims last True per row)")
    ap.add_argument("--no-trim", action="store_true",
                    help="Skip last-True trimming of label_mask")
    ap.add_argument("--output", "-o", default=None, help="Save report as JSON")
    args = ap.parse_args()

    tag_on  = args.tag_on  or args.tag
    tag_off = args.tag_off or args.tag

    _print_header(args.dir_on, args.dir_off, args.dir_off2,
                  tag_on, args.mask_file, args.layer)

    # Resolve label mask (with trim)
    label_mask = _resolve_label_mask(args.mask_file, args.dir_off,
                                     trim_last=not args.no_trim)
    if label_mask is not None:
        print(f"  label_mask: {_fmt_shape(tuple(label_mask.shape))}  "
              f"active={int(label_mask.sum())}\n")

    all_results: list[CheckResult] = []
    stop = False

    # ── ① position_ids ──
    r = cmp_position_ids(args.dir_on, args.dir_off)
    if r:
        all_results.append(r)
        _print_pos_ids(r)
        if not r.passed:
            stop = True

    # ── ② RoPE encoding ──
    if not stop:
        r = cmp_rope_emb(args.dir_on, args.dir_off)
        if r:
            all_results.append(r)
            _print_rope(r)
            if not r.passed:
                stop = True

    # ── ③ attn_output per-layer cos ──
    if not stop:
        r = cmp_attn_layer(args.dir_on, args.dir_off, args.layer)
        if r:
            all_results.append(r)
            _print_per_layer(r)

    # ── ④ First-token metrics ──
    if not stop:
        for r in cmp_first_token(args.dir_on, args.dir_off):
            all_results.append(r)
            _print_first_token(r)

    # ── ⑤ 2D comparison (entropy / logp) ──
    if tag_on:
        for fname, cname in [("entropy", "entropy"), ("logprobs", "logp")]:
            fn = f"{fname}_{tag_on}.pt"
            t1 = _load_tensor(args.dir_on, fn)
            t2 = _load_tensor(args.dir_off, f"{fname}_{tag_off or tag_on}.pt")
            if t1 is None or t2 is None:
                continue
            r = cmp_2d(args.dir_on, args.dir_off, fn,
                       f"{cname}_{tag_on}", label_mask, args.atol)
            all_results.append(r)
            print(_SEP_THIN)
            m = r.metrics
            print(f"  [{r.name}]  shape={m.get('shape','—')}  active={m.get('active','—')}  "
                  f"abs_max={m.get('abs_max',0):.6e}  rel_max={m.get('rel_max',0):.6e}  "
                  f"pearson={m.get('pearson_r',float('nan')):.8f}")
            print(f"  abs_mean={m.get('abs_mean',0):.6e}  rel_mean={m.get('rel_mean',0):.6e}  "
                  f"{_CHECK if r.passed else _CROSS} {'PASS' if r.passed else 'FAIL'}")
            print()

    # ── ⑥ OFF vs OFF baseline ──
    if args.dir_off2:
        print(_SEP_DOUBLE + "\n  [BASELINE]  OFF vs OFF2 Noise Floor\n" + _SEP_DOUBLE)
        # position_ids baseline
        rb = cmp_position_ids(args.dir_off, args.dir_off2)
        if rb:
            rb.name = "bl_pos_ids"
            print(f"  bl_pos_ids: {_CHECK if rb.passed else _CROSS} {'PASS' if rb.passed else 'FAIL'}\n")
            all_results.append(rb)
        # 2D baseline
        if tag_on:
            for fname, cname in [("entropy", "entropy"), ("logprobs", "logp")]:
                fn = f"{fname}_{tag_on}.pt"
                r = cmp_2d(args.dir_off, args.dir_off2, fn,
                           f"bl_{cname}_{tag_on}", label_mask, args.atol)
                all_results.append(r)
                m = r.metrics
                print(_SEP_THIN)
                print(f"  [{r.name}]  active={m.get('active','—')}  "
                      f"abs_max={m.get('abs_max',0):.6e}  rel_max={m.get('rel_max',0):.6e}  "
                      f"pearson={m.get('pearson_r',float('nan')):.8f}")
                print()

    # ── Summary ──
    _print_summary(all_results)

    if args.output:
        _dump_json(all_results, args.output, args.dir_on, args.dir_off,
                   tag_on, args.dir_off2)


if __name__ == "__main__":
    main()
