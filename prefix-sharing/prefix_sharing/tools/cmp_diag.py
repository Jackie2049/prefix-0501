"""Unified precision comparison — ON vs OFF diagnostic dumps.

Auto-detects available dump files and selects the right comparison strategy:

- Packed (attn_output.pt / logits.pt): cu_seqlens + prefix_lens row mapping,
  cosine similarity, dim buckets, verdict.

- 2D (logprobs_{tag}.pt / entropy_{tag}.pt): label_mask → Pearson r, top-N.

- 3D (logits_{tag}.pt): element-wise + per-position max-over-vocab diff.

Usage:
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --tag train
    python cmp_diag.py --dir-on ./dump_on --dir-off ./dump_off --output report.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

import torch

# ── Constants ──────────────────────────────────────────────────
_SEP_DOUBLE = "=" * 70
_SEP_SINGLE = "-" * 70
_SEP_THIN = "─" * 70
CHECK = "\u2713"
CROSS = "\u2717"
TOP_N = 20


def _fmt_shape(s: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in s) + ")"


# ── Packed: load & compare ─────────────────────────────────────

def _load_packed(dir_path: str, output_file: str,
                 cu_fname: str = "cu_seqlens_q.pt") -> dict | None:
    """Load a packed-format dump. Returns None if file missing.

    cu_fname: metadata filename (e.g. "cu_seqlens_q.pt" for attn,
    "cu_seqlens_q_logits.pt" for logits). Falls back to "cu_seqlens_q.pt"
    if the specific file is not found.
    """
    fp = os.path.join(dir_path, output_file)
    if not os.path.exists(fp):
        return None
    # Try specific cu_seqlens file, fallback to default
    cu_fp = os.path.join(dir_path, cu_fname)
    if not os.path.exists(cu_fp):
        cu_fp = os.path.join(dir_path, "cu_seqlens_q.pt")
        if not os.path.exists(cu_fp):
            return None
    pl_fp = os.path.join(dir_path, "prefix_lens.pt")
    if not os.path.exists(pl_fp):
        return None
    return {
        "output": torch.load(fp, weights_only=True).float(),
        "cu_seqlens": torch.load(cu_fp, weights_only=True),
        "prefix_lens": torch.load(pl_fp, weights_only=True),
    }


def _diagnose_packed(a: torch.Tensor, b: torch.Tensor, prefix_len: int,
                     max_tokens: int = 3) -> bool:
    """Print per-token diagnostics for packed tensors. Returns True if clean."""
    n_tokens, d = a.shape
    n_show = min(n_tokens, max_tokens)

    diff_abs = (a - b).abs()                                     # [N, D]
    token_max_diff, _ = diff_abs.max(dim=-1)                     # [N]
    worst_indices = token_max_diff.topk(min(n_show, n_tokens)).indices

    eps = 1e-8
    a_norm = a.norm(dim=-1) + eps
    b_norm = b.norm(dim=-1) + eps
    dot = (a * b).sum(dim=-1)
    cos_sim = dot / (a_norm * b_norm)

    rel_err = diff_abs / (b.abs() + eps)

    thresholds = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    dim_buckets = {t: (diff_abs > t).sum(dim=-1).float() for t in thresholds}

    all_cos = cos_sim.tolist()
    avg_cos = sum(all_cos) / len(all_cos)
    min_cos = min(all_cos)
    avg_max = sum(token_max_diff.tolist()) / len(token_max_diff.tolist())
    avg_mean_rel = sum(rel_err.mean(dim=-1).tolist()) / len(all_cos)
    total_dims = n_tokens * d

    print(f"  n_tokens={n_tokens} dim={d}")
    print(f"  cosine_sim:    avg={avg_cos:.6f}  min={min_cos:.6f}")
    print(f"  max_abs_diff:   avg={avg_max:.6e}  max={max(token_max_diff.tolist()):.6e}")
    print(f"  mean_rel_err:   avg={avg_mean_rel:.6e}")
    print(f"  dim_frac >1e-5: {(dim_buckets[1e-5].sum().item())/total_dims:.4f}",
          f">1e-4: {(dim_buckets[1e-4].sum().item())/total_dims:.4f}",
          f">1e-3: {(dim_buckets[1e-3].sum().item())/total_dims:.4f}",
          f">1e-2: {(dim_buckets[1e-2].sum().item())/total_dims:.4f}",
          f">1e-1: {(dim_buckets[1e-1].sum().item())/total_dims:.4f}")

    for rank, idx in enumerate(worst_indices.tolist()):
        abs_pos = prefix_len + idx
        cos = cos_sim[idx].item()
        max_d = token_max_diff[idx].item()
        mean_rel = rel_err[idx].mean().item()
        bucket_str = " ".join(
            f">{t}={dim_buckets[t][idx].item():.0f}/{d}"
            for t in thresholds[:4]
        )
        print(f"  worst#{rank+1} abs_pos={abs_pos} cos={cos:.6f} "
              f"max_diff={max_d:.4e} mean_rel={mean_rel:.4e}  [{bucket_str}]")

    # Verdict
    is_clean = avg_cos > 0.9999 and min_cos > 0.999
    if is_clean:
        print(f"  VERDICT: PRECISION NOISE (cosine ~1, likely bf16/fp32 rounding)")
    elif avg_cos > 0.99:
        print(f"  VERDICT: SUSPICIOUS (small systematic deviation)")
    else:
        print(f"  VERDICT: COMPUTATION ERROR (fundamentally wrong output)")
    return is_clean


def compare_packed(
    dir_on: str, dir_off: str, output_file: str, name: str,
    atol: float, rtol: float = 1e-5,
) -> dict | None:
    """Compare a packed-format dump (attn_output.pt or logits.pt)."""
    # Use dump-type-specific cu_seqlens filename (with fallback)
    cu_fname = "cu_seqlens_q_logits.pt" if name == "logits" else "cu_seqlens_q.pt"
    on = _load_packed(dir_on, output_file, cu_fname)
    off = _load_packed(dir_off, output_file, cu_fname)
    if on is None or off is None:
        return None

    on_out = on["output"]
    off_out = off["output"]
    cu_on = on["cu_seqlens"]
    cu_off = off["cu_seqlens"]
    pl = on["prefix_lens"]

    batch = min(cu_on.shape[0], cu_off.shape[0]) - 1
    is_attn = on_out.dim() == 3  # [N, 1, hidden]

    total_tokens = 0
    global_max_diff = 0.0
    ok_rows = 0
    fail_rows = 0

    for i in range(batch):
        pf = int(pl[i])
        if pf <= 0:
            continue

        # Compute suffix lengths independently from each side's cu_seqlens
        on_suffix_len = int(cu_on[i + 1] - cu_on[i])
        off_full_len = int(cu_off[i + 1] - cu_off[i])
        off_suffix_len = max(0, off_full_len - pf)

        on_start = int(cu_on[i])
        off_start = int(cu_off[i]) + pf

        # Validate ON bounds
        if on_start < 0 or on_start + on_suffix_len > on_out.shape[0]:
            print(f"[WARN] {name} row[{i}] prefix_len={pf}: ON out of bounds, skip")
            continue
        # Validate OFF bounds
        if off_start < 0 or off_start + off_suffix_len > off_out.shape[0]:
            print(f"[WARN] {name} row[{i}] prefix_len={pf}: OFF out of bounds, skip")
            continue
        # Validate suffix lengths match
        if on_suffix_len != off_suffix_len:
            print(f"[WARN] {name} row[{i}] prefix_len={pf}: "
                  f"suffix len mismatch (ON={on_suffix_len}, OFF={off_suffix_len}), skip")
            continue
        if on_suffix_len == 0:
            print(f"[WARN] {name} row[{i}] prefix_len={pf}: zero suffix tokens, skip")
            continue

        suffix_len = on_suffix_len

        if is_attn:
            a = on_out[on_start:on_start + suffix_len, 0, :]
            b = off_out[off_start:off_start + suffix_len, 0, :]
        else:
            a = on_out[on_start:on_start + suffix_len, :]
            b = off_out[off_start:off_start + suffix_len, :]

        diff_abs = (a - b).abs()
        row_max = diff_abs.max().item()
        row_mismatch = (~torch.isclose(a, b, rtol=rtol, atol=atol)).any(dim=-1)
        n_mismatch = row_mismatch.sum().item()

        if n_mismatch > 0:
            fail_rows += 1
            global_max_diff = max(global_max_diff, row_max)
            print(f"row[{i}] prefix_len={pf}: "
                  f"{n_mismatch}/{suffix_len} tokens exceed atol={atol}"
                  f" (max_abs_diff={row_max:.6e})")
            _diagnose_packed(a, b, pf)
        else:
            ok_rows += 1
            print(f"row[{i}] prefix_len={pf}: ALL MATCH ({suffix_len} tokens)")

        total_tokens += suffix_len

    print()
    print(_SEP_DOUBLE)
    print(f"  [{name}] ROWS: {ok_rows} OK, {fail_rows} FAIL")
    print(f"  TOTAL_SUFFIX_TOKENS: {total_tokens}")
    print(f"  MAX_ABS_DIFF: {global_max_diff:.6e}")
    print(_SEP_DOUBLE)

    return {
        "name": name, "ok_rows": ok_rows, "fail_rows": fail_rows,
        "total_tokens": total_tokens, "max_abs_diff": global_max_diff,
        "passed": fail_rows == 0,
    }


# ── 2D: load & compare (reuses compare_logprobs.py logic) ────────

@dataclass
class CompareResult:
    name: str
    shape: tuple[int, ...] = ()
    max_diff: float = float("nan")
    mean_diff: float = float("nan")
    n_mismatch: int = -1
    n_total: int | None = None
    passed: bool = False
    shape_on: tuple[int, ...] | None = None
    shape_off: tuple[int, ...] | None = None
    pearson_r: float | None = None

    @property
    def shape_mismatch(self) -> bool:
        return self.shape_on is not None and self.shape_off is not None


def _load_label_mask(dir_path: str) -> torch.Tensor | None:
    path = os.path.join(dir_path, "label_mask.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).to(torch.bool)


def _load_label(dir_path: str) -> torch.Tensor | None:
    path = os.path.join(dir_path, "label.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path, weights_only=True).float()


def _load_2d_tensor(dir_path: str, filename: str) -> torch.Tensor | None:
    fp = os.path.join(dir_path, filename)
    if not os.path.exists(fp):
        return None
    return torch.load(fp, weights_only=True).float()


def _pearson_r(t1: torch.Tensor, t2: torch.Tensor, mask: torch.Tensor) -> float:
    x = t1[mask].to(torch.float64)
    y = t2[mask].to(torch.float64)
    n = x.numel()
    if n < 2:
        return float("nan")
    mean_x = x.mean()
    mean_y = y.mean()
    cov = ((x - mean_x) * (y - mean_y)).sum()
    std_x = ((x - mean_x) ** 2).sum().sqrt()
    std_y = ((y - mean_y) ** 2).sum().sqrt()
    if std_x == 0 or std_y == 0:
        return float("nan")
    return float(cov / (std_x * std_y))


def _make_2d_result(name: str, diff: torch.Tensor, mask: torch.Tensor,
                    atol: float, t1: torch.Tensor | None = None,
                    t2: torch.Tensor | None = None) -> CompareResult:
    n_mismatch = int((diff > atol).sum().item())
    max_diff = float(diff[mask].max().item()) if mask.any() else 0.0
    mean_diff = float(diff[mask].mean().item()) if mask.any() else 0.0
    n_total = int(mask.sum().item())
    pr = _pearson_r(t1, t2, mask) if t1 is not None and t2 is not None else None
    return CompareResult(
        name=name, shape=tuple(diff.shape),
        max_diff=max_diff, mean_diff=mean_diff,
        n_mismatch=n_mismatch, n_total=n_total,
        passed=(n_mismatch == 0), pearson_r=pr,
    )


def _print_2d_detail(
    name: str, t1: torch.Tensor, t2: torch.Tensor,
    diff: torch.Tensor, mask: torch.Tensor, atol: float,
    label: torch.Tensor | None,
):
    B, L = t1.shape
    active = int(mask.sum().item())
    n_mismatch = int((diff > atol).sum().item())
    max_diff = float(diff[mask].max().item()) if mask.any() else 0.0
    mean_diff = float(diff[mask].mean().item()) if mask.any() else 0.0
    pr = _pearson_r(t1, t2, mask)

    print(_SEP_THIN)
    print(f"─── {name}  shape=({B}, {L})  active_tokens={active}  pearson_r={pr:.8f}")
    print(f"    max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")
    print(_SEP_THIN)

    if n_mismatch == 0:
        print(f"  {CHECK} ALL MATCH")
        print()
        return

    mismatch_pos = (diff > atol).nonzero(as_tuple=False)
    n_total_mismatch = mismatch_pos.size(0)
    mismatch_vals = diff[mismatch_pos[:, 0], mismatch_pos[:, 1]]
    sorted_order = mismatch_vals.argsort(descending=True)
    top_indices = mismatch_pos[sorted_order[:TOP_N]]
    n_show = min(TOP_N, n_total_mismatch)

    print(f"  {CROSS} 差异最大的 {n_show}/{n_total_mismatch} 个不匹配位置 (按 diff 降序):")
    print(f"  {'POS':>10s}  {'TOKEN':>10s}  {'ON':>14s}  {'OFF':>14s}  {'DIFF':>14s}  {'REL%':>10s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}")

    for idx in top_indices:
        b, p = idx[0].item(), idx[1].item()
        tok = str(int(label[b, p].item())) if label is not None else "—"
        on_val = t1[b, p].item()
        off_val = t2[b, p].item()
        d = diff[b, p].item()
        rel = abs(d / max(abs(on_val), abs(off_val), 1e-8)) * 100
        print(f"  [{b:>4d},{p:>4d}]  {tok:>10s}  {on_val:>14.6f}  {off_val:>14.6f}  {d:>14.6e}  {rel:>9.2f}%")

    per_row = (diff > atol).sum(dim=1)
    print(f"\n  Row breakdown:")
    for b in range(B):
        n_row = int(per_row[b].item())
        if n_row > 0:
            row_max = float(diff[b].max().item())
            row_max_pos = int(diff[b].argmax().item())
            lh = f" token={int(label[b, row_max_pos].item())}" if label is not None else ""
            print(f"    row[{b}]: {n_row}/{L} mismatches  (max_diff={row_max:.6e} @ col={row_max_pos}{lh})")
    print()


def compare_2d_item(
    dir_on: str, dir_off: str, filename: str, name: str,
    atol: float, label_mask: torch.Tensor | None,
    label: torch.Tensor | None,
) -> CompareResult | None:
    """Compare a single 2D tensor file between on/off dumps."""
    t1 = _load_2d_tensor(dir_on, filename)
    t2 = _load_2d_tensor(dir_off, filename)
    if t1 is None or t2 is None:
        return None
    if t1.shape != t2.shape:
        return CompareResult(name=name, shape_on=tuple(t1.shape),
                             shape_off=tuple(t2.shape))

    compare_mask = label_mask
    if compare_mask is None or compare_mask.shape != t1.shape:
        compare_mask = (t2 != 0)
    diff, mask = (t1 - t2).abs(), torch.logical_and(
        compare_mask.to(diff.device), diff == diff)
    return _make_2d_result(name, diff, mask, atol, t1=t1, t2=t2)


def compare_2d(
    dir_on: str, dir_off: str, tag: str, atol: float,
) -> tuple[list[CompareResult], dict]:
    """Compare all 2D dumps with the given tag."""
    results: list[CompareResult] = []
    detail_data: list[dict] = []

    label_mask = _load_label_mask(dir_on)
    label = _load_label(dir_on)
    if label is not None:
        print(f"label  shape={_fmt_shape(tuple(label.shape))}")
        print()

    # entropy
    r = compare_2d_item(dir_on, dir_off, f"entropy_{tag}.pt",
                        f"entropy_{tag}", atol, label_mask, label)
    if r is not None:
        results.append(r)
        if not r.shape_mismatch:
            t1 = _load_2d_tensor(dir_on, f"entropy_{tag}.pt")
            t2 = _load_2d_tensor(dir_off, f"entropy_{tag}.pt")
            cm = label_mask if label_mask is not None and label_mask.shape == t1.shape else (t2 != 0)
            diff, mask = (t1 - t2).abs(), torch.logical_and(cm.to(t1.device), (t1 - t2).abs() == (t1 - t2).abs())
            detail_data.append({"type": "2d", "name": f"entropy_{tag}",
                                "t1": t1, "t2": t2, "diff": diff, "mask": mask,
                                "atol": atol, "label": label, "result": r})

    # logprobs
    r = compare_2d_item(dir_on, dir_off, f"logprobs_{tag}.pt",
                        f"logprobs_{tag}", atol, label_mask, label)
    if r is None:
        r = compare_2d_item(dir_on, dir_off, "logprobs.pt",
                            "logprobs", atol, label_mask, label)
    if r is not None:
        results.append(r)
        if not r.shape_mismatch:
            fp = f"logprobs_{tag}.pt" if os.path.exists(os.path.join(dir_on, f"logprobs_{tag}.pt")) else "logprobs.pt"
            t1 = _load_2d_tensor(dir_on, fp)
            t2 = _load_2d_tensor(dir_off, fp)
            cm = label_mask if label_mask is not None and label_mask.shape == t1.shape else (t2 != 0)
            diff, mask = (t1 - t2).abs(), torch.logical_and(cm.to(t1.device), (t1 - t2).abs() == (t1 - t2).abs())
            detail_data.append({"type": "2d", "name": fp.replace(".pt",""),
                                "t1": t1, "t2": t2, "diff": diff, "mask": mask,
                                "atol": atol, "label": label, "result": r})

    # logits (3D)
    t1 = _load_2d_tensor(dir_on, f"logits_{tag}.pt")
    t2 = _load_2d_tensor(dir_off, f"logits_{tag}.pt")
    if t1 is not None and t2 is not None and t1.dim() == 3:
        if t1.shape != t2.shape:
            r = CompareResult(name=f"logits_{tag}", shape_on=tuple(t1.shape),
                              shape_off=tuple(t2.shape))
        else:
            diff = (t1 - t2).abs()
            mask = torch.ones_like(diff, dtype=torch.bool)
            r = _make_2d_result(f"logits_{tag}", diff, mask, atol,
                                t1=t1, t2=t2)
            delta_per_pos = diff.max(dim=-1).values
            detail_data.append({"type": "3d", "name": f"logits_{tag}",
                                "t1": t1, "t2": t2, "diff": diff,
                                "diff_per_pos": delta_per_pos, "mask": mask,
                                "atol": atol, "label": label, "result": r})
        results.append(r)

    return results, {"detail_data": detail_data, "label": label}


# ── 3D detail print ────────────────────────────────────────────

def _print_3d_detail(
    name: str, t1: torch.Tensor, t2: torch.Tensor,
    diff: torch.Tensor, diff_per_pos: torch.Tensor, atol: float,
    label: torch.Tensor | None,
):
    B, L, V = t1.shape
    n_total = int(diff.numel())
    n_mismatch = int((diff > atol).sum().item())
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    pr = _pearson_r(t1, t2, torch.ones_like(diff, dtype=torch.bool))
    mismatched_positions = int((diff_per_pos > atol).sum().item())

    print(_SEP_THIN)
    print(f"─── {name}  shape=({B}, {L}, {V})  total_elements={n_total}  pearson_r={pr:.8f}")
    print(f"    element-wise:  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")
    print(f"    per-position (max over vocab):  {mismatched_positions}/{B*L} positions affected")
    print(_SEP_THIN)

    if n_mismatch == 0:
        print(f"  {CHECK} ALL MATCH")
        print()
        return

    n_show_elem = min(TOP_N, n_mismatch)
    flat_top = diff.view(-1).topk(n_show_elem)
    print(f"  {CROSS} 差异最大的 {flat_top.values.size(0)} 个元素, 按 diff 降序:")
    print(f"  {'POS':>20s}  {'TOKEN':>10s}  {'VOCAB':>10s}  {'ON':>14s}  {'OFF':>14s}  {'DIFF':>14s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}")
    for val, flat_idx in zip(flat_top.values, flat_top.indices):
        b = int(flat_idx // (L * V))
        rest = int(flat_idx % (L * V))
        l_idx = int(rest // V)
        v = int(rest % V)
        tok = str(int(label[b, l_idx].item())) if label is not None else "—"
        print(f"  [{b:>4d},{l_idx:>4d},{v:>4d}]  {tok:>10s}  {v:>10d}  "
              f"{t1[b,l_idx,v]:>14.6f}  {t2[b,l_idx,v]:>14.6f}  {val:>14.6e}")

    n_show_pos = min(TOP_N, B * L)
    top_pos = diff_per_pos.view(-1).topk(n_show_pos)
    print(f"\n  跨 vocab 差异最大的 {n_show_pos} 个 (b, l) 位置:")
    print(f"  {'POS':>10s}  {'TOKEN':>10s}  {'MAX_DIFF':>16s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*16}")
    for val, flat_idx in zip(top_pos.values, top_pos.indices):
        b = int(flat_idx // L)
        l_idx = int(flat_idx % L)
        tok = str(int(label[b, l_idx].item())) if label is not None else "—"
        print(f"  [{b:>4d},{l_idx:>4d}]  {tok:>10s}  {val:>16.6e}")
    print()


# ── Summary table ───────────────────────────────────────────────

def _print_summary(all_results: list[dict], atol: float):
    """Print a unified summary table from all comparison results."""
    print(_SEP_DOUBLE)
    print(f"  SUMMARY (atol={atol:.1e})")
    print(_SEP_DOUBLE)
    hdr = (f"  {'NAME':<24s} {'SHAPE':<20s} {'MAX_DIFF':>14s}  "
           f"{'MISMATCH':>12s}  {'PEARSON_R':>10s}  {'STATUS':>8s}")
    print(hdr)
    print(f"  {'─'*24} {'─'*20} {'─'*14}  {'─'*12}  {'─'*10}  {'─'*8}")

    for r in all_results:
        status = f"  {CHECK} PASS" if r["passed"] else f"  {CROSS} FAIL"
        mm = f"{r.get('n_mismatch', '—')}/{r.get('n_total', '—')}"
        pr = f"{r.get('pearson_r', float('nan')):>10.6f}" if 'pearson_r' in r and r['pearson_r'] is not None else "        —"
        print(f"  {r['name']:<24s} {r.get('shape_str', '—'):<20s} "
              f"{r.get('max_diff', float('nan')):>14.6e}  {mm:>12s}  {pr}  {status}")
    print(_SEP_DOUBLE)
    print()


# ── JSON output ─────────────────────────────────────────────────

def _dump_json(all_results: list[dict], output_path: str, atol: float,
               dir_on: str, dir_off: str, tag: str | None):
    record = {
        "atol": atol,
        "dir_on": dir_on,
        "dir_off": dir_off,
        "tag": tag,
        "results": [
            {k: v for k, v in r.items()
             if not isinstance(v, torch.Tensor)}
            for r in all_results
        ],
        "all_passed": all(r["passed"] for r in all_results),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"  Precision report saved to: {output_path}")
    print()


# ── Main ────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Unified precision comparison — ON vs OFF diagnostic dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--dir-on", required=True, help="ON dump directory")
    ap.add_argument("--dir-off", required=True, help="OFF dump directory")
    ap.add_argument("--tag", choices=["old", "train"], default=None,
                    help="Dump tag for 2D files (old = forward_only, train = training)")
    ap.add_argument("--atol", type=float, default=1e-3,
                    help="Absolute tolerance (default: 1e-3 for packed, 1e-5 for 2D)")
    ap.add_argument("--rtol", type=float, default=1e-5,
                    help="Relative tolerance (packed only)")
    ap.add_argument("--output", "-o", type=str, default=None,
                    help="Save report as JSON")
    args = ap.parse_args()

    dir_on = args.dir_on
    dir_off = args.dir_off

    print(_SEP_DOUBLE)
    print(f"  Prefix-Sharing Diag Report")
    print(f"  ON : {dir_on}")
    print(f"  OFF: {dir_off}")
    if args.tag:
        print(f"  TAG: {args.tag}")
    print(_SEP_DOUBLE)
    print()

    all_results: list[dict] = []
    detail_2d = None  # will be set by compare_2d

    # 1. Try packed dumps (attn_output.pt, logits.pt)
    for fname, label in [("attn_output.pt", "attn_output"),
                          ("logits.pt", "logits")]:
        r = compare_packed(dir_on, dir_off, fname, label, args.atol, args.rtol)
        if r is not None:
            all_results.append(r)

    # 2. Try 2D dumps with --tag
    if args.tag:
        results_2d, meta = compare_2d(dir_on, dir_off, args.tag, args.atol)
        detail_2d = meta
        for r in results_2d:
            all_results.append({
                "name": r.name,
                "shape_str": _fmt_shape(r.shape) if r.shape else "mismatch",
                "max_diff": r.max_diff,
                "mean_diff": r.mean_diff,
                "n_mismatch": r.n_mismatch,
                "n_total": r.n_total,
                "pearson_r": r.pearson_r,
                "passed": r.passed,
            })

    # 3. Legacy (no --tag): try logprobs.pt + input_ids.pt
    else:
        tag = None
        fp = os.path.join(dir_on, "logprobs.pt")
        if os.path.exists(fp):
            tag = "legacy"
            r = compare_2d_item(dir_on, dir_off, "logprobs.pt", "logprobs",
                                args.atol, None, None)
            if r is not None:
                all_results.append({
                    "name": r.name,
                    "shape_str": _fmt_shape(r.shape) if r.shape else "mismatch",
                    "max_diff": r.max_diff,
                    "mean_diff": r.mean_diff,
                    "n_mismatch": r.n_mismatch,
                    "n_total": r.n_total,
                    "pearson_r": r.pearson_r,
                    "passed": r.passed,
                })
            # Also try logprobs_{train/old}.pt
            for try_tag in ["old", "train"]:
                fp2 = os.path.join(dir_on, f"logprobs_{try_tag}.pt")
                if os.path.exists(fp2):
                    r = compare_2d_item(dir_on, dir_off,
                                        f"logprobs_{try_tag}.pt",
                                        f"logprobs_{try_tag}",
                                        args.atol, None, None)
                    if r is not None:
                        all_results.append({
                            "name": r.name,
                            "shape_str": _fmt_shape(r.shape) if r.shape else "mismatch",
                            "max_diff": r.max_diff,
                            "mean_diff": r.mean_diff,
                            "n_mismatch": r.n_mismatch,
                            "n_total": r.n_total,
                            "pearson_r": r.pearson_r,
                            "passed": r.passed,
                        })

    if not all_results:
        print("No dump files found in either directory.")
        return

    # Print summary table
    _print_summary(all_results, args.atol)

    # Print details for 2D/3D dumps
    if detail_2d is not None:
        for d in detail_2d["detail_data"]:
            r = d["result"]
            if d["type"] == "2d" and not r.passed:
                _print_2d_detail(d["name"], d["t1"], d["t2"],
                                 d["diff"], d["mask"], d["atol"],
                                 d["label"])
            elif d["type"] == "3d" and not r.passed:
                _print_3d_detail(d["name"], d["t1"], d["t2"],
                                 d["diff"], d["diff_per_pos"],
                                 d["atol"], d["label"])

    # JSON output
    if args.output:
        _dump_json(all_results, args.output, args.atol,
                   dir_on, dir_off, args.tag)


if __name__ == "__main__":
    main()
