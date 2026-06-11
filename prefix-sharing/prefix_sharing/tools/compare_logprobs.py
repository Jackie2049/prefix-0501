"""
端到端精度对比工具——对比开启/关闭 prefix-sharing 两遍独立运行的 dump。

支持三种 dump 类型，通过 --tag 切换：
  --tag old    : 对比 logits_old.pt + entropy_old.pt + label.pt  （推理前向，forward_only=True）
  --tag train  : 对比 logits_train.pt + entropy_train.pt + label.pt（训练前向，forward_only=False）
  (不设 --tag) : 对比 logprobs.pt + input_ids.pt（legacy，向前兼容）

--tag 模式下若目录中存在 logprobs.pt，也会自动加载一并对比。

完整流程：
  # 第一遍：开启 prefix-sharing
  ENABLE_PREFIX_SHARING=1 PREFIX_SHARING_DUMP_DIR=./dump_on python -m verl.trainer.main_ppo ...

  # 第二遍：关闭 prefix-sharing（其余参数完全相同）
  ENABLE_PREFIX_SHARING=0 PREFIX_SHARING_DUMP_DIR=./dump_off python -m verl.trainer.main_ppo ...

  # 对比
  python compare_logprobs.py -1 ./dump_on -2 ./dump_off --tag train
  python compare_logprobs.py -1 ./dump_on -2 ./dump_off --tag old --atol 1e-5
  python compare_logprobs.py -1 ./dump_on -2 ./dump_off   # legacy logprobs.pt
"""

import argparse
import os
import pathlib
import sys

import torch


# ──────────────────────────────────────────────
# 公共工具
# ──────────────────────────────────────────────

def cast_to_dtype(t: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """将 tensor 转成目标精度再回升至 float32，模拟混合精度下的量化误差。"""
    if dtype_str == "bfloat16":
        return t.to(torch.bfloat16).to(torch.float32)
    elif dtype_str == "float16":
        return t.to(torch.float16).to(torch.float32)
    elif dtype_str == "float32":
        return t
    else:
        sys.exit(f"error: unsupported dtype: {dtype_str}")


def load_tensor(filepath: str, dtype: str | None = None) -> torch.Tensor:
    """加载单个 tensor 文件，可选精度转换。"""
    if not os.path.exists(filepath):
        sys.exit(f"error: file not found: {filepath}")
    t = torch.load(filepath, weights_only=True)
    if dtype:
        t = cast_to_dtype(t, dtype)
    return t.float()


# ──────────────────────────────────────────────
# Legacy: logprobs.pt + input_ids.pt
# ──────────────────────────────────────────────

def load_legacy_dump(dump_dir: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """加载 legacy dump: (logprobs, input_ids)。"""
    dp = pathlib.Path(dump_dir)
    if not dp.exists():
        sys.exit(f"error: dump directory not found: {dump_dir}")

    lp_path = dp / "logprobs.pt"
    if lp_path.exists():
        lp = torch.load(str(lp_path), weights_only=True)
        ids_path = dp / "input_ids.pt"
        ids = torch.load(str(ids_path), weights_only=True) if ids_path.exists() else None
        return lp, ids

    # 多 rank 模式
    lps = {}
    idss = {}
    for fname in sorted(os.listdir(dump_dir)):
        if fname.startswith("logprobs_rank"):
            rank = int(fname.split("rank")[1].split(".")[0])
            lps[rank] = torch.load(os.path.join(dump_dir, fname), weights_only=True)
        elif fname.startswith("input_ids_rank"):
            rank = int(fname.split("rank")[1].split(".")[0])
            idss[rank] = torch.load(os.path.join(dump_dir, fname), weights_only=True)
    if not lps:
        sys.exit(f"error: no logprobs dump found in {dump_dir}")
    lp = torch.cat([lps[r] for r in sorted(lps.keys())], dim=0)
    ids = torch.cat([idss[r] for r in sorted(idss.keys())], dim=0) if idss else None
    return lp, ids


# ──────────────────────────────────────────────
# 对比结果结构
# ──────────────────────────────────────────────

class CompareResult:
    """单个对比项的结果汇总。"""
    def __init__(self, name: str, shape: tuple[int, ...], max_diff: float, mean_diff: float,
                 n_mismatch: int, n_total: int | None = None, passed: bool = False,
                 *, shape_on: tuple[int, ...] | None = None, shape_off: tuple[int, ...] | None = None):
        self.name = name
        self.shape = shape  # 如果 shape 一致则存公共 shape；不一致则存空 tuple
        self.max_diff = max_diff
        self.mean_diff = mean_diff
        self.n_mismatch = n_mismatch
        self.n_total = n_total
        self.passed = passed
        self.shape_on = shape_on
        self.shape_off = shape_off

    @property
    def shape_mismatch(self) -> bool:
        return self.shape_on is not None and self.shape_off is not None

    @staticmethod
    def from_2d_diff(name: str, diff: torch.Tensor, mask: torch.Tensor, atol: float,
                     n_total: int | None = None) -> "CompareResult":
        n_mismatch = (diff > atol).sum().item()
        max_diff = diff[mask].max().item() if mask.any() else 0.0
        mean_diff = diff[mask].mean().item() if mask.any() else 0.0
        return CompareResult(
            name=name, shape=tuple(diff.shape),
            max_diff=max_diff, mean_diff=mean_diff,
            n_mismatch=n_mismatch, n_total=n_total or int(mask.sum().item()),
            passed=(n_mismatch == 0),
        )

    @staticmethod
    def shape_mismatch_result(name: str, shape_on: tuple[int, ...], shape_off: tuple[int, ...]) -> "CompareResult":
        """创建 shape 不匹配的结果。"""
        return CompareResult(
            name=name, shape=(),
            max_diff=float("nan"), mean_diff=float("nan"),
            n_mismatch=-1, n_total=None, passed=False,
            shape_on=shape_on, shape_off=shape_off,
        )


def _compute_2d_diff(t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 2D 张量的绝对差异和有效 mask。"""
    diff = (t1 - t2).abs()
    mask = (t1 != 0) | (t2 != 0)
    return diff, mask


def _compute_3d_diff(t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 3D 张量的绝对差异和有效 mask（全量参与）。"""
    diff = (t1 - t2).abs()
    mask = torch.ones_like(diff, dtype=torch.bool)
    return diff, mask


# ──────────────────────────────────────────────
# 输出工具
# ──────────────────────────────────────────────

_SEP_DOUBLE = "=" * 70
_SEP_SINGLE = "-" * 70
_SEP_THIN   = "─" * 70

CHECK = "✓"
CROSS = "✗"


def _fmt_shape(s: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(x) for x in s) + ")"


def _print_header(tag: str, dir1: str, dir2: str):
    print(_SEP_DOUBLE)
    print(f"  Prefix-Sharing Precision Report  —  tag={tag}")
    print(f"  ON : {dir1}")
    print(f"  OFF: {dir2}")
    print(_SEP_DOUBLE)
    print()


def _print_summary_table(results: list[CompareResult], atol: float):
    """横向对比表格：一眼看出谁通过、谁异常。"""
    print(_SEP_DOUBLE)
    print(f"  SUMMARY  (atol={atol:.1e})")
    print(_SEP_DOUBLE)
    print(f"  {'NAME':<20s} {'SHAPE (ON / OFF)':<34s} {'MAX_DIFF':>14s}  {'MEAN_DIFF':>14s}  {'MISMATCHES':>12s}  {'STATUS':>8s}")
    print(f"  {'─'*20} {'─'*34} {'─'*14}  {'─'*14}  {'─'*12}  {'─'*8}")

    for r in results:
        if r.shape_mismatch:
            status = f"  {CROSS} SHAPE"
            shape_str = f"{_fmt_shape(r.shape_on)} / {_fmt_shape(r.shape_off)}"
            print(f"  {r.name:<20s} {shape_str:<34s} {'—':>14s}  {'—':>14s}  {'—':>12s}  {status}")
        else:
            status = f"  {CHECK} PASS" if r.passed else f"  {CROSS} FAIL"
            n_str = f"{r.n_mismatch}/{r.n_total}" if r.n_total else str(r.n_mismatch)
            print(f"  {r.name:<20s} {_fmt_shape(r.shape):<34s} {r.max_diff:>14.6e}  {r.mean_diff:>14.6e}  {n_str:>12s}  {status}")
    print(_SEP_DOUBLE)
    print()


def _print_pass(name: str, shape: tuple[int, ...], n_active: int):
    """单条目全对通过的简短输出。"""
    print(f"─── {name}  shape={_fmt_shape(shape)}  active_tokens={n_active} ───")
    print(f"  {CHECK} ALL MATCH — consistent with baseline")
    print()


def _print_shape_mismatch(name: str, shape_on: tuple[int, ...], shape_off: tuple[int, ...],
                          t1: torch.Tensor, t2: torch.Tensor):
    """打印 shape 不匹配的诊断信息。"""
    print(_SEP_THIN)
    print(f"─── {name}  SHAPE MISMATCH ───")
    print(f"    sharing ON  shape: {_fmt_shape(shape_on)}   (packed / trimmed)")
    print(f"    sharing OFF shape: {_fmt_shape(shape_off)}   (full unpadded)")
    print(f"    ON/OFF ratio: {shape_on[1] / shape_off[1]:.4f}x  (seq dim)")
    print()
    print(f"  {CROSS} Cannot compare element-wise — prefix-sharing changes sequence layout.")
    print(f"     This is EXPECTED: packing trims redundant prefix tokens, reducing seq_len.")
    print(f"     To verify correctness, run with identical seq_len (e.g. no packing)")
    print(f"     or compare per-token values after unpacking to the original layout.")
    print(f"  ON  stats:  min={t1.min().item():.4f}  max={t1.max().item():.4f}  mean={t1.mean().item():.4f}")
    print(f"  OFF stats:  min={t2.min().item():.4f}  max={t2.max().item():.4f}  mean={t2.mean().item():.4f}")
    print()


def _print_2d_detail(
    name: str, t1: torch.Tensor, t2: torch.Tensor,
    diff: torch.Tensor, mask: torch.Tensor, atol: float,
    label_ids: torch.Tensor | None,
):
    """打印 2D 张量的详细不匹配信息。"""
    B, L = t1.shape
    active = mask.sum().item()
    n_mismatch = (diff > atol).sum().item()
    max_diff = diff[mask].max().item() if mask.any() else 0.0
    mean_diff = diff[mask].mean().item() if mask.any() else 0.0

    print(_SEP_THIN)
    print(f"─── {name}  shape=({B}, {L})  active_tokens={active} ───")
    print(f"    max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")
    print(_SEP_THIN)

    if n_mismatch == 0:
        print(f"  {CHECK} ALL MATCH")
        print()
        return

    mismatch_pos = (diff > atol).nonzero(as_tuple=False)
    n_show = min(20, mismatch_pos.size(0))
    label_suffix = "" if label_ids is None else f" (label token id)"

    print(f"  {CROSS} 前 {n_show} 个不匹配位置 (on=sharing_ON  off=sharing_OFF):")
    print(f"  {'POS':>10s}  {'TOKEN':>10s}  {'ON':>14s}  {'OFF':>14s}  {'DIFF':>14s}  {'REL%':>10s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*10}")

    for idx in mismatch_pos[:n_show]:
        b, p = idx[0].item(), idx[1].item()
        tok = str(label_ids[b, p].item()) if label_ids is not None else "—"
        on_val = t1[b, p].item()
        off_val = t2[b, p].item()
        d = diff[b, p].item()
        rel = abs(d / max(abs(on_val), abs(off_val), 1e-8)) * 100
        print(f"  [{b:>4d},{p:>4d}]  {tok:>10s}  {on_val:>14.6f}  {off_val:>14.6f}  {d:>14.6e}  {rel:>9.2f}%")

    # 每行汇总
    per_row = (diff > atol).sum(dim=1)
    print(f"\n  Row summary:")
    for b in range(B):
        n_row = int(per_row[b].item())
        if n_row > 0:
            row_max = diff[b].max().item()
            row_max_pos = diff[b].argmax().item()
            label_hint = ""
            if label_ids is not None and label_ids.shape[1] > row_max_pos:
                label_hint = f" token={label_ids[b, row_max_pos].item()}"
            print(f"    row[{b}]: {n_row}/{L} mismatches  "
                  f"(max_diff={row_max:.6e} @ col={row_max_pos}{label_hint})")
    print()


def _print_3d_detail(
    name: str, t1: torch.Tensor, t2: torch.Tensor,
    diff: torch.Tensor, atol: float,
    label_ids: torch.Tensor | None,
):
    """打印 3D 张量的详细不匹配信息。"""
    B, L, V = t1.shape
    n_total = int(diff.numel())
    n_mismatch = (diff > atol).sum().item()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # 每 (b, l) 位置在 vocab 维度上的最大差异
    diff_per_pos = diff.max(dim=-1).values  # [B, L]
    mismatched_positions = int((diff_per_pos > atol).sum().item())

    print(_SEP_THIN)
    print(f"─── {name}  shape=({B}, {L}, {V})  total_elements={n_total} ───")
    print(f"    element-wise:  max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")
    print(f"    per-position (max over vocab):  {mismatched_positions}/{B*L} positions affected")
    print(_SEP_THIN)

    if n_mismatch == 0:
        print(f"  {CHECK} ALL MATCH")
        print()
        return

    # 差异最大的 topk 元素
    flat_top = diff.view(-1).topk(min(10, n_mismatch))
    print(f"  {CROSS} 差异最大的 {flat_top.values.size(0)} 个元素 (b, l, vocab_index):")
    print(f"  {'POS':>20s}  {'TOKEN':>10s}  {'VOCAB_IDX':>10s}  {'ON':>14s}  {'OFF':>14s}  {'DIFF':>14s}")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}")
    for val, flat_idx in zip(flat_top.values, flat_top.indices):
        b = int(flat_idx // (L * V))
        rest = int(flat_idx % (L * V))
        l = int(rest // V)
        v = int(rest % V)
        tok = str(label_ids[b, l].item()) if label_ids is not None else "—"
        print(f"  [{b:>4d},{l:>4d},{v:>4d}]  {tok:>10s}  {v:>10d}  {t1[b,l,v]:>14.6f}  {t2[b,l,v]:>14.6f}  {val:>14.6e}")

    # 每 (b, l) 位置跨 vocab 最大差异 topk
    n_show = min(10, B * L)
    top_pos = diff_per_pos.view(-1).topk(n_show)
    print(f"\n  跨 vocab 差异最大的 {n_show} 个 (b, l) 位置:")
    print(f"  {'POS':>10s}  {'TOKEN':>10s}  {'MAX_DIFF':>16s}")
    print(f"  {'─'*10}  {'─'*10}  {'─'*16}")
    for val, flat_idx in zip(top_pos.values, top_pos.indices):
        b = int(flat_idx // L)
        l = int(flat_idx % L)
        tok = str(label_ids[b, l].item()) if label_ids is not None else "—"
        print(f"  [{b:>4d},{l:>4d}]  {tok:>10s}  {val:>16.6e}")
    print()


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def compare_legacy(dir1: str, dir2: str, atol: float, dtype: str | None):
    """对比 legacy logprobs.pt。"""
    d1, d1_ids = load_legacy_dump(dir1)
    d2, d2_ids = load_legacy_dump(dir2)

    if dtype:
        d1 = cast_to_dtype(d1, dtype)
        d2 = cast_to_dtype(d2, dtype)

    _print_header("", dir1, dir2)
    diff, mask = _compute_2d_diff(d1, d2)
    r = CompareResult.from_2d_diff("logprobs", diff, mask, atol)
    _print_summary_table([r], atol)
    if r.passed:
        _print_pass("logprobs", r.shape, int(mask.sum().item()))
    else:
        _print_2d_detail("logprobs", d1, d2, diff, mask, atol, label_ids=d1_ids)


def compare_tagged(dir1: str, dir2: str, tag: str, atol: float, dtype: str | None):
    """对比新格式 dump: logits_{tag}.pt + entropy_{tag}.pt + label.pt + (可选) logprobs.pt。"""

    _print_header(tag, dir1, dir2)

    # 加载 label（两面共用）
    label = None
    label_path1 = os.path.join(dir1, "label.pt")
    if os.path.exists(label_path1):
        label = load_tensor(label_path1)
        print(f"label  shape={_fmt_shape(tuple(label.shape))}")
        print()

    # ── 收集所有对比结果 ──
    results: list[CompareResult] = []
    detail_data: list[dict] = []  # 存不匹配项的原始数据，后续展开详情

    # entropy
    ent_file = f"entropy_{tag}.pt"
    ent1_path = os.path.join(dir1, ent_file)
    ent2_path = os.path.join(dir2, ent_file)
    if os.path.exists(ent1_path) and os.path.exists(ent2_path):
        ent1 = load_tensor(ent1_path, dtype)
        ent2 = load_tensor(ent2_path, dtype)
        if ent1.shape != ent2.shape:
            r = CompareResult.shape_mismatch_result(
                f"entropy_{tag}", tuple(ent1.shape), tuple(ent2.shape))
            results.append(r)
            detail_data.append({"type": "shape_mismatch", "name": f"entropy_{tag}",
                                "t1": ent1, "t2": ent2, "result": r})
        else:
            diff, mask = _compute_2d_diff(ent1, ent2)
            r = CompareResult.from_2d_diff(f"entropy_{tag}", diff, mask, atol)
            results.append(r)
            detail_data.append({
                "type": "2d", "name": f"entropy_{tag}",
                "t1": ent1, "t2": ent2, "diff": diff, "mask": mask, "atol": atol,
                "label_ids": label, "result": r,
            })

    # logits
    log_file = f"logits_{tag}.pt"
    log1_path = os.path.join(dir1, log_file)
    log2_path = os.path.join(dir2, log_file)
    if os.path.exists(log1_path) and os.path.exists(log2_path):
        log1 = load_tensor(log1_path, dtype)
        log2 = load_tensor(log2_path, dtype)
        if log1.shape != log2.shape:
            r = CompareResult.shape_mismatch_result(
                f"logits_{tag}", tuple(log1.shape), tuple(log2.shape))
            results.append(r)
            detail_data.append({"type": "shape_mismatch", "name": f"logits_{tag}",
                                "t1": log1, "t2": log2, "result": r})
        else:
            diff, mask = _compute_3d_diff(log1, log2)
            n_mismatch = (diff > atol).sum().item()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            r = CompareResult(
                name=f"logits_{tag}",
                shape=tuple(log1.shape),
                max_diff=max_diff,
                mean_diff=mean_diff,
                n_mismatch=int(n_mismatch),
                n_total=int(diff.numel()),
                passed=(n_mismatch == 0),
            )
            results.append(r)
            detail_data.append({
                "type": "3d", "name": f"logits_{tag}",
                "t1": log1, "t2": log2, "diff": diff, "mask": mask, "atol": atol,
                "label_ids": label, "result": r,
            })

    # logprobs (legacy，如果存在就一起对比)
    lp_path1 = os.path.join(dir1, "logprobs.pt")
    lp_path2 = os.path.join(dir2, "logprobs.pt")
    if os.path.exists(lp_path1) and os.path.exists(lp_path2):
        lp1 = load_tensor(lp_path1)
        lp2 = load_tensor(lp_path2)
        if dtype:
            lp1 = cast_to_dtype(lp1, dtype)
            lp2 = cast_to_dtype(lp2, dtype)
        if lp1.shape != lp2.shape:
            r = CompareResult.shape_mismatch_result(
                "logprobs", tuple(lp1.shape), tuple(lp2.shape))
            results.append(r)
            detail_data.append({"type": "shape_mismatch", "name": "logprobs",
                                "t1": lp1, "t2": lp2, "result": r})
        else:
            diff, mask = _compute_2d_diff(lp1, lp2)
            r = CompareResult.from_2d_diff("logprobs", diff, mask, atol)
            results.append(r)
            detail_data.append({
                "type": "2d", "name": "logprobs",
                "t1": lp1, "t2": lp2, "diff": diff, "mask": mask, "atol": atol,
                "label_ids": label, "result": r,
            })

    if not results:
        print("No dump files found.")
        return

    # ── 输出 ──
    _print_summary_table(results, atol)

    # 判断逻辑：logits 坏 → entropy 一定坏（entropy 从 logits 算的）
    # 只展开 fail 项的详情
    for d in detail_data:
        r = d["result"]
        if d["type"] == "shape_mismatch":
            _print_shape_mismatch(d["name"], r.shape_on, r.shape_off, d["t1"], d["t2"])
        elif r.passed:
            if d["type"] == "2d":
                _print_pass(d["name"], r.shape, int(d["mask"].sum().item()))
            else:
                print(f"─── {d['name']}  shape={_fmt_shape(r.shape)} ───")
                print(f"  {CHECK} ALL MATCH")
                print()
        else:
            if d["type"] == "2d":
                _print_2d_detail(d["name"], d["t1"], d["t2"], d["diff"], d["mask"],
                                 d["atol"], d["label_ids"])
            else:
                _print_3d_detail(d["name"], d["t1"], d["t2"], d["diff"],
                                 d["atol"], d["label_ids"])


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare prefix-sharing dump tensors (logprobs / logits / entropy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-1", "--sharing-on", required=True,
                        help="Dump directory from sharing ON run")
    parser.add_argument("-2", "--sharing-off", required=True,
                        help="Dump directory from sharing OFF run")
    parser.add_argument("--atol", type=float, default=1e-5,
                        help="Absolute tolerance (default: 1e-5)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=None,
                        help="Cast both sides to this dtype before comparing")
    parser.add_argument("--tag", choices=["old", "train"], default=None,
                        help="Dump tag: 'old' for forward_only=True dumps, "
                             "'train' for forward_only=False dumps. "
                             "Omit for legacy logprobs.pt comparison.")
    args = parser.parse_args()

    if args.tag is not None:
        compare_tagged(args.sharing_on, args.sharing_off, args.tag,
                       atol=args.atol, dtype=args.dtype)
    else:
        compare_legacy(args.sharing_on, args.sharing_off,
                       atol=args.atol, dtype=args.dtype)
