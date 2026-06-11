"""
端到端精度对比工具——对比开启/关闭 prefix-sharing 两遍独立运行的 dump。

支持三种 dump 类型，通过 --tag 切换：
  --tag old    : 对比 logits_old.pt + entropy_old.pt + label.pt  （推理前向，forward_only=True）
  --tag train  : 对比 logits_train.pt + entropy_train.pt + label.pt（训练前向，forward_only=False）
  (不设 --tag) : 对比 logprobs.pt + input_ids.pt（legacy，向前兼容）

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
    """将 tensor 转成目标精度再回升至 float32，模拟混合精度下的量化误差。

    模型以 bf16/fp16 运行时，输出在计算过程存在中间精度损失。
    通过先降精度再回升，可以模拟这种误差，避免 dump 时已是 float32
    而掩盖实际运行时的不一致。
    """
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


def compare_legacy(dir1: str, dir2: str, atol: float, dtype: str | None):
    """对比 legacy logprobs.pt。"""
    d1, d1_ids = load_legacy_dump(dir1)
    d2, d2_ids = load_legacy_dump(dir2)

    if dtype:
        d1 = cast_to_dtype(d1, dtype)
        d2 = cast_to_dtype(d2, dtype)

    _compare_2d(d1, d2, atol, label_ids=d1_ids, name="logprobs")


# ──────────────────────────────────────────────
# 新 dump: logits_{tag}.pt + entropy_{tag}.pt + label.pt
# ──────────────────────────────────────────────


def compare_tagged(dir1: str, dir2: str, tag: str, atol: float, dtype: str | None):
    """对比新格式 dump: logits_{tag}.pt + entropy_{tag}.pt, 以及 label.pt。"""

    # 加载 label（两面共用）
    label = None
    label_path1 = os.path.join(dir1, "label.pt")
    if os.path.exists(label_path1):
        label = load_tensor(label_path1)
        print(f"label  shape={tuple(label.shape)}")

    # ── entropy ──
    ent_file = f"entropy_{tag}.pt"
    ent1_path = os.path.join(dir1, ent_file)
    ent2_path = os.path.join(dir2, ent_file)
    if os.path.exists(ent1_path) and os.path.exists(ent2_path):
        ent1 = load_tensor(ent1_path, dtype)
        ent2 = load_tensor(ent2_path, dtype)
        print(f"\n─── entropy_{tag}.pt ───")
        _compare_2d(ent1, ent2, atol, label_ids=label, name=f"entropy_{tag}")
    else:
        print(f"\n─── entropy_{tag}.pt ─── SKIP (file not found)")

    # ── logits ──
    log_file = f"logits_{tag}.pt"
    log1_path = os.path.join(dir1, log_file)
    log2_path = os.path.join(dir2, log_file)
    if os.path.exists(log1_path) and os.path.exists(log2_path):
        log1 = load_tensor(log1_path, dtype)
        log2 = load_tensor(log2_path, dtype)
        print(f"\n─── logits_{tag}.pt ───")
        _compare_3d(log1, log2, atol, label_ids=label, name=f"logits_{tag}")
    else:
        print(f"\n─── logits_{tag}.pt ─── SKIP (file not found)")


# ──────────────────────────────────────────────
# 通用对比核心
# ──────────────────────────────────────────────


def _compare_2d(
    t1: torch.Tensor,
    t2: torch.Tensor,
    atol: float,
    label_ids: torch.Tensor | None = None,
    name: str = "tensor",
):
    """逐元素对比两个 2D tensor [B, L]。"""
    assert t1.shape == t2.shape, f"shape mismatch: {t1.shape} vs {t2.shape}"

    B, L = t1.shape
    diff = (t1 - t2).abs()

    # 只关注非零位置（两边至少有一边非零）
    mask = (t1 != 0) | (t2 != 0)
    active = mask.sum().item()
    n_mismatch = (diff > atol).sum().item()

    max_diff = diff[mask].max().item() if mask.any() else 0.0
    mean_diff = diff[mask].mean().item() if mask.any() else 0.0
    # 每位置最大差异
    per_pos = diff.max(dim=0) if diff.ndim > 1 else diff

    print(f"shape=({B}, {L})  active_tokens={active}")
    print(f"max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")

    if n_mismatch == 0:
        print(f"ALL MATCH — {name} consistent with baseline")
        return

    mismatch_pos = (diff > atol).nonzero(as_tuple=False)
    n_show = min(20, mismatch_pos.size(0))
    print(f"\n前 {n_show} 个不匹配位置:")
    for idx in mismatch_pos[:n_show]:
        b, p = idx[0].item(), idx[1].item()
        parts = [f"[{b},{p}]"]
        if label_ids is not None:
            parts.append(f"token={label_ids[b, p].item()}")
        parts.append(f"on={t1[b, p]:.6f}")
        parts.append(f"off={t2[b, p]:.6f}")
        parts.append(f"diff={diff[b, p]:.6e}")
        print("  " + "  ".join(parts))

    # 汇总每行不匹配数
    per_row = (diff > atol).sum(dim=1)
    for b in range(B):
        if per_row[b].item() > 0:
            print(f"  row[{b}]: {per_row[b].item()} mismatches "
                  f"(max_per_pos_diff={per_pos[:, b].max().item():.6e}"
                  f" @ pos {per_pos[:, b].argmax().item()})")


def _compare_3d(
    t1: torch.Tensor,
    t2: torch.Tensor,
    atol: float,
    label_ids: torch.Tensor | None = None,
    name: str = "tensor",
):
    """逐元素对比两个 3D tensor [B, L, V]（logits）。

    报告逐元素整体差异 + 每 (b, l) 位置跨 vocab 维度的最大差异。
    """
    assert t1.shape == t2.shape, f"shape mismatch: {t1.shape} vs {t2.shape}"

    B, L, V = t1.shape
    diff = (t1 - t2).abs()

    # 逐元素整体
    n_total = diff.numel()
    n_mismatch = (diff > atol).sum().item()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    # 每 (b, l) 位置在 vocab 维度上的最大差异
    diff_per_pos = diff.max(dim=-1).values  # [B, L]
    mismatched_positions = (diff_per_pos > atol).sum().item()

    print(f"shape=({B}, {L}, {V})  total_elements={n_total}")
    print(f"element-wise: max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")
    print(f"per-position (max over vocab): mismatched_positions={mismatched_positions}")

    if n_mismatch == 0:
        print(f"ALL MATCH — {name} consistent with baseline")
        return

    # 找出差异最大的位置
    flat_top = diff.view(-1).topk(min(10, n_mismatch))
    print(f"\n差异最大的 {flat_top.values.size(0)} 个元素:")
    for val, flat_idx in zip(flat_top.values, flat_top.indices):
        b = flat_idx // (L * V)
        rest = flat_idx % (L * V)
        l = rest // V
        v = rest % V
        token_hint = f"token={label_ids[b, l].item()}" if label_ids is not None else ""
        print(
            f"  [{b},{l},{v}] {token_hint}  "
            f"on={t1[b, l, v]:.6f}  off={t2[b, l, v]:.6f}  diff={val:.6e}"
        )

    # 汇总每行每位置跨 vocab 的最大差异
    n_show = min(10, B * L)
    top_pos = diff_per_pos.view(-1).topk(n_show)
    print(f"\n跨 vocab 差异最大的 {n_show} 个 (b, l) 位置:")
    for val, flat_idx in zip(top_pos.values, top_pos.indices):
        b = flat_idx // L
        l = flat_idx % L
        token_hint = f"token={label_ids[b, l].item()}" if label_ids is not None else ""
        print(f"  [{b},{l}] {token_hint}  max_diff_over_vocab={val:.6e}")


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
