"""
端到端 logprobs 精度对比工具。

比较开启/关闭 prefix-sharing 两遍独立运行的 logprobs dump，验证精度一致性。

完整流程：
  # 第一遍：开启 prefix-sharing
  ENABLE_PREFIX_SHARING=1 PREFIX_SHARING_DUMP_DIR=./dump_on python -m verl.trainer.main_ppo ...

  # 第二遍：关闭 prefix-sharing（其余参数完全相同）
  ENABLE_PREFIX_SHARING=0 PREFIX_SHARING_DUMP_DIR=./dump_off python -m verl.trainer.main_ppo ...

  # 对比
  python compare_logprobs.py --sharing-on ./dump_on --sharing-off ./dump_off
  python compare_logprobs.py -1 ./dump_on -2 ./dump_off --atol 1e-5
  python compare_logprobs.py -1 ./dump_on -2 ./dump_off --dtype bfloat16
"""

import argparse
import os
import pathlib
import sys

import torch


def load_dump(dump_dir: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """加载 dump 目录中的 (logprobs, input_ids)。

    支持两种命名约定：
    - 单 rank:  logprobs.pt / input_ids.pt
    - 多 rank:  logprobs_rank{rank}.pt / input_ids_rank{rank}.pt
    多 rank 时 concat 在一起。
    """
    dp = pathlib.Path(dump_dir)
    if not dp.exists():
        sys.exit(f"error: dump directory not found: {dump_dir}")

    # 单 rank 模式
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


def cast_to_dtype(t: torch.Tensor, dtype_str: str) -> torch.Tensor:
    """将 tensor 转成目标精度再回升至 float32，模拟混合精度下的量化误差。

    模型以 bf16/fp16 运行时，logprobs 在计算过程存在中间精度损失。
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


def compare(dir1: str, dir2: str, atol: float = 1e-5, dtype: str | None = None):
    """加载两边 dump 并逐元素对比，输出差异报告。

    Args:
        dir1: 开启 prefix-sharing 的 dump 目录。
        dir2: 关闭 prefix-sharing（baseline）的 dump 目录。
        atol: 绝对容差，超出此值视为不匹配。
        dtype: 模拟混合精度误差，如 'bfloat16'。
    """
    d1_lp, d1_ids = load_dump(dir1)
    d2_lp, d2_ids = load_dump(dir2)

    if dtype:
        d1_lp = cast_to_dtype(d1_lp, dtype)
        d2_lp = cast_to_dtype(d2_lp, dtype)

    assert d1_lp.shape == d2_lp.shape, f"shape mismatch: {d1_lp.shape} vs {d2_lp.shape}"

    batch_size, seq_len = d1_lp.shape

    diff = (d1_lp - d2_lp).abs()

    # 只关注非零位置（两边至少有一边非零）
    mask = (d1_lp != 0) | (d2_lp != 0)
    active_span = mask.sum().item()
    n_mismatch = (diff > atol).sum().item()

    max_diff = diff[mask].max().item() if mask.any() else 0.0
    mean_diff = diff[mask].mean().item() if mask.any() else 0.0

    print(f"shape=({batch_size}, {seq_len})  active_tokens={active_span}")
    print(f"max_diff={max_diff:.6e}  mean_diff={mean_diff:.6e}  mismatched={n_mismatch}")

    all_match = True

    if n_mismatch > 0:
        all_match = False
        mismatch_positions = (diff > atol).nonzero(as_tuple=False)
        print(f"\n前 {min(20, mismatch_positions.size(0))} 个不匹配位置:")
        for idx in mismatch_positions[:20]:
            b, p = idx[0].item(), idx[1].item()
            parts = [f"[{b},{p}]"]
            if d1_ids is not None:
                parts.append(f"token={d1_ids[b, p].item()}")
            parts.append(f"on={d1_lp[b, p]:.6f}")
            parts.append(f"off={d2_lp[b, p]:.6f}")
            parts.append(f"diff={diff[b, p]:.6e}")
            print("  " + "  ".join(parts))

    if all_match:
        print("\nALL MATCH — prefix-sharing logprobs consistent with baseline")
    else:
        print(f"\nMISMATCH FOUND — {n_mismatch} positions exceed atol={atol}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare prefix-sharing logprobs dumps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-1", "--sharing-on", required=True, help="Dump directory from sharing ON run")
    parser.add_argument("-2", "--sharing-off", required=True, help="Dump directory from sharing OFF run")
    parser.add_argument(
        "--atol", type=float, default=1e-5, help="Absolute tolerance (default: 1e-5)"
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Cast both sides to this dtype before comparing (simulates reduced precision)",
    )
    args = parser.parse_args()
    compare(args.sharing_on, args.sharing_off, atol=args.atol, dtype=args.dtype)
