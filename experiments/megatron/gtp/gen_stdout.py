"""把《Megatron-LM 深度剖析（04）：GTP 权重重 materialization 的切分语义》
正文里的形状推演与不变量逐条重算，输出与正文逐格对账。

对应文章：
  https://lrypcy.github.io/2026/10/11/megatron-04-generalized-tensor-parallelism/

用法（在本目录）：
    ~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
    ~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏

退出码非 0 表示有对账失败项，可直接当 CI 用。

四张表：
  §3  gtp_remat_shard_dim0 的形状推演（padding 生效 / 不生效两种情形）
  §4  五组不变量 I1–I5
  §5  no-pad 模式的断言行为
  §6  与 activation-centric TP 的形状对照
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                     # noqa: E402
from gtp_shard import (                                 # noqa: E402
    SEED,
    activation_centric_shapes,
    check_invariants,
    per_rank_shapes,
    roundtrip,
    shard_dim0,
)

# trailing 维取Llama-3 8B 的 ffn = 14336，与正文 §3 的算例一致
TRAILING = (14336,)
failures: list[str] = []


def ck(tag: str, got, want, tol: float = 1e-9) -> None:
    """对账一个数字。确定性实验下容差极紧。"""
    denom = abs(want) if want else 1.0
    ok = abs(got - want) / denom <= tol
    if not ok:
        failures.append(f"{tag}: 脚本 {got:.9g} vs 正文 {want:.9g}")
    print(f"  {tag:<40}{got:>14.6f}{want:>14.6f}   {'ok' if ok else 'MISMATCH'}")


def bool_ck(tag: str, ok: bool) -> None:
    print(f"  {tag:<40}{str(ok):>14}{'True':>14}   {'ok' if ok else 'MISMATCH'}")
    if not ok:
        failures.append(f"{tag}: 脚本 {ok} vs 正文 True")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="只打屏，不写 results/")
    args = ap.parse_args(argv)

    buf = io.StringIO()
    real = sys.stdout
    if not args.stdout:
        sys.stdout = buf

    print("=" * 84)
    print("gen_stdout.py —— 《深度剖析（04）GTP 切分语义》逐格对账")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print(f"pad_for_alignment=16（GTPRematConfig 默认）| trailing={TRAILING} | seed={SEED}")
    print("=" * 84)

    # ── §3 形状推演 ────────────────────────────────────────────────
    print("\n[§3-a] padding 不生效：标准维度都已是 alignment 的倍数")
    print(f"  {'hidden':>7}{'size':>5}{'align':>7}{'pad':>5}{'padded':>8}{'shard':>7}{'shard%16':>10}{'每rank形状':>18}")
    aligned = []
    for dim0, size in [(4096, 4), (4096, 8), (5120, 8), (11008, 8), (13824, 8)]:
        i = per_rank_shapes(dim0, TRAILING, size)
        aligned.append((dim0, size, i))
        print(f"  {dim0:>7}{size:>5}{i['alignment']:>7}{i['pad_length']:>5}{i['padded']:>8}"
              f"{i['shard_dim0']:>7}{i['shard_dim0'] % 16:>10}{str(i['ranks'][0]['shape']):>18}")
        if i["pad_length"] != 0:
            failures.append(f"§3-a hidden={dim0} size={size} 期望 pad=0，实得 {i['pad_length']}")
        if i["shard_dim0"] % 16 != 0:
            failures.append(f"§3-a hidden={dim0} size={size} shard 非 16 的倍数")
    ck("§3-a hidden=4096 size=8 shard", aligned[1][2]["shard_dim0"], 512)

    print("\n[§3-b] padding 生效：非标准维度或 size 不整除标准维度")
    print(f"  {'hidden':>7}{'size':>5}{'align':>7}{'pad':>5}{'padded':>8}{'shard':>7}{'shard%16':>10}{'浪费%':>9}")
    padded_cases = []
    for dim0, size in [(4100, 4), (777, 4), (11008, 3), (11008, 6)]:
        i = per_rank_shapes(dim0, TRAILING, size)
        r = roundtrip(dim0, TRAILING, size)
        padded_cases.append((dim0, size, i, r))
        print(f"  {dim0:>7}{size:>5}{i['alignment']:>7}{i['pad_length']:>5}{i['padded']:>8}"
              f"{i['shard_dim0']:>7}{i['shard_dim0'] % 16:>10}{r['waste_ratio'] * 100:>9.3f}")
        if i["shard_dim0"] % 16 != 0:
            failures.append(f"§3-b hidden={dim0} size={size} shard 非 16 的倍数")
    ck("§3-b hidden=4100 size=4 pad", padded_cases[0][2]["pad_length"], 60)
    ck("§3-b hidden=4100 size=4 shard", padded_cases[0][2]["shard_dim0"], 1040)
    ck("§3-b hidden=777 size=4 浪费%", padded_cases[1][3]["waste_ratio"] * 100, 7.078507, 1e-5)

    # ── §4 五组不变量 ─────────────────────────────────────────────
    print("\n[§4] 五组不变量 I1–I5（覆盖 8 组配置）")
    cases = [(4096, 4), (4096, 8), (5120, 8), (4100, 4), (777, 4), (11008, 3), (11008, 8), (28672, 2)]
    print(f"  {'hidden':>7}{'size':>5}  " + "".join(f"{k.split('_')[0]:>6}" for k in
          ["I1", "I2", "I3", "I4a", "I4b", "I5"]) + "   往返")
    for dim0, size in cases:
        info = per_rank_shapes(dim0, TRAILING, size)
        inv = check_invariants(info)
        rt = roundtrip(dim0, TRAILING, size)
        flags = [
            inv["I1_padded_aligned"],
            inv["I2_shard_is_multiple_of_pad_align"],
            inv["I3_contiguous_cover"],
            inv["I4_data_contiguous"],
            inv["I4_pad_is_tail_only"],
            rt["restore_matches_truth"],
        ]
        print(f"  {dim0:>7}{size:>5}  " + "".join(f"{str(f):>6}" for f in flags)
              + f"   {'一致' if rt['restore_matches_truth'] else 'MISMATCH'}")
        for f in flags:
            if not f:
                failures.append(f"§4 hidden={dim0} size={size} 某条不变量为 False")
        if not rt["rebuild_matches_padded"]:
            failures.append(f"§4 hidden={dim0} size={size} 拼接未还原 padded 张量")

    # ── §5 no-pad 模式 ─────────────────────────────────────────────
    print("\n[§5] no-pad 模式（pad_for_alignment=0）")
    for dim0, expect_assert in [(4096, False), (4100, False), (4101, True), (777, True)]:
        try:
            s, p = shard_dim0(dim0, 4, pad_for_alignment=0)
            got_assert = False
            print(f"  hidden={dim0:>5} ({dim0}%4={dim0 % 4}): shard={s:>5} pad={p}  未抛断言")
        except AssertionError:
            got_assert = True
            print(f"  hidden={dim0:>5} ({dim0}%4={dim0 % 4}): ✓ 抛 AssertionError")
        bool_ck(f"§5 hidden={dim0} 是否抛断言", got_assert == expect_assert)
    np_info = per_rank_shapes(4096, TRAILING, 4, pad_for_alignment=0)
    ck("§5 no-pad hidden=4096 shard", np_info["shard_dim0"], 1024)

    # ── §6 与 activation-centric 的形状对照 ────────────────────────
    print("\n[§6] 与 activation-centric TP 的形状对照（cut_axis 不同）")
    ac = activation_centric_shapes(4096, 4096, 4)
    gtp = per_rank_shapes(4096, (4096,), 4)
    print(f"  activation-centric: 激活 {ac['act_shape_full']} → 每 rank {ac['act_shape_per_rank']}")
    print(f"                      权重 {ac['weight_shape_full']} → 每 rank {ac['weight_shape_per_rank']}")
    print(f"  GTP:                权重第 0 维 {gtp['dim0']} → 每 rank {gtp['ranks'][0]['shape']}"
          f"（沿 dim 0 切，gtp_remat={gtp['gtp_remat_size']}）")
    print(f"  对照: activation-centric 切激活最后一维；GTP 切权重第 0 维并在使用时all-gather 回全量")
    bool_ck("§6 activation-centric 每 rank 激活形状正确", ac["act_shape_per_rank"] == (1, 1, 1024))
    bool_ck("§6 GTP 每 rank 权重形状正确", gtp["ranks"][0]["shape"] == (1024, 4096))

    print("\n" + "=" * 84)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for f in failures:
            print("  !! " + f)
    else:
        print("对账通过：正文四张表的每一个数字都与本目录脚本实跑一致。")
    print("=" * 84)

    sys.stdout = real
    text = buf.getvalue()
    if args.stdout:
        print(text)
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        out = os.path.join(here, "results", "stdout.txt")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写入 {out}（{len(text.splitlines())} 行）")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())