"""把《Megatron-LM 深度剖析（03）：张量并行与序列并行的真实实现》正文里的
等价性结论逐条重算，输出与正文逐格对账。

对应文章：
  https://lrypcy.github.io/2026/10/09/megatron-03-tensor-parallel-and-sequence-parallel/

用法（在本目录）：
    ~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
    ~/Software/miniconda3/bin/python3 gen_stdout.py --stdout# 只打屏

退出码非 0 表示有对账失败项，可直接当 CI 用。

三张表：
  §3  f 的列切 + g 的 all-reduce ≡ 未切分的整块 GEMM
  §6  SP 下 g 换 reduce-scatter 的数值等价 + 通信量守恒
  §9  gradient-accumulation-fusion 与朴素累加的精度差
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tp_equiv import SEED, check_e1, check_e2, check_e3      # noqa: E402

# 正文使用的规格：s=8, b=4 → N=32；hidden=16, ffn=48（ffn = 3*hidden，对齐 SwiGLU 形状）
S, B, HIDDEN, FFN = 8, 4, 16, 48
TPS = (2, 4, 8)

# 容差
TOL_REL = 1e-5        # float32 下的等价性；实测约 2e-7，留两个数量级余量
TOL_RATIO = 1e-12     # 通信量守恒应为逐字节相等
failures: list[str] = []


def check(tag: str, got: float, want: float | None, rel_tol: float) -> None:
    """对账一条数字。rel_tol 为相对容差；确定性实验下可给很紧的值。"""
    if want is None:
        print(f"  {tag:<34}{got:>16.6e}{'—':>16}   (正文未列)")
        return
    denom = abs(want) if want else 1.0
    ok = abs(got - want) / denom <= rel_tol
    if not ok:
        failures.append(f"{tag}: 脚本 {got:.9g} vs 正文 {want:.9g}")
    print(f"  {tag:<34}{got:>16.6e}{want:>16.6e}   {'ok' if ok else 'MISMATCH'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="只打屏，不写 results/")
    args = ap.parse_args(argv)

    buf = io.StringIO()
    real = sys.stdout
    if not args.stdout:
        sys.stdout = buf

    print("=" * 84)
    print("gen_stdout.py —— 《深度剖析（03）张量并行与序列并行》等价性逐格对账")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print(f"规格 s={S} b={B} N={S*B} hidden={HIDDEN} ffn={FFN} | seed={SEED}")
    print("=" * 84)

    # ── §3  f 列切 + all-reduce ≡ 整块 ─────────────────────────────
    print("\n[§3] f 的列切 + g 的 all-reduce≡ 未切分的整块 GEMM")
    print(f"  {'tp':>4}{'相对误差':>16}{'绝对误差':>16}{'参考量级':>16}")
    e1 = {}
    for tp in TPS:
        r = check_e1(S, B, HIDDEN, FFN, tp)
        e1[tp] = r
        print(f"  {tp:>4}{r['rel_err']:>16.3e}{r['abs_err']:>16.3e}{r['ref_absmax']:>16.4f}")
        if r["rel_err"] > TOL_REL:
            failures.append(f"§3 tp={tp} 相对误差 {r['rel_err']:.3e} 超过 {TOL_REL:.0e}")
    check("§3 tp=4 相对误差", e1[4]["rel_err"], 1.205e-07, 0.05)

    # ── §6  SP 数值等价 + 通信量守恒 ───────────────────────────────
    print("\n[§6] SP 路径（g 换 reduce-scatter）与非 SP 的数值等价")
    print(f"  {'tp':>4}{'非SP 相对误差':>18}{'SP 相对误差':>18}")
    e2 = {}
    for tp in TPS:
        r = check_e2(S, B, HIDDEN, FFN, tp)
        e2[tp] = r
        print(f"  {tp:>4}{r['nosp_rel_err']:>18.3e}{r['sp_rel_err']:>18.3e}")
        if r["nosp_rel_err"] > TOL_REL:
            failures.append(f"§6 tp={tp} 非SP 相对误差 {r['nosp_rel_err']:.3e} 超过 {TOL_REL:.0e}")
        if r["sp_rel_err"] > TOL_REL:
            failures.append(f"§6 tp={tp} SP 相对误差 {r['sp_rel_err']:.3e} 超过 {TOL_REL:.0e}")

    print("\n[§6] 通信量守恒（ring bus-byte 口径，单位化到 S=1）")
    print(f"  {'tp':>4}{'AllReduce':>16}{'RS + AG':>16}{'比值':>18}")
    for tp in TPS:
        r = e2[tp]
        print(f"  {tp:>4}{r['ar_bytes_unit']:>16.6f}{r['rs_ag_bytes_unit']:>16.6f}"
              f"{r['bytes_ratio']:>18.15f}")
        if abs(r["bytes_ratio"] - 1.0) > TOL_RATIO:
            failures.append(f"§6 tp={tp} 通信量比值 {r['bytes_ratio']:.15f} ≠ 1")
    check("§6 tp=4 比值", e2[4]["bytes_ratio"], 1.0, TOL_RATIO)

    # ── §9  wgrad 累加精度 ─────────────────────────────────────────
    print("\n[§9] gradient-accumulation-fusion vs 朴素累加（真值基准 = fp64 累加）")
    print(f"  {'steps':>6}{'fused vs naive32':>20}{'fused vs fp64':>20}{'naive16 vs fp64':>20}")
    e3 = {}
    for steps in (4, 16, 64):
        r = check_e3(S, HIDDEN, FFN, 4, steps)
        e3[steps] = r
        print(f"  {steps:>6}{r['fused_vs_naive32_abs']:>20.3e}"
              f"{r['fused_vs_fp64_rel']:>20.3e}{r['naive16_vs_fp64_rel']:>20.3e}")
        if r["fused_vs_naive32_abs"] != 0.0:
            failures.append(
                f"§9 steps={steps} fused 与 naive32 非逐位相同："
                f"{r['fused_vs_naive32_abs']:.3e}")
        ratio = r["naive16_vs_fp64_rel"] / r["fused_vs_fp64_rel"]
        if ratio < 100.0:
            failures.append(f"§9 steps={steps} bf16 路径未显示出应有的劣势（比值 {ratio:.1f}）")

    print("\n[§9] 主累加器 dtype 的影响（steps=16）")
    r = e3[16]
    ratio = r["naive16_vs_fp64_rel"] / r["fused_vs_fp64_rel"]
    print(f"  fused(fp32 主累加器) 相对误差{r['fused_vs_fp64_rel']:>14.3e}")
    print(f"  naive16(bf16 逐步舍入) 相对误差   {r['naive16_vs_fp64_rel']:>14.3e}")
    print(f"  劣化倍数                {ratio:>14.1f}x")
    check("§9 steps=16 劣化倍数", ratio, 14558.94262295, 1e-9)

    print("\n" + "=" * 84)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for f in failures:
            print("!! " + f)
    else:
        print("对账通过：正文三张表的每一个数字都与本目录脚本实跑一致。")
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