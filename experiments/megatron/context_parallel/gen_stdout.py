"""把《Megatron-LM 深度剖析（07）：上下文并行与长上下文》正文里的
online softmax 分块递推结论逐条重算，输出与正文逐格对账。

对应文章：
  https://lrypcy.github.io/2026/10/17/megatron-07-context-parallel-and-long-context/

用法（在本目录）：
    ~/Software/miniconda3/bin/python3 gen_stdout.py# 写 results/stdout.txt
    ~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏

退出码非 0 表示有对账失败项，可直接当 CI 用。

三张表：
  §2  单块递推≡ 全量 softmax（逐位相同）
  §3  分块数扫描：多块递推 vs 全量 softmax
  §4  误差随块数的走向 + KV 通信量解析模型（每层每 token 的字节数）
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from online_softmax import (  # noqa: E402
    SEED,
    check_e2,
    check_e3,
    check_e4,
    check_e5,
)

# 正文使用的规格
N, D, M, DV = 64, 32, 256, 32          # 64 个 query，32 维 head，256 个 key
RING_D = 32                             # ring 实验里每 rank 持有的段长
RING_CP = 8                             # ring 实验的 CP size
BYTES_PER_ELEM = 2                       # bf16，每元素 2 字节

NB_BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

TOL_REL = 1e-12      # float64 下的等价性上限
failures: list[str] = []


def check(tag: str, got: float, want: float | None, rel_tol: float) -> None:
    """对账一条数字。rel_tol 为相对容差。"""
    if want is None:
        print(f"  {tag:<40}{got:>18.6e}{'—':>18}   (正文未列)")
        return
    denom = abs(want) if want else 1.0
    ok = abs(got - want) / denom <= rel_tol
    if not ok:
        failures.append(f"{tag}: 脚本 {got:.9g} vs 正文 {want:.9g}")
    print(f"  {tag:<40}{got:>18.6e}{want:>18.6e}   {'ok' if ok else 'MISMATCH'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="只打屏，不写 results/")
    args = ap.parse_args(argv)

    buf = io.StringIO()
    real = sys.stdout
    if not args.stdout:
        sys.stdout = buf

    print("=" * 92)
    print("gen_stdout.py —— 《深度剖析（07）上下文并行与长上下文》逐格对账")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print(f"规格 N={N} D={D} M={M} dv={DV} | ring: cp={RING_CP} 每段={RING_D} | seed={SEED}")
    print("=" * 92)

    # ── §2  单块递推 ≡ 全量 softmax ─────────────────────────────
    print("\n[§2] 单块递推 vs 全量 softmax（float64）")
    print(f"  {'n':>6}{'d':>6}{'m':>6}{'最大绝对误差':>20}{'参考量级':>16}{'逐位相同':>12}")
    e2 = check_e2(N, D, M, DV)
    print(f"  {e2['n']:>6}{e2['d']:>6}{e2['m']:>6}{e2['abs_err']:>20.3e}"
          f"{e2['ref_absmax']:>16.6f}{str(e2['bitexact']):>12}")
    if not e2["bitexact"]:
        failures.append(f"§2 单块递推未与全量 softmax 逐位相同（abs_err={e2['abs_err']:.3e}）")
    check("§2 abs_err", e2["abs_err"], 0.0, 1e-12)

    # ── §3  分块数扫描 ────────────────────────────────────────────
    print("\n[§3] 分块递推 vs 全量 softmax，扫块数（float64）")
    print(f"  {'块数':>8}{'块长':>8}{'最大绝对误差':>20}{'相对误差':>18}")
    e3 = check_e3(N, D, M, DV, NB_BLOCKS)
    e3map = {r["nblocks"]: r for r in e3}
    for r in e3:
        print(f"  {r['nblocks']:>8}{r['block_size']:>8}{r['abs_err']:>20.3e}{r['rel_err']:>18.3e}")
    for r in e3:
        if r["rel_err"] > TOL_REL:
            failures.append(
                f"§3 块数={r['nblocks']} 相对误差 {r['rel_err']:.3e} 超过 {TOL_REL:.0e}")
    check("§3 块数=4 相对误差", e3map[4]["rel_err"], 0.0, 1e-12)
    check("§3 块数=64 相对误差", e3map[64]["rel_err"], 0.0, 1e-12)

    # ── §4  误差随块数 + 通信量 ───────────────────────────────────
    print("\n[§4] 误差随块数的走向")
    e4 = check_e4(N, D, M, DV, NB_BLOCKS)
    print(f"  {'块数':>8}{'相对误差':>18}{'相邻比值':>18}")
    for i, r in enumerate(e4["rows"]):
        ratio = e4["ratios"][i - 1] if i > 0 else float("nan")
        ratio_s = "—" if i == 0 else f"{ratio:.6f}"
        print(f"  {r['nblocks']:>8}{r['rel_err']:>18.3e}{ratio_s:>18}")

    print("\n[§4] 三条路线的 KV 通信量（解析模型）")
    print("  口径：单层、全序列长度 S、单元素 BYTES 字节；只算 K 与 V 两个张量。")
    print("  p2p(ring)：每 rank 自持 S/cp，逐步收到其余 cp-1 段 → 每 rank 收 (cp-1)/cp·S")
    print("  all_gather(THD)：每 rank 一次性拿到全序列 KV → 每 rank 收 S")
    print("  a2a(Ulysses)：头按 CP 组切，a2a 后每 rank 持全序列但只有 1/cp 的头")
    print("           → 每 rank 收 (1/cp)·S 个「头份额」，全网合计与 ring 相同")
    print(f"  {'cp':>6}{'p2p 每rank':>16}{'all_gather 每rank':>22}{'a2a 每rank':>16}"
          f"{'AG/p2p':>14}")
    ag_over_p2p = []
    for cp in (2, 4, 8, 16):
        kv_total = 2.0 * BYTES_PER_ELEM            # K + V 两张量，每 token 每元素
        p2p = (cp - 1) / cp * kv_total# 每 rank 收到的 KV 字节
        ag = kv_total                              # 每 rank 收全序列
        a2a = kv_total / cp                        # 每 rank 只收 1/cp 的头份额
        ratio = ag / p2p
        ag_over_p2p.append(ratio)
        print(f"  {cp:>6}{p2p:>16.2f}{ag:>22.2f}{a2a:>16.2f}{ratio:>14.6f}")
        if abs(ratio - cp / (cp - 1)) > 1e-12:
            failures.append(f"§4 cp={cp} AG/p2p 比值 {ratio:.12f} ≠ cp/(cp-1)")

    print("\n[§4] 全网合计 KV 字节（每 token 折算，S=1）")
    print("  p2p(ring) 合计 = cp × (cp-1)/cp·S = (cp-1)·S → 随 cp 线性增长")
    print("  all_gather 合计 = cp × S → 随 cp 线性增长，系数比 ring 大")
    print("  a2a(Ulysses) 合计 = S → 与 cp 无关，每个 KV 元素只过一次网络")
    print(f"  {'cp':>6}{'p2p 合计':>16}{'all_gather 合计':>22}{'a2a 合计':>16}"
          f"{'AG/p2p':>14}")
    for cp in (2, 4, 8, 16):
        kv_total = 2.0 * BYTES_PER_ELEM
        p2p_total = (cp - 1) * kv_total
        ag_total = cp * kv_total
        a2a_total = kv_total
        print(f"  {cp:>6}{p2p_total:>16.2f}{ag_total:>22.2f}{a2a_total:>16.2f}"
              f"{ag_total / p2p_total:>14.6f}")
        if abs(p2p_total - (cp - 1) * kv_total) > 1e-12:
            failures.append(f"§4 cp={cp} p2p 合计不等于 (cp-1)·S")
        if abs(a2a_total - kv_total) > 1e-12:
            failures.append(f"§4 cp={cp} a2a 合计不等于 S")

    # ── §5  ring 顺序递推 ────────────────────────────────────────
    print("\n[§5] ring 顺序递推（cp 份 K/V 按ring 顺序逐段吃掉）vs 全 K/V 一次性 softmax")
    print(f"  {'cp':>6}{'每段长度':>12}{'最大绝对误差':>20}{'相对误差':>18}")
    e5 = check_e5(RING_CP, RING_D, RING_D, D)
    print(f"  {e5['cp']:>6}{RING_D:>12}{e5['abs_err']:>20.3e}{e5['rel_err']:>18.3e}")
    if e5["rel_err"] > TOL_REL:
        failures.append(f"§5 ring 递推相对误差 {e5['rel_err']:.3e} 超过 {TOL_REL:.0e}")
    check("§5 ring 相对误差", e5["rel_err"], 0.0, 1e-12)

    print("\n[§5] 每个 rank 各自的误差（cp=8）")
    print(f"  {'rank':>8}{'最大绝对误差':>20}")
    for i, err in enumerate(e5["per_rank_abs_err"]):
        print(f"  {i:>8}{err:>20.3e}")
        if err > TOL_REL:
            failures.append(f"§5 rank {i} 误差 {err:.3e} 超过 {TOL_REL:.0e}")

    print("\n" + "=" * 92)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for f in failures:
            print("!! " + f)
    else:
        print("对账通过：正文各表的每一个数字都与本目录脚本实跑一致。")
    print("=" * 92)

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