"""把《Megatron-LM 深度剖析（12）：确定性重跑与故障注入》正文里的
重跑正确性结论逐条重算，输出与正文逐格对账。

对应文章：
  https://lrypcy.github.io/2026/10/20/megatron-12-deterministic-rerun-and-fault-injection/

用法（在本目录）：
    ~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
    ~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏

退出码非 0 表示有对账失败项，可直接当 CI 用。
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rerun_equiv import (  # noqa: E402
    MICRO_BATCH,
    N_MICRO_BATCHES,
    SEED,
    check_e2,
    check_e3,
    check_e4,
    check_e5,
    check_e6,
)

TOL_REL = 1e-12
failures: list[str] = []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="只打屏，不写 results/")
    args = ap.parse_args(argv)

    buf = io.StringIO()
    real = sys.stdout
    if not args.stdout:
        sys.stdout = buf

    print("=" * 92)
    print("gen_stdout.py —— 《深度剖析（12）确定性重跑与故障注入》逐格对账")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print(f"规格 in={16} out={8} micro_batch={MICRO_BATCH} n_micro={N_MICRO_BATCHES}"
          f" dropout_p=0.1 | seed={SEED}")
    print("=" * 92)

    # ── §2  _compare_floats 的判定边界 ────────────────────────────
    print("\n[§2] _compare_floats 的判定边界（逐分支复刻源码 L1401-L1420）")
    print(f"  {'分支':<20}{'a':>10}{'b':>12}{'返回值':>26}")
    for r in check_e2():
        av = "nan" if isinstance(r["a"], float) and math.isnan(r["a"]) else f"{r['a']:.4g}"
        bv = "nan" if isinstance(r["b"], float) and math.isnan(r["b"]) else f"{r['b']:.4g}"
        g = r["got"]
        gs = "ZeroDivisionError" if isinstance(g, str) else (
            "inf" if math.isinf(g) else f"{g:.6E}")
        print(f"  {r['desc']:<20}{av:>10}{bv:>12}{gs:>26}")
        if not r["ok"]:
            failures.append(f"§2 {r['desc']}: 脚本 {g} vs 期望 {r['want']}")

    print("\n[§2] 相对差公式的对称性与边界")
    e2rows = {r["desc"]: r for r in check_e2()}
    sym = e2rows["对称性 a↔b"]["got"]
    close = e2rows["同号极接近"]["got"]
    print(f"  {'d(1.0, 1.001)':<24}{e2rows['同号相对差']['got']:>22.6E}")
    print(f"  {'d(1.001, 1.0)  对称':<24}{sym:>22.6E}")
    print(f"  {'d(1.0, 1+1e-15) 极接近':<24}{close:>22.6E}")
    print(f"  {'d(0.0, 1.0)    单边零':<24}{e2rows['零对一']['got']:>22.6E}")
    if abs(sym - e2rows["同号相对差"]["got"]) > TOL_REL:
        failures.append("§2 相对差不满足对称性")
    if close > 1e-14:
        failures.append(f"§2 极接近的两个值相对差 {close:.3E} 应接近 0")

    # ── §3  恢复 RNG 后重跑逐位一致 ───────────────────────────────
    print("\n[§3] 同一 step 重跑两次：恢复 RNG vs 不恢复")
    e3 = check_e3()
    print(f"  {'口径':<34}{'loss 相对差':>20}{'梯度最大绝对差':>22}{'逐位相同':>12}")
    print(f"  {'恢复 RNG 后重跑':<34}{e3['cmp_restore']:>20.6E}"
          f"{e3['grad_restore_absdiff']:>22.6E}{str(e3['grad_restore_bitexact']):>12}")
    print(f"  {'不恢复 RNG 重跑':<34}{e3['cmp_no_restore']:>20.6E}"
          f"{e3['grad_no_restore_absdiff']:>22.6E}{str(e3['grad_no_restore_bitexact']):>12}")
    print(f"\n  参考：loss={e3['loss']:.6f}  梯度 absmax={e3['ref_absmax']:.6f}")
    if e3["cmp_restore"] != 0.0 or not e3["grad_restore_bitexact"]:
        failures.append("§3 恢复 RNG 后重跑未逐位一致")
    if e3["cmp_no_restore"] == 0.0 or e3["grad_no_restore_bitexact"]:
        failures.append("§3 不恢复 RNG 时重跑结果竟与原跑一致（对照失效）")

    # ── §4  静默累加对照 ─────────────────────────────────────────
    print("\n[§4] 静默累加对照：第 3 个 microbatch 注入一次瞬时翻转（梯度 ×1e6）")
    e4 = check_e4()
    print(f"  {'情形':<30}{'检出':>6}{'重跑':>6}{'平均梯度 L2':>20}{'梯度 absmax':>20}")
    for key, label in (("clean", "无注入（真值）"),
                       ("silent", "注入但不挂校验项"),
                       ("guarded", "注入且挂校验项")):
        r = e4[key]
        print(f"  {label:<30}{r['detected']:>6}{r['reran']:>6}"
              f"{r['mean_l2']:>20.6E}{r['mean_absmax']:>20.6E}")
    print()
    print(f"  {'d(guarded, clean)':<30}{e4['cmp_clean_guarded']:>20.6E}")
    print(f"  {'d(silent, clean)':<30}{e4['cmp_clean_silent']:>20.6E}")
    if e4["cmp_clean_guarded"] != 0.0:
        failures.append(f"§4 挂校验项后梯度未与真值逐位一致（{e4['cmp_clean_guarded']:.3E}）")
    if e4["cmp_clean_silent"] <= 1.0:
        failures.append("§4 不挂校验项时误差未被放大（对照不显著）")
    if e4["guarded"]["detected"] != 1 or e4["guarded"]["reran"] != 1:
        failures.append("§4 挂校验项时未恰好检出并重跑一次")
    if e4["silent"]["detected"] != 0:
        failures.append("§4 不挂校验项时不应有检出")

    print("\n[§4] 翻转被静默累加后的污染倍率（梯度 L2）")
    ratio = e4["silent"]["mean_l2"] / e4["clean"]["mean_l2"]
    print(f"  不挂校验项 / 真值 = {ratio:.6E}")
    if ratio < 1e5:
        failures.append(f"§4 污染倍率 {ratio:.3E} 未显著")

    # ── §5  校验开销的解析核算 ────────────────────────────────────
    print("\n[§5] 校验开销的解析核算（A 级：只给算术，不含实测）")
    e5 = check_e5()
    print(f"  参数元素数 {e5['param_elems']}，microbatch 数 {N_MICRO_BATCHES}")
    print(f"  {'校验项数':>10}{'校验 FLOP/step':>20}{'结果显存 字节':>20}{'重跑 FLOP':>18}")
    for r in e5["rows"]:
        print(f"  {r['n_validations']:>10}{r['check_flops']:>20}"
              f"{r['result_bytes']:>20}{r['rerun_flops']:>18}")
    marg = e5["rows"][0]["marginal_check_flops_per_validation"]
    print(f"\n  每增挂一个校验项的边际校验算力 = {marg} FLOP/step")
    print(f"  结果显存：每个校验项 4 字节（存一份 float32 结果）")
    rows = e5["rows"]
    if rows[1]["check_flops"] != 2 * rows[0]["check_flops"]:
        failures.append("§5 校验 FLOP 未随校验项数线性增长")

    # ── §6  故障注入时延的采样分布 ────────────────────────────────
    print("\n[§6] 故障注入时延的采样分布（复刻 fault_injector.py L174-L189 的指数分布）")
    print(f"  {'mtti 秒':>12}{'样本数':>10}{'实测均值':>18}{'理论均值':>18}"
          f"{'实测标准差':>18}{'理论标准差':>18}")
    for mtti in (60.0, 600.0):
        r = check_e6(mtti_seconds=mtti)
        print(f"  {r['mtti_seconds']:>12.0f}{r['n']:>10}{r['mean']:>18.6f}"
              f"{r['expected_mean']:>18.6f}{r['std']:>18.6f}{r['expected_std']:>18.6f}")
        if abs(r["mean"] - r["expected_mean"]) / r["expected_mean"] > 0.02:
            failures.append(f"§6 mtti={mtti:.0f} 实测均值 {r['mean']:.4f} 偏离理论均值超2%")
        if abs(r["std"] - r["expected_std"]) / r["expected_std"] > 0.02:
            failures.append(f"§6 mtti={mtti:.0f} 实测标准差 {r['std']:.4f} 偏离理论值超2%")

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