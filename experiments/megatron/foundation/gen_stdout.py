"""把《Megatron-LM 深度剖析（00）：代码地图、配置系统与一次迭代的控制流全图》
正文里的四张表逐行重算一遍，输出与正文逐格对账。

对应文章：
  https://lrypcy.github.io/2026/09/28/megatron-00-foundation/

用法（在本目录）：
    ~/Software/miniconda3/bin/python3 gen_stdout.py            # 写 results/stdout.txt
    ~/Software/miniconda3/bin/python3 gen_stdout.py --stdout   # 只打屏，不写文件

四张表：
  §4.1  参数量自检——公开模型的参数量是确定的，n_params() 必须能对上
  §4.2  交叉验证——独立实现 vs Megatron 官方公式
  §4.4  重计算不计入——官方公式收尾是 return flops_fwd * 3
  §4.5  6ND 是什么，以及它差在哪

单位约定：脚本内部一律 FLOP/token，正文写 GFLOP（= 脚本值 / 1e9）。
「精确」一列默认是 causal 折半口径；§4.5 第二张表用未折半口径，两张表不可混用。
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs import MODELS, BenchConfig          # noqa: E402
from flops import (                             # noqa: E402
    cross_validate,
    training_flops_per_token,
)

GB = 1e9

# 正文 §4.1 / §4.2 / §4.5 的表格数值。SEED 不适用——本箱是纯解析，无随机数。
# fmt 里写的是正文的有效数字（3 位小数），容差按正文有效位定。
ARTICLE = {
    "llama-7b": {
        "seq": 4096,
        "total": 6.738, "no_embed": 6.476, "gated": True,
        "official": 42.864, "sixnd": 38.856, "off_6nd_pct": 10.3,
        "unfolded_exact": 46.08, "unfolded_pct": 18.6,
        "recompute_none": 46.08, "recompute_full": 61.18,
    },
    "llama3-8b": {
        "seq": 8192,
        "total": 8.030, "no_embed": 6.979, "gated": True,
        "official": 51.470, "sixnd": 41.876, "off_6nd_pct": 22.9,
        "unfolded_exact": None, "unfolded_pct": 38.3,
        "recompute_none": None, "recompute_full": None,
    },
    "qwen-14b": {
        "seq": 8192,
        "total": 14.769, "no_embed": 13.212, "gated": True,
        "official": 96.023, "sixnd": 79.272, "off_6nd_pct": 21.1,
        "unfolded_exact": None, "unfolded_pct": None,
        "recompute_none": None, "recompute_full": None,
    },
    "gpt3-175b": {
        "seq": 2048,
        "total": 175.181, "no_embed": 173.946, "gated": False,
        "official": 1061.878, "sixnd": 1043.677, "off_6nd_pct": 1.7,
        "unfolded_exact": None, "unfolded_pct": None,
        "recompute_none": None, "recompute_full": None,
    },
}

TOL = 0.05          # 绝对容差（有效位 3 位小数 → 0.001，宽松到 0.05 覆盖舍入）
failures: list[str] = []


def check(tag: str, got: float, want: float | None, unit: str) -> None:
    if want is None:
        # 正文没给这个数（只给了同组的百分比），只报不判
        print(f"  {tag:<34}{got:>14.3f}{unit}{'—':>14}{'   (正文未列)'}")
        return
    ok = abs(got - want) <= TOL
    if not ok:
        failures.append(f"{tag}: 脚本 {got:.3f}{unit} vs 正文 {want:.3f}{unit}")
    print(f"  {tag:<34}{got:>14.3f}{unit}{want:>14.3f}{unit}   {'ok' if ok else 'MISMATCH'}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true", help="只打屏，不写 results/")
    args = ap.parse_args(argv)

    buf = io.StringIO()
    real = sys.stdout
    if not args.stdout:
        sys.stdout = buf

    print("=" * 78)
    print("gen_stdout.py —— 《Megatron-LM 深度剖析（00）》四张表逐格对账")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print("=" * 78)

    # ── §4.1 参数量自检 ──────────────────────────────────────────────
    print("\n[§4.1] 参数量自检（对齐公开模型公布值）")
    print(f"  {'模型':<12}{'total(B)':>12}{'正文':>10}{'no-embed(B)':>14}{'正文':>10}{'gated':>8}")
    for name, a in ARTICLE.items():
        m = MODELS[name]
        tot = m.n_params() / 1e9
        ne = m.n_params_no_embedding() / 1e9
        print(f"  {name:<12}{tot:>12.3f}{a['total']:>10.3f}{ne:>14.3f}{a['no_embed']:>10.3f}"
              f"{str(m.gated_mlp):>8}")
        check(f"{name} total", tot, a["total"], " B")
        check(f"{name} no-embed", ne, a["no_embed"], " B")
        if bool(m.gated_mlp) != a["gated"]:
            failures.append(f"{name} gated: 脚本 {m.gated_mlp} vs 正文 {a['gated']}")

    # ── §4.2 交叉验证 ────────────────────────────────────────────────
    print("\n[§4.2] 交叉验证：独立实现（causal 折半） vs Megatron 官方公式")
    print(f"  {'模型':<12}{'seq':>7}{'独立实现':>14}{'官方公式':>14}{'偏差%':>10}")
    for name, a in ARTICLE.items():
        cv = cross_validate(BenchConfig(model=name, seq_len=a["seq"]))
        ours = cv["ours_causal"] / GB
        off = cv["megatron_official"] / GB
        print(f"  {name:<12}{a['seq']:>7}{ours:>14.3f}{off:>14.3f}"
              f"{cv['ours_vs_official_pct']:>10.2f}")
        check(f"{name} 独立实现", ours, a["official"], " GFLOP")
        check(f"{name} 官方公式", off, a["official"], " GFLOP")

    # ── §4.4 重计算不计入 ────────────────────────────────────────────
    print("\n[§4.4] 重计算不计入：官方公式收尾 return flops_fwd * 3，开 recompute 不涨")
    for name, a in ARTICLE.items():
        if a["recompute_none"] is None or a["recompute_full"] is None:
            continue
        t = training_flops_per_token(BenchConfig(model=name, seq_len=a["seq"]))
        # training_flops_per_token 返回原始 FLOP，正文写 GFLOP
        none = t["total_none"] / GB
        full = t["total_full_recompute"] / GB
        sel = t["total_selective_core_attn"] / GB
        print(f"  {name:<12}none {none:>8.2f}  selective {sel:>8.2f}  full {full:>8.2f} GFLOP"
              f"   full/none = {full / none:.3f}")
        check(f"{name} total_none", none, a["recompute_none"], " GFLOP")
        check(f"{name} total_full", full, a["recompute_full"], " GFLOP")
        print(f"  {'':<12}实际算力增量 = {full / none:.3f}x  → +{100 * (full / none - 1):.1f}%")

    # ── §4.5 6ND 对照 ────────────────────────────────────────────────
    print("\n[§4.5-表1] 官方口径 vs 6ND")
    print(f"  {'模型':<12}{'seq':>7}{'官方口径':>13}{'6ND':>12}{'偏差%':>10}")
    for name, a in ARTICLE.items():
        cv = cross_validate(BenchConfig(model=name, seq_len=a["seq"]))
        print(f"  {name:<12}{a['seq']:>7}{cv['megatron_official'] / GB:>13.3f}"
              f"{cv['6nd'] / GB:>12.3f}{cv['official_vs_6nd_pct']:>10.1f}")
        check(f"{name} 官方口径", cv["megatron_official"] / GB, a["official"], " GFLOP")
        check(f"{name} 6ND", cv["6nd"] / GB, a["sixnd"], " GFLOP")
        check(f"{name} 偏差%", cv["official_vs_6nd_pct"], a["off_6nd_pct"], "%")

    print("\n[§4.5-表2] 独立实现、未折半口径 vs 6ND（偏差随序列长度放大）")
    for name, a in ARTICLE.items():
        if a["unfolded_pct"] is None:
            continue
        cv = cross_validate(BenchConfig(model=name, seq_len=a["seq"]))
        exact = cv["ours_no_causal"] / GB
        pct = 100.0 * (exact - cv["6nd"] / GB) / (cv["6nd"] / GB)
        print(f"  {name:<12}{a['seq']:>7}{exact:>13.3f}{pct:>12.1f}%")
        check(f"{name} 未折半精确", exact, a["unfolded_exact"], " GFLOP")
        check(f"{name} 未折半偏差%", pct, a["unfolded_pct"], "%")

    print("\n" + "=" * 78)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for f in failures:
            print("  !! " + f)
    else:
        print("对账通过：正文四张表的每一个数字都与本目录脚本实跑一致。")
    print("=" * 78)

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
