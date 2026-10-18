#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prefix-hitrate-ttft-model：前缀命中率 -> TTFT 的一阶模型核验

对应文章：《推理引擎（06）：benchmark 交叉验证》

SGLang v0.5.20 release notes（PR #34565，厂商自报 B 级）称：DeepSeek-V4-Flash +
共享 system prompt 下，token 命中率 43.8% -> 60.8%，mean TTFT 1.57s -> 1.07s。
本文用两个闭式模型核验这组数字**内部是否自洽**，并把流传的
「命中率每 +10pp，TTFT 约降 19%」从模型里推出来，而不是当口诀背。

  组 1  单请求 TTFT 一阶模型：TTFT(h) = (1 - h) * P
        - 两个上报点反解 P，再互相预测，看残差
        - 推出「每 +10pp 的 TTFT 降幅」随 h 的变化（不是常数！）
  组 2  聚合吞吐一阶模型：throughput(h) = base * (1 + k * h)
        - 用 v0.4 cache-aware LB 的两个点（20%->75%，82,665->158,596 tok/s）
          反解 k，验证模型自洽
        - 明确两组模型量纲不同，不能互相换算

纯 Python / CPU、确定性输出。--selftest 只跑断言。

运行：
    python3 run.py
    python3 run.py --selftest
"""

from __future__ import annotations

import json
import sys

# --------------------------------------------------------------------------
# 上报数字（SGLang release notes，厂商自报；此处只验其内部自洽性）
# --------------------------------------------------------------------------

# v0.5.20 / PR #34565：DeepSeek-V4-Flash + shared system prompt
H_BEFORE = 0.438
H_AFTER = 0.608
TTFT_BEFORE = 1.57  # s
TTFT_AFTER = 1.07  # s

# v0.4 cache-aware LB：8xA100，平衡共享前缀负载
H_LB_LOW = 0.20
H_LB_HIGH = 0.75
TP_LOW = 82_665.0  # tok/s
TP_HIGH = 158_596.0  # tok/s


def ttft_model(h: float, p: float) -> float:
    """一阶模型：命中部分零成本，未命中部分付全额 prefill。"""
    return (1.0 - h) * p


def group1() -> dict:
    print("=" * 74)
    print("组 1：单请求 TTFT 一阶模型  TTFT(h) = (1 - h) * P")
    print("=" * 74)

    # 用两个上报点分别反解 P
    p_before = TTFT_BEFORE / (1.0 - H_BEFORE)
    p_after = TTFT_AFTER / (1.0 - H_AFTER)
    print(f"由前点反解：P = {TTFT_BEFORE} / (1 - {H_BEFORE}) = {p_before:.4f} s")
    print(f"由后点反解：P = {TTFT_AFTER} / (1 - {H_AFTER}) = {p_after:.4f} s")

    # 互相预测的残差
    pred_after_from_before = ttft_model(H_AFTER, p_before)
    pred_before_from_after = ttft_model(H_BEFORE, p_after)
    resid_after = abs(pred_after_from_before - TTFT_AFTER)
    resid_before = abs(pred_before_from_after - TTFT_BEFORE)
    print(f"\n用 P_before 预测后点：{pred_after_from_before:.4f}s vs 上报 {TTFT_AFTER}s"
          f"（残差 {resid_after:.4f}s）")
    print(f"用 P_after 预测前点：{pred_before_from_after:.4f}s vs 上报 {TTFT_BEFORE}s"
          f"（残差 {resid_before:.4f}s）")

    # 取平均 P，输出全曲线预测表
    p_mean = (p_before + p_after) / 2.0
    print(f"\n平均 P = {p_mean:.4f}s，全曲线预测：")
    print(f"{'命中率 h':>10}{'TTFT (s)':>12}{'相对 h=43.8% 的变化':>22}")
    base = ttft_model(H_BEFORE, p_mean)
    for h in (0.20, 0.438, 0.50, 0.60, 0.608, 0.80, 0.95):
        t = ttft_model(h, p_mean)
        delta = t / base - 1.0
        mark = "  <- 上报后点" if abs(h - H_AFTER) < 1e-9 else (
            "  <- 上报前点" if abs(h - H_BEFORE) < 1e-9 else "")
        print(f"{h:>10.1%}{t:>12.3f}{delta:>+22.1%}{mark}")

    # 「每 +10pp 降幅」不是常数：d TTFT / dh = -P，相对降幅 = -P / TTFT(h)
    print("\n每 +10pp 命中率的 TTFT 相对降幅（= P / TTFT(h)，随 h 增大）：")
    for h in (0.20, 0.438, 0.60, 0.80):
        slope = p_mean / ttft_model(h, p_mean)
        print(f"  h={h:>5.1%}: 每 +10pp -> TTFT 降 {slope * 0.10:.1%}")

    # 自校验
    # 1. 两个反解 P 应当接近（差异 < 5% 认为一阶模型自洽）
    assert abs(p_before - p_after) / p_mean < 0.05, (p_before, p_after)
    # 2. 平均 P 下两个上报点都能复现到 3% 以内
    assert abs(ttft_model(H_BEFORE, p_mean) - TTFT_BEFORE) / TTFT_BEFORE < 0.03
    assert abs(ttft_model(H_AFTER, p_mean) - TTFT_AFTER) / TTFT_AFTER < 0.03
    # 3. 上报区间中点处的斜率应接近流传的「约 19% / 10pp」
    mid_h = (H_BEFORE + H_AFTER) / 2.0
    slope_mid = p_mean / ttft_model(mid_h, p_mean) * 0.10
    assert 0.15 <= slope_mid <= 0.25, slope_mid
    # 4. 单调性：命中率越高 TTFT 越低
    prev = float("inf")
    for h in [i / 100 for i in range(0, 101)]:
        t = ttft_model(h, p_mean)
        assert t <= prev + 1e-12
        prev = t

    print(f"\n[自校验] 两个反解 P 相差 {abs(p_before - p_after) / p_mean:.2%}（<5%），"
          f"一阶模型与两个上报点自洽")
    print(f"[自校验] 区间中点斜率 {slope_mid:.1%} / +10pp，落在流传「约 19%」附近 ——"
          f"该斜率是模型推论，不是独立证据")
    print()

    return {
        "p_from_before_s": p_before,
        "p_from_after_s": p_after,
        "p_mean_s": p_mean,
        "resid_after_s": resid_after,
        "resid_before_s": resid_before,
        "slope_at_midpoint_per_10pp": slope_mid,
    }


def group2() -> dict:
    print("=" * 74)
    print("组 2：聚合吞吐一阶模型  throughput(h) = base * (1 + k * h)")
    print("=" * 74)

    # 由两点反解 base 与 k
    #   TP_low  = base * (1 + k * h_low)
    #   TP_high = base * (1 + k * h_high)
    ratio = TP_HIGH / TP_LOW
    # ratio = (1 + k*h_high) / (1 + k*h_low)
    # => ratio * (1 + k*h_low) = 1 + k*h_high
    # => k * (ratio*h_low - h_high) = 1 - ratio
    k = (1.0 - ratio) / (ratio * H_LB_LOW - H_LB_HIGH)
    base = TP_LOW / (1.0 + k * H_LB_LOW)
    print(f"两点：h={H_LB_LOW:.0%} -> {TP_LOW:,.0f} tok/s；"
          f"h={H_LB_HIGH:.0%} -> {TP_HIGH:,.0f} tok/s")
    print(f"反解：k = {k:.4f}，base = {base:,.0f} tok/s")

    # 相对斜率不是常数：dTP/TP = k / (1 + k*h)，随 h 增大而减小。
    # 传播中的「每 +10pp 约 +14%」是区间粗估；报告区间中点的瞬时斜率。
    h_mid = (H_LB_LOW + H_LB_HIGH) / 2.0
    slope_mid_tp = k / (1.0 + k * h_mid) * 0.10
    print(f"命中率每 +10pp 的吞吐相对增益（= k/(1+kh)，随 h 增大而减小）：")
    for h in (0.20, h_mid, 0.75, 0.95):
        print(f"  h={h:>5.1%}: 每 +10pp -> 吞吐 +{k / (1.0 + k * h) * 0.10:.1%}")
    print(f"区间中点 h={h_mid:.1%} 处：每 +10pp 约 +{slope_mid_tp:.1%}"
          f"（区间平均粗估约 +14%，量级一致）")

    # 模型复现两个点
    for h, tp in ((H_LB_LOW, TP_LOW), (H_LB_HIGH, TP_HIGH)):
        pred = base * (1.0 + k * h)
        assert abs(pred - tp) / tp < 1e-9, (h, pred, tp)
    print("[自校验] 模型精确复现两个上报点（两点两参，无残差自由度）")

    # 与组 1 的量纲对照：明确不能换算
    p_mean = (TTFT_BEFORE / (1 - H_BEFORE) + TTFT_AFTER / (1 - H_AFTER)) / 2
    slope_ttft_mid = p_mean / ttft_model((H_BEFORE + H_AFTER) / 2, p_mean) * 0.10
    print(f"\n对照：单请求 TTFT 每 +10pp 约 -{slope_ttft_mid:.1%}（组 1，中点位），"
          f"聚合吞吐每 +10pp 约 +{slope_mid_tp:.1%}（组 2，中点位）")
    print("两个数接近是巧合：一个是单请求延迟的相对变化，一个是集群吞吐的相对变化，")
    print("量纲与负载假设都不同，不能互相推导或混排。")

    # 自校验
    assert k > 0, "k 必须为正：命中率越高吞吐越高"
    assert 0.05 <= slope_mid_tp <= 0.30, slope_mid_tp
    # 模型在 h=0 与 h=1 处的行为
    assert base * (1 + k * 1.0) > TP_HIGH  # h=100% 比 75% 还高
    print(f"[自校验] k > 0；h=100% 时模型吞吐 {base * (1 + k):,.0f} tok/s"
          f"（> h=75% 的 {TP_HIGH:,.0f}）")
    print()

    return {
        "k": k,
        "base_tok_s": base,
        "throughput_gain_per_10pp_at_mid": slope_mid_tp,
        "ttft_drop_per_10pp_at_mid": slope_ttft_mid,
    }


def main(argv: list[str]) -> int:
    selftest = "--selftest" in argv
    out = {"ttft": group1(), "throughput": group2()}
    if selftest:
        print("--selftest: 全部断言通过")
        return 0
    print("=" * 74)
    print("结论")
    print("=" * 74)
    print(
        "v0.5.20 上报的命中率/TTFT 两点在 TTFT = (1-h)*P 一阶模型下自洽"
        f"（反解 P 相差 {abs(out['ttft']['p_from_before_s'] - out['ttft']['p_from_after_s']) / out['ttft']['p_mean_s']:.1%}）。\n"
        "「每 +10pp 降约 19%」是该模型在区间中点的斜率，随命中率变化，不是常数。\n"
        "吞吐侧同一直线模型在中点给出每 +10pp 约 +11%（区间平均粗估 +14%，量级一致）。\n"
        "两组模型量纲不同，不能互相换算。它们只是量级校核，不是对 release\n"
        "数字的独立复现——复现需要同硬件同负载重跑。"
    )
    with open("results/summary.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已写入 results/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
