"""投机解码的收益模型：接受率、最优草稿长度与加速比上界。

对应文章：《投机解码（02）：收益模型 —— α、γ、c 三个数决定一切》

基础设定沿用 Leviathan et al. 的 draft → verify 两阶段范式：
单步接受概率 α（独立假设）、草稿长度 γ、草稿/目标单次前向耗时比 c。

  · 每轮产出 token 数期望      E[T] = (1 - α^{γ+1}) / (1 - α)
  · 每轮耗时（以 t_T 为单位）  1 + γ c
  · 加速比                     S(α, γ, c) = (1 - α^{γ+1}) / ((1 - α)(1 + γ c))

三组检查，全部纯 numpy / CPU，固定 seed = 20261018，逐位可复现：

  1. 蒙特卡洛模拟 draft-verify 过程，与 E[T] 闭式对拍
  2. γ* 与 S* 的网格枚举 + 与一阶条件交叉验证
  3. 两条极限边界的数值验证

用法：
    python3 run.py              # 跑全部三组，打印表格
    python3 run.py --selftest   # 只跑断言（静默），适合挂 CI
    python3 run.py 1 2          # 只跑第 1、2 组
"""

import sys

import numpy as np

SEED = 20261018
N_TRIALS = 400_000     # 组 1 每格蒙特卡洛轮数
GAMMA_MAX = 64         # 组 2 枚举上界


# ---------------------------------------------------------------- 模型

def expected_tokens(alpha, gamma):
    """每轮产出 token 数期望 E[T] = (1 - α^{γ+1}) / (1 - α)。"""
    if abs(1.0 - alpha) < 1e-12:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def speedup(alpha, gamma, c):
    """加速比 S(α, γ, c)。"""
    return expected_tokens(alpha, gamma) / (1.0 + gamma * c)


def first_order_residual(alpha, gamma, c):
    """一阶条件 α^{γ+1} (c - (1 + γ c) ln α) - c，γ* 处应约为 0。"""
    return alpha ** (gamma + 1) * (c - (1.0 + gamma * c) * np.log(alpha)) - c


def simulate_round(rng, alpha, gamma, n_trials):
    """模拟 n_trials 轮 draft-verify，返回每轮产出 token 数数组。

    每轮：前 γ 个草稿逐个以概率 α 独立接受；首个被拒位置 j 处产出 j 个 token
    （j-1 个接受 + 1 个从残差分布重采的 bonus）；全部接受则产出 γ+1 个。
    """
    accepted = rng.random((n_trials, gamma)) < alpha
    # 首个被拒的下标；全接受则为 gamma（全 True 时 argmin 会返回 0，必须先判 all）
    all_accepted = accepted.all(axis=1)
    first_reject = np.where(all_accepted, gamma, accepted.argmin(axis=1))
    # 全接受 -> gamma + 1，否则 first_reject + 1
    tokens = np.where(first_reject == gamma, gamma + 1, first_reject + 1)
    return tokens.astype(np.float64)


# ---------------------------------------------------------------- 组 1

def group1():
    """蒙特卡洛与闭式的对拍。"""
    rng = np.random.default_rng(SEED)
    alphas = [0.5, 0.7, 0.9]
    gammas = [1, 2, 4, 8, 16]

    print("=" * 74)
    print("组 1  每轮产出 token 数：蒙特卡洛 vs 闭式  (N = %d 轮/格)" % N_TRIALS)
    print("=" * 74)
    print("  %-8s %6s %14s %14s %10s" % ("alpha", "gamma", "蒙特卡洛", "闭式", "相对误差"))
    print("  " + "-" * 60)

    worst = 0.0
    for a in alphas:
        for g in gammas:
            mc = float(simulate_round(rng, a, g, N_TRIALS).mean())
            ex = expected_tokens(a, g)
            rel = abs(mc - ex) / ex
            worst = max(worst, rel)
            print("  %-8.2f %6d %14.6f %14.6f %9.4f%%" % (a, g, mc, ex, rel * 100))
            assert rel < 0.01, (a, g, mc, ex)

    # 退化情形自校验：γ = 1 时 E[T] = 1 + α
    for a in alphas:
        assert abs(expected_tokens(a, 1) - (1.0 + a)) < 1e-12
    # γ = 0 时（不做投机）恒为 1，加速比恒为 1
    for a in alphas:
        assert abs(expected_tokens(a, 0) - 1.0) < 1e-12
        assert abs(speedup(a, 0, 0.05) - 1.0) < 1e-12
    # α = 0 时每轮恒产出 1 个 token，加速比 <= 1
    assert abs(expected_tokens(0.0, 8) - 1.0) < 1e-12
    assert speedup(0.0, 8, 0.05) < 1.0

    print("")
    print("  最大相对误差 %.4f%%（15 格全部 < 1%%）" % (worst * 100))
    print("  退化情形：γ=1 时 E[T] == 1+α；γ=0 时 E[T]==1 且 S==1；α=0 时 S < 1")
    print("")
    print("  结论：独立接受假设下，闭式 E[T] = (1-α^{γ+1})/(1-α) 与逐轮模拟一致。")
    print("        α=0 时加速比小于 1 —— 草稿全被拒还要付 γc 的账，这是负收益的来源。")
    print("")
    return worst


# ---------------------------------------------------------------- 组 2

def group2():
    """γ* 与 S* 的网格枚举。"""
    alphas = [0.40, 0.60, 0.80, 0.90, 0.95]
    cs = [0.01, 0.05, 0.10, 0.20, 0.50]

    print("=" * 74)
    print("组 2  最优草稿长度与加速比  (γ 枚举上界 = %d)" % GAMMA_MAX)
    print("=" * 74)
    print("  %-12s" % "alpha \\ c" + "".join("%16s" % ("c=%.2f" % c) for c in cs))
    print("  " + "-" * 92)

    grid = {}
    for a in alphas:
        row = []
        for c in cs:
            best_g, best_s = 0, speedup(a, 0, c)
            for g in range(1, GAMMA_MAX + 1):
                s = speedup(a, g, c)
                if s > best_s:
                    best_g, best_s = g, s
            row.append((best_g, best_s))
            grid[(a, c)] = (best_g, best_s)
        print("  %-12s" % ("alpha=%.2f" % a)
              + "".join("%16s" % ("g*=%d, S=%.2f" % (g, s)) for g, s in row))

    print("")

    # 交叉验证一：枚举出的 γ* 应落在一阶条件的连续根附近（±1 之内）
    # R(γ) 递减，R(γ*-1) > 0 > R(γ*+1) 即说明连续根落在 [γ*-1, γ*+1]
    checked = 0
    for (a, c), (g, _s) in grid.items():
        if 0 < g < GAMMA_MAX:
            r_prev = first_order_residual(a, g - 1, c)
            r_next = first_order_residual(a, g + 1, c)
            assert r_prev > 0.0 > r_next, (a, c, g, r_prev, r_next)
            checked += 1
    print("  交叉验证一：%d 个内部解处，一阶条件 R(γ*-1) > 0 > R(γ*+1)" % checked)
    print("             （连续根落在 γ* 的 ±1 之内，枚举解与解析一阶条件一致）")

    # 交叉验证二：单调性 —— γ* 随 c 增大不增，随 α 增大不减
    for a in alphas:
        gs = [grid[(a, c)][0] for c in cs]
        assert all(x >= y for x, y in zip(gs, gs[1:])), (a, gs)
    for c in cs:
        gs = [grid[(a, c)][0] for a in alphas]
        assert all(x <= y for x, y in zip(gs, gs[1:])), (c, gs)
    print("  交叉验证二：γ* 随 c 单调不增、随 α 单调不减（25 格全部成立）")

    # 交叉验证三：加速比不小于 1（否则应当关掉投机，即 γ*=0）
    for (a, c), (g, s) in grid.items():
        assert s >= 1.0 - 1e-12, (a, c, s)
    print("  交叉验证三：最优加速比恒 >= 1（25 格全部成立）")

    print("")
    neg = [(a, c) for (a, c), (g, s) in grid.items() if s < 1.05]
    print("  收益不足 5%% 的组合（即「不该开投机」的区域）：%s" %
          (", ".join("α=%.2f,c=%.2f" % t for t in sorted(neg)) or "无"))
    print("")
    print("  结论：γ* 对 c 极度敏感 —— 同一个 α 下，c 从 0.01 涨到 0.50 会把 γ* 压掉一个量级。")
    print("        「γ 设多大」没有普适答案，它由 c 决定，而 c 取决于草稿器形态。")
    print("")
    return grid


# ---------------------------------------------------------------- 组 3

def group3():
    """两条极限边界的数值验证。"""
    print("=" * 74)
    print("组 3  两条极限边界")
    print("=" * 74)

    # 边界一：c -> 0 时 S -> (1 - α^{γ+1}) / (1 - α)，即纯接受率收益
    print("")
    print("  边界一  c -> 0 时  S -> (1 - α^{γ+1}) / (1 - α)")
    print("  %-8s %6s %14s %14s %10s" % ("alpha", "gamma", "S(c=1e-6)", "E[T] 上界", "相对误差"))
    print("  " + "-" * 60)
    for a in [0.6, 0.8, 0.9]:
        for g in [4, 16]:
            s = speedup(a, g, 1e-6)
            bound = 1.0 / (1.0 - a)              # γ -> ∞ 的上界
            ex = expected_tokens(a, g)
            rel = abs(s - ex) / ex
            print("  %-8.2f %6d %14.6f %14.6f %9.6f%%" % (a, g, s, bound, rel * 100))
            assert rel < 1e-4
            assert s < bound                      # 有限 γ 严格小于 γ->∞ 的上界
    print("  有限 γ 的 S 严格小于 1/(1-α)；误差 < 1e-4")

    # 边界二：α -> 1 时 S -> (γ+1)/(1+γc)
    print("")
    print("  边界二  α -> 1 时  S -> (γ+1) / (1 + γ c)")
    print("  %-8s %6s %8s %14s %14s %10s" % ("alpha", "gamma", "c", "S", "极限值", "相对误差"))
    print("  " + "-" * 64)
    for g in [4, 16]:
        for c in [0.05, 0.20]:
            a = 1.0 - 1e-6
            s = speedup(a, g, c)
            limit = (g + 1.0) / (1.0 + g * c)
            rel = abs(s - limit) / limit
            print("  %-8.6f %6d %8.2f %14.6f %14.6f %9.6f%%" % (a, g, c, s, limit, rel * 100))
            assert rel < 1e-4

    print("")
    print("  两条边界的相对误差均 < 1e-4。")
    print("")
    print("  读法：1/(1-α) 是加速比的硬天花板 —— 与 γ 无关，只由接受率决定。")
    print("        α=0.9 时天花板是 10，无论草稿多长、多便宜都过不去。")
    print("        这正是「把 α 从 0.8 提到 0.9 比把 γ 从 8 加到 16 更值」的量化依据。")
    print("")
    return None


# ---------------------------------------------------------------- 入口

def selftest():
    group1()
    group2()
    group3()
    print("selftest: 3/3 组通过")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if "--selftest" in sys.argv:
        selftest()
        return

    groups = {"1": group1, "2": group2, "3": group3}
    picked = [groups[a] for a in args if a in groups] or list(groups.values())
    for fn in picked:
        fn()


if __name__ == "__main__":
    main()
