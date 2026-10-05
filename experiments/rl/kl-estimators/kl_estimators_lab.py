#!/usr/bin/env python3
"""KL 估计器 k1 / k2 / k3：值的偏差与方差、梯度的偏差与方差、维度与序列长度的影响。

约定（与配套博客一致）：样本 x ~ q，r = p(x)/q(x)，delta = log r = log p - log q，
估的是 reverse KL  KL(q||p) = E_q[-log r]。

    k1 = -log r                     无偏，可负
    k2 = 1/2 (log r)^2              有偏，恒非负
    k3 = (r - 1) - log r            无偏，恒非负

纯 numpy，CPU 可跑。分组运行（默认全跑）：

    python3 kl_estimators_lab.py            # 全部四组
    python3 kl_estimators_lab.py 1 2        # 只跑值估计与梯度无偏性
    python3 kl_estimators_lab.py --selftest # 只跑自校验
"""

import sys
import numpy as np

VOCAB = 2000      # 值估计那组的词表大小
VOCAB_G = 300     # 梯度那组用的小词表（梯度本身是 V 维向量）
TRIALS = 4000     # 值估计：每批 N 个样本，重复 TRIALS 批
MC = 400000       # 维度组 / 序列组的蒙特卡洛条数


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def three_estimators(logr):
    """给定一批样本的 log r，返回 (k1, k2, k3)。"""
    r = np.exp(logr)
    return -logr, 0.5 * logr ** 2, (r - 1.0) - logr


# --------------------------------------------------------------------------
# 1. 值估计：偏差与方差随批大小的变化
# --------------------------------------------------------------------------
def group_value():
    print("=" * 78)
    print("1. 值估计：同一批样本上三个估计器的偏差与方差")
    print("=" * 78)
    print("构造：词表 V=%d，q = softmax(theta)，p = softmax(theta + delta)，" % VOCAB)
    print("      delta ~ N(0, s^2 I)。真实 KL 用全词表求和算出，不靠采样。")
    print("      与《RL 符号与计算全景》§7.1 同一套构造（不同随机种子），")
    print("      那篇看的是单样本口径，这里看的是「批大小」这一维。\n")

    rng = np.random.default_rng(20261005)
    logits = rng.normal(0.0, 1.0, VOCAB)
    for tag, s in [("A 轻微漂移", 0.15), ("B 大幅漂移", 1.20)]:
        q = softmax(logits)
        p = softmax(logits + rng.normal(0.0, s, VOCAB))
        logr_all = np.log(p) - np.log(q)
        kl_true = float(np.sum(q * (-logr_all)))

        print("--- regime %s（logit 扰动标准差 %.2f），真实 KL = %.5f ---" % (tag, s, kl_true))
        print("%6s %10s %12s %12s %12s %10s" % ("N", "估计器", "均值", "偏差", "标准差", "std/真值"))
        for N in (1, 16, 256):
            est = np.empty((TRIALS, 3))
            for t in range(TRIALS):
                x = rng.choice(VOCAB, size=N, p=q)
                k1v, k2v, k3v = three_estimators(logr_all[x])
                est[t] = (k1v.mean(), k2v.mean(), k3v.mean())
            for j, name in enumerate(("k1", "k2", "k3")):
                col = est[:, j]
                print("%6d %10s %12.5f %12.5f %12.5f %9.2fx"
                      % (N, name, col.mean(), col.mean() - kl_true, col.std(),
                         col.std() / kl_true))
            neg = float(np.mean(est[:, 0] < 0))
            print("%6s %10s   负值占比 %.1f%%（k2/k3 恒非负，不列）" % ("", "k1", 100 * neg))
        print()


# --------------------------------------------------------------------------
# 2. 梯度无偏性：精确期望 + 蒙特卡洛方差
# --------------------------------------------------------------------------
def group_grad():
    print("=" * 78)
    print("2. 梯度：谁无偏、谁有偏，以及各自的方差")
    print("=" * 78)
    rng = np.random.default_rng(7)
    th_q = rng.normal(0.0, 1.0, VOCAB_G)
    q = softmax(th_q)
    p = softmax(th_q + rng.normal(0.0, 0.35, VOCAB_G))
    logr_all = np.log(p) - np.log(q)
    r_all = np.exp(logr_all)

    # softmax 参数化：nabla_theta log p(x) = e_x - p
    # 真梯度  nabla KL(q||p_theta) = p - q
    grad_true = p - q
    kl_true = float(np.sum(q * (-logr_all)))
    print("词表 V=%d，真实 KL = %.5f，||真梯度|| = %.5f" % (VOCAB_G, kl_true, np.linalg.norm(grad_true)))
    print("真梯度取闭式 p - q（由 nabla KL = -E_q[nabla log p] 与 nabla log p = e_x - p 推出）\n")

    def expect(w):
        """E_q[w(x) * nabla log p(x)] = (q*w) - (sum q*w) * p"""
        return q * w - float(np.sum(q * w)) * p

    print("--- 精确期望（不做采样，直接对词表求和）---")
    print("%10s %18s %10s" % ("", "相对偏差", "判定"))
    for name, g in (("grad k1", -expect(np.ones(VOCAB_G))),
                    ("grad k2", expect(logr_all)),
                    ("grad k3", expect(r_all) - expect(np.ones(VOCAB_G)))):
        rel = np.linalg.norm(g - grad_true) / np.linalg.norm(grad_true)
        print("%10s %18.6f %10s" % (name, rel, "无偏" if rel < 1e-9 else "有偏"))
    print("  依据：E_q[r * nabla log p] = sum_x p(x)(e_x - p) = 0，实测范数 %.2e"
          % np.linalg.norm(expect(r_all)))
    print("  而 E_q[log r * nabla log p] 不为零，这就是 k2 梯度有偏的来源\n")

    print("--- 蒙特卡洛方差（每批 64 个样本，%d 批）---" % 400)
    B, N = 400, 64
    idx = rng.choice(VOCAB_G, size=(B, N), p=q)
    lr = logr_all[idx]
    rv = np.exp(lr)
    tally = np.zeros((3, B, VOCAB_G))
    for b in range(B):
        xb = idx[b]
        counts = np.bincount(xb, minlength=VOCAB_G) / N
        score_mean = counts - p                       # (1/N) sum (e_x - p)
        tally[0, b] = -score_mean                                    # grad k1
        tally[1, b] = np.bincount(xb, weights=lr[b], minlength=VOCAB_G) / N - lr[b].mean() * p
        tally[2, b] = (np.bincount(xb, weights=rv[b] - 1.0, minlength=VOCAB_G) / N
                       - (rv[b] - 1.0).mean() * p)
    print("%10s %22s" % ("", "批均值梯度的总方差"))
    for i, name in enumerate(("grad k1", "grad k2", "grad k3")):
        print("%10s %22.4e" % (name, tally[i].var(axis=0).sum()))
    print()


# --------------------------------------------------------------------------
# 3. 维度：grad k3 的方差随维度爆炸
# --------------------------------------------------------------------------
def group_dim():
    print("=" * 78)
    print("3. 维度：grad k3 带着 (r-1) 因子，方差随维度指数增长")
    print("=" * 78)
    print("构造：q = N(0, I_D)，p = N(mu, I_D)，对 p 的参数 mu 求梯度。")
    print("     log r = mu*S - D*mu^2/2，S = sum_d x_d ~ N(0, D)（可直采，不必存 D 维样本）。")
    print("     真实 KL = D*mu^2/2，真实梯度 d KL/d mu = D*mu。\n")
    mu = 0.05
    rng = np.random.default_rng(11)
    print("%6s %10s %12s %14s %14s %16s"
          % ("D", "真实 KL", "真实梯度", "std(grad k1)", "std(grad k3)", "E[(r-1)^2] 解析"))
    for D in (1, 10, 100, 1000):
        S = rng.normal(0.0, np.sqrt(D), MC)
        logr = mu * S - 0.5 * D * mu * mu
        r = np.exp(logr)
        dlogr = S - D * mu                    # d log r / d mu
        g1 = -dlogr
        g3 = (r - 1.0) * dlogr
        s1, s3 = g1.std(), g3.std()
        # 解析：r 是对数正态，Var(r) = exp(D*mu^2) - 1
        print("%6d %10.3f %12.3f %14.4g %14.4g %16.4g"
              % (D, 0.5 * D * mu * mu, D * mu, s1, s3, np.exp(D * mu * mu) - 1.0))
    print("\n  最后一列是解析值，不靠采样：r 为对数正态，E[r]=1 而 Var(r)=exp(D*mu^2)-1，")
    print("  随维度指数增长。grad k3 里乘了 (r-1)，二阶矩被这一项直接放大；grad k1 没有。")
    print("  注意 std(grad k3) 那一列本身是重尾量的蒙特卡洛估计，换 M 会明显变化，")
    print("  **不要拿它的具体数值当结论**，要看的是最后一列的解析增长律。")
    print("  两个梯度的均值都对得上真实梯度（无偏），差别只在方差。\n")


# --------------------------------------------------------------------------
# 4. 序列长度：有限样本下 k3 系统性低估
# --------------------------------------------------------------------------
def group_seq():
    print("=" * 78)
    print("4. 序列长度：E_q[r] = 1 靠极稀有样本撑起来，k3 在有限样本下系统性低估")
    print("=" * 78)
    print("构造：每个 token 的 log r ~ N(-sigma^2/2, sigma^2)（这样 E[e^{log r}] = 1），")
    print("     序列级 log r = 各 token 之和 ~ N(-T*sigma^2/2, T*sigma^2)（同样可直采）。")
    print("     真实 KL = T*sigma^2/2。\n")
    rng = np.random.default_rng(13)
    print("%6s %6s %10s %12s %14s %12s" % ("T", "sigma", "真实 KL", "mean k1", "mean k2", "mean k3"))
    for sigma in (0.2, 0.5):
        for T in (16, 64, 256, 1024):
            d = rng.normal(-0.5 * T * sigma * sigma, np.sqrt(T) * sigma, MC)
            k1, k2, k3 = three_estimators(d)
            print("%6d %6.1f %10.3f %12.3f %14.3f %12.3f"
                  % (T, sigma, 0.5 * T * sigma * sigma, k1.mean(), k2.mean(), k3.mean()))
    print("\n  k3 与真值的差 = 样本里 (r-1) 的均值，它恒为负：")
    print("  E[r]=1 由极稀有样本贡献，Var(r)=exp(T*sigma^2)-1，样本量再大也采不到那条尾巴，")
    print("  于是 mean(r) -> 0，差值趋向 -1 nat。T*sigma^2 不大时这一项只是噪声（见 T=256/sigma=0.2 行）。")
    print("  k2 则随 T 二次发散（T=1024/sigma=0.2 时 229.92 对真值 20.48）。\n")


# --------------------------------------------------------------------------
# 自校验：构造真值 <-> 反算值
# --------------------------------------------------------------------------
def selftest():
    print("=" * 78)
    print("自校验：构造真值与反算值必须对得上")
    print("=" * 78)
    ok = True

    rng = np.random.default_rng(0)
    q = softmax(rng.normal(0, 1, 500))
    p = softmax(np.log(q) + rng.normal(0, 0.4, 500))
    lr = np.log(p) - np.log(q)
    kl = float(np.sum(q * (-lr)))
    # k1 的期望就是 KL（定义）
    k1 = float(np.sum(q * (-lr)))
    # k3 的期望：E_q[r-1] 必须为 0
    cv = float(np.sum(q * (np.exp(lr) - 1.0)))
    # k2 的期望与 KL 不同（这是它「有偏」的定义）
    k2 = float(np.sum(q * 0.5 * lr ** 2))
    print("  E_q[k1] - KL            = %+.3e    (应为 0)" % (k1 - kl))
    print("  E_q[r-1] 控制变量       = %+.3e    (应为 0)" % cv)
    print("  E_q[k2] - KL            = %+.3e    (非 0，即 k2 的偏差)" % (k2 - kl))
    ok &= abs(k1 - kl) < 1e-12 and abs(cv) < 1e-12 and k2 > kl

    # 高斯闭式：KL(N(0,1) || N(mu,1)) = mu^2/2，用 MC 反算
    mu, D, M = 0.05, 1000, 400000
    S = rng.normal(0.0, np.sqrt(D), M)
    d = mu * S - 0.5 * D * mu * mu
    print("  高斯 KL 真值 %.5f，MC 反算 %.5f（差 %+.2e，应为小量）"
          % (0.5 * D * mu * mu, (-d).mean(), (-d).mean() - 0.5 * D * mu * mu))
    ok &= abs((-d).mean() - 0.5 * D * mu * mu) < 5e-3

    # 序列闭式：T*sigma^2/2
    sigma, T = 0.2, 64
    d = rng.normal(-0.5 * T * sigma * sigma, np.sqrt(T) * sigma, M)
    print("  序列 KL 真值 %.5f，MC 反算 %.5f（差 %+.2e，应为小量）"
          % (0.5 * T * sigma * sigma, (-d).mean(), (-d).mean() - 0.5 * T * sigma * sigma))
    ok &= abs((-d).mean() - 0.5 * T * sigma * sigma) < 5e-3

    print("\n  自校验 %s" % ("通过" if ok else "失败"))
    return 0 if ok else 1


GROUPS = {"1": group_value, "2": group_grad, "3": group_dim, "4": group_seq}


def main(argv):
    if "--selftest" in argv:
        return selftest()
    picked = [a for a in argv if a in GROUPS] or sorted(GROUPS)
    print("KL 估计器实验室 · 分组 %s\n" % ",".join(picked))
    for g in picked:
        GROUPS[g]()
    print("完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
