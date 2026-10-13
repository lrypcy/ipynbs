#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VAE 核心机制的数值验证（纯 numpy，CPU 可跑，固定 seed）。

对应文章：lrypcy.github.io 的《变分自编码器（01）：ELBO、重参数化与潜空间》。

五组检查，每组都做「解析真值 ↔ 数值反算」自校验：

  1. 高斯后验 KL 的闭式解 vs 蒙特卡洛估计
  2. ELBO 的紧度恒等式 log p(x) = ELBO + KL(q || p(z|x))，以及「最优 q 就是真后验」
  3. 重参数化梯度 vs score-function（REINFORCE）梯度的方差对比
  4. 平方误差下的模态平均为什么输出模糊（与「选单模态」的定量权衡）
  5. beta 扫描：KL 权重过大怎样把潜维度压死

用法：
    python3 run.py            # 跑全部
    python3 run.py --selftest # 只跑自校验断言
"""

import argparse
import sys

import numpy as np

SEED = 20261013
rng = np.random.default_rng(SEED)

LINE = "-" * 72


def section(title):
    print()
    print(LINE)
    print(title)
    print(LINE)


# ----------------------------------------------------------------------------
# 1. 高斯后验的 KL：闭式解 vs 蒙特卡洛
# ----------------------------------------------------------------------------
def kl_closed_form(mu, logvar):
    """D_KL( N(mu, diag(exp(logvar))) || N(0, I) )，闭式解（nats，逐维求和）。"""
    return 0.5 * np.sum(mu ** 2 + np.exp(logvar) - 1.0 - logvar, axis=-1)


def kl_monte_carlo(mu, logvar, n_samples, rng):
    """同一个 KL 的蒙特卡洛估计：E_q[ log q(z) - log p(z) ]。"""
    d = mu.shape[-1]
    sigma = np.exp(0.5 * logvar)
    eps = rng.standard_normal((n_samples, d))
    z = mu + sigma * eps
    # log N(z; mu, sigma^2)
    log_q = -0.5 * np.sum(((z - mu) / sigma) ** 2 + 2.0 * np.log(sigma)
                          + np.log(2.0 * np.pi), axis=-1)
    # log N(z; 0, I)
    log_p = -0.5 * np.sum(z ** 2 + np.log(2.0 * np.pi), axis=-1)
    return float(np.mean(log_q - log_p))


def lab_1():
    section("1. 高斯后验的 KL：闭式解 vs 蒙特卡洛估计")
    d = 8
    mu = rng.normal(0.0, 0.8, size=d)
    logvar = rng.normal(-0.3, 0.6, size=d)
    truth = float(kl_closed_form(mu, logvar))

    print(f"潜维度 d = {d}")
    print(f"闭式解  D_KL = {truth:.6f} nats")
    print()
    print(f"{'样本数':>10} | {'蒙特卡洛估计':>16} | {'绝对误差':>12} | {'标准误(估)':>12}")
    print("-" * 60)
    for n in (1_000, 10_000, 100_000, 1_000_000):
        # 独立重复 20 次，给出误差的标准差（即蒙特卡洛标准误）
        ests = np.array([kl_monte_carlo(mu, logvar, n, rng) for _ in range(20)])
        err = float(np.mean(np.abs(ests - truth)))
        print(f"{n:>10} | {float(np.mean(ests)):>16.6f} | {err:>12.6f} | "
              f"{float(np.std(ests)):>12.6f}")

    # 自校验：大样本下估计必须收敛到闭式解
    est = kl_monte_carlo(mu, logvar, 2_000_000, rng)
    assert abs(est - truth) < 0.01, (est, truth)
    # 自校验：mu=0, sigma=1 时 KL 必须为 0
    zero = float(kl_closed_form(np.zeros(d), np.zeros(d)))
    assert abs(zero) < 1e-12, zero
    print()
    print(f"自校验通过：N=2e6 时 MC 估计 {est:.6f} 与闭式解 {truth:.6f} 相差 "
          f"{abs(est - truth):.6f}；mu=0, logvar=0 时 KL = {zero:.1e}")
    return {"closed_form": truth, "mc_2e6": est}


# ----------------------------------------------------------------------------
# 2. ELBO 的紧度恒等式
# ----------------------------------------------------------------------------
def lab_2():
    section("2. ELBO 的紧度恒等式：log p(x) = ELBO + KL(q || p(z|x))")
    # 一维 toy：p(z) = N(0, 1)，p(x|z) = N(z, sigma_x^2)
    sigma_x = 0.8
    x = 0.70

    # 边际：p(x) = N(0, 1 + sigma_x^2)
    log_px = -0.5 * (np.log(2 * np.pi * (1 + sigma_x ** 2))
                     + x ** 2 / (1 + sigma_x ** 2))
    # 真后验：p(z|x) = N(m, s^2)
    m = x / (1 + sigma_x ** 2)
    s2 = sigma_x ** 2 / (1 + sigma_x ** 2)
    s = np.sqrt(s2)

    def elbo(mu, logvar):
        var = np.exp(logvar)
        # E_q[log p(x|z)] = -0.5 log(2 pi sigma_x^2) - E[(x-z)^2] / (2 sigma_x^2)
        recon = -0.5 * np.log(2 * np.pi * sigma_x ** 2) \
                - ((x - mu) ** 2 + var) / (2 * sigma_x ** 2)
        return recon - 0.5 * (mu ** 2 + var - 1.0 - logvar)

    def kl_gauss(mu, var, m2, v2):
        """KL( N(mu, var) || N(m2, v2) ) 的闭式解。"""
        return 0.5 * (np.log(v2 / var) + (var + (mu - m2) ** 2) / v2 - 1.0)

    print(f"sigma_x = {sigma_x}, 观测 x = {x}")
    print(f"真后验 p(z|x) = N({m:.6f}, {s2:.6f})")
    print(f"log p(x)（解析）     = {log_px:.10f}")
    print()
    print("随机抽 6 组 (mu, logvar)，检查恒等式：")
    print(f"{'mu':>10} {'logvar':>10} | {'ELBO':>14} {'KL(q||p(z|x))':>16} "
          f"{'两者之和':>14} {'残差':>12}")
    print("-" * 82)
    max_res = 0.0
    for _ in range(6):
        mu = float(rng.normal(0.0, 1.2))
        logvar = float(rng.normal(0.0, 1.0))
        var = np.exp(logvar)
        e = float(elbo(mu, logvar))
        k = float(kl_gauss(mu, var, m, s2))
        res = abs(e + k - log_px)
        max_res = max(max_res, res)
        print(f"{mu:>10.4f} {logvar:>10.4f} | {e:>14.8f} {k:>16.8f} "
              f"{e + k:>14.8f} {res:>12.2e}")
    assert max_res < 1e-9, max_res

    # 网格搜索最大化 ELBO，检查最优点是否就是真后验
    mus = np.linspace(-2.0, 2.0, 4001)
    logvars = np.linspace(-6.0, 3.0, 4001)
    MM, LL = np.meshgrid(mus, logvars, indexing="ij")
    E = elbo(MM, LL)
    idx = np.unravel_index(np.argmax(E), E.shape)
    mu_hat, lv_hat = float(mus[idx[0]]), float(logvars[idx[1]])
    sigma_hat = np.exp(0.5 * lv_hat)

    print()
    print("网格搜索 max_{mu, sigma} ELBO：")
    print(f"  数值最优   mu = {mu_hat:.6f},  sigma = {sigma_hat:.6f},  "
          f"ELBO = {float(E[idx]):.8f}")
    print(f"  真后验     mu = {m:.6f},  sigma = {s:.6f}")
    print(f"  |Δmu| = {abs(mu_hat - m):.2e},  |Δsigma| = {abs(sigma_hat - s):.2e},  "
          f"log p(x) - ELBO* = {log_px - float(E[idx]):.2e}")
    assert abs(mu_hat - m) < 1e-3 and abs(sigma_hat - s) < 1e-3

    # 自校验：ELBO 必须处处 <= log p(x)（下界性质）
    gap = log_px - E
    assert gap.min() > -1e-12, gap.min()
    print()
    print(f"自校验通过：{E.size} 个网格点上 log p(x) - ELBO 的最小值 = {gap.min():.2e} >= 0"
          "（下界性质成立）")
    return {"log_px": log_px, "elbo_star": float(E[idx])}


# ----------------------------------------------------------------------------
# 3. 重参数化梯度 vs score-function 梯度
# ----------------------------------------------------------------------------
def lab_3():
    section("3. 重参数化梯度 vs score-function（REINFORCE）梯度的方差")
    # 目标 J(mu) = E_{z ~ N(mu, sigma^2)}[ (z - c)^2 ]，真梯度 dJ/dmu = 2(mu - c)
    sigma = 1.0
    c = 2.0
    mus = np.array([-1.0, 0.0, 1.0])
    n = 20_000

    print(f"目标 J(mu) = E[(z - c)^2]，z ~ N(mu, sigma^2)，sigma = {sigma}, c = {c}")
    print(f"{'mu':>6} | {'真梯度':>10} | {'重参数化 均值/方差':>26} | "
          f"{'REINFORCE 均值/方差':>26} | {'方差比':>10}")
    print("-" * 96)
    ratios = []
    for mu in mus:
        eps = rng.standard_normal(n)
        z = mu + sigma * eps
        true_grad = 2.0 * (mu - c)

        g_rep = 2.0 * (z - c)                       # d/dmu (z-c)^2，z = mu + sigma eps
        g_sf = (z - c) ** 2 * (z - mu) / sigma ** 2  # f(z) * d log q / dmu

        vr, vs = float(np.var(g_rep)), float(np.var(g_sf))
        ratios.append(vs / vr)
        print(f"{mu:>6.1f} | {true_grad:>10.4f} | "
              f"{float(np.mean(g_rep)):>10.4f} / {vr:>11.4f}     | "
              f"{float(np.mean(g_sf)):>10.4f} / {vs:>11.4f}     | {vs / vr:>10.1f}x")

    # 自校验：两个估计器都必须无偏（样本均值贴近真梯度）
    for mu in mus:
        eps = rng.standard_normal(n)
        z = mu + sigma * eps
        g_rep = float(np.mean(2.0 * (z - c)))
        g_sf = float(np.mean((z - c) ** 2 * (z - mu) / sigma ** 2))
        true_grad = 2.0 * (mu - c)
        assert abs(g_rep - true_grad) < 0.05, (mu, g_rep, true_grad)
        assert abs(g_sf - true_grad) < 0.10, (mu, g_sf, true_grad)
    print()
    print(f"自校验通过：两个估计器的样本均值都落在真梯度 0.05 以内；"
          f"REINFORCE 的方差是重参数化的 {min(ratios):.0f}~{max(ratios):.0f} 倍")
    return {"var_ratio": ratios}


# ----------------------------------------------------------------------------
# 4. 平方误差的模态平均为什么模糊
# ----------------------------------------------------------------------------
def lab_4():
    section("4. 平方误差的模态平均：为什么 MSE 最优解是「糊」的")
    # 两条同样合理的重建：阶跃边缘落在第 30 或第 34 个像素，后验各 0.5
    L = 64
    idx = np.arange(L)
    y1 = (idx >= 30).astype(float)
    y2 = (idx >= 34).astype(float)
    mse_opt = 0.5 * (y1 + y2)          # 最小化期望平方误差的唯一最优解

    def sharpness(v):
        """锐度：一阶梯度的峰值（阶跃信号为 1.0，被摊开后小于 1）。"""
        return float(np.max(np.abs(np.diff(v))))

    def hf_energy(v):
        """高频能量：二阶差分的 L2 范数（模糊化后显著下降）。"""
        return float(np.linalg.norm(np.diff(v, n=2)))

    print(f"信号长度 L = {L}；两个等概率模态：边缘在 30 / 边缘在 34")
    print()
    print(f"{'解码策略':<22} | {'期望 MSE':>12} | {'边缘锐度':>10} | {'高频能量':>10}")
    print("-" * 64)

    def report(name, predictor, mode):
        if mode == "mean":
            mse = float(np.mean((mse_opt - y1) ** 2) * 0.5
                        + np.mean((mse_opt - y2) ** 2) * 0.5)
            shp, hf = sharpness(mse_opt), hf_energy(mse_opt)
        elif mode == "single":
            mse = float(np.mean((predictor - y1) ** 2) * 0.5
                        + np.mean((predictor - y2) ** 2) * 0.5)
            shp, hf = sharpness(predictor), hf_energy(predictor)
        else:  # 随机挑一个模态
            mse = float(0.5 * np.mean((y1 - y1) ** 2) + 0.5 * np.mean((y2 - y2) ** 2)
                        + 0.0)
            mse = float(0.5 * np.mean((y1 - y2) ** 2))
            shp, hf = sharpness(y1), hf_energy(y1)
        print(f"{name:<22} | {mse:>12.4f} | {shp:>10.4f} | {hf:>10.4f}")
        return mse, shp, hf

    r_mean = report("MSE 最优（模态平均）", mse_opt, "mean")
    r_one = report("总是选 y1", y1, "single")
    r_rand = report("随机选一个模态", None, "random")

    # 自校验：MSE 最优解的期望 MSE 必须严格最小（解析值 0.25 * ||y1-y2||^2）
    d2 = float(np.mean((y1 - y2) ** 2))          # 逐元素平均，与上面的 MSE 同口径
    assert abs(r_mean[0] - 0.25 * d2) < 1e-9
    assert abs(r_one[0] - 0.50 * d2) < 1e-9
    assert r_mean[0] < r_one[0]
    # 自校验：MSE 最优解的锐度必须只有真实模态的一半
    assert abs(r_mean[1] - 0.5 * sharpness(y1)) < 1e-9
    print()
    print(f"自校验通过：MSE 最优解的期望 MSE = 0.25·||y1-y2||² = {r_mean[0]:.4f}，"
          f"严格小于选单模态的 0.50·||y1-y2||² = {r_one[0]:.4f}；"
          f"但锐度只有 {r_mean[1]:.2f}（真实模态为 {sharpness(y1):.2f}）")
    print("结论：MSE 用「牺牲锐度」换「最低期望误差」——这正是像素级 VAE 显得糊的机制。")
    return {"mse_mean": r_mean[0], "mse_single": r_one[0],
            "sharp_mean": r_mean[1], "sharp_true": sharpness(y1)}


# ----------------------------------------------------------------------------
# 5. beta 扫描：KL 权重怎样压死潜维度
# ----------------------------------------------------------------------------
def _make_data(n, d_obs, d_true, rng):
    w = rng.normal(0.0, 1.0 / np.sqrt(d_true), size=(d_true, d_obs))
    z = rng.standard_normal((n, d_true))
    x = z @ w + 0.05 * rng.standard_normal((n, d_obs))
    return x


def _clip_update(u, max_norm=2.0):
    """梯度裁剪（纯数值保护）：先抹掉非有限值，再按整体范数缩放。"""
    u = np.where(np.isfinite(u), u, 0.0)
    nrm = float(np.linalg.norm(u))
    if nrm == 0.0 or nrm <= max_norm:
        return u
    return u * (max_norm / nrm)


def _train_vae(x, d_lat, beta, steps=8000, lr=0.005, sigma_x=0.5, seed=0):
    """线性高斯 VAE 的全批量梯度下降。

    数值保护三件套（都不改变优化目标，只保证不失稳）：
      logvar 裁到 [-10, 1]、梯度范数裁剪、学习率衰减。
    """
    rng = np.random.default_rng(seed)
    n, d_obs = x.shape
    we_mu = rng.normal(0.0, 0.01, size=(d_obs, d_lat))
    we_lv = np.zeros((d_obs, d_lat))                 # 初始 sigma = 1
    wd = rng.normal(0.0, 1.0 / np.sqrt(d_lat), size=(d_lat, d_obs))

    for t in range(steps):
        lr_t = lr / (1.0 + t / 2000.0)
        eps = rng.standard_normal((n, d_lat))
        mu = x @ we_mu
        logvar = np.clip(x @ we_lv, -10.0, 1.0)
        sig = np.exp(0.5 * logvar)
        z = mu + sig * eps
        rec = z @ wd

        # 重建项：逐元素求和后对 batch 取平均，量纲与 KL 项一致
        g_rec = (rec - x) / (sigma_x ** 2 * n)
        g_wd = z.T @ g_rec
        g_z = g_rec @ wd.T

        g_mu = g_z + beta * mu / n
        g_lv = g_z * eps * 0.5 * sig + beta * 0.5 * (np.exp(logvar) - 1.0) / n

        wd -= lr_t * _clip_update(g_wd)
        we_mu -= lr_t * _clip_update(x.T @ g_mu)
        we_lv -= lr_t * _clip_update(x.T @ g_lv)

    mu = x @ we_mu
    logvar = np.clip(x @ we_lv, -10.0, 1.0)
    kl_per_dim = np.mean(0.5 * (mu ** 2 + np.exp(logvar) - 1.0 - logvar), axis=0)
    z = mu + np.exp(0.5 * logvar) * rng.standard_normal((n, d_lat))
    rmse = float(np.sqrt(np.mean((z @ wd - x) ** 2)))
    return rmse, kl_per_dim


def lab_5():
    section("5. beta 扫描：KL 权重过大如何压死潜维度")
    n, d_obs, d_true, d_lat = 1024, 24, 4, 8
    x = _make_data(n, d_obs, d_true, rng)
    print(f"合成数据：N = {n}, 观测维 D = {d_obs}, 真实隐因子 K = {d_true}, "
          f"模型潜维 d = {d_lat}")
    print(f"线性高斯 VAE，全批量梯度下降 8000 步，lr = 0.005（带衰减），sigma_x = 0.5")
    print()
    print(f"{'beta':>6} | {'重建 RMSE':>12} | {'KL 总和':>12} | "
          f"{'活跃维度数':>12} | {'最大单维 KL':>14}")
    print("-" * 68)
    rows = []
    for beta in (0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        rmse, kld = _train_vae(x, d_lat, beta)
        active = int(np.sum(kld > 0.01))
        rows.append((beta, rmse, float(kld.sum()), active, float(kld.max())))
        print(f"{beta:>6.2f} | {rmse:>12.5f} | {float(kld.sum()):>12.5f} | "
              f"{active:>12d} | {float(kld.max()):>14.5f}")

    # 自校验：beta 单调上升时，KL 总和单调下降、活跃维度数不上升
    kls = [r[2] for r in rows]
    acts = [r[3] for r in rows]
    assert all(kls[i] >= kls[i + 1] - 1e-6 for i in range(len(kls) - 1)), kls
    assert all(acts[i] >= acts[i + 1] for i in range(len(acts) - 1)), acts
    assert acts[0] > acts[-1], acts
    print()
    print(f"自校验通过：beta 从 {rows[0][0]} 升到 {rows[-1][0]}，"
          f"KL 总和 {kls[0]:.4f} → {kls[-1]:.4f}（单调下降），"
          f"活跃维度 {acts[0]} → {acts[-1]}（单调不升）")
    print("结论：KL 项是一把剪刀——权重越大剪掉越多潜维度；后验坍塌就是剪到 0 的极限情形。")
    return {"rows": rows}


def selftest():
    """只跑断言，静默模式。"""
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lab_1()
        lab_2()
        lab_3()
        lab_4()
        lab_5()
    print("selftest: 全部断言通过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    print(f"VAE 核心机制数值验证 | numpy {np.__version__} | seed = {SEED}")
    lab_1()
    lab_2()
    lab_3()
    lab_4()
    lab_5()
    print()
    print(LINE)
    print("全部检查完成。")
    print(LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
