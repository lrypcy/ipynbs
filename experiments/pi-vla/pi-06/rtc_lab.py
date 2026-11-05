# -*- coding: utf-8 -*-
"""06 篇（RTC）的合成探针：把「延迟下动作块怎么接」拆成可解析的高斯世界。

论文 A：*Real-Time Execution of Action Chunking Flow Policies*，arXiv:2506.07339v2。

不是复现真机 / Kinetix 成绩（那些数字只在曲线图里，无数值表）。这里把
论文 §2–§3 与附录 A.2 / A.4 的四个机制逐个钉死：

  A 组  Eq.(5) 软掩码形状、ΠGDM guidance 权重裁剪阈值 4.25、闭式后验自校验
  B 组  inference delay d 下 naive / hard / soft / TE / sync 的切换点加速度
  C 组  β 裁剪与去噪步数 n：第一步超调增益 |1−(k+β)/n| 何时 >1
  D 组  execution horizon s、延迟鲁棒性、sync 的线性退化、guidance 抑制合法换 mode

世界是**线性高斯流**：真实 flow policy 的 inpainting / ΠGDM guidance 在这个
世界里退化成「带权重观测的高斯后验」，有闭式解，因此可扫很多 seed 而不必
真的跑 5 步去噪。C 组另用离散迭代复现「有限步 + 大权重 → 单步超调」。

时间轴按 Alg.1 的索引对齐：新 chunk 索引 i 对应绝对时刻 s+i，
冻结目标 Y[i]=A_cur[s+i]；新 chunk 基于 d 步前的观测生成，故先验相位滞后 d。
延迟越大，陈旧预测与当前状态的偏差越大、策略模式分叉概率越高。

纯 numpy / CPU；运行：~/Software/miniconda3/bin/python rtc_lab.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from series_common import (  # noqa: E402
    Check, soft_mask, ar_kernel, cond_sample, jerk, mode_profile, mean_std,
)

CK = Check()

# ---- 与论文 §4.2 / Table 4 对齐的常量 --------------------------------------
H = 50                 # 真机动作块长度
S = 25                 # 真机 execution horizon
SIGMA = 0.15           # 世界的过程噪声
TAU2 = 0.01            # 观测噪声方差（软约束强度的分母）
N_SEEDS = 80
BASE_SEED = 20261106
DELAYS = (0, 6, 11, 16)   # d≈0 / 6 / 11 / 16 对应无注入 / 基线 / +100 ms / +200 ms
THR_J = 0.42           # 切换点最大加速度阈值（OOD 代理）
THR_A = 0.30           # 新段平均动作幅值阈值（动作是否有效）

METHODS = ("naive", "hard", "exp", "linear", "te", "sync")
KIND = dict(naive="exp", hard="hard", exp="exp", linear="linear", te="exp", sync="exp")
OFF = dict(hard=101, exp=202, linear=303)


def profile(mode: float) -> np.ndarray:
    """周期平滑轨迹：两个相反 mode 的预测逐点平均会互相抵消（TE 的失效）。"""
    i = np.arange(H)
    return mode * (np.sin(2.0 * np.pi * i / H + 0.6)
                   + 0.35 * np.sin(6.0 * np.pi * i / H + 0.3))


def aligned(mode: float, d: int, s: int = S) -> np.ndarray:
    """新 chunk 索引 i 对应时刻 s+i；d 步前的观测给出的是时刻 s+i−d 的预测。

    d=0 时与 Y 完全同相；d>0 时相位滞后 d，滞后越大与当前状态的偏差越大。
    """
    idx = (s + np.arange(H) - d) % H
    return profile(mode)[idx]


def guidance_weight(tau: float, beta: float) -> float:
    """Eq.(2) 的 min(β, (1−τ)/(τ r_τ²))，r_τ²=(1−τ)²/(τ²+(1−τ)²)。

    τ=0 处解析式发散，论文的 β 裁剪把它截成有限值。
    """
    if tau == 0.0:
        return beta
    return min(beta, (tau * tau + (1.0 - tau) ** 2) / (tau * (1.0 - tau)))


def trial(seed: int, d: int, s: int = S, *, forced_mode: float | None = None) -> dict:
    """一次「旧 chunk 已执行 s 步，新 chunk 待生成」的世界快照。

    旧 chunk 走 mode +1；新 chunk 的先验 mode 以 p_flip=min(0.5, 0.03d) 翻到 −1
    （观测越陈旧，策略分叉越可能）。所有方法共享同一组随机数。
    """
    rng = np.random.default_rng(seed)
    Sig = ar_kernel(H, rho=0.85, sigma=SIGMA)
    prev = profile(1.0) + rng.multivariate_normal(
        np.zeros(H), Sig, method="cholesky"
    )
    Y = np.zeros(H)
    Y[: H - s] = prev[s:]            # Alg.1：A_prev = A_cur[s..H-1] 右填充到 H
    p_flip = min(0.5, 0.03 * d)
    if forced_mode is None:
        m = -1.0 if rng.random() < p_flip else 1.0
    else:
        m = forced_mode
    eps = rng.multivariate_normal(np.zeros(H), Sig, method="cholesky")
    eps2 = rng.multivariate_normal(np.zeros(H), Sig, method="cholesky")
    return dict(
        seed=seed, s=s, d=d, Sig=Sig, prev=prev, Y=Y, m=m, p_flip=p_flip,
        mu=aligned(m, d, s),
        free=aligned(m, d, s) + eps,
        other=aligned(-m, d, s) + eps2,
        sync=aligned(1.0, 0, s) + eps,       # 同步推理：每次都拿到当前观测
        # 反应性实验固定相位（与 mu 同为 d 步前的对齐），只隔离「换 mode」这件事
        target_react=aligned(-1.0, d, s),
    )


def chunk_for(method: str, t: dict, d: int) -> tuple[np.ndarray, np.ndarray]:
    """按方法生成新 chunk，返回 (A_new, W)。"""
    W = soft_mask(H, t["s"], d, KIND[method])
    if method == "sync":
        return t["sync"], W
    if method == "naive":
        return t["free"], W
    if method == "te":
        # temporal ensembling：把两个相反 mode 的预测逐点平均 → 互相抵消
        return 0.5 * (t["free"] + t["other"]), W
    prng = np.random.default_rng(t["seed"] + OFF[method])
    a, _, _ = cond_sample(
        t["mu"], t["Sig"], np.arange(H), t["Y"], W, TAU2, prng
    )
    return a, W


def measure(t: dict, a: np.ndarray, d: int) -> dict:
    """切换点的最大加速度、新段动作幅值、重叠区一致性、是否算成功。

    joint = 已经按旧 chunk 执行的 s+d 步 + 新 chunk 从索引 d 起的动作。
    取切换点附近 7 个点算二阶差分，避开旧 chunk 自身的背景噪声。
    """
    s = t["s"]
    joint = np.concatenate([t["prev"][: s + d], a[d:]])
    k = s + d
    j = jerk(joint[k - 3: k + 4])
    amp = float(np.mean(np.abs(a[d:])))
    if d < H - s:
        overlap = float(np.sqrt(np.mean((a[d: H - s] - t["Y"][d: H - s]) ** 2)))
    else:
        overlap = 0.0
    return dict(jerk=j, chunk_jerk=jerk(a), amp=amp, overlap=overlap,
                ok=bool(j < THR_J and amp > THR_A))


def sweep(method: str, d: int, s: int = S, n: int = N_SEEDS,
          *, forced_mode: float | None = None) -> dict:
    """固定 seed 池；同 d 下不同 s 复用同一批随机数，只改变世界的对齐。"""
    vals = []
    for i in range(n):
        t = trial(BASE_SEED + 1000 * d + i, d, s=s, forced_mode=forced_mode)
        a, _ = chunk_for(method, t, d)
        vals.append(measure(t, a, d))
    j_mean, _ = mean_std([v["jerk"] for v in vals])
    a_mean, _ = mean_std([v["amp"] for v in vals])
    o_mean, _ = mean_std([v["overlap"] for v in vals])
    cj_mean, _ = mean_std([v["chunk_jerk"] for v in vals])
    return dict(
        jerk=j_mean, chunk_jerk=cj_mean, amp=a_mean, overlap=o_mean,
        success=float(np.mean([v["ok"] for v in vals])),
    )


def guided_denoise(beta: float, t: dict, *, d: int = 15, s: int = S,
                   n: int = 5, seed: int = 4242) -> dict:
    """GuidedInference 的离散版：把 ΠGDM 的 VJP 近似成单位算子。

    A^{j+1} = A^j + (k(μ−A^j) + w_j (Y−A^j)⊙W) / n，
    第一步 τ=0 的收缩系数是 gain = 1 − (k + β)/n（W 冻结区 = 1）。
    |gain|>1 即单步超调发散，这正是附录 A.2 / Fig.7 的 chunk 发散条件。
    """
    rng = np.random.default_rng(seed)
    W = soft_mask(H, s, d, "exp")
    Y = t["Y"]
    k = 0.5
    A = rng.multivariate_normal(np.zeros(H), t["Sig"], method="cholesky")
    first = None
    weights = []
    for step in range(n):
        tau = step / n
        wgt = guidance_weight(tau, beta)
        weights.append(wgt)
        v = k * (0.0 - A)
        A = A + (v + wgt * (Y - A) * W) / n
        if step == 0:
            first = A.copy()
    return dict(
        A=A, first=first, weights=weights,
        gain=1.0 - (k + beta) / n,
        norm=float(np.linalg.norm(A) / np.sqrt(H)),
        first_norm=float(np.linalg.norm(first) / np.sqrt(H)),
        first_jerk=jerk(first),
        final_jerk=jerk(A),
    )


# ===================================================================== A 组
CK.banner("A 组：Eq.(5) 软掩码、guidance 权重裁剪、闭式后验自校验")

d0, s0 = 11, S
W_exp = soft_mask(H, s0, d0, "exp")
W_lin = soft_mask(H, s0, d0, "linear")
W_hard = soft_mask(H, s0, d0, "hard")

CK.ok("前 d 个权重全为 1（冻结到已知会执行的值）",
      np.allclose(W_exp[:d0], 1.0), f"d={d0}，sum={W_exp[:d0].sum():.1f}")
CK.ok("末 s 个权重全为 0（超出前一 chunk，须全新生成）",
      np.allclose(W_exp[H - s0:], 0.0), f"s={s0}，sum={W_exp[H - s0:].sum():.1f}")
mid = W_exp[d0: H - s0]
CK.ok("中间 H−s−d 个权重严格单调递减",
      bool(np.all(np.diff(mid) < 0)), f"长度 {len(mid)}，首 {mid[0]:.4f} 末 {mid[-1]:.4f}")
CK.ok("中间段首尾都在 (0,1) 内",
      0.0 < mid[-1] < mid[0] < 1.0, f"{mid[0]:.4f} → {mid[-1]:.4f}")
CK.ok("指数衰减全程不高于线性衰减（越远的动作越不确定）",
      bool(np.all(W_exp[d0: H - s0] <= W_lin[d0: H - s0] + 1e-12)),
      f"指数和 {mid.sum():.4f} 线性 {W_lin[d0:H-s0].sum():.4f}")
CK.ok("hard masking 的中间段权重为 0（只冻结、不引导）",
      np.allclose(W_hard[d0:], 0.0), f"hard 总和 {W_hard.sum():.1f}")
CK.ok("d > H−s 时中间段消失，软掩码退化成 hard",
      np.allclose(soft_mask(H, s0, H - s0 + 3, "exp"),
                  soft_mask(H, s0, H - s0 + 3, "hard")), "约束 d ≤ H−s 的来源")

w5 = [guidance_weight(j / 5, 5.0) for j in range(5)]
CK.ok("n=5 时 τ=0.2 的未裁剪 guidance 权重是 4.25",
      abs(w5[1] - 4.25) < 1e-9, f"序列 {[round(x, 4) for x in w5]}")
CK.ok("β=5 时只有 τ=0 那一步被裁剪",
      all(guidance_weight(j / 5, 5.0) == min(5.0, w5[j]) for j in range(1, 5))
      and w5[0] == 5.0, f"β≥4.25 的出处（附录 A.2）")
CK.ok("τ=0 处解析式发散，β 是唯一的上界",
      guidance_weight(0.0, 150.0) == 150.0, "未裁剪为 ∞")

Sig0 = ar_kernel(H, rho=0.85, sigma=SIGMA)
r0 = np.random.default_rng(7)
_, m0, S0 = cond_sample(profile(1.0), Sig0, np.arange(H),
                        np.zeros(H), np.zeros(H), TAU2, r0)
CK.ok("W=0 时后验等于先验（掩码不引入任何偏置）",
      np.max(np.abs(m0 - profile(1.0))) < 1e-8
      and np.max(np.abs(S0 - Sig0)) < 1e-8, "闭式解退化检查")
y_hard = profile(1.0)
a_h, m_h, _ = cond_sample(np.zeros(H), Sig0, np.arange(H), y_hard,
                          np.ones(H), TAU2, np.random.default_rng(8))
rmse_post = float(np.sqrt(np.mean((m_h - y_hard) ** 2)))
rmse_prior = float(np.sqrt(np.mean(y_hard ** 2)))
CK.ok("W=1 且 τ²=0.01 时后验均值被拉到观测值附近",
      rmse_post < 0.1 * rmse_prior,
      f"后验 RMSE {rmse_post:.4f}，先验 RMSE {rmse_prior:.4f}")


# ===================================================================== B 组
CK.banner("B 组：inference delay 下各方法的切换点加速度与动作有效性")

B = {(m, d): sweep(m, d) for m in METHODS for d in DELAYS}
print("\n  方法      d   切换点最大加速度   新段幅值   重叠区误差   成功率")
for d in DELAYS:
    for m in METHODS:
        r = B[(m, d)]
        print(f"  {m:<8} {d:>3}   {r['jerk']:>10.4f}   {r['amp']:>9.4f}"
              f"   {r['overlap']:>9.4f}   {r['success']:>6.3f}")

CK.ok("d=0 时无延迟，naive 的切换点加速度也低",
      B[("naive", 0)]["jerk"] < THR_J, f"naive d=0：{B[('naive',0)]['jerk']:.4f}")
CK.ok("延迟从 0 涨到 16，naive async 的切换点加速度显著上升",
      B[("naive", 16)]["jerk"] > B[("naive", 0)]["jerk"] + 0.30,
      f"{B[('naive',0)]['jerk']:.4f} → {B[('naive',16)]['jerk']:.4f}")
for d in (6, 11, 16):
    CK.ok(f"d={d}：soft(exp) 的加速度低于 hard masking",
          B[("exp", d)]["jerk"] < B[("hard", d)]["jerk"],
          f"{B[('exp',d)]['jerk']:.4f} vs {B[('hard',d)]['jerk']:.4f}")
    CK.ok(f"d={d}：soft(exp) 的加速度低于 naive async",
          B[("exp", d)]["jerk"] < B[("naive", d)]["jerk"],
          f"{B[('exp',d)]['jerk']:.4f} vs {B[('naive',d)]['jerk']:.4f}")
    CK.ok(f"d={d}：soft(exp) 的成功率高于 naive async",
          B[("exp", d)]["success"] > B[("naive", d)]["success"],
          f"{B[('exp',d)]['success']:.3f} vs {B[('naive',d)]['success']:.3f}")

CK.ok("指数与线性衰减非常接近（附录 A.4 的软掩码调度消融）",
      abs(B[("exp", 11)]["jerk"] - B[("linear", 11)]["jerk"]) < 0.08,
      f"exp {B[('exp',11)]['jerk']:.4f} vs linear {B[('linear',11)]['jerk']:.4f}")
CK.ok("TE 把 chunk 内的加速度压下来了（Fig.2：TE 降低加速度）",
      B[("te", 11)]["chunk_jerk"] < B[("naive", 11)]["chunk_jerk"],
      f"TE {B[('te',11)]['chunk_jerk']:.4f} vs naive {B[('naive',11)]['chunk_jerk']:.4f}")
CK.ok("但 TE 的动作幅值塌掉（平均掉成噪声，动作无效）",
      B[("te", 11)]["amp"] < 0.6 * B[("exp", 11)]["amp"],
      f"TE {B[('te',11)]['amp']:.4f} vs soft {B[('exp',11)]['amp']:.4f}")
CK.ok("TE 即使 d=0 也差（Fig.5 的定性结论）",
      B[("te", 0)]["success"] < B[("exp", 0)]["success"],
      f"TE {B[('te',0)]['success']:.3f} vs soft {B[('exp',0)]['success']:.3f}")
CK.ok("soft 的成功率对注入延迟几乎不退化（Fig.6 的完全鲁棒）",
      B[("exp", 16)]["success"] >= B[("exp", 0)]["success"] - 0.10,
      f"d=0 {B[('exp',0)]['success']:.3f} → d=16 {B[('exp',16)]['success']:.3f}")
CK.ok("naive 的成功率随延迟明显退化",
      B[("naive", 0)]["success"] - B[("naive", 16)]["success"] > 0.15,
      f"d=0 {B[('naive',0)]['success']:.3f} → d=16 {B[('naive',16)]['success']:.3f}")
CK.ok("RTC 的切换点加速度对延迟不敏感（不随 d 单调恶化）",
      max(B[("exp", d)]["jerk"] for d in DELAYS)
      - min(B[("exp", d)]["jerk"] for d in DELAYS) < 0.25,
      " → ".join(f"{B[('exp',d)]['jerk']:.4f}" for d in DELAYS))


# ===================================================================== C 组
CK.banner("C 组：β 裁剪与去噪步数（附录 A.2 / Fig.7）")

t_beta = trial(777, 15)
BETAS = (5, 20, 50, 150)
C5, C100, CGAIN = {}, {}, {}
print("\n    β     n=5 首步范数   n=5 首步加速度   n=5 |gain|   n=100 首步范数   n=100 |gain|")
for beta in BETAS:
    r5 = guided_denoise(beta, t_beta, n=5)
    r100 = guided_denoise(beta, t_beta, n=100)
    C5[beta] = r5
    C100[beta] = r100
    CGAIN[beta] = abs(r5["gain"])
    print(f"  {beta:>4}   {r5['first_norm']:>12.4f}   {r5['first_jerk']:>14.4f}"
          f"   {abs(r5['gain']):>10.4f}   {r100['first_norm']:>16.4f}"
          f"   {abs(r100['gain']):>13.4f}")
print(f"  β=5 的 guidance 权重序列：{[round(x, 4) for x in C5[5]['weights']]}")

CK.ok("β=5（论文设定）在 n=5 步下第一步不超调（|gain|<1）",
      CGAIN[5] < 1.0, f"|gain|={CGAIN[5]:.4f}")
CK.ok("β=20 / 50 / 150 在 n=5 步下第一步超调发散",
      all(CGAIN[b] > 1.0 for b in (20, 50, 150)),
      " → ".join(f"β={b}: {CGAIN[b]:.2f}" for b in (20, 50, 150)))
CK.ok("n=5 下 β 越大首步最大加速度（jerkiness / OOD 代理）越大",
      all(C5[BETAS[i + 1]]["first_jerk"] > C5[BETAS[i]]["first_jerk"]
          for i in range(3)),
      " → ".join(f"{C5[b]['first_jerk']:.3f}" for b in BETAS))
CK.ok("n=100 时即使 β=150 也不超调（超调来自大步长，不是 guidance 本身）",
      abs(C100[150]["gain"]) < 1.0, f"|gain|={abs(C100[150]['gain']):.4f}")
CK.ok("裁剪阈值：β=5 时 τ=0 的权重恰为 5，其余步都小于 5",
      C5[5]["weights"][0] == 5.0 and all(w < 5.0 for w in C5[5]["weights"][1:]),
      f"权重序列 {[round(x, 4) for x in C5[5]['weights']]}")


# ===================================================================== D 组
CK.banner("D 组：execution horizon、延迟鲁棒性与 guidance 的代价")

S_GRID = (10, 15, 20, 25)
D_s = {s: sweep("exp", 6, s=s) for s in S_GRID}
D_s_naive = {s: sweep("naive", 6, s=s) for s in S_GRID}
print("\n   s    soft 加速度   soft 成功率   naive 加速度   naive 成功率")
for s in S_GRID:
    print(f"  {s:>3}   {D_s[s]['jerk']:>10.4f}   {D_s[s]['success']:>11.3f}"
          f"   {D_s_naive[s]['jerk']:>12.4f}   {D_s_naive[s]['success']:>12.3f}")

CK.ok("execution horizon 越小，重叠引导越多，soft 的切换点加速度不增",
      all(D_s[S_GRID[i + 1]]["jerk"] <= D_s[S_GRID[i]]["jerk"] + 0.04
          for i in range(len(S_GRID) - 1)),
      " → ".join(f"{D_s[s]['jerk']:.4f}" for s in S_GRID))
CK.ok("只有 RTC 随 s 减小成功率不降（Fig.5：仅 RTC 与 BID 有此性质）",
      all(D_s[S_GRID[i + 1]]["success"] <= D_s[S_GRID[i]]["success"] + 0.08
          for i in range(len(S_GRID) - 1)),
      " → ".join(f"{D_s[s]['success']:.3f}" for s in S_GRID))
CK.ok("把 execution horizon 压到最小也救不了 naive（没有重叠区引导可用）",
      all(D_s_naive[s]["jerk"] > D_s[s]["jerk"] + 0.15 for s in S_GRID),
      " → ".join(f"s={s}: {D_s_naive[s]['jerk']:.4f} vs {D_s[s]['jerk']:.4f}"
                 for s in S_GRID))

# 同步推理的动作质量与延迟无关，代价只有「执行 s 步后停下等 d 步」
q0 = B[("sync", 0)]["success"]
print("\n   d    sync throughput   RTC throughput   sync 停顿步数")
D_tp = {}
for d in (0, 6, 11, 16):
    tp_sync = q0 * S / (S + d)
    tp_rtc = B[("exp", d)]["success"]
    D_tp[d] = (tp_sync, tp_rtc)
    print(f"  {d:>3}   {tp_sync:>15.4f}   {tp_rtc:>14.4f}   {d:>12d}")

CK.ok("synchronous 的 throughput 随延迟下降，RTC 不随延迟下降",
      D_tp[16][0] < D_tp[0][0] - 0.10
      and D_tp[16][1] >= D_tp[0][1] - 0.10,
      f"sync {D_tp[0][0]:.4f}→{D_tp[16][0]:.4f}；RTC {D_tp[0][1]:.4f}→{D_tp[16][1]:.4f}")
CK.ok("同步退化完全来自停顿：tp(d)/tp(0) 就是 s/(s+d)",
      abs(D_tp[16][0] / D_tp[0][0] - S / (S + 16)) < 1e-9,
      f"d=16 比值 {D_tp[16][0]/D_tp[0][0]:.4f}，理论 {S/(S+16):.4f}")

# 反应性边界：新观测明确要求换到 mode −1，看 guidance 会不会把合法切换压掉
react_err = {}
for m in ("naive", "hard", "exp"):
    errs = []
    for i in range(N_SEEDS):
        t = trial(BASE_SEED + 11000 + i, 11, forced_mode=-1.0)
        a, _ = chunk_for(m, t, 11)
        errs.append(np.sqrt(np.mean(
            (a[11: H - S] - t["target_react"][11: H - S]) ** 2)))
    react_err[m] = float(np.mean(errs))
print("  换 mode 时与理想块的 RMSE：" +
      "  ".join(f"{m}={react_err[m]:.4f}" for m in ("naive", "hard", "exp")))
CK.ok("guidance 会抑制合法的策略切换：soft 与理想块的偏差大于 naive",
      react_err["exp"] > react_err["naive"] + 0.10,
      f"soft {react_err['exp']:.4f} vs naive {react_err['naive']:.4f}")
CK.ok("软掩码的中间段正是这个代价的来源（hard 更接近 naive）",
      react_err["hard"] < react_err["exp"],
      f"hard {react_err['hard']:.4f} vs soft {react_err['exp']:.4f}")


# ===================================================================== 回填表
CK.banner("正文回填表（以下数字与上面逐条对齐，可直接写进文章）")
print(f"  掩码：d=11 / s=25 / H=50，中间段 {mid[0]:.4f} → {mid[-1]:.4f}，"
      f"指数和 {mid.sum():.4f}，线性和 {W_lin[d0:H-s0].sum():.4f}")
print(f"  guidance 权重（n=5, β=5）：{[round(x, 4) for x in w5]}")
for d in DELAYS:
    print(f"  d={d:>2}  " + "  ".join(
        f"{m}: jerk={B[(m,d)]['jerk']:.4f} amp={B[(m,d)]['amp']:.4f} "
        f"succ={B[(m,d)]['success']:.3f}" for m in ("naive", "hard", "exp", "linear", "te")))
print("  β 扫描 n=5：" + "  ".join(
    f"β={b}: gain={abs(C5[b]['gain']):.2f} first_jerk={C5[b]['first_jerk']:.4f} "
    f"first_norm={C5[b]['first_norm']:.4f}" for b in BETAS))
print("  β 扫描 n=100：" + "  ".join(
    f"β={b}: gain={abs(C100[b]['gain']):.4f} first_norm={C100[b]['first_norm']:.4f}"
    for b in BETAS))
print("  s 扫描 soft jerk：" + "  ".join(
    f"s={s}: {D_s[s]['jerk']:.4f}" for s in S_GRID))
print("  s 扫描 soft succ：" + "  ".join(
    f"s={s}: {D_s[s]['success']:.3f}" for s in S_GRID))
print("  s 扫描 naive jerk：" + "  ".join(
    f"s={s}: {D_s_naive[s]['jerk']:.4f}" for s in S_GRID))
print("  TE chunk 内加速度：" + "  ".join(
    f"d={d}: te={B[('te',d)]['chunk_jerk']:.4f} "
    f"naive={B[('naive',d)]['chunk_jerk']:.4f}" for d in DELAYS))
print("  throughput：" + "  ".join(
    f"d={d}: sync={D_tp[d][0]:.4f} rtc={D_tp[d][1]:.4f}" for d in (0, 6, 11, 16)))
print("  换 mode RMSE：" + "  ".join(
    f"{m}={react_err[m]:.4f}" for m in ("naive", "hard", "exp")))

CK.summary()
