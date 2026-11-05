# -*- coding: utf-8 -*-
"""07 篇（RECAP / π*0.6）合成探针：把「advantage 条件化」拆成可解析的高斯世界。

论文 B：*π*0.6: a VLA That Learns From Experience*，arXiv:2511.14759v2。
RECAP = RL with Experience and Corrections via Advantage-conditioned Policies。

**只验机制，不重算论文成绩**（真机 throughput >2×、失败率减半、13 小时 espresso
等都是引用，见 README 的边界声明）。本探针把论文里最容易含混过去的四件事钉死：

  自校验  201-bin distributional value 的梯度（手写 vs 有限差分）；构造真值 ↔ 反算值
  A 组    advantage 二值化与条件化：连续 advantage 加权 vs 二值 I；β=1 的 Bayes 等价
          （论文 §III、§IV-B Eq.(2)/Eq.(3)、§V-B/§V-D、A-F）
  B 组    expert interventions / on-policy 数据：离线数据分布 vs 部署分布偏移，
          只加离线 vs 加 on-policy 纠正，迭代轮数是否收敛（Algorithm 1、§V-D、§VI-C2）
  C 组    条件化 positive 与拒绝采样 / best-of-N 的等价性边界（§III、§IV-B、A-E）
  D 组    代价与边界：二值化丢的信息、value 标定误差如何传导、何时迭代反而变差
          （§IV-A、§V-D、§VI-C3）

世界是**标量动作的单步高斯 bandit**：行为策略 π_ref 是「靠近成功的 demo 模式」与
「偏离成功的次优模式」的混合；回报 V(a) = −(a−θ)²/A_MAX² ∈ (−1,0]（θ=成功点，
V=0 即成功；对应论文「距成功剩余步数的负值，归一化到 (−1,0)」）。advantage
A(a)=V(a)−V_ref（V_ref 是 behavior policy 的 MC 价值估计；单步世界是 N=50 lookahead
的退化）。I=𝟙[A>ε_l]，ε_l 取使约 30% 数据为正的阈值（论文 §V-D / A-F）。因为
V 是 a² 的单调仿射，I=True 等价于 |a−θ|<τ：正 advantage 区间是成功点邻域，
π_ref(a|I) 就是把行为策略截断到该邻域——全部量在细网格上**确定性可算**。

纯 numpy / CPU / 固定 seed / 多 seed 平均 / 每组末尾带断言 / 最后打印断言总数。
运行：~/Software/miniconda3/bin/python recap_lab.py      约 4 秒
"""
from __future__ import annotations

import os
import sys

import numpy as np

# series_common.py 在 pi-vla 根目录；lab 在 pi-07/ 下，故取上两级目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from series_common import Check, mean_std  # noqa: E402

CK = Check()

# ---- 论文常量（引用，不参与合成世界的自由度） ----------------------------------
B_BINS = 201          # distributional value 的离散 bins（§IV-A Eq.1）
Q_POS = 0.30          # 正 advantage 比例：§V-D 的 30% percentile / A-F 的 30% 预训练
N_LOOKAHEAD = 50      # post-training advantage 的 N=50 lookahead（A-F）
BETA_DEPLOY = 1       # 部署默认 β=1（§VI-B）
CFG_RANGE = (1.5, 2.5)  # 实用 CFG β 区间（A-E）
ADV_DROPOUT = 0.30    # advantage conditioning dropout（A-F）

# ---- 合成世界的自由度（自定，与论文真机数字无关） ------------------------------
A_MAX = 10.0                       # 回报归一化分母，使 V∈(−1,0]
GRID_LO, GRID_HI, GRID_STEP = -8.0, 10.0, 0.002
A_GRID = np.arange(GRID_LO, GRID_HI + 1e-9, GRID_STEP)   # 动作网格
THETA = 0.0                        # 成功点 / 最优动作
MU_GOOD, MU_BAD, SIG_REF, W_BAD = 0.6, 2.4, 1.0, 0.55   # π_ref 双模态

# B 组：两状态 contextual bandit（部署偏移到 demo 没覆盖的状态）
THETA0, THETA1 = 0.0, 2.0         # 状态 0（ID/demo）/ 状态 1（部署/OOD）的最优动作
SIG_EXEC = 0.5                     # 部署执行噪声，给出 J1 的噪声下界
SIG_ROLL, N_ROLL = 1.0, 80         # on-policy autonomous rollout
SIG_CORR, N_CORR = 0.5, 40         # expert intervention（强制 I=True）
K_ITERS, N_SEEDS = 6, 40           # 迭代轮数 / seed 数

# D 组：标定噪声多 seed 平均
N_SEEDS_CAL = 24

_rng_world = np.random.default_rng(20261107)


def _normal_pdf(x, mu, sig):
    return np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))


# π_ref 在网格上的概率质量（含次优数据的混合行为策略）
F_REF = ((1 - W_BAD) * _normal_pdf(A_GRID, MU_GOOD, SIG_REF)
         + W_BAD * _normal_pdf(A_GRID, MU_BAD, SIG_REF))
P_REF = F_REF / F_REF.sum()
V_GRID = -(A_GRID - THETA) ** 2 / A_MAX ** 2          # V(a) ∈ (−1,0]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def advantage_mask(p, adv, q=Q_POS):
    """I = 𝟙[A>ε_l]，ε_l 取使质量占比 q 的 A 为 True 的分位阈值。

    返回 (mask, ε_l)。按 p 加权取 adv 最大的 q 比例，故 Pr(I)=q（含边界点误差）。
    """
    order = np.argsort(adv)[::-1]
    cs = np.cumsum(p[order])
    j = int(np.searchsorted(cs, q))
    mask = np.zeros_like(adv, dtype=bool)
    mask[order[: j + 1]] = True
    return mask, float(adv[order[j]])


def discrete_entropy(q):
    """网格上的离散熵（只做相对比较；未除 Δa，注明是相对读数）。"""
    nz = q[q > 0]
    return float(-np.sum(nz * np.log(nz)))


def kl(q, p):
    nz = q > 0
    return float(np.sum(q[nz] * np.log(q[nz] / p[nz])))


def policy_quality(q):
    return float(q @ V_GRID)                # J(π)=E_π[V]，越高越好，上界 0


# ===================================================================== 自校验
CK.banner("自校验：201-bin distributional value 梯度；构造真值 ↔ 反算值；世界不变量")

# ---- 1. 手写梯度 vs 有限差分（论文 §IV-A：p_φ(V|o) ∈ Δ_B，V=Σ_b p_b v(b)）
_edges = np.linspace(-1.0, 0.0, B_BINS + 1)
_vbin = 0.5 * (_edges[:-1] + _edges[1:])
_phi0 = _rng_world.normal(0, 0.5, size=B_BINS)
_Vtrue = -0.0413


def _dist_value_loss_grad(phi, vbin, vtrue):
    p = softmax(phi)
    vf = float(p @ vbin)
    dL = vf - vtrue
    g = dL * p * (vbin - vf)
    return 0.5 * dL ** 2, g


_L0, _g0 = _dist_value_loss_grad(_phi0, _vbin, _Vtrue)
_d = _rng_world.normal(size=B_BINS)
_d /= np.linalg.norm(_d)
_eps = 1e-6
_Lp, _ = _dist_value_loss_grad(_phi0 + _eps * _d, _vbin, _Vtrue)
_Lm, _ = _dist_value_loss_grad(_phi0 - _eps * _d, _vbin, _Vtrue)
_num = (_Lp - _Lm) / (2 * _eps)
_ana = float(_g0 @ _d)
_rel_grad = abs(_ana - _num) / max(abs(_num), 1e-12)
CK.ok("201-bin distributional value 的手写梯度与有限差分一致",
      _rel_grad < 1e-6, f"相对误差 {_rel_grad:.2e}（B={B_BINS}）")

# ---- 2. V_ref 三路对照：解析高斯二阶矩 ↔ 网格积分 ↔ 201-bin 直方图
_Ea2 = (1 - W_BAD) * (MU_GOOD ** 2 + SIG_REF ** 2) \
    + W_BAD * (MU_BAD ** 2 + SIG_REF ** 2)
V_REF_ANA = -_Ea2 / A_MAX ** 2
V_REF_GRID = float(P_REF @ V_GRID)
_hist, _ = np.histogram(V_GRID, bins=_edges, weights=P_REF)
V_REF_HIST = float((_hist * _vbin).sum() / _hist.sum())
CK.ok("V_ref 解析值 ↔ 网格积分一致（构造真值 ↔ 反算值）",
      abs(V_REF_ANA - V_REF_GRID) < 5e-4,
      f"解析 {V_REF_ANA:.4f} 网格 {V_REF_GRID:.4f}")
CK.ok("V_ref 解析值 ↔ 201-bin 直方图反算一致",
      abs(V_REF_ANA - V_REF_HIST) < 5e-4,
      f"解析 {V_REF_ANA:.4f} 直方图 {V_REF_HIST:.4f}（B={B_BINS}）")

# ---- 3. 世界不变量
CK.ok("π_ref 在网格上归一", abs(P_REF.sum() - 1.0) < 1e-12,
      f"Σp={P_REF.sum():.6f}")
CK.ok("θ=0 是 V 的全局最大（成功点 V=0）",
      abs(V_GRID[np.argmin(np.abs(A_GRID - THETA))]) < 1e-12
      and V_GRID.max() <= 1e-12, f"V(θ)={V_GRID[np.argmin(np.abs(A_GRID-THETA))]:.4f}")

# advantage、二值化阈值
ADV = V_GRID - V_REF_GRID
MASK_I, EPS_L = advantage_mask(P_REF, ADV, Q_POS)
PR_I = float(P_REF[MASK_I].sum())
# |a−θ|<τ 的等价刻画
_absa = np.abs(A_GRID - THETA)
_order = np.argsort(_absa)
_cs = np.cumsum(P_REF[_order])
TAU = float(_absa[_order[np.searchsorted(_cs, Q_POS)]])
CK.ok("I 的正比例落在 30% 分位目标附近",
      abs(PR_I - Q_POS) < 0.005, f"Pr(I)={PR_I:.4f}，目标 {Q_POS:.2f}")
CK.ok("二值 I 与「|a−θ|<τ」是同一集合（V 单调仿射）",
      float(np.mean(MASK_I == (_absa < TAU))) > 0.999,
      f"τ={TAU:.4f}，重合度 {np.mean(MASK_I==(_absa<TAU)):.4f}")

# 几个核心策略分布（A/C/D 共用）
Q_COND = P_REF * MASK_I
Q_COND /= Q_COND.sum()               # π_ref(a|I) = β=1 条件化 = 二值接受拒绝采样
J_REF = policy_quality(P_REF)
J_COND = policy_quality(Q_COND)
H_REF, H_COND = discrete_entropy(P_REF), discrete_entropy(Q_COND)
KL_COND = kl(Q_COND, P_REF)


# ===================================================================== A 组
CK.banner("A 组：advantage 二值化与条件化（连续加权 vs 二值 I；β=1 的 Bayes 等价）")

# ---- 连续 advantage 加权：论文 §III KL 结果 π̂ ∝ π_ref·exp(A/β)。
# β 用 A 的标准差归一成无量纲温度 t：β = t·std(A)。
STD_A = float(np.sqrt(P_REF @ (ADV - (P_REF @ ADV)) ** 2))
T_GRID = (0.15, 0.3, 0.6, 1.0, 2.0, 4.0, 10.0)
CONT = {}
for t in T_GRID:
    beta = t * STD_A
    w = np.exp((ADV - ADV.max()) / beta)
    q = P_REF * w
    q /= q.sum()
    CONT[t] = dict(beta=beta, J=policy_quality(q), H=discrete_entropy(q),
                   KL=kl(q, P_REF), q=q)
T_BEST = min(T_GRID, key=lambda t: -CONT[t]["J"])   # J 最大（最接近 0）的温度
J_CONT_BEST = CONT[T_BEST]["J"]

print("\n  A 组：连续 advantage 加权 π̂∝π_ref·exp(A/β)（β=t·std(A)）")
print("     t       β        J(π̂)      熵 H     KL(π̂‖π_ref)")
for t in T_GRID:
    c = CONT[t]
    print(f"  {t:>5.2f}  {c['beta']:>8.5f}  {c['J']:>10.4f}  {c['H']:>8.4f}"
          f"  {c['KL']:>10.4f}")

# ---- β=1 的 Bayes 重写（论文 Eq.2）：π̂(a) ∝ π_ref(a)·p(I|A)^β，
#      p(I|A)=π_ref(a|I)/π_ref(a)。β=1 时 π̂=π_ref(a|I)。
p_I_given_A = Q_COND / P_REF                    # = 𝟙(I)/Pr(I)，区间内为常数
Q_BAYES_B1 = P_REF * (p_I_given_A ** BETA_DEPLOY)
Q_BAYES_B1 /= Q_BAYES_B1.sum()
_bayes_eq = float(np.max(np.abs(Q_BAYES_B1 - Q_COND)))

# 「β=1 条件化 positive 对数据做了什么」：硬筛选到正 advantage 区间
R_KEEP = float((Q_COND @ V_GRID))              # 保留数据平均回报 = J_cond
_drop = P_REF * (~MASK_I)
_drop = _drop / _drop.sum()
R_DROP = float(_drop @ V_GRID)                 # 被丢弃数据平均回报
MEAN_REF = float(P_REF @ A_GRID)
MEAN_COND = float(Q_COND @ A_GRID)

print("\n  A 组：β=1 条件化对数据做了什么（硬筛选 + 重新归一化）")
print(f"    保留比例 q = {PR_I:.4f}；正 advantage 区间 |a−θ| < τ = {TAU:.4f}")
print(f"    ε_l = {EPS_L:.4f}；V_ref = {V_REF_GRID:.4f}")
print(f"    保留样本平均回报 {R_KEEP:.4f}，被丢弃样本平均回报 {R_DROP:.4f}")
print(f"    行为策略均值 {MEAN_REF:.4f} → 条件策略均值 {MEAN_COND:.4f}")
print(f"    J(π_ref) {J_REF:.4f} → J(β=1 条件化) {J_COND:.4f}；"
      f"KL {KL_COND:.4f}；熵 {H_REF:.4f} → {H_COND:.4f}")

CK.ok("二值条件化（β=1）把策略质量从 J_ref 大幅拉向成功点",
      J_COND > J_REF + 0.02,
      f"J_ref {J_REF:.4f} → J_cond {J_COND:.4f}（两者为原始读数）")
CK.ok("β=1 的 Bayes 重写 π̂∝π_ref·p(I|A) 逐点等于 π_ref(a|I)",
      _bayes_eq < 1e-10, f"最大偏差 {_bayes_eq:.2e}（论文 Eq.(2)）")
CK.ok("条件化把动作均值朝成功点 θ 移动",
      abs(MEAN_COND - THETA) < abs(MEAN_REF - THETA),
      f"{MEAN_REF:.4f} → {MEAN_COND:.4f}，θ={THETA:.1f}")
CK.ok("β=1 等价于「按 I 硬筛选数据」：保留回报高于丢弃回报",
      R_KEEP > R_DROP + 0.02,
      f"保留 {R_KEEP:.4f} vs 丢弃 {R_DROP:.4f}（两原始读数）")
CK.ok("保留比例就是 30% 分位阈值设计出来的 q",
      abs(PR_I - Q_POS) < 0.005, f"q={PR_I:.4f}")
CK.ok("条件化收缩了策略支撑（熵下降、KL>0）",
      H_COND < H_REF and KL_COND > 0,
      f"H {H_REF:.4f}→{H_COND:.4f}，KL {KL_COND:.4f}")
CK.ok("连续加权里存在温度使 J 不低于二值条件化（连续保留了 A 的大小信息）",
      J_CONT_BEST >= J_COND - 0.002,
      f"连续最优 t={T_BEST:.2f} J={J_CONT_BEST:.4f} vs 二值 J={J_COND:.4f}")


# ===================================================================== B 组
CK.banner("B 组：on-policy 数据 / expert interventions / 迭代（分布偏移下的 Algorithm 1）")


def j1_of(mu):
    """状态 1 部署回报：J1(μ)=−((μ−θ1)²+σ_exec²)/A_MAX²（越高越好）。"""
    return -((mu - THETA1) ** 2 + SIG_EXEC ** 2) / A_MAX ** 2


def b_run(seed, mode, K=K_ITERS, c=0.0):
    """一轮 RECAP 迭代的有限样本版。mode ∈ {offline, auto, corr}。

    offline：只用离线 demo（状态 0），状态 1 无数据 → 回退到 ID 策略 μ1=0。
    auto   ：状态 1 收集 on-policy autonomous rollouts（覆盖 OOD，无纠正）。
    corr   ：另加 expert interventions（强制 I=True，动作 ≈ θ1+c）。
    每轮用**迄今全部累积的 I=True 样本**重新提策略（论文 §IV-B「用全部数据」）。
    """
    rng = np.random.default_rng(seed)
    mu1 = THETA0                       # 初始：把 ID 策略搬到 OOD 状态
    hist = []
    js = [j1_of(mu1)]
    for _k in range(K):
        if mode == "offline":
            js.append(j1_of(mu1))      # 无新数据，μ1 恒定
            continue
        a_roll = mu1 + SIG_ROLL * rng.normal(size=N_ROLL)
        if mode == "corr":
            a_corr = (THETA1 + c) + SIG_CORR * rng.normal(size=N_CORR)
            new = np.concatenate([a_roll, a_corr])
        else:
            new = a_roll
        # value function 给本轮数据打 advantage：I=True 当 |a−θ1| 落在 top-q
        v1 = -(new - THETA1) ** 2 / A_MAX ** 2
        order = np.argsort(v1)[::-1]
        j = max(1, int(round(Q_POS * len(new))))
        keep = np.zeros(len(new), dtype=bool)
        keep[order[:j]] = True
        if mode == "corr":
            keep[N_ROLL:] = True        # 人工纠正强制 I=True（论文 §IV-B）
        hist.append(new[keep])
        allpos = np.concatenate(hist)
        mu1 = float(allpos.mean())
        js.append(j1_of(mu1))
    return np.array(js)


def b_avg(mode, c=0.0):
    runs = np.array([b_run(9000 + s, mode, c=c) for s in range(N_SEEDS)])
    return runs.mean(axis=0)


J_OFF = b_avg("offline")
J_AUTO = b_avg("auto")
J_CORR = b_avg("corr")
J1_FLOOR = -SIG_EXEC ** 2 / A_MAX ** 2          # 解析最优下界（μ1=θ1）

print("\n  B 组：状态 1 部署回报 J1（K=0..6 轮，40 seed 平均；越高越好）")
print("    轮 k    offline(只离线)   +autonomous   +corrections")
for k in range(K_ITERS + 1):
    print(f"    {k:>4}    {J_OFF[k]:>14.4f}   {J_AUTO[k]:>12.4f}   {J_CORR[k]:>14.4f}")
print(f"    解析最优下界 J1(μ1=θ1) = {J1_FLOOR:.4f}")

_gain_corr = [J_CORR[k + 1] - J_CORR[k] for k in range(K_ITERS)]
_gain_auto = [J_AUTO[k + 1] - J_AUTO[k] for k in range(K_ITERS)]

CK.ok("只加离线数据：部署回报不随迭代提升（OOD 状态从未被覆盖）",
      abs(J_OFF[-1] - J_OFF[0]) < 1e-9,
      f"{J_OFF[0]:.4f} → {J_OFF[-1]:.4f}")
CK.ok("加 on-policy 纠正后，部署回报相对离线基线显著提升",
      J_CORR[-1] - J_OFF[-1] > 0.02,
      f"offline {J_OFF[-1]:.4f} → corr {J_CORR[-1]:.4f}（两原始读数）")
CK.ok("只加 autonomous rollouts 也能改进，但慢于加 corrections",
      J_AUTO[-1] > J_OFF[-1] + 0.005 and J_CORR[-1] > J_AUTO[-1],
      f"auto {J_AUTO[-1]:.4f} vs corr {J_CORR[-1]:.4f}")
CK.ok("corrections 的收益来自强制 I=True 的好动作（覆盖 OOD + 指向 θ1）",
      J_CORR[1] > J_AUTO[1],
      f"第 1 轮 corr {J_CORR[1]:.4f} vs auto {J_AUTO[1]:.4f}")
CK.ok("迭代增益逐轮递减（Algorithm 1 多轮改进趋于收敛）",
      _gain_corr[0] > _gain_corr[-1] and all(
          _gain_corr[i] >= _gain_corr[i + 1] - 1e-6 for i in range(len(_gain_corr) - 1)),
      "增益 " + " → ".join(f"{g:.4f}" for g in _gain_corr))
CK.ok("K 轮后接近解析最优下界（seed 平均反算 ↔ 构造真值）",
      abs(J_CORR[-1] - J1_FLOOR) < 0.012,
      f"末值 {J_CORR[-1]:.4f} vs 解析下界 {J1_FLOOR:.4f}")
CK.ok("一轮就吃到大部分收益（论文 §V-D：实践中一轮常已显著改进）",
      (J_CORR[1] - J_OFF[0]) > 0.6 * (J_CORR[-1] - J_OFF[0]),
      f"首轮 {J_CORR[1]-J_OFF[0]:.4f} / 全轮 {J_CORR[-1]-J_OFF[0]:.4f}")


# ===================================================================== C 组
CK.banner("C 组：条件化 positive 与拒绝采样 / best-of-N 的等价性边界")


def best_of_N(N):
    """从 π_ref 独立采 N 个动作取 R 最大（= |a−θ| 最小）。

    用网格分布的 order statistics 闭式：Pr(min u ∈ (u_{j-1},u_j])
    =(1−F_{j-1})^N −(1−F_j)^N。返回 (J, 选中 u 的分布, 该分布熵)。
    """
    u = _absa
    order = np.argsort(u)
    us = u[order]
    pj = P_REF[order]
    F = np.cumsum(pj)
    Fprev = np.concatenate([[0.0], F[:-1]])
    prob = (1 - Fprev) ** N - (1 - F) ** N
    prob = np.clip(prob, 0.0, None)
    prob /= prob.sum()
    J = float(-np.sum(us ** 2 * prob) / A_MAX ** 2)
    H = discrete_entropy(prob)
    return J, H, prob, us


N_GRID = (1, 2, 3, 4, 5, 8, 16, 64, 256)
BO = {N: best_of_N(N) for N in N_GRID}
N_STAR = min(N_GRID, key=lambda N: abs(BO[N][0] - J_COND))

# 两种机制真正的区别：条件化在整个正 advantage 区间上有支撑（偶尔采到区间边缘），
# best-of-N 是极值顺序统计量，几乎只落在 θ 附近。用「区间边缘尾部质量」量化。
TAIL_THR = 0.7
TAIL_COND = float(Q_COND[_absa > TAIL_THR].sum())
TAIL_BON = float(BO[N_STAR][2][BO[N_STAR][3] > TAIL_THR].sum())

# 拒绝采样（二值接受、保留原密度）的分布 = π_ref(a|I)
Q_REJ = P_REF * MASK_I
Q_REJ /= Q_REJ.sum()
J_REJ = policy_quality(Q_REJ)
# best-of-N 与 conditioning 的分布差异：用 |a−θ| 的熵对照
_absa_sorted_mask = MASK_I
H_U_COND = discrete_entropy((P_REF * _absa_sorted_mask)
                            / (P_REF * _absa_sorted_mask).sum())
H_U_BON = BO[N_STAR][1]

print("\n  C 组：best-of-N 取最大回报 vs β=1 条件化 / 二值接受拒绝采样")
print("     N     J(best-of-N)   选中 |a−θ| 的熵")
for N in N_GRID:
    print(f"  {N:>4}    {BO[N][0]:>12.4f}    {BO[N][1]:>10.4f}")
print(f"  J(β=1 条件化) = {J_COND:.4f}；J(二值接受拒绝采样) = {J_REJ:.4f}")
print(f"  数值最接近的 N* = {N_STAR}（J={BO[N_STAR][0]:.4f}）")
print(f"  |a−θ| 分布熵：条件化 {H_U_COND:.4f} vs best-of-{N_STAR} {H_U_BON:.4f}")
print(f"  区间边缘尾部质量 Pr(|a−θ|>{TAIL_THR:.1f})："
      f"条件化 {TAIL_COND:.4f} vs best-of-{N_STAR} {TAIL_BON:.4f}")

CK.ok("best-of-1 就是行为策略（order statistics 自校验）",
      abs(BO[1][0] - J_REF) < 1e-9,
      f"J(bo1) {BO[1][0]:.4f} vs J_ref {J_REF:.4f}")
CK.ok("二值接受拒绝采样的分布就是 π_ref(a|I)（与 β=1 条件化等价）",
      np.max(np.abs(Q_REJ - Q_COND)) < 1e-12 and abs(J_REJ - J_COND) < 1e-12,
      f"J_rej {J_REJ:.4f} = J_cond {J_COND:.4f}")
CK.ok("best-of-N 的期望回报随 N 单调增、上界是成功点 V=0",
      all(BO[N_GRID[i + 1]][0] >= BO[N_GRID[i]][0] - 1e-9
          for i in range(len(N_GRID) - 1)) and BO[N_GRID[-1]][0] <= 1e-9,
      " → ".join(f"{BO[N][0]:.4f}" for N in N_GRID))
CK.ok("N→∞ 时 best-of-N 收敛到最优动作（J→0）",
      BO[256][0] > -0.001, f"J(bo256)={BO[256][0]:.4f}")
CK.ok("存在有限 N 使 best-of-N 的期望回报≈条件化，但二者不是同一分布",
      abs(BO[N_STAR][0] - J_COND) < 0.004,
      f"N*={N_STAR}：J {BO[N_STAR][0]:.4f} vs {J_COND:.4f}")
CK.ok("反直觉：J 最接近时连 |a−θ| 熵都几乎相同（回报+熵都分不开机制），"
      "但条件化在区间边缘仍有质量、best-of-N 几乎只落在 θ 附近（分布本质不同）",
      abs(H_U_BON - H_U_COND) < 0.5 and TAIL_COND > TAIL_BON + 0.02,
      f"熵 {H_U_COND:.4f} vs {H_U_BON:.4f}；"
      f"Pr(|a−θ|>0.7) cond {TAIL_COND:.4f} vs bo{N_STAR} {TAIL_BON:.4f}")
CK.ok("best-of-N 的代价是 N 倍采样；条件化从单模型一次采样即可",
      N_STAR > 1, f"等价点 N*={N_STAR}（部署需 N* 倍候选）")


# ===================================================================== D 组
CK.banner("D 组：代价与边界（二值化信息损失、value 标定误差传导、何时迭代反而变差）")

# ---- 1. 二值化丢的信息：A 的方差分解 Var(A)=Var(E[A|I]) + E[Var(A|I)]
_mean_A = float(P_REF @ ADV)
_var_A = float(P_REF @ (ADV - _mean_A) ** 2)
_between = 0.0
_within = 0.0
for m in (MASK_I, ~MASK_I):
    pm = P_REF[m] / P_REF[m].sum()
    am = ADV[m]
    mu_m = float(pm @ am)
    _between += P_REF[m].sum() * (mu_m - _mean_A) ** 2
    _within += P_REF[m].sum() * float(pm @ (am - mu_m) ** 2)
INFO_LOSS = _within / _var_A

# ---- 2. value 标定误差如何传导到策略
#  (a) 仿射平移 V_est=V+δ：advantage 排序不变 → I 不变 → J 不变（二值化免疫）
DELTA_GRID = (-0.05, -0.01, 0.0, 0.01, 0.05)
J_AFFINE = {}
for delta in DELTA_GRID:
    m, _ = advantage_mask(P_REF, ADV + delta, Q_POS)
    q = P_REF * m
    q /= q.sum()
    J_AFFINE[delta] = policy_quality(q)

#  (b) 与 a 相关的非线性标定噪声 V_est=V+ξ(a)：打乱 advantage 排序 → I 错标
SIG_CAL_GRID = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
J_CAL = {}
for sig in SIG_CAL_GRID:
    vals = []
    for s in range(N_SEEDS_CAL):
        rng = np.random.default_rng(31000 + s)
        noise = rng.normal(0.0, sig, size=len(A_GRID)) if sig > 0 \
            else np.zeros_like(A_GRID)
        adv_est = (V_GRID + noise) - (V_REF_GRID + float(P_REF @ noise))
        m, _ = advantage_mask(P_REF, adv_est, Q_POS)
        q = P_REF * m
        q /= q.sum()
        vals.append(policy_quality(q))
    J_CAL[sig], _ = mean_std(vals)

# ---- 3. 迭代反而变差：坏 corrections（θ1+c 偏离 θ1，论文「纠正本身非最优监督」）
C_GRID = (0.0, 1.0, 2.0, 3.0, 4.0)
J_BADCORR = {}
for c in C_GRID:
    J_BADCORR[c] = b_avg("corr", c=c)[-1]

print("\n  D 组 1：二值化把连续 A 压成 1 bit，丢了多少信息")
print(f"    Var(A) 总 {_var_A:.6f} = 组间(二值可解释) {_between:.6f}"
      f" + 组内(丢失) {_within:.6f}")
print(f"    组内方差占比 = {INFO_LOSS:.4f}；但 J 仍从 {J_REF:.4f} 提到 {J_COND:.4f}")
print(f"    连续加权最优 J = {J_CONT_BEST:.4f}（t={T_BEST:.2f}），二值 J = {J_COND:.4f}")
print("\n  D 组 2：value function 标定误差传导到策略")
print("     仿射平移 δ      J(条件化)")
for delta in DELTA_GRID:
    print(f"     {delta:>8.3f}      {J_AFFINE[delta]:.4f}")
print("     标定噪声 σ      J(条件化，24 seed 平均)")
for sig in SIG_CAL_GRID:
    print(f"     {sig:>8.3f}      {J_CAL[sig]:.4f}")
print("\n  D 组 3：坏 corrections 让迭代反而变差（c = 纠正动作偏离 θ1）")
print("     纠正偏差 c      K 轮后 J1")
for c in C_GRID:
    print(f"     {c:>8.2f}      {J_BADCORR[c]:.4f}")
print(f"     离线基线（不加任何数据）J1 = {J_OFF[-1]:.4f}")

CK.ok("二值化确实丢了 A 的大部分数值信息（组内方差占比 > 0.5）",
      INFO_LOSS > 0.5, f"组内占比 {INFO_LOSS:.4f}")
CK.ok("但保留的「正/负」方向信息足够：二值 J 仍远优于 J_ref",
      J_COND > J_REF + 0.02, f"{J_REF:.4f} → {J_COND:.4f}")
CK.ok("value 的仿射标定误差（平移 δ）不改变二值 I → 策略质量免疫（反直觉）",
      max(J_AFFINE.values()) - min(J_AFFINE.values()) < 1e-12,
      "δ 扫描 " + " / ".join(f"{J_AFFINE[d]:.4f}" for d in DELTA_GRID))
CK.ok("但非线性标定噪声打乱 advantage 排序，J 随 σ 单调不增",
      all(J_CAL[SIG_CAL_GRID[i + 1]] <= J_CAL[SIG_CAL_GRID[i]] + 1e-6
          for i in range(len(SIG_CAL_GRID) - 1)),
      " → ".join(f"{J_CAL[s]:.4f}" for s in SIG_CAL_GRID))
CK.ok("标定噪声让 J 明显下降（σ=0.05 相对无噪声降幅 > 0.01）",
      J_CAL[0.0] - J_CAL[0.05] > 0.01,
      f"σ=0 {J_CAL[0.0]:.4f} → σ=0.05 {J_CAL[0.05]:.4f}（两原始读数）")
CK.ok("反直觉：二值化对 value 标定误差相当鲁棒——σ 放大到 0.2（远超 V 量级）"
      "J 仍优于 J_ref，改进没有被完全抹平",
      J_CAL[0.2] > J_REF + 0.005 and J_CAL[0.2] < J_CAL[0.0] - 0.02,
      f"σ=0.2 {J_CAL[0.2]:.4f} vs J_ref {J_REF:.4f}、σ=0 {J_CAL[0.0]:.4f}")
CK.ok("坏 corrections 让迭代反而变差：c 增大时 K 轮后 J1 单调下降",
      all(J_BADCORR[C_GRID[i + 1]] < J_BADCORR[C_GRID[i]]
          for i in range(len(C_GRID) - 1)),
      " → ".join(f"c={c:.0f}: {J_BADCORR[c]:.4f}" for c in C_GRID))
CK.ok("纠正偏差过大时，加 corrections 比不加（离线基线）更差（反直觉）",
      J_BADCORR[C_GRID[-1]] < J_OFF[-1],
      f"c={C_GRID[-1]:.0f} 的 J1 {J_BADCORR[C_GRID[-1]]:.4f} "
      f"vs 离线基线 {J_OFF[-1]:.4f}（对应论文「纠正非最优监督」）")


# ===================================================================== 正文回填表
CK.banner("正文回填表（以下数字与上面逐条对齐，可直接写进文章；均为原始读数）")
print(f"  [世界] A_MAX={A_MAX:.1f}  θ={THETA:.1f}  π_ref: μ_g={MU_GOOD:.1f}"
      f" μ_b={MU_BAD:.1f} σ={SIG_REF:.1f} w_bad={W_BAD:.2f}")
print(f"  [世界] V_ref 解析={V_REF_ANA:.4f} 网格={V_REF_GRID:.4f}"
      f" 201bin直方图={V_REF_HIST:.4f} 梯度相对误差={_rel_grad:.2e}")
print(f"  [二值化] q={PR_I:.4f}  τ={TAU:.4f}  ε_l={EPS_L:.4f}"
      f"  std(A)={STD_A:.5f}  B={B_BINS}  N_lookahead={N_LOOKAHEAD}")
print(f"  [论文常量] 部署 β={BETA_DEPLOY}  CFG β∈[{CFG_RANGE[0]},{CFG_RANGE[1]}]"
      f"  advantage dropout={ADV_DROPOUT:.2f}  ε_l 取 30% 分位")
print(f"  [A组] J_ref={J_REF:.4f}  J_cond(β=1)={J_COND:.4f}"
      f"  J_rej(拒绝采样)={J_REJ:.4f}  J_cont_best(t={T_BEST:.2f})={J_CONT_BEST:.4f}")
print(f"  [A组] 保留回报={R_KEEP:.4f} 丢弃回报={R_DROP:.4f}"
      f" 行为均值={MEAN_REF:.4f} 条件均值={MEAN_COND:.4f}")
print(f"  [A组] 熵 H_ref={H_REF:.4f} H_cond={H_COND:.4f}  KL_cond={KL_COND:.4f}"
      f"  Bayes等价最大偏差={_bayes_eq:.2e}")
print("  [A组] 连续加权扫描 t:J:H:KL = " + "  ".join(
    f"t={t:.2f}:{CONT[t]['J']:.4f}:{CONT[t]['H']:.4f}:{CONT[t]['KL']:.4f}"
    for t in T_GRID))
print("  [B组] J_offline(K=0..6) = " + "  ".join(f"{v:.4f}" for v in J_OFF))
print("  [B组] J_autonomous(K=0..6) = " + "  ".join(f"{v:.4f}" for v in J_AUTO))
print("  [B组] J_corrections(K=0..6) = " + "  ".join(f"{v:.4f}" for v in J_CORR))
print(f"  [B组] 解析最优下界 J1_floor={J1_FLOOR:.4f}"
      f"  corr逐轮增益=" + "  ".join(f"{g:.4f}" for g in _gain_corr))
print("  [C组] J_bestofN(N:J) = " + "  ".join(
    f"N={N}:{BO[N][0]:.4f}" for N in N_GRID))
print(f"  [C组] N*={N_STAR}  J_boN*={BO[N_STAR][0]:.4f}"
      f"  H_u_cond={H_U_COND:.4f}  H_u_boN*={H_U_BON:.4f}")
print(f"  [C组] Pr(|a−θ|>0.7)：条件化={TAIL_COND:.4f}"
      f"  best-of-{N_STAR}={TAIL_BON:.4f}")
print("  [D组] 方差 VarA=" + f"{_var_A:.6f}" + " 组间=" + f"{_between:.6f}"
      + " 组内=" + f"{_within:.6f}" + f" 信息损失占比={INFO_LOSS:.4f}")
print("  [D组] 仿射平移 δ:J = " + "  ".join(
    f"δ={d:.3f}:{J_AFFINE[d]:.4f}" for d in DELTA_GRID))
print("  [D组] 标定噪声 σ:J = " + "  ".join(
    f"σ={s:.3f}:{J_CAL[s]:.4f}" for s in SIG_CAL_GRID))
print("  [D组] 坏纠正 c:J1 = " + "  ".join(
    f"c={c:.1f}:{J_BADCORR[c]:.4f}" for c in C_GRID)
    + f"  离线基线={J_OFF[-1]:.4f}")
print("  注（反直觉/边界，写入正文须照此口径）：")
print("   · β=1 条件化 = 按 I 硬筛选数据；但 Eq.(3) 训练仍用全部数据学无条件+条件两项，")
print("     二者不矛盾。硬二值指示下 p(I|A)=𝟙/Pr(I) 为常数，CFG β 在理论 Bayes 形式里退化，")
print("     论文的 β>1 锐化依赖 learned conditional 与 unconditional 的连续似然比，本探针未建模。")
print("   · 二值化对 value 的仿射标定误差免疫（I 只依赖 A 排序）；对非线性噪声也相当鲁棒：")
print(f"     σ 放大到 0.2（远超 V 量级）J 仍为 {J_CAL[0.2]:.4f}、"
      f"优于 J_ref {J_REF:.4f}，改进没被抹平。")
print("   · best-of-N（取最大）≠ 条件化；仅「二值接受拒绝采样」与 β=1 条件化分布相同。")
print("     反直觉：在 J 最接近的 N* 处，两者连 |a−θ| 的熵都几乎相等，")
print("     回报 + 熵都分不开机制；真正的区别在区间边缘尾部质量与 N* 倍采样代价。")
print("   · 坏 corrections 会让迭代比不加数据更差，对应论文「纠正本身非最优监督」。")

CK.summary()
