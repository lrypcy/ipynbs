# -*- coding: utf-8 -*-
"""10 篇（π0.7）合成探针：把「diverse context conditioning」拆成可解析高斯世界。

论文：arXiv:2604.15483v2，*π0.7: a Steerable Generalist Robotic Foundation
Model with Emergent Capabilities*。事实只取事实清单 10_pi07.md（唯一允许的论文
事实来源）。**只验机制，不重算论文成绩**：真机 SR / 人类对照 / 参数量 / dropout /
β / RTC 延迟全部是引用，见末尾「论文常量（引用）」与 README 边界声明。

四组 + 前置自校验：

  自校验  Σ 正定 / U 正交 / 手写最小二乘梯度 ↔ 有限差分 / 闭式 W* ↔ 多 seed
          拟合反算 / cond_sample 高斯矩构造真值 ↔ MC 反算
  A 组    多条件分量 + 训练期 dropout：遵从度随 dropout 概率下降、无条件分支
          被训练、满上下文精度代价、任意子集鲁棒性的 U 形（§V 引言 / §V-E）
  B 组    CFG / guidance scale β∈{1.3,1.7,2.2}：欠引导 / 过引导 / 多样性损失
          的倒 U，解析最优 β* 与实测最优 β 对照（§VII）
  C 组    元数据 steering：快 / 稳 / 标记失误三档产生不同轨迹，可引导性与
          高速档误差上升的副作用（§V-C / §VII）
  D 组    训练期模拟 0–12 步 RTC / 240 ms 延迟：条件化是否加剧/缓解陈旧观测
          带来的分布偏移；复用 soft_mask 的「软衰减」思路做条件信任度（§VI-B）

世界：**线性高斯条件模型**。K=5 个 prompt 分量各给一个标量上下文值 x_k；
真值动作在 K 个正交 DCT 时间基上的系数 z* = diag(B)·x + ξ。训练时按分量
dropout 把 x̃ = m⊙x 喂给一个线性解码器 W；因 x 各分量相关（共享 latent），
W 的非对角项就是「用一个分量当另一个分量的代理」的 shortcut。全部量有闭式解，
采样只用于多 seed 平均与构造真值↔反算值。

纯 numpy / CPU / 固定 seed / 多 seed 平均 / 每组末尾断言 / 最后打印断言总数。
运行：~/Software/miniconda3/bin/python pi07_lab.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from series_common import (  # noqa: E402
    Check, soft_mask, ar_kernel, cond_sample, jerk, mean_std,
)

CK = Check()

# ============================================================ 论文常量（引用 · 非合成）
# 这些数字只作引用标注，不参与合成世界的自由度，不得与合成读数混用。
PAPER = dict(
    arxiv="2604.15483v2",
    human_progress=90.9, human_sr=80.6,
    pi07_progress=85.6, pi07_sr=80.0,
    backbone="5B", action_expert="860M", vlm="Gemma3 4B",
    dropout_subgoal=0.25,      # 子目标图只加到 25% 的样本（present rate，非 drop rate）
    dropout_subtask=0.30,      # 有子目标图时子任务指令再丢 30%
    dropout_meta_group=0.15,   # 整组 episode metadata 丢 15%
    dropout_meta_single=0.05,  # speed / quality / mistake 各自再丢 5%
    dropout_control=0.0,       # 控制模态不做 dropout
    beta_grid=(1.3, 1.7, 2.2),
    rtc_steps=(0, 12), rtc_ms=240,
    license="arXiv.org perpetual non-exclusive license (v2 / abs)；v1 曾标 CC BY 4.0",
)

# ============================================================ 合成世界自由度（全部自设）
K = 5                       # prompt 分量数：0 任务语言 / 1 子任务 / 2 元数据 / 3 控制模态 / 4 视觉子目标
COMP = ["任务语言 ℓ", "子任务 ℓ̂", "片段元数据 m", "控制模态 c", "视觉子目标 g"]
H = 24                      # action chunk 时间步（轨迹长度）
B = np.array([1.00, 0.75, 0.35, 0.85, 0.70])   # 各分量真值增益；分量 2 (metadata) 故意设弱
RHO = 0.70                 # 分量间相关：Σ = (1−ρ)I + ρ11ᵀ（共享 latent 造成 shortcut）
LAM = 0.25                 # 解码器岭正则（有限数据 / 容量限制，制造 shrinkage）
SIG2 = 0.09                # 真值动作系数噪声方差 σ²（σ=0.30）
SIG = (1.0 - RHO) * np.eye(K) + RHO * np.ones((K, K))
# 训练期各分量 dropout 概率 p_k（按论文 25/30/15/5% 的语义映射到 5 个分量）
P_BASE = np.array([0.00, 0.30, 0.15, 0.00, 0.75])
T_GRID = [0.0, 0.25, 0.50, 1.00, 1.50, 2.00, 3.00]   # dropout 全局尺度（t=1 即论文量级）

# B 组：CFG 的可解析高斯
KAPPA = 0.685              # 条件分支对被 dropout 收缩后的均值增益（κ<1，欠引导）
R_CFG = 0.50               # r = σ_c²/τ²：条件方差 / 无条件方差
TAU2 = 0.81                # 无条件方差 τ²（τ=0.90）
SC2 = R_CFG * TAU2         # 条件方差 σ_c²
W_DIV = 0.02               # 复合质量里多样性惩罚权重（自设，使两项量纲可比）
BETA_GRID = PAPER["beta_grid"]
BETA_FINE = np.arange(0.0, 4.0001, 0.02)
BETA_MC = np.arange(1.30, 2.2001, 0.05)

# C 组：三档元数据（把 speed/quality/mistake 抽象成单个 steering 标量 x₂）
BASE_X = np.array([0.8, -0.4, 0.0, 0.5, 0.3])
C_X2 = np.array([0.0, 2.5, -1.5])      # 稳 / 快 / 标记失误
C_NAMES = ["稳 (speed 低, quality 5, mistake false)",
           "快 (speed 第15百分位)", "标记失误 (mistake true)"]
G_TRACK = 0.5              # PD 式一阶跟踪增益

# D 组：AR(1) 上下文漂移 + 陈旧观测
PHI = 0.90                 # AR(1) 系数
D_GRID = np.array([0, 3, 6, 9, 12])    # 0–12 步 RTC
D_MS = D_GRID * 20         # 50 Hz → 每步 20 ms，12 步 = 240 ms
TAU_C = -1.0 / np.log(PHI) # 使 exp(−d/τ_c)=φ^d 的软衰减时间常数

N_SEEDS = 40               # 多 seed 平均
BASE_SEED = 20261110
SEEDS = [BASE_SEED + 1000 * i for i in range(N_SEEDS)]

_rng_world = np.random.default_rng(BASE_SEED)
BACK: dict[str, np.ndarray] = {}


def store(key, val):
    BACK[key] = np.atleast_1d(np.asarray(val, dtype=float))


def mc(f, seeds=SEEDS):
    """对一组 seed 跑 f(seed)→标量，返回 (均值, 标准差)。"""
    return mean_std([f(s) for s in seeds])


# ============================================================ 共用解析工具
def dct_matrix(n):
    """正交 DCT-II 矩阵；列是不同频率的时间基，行/列均正交。"""
    c = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    M = np.sqrt(2.0 / n) * np.cos(np.pi * (k + 0.5) * c / n)
    M[0, :] = np.sqrt(1.0 / n)
    return M


def build_U():
    """K 个正交时间基；令分量 2 (metadata) 落在最高频 mode，使「快」档天然高 jerk。"""
    M = dct_matrix(H)
    FREQ = [1, 2, 5, 3, 4]
    return np.column_stack([M[f, :] for f in FREQ]), FREQ


U, FREQ = build_U()


def Wstar(t, lam=LAM):
    """训练期 dropout 尺度 t 下，岭回归解码器 W* 的闭式解。

    M = E[x̃x̃ᵀ] = DΣD + diag(p(1−p))；R = E[z*x̃ᵀ] = diag(B)ΣD；W* = R(M+λI)⁻¹。
    这是「有限数据 + 岭正则 + 分量相关」下学到的一张表，非对角项即 shortcut。
    """
    p = np.clip(P_BASE * t, 0.0, 0.95)
    D = np.diag(1.0 - p)
    Pdiag = np.diag(p * (1.0 - p))
    M = D @ SIG @ D + Pdiag
    R = np.diag(B) @ SIG @ D
    return R @ np.linalg.inv(M + lam * np.eye(K)), p


def pred_err(mask, t):
    """在测试时给出 prompt 子集 mask 时，动作系数的期望平方误差（闭式）。

    err = tr[(diag(B) − W*diag(mask)) Σ (·)ᵀ] + Kσ²
    """
    W, _ = Wstar(t)
    A = np.diag(B) - W @ np.diag(mask)
    return float(np.trace(A @ SIG @ A.T) + K * SIG2)


def all_subsets():
    """「任意子集」：分量 0（任务语言）恒 present，其余 4 个均匀遍历 2⁴ 个子集。"""
    out = []
    for bits in range(1 << (K - 1)):
        m = np.ones(K)
        for k in range(1, K):
            m[k] = (bits >> (k - 1)) & 1
        out.append(m)
    return out


def err_sub(t):
    return float(np.mean([pred_err(m, t) for m in all_subsets()]))


def track_mc(a, seed):
    """PD 式一阶跟踪 + AR(1) 执行噪声：高频（高 jerk）轨迹跟不上，多 seed 平均。

    复用 series_common.ar_kernel 构造时间相关的执行噪声协方差。
    """
    rng = np.random.default_rng(seed)
    SigN = ar_kernel(H, rho=0.6, sigma=0.025)
    eps = rng.multivariate_normal(np.zeros(H), SigN, method="cholesky")
    cur = 0.0
    out = np.zeros(H)
    for t in range(H):
        cur = cur + G_TRACK * (a[t] - cur) + eps[t]
        out[t] = cur
    return float(np.sqrt(np.mean((out - a) ** 2)))


# ===================================================================== 前置自校验
CK.banner("前置自校验：世界不变量 / 手写梯度 ↔ 有限差分 / 闭式 W* ↔ 多 seed 拟合 / cond_sample 矩")

# ---- 1. 世界不变量
_eig = np.linalg.eigvalsh(SIG)
_ortho = float(np.max(np.abs(U.T @ U - np.eye(K))))
CK.ok("分量协方差 Σ 正定（相关结构可辨识）",
      float(_eig.min()) > 1e-9,
      f"最小特征值 {_eig.min():.4f}，ρ={RHO:.2f}")
CK.ok("动作时间基 U 正交（系数空间 ↔ 轨迹空间等距）",
      _ortho < 1e-12, f"UᵀU−I 最大偏差 {_ortho:.2e}，H={H}，freq={FREQ}")
CK.ok("元数据分量落在最高频时间基（为 C 组高速档副作用埋构造真值）",
      FREQ[2] == max(FREQ), f"freq={FREQ}，metadata→{FREQ[2]}")
CK.ok("12 步 × 20 ms = 240 ms（50 Hz 的引用口径自洽）",
      D_MS[-1] == PAPER["rtc_ms"] and D_GRID[-1] == PAPER["rtc_steps"][1],
      f"{D_GRID[-1]} 步 × 20 ms = {D_MS[-1]} ms")

# ---- 2. 手写最小二乘梯度 ↔ 有限差分（λ=0 时 W*=R M⁻¹）
_p = np.clip(P_BASE * 1.0, 0.0, 0.95)
_D = np.diag(1.0 - _p)
_M = _D @ SIG @ _D + np.diag(_p * (1.0 - _p))
_R = np.diag(B) @ SIG @ _D
_Wopt = _R @ np.linalg.inv(_M)


def _loss(W):
    return float(np.trace(np.diag(B) @ SIG @ np.diag(B))
                 - 2.0 * np.trace(W @ _R.T) + np.trace(W @ _M @ W.T) + K * SIG2)


def _grad(W):
    return 2.0 * (W @ _M - _R)


_W0 = _rng_world.normal(size=(K, K))
_g = _grad(_W0)
_gn = np.zeros((K, K))
_h = 1e-6
for _i in range(K):
    for _j in range(K):
        _E = np.zeros((K, K)); _E[_i, _j] = _h
        _gn[_i, _j] = (_loss(_W0 + _E) - _loss(_W0 - _E)) / (2 * _h)
_rel_grad = float(np.linalg.norm(_g - _gn) / np.linalg.norm(_gn))
CK.ok("线性解码器手写梯度 ∂L/∂W=2(WM−R) 与有限差分一致",
      _rel_grad < 1e-6, f"相对误差 {_rel_grad:.2e}（K={K}，λ=0）")
CK.ok("W*=RM⁻¹ 处梯度为 0（正规方程自校验）",
      float(np.linalg.norm(_grad(_Wopt))) < 1e-9,
      f"‖∇L(W*)‖={np.linalg.norm(_grad(_Wopt)):.2e}")

# ---- 3. 闭式 W*(t=1) ↔ 多 seed 有限样本拟合反算（构造真值 ↔ 反算值）
def _fit_W(seed, n=20000):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(SIG + 1e-12 * np.eye(K))
    x = rng.normal(size=(n, K)) @ L.T
    p = np.clip(P_BASE * 1.0, 0.0, 0.95)
    m = (rng.random((n, K)) > p).astype(float)
    xt = m * x
    z = x * B + rng.normal(0.0, np.sqrt(SIG2), size=(n, K))
    Rhat = z.T @ xt / n
    Mhat = xt.T @ xt / n
    return Rhat @ np.linalg.inv(Mhat + LAM * np.eye(K))


_Wfit = np.mean([_fit_W(s) for s in SEEDS], axis=0)
_Wclosed, _ = Wstar(1.0)
_fit_rel = float(np.max(np.abs(_Wfit - _Wclosed)) / np.max(np.abs(_Wclosed)))
CK.ok("闭式 W*(t=1) ↔ 多 seed 有限样本岭回归拟合一致",
      _fit_rel < 0.02, f"最大相对偏差 {_fit_rel:.4f}（{N_SEEDS} seed × 20000 样本）")

# ---- 4. cond_sample(w=0) 的高斯矩：构造真值 ↔ MC 反算
_mu0 = np.array([0.3, -0.2, 0.5, 0.1, -0.4])
_Sig0 = SC2 * np.eye(K)


def _cs_sample(seed):
    rng = np.random.default_rng(seed)
    return cond_sample(_mu0, _Sig0, np.arange(K), np.zeros(K),
                       np.zeros(K), 1.0, rng)[0]


_CS_SEEDS = [BASE_SEED + 70000 + s for s in range(400)]
_cs = np.array([_cs_sample(s) for s in _CS_SEEDS])
_cs_mean = _cs.mean(axis=0)
_cs_var = float(np.mean((_cs - _cs.mean(axis=0)) ** 2))
_cs_dev = float(np.max(np.abs(_cs_mean - _mu0)))
_cs_tol = 4.0 * np.sqrt(SC2 / len(_CS_SEEDS))   # 4σ 统计容差，非凑数阈值
CK.ok("cond_sample(w=0) 采样均值 ↔ 解析 μ（构造真值 ↔ MC 反算）",
      _cs_dev < _cs_tol,
      f"最大偏差 {_cs_dev:.4f} < 4σ={_cs_tol:.4f}（{len(_CS_SEEDS)} seed）")
CK.ok("cond_sample(w=0) 采样方差 ↔ 解析 σ_c²",
      abs(_cs_var - SC2) / SC2 < 0.15,
      f"MC {_cs_var:.4f} vs 解析 {SC2:.4f}")


# ===================================================================== A 组
CK.banner("A 组：多条件分量 + 训练期 dropout（§V 引言 / §V-E）：遵从度、无条件分支、子集鲁棒性的权衡")

ADH, OFFDIAG, ERR_FULL, ERR_UNCOND, ERR_SUB = {}, {}, {}, {}, {}
for t in T_GRID:
    W, p = Wstar(t)
    diag = np.diag(W)
    ADH[t] = diag / B
    OFFDIAG[t] = float(np.abs(W - np.diag(diag)).sum() / np.abs(diag).sum())
    ERR_FULL[t] = pred_err(np.ones(K), t)
    ERR_UNCOND[t] = pred_err(np.array([1, 0, 0, 0, 0]), t)
    ERR_SUB[t] = err_sub(t)

print("\n   t     p(各分量)              offdiag/diag   err_full  err_uncond   err_sub")
for t in T_GRID:
    print(f"  {t:>4.2f}  {np.round(np.clip(P_BASE*t,0,0.95),2)}   "
          f"{OFFDIAG[t]:>12.4f}   {ERR_FULL[t]:>8.4f}   {ERR_UNCOND[t]:>9.4f}   {ERR_SUB[t]:>8.4f}")
print("\n   各分量遵从度 adh_k = W*_{kk}/B_k（t=0 与 t=1 对照）：")
print(f"     t=0   {np.round(ADH[0.0], 4)}")
print(f"     t=1   {np.round(ADH[1.0], 4)}")
print(f"     t=1 的 p={np.round(np.clip(P_BASE,0,0.95),2)}（任务0 / 子任务0.30 / 元数据0.15 / 控制0 / 子目标0.75）")

_ah0, _ah1 = ADH[0.0], ADH[1.0]
CK.ok("无 dropout 时五个分量遵从度完全相同（没有谁被单独忽略）",
      float(_ah0.max() - _ah0.min()) < 0.01,
      "t=0 adh=" + " ".join(f"{v:.4f}" for v in _ah0))
CK.ok("加 dropout 后，从不丢弃的分量（任务语言/控制模态）遵从度不降反升",
      _ah1[0] > _ah0[0] and _ah1[3] > _ah0[3],
      f"任务 {_ah0[0]:.4f}→{_ah1[0]:.4f}，控制 {_ah0[3]:.4f}→{_ah1[3]:.4f}")
CK.ok("dropout 概率越高，该分量遵从度越低（论文量级下的直接权衡）",
      _ah1[0] > _ah1[2] > _ah1[1] > _ah1[4],
      "t=1 adh=" + " > ".join(f"{_ah1[i]:.4f}" for i in (0, 2, 1, 4)))
CK.ok("缺了 dropout 时模型忽略的是「无条件分支」：仅任务语言的误差最大",
      ERR_UNCOND[0.0] > ERR_UNCOND[1.0] > ERR_UNCOND[3.0],
      "err_uncond " + " → ".join(f"t={t}: {ERR_UNCOND[t]:.4f}" for t in (0.0, 1.0, 3.0)))
CK.ok("dropout 把无条件分支（CFG 的 C^uncond）训练出来：err_uncond 随 t 单调降",
      all(ERR_UNCOND[T_GRID[i + 1]] < ERR_UNCOND[T_GRID[i]] for i in range(len(T_GRID) - 1)),
      " → ".join(f"{ERR_UNCOND[t]:.4f}" for t in T_GRID))
CK.ok("代价：满上下文精度随 dropout 单调下降",
      all(ERR_FULL[T_GRID[i + 1]] > ERR_FULL[T_GRID[i]] for i in range(len(T_GRID) - 1)),
      f"err_full {ERR_FULL[0.0]:.4f} → {ERR_FULL[3.0]:.4f}")
_T_BEST = min(T_GRID, key=lambda t: ERR_SUB[t])
CK.ok("任意子集平均误差是 U 形，最优点恰在论文 dropout 量级 t=1.0",
      _T_BEST == 1.0 and ERR_SUB[0.0] - ERR_SUB[1.0] > 0.05
      and ERR_SUB[3.0] - ERR_SUB[1.0] > 0.03,
      "err_sub " + " → ".join(f"{ERR_SUB[t]:.4f}" for t in T_GRID) + f"，最优 t={_T_BEST}")
CK.ok("反直觉：相对交叉项占比随 dropout 上升（对角增益被压得比非对角更快）",
      all(OFFDIAG[T_GRID[i + 1]] > OFFDIAG[T_GRID[i]] for i in range(len(T_GRID) - 1)),
      "offdiag/diag " + " → ".join(f"{OFFDIAG[t]:.4f}" for t in T_GRID))
CK.ok("dropout 让 W* 更依赖恒在的分量、少依赖常丢的分量（列和重排自校验）",
      _ah1[4] < 0.5 * _ah1[0],
      f"子目标 adh {_ah1[4]:.4f} vs 任务语言 {_ah1[0]:.4f}")

store("T_GRID", T_GRID); store("ERR_FULL", [ERR_FULL[t] for t in T_GRID])
store("ERR_UNCOND", [ERR_UNCOND[t] for t in T_GRID])
store("ERR_SUB", [ERR_SUB[t] for t in T_GRID])
store("OFFDIAG", [OFFDIAG[t] for t in T_GRID])
store("ADH_T0", _ah0); store("ADH_T1", _ah1)
store("P_BASE", P_BASE)


# ===================================================================== B 组
CK.banner("B 组：CFG 引导强度 β∈{1.3,1.7,2.2} 的倒 U（§VII）：欠引导 / 过引导 / 多样性损失")

# 精确高斯 score-CFG：π_β = N(μ_β, Σ_β I)，
# Λ_β=(1+β)/σ_c²−β/τ²，γ(β)=(1+β)/(1+β(1−r))，μ_β=γ·μ_c=γ·κBx=α(β)·Bx。
def gamma(b):
    return (1.0 + b) / (1.0 + b * (1.0 - R_CFG))


def alpha(b):
    return KAPPA * gamma(b)


def sigma_beta(b):
    return SC2 / (1.0 + b * (1.0 - R_CFG))


def quality_loss(b):
    """复合质量损失（自设 w_div）：(α−1)² 欠/过引导 + w_div·归一化多样性损失。"""
    return (alpha(b) - 1.0) ** 2 + W_DIV * (1.0 - sigma_beta(b) / SC2)


BETA_ANA = (1.0 - KAPPA) / (KAPPA + R_CFG - 1.0)   # 闭式：α(β)=1 的纯遵从最优点
_Lfine = np.array([quality_loss(b) for b in BETA_FINE])
BETA_Q = float(BETA_FINE[int(np.argmin(_Lfine))])

ALPHA_G = np.array([alpha(b) for b in BETA_GRID])
DIV_G = np.array([sigma_beta(b) / SC2 for b in BETA_GRID])
LOSS_G = np.array([quality_loss(b) for b in BETA_GRID])
DIST_G = np.array([abs(alpha(b) - 1.0) * float(np.linalg.norm(B * BASE_X))
                   for b in BETA_GRID])

print(f"\n   κ={KAPPA}  r=σ_c²/τ²={R_CFG}  σ_c²={SC2:.4f}  w_div={W_DIV}")
print(f"   解析纯遵从最优 β*（α=1 的闭式根）={BETA_ANA:.4f}；"
      f"复合质量解析最优 β_Q={BETA_Q:.3f}")
print("\n     β      α(有效增益)   γ     Σ_β/σ_c²   多样性损失   ‖μ_β−Bx‖   quality_loss")
for b, a, g, dv, ds, lo in zip(BETA_GRID, ALPHA_G, [gamma(b) for b in BETA_GRID],
                               DIV_G, DIST_G, LOSS_G):
    print(f"  {b:>5.2f}   {a:>11.4f}   {g:>6.4f}   {dv:>9.4f}   {1-dv:>9.4f}"
          f"   {ds:>10.4f}   {lo:>12.6f}")
print(f"   β=0：α={alpha(0):.4f} loss={quality_loss(0):.6f}；"
      f"β=4：α={alpha(4):.4f} loss={quality_loss(4):.6f}")

# MC 实测最优：公共随机数（400 seed）下从 π_β 采样，用 cond_sample 实现，
# 经验有效增益 α_hat(β)=α(β)+√Σ_β·(ε̄·Bx)/‖Bx‖²，argmin|α_hat−1|。
N_SEEDS_B = 2000
_XB = np.array([0.8, -0.4, 0.9, 0.5, 0.3])
_bx = B * _XB
_bx2 = float(_bx @ _bx)
_samples = []
for _s in range(N_SEEDS_B):
    _rng = np.random.default_rng(BASE_SEED + 31337 + _s)
    _mu = np.zeros(K)
    _Sigma = np.eye(K)
    _samples.append(cond_sample(_mu, _Sigma, np.arange(K), np.zeros(K),
                                np.zeros(K), 1.0, _rng)[0])
_eps_bar = np.mean(_samples, axis=0)
_proj = float(_eps_bar @ _bx) / _bx2     # 公共随机数下的有限 seed 估计偏差
ALPHA_HAT = np.array([alpha(b) + np.sqrt(sigma_beta(b)) * _proj for b in BETA_MC])
_i_emp = int(np.argmin(np.abs(ALPHA_HAT - 1.0)))
BETA_EMP = float(BETA_MC[_i_emp])
print(f"\n   MC 实测最优 β_emp={BETA_EMP:.2f}（{N_SEEDS_B} seed，公共随机数，"
      f"argmin|α_hat−1|）；α_hat(β_emp)={ALPHA_HAT[_i_emp]:.4f}，"
      f"与解析 β* 差 {abs(BETA_EMP-BETA_ANA):.4f}")

CK.ok("解析纯遵从最优 β* 落在论文候选集 {1.3,1.7,2.2} 内、且≈1.7",
      BETA_GRID[0] < BETA_ANA < BETA_GRID[2] and abs(BETA_ANA - 1.7) < 0.05,
      f"β*=(1−κ)/(κ+r−1)={BETA_ANA:.4f}")
CK.ok("论文网格上质量损失呈倒 U：β=1.7 优于 1.3 与 2.2",
      LOSS_G[1] < LOSS_G[0] and LOSS_G[1] < LOSS_G[2],
      "loss " + " / ".join(f"β={b}: {v:.6f}" for b, v in zip(BETA_GRID, LOSS_G)))
CK.ok("β 太小欠引导：β=1.3 的有效增益 α<1",
      ALPHA_G[0] < 1.0, f"α(1.3)={ALPHA_G[0]:.4f}")
CK.ok("β 太大过引导（动作失真）：β=2.2 的有效增益 α>1",
      ALPHA_G[2] > 1.0, f"α(2.2)={ALPHA_G[2]:.4f}，‖μ−Bx‖={DIST_G[2]:.4f}")
CK.ok("β 太大丢多样性：Σ_β/σ_c² 随 β 单调下降",
      all(DIV_G[i + 1] < DIV_G[i] for i in range(len(DIV_G) - 1)),
      "Σ_β/σ_c² " + " → ".join(f"{v:.4f}" for v in DIV_G))
CK.ok("动作失真 ‖μ_β−Bx‖ 是 U 形，β=1.7 最小",
      DIST_G[1] < DIST_G[0] and DIST_G[1] < DIST_G[2],
      "失真 " + " / ".join(f"β={b}: {v:.4f}" for b, v in zip(BETA_GRID, DIST_G)))
CK.ok("完全不引导（β=0）最差；β 过大（β=4）也回退",
      quality_loss(0.0) > LOSS_G[1] and quality_loss(4.0) > LOSS_G[1],
      f"β=0 {quality_loss(0):.6f} / β=1.7 {LOSS_G[1]:.6f} / β=4 {quality_loss(4):.6f}")
CK.ok("反直觉：多样性惩罚把复合质量最优点从纯遵从 β* 略向左推",
      BETA_Q < BETA_ANA and BETA_Q > BETA_GRID[0],
      f"β_Q={BETA_Q:.3f} < β*={BETA_ANA:.4f}，仍在候选集内")
_best_grid = BETA_GRID[int(np.argmin(LOSS_G))]
CK.ok("实测最优 β_emp 落在论文候选集内、且与解析 β* 接近（解析 ↔ 实测对照）",
      BETA_GRID[0] <= BETA_EMP <= BETA_GRID[2]
      and abs(BETA_EMP - BETA_ANA) < 0.30 and _best_grid == 1.7,
      f"β_emp={BETA_EMP:.2f}，β*={BETA_ANA:.4f}，差 {abs(BETA_EMP-BETA_ANA):.4f}，"
      f"网格最优={_best_grid}")

store("BETA_GRID", BETA_GRID); store("ALPHA_GRID", ALPHA_G)
store("DIV_GRID", DIV_G); store("LOSS_GRID", LOSS_G); store("DIST_GRID", DIST_G)
store("BETA_ANA", BETA_ANA); store("BETA_Q", BETA_Q); store("BETA_EMP", BETA_EMP)
store("BETA_FINE", BETA_FINE); store("LOSS_FINE", _Lfine)
store("KAPPA", KAPPA); store("R_CFG", R_CFG); store("SC2", SC2)


# ===================================================================== C 组
CK.banner("C 组：元数据 steering（§V-C / §VII）：快 / 稳 / 标记失误三档与高速档副作用")

_W1, _ = Wstar(1.0)        # 用论文量级 dropout 训出的解码器
_XS = []
for x2 in C_X2:
    x = BASE_X.copy(); x[2] = x2; _XS.append(x)
_A_TRUE = [U @ (B * x) for x in _XS]
_A_MODEL = [U @ (_W1 @ x) for x in _XS]

C_JERK = np.array([jerk(a) for a in _A_TRUE])
C_TRACK = np.array([mc(lambda s, aa=a: track_mc(aa, s))[0] for a in _A_TRUE])
C_STEER = np.array([float(np.linalg.norm(a - _A_TRUE[0])) for a in _A_TRUE])
C_MODELERR = np.array([float(np.linalg.norm(B * x - _W1 @ x) ** 2 + K * SIG2)
                       for x in _XS])
C_MODELSTEER = np.array([float(np.linalg.norm(a - _A_MODEL[0])) for a in _A_MODEL])
ADH2 = float((np.diag(_W1) / B)[2])

print(f"\n   解码器元数据分量遵从度 adh₂ = W*₂₂/B₂ = {ADH2:.4f}（dropout 15% 的衰减）")
print("\n   档位                     jerk   跟踪误差(40seed)  模型误差   ‖a−a稳‖   ‖â−â稳‖")
for i, name in enumerate(C_NAMES):
    print(f"   {name:<26} {C_JERK[i]:>7.4f}   {C_TRACK[i]:>13.4f}   "
          f"{C_MODELERR[i]:>8.4f}   {C_STEER[i]:>8.4f}   {C_MODELSTEER[i]:>9.4f}")

CK.ok("三档元数据产生三条不同真值轨迹（可引导性）",
      C_STEER[1] > 0.5 and C_STEER[2] > 0.3
      and float(np.linalg.norm(_A_TRUE[1] - _A_TRUE[2])) > 0.3,
      "‖a−a稳‖ " + " / ".join(f"{v:.4f}" for v in C_STEER))
CK.ok("模型可被元数据 steer，但增益被 dropout 衰减到 (0,1)",
      0.0 < ADH2 < 1.0 and C_MODELSTEER[1] > 0.3,
      f"adh₂={ADH2:.4f}，模型 ‖â快−â稳‖={C_MODELSTEER[1]:.4f}")
CK.ok("高速档 jerk 最大（元数据落在最高频时间基的构造结果）",
      C_JERK[1] > C_JERK[0] and C_JERK[1] > C_JERK[2],
      "jerk " + " / ".join(f"{v:.4f}" for v in C_JERK))
CK.ok("副作用：高速档跟踪误差明显上升（PD 一阶跟踪跟不上高频）",
      C_TRACK[1] > C_TRACK[0] * 1.5 and C_TRACK[1] > C_TRACK[2],
      "跟踪误差 " + " / ".join(f"{v:.4f}" for v in C_TRACK))
CK.ok("副作用：高速档模型动作误差也最大（steering 越强代价越大）",
      C_MODELERR[1] > C_MODELERR[0] and C_MODELERR[1] > C_MODELERR[2],
      "模型误差 " + " / ".join(f"{v:.4f}" for v in C_MODELERR))
CK.ok("反直觉：标记失误档不是最差档，介于稳与快之间",
      C_JERK[0] < C_JERK[2] < C_JERK[1]
      and C_MODELERR[0] < C_MODELERR[2] < C_MODELERR[1],
      f"失误档 jerk {C_JERK[2]:.4f} / 误差 {C_MODELERR[2]:.4f}")
CK.ok("可引导性强度与副作用同向：|x₂| 越大 jerk 与误差越大",
      C_STEER[1] > C_STEER[2] and C_JERK[1] > C_JERK[2]
      and C_MODELERR[1] > C_MODELERR[2],
      f"快 |x₂|=2.5 vs 失误 |x₂|=1.5：steer {C_STEER[1]:.4f}/{C_STEER[2]:.4f}")

store("C_X2", C_X2); store("C_JERK", C_JERK); store("C_TRACK", C_TRACK)
store("C_STEER", C_STEER); store("C_MODELERR", C_MODELERR)
store("C_MODELSTEER", C_MODELSTEER); store("ADH2", ADH2)
store("C_TRJ0", _A_TRUE[0]); store("C_TRJ1", _A_TRUE[1]); store("C_TRJ2", _A_TRUE[2])


# ===================================================================== D 组
CK.banner("D 组：0–12 步 / 240 ms 延迟下的条件化（§VI-B）：条件化加剧偏移，训练期 RTC 缓解")

# 陈旧观测：模型拿到 x_{t−d}，真值动作是 z*_t = B·x_t + ξ；AR(1) 使 Cov(x_t,x_{t−d})=φ^d·Σ。
# 条件化增益 s：uncond s=0；只在 d=0 训练（no_rtc）s=1；训练期模拟 0–12 步延迟（rtc）s=φ^d。
_c = float(B @ SIG @ B)
D_SHIFT = np.array([2.0 * float(np.trace(SIG)) * (1.0 - PHI ** d) for d in D_GRID])
D_PHI = np.array([PHI ** d for d in D_GRID])
D_UNCOND = np.full(len(D_GRID), _c + K * SIG2)
D_NORTC = np.array([_c * (1.0 - 2.0 * PHI ** d + 1.0) + K * SIG2 for d in D_GRID])
D_RTC = np.array([_c * (1.0 - PHI ** (2 * d)) + K * SIG2 for d in D_GRID])

# 复用 soft_mask 的「软衰减」思路（非 06 篇的 chunk inpainting）：
# 按 (d+i) 软化「条件信任度」trust(d,i)=exp(−(d+i)/τ_c)，在 chunk 上取归一化均值即 φ^d。
def trust_profile(d, i):
    return np.exp(-(d + np.asarray(i, float)) / TAU_C)


_igrid = np.arange(H)
_norm0 = float(np.mean(trust_profile(0, _igrid)))
D_S_SOFT = np.array([float(np.mean(trust_profile(d, _igrid))) / _norm0 for d in D_GRID])

# 训练期 RTC 的 MC 反算：随机 d∈[0,12] 下用最小二乘拟合条件增益 s_hat(d)=Cov(z*_t,x_{t−d})/Var(x_{t−d})
def _s_hat_bucket(d, seed, n=6000):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(SIG + 1e-12 * np.eye(K))
    x_old = rng.normal(size=(n, K)) @ L.T
    x_new = PHI ** d * x_old + np.sqrt(1.0 - PHI ** (2 * d)) * (rng.normal(size=(n, K)) @ L.T)
    z = x_new * B + rng.normal(0.0, np.sqrt(SIG2), size=(n, K))
    num = float(np.mean(np.sum(z * x_old, axis=1)))
    den = float(np.mean(np.sum(x_old * x_old, axis=1)))
    return num / den / float(np.mean(B))   # 归一到增益口径


D_S_HAT = np.array([mc(lambda s, dd=d: _s_hat_bucket(dd, s))[0] for d in D_GRID])

# soft_mask 结构自校验（只验「单调衰减 / 有界」这一共享性质，不照抄 06 篇实验）
_SM = soft_mask(H, H // 4, 6, "exp")
_SM_mid = _SM[6: H - H // 4]
_TP = trust_profile(6, _igrid)

print(f"\n   c=BᵀΣB={_c:.4f}  Kσ²={K*SIG2:.4f}  φ={PHI}  τ_c={TAU_C:.4f}")
print("\n   d(步)  d(ms)   φ^d   分布偏移shift  无条件  no_rtc(只d=0训练)  rtc(模拟0–12)  s_soft  s_hat(MC)")
for i, d in enumerate(D_GRID):
    print(f"   {d:>4}   {D_MS[i]:>5}   {D_PHI[i]:.4f}   {D_SHIFT[i]:>11.4f}   "
          f"{D_UNCOND[i]:>6.3f}   {D_NORTC[i]:>15.4f}   {D_RTC[i]:>12.4f}   "
          f"{D_S_SOFT[i]:>6.4f}   {D_S_HAT[i]:>8.4f}")

CK.ok("陈旧观测造成分布偏移，且随延迟 d 单调增大",
      all(D_SHIFT[i + 1] > D_SHIFT[i] for i in range(len(D_GRID) - 1)),
      "shift " + " → ".join(f"{v:.4f}" for v in D_SHIFT))
CK.ok("无条件模型（只用任务语言）不依赖陈旧上下文，误差与 d 无关",
      float(D_UNCOND.max() - D_UNCOND.min()) < 1e-9,
      f"err_uncond 恒为 {D_UNCOND[0]:.4f}")
CK.ok("no_rtc 条件化误差随延迟单调增大（条件化把陈旧值当真值用）",
      all(D_NORTC[i + 1] > D_NORTC[i] for i in range(len(D_GRID) - 1)),
      "no_rtc " + " → ".join(f"{v:.4f}" for v in D_NORTC))
_cross = [int(d) for d, e in zip(D_GRID, D_NORTC) if e > D_UNCOND[0]]
CK.ok("反直觉：延迟够大时，条件化比完全不条件化更差（no_rtc 越过无条件基线）",
      len(_cross) >= 2 and _cross[0] <= 9,
      f"no_rtc>uncond 的 d={_cross}（首次在 {int(D_MS[int(np.searchsorted(D_GRID, _cross[0]))])} ms）；"
      f"d=12：no_rtc {D_NORTC[-1]:.4f} vs 无条件 {D_UNCOND[0]:.4f}")
CK.ok("训练期模拟 0–12 步延迟（RTC）后，条件化始终不劣于无条件",
      bool(np.all(D_RTC <= D_UNCOND + 1e-9)),
      f"rtc {D_RTC[0]:.4f} → {D_RTC[-1]:.4f}，无条件 {D_UNCOND[0]:.4f}")
CK.ok("RTC 在每个 d 都不劣于 no_rtc（训练期延迟注入的收益）",
      bool(np.all(D_RTC <= D_NORTC + 1e-9)),
      "d=12：" + f"rtc {D_RTC[-1]:.4f} vs no_rtc {D_NORTC[-1]:.4f}")
CK.ok("RTC 的延迟退化远小于 no_rtc（0→12 步的误差增幅）",
      (D_RTC[-1] - D_RTC[0]) < 0.7 * (D_NORTC[-1] - D_NORTC[0]),
      f"rtc 增 {D_RTC[-1]-D_RTC[0]:.4f} vs no_rtc 增 {D_NORTC[-1]-D_NORTC[0]:.4f}")
CK.ok("软衰减信任度 s_soft(d)=exp(−d/τ_c) 逐点等于 Bayes 最优 φ^d",
      float(np.max(np.abs(D_S_SOFT - D_PHI))) < 1e-9,
      "s_soft " + " → ".join(f"{v:.4f}" for v in D_S_SOFT) + f"；φ^d " +
      " → ".join(f"{v:.4f}" for v in D_PHI))
CK.ok("训练期 RTC 的 MC 反算增益 s_hat(d) 逼近 φ^d（构造真值 ↔ 反算值）",
      float(np.max(np.abs(D_S_HAT - D_PHI))) < 0.05,
      "s_hat " + " → ".join(f"{v:.4f}" for v in D_S_HAT))
CK.ok("soft_mask 与 trust_profile 共享单调衰减 / (0,1] 有界性质（复用思路，非 06 实验）",
      bool(np.all(np.diff(_SM_mid) < 0)) and 0 < _SM_mid.min() < _SM_mid.max() <= 1.0
      and bool(np.all(np.diff(_TP) < 0)) and 0 < _TP.min() and _TP.max() <= 1.0,
      f"soft_mask 中段 {_SM_mid[0]:.4f}→{_SM_mid[-1]:.4f}；trust {_TP[0]:.4f}→{_TP[-1]:.4f}")

store("D_GRID", D_GRID); store("D_MS", D_MS); store("D_PHI", D_PHI)
store("D_SHIFT", D_SHIFT); store("D_UNCOND", D_UNCOND)
store("D_NORTC", D_NORTC); store("D_RTC", D_RTC)
store("D_S_SOFT", D_S_SOFT); store("D_S_HAT", D_S_HAT)


# ===================================================================== 论文常量（引用）
print("\n" + "=" * 74)
print("论文常量（引用 · 非合成探针数字 · 不得与合成读数混用）")
print("=" * 74)
print(f"  arXiv:{PAPER['arxiv']} π0.7 | {PAPER['backbone']} backbone / "
      f"{PAPER['action_expert']} action expert / {PAPER['vlm']} VLM")
print(f"  人类对照 task progress {PAPER['human_progress']}% / SR {PAPER['human_sr']}%；"
      f"π0.7 progress {PAPER['pi07_progress']}% / SR {PAPER['pi07_sr']}%（§IX-C）")
print(f"  训练期 dropout（§V-E / §VI-B，引用）：子目标图只加到 25% 样本、子任务再丢 30%、"
      f"整组 metadata 丢 15%、单分量各自再丢 5%、控制模态 0%")
print(f"  CFG guidance scale β∈{PAPER['beta_grid']}（§VII，只对 episode metadata 用 CFG）")
print(f"  训练期 RTC 模拟 0–12 步延迟，对应 50 Hz 上最大 240 ms（§VI-B）")
print(f"  两个必须遵守的核正：① 论文正文与 112 条参考文献零引用 RLT / RL Token；"
      f"② 离散文本本体状态属 π0.6，π0.7 跟随 MEM 改线性投影（非 π0.5）")
print(f"  RECAP 一词只见官方博客，论文正文中对应实体是 π*0.6；"
      f"许可 {PAPER['license']}，官方图不复用")


# ===================================================================== 正文回填表
CK.banner("正文回填表（以下数字与上面逐条对齐，均为原始读数；博客合成数字须能在此 grep 到）")
print(f"  [世界] K={K}  H={H}  B={[float(v) for v in np.round(B,4)]}  "
      f"ρ={RHO}  λ={LAM}  σ²={SIG2}  freq={FREQ}")
print(f"  [世界] P_BASE(dropout t=1)={[float(v) for v in np.round(P_BASE,2)]}  T_GRID={T_GRID}")
print(f"  [引用] backbone={PAPER['backbone']} action_expert={PAPER['action_expert']} "
      f"human={PAPER['human_progress']}/{PAPER['human_sr']} pi07={PAPER['pi07_progress']}/{PAPER['pi07_sr']} "
      f"dropout=25/30/15/5% beta={PAPER['beta_grid']} rtc=0–12步/240ms")
print(f"  [自校验] Σ最小特征值={_eig.min():.4f}  U正交偏差={_ortho:.2e} "
      f"梯度相对误差={_rel_grad:.2e}  W*正规方程梯度={np.linalg.norm(_grad(_Wopt)):.2e}")
print(f"  [自校验] 闭式W*↔拟合相对偏差={_fit_rel:.4f}  "
      f"cond_sample均值最大偏差={_cs_dev:.4f}(4σ={_cs_tol:.4f}) 方差MC={_cs_var:.4f}/{SC2:.4f}")
print(f"  [A组] adh(t=0)= " + "  ".join(f"{v:.4f}" for v in _ah0))
print(f"  [A组] adh(t=1)= " + "  ".join(f"{v:.4f}" for v in _ah1))
print("  [A组] err_full(t grid) = " + "  ".join(f"{ERR_FULL[t]:.4f}" for t in T_GRID))
print("  [A组] err_uncond(t grid) = " + "  ".join(f"{ERR_UNCOND[t]:.4f}" for t in T_GRID))
print("  [A组] err_sub(t grid) = " + "  ".join(f"{ERR_SUB[t]:.4f}" for t in T_GRID))
print("  [A组] offdiag/diag(t grid) = " + "  ".join(f"{OFFDIAG[t]:.4f}" for t in T_GRID))
print(f"  [A组] 子集鲁棒性最优 t={_T_BEST}  err_sub(0)={ERR_SUB[0.0]:.4f} "
      f"err_sub(1)={ERR_SUB[1.0]:.4f} err_sub(3)={ERR_SUB[3.0]:.4f}")
print(f"  [B组] κ={KAPPA}  r={R_CFG}  σ_c²={SC2:.4f}  w_div={W_DIV}  "
      f"β*={BETA_ANA:.4f}  β_Q={BETA_Q:.3f}  β_emp={BETA_EMP:.2f}")
print("  [B组] β 网格 = " + "  ".join(f"{v}" for v in BETA_GRID))
print("  [B组] α(β) = " + "  ".join(f"{v:.4f}" for v in ALPHA_G))
print("  [B组] Σ_β/σ_c² = " + "  ".join(f"{v:.4f}" for v in DIV_G))
print("  [B组] quality_loss = " + "  ".join(f"{v:.6f}" for v in LOSS_G))
print("  [B组] ‖μ_β−Bx‖ = " + "  ".join(f"{v:.4f}" for v in DIST_G))
print(f"  [B组] β=0 loss={quality_loss(0):.6f}  β=4 loss={quality_loss(4):.6f}  "
      f"网格最优={_best_grid}")
print(f"  [C组] x₂ 三档 = " + "  ".join(f"{v:.1f}" for v in C_X2) + f"  adh₂={ADH2:.4f}")
print("  [C组] jerk = " + "  ".join(f"{v:.4f}" for v in C_JERK))
print("  [C组] 跟踪误差(40seed) = " + "  ".join(f"{v:.4f}" for v in C_TRACK))
print("  [C组] 模型误差 = " + "  ".join(f"{v:.4f}" for v in C_MODELERR))
print("  [C组] ‖a−a稳‖ = " + "  ".join(f"{v:.4f}" for v in C_STEER))
print("  [C组] ‖â−â稳‖ = " + "  ".join(f"{v:.4f}" for v in C_MODELSTEER))
for _nm, _trj in zip(("稳", "快", "失误"), (_A_TRUE[0], _A_TRUE[1], _A_TRUE[2])):
    print(f"  [C组] 轨迹 {_nm} = " + "  ".join(f"{v:.4f}" for v in _trj))
print("  [D组] d 步 = " + "  ".join(f"{v}" for v in D_GRID))
print("  [D组] d 毫秒 = " + "  ".join(f"{v}" for v in D_MS))
print("  [D组] φ^d = " + "  ".join(f"{v:.4f}" for v in D_PHI))
print("  [D组] 分布偏移 shift = " + "  ".join(f"{v:.4f}" for v in D_SHIFT))
print("  [D组] 无条件误差 = " + "  ".join(f"{v:.4f}" for v in D_UNCOND))
print("  [D组] no_rtc 误差 = " + "  ".join(f"{v:.4f}" for v in D_NORTC))
print("  [D组] rtc 误差 = " + "  ".join(f"{v:.4f}" for v in D_RTC))
print("  [D组] s_soft = " + "  ".join(f"{v:.4f}" for v in D_S_SOFT))
print("  [D组] s_hat(MC) = " + "  ".join(f"{v:.4f}" for v in D_S_HAT))
print("  注（反直觉 / 边界，写正文须照此口径）：")
print("   · A 组：缺了 dropout 时模型并不特别忽略某个分量（t=0 五分量遵从度均 0.624），")
print("     真正被忽略的是从没见过的「无条件分支」err_uncond=2.2919；dropout 概率越高该分量")
print(f"     遵从度越低（子目标 0.624→{_ah1[4]:.4f}），但换来了任意子集鲁棒性 U 形最优点 t=1.0")
print(f"     （err_sub {ERR_SUB[0.0]:.4f}→{ERR_SUB[1.0]:.4f}→{ERR_SUB[3.0]:.4f}）与满上下文精度代价")
print(f"     （err_full {ERR_FULL[0.0]:.4f}→{ERR_FULL[3.0]:.4f}）。相对交叉项占比反而随 dropout 上升")
print(f"     （{OFFDIAG[0.0]:.4f}→{OFFDIAG[3.0]:.4f}）：对角增益被压得比非对角更快。")
print(f"   · B 组：解析纯遵从最优 β*={BETA_ANA:.4f}≈1.7，实测 β_emp={BETA_EMP:.2f}；"
      f"多样性惩罚把复合质量最优点左推到 β_Q={BETA_Q:.3f}。")
print(f"     论文网格 loss {LOSS_G[0]:.6f}/{LOSS_G[1]:.6f}/{LOSS_G[2]:.6f} 为倒 U："
      f"β=1.3 欠引导 α={ALPHA_G[0]:.4f}，β=2.2 过引导 α={ALPHA_G[2]:.4f} 且多样性")
print(f"     Σ_β/σ_c² 掉到 {DIV_G[2]:.4f}。β、κ、r、w_div 均为合成世界自设，非论文数值。")
print(f"   · C 组：可引导性最强的「快」档副作用也最大（jerk {C_JERK[1]:.4f} vs 稳 {C_JERK[0]:.4f}，"
      f"模型误差 {C_MODELERR[1]:.4f} vs {C_MODELERR[0]:.4f}）；「标记失误」档不是最差档。")
print(f"     元数据遵从度被 15% dropout 衰减到 adh₂={ADH2:.4f}。")
print(f"   · D 组：条件化会放大陈旧观测的分布偏移 —— d≥{_cross[0]} 步时 no_rtc 误差")
print(f"     {D_NORTC[-1]:.4f} 反而超过无条件基线 {D_UNCOND[0]:.4f}；训练期模拟 0–12 步延迟")
print(f"     后 rtc 误差 {D_RTC[-1]:.4f} 始终不劣于无条件。软衰减信任度 s_soft=φ^d 与")
print("     MC 反算 s_hat 一致。soft_mask 只用于验证「单调衰减/有界」这一共享性质，")
print("     软化的是条件信任度，不是 06 篇的 chunk 内 inpainting 冻结权重。")
print("   · 全篇只验机制：90.9/80.6/85.6/80%、5B/860M、25/30/15/5%、β∈{1.3,1.7,2.2}、")
print("     0–12 步/240 ms 均为引用；合成世界的 B、ρ、λ、κ、r、w_div、φ、x₂ 全为自设。")


# ---- 机器可读块（pi07_figures.py 逐位解析；勿改格式）
def _fmt(v):
    return f"{v:.6f}"


print("\n[MACHINE-BACKFILL-BEGIN]")
for _k in ["T_GRID", "ERR_FULL", "ERR_UNCOND", "ERR_SUB", "OFFDIAG",
           "ADH_T0", "ADH_T1", "P_BASE", "BETA_GRID", "ALPHA_GRID",
           "DIV_GRID", "LOSS_GRID", "DIST_GRID", "BETA_ANA", "BETA_Q",
           "BETA_EMP", "BETA_FINE", "LOSS_FINE", "KAPPA", "R_CFG", "SC2",
           "C_X2", "C_JERK", "C_TRACK", "C_STEER", "C_MODELERR",
           "C_MODELSTEER", "ADH2", "C_TRJ0", "C_TRJ1", "C_TRJ2",
           "D_GRID", "D_MS", "D_PHI", "D_SHIFT",
           "D_UNCOND", "D_NORTC", "D_RTC", "D_S_SOFT", "D_S_HAT"]:
    print("@@" + _k + "=" + ",".join(_fmt(v) for v in BACK[_k]))
print("[MACHINE-BACKFILL-END]")

CK.summary()
