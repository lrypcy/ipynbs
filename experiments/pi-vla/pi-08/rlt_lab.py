# -*- coding: utf-8 -*-
"""08 篇（RLT / RL Token）合成探针：把「冻结 VLA 上挂一个小 actor-critic」拆成可解析的高斯世界。

论文 C：*RL Token: Bootstrapping Online RL with Vision-Language-Action Models*，
arXiv:2604.23073v2（Physical Intelligence，30 Apr 2026）。事实只取 `08_rlt.md` 清单。

**只验机制，不重算论文成绩。** 论文里的 20%→65%、最高 3× 提速、15 min–5 h 真机数据、
400–1000 episodes、1–10 h 示范、50% dropout 等全部是引用，本探针一个都没有重算。

把论文里最容易含混过去的四件事钉死：

  自校验  线性 encoder–decoder 重建损失的手写梯度 ↔ 有限差分；构造真值 ↔ MC 反算
  A 组    RL token bottleneck（§IV-A Eq.1/Eq.2）：bottleneck 维度 ↔ 重建误差 ↔
          下游控制精度的三段关系，及「维度够用就饱和」的拐点（= 内在秩 r）
  B 组    冻结 backbone + 小 critic（§IV-B Eq.3、Alg.1、App.-B）：
          TD3 双 Q 抑制过估计（与单 Q 对照）、冻结特征下 critic 的可学习性、
          chunk-level 回报 vs step-level 回报的信用分配方差
  C 组    参考动作 dropout 50% 与 BC anchor β（§IV-B Eq.4/Eq.5、App.-B）：
          「不给参考还能不能学」「dropout 太低退化成复制 BC」「anchor 太强压住 RL 改进」
  D 组    边界：token 维度过小、回报稀疏、离线数据偏差下的失效模式

世界是**线性高斯**：VLA 最后层 token 序列由低秩任务码 v∈R^r 经冻结线性映射加噪生成；
encoder–decoder 的 L2 重建损失在这个世界里退化成秩-b 的 PCA 投影（Bayes 最优），
有闭式解，因此可扫很多 seed 而不必真的训 transformer。actor 把 VLA 参考块当作
「局部编辑」的起点（Eq.5 的闭式最优 a=(â*+βã)/(1+β)），critic 用双 Q 离策略 TD。

论文常量（引用，不参与合成世界的自由度）：π0.6 / 50 Hz / H=50(=1 s) / d=14 /
C=10 / chunk=140 维 / 稀疏二值 reward / 50% dropout / 2 腕+1 基座相机。
**未给出**：RL token 瓶颈维数、β 具体数值、γ、σ、warmup 步数、α。探针里的
b（bottleneck 维）、β、γ、σ 全部是自设，不能冒充论文数值。

纯 numpy / CPU / 固定 seed / 多 seed 平均 / 每组末尾断言 / 最后打印断言总数。
运行：~/Software/miniconda3/bin/python rlt_lab.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# series_common.py 在 pi-vla 根目录；lab 在 pi-08/ 下，故取上两级目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from series_common import Check, mean_std  # noqa: E402

CK = Check()

# ============================================================ 论文常量（引用，非合成）
H_CHUNK = 50            # VLA 动作块长度，1 s @ 50 Hz（§III / §VI-A）
C_RL = 10               # RL 动作块长度，C < H（§III / App.-B / §VI-C）
D_ACT = 14              # 每步动作维度（§VI-A）
CHUNK_DIM = C_RL * D_ACT   # 140 维 chunked action
CTRL_HZ = 50            # 控制频率（§VI-A / App.-C）
REF_DROPOUT = 0.50      # 参考动作 dropout 50%（App.-B）
N_WRISTCAM = 2          # 腕部相机路数（§VI-A）
N_BASECAM = 1           # 基座相机路数（§VI-A）
# 引用成绩：20%→65%（§I）、up to 3×（Abstract）、15 min–5 h（§VI-B）、400–1000 episodes（§VI-B）
# 论文没有表格，逐任务精确成功率/吞吐只在柱状图里；本探针不引用、不重算。

# ============================================================ 合成世界自由度（自设）
R = 6                                   # 内在秩 / 任务码维度（探针自设）
LAM = np.array([4.0, 3.0, 2.0, 1.2, 0.7, 0.4])   # 信号方差，递减，有明显谱间隙
SIG2 = 0.09                             # 观测噪声方差 σ²（σ=0.3）
SIG = np.sqrt(SIG2)
M_TOK, D_TOK = 12, 5                   # VLA 最后层 token 序列：M×d_z = 12×5
D_EMB = M_TOK * D_TOK                  # 60 维扁平 token 序列
B_GRID = list(range(1, 13))            # bottleneck 维度扫描 1..12
W_CTRL = np.array([3.0, 2.4, 1.8, 1.3, 0.9, 0.6])  # 动作各方向的「重要性」‖B_j‖²
SIG_FEAT = 0.20                        # RL token 对 v 的估计噪声（B 组 critic 用）

N_SEEDS = 16
BASE_SEED = 20261108
SEEDS = [BASE_SEED + 1000 * i for i in range(N_SEEDS)]

# C 组：局部动作编辑 + 参考 dropout 的可解析模型
DELTA = 0.8             # VLA 参考的系统偏置 δ
SIG_V = 0.4             # VLA 参考噪声 σ_v
TAU_Q = np.sqrt(2.0)    # critic 对最优动作的估计噪声（τ_Q²=2.0 > D，故 β=0 最差）
SIG_EXPL = 0.2          # 高斯策略探索噪声 σ_expl（Eq.4 的小固定 std）
N_BUF = 2000            # 训练 buffer（状态通路的有效样本量）
C_CORRUPT = 3.0         # 测试时的参考块污染幅度
P0 = 0.15               # dropout→状态通路「有效训练度」的半饱和常数
D_C = DELTA ** 2 + SIG_V ** 2          # VLA 先验的误差量级 = 0.80
V_C = TAU_Q ** 2                       # critic 噪声量级 = 2.00
BETA_STAR = V_C / D_C                  # 解析最优 β = 2.5
BETAS = [0.0, 0.2, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 50.0, 200.0]
DROPOUTS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]

# B 组
K_ACT = 12              # 候选动作数（双 Q 过估计的标准离散抽象）
TAU_DQ = 1.0            # critic 估计噪声 std
GAMMA = 0.9             # 自设折扣（论文 γ 未给出）
NQ = 4000               # 双 Q 单步 MC seed 数
H_STEPS = 50            # 合成 episode 控制步数（对齐 H=50）
N_STEP = H_STEPS                       # step-level 决策数 = 50
N_CHUNK = H_STEPS // C_RL              # chunk-level 决策数 = 5
P_GOOD_GRID = [0.9, 0.7, 0.5]         # 稀疏二值 reward 的「单决策成功概率」

_rng_world = np.random.default_rng(BASE_SEED)

# 冻结 backbone：正交基 QFULL，前 R 列是信号方向（列范数²=LAM）
QFULL = np.linalg.qr(_rng_world.normal(size=(D_EMB, D_EMB)))[0]
A_MAP = QFULL[:, :R] * np.sqrt(LAM)    # x = A_MAP v + η，D_EMB×R
# Cov(x) 在 QFULL 基下对角：前 R 个特征值 LAM+σ²，其余 σ²
EIG = np.concatenate([LAM + SIG2, np.full(D_EMB - R, SIG2)])
EIG_SORT = np.sort(EIG)[::-1]
EIG_CS = np.cumsum(EIG_SORT)
EIG_TOTAL = EIG_SORT.sum()

# 机器可读回填块（配图逐位解析这里）
BACK: dict[str, np.ndarray] = {}


def store(key, val):
    BACK[key] = np.atleast_1d(np.asarray(val, dtype=float))


def mc(f, seeds):
    """对一组 seed 跑 f(seed)→标量，返回 (均值, 标准差)。"""
    return mean_std([f(s) for s in seeds])


# ===================================================================== 自校验
CK.banner("自校验：世界不变量、构造真值↔反算值、encoder/decoder 重建梯度（手写 vs 有限差分）")

# ---- 1. 世界不变量
CK.ok("Cov(x) 谱：前 r 个特征值 = LAM+σ²，其余 = σ²（构造真值）",
      np.allclose(EIG_SORT[:R], LAM + SIG2)
      and np.allclose(EIG_SORT[R:], SIG2),
      f"前 r {EIG_SORT[:R].round(3).tolist()}，噪声 {EIG_SORT[R]:.4f}，D={D_EMB}")
CK.ok("冻结 backbone 的列范数² 恰为 LAM（信号功率）",
      np.allclose(np.sum(A_MAP ** 2, axis=0), LAM),
      f"Σ‖a_j‖² = {np.sum(A_MAP**2,axis=0).round(2).tolist()}，Σλ={LAM.sum():.1f}")

# ---- 2. 线性 encoder–decoder 重建损失的手写梯度 vs 有限差分
# L = ½‖D E x − z̄‖²，z̄ = sg(x)（stop-gradient：目标当作数据，不回传 backbone）
_b0 = 3
_E0 = _rng_world.normal(0, 0.12, size=(_b0, D_EMB))
_D0 = _rng_world.normal(0, 0.12, size=(D_EMB, _b0))
_x0 = (np.sqrt(LAM) * _rng_world.normal(size=R)) @ QFULL[:, :R].T \
    + SIG * _rng_world.normal(size=D_EMB)
_tgt0 = _x0.copy()        # z̄ = sg(x)


def loss_grad(E, D, x, tgt):
    z = E @ x                  # (b,)
    xhat = D @ z               # (D,)
    r = xhat - tgt
    L = 0.5 * float(r @ r)
    gD = np.outer(r, z)                 # ∂L/∂D
    gz = D.T @ r                        # ∂L/∂z
    gE = np.outer(gz, x)                # ∂L/∂E
    return L, gE, gD


_L0, _gE, _gD = loss_grad(_E0, _D0, _x0, _tgt0)
_dE = _rng_world.normal(size=_E0.shape)
_dD = _rng_world.normal(size=_D0.shape)
_nrm = np.sqrt((_dE ** 2).sum() + (_dD ** 2).sum())
_dE /= _nrm
_dD /= _nrm
_eps = 1e-6
_Lp, _, _ = loss_grad(_E0 + _eps * _dE, _D0 + _eps * _dD, _x0, _tgt0)
_Lm, _, _ = loss_grad(_E0 - _eps * _dE, _D0 - _eps * _dD, _x0, _tgt0)
_num = (_Lp - _Lm) / (2 * _eps)
_ana = float((_gE * _dE).sum() + (_gD * _dD).sum())
_rel_grad = abs(_ana - _num) / max(abs(_num), 1e-12)
CK.ok("encoder/decoder 重建损失的手写梯度 ↔ 有限差分一致",
      _rel_grad < 1e-6, f"相对误差 {_rel_grad:.2e}（b={_b0}, D={D_EMB}）")
store("GRAD_REL", _rel_grad)

# ---- 3. PCA 解析重建误差 ↔ MC 反算（构造真值↔反算值）
def recon_analytic(b):
    """秩-b PCA 投影的最小重建 MSE = 丢弃的协方差特征值之和。"""
    return float(EIG_TOTAL - EIG_CS[b - 1])


def recon_mc(b, seed, n=4000):
    rng = np.random.default_rng(seed)
    Vb = QFULL[:, :b]
    v = rng.normal(size=(n, R))
    x = (v * np.sqrt(LAM)) @ QFULL[:, :R].T + rng.normal(0, SIG, size=(n, D_EMB))
    xhat = (x @ Vb) @ Vb.T
    return float(np.mean(np.sum((x - xhat) ** 2, axis=1)))


RECON = {b: recon_analytic(b) for b in B_GRID}
_recon_chk = {b: mc(lambda s, bb=b: recon_mc(bb, s), SEEDS) for b in (1, 3, 6, 12)}
_recon_rel = max(abs(_recon_chk[b][0] - RECON[b]) / RECON[b] for b in _recon_chk)
CK.ok("PCA 解析重建误差 ↔ MC 反算一致（多 seed 平均）",
      _recon_rel < 0.03,
      "  ".join(f"b={b}: 解析 {RECON[b]:.3f} / MC {_recon_chk[b][0]:.3f}"
                for b in (1, 3, 6, 12)) + f"；最大相对误差 {_recon_rel:.3f}")

# ---- 4. 下游控制误差 解析 ↔ MC 反算
def control_analytic(b):
    """控制误差 = Σ_j w_j·E[(v̂_j−v_j)²]；保留方向为 Wiener MSE σ²/(λ_j+σ²)，丢弃方向为 1。"""
    k = min(b, R)
    kept = np.sum(W_CTRL[:k] * SIG2 / (LAM[:k] + SIG2))
    dropped = np.sum(W_CTRL[k:])
    return float(kept + dropped)


def control_mc(b, seed, n=8000):
    rng = np.random.default_rng(seed)
    k = min(b, R)
    v = rng.normal(size=(n, R))
    x = (v * np.sqrt(LAM)) @ QFULL[:, :R].T + rng.normal(0, SIG, size=(n, D_EMB))
    z = x @ QFULL[:, :b]                       # (n,b) RL token
    o = z[:, :k]                               # 保留的信号方向观测
    vhat = np.zeros((n, R))
    vhat[:, :k] = o * (np.sqrt(LAM[:k]) / (LAM[:k] + SIG2))[None, :]
    err = np.sum(W_CTRL[None, :] * (vhat - v) ** 2, axis=1)
    return float(err.mean())


CTRL = {b: control_analytic(b) for b in B_GRID}
CTRL_ORACLE = control_analytic(D_EMB)          # b=60：拿到全部维度
_ctrl_chk = {b: mc(lambda s, bb=b: control_mc(bb, s), SEEDS) for b in (1, 3, 6, 12)}
_ctrl_rel = max(abs(_ctrl_chk[b][0] - CTRL[b]) / CTRL[b] for b in _ctrl_chk)
CK.ok("下游控制误差 解析 ↔ MC 反算一致（多 seed 平均）",
      _ctrl_rel < 0.04,
      "  ".join(f"b={b}: {CTRL[b]:.4f}/{_ctrl_chk[b][0]:.4f}" for b in (1, 3, 6, 12))
      + f"；最大相对误差 {_ctrl_rel:.3f}")

store("B_GRID", B_GRID)
store("RECON", [RECON[b] for b in B_GRID])
store("CTRL", [CTRL[b] for b in B_GRID])


# ===================================================================== A 组
CK.banner("A 组：RL token bottleneck —— 维度 ↔ 重建误差 ↔ 下游控制精度的三段关系与拐点")

print("\n   b   L_ro 重建误差(解析)   下游控制误差(解析)   边际重建增益")
for b in B_GRID:
    gain = recon_analytic(b - 1) - RECON[b] if b > 1 else EIG_TOTAL - RECON[b]
    print(f"  {b:>2}   {RECON[b]:>16.4f}   {CTRL[b]:>16.4f}   {gain:>14.4f}")
print(f"   b=0（无 RL token）：重建 {EIG_TOTAL:.4f}，控制 {W_CTRL.sum():.4f}")
print(f"   b={D_EMB}（全维 oracle）：控制 {CTRL_ORACLE:.4f}")

_GAIN = np.array([(EIG_TOTAL - RECON[1])] +
                 [RECON[b - 1] - RECON[b] for b in B_GRID[1:]])
KNEE = min(b for b in B_GRID if abs(CTRL[b] - CTRL_ORACLE) < 1e-9)

CK.ok("重建误差 RECON(b) 随 bottleneck 维度单调不增",
      all(RECON[B_GRID[i + 1]] <= RECON[B_GRID[i]] + 1e-9 for i in range(len(B_GRID) - 1)),
      " → ".join(f"{RECON[b]:.3f}" for b in B_GRID))
CK.ok("控制误差 CTRL(b) 随 bottleneck 维度单调不增",
      all(CTRL[B_GRID[i + 1]] <= CTRL[B_GRID[i]] + 1e-9 for i in range(len(B_GRID) - 1)),
      " → ".join(f"{CTRL[b]:.4f}" for b in B_GRID))
CK.ok("第 1 段（b<r）：维度不足，重建与控制误差都高（b=1 控制误差 ≥ 4× 饱和值）",
      CTRL[1] > 4 * CTRL[R] and RECON[1] > 2 * RECON[R],
      f"b=1 控制 {CTRL[1]:.4f} vs 饱和 {CTRL[R]:.4f}（{CTRL[1]/CTRL[R]:.1f}×）；"
      f"重建 {RECON[1]:.3f} vs {RECON[R]:.3f}")
CK.ok("第 2 段（拐点）：控制误差的饱和拐点 = 内在秩 r",
      KNEE == R,
      f"拐点 b={KNEE}，内在秩 r={R}，饱和值 {CTRL_ORACLE:.4f}")
CK.ok("第 3 段（b≥r）：控制精度完全饱和（b=6 与 b=60 全维 oracle 相同）",
      abs(CTRL[R] - CTRL_ORACLE) < 1e-9 and abs(CTRL[12] - CTRL[R]) < 1e-9,
      f"b=6 {CTRL[6]:.4f} = b=12 {CTRL[12]:.4f} = b=60 {CTRL_ORACLE:.4f}")
CK.ok("『维度够用就饱和』：b=r=6 已达全维 oracle 的控制精度，压缩比 60/6=10×",
      abs(CTRL[R] - CTRL_ORACLE) < 1e-9 and D_EMB / R == 10,
      f"b=6 控制 {CTRL[R]:.4f}，b=60 {CTRL_ORACLE:.4f}，D/r={D_EMB/R:.0f}×")
CK.ok("反直觉：b>r 后 L_ro 仍每维降 σ²=0.09（拟合噪声），但控制精度已不再提升",
      (RECON[6] - RECON[12]) > 0.4
      and all(abs(g - SIG2) < 1e-9 for g in _GAIN[R:]),
      f"b=6→12 重建仍降 {RECON[6]-RECON[12]:.3f}（=6σ²），而控制恒 {CTRL[6]:.4f}")

store("RECON_GAIN", _GAIN)
store("CTRL_FLOOR", CTRL_ORACLE)
store("KNEE", KNEE)


# ===================================================================== B 组
CK.banner("B 组：冻结 backbone + 小 critic —— 双 Q 过估计、critic 可学习性、chunk/step 信用分配")

# ---- B1：TD3 双 Q 抑制过估计（单 Q 对照）
def dq_bias(seed):
    rng = np.random.default_rng(seed)
    e1 = rng.normal(0, TAU_DQ, K_ACT)
    e2 = rng.normal(0, TAU_DQ, K_ACT)
    e0 = rng.normal(0, TAU_DQ, K_ACT)
    single = e1.max()                          # 单 Q 贪心：max_k Q̂_k（最大化偏差）
    decoupled = e2[np.argmax(e1)]             # Double-DQN：Q1 选、Q2 评
    ap = np.argmax(e0)                        # 目标策略动作（独立估计选出）
    clipped = min(e1[ap], e2[ap])             # TD3 clipped double Q：min(Q1',Q2')
    return single, decoupled, clipped


_dq = np.array([dq_bias(BASE_SEED + 90000 + i) for i in range(NQ)])
DQ_SINGLE, DQ_DECOUPLED, DQ_CLIPPED = _dq.mean(axis=0)
# E[max of K N(0,1)]：用一次大样本标定解析参考值
_emax = np.random.default_rng(7).normal(size=(200000, K_ACT)).max(axis=1).mean()
_CV_MIN = -TAU_DQ / np.sqrt(np.pi)            # E[min of 2 N(0,1)] = −τ/√π

print("\n   单步过估计偏差（真值 q0=0，K=12，τ=1，4000 seed 平均）")
print(f"     单 Q 贪心 (max)      bias = {DQ_SINGLE:+.4f}   解析参考 τ·E[max_K] = {TAU_DQ*_emax:+.4f}")
print(f"     解耦双 Q (选/评分离) bias = {DQ_DECOUPLED:+.4f}")
print(f"     TD3 clipped (min)    bias = {DQ_CLIPPED:+.4f}   解析参考 −τ/√π = {_CV_MIN:+.4f}")

CK.ok("单 Q 贪心显著高估（maximization bias，bias > 0 且 > 1.2）",
      DQ_SINGLE > 1.2, f"bias={DQ_SINGLE:+.4f}")
CK.ok("单步偏差的构造真值↔MC 反算：单 Q ≈ τ·E[max_K]，clipped ≈ −τ/√π",
      abs(DQ_SINGLE - TAU_DQ * _emax) < 0.06
      and abs(DQ_CLIPPED - _CV_MIN) < 0.06,
      f"单Q {DQ_SINGLE:+.4f} vs {TAU_DQ*_emax:+.4f}；clipped {DQ_CLIPPED:+.4f} vs {_CV_MIN:+.4f}")
CK.ok("双 Q 抑制过估计：|bias(decoupled)| 与 |bias(clipped)| 都远小于单 Q",
      abs(DQ_DECOUPLED) < 0.1 * abs(DQ_SINGLE)
      and abs(DQ_CLIPPED) < 0.5 * abs(DQ_SINGLE),
      f"single {DQ_SINGLE:+.4f} / decoupled {DQ_DECOUPLED:+.4f} / clipped {DQ_CLIPPED:+.4f}")
CK.ok("诚实边界：TD3 的 clipped min 用可控『小幅低估』换掉单 Q 的『大幅高估』",
      DQ_CLIPPED < 0 and abs(DQ_CLIPPED) < abs(DQ_SINGLE),
      f"clipped {DQ_CLIPPED:+.4f}（略低估） vs single {DQ_SINGLE:+.4f}（高估）")

# 迭代 TD：单 Q 的高估会随引导次数复合放大
def td_compound(seed, mode, T=40):
    rng = np.random.default_rng(seed)
    V = 0.0
    for _ in range(T):
        if mode == "single":
            qh = V + rng.normal(0, TAU_DQ, K_ACT)
            target = 1.0 + GAMMA * qh.max()
        else:
            q1 = V + rng.normal(0, TAU_DQ)
            q2 = V + rng.normal(0, TAU_DQ)
            target = 1.0 + GAMMA * min(q1, q2)
        V = 0.5 * V + 0.5 * target
    return V


V_TRUE = 1.0 / (1 - GAMMA)
_td_single, _s1 = mc(lambda s: td_compound(s, "single"), SEEDS)
_td_clipped, _s2 = mc(lambda s: td_compound(s, "clipped"), SEEDS)
print(f"   迭代 TD（40 步，{N_SEEDS} seed）：真值 V*={V_TRUE:.2f}，"
      f"单 Q {_td_single:.2f}（+{_td_single-V_TRUE:.2f}），"
      f"clipped {_td_clipped:.2f}（{_td_clipped-V_TRUE:+.2f}）")
CK.ok("迭代引导下单 Q 高估被复合放大、远超真值",
      _td_single > V_TRUE + 5.0, f"V_single={_td_single:.2f} vs V*={V_TRUE:.2f}")
CK.ok("clipped double Q 把迭代偏差压到远小于单 Q（代价是转入小幅低估）",
      abs(_td_clipped - V_TRUE) < abs(_td_single - V_TRUE) and _td_clipped < V_TRUE,
      f"|clipped−V*|={abs(_td_clipped-V_TRUE):.2f} < |single−V*|={abs(_td_single-V_TRUE):.2f}")

store("DQ_SINGLE", DQ_SINGLE)
store("DQ_DECOUPLED", DQ_DECOUPLED)
store("DQ_CLIPPED", DQ_CLIPPED)
store("TD_VTRUE", V_TRUE)
store("TD_SINGLE", _td_single)
store("TD_CLIPPED", _td_clipped)

# ---- B2：冻结特征下 critic 的可学习性（RL token vs 通用 CV 编码器 / w/o RL token）
def critic_features(v, a, vhat, xi, mode, big=False):
    feat = vhat if mode == "good" else xi       # good=含 RL token；bad=无任务信息的冻结特征
    cols = [np.ones(len(a)), a, a ** 2, feat, a * feat, feat ** 2]
    if big:
        rng = np.random.default_rng(12345)
        extra = rng.normal(size=(len(a), 40))
        cols.append(extra)
    return np.column_stack([np.asarray(c).reshape(len(a), -1) for c in cols])


def q_star(v, a):
    return 1.0 - np.sum((a - v) ** 2, axis=1) / R


def critic_qmse(seed, mode, n_buf=2000, big=False):
    rng = np.random.default_rng(seed)
    n_te = 1200
    n = n_buf + n_te
    v = rng.normal(size=(n, R))
    a = v + rng.normal(0, 0.7, size=(n, R))       # 探索动作散布在 v 附近
    vhat = v + SIG_FEAT * rng.normal(size=(n, R)) # RL token = v 的带噪估计（冻结）
    xi = rng.normal(size=(n, R))                   # 通用冻结特征：与 v 独立（ResNet-10 抽象）
    P = critic_features(v, a, vhat, xi, mode, big)
    q = q_star(v, a)
    Ptr, qtr, Pte, qte = P[:n_buf], q[:n_buf], P[n_buf:], q[n_buf:]
    lam = 1e-3 * np.trace(Ptr.T @ Ptr) / Ptr.shape[1]
    w = np.linalg.solve(Ptr.T @ Ptr + lam * np.eye(Ptr.shape[1]), Ptr.T @ qtr)
    return float(np.mean((Pte @ w - qte) ** 2))


Q_GOOD_SMALL, _ = mc(lambda s: critic_qmse(s, "good", 2000, False), SEEDS)
Q_GOOD_BIG, _ = mc(lambda s: critic_qmse(s, "good", 2000, True), SEEDS)
Q_BAD_SMALL, _ = mc(lambda s: critic_qmse(s, "bad", 2000, False), SEEDS)
Q_BAD_BIG, _ = mc(lambda s: critic_qmse(s, "bad", 2000, True), SEEDS)
Q_GOOD_N50, _ = mc(lambda s: critic_qmse(s, "good", 50, False), SEEDS)

print("\n   小 critic 的 Q-MSE（16 seed 平均；越低越好）")
print(f"     好特征(含 RL token) small={Q_GOOD_SMALL:.4f}  big(多 40 随机特征)={Q_GOOD_BIG:.4f}")
print(f"     坏特征(通用 CV)    small={Q_BAD_SMALL:.4f}  big={Q_BAD_BIG:.4f}")
print(f"     好特征 buffer=50    Q-MSE={Q_GOOD_N50:.4f}（buffer 更大后 {Q_GOOD_SMALL:.4f}）")

CK.ok("冻结特征含任务信息时，小线性 critic 可学到低 Q-MSE（可学习性）",
      Q_GOOD_SMALL < 0.08, f"Q-MSE={Q_GOOD_SMALL:.4f}（backbone 全程冻结、不回传梯度）")
CK.ok("critic 误差随 buffer 增大下降（确实在学习，非固定）",
      Q_GOOD_SMALL < Q_GOOD_N50,
      f"buffer 50 {Q_GOOD_N50:.4f} → 2000 {Q_GOOD_SMALL:.4f}")
CK.ok("冻结 backbone 不是瓶颈：加 critic 容量（big）几乎不掉点",
      abs(Q_GOOD_BIG - Q_GOOD_SMALL) / Q_GOOD_SMALL < 0.20,
      f"small {Q_GOOD_SMALL:.4f} vs big {Q_GOOD_BIG:.4f}")
CK.ok("w/o RL token（换成无任务信息的通用特征）Q-MSE 显著更高，容量也救不回",
      Q_BAD_SMALL > 3 * Q_GOOD_SMALL and Q_BAD_BIG > 3 * Q_GOOD_SMALL,
      f"bad {Q_BAD_SMALL:.4f}/{Q_BAD_BIG:.4f} vs good {Q_GOOD_SMALL:.4f}"
      f"（对应论文去 RL token→ResNet-10 的 throughput 下降 50%）")

store("Q_GOOD_SMALL", Q_GOOD_SMALL)
store("Q_GOOD_BIG", Q_GOOD_BIG)
store("Q_BAD_SMALL", Q_BAD_SMALL)
store("Q_BAD_BIG", Q_BAD_BIG)
store("Q_GOOD_N50", Q_GOOD_N50)

# ---- B3：chunk-level vs step-level 回报的信用分配方差
def pg_var_mc(seed, N, n_ep=24000):
    """REINFORCE 梯度估计量的方差；r_i=a_i，a_i~N(0,1) → 解析 Var=2N²。"""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n_ep, N))
    G = A.sum(axis=1)
    g = G * G                                  # ĝ = G·(Σa_i)/σ² = G² (σ=1)
    return float(np.var(g))


VAR_STEP, _ = mc(lambda s: pg_var_mc(s, N_STEP), SEEDS)
VAR_CHUNK, _ = mc(lambda s: pg_var_mc(s, N_CHUNK), SEEDS)
VAR_RATIO = VAR_STEP / VAR_CHUNK
print(f"\n   信用分配：REINFORCE 梯度方差（决策数 step={N_STEP} / chunk={N_CHUNK}）")
print(f"     step Var={VAR_STEP:.1f}  chunk Var={VAR_CHUNK:.1f}  比值={VAR_RATIO:.1f}"
      f"（解析 2N²：{2*N_STEP**2} / {2*N_CHUNK**2}）")
print(f"     单 episode 决策数 {N_STEP} → {N_CHUNK}（缩短 C={C_RL}×）；"
      f"TD 引导传播次数 {N_STEP} → {N_CHUNK}")

CK.ok("解析构造真值↔MC：REINFORCE 方差 ∝ N²（step 与 chunk 都对齐 2N²）",
      abs(VAR_STEP - 2 * N_STEP ** 2) / (2 * N_STEP ** 2) < 0.08
      and abs(VAR_CHUNK - 2 * N_CHUNK ** 2) / (2 * N_CHUNK ** 2) < 0.08,
      f"step {VAR_STEP:.0f} vs 5000；chunk {VAR_CHUNK:.0f} vs 50")
CK.ok("chunk-level 把信用分配的梯度方差压低一个量级（比值 ≥ C=10）",
      VAR_RATIO >= C_RL, f"比值 {VAR_RATIO:.1f}（C={C_RL}，本模型给 C²=100）")
CK.ok("Eq.(3) 的 chunk 化把单 episode 的决策/引导传播次数缩短 C=10 倍",
      N_STEP // N_CHUNK == C_RL, f"{N_STEP} → {N_CHUNK}")

# chunk 内 C 步奖励求和的「单目标方差」反而更大（诚实边界，不硬说全面更小）
_gseries = np.array([GAMMA ** (2 * k) for k in range(C_RL)]).sum()
print(f"     诚实边界：Σ_{{k<C}}γ^k r 的单目标奖励方差 = 单步的 {_gseries:.2f}×"
      f"（把 C 步噪声加进来）；减小的是决策数/引导传播，不是单个目标的奖励和方差")
CK.ok("诚实：chunk 目标的 C 步奖励和方差高于单步（C 步噪声累加）",
      _gseries > 1.0, f"方差比 {_gseries:.2f}（γ={GAMMA}, C={C_RL}）")

# 稀疏二值 reward：成功需 N 个决策全好
SUCC_STEP, SUCC_CHUNK = {}, {}
for pg in P_GOOD_GRID:
    SUCC_STEP[pg] = pg ** N_STEP
    SUCC_CHUNK[pg] = pg ** N_CHUNK
    cv2 = (1 - SUCC_CHUNK[pg]) / SUCC_CHUNK[pg]
    print(f"     p_good={pg:.1f}: P(succ) step {SUCC_STEP[pg]:.2e} / chunk {SUCC_CHUNK[pg]:.4f}"
          f"；平均 {1/SUCC_CHUNK[pg]:.1f} 个 episode 才见一次成功(chunk)；"
          f"chunk 回报相对方差 CV²={cv2:.1f}")
CK.ok("稀疏二值 reward 下 chunk-level 的成功率远高于 step-level（信用分配可行）",
      all(SUCC_CHUNK[pg] > SUCC_STEP[pg] * 100 for pg in P_GOOD_GRID),
      "  ".join(f"p={pg}: {SUCC_CHUNK[pg]:.3f} vs {SUCC_STEP[pg]:.1e}" for pg in P_GOOD_GRID))
CK.ok("回报越稀疏（p_good↓）相对方差 CV² 越大（二值回报的信息下界）",
      all((1 - SUCC_CHUNK[P_GOOD_GRID[i + 1]]) / SUCC_CHUNK[P_GOOD_GRID[i + 1]]
          > (1 - SUCC_CHUNK[P_GOOD_GRID[i]]) / SUCC_CHUNK[P_GOOD_GRID[i]]
          for i in range(len(P_GOOD_GRID) - 1)),
      "  ".join(f"p={pg}: CV²={(1-SUCC_CHUNK[pg])/SUCC_CHUNK[pg]:.1f}" for pg in P_GOOD_GRID))

store("VAR_STEP", VAR_STEP)
store("VAR_CHUNK", VAR_CHUNK)
store("VAR_RATIO", VAR_RATIO)
store("REWARDSUM_VAR_RATIO", _gseries)
store("P_GOOD", P_GOOD_GRID)
store("SUCC_STEP", [SUCC_STEP[p] for p in P_GOOD_GRID])
store("SUCC_CHUNK", [SUCC_CHUNK[p] for p in P_GOOD_GRID])


# ===================================================================== C 组
CK.banner("C 组：参考动作 dropout 50% 与 BC anchor β（Eq.4/Eq.5）—— 局部动作编辑")

# ---- C1：β 扫描（局部编辑 a = ã + κ(â*−ã)，κ=1/(1+β)，Eq.5 的闭式最优）
def beta_error_analytic(beta):
    k = 1.0 / (1.0 + beta)
    return float((1 - k) ** 2 * D_C + k ** 2 * V_C + SIG_EXPL ** 2)


def beta_error_mc(beta, seed, n=20000):
    rng = np.random.default_rng(seed)
    k = 1.0 / (1.0 + beta)
    v = rng.normal(size=(n, R))
    xi = TAU_Q * rng.normal(size=(n, R))        # critic 对最优动作的噪声
    eta = SIG_V * rng.normal(size=(n, R))       # VLA 参考噪声
    a_ref = v + DELTA + eta                     # ã
    a_hat = v + xi                              # â*
    a = a_ref + k * (a_hat - a_ref) + SIG_EXPL * rng.normal(size=(n, R))
    return float(np.mean((a - v) ** 2))


def beta_success_mc(beta, seed, n=20000):
    rng = np.random.default_rng(seed)
    k = 1.0 / (1.0 + beta)
    v = rng.normal(size=(n, R))
    xi = TAU_Q * rng.normal(size=(n, R))
    eta = SIG_V * rng.normal(size=(n, R))
    a_ref = v + DELTA + eta
    a_hat = v + xi
    a = a_ref + k * (a_hat - a_ref) + SIG_EXPL * rng.normal(size=(n, R))
    return float(np.mean(np.mean((a - v) ** 2, axis=1) < 0.5))


C_ERR, C_ERR_MC, C_SUCC = [], [], []
for beta in BETAS:
    C_ERR.append(beta_error_analytic(beta))
    C_ERR_MC.append(mc(lambda s, bb=beta: beta_error_mc(bb, s), SEEDS)[0])
    C_SUCC.append(mc(lambda s, bb=beta: beta_success_mc(bb, s), SEEDS)[0])

print("\n   β 扫描（κ=1/(1+β)；D=δ²+σ_v²=0.80，V=τ_Q²=2.00，解析最优 β*=V/D=2.5）")
print("     β     动作误差(解析)   动作误差(MC)   成功率(MC)")
for beta, e, em, sc in zip(BETAS, C_ERR, C_ERR_MC, C_SUCC):
    print(f"  {beta:>6.1f}   {e:>14.4f}   {em:>12.4f}   {sc:>9.3f}")

_i_min = int(np.argmin(C_ERR))
CK.ok("β 的解析↔MC 反算一致（局部编辑闭式最优）",
      max(abs(C_ERR[i] - C_ERR_MC[i]) / C_ERR[i] for i in range(len(BETAS))) < 0.04,
      f"最大相对误差 {max(abs(C_ERR[i]-C_ERR_MC[i])/C_ERR[i] for i in range(len(BETAS))):.3f}")
CK.ok("β 与动作误差是倒 U 形：存在内部最优，且解析 β*=V/D=2.5 命中",
      C_ERR[0] > C_ERR[_i_min] and C_ERR[-1] > C_ERR[_i_min]
      and abs(BETAS[_i_min] - BETA_STAR) < 1e-9,
      f"网格最小 β={BETAS[_i_min]}，误差 {C_ERR[_i_min]:.4f}；解析 β*={BETA_STAR}")
CK.ok("β=0（论文 w/o BC Regularizer）在本探针里是最差的单项：比满锚定还差",
      C_ERR[0] > C_ERR[-1] and C_ERR[0] == max(C_ERR),
      f"β=0 {C_ERR[0]:.4f} > β=200 {C_ERR[-1]:.4f} > β*=2.5 {C_ERR[_i_min]:.4f}"
      f"（对应论文『β=0 最大的单项性能下降』）")
CK.ok("anchor 太强会压住 RL 改进：β=50/200 误差显著高于 β*",
      C_ERR[-1] > C_ERR[_i_min] * 1.3 and C_ERR[-2] > C_ERR[_i_min] * 1.2,
      "  ".join(f"β={b}: {e:.4f}" for b, e in zip(BETAS[-3:], C_ERR[-3:]))
      + f"；β*={C_ERR[_i_min]:.4f}")
CK.ok("成功率也呈倒 U 形（峰值在 β* 邻域；阈值化成功率与 MSE 不必逐点同峰）",
      C_SUCC[_i_min] >= max(C_SUCC) - 0.02
      and C_SUCC[0] < C_SUCC[_i_min] and C_SUCC[-1] < C_SUCC[_i_min],
      f"β=0 {C_SUCC[0]:.3f} → 峰值 {max(C_SUCC):.3f}(β={BETAS[int(np.argmax(C_SUCC))]}) "
      f"→ β* {C_SUCC[_i_min]:.3f} → β=200 {C_SUCC[-1]:.3f}")

store("BETAS", BETAS)
store("C_ERR", C_ERR)
store("C_ERR_MC", C_ERR_MC)
store("C_SUCC", C_SUCC)
store("BETA_STAR", BETA_STAR)

# ---- C2：参考动作 dropout（状态通路的有效训练度 s(p)=p/(p+P0)，复制依赖 ρ=1−s）
KAPPA = 1.0 / (1.0 + BETA_STAR)


def s_of_p(p):
    return p / (p + P0)


def rho_of_p(p):
    return 1.0 - s_of_p(p)


def sigma_s2(p):
    return SIG_V ** 2 + 3.0 / (p * N_BUF + 1.0)


def _mse_decompose(bias, c_v, c_eta, c_eps, c_xi, ss2):
    """a−v = bias + c_v·v + c_eta·η + c_eps·ε_s + c_xi·ξ + noise 的 E[(a−v)²]。

    bias（VLA 偏置 δ、污染 c）是确定性常量，其平方必须整项计入，不能拆成
    δ²+c² 而漏掉交叉项 2δc —— 这是上一版解析式与 MC 对不上的原因。
    """
    return (bias ** 2 + c_v ** 2 * 1.0 + c_eta ** 2 * SIG_V ** 2
            + c_eps ** 2 * ss2 + c_xi ** 2 * V_C + SIG_EXPL ** 2)


def dropout_metrics_analytic(p):
    """返回 (masked_err, corrupted_err, normal_err)；与 MC 逐项同构。"""
    s, rho, ss2 = s_of_p(p), rho_of_p(p), sigma_s2(p)
    om = 1 - KAPPA
    g_v = rho + (1 - rho) * s
    # 正常推理：bias=om·δ，v 系数 om(g_v−1)，η/ε_s/ξ 系数 omρ / om(1−ρ) / κ
    normal = _mse_decompose(om * DELTA, om * (g_v - 1), om * rho,
                            om * (1 - rho), KAPPA, ss2)
    # 参考隐藏：bias=om·δ，v 系数 om(s−1)，ε_s 系数 om（状态通路未训练度收缩 v）
    masked = _mse_decompose(om * DELTA, om * (s - 1), 0.0, om, KAPPA, ss2)
    # 参考污染：确定性偏置变成 om·(δ+ρc)，交叉项 2δρc 不可漏
    corrupted = _mse_decompose(om * (DELTA + rho * C_CORRUPT), om * (g_v - 1),
                               om * rho, om * (1 - rho), KAPPA, ss2)
    return masked, corrupted, normal


def dropout_metrics_mc(p, seed, n=20000):
    rng = np.random.default_rng(seed)
    s, rho, ss2 = s_of_p(p), rho_of_p(p), sigma_s2(p)
    om = 1 - KAPPA
    v = rng.normal(size=(n, R))
    xi = TAU_Q * rng.normal(size=(n, R))
    eta = SIG_V * rng.normal(size=(n, R))
    eps_s = np.sqrt(ss2) * rng.normal(size=(n, R))
    noise = SIG_EXPL * rng.normal(size=(n, R))
    a_ref = v + DELTA + eta
    a_hat = v + xi
    a_state = s * v + DELTA + eps_s          # 状态通路（未训练度 (1−s) 收缩 v）
    a_corrupt = a_ref + C_CORRUPT
    # 正常推理
    a_normal = om * (rho * a_ref + (1 - rho) * a_state) + KAPPA * a_hat + noise
    # 参考隐藏
    a_masked = om * a_state + KAPPA * a_hat + noise
    # 参考污染
    a_corr = om * (rho * a_corrupt + (1 - rho) * a_state) + KAPPA * a_hat + noise
    return (float(np.mean((a_masked - v) ** 2)),
            float(np.mean((a_corr - v) ** 2)),
            float(np.mean((a_normal - v) ** 2)))


MASK, CORR, NORM, RHO = [], [], [], []
for p in DROPOUTS:
    ma, co, no = dropout_metrics_analytic(p)
    mc_vals = np.array([dropout_metrics_mc(p, s) for s in SEEDS])
    MASK.append(mc_vals[:, 0].mean())
    CORR.append(mc_vals[:, 1].mean())
    NORM.append(mc_vals[:, 2].mean())
    RHO.append(rho_of_p(p))

print("\n   dropout 扫描（β=β*=2.5；s(p)=p/(p+0.15) 为状态通路有效训练度，ρ=1−s 为复制依赖）")
print("     p     masked(无参考)   corrupted(参考污染)   normal(参考正常)   ρ(复制依赖)")
for p, m, c, n, r in zip(DROPOUTS, MASK, CORR, NORM, RHO):
    print(f"  {p:>5.2f}   {m:>14.4f}   {c:>18.4f}   {n:>16.4f}   {r:>12.4f}")

CK.ok("参考动作 dropout 解析↔MC 一致（状态通路/复制依赖模型）",
      max(abs(dropout_metrics_analytic(p)[i] - [MASK, CORR, NORM][i][j])
          / max(dropout_metrics_analytic(p)[i], 1e-9)
          for j, p in enumerate(DROPOUTS) for i in range(3)) < 0.06,
      "三组指标各 6 档的最大相对误差 < 0.06")
CK.ok("『不给参考动作时还能不能学』：dropout 足够时 masked 误差大幅下降",
      MASK[0] > 2.5 * MASK[3],
      f"p=0 {MASK[0]:.4f} → p=0.5 {MASK[3]:.4f}（{MASK[0]/MASK[3]:.1f}×）")
CK.ok("『dropout 太低退化成复制 BC』：p=0 时参考被污染会被原样复制",
      CORR[0] > 3 * CORR[3],
      f"corrupted p=0 {CORR[0]:.4f} → p=0.5 {CORR[3]:.4f}")
CK.ok("复制依赖 ρ 随 dropout 单调下降（独立状态通路被保住）",
      all(RHO[i + 1] < RHO[i] for i in range(len(RHO) - 1)),
      "  ".join(f"p={p}:{r:.3f}" for p, r in zip(DROPOUTS, RHO)))
CK.ok("诚实边界：50% 已兑现绝大部分收益，但并非严格饱和（再升到 0.9 只再降 2.7%）",
      MASK[0] / MASK[3] > 3.5 and MASK[3] / MASK[-1] < 1.05,
      f"p=0 {MASK[0]:.4f} → 0.5 {MASK[3]:.4f}（降 {MASK[0]/MASK[3]:.1f}×）；"
      f"0.5→0.9 仅 {MASK[3]:.4f}→{MASK[-1]:.4f}（{(1-MASK[-1]/MASK[3])*100:.1f}%）")
CK.ok("dropout 不损害正常推理（参考始终提供）：normal 误差在扫描里基本持平",
      max(NORM) - min(NORM) < 0.05,
      "  ".join(f"p={p}:{n:.4f}" for p, n in zip(DROPOUTS, NORM)))

store("DROPOUTS", DROPOUTS)
store("MASK", MASK)
store("CORR", CORR)
store("NORM", NORM)
store("RHO", RHO)


# ===================================================================== D 组
CK.banner("D 组：边界失效模式 —— token 维度过小 / 回报稀疏 / 离线数据偏差")

# ---- D1：token 维度过小（接 A 组控制误差）
print("\n   token 维度过小：控制误差（越低越好）")
for b in (1, 2, 3, 6, 12):
    print(f"     b={b:>2}：控制误差 {CTRL[b]:.4f}（相对饱和 {CTRL[b]/CTRL_ORACLE:.1f}×）")
CK.ok("b=1（极小 RL token）控制误差是饱和值的 4 倍以上（信息被掐断）",
      CTRL[1] > 4 * CTRL_ORACLE,
      f"b=1 {CTRL[1]:.4f} vs 饱和 {CTRL_ORACLE:.4f}（{CTRL[1]/CTRL_ORACLE:.1f}×）")
CK.ok("b 从 1 增到 r=6 是恢复控制精度的关键，之后不再提升",
      CTRL[1] > CTRL[2] > CTRL[3] > CTRL[6] and abs(CTRL[6] - CTRL[12]) < 1e-9,
      " → ".join(f"{CTRL[b]:.4f}" for b in (1, 2, 3, 6, 12)))

# ---- D2：回报稀疏（接 B3 稀疏二值成功）
print("\n   回报稀疏：chunk-level 成功率与『见到一次成功所需 episode 数』")
for pg in P_GOOD_GRID:
    n_ep = 1.0 / SUCC_CHUNK[pg]
    print(f"     p_good={pg:.1f}：P(succ|chunk)={SUCC_CHUNK[pg]:.4f}，"
          f"平均需 {n_ep:.1f} 个 episode 才见一次成功")
CK.ok("即便 chunk 化，回报过稀疏时学习信号仍会枯竭（p_good=0.5 需 30+ episode 才见一次成功）",
      (1.0 / SUCC_CHUNK[0.5]) > 30,
      f"p_good=0.5：1/P(succ)={1.0/SUCC_CHUNK[0.5]:.1f}（chunk 化也不凭空造信息）")
CK.ok("chunk 化把 step-level 几乎不可学的稀疏任务变得可学",
      SUCC_CHUNK[0.5] > 0.01 and SUCC_STEP[0.5] < 1e-10,
      f"chunk {SUCC_CHUNK[0.5]:.4f} vs step {SUCC_STEP[0.5]:.2e}")

# ---- D3：离线数据偏差（buffer 只有 VLA warmup 数据，动作全在 ã 附近）
def qmse_offline(seed, mode):
    """critic 在『只覆盖 VLA 参考动作』的 buffer 上训练，测其在最优动作区的 Q 误差。"""
    rng = np.random.default_rng(seed)
    n_buf, n_te = 2000, 1200
    n = n_buf + n_te
    v = rng.normal(size=(n, R))
    vhat = v + SIG_FEAT * rng.normal(size=(n, R))
    xi = rng.normal(size=(n, R))
    if mode == "vla_only":
        # 离线 warmup：动作全在带偏置的 VLA 参考 ã=v+δ 附近，从不靠近最优 a*=v
        a_buf = v[:n_buf] + DELTA + 0.05 * rng.normal(size=(n_buf, R))
    else:
        # 加入 on-policy / 人类干预：一半样本覆盖到 a* 邻域
        a_buf = v[:n_buf] + DELTA + 0.05 * rng.normal(size=(n_buf, R))
        mix = rng.random(n_buf) < 0.5
        a_buf[mix] = v[:n_buf][mix] + 0.2 * rng.normal(size=(int(mix.sum()), R))
    a_te = v[n_buf:] + 0.2 * rng.normal(size=(n_te, R))   # 测试：最优动作区
    P_buf = critic_features(v[:n_buf], a_buf, vhat[:n_buf], xi[:n_buf], "good")
    P_te = critic_features(v[n_buf:], a_te, vhat[n_buf:], xi[n_buf:], "good")
    q_buf = q_star(v[:n_buf], a_buf)
    q_te = q_star(v[n_buf:], a_te)
    lam = 1e-3 * np.trace(P_buf.T @ P_buf) / P_buf.shape[1]
    w = np.linalg.solve(P_buf.T @ P_buf + lam * np.eye(P_buf.shape[1]), P_buf.T @ q_buf)
    return float(np.mean((P_te @ w - q_te) ** 2))


Q_VLA_ONLY, _ = mc(lambda s: qmse_offline(s, "vla_only"), SEEDS)
Q_WITH_ONPOLICY, _ = mc(lambda s: qmse_offline(s, "onpolicy"), SEEDS)
print(f"\n   离线数据偏差：最优动作区的 Q-MSE（16 seed 平均）")
print(f"     仅 VLA warmup 数据   {Q_VLA_ONLY:.4f}（动作只覆盖 ã，最优区外推误差大）")
print(f"     + on-policy/干预数据  {Q_WITH_ONPOLICY:.4f}（覆盖 a* 邻域后误差下降）")
CK.ok("只用 VLA warmup 离线数据：最优动作区 Q 误差大（分布偏差/外推）",
      Q_VLA_ONLY > 0.10, f"Q-MSE={Q_VLA_ONLY:.4f}")
CK.ok("补上 on-policy / 人类干预数据后，最优动作区 Q 误差显著下降",
      Q_WITH_ONPOLICY < 0.5 * Q_VLA_ONLY,
      f"{Q_VLA_ONLY:.4f} → {Q_WITH_ONPOLICY:.4f}（对应论文 §V 干预替换 buffer 中 VLA 参考）")

store("Q_VLA_ONLY", Q_VLA_ONLY)
store("Q_WITH_ONPOLICY", Q_WITH_ONPOLICY)
store("CTRL_B1", CTRL[1])
store("CTRL_B2", CTRL[2])
store("CTRL_B3", CTRL[3])
store("CTRL_B6", CTRL[6])
store("CTRL_B12", CTRL[12])


# ===================================================================== 论文常量（引用）
print("\n" + "=" * 74)
print("论文常量（引用 · 非合成探针数字 · 不得与合成读数混用）")
print("=" * 74)
print(f"  基座 π0.6 | 50 Hz | H={H_CHUNK}(=1 s) | d={D_ACT} | C={C_RL} | "
      f"chunk={CHUNK_DIM} 维 | C<H | {N_WRISTCAM} 腕相机 + {N_BASECAM} 基座相机")
print("  稀疏二值 reward（r_T=1/0）| 参考动作 dropout 50%（训练置零/推理始终给）|")
print("  去 RL token→ResNet-10 吞吐降 50% | w/o Chunk C=1 学不动 | β=0 最大单项下降")
print("  【成绩引用】20%→65%（§I）| up to 3× 提速（Abstract）| 15 min–5 h 真机数据（§VI-B）|")
print("              400–1000 episodes（§VI-B）| 1–10 h 示范 / 2000–10000 steps（§VI-B/App.-B）")
print("  【论文未给出】RL token 瓶颈维数、β 具体数值、γ、σ、warmup 步数、α ——")
print("               探针里的 b、β=2.5、γ=0.9、σ=0.2 全部自设，不能冒充论文数值")


# ===================================================================== 正文回填表
CK.banner("正文回填表（以下数字与上面逐条对齐，均为原始读数；博客合成数字须能在此 grep 到）")
print(f"  [世界] r={R}  D={D_EMB}  LAM={LAM.tolist()}  σ²={SIG2}  Σλ={LAM.sum():.1f}  "
      f"噪声下界(D−r)σ²={(D_EMB-R)*SIG2:.2f}")
print(f"  [自校验] 手写梯度↔有限差分相对误差={_rel_grad:.2e}；"
      f"PCA解析↔MC最大相对误差={_recon_rel:.3f}；控制解析↔MC={_ctrl_rel:.3f}")
print(f"  [A组] b=1..12 重建 RECON = " + "  ".join(f"{RECON[b]:.4f}" for b in B_GRID))
print(f"  [A组] b=1..12 控制 CTRL  = " + "  ".join(f"{CTRL[b]:.4f}" for b in B_GRID))
print(f"  [A组] 边际重建增益       = " + "  ".join(f"{g:.4f}" for g in _GAIN))
print(f"  [A组] 拐点 b={KNEE}（=内在秩 r={R}）饱和控制 CTRL_floor={CTRL_ORACLE:.4f} "
      f"(b=60 oracle 同值)；b=6→12 重建仍降 {RECON[6]-RECON[12]:.4f}=6σ²")
print(f"  [B组] 双Q单步偏差 single={DQ_SINGLE:+.4f} decoupled={DQ_DECOUPLED:+.4f} "
      f"clipped={DQ_CLIPPED:+.4f}；解析 τE[max]={TAU_DQ*_emax:+.4f} −τ/√π={_CV_MIN:+.4f}")
print(f"  [B组] 迭代TD V*={V_TRUE:.2f} single={_td_single:.2f} clipped={_td_clipped:.2f}")
print(f"  [B组] 小critic Q-MSE good_small={Q_GOOD_SMALL:.4f} good_big={Q_GOOD_BIG:.4f} "
      f"bad_small={Q_BAD_SMALL:.4f} bad_big={Q_BAD_BIG:.4f} good_n50={Q_GOOD_N50:.4f}")
print(f"  [B组] 信用分配 Var_step={VAR_STEP:.1f} Var_chunk={VAR_CHUNK:.1f} "
      f"比值={VAR_RATIO:.1f}（解析 2N²=5000/50）；决策 {N_STEP}→{N_CHUNK}；"
      f"C步奖励和方差比={_gseries:.2f}×")
print("  [B组] 稀疏二值 P(succ) step = " + "  ".join(
    f"p={pg}:{SUCC_STEP[pg]:.2e}" for pg in P_GOOD_GRID))
print("  [B组] 稀疏二值 P(succ) chunk = " + "  ".join(
    f"p={pg}:{SUCC_CHUNK[pg]:.4f}" for pg in P_GOOD_GRID))
print(f"  [C组] β*=V/D={BETA_STAR}（V={V_C} D={D_C}）")
print("  [C组] β 扫描 动作误差(解析) = " + "  ".join(
    f"β={b}:{e:.4f}" for b, e in zip(BETAS, C_ERR)))
print("  [C组] β 扫描 动作误差(MC)   = " + "  ".join(
    f"β={b}:{e:.4f}" for b, e in zip(BETAS, C_ERR_MC)))
print("  [C组] β 扫描 成功率(MC)     = " + "  ".join(
    f"β={b}:{s:.3f}" for b, s in zip(BETAS, C_SUCC)))
print("  [C组] dropout 扫描 p          = " + "  ".join(f"{p:.2f}" for p in DROPOUTS))
print("  [C组] masked(无参考) 误差     = " + "  ".join(f"{m:.4f}" for m in MASK))
print("  [C组] corrupted(参考污染)误差 = " + "  ".join(f"{c:.4f}" for c in CORR))
print("  [C组] normal(参考正常) 误差   = " + "  ".join(f"{n:.4f}" for n in NORM))
print("  [C组] 复制依赖 ρ             = " + "  ".join(f"{r:.4f}" for r in RHO))
print(f"  [D组] token过小 控制误差 b1={CTRL[1]:.4f} b2={CTRL[2]:.4f} "
      f"b3={CTRL[3]:.4f} b6={CTRL[6]:.4f} b12={CTRL[12]:.4f}")
print(f"  [D组] 稀疏 p_good=0.5 见到一次成功需 {1.0/SUCC_CHUNK[0.5]:.1f} episode(chunk)")
print(f"  [D组] 离线偏差 最优区 Q-MSE vla_only={Q_VLA_ONLY:.4f} "
      f"with_onpolicy={Q_WITH_ONPOLICY:.4f}")
print("  注（反直觉/边界，写正文须照此口径）：")
print("   · b≥r 后 L_ro 仍因拟合噪声每维降 σ²=0.09，但下游控制精度已完全饱和——")
print("     别拿重建损失继续往上堆 RL token 维度。")
print("   · TD3 的 clipped 双 Q 是『小幅低估』换掉单 Q 的『大幅高估』，不是无偏。")
print("   · chunk 级单目标的 C 步奖励和方差比单步大 4.62×；chunk 化减小的是决策数/引导传播")
print("     次数（信用分配），不是单个 TD 目标内的噪声。")
print("   · β=0 在本探针里比满锚定 β=200 还差（critic 噪声 τ_Q²>D），印证论文『最大单项下降』；")
print("     但 β≫β* 也会把动作粘在带偏置的 VLA 参考上、压住 RL 改进，是倒 U 形而非单调。")
print("   · 阈值化成功率与 MSE 不必逐点同峰：本探针成功率峰值在 β=2（0.407），")
print("     β*=2.5 为 0.403，二者差 0.004；倒 U 形结论不变。")
print("   · 参考动作 dropout 的 50% 并非严格饱和：p=0.5 的 masked 误差 0.6405，")
print("     升到 0.9 仍再降 2.7%（0.6231）；50% 兑现的是绝大部分收益，不是全部。")
print("   · 回报过稀疏时 chunk 化也不凭空造信息：p_good=0.5 仍需 32 个 episode 才见一次成功。")

# ---- 机器可读块（rlt_figures.py 逐位解析；勿改格式）
def _fmt(v):
    # 极小值用科学记数，避免 6 位小数被压成 0.000000（log 轴画不出来）
    return f"{v:.3e}" if (v != 0.0 and abs(v) < 1e-4) else f"{v:.6f}"


print("\n[MACHINE-BACKFILL-BEGIN]")
for _k in ["B_GRID", "RECON", "CTRL", "RECON_GAIN", "KNEE", "CTRL_FLOOR",
           "DQ_SINGLE", "DQ_DECOUPLED", "DQ_CLIPPED", "TD_VTRUE", "TD_SINGLE", "TD_CLIPPED",
           "Q_GOOD_SMALL", "Q_GOOD_BIG", "Q_BAD_SMALL", "Q_BAD_BIG", "Q_GOOD_N50",
           "VAR_STEP", "VAR_CHUNK", "VAR_RATIO", "REWARDSUM_VAR_RATIO",
           "P_GOOD", "SUCC_STEP", "SUCC_CHUNK",
           "BETAS", "C_ERR", "C_ERR_MC", "C_SUCC", "BETA_STAR",
           "DROPOUTS", "MASK", "CORR", "NORM", "RHO",
           "Q_VLA_ONLY", "Q_WITH_ONPOLICY",
           "CTRL_B1", "CTRL_B2", "CTRL_B3", "CTRL_B6", "CTRL_B12",
           "GRAD_REL"]:
    print("@@" + _k + "=" + ",".join(_fmt(v) for v in BACK[_k]))
print("[MACHINE-BACKFILL-END]")

CK.summary()
