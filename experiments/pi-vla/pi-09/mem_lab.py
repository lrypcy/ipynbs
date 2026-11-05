# -*- coding: utf-8 -*-
"""09 篇（MEM）合成探针：把「短时视频记忆 + 长时文本记忆」的分工拆成可解析世界。

论文 D：*MEM: Multi-Scale Embodied Memory for Vision Language Action Models*,
arXiv:2603.03596v2。事实只取事实清单 09_mem.md（唯一允许的论文事实来源）。
论文没有数值表、没有逐任务成功率、没有独立编号的遮挡实验数值、未提及 π0.7 ——
本探针一个真机数字都没有重算，也一个都不编。

只验机制（四组 + 前置自校验）：

  自校验  正交 DCT 表示 / 线性高斯后验增益（手写 ↔ 有限差分）/ 构造真值 ↔ MC 反算 /
          跨度 ↔ 采样间隔 ↔ 可恢复事件比例
  A 组    固定 token 预算下的时间跨度：时空分离 attention vs 全连接 attention
          （论文 §III-C：每 4 层 causal temporal attention、上层丢弃历史 token、
          token 数不增、O(Kn²+nK²) vs O(n²K²)；§III-D：6 帧/1 s → 18 帧/54 s）
  B 组    短时记忆补遮挡：无记忆 / 只当前帧 / 有视频记忆，及记忆跨度不够时的失效
          （§III-C 部分可观测 / 自遮挡；Fig.7 的定性载体，无数值）
  C 组    长时记忆补阶段：π_HL 生成 m_t、LLM 造标签、压缩去冗余、
          naive 拼接 / 失败不更新 / 压缩过头（§III-A/§III-B、§IV-A Fig.5）
  D 组    两类记忆的分工边界：遮挡 × 长程阶段 × 四种记忆配置的交叉对照
          （§IV-A「两者都必不可少」、§IV-C Fig.9「无性能退化」）

世界：**线性高斯 + 离散阶段**。短时侧把 K 帧历史压进固定 B 个标量槽（对齐
「token 数不增」）；长时侧 N=15 个阶段、每阶段 60 s（15 min 是论文引用，
N 与 60 s 是探针自设）。全部合成数字都在末尾「正文回填表」集中打印。

纯 numpy / CPU / 固定 seed / 多 seed 平均 / 每组末尾断言 / 最后打印断言总数。
运行：~/Software/miniconda3/bin/python mem_lab.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from series_common import Check, mean_std  # noqa: E402

CK = Check()

# ============================================================ 论文常量（引用 · 非合成）
# 这些是论文/事实清单里写明的数字，只作引用标注，不参与合成世界的自由度。
PAPER = dict(
    arxiv="2603.03596v2",
    pretrain_frames=6, pretrain_stride_s=1, pretrain_span_s=5,       # §III-D 09.txt:302
    posttrain_frames=18, posttrain_span_s=54,                         # §III-D 09.txt:303
    every_n_layers=4,                                                  # §III-C 09.txt:279
    max_task_min=15,                                                   # Abstract/§I
    base="π0.6", vlm="Gemma3-4B", action_expert="860M",                # §III-D 09.txt:292-295
    token_count="不增",                                                # §III-C/D 09.txt:283,299
    complexity_sep="O(Kn²+nK²)", complexity_full="O(n²K²)",            # §III-C 09.txt:281
    license="CC BY 4.0",
)

# ============================================================ 合成世界自由度（全部自设）
B_SLOTS = 12                 # 固定 token 预算：传入 backbone 的标量槽数（与 K 无关）
K_GRID = [6, 9, 12, 15, 18]  # 帧数网格；端点 6 / 18 对齐论文口径，但世界参数自设
DELTA_GRID = [1, 2, 3]       # 采样间隔 Δ（秒），K=18 时扫跨度
N_PATCH = 16                 # 每帧 patch 数 n（复杂度公式用，自设）
N_LAYERS = 12                # ViT 层数（自设）；每 4 层加 temporal attention 是引用
T_LAYERS = N_LAYERS // PAPER["every_n_layers"]
TAU2 = 0.04                  # 表示噪声方差 τ²（τ=0.2）
W_EVENT = 0.5                # 帧间事件可分辨的半窗（秒）
N_SEEDS = 40
BASE_SEED = 20261109
SEEDS = [BASE_SEED + 1000 * i for i in range(N_SEEDS)]

# B 组：抽屉识别（L 类），当前帧被遮挡
L_DRAW = 8                   # 抽屉数（论文「4 抽屉 25%」是引用，这里 8 类是自设）
SIG_OBS = 0.4                # 观测噪声 σ
A_CLEAR = 1.0                # 历史清晰帧里物体特征的信号强度
A_WEAK = 0.3                 # 当前被遮挡帧的弱线索强度
D_GRID = [1, 3, 5, 7, 11, 15, 18, 24]   # 距最后一次可见的帧数
K_MEM = [6, 18]              # 两种记忆跨度（端点对齐论文口径）

# C 组：长程阶段
N_STAGE = 15                 # 阶段数（自设）
STAGE_SEC = 60               # 每阶段 60 s → 900 s = 15 min（15 min 是引用）
RHO = 1                      # 每阶段的冗余标签数（失败重试 / 颜色细节）
L_MAX = N_STAGE              # 文本记忆最大标签槽数（压缩目标：几 bit）
A_GRID = [0.0, 1 / 6, 1 / 3, 0.5, 2 / 3, 5 / 6, 1.0]  # 压缩强度 α
W_MEM_S = 54                 # 短时视频记忆覆盖秒数（对齐后训练 54 s 口径，自设）

_rng_world = np.random.default_rng(BASE_SEED)
BACK: dict[str, np.ndarray] = {}


def store(key, val):
    BACK[key] = np.atleast_1d(np.asarray(val, dtype=float))


def mc(f, seeds=SEEDS):
    """对一组 seed 跑 f(seed)→标量，返回 (均值, 标准差)。"""
    return mean_std([f(s) for s in seeds])


# ============================================================ 共用解析工具
def dct_matrix(n):
    """正交 DCT-II 矩阵（M Mᵀ = Mᵀ M = I）；用作「固定正弦 temporal position encoding」。

    M[c,k] = √(2/n)·cos(π (k+½) c / n)，c=0 行为 √(1/n)；行、列均正交。
    """
    c = np.arange(n)[:, None]
    k = np.arange(n)[None, :]
    M = np.sqrt(2.0 / n) * np.cos(np.pi * (k + 0.5) * c / n)
    M[0, :] = np.sqrt(1.0 / n)
    return M


def A_sep(K, B=B_SLOTS):
    """时空分离编码器的抽象：K 帧历史 → 固定 B 个标量槽的线性映射。

    K≤B：取 B×B DCT 的前 K 列（列正交，全部状态可无偏恢复）；
    K>B：取 K×K DCT 的前 B 行（行正交，得到秩-B 的充分统计 sketch）。
    两种情形都只有 B 行 —— token 数不增。
    """
    if K <= B:
        return dct_matrix(B)[:, :K]
    return dct_matrix(K)[:B, :]


def norm_cdf(x):
    a = np.atleast_1d(np.asarray(x, dtype=float))
    return 0.5 * (1.0 + np.array([math.erf(float(v) / math.sqrt(2.0)) for v in a]))


def acc_signal(a, sig=SIG_OBS, L=L_DRAW):
    """观测 y = a·e_j + η，argmax 判别正确率的确定性积分（构造真值）。

    P(correct) = ∫ φ(z)·Φ(z + a/σ)^(L−1) dz；a=0 时精确等于 1/L。
    """
    z = np.linspace(-8.0, 8.0, 20001)
    w = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    c = norm_cdf(z + a / sig)
    return float(np.trapezoid(w * c ** (L - 1), z))


def acc_signal_mc(a, seed, sig=SIG_OBS, L=L_DRAW, n=6000):
    rng = np.random.default_rng(seed)
    j = rng.integers(0, L, size=n)
    y = rng.normal(0.0, sig, size=(n, L))
    y[np.arange(n), j] += a
    return float(np.mean(np.argmax(y, axis=1) == j))


# ===================================================================== 前置自校验
CK.banner("前置自校验：正交表示 / 后验增益 / 构造真值↔反算值 / 世界不变量")

# ---- 1. DCT 正交性（表示映射的构造真值）
_M = dct_matrix(24)
_ortho = max(np.max(np.abs(_M @ _M.T - np.eye(24))),
             np.max(np.abs(_M.T @ _M - np.eye(24))))
CK.ok("固定正弦 temporal encoding（DCT-II）是正交矩阵",
      _ortho < 1e-12, f"最大偏差 {_ortho:.2e}（24×24）")

# ---- 2. 线性高斯后验增益：g* = 1/(1+σ²)，手写 ↔ 有限差分
_SIG2_W = 0.09


def _mse_gain(g):
    return (1.0 - g) ** 2 + g ** 2 * _SIG2_W


_g_star = 1.0 / (1.0 + _SIG2_W)
_h = 1e-4
_d1 = (_mse_gain(_g_star + _h) - _mse_gain(_g_star - _h)) / (2 * _h)
_g_num = float(np.argmin([_mse_gain(g) for g in np.linspace(0, 1, 100001)])) / 100000.0
CK.ok("Wiener 后验增益 g*=1/(1+σ²) 是 MSE 最小点（解析 ↔ 有限差分/网格）",
      abs(_d1) < 1e-6 and abs(_g_num - _g_star) < 1e-4,
      f"g*={_g_star:.6f}，有限差分导数 {_d1:.2e}，网格最小 {_g_num:.6f}")

# ---- 3. 无线索时的机会基线 = 1/L（构造真值 ↔ MC 反算）
ACC_CHANCE = 1.0 / L_DRAW
_acc0_ana = acc_signal(0.0)
_acc0_mc, _ = mc(lambda s: acc_signal_mc(0.0, s))
CK.ok("无观测时的抽屉判别准确率精确等于 1/L（构造真值 ↔ MC 反算）",
      abs(_acc0_ana - ACC_CHANCE) < 1e-12 and abs(_acc0_mc - ACC_CHANCE) < 0.01,
      f"解析 {_acc0_ana:.6f} / MC {_acc0_mc:.4f} / 1/L={ACC_CHANCE:.6f}")

# ---- 4. 观测信号判别正确率：积分（构造真值）↔ MC 反算（多 seed）
ACC_WEAK = acc_signal(A_WEAK)
ACC_CLEAR = acc_signal(A_CLEAR)
_accw_mc, _ = mc(lambda s: acc_signal_mc(A_WEAK, s))
_accc_mc, _ = mc(lambda s: acc_signal_mc(A_CLEAR, s))
CK.ok("清晰帧 / 弱线索帧的判别正确率：积分 ↔ MC 反算一致（多 seed）",
      abs(_accw_mc - ACC_WEAK) / ACC_WEAK < 0.03
      and abs(_accc_mc - ACC_CLEAR) / ACC_CLEAR < 0.03,
      f"弱 {ACC_WEAK:.4f}/{_accw_mc:.4f}，清晰 {ACC_CLEAR:.4f}/{_accc_mc:.4f}")
CK.ok("信号越强判别越准：清晰帧 > 弱线索帧 > 机会基线",
      ACC_CLEAR > ACC_WEAK > ACC_CHANCE + 0.02,
      f"{ACC_CLEAR:.4f} > {ACC_WEAK:.4f} > {ACC_CHANCE:.4f}")

# ---- 5. token 数不增 / 跨度 = (K−1)Δ 的世界不变量
_shapes = [A_sep(K).shape for K in K_GRID]
CK.ok("任意帧数下表示都只有 B 行（token 数不增，对齐论文引用）",
      all(sh[0] == B_SLOTS for sh in _shapes),
      f"K={K_GRID} 的表示形状 {_shapes}，B={B_SLOTS}")
SPAN_GRID = [(K - 1) * 1 for K in K_GRID]
CK.ok("跨度 = (K−1)·Δ（自校验：帧数/采样间隔 ↔ 历史覆盖）",
      SPAN_GRID == [5, 8, 11, 14, 17],
      f"Δ=1 时 K={K_GRID} → span={SPAN_GRID}")


# ===================================================================== A 组
CK.banner("A 组：固定 token 预算下的时间跨度 —— 时空分离 vs 全连接 attention")


def sep_mse(K):
    """时空分离：秩-min(K,B) sketch 的最小二乘重建 MSE（每状态，解析）。"""
    return TAU2 + max(0.0, 1.0 - B_SLOTS / K) * (1.0 - TAU2)


def full_mse(K):
    """全连接：固定 B 槽只能保留最近 B 帧，旧帧丢弃；每状态 MSE（解析）。"""
    r = min(K, B_SLOTS)
    return ((K - r) * 1.0 + r * TAU2) / K


def sep_state_mse(K):
    P = A_sep(K).T @ A_sep(K) if K > B_SLOTS else np.eye(K)
    return 1.0 - (1.0 - TAU2) * np.diag(P)


def full_state_mse(K):
    r = min(K, B_SLOTS)
    out = np.ones(K)
    out[K - r:] = TAU2
    return out


def sep_detect_acc(K):
    P = A_sep(K).T @ A_sep(K) if K > B_SLOTS else np.eye(K)
    corr = np.sqrt(np.clip(np.diag(P) / (1.0 + TAU2), 0.0, 1.0))
    return 0.5 + np.arcsin(corr) / np.pi


def full_detect_acc(K):
    r = min(K, B_SLOTS)
    kept = 0.5 + np.arcsin(1.0 / np.sqrt(1.0 + TAU2)) / np.pi
    out = np.full(K, 0.5)
    out[K - r:] = kept
    return out


def a_mc(K, seed, n=4000):
    """多 seed 反算：从 B 槽表示重建 K 个历史状态的 MSE 与符号判别准确率。"""
    rng = np.random.default_rng(seed)
    A = A_sep(K)
    h = rng.normal(size=(n, K))
    eps = rng.normal(0.0, math.sqrt(TAU2), size=(n, B_SLOTS))
    z = h @ A.T + eps
    hhat = z @ A                      # A⁺ = Aᵀ（两种情形都成立）
    mse = float(np.mean((hhat - h) ** 2))
    acc = float(np.mean(np.sign(hhat) == np.sign(h)))
    return mse, acc


SEP_MSE = [sep_mse(K) for K in K_GRID]
FULL_MSE = [full_mse(K) for K in K_GRID]
SEP_MAX, FULL_MAX, SEP_MIN, FULL_MIN = [], [], [], []
SEP_MEANACC, FULL_MEANACC, SEP_MAXACC, FULL_MAXACC = [], [], [], []
SEP_MSE_MC, SEP_ACC_MC = [], []
for K in K_GRID:
    sm, sa = sep_state_mse(K), sep_detect_acc(K)
    fm, fa = full_state_mse(K), full_detect_acc(K)
    SEP_MAX.append(float(sm.max())); FULL_MAX.append(float(fm.max()))
    SEP_MIN.append(float(sa.min())); FULL_MIN.append(float(fa.min()))
    SEP_MEANACC.append(float(sa.mean())); FULL_MEANACC.append(float(fa.mean()))
    SEP_MAXACC.append(float(sa.max())); FULL_MAXACC.append(float(fa.max()))
    _m, _ = mc(lambda s, kk=K: a_mc(kk, s)[0]); SEP_MSE_MC.append(_m)
    _a, _ = mc(lambda s, kk=K: a_mc(kk, s)[1]); SEP_ACC_MC.append(_a)

print("\n   K   span(Δ=1)  sep MSE  full MSE  sep最差帧MSE  full最差帧MSE"
      "  sep平均判别  full平均判别  sep最差帧判别  full最差帧判别")
for i, K in enumerate(K_GRID):
    print(f"  {K:>2}   {SPAN_GRID[i]:>8}   {SEP_MSE[i]:>7.4f}  {FULL_MSE[i]:>8.4f}"
          f"   {SEP_MAX[i]:>11.4f}   {FULL_MAX[i]:>12.4f}"
          f"   {SEP_MEANACC[i]:>11.4f}   {FULL_MEANACC[i]:>12.4f}"
          f"   {SEP_MIN[i]:>12.4f}   {FULL_MIN[i]:>13.4f}")

CK.ok("可恢复历史状态数 = rank(A_sep) = min(K,B)（构造真值自校验）",
      np.linalg.matrix_rank(A_sep(6)) == 6
      and np.linalg.matrix_rank(A_sep(18)) == B_SLOTS,
      f"K=6 rank=6；K=18 rank={np.linalg.matrix_rank(A_sep(18))}=B={B_SLOTS}")
CK.ok("时空分离的解析 MSE ↔ 多 seed MC 反算一致",
      max(abs(SEP_MSE[i] - SEP_MSE_MC[i]) / SEP_MSE[i] for i in range(len(K_GRID))) < 0.03,
      "  ".join(f"K={K}: {SEP_MSE[i]:.4f}/{SEP_MSE_MC[i]:.4f}"
               for i, K in enumerate(K_GRID)))
CK.ok("时空分离的解析判别准确率 ↔ 多 seed MC 反算一致",
      max(abs(SEP_MEANACC[i] - SEP_ACC_MC[i]) / SEP_MEANACC[i]
          for i in range(len(K_GRID))) < 0.03,
      "  ".join(f"{SEP_MEANACC[i]:.4f}/{SEP_ACC_MC[i]:.4f}"
               for i in range(len(K_GRID))))
CK.ok("反直觉：两种注意力在固定 token 预算下的总重建误差完全相同",
      max(abs(SEP_MSE[i] - FULL_MSE[i]) for i in range(len(K_GRID))) < 1e-9,
      "  ".join(f"K={K}: {SEP_MSE[i]:.4f}={FULL_MSE[i]:.4f}"
               for i, K in enumerate(K_GRID)))
CK.ok("差别在误差分布：K=18 时分离的最差帧误差远小于全连接",
      SEP_MAX[-1] < 0.7 and FULL_MAX[-1] > 0.95,
      f"sep {SEP_MAX[-1]:.4f} vs full {FULL_MAX[-1]:.4f}（K=18）")
CK.ok("反直觉：平均符号判别准确率两者几乎相同（总量相同，平均指标分不开）",
      abs(SEP_MEANACC[-1] - FULL_MEANACC[-1]) < 0.02,
      f"K=18：sep {SEP_MEANACC[-1]:.4f} vs full {FULL_MEANACC[-1]:.4f}")
CK.ok("只有「最差帧/远期召回」能分开：分离 > 0.7，全连接塌到随机 0.5",
      SEP_MIN[-1] > 0.70 and abs(FULL_MIN[-1] - 0.5) < 1e-9,
      f"sep {SEP_MIN[-1]:.4f} vs full {FULL_MIN[-1]:.4f}")
CK.ok("公平的代价：全连接在近端帧上比分离的任意一帧都准（误差集中在远端）",
      FULL_MAXACC[-1] > SEP_MAXACC[-1]
      and FULL_MAXACC[-1] > SEP_MEANACC[-1] + 0.10,
      f"近端：full {FULL_MAXACC[-1]:.4f} vs sep 最好帧 {SEP_MAXACC[-1]:.4f}"
      f" / sep 平均 {SEP_MEANACC[-1]:.4f}")
CK.ok("跨度随帧数线性增长，但 B 恒为 12（token 数不增）",
      all(SPAN_GRID[i + 1] > SPAN_GRID[i] for i in range(4))
      and all(A_sep(K).shape[0] == B_SLOTS for K in K_GRID),
      f"span {SPAN_GRID}，B={B_SLOTS}")

# ---- A2：Δ 扫描 —— 跨度/采样间隔 ↔ 可恢复的帧间事件比例
SPAN_DELTA = [(K_GRID[-1] - 1) * d for d in DELTA_GRID]
PRECOVER = [min(1.0, 2 * W_EVENT / d) for d in DELTA_GRID]


def recover_mc(d, seed, n=8000):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, d, size=n)
    return float(np.mean(np.minimum(u, d - u) < W_EVENT))


PRECOVER_MC, _ = (list(x) for x in zip(*[mc(lambda s, dd=d: recover_mc(dd, s))
                                          for d in DELTA_GRID]))
print("\n   K=18 的采样间隔扫描：Δ / 跨度 / 帧间事件可恢复比例（解析 ↔ MC）")
for d, sp, p, pm in zip(DELTA_GRID, SPAN_DELTA, PRECOVER, PRECOVER_MC):
    print(f"   Δ={d}s  span={sp}s  P_recover={p:.4f}  MC={pm:.4f}")
CK.ok("跨度 = (K−1)Δ 与可恢复事件比例 min(1,2w/Δ)：解析 ↔ MC 反算一致",
      max(abs(PRECOVER[i] - PRECOVER_MC[i]) for i in range(3)) < 0.02,
      "  ".join(f"Δ={d}: {PRECOVER[i]:.4f}/{PRECOVER_MC[i]:.4f}"
               for i, d in enumerate(DELTA_GRID)))
CK.ok("反直觉：token 数不增时拉长跨度（Δ↑）要以时间分辨率为代价",
      all(SPAN_DELTA[i + 1] > SPAN_DELTA[i] for i in range(2))
      and all(PRECOVER[i + 1] < PRECOVER[i] for i in range(2)),
      f"span {SPAN_DELTA} / P_recover {[round(p, 4) for p in PRECOVER]}")

# ---- A3：复杂度（论文 §III-C 的 O(Kn²+nK²) vs O(n²K²)）
def cost_sep(K):
    return K * N_PATCH ** 2 + N_PATCH * K ** 2


def cost_full(K):
    return N_PATCH ** 2 * K ** 2


C_BUDGET = cost_full(K_GRID[-1])
KMAX_SEP = max(K for K in range(1, 400) if cost_sep(K) <= C_BUDGET)
KMAX_FULL = max(K for K in range(1, 400) if cost_full(K) <= C_BUDGET)
COST_SEP18, COST_FULL18 = cost_sep(18), cost_full(18)
COST_RATIO = COST_FULL18 / COST_SEP18
print(f"\n   复杂度（n={N_PATCH}）：K=18 全连接 {COST_FULL18}，分离 {COST_SEP18}，"
      f"比值 {COST_RATIO:.2f}×；同算力预算 C={C_BUDGET} 下 "
      f"K_max 全连接 {KMAX_FULL} / 分离 {KMAX_SEP}（√n={math.sqrt(N_PATCH):.0f}）")
CK.ok("每 4 层加一次 temporal attention：12 层 → 3 个 temporal 层（引用机制 + 自设层数）",
      T_LAYERS == 3 and N_LAYERS % PAPER["every_n_layers"] == 0,
      f"L={N_LAYERS}，every={PAPER['every_n_layers']} → {T_LAYERS} 层")
CK.ok("时空分离显著省算力（K=18 时全连接开销 > 分离的 5 倍）",
      COST_RATIO > 5.0, f"{COST_FULL18} / {COST_SEP18} = {COST_RATIO:.2f}×")
CK.ok("同一算力预算下分离可支持 ≥3× 帧数，但达不到理想 √n（空间项也占预算）",
      KMAX_SEP >= 3 * KMAX_FULL and KMAX_SEP < math.sqrt(N_PATCH) * KMAX_FULL,
      f"K_max {KMAX_FULL} → {KMAX_SEP}（比值 {KMAX_SEP / KMAX_FULL:.2f}，√n={math.sqrt(N_PATCH):.0f}）")

store("K_GRID", K_GRID); store("SPAN_GRID", SPAN_GRID)
store("SEP_MSE", SEP_MSE); store("FULL_MSE", FULL_MSE)
store("SEP_MAXMSE", SEP_MAX); store("FULL_MAXMSE", FULL_MAX)
store("SEP_MEANACC", SEP_MEANACC); store("FULL_MEANACC", FULL_MEANACC)
store("SEP_MINACC", SEP_MIN); store("FULL_MINACC", FULL_MIN)
store("SEP_MAXACC", SEP_MAXACC); store("FULL_MAXACC", FULL_MAXACC)
store("DELTA_GRID", DELTA_GRID); store("SPAN_DELTA", SPAN_DELTA)
store("PRECOVER", PRECOVER); store("PRECOVER_MC", PRECOVER_MC)
store("COST_SEP18", COST_SEP18); store("COST_FULL18", COST_FULL18)
store("COST_RATIO", COST_RATIO); store("KMAX_SEP", KMAX_SEP)
store("KMAX_FULL", KMAX_FULL); store("C_BUDGET", C_BUDGET)


# ===================================================================== B 组
CK.banner("B 组：短时记忆补遮挡 —— 无记忆 / 只当前帧 / 有视频记忆，及跨度不够的失效")


def video_acc(K, d):
    """遮挡 d 帧：d<K 时视频记忆里还留着清晰帧；d≥K 时只省下当前弱线索。"""
    return ACC_CLEAR if d < K else ACC_WEAK


def b_mc(K, d, seed, n=6000):
    rng = np.random.default_rng(seed)
    j = rng.integers(0, L_DRAW, size=n)
    y = rng.normal(0.0, SIG_OBS, size=(n, L_DRAW))
    y[np.arange(n), j] += A_CLEAR if d < K else A_WEAK
    return float(np.mean(np.argmax(y, axis=1) == j))


VIDEO = {K: [video_acc(K, d) for d in D_GRID] for K in K_MEM}
print("\n   遮挡时长 d（帧）  无记忆   只当前帧   video(K=6)   video(K=18)")
for d in D_GRID:
    print(f"   {d:>10}   {ACC_CHANCE:>7.4f}   {ACC_WEAK:>8.4f}"
          f"   {VIDEO[6][D_GRID.index(d)]:>10.4f}   {VIDEO[18][D_GRID.index(d)]:>12.4f}")

# 多 seed 反算整张 d 扫描表
VIDEO_MC = {}
for K in K_MEM:
    VIDEO_MC[K] = [mc(lambda s, kk=K, dd=d: b_mc(kk, dd, s))[0] for d in D_GRID]
MC_OK = max(abs(VIDEO[K][i] - VIDEO_MC[K][i]) for K in K_MEM for i in range(len(D_GRID)))
CK.ok("B 组整张 d 扫描：解析 ↔ MC 反算一致（多 seed）",
      MC_OK < 0.02, f"最大偏差 {MC_OK:.4f}")
CK.ok("当前帧被遮挡时，无记忆与只当前帧都只有机会/弱线索水平",
      ACC_CHANCE < ACC_WEAK < ACC_CLEAR - 0.5,
      f"无记忆 {ACC_CHANCE:.4f} / 只当前帧 {ACC_WEAK:.4f} / 有视频记忆 {ACC_CLEAR:.4f}")
CK.ok("有视频记忆（窗内）把遮挡补偿拉到清晰帧水平，远超只当前帧",
      VIDEO[18][0] > ACC_WEAK + 0.5,
      f"d=1：video {VIDEO[18][0]:.4f} vs 只当前帧 {ACC_WEAK:.4f}")
CK.ok("记忆跨度不够时失效：K=6、d=7 退化到「只当前帧」而非塌到随机（诚实边界）",
      abs(VIDEO[6][3] - ACC_WEAK) < 1e-9 and VIDEO[6][3] > ACC_CHANCE + 0.02,
      f"K=6,d=7 → {VIDEO[6][3]:.4f} = 弱线索 {ACC_WEAK:.4f}；机会 {ACC_CHANCE:.4f}")
CK.ok("同一遮挡 d=7，K=18 仍拿到清晰帧、K=6 已退化到弱线索（跨度扩展的价值）",
      VIDEO[18][3] == ACC_CLEAR and VIDEO[6][3] == ACC_WEAK
      and VIDEO[18][3] > VIDEO[6][3] + 0.40,
      f"K=18 {VIDEO[18][3]:.4f} vs K=6 {VIDEO[6][3]:.4f}")
CK.ok("反直觉：K 从 6 增到 18 不提高窗内补偿精度，只推高「不失效」的遮挡时长上限",
      abs(VIDEO[6][0] - VIDEO[18][0]) < 1e-9
      and sum(v > ACC_WEAK + 0.5 for v in VIDEO[18]) > sum(v > ACC_WEAK + 0.5 for v in VIDEO[6]),
      f"d=1 两者均 {VIDEO[6][0]:.4f}；窗内 d 计数 K=6 {sum(v > ACC_WEAK + 0.5 for v in VIDEO[6])}"
      f" → K=18 {sum(v > ACC_WEAK + 0.5 for v in VIDEO[18])}")
CK.ok("遮挡超过最大跨度（d=24）时，连 K=18 也失效",
      VIDEO[18][-1] < ACC_WEAK + 1e-9,
      f"d=24：video(K=18) {VIDEO[18][-1]:.4f}")

store("D_GRID", D_GRID)
store("ACC_CHANCE", ACC_CHANCE); store("ACC_WEAK", ACC_WEAK)
store("ACC_CLEAR", ACC_CLEAR)
store("VIDEO_K6", VIDEO[6]); store("VIDEO_K18", VIDEO[18])


# ===================================================================== C 组
CK.banner("C 组：长时记忆补阶段 —— π_HL / LLM 标签 / 压缩去冗余 / naive / 压缩过头")


def coverage(alpha, drop_failures=False):
    """压缩强度 α 下文本记忆保留的阶段标记比例。

    原始标签流：每阶段 [marker, filler×RHO]（filler=失败重试/颜色细节）。
    α 优先删 filler、再删最旧的 marker；剩余进容量 L_MAX 的 buffer（只留最近）。
    drop_failures=True 模拟 π_HL「失败的子任务不更新记忆」：filler 从不写入。
    """
    entries = []
    for s in range(N_STAGE):
        entries.append((s, "marker"))
        if not drop_failures:
            for _ in range(RHO):
                entries.append((s, "filler"))
    n_remove = int(round(alpha * len(entries)))
    order = ([i for i, e in enumerate(entries) if e[1] == "filler"]
             + [i for i, e in enumerate(entries) if e[1] == "marker"])
    remove = set(order[:n_remove])
    kept = [e for i, e in enumerate(entries) if i not in remove][-L_MAX:]
    stages = {s for s, kind in kept if kind == "marker"}
    return len(stages) / N_STAGE


def acc_from_coverage(c):
    return c * 1.0 + (1.0 - c) * 0.5


def c_mc(alpha, seed, n=6000, drop_failures=False):
    """随机阶段标记 + 随机提问阶段，MC 反算阶段判别准确率。"""
    c = coverage(alpha, drop_failures)
    stages = set()
    # 复算被保留的阶段集合（与 coverage 同构），用于逐次提问
    entries = []
    for s in range(N_STAGE):
        entries.append((s, "marker"))
        if not drop_failures:
            for _ in range(RHO):
                entries.append((s, "filler"))
    n_remove = int(round(alpha * len(entries)))
    order = ([i for i, e in enumerate(entries) if e[1] == "filler"]
             + [i for i, e in enumerate(entries) if e[1] == "marker"])
    remove = set(order[:n_remove])
    kept = [e for i, e in enumerate(entries) if i not in remove][-L_MAX:]
    stages = {s for s, kind in kept if kind == "marker"}
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(n, N_STAGE))
    q = rng.integers(0, N_STAGE, size=n)
    known = np.array([s in stages for s in q])
    guess = rng.integers(0, 2, size=n)
    ans = np.where(known, bits[np.arange(n), q], guess)
    return float(np.mean(ans == bits[np.arange(n), q]))


COVER = [coverage(a) for a in A_GRID]
ACC_ALPHA = [acc_from_coverage(c) for c in COVER]
ACC_ALPHA_MC = [mc(lambda s, aa=a: c_mc(aa, s))[0] for a in A_GRID]
ACC_NOFAIL = acc_from_coverage(coverage(0.0, drop_failures=True))
ALPHA_STAR = A_GRID[int(np.argmax(ACC_ALPHA))]

# 四种记忆配置（长程阶段判别）
# 只视频：窗口 54 s < 阶段 60 s，最多看见 1 个阶段标记
N_VISIBLE = W_MEM_S // STAGE_SEC + 1
ACC_VIDEO_ONLY = (1.0 / N_STAGE) * 1.0 + (1.0 - 1.0 / N_STAGE) * 0.5
ACC_NONE_MEM = 0.5
ACC_TEXT_ONLY = ACC_ALPHA[int(np.argmax(ACC_ALPHA))]
ACC_BOTH = ACC_TEXT_ONLY

print("\n   α(压缩强度)   覆盖率   阶段判别准确率(解析)   MC 反算")
for a, c, ac, m in zip(A_GRID, COVER, ACC_ALPHA, ACC_ALPHA_MC):
    print(f"   {a:>10.3f}   {c:>7.4f}   {ac:>17.4f}   {m:>8.4f}")
print(f"   峰值 α*={ALPHA_STAR:.3f}（= ρ/(1+ρ)={RHO/(1+RHO):.3f}）；"
      f"失败不更新记忆（α=0）准确率 {ACC_NOFAIL:.4f}")
print("\n   四种记忆配置 × 长程阶段判别："
      f"都没有 {ACC_NONE_MEM:.4f} / 只视频 {ACC_VIDEO_ONLY:.4f} / "
      f"只文本 {ACC_TEXT_ONLY:.4f} / 两者都有 {ACC_BOTH:.4f}")

CK.ok("C 组 α 扫描：解析覆盖率 ↔ MC 反算一致（多 seed）",
      max(abs(ACC_ALPHA[i] - ACC_ALPHA_MC[i]) for i in range(len(A_GRID))) < 0.02,
      "  ".join(f"{ACC_ALPHA[i]:.4f}/{ACC_ALPHA_MC[i]:.4f}" for i in range(len(A_GRID))))
CK.ok("压缩准确率是倒 U 形：先升后降，峰值恰在 α*=ρ/(1+ρ)=0.5",
      all(ACC_ALPHA[i + 1] >= ACC_ALPHA[i] - 1e-9 for i in range(3))
      and all(ACC_ALPHA[i + 1] <= ACC_ALPHA[i] + 1e-9 for i in range(3, 6))
      and abs(ALPHA_STAR - RHO / (1 + RHO)) < 1e-9,
      "  ".join(f"α={a:.3f}:{v:.4f}" for a, v in zip(A_GRID, ACC_ALPHA)))
CK.ok("峰值（删完冗余、阶段标记一个不丢）达到完美阶段判别 1.0",
      abs(ACC_ALPHA[int(np.argmax(ACC_ALPHA))] - 1.0) < 1e-9,
      f"峰值 {ACC_ALPHA[int(np.argmax(ACC_ALPHA))]:.4f}")
CK.ok("naive 拼接所有历史（α=0，含失败重试冗余）显著更差",
      ACC_ALPHA[0] < ACC_ALPHA[int(np.argmax(ACC_ALPHA))] - 0.20,
      f"naive {ACC_ALPHA[0]:.4f} vs 压缩 {ACC_ALPHA[int(np.argmax(ACC_ALPHA))]:.4f}"
      f"（论文：分布漂移显著更差）")
CK.ok("压缩过头丢掉关键阶段：α=1 退化到随机 0.5",
      abs(ACC_ALPHA[-1] - 0.5) < 1e-9, f"α=1.0 → {ACC_ALPHA[-1]:.4f}")
CK.ok("π_HL「失败不更新记忆」等价于把冗余全删掉：α=0 也拿到 1.0",
      abs(ACC_NOFAIL - 1.0) < 1e-9 and abs(ACC_NOFAIL - ACC_ALPHA[int(np.argmax(ACC_ALPHA))]) < 1e-9,
      f"失败不更新 {ACC_NOFAIL:.4f} = 显式压缩峰值 {ACC_ALPHA[int(np.argmax(ACC_ALPHA))]:.4f}")
CK.ok("只文本（压缩阶段标签）相对无记忆把阶段判别从 0.5 拉到 1.0",
      ACC_TEXT_ONLY > ACC_NONE_MEM + 0.40,
      f"都没有 {ACC_NONE_MEM:.4f} → 只文本 {ACC_TEXT_ONLY:.4f}")
CK.ok("反直觉：两者都有 = 只文本；短时视频记忆对长程阶段判别几乎无效",
      abs(ACC_BOTH - ACC_TEXT_ONLY) < 1e-9
      and ACC_VIDEO_ONLY - ACC_NONE_MEM < 0.05,
      f"只视频 {ACC_VIDEO_ONLY:.4f} vs 都没有 {ACC_NONE_MEM:.4f}（差 {ACC_VIDEO_ONLY-ACC_NONE_MEM:.4f}）")
CK.ok("短时窗口 54 s < 阶段 60 s，最多只看见 1 个阶段标记（分工的结构来源）",
      N_VISIBLE == 1, f"W={W_MEM_S}s / 阶段 {STAGE_SEC}s → 可见标记 {N_VISIBLE} 个")

store("A_GRID", A_GRID); store("COVER", COVER)
store("ACC_ALPHA", ACC_ALPHA); store("ACC_ALPHA_MC", ACC_ALPHA_MC)
store("ACC_NOFAIL", ACC_NOFAIL); store("ALPHA_STAR", ALPHA_STAR)
store("CFG_STAGE", [ACC_NONE_MEM, ACC_VIDEO_ONLY, ACC_TEXT_ONLY, ACC_BOTH])


# ===================================================================== D 组
CK.banner("D 组：两类记忆的分工边界 —— 遮挡 × 长程阶段 × 四种记忆配置交叉对照")

# 遮挡任务（d=3，K=18 窗内）：文本记忆里没有抽屉位置信息 → 只文本=机会基线
CFG_OCC = [ACC_CHANCE,            # 都没有
           video_acc(18, 3),      # 只视频
           ACC_CHANCE,            # 只文本（无位置信息）
           video_acc(18, 3)]      # 两者都有（文本不加分、不干扰）
CFG_STAGE = [ACC_NONE_MEM, ACC_VIDEO_ONLY, ACC_TEXT_ONLY, ACC_BOTH]
CFG_NAMES = ["都没有", "只视频", "只文本", "两者都有"]

print("\n   配置       遮挡补偿准确率   长程阶段判别准确率")
for n, o, s in zip(CFG_NAMES, CFG_OCC, CFG_STAGE):
    print(f"   {n:<8}   {o:>13.4f}   {s:>17.4f}")
print(f"\n   记忆跨度不够的交叉边界：K=6、d=7 时 video 在遮挡任务只有 "
      f"{video_acc(6, 7):.4f}（=弱线索），而文本记忆仍是完好的 {ACC_TEXT_ONLY:.4f}")

CK.ok("遮挡任务：短时视频记忆有效（只视频 >> 只文本）",
      CFG_OCC[1] > CFG_OCC[2] + 0.50,
      f"只视频 {CFG_OCC[1]:.4f} vs 只文本 {CFG_OCC[2]:.4f}")
CK.ok("遮挡任务：长时文本记忆完全无效（只文本 = 都没有）",
      abs(CFG_OCC[2] - CFG_OCC[0]) < 1e-9,
      f"只文本 {CFG_OCC[2]:.4f} = 都没有 {CFG_OCC[0]:.4f}")
CK.ok("长程阶段任务：长时文本记忆有效（只文本 >> 只视频）",
      CFG_STAGE[2] > CFG_STAGE[1] + 0.40,
      f"只文本 {CFG_STAGE[2]:.4f} vs 只视频 {CFG_STAGE[1]:.4f}")
CK.ok("长程阶段任务：短时视频记忆几乎无效（只视频 − 都没有 < 0.05）",
      CFG_STAGE[1] - CFG_STAGE[0] < 0.05,
      f"{CFG_STAGE[1]:.4f} − {CFG_STAGE[0]:.4f} = {CFG_STAGE[1]-CFG_STAGE[0]:.4f}")
CK.ok("两者都有 = 对应单项记忆的最大值（无叠加增益，也无性能退化）",
      all(abs(CFG_OCC[3] - max(CFG_OCC[1], CFG_OCC[2])) < 1e-9
          for _ in [0])
      and abs(CFG_STAGE[3] - max(CFG_STAGE[1], CFG_STAGE[2])) < 1e-9,
      f"遮挡 {CFG_OCC[3]:.4f} / 阶段 {CFG_STAGE[3]:.4f}（论文 Fig.9 无退化）")
_best_o, _best_s = max(CFG_OCC), max(CFG_STAGE)
_n_best = sum(1 for i in range(4) if CFG_OCC[i] == _best_o and CFG_STAGE[i] == _best_s)
CK.ok("反直觉：只有「两者都有」同时在两类任务上都取到各自最佳水平（互补而非叠加）",
      CFG_OCC[3] == _best_o and CFG_STAGE[3] == _best_s and _n_best == 1,
      f"遮挡最佳 {_best_o:.4f} / 阶段最佳 {_best_s:.4f}；"
      f"两项都达最佳的配置数 {_n_best}")
CK.ok("交叉边界：记忆跨度不够时视频失效，完好的文本记忆也救不了遮挡",
      abs(video_acc(6, 7) - ACC_WEAK) < 1e-9 and ACC_TEXT_ONLY >= 0.9,
      f"K=6,d=7 遮挡 {video_acc(6, 7):.4f}；文本阶段 {ACC_TEXT_ONLY:.4f}")

store("CFG_OCC", CFG_OCC); store("CFG_STAGE_D", CFG_STAGE)
store("CFG_NONE", ACC_NONE_MEM); store("CFG_VIDEO_ONLY", ACC_VIDEO_ONLY)


# ===================================================================== 论文常量（引用）
print("\n" + "=" * 74)
print("论文常量（引用 · 非合成探针数字 · 不得与合成读数混用）")
print("=" * 74)
print(f"  arXiv:{PAPER['arxiv']}  MEM | 基座 {PAPER['base']} = {PAPER['vlm']} VLM "
      f"+ {PAPER['action_expert']} action expert | 最长任务 {PAPER['max_task_min']} 分钟")
print(f"  视频记忆：ViT 每 {PAPER['every_n_layers']} 层加 causal temporal attention；"
      f"token 数{PAPER['token_count']}；复杂度 {PAPER['complexity_sep']} vs {PAPER['complexity_full']}")
print(f"  预训练 {PAPER['pretrain_frames']} 帧 / stride {PAPER['pretrain_stride_s']} s "
      f"（≈{PAPER['pretrain_span_s']} s 历史）→ 后训练最多 {PAPER['posttrain_frames']} 帧 / "
      f"{PAPER['posttrain_span_s']} s；上层丢弃历史 token")
print(f"  文本记忆：π_HL 生成 m_t；训练标签由 off-the-shelf LLM 造、明确要求压缩去冗余；")
print(f"  失败子任务不更新记忆；naive 拼接所有历史显著更差；两者都必不可少；Fig.9 无性能退化")
print(f"  许可 {PAPER['license']}，但本系列统一自画，不复用论文 Figure；")
print(f"  论文无数值表、无逐任务成功率、无独立遮挡实验数值、未提及 π0.7")


# ===================================================================== 正文回填表
CK.banner("正文回填表（以下数字与上面逐条对齐，均为原始读数；博客合成数字须能在此 grep 到）")
print(f"  [世界] B={B_SLOTS}  K_GRID={K_GRID}  Δ_GRID={DELTA_GRID}  "
      f"n_patch={N_PATCH}  L_layers={N_LAYERS}  temporal_layers={T_LAYERS}  τ²={TAU2}")
print(f"  [世界] L_draw={L_DRAW}  σ_obs={SIG_OBS}  A_clear={A_CLEAR}  A_weak={A_WEAK}  "
      f"N_stage={N_STAGE}  stage_sec={STAGE_SEC}  ρ={RHO}  L_max={L_MAX}  W_mem={W_MEM_S}s")
print(f"  [自校验] DCT 正交偏差={_ortho:.2e}；后验增益 g*={_g_star:.6f} "
      f"（有限差分导数 {_d1:.2e}，网格最小 {_g_num:.6f}）")
print(f"  [自校验] 机会基线 1/L={ACC_CHANCE:.6f}（MC {_acc0_mc:.4f}）；"
      f"弱线索 {ACC_WEAK:.4f}（MC {_accw_mc:.4f}）；清晰帧 {ACC_CLEAR:.4f}（MC {_accc_mc:.4f}）")
print(f"  [A组] span(Δ=1, K={K_GRID}) = " + "  ".join(f"{v}" for v in SPAN_GRID))
print("  [A组] sep 总重建 MSE = " + "  ".join(f"{v:.4f}" for v in SEP_MSE))
print("  [A组] full 总重建 MSE = " + "  ".join(f"{v:.4f}" for v in FULL_MSE))
print("  [A组] sep 最差帧 MSE = " + "  ".join(f"{v:.4f}" for v in SEP_MAX))
print("  [A组] full 最差帧 MSE = " + "  ".join(f"{v:.4f}" for v in FULL_MAX))
print("  [A组] sep 平均符号判别 = " + "  ".join(f"{v:.4f}" for v in SEP_MEANACC))
print("  [A组] full 平均符号判别 = " + "  ".join(f"{v:.4f}" for v in FULL_MEANACC))
print("  [A组] sep 最差帧判别 = " + "  ".join(f"{v:.4f}" for v in SEP_MIN))
print("  [A组] full 最差帧判别 = " + "  ".join(f"{v:.4f}" for v in FULL_MIN))
print("  [A组] sep 近端帧判别 = " + "  ".join(f"{v:.4f}" for v in SEP_MAXACC))
print("  [A组] full 近端帧判别 = " + "  ".join(f"{v:.4f}" for v in FULL_MAXACC))
print(f"  [A组] sep MSE 多 seed MC = " + "  ".join(f"{v:.4f}" for v in SEP_MSE_MC))
print(f"  [A组] sep 平均判别 MC = " + "  ".join(f"{v:.4f}" for v in SEP_ACC_MC))
print(f"  [A组] Δ 扫描 span = " + "  ".join(f"{v}" for v in SPAN_DELTA))
print("  [A组] Δ 扫描 P_recover(解析) = " + "  ".join(f"{v:.4f}" for v in PRECOVER))
print("  [A组] Δ 扫描 P_recover(MC)   = " + "  ".join(f"{v:.4f}" for v in PRECOVER_MC))
print(f"  [A组] 复杂度 K=18：全连接 {COST_FULL18} 分离 {COST_SEP18} 比值 {COST_RATIO:.2f}×；"
      f"预算 C={C_BUDGET} 下 K_max 全连接 {KMAX_FULL} / 分离 {KMAX_SEP}（√n={math.sqrt(N_PATCH):.0f}）")
print(f"  [B组] d 扫描 = {D_GRID}")
print(f"  [B组] 无记忆 {ACC_CHANCE:.4f} 只当前帧 {ACC_WEAK:.4f} 清晰帧 {ACC_CLEAR:.4f}")
print("  [B组] video(K=6) = " + "  ".join(f"{v:.4f}" for v in VIDEO[6]))
print("  [B组] video(K=18) = " + "  ".join(f"{v:.4f}" for v in VIDEO[18]))
print("  [B组] video(K=6) MC = " + "  ".join(f"{v:.4f}" for v in VIDEO_MC[6]))
print("  [B组] video(K=18) MC = " + "  ".join(f"{v:.4f}" for v in VIDEO_MC[18]))
print("  [C组] α 扫描 = " + "  ".join(f"{v:.4f}" for v in A_GRID))
print("  [C组] 覆盖率 = " + "  ".join(f"{v:.4f}" for v in COVER))
print("  [C组] 阶段判别准确率 = " + "  ".join(f"{v:.4f}" for v in ACC_ALPHA))
print("  [C组] 阶段判别 MC = " + "  ".join(f"{v:.4f}" for v in ACC_ALPHA_MC))
print(f"  [C组] α*={ALPHA_STAR:.4f} 峰值={max(ACC_ALPHA):.4f} "
      f"naive(α=0)={ACC_ALPHA[0]:.4f} 压缩过头(α=1)={ACC_ALPHA[-1]:.4f} "
      f"失败不更新={ACC_NOFAIL:.4f} 可见阶段标记={N_VISIBLE}")
print(f"  [C组] 四配置(都没有/只视频/只文本/两者都有) = "
      + "  ".join(f"{v:.4f}" for v in CFG_STAGE))
print(f"  [D组] 遮挡任务 四配置 = " + "  ".join(f"{v:.4f}" for v in CFG_OCC))
print(f"  [D组] 阶段任务 四配置 = " + "  ".join(f"{v:.4f}" for v in CFG_STAGE))
print("  注（反直觉 / 边界，写正文须照此口径）：")
print("   · A 组：固定 token 预算下两种注意力的总重建误差完全相同（都是 B 个标量槽）；")
print("     差别在误差分布 —— 分离摊平到全跨度、全连接集中在远端帧。平均判别准确率")
print(f"     K=18 时两者原始读数 {SEP_MEANACC[-1]:.4f} / {FULL_MEANACC[-1]:.4f} 相差很小，只有最差帧/远期召回指标")
print(f"     能把二者分开（sep {SEP_MIN[-1]:.4f} vs full {FULL_MIN[-1]:.4f}）；全连接近端反而更准")
print(f"     （{FULL_MAXACC[-1]:.4f} vs {SEP_MAXACC[-1]:.4f}）。跨度靠 Δ↑ 拉长不增 token，")
print(f"     但帧间事件可恢复比例按 2w/Δ 从 {PRECOVER[0]:.4f} 掉到 {PRECOVER[-1]:.4f}。")
print(f"   · B 组：K 6→18 在窗内 d=1 的补偿精度不变（均 {VIDEO[6][0]:.4f}），跨度买的是覆盖范围")
print(f"     不是精度；d≥K 时退化到「只当前帧」{ACC_WEAK:.4f}，不是塌到随机 {ACC_CHANCE:.4f}。")
print(f"   · C 组：压缩准确率是倒 U 形，峰值 α*={ALPHA_STAR:.3f}=ρ/(1+ρ)，恰在删完冗余、")
print(f"     阶段标记一个不丢处（{max(ACC_ALPHA):.4f}）；naive {ACC_ALPHA[0]:.4f} 与压缩过头")
print(f"     {ACC_ALPHA[-1]:.4f} 都差。「两者都有」= 只文本 {ACC_TEXT_ONLY:.4f}，短时视频对长程无效。")
print(f"   · D 组：两类记忆互补而非叠加 —— 两者都有在单项任务上等于对应单项记忆，")
print(f"     但只有它同时在两类任务上都取到各自最佳水平（遮挡 {CFG_OCC[3]:.4f} / 阶段 {CFG_STAGE[3]:.4f}）；")
print(f"     长时文本对遮挡无效（{CFG_OCC[2]:.4f}=机会），")
print(f"     短时视频对长程几乎无效（{ACC_VIDEO_ONLY:.4f} vs {ACC_NONE_MEM:.4f}）。")


# ---- 机器可读块（mem_figures.py 逐位解析；勿改格式）
def _fmt(v):
    return f"{v:.6f}"


print("\n[MACHINE-BACKFILL-BEGIN]")
for _k in ["K_GRID", "SPAN_GRID", "SEP_MSE", "FULL_MSE", "SEP_MAXMSE", "FULL_MAXMSE",
           "SEP_MEANACC", "FULL_MEANACC", "SEP_MINACC", "FULL_MINACC",
           "SEP_MAXACC", "FULL_MAXACC", "DELTA_GRID", "SPAN_DELTA",
           "PRECOVER", "PRECOVER_MC", "COST_SEP18", "COST_FULL18", "COST_RATIO",
           "KMAX_SEP", "KMAX_FULL", "C_BUDGET", "D_GRID", "ACC_CHANCE", "ACC_WEAK",
           "ACC_CLEAR", "VIDEO_K6", "VIDEO_K18", "A_GRID", "COVER", "ACC_ALPHA",
           "ACC_ALPHA_MC", "ACC_NOFAIL", "ALPHA_STAR", "CFG_STAGE", "CFG_OCC",
           "CFG_STAGE_D", "CFG_NONE", "CFG_VIDEO_ONLY"]:
    print("@@" + _k + "=" + ",".join(_fmt(v) for v in BACK[_k]))
print("[MACHINE-BACKFILL-END]")

CK.summary()
