# -*- coding: utf-8 -*-
"""Hi Robot companion lab —— 把「开放式指令」拆成子任务，拆了之后换来什么。

对应文章：π 系列具身模型（04）Hi Robot（arXiv 2502.19417）。

论文层面的数字（三个平台的 IA / TP、GPT-4o 对照、专家上限）是引用，本实验不重算。
这里只负责四件能在 CPU 上钉死的事：

  A 组  分解到底换来什么：同一批演示、同一个预算，分层与扁平差多少，差在哪
  B 组  高层多久重算一次：1 Hz 与「事件触发」是有代价的设计参数，不是随手写的
  C 组  接口的脆弱处：高层说出的技能必须落在低层认识的措辞上
  D 组  合成数据喂谁：为什么合成标注只喂高层，而不喂低层

纯 numpy / CPU / 固定 seed / 每组末尾断言 / 最后打出断言总数。

一个必须说清的前提：本实验验的是**机制**，不是论文成绩。
世界里的「技能」是 8 维动作原型，「约束」是 4 个属性位上的筛选谓词，
规模比真机小好几个数量级；所有结论只在「合成探针」的范围内主张。
"""

import numpy as np

import hirobot_lib as L
from hirobot_lib import (D_A, D_C, D_X, DONE, M, S, World, Adam, bwd, fwd, init_mlp,
                         mse_grad, onehot, sample_batch, softmax_ce)

SEED = 20261104
N_ASSERT = 0


def ok(cond, msg):
    global N_ASSERT
    assert cond, f"FAILED: {msg}"
    N_ASSERT += 1
    print(f"  [ok] {msg}")


def section(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def banner(t):
    print(f"\n--- {t} ---")


W = World(0)
L.SIG_ETA = 0.15          # 噪声下调到「信号 : 噪声 ≈ 4 : 1」，留出可分辨的动态范围


# ------------------------------------------------------------------ 采样器
def sample_by_probs(cs, probs, n, rng):
    """按给定分布从若干约束里采 n 个样本。"""
    cnt = rng.multinomial(n, np.asarray(probs, dtype=float) / np.sum(probs))
    xs, cvs, ss = [], [], []
    for c, m in zip(cs, cnt, strict=True):
        if m == 0:
            continue
        x = W._sample_scene(m, rng)
        xs.append(x)
        cvs.append(np.repeat(W.cvec(*c)[None], m, axis=0))
        ss.append(W.skill(x, *c))
    x = np.concatenate(xs)
    cv = np.concatenate(cvs)
    s = np.concatenate(ss)
    return cv, x, s, W.action(x, s, noise=True, rng=rng)


# ------------------------------------------------------------------ 训练
def train_pair(cs, steps, *, seed=1, bs=64, lr=3e-3, probs=None):
    """同一批演示同时训出扁平 / 高层 / 低层三个网络。返回 (flat, hi, lo)。"""
    rng = np.random.default_rng(seed)
    flat = init_mlp(D_C + D_X, D_A, rng)
    hi = init_mlp(D_C + D_X, S, rng)
    lo = init_mlp(S + D_X, D_A, rng)
    oF, oH, oL = Adam(flat, lr), Adam(hi, lr), Adam(lo, lr)
    for _ in range(steps):
        if probs is None:
            cv, xb, s, a = sample_batch(W, cs, bs // len(cs) + 1, rng)
            cv, xb, s, a = cv[:bs], xb[:bs], s[:bs], a[:bs]
        else:
            cv, xb, s, a = sample_by_probs(cs, probs, bs, rng)
        Xf = np.concatenate([cv, xb], axis=1)
        H, o = fwd(flat, Xf)
        _, d = mse_grad(o, a)
        oF.step(flat, bwd(flat, Xf, H, d))
        H, o = fwd(hi, Xf)
        _, d = softmax_ce(o, s)
        oH.step(hi, bwd(hi, Xf, H, d))
        Xl = np.concatenate([onehot(s, S), xb], axis=1)
        H, o = fwd(lo, Xl)
        _, d = mse_grad(o, a)
        oL.step(lo, bwd(lo, Xl, H, d))
    return flat, hi, lo


def evaluate(flat, hi, lo, cs, *, n=500, seed=101, oracle=False):
    """返回 (扁平准确率, 分层准确率, 扁平相对误差, 分层相对误差)。"""
    rng = np.random.default_rng(seed)
    cv, xb, s, a = sample_batch(W, cs, n, rng)
    Xf = np.concatenate([cv, xb], axis=1)
    _, of = fwd(flat, Xf)
    _, lh = fwd(hi, Xf)
    sh = s if oracle else np.argmax(lh, axis=1)
    _, ol = fwd(lo, np.concatenate([onehot(sh, S), xb], axis=1))
    sf = W.implied_skill(of, xb)          # 论文给扁平基线的打分协议：反推它想做哪一步
    return (float(np.mean(sf == s)), float(np.mean(sh == s)),
            W.rel_err(of, a), W.rel_err(ol, a))


# ================================================================== 自校验
section("自校验：手写反向 vs 有限差分")
rng0 = np.random.default_rng(SEED)
_p = init_mlp(D_C + D_X, D_A, rng0, hid=16)
_xb = W._sample_scene(8, rng0)
_cv = np.repeat(W.cvec_batch(W.train_pool[:2]), 4, axis=0)
_X = np.concatenate([_cv, _xb], axis=1)
_y = W.action(_xb, W.skill(_xb, *W.train_pool[0])[:4].tolist() + W.skill(_xb, *W.train_pool[1])[:4].tolist(),
              noise=False)
_H, _o = fwd(_p, _X)
_, _d = mse_grad(_o, _y)
_g = bwd(_p, _X, _H, _d)
eps = 1e-6
maxrel = 0.0
for li, gi in enumerate(_g):
    flat_g = gi.reshape(-1)
    for k in range(0, flat_g.size, max(1, flat_g.size // 12)):
        idx = np.unravel_index(k, gi.shape)
        old = _p[li][idx]
        _p[li][idx] = old + eps
        _, o1 = fwd(_p, _X); l1 = float(np.mean(np.square(o1 - _y)))
        _p[li][idx] = old - eps
        _, o2 = fwd(_p, _X); l2 = float(np.mean(np.square(o2 - _y)))
        _p[li][idx] = old
        num = (l1 - l2) / (2 * eps)
        den = abs(flat_g[k]) + 1e-12
        maxrel = max(maxrel, abs(num - flat_g[k]) / den)
ok(maxrel < 1e-5, f"手写反向与有限差分一致（最大相对误差 {maxrel:.1e} < 1e-5）")

_xb = W._sample_scene(4096, rng0)
_a0 = W.action(_xb, W.skill(_xb, *W.train_pool[0]), noise=False)
_a1 = W.action(_xb, W.skill(_xb, *W.train_pool[0]), noise=True)
FLOOR = W.noise_floor(_a1)
ok(abs(np.mean(np.square(_xa := (_a1 - _a0))) - L.SIG_ETA ** 2) < 5e-3,
   f"噪声项可分离：E‖η‖² 实测 {float(np.mean(np.square(_xa))):.4f} ≈ σ_η²={L.SIG_ETA ** 2:.4f}")
print(f"  噪声下界（相对动作误差）= {FLOOR:.4f}")


# ================================================================== A 组
section("A 组：分解换来什么（同一批演示，同一预算）")
banner("A1：训练步数扫描（C=8 个训练约束，3 个 seed 平均，未见场景）")
CS8 = W.train_pool[:8]
STEPS = [250, 500, 1000, 2000, 4000]
CURVE = {}
for st in STEPS:
    acc = np.zeros(4)
    for sd in (1, 2, 3):
        f, h, l = train_pair(CS8, st, seed=sd)
        acc += np.asarray(evaluate(f, h, l, CS8)) / 3.0
    CURVE[st] = acc
    print(f"  steps={st:5d}  技能准确率 flat={acc[0]:.3f} hier={acc[1]:.3f} "
          f"(+{acc[1] - acc[0]:+.3f})   动作相对误差 flat={acc[2]:.3f} hier={acc[3]:.3f}")

_f, _h, _l = train_pair(CS8, 4000, seed=1)
ORC = evaluate(_f, _h, _l, CS8, oracle=True)
print(f"  参照：只给低层真值技能（高层上界）动作误差 = {ORC[3]:.4f}，噪声下界 = {FLOOR:.4f}")

ok(all(CURVE[st][1] > CURVE[st][0] for st in STEPS),
   "分层在技能准确率上全程领先扁平（5 个步数档位全部为正）")
ok(CURVE[250][2] < CURVE[250][3] and CURVE[4000][2] > CURVE[4000][3],
   "动作 MSE 上扁平先赢后输：早期靠「几个技能原型的加权叠加」压低 MSE")
ok(CURVE[2000][1] >= CURVE[4000][0] - 0.01,
   f"分层 2000 步（{CURVE[2000][1]:.3f}）≈ 扁平 4000 步（{CURVE[4000][0]:.3f}）：数据效率约 2 倍")
ok(ORC[3] < FLOOR * 1.6 and ORC[3] > FLOOR * 0.6,
   f"高层给真值时低层误差 {ORC[3]:.4f} 紧贴噪声下界 {FLOOR:.4f}：瓶颈确实在高层")
GAP8 = CURVE[4000][1] - CURVE[4000][0]
print(f"  4000 步的准确率差距 = {GAP8:+.3f}")

banner("A2：训练约束数 C 的影响（固定 2000 步）")
GAPC = {}
for C in (2, 4, 8):
    cs = W.train_pool[:C]
    acc = np.zeros(4)
    for sd in (1, 2, 3):
        f, h, l = train_pair(cs, 2000, seed=sd)
        acc += np.asarray(evaluate(f, h, l, cs)) / 3.0
    GAPC[C] = acc
    print(f"  C={C}  flat={acc[0]:.3f} hier={acc[1]:.3f} (+{acc[1] - acc[0]:+.3f})  "
          f"mse flat={acc[2]:.3f} hier={acc[3]:.3f}")
gaps = [GAPC[c][1] - GAPC[c][0] for c in (2, 4, 8)]
ok(min(gaps) > 0.05,
   f"三种约束规模下分层都领先 5 个百分点以上（实测 {['%+.3f' % g for g in gaps]}）")
ok(abs(max(gaps) - min(gaps)) < 0.05,
   "差距量级不随约束数单调放大：优势来自「拆成离散分类 + 连续回归两步」，不是来自组合爆炸")


# ================================================================== B 组
section("B 组：高层多久重算一次（1 Hz 与事件触发都是设计参数，不是随手写的）")

L_SUB = 3          # 一个原子技能需要连续执行 3 步（论文：原子技能持续 1–3 秒）
N_SUB = 10         # 一个 episode 有 10 个子任务
T_MAX = 150        # 步数预算（足够跑完，用来量「要多少步」而不是「跑不完多少」）
EPS_HI = 0.15      # 高层每次调用说错的概率
N_EP = 400         # episode 数
RS = [1, 2, 3, 4, 6, 12]


def run_episode(R, rng, *, eps=EPS_HI, event=False, interrupt=False):
    """跑一个 episode。

    低层一次只认一条命令；命令一变，手上这段就重新起（低层不保证能接上）。
    高层每 R 步重算一次；`event=True` 时用户一插话就额外触发一次重算。
    `interrupt=True` 时，插话时刻随机落在某个子任务中间，之后剩余计划整个换掉。
    """
    truth = rng.integers(0, S - 1, size=N_SUB)
    t_int = int(rng.integers(L_SUB * 3, L_SUB * 7)) if interrupt else None
    k = run_ = 0
    cmd = -1
    calls = 0
    lat = None
    wasted = 0
    fired = False
    t = 0
    for t in range(T_MAX):
        changed = t_int is not None and t >= t_int
        if changed and not fired:                       # 插话到达 → 重排剩余计划
            truth[k:] = rng.integers(0, S - 1, size=N_SUB - k)
            fired = True
        if t % R == 0 or (event and t == t_int):        # 周期重算 / 事件触发
            calls += 1
            want = truth[min(k, N_SUB - 1)]
            c = want if rng.random() > eps else int(rng.integers(0, S - 1))
            if c != cmd:
                cmd, run_ = c, 0                        # 换命令：手上这段作废
        if changed:
            if cmd != truth[min(k, N_SUB - 1)]:
                wasted += 1                             # 还在按旧计划动手
            elif lat is None:
                lat = t - t_int                         # 命令首次跟上新指令
        run_ += 1
        if cmd == truth[min(k, N_SUB - 1)] and run_ >= L_SUB:
            k += 1
            run_ = 0
            if k >= N_SUB:
                break
    return dict(steps=t + 1, calls=calls, done=k,
                lat=(lat if lat is not None else T_MAX), wasted=wasted)


def sweep(**kw):
    out = {}
    for R in RS:
        rg = np.random.default_rng(kw.get("seed0", 2000) + R)
        acc = np.zeros(4)
        for _ in range(N_EP):
            r = run_episode(R, rg, **{k: v for k, v in kw.items() if k != "seed0"})
            acc += np.asarray([r["steps"], r["calls"], r["lat"], r["wasted"]]) / N_EP
        out[R] = acc
    return out


banner("B1：重算周期 R 的取舍（没有中途纠正，高层 15% 概率说错）")
B1 = sweep()
STEP, CALL = {}, {}
for R in RS:
    STEP[R], CALL[R] = B1[R][0], B1[R][1]
    print(f"  R={R:2d}  完成 {N_SUB} 个子任务要 {B1[R][0]:6.1f} 步   高层调用 {B1[R][1]:6.1f} 次")
R_STAR = min(STEP, key=STEP.get)
ok(R_STAR == L_SUB,
   f"最省步数的重算周期 = 子任务长度 L={L_SUB}（实测最优 R={R_STAR}，{STEP[R_STAR]:.1f} 步）")
ok(STEP[1] > STEP[L_SUB] and CALL[1] > CALL[L_SUB],
   f"每步重算被支配：R=1 要 {STEP[1]:.1f} 步 / {CALL[1]:.1f} 次调用，"
   f"R={L_SUB} 只要 {STEP[L_SUB]:.1f} 步 / {CALL[L_SUB]:.1f} 次 —— 步数和算力两个都更差")
ok(STEP[2] > STEP[L_SUB] and CALL[2] > CALL[L_SUB],
   f"周期不与子任务长度对齐同样被支配：R=2 要 {STEP[2]:.1f} 步 / {CALL[2]:.1f} 次调用")
ok(all(STEP[RS[i]] < STEP[RS[i + 1]] for i in range(RS.index(L_SUB), len(RS) - 1)),
   "R 超过子任务长度后步数单调上升：两次高层调用之间已经该换技能了，低层还在做旧的")
ok(all(CALL[RS[i]] > CALL[RS[i + 1]] for i in range(RS.index(4), len(RS) - 1)),
   f"R≥4 后高层调用次数单调下降（{CALL[4]:.1f} → {CALL[6]:.1f} → {CALL[12]:.1f}）："
   "R>L 是拿「多花墙钟时间」换「少花高层算力」，落在帕累托前沿上")
ok(CALL[1] / CALL[L_SUB] > 3.0,
   f"R={L_SUB} 相比每步重算省 {CALL[1] / CALL[L_SUB]:.1f} 倍高层算力"
   f"（{CALL[1]:.0f} → {CALL[L_SUB]:.0f} 次）")

banner("B2：中途纠正 —— 周期重算 vs 事件触发（插话时刻随机落在子任务中间）")
B2P = sweep(interrupt=True, seed0=3100)
B2E = sweep(interrupt=True, event=True, seed0=3100)
LAT_P, LAT_E, WST_P, WST_E = {}, {}, {}, {}
for R in RS:
    LAT_P[R], WST_P[R] = B2P[R][2], B2P[R][3]
    LAT_E[R], WST_E[R] = B2E[R][2], B2E[R][3]
    print(f"  R={R:2d}  纯周期：延迟 {LAT_P[R]:5.2f} 步 浪费 {WST_P[R]:5.2f} 步   "
          f"事件触发：延迟 {LAT_E[R]:4.2f} 步 浪费 {WST_E[R]:4.2f} 步")
ok(all(LAT_E[R] <= 1.0 for R in RS),
   "事件触发把纠正延迟压到 1 步以内：收到新指令立刻重算，与周期无关")
ok(all(LAT_E[R] < LAT_P[R] for R in RS if R > 1),
   "R>1 时事件触发的延迟严格小于纯周期重算")
ok(LAT_P[12] > 5.0 and abs(LAT_P[6] - 3.0) < 1.0,
   f"纯周期重算的延迟约为 R 的一半（R=6 → {LAT_P[6]:.2f} 步，R=12 → {LAT_P[12]:.2f} 步）")
ok(WST_P[12] > WST_E[12] + 3.0,
   f"R=12 时事件触发少浪费 {WST_P[12] - WST_E[12]:.1f} 步"
   f"（{WST_P[12]:.1f} → {WST_E[12]:.1f}）：重算周期可以放慢，代价由另一条触发通路补上")


# ================================================================== C 组
section("C 组：接口的脆弱处 —— 高层说出的措辞必须落在低层认识的集合里")

_f, _h, _l = train_pair(CS8, 4000, seed=1)
_rng = np.random.default_rng(55)
_xb = W._sample_scene(4000, _rng)
_s = W.skill(_xb, *CS8[0])
_a = W.action(_xb, _s, noise=False)


def paraphrase(sk, kappa, rng_u=None):
    """高层用了同义改写：标签向量偏离规范 one-hot。kappa=0 就是规范措辞。"""
    base = onehot(sk, S)
    g = np.random.default_rng(7) if rng_u is None else rng_u
    u = g.normal(size=(S, S))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    kap = np.atleast_1d(np.asarray(kappa, dtype=float))
    if kap.size == len(base):
        kap = kap[:, None]
    v = base + kap * u[np.asarray(sk)]
    return v / np.linalg.norm(v, axis=1, keepdims=True)


banner("C1：低层对标签扰动的敏感度")
KS = [0.0, 0.25, 0.5, 1.0, 2.0]
ERR_K = {}
for kp in KS:
    _, o = fwd(_l, np.concatenate([paraphrase(_s, kp), _xb], axis=1))
    ERR_K[kp] = W.rel_err(o, _a)
    print(f"  kappa={kp:4.2f}  动作相对误差 = {ERR_K[kp]:.4f}")
_rs = np.random.default_rng(9).integers(0, S, size=len(_s))
_, _or = fwd(_l, np.concatenate([onehot(_rs, S), _xb], axis=1))
ERR_RAND = W.rel_err(_or, _a)
print(f"  参照：标签完全随机 = {ERR_RAND:.4f}，规范措辞 = {ERR_K[0.0]:.4f}")
ok(ERR_K[0.0] < 0.02,
   f"规范措辞下低层几乎无误差（{ERR_K[0.0]:.4f}）：技能原型 + 场景修正都学会了")
ok(all(ERR_K[KS[i]] < ERR_K[KS[i + 1]] for i in range(len(KS) - 1)),
   "误差随措辞偏离程度单调上升")
ok(ERR_K[2.0] > 20 * ERR_K[0.0],
   f"kappa=2 的误差是规范措辞的 {ERR_K[2.0] / ERR_K[0.0]:.0f} 倍，"
   f"已到「标签随机」{ERR_RAND:.3f} 的 {ERR_K[2.0] / ERR_RAND:.0%}：低层彻底认不出来")

banner("C2：端到端 —— 高层有多常说改写，以及标签扰动增强能不能救")
RHOS = [0.0, 0.1, 0.2, 0.4, 0.8]
E2E = {}
for rho in RHOS:
    e = (1 - rho) * ERR_K[0.0] + rho * ERR_K[1.0]
    E2E[rho] = e
    print(f"  改写比例 rho={rho:4.2f}  端到端误差 = {e:.4f}")
ok(all(E2E[RHOS[i]] < E2E[RHOS[i + 1]] for i in range(len(RHOS) - 1)),
   "端到端误差随改写比例单调上升（每 10% 的改写都要折算成动作误差）")

_rng2 = np.random.default_rng(3)
_lo_aug = init_mlp(S + D_X, D_A, _rng2)
_oA = Adam(_lo_aug, 3e-3)
for _ in range(4000):
    cv, xb2, s2, a2 = sample_batch(W, CS8, 9, _rng2)
    lab = onehot(s2, S)
    m = _rng2.random(len(s2)) < 0.3                     # 30% 的标签做扰动增强
    if m.any():
        lab[m] = paraphrase(s2[m], _rng2.uniform(0.3, 1.2, size=int(m.sum())),
                            rng_u=np.random.default_rng(int(_rng2.integers(1 << 30))))
    H, o = fwd(_lo_aug, np.concatenate([lab, xb2], axis=1))
    _, d = mse_grad(o, a2)
    _oA.step(_lo_aug, bwd(_lo_aug, np.concatenate([lab, xb2], axis=1), H, d))
_, _oa = fwd(_lo_aug, np.concatenate([paraphrase(_s, 1.0), _xb], axis=1))
ERR_AUG = W.rel_err(_oa, _a)
_, _ob = fwd(_lo_aug, np.concatenate([onehot(_s, S), _xb], axis=1))
ERR_AUG0 = W.rel_err(_ob, _a)
print(f"  增强训练后：kappa=1 的误差 {ERR_K[1.0]:.4f} → {ERR_AUG:.4f}；"
      f"规范措辞 {ERR_AUG0:.4f}")
ok(ERR_AUG < ERR_K[1.0] * 0.7,
   f"低层训练时对标签做扰动增强，kappa=1 的退化从 {ERR_K[1.0]:.4f} 降到 {ERR_AUG:.4f}")
ok(ERR_AUG0 < ERR_K[0.0] + 0.02,
   f"增强的代价很小：规范措辞下 {ERR_AUG0:.4f} 与不增强的 {ERR_K[0.0]:.4f} 同量级")


# ================================================================== D 组
section("D 组：合成标注喂谁 —— 高层的数据分布才是瓶颈")

P_LAB = np.array([0.85] + [0.15 / 7] * 7)          # 人标注的演示：一个约束占八成半
CS_SYN = CS8 + W.test_c                            # 合成标注：单位约束均匀 + 复合约束
P_SYN = np.full(len(CS_SYN), 1.0 / len(CS_SYN))

FREQ = [CS8[0]]
RARE = CS8[4:]
COMP = W.test_c
GROUPS = {"常见单位约束": FREQ, "稀有单位约束": RARE, "复合约束": COMP}

RES_D, RES_DO = {}, {}
P_MIX = 0.6 * np.concatenate([P_LAB, np.zeros(len(W.test_c))]) + 0.4 * P_SYN
for tag, probs in (("labeled", P_LAB), ("labeled+syn", P_MIX)):
    cs = CS8 if tag == "labeled" else CS_SYN
    acc = {g: np.zeros(4) for g in GROUPS}
    orc = {g: 0.0 for g in GROUPS}
    for sd in (1, 2, 3):
        f, h, l = train_pair(cs, 2000, seed=sd, probs=probs)
        for g, gcs in GROUPS.items():
            acc[g] += np.asarray(evaluate(f, h, l, gcs, n=400)) / 3.0
            orc[g] += evaluate(f, h, l, gcs, n=400, oracle=True)[3] / 3.0
    RES_D[tag] = acc
    RES_DO[tag] = orc
    line = "  ".join(f"{g} {acc[g][1]:.3f}" for g in GROUPS)
    print(f"  {tag:12s} 高层技能准确率：{line}")

d_freq = RES_D["labeled+syn"]["常见单位约束"][1] - RES_D["labeled"]["常见单位约束"][1]
d_rare = RES_D["labeled+syn"]["稀有单位约束"][1] - RES_D["labeled"]["稀有单位约束"][1]
d_comp = RES_D["labeled+syn"]["复合约束"][1] - RES_D["labeled"]["复合约束"][1]
print(f"  合成数据带来的变化：常见 {d_freq:+.3f} / 稀有 {d_rare:+.3f} / 复合 {d_comp:+.3f}")
ok(d_rare > 0.03,
   f"合成标注主要补的是稀有约束（{RES_D['labeled']['稀有单位约束'][1]:.3f} → "
   f"{RES_D['labeled+syn']['稀有单位约束'][1]:.3f}，{d_rare:+.3f}）")
ok(d_comp > 0.10,
   f"复合（多从句）指令几乎全靠合成标注（{RES_D['labeled']['复合约束'][1]:.3f} → "
   f"{RES_D['labeled+syn']['复合约束'][1]:.3f}，{d_comp:+.3f}）")
ok(d_freq < 0 and abs(d_freq) < 0.10,
   f"代价落在头部：最常见的那个约束降了 {abs(d_freq):.3f}（合成数据稀释了它的样本份额）")
ok(d_comp > d_freq,
   "合成数据的收益集中在「人标注演示覆盖不到的指令」，不是均匀抬高所有指令")

banner("D2：合成标注只动高层不动低层（直接给低层真值技能时两组的误差）")
for g in GROUPS:
    print(f"  {g:8s} 低层单独误差  labeled {RES_DO['labeled'][g]:.4f}   "
          f"labeled+syn {RES_DO['labeled+syn'][g]:.4f}")
ok(all(abs(RES_DO["labeled+syn"][g] - RES_DO["labeled"][g]) < 0.03 for g in GROUPS),
   "两组在「直接给真值技能」时低层误差一致（差 < 0.03）："
   "合成标注改变的是高层看到的指令分布，低层的技能原型没被动过")


# ================================================================== 汇总
section(f"汇总：{N_ASSERT} 条断言全部通过")
print("\n  A 组：分层 vs 扁平（C=8 个训练约束，3 个 seed 平均，未见场景）")
for st in STEPS:
    print(f"    {st:5d} 步   技能准确率 flat {CURVE[st][0]:.3f} / hier {CURVE[st][1]:.3f}"
          f"   动作相对误差 flat {CURVE[st][2]:.3f} / hier {CURVE[st][3]:.3f}")
print(f"    数据效率：hier 2000 步 {CURVE[2000][1]:.3f} ≈ flat 4000 步 {CURVE[4000][0]:.3f}")
print(f"    上界参照：高层给真值时低层误差 {ORC[3]:.4f}，噪声下界 {FLOOR:.4f}")
print(f"  B 组：重算周期 R（子任务长度 L={L_SUB}，高层 {EPS_HI:.0%} 概率说错）")
for R in RS:
    print(f"    R={R:2d}   完成 {N_SUB} 个子任务要 {STEP[R]:6.1f} 步，高层调用 {CALL[R]:6.1f} 次")
print(f"    最优 R = {R_STAR}（= L）；R=1 与 R=2 在步数和调用数上都被 R=L 支配")
print(f"    中途纠正延迟：纯周期 {LAT_P[12]:.1f} 步（R=12）vs 事件触发 {LAT_E[12]:.1f} 步")
print(f"  C 组：高层措辞偏离规范 one-hot 的程度 kappa")
for kp in KS:
    print(f"    kappa={kp:4.2f}   动作相对误差 {ERR_K[kp]:.4f}")
print(f"    参照：标签完全随机 {ERR_RAND:.4f}；标签扰动增强后 kappa=1 从 "
      f"{ERR_K[1.0]:.4f} 降到 {ERR_AUG:.4f}")
print(f"  D 组：合成标注带来的高层技能准确率变化（2000 步，3 seed 平均）")
for g in GROUPS:
    print(f"    {g:8s}  labeled {RES_D['labeled'][g][1]:.3f}  →  "
          f"labeled+syn {RES_D['labeled+syn'][g][1]:.3f}   "
          f"({RES_D['labeled+syn'][g][1] - RES_D['labeled'][g][1]:+.3f})")
print(f"\n  断言总数：{N_ASSERT}")
