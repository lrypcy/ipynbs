# -*- coding: utf-8 -*-
"""π0.5 companion lab：合成世界 + 单模型两阶段 / 共训练的公共内核。

这一篇在 Hi Robot 探针（`../hi-robot/`）的世界上加三个结构，分别对应 π0.5
（arXiv:2504.16054）的三件事：

1. **外观 → 语义的间接层**（对应 web 数据共训练）
   场景里的每个物体是一个**类型** t ∈ 0..T-1，观测给的是外观向量 φ_t，
   真值决策用的是语义属性 z_t ∈ {±1}^M。模型要先学一个槽编码器
   ψ: φ → ẑ，才能做「按属性筛选」的技能选择。web 数据（WD）直接给
   (φ_t, z_t) 对——**包括机器人演示里没出现过的类型**（web-only 类型）。
   没有 WD 时，ψ 只能从技能标签里间接学，web-only 类型的语义是瞎猜的。

2. **环境上下文向量**（对应多环境 scaling）
   每个环境 e 有一个上下文 κ_e ∈ R^4，动作里带一个环境相关项
   (v·x)·u(κ_e)，u 是一个**固定的非线性映射**（κ 进观测——新家是看得见的，
   u(·) 要从数据里学）。环境数少时 u(·) 只能背下来那几个 κ，环境数多了
   才能泛化到新 κ——这就是论文 Fig.8 那条 3→104 曲线的合成对应物。

3. **implicit HL 对照**（对应论文 Fig.13）
   「共训练时带上子任务标签、部署时只跑低层」——动作头直接条件于原始
   约束而不是预测出的技能。对照：joint（两阶段推理）/ implicit / flat
   （连子任务监督都没有）。

符号沿用 04 篇：约束 c 是 M×3 逐位 one-hot，技能 s* 由 W_g 打分选出，
动作 a* = μ_s + B x_z + 环境项 + η。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 世界常量
M = 4            # 每个类型的语义属性位数（±1）
T = 10           # 物体类型数：7 个「机器人演示里出现过」+ 3 个「web-only」
K = 5            # 每个场景的槽位数
S = 6            # 技能数
D_A = 8          # 动作维度
DONE = S - 1     # 「收工」伪技能
D_C = M * 3      # 约束 one-hot
D_PHI = 8        # 外观向量维度（观测给这个，不给 z）
D_X = K * (M + 1)   # 语义场景向量：每槽 [z (M), done (1)]
D_CTX = 4        # 环境上下文维度
N_SEEN = 7       # 机器人演示覆盖的类型数
HID = 64         # 各网络隐藏宽度
SIG_ETA = 0.15   # 不可预测噪声

N_MM_ENV = 8     # MM 数据的环境数（约 400 小时 / 目标平台的量级）
N_ME_ENV = 60    # ME 数据额外带来的环境数
N_TEST_ENV = 6   # 测试环境数


def _orthonormal(rng, n, d):
    """n 个 d 维单位正交行（n ≤ d）。"""
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q[:, :n].T


def _simplex(k: int, d: int) -> np.ndarray:
    """k 行正单纯形（k = d+1）：范数相同、两两内积相同、和为 0。"""
    C = np.eye(k) - np.full((k, k), 1.0 / k)
    ev, U = np.linalg.eigh(C)
    return U[:, -d:] * np.sqrt(ev[-d:])


def _all_z(m: int = M) -> np.ndarray:
    """枚举 {±1}^m 的全部 2^m 个属性模式。"""
    g = np.array(np.meshgrid(*[[-1.0, 1.0]] * m, indexing="ij")).reshape(m, -1).T
    return g


class World:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rng = rng

        # ---- 类型与语义：7 个 seen 类型 + 3 个 web-only 类型，属性模式互不相同
        patterns = _all_z()
        pick = rng.permutation(len(patterns))[:T]
        self.z_types = patterns[pick]                     # (T, M)
        self.seen_types = np.arange(N_SEEN)
        self.web_types = np.arange(N_SEEN, T)
        # 外观：10 个彼此拉开距离的随机单位向量（ψ 要从 φ 恢复 z）
        phi = rng.normal(size=(T, D_PHI))
        phi /= np.linalg.norm(phi, axis=1, keepdims=True)
        self.phi_types = phi

        # ---- 动作内核（同 04 篇）
        self.B = rng.normal(size=(D_A, D_X))
        self._calibrate_B()
        self.mu = _orthonormal(rng, S, D_A)
        self.W_g = self._balanced_scorer(rng)

        # ---- 环境上下文：u(κ) = Cκ + tanh(κ W₂ᵀ) C₂（带固定的非线性分量，
        #      让「从很少几个环境学 u(·)」真的学不出来），动作环境项 = (v·x)·u
        self.C = rng.normal(size=(D_A, D_CTX))
        self.W2 = rng.normal(size=(16, D_CTX))
        self.C2 = rng.normal(size=(D_A, 16))
        self.v = rng.normal(size=D_X)
        self.v /= np.linalg.norm(self.v)
        self._calibrate_C()

        # ---- 约束池（同 04 篇）
        self.single = [((j,), ()) for j in range(M)] + [((), (j,)) for j in range(M)]
        self.composite = [((j,), ((j + 1) % M,)) for j in range(M)]
        order = rng.permutation(len(self.single))
        self.train_pool = [self.single[i] for i in order]

    # ------------------------------------------------------------- 类型语义
    def _balanced_scorer(self, rng, tries: int = 4000):
        """W_g：单纯形 + 旋转，要求 argmax 在 {±1}^M 上分布均衡（同 04 篇）。"""
        base = _simplex(S - 1, M)
        grid = _all_z()
        best, best_gap = None, -1.0
        for _ in range(tries):
            Q, _ = np.linalg.qr(rng.normal(size=(M, M)))
            W = base @ Q
            cnt = np.bincount(np.argmax(grid @ W.T, axis=1), minlength=S - 1)
            p = cnt / cnt.sum()
            if p.min() > best_gap:
                best, best_gap = W, p.min()
            if p.min() >= 0.15 and p.max() <= 0.30:
                return W
        return best

    def _calibrate_B(self):
        """缩放 B 使 E‖Bx‖² = 0.25（逐维平均口径，目标除以 D_A——04 篇踩过的坑）。"""
        x = self.sample_semantic(8192, np.arange(T))[0]
        cur = float(np.mean(np.square(x @ self.B.T)))
        self.B *= np.sqrt((0.25 / D_A) / max(cur, 1e-12))

    def env_u(self, kap: np.ndarray) -> np.ndarray:
        """环境项的方向向量 u(κ) = Cκ + tanh(κ W₂ᵀ) C₂（固定的非线性映射）。"""
        return kap @ self.C.T + np.tanh(kap @ self.W2.T) @ self.C2.T

    def _calibrate_C(self):
        """缩放 (C, C2) 使环境项 E‖(v·x)·u(κ)‖² = 0.25（与 Bx 项同量级）。"""
        x = self.sample_semantic(8192, np.arange(T))[0]
        kap = self.rng.normal(size=(len(x), D_CTX))
        env = (x @ self.v)[:, None] * self.env_u(kap)
        cur = float(np.mean(np.square(env)))
        sc = np.sqrt((0.25 / D_A) / max(cur, 1e-12))
        self.C *= sc
        self.C2 *= sc

    # ------------------------------------------------------------- 场景采样
    def sample_scene(self, n: int, types, rng=None):
        """采 n 个场景。返回 (z (n,K,M), types (n,K), done (n,K), 槽位类型表)。"""
        g = self.rng if rng is None else rng
        ty = g.choice(np.asarray(types), size=(n, K), replace=True)
        z = self.z_types[ty]                              # (n, K, M)
        p_done = g.uniform(0.0, 0.6, size=(n, 1))
        done = np.where(g.random((n, K)) < p_done, 1.0, -1.0)
        return z, ty, done

    def semantic_vec(self, z, done) -> np.ndarray:
        """(z, done) → 语义场景向量 x（模型与真值共用的表示）。"""
        n = z.shape[0]
        return np.concatenate([z.reshape(n, K * M), done.reshape(n, K)], axis=1)

    def sample_semantic(self, n: int, types, rng=None):
        """直接返回语义场景 x（D 组 / 标定用，跳过外观）。"""
        z, _, done = self.sample_scene(n, types, rng)
        return self.semantic_vec(z, done), z, done

    def observe(self, ty: np.ndarray) -> np.ndarray:
        """槽位类型表 → 外观观测 (n, K, D_PHI)。"""
        return self.phi_types[ty]

    # ------------------------------------------------------------- 真值
    def target(self, x: np.ndarray, r, rbar) -> np.ndarray:
        """满足约束且未处理的、下标最小的槽位；没有则 −1。"""
        t = np.atleast_2d(x).reshape(-1, K, M + 1)
        z, done = t[:, :, :M], t[:, :, M]
        ok = np.ones(z.shape[:2], dtype=bool)
        for j in r:
            ok &= z[:, :, j] > 0
        for j in rbar:
            ok &= z[:, :, j] < 0
        ok &= done < 0
        has = ok.any(axis=1)
        return np.where(has, np.argmax(ok, axis=1), -1)

    def skill(self, x: np.ndarray, r, rbar) -> np.ndarray:
        return self.skill_vec(x, self.cvec(r, rbar)[None, :])[0]

    def skill_vec(self, x: np.ndarray, cv: np.ndarray) -> np.ndarray:
        """向量化真值技能：cv (n, M*3) 逐位 one-hot，每个样本一个约束。"""
        t = np.atleast_2d(x).reshape(-1, K, M + 1)
        z, done = t[:, :, :M], t[:, :, M]
        c3 = cv.reshape(-1, M, 3)
        r1, r0 = c3[:, :, 0], c3[:, :, 1]                 # (n, M)
        ok1 = (r1[:, None, :] > 0) & (z > 0)
        ok0 = (r0[:, None, :] > 0) & (z < 0)
        off = (r1[:, None, :] == 0) & (r0[:, None, :] == 0)
        ok = (ok1 | ok0 | off).all(axis=2)                # (n, K)
        ok &= done < 0
        has = ok.any(axis=1)
        idx = np.where(has, np.argmax(ok, axis=1), 0)
        zt = z[np.arange(len(z)), idx]                    # (n, M)
        sk = np.argmax(zt @ self.W_g.T, axis=-1)
        return np.where(has, sk, DONE)

    def action(self, x: np.ndarray, s: np.ndarray, kap: np.ndarray,
               *, noise: bool = True, rng=None) -> np.ndarray:
        """真值动作 a* = μ_s + Bx + (v·x)·u(κ) + η。"""
        x = np.atleast_2d(x)
        kap = np.atleast_2d(kap)
        env = (x @ self.v)[:, None] * self.env_u(kap)
        a = self.mu[np.asarray(s).reshape(-1)] + x @ self.B.T + env
        if noise:
            g = self.rng if rng is None else rng
            a = a + SIG_ETA * g.normal(size=a.shape)
        return a

    def oracle_env_gap(self, x: np.ndarray, kap: np.ndarray) -> float:
        """环境项的相对量级（给「完全不看 κ 的模型」当前场景的误差下界）。"""
        x = np.atleast_2d(x)
        kap = np.atleast_2d(kap)
        env = (x @ self.v)[:, None] * self.env_u(kap)
        return float(np.mean(np.square(env)))

    def rel_err(self, a_hat, a_star) -> float:
        num = float(np.mean(np.square(a_hat - a_star)))
        den = float(np.mean(np.square(a_star - a_star.mean(axis=0))))
        return num / max(den, 1e-12)

    def cvec(self, r, rbar) -> np.ndarray:
        v = np.zeros((M, 3))
        rset, rbset = set(r), set(rbar)
        for j in range(M):
            v[j, 0 if j in rset else (1 if j in rbset else 2)] = 1.0
        return v.reshape(-1)

    def cvec_batch(self, cs) -> np.ndarray:
        return np.stack([self.cvec(*c) for c in cs])


# ---------------------------------------------------------------- 数据批
def make_batch(world: World, cs, types, kaps, n_per: int, rng):
    """混合采样一批 (约束, 场景, κ)，全部向量化。

    返回 dict：cv (n,D_C)、obs (n,K,D_PHI)、x (n,D_X)、done (n,K)、
    kap (n,D_CTX)、s* (n,)、a* (n,D_A)。每个样本独立抽约束、类型与 κ。
    """
    n = len(cs) * n_per
    ty = rng.choice(np.asarray(types), size=(n, K), replace=True)
    z = world.z_types[ty]
    p_done = rng.uniform(0.0, 0.6, size=(n, 1))
    done = np.where(rng.random((n, K)) < p_done, 1.0, -1.0)
    x = world.semantic_vec(z, done)
    ci = rng.integers(0, len(cs), size=n)
    cv = world.cvec_batch([cs[i] for i in ci])
    ki = rng.integers(0, len(kaps), size=n)
    kap = kaps[ki]
    s = world.skill_vec(x, cv)
    a = world.action(x, s, kap, noise=True, rng=rng)
    return dict(cv=cv, obs=world.observe(ty), x=x, done=done, ty=ty,
                kap=kap, s=s, a=a)


# ---------------------------------------------------------------- 小网络
def init_mlp(din: int, dout: int, rng, hid: int = HID):
    s1 = np.sqrt(2.0 / din)
    s2 = np.sqrt(1.0 / hid)
    return [rng.normal(0, s1, (din, hid)), np.zeros(hid),
            rng.normal(0, s2, (hid, dout)), np.zeros(dout)]


def fwd(p, X):
    H = np.tanh(X @ p[0] + p[1])
    return H, H @ p[2] + p[3]


def bwd(p, X, H, dY):
    """参数梯度。约定：dY 已经是 dL/dY（含全部归一化），这里不再除以 n。

    （hirobot_lib 的版本在 bwd 里再除一次 n，恰好被「批大小 = 动作维度」
    的自校验批次掩盖；本篇批次 128 ≠ 动作维度 8，必须把口径分开。）
    """
    dH = (dY @ p[2].T) * (1.0 - H * H)
    return [X.T @ dH, dH.sum(axis=0), H.T @ dY, dY.sum(axis=0)]


def mse_grad(out, target):
    """损失 = 每样本向量范数平方、对 batch 取平均（flow matching 的 ‖·‖² 口径）。

    这样打印的损失与 dL/dout 严格对应：d = 2(out-target)/n。
    （np.mean(np.square(...)) 会对输出维也取平均，和这个梯度差一个 D_A。）
    """
    n = out.shape[0]
    diff = out - target
    loss = float(np.mean(np.sum(np.square(diff), axis=-1)))
    return loss, 2.0 * diff / n


def softmax_ce(logits, y):
    n = logits.shape[0]
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    pr = e / e.sum(axis=-1, keepdims=True)
    loss = float(-np.mean(np.log(pr[np.arange(n), y] + 1e-12)))
    d = pr.copy()
    d[np.arange(n), y] -= 1.0
    return loss, d / n


class Adam:
    def __init__(self, params, lr: float = 3e-3):
        self.t = 0
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            p -= self.lr * (self.m[i] / (1 - b1 ** self.t)) / (
                np.sqrt(self.v[i] / (1 - b2 ** self.t)) + eps)


def onehot(idx: np.ndarray, n: int) -> np.ndarray:
    o = np.zeros((len(idx), n))
    o[np.arange(len(idx)), np.asarray(idx)] = 1.0
    return o


# ---------------------------------------------------------------- 模型
class Pi05Net:
    """单模型两阶段：槽编码器 ψ（语义）+ 共享主干 g + 技能头 + 动作头。

    变体开关
    --------
    use_skill_input : 动作头是否条件于技能（False = implicit / flat 的
        「只跑低层」形态，动作头直接看约束）。
    train_skill     : 是否训练技能头（False = flat，连子任务监督都没有）。
    """

    def __init__(self, world: World, rng, *, use_skill_input: bool = True,
                 train_skill: bool = True):
        self.w = world
        self.use_skill_input = use_skill_input
        self.train_skill = train_skill
        self.psi = init_mlp(D_PHI, M, rng)                     # 外观 → 语义
        self.trunk = init_mlp(D_C + D_X, HID, rng)
        self.skill_head = init_mlp(HID, S, rng) if train_skill else None
        din = HID + (S if use_skill_input else D_C) + D_CTX
        self.act_head = init_mlp(din, D_A, rng)
        self.params = self.psi + self.trunk + self.act_head
        if self.skill_head is not None:
            self.params = self.params + self.skill_head

    # ---- 前向
    def zhat(self, obs):
        """外观 → 每槽语义 ẑ（tanh 压到 ±1）。"""
        n, kk, _ = obs.shape
        _, out = fwd(self.psi, obs.reshape(n * kk, D_PHI))
        return np.tanh(out).reshape(n, kk, M)

    def scene_rep(self, obs, done):
        """(obs, done) → 模型视角的语义场景向量。"""
        n = obs.shape[0]
        return np.concatenate([self.zhat(obs).reshape(n, K * M),
                               done.reshape(n, K)], axis=1)

    def features(self, cv, xs):
        _, h = fwd(self.trunk, np.concatenate([cv, xs], axis=1))
        return h

    def skill_logits(self, feat):
        _, out = fwd(self.skill_head, feat)
        return out

    def act_out(self, feat, cv, xs, kap, s):
        if self.use_skill_input:
            z_in = np.concatenate([feat, onehot(s, S), kap], axis=1)
        else:
            z_in = np.concatenate([feat, cv, kap], axis=1)
        _, out = fwd(self.act_head, z_in)
        return out
