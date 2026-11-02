# -*- coding: utf-8 -*-
"""Hi Robot companion lab：合成世界 + 分层 / 扁平策略的公共内核。

世界设定
--------
桌上有 K 个物体，每个物体有 M 位二值属性（可回收 / 是垃圾 / 黄色 / 易碎 …），
编码成 ±1。用户指令是一个**作用在属性位上的筛选谓词**（下称「约束」c）：

    把「满足 c 的、还没处理的」物体逐个收走。

一个约束写成 (r, rbar)：r 里的位必须为 +1，rbar 里的位必须为 −1，其余位不关心。
表示成一个 M×3 的逐位 one-hot（每位取 require-1 / require-0 / 不关心），
这样**每个属性位是一个独立的输入槽**——组合结构就是从这里来的：
训练只给单位约束，测试给「两位合取」的复合约束。

低层有 S 个技能（0..S-2 是「对目标物体做某种操作」，S-1 是 done）。
技能只决定**这一段动作的原型** μ_s，与用户约束无关。
高层要做的是：看 (约束, 桌面状态) → 判断这一步该执行哪个技能。

真值
----
  target = 满足约束且未处理的、下标最小的物体
  s*     = DONE（若无 target）否则 argmax_s (W_g @ z_target)_s，s ∈ 0..S-2
  a*     = μ_{s*} + B x + η            η 是不可预测噪声，构成动作误差下界

两个待学的映射
--------------
  高层 h: (c, x) → s        交叉熵，学「按约束挑物体 + 由物体属性选技能」
  低层 l: (s, x) → a        均方误差，学「技能 → 动作原型」+ 场景修正
  扁平 f: (c, x) → a        均方误差，必须把上面两步压进一个连续映射里

评分协议（对齐论文 §5.1）
------------------------
论文对**没有可解释语言输出**的扁平基线，是用评估者「反推它想做哪一步」来打分的。
这里照做：扁平只输出连续动作 â，评估者用**自己知道的** (μ, B, x) 把它反解成
``implied_skill(â, x) = argmin_s ‖â − (μ_s + Bx)‖``。
这个反解对扁平是**偏袒**的（它白拿了真值的场景修正项），所以差距只会低估不会高估。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 世界常量
M = 4            # 每个物体的属性位数
K = 5            # 桌上物体数
S = 6            # 技能数
D_A = 8          # 动作维度
DONE = S - 1     # 「收工」这个伪技能的下标
D_X = K * (M + 1)   # 场景向量：每个物体 M 位属性(±1) + 1 位「已处理」(±1)
D_C = M * 3      # 约束向量：每位 3 选 1 的 one-hot
HID = 64         # 各网络的隐藏层宽度

SIG_ETA = 0.30   # 不可预测噪声的标准差


def _orthonormal(rng, n, d):
    """n 个 d 维单位正交行（n ≤ d）。"""
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q[:, :n].T


def _simplex(k: int, d: int) -> np.ndarray:
    """k 个 d 维行向量组成正单纯形：范数相同、两两内积相同、和为 0（要求 k = d+1）。"""
    C = np.eye(k) - np.full((k, k), 1.0 / k)
    ev, U = np.linalg.eigh(C)              # 特征值升序
    return U[:, -d:] * np.sqrt(ev[-d:])    # (k, d)


def _all_z(m: int = M) -> np.ndarray:
    """枚举 {±1}^m 的全部 2^m 个属性模式。"""
    g = np.array(np.meshgrid(*[[-1.0, 1.0]] * m, indexing="ij")).reshape(m, -1).T
    return g


class World:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.B = rng.normal(size=(D_A, D_X))
        self._calibrate_B()
        self.mu = _orthonormal(rng, S, D_A)
        self.W_g = self._balanced_scorer(rng)
        # 约束池：8 个「单位约束」+ 4 个「两位合取」的复合约束
        self.single = [((j,), ()) for j in range(M)] + [((), (j,)) for j in range(M)]
        self.composite = [((j,), ((j + 1) % M,)) for j in range(M)]
        order = rng.permutation(len(self.single))
        self.train_pool = [self.single[i] for i in order]   # 训练从这 8 个里取前 n 个
        self.test_c = self.composite                         # 测试固定为 4 个复合约束

    # ---------------------------------------------------------------- 世界构件
    def _calibrate_B(self):
        """缩放到 E‖Bx‖² = 0.25（x 零均值，所以这也等于 Var(Bx)）。

        注意 `np.mean(np.square(...))` 是**逐维平均**，所以目标要除以 D_A。
        """
        x = self._sample_scene(8192)
        cur = float(np.mean(np.square(x @ self.B.T)))      # 逐维
        self.B *= np.sqrt((0.25 / D_A) / max(cur, 1e-12))

    def _balanced_scorer(self, rng, tries: int = 4000):
        """技能打分矩阵 W_g：单纯形 + 随机旋转，要求 argmax 在 {±1}^M 上分布均衡。"""
        base = _simplex(S - 1, M)
        grid = _all_z()                                  # (2^M, M)
        best, best_gap = None, -1.0
        for t in range(tries):
            Q, _ = np.linalg.qr(rng.normal(size=(M, M)))
            W = base @ Q
            cnt = np.bincount(np.argmax(grid @ W.T, axis=1), minlength=S - 1)
            p = cnt / cnt.sum()
            gap = float(p.min())
            if gap > best_gap:
                best, best_gap = W, gap
            if p.min() >= 0.15 and p.max() <= 0.30:
                return W
        return best

    # ---------------------------------------------------------------- 约束编码
    @staticmethod
    def cvec(r, rbar) -> np.ndarray:
        """(r, rbar) → M×3 逐位 one-hot。通道 0=require-1，1=require-0，2=不关心。"""
        v = np.zeros((M, 3), dtype=np.float64)
        rset, rbset = set(r), set(rbar)
        for j in range(M):
            v[j, 0 if j in rset else (1 if j in rbset else 2)] = 1.0
        return v.reshape(-1)

    def cvec_batch(self, cs) -> np.ndarray:
        return np.stack([self.cvec(*c) for c in cs])

    # ---------------------------------------------------------------- 场景采样
    def _sample_scene(self, n: int, rng=None) -> np.ndarray:
        """n 个场景 (n, D_X)：属性 ±1（各半），已处理比例 ~ U(0, 0.6)，编码成 ±1。"""
        g = self.rng if rng is None else rng
        z = np.where(g.random((n, K, M)) < 0.5, -1.0, 1.0)
        p_done = g.uniform(0.0, 0.6, size=(n, 1))
        done = np.where(g.random((n, K, 1)) < p_done[:, :, None], 1.0, -1.0)
        return np.concatenate([z, done], axis=-1).reshape(n, D_X)

    def split(self, x: np.ndarray):
        """(n, D_X) → (z (n,K,M), done (n,K))，done = +1 表示已处理。"""
        t = np.atleast_2d(x).reshape(-1, K, M + 1)
        return t[:, :, :M], t[:, :, M]

    def target(self, x: np.ndarray, r, rbar) -> np.ndarray:
        """每个场景里「满足约束且未处理的、下标最小的物体」下标；没有则 −1。"""
        z, done = self.split(x)
        ok = np.ones(z.shape[:2], dtype=bool)
        for j in r:
            ok &= z[:, :, j] > 0
        for j in rbar:
            ok &= z[:, :, j] < 0
        ok &= done < 0
        has = ok.any(axis=1)
        return np.where(has, np.argmax(ok, axis=1), -1)

    def skill(self, x: np.ndarray, r, rbar) -> np.ndarray:
        """真值的下一步技能 s*。"""
        idx = self.target(x, r, rbar)
        z, _ = self.split(x)
        out = np.full(z.shape[0], DONE, dtype=np.int64)
        m = idx >= 0
        if m.any():
            out[m] = np.argmax(z[m, idx[m]] @ self.W_g.T, axis=-1)
        return out

    def action(self, x: np.ndarray, s: np.ndarray, *, noise: bool = True,
               rng=None) -> np.ndarray:
        """真值动作 a* = μ_s + Bx（+ η）。"""
        x = np.atleast_2d(x)
        a = self.mu[np.asarray(s).reshape(-1)] + x @ self.B.T
        if noise:
            g = self.rng if rng is None else rng
            a = a + SIG_ETA * g.normal(size=a.shape)
        return a

    def implied_skill(self, a_hat: np.ndarray, x: np.ndarray) -> np.ndarray:
        """把连续动作反解成「它想执行哪个技能」（论文给扁平基线打分的做法）。"""
        x = np.atleast_2d(x)
        base = x @ self.B.T                       # (n, D_A)
        d = (a_hat[:, None, :] - (self.mu[None, :, :] + base[:, None, :])) ** 2
        return np.argmin(d.sum(axis=-1), axis=-1)

    # ---------------------------------------------------------------- 归一化基准
    def rel_err(self, a_hat: np.ndarray, a_star: np.ndarray) -> float:
        """相对动作误差：分子是 MSE，分母是「预测 a* 的均值」的 MSE。"""
        num = float(np.mean(np.square(a_hat - a_star)))
        den = float(np.mean(np.square(a_star - a_star.mean(axis=0))))
        return num / max(den, 1e-12)

    def noise_floor(self, a_star: np.ndarray) -> float:
        """不可预测噪声对应的相对误差下界（分子分母同为「逐维平均」）。"""
        return SIG_ETA ** 2 / max(
            float(np.mean(np.square(a_star - a_star.mean(axis=0)))), 1e-12)


# ---------------------------------------------------------------- 约束采样
def sample_batch(world: World, cs, n_per: int, rng):
    """给一组约束各采 n_per 个场景，返回 (约束 one-hot, x, s*, a*)。"""
    x = world._sample_scene(len(cs) * n_per, rng)
    cv = np.repeat(world.cvec_batch(cs), n_per, axis=0)
    s = np.concatenate([world.skill(x[i * n_per:(i + 1) * n_per], *c)
                        for i, c in enumerate(cs)])
    a = world.action(x, s, noise=True, rng=rng)
    return cv, x, s, a


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
    n = X.shape[0]
    dH = (dY @ p[2].T) * (1.0 - H * H)
    return [X.T @ dH / n, dH.sum(axis=0) / n, H.T @ dY / n, dY.sum(axis=0) / n]


def mse_grad(out, target):
    return float(np.mean(np.square(out - target))), 2.0 * (out - target) / out.shape[0]


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
    def __init__(self, params, lr: float = 3e-3, b1: float = 0.9,
                 b2: float = 0.999, eps: float = 1e-8):
        self.t = 0
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads, strict=True)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


def onehot(idx: np.ndarray, n: int) -> np.ndarray:
    o = np.zeros((len(idx), n))
    o[np.arange(len(idx)), np.asarray(idx)] = 1.0
    return o
