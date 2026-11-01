"""扩散模型后训练对齐的数值核验：DRaFT / ReFL / Diffusion-DPO（PyTorch 版）。

配套博文：《扩散模型的后训练对齐：奖励只给一次，信用怎么摊到每一步》
（lrypcy.github.io）

玩具：一维三峰高斯混合 + **指数倾斜族**。因为倾斜「x 的指数」在扩散下闭包
（高斯仍高斯、混合仍混合），所以每个 θ 都有解析 score、解析样本、解析
log 密度与解析 KL —— 三条对齐路线可以在同一个可解析的目标上直接对比。

纯 torch + matplotlib，纯 CPU、float64。每组末尾自断言，汇总断言总数。
用法：
    python post_train_lab.py                 # 全部 A–E
    python post_train_lab.py B C             # 只跑指定组
    python post_train_lab.py A --figdir DIR  # 额外输出 F 组（配图）
"""

import math
import sys

import torch

torch.set_default_dtype(torch.float64)

T_MAX = 1000.0
BETA0, BETA1 = 1e-4, 0.02
T_END = 0.01
K_STEPS = 20                 # 采样链的步数
LAM = 2.0                    # KL 系数 λ，最优倾斜 θ* = 1/λ = 0.5
SEED = 20261103

MU = torch.tensor([-3.0, 0.0, 3.0])
W = torch.tensor([1 / 3, 1 / 3, 1 / 3])
SD = 0.5


def fmt(v, n=4):
    return f"{v:.{n}f}"


def gen(seed):
    return torch.Generator().manual_seed(seed)


def assert_close(name, got, want, tol, state):
    err = float((torch.as_tensor(got).detach() - torch.as_tensor(want).detach()).abs().max())
    assert err <= tol, f"[{name}] err={err:.3e} > tol={tol:.3e}"
    state.append((name, err))


def assert_true(name, cond, state):
    assert cond, f"[{name}] 断言失败"
    state.append((name, 0.0))


def section(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# ------------------------------------------------------------ 调度
def abar(t):
    t = torch.as_tensor(t)
    return torch.exp(-(BETA0 * t + 0.5 * (BETA1 - BETA0) * t * t / T_MAX))


def alpha_of(t):
    return torch.sqrt(abar(t))


def sigma_of(t):
    return torch.sqrt(1.0 - abar(t))


LAM_T = float(torch.log(alpha_of(T_MAX) / sigma_of(T_MAX)))
LAM_END = float(torch.log(alpha_of(T_END) / sigma_of(T_END)))


def as_from_lam(l):
    e = math.exp(2.0 * l)
    s = 1.0 / math.sqrt(1.0 + e)
    return s * math.exp(l), s


# ------------------------------------------------- 倾斜族（θ 是唯一的参数）
def tilted_weights(theta):
    """p_θ(x) ∝ p_0(x) e^{θx} 之后各分量的权重。"""
    logits = torch.log(W) + theta * MU + 0.5 * theta ** 2 * SD ** 2
    return torch.softmax(logits, dim=0)


def log_Z(theta):
    """Z(θ) = E_{p_0}[e^{θx}] 的闭式。"""
    logits = torch.log(W) + theta * MU + 0.5 * theta ** 2 * SD ** 2
    return torch.logsumexp(logits, dim=0)


def mean_theta(theta):
    """m(θ) = E_{p_θ}[x] = d/dθ log Z。"""
    return MU + theta * SD ** 2


def var_theta(theta):
    """m'(θ) = Var_{p_θ}[x]。倾斜既平移分量均值、又给分量重新加权，
    所以是「峰内方差 + 峰间方差」两部分，不等于 SD²。"""
    w, m = tilted_weights(theta), mean_theta(theta)
    mu = torch.dot(w, m)
    return SD ** 2 + torch.dot(w, (m - mu) ** 2)


def sample_theta(theta, n, g):
    w = tilted_weights(theta)
    k = torch.multinomial(w.detach(), n, replacement=True, generator=g)
    return mean_theta(theta).detach()[k] + SD * torch.randn(n, generator=g)


def logp_theta(x, theta):
    """log p_θ(x) 的闭式（x 可以是张量，θ 是标量张量或 float）。"""
    w = tilted_weights(theta)
    m = mean_theta(theta)
    z = (x[:, None] - m[None]) ** 2 / (2 * SD ** 2)
    return torch.logsumexp(torch.log(w)[None] - z - math.log(SD * math.sqrt(2 * math.pi)), dim=1)


def kl_theta(theta):
    """KL(p_θ || p_0) = θ m(θ) - log Z(θ)。"""
    w, m = tilted_weights(theta), mean_theta(theta)
    return theta * torch.dot(w, m) - log_Z(theta)


# ------------------------------------------------- 扩散后的倾斜混合（解析）
def diff_params(theta, t):
    """第 k 个分量在时刻 t 的均值与方差（倾斜先作用在 x_0 上，再扩散）。"""
    a, s = alpha_of(t), sigma_of(t)
    mean = a * mean_theta(theta)
    var = a ** 2 * SD ** 2 + s ** 2
    return mean, var


def score_t(x, theta, t):
    a, s = alpha_of(t), sigma_of(t)
    mean, var = diff_params(theta, t)
    logits = torch.log(tilted_weights(theta))[None] - (x[:, None] - mean[None]) ** 2 / (2 * var) - 0.5 * math.log(var)
    r = torch.softmax(logits, dim=1)
    return (r * (-(x[:, None] - mean[None]) / var)).sum(1)


def x0hat(x, theta, t):
    a, s = alpha_of(t), sigma_of(t)
    return (x + s ** 2 * score_t(x, theta, t)) / a


def lam_grid(K):
    return [LAM_T + (LAM_END - LAM_T) * i / K for i in range(K + 1)]


def sample_chain(theta, x_init, K=K_STEPS):
    """跑一遍确定性采样链，返回 (x_0, 中间状态列表)。θ 参与图，可反传。"""
    lams = lam_grid(K)
    x = x_init
    xs = [x]
    for i in range(K):
        la, lb = lams[i], lams[i + 1]
        a, s = as_from_lam(la)
        _, s_b = as_from_lam(lb)
        xh = _x0hat_as(x, theta, a, s)
        u = x / s + (math.exp(lb) - math.exp(la)) * xh
        x = u * s_b
        xs.append(x)
    return x, xs


def _score_as(x, theta, a, s):
    """给定 (α,σ) 的 score（不经过 t）。"""
    mean = a * mean_theta(theta)
    var = a ** 2 * SD ** 2 + s ** 2
    logits = torch.log(tilted_weights(theta))[None] - (x[:, None] - mean[None]) ** 2 / (2 * var) - 0.5 * math.log(var)
    r = torch.softmax(logits, dim=1)
    return (r * (-(x[:, None] - mean[None]) / var)).sum(1)


def _x0hat_as(x, theta, a, s):
    return (x + s ** 2 * _score_as(x, theta, a, s)) / a


def chain_grads(theta, xs):
    """每一步的 dx_k/dθ（x_init 与 θ 无关，记 0）。"""
    out = [0.0]
    for xk in xs[1:]:
        out.append(float(torch.autograd.grad(xk.sum(), theta, retain_graph=True)[0]))
    return out


# ------------------------------------------------------------------ A
def exp_A(state):
    """倾斜族的解析量：logZ 的导数、KL 闭式、score 的 autograd 对照、最优 θ*。"""
    section("A 组：倾斜族的解析量自洽")
    g = gen(SEED)

    # d/dθ log Z = m(θ)
    for th in (0.0, 0.3, 0.5, 1.0):
        t = torch.tensor(th, requires_grad=True)
        gz = torch.autograd.grad(log_Z(t), t)[0]
        assert_close(f"θ={th} dlogZ/dθ = m(θ)", gz, torch.dot(tilted_weights(torch.tensor(th)), mean_theta(torch.tensor(th))), 1e-10, state)

    # KL 非负、在 θ=0 处为 0
    assert_close("KL(θ=0) = 0", kl_theta(torch.tensor(0.0)), torch.tensor(0.0), 1e-12, state)
    assert_true("KL(θ=0.5) > 0", float(kl_theta(torch.tensor(0.5))) > 0, state)

    # 解析 score 与 autograd 对照（任意 θ 都要成立）
    for th in (0.0, 0.5):
        for t in (50.0, 400.0, 900.0):
            a, s = float(alpha_of(t)), float(sigma_of(t))
            x = sample_theta(torch.tensor(th), 256, g)
            x = (a * x + s * torch.randn(256, generator=g)).requires_grad_(True)
            mean, var = a * mean_theta(torch.tensor(th)), a ** 2 * SD ** 2 + s ** 2
            logits = torch.log(tilted_weights(torch.tensor(th)))[None] - (x[:, None] - mean[None]) ** 2 / (2 * var) - 0.5 * math.log(var)
            lp = torch.logsumexp(logits, dim=1).sum()
            grad = torch.autograd.grad(lp, x)[0]
            assert_close(f"θ={th} t={t:g} 解析 score = autograd", score_t(x, torch.tensor(th), t), grad, 1e-9, state)

    # 目标函数 J(θ) = m(θ) - λ·KL 的驻点就是 θ* = 1/λ
    def J(th):
        t = torch.tensor(th)
        return torch.dot(tilted_weights(t), mean_theta(t)) - LAM * kl_theta(t)

    ths = torch.linspace(0.0, 1.2, 1201, requires_grad=True)
    vals = torch.stack([J(float(v)) for v in ths.detach()])
    best = float(ths.detach()[int(vals.argmax())])
    assert_close("数值最优 θ* = 1/λ", best, 1.0 / LAM, 2e-3, state)
    print(f"  λ = {LAM}，θ* = 1/λ = {1 / LAM:.4f}，数值最优 {best:.4f}")
    print(f"  θ=0.5 时 KL = {float(kl_theta(torch.tensor(0.5))):.4f}，E[x] = {float(torch.dot(tilted_weights(torch.tensor(0.5)), mean_theta(torch.tensor(0.5)))):.4f}")


# ------------------------------------------------------------------ B
def exp_B(state):
    """信用分配剖面：一条链上的梯度贡献怎么分布。"""
    section("B 组：奖励梯度沿采样链的分布")
    g = gen(SEED + 1)
    theta = torch.tensor(0.5, requires_grad=True)
    x_init = torch.randn(4096, generator=g)

    x0, xs = sample_chain(theta, x_init)
    grads = chain_grads(theta, xs)          # 差分即该步的贡献
    contrib = [grads[i + 1] - grads[i] for i in range(len(grads) - 1)]
    total = grads[-1]
    share = [abs(c) / sum(abs(v) for v in contrib) for c in contrib]

    print(f"  链长 K = {K_STEPS}，总梯度 dx_0/dθ = {total:.4f}")
    print(f"  {'step':>6}{'占比':>9}{'累计':>9}   {'step':>6}{'占比':>9}{'累计':>9}")
    half = len(share) // 2
    for a in range(half):
        b = a + half
        ca = sum(share[: a + 1]) * 100
        cb = sum(share[: b + 1]) * 100
        print(f"  {a + 1:>6}{share[a] * 100:>8.2f}%{ca:>8.2f}%   {b + 1:>6}{share[b] * 100:>8.2f}%{cb:>8.2f}%")

    peak = int(max(range(len(share)), key=lambda i: share[i])) + 1
    last5 = sum(share[-5:])
    first10 = sum(share[:10])
    top3 = sum(sorted(share)[-3:])
    print(f"\n  峰值在第 {peak} 步（{share[peak - 1] * 100:.2f}%）；前 10 步累计 {first10 * 100:.2f}%；"
          f"最后 5 步合计 {last5 * 100:.3f}%；贡献最大的三步合计 {top3 * 100:.2f}%")

    assert_true(f"最后一步几乎不承担信用（{share[-1] * 100:.3f}% < 0.5%）", share[-1] < 0.005, state)
    assert_true(f"最后 5 步合计可忽略（{last5 * 100:.3f}% < 1%）", last5 < 0.01, state)
    assert_true(f"信用集中在中段而非两端（峰值第 {peak} 步落在 5–12）", 5 <= peak <= 12, state)
    assert_true(f"前 10 步拿走 90% 以上（{first10 * 100:.2f}%）", first10 > 0.90, state)
    assert_true("所有步的贡献同号（没有相互抵消）", all(c > 0 for c in contrib) or all(c < 0 for c in contrib), state)

    print("\n  链长对「只反传最后一步」的影响：")
    print(f"  {'K':>4} {'最后一步占比':>14} {'只用最后一步的相对偏差':>26}")
    rows = []
    for K in (5, 10, 20, 40):
        th2 = torch.tensor(0.5, requires_grad=True)
        x0k, xsk = sample_chain(th2, x_init, K=K)
        gk = chain_grads(th2, xsk)
        ck = [gk[i + 1] - gk[i] for i in range(len(gk) - 1)]
        sh = abs(ck[-1]) / sum(abs(v) for v in ck)
        rows.append(sh)
        print(f"  {K:>4} {sh * 100:>13.3f}% {(ck[-1] / gk[-1] - 1) * 100:>25.2f}%")
    assert_true("链越长，最后一步占比越小", rows[0] > rows[-1], state)


# ------------------------------------------------------------------ C
def exp_C(state):
    """四个估计器的偏差与方差。"""
    section("C 组：梯度估计器的偏差与方差（B=32 样本，200 次重复）")
    g = gen(SEED + 2)
    theta0 = 0.5
    B, R = 32, 200
    mprime = float(var_theta(torch.tensor(theta0)))

    def pathwise(th, xi):
        th_t = torch.tensor(th, requires_grad=True)
        x0, _ = sample_chain(th_t, xi)
        return float(torch.autograd.grad(x0.sum(), th_t)[0]) / xi.shape[0]

    def score_direct(th, xi):
        """样本直接来自 p_θ（密度已知）时的 score-function 估计。"""
        x0 = sample_theta(torch.tensor(th), xi.shape[0], g)
        th_t = torch.tensor(th, requires_grad=True)
        dlp = torch.autograd.grad(logp_theta(x0, th_t).sum(), th_t)[0]
        return float((x0 * dlp).sum()) / xi.shape[0]

    def score_chain(th, xi):
        """样本来自采样链（这是真扩散模型里唯一能拿到的样本）。"""
        with torch.no_grad():
            x0, _ = sample_chain(torch.tensor(th), xi)
        th_t = torch.tensor(th, requires_grad=True)
        dlp = torch.autograd.grad(logp_theta(x0, th_t).sum(), th_t)[0]
        return float((x0 * dlp).sum()) / xi.shape[0]

    def last_step(th, xi):
        th_t = torch.tensor(th, requires_grad=True)
        x0, xs = sample_chain(th_t, xi, K=K_STEPS)
        gk = chain_grads(th_t, xs)
        return (gk[-1] - gk[-2]) / xi.shape[0]

    est = [("pathwise（DRaFT，穿链反传）", pathwise),
           ("score-function，样本来自 p_θ", score_direct),
           ("score-function，样本来自采样链", score_chain),
           ("只反传最后一步", last_step)]

    print(f"  解析 m'(θ) = Var_θ[x] = {mprime:.5f}（= SD² {SD ** 2:.4f} + 峰间方差 {mprime - SD ** 2:.4f}）")
    stats = {}
    for name, f in est:
        vals = torch.tensor([f(theta0, torch.randn(B, generator=g)) for _ in range(R)])
        bias = float(vals.mean()) - mprime
        sd = float(vals.std())
        stats[name] = (bias, sd)
        se = sd / math.sqrt(R)
        print(f"  {name:>30}  均值 {float(vals.mean()):>9.5f}  偏差 {bias:>+9.5f}  标准差 {sd:>9.5f}  标准误 {se:>7.4f}")

    b1, s1 = stats["pathwise（DRaFT，穿链反传）"]
    b2, s2 = stats["score-function，样本来自 p_θ"]
    b3, s3 = stats["score-function，样本来自采样链"]
    b4, s4 = stats["只反传最后一步"]
    assert_true(f"pathwise 无偏（|偏差| = {abs(b1):.5f} < 3σ/√R）", abs(b1) < 3 * s1 / math.sqrt(R), state)
    assert_true(f"样本取自 p_θ 时 score-function 无偏（|偏差| = {abs(b2):.5f} < 3σ/√R）",
                abs(b2) < 3 * s2 / math.sqrt(R), state)
    assert_true(f"样本取自采样链时 score-function 严重有偏（偏差 {b3:+.4f} > 0.5×真值）", abs(b3) > 0.5 * mprime, state)
    assert_true(f"只反传最后一步拿不到梯度（均值 {b4 + mprime:.6f} ≈ 0）", abs(b4 + mprime) < 1e-3, state)
    assert_true(f"score-function 的方差远大于 pathwise（{s2:.4f} ≫ {s1:.4f}）", s2 > s1, state)
    print(f"  → 穿链反传与「已知密度」的 score-function 都无偏：前者标准差 {s1:.3f}，后者 {s2:.3f}（{s2 / s1:.1f} 倍）。")
    print(f"    但真模型里拿不到 p_θ(x_0)，用链上样本硬套 log-derivative 的偏差是 {abs(b3) / mprime * 100:.0f}% ——")
    print("    这就是 ReFL 不走原始 REINFORCE、改用奖励加权回归的原因。")


# ------------------------------------------------------------------ D
def exp_D(state):
    """三条路线在同样样本预算下逼近 θ*。"""
    section("D 组：三条路线的样本效率（目标 |θ̂-θ*| ≤ 0.05，上限 40 万条）")
    g = gen(SEED + 3)
    theta_star = 1.0 / LAM
    B = 64
    CAP = 400000

    def clip(v, c=2.0):
        return max(-c, min(c, v))

    def run_pathwise(eta=0.10):
        th, used, hist = 0.0, 0, []
        while used < CAP:
            xi = torch.randn(B, generator=g)
            th_t = torch.tensor(th, requires_grad=True)
            x0, _ = sample_chain(th_t, xi)
            ghat = float(torch.autograd.grad(x0.sum(), th_t)[0]) / B
            th = th + eta * clip(ghat) * (1 - LAM * th)
            used += B
            hist.append(th)
            if abs(th - theta_star) < 0.05:
                return th, used
        return th, used

    def run_reinforce(eta=0.005):
        th, used = 0.0, 0
        while used < CAP:
            th_t = torch.tensor(th, requires_grad=True)
            x0 = sample_theta(th_t, B, g)
            dlp = torch.autograd.grad(logp_theta(x0, th_t).sum(), th_t)[0]
            ghat = float((x0.detach() * dlp).sum()) / B
            th = th + eta * clip(ghat) * (1 - LAM * th)
            th = max(-0.5, min(1.5, th))
            used += B
            if abs(th - theta_star) < 0.05:
                return th, used
        return th, used

    def run_dpo(beta=1.0 / LAM, eta=0.5):
        th, used = 0.0, 0
        while used < CAP:
            y1 = sample_theta(torch.tensor(0.0), B, g)
            y2 = sample_theta(torch.tensor(0.0), B, g)
            pref = torch.rand(B, generator=g) < torch.sigmoid(y1 - y2)
            xp = torch.where(pref, y1, y2)
            xm = torch.where(pref, y2, y1)
            th_t = torch.tensor(th, requires_grad=True)
            logr = beta * (logp_theta(xp, th_t) - logp_theta(xm, th_t))
            loss = -torch.log(torch.sigmoid(logr)).mean()
            grad = torch.autograd.grad(loss, th_t)[0]
            th = th - eta * clip(float(grad)) 
            th = max(-0.5, min(1.5, th))
            used += 2 * B
            if abs(th - theta_star) < 0.05:
                return th, used
        return th, used

    print(f"  θ* = {theta_star:.4f}")
    res = {}
    for name, f in (("DRaFT（穿链反传）", run_pathwise), ("REINFORCE（策略梯度）", run_reinforce),
                    ("Diffusion-DPO", run_dpo)):
        th, used = f()
        res[name] = used
        flag = "达到" if abs(th - theta_star) < 0.05 else "未达到"
        print(f"  {name:>22}  θ̂ = {th:.4f}   |θ̂-θ*| = {abs(th - theta_star):.4f}   样本 {used:>7}  {flag}")
        assert_true(f"{name} 在预算内达成目标", abs(th - theta_star) < 0.05, state)

    print(f"  样本效率比：REINFORCE / DRaFT = {res['REINFORCE（策略梯度）'] / res['DRaFT（穿链反传）']:.1f}×，"
          f"DPO / DRaFT = {res['Diffusion-DPO'] / res['DRaFT（穿链反传）']:.1f}×")
    assert_true(f"REINFORCE 比 DRaFT 慢两个数量级以上（{res['REINFORCE（策略梯度）'] / res['DRaFT（穿链反传）']:.0f}×）",
                res["REINFORCE（策略梯度）"] > 100 * res["DRaFT（穿链反传）"], state)
    assert_true(f"DPO 与 DRaFT 同量级（比值 {res['Diffusion-DPO'] / res['DRaFT（穿链反传）']:.2f}）",
                0.4 <= res["Diffusion-DPO"] / res["DRaFT（穿链反传）"] <= 3.0, state)


# ------------------------------------------------------------------ E
def exp_E(state):
    """显存账：反向图随链长线性增长。"""
    section("E 组：反向图的规模 vs 链长")
    g = gen(SEED + 4)
    xi = torch.randn(256, generator=g)

    def graph_nodes(x):
        # 必须把节点对象留住：只存 id 的话包装器会被回收、id 被复用，遍历会提前结束
        keep, seen, stack = [], set(), [x.grad_fn]
        while stack:
            f = stack.pop()
            if f is None or id(f) in seen:
                continue
            keep.append(f)
            seen.add(id(f))
            for nxt, _ in f.next_functions:
                stack.append(nxt)
        return len(keep)

    print(f"  {'K':>4} {'计算图节点数':>14} {'相对 K=5':>10}")
    base = None
    for K in (5, 10, 20, 40):
        th = torch.tensor(0.5, requires_grad=True)
        x0, _ = sample_chain(th, xi, K=K)
        n = graph_nodes(x0)
        base = base or n
        print(f"  {K:>4} {n:>14} {n / base:>9.2f}×")
    # 线性：K 翻倍节点数翻倍
    th = torch.tensor(0.5, requires_grad=True)
    n20 = graph_nodes(sample_chain(th, xi, K=20)[0])
    th2 = torch.tensor(0.5, requires_grad=True)
    n40 = graph_nodes(sample_chain(th2, xi, K=40)[0])
    assert_true(f"K 从 20 到 40，图节点数接近翻倍（{n20} → {n40}）", 1.7 <= n40 / n20 <= 2.3, state)


# ------------------------------------------------------------------ F
def exp_F(figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = gen(SEED + 5)
    theta = torch.tensor(0.5, requires_grad=True)
    x_init = torch.randn(4096, generator=g)
    x0, xs = sample_chain(theta, x_init)
    grads = chain_grads(theta, xs)
    contrib = [abs(grads[i + 1] - grads[i]) for i in range(len(grads) - 1)]
    share = [c / sum(contrib) for c in contrib]

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=160)
    ax.bar(range(1, len(share) + 1), share, color="#378ADD", width=0.7)
    ax.set_xlabel("sampling step (noise → data)")
    ax.set_ylabel("share of ∂x₀/∂θ")
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{figdir}/post_train_credit.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=160)
    xs_np = [float(v.detach().mean()) for v in xs]
    ax.plot(range(len(xs_np)), xs_np, marker="o", ms=3.2, lw=1.4, color="#0F6E56")
    ax.set_xlabel("sampling step")
    ax.set_ylabel("batch mean of x_t")
    ax.grid(True, alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(f"{figdir}/post_train_trajectory.png")
    plt.close(fig)
    print(f"  已写出 {figdir}/post_train_credit.png 与 post_train_trajectory.png")


# ---------------------------------------------------------------- main
def main(argv):
    groups = [a for a in argv if not a.startswith("-")]
    figdir = None
    if "--figdir" in argv:
        figdir = argv[argv.index("--figdir") + 1]
        groups = [a for a in groups if a != figdir]

    groups = groups or ["A", "B", "C", "D", "E"]
    state = []
    table = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E}
    for gname in groups:
        table[gname](state)
    if figdir:
        exp_F(figdir)

    print()
    print("=" * 78)
    print(f"断言总数 = {len(state)}")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1:])
