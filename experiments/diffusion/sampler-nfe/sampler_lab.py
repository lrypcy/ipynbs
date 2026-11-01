"""扩散采样器的数值核验：从 DDIM 到 DPM-Solver（PyTorch 版）。

配套博文：《扩散模型采样器：把反向过程写成 ODE 之后》
（lrypcy.github.io），ddpm-deep-dive 的续篇。

核心玩具：2D 八峰高斯混合。VP 扩散后的边际 p_t 仍是高斯混合，
所以 score 与后验均值 x̂_0 都是**解析**的 —— 没有网络近似误差，
测出来的每一点都只来自离散化，这正是比较求解器需要的条件。

时间用**连续**表示（t∈[0,T]，线性 β），不落到整数步上：
这样终点可以取到 σ≈1e-3，误差本底足够低，阶数才拟合得出来。

纯 torch + matplotlib，纯 CPU、float64。每组末尾自断言，汇总断言总数。
用法：
    python sampler_lab.py                 # 全部 A–F
    python sampler_lab.py B D             # 只跑指定组
    python sampler_lab.py --figdir DIR    # 额外输出 G 组（配图）
"""

import math
import sys

import torch

torch.set_default_dtype(torch.float64)     # 阶数拟合要看 1e-3 以下的误差，float32 不够

T_MAX = 1000.0
BETA0, BETA1 = 1e-4, 0.02
T_END = 0.01                                # 积分终点，σ≈1e-3
SEED = 20261102

R_RING, SD = 3.0, 0.30                      # 八峰半径 / 峰内标准差
N_SAMPLE = 40000
N_PROJ = 256
ORDER1 = ("euler_x", "ddim", "dpm2m")
NAME = {"euler_x": "Euler(x)", "midpoint_x": "Midpoint(x)", "ddim": "DDIM",
        "dpm2": "DPM-Solver-2", "dpm2m": "DPM-Solver-2M"}


def fmt(v, n=6):
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


# ------------------------------------------------------------ 连续调度
def beta(t):
    return BETA0 + (BETA1 - BETA0) * t / T_MAX


def abar(t):
    """ᾱ(t) = exp(-∫_0^t β)，线性 β 时有闭式。t 可以是 python float 或 tensor。"""
    t = torch.as_tensor(t)
    return torch.exp(-(BETA0 * t + 0.5 * (BETA1 - BETA0) * t * t / T_MAX))


def alpha_of(t):
    return torch.sqrt(abar(t))


def sigma_of(t):
    return torch.sqrt(1.0 - abar(t))


def lam_of(t):
    """λ_t = log(α_t/σ_t)。"""
    a, s = alpha_of(t), sigma_of(t)
    return torch.log(a / s)


def as_from_lam(l):
    """由 λ 反解 (α, σ)：α = e^λ/√(1+e^{2λ})，σ = 1/√(1+e^{2λ})。"""
    e = math.exp(2.0 * l)
    s = 1.0 / math.sqrt(1.0 + e)
    return s * math.exp(l), s


LAM_T = float(lam_of(T_MAX))       # ≈ -5.03
LAM_END = float(lam_of(T_END))     # ≈ +6.9


# ------------------------------------------------------------- 八峰玩具
def centers():
    ang = torch.arange(8) * (2 * math.pi / 8)
    return torch.stack([R_RING * torch.cos(ang), R_RING * torch.sin(ang)], 1)


MU = centers()
W = torch.full((8,), 1.0 / 8)


def comp_var(a, s):
    """第 k 个分量在 (α,σ) 时刻的方差：α²σ_data² + σ²。"""
    return a * a * SD * SD + s * s


def _logp_parts(x, a, s):
    v = comp_var(a, s)
    m = a * MU
    lp = -0.5 * ((x[:, None, :] - m[None]) ** 2).sum(-1) / v - math.log(v) - math.log(2 * math.pi)
    return lp + torch.log(W), v, m


def logp(x, a, s):
    lp, _, _ = _logp_parts(x, a, s)
    return torch.logsumexp(lp, dim=1)


def resp(x, a, s):
    lp, _, _ = _logp_parts(x, a, s)
    return torch.softmax(lp, dim=1)


def score(x, a, s):
    """∇_x log p_t(x)，解析式。"""
    r = resp(x, a, s)
    _, v, m = _logp_parts(x, a, s)
    return (r[..., None] * (-(x[:, None, :] - m[None]) / v)).sum(1)


def x0hat(x, a, s):
    """后验均值 E[x_0|x_t=x] = (x + σ²∇log p_t)/α，解析式。"""
    return (x + s * s * score(x, a, s)) / a


def sample_data(n, g):
    k = torch.multinomial(W, n, replacement=True, generator=g)
    return MU[k] + SD * torch.randn((n, 2), generator=g)


def sample_mixture(a, s, n, g):
    """从 p_{(α,σ)} 精确采样（不是从 N(0,I) 近似）。"""
    k = torch.multinomial(W, n, replacement=True, generator=g)
    return a * MU[k] + math.sqrt(comp_var(a, s)) * torch.randn((n, 2), generator=g)


# --------------------------------------------------------------- 度量
def sliced_w2(X, Y, g, n_proj=N_PROJ):
    """sliced W2 的平方根：固定随机方向上的分位数均方差的平均。"""
    d = torch.randn((2, n_proj), generator=g)
    d = d / d.norm(dim=0, keepdim=True)
    xs = (X @ d).sort(0).values
    ys = (Y @ d).sort(0).values
    return float(((xs - ys) ** 2).mean(0).mean().sqrt())


# --------------------------------------------------------------- 求解器
def lam_nodes(nfe, t_end=T_END, t_start=T_MAX):
    """均匀 λ 网格（DPM-Solver 论文的默认取法）。"""
    return [LAM_T + (LAM_END - LAM_T) * i / nfe for i in range(nfe + 1)]


def t_nodes(nfe, t_end=T_END, t_start=T_MAX):
    """均匀 t 网格（DDIM 常见实现的取法）。"""
    return [t_start + (t_end - t_start) * i / nfe for i in range(nfe + 1)]


def nodes(mode, nfe):
    return lam_nodes(nfe) if mode == "lam" else [float(lam_of(t)) for t in t_nodes(nfe)]


def _rhs_x(x, l):
    """x 空间 ODE：dx/dλ = α(x̂_0 - αx)。"""
    a, s = as_from_lam(l)
    return a * (x0hat(x, a, s) - a * x)


def solve(x, lams, kind):
    """沿 λ 序列积分，返回 (x̂_0 估计, 实际网络调用次数)。"""
    nfe = 0
    prev = None
    for i in range(len(lams) - 1):
        la, lb = lams[i], lams[i + 1]
        a_a, s_a = as_from_lam(la)
        a_b, s_b = as_from_lam(lb)
        if kind in ("euler_x", "midpoint_x"):
            h = lb - la
            k1 = _rhs_x(x, la)
            nfe += 1
            if kind == "euler_x":
                x = x + h * k1
            else:
                lm = 0.5 * (la + lb)
                x = x + h * _rhs_x(x + 0.5 * h * k1, lm)
                nfe += 1
        else:
            u = x / s_a
            xh = x0hat(x, a_a, s_a)
            nfe += 1
            if kind == "dpm2m":
                # 多步变体：用上一步的 x̂_0 做斜率，每步只需 1 次前向
                slope = 0.0 if prev is None else (xh - prev[0]) / (la - prev[1])
                i0 = math.exp(lb) - math.exp(la)
                i1 = (math.exp(lb) * (lb - 1) - math.exp(la) * (la - 1)) - la * i0
                u = u + i0 * xh + i1 * slope
                prev = (xh, la)
            elif kind == "ddim":
                u = u + (math.exp(lb) - math.exp(la)) * xh
            else:
                lm = 0.5 * (la + lb)
                u_m = u + (math.exp(lm) - math.exp(la)) * xh
                x_m = u_m * as_from_lam(lm)[1]
                xh_m = x0hat(x_m, *as_from_lam(lm))
                slope = (xh_m - xh) / (lm - la)
                i0 = math.exp(lb) - math.exp(la)
                i1 = (math.exp(lb) * (lb - 1) - math.exp(la) * (la - 1)) - la * i0
                u = u + i0 * xh + i1 * slope
                nfe += 1
            x = u * s_b
    a_e, s_e = as_from_lam(lams[-1])
    return x0hat(x, a_e, s_e), nfe


def run(kind, nfe, x_init, mode="lam"):
    steps = nfe if kind in ORDER1 else max(1, nfe // 2)
    return solve(x_init.clone(), nodes(mode, steps), kind)


# ------------------------------------------------------------------ A
def exp_A(state):
    """解析 score / x̂_0 的正确性：autograd 对照 + 归一化 + 大 t 极限。"""
    section("A 组：解析 score 与后验均值 x̂_0 的正确性")
    g = gen(SEED)
    for t in (1.0, 150.0, 500.0, 1000.0):
        a, s = float(alpha_of(t)), float(sigma_of(t))
        x = sample_mixture(a, s, 512, g).requires_grad_(True)
        grad = torch.autograd.grad(logp(x, a, s).sum(), x)[0]
        assert_close(f"t={t:g} 解析 score = autograd ∇log p_t", score(x, a, s), grad, 1e-9, state)

        sh = score(x, a, s)
        assert_close(f"t={t:g} 自洽 α x̂_0 + σ ε̂ = x", a * x0hat(x, a, s) + s * (-s * sh), x, 1e-9, state)

    grid = torch.linspace(-9, 9, 361)
    Xg = torch.cartesian_prod(grid, grid)
    for t in (10.0, 500.0, 1000.0):
        a, s = float(alpha_of(t)), float(sigma_of(t))
        mass = float(torch.exp(logp(Xg, a, s)).sum()) * (18 / 360) ** 2
        assert_true(f"t={t:g} p_t 归一化（网格积分 {mass:.6f}）", abs(mass - 1) < 3e-3, state)

    a, s = float(alpha_of(T_MAX)), float(sigma_of(T_MAX))
    xT = sample_mixture(a, s, 200000, g)
    var = float(xT.var(0).mean())
    assert_true(f"t=T 时 p_T 方差 ≈ 1（得 {var:.4f}）", abs(var - 1) < 0.02, state)
    print(f"  p_T 方差 = {fmt(var, 4)}，λ 区间 = [{LAM_T:.3f}, {LAM_END:.3f}]，σ_end = {float(sigma_of(T_END)):.2e}")


# ------------------------------------------------------------------ B
def exp_B(state):
    """λ 空间的推导链：dλ/dt = -β/(2σ²)，以及 dx/dλ 的两种写法。"""
    section("B 组：λ 空间 ODE 的推导核验")
    g = gen(SEED + 1)
    for t in (5.0, 200.0, 600.0, 1000.0):
        a, s, b = float(alpha_of(t)), float(sigma_of(t)), float(beta(t))
        d_ana = -b / (2 * s * s)
        h = 1e-2
        if h < t < T_MAX - h:
            d_fd = float((lam_of(t + h) - lam_of(t - h)) / (2 * h))
        else:
            d_fd = float((lam_of(t) - lam_of(t - h)) / h) if t >= T_MAX - h else float((lam_of(t + h) - lam_of(t)) / h)
        assert_close(f"t={t:g} dλ/dt 解析式 = 差分", d_ana, d_fd, 1e-5, state)

        x = sample_mixture(a, s, 256, g)
        xh = x0hat(x, a, s)
        eh = (x - a * xh) / s
        da, ds = -0.5 * b * a, 0.5 * b * a * a / s        # α', σ' 直接由 β 给出
        chain = (da * xh + ds * eh) / d_ana               # dx/dλ = (dx/dt)/(dλ/dt)
        assert_close(f"t={t:g} 链式法则 = α(x̂_0-αx)", chain, _rhs_x(x, float(lam_of(t))), 1e-9, state)
        print(f"  t={t:4g}  dλ/dt = {fmt(d_ana, 4)}   |dx/dλ| = {fmt(float(chain.norm(dim=1).mean()), 4)}")

    # u = x/σ 换元后 du/dλ = e^λ x̂_0，等价于 dx/dλ = α x̂_0 - α² x
    l = 1.3
    a, s = as_from_lam(l)
    x = sample_mixture(a, s, 128, g)
    assert_close("σ·du/dλ = dx/dλ + α²x（换元自洽）",
                 s * math.exp(l) * x0hat(x, a, s) - a * a * x, _rhs_x(x, l), 1e-9, state)


# ------------------------------------------------------------------ C
def exp_C(state):
    """DDIM 更新 = 「区间内 x̂_0 取常数」时的精确积分。"""
    section("C 组：DDIM 是常数 x̂_0 假设下的精确解")
    g = gen(SEED + 2)
    la, lb = float(lam_of(800.0)), float(lam_of(600.0))
    a_a, s_a = as_from_lam(la)
    a_b, s_b = as_from_lam(lb)
    x = sample_mixture(a_a, s_a, 64, g)
    xh = x0hat(x, a_a, s_a)
    closed = (x / s_a + (math.exp(lb) - math.exp(la)) * xh) * s_b
    paper = (s_b / s_a) * x + (a_b - a_a * s_b / s_a) * xh
    assert_close("DDIM 两种写法（u 积分式 / 论文式）同值", closed, paper, 1e-12, state)

    # 一阶方法的局部截断误差：把 x̂_0 视为常数时，区间内真实变化被全部丢掉
    h = lb - la
    mid = 0.5 * (la + lb)
    xh_m = x0hat(x * 0 + closed, *as_from_lam(mid))
    assert_true(f"区间宽度 |h|={h:.4f}，x̂_0 在区间内的变化不为零（一阶误差来源）",
                float((xh_m - xh).norm(dim=1).mean()) > 1e-3, state)

    a0, s0 = float(alpha_of(1.0)), float(sigma_of(1.0))
    x0 = sample_data(64, g)
    eps = torch.randn(x0.shape, generator=gen(7))
    rec = (a0 * x0 + s0 * eps - s0 * eps) / a0
    assert_close("给定真实 ε 时 x̂_0 反解回 x_0", rec, x0, 1e-10, state)
    print(f"  t=1 时 σ = {fmt(s0, 6)}；积分到 t_end={T_END} 时 σ = {float(sigma_of(T_END)):.2e}")


# ------------------------------------------------------------------ D
def reference(x_init, nfe=4000):
    """同一批初值在高精度网格上的解，作为「精确解」——离散化误差的零点。"""
    return run("midpoint_x", nfe, x_init)[0]


def exp_D(state):
    """四个求解器对同一个参考解的离散化误差。"""
    section("D 组：离散化误差（对同一批初值的高精度参考解）")
    g = gen(SEED + 3)
    aT, sT = float(alpha_of(T_MAX)), float(sigma_of(T_MAX))
    x_init = sample_mixture(aT, sT, 8000, g)
    x_ref = reference(x_init)
    kinds = ["euler_x", "midpoint_x", "ddim", "dpm2", "dpm2m"]
    nfes = [5, 10, 20, 40, 80]

    print(f"{'NFE':>5} " + "".join(f"{NAME[k]:>14}" for k in kinds))
    table = {}
    for nfe in nfes:
        row = []
        for k in kinds:
            x_end, used = run(k, nfe, x_init)
            assert_true(f"{NAME[k]}@NFE={nfe} 网络调用 {used} ≤ {nfe}", used <= nfe, state)
            row.append(float((x_end - x_ref).norm(dim=1).mean()))
        table[nfe] = row
        print(f"{nfe:>5} " + "".join(f"{v:>14.5f}" for v in row))

    for j, k in enumerate(kinds):
        seq = [table[n][j] for n in nfes]
        assert_true(f"{NAME[k]} 离散化误差随 NFE 单调下降",
                    all(seq[i + 1] < seq[i] for i in range(len(seq) - 1)), state)

    e = {k: table[nfes[-1]][j] for j, k in enumerate(kinds)}
    assert_true("NFE=80：DPM-2 最优", e["dpm2"] == min(e.values()), state)
    assert_true("NFE=80：DPM-2 优于 DDIM", e["dpm2"] < e["ddim"], state)
    assert_true("NFE=80：Midpoint(x) 优于 Euler(x)", e["midpoint_x"] < e["euler_x"], state)

    # 低预算下二阶方法反而更差：每次迭代两次网络调用，步数少一半
    assert_true("NFE=20：Midpoint(x) 劣于 Euler(x)（步数减半吃掉阶数优势）",
                table[20][1] > table[20][0], state)
    assert_true("NFE=40：Midpoint(x) 反超 Euler(x)", table[40][1] < table[40][0], state)
    # 一阶的两种写法：解析积掉线性部分在粗网格上并不占优
    assert_true("NFE=80：DDIM 劣于 Euler(x)（e^λ 放大 x̂_0 的估计误差）",
                e["ddim"] > e["euler_x"], state)

    print("\n  NFE=80 同预算：" + "，".join(f"{NAME[k]} {e[k]:.5f}" for k in kinds))
    print(f"  DPM-2 相对 DDIM 改善 {e['ddim'] / e['dpm2']:.2f}×，相对 Euler(x) 改善 {e['euler_x'] / e['dpm2']:.2f}×")

    # 参考解确实落在数据分布上；同时给出 sliced W2 自身的分辨力本底
    g2 = gen(SEED + 33)
    ref_all = sample_data(40000, g2)
    ref_b = sample_data(40000, g2)
    floor = sliced_w2(ref_all[:8000], ref_b[:8000], g2)   # 与 x_ref 同规模，才可比
    d_ref = sliced_w2(x_ref, ref_all[:8000], g2)
    print(f"\n  参考解到数据的 sliced W2 = {d_ref:.5f}；两个独立样本间的同统计量 = {floor:.5f}（估计本底）")
    assert_true(f"参考解与数据可分（{d_ref:.4f} 不低于本底 {floor:.4f} 的一半）", d_ref > 0.5 * floor, state)
    assert_true(f"参考解到数据距离接近本底（{d_ref:.4f} ≤ 1.6×{floor:.4f}）", d_ref <= 1.6 * floor, state)
    return table


# ------------------------------------------------------------------ E
def exp_E(state):
    """误差阶：log(err) 对 log(NFE) 的斜率。"""
    section("E 组：收敛阶拟合")
    g = gen(SEED + 4)
    aT, sT = float(alpha_of(T_MAX)), float(sigma_of(T_MAX))
    x_init = sample_mixture(aT, sT, 8000, g)
    x_ref = reference(x_init)
    nfes = [8, 16, 32, 64]
    kinds = ["euler_x", "ddim", "midpoint_x", "dpm2", "dpm2m"]
    order = {}
    for k in kinds:
        errs = [float((run(k, n, x_init)[0] - x_ref).norm(dim=1).mean()) for n in nfes]
        lx = torch.log(torch.tensor([float(n) for n in nfes]))
        ly = torch.log(torch.tensor(errs))
        b = float(((lx - lx.mean()) * (ly - ly.mean())).sum() / ((lx - lx.mean()) ** 2).sum())
        order[k] = -b
        print(f"  {NAME[k]:>14}  斜率 = {-b:.3f}   误差 {errs[0]:.5f} → {errs[-1]:.6f}")
    assert_true(f"Euler(x) 阶数 ≈1（得 {order['euler_x']:.2f}）", 0.85 <= order["euler_x"] <= 1.2, state)
    assert_true(f"DDIM 阶数 ≈1（得 {order['ddim']:.2f}）", 0.85 <= order["ddim"] <= 1.2, state)
    assert_true(f"Midpoint(x) 阶数 ≈2（得 {order['midpoint_x']:.2f}）", 1.7 <= order["midpoint_x"] <= 2.3, state)
    assert_true(f"DPM-Solver-2 阶数 ≈2（得 {order['dpm2']:.2f}）", 1.7 <= order["dpm2"] <= 2.3, state)
    assert_true(f"DPM-Solver-2M 阶数 >1.5（得 {order['dpm2m']:.2f}，低预算起步段拉高斜率）",
                order["dpm2m"] > 1.5, state)


# ------------------------------------------------------------------ F
def exp_F(state):
    """取点规则：均匀 t 还是均匀 λ。"""
    section("F 组：取点规则（uniform-t vs uniform-λ）")
    g = gen(SEED + 5)
    aT, sT = float(alpha_of(T_MAX)), float(sigma_of(T_MAX))
    x_init = sample_mixture(aT, sT, 8000, g)
    x_ref = reference(x_init)
    print(f"{'NFE':>5} {'DDIM(t)':>12} {'DDIM(λ)':>12} {'DPM2(t)':>12} {'DPM2(λ)':>12}")
    for nfe in (10, 20, 40):
        vals = []
        for k in ("ddim", "dpm2"):
            for mode in ("t", "lam"):
                vals.append(float((run(k, nfe, x_init, mode)[0] - x_ref).norm(dim=1).mean()))
        print(f"{nfe:>5} {vals[0]:>12.5f} {vals[1]:>12.5f} {vals[2]:>12.5f} {vals[3]:>12.5f}")
        assert_true(f"NFE={nfe} 二阶偏好 uniform-λ（DPM2(λ) < DPM2(t)）", vals[3] < vals[2], state)
        assert_true(f"NFE={nfe} 一阶偏好 uniform-t（DDIM(t) < DDIM(λ)）", vals[0] < vals[1], state)
    print("  同一个网格换一个求解器就换一个结论：阶数与取点规则必须配套，不能各自最优再拼起来")


# ------------------------------------------------------------------ G
def exp_G(figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = gen(SEED + 6)
    aT, sT = float(alpha_of(T_MAX)), float(sigma_of(T_MAX))
    x_init = sample_mixture(aT, sT, 8000, g)
    x_ref = reference(x_init)
    nfes = [5, 10, 20, 40, 80, 160, 320]
    kinds = ["euler_x", "midpoint_x", "ddim", "dpm2", "dpm2m"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
    for k in kinds:
        ys = [float((run(k, n, x_init)[0] - x_ref).norm(dim=1).mean()) for n in nfes]
        ax.loglog(nfes, ys, marker="o", ms=3.5, lw=1.4, label=NAME[k])
    ax.set_xlabel("NFE (network evaluations)")
    ax.set_ylabel("mean L2 to high-accuracy reference")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{figdir}/sampler_nfe_error.png")
    plt.close(fig)

    fig, axs = plt.subplots(1, 5, figsize=(13, 2.8), dpi=160)
    g2 = gen(SEED + 66)
    data = sample_data(6000, g2)
    for ax, k in zip(axs, kinds):
        x_end, _ = run(k, 10, x_init[:6000])
        ax.scatter(data[:, 0], data[:, 1], s=1.2, c="#B5D4F4", alpha=0.6, lw=0)
        ax.scatter(x_end[:, 0], x_end[:, 1], s=1.2, c="#185FA5", alpha=0.6, lw=0)
        ax.set_title(f"{NAME[k]} @ NFE=10", fontsize=8)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(f"{figdir}/sampler_clouds.png")
    plt.close(fig)
    print(f"  已写出 {figdir}/sampler_nfe_error.png 与 sampler_clouds.png")


# ---------------------------------------------------------------- main
def main(argv):
    groups = [a for a in argv if not a.startswith("-")]
    figdir = None
    if "--figdir" in argv:
        figdir = argv[argv.index("--figdir") + 1]
        groups = [a for a in groups if a != figdir]

    groups = groups or ["A", "B", "C", "D", "E", "F"]
    state, shared = [], {}
    table = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "F": exp_F}
    for gname in groups:
        if gname == "E":
            exp_E(state)
        else:
            table[gname](state)
    if figdir:
        exp_G(figdir)

    print()
    print("=" * 78)
    print(f"断言总数 = {len(state)}")
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv[1:])
