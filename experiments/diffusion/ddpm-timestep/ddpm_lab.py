"""DDPM 时间步 / 调度 / 采样器 · 八组核验（PyTorch 版）。

配套博文：《DDPM 深度剖析：前向闭式、时间步与 DDIM 采样》
（lrypcy.github.io）

纯 torch + matplotlib，纯 CPU、float64。每组末尾自断言，汇总断言总数。
用法：
    python ddpm_lab.py                      # 全部 A、A2、B–F、H、W
    python ddpm_lab.py A2 C                 # 只跑指定组
    python ddpm_lab.py --figdir DIR         # 额外输出 G 组（配图）
"""

import sys

import torch

torch.set_default_dtype(torch.float64)     # int/group 里的恒等式要 float64 才站得住

T_DEFAULT = 1000
BETA_START, BETA_END = 1e-4, 0.02
COS_S = 0.008
SEED = 20261101


def fmt(v, n=4):
    return f"{v:.{n}f}"


def gen(seed):
    return torch.Generator().manual_seed(seed)


def randn(shape, g):
    return torch.randn(shape, generator=g)


def assert_close(name, got, want, tol, state):
    err = float((torch.as_tensor(got) - torch.as_tensor(want)).abs().max())
    assert err <= tol, f"[{name}] err={err:.3e} > tol={tol:.3e}"
    state.append((name, err))
    print(f"    {name:<52s} err = {err:.3e}  [ok]")
    return err


# ---------------------------------------------------------------- schedule
def linear_betas(T=T_DEFAULT, b0=BETA_START, b1=BETA_END):
    """DDPM 原论文 §4 的设置：beta_t 线性从 1e-4 到 0.02，T=1000。"""
    return torch.linspace(b0, b1, T)


def betas_for_alpha_bar(alpha_bar_fn, T=T_DEFAULT, max_beta=0.999):
    """Improved DDPM §3.2：先给 \\bar alpha 曲线，再反解 beta_t = 1 - \\bar a_t / \\bar a_{t-1}。"""
    u = torch.arange(1, T + 1, dtype=torch.float64) / T
    ab = alpha_bar_fn(u)
    betas = 1.0 - ab / torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])
    return torch.clamp(betas, 0.0, max_beta)


def cosine_betas(T=T_DEFAULT, s=COS_S):
    def f(u):
        ang = ((u + s) / (1.0 + s)) * torch.pi / 2.0
        return torch.cos(ang) ** 2 / torch.cos(torch.tensor(s * torch.pi / 2.0)) ** 2

    return betas_for_alpha_bar(f, T)


def cumulative(betas):
    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    return alphas, abar


def pad1(abar):
    """补上 t=0 的位置：\\bar alpha_0 = 1。索引即时间步。"""
    return torch.cat([torch.ones(1, dtype=torch.float64), abar])


# ---------------------------------------------------------------- forward
def q_sample(x0, abar_t, eps):
    """闭式：x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps。"""
    return abar_t.sqrt() * x0 + (1.0 - abar_t).sqrt() * eps


# ---------------------------------------------------------------- group A
def exp_A(rng, T=T_DEFAULT):
    """递推 t 次 == 一步闭式（分布层面）。"""
    print("\n[A] 逐步递推 vs 一步闭式")
    betas = linear_betas(T)
    _, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 4
    x0 = randn(d, rng)
    x0 = x0 / x0.norm() * d ** 0.5
    N = 200000

    for t in (1, 10, 100, 500, 1000):
        # 逐维、逐步递推 t 次
        x = x0[None, :].repeat(N, 1)
        for s in range(1, t + 1):
            x = (1.0 - betas[s - 1]).sqrt() * x + betas[s - 1].sqrt() * randn((N, d), rng)
        mean_emp = x.mean(dim=0)
        var_emp = x.var(dim=0).mean()
        want_mean = abar_p[t].sqrt() * x0
        want_var = 1.0 - abar_p[t]
        mc = 0.05 * max(want_var.sqrt().item(), 1e-6)   # Monte Carlo 误差容忍
        st = []
        assert_close(f"t={t:<4d} 递推均值 = sqrt(abar_t) x0", mean_emp, want_mean, 0.03, st)
        assert_close(f"t={t:<4d} 递推方差 = 1 - abar_t", var_emp, want_var, max(0.05, mc), st)
    return 10


# ---------------------------------------------------------------- group A2
def exp_A2(T=T_DEFAULT):
    """方差账本：由 α_t + β_t = 1 推出的三条恒等式，以及取对数后的加性。"""
    print("\n[A2] 方差账本：闭式推导里用到的三条代数关系")
    betas = linear_betas(T)
    alphas, abar = cumulative(betas)
    abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])   # ᾱ_0 = 1
    st = []

    # 两层展开：把 x_{t-2} 推到 x_t 的方差恰好是 1 - α_t α_{t-1}
    lhs = alphas[1:] * betas[:-1] + betas[1:]
    rhs = 1.0 - alphas[1:] * alphas[:-1]
    assert_close("两层展开 α_t β_{t-1} + β_t = 1 - α_t α_{t-1}", lhs, rhs, 1e-12, st)

    # 归纳一步：α_t (1 - ᾱ_{t-1}) + β_t = 1 - ᾱ_t
    lhs = alphas * (1.0 - abar_prev) + betas
    assert_close("归纳一步 α_t (1-ᾱ_{t-1}) + β_t = 1-ᾱ_t", lhs, 1.0 - abar, 1e-12, st)

    # 望远镜求和：第 s 步注入的 β_s 之后被每一步 α_u 衰减，总和 = 1 - ᾱ_t
    # ∏_{u=s+1}^{t} α_u = ᾱ_t / ᾱ_s，所以 Σ_{s≤t} β_s ∏_{u>s} α_u = ᾱ_t Σ_{s≤t} β_s/ᾱ_s
    tel = abar * torch.cumsum(betas / abar, 0)
    assert_close("望远镜求和 Σ_s β_s ∏_{u>s} α_u = 1-ᾱ_t", tel, 1.0 - abar, 1e-9, st)

    # 取对数后 β 变加性：log ᾱ_T ≈ -Σβ - ½Σβ²
    sum_b = float(betas.sum())
    half_sq = float(0.5 * (betas ** 2).sum())
    approx = float(torch.exp(torch.tensor(-sum_b - half_sq)))
    exact = float(abar[-1])
    rel = abs(approx - exact) / exact
    print(f"    Σβ_s = {fmt(sum_b, 4)}，½Σβ_s² = {fmt(half_sq, 4)}，"
          f"exp(-Σ-½Σ²) = {approx:.4e}，实测 ᾱ_T = {exact:.4e}，相对误差 {rel:.2e}")
    assert rel < 2e-2, f"[对数展开] 相对误差 {rel:.3e} > 2e-2"
    st.append(("对数展开 ᾱ_T ≈ exp(-Σβ - ½Σβ²)", rel))
    return 4


# ---------------------------------------------------------------- group W
def exp_W(T=T_DEFAULT):
    """ELBO 的时间权重：L_simple 把每一项的系数都换成 1，丢掉的到底是什么。"""
    print("\n[W] ELBO 的时间权重：L_simple 丢掉的是什么")
    betas = linear_betas(T)
    alphas, abar = cumulative(betas)
    st = []

    # 只看 t >= 2：t = 1 时 ᾱ_0 = 1，后验方差为 0，那一项由解码项 L_0 接管
    b, a = betas[1:], alphas[1:]
    at, ap = abar[1:], abar[:-1]
    beta_tilde = (1.0 - ap) / (1.0 - at) * b
    w_kernel = b ** 2 / (2.0 * beta_tilde * a * (1.0 - at))   # 从 KL 的 1/(2β̃) 推出来
    w_closed = b / (2.0 * a * (1.0 - ap))                     # 化简后的闭式
    assert_close("权重 β²/(2β̃ α (1-ᾱ_t)) = β/(2α(1-ᾱ_{t-1}))", w_kernel, w_closed, 1e-12, st)

    cells = "  ".join(f"t={t}: {float(w_closed[t - 2]):.4f}" for t in (2, 10, 100, 500, 1000))
    print(f"    {cells}")
    ratio = float(w_closed[0] / w_closed[498])                # t=2 与 t=500 的比值
    print(f"    t=2 的权重是 t=500 的 {ratio:.1f} 倍 → L_simple 等于相对抬高大 t")

    assert float(w_closed.max()) < 1.0, "[权重] 不该出现大于 1 的系数"
    st.append(("权重上界 < 1", 0.0))
    assert 50.0 < ratio < 500.0, f"[权重比] {ratio:.1f} 不在预期量级"
    st.append(("t=2 与 t=500 的权重比", ratio))
    return 3


# ---------------------------------------------------------------- group B
def exp_B(rng, T=T_DEFAULT):
    """t 决定噪声等级，eps 决定具体实例：同一个 t 的两次采样不同但同分布。"""
    print("\n[B] 同一 t、不同 eps：实例不同，等级相同")
    betas = linear_betas(T)
    _, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 8
    x0 = randn(d, rng)
    x0 = x0 / x0.norm()
    t = 500
    N = 200000
    eps = randn((2, N, d), rng)
    xs = q_sample(x0[None, None, :], abar_p[t], eps)          # (2, N, d)
    dist = torch.linalg.vector_norm(xs[0] - xs[1], dim=1).mean()

    # E||x^(1) - x^(2)||^2 = 2 d (1 - abar_t)
    want_dist_sq = 2.0 * d * (1.0 - abar_p[t])
    emp_sq = (torch.linalg.vector_norm(xs[0] - xs[1], dim=1) ** 2).mean()

    st = []
    assert_close("两实例间的均方距离 = 2 d (1 - abar_t)", emp_sq, want_dist_sq, 0.6, st)
    for i in range(2):
        assert_close(f"实例 {i+1} 的方差 = 1 - abar_t", xs[i].var(), 1.0 - abar_p[t], 0.02, st)
    print(f"    abar_{t} = {fmt(abar_p[t].item(), 6)}  E||x1-x2|| = {fmt(dist.item())}")
    return 3


# ---------------------------------------------------------------- group C
def exp_C(rng, T=T_DEFAULT):
    """后验均值三条写法逐点相等。"""
    print("\n[C] 后验均值：三种写法是同一个向量")
    betas = linear_betas(T)
    alphas, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 6
    x0 = randn(d, rng)
    eps = randn(d, rng)
    st = []

    for t in (2, 50, 300, 900, 1000):
        xt = q_sample(x0, abar_p[t], eps)
        a_t = alphas[t - 1]
        b_t = betas[t - 1]
        ab_prev = abar_p[t - 1]

        # 写法一：DDPM 论文式 (7) 的后验均值
        mu_1 = (ab_prev.sqrt() * b_t) / (1.0 - abar_p[t]) * x0 + \
               (a_t.sqrt() * (1.0 - ab_prev)) / (1.0 - abar_p[t]) * xt

        # 写法二：用 eps 表达的一步去噪（Ho et al. Algorithm 2）
        mu_2 = (xt - b_t / (1.0 - abar_p[t]).sqrt() * eps) / a_t.sqrt()

        # 写法三：先反解 x_hat0，再按 diffusers 的系数加权组合
        x_hat0 = (xt - (1.0 - abar_p[t]).sqrt() * eps) / abar_p[t].sqrt()
        c1 = ab_prev.sqrt() * b_t / (1.0 - abar_p[t])
        c2 = a_t.sqrt() * (1.0 - ab_prev) / (1.0 - abar_p[t])
        mu_3 = c1 * x_hat0 + c2 * xt

        assert_close(f"t={t:<4d} 论文式均值 = eps 式均值", mu_1, mu_2, 1e-10, st)
        assert_close(f"t={t:<4d} x_hat0 组合式 = 论文式均值", mu_3, mu_1, 1e-10, st)
        # 后验方差的两条写法
        beta_tilde = (1.0 - ab_prev) / (1.0 - abar_p[t]) * b_t
        assert_close(f"t={t:<4d} beta_tilde 一致", beta_tilde,
                     (1.0 - ab_prev) / (1.0 - abar_p[t]) * (1.0 - a_t), 1e-12, st)
    return 15


# ---------------------------------------------------------------- group D
def ddim_step(x, eps_hat, m, p, abar_p, z, eta):
    """diffusers DDIM 的更新式（time index m -> p，p < m，p 可为 0）。"""
    abar_m, abar_next = abar_p[m], abar_p[p]
    x_hat0 = (x - (1.0 - abar_m).sqrt() * eps_hat) / abar_m.sqrt()
    var_term = 1.0 - abar_m / abar_next
    sigma2 = eta * (1.0 - abar_next) / (1.0 - abar_m) * var_term
    coef = torch.clamp(1.0 - abar_next - sigma2, min=0.0).sqrt()
    return abar_next.sqrt() * x_hat0 + coef * eps_hat + sigma2.sqrt() * z, x_hat0, sigma2


def make_error_operator(d, seed=SEED):
    """一个固定的线性“网络残差”：预测误差取 rho W x_t。

    误差必须**依赖当前状态**：如果每步注入的是与 x_t 无关的白噪声，那么
    x_hat0 = (x_t - sqrt(1-abar_t) eps_hat)/sqrt(abar_t) = x0 - 稍后 x_t 无关的量，
    与历史完全无关，轨迹每步都在重置，结果只由最后一段决定——那样的 toy
    里 eta 与中间路径都不起作用，测出来的“步数越多越准”是假的。
    """
    return randn((d, d), gen(seed)) / d ** 0.5


def ddim_trajectory(x0, x_start, steps, eta, rho, seed, abar_p, W=None):
    """跑一条 DDIM 轨迹并返回终点。T 由 abar_p 的长度推出。"""
    g = gen(seed)
    big_t = abar_p.numel() - 1
    tau = torch.linspace(big_t, 0, steps + 1).round().long()
    x = x_start.clone()
    for i in range(steps):
        m, p = int(tau[i]), int(tau[i + 1])
        eps_true = (x - abar_p[m].sqrt() * x0) / (1.0 - abar_p[m]).sqrt()
        err = rho * (W @ x) if W is not None else rho * randn(x.shape, g)
        z = randn(x.shape, g)
        x, _, _ = ddim_step(x, eps_true + err, m, p, abar_p, z, eta)
    return x


def exp_D(rng, T=T_DEFAULT):
    """DDIM：oracle 下任意步数直达 x0；模型有误差时，误差随步数如何变化。"""
    print("\n[D] DDIM 的步数与误差")
    betas = linear_betas(T)
    _, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 16
    W = make_error_operator(d)
    x0 = randn(d, rng)
    x0 = x0 / x0.norm()

    def run(steps, rho, seed):
        g = gen(seed)
        xa = randn(d, g)
        xa = xa / xa.norm()
        xT = q_sample(xa, abar_p[T], randn(d, g))
        x = ddim_trajectory(xa, xT, steps, 0.0, rho, seed, abar_p, W)
        return torch.linalg.vector_norm(x - xa) / xa.norm()

    st = []
    # (1) oracle：任意步数都精确落到 x0
    print("    eta=0、oracle 噪声：")
    for steps in (1, 2, 5, 20, 50, 200, 1000):
        err = run(steps, 0.0, seed=1).item()
        assert err < 1e-8, f"[oracle steps={steps}] err={err:.3e}"
        print(f"      steps={steps:<5d}  相对误差 = {err:.2e}  [ok]")
        st.append((f"oracle steps={steps}", err))

    # (2) 一步到位的误差放大：x_final - x0 = -sqrt((1-abar_T)/abar_T) * delta，逐点可证
    eps = randn(d, rng)
    xT = q_sample(x0, abar_p[T], eps)
    rho_step, g7 = 0.01, gen(7)
    delta = rho_step * randn(d, g7)
    eps_hat = eps + delta
    x_final, _, _ = ddim_step(xT, eps_hat, T, 0, abar_p, torch.zeros(d), 0.0)
    ampl = ((1.0 - abar_p[T]) / abar_p[T]).sqrt()
    assert_close("单步误差 = sqrt((1-abar_T)/abar_T) * ||delta||",
                 torch.linalg.vector_norm(x_final - x0),
                 ampl * torch.linalg.vector_norm(delta), 1e-9, st)
    print(f"      abar_T = {abar_p[T].item():.3e}  放大系数 sqrt((1-abar_T)/abar_T) = {fmt(ampl.item(), 1)}")

    # (3) 有模型误差时：误差随步数单调下降
    print("    eta=0、每步注入 1% 相对误差（误差取 rho W x_t，依赖当前状态）：")
    prev, rows = float("inf"), []
    for steps in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        errs = [run(steps, rho_step, seed=100 + k).item() for k in range(20)]
        m = float(sum(errs) / len(errs))
        rows.append((steps, m))
        print(f"      steps={steps:<5d}  平均相对误差 = {m:.4f}")
        assert m <= prev * 1.02, f"steps={steps} 误差回升：{m} > {prev}"
        prev = m
        st.append((f"steps={steps} 单调", m))
    print("    误差随 U-Net 调用次数单调下降，1000 步时已压到 1 步时的 "
          f"{rows[-1][1] / rows[0][1]:.2e} 倍 -> [ok]")
    return len(st)


# ---------------------------------------------------------------- group E
def snr_db(abar_p):
    return 10.0 * torch.log10(abar_p / torch.clamp(1.0 - abar_p, 1e-300))


def exp_E(T=T_DEFAULT):
    """线性 vs 余弦：步数到底花在哪些 SNR 区间。"""
    print("\n[E] 线性调度 vs 余弦调度")
    st = []
    out = {}
    for name, betas in (("linear", linear_betas(T)), ("cosine", cosine_betas(T))):
        _, abar = cumulative(betas)
        abar_p = pad1(abar)
        snr = snr_db(abar_p[1:])
        zero = int(torch.argmax((snr < 0.0).float())) + 1        # 第一次跌到 0 dB 以下
        bands = {
            "clean  (> 10 dB)": float((snr > 10.0).float().mean()),
            "mid    (-10..10 dB)": float(((snr > -10.0) & (snr <= 10.0)).float().mean()),
            "noise  (< -10 dB)": float((snr <= -10.0).float().mean()),
        }
        btab = betas[[0, 99, 299, 499, 999]]
        out[name] = dict(snr=snr, zero=zero, bands=bands, btab=btab, abar_p=abar_p, betas=betas)
        print(f"    {name:<7s} beta@[1,100,300,500,1000] = "
              + ", ".join(fmt(v.item(), 5) for v in btab))
        print(f"    {name:<7s} SNR 首次 < 0 dB 于 t = {zero:<4d} ｜ abar_T = {abar_p[T].item():.3e}")
        for k, v in bands.items():
            print(f"      {k}: {100 * v:5.1f}% 的步数")
    # 断言：余弦把更多步数放在中等 SNR（信息量最大）的区间
    cm = out["cosine"]["bands"]["mid    (-10..10 dB)"]
    lm = out["linear"]["bands"]["mid    (-10..10 dB)"]
    cn = out["cosine"]["bands"]["noise  (< -10 dB)"]
    ln = out["linear"]["bands"]["noise  (< -10 dB)"]
    print(f"    中等 SNR 区间占比：cosine {100 * cm:.1f}% > linear {100 * lm:.1f}%")
    print(f"    几乎纯噪声区间占比：cosine {100 * cn:.1f}% < linear {100 * ln:.1f}%")
    assert cm > lm, "余弦应把更多步数放在中等 SNR 区间"
    assert cn < ln, "余弦应在几乎纯噪声的区间上少花步数"
    print("    -> [ok]")
    # 断言：论文表上的线性 beta 取值
    expected = torch.tensor([1e-4, 0.0021, 0.0060, 0.0100, 0.0200], dtype=torch.float64)
    assert_close("线性 beta 取值 ≈ 论文/Clipping 表", out["linear"]["btab"], expected, 5e-4, st)
    # 单调性
    for name in ("linear", "cosine"):
        assert bool(torch.all(torch.diff(out[name]["abar_p"]) < 0)), f"{name} abar 应单调下降"
        st.append((f"{name} abar 单调", 0.0))
    return len(st) + 2


# ---------------------------------------------------------------- group F
def set_timesteps(num_train, num_inference, spacing, steps_offset=0):
    """从训练时间步里挑出推理时间步的三种常见规则（降序返回）。

    - endpoint   端点插值：round(i * (T-1) / (N-1))，必含两端 T-1 与 0
    - stride     整数步长：round(i * (T // N))，步长取整，不含 T-1；再加 steps_offset
    - trailing   后缘对齐：round((i+1) * T / N) - 1，最小值不落在 0 上
    """
    i = torch.arange(num_inference, dtype=torch.float64)
    if spacing == "endpoint":
        ts = torch.round(i * (num_train - 1) / max(num_inference - 1, 1))
    elif spacing == "stride":
        ts = torch.round(i * (num_train // num_inference)) + steps_offset
    elif spacing == "trailing":
        ts = torch.round((i + 1) * num_train / num_inference) - 1
    else:
        raise ValueError(spacing)
    ts = ts.long()
    return torch.flip(ts, dims=[0])


def exp_F(T=T_DEFAULT):
    """推理时间步怎么从 1000 个里挑出来：同样 50 次调用，采样路径可以不同。"""
    print("\n[F] set_timesteps：三种取点规则")
    st = []
    for N in (5, 10, 20, 50):
        print(f"    N={N}")
        for name in ("endpoint", "stride", "trailing"):
            ts = set_timesteps(T, N, name)
            assert len(ts) == N, f"{name} N={N} 长度不对"
            assert bool(torch.all(torch.diff(ts) < 0)), f"{name} N={N} 必须严格降序"
            assert int(ts.min()) >= 0 and int(ts.max()) < T, f"{name} N={N} 越界"
            assert len(set(ts.tolist())) == N, f"{name} N={N} 有重复"
            st.append((f"{name} N={N} 不变量", 0.0))
            print(f"      {name:<9s}: {[int(v) for v in ts]}")
    ts_ep = set_timesteps(T, 50, "endpoint")
    ts_st = set_timesteps(T, 50, "stride")
    ts_tr = set_timesteps(T, 50, "trailing")
    print(f"    50 步时 stride 的最大时间步 = {int(ts_st.max())}，trailing = {int(ts_tr.max())}，"
          f"endpoint = {int(ts_ep.max())}")
    assert int(ts_st.max()) < int(ts_ep.max()), "整数步长版不会覆盖最大训练时间步"
    st.append(("stride 不覆盖 T-1", 0.0))
    print("    同样 50 次 U-Net 调用，stride 与 endpoint 走的是两条不同的采样路径 -> [ok]")
    return len(st)


# ---------------------------------------------------------------- group H
def exp_H(rng, T=T_DEFAULT):
    """最后一里地的随机项：它给的是多样性，不是精度。"""
    print("\n[H] DDPM 的随机项在做什么")
    betas = linear_betas(T)
    alphas, abar = cumulative(betas)
    abar_p = pad1(abar)
    d = 16
    st = []

    x0 = randn(d, rng)
    x0 = x0 / x0.norm()
    xT = q_sample(x0, abar_p[T], randn(d, rng))     # 固定同一个出发点 x_T
    W = make_error_operator(d)

    def run(steps, eta, rho, seed, *, tgt=x0, start=xT, op=W):
        """默认用上面固定的 (x_T, 误差算子)；也可以按组传入不同的。"""
        return ddim_trajectory(tgt, start, steps, eta, rho, seed, abar_p,
                               None if rho == 0.0 else op)

    def rel_norm(xs, ref):
        return torch.linalg.vector_norm(xs - ref, dim=1) / ref.norm()

    # (1) 随机项相对目标尺度的占比：最后一里地反而不小
    print("    随机项 sigma_t 与目标尺度 sqrt(1 - abar_{t-1})：")
    for t in (1000, 750, 500, 250, 100, 10, 2):
        b_tilde = (1.0 - abar_p[t - 1]) / (1.0 - abar_p[t]) * betas[t - 1]
        sig = b_tilde.sqrt()
        scale = (1.0 - abar_p[t - 1]).sqrt()
        ratio = 100 * (sig / scale).item()
        print(f"      t={t:<5d} sigma_t = {fmt(sig.item(), 4)} ｜ sqrt(1-abar_{{t-1}}) = "
              f"{fmt(scale.item(), 4)} ｜ 占比 = {ratio:5.2f}%")
        assert 0.0 < sig.item() < 0.25, "后验标准差不应超过 0.25"
        st.append((f"sigma_{t} 合理", sig.item()))
    assert_close("第一步的随机项 = sqrt(beta_1)", betas[0].sqrt(),
                 torch.tensor(BETA_START).sqrt(), 1e-12, st)

    # (2) oracle 下：eta 根本不起作用——完美模型不需要也没拿随机项换来多样性
    print("    oracle 噪声、同一起点 x_T，换 20 条随机流：")
    for eta in (0.0, 1.0):
        xs = torch.stack([run(1000, eta, 0.0, seed=500 + k) for k in range(20)])
        mean_x = xs.mean(dim=0)
        spread = float(rel_norm(xs, mean_x).mean())
        dev = float(rel_norm(xs, x0).mean())
        print(f"      eta={eta:<4.0f}  mean||x_k - x0||/||x0|| = {fmt(dev, 5)}"
              f"  ｜  多样度 mean||x_k - x_bar||/||x0|| = {fmt(spread, 5)}")
        assert dev < 1e-8, f"eta={eta} oracle 下应精确回到 x0"
        st.append((f"eta={eta} oracle 回到 x0", dev))
    print("      -> 模型完美时 ODE(eta=0) 与 SDE(eta=1) 殊途同归，随机项是白注入")

    # (3) 有模型误差时：随机项才显出作用——多要多样性，就得多付精度
    print("    每步注入 1% 相对误差：8 组不同的 (x0, x_T, 误差算子) x 10 条随机流取平均：")
    n_trial, n_flow = 8, 10
    prev_spr = float("inf")
    for steps in (20, 50, 100):
        err_sum = {0.0: 0.0, 1.0: 0.0}
        spr_sum = {0.0: 0.0, 1.0: 0.0}
        for trial in range(n_trial):
            g3 = gen(9000 + trial)
            xa = randn(d, g3)
            xa = xa / xa.norm()
            xsa = q_sample(xa, abar_p[T], randn(d, g3))
            op = make_error_operator(d, seed=SEED + 11111 + trial)
            for eta in (0.0, 1.0):
                xs = torch.stack([run(steps, eta, 0.01, seed=700 + k, tgt=xa, start=xsa, op=op)
                                  for k in range(n_flow)])
                err_sum[eta] += float(rel_norm(xs, xa).mean())
                spr_sum[eta] += float(rel_norm(xs, xs.mean(dim=0)).mean())
        err = {e: err_sum[e] / n_trial for e in err_sum}
        spr = {e: spr_sum[e] / n_trial for e in spr_sum}
        print(f"      steps={steps:<4d}  eta=0: 误差 {fmt(err[0.0], 5)} ｜ 多样度 {fmt(spr[0.0], 5)}"
              f"  ｜  eta=1: 误差 {fmt(err[1.0], 5)} ｜ 多样度 {fmt(spr[1.0], 5)}"
              f"  ｜  误差比 {fmt(err[1.0] / err[0.0], 3)}x"
              f" ｜ 多样度占偏差 {100 * spr[1.0] / err[1.0]:5.1f}%")
        # eta=0 是确定性映射：换随机流换不出第二个样本
        assert spr[0.0] < 1e-12, f"steps={steps} eta=0 必须是确定的"
        # eta=1 才是一个采样器：同一 x_T 得到一片样本
        assert spr[1.0] > 0.0, f"steps={steps} eta=1 应给出互不相同的样本"
        # 但它并没有把 x0 找得更准：精度是同一个量级，不是数量级的差别
        assert 0.5 < err[1.0] / err[0.0] < 2.0, f"steps={steps} 随机项不应改变精度的量级"
        # 多样性本身也随步数走：步数越少，注入的 sigma 越大，样本铺得越开
        assert spr[1.0] < prev_spr, f"steps={steps} 多样性应随步数下降"
        prev_spr = spr[1.0]
        st.append((f"steps={steps} eta=0 确定性", spr[0.0]))
        st.append((f"steps={steps} eta=1 多样性", spr[1.0]))
        st.append((f"steps={steps} 精度同量级", err[1.0] / err[0.0]))
    print("      -> 随机项的作用是给出一片样本，而不是把 x0 找得更准；它的散布量本身也随步数下降")
    return len(st)


# ---------------------------------------------------------------- figures
def exp_G(figdir, T=T_DEFAULT):
    """生成博文配图。"""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    betas_lin = linear_betas(T)
    betas_cos = cosine_betas(T)
    _, abar_lin = cumulative(betas_lin)
    _, abar_cos = cumulative(betas_cos)
    abar_lin_p, abar_cos_p = pad1(abar_lin), pad1(abar_cos)
    t = torch.arange(1, T + 1, dtype=torch.float64).numpy()

    plt.rcParams.update({
        "font.family": ["Heiti TC", "Songti SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11,
    })

    # 1) beta 与 abar 曲线
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax[0].plot(t, betas_lin.numpy(), color="#c0392b", lw=2, label="linear")
    ax[0].plot(t, betas_cos.numpy(), color="#2471a3", lw=2, label="cosine")
    ax[0].set_xlabel("timestep $t$"); ax[0].set_ylabel(r"$\beta_t$")
    ax[0].set_title("per-step noise strength"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(t, abar_lin_p[1:].numpy(), color="#c0392b", lw=2, label="linear")
    ax[1].plot(t, abar_cos_p[1:].numpy(), color="#2471a3", lw=2, label="cosine")
    ax[1].axhline(0, color="gray", lw=.8, ls=":")
    ax[1].set_xlabel("timestep $t$"); ax[1].set_ylabel(r"$\bar\alpha_t$")
    ax[1].set_title("cumulative signal retention"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_schedule_beta_alpha.png", dpi=150); plt.close(fig)

    # 2) SNR 曲线
    snr_l = snr_db(abar_lin_p[1:]).numpy(); snr_c = snr_db(abar_cos_p[1:]).numpy()
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.plot(t, snr_l, color="#c0392b", lw=2, label="linear")
    ax.plot(t, snr_c, color="#2471a3", lw=2, label="cosine")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.axhline(-10, color="#7b7d7d", lw=1, ls=":")
    ax.text(T * .52, -9.2, "SNR = -10 dB (nearly pure noise)", color="#7b7d7d", fontsize=10)
    ax.set_xlabel("timestep $t$"); ax.set_ylabel(r"SNR  $= \bar\alpha_t/(1-\bar\alpha_t)$   [dB]")
    ax.set_title("Signal-to-noise ratio along the forward process")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_snr_linear_vs_cosine.png", dpi=150); plt.close(fig)

    # 3) 加噪演进图（合成图片）
    g = gen(SEED)
    n = 64
    yy, xx = torch.meshgrid(torch.arange(n) / n, torch.arange(n) / n, indexing="ij")
    img = torch.zeros(n, n, 3, dtype=torch.float64)
    img[..., 0] = torch.clamp(1.3 - torch.hypot(xx - .35, yy - .4) * 1.8, 0, 1)      # 圆盘
    bar = ((xx > .62) & (xx < .72) & (yy > .18) & (yy < .82)).double()               # 竖条
    img[..., 1] = torch.clamp(torch.maximum(bar, 1.1 - torch.hypot(xx - .8, yy - .7) * 2.4), 0, 1)
    img[..., 2] = torch.clamp((yy - .5).abs() * 1.6, 0, 1)
    img = img * 2 - 1                                                                # [-1, 1]
    steps = [0, 2, 10, 50, 100, 200, 400, 700, 1000]
    fig, ax = plt.subplots(1, len(steps), figsize=(15.5, 2.4))
    for i, s in enumerate(steps):
        cur = img.clone() if s == 0 else q_sample(img, abar_lin_p[s], randn(img.shape, g))
        cur = torch.clamp(cur, -1, 1).numpy()
        ax[i].imshow((cur + 1) / 2)
        snr = "clean" if s == 0 else f"{snr_l[s - 1]:.1f} dB"
        ax[i].set_title(f"$t={s}$\n{snr}", fontsize=9)
        ax[i].axis("off")
    fig.suptitle(r"$x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$   (linear schedule, $T=1000$)",
                 y=1.12, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_noise_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) DDIM 步数 vs 误差
    steps_grid = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    means, sds = [], []
    for s in steps_grid:
        e = torch.tensor([one_run(s, 0.0, 0.01, 100 + k, T, abar_lin_p) for k in range(20)])
        means.append(float(e.mean())); sds.append(float(e.std()))
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.errorbar(steps_grid, means, yerr=sds, marker="o", color="#2471a3", capsize=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of U-Net calls (sampling steps)")
    ax.set_ylabel("relative error of $x_0$")
    ax.set_title("DDIM (eta=0) with 1% per-step prediction error")
    ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_ddim_error_vs_steps.png", dpi=150); plt.close(fig)

    # 5) 时间步取点策略
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    for name, ts in (("endpoint", set_timesteps(T, 50, "endpoint")),
                     ("trailing", set_timesteps(T, 50, "trailing"))):
        y = 1 if name == "endpoint" else 0
        ax.scatter(ts.numpy(), np_full(float(y), len(ts)), s=14,
                   label=name, color="#c0392b" if name == "endpoint" else "#2471a3")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["trailing", "endpoint"])
    ax.set_xlabel("training timestep selected by set_timesteps()")
    ax.set_xlim(-20, T + 20); ax.legend(loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_timesteps_selection.png", dpi=150); plt.close(fig)

    print(f"    配图已写入 {figdir}")
    return 0


def np_full(v, n):
    """matplotlib scatter 用：造一条长度 n 的常数序列（避免依赖 numpy）。"""
    return [v] * n


def one_run(steps, eta, rho, seed, T, abar_p):
    """给配图用的一次 DDIM 采样：与 D/H 组共用同一套误差模型。"""
    d = 16
    W = make_error_operator(d)
    g = gen(seed)
    x0 = randn(d, g); x0 = x0 / x0.norm()
    xT = q_sample(x0, abar_p[T], randn(d, g))
    x = ddim_trajectory(x0, xT, steps, eta, rho, seed, abar_p, W)
    return torch.linalg.vector_norm(x - x0) / x0.norm()


def main(argv):
    figdir = None
    groups = []
    skip = False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--figdir":
            idx = argv.index(a) + 1
            figdir = argv[idx] if idx < len(argv) else None
            skip = True
            continue
        if a.startswith("-"):
            continue
        groups.append(a)
    groups = groups or ["A", "A2", "B", "C", "D", "E", "F", "H", "W"]
    rng = gen(SEED)
    total = 0
    print("=" * 72)
    print("DDPM 时间步 / 调度 / 采样器 · 核验（PyTorch）")
    print(f"设置：T={T_DEFAULT}，linear beta {BETA_START}→{BETA_END}，"
          f"cosine offset s={COS_S}，dtype={torch.get_default_dtype()}，"
          f"torch={torch.__version__}")
    print("=" * 72)
    table = {"A": exp_A, "A2": exp_A2, "B": exp_B, "C": exp_C, "D": exp_D,
             "E": exp_E, "F": exp_F, "H": exp_H, "W": exp_W}
    for g in groups:
        total += table[g](rng) if g in ("A", "B", "C", "D", "H") else table[g]()
    if figdir:
        exp_G(figdir)
    print("=" * 72)
    print(f"断言总数 = {total}")
    print("=" * 72)
    return total


if __name__ == "__main__":
    main(sys.argv)
