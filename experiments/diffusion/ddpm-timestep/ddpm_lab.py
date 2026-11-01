"""DDPM 时间步 / 调度 / 采样器 · 六组核验。

配套博文：《DDPM 深度剖析：前向闭式、时间步与 DDIM 采样》
（lrypcy.github.io）

纯 numpy + matplotlib，纯 CPU。每组末尾自断言，汇总断言总数。
用法：
    python3 ddpm_lab.py                      # 全部 A–F
    python3 ddpm_lab.py A C                  # 只跑指定组
    python3 ddpm_lab.py --figdir DIR         # 额外输出 G 组（配图）
"""

import sys

import numpy as np

T_DEFAULT = 1000
BETA_START, BETA_END = 1e-4, 0.02
COS_S = 0.008
TOL = 1e-9


def fmt(v, n=4):
    return f"{v:.{n}f}"


def assert_close(name, got, want, tol, state):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    assert err <= tol, f"[{name}] err={err:.3e} > tol={tol:.3e}"
    state.append((name, err))
    print(f"    {name:<52s} err = {err:.3e}  [ok]")
    return err


# ---------------------------------------------------------------- schedule
def linear_betas(T=T_DEFAULT, b0=BETA_START, b1=BETA_END):
    """DDPM 原论文 §4 的设置：beta_t 线性从 1e-4 到 0.02，T=1000。"""
    return np.linspace(b0, b1, T)


def betas_for_alpha_bar(alpha_bar_fn, T=T_DEFAULT, max_beta=0.999):
    """Improved DDPM §3.2：先给 \bar alpha 曲线，再反解 beta_t = 1 - \bar a_t / \bar a_{t-1}。"""
    t = np.arange(1, T + 1)
    ab = alpha_bar_fn(t / T)
    betas = 1.0 - ab / np.concatenate([[1.0], ab[:-1]])
    return np.clip(betas, 0.0, max_beta)


def cosine_betas(T=T_DEFAULT, s=COS_S):
    def f(u):
        ang = ((u + s) / (1.0 + s)) * np.pi / 2.0
        return np.cos(ang) ** 2 / np.cos(s * np.pi / 2.0) ** 2

    return betas_for_alpha_bar(f, T)


def cumulative(betas):
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return alphas, abar


def pad1(abar):
    """补上 t=0 的位置：\bar alpha_0 = 1。索引即时间步。"""
    return np.concatenate([[1.0], abar])


# ---------------------------------------------------------------- forward
def q_sample(x0, abar_p, eps):
    """闭式：x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps。"""
    return np.sqrt(abar_p) * x0 + np.sqrt(1.0 - abar_p) * eps


# ---------------------------------------------------------------- group A
def exp_A(rng, T=T_DEFAULT):
    """递推 t 次 == 一步闭式（分布层面）。"""
    print("\n[A] 逐步递推 vs 一步闭式")
    betas = linear_betas(T)
    _, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 4
    x0 = rng.normal(size=d)
    x0 = x0 / np.linalg.norm(x0) * np.sqrt(d)
    N = 200000

    for t in (1, 10, 100, 500, 1000):
        eps = rng.normal(size=(N, d))
        # 逐维、逐步递推 t 次
        x = np.repeat(x0[None, :], N, axis=0)
        for s in range(1, t + 1):
            x = np.sqrt(1.0 - betas[s - 1]) * x + np.sqrt(betas[s - 1]) * rng.normal(size=(N, d))
        mean_emp = x.mean(axis=0)
        var_emp = x.var(axis=0).mean()
        want_mean = np.sqrt(abar_p[t]) * x0
        want_var = 1.0 - abar_p[t]
        mc = 0.05 * max(np.sqrt(want_var), 1e-6)  # Monte Carlo 误差容忍
        assert_close(f"t={t:<4d} 递推均值 = sqrt(abar_t) x0", mean_emp, want_mean, 0.03, st := [])
        assert_close(f"t={t:<4d} 递推方差 = 1 - abar_t", var_emp, want_var, max(0.05, mc), st)
    return 10


# ---------------------------------------------------------------- group B
def exp_B(rng, T=T_DEFAULT):
    """t 决定噪声等级，eps 决定具体实例：同一个 t 的两次采样不同但同分布。"""
    print("\n[B] 同一 t、不同 eps：实例不同，等级相同")
    betas = linear_betas(T)
    _, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 8
    x0 = rng.normal(size=d)
    x0 /= np.linalg.norm(x0)
    t = 500
    N = 200000
    eps = rng.normal(size=(2, N, d))
    xs = q_sample(x0[None, None, :], abar_p[t], eps)          # (2, N, d)
    dist = np.linalg.norm(xs[0] - xs[1], axis=1).mean()

    # E||x^(1) - x^(2)||^2 = 2 d (1 - abar_t)
    want_dist_sq = 2.0 * d * (1.0 - abar_p[t])
    emp_sq = (np.linalg.norm(xs[0] - xs[1], axis=1) ** 2).mean()

    st = []
    assert_close("两实例间的均方距离 = 2 d (1 - abar_t)", emp_sq, want_dist_sq, 0.6, st)
    for i in range(2):
        assert_close(f"实例 {i+1} 的方差 = 1 - abar_t", xs[i].var(), 1.0 - abar_p[t], 0.02, st)
    print(f"    abar_{t} = {fmt(abar_p[t], 6)}  E||x1-x2|| = {fmt(dist)}")
    return 3


# ---------------------------------------------------------------- group C
def exp_C(rng, T=T_DEFAULT):
    """后验均值三条写法逐点相等。"""
    print("\n[C] 后验均值：三种写法是同一个向量")
    betas = linear_betas(T)
    alphas, abar = cumulative(betas)
    abar_p = pad1(abar)

    d = 6
    x0 = rng.normal(size=d)
    eps = rng.normal(size=d)
    st = []

    for t in (2, 50, 300, 900, 1000):
        xt = q_sample(x0, abar_p[t], eps)
        a_t = alphas[t - 1]
        b_t = betas[t - 1]
        ab_prev = abar_p[t - 1]

        # 写法一：DDPM 论文式 (7) 的后验均值
        mu_1 = (np.sqrt(ab_prev) * b_t) / (1.0 - abar_p[t]) * x0 + \
               (np.sqrt(a_t) * (1.0 - ab_prev)) / (1.0 - abar_p[t]) * xt

        # 写法二：用 eps 表达的一步去噪（Ho et al. Algorithm 2）
        mu_2 = (xt - b_t / np.sqrt(1.0 - abar_p[t]) * eps) / np.sqrt(a_t)

        # 写法三：先反解 x_hat0，再按 diffusers 的系数加权组合
        x_hat0 = (xt - np.sqrt(1.0 - abar_p[t]) * eps) / np.sqrt(abar_p[t])
        c1 = np.sqrt(ab_prev) * b_t / (1.0 - abar_p[t])
        c2 = np.sqrt(a_t) * (1.0 - ab_prev) / (1.0 - abar_p[t])
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
    abar_m, abar_p_ = abar_p[m], abar_p[p]
    x_hat0 = (x - np.sqrt(1.0 - abar_m) * eps_hat) / np.sqrt(abar_m)
    var_term = 1.0 - abar_m / abar_p_
    sigma2 = eta * (1.0 - abar_p_) / (1.0 - abar_m) * var_term
    coef = np.sqrt(np.clip(1.0 - abar_p_ - sigma2, 0.0, None))
    return np.sqrt(abar_p_) * x_hat0 + coef * eps_hat + np.sqrt(sigma2) * z, x_hat0, sigma2


def make_error_operator(d, seed=20261101):
    """一个固定的线性“网络残差”：预测误差取 \rho\,W x_t。

    误差必须**依赖当前状态**：如果每步注入的是与 x_t 无关的白噪声，那么
    x_hat0 = (x_t - sqrt(1-abar_t) eps_hat)/sqrt(abar_t) = x0 - sqrt(1-abar_t) rho xi/sqrt(abar_t)
    与历史完全无关，轨迹每步都在重置，最终结果只由最后一段决定——那样的 toy
    里 eta 与中间路径都不起作用，测出来的“步数越多越准”是假的。
    """
    return np.random.default_rng(seed).normal(size=(d, d)) / np.sqrt(d)


def ddim_trajectory(x0, x_start, steps, eta, rho, seed, abar_p, W=None):
    """跑一条 DDIM 轨迹并返回终点。T 由 abar_p 的长度推出。"""
    rng2 = np.random.default_rng(seed)
    big_t = len(abar_p) - 1
    tau = np.linspace(big_t, 0, steps + 1).round().astype(int)
    x = np.array(x_start, dtype=float)
    for i in range(steps):
        m, p = int(tau[i]), int(tau[i + 1])
        eps_true = (x - np.sqrt(abar_p[m]) * x0) / np.sqrt(1.0 - abar_p[m] + 1e-300)
        err = rho * (W @ x) if W is not None else rho * rng2.normal(size=x.shape)
        z = rng2.normal(size=x.shape)
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
    x0 = rng.normal(size=d)
    x0 /= np.linalg.norm(x0)

    def run(steps, rho, seed):
        rng2 = np.random.default_rng(seed)
        x0 = rng2.normal(size=d)
        x0 /= np.linalg.norm(x0)
        xT = q_sample(x0, abar_p[T], rng2.normal(size=d))
        x = ddim_trajectory(x0, xT, steps, 0.0, rho, seed, abar_p, W)
        return np.linalg.norm(x - x0) / np.linalg.norm(x0)

    st = []
    # (1) oracle：任意步数都精确落到 x0
    print("    eta=0、oracle 噪声：")
    for steps in (1, 2, 5, 20, 50, 200, 1000):
        err = run(steps, 0.0, seed=1)
        assert err < 1e-8, f"[oracle steps={steps}] err={err:.3e}"
        print(f"      steps={steps:<5d}  相对误差 = {err:.2e}  [ok]")
        st.append((f"oracle steps={steps}", err))

    # (2) 一步到位的误差放大：x_final - x0 = -sqrt((1-abar_T)/abar_T) * delta，逐点可证
    eps = rng.normal(size=d)
    xT = q_sample(x0, abar_p[T], eps)
    rho_step, rng2 = 0.01, np.random.default_rng(7)
    delta = rho_step * rng2.normal(size=d)
    eps_hat = eps + delta
    x_final, _, _ = ddim_step(xT, eps_hat, T, 0, abar_p, np.zeros(d), 0.0)
    ampl = np.sqrt((1.0 - abar_p[T]) / abar_p[T])
    assert_close("单步误差 = sqrt((1-abar_T)/abar_T) * ||delta||",
                 np.linalg.norm(x_final - x0), ampl * np.linalg.norm(delta), 1e-9, st)
    print(f"      abar_T = {abar_p[T]:.3e}  放大系数 sqrt((1-abar_T)/abar_T) = {fmt(ampl, 1)}")

    # (3) 有模型误差时：误差随步数单调下降
    print("    eta=0、每步注入 1% 相对误差（误差取 rho * W x_t，依赖当前状态）：")
    prev, rows = float("inf"), []
    for steps in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
        errs = [run(steps, rho_step, seed=100 + k) for k in range(20)]
        m = float(np.mean(errs))
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
    return 10.0 * np.log10(abar_p / np.clip(1.0 - abar_p, 1e-300, None))


def exp_E(T=T_DEFAULT):
    """线性 vs 余弦：步数到底花在哪些 SNR 区间。"""
    print("\n[E] 线性调度 vs 余弦调度")
    st = []
    out = {}
    for name, betas in (("linear", linear_betas(T)), ("cosine", cosine_betas(T))):
        _, abar = cumulative(betas)
        abar_p = pad1(abar)
        snr = snr_db(abar_p[1:])
        zero = int(np.argmax(snr < 0.0)) + 1                    # 第一次跌到 0 dB 以下
        bands = {
            "clean  (> 10 dB)": float(np.mean(snr > 10.0)),
            "mid    (-10..10 dB)": float(np.mean((snr > -10.0) & (snr <= 10.0))),
            "noise  (< -10 dB)": float(np.mean(snr <= -10.0)),
        }
        btab = betas[[0, 99, 299, 499, 999]]
        out[name] = dict(snr=snr, zero=zero, bands=bands, btab=btab, abar_p=abar_p, betas=betas)
        print(f"    {name:<7s} beta@[1,100,300,500,1000] = "
              + ", ".join(fmt(v, 5) for v in btab))
        print(f"    {name:<7s} SNR 首次 < 0 dB 于 t = {zero:<4d} ｜ abar_T = {abar_p[T]:.3e}")
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
    # 断言：论文表上的线性 beta 取值（Clipping 里那张表）
    expected = np.array([1e-4, 0.0021, 0.0060, 0.0100, 0.0200])
    assert_close("线性 beta 取值 ≈ 论文/Clipping 表", out["linear"]["btab"], expected, 5e-4, st)
    # 单调性
    for name in ("linear", "cosine"):
        assert np.all(np.diff(out[name]["abar_p"]) < 0), f"{name} abar 应单调下降"
        st.append((f"{name} abar 单调", 0.0))
    return len(st) + 2


# ---------------------------------------------------------------- group F
def set_timesteps(num_train, num_inference, spacing, steps_offset=0):
    """从训练时间步里挑出推理时间步的三种常见规则（降序返回）。

    - endpoint   端点插值：round(i * (T-1) / (N-1))，必含两端 T-1 与 0
    - stride     整数步长：round(i * (T // N))，步长取整，不含 T-1；再加 steps_offset
    - trailing   后缘对齐：round((i+1) * T / N) - 1，必定不含 0 以外的右端点错位
    """
    i = np.arange(num_inference)
    if spacing == "endpoint":
        ts = np.round(i * (num_train - 1) / max(num_inference - 1, 1))
    elif spacing == "stride":
        ts = np.round(i * (num_train // num_inference)) + steps_offset
    elif spacing == "trailing":
        ts = np.round((i + 1) * num_train / num_inference) - 1
    else:
        raise ValueError(spacing)
    ts = ts.astype(np.int64)
    return ts[::-1]


def exp_F(T=T_DEFAULT):
    """推理时间步怎么从 1000 个里挑出来：同样 50 次调用，采样路径可以不同。"""
    print("\n[F] set_timesteps：三种取点规则")
    st = []
    for N in (5, 10, 20, 50):
        print(f"    N={N}")
        for name, kw in (("endpoint", {}), ("stride", {}), ("trailing", {})):
            ts = set_timesteps(T, N, name, **kw)
            assert len(ts) == N, f"{name} N={N} 长度不对"
            assert np.all(np.diff(ts) < 0), f"{name} N={N} 必须严格降序"
            assert ts.min() >= 0 and ts.max() < T, f"{name} N={N} 越界"
            assert len(set(ts.tolist())) == N, f"{name} N={N} 有重复"
            st.append((f"{name} N={N} 不变量", 0.0))
            print(f"      {name:<9s}: {[int(v) for v in ts]}")
    ts_ep = set_timesteps(T, 50, "endpoint")
    ts_st = set_timesteps(T, 50, "stride")
    ts_tr = set_timesteps(T, 50, "trailing")
    print(f"    50 步时 stride 的最大时间步 = {ts_st.max()}，trailing = {ts_tr.max()}，"
          f"endpoint = {ts_ep.max()}")
    assert ts_st.max() < ts_ep.max(), "整数步长版不会覆盖最大训练时间步"
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

    x0 = rng.normal(size=d)
    x0 /= np.linalg.norm(x0)
    xT = q_sample(x0, abar_p[T], rng.normal(size=d))     # 固定同一个出发点 x_T
    W = make_error_operator(d)

    def run(steps, eta, rho, seed, *, tgt=x0, start=xT, op=W):
        """默认用上面固定的 (x_T, 误差算子)；也可以按组传入不同的。"""
        return ddim_trajectory(tgt, start, steps, eta, rho, seed, abar_p,
                               None if rho == 0.0 else op)

    # (1) 随机项相对目标尺度的占比：最后一里地反而不小
    print("    随机项 sigma_t 与目标尺度 sqrt(1 - abar_{t-1})：")
    for t in (1000, 750, 500, 250, 100, 10, 2):
        b_tilde = (1.0 - abar_p[t - 1]) / (1.0 - abar_p[t]) * betas[t - 1]
        sig = np.sqrt(b_tilde)
        scale = np.sqrt(1.0 - abar_p[t - 1])
        print(f"      t={t:<5d} sigma_t = {fmt(sig, 4)} ｜ sqrt(1-abar_{{t-1}}) = "
              f"{fmt(scale, 4)} ｜ 占比 = {100 * sig / scale:5.2f}%")
        assert 0.0 < sig < 0.25, "后验标准差不应超过 0.25"
        st.append((f"sigma_{t} 合理", sig))
    assert_close("第一步的随机项 = sqrt(beta_1)", np.sqrt(betas[0]),
                 np.sqrt(BETA_START), 1e-12, st)

    # (2) oracle 下：eta 根本不起作用——完美模型不需要也没拿随机项换来多样性
    print("    oracle 噪声、同一起点 x_T，换 20 条随机流：")
    for eta in (0.0, 1.0):
        xs = np.stack([run(1000, eta, 0.0, seed=500 + k) for k in range(20)])
        mean_x = xs.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(xs - mean_x, axis=1)) / np.linalg.norm(x0))
        dev = float(np.mean(np.linalg.norm(xs - x0, axis=1)) / np.linalg.norm(x0))
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
            rng3 = np.random.default_rng(9000 + trial)
            xa = rng3.normal(size=d)
            xa /= np.linalg.norm(xa)
            xsa = q_sample(xa, abar_p[T], rng3.normal(size=d))
            op = make_error_operator(d, seed=31337 + trial)
            for eta in (0.0, 1.0):
                xs = np.stack([run(steps, eta, 0.01, seed=700 + k, tgt=xa, start=xsa, op=op)
                               for k in range(n_flow)])
                err_sum[eta] += float(np.mean(np.linalg.norm(xs - xa, axis=1)) / np.linalg.norm(xa))
                spr_sum[eta] += float(np.mean(np.linalg.norm(xs - xs.mean(axis=0), axis=1))
                                      / np.linalg.norm(xa))
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
    t = np.arange(1, T + 1)

    plt.rcParams.update({
        "font.family": ["Heiti TC", "Songti SC", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11,
    })

    # 1) beta 与 abar 曲线
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.0))
    ax[0].plot(t, betas_lin, color="#c0392b", lw=2, label="linear")
    ax[0].plot(t, betas_cos, color="#2471a3", lw=2, label="cosine")
    ax[0].set_xlabel("timestep $t$"); ax[0].set_ylabel(r"$\beta_t$")
    ax[0].set_title("per-step noise strength"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(t, abar_lin_p[1:], color="#c0392b", lw=2, label="linear")
    ax[1].plot(t, abar_cos_p[1:], color="#2471a3", lw=2, label="cosine")
    ax[1].axhline(0, color="gray", lw=.8, ls=":")
    ax[1].set_xlabel("timestep $t$"); ax[1].set_ylabel(r"$\bar\alpha_t$")
    ax[1].set_title("cumulative signal retention"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_schedule_beta_alpha.png", dpi=150); plt.close(fig)

    # 2) SNR 曲线
    snr_l = snr_db(abar_lin_p[1:]); snr_c = snr_db(abar_cos_p[1:])
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
    rng = np.random.default_rng(20261101)
    n = 64
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = np.zeros((n, n, 3))
    img[..., 0] = np.clip(1.3 - np.hypot(xx - .35, yy - .4) * 1.8, 0, 1)      # 圆盘
    bar = ((xx > .62) & (xx < .72) & (yy > .18) & (yy < .82))                 # 竖条
    img[..., 1] = np.clip(np.maximum(bar, 1.1 - np.hypot(xx - .8, yy - .7) * 2.4), 0, 1)
    img[..., 2] = np.clip(np.abs(yy - .5) * 1.6, 0, 1)
    img = img * 2 - 1                                                          # [-1, 1]
    steps = [0, 2, 10, 50, 100, 200, 400, 700, 1000]
    fig, ax = plt.subplots(1, len(steps), figsize=(15.5, 2.4))
    for i, s in enumerate(steps):
        if s == 0:
            cur = img.copy()
        else:
            eps = rng.normal(size=img.shape)
            cur = q_sample(img, abar_lin_p[s], eps)
        cur = np.clip(cur, -1, 1)
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
        e = []
        for k in range(20):
            r = _one_run(s, 0.0, 0.01, 100 + k, T, abar_lin_p)
            e.append(r)
        means.append(float(np.mean(e))); sds.append(float(np.std(e)))
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
        ax.scatter(ts, np.full_like(ts, y, dtype=float), s=14,
                   label=name, color="#c0392b" if name == "endpoint" else "#2471a3")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["trailing", "endpoint"])
    ax.set_xlabel("training timestep selected by set_timesteps()")
    ax.set_xlim(-20, T + 20); ax.legend(loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/ddpm_timesteps_selection.png", dpi=150); plt.close(fig)

    print(f"    配图已写入 {figdir}")
    return 0


def _one_run(steps, eta, rho, seed, T, abar_p):
    """给配图用的一次 DDIM 采样：与 D/H 组共用同一套误差模型。"""
    d = 16
    W = make_error_operator(d)
    rng2 = np.random.default_rng(seed)
    x0 = rng2.normal(size=d); x0 /= np.linalg.norm(x0)
    xT = q_sample(x0, abar_p[T], rng2.normal(size=d))
    x = ddim_trajectory(x0, xT, steps, eta, rho, seed, abar_p, W)
    return np.linalg.norm(x - x0) / np.linalg.norm(x0)


def main(argv):
    figdir = None
    args = []
    skip = False
    for a in argv[1:]:
        if skip:
            skip = False
            continue
        if a == "--figdir":
            figdir = argv[argv.index(a) + 1] if argv.index(a) + 1 < len(argv) else None
            skip = True
            continue
        if a.startswith("-"):
            continue
        args.append(a)
    groups = args or ["A", "B", "C", "D", "E", "F", "H"]
    rng = np.random.default_rng(20261101)
    total = 0
    print("=" * 72)
    print("DDPM 时间步 / 调度 / 采样器 · 核验")
    print(f"设置：T={T_DEFAULT}，linear beta {BETA_START}→{BETA_END}，cosine offset s={COS_S}")
    print("=" * 72)
    table = {"A": exp_A, "B": exp_B, "C": exp_C, "D": exp_D, "E": exp_E, "F": exp_F, "H": exp_H}
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
