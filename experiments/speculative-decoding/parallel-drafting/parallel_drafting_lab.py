"""并行草稿与自适应验证：五个可核验的合成探针。

纯 CPU / numpy / 固定 seed 20261117。所有数字由「构造真值 ↔ 反算值」自校验，
每组末尾带断言，最后一个 cell 打印断言总数。

组 1  用 Red Hat 公布的 P-EAGLE 逐位置接受率反算平均接受长度，检验常数 α 假设
组 2  草稿成本从 γ·c 换成常数 c_b 之后，最优草稿长度与收益曲面怎么变
组 3  树验证的祖先掩码矩阵，逐格对照八节点候选树
组 4  置信度调度只截断候选前缀时，输出分布是否仍等于目标分布（蒙特卡洛）
组 5  margin 接受规则放宽到什么程度开始有损（TV 距离扫描）

运行：~/Software/miniconda3/bin/python parallel_drafting_lab.py
切 notebook：~/Software/miniconda3/envs/torch/bin/python make_notebook.py
"""

# %%
import numpy as np

RNG = np.random.default_rng(20261117)
OK = []
FAIL = []


def check(cond, msg):
    (OK if bool(cond) else FAIL).append(msg)
    if not bool(cond):
        print(f"  [FAIL] {msg}")


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def tv(a, b):
    """总变差距离，输入为已归一化的离散分布。"""
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def speedup(alpha, gamma, draft_cost, verify_slope=0.0):
    """一阶收益模型：分子是每轮产出期望，分母是「一次验证 + 草稿 + 验证增量」。

    draft_cost 可以是 γ*c（串行草稿）或常数 c_b（并行草稿）。
    verify_slope 描述候选数增加带来的验证侧增量，两种形态共用。
    """
    gamma = np.asarray(gamma, dtype=float)
    produced = (1.0 - alpha ** (gamma + 1.0)) / (1.0 - alpha)
    cost = 1.0 + draft_cost + verify_slope * gamma
    return produced / cost


# %%
def g1_acceptance_curve():
    section("组 1｜逐位置接受率反算：常数 α 假设高估了 18%")

    # Red Hat Developer, "Speeding up LLM inference with P-EAGLE in vLLM Speculators"
    # 表格 Average 行：Pos 1..Pos 7 的接受率，以及官方给出的 Avg Length
    a = np.array([0.749, 0.499, 0.319, 0.204, 0.131, 0.085, 0.055])
    official_avg_len = 3.044
    K = len(a)

    # 构造真值：每轮产出 = 1（拒绝处的修正 token 或 bonus token）+ 各位置接受概率之和
    e_t = 1.0 + a.sum()
    print(f"官方 Avg Length            : {official_avg_len:.3f}")
    print(f"1 + Σ a_i 反算              : {e_t:.3f}")
    print(f"差值（官方四舍五入）        : {abs(e_t - official_avg_len):.3f}")

    ratios = a[1:] / a[:-1]
    r = ratios.mean()
    print("\n相邻位置接受率之比 a_(i+1)/a_i：")
    for i, v in enumerate(ratios, start=1):
        print(f"  {i}→{i+1}: {v:.4f}")
    print(f"均值 {r:.4f}  标准差 {ratios.std():.4f}")

    # 常数 α 模型（把位置 1 的接受率当成全局 α）与双参数几何衰减模型对照
    alpha_flat = a[0]
    e_flat = (1.0 - alpha_flat ** (K + 1)) / (1.0 - alpha_flat)
    e_geom = 1.0 + a[0] * (1.0 - r ** K) / (1.0 - r)
    over = (e_flat - e_t) / e_t * 100.0
    err_geom = abs(e_geom - e_t) / e_t * 100.0
    print(f"\n常数 α={alpha_flat:.3f} 模型 E[T] : {e_flat:.4f}  高估 {over:.2f}%")
    print(f"双参数 a1*r^(i-1) 模型 E[T]: {e_geom:.4f}  误差 {err_geom:.2f}%")

    check(abs(e_t - official_avg_len) < 0.01, "组1a 反算 E[T] 与官方 Avg Length 一致")
    check(np.all((ratios > 0.60) & (ratios < 0.70)), "组1b 相邻比值落在 0.60~0.70")
    check(ratios.std() < 0.02, "组1c 相邻比值近似常数（几何衰减）")
    check(over > 15.0, "组1d 常数 α 模型显著高估每轮产出")
    check(err_geom < 2.0, "组1e 双参数几何模型误差小于 2%")
    assert abs(e_t - official_avg_len) < 0.01 and err_geom < 2.0


# %%
def g2_cost_shape():
    section("组 2｜草稿成本换形状：γ* 变大，但验证成本仍随 γ 增长")

    alpha = 0.647          # 组 1 反算出的几何衰减率
    c = 0.08               # 串行草稿：单次前向相对目标前向的成本
    c_b = 0.08             # 并行草稿：一整个 block 一次前向，成本与 γ 解耦
    beta = 0.02            # 验证侧随候选数增加的增量（两种形态共用）
    grid = np.arange(1, 33)

    s_ar = speedup(alpha, grid, c * grid, beta)
    s_par = speedup(alpha, grid, c_b, beta)
    g_ar = int(grid[np.argmax(s_ar)])
    g_par = int(grid[np.argmax(s_par)])

    print(f"{'γ':>4} | {'串行 S':>9} | {'并行 S':>9}")
    for g in (1, 2, 4, 8, 12, 16, 24, 32):
        print(f"{g:>4} | {speedup(alpha, g, c*g, beta):>9.4f} | {speedup(alpha, g, c_b, beta):>9.4f}")
    print(f"\nγ*：串行 {g_ar}  →  并行 {g_par}")
    print(f"S*：串行 {s_ar.max():.4f}  →  并行 {s_par.max():.4f}")

    # 曲面平坦度：γ 从 8 拉到 16 时收益的跌幅
    drop_ar = 1.0 - speedup(alpha, 16, c*16, beta) / speedup(alpha, 8, c*8, beta)
    drop_par = 1.0 - speedup(alpha, 16, c_b, beta) / speedup(alpha, 8, c_b, beta)
    print(f"γ 8→16 的收益跌幅：串行 {drop_ar*100:.2f}%  →  并行 {drop_par*100:.2f}%")

    check(g_par > g_ar, "组2a 并行草稿的最优 γ 更大")
    check(s_par.max() > s_ar.max(), "组2b 并行草稿的峰值收益更高")
    check(drop_par < drop_ar / 2, "组2c 并行草稿把 γ 8→16 的跌幅压到串行的一半以下")
    check(drop_par > 0.05, "组2d 验证侧增量仍在，γ 不能无限拉长")
    assert g_par > g_ar and drop_par < drop_ar


# %%
def g3_tree_mask():
    section("组 3｜树验证掩码：每个候选节点只看自己的祖先")

    # 八节点候选树：Root → A → {B,C,D}，B → {E,F}，D → G
    names = ["R", "A", "B", "C", "D", "E", "F", "G"]
    parent = {"R": None, "A": "R", "B": "A", "C": "A", "D": "A",
              "E": "B", "F": "B", "G": "D"}
    idx = {n: i for i, n in enumerate(names)}

    def ancestors(n):
        out = []
        while parent[n] is not None:
            n = parent[n]
            out.append(n)
        return out

    M = np.zeros((len(names), len(names)), dtype=int)
    for i, n in enumerate(names):
        for j in names:
            if j == n or j in ancestors(n):
                M[i, idx[j]] = 1

    # 逐行打印（行=Query，列=Key）
    print("    " + " ".join(f"{n:>2}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>3} " + " ".join(f"{v:>2}" for v in M[i]))

    expected = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 0, 1, 0, 0, 1],
    ])
    print(f"\n非零元个数（每条路径的可见 KV 数之和）：{M.sum()}")
    print(f"与对照矩阵逐格一致：{np.array_equal(M, expected)}")

    check(np.array_equal(M, expected), "组3a 祖先掩码与八节点树逐格一致")
    check(np.all(np.diag(M) == 1), "组3b 每个节点都能看到自己")
    check(np.all(np.triu(M, 1) == 0), "组3c 掩码在拓扑序下是下三角")
    check(M[idx["E"], idx["F"]] == 0 and M[idx["F"], idx["E"]] == 0,
          "组3d 同父兄弟分支互不可见")
    check(M[idx["G"], idx["D"]] == 1 and M[idx["G"], idx["B"]] == 0,
          "组3e G 看 D 与 A，但不看另一条分支的 B")
    assert np.array_equal(M, expected)


# %%
def g4_confidence_truncation():
    section("组 4｜置信度调度只截断前缀时，输出分布仍等于目标分布")

    p = np.array([0.50, 0.30, 0.15, 0.05])
    q = np.array([0.40, 0.35, 0.20, 0.05])
    residual = np.maximum(p - q, 0.0)
    residual /= residual.sum()
    V = len(p)
    N = 200_000

    def target_sample(n):
        return RNG.choice(V, size=n, p=p)

    def sd_tokens(gamma, n):
        """一轮投机解码输出的整段 token 序列，向量化实现。

        接受前缀 + 一个修正 token（拒绝处从残差采样）或 bonus token（全接受时从 p 采样）。
        """
        draft = RNG.choice(V, size=(n, gamma), p=q)
        u = RNG.random((n, gamma))
        acc = u < np.minimum(1.0, p[draft] / q[draft])
        any_reject = ~acc.all(axis=1)
        k = np.where(any_reject, np.argmax(~acc, axis=1), gamma)
        mask = np.arange(gamma)[None, :] < k[:, None]
        accepted = draft[mask]
        corr = RNG.choice(V, size=n, p=residual)
        bonus = RNG.choice(V, size=n, p=p)
        extra = np.where(any_reject, corr, bonus)
        return np.concatenate([accepted, extra]), k + 1

    def empirical(samples):
        return np.bincount(samples, minlength=V) / len(samples)

    base = empirical(target_sample(N))
    short, lens2 = sd_tokens(2, N)
    full, lens4 = sd_tokens(4, N)
    e_short, e_full = empirical(short), empirical(full)

    print("纯目标采样        :", np.round(base, 5))
    print("截断到 γ=2 的投机 :", np.round(e_short, 5))
    print("完整 γ=4 的投机   :", np.round(e_full, 5))
    print(f"TV(截断, 目标)  : {tv(e_short, base):.5f}")
    print(f"TV(完整, 目标)  : {tv(e_full, base):.5f}")

    # 截断的代价不在分布，在每轮产出多少个 token
    m2, m4 = float(lens2.mean()), float(lens4.mean())
    print(f"每轮产出期望：γ=2 → {m2:.4f}，γ=4 → {m4:.4f}")

    check(tv(base, p) < 0.005, "组4a 纯目标采样的经验分布收敛到 p")
    check(tv(e_short, base) < 0.01, "组4b 截断前缀后输出分布仍等于 p")
    check(tv(e_full, base) < 0.01, "组4c 完整投机后输出分布仍等于 p")
    check(m4 > m2, "组4d 截断只是减少每轮产出，不改变分布")
    assert tv(e_short, base) < 0.01 and tv(e_full, base) < 0.01


# %%
def g5_margin_lossy():
    section("组 5｜margin 接受规则：放宽阈值不是单调地变坏")

    p = np.array([0.51, 0.47, 0.02])
    q = np.array([0.40, 0.45, 0.15])
    V = len(p)
    top = p.max()
    margins = p / top
    print("候选 token 的 margin = p(x)/max(p)：", np.round(margins, 4))

    def output_distribution(tau):
        """严格拒绝采样：y~p 才是真值；x~q 与 y 不等时，margin 够高就放行 x。

        τ=1 时没有任何 x 满足 margin > τ，等于退回严格采样，输出分布就是 p。
        """
        out = np.zeros(V)
        for x in range(V):
            for y in range(V):
                prob = q[x] * p[y]
                if x == y or margins[x] > tau:
                    out[x] += prob
                else:
                    out[y] += prob
        return out

    taus = [1.00, 0.98, 0.95, 0.92, 0.90, 0.85, 0.50]
    tvs = []
    print(f"\n{'τ':>6} | {'被 margin 直接放行 mass':>22} | {'TV(输出, p)':>12}")
    for tau in taus:
        accepted_mass = q[margins > tau].sum()
        out = output_distribution(tau)
        d = tv(out, p)
        tvs.append(d)
        print(f"{tau:>6.2f} | {accepted_mass:>22.4f} | {d:>12.5f}")

    check(tvs[0] < 1e-12, "组5a τ=1 时退回严格采样，TV 为零")
    check(tvs[0] < min(tvs[1:]), "组5b 任何非空放行集合都带来分布偏差")
    check(tvs[3] < tvs[1], "组5c 放行集合扩大后 TV 反而下降（反直觉）")
    check(tvs[-1] > 0.05, "组5d 放宽到 τ=0.5 时有损性显著")
    check(0.47 / 0.51 > 0.92, "组5e 例子里的次优 token B 落在 0.92 阈值附近")
    assert tvs[0] < 1e-12 and tvs[-1] > 0.05


# %%
def main():
    for g in (g1_acceptance_curve, g2_cost_shape, g3_tree_mask,
              g4_confidence_truncation, g5_margin_lossy):
        g()
    print("\n" + "=" * 72)
    print(f"总计 {len(OK)}/{len(OK) + len(FAIL)} 条断言通过")
    if FAIL:
        print("失败：", FAIL)
        raise SystemExit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
