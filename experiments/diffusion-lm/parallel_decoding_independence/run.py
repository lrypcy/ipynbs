"""并行解码的代价：tau-leaping 的条件独立假设，以及 unmasking 策略的消融。

纯 numpy、CPU、固定随机种子。两组验证：

A. 两个强相关 token：一步解 1 个 vs 一步解 2 个，对联合分布的拟合差多少
B. L=10 的一阶马尔可夫链上，random / confidence-first / semi-AR 三种 unmasking 策略
   在不同的“步数预算”下的质量-步数曲线（每步提交多个位置 = 条件独立近似）

运行：
    python run.py            # 结果打印到 stdout，并写入 results/summary.md
"""

import numpy as np
import os

SEED = 20261012
rng = np.random.default_rng(SEED)
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300))))


# ---------------------------------------------------------------- A ---------
def part_a():
    joint = np.array([[0.45, 0.05], [0.05, 0.45]])   # 两个 token 强相关
    marg0 = joint.sum(axis=1)
    cond = joint / marg0[:, None]

    # 串行：先解第 1 个（边缘），再按条件解第 2 个 —— 精确还原联合
    seq_est = np.zeros((2, 2))
    for a in range(2):
        for b in range(2):
            seq_est[a, b] = marg0[a] * cond[a, b]

    # 并行（一步 t-leaping）：两个都从边缘独立采样
    par_est = np.outer(marg0, joint.sum(axis=0))

    p_consistent = joint[0, 0] + joint[1, 1]           # 真实分布下“两个 token 一致”的概率
    par_consistent = par_est[0, 0] + par_est[1, 1]

    log("## A tau-leaping 的条件独立代价（两个强相关 token）")
    log(f"  真实联合 P(一致) = {p_consistent:.4f}")
    log(f"  串行（1 个/步）  KL = {kl(joint, seq_est):.6f}   P(一致) = {seq_est[0,0]+seq_est[1,1]:.4f}")
    log(f"  并行（2 个/步）  KL = {kl(joint, par_est):.6f}   P(一致) = {par_consistent:.4f}")
    log(f"  P(一致) 的相对误差：串行 {(seq_est[0,0]+seq_est[1,1]-p_consistent)/p_consistent:+.2%}    并行 {(par_consistent-p_consistent)/p_consistent:+.2%}")
    log("")

    # 三 token 版本：一步解 1 / 2 / 3 个
    Vt = 2
    w = np.array([0.5, 0.5])
    Tm = np.array([[0.9, 0.1], [0.1, 0.9]])
    Lc = 3
    true_joint = np.zeros([Vt] * Lc)
    for idx in np.ndindex(*([Vt] * Lc)):
        pr = w[idx[0]]
        for i in range(1, Lc):
            pr *= Tm[idx[i - 1], idx[i]]
        true_joint[idx] = pr
    true_flat = true_joint.reshape(-1)

    def product_of_marginals(n_commit):
        """前 n_commit 个位置用边缘独立，其余用真实条件（模拟“每步提交 n 个”）。"""
        marg = np.zeros([Vt] * Lc)
        idxs = np.array(np.meshgrid(*[np.arange(Vt)] * Lc, indexing="ij")).reshape(Lc, -1).T
        for idx in idxs:
            pr = 1.0
            for i in range(n_commit):          # 被提交的位置：条件独立
                pr *= w[idx[i]] if i == 0 else true_joint.sum(axis=tuple(j for j in range(Lc) if j != i)).reshape(-1)[0] * 0  # 占位，下面重写
            marg[tuple(idx)] = pr
        return marg

    # 直接算：提交集合 S 内用边缘之积，其余按真实链式条件
    def approx_joint(n_commit):
        out = np.zeros([Vt] * Lc)
        marg_i = []
        for i in range(Lc):
            m = np.zeros(Vt)
            for v in range(Vt):
                tot = 0.0
                for idx in np.ndindex(*([Vt] * Lc)):
                    if idx[i] != v:
                        continue
                    pr = w[idx[0]]
                    for k in range(1, Lc):
                        pr *= Tm[idx[k - 1], idx[k]]
                    tot += pr
                m[v] = tot
            marg_i.append(m)
        for idx in np.ndindex(*([Vt] * Lc)):
            pr = 1.0
            for i in range(Lc):
                if i < n_commit:                # 一步内提交：用边缘
                    pr *= marg_i[i][idx[i]]
                else:                           # 后续步：用真实条件
                    pr *= w[idx[i]] if i == 0 else Tm[idx[i - 1], idx[i]]
            out[idx] = pr
        return out / out.sum()

    log("  三 token 链式分布，逐步增大“每步提交数”：")
    for k in (1, 2, 3):
        est = approx_joint(k)
        log(f"    每步提交 {k} 个：KL(真实 || 近似) = {kl(true_flat, est.reshape(-1)):.6f}")
    log("")
    return kl(joint, seq_est), kl(joint, par_est)


# ---------------------------------------------------------------- B ---------
def make_chain(V, L):
    w = rng.dirichlet(np.ones(V) * 2.0)
    Tm = np.stack([rng.dirichlet(np.ones(V) * 1.2) for _ in range(V)])
    return w, Tm


def forward_backward(w, Tm, evid):
    """evid[i] 为 None 表示未观测，否则为 one-hot；返回每个位置的边缘分布。"""
    V = len(w)
    L = len(evid)
    e = [np.ones(V) if evid[i] is None else np.eye(V)[int(evid[i])] for i in range(L)]

    a = [np.zeros(V) for _ in range(L)]
    a[0] = w * e[0]
    a[0] /= a[0].sum()
    for i in range(1, L):
        a[i] = (a[i - 1] @ Tm) * e[i]
        a[i] /= max(a[i].sum(), 1e-300)

    b = [np.ones(V) for _ in range(L)]
    for i in range(L - 2, -1, -1):
        b[i] = Tm @ (b[i + 1] * e[i + 1])
        b[i] /= max(b[i].sum(), 1e-300)

    m = [a[i] * b[i] for i in range(L)]
    m = [x / max(x.sum(), 1e-300) for x in m]
    return m


def viterbi(w, Tm, L):
    V = len(w)
    delta = [np.log(w)]
    psi = [np.zeros(L, dtype=int)]
    for i in range(1, L):
        m = delta[-1][:, None] + np.log(Tm)
        psi.append(np.argmax(m, axis=0))
        delta.append(np.max(m, axis=0))
    path = [int(np.argmax(delta[-1]))]
    for i in range(L - 1, 0, -1):
        path.append(int(psi[i][path[-1]]))
    return path[::-1]


def part_b(V=4, L=10, trials=400):
    w, Tm = make_chain(V, L)
    ref = viterbi(w, Tm, L)
    logw, logT = np.log(w), np.log(Tm)

    def logp(x):
        return logw[x[0]] + sum(logT[x[i - 1], x[i]] for i in range(1, L))

    log(f"## B unmasking 策略消融（V={V}, L={L}, 参考序列 = Viterbi，trials={trials}）")
    log(f"  参考序列的 -log p = {-logp(np.array(ref)):.4f}")
    log("")
    log("  两套指标：greedy 模式（提交时取 argmax）看与 Viterbi 的一致率；")
    log("            sampling 模式（提交时按边缘采样）看生成序列在真实分布下的 -log p。")
    log("")
    log("  策略            步数  每步提交   一致率(greedy)   -log p(sampling)")

    def decode(strategy, steps, greedy):
        k = int(np.ceil(L / steps))
        obs = [None] * L
        remaining = list(range(L))
        for _ in range(steps):
            if not remaining:
                break
            m = forward_backward(w, Tm, obs)
            conf = np.array([m[i].max() for i in remaining])
            if strategy == "random":
                pick = list(rng.choice(remaining, size=min(k, len(remaining)), replace=False))
            elif strategy == "confidence":
                order = np.argsort(-conf)
                pick = [remaining[j] for j in order[:min(k, len(remaining))]]
            else:
                pick = sorted(remaining)[:min(k, len(remaining))]
            for i in pick:                      # 同一批内互相独立 = 条件独立近似
                obs[i] = int(np.argmax(m[i])) if greedy else int(rng.choice(V, p=m[i] / m[i].sum()))
                remaining.remove(i)
        if any(o is None for o in obs):
            m = forward_backward(w, Tm, obs)
            obs = [o if o is not None else int(np.argmax(m[i])) for i, o in enumerate(obs)]
        return np.array(obs)

    results = {}
    for strategy in ("random", "confidence", "semi_ar"):
        for steps in (1, 2, 5, 10):
            agree = [np.mean(decode(strategy, steps, True) == np.array(ref)) for _ in range(trials)]
            nlp = [-logp(decode(strategy, steps, False)) for _ in range(trials)]
            results[(strategy, steps)] = (float(np.mean(agree)), float(np.mean(nlp)))
            log(f"  {strategy:14s}  {steps:3d}   {int(np.ceil(L/steps)):6d}   {np.mean(agree):13.3%}   {np.mean(nlp):14.4f}")
    log("")
    log("  读法：步数越少（每步提交越多），条件独立近似越粗糙，一致率与 -log p 同时变差；")
    log("        在同样的步数预算下，先提交高置信位置（confidence）通常优于随机提交。")
    log("")
    return results


def main():
    log(f"# 并行解码代价与 unmasking 策略消融（seed={SEED}）")
    log("")
    part_a()
    part_b()

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results", "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
