"""掩码扩散（MDLM）ELBO 的数值验证。

纯 numpy、CPU、固定随机种子，无需 GPU 与任何外部数据。四组验证：

A. 前向核 / 反向核的转移概率与理论值一致（吸收态，alpha_t = 1 - t）
B. 固定 15% 掩码 vs t ~ U[0,1]：训练出来的模型能不能从全 [MASK] 生成
C. 训练损失确实是 -E[log p(x0)] 的上界；去掉 1/t 权重后不再是同一个量
D. 1/t 权重如何重新分配各掩码率区间的贡献，以及它对估计方差的影响

运行：
    python run.py            # 结果打印到 stdout，并写入 results/summary.md
"""

import numpy as np
import os

SEED = 20261012
V, L = 4, 3            # 词表大小、序列长度（保持小，便于精确枚举）
MASK = V               # 0..V-1 是真实 token，V 表示 [MASK]
NSTATE = (V + 1) ** L  # 含掩码的上下文状态数 5^3 = 125
NSEQ = V ** L          # 干净序列数 4^3 = 64

rng = np.random.default_rng(SEED)
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def ctx_index(ctx):
    """上下文 -> 索引。与 meshgrid(indexing="ij") 的行序一致：高位在前。"""
    idx = 0
    for p in range(L):
        idx += int(ctx[p]) * (V + 1) ** (L - 1 - p)
    return idx


def build_tables(p_data):
    """枚举所有序列 / 上下文，预计算一致性布尔表与真实后验收敛器。"""
    seqs = np.array(np.meshgrid(*[np.arange(V)] * L, indexing="ij")).reshape(L, -1).T
    ctxs = np.array(np.meshgrid(*[np.arange(V + 1)] * L, indexing="ij")).reshape(L, -1).T

    consistent = np.zeros((NSEQ, NSTATE), dtype=bool)
    for c in range(NSTATE):
        ok = np.ones(NSEQ, dtype=bool)
        for p in range(L):
            if ctxs[c, p] != MASK:
                ok &= (seqs[:, p] == ctxs[c, p])
        consistent[:, c] = ok

    post = np.zeros((NSTATE, L, V))
    mass = consistent.T @ p_data
    for i in range(L):
        for v in range(V):
            sel = seqs[:, i] == v
            num = consistent[sel, :].T @ p_data[sel]
            post[:, i, v] = np.where(mass > 0, num / np.maximum(mass, 1e-300), 1.0 / V)
    return seqs, ctxs, consistent, post


# ---------------------------------------------------------------- A ---------
def part_a(n=400_000):
    """吸收态下每个位置有一个“掩码时刻”tau ~ U[0,1]：t 时刻被掩码 <=> tau <= t。"""
    s_val, t_val = 0.30, 0.70
    tau = rng.random((n, 4))
    masked_s, masked_t = tau <= s_val, tau <= t_val

    unmasked_at_s = ~masked_s
    emp_fwd = np.sum(masked_t & unmasked_at_s) / np.sum(unmasked_at_s)
    theo_fwd = (t_val - s_val) / (1 - s_val)          # (alpha_s - alpha_t) / alpha_s

    emp_rev = np.sum((~masked_s) & masked_t) / np.sum(masked_t)
    theo_rev = (t_val - s_val) / t_val                # (alpha_s - alpha_t) / (1 - alpha_t)

    log("## A 前向核 / 反向核的转移概率（吸收态，alpha_t = 1 - t）")
    log(f"  前向 P(mask@t | unmask@s), s={s_val}, t={t_val}")
    log(f"    经验 {emp_fwd:.5f}   理论 (alpha_s-alpha_t)/alpha_s = {theo_fwd:.5f}   偏差 {abs(emp_fwd-theo_fwd):.5f}")
    log(f"  反向 P(unmask@s | mask@t)")
    log(f"    经验 {emp_rev:.5f}   理论 (alpha_s-alpha_t)/(1-alpha_t) = {theo_rev:.5f}   偏差 {abs(emp_rev-theo_rev):.5f}")
    log("")


# ---------------------------------------------------------------- B ---------
def make_features(ctx_row, i):
    """预测第 i 位时的特征：其余各位的 one-hot（含 [MASK] 那一档）。"""
    feats = []
    for j in range(L):
        if j == i:
            continue
        onehot = np.zeros(V + 1)
        onehot[int(ctx_row[j])] = 1.0
        feats.append(onehot)
    return np.concatenate(feats)


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_predictor(p_data, seqs, scheme, steps=20000, batch=64, lr=0.5):
    """tabular softmax 预测器；scheme 决定训练时见到的掩码率分布。"""
    d = (L - 1) * (V + 1)
    W = np.zeros((V, d))
    b = np.zeros(V)
    seq_prob = p_data / p_data.sum()
    for _ in range(steps):
        idxs = rng.choice(NSEQ, size=batch, p=seq_prob)
        F, Y = [], []
        for s in idxs:
            x = seqs[s]
            t = 0.15 if scheme == "fixed15" else float(rng.random())
            ctx = x.copy()
            for p in range(L):
                if rng.random() < t:
                    ctx[p] = MASK
            for i in range(L):
                if ctx[i] == MASK:
                    F.append(make_features(ctx, i))
                    Y.append(x[i])
        if not F:
            continue
        F = np.array(F)
        Y = np.array(Y)
        prob = softmax_rows(F @ W.T + b)
        prob[np.arange(len(Y)), Y] -= 1.0
        prob /= len(Y)
        W -= lr * (prob.T @ F)
        b -= lr * prob.sum(axis=0)
    return W, b


def predictor_to_post(W, b, ctxs):
    """把线性 softmax 预测器在全部 125 个上下文上展开成后验收敛器表。"""
    post = np.zeros((NSTATE, L, V))
    for c in range(NSTATE):
        for i in range(L):
            f = make_features(ctxs[c], i)
            post[c, i] = softmax_rows((W @ f + b)[None])[0]
    return post


def reverse_chain(post, K=64):
    """按反向核传播状态分布，返回反向过程诱导的序列分布 p_theta(x0)。"""
    grid = np.linspace(1.0, 0.0, K + 1)
    dist = np.zeros(NSTATE)
    dist[ctx_index(np.full(L, MASK))] = 1.0
    for k in range(K):
        t, s = grid[k], grid[k + 1]
        p_restore = (t - s) / t                       # (alpha_s - alpha_t) / (1 - alpha_t)
        T = np.zeros((NSTATE, NSTATE))
        for c in range(NSTATE):
            if dist[c] == 0:
                continue
            ctx = ctxs_cache[c]
            stay = 1.0
            for p in range(L):
                if ctx[p] == MASK:
                    for v in range(V):
                        nxt = ctx.copy()
                        nxt[p] = v
                        T[ctx_index(nxt), c] += p_restore * post[c, p, v]
                    stay -= p_restore * post[c, p, :].sum()
            T[c, c] += max(stay, 0.0)
        dist = T @ dist
    p_model = np.zeros(NSEQ)
    for s in range(NSEQ):
        p_model[s] = dist[ctx_index(seqs_cache[s])]
    return p_model / p_model.sum()


def kl(p, q):
    return float(np.sum(p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300))))


def part_b():
    """在 L=8 的一阶马尔可夫链上重做：长序列下 15% 掩码几乎见不到高掩码率上下文。

    指标：
      CE@t       —— 掩码率 t 处的平均预测交叉熵（越低越会填）
      gen_nll    —— 用反向过程采样出的序列在真实分布下的 -log p（越接近熵越好）
    """
    LB, M, NTRAIN = 6, 4000, 200_000
    NCTX = (V + 1) ** LB
    pi = rng.dirichlet(np.ones(V) * 2.0)
    Tm = np.stack([rng.dirichlet(np.ones(V) * 1.5) for _ in range(V)])
    log_pi, log_T = np.log(pi), np.log(Tm)

    def logp_seq(x):
        return log_pi[x[0]] + sum(log_T[x[i - 1], x[i]] for i in range(1, LB))

    entropy = -np.sum(pi * log_pi) - (LB - 1) * np.sum(pi * np.sum(Tm * log_T, axis=1))

    def sample_data(n):
        out = np.empty((n, LB), dtype=int)
        out[:, 0] = rng.choice(V, size=n, p=pi)
        for i in range(1, LB):
            for k in range(V):
                sel = out[:, i - 1] == k
                if sel.any():
                    out[sel, i] = rng.choice(V, size=int(sel.sum()), p=Tm[k])
        return out

    def ctx_idx(ctx):
        idx = 0
        for p in range(LB):
            idx += int(ctx[p]) * (V + 1) ** (LB - 1 - p)
        return idx

    def train_counts(scheme):
        """查表式 MLE：不做权值共享，没见过的上下文只能回退到均匀分布。"""
        counts = np.zeros((NCTX, LB, V))
        X = sample_data(NTRAIN)
        all_mask_seen = 0
        for x in X:
            t = 0.15 if scheme == "fixed15" else float(rng.random())
            ctx = x.copy()
            for p in range(LB):
                if rng.random() < t:
                    ctx[p] = MASK
            if not np.any(ctx == MASK):
                continue
            if np.all(ctx == MASK):
                all_mask_seen += 1
            c = ctx_idx(ctx)
            for i in range(LB):
                if ctx[i] == MASK:
                    counts[c, i, int(x[i])] += 1.0
        return counts, all_mask_seen

    def posterior(counts, alpha=0.5):
        s = counts.sum(axis=2, keepdims=True)
        return (counts + alpha) / (s + alpha * V)

    def eval_ce(post, t, draws=4000):
        X = sample_data(draws)
        tot, cnt = 0.0, 0
        for x in X:
            ctx = x.copy()
            for p in range(LB):
                if rng.random() < t:
                    ctx[p] = MASK
            c = ctx_idx(ctx)
            for i in range(LB):
                if ctx[i] == MASK:
                    tot += -np.log(post[c, i, int(x[i])])
                    cnt += 1
        return tot / max(cnt, 1)

    def generate(post, n, steps=LB):
        grid = np.linspace(1.0, 0.0, steps + 1)
        outs = []
        for _ in range(n):
            ctx = np.full(LB, MASK)
            for k in range(steps):
                t, s = grid[k], grid[k + 1]
                p_restore = (t - s) / t
                c = ctx_idx(ctx)
                for i in range(LB):
                    if ctx[i] == MASK and rng.random() < p_restore:
                        ctx[i] = rng.choice(V, p=post[c, i] / post[c, i].sum())
            outs.append(ctx.copy())
        return np.array(outs)

    log(f"## B 固定 15% 掩码 vs t ~ U[0,1]（L={LB} 一阶马尔可夫链，查表式 MLE，无权值共享）")
    log(f"  参考：真实分布的熵 H(p_data) = {entropy:.4f}；均匀分布基线 -log p = {LB * np.log(V):.4f}")
    out = {}
    for scheme in ("fixed15", "uniform"):
        counts, seen = train_counts(scheme)
        post = posterior(counts)
        ce15, ce50, ce10 = eval_ce(post, 0.15), eval_ce(post, 0.50), eval_ce(post, 1.00)
        gen = generate(post, M)
        gen_nll = float(np.mean([-logp_seq(g) for g in gen]))
        out[scheme] = (ce15, ce50, ce10, gen_nll, seen)
        log(f"  训练掩码率={scheme:8s}  全掩码上下文被观测 {seen:7d} 次 / {NTRAIN}")
        log(f"      CE@0.15={ce15:.4f}  CE@0.50={ce50:.4f}  CE@1.00={ce10:.4f}   生成 -log p={gen_nll:.4f}（超出熵 {gen_nll-entropy:+.4f}）")
    log(f"  全掩码处 CE 之比 fixed/uniform = {out['fixed15'][2]/out['uniform'][2]:.3f}")
    log(f"  生成样本超出熵的量之比 fixed/uniform = {(out['fixed15'][3]-entropy)/max(out['uniform'][3]-entropy, 1e-12):.3f}")
    log("")
    return out


def eval_ce_tabular(post, p_data, seqs, t, draws=4000):
    """给定后验收敛器表，用蒙特卡洛估计掩码率 t 处的平均交叉熵。"""
    tot, cnt = 0.0, 0
    for _ in range(draws):
        s = rng.choice(NSEQ, p=p_data / p_data.sum())
        x = seqs[s]
        ctx = x.copy()
        for p in range(L):
            if rng.random() < t:
                ctx[p] = MASK
        c = ctx_index(ctx)
        for p in range(L):
            if ctx[p] == MASK:
                tot += -np.log(max(post[c, p, int(x[p])], 1e-300))
                cnt += 1
    return tot / max(cnt, 1)


# ---------------------------------------------------------------- C ---------
def part_c(p_data, seqs, post):
    """精确计算反向过程诱导的 p_theta、NLL，以及带 / 不带 1/t 的两个量。"""
    p_model = reverse_chain(post)
    nll = -float(np.sum(p_data * np.log(np.maximum(p_model, 1e-300))))

    M = 2000
    ts = (np.arange(M) + 0.5) / M
    bound_w = np.zeros(NSEQ)
    bound_u = np.zeros(NSEQ)
    contrib_w = np.zeros(M)
    contrib_u = np.zeros(M)
    for s in range(NSEQ):
        x = seqs[s]
        for mi, t in enumerate(ts):
            for pattern in range(1, 1 << L):
                prob, ce = 1.0, 0.0
                ctx = x.copy()
                for p in range(L):
                    if pattern & (1 << p):
                        prob *= t
                        ctx[p] = MASK
                    else:
                        prob *= (1 - t)
                c = ctx_index(ctx)
                for p in range(L):
                    if pattern & (1 << p):
                        ce += -np.log(max(post[c, p, int(x[p])], 1e-300))
                bound_w[s] += prob * ce / t / M
                bound_u[s] += prob * ce / M
    for mi, t in enumerate(ts):
        for s in range(NSEQ):
            x = seqs[s]
            for pattern in range(1, 1 << L):
                prob, ce = 1.0, 0.0
                ctx = x.copy()
                for p in range(L):
                    if pattern & (1 << p):
                        prob *= t
                        ctx[p] = MASK
                    else:
                        prob *= (1 - t)
                c = ctx_index(ctx)
                for p in range(L):
                    if pattern & (1 << p):
                        ce += -np.log(max(post[c, p, int(x[p])], 1e-300))
                contrib_w[mi] += p_data[s] * prob * ce / t
                contrib_u[mi] += p_data[s] * prob * ce

    elbo_w = float(np.sum(p_data * bound_w))
    elbo_u = float(np.sum(p_data * bound_u))
    log("## C 训练损失与 NLL 的关系（真实后验收敛器，精确枚举 + 数值积分）")
    log(f"  -E[log p_theta(x0)]              = {nll:.4f}")
    log(f"  带 1/t 权重的量 E[(1/t) sum ce]   = {elbo_w:.4f}   （差 {elbo_w - nll:+.4f}，{'是上界' if elbo_w >= nll else '不是上界'}）")
    log(f"  去掉 1/t 的量 E[sum ce]           = {elbo_u:.4f}   （差 {elbo_u - nll:+.4f}，{'是上界' if elbo_u >= nll else '不是上界'}）")
    log(f"  两者之比 带/不带 = {elbo_w / max(elbo_u, 1e-12):.3f}  —— 1/t 把量纲抬回“每个 token 的均值”，否则高掩码率样本被系统性压低")
    log("")
    return nll, elbo_w, elbo_u, ts, contrib_w, contrib_u


# ---------------------------------------------------------------- D ---------
def part_d(p_data, seqs, post, ts, contrib_w, contrib_u):
    log("## D 1/t 权重对各掩码率区间贡献的重新分配（占比）")
    tot_w, tot_u = contrib_w.sum(), contrib_u.sum()
    for lo, hi in [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]:
        sel = (ts >= lo) & (ts < hi)
        log(f"  t in [{lo:.2f},{hi:.2f})   带权重 {contrib_w[sel].sum()/tot_w:6.2%}    不带权重 {contrib_u[sel].sum()/tot_u:6.2%}")

    def mc_estimator(weighted, n, reps=2000):
        vals = np.empty(reps)
        for r in range(reps):
            acc = 0.0
            for _ in range(n):
                s = rng.choice(NSEQ, p=p_data / p_data.sum())
                x = seqs[s]
                t = float(rng.random())
                ctx = x.copy()
                for p in range(L):
                    if rng.random() < t:
                        ctx[p] = MASK
                c = ctx_index(ctx)
                ce = 0.0
                for p in range(L):
                    if ctx[p] == MASK:
                        ce += -np.log(max(post[c, p, int(x[p])], 1e-300))
                acc += ce / t if weighted else ce
            vals[r] = acc / n
        return float(np.var(vals))

    var_w = mc_estimator(True, 16)
    var_u = mc_estimator(False, 16)
    log(f"  蒙特卡洛估计方差（n=16，2000 次重复）：带 1/t {var_w:.4f}    不带 {var_u:.4f}    比值 {var_w/max(var_u,1e-12):.3f}")
    log("")


def main():
    global seqs_cache, ctxs_cache
    log(f"# 掩码扩散 ELBO 数值验证（seed={SEED}, V={V}, L={L}）")
    log("")
    p_raw = rng.dirichlet(np.ones(NSEQ) * 0.7)
    p_data = p_raw / p_raw.sum()
    seqs, ctxs, consistent, post = build_tables(p_data)
    seqs_cache, ctxs_cache = seqs, ctxs

    part_a()
    part_b()
    nll, elbo_w, elbo_u, ts, cw, cu = part_c(p_data, seqs, post)
    part_d(p_data, seqs, post, ts, cw, cu)

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "results", "summary.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
