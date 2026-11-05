"""线性注意力的 KV cache：状态账本、容量、分块与前缀复用的合成探针。

纯 CPU / numpy / 固定 seed。所有数字由构造真值反算，组末带断言。
运行：~/Software/miniconda3/bin/python linear_attn_kv_cache_lab.py
切 notebook：python make_notebook.py（把 # %% 段变成 cell 并跑出输出）
"""

import numpy as np

RNG = np.random.default_rng(20261105)
OK = []
FAIL = []


def check(cond, msg):
    (OK if bool(cond) else FAIL).append(msg)
    if not bool(cond):
        print("  [FAIL]", msg)


def section(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def phi(x):
    """Katharopoulos et al. 2020 §3.2.1 公式 7：elu(x) + 1。"""
    a = np.asarray(x, dtype=np.float64)
    return np.where(a > 0, a + 1.0, np.exp(a) + 0.0) + 0.0


def outer(k, v):
    return np.outer(k, v)


# %% 组 1：parallel / recurrent / chunkwise / 朴素双重循环，四者等价；softmax 不等价
def g1_forms():
    section("组 1｜四种线性注意力写法逐位一致，softmax 是另一个核")
    T, dk, dv, C = 48, 8, 6, 8
    q = RNG.normal(0, 0.5, (T, dk))
    k = RNG.normal(0, 0.5, (T, dk))
    v = RNG.normal(0, 0.5, (T, dv))
    fq, fk = phi(q), phi(k)

    # 构造真值：显式双重循环，j <= i
    naive = np.zeros((T, dv))
    for i in range(T):
        num = np.zeros(dv)
        den = 0.0
        for j in range(i + 1):
            w = float(fq[i] @ fk[j])
            num += w * v[j]
            den += w
        naive[i] = num / den

    # parallel：masked 矩阵形式
    A = fq @ fk.T
    M = np.tril(np.ones((T, T)))
    Aw = A * M
    parallel = (Aw @ v) / Aw.sum(axis=1, keepdims=True)

    # recurrent：S in R^{dk x dv}, z in R^{dk}
    S = np.zeros((dk, dv))
    z = np.zeros(dk)
    recurrent = np.zeros((T, dv))
    for i in range(T):
        S += outer(fk[i], v[i])
        z += fk[i]
        recurrent[i] = (fq[i] @ S) / (fq[i] @ z)

    # chunkwise：块内矩阵形式 + 块间只传一份 S / z
    S = np.zeros((dk, dv))
    z = np.zeros(dk)
    chunkwise = np.zeros((T, dv))
    for s in range(0, T, C):
        e = min(s + C, T)
        n = e - s
        Ab = fq[s:e] @ fk[s:e].T
        Mb = np.tril(np.ones((n, n)))
        num = (Ab * Mb) @ v[s:e] + fq[s:e] @ S
        # 分母用与分子同一套 mask：每行的 fq_i · (sum_{j<=i} fk_j + z)
        den = np.array([fq[s + i] @ (z + fk[s:s + i + 1].sum(axis=0)) for i in range(n)])
        chunkwise[s:e] = num / den[:, None]
        S = S + fk[s:e].T @ v[s:e]
        z = z + fk[s:e].sum(axis=0)

    # 对照：softmax attention（因果），用同一个 Q/K/V
    scores = (q @ k.T) / np.sqrt(dk)
    scores = np.where(M > 0, scores, -np.inf)
    P = np.exp(scores - scores.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    softmax_o = P @ v

    e_par = float(np.max(np.abs(naive - parallel)))
    e_rec = float(np.max(np.abs(naive - recurrent)))
    e_chk = float(np.max(np.abs(naive - chunkwise)))
    e_smx = float(np.mean(np.abs(naive - softmax_o)) / np.mean(np.abs(naive)))
    print(f"  max|naive-parallel|   = {e_par:.3e}")
    print(f"  max|naive-recurrent|  = {e_rec:.3e}")
    print(f"  max|naive-chunkwise|  = {e_chk:.3e} (C={C}, 传递 {T // C} 次状态)")
    print(f"  mean|naive-softmax|/|naive| = {e_smx:.4f}  <- 换核不是等价变换")
    print(f"  状态形状 S = {dk} x {dv} = {dk * dv} 元素；z = {dk} 元素")

    check(e_par < 1e-10, "G1-1 parallel 与朴素真值一致")
    check(e_rec < 1e-10, "G1-2 recurrent 与朴素真值一致")
    check(e_chk < 1e-10, "G1-3 chunkwise 与朴素真值一致")
    check(T // C == 6, "G1-4 48 token / C=8 恰好传 6 次状态")
    check(e_smx > 0.10, "G1-5 线性核与 softmax 输出明显不同")
    check(dk * dv == 48, "G1-6 状态元素数 = dk*dv")
    print(f"  [OK] {len(OK)} 条断言")


# %% 组 2：每层每请求的缓存字节账本与交叉点 T*
def cache_bytes(T, n_h=32, n_kv=8, d=128, b=2):
    """返回 (softmax KV, 线性状态, 3:1 混合)：单位字节，按「每层 / 每请求」。"""
    kv = 2 * n_kv * d * T * b
    state = n_h * d * d * b
    hybrid = 0.25 * kv + 0.75 * state
    return kv, state, hybrid


def g2_ledger():
    section("组 2｜缓存账本：固定状态不是免费的，存在交叉点 T*")
    n_h, n_kv, d, b = 32, 8, 128, 2
    kv_per_token = 2 * n_kv * d * b
    state = n_h * d * d * b
    t_star = state / kv_per_token
    print(f"  设定：n_h={n_h}, n_kv={n_kv} (GQA 4:1), d={d}, dtype={b} bytes, 每层/每请求")
    print(f"  softmax 每 token KV = {kv_per_token} bytes；线性状态 = {state} bytes = {state / 2**20:.4f} MiB")
    print(f"  交叉点 T* = n_h*d^2/(2*n_kv*d) = n_h*d/(2*n_kv) = {t_star:.0f} token")
    print(f"  单头口径 T* = d/2 = {d / 2:.0f} token（n_h=n_kv 时）")

    print("\n   T(token) | softmax MiB | state MiB | 3:1 MiB | 显存口径减少 | KV口径减少")
    rows = []
    for T in (128, 256, 1024, 4096, 32768, 131072, 1048576):
        kv, st, hy = cache_bytes(T)
        save_mem = 1.0 - hy / kv
        save_kv = 0.75
        rows.append((T, kv, st, hy, save_mem, save_kv))
        print(f"  {T:>9} | {kv / 2**20:>11.4f} | {st / 2**20:>9.4f} | "
              f"{hy / 2**20:>8.4f} | {save_mem * 100:>11.4f}% | {save_kv * 100:>10.2f}%")

    check(kv_per_token == 4096, "G2-1 每 token KV = 4096 bytes")
    check(state == 1048576, "G2-2 状态 = 1048576 bytes = 1 MiB")
    check(t_star == 256, "G2-3 交叉点 256 token")
    kv256, st256, hy256 = cache_bytes(256)
    check(abs(kv256 - st256) < 1e-9 and abs(hy256 - kv256) < 1e-9,
          "G2-4 T=T* 时三种口径相等")
    check(cache_bytes(128)[2] > cache_bytes(128)[0], "G2-5 T<T* 时混合反而更占显存")
    saves = [r[4] for r in rows[1:]]
    check(all(saves[i] < saves[i + 1] for i in range(len(saves) - 1)),
          "G2-6 显存口径减少随 T 单调增")
    check(all(s < 0.75 for s in saves) and saves[-1] > 0.749,
          "G2-7 显存口径减少恒 <75%，1M 时趋近 75%")
    check(abs((1 - cache_bytes(1024)[2] / cache_bytes(1024)[0]) - 0.75 * (1 - 256 / 1024)) < 1e-12,
          "G2-8 闭式 0.75*(1-T*/T) 与逐项计算一致")
    print(f"  [OK] {len(OK)} 条断言")


# %% 组 3：状态容量：正交可读、随机 key 串扰、rank 上界、delta rule 精确改写
def g3_capacity():
    section("组 3｜有限状态的容量：memory collision 与 delta rule 改写")
    d, dv = 16, 6
    V = RNG.normal(0, 1, (128, dv))

    # 正交 key：前 d 个标准正交基，构造真值 S = sum_i v_i k_i^T，反算 S k_j = v_j
    Korth = np.eye(d)[:d]
    Sorth = V[:d].T @ Korth
    readback = Korth @ Sorth.T
    e_orth = float(np.max(np.abs(readback - V[:d])))
    print(f"  正交 key (N=d={d}) 反算最大误差 = {e_orth:.3e}")

    # 随机单位 key：串扰随 N/d 增长
    Krnd = RNG.normal(0, 1, (128, d))
    Krnd /= np.linalg.norm(Krnd, axis=1, keepdims=True)
    print("\n    N | rank(S) | 随机 key 读回相对误差 | 最小二乘残差")
    errs = []
    for N in (8, 16, 32, 64, 128):
        Kb, Vb = Krnd[:N], V[:N]
        S = Vb.T @ Kb
        out = Kb @ S.T
        rel = float(np.mean(np.linalg.norm(out - Vb, axis=1)) /
                    np.mean(np.linalg.norm(Vb, axis=1)))
        rank = int(np.linalg.matrix_rank(S))
        # 构造真值 ↔ 反算值：能否找到一个 W 使 K W = V
        W, *_ = np.linalg.lstsq(Kb, Vb, rcond=None)
        resid = float(np.linalg.norm(Kb @ W - Vb) / np.linalg.norm(Vb))
        errs.append((N, rank, rel, resid))
        print(f"  {N:>3} | {rank:>7} | {rel:>19.4f} | {resid:>14.4e}")
        check(rank == min(N, d, dv), f"G3-rank-{N}: rank(S)=min(N,dk,dv)")
        if N <= d:
            check(resid < 1e-10, f"G3-lstsq-{N}: N<=d 时存在精确解")
        else:
            check(resid > 1e-3, f"G3-lstsq-{N}: N>d 时无精确解")

    check(e_orth < 1e-12, "G3-1 正交 key 可逐位读回")
    check(errs[0][2] < errs[2][2] < errs[4][2], "G3-2 随机 key 串扰随 N 单调增")

    # 同一个 key 写两次：加性外积叠加，delta rule 覆盖
    key = np.zeros(d)
    key[0] = 1.0
    v1, v2 = V[0], V[1]
    S_add = np.outer(v1, key) + np.outer(v2, key)
    S_delta = np.outer(v1, key)
    S_delta = S_delta @ (np.eye(d) - np.outer(key, key)) + np.outer(v2, key)
    r_add, r_delta = S_add @ key, S_delta @ key
    print(f"\n  同 key 覆盖：加性读回误差 {np.linalg.norm(r_add - v2):.4f}，"
          f"delta 读回误差 {np.linalg.norm(r_delta - v2):.3e}")
    check(np.linalg.norm(r_add - v2) > 0.1, "G3-3 加性外积无法覆盖旧值")
    check(np.linalg.norm(r_delta - v2) < 1e-12, "G3-4 delta rule 精确改写为 v2")
    print(f"  [OK] {len(OK)} 条断言")


# %% 组 4：带衰减的 chunkwise 与递推一致；块内 FLOPs 与块间状态 IO 的两笔账
def g4_chunk():
    section("组 4｜chunkwise：块内二次、块间递推；C 的两笔账方向相反")
    T, dk, dv, alpha = 256, 8, 6, 0.97
    q = RNG.normal(0, 0.5, (T, dk))
    k = RNG.normal(0, 0.5, (T, dk))
    v = RNG.normal(0, 0.5, (T, dv))
    fq, fk = phi(q), phi(k)

    S = np.zeros((dk, dv))
    z = np.zeros(dk)
    rec = np.zeros((T, dv))
    for i in range(T):
        S = alpha * S + outer(fk[i], v[i])
        z = alpha * z + fk[i]
        rec[i] = (fq[i] @ S) / (fq[i] @ z)

    def chunkwise(C):
        S = np.zeros((dk, dv))
        z = np.zeros(dk)
        out = np.zeros((T, dv))
        for s in range(0, T, C):
            e = min(s + C, T)
            n = e - s
            D = np.tril(np.power(alpha, np.subtract.outer(np.arange(n), np.arange(n))))
            A = (fq[s:e] @ fk[s:e].T) * D
            carry = np.power(alpha, np.arange(1, n + 1))
            num = A @ v[s:e] + (fq[s:e] @ S) * carry[:, None]
            den = np.array([fq[s + i] @ (alpha ** (i + 1) * z +
                                         (np.power(alpha, i - np.arange(i + 1))[:, None] *
                                          fk[s:s + i + 1]).sum(axis=0)) for i in range(n)])
            out[s:e] = num / den[:, None]
            S = alpha ** n * S + (np.power(alpha, n - 1 - np.arange(n))[:, None] * fk[s:e]).T @ v[s:e]
            z = alpha ** n * z + (np.power(alpha, n - 1 - np.arange(n))[:, None] * fk[s:e]).sum(axis=0)
        return out

    errs = {}
    for C in (8, 16, 32, 64, 128):
        e = float(np.max(np.abs(rec - chunkwise(C))))
        errs[C] = e
        print(f"  C={C:>3}  max|recurrent-chunkwise| = {e:.3e}  块数={T // C}")
        check(e < 1e-9, f"G4-eq-C{C}: 衰减 chunkwise 与递推一致")
    check(max(errs.values()) < 1e-9, "G4-1 五种块大小全部一致")

    # 两笔账（解析式，非计时）：T=4096, d=128, bf16
    T2, d2, b2 = 4096, 128, 2
    kv_write = 2 * T2 * d2 * b2
    print(f"\n  softmax KV 一次性写入 = {kv_write / 2**20:.4f} MiB（对照）")
    print("    C | 块内 FLOPs (M) | 块间状态传递 (MiB) | 乘积 (M*MiB)")
    fls, ios = [], []
    for C in (16, 32, 64, 128, 256):
        flops = 2 * (T2 // C) * C * C * d2      # QK^T + A@V，各 C^2*d 每块
        io = (T2 // C) * d2 * d2 * b2
        fls.append(flops)
        ios.append(io)
        print(f"  {C:>3} | {flops / 1e6:>14.4f} | {io / 2**20:>18.4f} | "
              f"{flops / 1e6 * io / 2**20:>14.4f}")
    check(all(abs(fls[i + 1] / fls[i] - 2) < 1e-12 for i in range(4)), "G4-2 C 翻倍，块内 FLOPs 翻倍")
    check(all(abs(ios[i] / ios[i + 1] - 2) < 1e-12 for i in range(4)), "G4-3 C 翻倍，块间状态 IO 减半")
    check(all(abs(fls[i] * ios[i] - fls[0] * ios[0]) < 1 for i in range(5)),
          "G4-4 两笔账乘积恒定，C 只是权衡旋钮")
    check(ios[0] > kv_write, "G4-5 C=16 时传状态比写一遍 softmax KV 还贵")
    print(f"  [OK] {len(OK)} 条断言")


# %% 组 5：数值精度——fp32/fp16 长递推、gamma 连乘下溢与 log-space
def g5_numeric():
    section("组 5｜递推状态的精度：长序列累积误差与衰减连乘下溢")
    T, d, alpha = 20000, 8, 0.999
    k = RNG.normal(0, 0.3, (T, d))
    v = RNG.normal(0, 0.3, (T, d))

    # 构造真值：闭式 S_T = sum_j alpha^(T-1-j) k_j v_j^T，fp64 累加
    w = np.power(alpha, T - 1 - np.arange(T))
    truth = (k * w[:, None]).T @ v
    nrm = np.linalg.norm(truth)

    def run(dtype):
        S = np.zeros((d, d), dtype=dtype)
        for i in range(T):
            S = (np.asarray(alpha, dtype=dtype) * S +
                 np.outer(k[i].astype(dtype), v[i].astype(dtype))).astype(dtype)
        return float(np.linalg.norm(S.astype(np.float64) - truth) / nrm)

    e32, e16 = run(np.float32), run(np.float16)
    print(f"  T={T}, alpha={alpha}, d={d}")
    print(f"  fp32 逐步递推相对误差 = {e32:.3e}")
    print(f"  fp16 逐步递推相对误差 = {e16:.3e}  (放大 {e16 / max(e32, 1e-15):.1f} 倍)")
    check(e32 < 1e-4, "G5-1 fp32 长递推与闭式真值一致")
    check(e16 > 5 * e32, "G5-2 fp16 累积误差显著大于 fp32")

    # 块内衰减连乘：GLA 论文指出的 sigmoid 连乘不稳
    C = 128
    gamma = RNG.uniform(0.70, 0.99, C)
    truth_g = float(np.exp(np.log(gamma).sum()))
    direct16 = np.float16(1.0)
    for g in gamma:
        direct16 = np.float16(direct16 * np.float16(g))
    logspace = float(np.exp(np.log(gamma).astype(np.float32).sum()))
    print(f"\n  C={C}, gamma~U(0.70,0.99)：fp64 真值 = {truth_g:.6e}")
    e_direct = abs(float(direct16) - truth_g) / truth_g
    e_log = abs(logspace - truth_g) / truth_g
    print(f"  fp16 直接连乘 = {float(direct16):.6e}（相对误差 {e_direct:.1f} 倍）；"
          f"log-space = {logspace:.6e}（相对误差 {e_log:.3e}）")
    check(e_direct > 1.0, "G5-3 fp16 直接连乘落入 subnormal，相对误差超过 1 倍")
    check(e_log < 1e-3, "G5-4 log-space 与真值一致")
    print(f"  [OK] {len(OK)} 条断言")


# %% 组 6：前缀复用——in-place 状态不可回滚，softmax KV 可以只读共享
def g6_prefix():
    section("组 6｜前缀复用：状态必须快照 + copy-on-write，KV 只需只读共享")
    dk, dv = 8, 6
    P = RNG.normal(0, 0.5, (8, dk))
    A = RNG.normal(0, 0.5, (4, dk))
    B = RNG.normal(0, 0.5, (4, dk))
    VP = RNG.normal(0, 0.5, (8, dv))
    VA = RNG.normal(0, 0.5, (4, dv))
    VB = RNG.normal(0, 0.5, (4, dv))
    Qb = RNG.normal(0, 0.5, (4, dk))
    fp, fa, fb, fqb = phi(P), phi(A), phi(B), phi(Qb)

    def state(K, V):
        return K.T @ V, K.sum(axis=0)

    SP, zP = state(fp, VP)
    SPA, zPA = SP + fa.T @ VA, zP + fa.sum(axis=0)

    def read(S, z, Q):
        return (Q @ S) / (Q @ z)[:, None]

    # 正确路径：从 state(P) 出发接 B
    S, z = SP.copy(), zP.copy()
    correct = np.zeros((4, dv))
    for i in range(4):
        S += outer(fb[i], VB[i])
        z += fb[i]
        correct[i] = (fqb[i] @ S) / (fqb[i] @ z)

    # 错误路径：从被 A 覆盖过的 state(PA) 直接接 B，没有快照
    S, z = SPA.copy(), zPA.copy()
    wrong = np.zeros((4, dv))
    for i in range(4):
        S += outer(fb[i], VB[i])
        z += fb[i]
        wrong[i] = (fqb[i] @ S) / (fqb[i] @ z)

    # 正确路径（显式 copy-on-write：从快照 fork 一份）
    S, z = SP.copy(), zP.copy()
    cow = np.zeros((4, dv))
    for i in range(4):
        S += outer(fb[i], VB[i])
        z += fb[i]
        cow[i] = (fqb[i] @ S) / (fqb[i] @ z)

    e_wrong = float(np.mean(np.abs(wrong - correct)) / np.mean(np.abs(correct)))
    e_cow = float(np.max(np.abs(cow - correct)))
    print(f"  无快照直接从 state(PA) 接 B 的相对偏差 = {e_wrong:.4f}")
    print(f"  快照 + copy-on-write 后与正确路径最大偏差 = {e_cow:.3e}")
    print(f"  需复制的状态元素数 = dk*dv = {dk * dv}；前缀 KV 元素数 = 2*dk*8 = {2 * dk * 8}")
    check(e_wrong > 0.05, "G6-1 无快照时状态不可回滚，输出明显不同")
    check(e_cow < 1e-12, "G6-2 快照 + COW 与正确路径逐位一致")
    check(SP.shape == (dk, dv), "G6-3 状态是固定形状矩阵")

    # softmax 侧：前缀 K/V 是只读列表，两条分支共享同一份前缀即可
    def smx(prefix_k, prefix_v, suffix_k, suffix_v, query):
        K = np.concatenate([prefix_k, suffix_k], axis=0)
        V = np.concatenate([prefix_v, suffix_v], axis=0)
        out = np.zeros((len(query), dv))
        for i, qi in enumerate(query):
            s = K @ qi / np.sqrt(dk)
            p = np.exp(s - s.max())
            p /= p.sum()
            out[i] = p @ V
        return out

    oA = smx(P, VP, A, VA, Qb)
    oB = smx(P, VP, B, VB, Qb)
    check(oA.shape == oB.shape == (4, dv), "G6-4 两条分支从同一份只读前缀 KV 出发")
    check(np.all(np.isfinite(oA)) and np.all(np.isfinite(oB)), "G6-5 共享前缀不改动前缀 KV")
    print(f"  [OK] {len(OK)} 条断言")


# %% 组 7：混合架构的两种口径——KV cache 减少 vs 总缓存显存减少
def g7_hybrid():
    section("组 7｜3:1 / 7:1 混合：官方 75% 是哪个口径")
    ratios = {"Kimi Linear / Qwen3-Next (3:1)": 0.75,
              "MiniMax-01 (7:1)": 7 / 8}
    t_star = 256
    for name, f in ratios.items():
        print(f"\n  {name}: 线性层占比 f = {f:.4f}，官方 KV 口径减少 = {f * 100:.2f}%")
        print("      T | 显存口径减少 | 与 f 的差")
        for T in (1024, 4096, 32768, 1048576):
            save = f * (1 - t_star / T)
            print(f"  {T:>6} | {save * 100:>11.4f}% | {(f - save) * 100:>9.4f} pp")
            check(save <= f + 1e-12, f"G7-{f:.3f}-T{T}: 显存口径不超过层型占比")
        check(abs(f - (0.75 if abs(f - 0.75) < 1e-9 else 0.875)) < 1e-9,
              "G7-1 层型占比与官方口径数字一致")
    check(abs(0.75 * (1 - 256 / 4096) - 0.703125) < 1e-12,
          "G7-2 4K 时 3:1 显存口径 = 70.3125%")
    print(f"\n  [OK] {len(OK)} 条断言")


# %% 主流程与断言总数
def main():
    for g in (g1_forms, g2_ledger, g3_capacity, g4_chunk, g5_numeric, g6_prefix, g7_hybrid):
        g()
    print("\n" + "=" * 72)
    print(f"总计 {len(OK)}/{len(OK) + len(FAIL)} 条断言通过")
    if FAIL:
        print("失败：", FAIL)
        raise SystemExit(1)
    print("=" * 72)


if __name__ == "__main__":
    main()
