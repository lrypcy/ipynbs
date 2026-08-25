"""QoQ/QServe 与 QQQ 实验代码（第 13 篇配套，与正文代码块一致）
依赖：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

def sym_quant(A, bits):
    s = np.abs(A).max() / (2 ** (bits - 1) - 1)
    return np.round(A / s) * s

def asym_quant_cols(A, bits):
    """逐通道非对称量化：zero-point 平移 + 均匀网格"""
    lo, hi = A.min(0), A.max(0)
    zp = (lo + hi) / 2
    s = np.where(hi > lo, (hi - lo) / (2 ** bits - 1), 1.0)
    return zp + s * np.round((A - zp) / s)

def sym_quant_cols(A, bits):
    out = np.empty_like(A)
    for j in range(A.shape[1]):
        out[:, j] = sym_quant(A[:, j], bits)
    return out

def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

# ---------- Demo A: SmoothAttention ----------
T, d = 1024, 128
Q = rng.normal(0, 1.0, (T, d))
K = rng.normal(0, 1.0, (T, d))
oc = rng.choice(d, 6, replace=False)
for j in oc:                                      # K 的离群通道：大均值+小波动
    K[:, j] = rng.uniform(20, 40) + 0.5 * rng.normal(0, 1, T)

S_ref = (Q @ K.T) / np.sqrt(d)
S_naive = (Q @ sym_quant(K, 8).T) / np.sqrt(d)    # 直接对 K 做 per-tensor int8
lam = np.abs(K).max(0) ** 0.6                     # SmoothAttention 型逐通道缩放
K_s, Q_s = K / lam, Q * lam                       # 等效迁移：λ 折到 Q 一侧
S_smooth = (Q_s @ sym_quant(K_s, 8).T) / np.sqrt(d)
print("[DemoA] attention 分数矩阵的重构相对误差（K 侧 per-tensor int8）")
print(f"  直接量化 K      : {rel_err(S_naive, S_ref):.4f}")
print(f"  SmoothAttention : {rel_err(S_smooth, S_ref):.4f}"
      f"   （降低 {(1-rel_err(S_smooth,S_ref)/rel_err(S_naive,S_ref))*100:.1f}%）")

# ---------- Demo B: LoRC 低秩补偿 ----------
M = N = 256
W = rng.normal(0, 0.02, (M, N))
X = rng.normal(0, 1.0, (2048, N))
Y_ref = X @ W.T
Wq = np.stack([sym_quant(W[i], 4) for i in range(M)])     # RTN-W4 权重
delta = W - Wq                                            # 量化残差
sv = np.linalg.svd(delta, compute_uv=False)
print("\n[DemoB] W4(RTN) 层输出误差 + rank-r fp16 低秩补偿")
print(f"  补偿前          : {rel_err(X @ Wq.T, Y_ref):.4f}")
print(f"  （残差 top-8 奇异值能量占比 = {(sv[:8]**2).sum()/(sv**2).sum():.2f}，"
      f"越低则低秩可压缩性越差）")
for r in [4, 8, 16]:
    U, s, Vt = np.linalg.svd(delta, full_matrices=False)
    corr = U[:, :r] @ np.diag(s[:r]) @ Vt[:r]
    overhead = r * (M + N) / (M * N) * 100                # 相对权重量的 fp16 参数占比
    print(f"  rank-{r:<2d} 补偿后 : {rel_err(X @ (Wq + corr).T, Y_ref):.4f}"
          f"   （存储开销 {overhead:.2f}%）")

# ---------- Demo C: KV cache 非对称 int4 ----------
Vc = rng.normal(0, 1.0, (2048, d))
occ = rng.choice(d, 10, replace=False)
for j in occ:                                             # 少数通道带稳定正偏移
    Vc[:, j] = rng.uniform(6, 14) + rng.normal(0, 0.8, 2048)
v_sym, v_asym = sym_quant_cols(Vc, 4), asym_quant_cols(Vc, 4)
print("\n[DemoC] KV cache(V) 的 4-bit 重建相对误差")
print(f"  逐通道对称 int4   : {rel_err(v_sym, Vc):.4f}")
print(f"  逐通道非对称 int4 : {rel_err(v_asym, Vc):.4f}   "
      f"（降低 {(1-rel_err(v_asym,Vc)/rel_err(v_sym,Vc))*100:.1f}%）")
