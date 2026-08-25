"""OliVe 实验代码（第 12 篇配套，与正文代码块一致）
依赖：numpy
"""
import numpy as np

rng = np.random.default_rng(0)

LEVELS_M = np.array([1.0, 1.25, 1.5, 1.75])      # 迷你浮点尾数(m2)

def e2m1_shared(x):
    """4-bit E2M1 思想：±{0,.5,1,1.5,2,3,4,6} × 2^k，k 由全体元素 absmax 决定"""
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    k = int(np.floor(np.log2(np.abs(x).max() / 6)))
    grid = levels * 2.0 ** k
    return grid[np.argmin(np.abs(x[:, None] - grid[None, :]), axis=1)]

def narrow_saturate(x, k_normal):
    """窄格式(与正常值同一网格)，超界即饱和——'没有宽格式'的近似"""
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])
    grid = levels * 2.0 ** k_normal
    return np.clip(grid[np.argmin(np.abs(x[:, None] - grid[None, :]), axis=1)],
                   -grid[-1], grid[-1])

def abfloat(x, z_lo=3, z_hi=10):
    """教学版 abfloat(s1,z4,m2)：可表示幅值 = {1,1.25,1.5,1.75}×2^z, z∈[z_lo,z_hi]
       关键性质：幅值 < 2^z_lo 一律冲刷为 0 —— 编码区间整体跳过正常数值范围"""
    xq = np.zeros_like(x)
    nz = np.abs(x) > 0
    a = np.abs(x[nz])
    z = np.clip(np.floor(np.log2(a)), z_lo, z_hi).astype(int)
    scale = 2.0 ** z
    m = LEVELS_M[np.argmin(np.abs(a / scale - LEVELS_M[:, None]), axis=0)]
    xq[nz] = np.sign(x[nz]) * m * scale
    return xq

def rel_err(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

# ---------- 场景：正常值 N(0,1) + 2% 的 16~64 倍离群值 ----------
n = 4096
x = rng.normal(0, 1.0, n)
pos = rng.choice(n, int(n * 0.02), replace=False)
x[pos] = rng.uniform(16.0, 64.0, len(pos))
is_out = np.abs(x) > 8.0
k_normal = int(np.floor(np.log2(np.abs(x[~is_out]).max() / 6)))   # 正常值自己的指数

# ---------- 三种编码 ----------
q1 = e2m1_shared(x)                                    # 方案1：全员共享网格(被劫持)
q2 = x.copy()                                          # 方案2：victim 置零 + 窄格式饱和
q2[~is_out] = narrow_saturate(x[~is_out], k_normal)
q2[is_out] = narrow_saturate(x[is_out], k_normal)
q3 = x.copy()                                          # 方案3：OVP
q3[~is_out] = e2m1_shared(x[~is_out])                  #   正常对：干净的细网格
q3[is_out] = abfloat(x[is_out])                        #   离群值：宽格式
n_victim = 0
for a_ in range(0, n, 2):                              # 固定步长相邻配对
    b_ = a_ + 1
    if is_out[a_] != is_out[b_]:                       #   (outlier, victim) 对
        q2[b_ if is_out[a_] else a_] = 0.0             #   两方案的 victim 同样牺牲
        q3[b_ if is_out[a_] else a_] = 0.0
        n_victim += 1

print(f"[DemoA] 同槽位预算下的重构相对误差（离群 {is_out.mean()*100:.0f}%，"
      f"victim 约 {n_victim/n*100:.1f}%；k_normal={k_normal} vs 共享 k="
      f"{int(np.floor(np.log2(np.abs(x).max()/6)))}）")
print(f"  方案1 全员共享窄网格            : {rel_err(q1, x):.4f}")
print(f"  方案2 victim置零+窄格式饱和     : {rel_err(q2, x):.4f}")
print(f"  方案3 OVP(细网格+abfloat+victim): {rel_err(q3, x):.4f}")

# ---------- Demo B: 范围覆盖对比 ----------
print("\n[DemoB] 可表示范围对比")
print(f"  abfloat(z∈[3,10]) 覆盖 : [{LEVELS_M[0]*2**3:.1f}, {LEVELS_M[-1]*2**10:.0f}]，"
      f"且 (0,{2**3}) 区间无任何编码 —— '跳过'正常区间")
v_test = rng.uniform(16.0, 64.0, 200)
errs_ab = [rel_err(abfloat(np.array([v])), np.array([v])) for v in v_test]
errs_sat = [rel_err(narrow_saturate(np.array([v]), k_normal), np.array([v])) for v in v_test]
print(f"  离群值(16~64)平均表示误差 : abfloat={np.mean(errs_ab)*100:.1f}%  "
      f"窄格式饱和={np.mean(errs_sat)*100:.1f}%")
