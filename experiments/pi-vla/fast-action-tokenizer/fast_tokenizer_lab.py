"""FAST 动作分词实验：DCT + 量化 + 列优先展平 + BPE。

配套文章 `_posts/2026-11-03-pi-vla-02-fast-action-tokenizer.md`。共用件在
`fast_tokenizer_lib.py`，配图在 `fast_tokenizer_figures.py`。

五个组：
  A 同一秒动作、采样更密时，分箱 token 数 = 维度 x 步数，FAST 亚线性增长
  B 量化尺度 gamma 是唯一压缩旋钮：round(gamma*c) 的归零阈值是 1/(2*gamma)
  C BPE 的独立贡献：去掉 BPE 只剩非零系数，仍比分箱少，但明显多于 BPE
  D 列优先 vs 行优先展平：同样的系数预算，先给低频能重建多少信号
  E 同 token 预算下的重建误差：分箱只能靠降采样换 token，FAST 可以直接截频

只依赖 numpy。运行：python3 fast_tokenizer_lab.py
"""

import numpy as np

import fast_tokenizer_lib as F
from fast_tokenizer_lib import (  # noqa: F401
    bin_tokens,
    bpe_train,
    dct_matrix,
    dct_quantize,
    flatten,
    fast_token_count,
    make_dataset,
    normalize,
    section,
)

check = F.check


# %% [markdown]
# ## A. 采样更密，token 数怎么变
#
# 分箱的 token 数是 `维度 x 步数`，控制频率翻一倍就翻一倍。FAST 压的是信号的
# **带宽**，不是采样点数，所以同一个动作块采样加密 10 倍，token 数只该涨一点。
# A2 把这件事拆开验证：固定采样率、只改带宽，看 token 数跟谁走。

# %%
section("A1. 分箱 token 数 = 维度 x 步数；FAST 跟着带宽走")
DIM = 7
H_LIST = [5, 10, 20, 50]
VOCAB = 512
rows = {}
for h in H_LIST:
    act = normalize(make_dataset(32, DIM, h, seed=100 + h))
    q, recon_bin = bin_tokens(act)
    coef = dct_quantize(act)
    merges = bpe_train(flatten(coef), vocab_size=VOCAB)
    n_fast = fast_token_count(coef, merges)
    n_nonzero = float(np.mean((coef != 0).sum(axis=(1, 2))))
    recon_fast = np.einsum("cdk,kh->cdh", coef / 10.0, dct_matrix(h))
    rows[h] = dict(n_bin=DIM * h, n_fast=n_fast, n_nonzero=n_nonzero,
                   rmse_bin=float(np.sqrt(np.mean((recon_bin - act) ** 2))),
                   rmse_fast=float(np.sqrt(np.mean((recon_fast - act) ** 2))))
    print(f"  H={h:3d}  分箱 {rows[h]['n_bin']:4d}  "
          f"FAST {n_fast:6.1f}  非零系数 {n_nonzero:6.1f}  "
          f"压缩比 {rows[h]['n_bin'] / n_fast:5.2f}x")

check(all(rows[h]["n_bin"] == DIM * h for h in H_LIST), "分箱 token 数必须恰好是 dim*H")
check(rows[50]["n_fast"] / rows[5]["n_fast"] < 3.0, "采样率涨 10 倍，FAST token 数不该涨 3 倍")
ratios = [rows[h]["n_bin"] / rows[h]["n_fast"] for h in H_LIST]
check(all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1)), "压缩比应随频率单调上升")
check(ratios[-1] > 4.0, "H=50 时压缩比应显著大于 1")
check(all(rows[h]["n_fast"] < rows[h]["n_nonzero"] for h in H_LIST), "BPE 应比裸非零系数更少")
check(rows[50]["rmse_fast"] < 0.1, "gamma=10 时 FAST 重建误差应很小")

section("A2. 固定采样率、只改带宽：token 数跟带宽走，不跟采样率走")
print("  带宽上限   非零系数   FAST token   分箱 token")
bw_rows = []
for fmax in [1.0, 1.5, 2.5, 4.0]:
    act = normalize(make_dataset(32, DIM, 50, seed=170, fmax=fmax))
    coef = dct_quantize(act)
    merges = bpe_train(flatten(coef), vocab_size=VOCAB)
    n_fast = fast_token_count(coef, merges)
    n_nz = float(np.mean((coef != 0).sum(axis=(1, 2))))
    bw_rows.append((fmax, n_nz, n_fast))
    print(f"  {fmax:6.1f} Hz   {n_nz:8.1f}   {n_fast:9.1f}   {DIM * 50:9d}")
check(all(bw_rows[i][2] < bw_rows[i + 1][2] for i in range(len(bw_rows) - 1)),
      "带宽越宽，token 数应单调增加")
check(bw_rows[0][2] < 0.7 * bw_rows[-1][2], "带宽 1Hz 与 4Hz 的 token 数应差出 30% 以上")
check(all(r[2] < DIM * 50 for r in bw_rows), "所有带宽下 FAST 都应少于分箱")

# %% [markdown]
# ## B. gamma 是唯一的压缩旋钮
#
# `round(gamma*c)` 有一个精确的归零阈值：`|c| < 1/(2*gamma)` 必被舍入到 0。
# 所以"保留多少个系数"不是超参，是分布形状与 gamma 共同决定的结果——这也是
# 论文里找不到"保留几个 DCT 系数"这个数字的原因。
#
# 方向要说清楚：**gamma 越大越精细，非零系数越多、token 越多**，压缩与保真
# 是同一条轴的两端。

# %%
section("B. gamma 的归零阈值、保留比例与保真度")
act = normalize(make_dataset(32, DIM, 50, seed=150))
h = 50
d = dct_matrix(h)
coefs_raw = act @ d.T
print("  gamma   非零占比   重建 RMSE   舍入阈值 1/(2gamma)")
gammas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
keep_frac, rmse_g = [], []
for g in gammas:
    cq = np.round(g * coefs_raw)
    keep = float(np.mean(cq != 0))
    recon = (cq / g) @ d
    rmse = float(np.sqrt(np.mean((recon - act) ** 2)))
    keep_frac.append(keep)
    rmse_g.append(rmse)
    print(f"  {g:5.1f}   {keep:8.4f}   {rmse:10.5f}   {1 / (2 * g):9.4f}")
    check(np.all(np.abs(coefs_raw[cq != 0]) >= 1 / (2 * g) - 1e-12),
          f"gamma={g}：非零系数的 |c| 必须 >= 1/(2*gamma)")
check(all(keep_frac[i] < keep_frac[i + 1] for i in range(len(keep_frac) - 1)),
      "gamma 越大，保留下来的系数越多")
check(all(rmse_g[i] > rmse_g[i + 1] for i in range(len(rmse_g) - 1)), "gamma 越大误差越小")
check(keep_frac[0] < 0.15, "gamma=1 时只有不到 15% 的系数活下来（平滑信号本身稀疏）")
check(keep_frac[-1] > 0.25, "gamma=50 时应有 25% 以上系数保留")
check(rmse_g[-1] < 0.01, "gamma=50 时重建误差应小于 0.01")
check(rmse_g[3] < 0.03, "gamma=10 的量化误差应低于全量程的 3%")

# 边际效率：每多花 1 个百分点系数，换来多少误差下降
eff = [(keep_frac[i + 1] - keep_frac[i]) / (rmse_g[i] - rmse_g[i + 1])
       for i in range(len(gammas) - 1)]
print("  边际效率（每降 0.01 RMSE 需要多留几个百分点系数）："
      + "  ".join(f"{e / 100:.2f}" for e in eff))
check(eff[-1] > 3 * eff[0], "越往高保真走，每一个百分点的系数越不划算")

# %% [markdown]
# ## C. BPE 到底贡献了多少
#
# 消融方式：把 BPE 拿掉，只保留"非零系数"作为 token（这是对 BPE 最有利的
# 假想基线——零免费）。剩下的是 BPE 真正的贡献。

# %%
section("C. BPE 的独立贡献")
for h in [20, 50]:
    for dim in [7, 14]:
        act = normalize(make_dataset(24, dim, h, seed=200 + h + dim))
        coef = dct_quantize(act)
        flat = flatten(coef)
        merges = bpe_train(flat, vocab_size=VOCAB)
        n_bpe = fast_token_count(coef, merges)
        n_nz = float(np.mean((coef != 0).sum(axis=(1, 2))))
        n_bin = dim * h
        print(f"  dim={dim:2d} H={h:2d}  分箱 {n_bin:4d}  "
              f"去BPE(非零) {n_nz:6.1f}  FAST {n_bpe:6.1f}  "
              f"BPE 再压 {n_nz / n_bpe:4.2f}x")
        check(n_bpe < n_nz, "带 BPE 必须比裸非零系数更少")
        check(n_nz < n_bin, "即使去掉 BPE 也应少于分箱")
        check(n_nz / n_bpe > 1.2, "BPE 的额外压缩应可测")
        if dim == 14 and h == 50:
            check(n_bin / n_bpe > 5.0, "14 维 50Hz 的压缩比应超过 5 倍")

# %% [markdown]
# ## D. 为什么必须列优先展平
#
# 自回归是从左往右吐 token 的，被截断或提前终止时只留下前缀。列优先把
# **所有维度的低频**排在最前，行优先则把**某个维度的全部频率**排在最前。
# 两种顺序在总 RMSE 上可能打平，但看每一维就会露馅：行优先在小预算下让
# 大部分维度直接留成全零。

# %%
section("D. 列优先 vs 行优先：同样的系数预算")
act = normalize(make_dataset(32, 14, 50, seed=250))
h, dim = 50, 14
d = dct_matrix(h)
coef = act @ d.T
flat_f = flatten(coef, order="freq_first")
flat_d = flatten(coef, order="dim_first")
print("  预算   列优先 均值RMSE  列优先 最差维  |  行优先 均值RMSE  行优先 最差维")
for n_freq in [1, 2, 4, 8, 16]:
    budget = n_freq * dim
    rec_c = (np.where(np.arange(h)[None, None, :] < n_freq, coef, 0.0)) @ d
    trunc = np.zeros_like(flat_d)
    trunc[:, :budget] = flat_d[:, :budget]
    rec_r = trunc.reshape(coef.shape) @ d
    per_dim_c = np.sqrt(np.mean((rec_c - act) ** 2, axis=(0, 2)))
    per_dim_r = np.sqrt(np.mean((rec_r - act) ** 2, axis=(0, 2)))
    print(f"  {budget:5d}   {np.sqrt(np.mean((rec_c - act) ** 2)):12.5f}"
          f"  {per_dim_c.max():10.5f}  |  "
          f"{np.sqrt(np.mean((rec_r - act) ** 2)):13.5f}  {per_dim_r.max():10.5f}")
    if n_freq <= 4:
        check(per_dim_r.max() > 1.5 * per_dim_c.max(),
              f"预算 {budget} 个系数时，行优先最差的一维应烂得多")
check(per_dim_r.max() > 0.7,
      "行优先即使花掉 224 个系数，最后一维仍是全零、误差几乎不动")
check(per_dim_c.max() < 0.01, "列优先花同样的 224 个系数，所有维度都已重建好")

# 顺带量一下两种展平顺序对 BPE 的影响（哪边更容易压）
coef_q_d = np.round(10 * coef).astype(np.int64)
n_tok_f = fast_token_count(coef_q_d, bpe_train(flatten(coef_q_d, order="freq_first"), vocab_size=VOCAB))
n_tok_d = fast_token_count(coef_q_d, bpe_train(flatten(coef_q_d, order="dim_first"), vocab_size=VOCAB))
print(f"  两种顺序的 token 数：列优先 {n_tok_f:.1f}  行优先 {n_tok_d:.1f}")

# %% [markdown]
# ## E. 同 token 预算下谁更省
#
# 分箱的结构性问题：token 数被 `维度 x 步数` 锁死，想省 token 只能**降采样**，
# 也就是直接丢时间分辨率。FAST 有真正的旋钮——少发几个高频系数。

# %%
section("E. token 预算 vs 重建误差")
act = normalize(make_dataset(32, 14, 50, seed=300))
h, dim = 50, 14
d = dct_matrix(h)
coef_raw = act @ d.T
coef_q = np.round(10 * coef_raw).astype(np.int64)
merges = bpe_train(flatten(coef_q), vocab_size=VOCAB)

fast_curve = []
for n_freq in range(1, h + 1):
    rec = (np.where(np.arange(h)[None, None, :] < n_freq, coef_q / 10.0, 0.0)) @ d
    rmse = float(np.sqrt(np.mean((rec - act) ** 2)))
    n_tok = fast_token_count(coef_q, merges, n_freq=n_freq)
    fast_curve.append((n_tok, rmse))
print("  FAST 曲线（token 数 -> RMSE）抽样：")
for n_freq in [1, 2, 4, 8, 16, 32, 50]:
    n_tok, rmse = fast_curve[n_freq - 1]
    print(f"    前 {n_freq:2d} 个频率  {n_tok:6.1f} token  RMSE {rmse:.5f}")

# 分箱 + 降采样：先按 256 档量化（这是分箱本来的精度），再每隔 s 步留一个
# token、线性插值补回。分箱想省 token 只有这一条路。
_, act_bin = bin_tokens(act, n_bins=256)
bin_curve = []
for s in [1, 2, 4, 8, 16, 50]:
    keep = np.arange(0, h, s)
    rec = np.empty_like(act)
    for ci in range(act.shape[0]):
        for di in range(dim):
            rec[ci, di] = np.interp(np.arange(h), keep, act_bin[ci, di, keep])
    rmse = float(np.sqrt(np.mean((rec - act) ** 2)))
    bin_curve.append((dim * len(keep), rmse))
print("  分箱曲线（token 数 -> RMSE）：")
for n_tok, rmse in bin_curve:
    print(f"    {n_tok:6d} token  RMSE {rmse:.5f}")

check(bin_curve[0][0] == dim * h, "分箱全分辨率 token 数 = dim*H")
# 找到各自到达 RMSE <= 0.02 的最小 token 预算
THR = 0.02
fast_need = min((t for t, e in fast_curve if e <= THR), default=None)
bin_need = min((t for t, e in bin_curve if e <= THR), default=None)
print(f"  达到 RMSE<={THR} 所需 token：FAST {fast_need:.1f}，分箱 {bin_need}")
check(fast_need is not None and fast_need < 90, "FAST 应在 90 token 内达到 0.02")
check(bin_need is not None and bin_need >= 300, "分箱必须花掉大半 token 才能达到同一精度")
check(bin_need / fast_need > 4.0, "同一精度下分箱的 token 开销应高数倍")

# 同等预算（约 64 token）下直接比
fast_64 = min(fast_curve, key=lambda p: abs(p[0] - 64))
bin_64 = min(bin_curve, key=lambda p: abs(p[0] - 64))
print(f"  64 token 附近：FAST {fast_64[1]:.5f}（{fast_64[0]:.1f} token），"
      f"分箱 {bin_64[1]:.5f}（{bin_64[0]} token）")
check(bin_64[1] > 5 * fast_64[1], "64 token 预算下分箱应差出 5 倍以上")
check(fast_curve[-1][1] > 0.005, "gamma=10 的量化误差下限应可见（FAST 不会归零）")

print("\n" + "=" * 74)
print(f"全部断言通过：{F.N_ASSERT} 条")
print("=" * 74)
