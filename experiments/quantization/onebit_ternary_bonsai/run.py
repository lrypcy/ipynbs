"""1-bit / 三元权重量化实验（扩展篇 E4 配套：BitNet b1.58 与 Bonsai）

依赖：numpy + matplotlib
运行：/Users/congyuan/Software/miniconda3/bin/python run.py

四个实验：
  ExpA 位宽账          —— ternary / binary 的真实 bits-per-weight 怎么算
  ExpB 字母表 x 粒度   —— 重建误差（含 INT4/INT3/INT2 参照系）
  ExpC 误差 -> 输出    —— 权重重建误差如何传到激活输出
  ExpD 三条路径        —— 训完再量化 / 随机起+环内 / FP 起点+环内微调（手写 STE）
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 0
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(SEED)
res = {}

LINE = "=" * 78


def bpw(alphabet_bits, group_size, scale_bits=16):
    """真实位宽 = 码本位宽 + 每组一个 scale 的摊销开销"""
    return alphabet_bits + scale_bits / group_size


# ============================================================ ExpA 位宽账
print(LINE)
print("ExpA  位宽账：字母表 + group scale 开销")
print(LINE)
print(f"{'format':38s} {'bpw':>7s} {'vs FP16':>8s}")
table_a = []
for name, b in [
    ("FP16 baseline", 16.0),
    ("INT4 g128", bpw(4.0, 128)),
    ("INT2 g128", bpw(2.0, 128)),
    ("Ternary ideal log2(3), no scale", np.log2(3)),
    ("Ternary g128 (PrismML)", bpw(np.log2(3), 128)),
    ("Ternary g128 (2-bit slot)", bpw(2.0, 128)),
    ("Binary g128 (GGUF Q1_0_g128)", bpw(1.0, 128)),
    ("Binary g128 (MLX scale+bias)", bpw(1.0, 128) + 16 / 128),
]:
    print(f"{name:38s} {b:7.3f} {16.0 / b:7.2f}x")
    table_a.append({"format": name, "bpw": round(float(b), 4), "vs_fp16": round(16.0 / b, 3)})
res["expA_bits_account"] = table_a

print("\n校验 PrismML 公开口径：")
print(f"  binary  g128 = {bpw(1.0,128):.4f} bpw, {16/bpw(1.0,128):.2f}x   (官方 1.125 / 14.2x)")
print(f"  ternary g128 = {bpw(np.log2(3),128):.4f} bpw, {16/bpw(np.log2(3),128):.2f}x  (官方 1.71 / 9.4x)")
print(f"  MLX 1-bit    = {bpw(1.0,128)+16/128:.4f} bpw           (官方 1.25)")
N = 27.3e9
print(f"  27.3B params: binary g128 -> {N*bpw(1.0,128)/8/1e9:.2f} GB (官方 3.9); "
      f"ternary g128 -> {N*bpw(np.log2(3),128)/8/1e9:.2f} GB (官方 5.9)")
res["expA_prismml_check"] = {
    "binary_bpw": round(bpw(1.0, 128), 4),
    "ternary_bpw": round(bpw(np.log2(3), 128), 4),
    "mlx_bpw": round(bpw(1.0, 128) + 16 / 128, 4),
    "b27_binary_gb": round(N * bpw(1.0, 128) / 8 / 1e9, 2),
    "b27_ternary_gb": round(N * bpw(np.log2(3), 128) / 8 / 1e9, 2),
}


# ============================================================ 量化器
def tern(W, axis=-1, ls=True):
    """三元量化：absmean 归一化 + round（BitNet b1.58 口径）"""
    g = np.abs(W).mean(axis=axis, keepdims=True) + 1e-12
    T = np.clip(np.round(W / g), -1, 1)
    if ls:
        s = (W * T).sum(axis=axis, keepdims=True) / (T * T).sum(axis=axis, keepdims=True) + 1e-12
    else:
        s = g
    return T * s


def intq(W, bits, axis=-1, ls=True):
    """对称均匀整数量化，absmax scale"""
    qmax = 2 ** (bits - 1) - 1
    amax = np.abs(W).max(axis=axis, keepdims=True) + 1e-12
    T = np.clip(np.round(W / amax * qmax), -qmax, qmax)
    if ls:
        s = (W * T).sum(axis=axis, keepdims=True) / (T * T).sum(axis=axis, keepdims=True) + 1e-12
    else:
        s = amax / qmax
    return T * s


def binq(W, axis=-1):
    """二值量化：sign + 最小二乘 scale"""
    T = np.sign(W)
    s = (W * T).sum(axis=axis, keepdims=True) / np.sum(np.ones_like(W), axis=axis, keepdims=True)
    return T * s


def grouped(W, g, fn):
    m, n = W.shape
    return fn(W.reshape(m, n // g, g), axis=-1).reshape(m, n)


# ============================================================ ExpB 字母表 x 粒度
M, Nw = 512, 512
W = rng.normal(0, 1.0, size=(M, Nw))
W = np.where(rng.random((M, Nw)) < 0.01, W * 8.0, W)     # 1% 重尾 outlier
W /= np.sqrt((W ** 2).mean())

print()
print(LINE)
print("ExpB  重建误差：字母表 x 粒度（W: 512x512 高斯 + 1% 重尾 outlier）")
print(LINE)
print(f"{'method':38s} {'bpw':>7s} {'rel err':>9s}")
rows_b = []
for bits in (4, 3, 2):
    for g in (None, 128, 32):
        Wh = tern(W, axis=None) if False else (
            intq(W, bits, axis=None) if g is None else grouped(W, g, lambda a, axis: intq(a, bits, axis)))
        b = bits if g is None else bpw(bits, g)
        e = float(np.linalg.norm(W - Wh) / np.linalg.norm(W))
        name = f"INT{bits} " + ("per-tensor" if g is None else f"g{g}")
        print(f"{name:38s} {b:7.3f} {e*100:8.2f}%")
        rows_b.append({"method": name, "alphabet": f"INT{bits}", "group": g,
                       "bpw": round(float(b), 4), "rel_err": round(e, 5)})
for g in (None, 256, 128, 64, 32):
    Wh = tern(W, axis=None) if g is None else grouped(W, g, lambda a, axis: tern(a, axis))
    b = np.log2(3) if g is None else bpw(np.log2(3), g)
    e = float(np.linalg.norm(W - Wh) / np.linalg.norm(W))
    name = "Ternary " + ("per-tensor" if g is None else f"g{g}")
    print(f"{name:38s} {b:7.3f} {e*100:8.2f}%")
    rows_b.append({"method": name, "alphabet": "ternary", "group": g,
                   "bpw": round(float(b), 4), "rel_err": round(e, 5)})
for g in (None, 128):
    Wh = binq(W, axis=None) if g is None else grouped(W, g, lambda a, axis: binq(a, axis))
    b = 1.0 if g is None else bpw(1.0, g)
    e = float(np.linalg.norm(W - Wh) / np.linalg.norm(W))
    name = "Binary " + ("per-tensor" if g is None else f"g{g}")
    print(f"{name:38s} {b:7.3f} {e*100:8.2f}%")
    rows_b.append({"method": name, "alphabet": "binary", "group": g,
                   "bpw": round(float(b), 4), "rel_err": round(e, 5)})
res["expB_recon"] = rows_b

# 粒度边际收益：从 per-tensor 细化到 g32，误差降了多少、位宽涨了多少
def gran_gain(alphabet):
    pts = {}
    for g in (None, 256, 128, 64, 32):
        if alphabet == "INT4":
            Wh = intq(W, 4, axis=None) if g is None else grouped(W, g, lambda a, axis: intq(a, 4, axis))
            b = 4.0 if g is None else bpw(4.0, g)
        elif alphabet == "INT2":
            Wh = intq(W, 2, axis=None) if g is None else grouped(W, g, lambda a, axis: intq(a, 2, axis))
            b = 2.0 if g is None else bpw(2.0, g)
        else:
            Wh = tern(W, axis=None) if g is None else grouped(W, g, lambda a, axis: tern(a, axis))
            b = np.log2(3) if g is None else bpw(np.log2(3), g)
        pts[g] = (float(b), float(np.linalg.norm(W - Wh) / np.linalg.norm(W)))
    return pts


print("\n粒度边际收益（per-tensor -> g32）：")
gain = {}
for ab in ("INT4", "INT2", "ternary"):
    pts = gran_gain(ab)
    e0, e1 = pts[None][1], pts[32][1]
    b0, b1 = pts[None][0], pts[32][0]
    print(f"  {ab:8s}: 误差 {e0*100:6.2f}% -> {e1*100:6.2f}%  (降 {(1-e1/e0)*100:5.1f}%)   "
          f"位宽 {b0:.3f} -> {b1:.3f} bpw (涨 {(b1/b0-1)*100:5.1f}%)")
    gain[ab] = {"err_pt": round(e0, 5), "err_g32": round(e1, 5),
                "err_drop_pct": round((1 - e1 / e0) * 100, 2),
                "bpw_pt": round(b0, 4), "bpw_g32": round(b1, 4),
                "bpw_gain_pct": round((b1 / b0 - 1) * 100, 2),
                "curve": {str(k): [round(v[0], 4), round(v[1], 5)] for k, v in pts.items()}}
res["expB_granularity_gain"] = gain

# 三元码本中 0 的占比
T_pt = np.clip(np.round(W / (np.abs(W).mean() + 1e-12)), -1, 1)
zero_ratio = float((T_pt == 0).mean())
print(f"\n三元码本中 0 的占比（per-tensor absmean）: {zero_ratio*100:.1f}%")
res["expB_ternary_zero_ratio"] = round(zero_ratio, 4)


# ============================================================ ExpC 误差 -> 输出
X = rng.normal(0, 1.0, size=(256, Nw))
X[:, rng.choice(Nw, size=Nw // 20, replace=False)] *= 10.0   # 5% outlier 通道
Y = X @ W.T
print()
print(LINE)
print("ExpC  权重误差 -> 输出误差（激活含 5% outlier 通道 x10）")
print(LINE)
print(f"{'method':38s} {'out rel err':>12s} {'cosine':>10s}")
rows_c = []
for name, Wh in [
    ("INT4 g128", grouped(W, 128, lambda a, axis: intq(a, 4, axis))),
    ("INT2 g128", grouped(W, 128, lambda a, axis: intq(a, 2, axis))),
    ("Ternary g128", grouped(W, 128, lambda a, axis: tern(a, axis))),
    ("Ternary per-tensor", tern(W, axis=None)),
    ("Binary g128", grouped(W, 128, lambda a, axis: binq(a, axis))),
]:
    Yh = X @ Wh.T
    rel = float(np.linalg.norm(Y - Yh) / np.linalg.norm(Y))
    cos = float((Y * Yh).sum() / (np.linalg.norm(Y) * np.linalg.norm(Yh)))
    print(f"{name:38s} {rel*100:11.2f}% {cos:10.4f}")
    rows_c.append({"method": name, "out_rel_err": round(rel, 5), "cosine": round(cos, 4)})
res["expC_output"] = rows_c


# ============================================================ ExpD 三条路径
print()
print(LINE)
print("ExpD  得到三元权重的三条路径（手写 STE，SGD 1200 步）")
print(LINE)
d_in, d_out = 64, 32
Xtr = rng.normal(0, 1, size=(4096, d_in))
Xte = rng.normal(0, 1, size=(1024, d_in))
W0 = rng.normal(0, 1 / np.sqrt(d_in), size=(d_out, d_in))
Ytr = Xtr @ W0.T + 0.05 * rng.normal(size=(4096, d_out))
Yte = Xte @ W0.T + 0.05 * rng.normal(size=(1024, d_out))


def q(Ws):
    """三元量化：per-output-feature absmean scale（BitNet 口径）"""
    g = np.abs(Ws).mean(axis=1, keepdims=True) + 1e-12
    return np.clip(np.round(Ws / g), -1, 1) * g


def step(Ws, in_loop, lr):
    """STE：前向用量化权重，反向 dW_hat/dW := 1"""
    Wf = q(Ws) if in_loop else Ws
    return Ws - lr * (((Xtr @ Wf.T - Ytr).T @ Xtr) / len(Xtr))


def mse(Wm):
    return float(np.mean(((Xte @ Wm.T) - Yte) ** 2))


STEPS, LR_FP, LR_STE = 1200, 0.05, 1e-3
Ws_fp = rng.normal(0, 1 / np.sqrt(d_in), size=(d_out, d_in)) * 0.1
for _ in range(STEPS):
    Ws_fp = step(Ws_fp, False, LR_FP)
A = mse(q(Ws_fp))

Ws_B = rng.normal(0, 1 / np.sqrt(d_in), size=(d_out, d_in)) * 0.1
Ws_C = Ws_fp.copy()
curve = {"A": [], "B": [], "C": []}
for t in range(STEPS):
    Ws_B = step(Ws_B, True, LR_STE)
    Ws_C = step(Ws_C, True, LR_STE)
    if t % 50 == 0 or t == STEPS - 1:
        curve["A"].append(A)
        curve["B"].append(mse(q(Ws_B)))
        curve["C"].append(mse(q(Ws_C)))
B, C = mse(q(Ws_B)), mse(q(Ws_C))
FP = mse(Ws_fp)

print(f"{'path':44s} {'test MSE':>12s}")
print(f"{'参考 全精度解（不量化，上界）':44s} {FP:12.5f}")
print(f"{'A 训完再量化 PTQ':44s} {A:12.5f}")
print(f"{'B 随机初始化 + 量化在环（原生低比特）':44s} {B:12.5f}")
print(f"{'C 全精度解 + 量化在环微调（QAT）':44s} {C:12.5f}")
print(f"\nC 相对 A 的改善: {(1-C/A)*100:.1f}%   A/C = {A/C:.2f}x   "
      f"距全精度上界仍有 {A/FP:.0f}x（A）/ {C/FP:.0f}x（C）")

# 无 STE 对照：round 的导数处处为 0 -> 权重永不更新
Ws_dead = rng.normal(0, 1 / np.sqrt(d_in), size=(d_out, d_in)) * 0.1
print(f"对照：无 STE（round 导数=0）1200 步后 test MSE = {mse(q(Ws_dead)):.5f}"
      f"（等于初始值，训练完全停滞）")
res["expD_paths"] = {
    "fp_upper_bound": round(FP, 6),
    "A_ptq": round(A, 6),
    "B_native_ste": round(B, 6),
    "C_qat_finetune": round(C, 6),
    "no_ste_baseline": round(mse(q(Ws_dead)), 6),
    "C_improve_vs_A_pct": round((1 - C / A) * 100, 2),
    "gap_to_fp_bound_A": round(A / FP, 1),
    "gap_to_fp_bound_C": round(C / FP, 1),
    "curve_step": 50,
    "curve": curve,
}

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(res, f, indent=2, ensure_ascii=False)


# ============================================================ 图
def plot_frontier():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ab, marker, color in [("INT4", "o", "tab:blue"), ("INT3", "s", "tab:cyan"),
                              ("INT2", "^", "tab:orange"), ("ternary", "D", "tab:red"),
                              ("binary", "v", "tab:green")]:
        xs = [r["bpw"] for r in rows_b if r["alphabet"] == ab]
        ys = [r["rel_err"] * 100 for r in rows_b if r["alphabet"] == ab]
        if xs:
            ax.plot(xs, ys, marker + "-", color=color, label=ab)
            for x, y, r in zip(xs, ys, [r for r in rows_b if r["alphabet"] == ab]):
                ax.annotate(r["method"].split()[-1], (x, y), textcoords="offset points",
                            xytext=(4, 3), fontsize=7, color=color)
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("true bits per weight")
    ax.set_ylabel("weight reconstruction rel. error (%)")
    ax.set_title("Alphabet x granularity: error vs true bit budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "alphabet_granularity_frontier.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_granularity_gain():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ab, color in [("INT4", "tab:blue"), ("INT2", "tab:orange"), ("ternary", "tab:red")]:
        pts = gain[ab]["curve"]
        keys = [None, 256, 128, 64, 32]
        xs = [float(pts[str(k)][0]) for k in keys]
        ys = [float(pts[str(k)][1]) * 100 for k in keys]
        ax.plot(range(len(keys)), ys, "o-", color=color, label=ab)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.annotate(f"{x:.2f}bpw", (i, y), textcoords="offset points",
                        xytext=(4, -9), fontsize=7, color=color)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["per-tensor", "g256", "g128", "g64", "g32"])
    ax.set_ylabel("weight reconstruction rel. error (%)")
    ax.set_title("Granularity payoff collapses as alphabet shrinks")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "granularity_diminishing_return.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_paths():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    xs = np.arange(len(curve["A"])) * 50
    ax.plot(xs, curve["A"], "o-", color="tab:blue", label="A: train FP -> quantize (PTQ)")
    ax.plot(xs, curve["B"], "s-", color="tab:orange", label="B: random init + STE in loop")
    ax.plot(xs, curve["C"], "^-", color="tab:green", label="C: FP init + STE finetune (QAT)")
    ax.axhline(FP, ls="--", color="gray", label=f"FP16 bound ({FP:.4f})")
    ax.set_xlabel("SGD step")
    ax.set_ylabel("test MSE")
    ax.set_title("Three routes to a ternary weight matrix")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    names = ["FP16\nbound", "A: PTQ", "B: native\nSTE", "C: QAT\nfinetune"]
    vals = [FP, A, B, C]
    colors = ["lightgray", "tab:blue", "tab:orange", "tab:green"]
    bars = ax.bar(names, vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("test MSE (log scale)")
    ax.set_title("Capacity gap dominates: ~100x from FP bound")
    ax.grid(alpha=0.3, axis="y")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.15, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, "three_routes_to_ternary.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


for f in (plot_frontier, plot_granularity_gain, plot_paths):
    print("\n[saved]", f())
print("\n[results] results/results.json")
