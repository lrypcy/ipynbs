# -*- coding: utf-8 -*-
"""
03 篇配套实验：AWQ 激活感知缩放搜索的最小可复现研究。

三个 Demo（与文章章节一一对应）：
  A. 显著性度量对比 —— 权重幅度 vs 激活幅度 vs 乘积指标，保护 top-1% 通道的误差回收率 (对应文章 §3)
  B. γ 缩放搜索     —— 输出 MSE 随缩放指数 γ 的 U 形曲线，及收益对量化粒度的依赖      (对应文章 §5)
  C. clip 搜索目标函数 —— 同一套网格搜索、两种目标函数（权重 MSE vs 激活加权 MSE）的对齐性 + 六配置最终对比 (对应文章 §6)

约定（沿用 00 篇符号字典；本篇新增缩放向量 kappa 与指数 gamma，
alpha 沿用 00 篇的裁剪比例专用记号，论文原文的 alpha 在本系列改记 gamma）：
  等效变换   W' = W · diag(kappa)，x' = x / kappa（W4A16 下激活侧不执行）
  反量化恢复 What = Q_g(W · diag(kappa)) / diag(kappa)
  对称网格   qmin=-2^(b-1), qmax=2^(b-1)-1；分组沿输入通道轴，组内共享 scale

运行：python run.py   （依赖 numpy + matplotlib，CPU 数十秒内完成）
输出：results/ 下 3 张 PNG + results.json + stdout 的 markdown 表格
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

QMIN = lambda b: -(1 << (b - 1))          # -2^{b-1}
QMAX = lambda b: (1 << (b - 1)) - 1       #  2^{b-1}-1

BITS = 4
GROUP = 128


# ----------------------------------------------------------------------
# 数据合成：权重各通道同分布（刻意不注入权重 outlier）；
# 激活幅度用混合模型模拟 LLM 的 emergent outlier：
#   约 0.8% 的通道尺度放大 12 倍（LLM.int8() 观察到的系统性大激活通道），
#   其余通道 log-uniform 温和差异。这样"显著通道"只能由激活定义——
#   正是 AWQ 论文 Figure 1 观察的前提。
# ----------------------------------------------------------------------
def make_data(rng, c_out=512, c_in=1024, t_cal=2048, t_test=2048,
              n_emerge=8, emerge_boost=12.0):
    W = rng.normal(0, 0.02, (c_out, c_in))
    sigma = np.exp(rng.uniform(np.log(0.4), np.log(1.8), size=c_in))
    idx_emerge = rng.choice(c_in, n_emerge, replace=False)
    sigma[idx_emerge] *= emerge_boost
    x_cal = rng.normal(0, 1.0, (t_cal, c_in)) * sigma
    x_test = rng.normal(0, 1.0, (t_test, c_in)) * sigma
    return W, sigma, x_cal, x_test, idx_emerge


# ----------------------------------------------------------------------
# 核心算子：分组对称 RTN（可选逐组裁剪比例 alpha）
# 正文 §2 的代码化：What = Q_g(W)；s_group = alpha * max|seg| / qmax
# ----------------------------------------------------------------------
def group_rtn(W, group=GROUP, bits=BITS, alphas=None, col_weight=None):
    """alphas=None 时按 max-scale（alpha=1）；否则逐组在 alphas 网格上搜索最优裁剪。

    目标函数由 col_weight 决定（正文 §6 的核心变量）:
      None          -> 组内权重 MSE          E[(Q(w)-w)^2]
      形状 (c_in,)   -> 激活加权 MSE           E[E[x_j^2]·(Q(w)-w)^2]  （AWQ auto_clip 的高效代理）
    返回 (反量化权重, 每组选中的 alpha 数组)。"""
    c_out, c_in = W.shape
    qmax = QMAX(bits)
    What = np.empty_like(W)
    chosen = []
    for g0 in range(0, c_in, group):
        seg = W[:, g0:g0 + group]
        cw = None if col_weight is None else col_weight[g0:g0 + group][None, :]
        gmax = np.max(np.abs(seg), axis=1, keepdims=True)          # (m,1)
        if alphas is None:
            a = np.ones(1)
            best_s = gmax / qmax
        else:
            losses = []
            for a in alphas:
                s = a * gmax / qmax
                q = np.clip(np.round(seg / s), QMIN(bits), qmax)
                err2 = (q * s - seg) ** 2
                losses.append(np.mean(err2 * cw) if cw is not None else np.mean(err2))
            k = int(np.argmin(losses))
            a = np.array([alphas[k]])
            best_s = a[0] * gmax / qmax
        q = np.clip(np.round(seg / best_s), QMIN(bits), qmax)
        What[:, g0:g0 + group] = q * best_s
        chosen.append(float(a[0]))
    return What, np.asarray(chosen)


def out_mse(W, What, x_test):
    """层输出 MSE：E||x W^T - x What^T||^2（逐元素平均）。"""
    return float(np.mean((x_test @ What.T - x_test @ W.T) ** 2))


def weight_snr_db(W, What):
    return float(10 * np.log10(np.var(W) / np.mean((W - What) ** 2)))


def awq_scale(act_max, gamma):
    """论文搜索族 kappa_j = a_j^gamma，再做几何均值归一化（只做通道间再分配，不引入整体膨胀）。"""
    s = np.power(np.where(act_max == 0, 1.0, act_max), gamma)
    return s / np.exp(np.mean(np.log(s)))


def apply_awq(W, act_max, gamma, group=GROUP, bits=BITS, clip_alphas=None,
              clip_col_weight=None):
    """AWQ 完整管线：等效变换 -> 分组（可选裁剪，目标可选）量化 -> 除回缩放。"""
    kappa = awq_scale(act_max, gamma)
    Ws = W * kappa[None, :]
    # 关键细节：反量化除回 kappa_j 后，缩放坐标系里的误差 eps_tilde 折算到
    # 原坐标要除以 kappa_j^2，因此激活加权目标在缩放后坐标系里
    # 列权应改为 E[x_j^2] / kappa_j^2（正文 §3.4 链式法则推导）。
    cw = None if clip_col_weight is None else clip_col_weight / kappa ** 2
    Wq, a_star = group_rtn(Ws, group, bits, clip_alphas, cw)
    return Wq / kappa[None, :], a_star


def per_input_channel_rtn(W, bits=BITS):
    """逐输入通道量化（每个输入通道独立 scale）：AWQ 缩放的理论失效场景。"""
    amax = np.max(np.abs(W), axis=0, keepdims=True)
    s = amax / QMAX(bits)
    return np.clip(np.round(W / s), QMIN(bits), QMAX(bits)) * s


# ----------------------------------------------------------------------
# Demo A：显著性度量对比（保护 top-1% 通道实验）
# ----------------------------------------------------------------------
def demo_a(W, x_cal, x_test, rng, k_frac=0.01):
    c_in = W.shape[1]
    k = max(1, int(round(c_in * k_frac)))                     # 约 1% 通道
    What_rtn, _ = group_rtn(W)
    eps = W - What_rtn                                        # 逐元素量化误差
    mse_base = out_mse(W, What_rtn, x_test)

    # 逐列误差贡献：E[x_j^2] * E[eps_ij^2]（正文 §3 的分解式代码化）
    act_energy = np.mean(x_cal ** 2, axis=0)
    contrib = act_energy * np.mean(eps ** 2, axis=0)

    act_stat = np.max(np.abs(x_cal), axis=0)                  # c_j = max_t |x_tj|
    w_colmax = np.max(np.abs(W), axis=0)
    prod_score = act_stat * w_colmax                          # 乘积指标 |c_j|·max_i|w_ij|

    def protected_mse(idx):
        What = What_rtn.copy()
        What[:, idx] = W[:, idx]                              # 显著列保持 FP16
        return out_mse(W, What, x_test)

    idx_w = np.argsort(w_colmax)[::-1][:k]
    idx_a = np.argsort(act_stat)[::-1][:k]
    idx_p = np.argsort(prod_score)[::-1][:k]
    rand_shares = [protected_mse(rng.choice(c_in, k, replace=False))
                   for _ in range(50)]

    # top-1% 误差能量集中度：按真实贡献排序 vs 按激活统计排序
    order_true = np.sort(contrib)[::-1]
    share_true = order_true[:k].sum() / contrib.sum()
    share_by_act = contrib[idx_a].sum() / contrib.sum()

    res = {
        "k": k,
        "top1pct_error_share_true": round(float(share_true), 4),
        "top1pct_error_share_by_activation": round(float(share_by_act), 4),
        "remaining_ratio": {
            "no_protect": 1.0,
            "random": round(float(np.mean(rand_shares) / mse_base), 4),
            "weight_mag": round(float(protected_mse(idx_w) / mse_base), 4),
            "act_mag": round(float(protected_mse(idx_a) / mse_base), 4),
            "product": round(float(protected_mse(idx_p) / mse_base), 4),
        },
        "overlap_act_weight_top": int(len(set(idx_a.tolist()) & set(idx_w.tolist()))),
    }

    labels = ["no protect", "random", "by weight mag", "by act mag", "by product"]
    keys = ["no_protect", "random", "weight_mag", "act_mag", "product"]
    vals = [res["remaining_ratio"][kk] * 100 for kk in keys]

    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150)
    bars = ax.bar(labels, vals,
                  color=["#999999", "#BBBBBB", "#C44E52", "#4C72B0", "#55A868"])
    ax.axhline(100, color="gray", ls="--", lw=1)
    ax.set_ylabel("remaining output MSE (% of RTN baseline)")
    ax.set_title(f"Protecting top-{k} (~1%) input channels, INT4 group={GROUP}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "awq_saliency_selection_comparison.png"))
    plt.close(fig)
    return res


# ----------------------------------------------------------------------
# Demo B：γ 缩放搜索曲线与粒度依赖
# ----------------------------------------------------------------------
def demo_b(W, x_cal, x_test):
    gammas = np.linspace(0.0, 1.0, 21)
    act_max = np.max(np.abs(x_cal), axis=0)
    What_rtn, _ = group_rtn(W)
    mse_base = out_mse(W, What_rtn, x_test)

    configs = [("group=256", 256), ("group=128", 128),
               ("group=64", 64), ("group=32", 32)]
    curves = {}
    summary = {}
    for name, g in configs:
        curve = []
        for gamma in gammas:
            What, _ = apply_awq(W, act_max, gamma, group=g)
            curve.append(out_mse(W, What, x_test) / mse_base)
        curves[name] = np.asarray(curve)
        i = int(np.argmin(curve))
        summary[name] = {"gamma_star": round(float(gammas[i]), 2),
                         "best_ratio": round(float(curve[i]), 4),
                         "gain_x": round(float(1.0 / curve[i]), 3)}

    # 理论失效场景：逐输入通道粒度下缩放应严格无效（曲线水平）。
    # 注意该配置的绝对水位由其自身粒度决定（每列 512 个元素共享一个 scale，
    # 比 group=128 更粗），平直性才是本图要验证的性质。
    pc_curve = []
    for gamma in gammas:
        s = awq_scale(act_max, gamma)
        What = per_input_channel_rtn(W * s[None, :]) / s[None, :]
        pc_curve.append(out_mse(W, What, x_test) / mse_base)
    pc_curve = np.asarray(pc_curve)
    curves["per-input-channel"] = pc_curve
    summary["per-input-channel"] = {
        "gamma_star": None,
        "best_ratio": round(float(pc_curve.min()), 6),
        "gain_x": round(float(1.0 / float(pc_curve.mean())), 6),
        "curve_std": round(float(pc_curve.std()), 8),
        "note": "flat (scaling is identity at same-axis granularity)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150)
    colors = {"group=256": "#8C8C8C", "group=128": "#4C72B0",
              "group=64": "#DD8452", "group=32": "#55A868"}
    for name, g in configs:
        axes[0].plot(gammas, curves[name], label=name, color=colors[name])
    axes[0].plot(gammas, pc_curve, "--", color="black", lw=1.2,
                 label="per-input-channel (theory: flat)")
    i128 = int(np.argmin(curves["group=128"]))
    axes[0].scatter([gammas[i128]], [curves["group=128"][i128]],
                    color="black", s=28, zorder=5)
    axes[0].annotate(f"$\\gamma^*$={gammas[i128]:.2f}",
                     (gammas[i128], curves["group=128"][i128]),
                     textcoords="offset points", xytext=(6, 8), fontsize=9)
    axes[0].set_xlabel(r"scaling exponent $\gamma$  ($s_j=c_j^{\,\gamma}$)")
    axes[0].set_ylabel("output MSE / RTN baseline")
    axes[0].set_title("AWQ scale search curve (INT4)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    gs = [256, 128, 64, 32]
    gains = [summary[f"group={g}"]["best_ratio"] for g in gs]
    axes[1].plot(gs, gains, "o-", color="#4C72B0", label="group-wise")
    axes[1].axhline(1.0, color="black", ls="--", lw=1.2,
                    label="per-input-channel (=1, no effect)")
    for g, v in zip(gs, gains):
        axes[1].annotate(f"{v:.3f}", (g, v), textcoords="offset points",
                         xytext=(4, -12), fontsize=8)
    axes[1].set_xlabel("group size g")
    axes[1].set_ylabel("best MSE / RTN baseline")
    axes[1].set_title("Scaling benefit vs granularity")
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "awq_gamma_scale_search.png"))
    plt.close(fig)
    return {"gammas": [round(float(g), 2) for g in gammas],
            "curves": {k: [round(float(v), 4) for v in vv] for k, vv in curves.items()},
            "summary": summary}


# ----------------------------------------------------------------------
# Demo C：clip 搜索的目标函数（同一网格搜索，不同目标）+ 六配置最终对比
# ----------------------------------------------------------------------
def demo_c(W, x_cal, x_test, gamma_fixed):
    alphas = np.linspace(0.3, 1.0, 29)
    What_rtn, _ = group_rtn(W)
    mse_base = out_mse(W, What_rtn, x_test)
    act_max = np.max(np.abs(x_cal), axis=0)
    act_energy = np.mean(x_cal ** 2, axis=0)           # 列权 E[x_j^2]（校准集口径，搜索时可见）

    # 两条逐组归一化目标函数曲线：权重 MSE vs 激活加权 MSE（观察谷底位置差异）
    seg_all, gmax_all = [], []
    for g0 in range(0, W.shape[1], GROUP):
        seg = W[:, g0:g0 + GROUP]
        seg_all.append(seg)
        gmax_all.append(np.max(np.abs(seg), axis=1, keepdims=True))
    norm_curves = {"weight_mse": [], "act_weighted": []}
    for a in alphas:
        tot = {"weight_mse": 0.0, "act_weighted": 0.0}
        tot_full = {"weight_mse": 0.0, "act_weighted": 0.0}
        for gi, (seg, gm) in enumerate(zip(seg_all, gmax_all)):
            cw = act_energy[gi * GROUP:(gi + 1) * GROUP][None, :]
            s_full = gm / QMAX(BITS)
            s = a * gm / QMAX(BITS)
            q = np.clip(np.round(seg / s), QMIN(BITS), QMAX(BITS))
            qf = np.clip(np.round(seg / s_full), QMIN(BITS), QMAX(BITS))
            e, ef = (q * s - seg) ** 2, (qf * s_full - seg) ** 2
            tot["weight_mse"] += float(np.mean(e))
            tot["act_weighted"] += float(np.mean(e * cw))
            tot_full["weight_mse"] += float(np.mean(ef))
            tot_full["act_weighted"] += float(np.mean(ef * cw))
        for k in tot:
            norm_curves[k].append(tot[k] / tot_full[k])
    norm_curves = {k: np.asarray(v) for k, v in norm_curves.items()}
    i_star = {k: int(np.argmin(v)) for k, v in norm_curves.items()}

    # 六种配置：目标函数的对齐性是唯一变量
    What_clip_w, a_only_w = group_rtn(W, GROUP, BITS, alphas)                 # 裁剪，目标=权重MSE
    What_clip_a, a_only_a = group_rtn(W, GROUP, BITS, alphas, act_energy)     # 裁剪，目标=激活加权
    What_awq, _ = apply_awq(W, act_max, gamma_fixed, GROUP, BITS, None)       # 只缩放
    What_full_w, a_full_w = apply_awq(W, act_max, gamma_fixed, GROUP, BITS, alphas)
    What_full_a, a_full_a = apply_awq(W, act_max, gamma_fixed, GROUP, BITS, alphas,
                                      clip_col_weight=act_energy)

    def report(name, What, extra=""):
        return {"config": name,
                "weight_snr_db": round(weight_snr_db(W, What), 2),
                "rel_output_err_pct": round(
                    float(np.linalg.norm(x_test @ What.T - x_test @ W.T)
                          / np.linalg.norm(x_test @ W.T) * 100), 3),
                "out_mse_ratio": round(out_mse(W, What, x_test) / mse_base, 4),
                "note": extra}

    rows = [
        report("RTN (max-scale)", What_rtn, "baseline"),
        report("clip only (obj=wMSE)", What_clip_w,
               f"mean alpha*={a_only_w.mean():.2f}"),
        report("clip only (obj=act-wMSE)", What_clip_a,
               f"mean alpha*={a_only_a.mean():.2f}"),
        report(f"scale only (gamma={gamma_fixed:.2f})", What_awq, "no clip"),
        report("scale+clip (obj=wMSE)", What_full_w,
               f"mean alpha*={a_full_w.mean():.2f}"),
        report("scale+clip (obj=act-wMSE)", What_full_a,
               f"mean alpha*={a_full_a.mean():.2f}"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), dpi=150)
    axes[0].plot(alphas, norm_curves["weight_mse"], color="#DD8452",
                 label="objective: weight MSE")
    axes[0].plot(alphas, norm_curves["act_weighted"], color="#4C72B0",
                 label="objective: act-weighted MSE")
    axes[0].axvline(alphas[i_star["weight_mse"]], color="#DD8452", ls="--", lw=1)
    axes[0].axvline(alphas[i_star["act_weighted"]], color="#4C72B0", ls="--", lw=1)
    axes[0].annotate(rf"$\alpha^*$={alphas[i_star['weight_mse']]:.2f}",
                     (alphas[i_star["weight_mse"]], norm_curves["weight_mse"][i_star["weight_mse"]]),
                     textcoords="offset points", xytext=(6, 12), fontsize=9, color="#DD8452")
    axes[0].annotate(rf"$\alpha^*$={alphas[i_star['act_weighted']]:.2f}",
                     (alphas[i_star["act_weighted"]], norm_curves["act_weighted"][i_star["act_weighted"]]),
                     textcoords="offset points", xytext=(-58, 12), fontsize=9, color="#4C72B0")
    axes[0].set_xlabel(r"group clipping ratio $\alpha$  ($s=\alpha\cdot\max|w|/q_{\max}$)")
    axes[0].set_ylabel("group objective / no-clip value")
    axes[0].set_title("Same grid search, two objectives")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)

    names = [r["config"] for r in rows]
    xs = np.arange(len(names))
    axes[1].bar(xs, [r["out_mse_ratio"] for r in rows],
                color=["#999999", "#DD8452", "#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(names, rotation=14, ha="right", fontsize=7.5)
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set_ylabel("output MSE / RTN baseline")
    axes[1].set_title("INT4 pipeline comparison (evaluated on true output MSE)")
    for x, r in zip(xs, rows):
        axes[1].text(x, r["out_mse_ratio"] + 0.008, f"{r['out_mse_ratio']:.3f}",
                     ha="center", fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "awq_clip_and_pipeline_comparison.png"))
    plt.close(fig)

    return {"alphas": [round(float(a), 3) for a in alphas],
            "norm_curves": {k: [round(float(v), 5) for v in vv]
                            for k, vv in norm_curves.items()},
            "alpha_star": {k: round(float(alphas[v]), 3) for k, v in i_star.items()},
            "rows": rows}


def main():
    rng = np.random.default_rng(0)
    W, sigma, x_cal, x_test, idx_emerge = make_data(rng)
    meta = {
        "shape": list(W.shape),
        "sigma_range": [round(float(sigma.min()), 3), round(float(sigma.max()), 3)],
        "dynamic_range_x": round(float(sigma.max() / sigma.min()), 1),
        "n_emerge_channels": int(len(idx_emerge)),
        "bits": BITS, "group": GROUP,
        "t_cal": x_cal.shape[0], "t_test": x_test.shape[0],
    }
    print(f"W {meta['shape']}, act sigma in [{meta['sigma_range'][0]}, "
          f"{meta['sigma_range'][1]}] (range {meta['dynamic_range_x']}x), "
          f"emergent channels={meta['n_emerge_channels']}, "
          f"INT{BITS}, group={GROUP}\n")

    print("Running Demo A: saliency metric comparison ...")
    ra = demo_a(W, x_cal, x_test, rng)
    print("Running Demo B: gamma scale search ...")
    rb = demo_b(W, x_cal, x_test)
    gamma_fixed = rb["summary"]["group=128"]["gamma_star"]
    print(f"Running Demo C: clip search + pipeline (gamma={gamma_fixed}) ...")
    rc = demo_c(W, x_cal, x_test, gamma_fixed)

    payload = {"meta": meta, "demo_a": ra, "demo_b": rb, "demo_c": rc}
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\n## Demo A: protect top-1% channels (remaining output MSE / RTN)\n")
    print("| selector | remaining ratio |")
    print("|---|---|")
    for kk in ["no_protect", "random", "weight_mag", "act_mag", "product"]:
        v = ra["remaining_ratio"][kk]
        print(f"| {kk} | {v:.3f} |")
    print(f"\ntop-1% true error-energy share = "
          f"{ra['top1pct_error_share_true']:.1%}; "
          f"share captured by activation ranking = "
          f"{ra['top1pct_error_share_by_activation']:.1%}; "
          f"|act∩weight| in top-{ra['k']} = {ra['overlap_act_weight_top']}")

    print("\n## Demo B: gamma search summary\n")
    print("| granularity | gamma* | best MSE ratio | gain (1/ratio) |")
    print("|---|---|---|---|")
    for name, sv in rb["summary"].items():
        gs = sv["gamma_star"] if sv["gamma_star"] is not None else "-"
        print(f"| {name} | {gs} | {sv['best_ratio']} | {sv['gain_x']} |")

    print("\n## Demo C: clip objective shapes & pipeline comparison\n")
    print(f"objective valley: weight-MSE alpha*={rc['alpha_star']['weight_mse']}, "
          f"act-weighted alpha*={rc['alpha_star']['act_weighted']}")
    print("| config | weight SNR (dB) | rel output err (%) | MSE ratio | note |")
    print("|---|---|---|---|---|")
    for r in rc["rows"]:
        print(f"| {r['config']} | {r['weight_snr_db']} | {r['rel_output_err_pct']} "
              f"| {r['out_mse_ratio']} | {r['note']} |")

    print("\nAll outputs saved to", OUT)


if __name__ == "__main__":
    main()
