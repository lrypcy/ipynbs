#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- PTQ 系列《11: RPTQ / QUIK / ATOM》配套实验。

移植自博客仓库 experiments_rptq_quik_atom.py，算法逻辑与超参不变
（SEED=0 与源脚本一致）。纯 numpy + matplotlib，CPU 秒级跑完。
复现场景：正常通道 N(0,1)、8 个离群通道随机散布的激活，权重统一
RTN per-channel W4（共同误差地板），对比三种激活处理：
  变体 0  naive per-group A4
  变体 1  RPTQ —— 按校准 absmax 排序重排后 per-group A4
  变体 2  ATOM —— 双路混合精度（离群列 int8 + 正常通道 int4 组量化）
附：置换等价性验证（重排在数学上是免费的）。

输出图到本目录 results/（相对本文件，从任意 cwd 运行均可）：
  rptq_group_absmax_reorder.png   重排前后各组 absmax 上界对比
  variant_errors_vs_floor.png     三变体相对误差 vs W4 噪声地板
另将全部关键数字写入 results/results.json。
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")   # 输出到本目录 results/
os.makedirs(OUT, exist_ok=True)

SEED = 0


# ---------- 公共构件 ----------
def sym_quant(A, bits):
    """对称 per-tensor 量化"""
    if A.size == 0:
        return A
    s = np.abs(A).max() / (2 ** (bits - 1) - 1)
    return np.round(A / s) * s


def group_act_quant(X, bits, g):
    """逐组(per-token×per-group-of-channels)对称量化激活"""
    T, d = X.shape
    Xq = np.empty_like(X)
    for j0 in range(0, d, g):
        blk = X[:, j0:j0 + g]
        s = np.abs(blk).max() / (2 ** (bits - 1) - 1)
        Xq[:, j0:j0 + g] = np.round(blk / s) * s
    return Xq


def rel_err(Y, Y_ref):
    return float(np.linalg.norm(Y - Y_ref) / np.linalg.norm(Y_ref))


def main():
    rng = np.random.default_rng(SEED)   # 与源脚本一致：default_rng(0)

    # ---------- 场景：正常通道 N(0,1)，8 个离群通道随机散布 ----------
    T, d, n_outlier, d_out, g = 2048, 128, 8, 256, 32
    X = rng.normal(0, 1.0, (T, d))
    oc = sorted(rng.choice(d, n_outlier, replace=False).tolist())
    mus = rng.uniform(20.0, 40.0, n_outlier)
    for k, j in enumerate(oc):
        X[:, j] = mus[k] + 0.5 * rng.normal(0, 1.0, T)     # 大均值+小波动
    W = rng.normal(0, 0.02, (d_out, d))
    Y_ref = X @ W.T

    # 权重统一 RTN per-channel 4-bit（所有变体相同：既公平、又构成共同的误差地板）
    Wq = np.stack([sym_quant(W[i], 4) for i in range(d_out)])
    e_floor = rel_err(X @ Wq.T, Y_ref)
    print(f"[地板] 仅权重量化噪声（激活 fp16）：相对误差 = {e_floor:.4f}")

    # ---------- 变体 0：naive per-group A4 ----------
    e0 = rel_err(group_act_quant(X, 4, g) @ Wq.T, Y_ref)

    # ---------- 变体 1：RPTQ —— 按校准 absmax 排序重排（原论文用 (min,max) k-means） ----------
    absmax_c = np.abs(X).max(0)
    order = np.argsort(absmax_c)                 # 从小到大 → 离群通道聚到尾端
    Xr, Wrq = X[:, order], Wq[:, order]          # 权重同序
    Xq1 = group_act_quant(Xr, 4, g)
    bounds_after = [round(float(np.abs(Xr[:, j0:j0+g]).max()), 2)
                    for j0 in range(0, d, g)]
    bounds_before = [round(float(np.abs(X[:, j0:j0+g]).max()), 2)
                     for j0 in range(0, d, g)]   # 仅作图用诊断量：重排前的组上界
    print("[RPTQ] 重排后各组的通道 absmax 上界:", bounds_after)
    e1 = rel_err(Xq1 @ Wrq.T, Y_ref)

    # ---------- 变体 2：ATOM —— 双路混合精度（离群列 int8 + 正常通道干净的 int4 组量化） ----------
    normal_mask = np.ones(d, dtype=bool); normal_mask[oc] = False
    Xn_q = group_act_quant(X[:, normal_mask], 4, g)
    Xo_q = sym_quant(X[:, oc], 8)
    e2 = rel_err(Xn_q @ Wq[:, normal_mask].T + Xo_q @ Wq[:, oc].T, Y_ref)

    print("[对照] 层输出相对误差（同一块 W4 权重噪声地板之上，只改变激活处理）:")
    variants = [("naive per-group A4         ", e0),
                ("RPTQ 重排 + per-group A4   ", e1),
                ("ATOM 双路(离群int8+主int4) ", e2)]
    for name, e in variants:
        print(f"  {name}: {e:.4f}   （为地板的 {e/e_floor:.2f}x）")

    # ---------- 附：置换等价性验证（重排在数学上是免费的） ----------
    P = np.eye(d)[order]
    lhs = X @ W.T
    rhs = (X @ P) @ (W @ P).T
    perm_err = float(np.abs(rhs - lhs).max())
    print(f"[置换等价性] ||XP(WP)^T − XW^T||_max = {perm_err:.2e}")

    # ============================================================
    # 图 A：重排前后各组的 absmax 上界（RPTQ 把离群值聚到尾端组）
    # ============================================================
    n_groups = d // g
    xs = np.arange(n_groups)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.bar(xs - 0.2, bounds_before, width=0.4, color="#C44E52",
           label="before reorder (natural layout)")
    ax.bar(xs + 0.2, bounds_after, width=0.4, color="#4C72B0",
           label="after RPTQ reorder (sorted by absmax)")
    ax.set_yscale("log")
    ax.set_xlabel(f"activation group index (g={g} channels each)")
    ax.set_ylabel("per-group absmax upper bound (log)")
    ax.set_title("RPTQ: sorting channels by absmax isolates outliers into tail groups")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    pa = os.path.join(OUT, "rptq_group_absmax_reorder.png")
    fig.savefig(pa, dpi=130)
    plt.close(fig)
    print(f"\n[save] {pa}")

    # ============================================================
    # 图 B：三变体相对误差 vs W4 地板
    # ============================================================
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    names = ["W4 floor\n(act fp16)", "naive\nper-group A4", "RPTQ reorder\n+ A4",
             "ATOM dual-path\n(outlier int8 + int4)"]
    vals = [e_floor, e0, e1, e2]
    bars = ax.bar(names, vals, color=["#999999", "#C44E52", "#DD8452", "#55A868"])
    for b, v, base in zip(bars, vals, [e_floor] * 4):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.03,
                f"{v:.4f}\n({v/e_floor:.2f}x floor)", ha="center", fontsize=8.5)
    ax.set_ylabel("layer output rel error")
    ax.set_title("Same W4 weight-noise floor, different activation handling")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    pb = os.path.join(OUT, "variant_errors_vs_floor.png")
    fig.savefig(pb, dpi=130)
    plt.close(fig)
    print(f"[save] {pb}")

    # ---- 全部关键数字落盘 results.json ----
    payload = {
        "config": {"SEED": SEED, "T": T, "d_in": d, "d_out": d_out,
                   "n_outlier": n_outlier, "group": g},
        "w4_weight_only_floor_relerr": round(e_floor, 4),
        "variants_relerr": {"naive_per_group_a4": round(e0, 4),
                            "rptq_reorder_a4": round(e1, 4),
                            "atom_dualpath": round(e2, 4)},
        "variants_over_floor_x": {"naive_per_group_a4": round(e0 / e_floor, 2),
                                  "rptq_reorder_a4": round(e1 / e_floor, 2),
                                  "atom_dualpath": round(e2 / e_floor, 2)},
        "rptq_group_absmax_bounds": {"before_reorder": bounds_before,
                                     "after_reorder": bounds_after},
        "permutation_equivalence_max_abs_dev": perm_err,
    }
    pj = os.path.join(OUT, "results.json")
    with open(pj, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[save] {pj}")


if __name__ == "__main__":
    main()
