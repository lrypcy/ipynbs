#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run.py -- 《PTQ（07）：QuaRot 与 SpinQuant》配套实验。

纯 numpy + matplotlib，CPU 秒级。复现旋转（Hadamard）如何压平激活
异常通道，让低比特量化的"scale 被离群值绑架"问题消失。

  (A) Walsh-Hadamard 快速变换（迭代 butterfly），验证正交性
  (B) 离群通道压平：旋转前后逐通道 max|.| 分布对比
  (C) 端到端 INT4：直接量化 X、W vs 旋转后量化（在线旋转激活 +
      离线预旋转权重），比较输出相对误差

数学骨架：y = xW = (xH)(WH^T)，H 为正交对合矩阵（HH^T=I）。
把激活在线旋转、权重一次性预旋转，层数学等价，但动态范围被"打散"。

对应文章：https://lrypcy.github.io/2026/08/24/ptq-07-quarot-spinquant/
输出图到 results/（相对本文件）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 0
T, K, N = 512, 256, 128          # token / 输入通道(=Hadamard 维度) / 输出通道
BITS = 4
QMAX = 2 ** (BITS - 1) - 1        # 7
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)


# ----------------------------------------------------------------------
# Walsh-Hadamard 快速变换（迭代 butterfly），仅支持 2 的幂
# ----------------------------------------------------------------------
def fast_wht(x):
    """沿最后一维做归一化 Walsh-Hadamard 变换，H = H^T = H^{-1}。"""
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError("size must be power of 2")
    h = x.astype(np.float64).copy()
    step = 1
    while step < n:
        h = h.reshape(-1, step, 2, n // (step * 2))
        a = h[:, :, 0, :].copy()
        b = h[:, :, 1, :].copy()
        h[:, :, 0, :] = a + b
        h[:, :, 1, :] = a - b
        h = h.reshape(-1, n)
        step *= 2
    return (h / np.sqrt(n)).reshape(x.shape).astype(np.float64)


def sym_quant(x, axis, qmax=QMAX):
    """对称逐 row/col 量化（axis=1: 每 token 一个 scale；axis=0: 每列）。"""
    s = np.max(np.abs(x), axis=axis, keepdims=True) / qmax
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(x / s), -qmax, qmax) * s


def rel_fro(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    H = fast_wht(np.eye(K))                      # 显式矩阵（K=256 可承受）

    # ---- 合成含离群通道的激活与权重 ----
    X = rng.standard_normal((T, K))
    out_ch = rng.choice(K, size=8, replace=False)
    X[:, out_ch] *= 15.0                         # 8 个异常大通道
    W = rng.standard_normal((K, N))

    # ---- (A) 正交性自检 ----
    err_orth = float(np.max(np.abs(H @ H - np.eye(K))))
    print("Walsh-Hadamard: K=%d, max|HH^T - I| = %.2e" % (K, err_orth))

    # ---- (B) 逐通道动态范围压平 ----
    mx_before = np.max(np.abs(X), axis=0)
    Xt = X @ H                                   # 在线旋转（等价 xH）
    mx_after = np.max(np.abs(Xt), axis=0)
    flat_ratio = float(np.max(mx_before) / np.max(mx_after))

    # ---- (C) 端到端 INT4 对比 ----
    # 直接量化：X 逐 token、W 逐列
    Xq = sym_quant(X, axis=1)
    Wq = sym_quant(W, axis=0)
    Y_direct = Xq @ Wq
    err_direct = rel_fro(Y_direct, X @ W)

    # 旋转路线：â = xH 在线量化；W' = HW 预旋转量化；y = â W'^T
    # （(xH)(HW)^T = x HH^T W^T = xW，层数学完全等价）
    W_rot = H @ W                                # (K,K)x(K,N) 沿输入维预旋转
    at_q = sym_quant(Xt, axis=1)
    Wrot_q = sym_quant(W_rot, axis=0)
    Y_rot = at_q @ Wrot_q                        # 无需转回：旋转已融进权重
    err_rot = rel_fro(Y_rot, X @ W)

    res = {
        "orth_err": err_orth,
        "chan_max_before_top5": [float(v) for v in np.sort(mx_before)[-5:]],
        "chan_max_after_max": float(np.max(mx_after)),
        "dynamic_range_shrink_x": flat_ratio,
        "int4_relerr_direct": err_direct,
        "int4_relerr_rotated": err_rot,
        "error_reduction_pct": float(100 * (1 - err_rot / err_direct)),
    }

    # ---- 图 ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    idx = np.argsort(mx_before)[::-1]
    ax[0].semilogy(mx_before[idx], "o-", ms=3, color="#C44E52",
                   label="before H (has outliers)")
    ax[0].semilogy(mx_after[idx], "o-", ms=3, color="#4C72B0",
                   label="after H (flattened)")
    ax[0].set_xlabel("channel (sorted by pre-rotation max)")
    ax[0].set_ylabel("per-channel max|x|")
    ax[0].set_title("Hadamard flattens channel dynamic range\n(max shrunk %.0fx)" % flat_ratio)
    ax[0].legend(fontsize=9)

    bars = ax[1].bar(["INT4\ndirect", "INT4\nHadamard-rotated"],
                     [err_direct, err_rot],
                     color=["#C44E52", "#4C72B0"], width=0.55)
    for b, v in zip(bars, [err_direct, err_rot]):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.05, f"{v:.3f}",
                   ha="center", fontsize=10)
    ax[1].set_ylabel("output rel. error  ||dy||/||y||")
    ax[1].set_title("End-to-end INT4 GEMM error\n(reduction %.1f%%)"
                    % res["error_reduction_pct"])
    ax[1].set_ylim(0, err_direct * 1.25)
    fig.tight_layout()
    p1 = os.path.join(RES, "hadamard_flatten_and_int4_error.png")
    fig.savefig(p1, dpi=130)

    # ---- 打印 ----
    print("旋转前 top-5 通道 max|x| :", np.round(res["chan_max_before_top5"], 1))
    print("旋转后全体通道最大 max|x| : %.1f  （动态范围收缩 %.0fx）"
          % (res["chan_max_after_max"], flat_ratio))
    print("-" * 60)
    print("INT4 直接量化   输出相对误差 : %.4f" % err_direct)
    print("INT4 Hadamard 旋转          : %.4f  （误差降低 %.1f%%）"
          % (err_rot, res["error_reduction_pct"]))
    print(f"[save] {p1}")

    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
