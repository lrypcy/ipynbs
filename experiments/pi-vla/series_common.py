# -*- coding: utf-8 -*-
"""π 系列 06–10 共用的合成实验内核。

只放三件跨篇复用的东西：断言框架、高斯条件采样（对应 flow matching 的
inpainting / guidance 在可解析情形下的闭式解）、软掩码与统计量。
每篇的 `lab.py` 自含自己世界的定义，读 README 即可复现。
"""
from __future__ import annotations

import numpy as np


class Check:
    """固定格式的断言收集器；最后一个 cell 打出总数，好和 README / 正文对账。"""

    def __init__(self):
        self.items: list[str] = []

    def banner(self, title: str) -> None:
        line = "=" * 74
        print(f"\n{line}\n{title}\n{line}")

    def ok(self, name: str, cond: bool, detail: str = "") -> None:
        tag = "[ok]" if bool(cond) else "[FAIL]"
        self.items.append(f"{tag} {name}" + (f"：{detail}" if detail else ""))
        print("  " + f"{tag} {name}" + (f"（{detail}）" if detail else ""))

    def summary(self) -> None:
        n_ok = sum(1 for a in self.items if a.startswith("[ok]"))
        self.banner("汇总")
        for a in self.items:
            print("  " + a)
        print(f"\n  断言总数：{len(self.items)}（通过 {n_ok}）")
        if n_ok != len(self.items):
            raise SystemExit("有断言未通过")


def soft_mask(H: int, s: int, d: int, kind: str = "exp") -> np.ndarray:
    """RTC Eq.(5) 的软掩码；kind ∈ {exp, linear, hard}。

    W_i = 1 (i<d)；中间按 c_i·(e^{c_i}−1)/(e−1) 指数衰减；W_i = 0 (i≥H−s)。
    """
    W = np.zeros(H)
    W[:d] = 1.0
    if kind == "hard":
        return W
    for i in range(d, H - s):
        c = (H - s - i) / (H - s - d + 1)
        W[i] = c * (np.exp(c) - 1.0) / (np.e - 1.0) if kind == "exp" else c
    return W


def ar_kernel(H: int, rho: float = 0.85, sigma: float = 0.30) -> np.ndarray:
    """时间相关的高斯协方差：相邻动作越近越相关，保证 chunk 内平滑。"""
    lag = np.abs(np.subtract.outer(np.arange(H), np.arange(H)))
    return sigma ** 2 * rho ** lag


def cond_sample(mu: np.ndarray, Sigma: np.ndarray, idx, y, w, tau2: float, rng):
    """软约束下的高斯后验采样。

    观测 y_i ~ N(A_i, tau2/w_i)：w_i 越大约束越强，w_i=0 等于没有观测。
    这是 ΠGDM guidance 在「线性高斯流」里的闭式对应物：
    后验精度 = 先验精度 + diag(w)/tau2，后验均值按同一权重吸收观测。
    返回 (样本, 后验均值, 后验协方差)。
    """
    n = len(mu)
    idx = np.asarray(idx, dtype=int)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    eye = 1e-9 * np.eye(n)
    Sinv = np.linalg.inv(Sigma + eye)
    D = np.zeros((n, n))
    D[idx, idx] = w / tau2
    Spost = np.linalg.inv(Sinv + D + eye)
    b = Sinv @ mu
    b[idx] += w * y / tau2
    mpost = Spost @ b
    return rng.multivariate_normal(mpost, Spost, method="cholesky"), mpost, Spost


def jerk(a: np.ndarray) -> float:
    """二阶离散差分的最大绝对值（论文 A.2 用最大加速度作 OOD 代理）。"""
    a = np.asarray(a, dtype=float)
    if len(a) < 3:
        return 0.0
    return float(np.max(np.abs(a[2:] - 2.0 * a[1:-1] + a[:-2])))


def mode_profile(H: int, mode: int, amp: float = 1.0) -> np.ndarray:
    """两个「策略模式」的动作块原型：绕行方向相反，端点都归零。"""
    t = np.linspace(0.0, np.pi, H)
    return mode * amp * np.sin(t) * (1.0 + 0.25 * np.sin(2.0 * t))


def mean_std(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))
