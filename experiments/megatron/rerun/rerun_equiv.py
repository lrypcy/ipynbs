"""确定性重跑的正确性验证：挂上校验项 vs 静默累加。

对应文章：
  https://lrypcy.github.io/2026/10/20/megatron-12-deterministic-rerun-and-fault-injection/

源码基准：megatron-core 0.20.0 / commit `60e039626`。

本模块把 `rerun_state_machine.py` 的判定逻辑抽成一个可跑的最小复现：
`_compare_floats` 的相对差、`_save_state` / `_restore_state` 的 RNG 恢复，
以及「不挂校验项 → 误差被静默累加」这条对照。纯 numpy float64，固定 seed。

四项断言：

  §2  `_compare_floats` 的判定边界：相等 / 双 NaN / 单 NaN / Inf / 相对差公式
  §3  恢复 RNG 后重跑逐位一致；不恢复则不同（这就是 `_save_state` 存 rng_state 的理由）
  §4  静默累加对照：同一处瞬时翻转，挂校验项时梯度逐位不变，不挂时梯度被污染
  §5  校验开销的解析核算：挂 N 个校验项每 step 的额外算力与显存

数学背景：源码 `_compare_floats`（`rerun_state_machine.py` L1401-L1420）的相对差是

$$d(a, b) = \left\lvert \frac{a - b}{a + b} \right\rvert \times 2$$

对称、且在 $$a + b \to 0$$ 时发散——这是「相对可比性」而非「绝对差」的度量。
`validate_result` 的判据是 $$d > \text{tolerance}$$（L623、L669）。
"""

from __future__ import annotations

import math

import numpy as np

SEED = 20261020

COMPARISON_MATCH = 0.0
COMPARISON_MISMATCH = math.inf

# 一个「模型」的形状：单层线性 + 均方损失，微批大小 4
IN_DIM = 16
OUT_DIM = 8
MICRO_BATCH = 4
N_MICRO_BATCHES = 8
DROPOUT_P = 0.1


# ────────────────────────────────────────────────────────────────────
# §2  `_compare_floats` 的判定边界
# ────────────────────────────────────────────────────────────────────
def compare_floats(a: float, b: float) -> float:
    """逐行复刻 `rerun_state_machine._compare_floats`（L1401-L1420）。

    判定顺序不能改：先判相等、再判双 NaN、再判 NaN/Inf 交叉，
    最后才算相对差。任一「类别不同」的组合直接返回 inf，不进入相对差计算。
    """
    if (a == b) or (math.isnan(a) and math.isnan(b)):
        return COMPARISON_MATCH
    if (
        (math.isnan(a) and not math.isnan(b))
        or (not math.isnan(a) and math.isnan(b))
        or (math.isinf(a) and not math.isinf(b))
        or (not math.isinf(a) and math.isinf(b))
        or (math.isnan(a) and math.isinf(b))
        or (math.isinf(a) and math.isnan(b))
    ):
        return COMPARISON_MISMATCH
    return math.fabs((a - b) / (a + b) * 2)


def check_e2() -> list[dict]:
    """把 compare_floats 的每个判定分支跑一遍。"""
    inf = math.inf
    nan = math.nan
    cases = [
        ("完全相等", 3.5, 3.5, COMPARISON_MATCH),
        ("双 NaN 视为相等", nan, nan, COMPARISON_MATCH),
        ("单边 NaN", nan, 3.5, COMPARISON_MISMATCH),
        ("单边 Inf", inf, 3.5, COMPARISON_MISMATCH),
        ("NaN vs Inf", nan, inf, COMPARISON_MISMATCH),
        ("双 Inf 同号", inf, inf, COMPARISON_MATCH),
        ("同号相对差", 1.0, 1.001, math.fabs((1.0 - 1.001) / (1.0 + 1.001) * 2)),
        ("对称性 a↔b", 1.0, 1.001, compare_floats(1.001, 1.0)),
        ("零对一", 0.0, 1.0, 2.0),
        ("同号极接近", 1.0, 1.0 + 1e-15, math.fabs((1.0 - (1.0 + 1e-15)) / (1.0 + (1.0 + 1e-15)) * 2)),
        ("异号且和为零", 1.0, -1.0, COMPARISON_MISMATCH),
    ]
    out = []
    for desc, a, b, want in cases:
        try:
            got = compare_floats(a, b)
        except ZeroDivisionError:
            # 源码在 a + b == 0 且两者有限不等时会抛 ZeroDivisionError：
            # Python 的 float 除零不是 inf 而是异常。源码用 `.item()` 取到的就是 Python float，
            # 所以这个边界在真实代码里同样是抛异常而不是返回 inf。本实验如实复现。
            got = "ZeroDivisionError"
        ok = (got == want) or (math.isnan(got) and math.isnan(want)) if isinstance(got, float) \
            else (got == "ZeroDivisionError" and desc == "异号且和为零")
        out.append({"desc": desc, "a": a, "b": b, "got": got, "want": want, "ok": ok})
    return out


# ────────────────────────────────────────────────────────────────────
# §3  恢复 RNG 后重跑逐位一致
# ────────────────────────────────────────────────────────────────────
def apply_random_dropout(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """模拟 dropout：按概率把元素置零。这是 forward 里唯一依赖 RNG 的地方。"""
    mask = rng.random(t.shape) >= DROPOUT_P
    return t * mask


def forward_backward(x, y, w0, rng) -> tuple[float, np.ndarray]:
    """一次 forward + backward（解析梯度，纯 numpy）。

    loss = mean((x_dropout @ w0 - y)^2)
    dL/dw0 = 2/N * (x_dropout @ w0 - y)^T @ x_dropout

    返回 (loss, grad)，grad 的形状同 w0。
    """
    xd = apply_random_dropout(x, rng)          # [mb, in]
    pred = xd @ w0                             # [mb, out]
    diff = pred - y                           # [mb, out]
    loss = float(np.mean(diff * diff))
    grad = (2.0 / diff.shape[0]) * (xd.T @ diff)   # [in, out]
    return loss, grad


def check_e3(seed: int = SEED) -> dict:
    """同一 step 跑两次：恢复 RNG vs 不恢复。

    这是 `_save_state`（L964-L983）存四个 RNG 状态、`_restore_state`（L985-L994）
    逐个 set 回去的直接复现。不恢复的那一路用的是**同一个 generator 继续往下走**
    ——这才是真实的失败模式：第一次 forward 消耗掉的随机数不会被退回。
    """
    rng_data = np.random.default_rng(seed)
    x = rng_data.standard_normal((MICRO_BATCH, IN_DIM))
    y = rng_data.standard_normal((MICRO_BATCH, OUT_DIM))
    w0 = rng_data.standard_normal((IN_DIM, OUT_DIM))

    # 第一次跑：在 forward 起始处记下 RNG 状态（= _save_state 的时机）
    rng_shared = np.random.default_rng(12345)
    state_saved = rng_shared.bit_generator.state
    loss_1, grad_1 = forward_backward(x, y, w0, rng_shared)

    # 第二次跑：不恢复 RNG，同一个 generator 继续往下走
    loss_norst, grad_norst = forward_backward(x, y, w0, rng_shared)

    # 第三次跑：恢复 RNG 后重跑（= _restore_state 后重跑）
    rng_shared.bit_generator.state = state_saved
    loss_rst, grad_rst = forward_backward(x, y, w0, rng_shared)

    return {
        "loss": loss_1,
        "ref_absmax": float(np.max(np.abs(grad_1))),
        "cmp_restore": compare_floats(loss_rst, loss_1),
        "cmp_no_restore": compare_floats(loss_norst, loss_1),
        "grad_restore_absdiff": float(np.max(np.abs(grad_rst - grad_1))),
        "grad_no_restore_absdiff": float(np.max(np.abs(grad_norst - grad_1))),
        "grad_restore_bitexact": bool(np.max(np.abs(grad_rst - grad_1)) == 0.0),
        "grad_no_restore_bitexact": bool(np.max(np.abs(grad_norst - grad_1)) == 0.0),
    }


# ────────────────────────────────────────────────────────────────────
# §4  静默累加对照
# ────────────────────────────────────────────────────────────────────
def run_gradient_accumulation(
    n_micro: int, flip_at: int, rng_seed: int, with_validation: bool, magnitude: float = 1e6
) -> dict:
    """跑一个 step 的梯度累加。

    flip_at:      在第几个 microbatch 上注入一次「瞬时翻转」（梯度整体乘一个因子）
    with_validation: True 时按 `is_unexpectedly_large` 的思路检测，
                     命中就恢复 RNG 重跑该 microbatch（L622-L636 的行为）

    还原的源码行为：检测到 → `_restore_state` → 重跑 → 结果逐位还原 → 接受。
    """
    rng_data = np.random.default_rng(rng_seed)
    grads = np.zeros((IN_DIM, OUT_DIM))
    detected = 0
    reran = 0

    for mb in range(n_micro):
        rng_mb = np.random.default_rng(1000 + mb)
        state_saved = rng_mb.bit_generator.state

        x = rng_data.standard_normal((MICRO_BATCH, IN_DIM))
        y = rng_data.standard_normal((MICRO_BATCH, OUT_DIM))
        w0 = rng_data.standard_normal((IN_DIM, OUT_DIM))
        _, grad = forward_backward(x, y, w0, rng_mb)

        # 注入瞬时翻转：梯度整体被放大，模拟位翻转 / 瞬态错误
        if mb == flip_at:
            grad = grad * magnitude

        if with_validation and np.max(np.abs(grad)) > 1e3:
            detected += 1
            rng_mb.bit_generator.state = state_saved   # _restore_state
            _, grad = forward_backward(x, y, w0, rng_mb)  # 重跑
            reran += 1

        grads += grad

    mean_grad = grads / n_micro
    return {
        "n_micro": n_micro,
        "flip_at": flip_at,
        "with_validation": with_validation,
        "detected": detected,
        "reran": reran,
        "mean_absmax": float(np.max(np.abs(mean_grad))),
        "mean_l2": float(np.linalg.norm(mean_grad)),
    }


def check_e4(seed: int = SEED) -> dict:
    """对照：同一处瞬时翻转，挂校验项 vs 不挂。"""
    clean = run_gradient_accumulation(N_MICRO_BATCHES, -1, seed, False)
    silent = run_gradient_accumulation(N_MICRO_BATCHES, 3, seed, False)
    guarded = run_gradient_accumulation(N_MICRO_BATCHES, 3, seed, True)
    return {
        "clean": clean, "silent": silent, "guarded": guarded,
        "cmp_clean_guarded": compare_floats(guarded["mean_l2"], clean["mean_l2"]),
        "cmp_clean_silent": compare_floats(silent["mean_l2"], clean["mean_l2"]),
    }


# ────────────────────────────────────────────────────────────────────
# §5  校验开销的解析核算（A 级，只给算术模型）
# ────────────────────────────────────────────────────────────────────
def validation_overhead_model(
    n_validations: int, param_elems: int, micro_batches: int, flops_per_elem: int = 2
) -> dict:
    """挂 N 个校验项的每 step 开销：只给算术，不假装实测。

    三项：
      1. 校验本身的算力：每次校验扫一遍结果张量，约 flops_per_elem FLOP/元素
      2. 存 initial_result 的显存：每个校验项存一份（float32，4 字节）
      3. 触发时重跑整个 step：micro_batches × param_elems × flops_per_elem
    """
    check_flops = n_validations * param_elems * flops_per_elem
    result_bytes = n_validations * 4
    rerun_flops = micro_batches * param_elems * flops_per_elem
    return {
        "n_validations": n_validations,
        "param_elems": param_elems,
        "check_flops": check_flops,
        "result_bytes": result_bytes,
        "rerun_flops": rerun_flops,
        "marginal_check_flops_per_validation": param_elems * flops_per_elem,
    }


def check_e5() -> dict:
    """扫挂载的校验项数，给出边际开销。"""
    param_elems = IN_DIM * OUT_DIM
    rows = [validation_overhead_model(n, param_elems, N_MICRO_BATCHES) for n in (1, 2, 4, 8, 16)]
    return {"rows": rows, "param_elems": param_elems}


# ────────────────────────────────────────────────────────────────────
# 故障注入时延的采样分布（复刻 `fault_injector.get_fault_delay`）
# ────────────────────────────────────────────────────────────────────
def sample_fault_delay(mtti_seconds: float, offset_seconds: float, rng: np.random.Generator) -> float:
    """复刻 `fault_injector.py` L174-L189 的指数分布采样。

    源码：`lambda_inj = 1.0 / mtti_seconds`
          `fault_delay = offset_seconds + (-log(1.0 - rng.random()) / lambda_inj)`

    这是标准的指数分布反函数法：$$X = -\ln(1 - U) / \lambda$$。
    """
    lambda_inj = 1.0 / mtti_seconds
    return offset_seconds + (-math.log(1.0 - rng.random()) / lambda_inj)


def check_e6(mtti_seconds: float = 60.0, offset_seconds: float = 0.0,
             n: int = 20000, seed: int = SEED) -> dict:
    """扫 mtti，验证采样出的时延均值≈mtti、方差≈mtti²。"""
    rng = np.random.default_rng(seed)
    xs = np.array([sample_fault_delay(mtti_seconds, offset_seconds, rng) for _ in range(n)])
    return {
        "mtti_seconds": mtti_seconds,
        "n": n,
        "mean": float(xs.mean()),
        "expected_mean": offset_seconds + mtti_seconds,
        "std": float(xs.std(ddof=1)),
        "expected_std": mtti_seconds,
        "min": float(xs.min()),
        "max": float(xs.max()),
    }