"""A 级解析：CUDA Graph 前后的 kernel launch 计数模型。

对应文章：
  https://lrypcy.github.io/2026/10/18/megatron-10-cuda-graph/

**本脚本只做算术，不做任何性能声称。**

## 为什么 launch 数可以被解析地数出来

CUDA Graph 的收益来自一个可数的量：CPU 侧每次 iteration 需要提交的 launch 次数。
把一次 transformer layer 的前向/反向拆成「可 launch 的 CUDA 操作清单」，
累加得到未图化时的 launch 数；图化后这段清单整体变成 **1 次 graph launch**。

## 必须分清的两个数（本脚本的核心诚实性约束）

| 量 | 图化前 | 图化后 | 说明 |
|---|---|---|---|
| **CPU 侧 launch 次数** | `N` | `1` | 这是 CUDA Graph 消掉的东西 |
| **device 侧 kernel 执行次数** | `N` | `N` | 图里仍然有 N 个 kernel node，只是由 driver 一次性提交 |

把这两者混为一谈是这类文章最常见的错误。图化不会让 GPU 少算一遍乘法，
它省掉的是 CPU 反复调用 launch API 的开销。因此本脚本**同时输出两列**，
并且**不输出任何时间/吞吐数字**——那需要真卡，本机没有。

## 算术的两条来源

1. **GEMM 个数**：从 Megatron 的 dense 层结构数出来（`linear_qkv` / `linear_proj`
   / `linear_fc1` / `linear_fc2`），这是源码结构决定的，不依赖 TE 版本。
2. **每类算子的 launch 数**：TE 的融合粒度会变（`fused_bias_swiglu` 等），
   所以「pointwise 算子拆成几个 kernel」是**显式假设**，在 `ASSUMPTIONS` 里列出，
   读者可自行改参数重算。脚本对 GEMM 数做敏感性扫描，展示结论对假设的依赖程度。

用法（在本目录）：
    ~/Software/miniconda3/bin/python3 launch_count.py            # 写 results/launch_count.txt
    ~/Software/miniconda3/bin/python3 launch_count.py --stdout   # 只打屏

退出码非 0 表示有对账失败项，可直接当 CI 用。
"""

from __future__ import annotations

import argparse
import io
import os
import sys

# --------------------------------------------------------------------------------------
# 显式假设：TE 融合粒度随版本变化，这里把它做成可调参数而不是写死的常数。
# 每个 (名称, 前向 launch 数, 反向 launch 数) 一行。
# --------------------------------------------------------------------------------------

# 一个 dense transformer layer 的前向算子清单。
# GEMM 行：linear_qkv / linear_proj / linear_fc1 / linear_fc2 —— 由 Megatron 模块结构决定。
# 其余行：融合粒度假设，TE 版本不同会变。
FORWARD_OPS: list[tuple[str, str, int, int]] = [
    # (算子名, 类别, 前向 launch 数, 反向 launch 数)
    ("input_layernorm", "pointwise", 1, 1),
    ("linear_qkv", "gemm", 1, 2),          # 反向 = dgrad + wgrad
    ("rope", "pointwise", 1, 1),
    ("attention_core", "gemm_like", 1, 1),  # FA/cuDNN 是一个 fused kernel，反向也是一个
    ("linear_proj", "gemm", 1, 2),
    ("bias_drop_add_residual", "pointwise", 1, 1),
    ("pre_mlp_layernorm", "pointwise", 1, 1),
    ("linear_fc1", "gemm", 1, 2),
    ("fused_swiglu", "pointwise", 1, 1),
    ("linear_fc2", "gemm", 1, 2),
    ("bias_drop_add_residual_mlp", "pointwise", 1, 1),
]

ASSUMPTIONS = """\
本模型的诚实边界：
  * GEMM 个数（4 个 / layer）来自 Megatron dense 层模块结构，与 TE 版本无关 —— 可靠。
  * 每个 GEMM 反向 = 2 次 launch（dgrad + wgrad）—— autograd 标准行为，可靠。
  * pointwise 算子的 launch 数是**融合粒度假设**，随 TE 版本变化。
    脚本对融合粒度做敏感性扫描，不把假设当结论。
  * 不含 MoE / Mamba / 分布式通信 launch —— 那些另有依赖，见文章正文。
  * 不含重计算（recompute）带来的额外 launch —— 那会**增加** launch 数，正文单独讨论。
"""


def layer_launches(forward_ops: list[tuple[str, str, int, int]]) -> dict[str, int]:
    """返回单个 dense layer 前向/反向的 launch 数。

    gemm 类的 forward_ops[i][2] 已按 1 计，反向已按 2 计，这里只累加。
    """
    fwd = sum(op[2] for op in forward_ops)
    bwd = sum(op[3] for op in forward_ops)
    return {"fwd": fwd, "bwd": bwd, "total": fwd + bwd}


def graph_table(num_layers: int, microbatches: int) -> dict[str, int]:
    """整 step 的 launch 账：CPU 侧 vs device 侧，图化前 vs 图化后。"""
    per = layer_launches(FORWARD_OPS)
    eager = per["total"] * num_layers * microbatches

    # 图化：每层的前向图 + 反向图各 1 次 graph launch
    graphed = 2 * num_layers * microbatches

    return {
        "per_layer_fwd": per["fwd"],
        "per_layer_bwd": per["bwd"],
        "per_layer_total": per["total"],
        "eager_cpu_launches": eager,
        "graph_cpu_launches": graphed,
        "device_kernel_nodes": eager,   # device 侧执行次数不变
        "cpu_launches_saved": eager - graphed,
    }


def optimizer_launch_model() -> dict[str, int]:
    """`capturable` Adam step 的 launch 计数模型。

    对应文章 §6（`optimizer_cuda_graph.py` 68 行孤儿）。

    一个 Adam 参数组的 step，按元素算子逐条数（foreach/fused 实现会把它们批成少量 kernel，
    这里给的是**未融合上界**；torch 的 `_fused_adam` 会显著更少）：

      exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)          -> 2
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=...)  -> 2
      bias_correction1 / bias_correction2 计算                  -> 2
      denom = (exp_avg_sq.sqrt() / bias_correction2).add_(eps)  -> 3
      step_size = lr / bias_correction1                        -> 1
      param.addcdiv_(exp_avg, denom, value=-step_size)          -> 1

    共 11 次。foreach 实现按「每类算子一次」批处理，与参数量无关；
    `_fused_adam` 把大部分收进单个 multi-tensor-apply kernel。
    图化后这 11 次（未融合）/ 约 4-6 次（foreach）整体变成 1 次 graph launch。
    """
    per_param_group_unfused = 11
    per_param_group_foreach = 5   # 5 类算子各批一次
    per_param_group_fused = 1     # _fused_adam 单 kernel

    return {
        "adam_unfused": per_param_group_unfused,
        "adam_foreach": per_param_group_foreach,
        "adam_fused": per_param_group_fused,
        "graphed": 1,
    }


def sensitivity(num_layers: int = 32, microbatches: int = 8) -> list[tuple[int, int, float]]:
    """对 pointwise 融合粒度做敏感性扫描。

    放大系数 k 含义：每个 pointwise 算子被拆成 k 个 kernel（k=1 是最激进融合，
    k=3 是「每个 pointwise 都拆得很碎」的悲观情形）。
    返回 (k, eager_launches, 相对 k=1 的倍数)。
    """
    rows: list[tuple[int, int, float]] = []
    base: int | None = None
    for k in (1, 2, 3):
        scaled = [
            (name, cat, f * k if cat == "pointwise" else f, b * k if cat == "pointwise" else b)
            for (name, cat, f, b) in FORWARD_OPS
        ]
        per = layer_launches(scaled)
        eager = per["total"] * num_layers * microbatches
        if base is None:
            base = eager
        rows.append((k, eager, eager / base))
    return rows


def _fmt_table(rows: list[tuple[str, str, int, int]], t: dict[str, int]) -> str:
    out = io.StringIO()
    out.write("  " + "算子".ljust(28) + "类别".ljust(12) + "fwd".rjust(5) + "bwd".rjust(6) + "\n")
    out.write("  " + "-" * 51 + "\n")
    for name, cat, f, b in rows:
        out.write("  " + name.ljust(28) + cat.ljust(12) + str(f).rjust(5) + str(b).rjust(6) + "\n")
    out.write("  " + "-" * 51 + "\n")
    out.write(
        "  " + "合计 / layer".ljust(40)
        + str(t["per_layer_fwd"]).rjust(5) + str(t["per_layer_bwd"]).rjust(6) + "\n"
    )
    return out.getvalue()


def report() -> str:
    out = io.StringIO()

    out.write("=" * 78 + "\n")
    out.write("CUDA Graph launch 计数模型（A 级解析，无 GPU，无性能声称）\n")
    out.write("=" * 78 + "\n\n")

    out.write("§1  单个 dense transformer layer 的算子清单与 launch 数\n")
    out.write("-" * 78 + "\n")
    L, M = 32, 8
    t = graph_table(L, M)
    out.write(_fmt_table(FORWARD_OPS, t))
    out.write(f"\n  配置：num_layers={L}  microbatches={M}\n\n")

    out.write("§2  整 step 的 launch 账（关键：CPU 侧 vs device 侧分开记）\n")
    out.write("-" * 78 + "\n")
    out.write(f"  {'量':<34}{'图化前':>12}{'图化后':>12}\n")
    out.write("  " + "-" * 58 + "\n")
    out.write(
        f"  {'CPU 侧 launch 次数':<30}{t['eager_cpu_launches']:>12}{t['graph_cpu_launches']:>12}\n"
    )
    out.write(
        f"  {'device 侧 kernel 执行次数':<28}{t['device_kernel_nodes']:>12}{t['device_kernel_nodes']:>12}\n"
    )
    out.write("  " + "-" * 58 + "\n")
    out.write(f"  CPU 侧省下：{t['cpu_launches_saved']} 次 / step\n")
    out.write(
        "  注意：device 侧执行次数**没有减少**。图化省的是 CPU 反复提交 launch 的开销，\n"
        "  不是让 GPU 少算一遍。把这行写成「kernel 变少」是错的。\n\n"
    )

    out.write("§3  对融合粒度假设的敏感性（pointwise 拆成 k 个 kernel）\n")
    out.write("-" * 78 + "\n")
    out.write(f"  {'k':>3}{'eager launch/step':>20}{'相对 k=1':>14}\n")
    out.write("  " + "-" * 37 + "\n")
    for k, eager, mult in sensitivity(L, M):
        out.write(f"  {k:>3}{eager:>20}{mult:>13.2f}x\n")
    out.write(
        "\n  k 只改变绝对值，不改变「CPU 侧 N -> 1」这个结构性事实。\n\n"
    )

    out.write("§4  optimizer step（对应 §6 孤儿节）\n")
    out.write("-" * 78 + "\n")
    o = optimizer_launch_model()
    out.write(f"  {'实现':<26}{'每参数组 launch':>18}\n")
    out.write("  " + "-" * 44 + "\n")
    for k, label in (
        ("adam_unfused", "Adam 逐元素（未融合上界）"),
        ("adam_foreach", "Adam foreach（批成 5 类）"),
        ("adam_fused", "_fused_adam（单 kernel）"),
        ("graphed", "图化后（capturable）"),
    ):
        out.write(f"  {label:<24}{o[k]:>18}\n")
    out.write(
        "\n  `capturable` 切的是「优化器能否被 capture」这个可用性边界，\n"
        "  不是「给 step 加个图」的开关。详见正文 §6。\n\n"
    )

    out.write("§5  本模型不覆盖的部分\n")
    out.write("-" * 78 + "\n")
    out.write(ASSUMPTIONS)
    out.write(
        "\n  图化后的**实际收益上界**必须引官方数据并注明卡型规模（B 级证据）。\n"
        "  本机无 GPU，本脚本不产出任何吞吐 / MFU / 毫秒数字。\n"
    )

    return out.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description="CUDA Graph launch 计数模型")
    ap.add_argument("--stdout", action="store_true", help="只打屏，不写 results/")
    ap.add_argument("--check", action="store_true", help="与冻结的 results/ 比对，有漂移则非 0 退出")
    args = ap.parse_args()

    text = report()

    # ---- 对账：不变量必须成立，否则非 0 退出 ----
    t = graph_table(32, 8)
    failures: list[str] = []

    def ck(tag: str, got, want) -> None:
        if got != want:
            failures.append(f"{tag}: got {got}, want {want}")

    # 反向 = 4 个 GEMM × 2 (dgrad+wgrad) + attention_core 1 + 7 个 pointwise × 1 = 15
    ck("GEMM 个数/layer", sum(1 for _, c, _, _ in FORWARD_OPS if c == "gemm"), 4)
    ck("前向 launch/layer", t["per_layer_fwd"], 11)
    ck("反向 launch/layer", t["per_layer_bwd"], 15)
    ck("每层合计", t["per_layer_total"], 26)
    ck("eager = per_layer * L * M", t["eager_cpu_launches"], 26 * 32 * 8)
    ck("graph = 2 * L * M", t["graph_cpu_launches"], 2 * 32 * 8)
    ck("device 侧执行数不变", t["device_kernel_nodes"], t["eager_cpu_launches"])
    ck("省下 = eager - graph", t["cpu_launches_saved"], t["eager_cpu_launches"] - t["graph_cpu_launches"])
    ck("Adam 未融合上界", optimizer_launch_model()["adam_unfused"], 11)
    ck("敏感性 k=1 基线", sensitivity()[0][1], 26 * 32 * 8)

    if failures:
        for f in failures:
            print(f"MISMATCH  {f}", file=sys.stderr)
        return 1

    header = "OK  10/10 断言通过（launch 计数模型不变量）\n\n"
    frozen = header + text

    if args.stdout:
        sys.stdout.write(frozen)
        return 0

    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "results")
    path = os.path.join(outdir, "launch_count.txt")

    if args.check:
        if not os.path.exists(path):
            print(f"MISMATCH  冻结结果不存在：{path}", file=sys.stderr)
            return 1
        with open(path, encoding="utf-8") as fh:
            have = fh.read()
        if have != frozen:
            print(f"MISMATCH  与冻结结果漂移：{path}", file=sys.stderr)
            print("        正文里的数字必须按数字字面量与本脚本输出对齐。", file=sys.stderr)
            return 1
        print(f"OK  与冻结结果一致：{path}")
        return 0

    os.makedirs(outdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(frozen)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
