"""A 级解析：GTP_remat 每 microbatch 每权重的通信字节预算。

只做算术，不做端到端性能声称。输出对账官方 GTP 设计文档 §1.3 的
"Communication volume breakdown" 表。

三个来源等级说明（AGENTS.md / RESEARCH_PLAN.md §1.1）：
- 本脚本是 A 级（解析脚本实跑），读者可自行复算。
- 官方文档那张表是 B 级参照物，两者逐格一致才算通过。
- 本脚本不声称任何吞吐 / MFU 结论。

公式（来自官方表的列定义）：
    Per-elem    = Data B/elem + Scale_inv B/elem      # 一份量化权重缓冲的线上字节
    Fwd AG      = Per-elem                            # 前向聚合一次
    Bwd AG      = Per-elem                            # 反向再聚合一次同一分片（columnwise 视角）
    Wgrad RS    = 2.0                                 # 梯度恒按 bf16 归约，与权重精度无关
    Total       = Fwd AG + Bwd AG + Wgrad RS

两种原生格式都没有 per-microbatch 的 amax all-reduce：
- MXFP8 是 microscale，缩放因子随数据一起走；
- NVFP4 的 block scale 在被聚合的缓冲里，per-tensor scale 在优化器步里定。
所以 Total 里不含 Fwd AR(amax) 一项。
"""

from __future__ import annotations

import sys

# 元素 = 一个权重矩阵元素（不含 padding）
BYTES_PER_BF16 = 2

# (格式名, 块大小, 每元素数据字节)
#   BF16  : 无块，每元素 2 字节
#   MXFP8 : 32 元素共享一个 E8M0 scale，每元素 1 字节数据 + 1/32 字节 scale
#   NVFP4 : 16 元素共享一个 E4M3 scale，每元素 4bit(0.5 字节) 数据 + 1/16 字节 scale
FORMATS: list[tuple[str, int | None, float]] = [
    ("BF16", None, 2.0),
    ("MXFP8", 32, 1.0),
    ("NVFP4", 16, 0.5),
]

WGRAD_RS_BYTES = 2.0  # bf16 梯度归约，与权重精度无关


def budget(block: int | None, data_b: float) -> dict[str, float]:
    """返回单个格式的每元素字节预算。"""
    if block is None:
        scale_b = 0.0
    else:
        # 一个 scale 字节摊到 block 个元素上
        scale_b = 1.0 / block
    per_elem = data_b + scale_b
    fwd_ag = per_elem
    bwd_ag = per_elem
    total = fwd_ag + bwd_ag + WGRAD_RS_BYTES
    return {
        "data": data_b,
        "scale": scale_b,
        "per_elem": per_elem,
        "fwd_ag": fwd_ag,
        "bwd_ag": bwd_ag,
        "wgrad_rs": WGRAD_RS_BYTES,
        "total": total,
    }


def main() -> int:
    rows = [(name, block, budget(block, data)) for name, block, data in FORMATS]
    baseline = rows[0][2]["total"]

    print("=" * 78)
    print("comm_volume.py —— GTP_remat 通信量预算（A 级解析，无性能声称）")
    print("口径：per-microbatch per-weight；假设 bf16 wgrad reduce-scatter")
    print("=" * 78)
    print()
    print(f"{'格式':<7}{'块':>5}{'数据B':>9}{'scaleB':>10}{'Per-elem':>11}"
          f"{'FwdAG':>9}{'BwdAG':>9}{'RS':>7}{'合计':>9}{'vs BF16':>10}")
    print("-" * 78)
    for name, block, b in rows:
        blk = str(block) if block else "n/a"
        print(f"{name:<7}{blk:>5}"
              f"{b['data']:>9.4f}{b['scale']:>10.4f}{b['per_elem']:>11.4f}"
              f"{b['fwd_ag']:>9.4f}{b['bwd_ag']:>9.4f}{b['wgrad_rs']:>7.4f}"
              f"{b['total']:>9.4f}{b['total']/baseline:>9.2f}x")

    print()
    print("── 逐格对账官方表（§1.3 Communication volume breakdown）──")
    # 官方文档给出的四列，顺序：Per-elem / Fwd AG / Bwd AG / Total / vs BF16
    official = {
        "BF16":  (2.0000, 2.0000, 2.0000, 6.0000, 1.00),
        "MXFP8": (1.0313, 1.0313, 1.0313, 4.0626, 0.68),
        "NVFP4": (0.5625, 0.5625, 0.5625, 3.1250, 0.52),
    }
    failures: list[str] = []

    def check(label: str, got: float, want: float, tol: float = 5e-4) -> None:
        ok = abs(got - want) <= tol
        if not ok:
            failures.append(f"{label}: 实算 {got:.4f} != 官方 {want:.4f}")
        print(f"  {label:<44}{got:>10.4f}{want:>10.4f}  {'ok' if ok else '!! 不一致'}")

    for name, _block, b in rows:
        o = official[name]
        print(f"  [{name}]")
        check(f"{name} Per-elem", b["per_elem"], o[0])
        check(f"{name} Fwd AG", b["fwd_ag"], o[1])
        check(f"{name} Bwd AG", b["bwd_ag"], o[2])
        check(f"{name} Total B/elem", b["total"], o[3])
        rel = b["total"] / baseline
        ok = abs(rel - o[4]) <= 0.005
        if not ok:
            failures.append(f"{name} vs BF16: 实算 {rel:.2f} != 官方 {o[4]:.2f}")
        print(f"  {'  vs BF16':<44}{rel:>10.2f}{o[4]:>10.2f}  {'ok' if ok else '!! 不一致'}")

    print()
    print("── 结论 ──")
    print("  聚合（AG）部分随精度收缩，但 wgrad RS 恒为 2.0 B/elem，不随之收缩。")
    for name, _block, b in rows:
        ag_share = 2 * b["per_elem"] / b["total"] * 100
        rs_share = b["wgrad_rs"] / b["total"] * 100
        print(f"  {name:<7} AG 两项占总预算 {ag_share:>3.0f}%，wgrad RS 占 {rs_share:>3.0f}%")
    print()
    print("=" * 78)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for f in failures:
            print("  !! " + f)
    else:
        print("对账通过：每一格都与官方 §1.3 通信量分解表一致。")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())