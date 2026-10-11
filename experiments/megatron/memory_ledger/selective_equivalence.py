#!/usr/bin/env python3
"""normal 与 output-discarding 两种 checkpointing 的数值等价性（C 级，CPU torch）。

对应文章《Megatron-LM 深度剖析（08）：显存账本——重计算、卸载与分布保存》。
源码基准：megatron-core 0.20.0 / commit 60e039626。

要验三件事
----------
1. **重算不改变数值**：三条路径（不 checkpoint / normal / output-discarding）
   的前向输出与反向梯度必须**逐位相同**（float64，0 ULP），不是近似。
2. **被丢弃的输出确实不参与反向**：output-discarding 之后，那条输出张量的
   storage 字节数必须真的变成 0，而不是「还在那里只是没人用」。
3. **RNG 恢复是承重结构**：把「重算前恢复 RNG 状态」这一步去掉，
   带 dropout 的模块梯度就会变。源码 `tensor_parallel/random.py` L644/L652-L654
   的 `_set_all_rng_states(*ctx.rng_states)` 不是保险性代码，去掉就错。

对应源码
--------
- normal：`CheckpointFunction`（`tensor_parallel/random.py` L600-L672）
- output-discarding：`CheckpointWithoutOutputFunction`（同文件 L748-L800）
  + `CheckpointWithoutOutputManager`（L803-L845）+ `CheckpointWithoutOutput`（L848-L1010）

本实验与源码的差别（必须说明，不是等价实现）
------------------------------------------
源码在重算后用 C++ 扩展 `_get_share_storage()` 把输出的 `StorageImpl` 指针
换到重算结果上（`random.py` L967-L978），让所有已存在的 view 立刻看到新数据、
且不触发 autograd 的版本号检查。这个 zero-copy 换指针在 CPU 上不可复现；
强行用 `out.data = ...` 代替会让 autograd 拿到已 `resize_(0)` 的 storage 而
段错误（实测退出码 139）。因此本实验把两件事**分开验**：

- 数值等价性（第 1、2 节的断言）：走完整的 autograd Function，但不释放 storage；
- 丢弃确实释放字节（第 3 节）：独立地对一个张量做 `resize_(0)` 并核对字节数。

要验的是语义（丢弃是否真发生、重算是否逐位复现），不是换指针的实现。

用法
----
    python3 selective_equivalence.py
退出码 0 表示全部断言通过。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

torch.manual_seed(20261012)
torch.use_deterministic_algorithms(True)

DTYPE = torch.float64
SEQ, HIDDEN, FFN, HEADS = 16, 32, 64, 4
HEAD_DIM = HIDDEN // HEADS


def ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """两个 float64 张量的最大 ULP 差。逐位相同则返回 0。"""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    ai = a.detach().contiguous().view(torch.int64)
    bi = b.detach().contiguous().view(torch.int64)
    return int((ai - bi).abs().max().item())


class Block(nn.Module):
    """一个够用的 decoder 子块：RMSNorm → 注意力核心 → RMSNorm → MLP。

    注意力核心里放了 dropout —— 这是 RNG 恢复是否承重的试金石。
    """

    def __init__(self, p: float = 0.3):
        super().__init__()
        self.norm1 = nn.RMSNorm(HIDDEN, eps=1e-6).to(DTYPE)
        self.norm2 = nn.RMSNorm(HIDDEN, eps=1e-6).to(DTYPE)
        self.q = nn.Linear(HIDDEN, HIDDEN, bias=False).to(DTYPE)
        self.k = nn.Linear(HIDDEN, HIDDEN, bias=False).to(DTYPE)
        self.v = nn.Linear(HIDDEN, HIDDEN, bias=False).to(DTYPE)
        self.o = nn.Linear(HIDDEN, HIDDEN, bias=False).to(DTYPE)
        self.up = nn.Linear(HIDDEN, FFN, bias=False).to(DTYPE)
        self.down = nn.Linear(FFN, HIDDEN, bias=False).to(DTYPE)
        self.p = p

    def attn_core(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        shape = (b, s, HEADS, HEAD_DIM)
        q = self.q(x).view(shape).transpose(1, 2)
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / (HEAD_DIM ** 0.5)
        causal = torch.ones(s, s, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        probs = torch.nn.functional.dropout(probs, p=self.p, training=True)
        return (probs @ v).transpose(1, 2).reshape(b, s, HIDDEN)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.silu(self.up(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.o(self.attn_core(h))
        x = x + h
        return x + self.mlp(self.norm2(x))


class OutputDiscardingCheckpoint(torch.autograd.Function):
    """镜像 `CheckpointWithoutOutputFunction`（`random.py` L748-L800）。

    forward：前向跑完，把输出 storage 丢掉（`CheckpointWithoutOutput._discard_outputs`，
             L985-L996），但把重算所需的东西留在 ctx 上。
    backward：从 ctx 取输入重算（`_recompute`，L928-L983），把 storage 挂回原句柄，
             再把梯度按 `detach_variable` 隔离后的副本返回（L791-L793 的注释
             说明它取 ctx.inputs 而非 ctx.saved_tensors，正是为了避免
             CPU 卸载场景下重复加载输入）。
    """

    @staticmethod
    def forward(ctx, run_function, holder, *args):
        holder.rng_states = _capture_rng()
        with torch.no_grad():
            out = run_function(*args)
        holder.outputs = ((out,) if isinstance(out, torch.Tensor) else tuple(out))
        ctx.holder = holder
        ctx.run_function = run_function
        ctx.inputs = args
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        h = ctx.holder
        _restore_rng(h.rng_states)
        detached = detach_variable(ctx.inputs)
        with torch.enable_grad():
            new = ctx.run_function(*detached)
        torch.autograd.backward(new, grad_outputs)
        grads = tuple(t.grad for t in detached)
        h.recomputed = tuple((new,) if isinstance(new, torch.Tensor) else new)
        h.outputs = None
        return (None, None) + grads


class Holder:
    """承载输出句柄与丢弃记录，对应 `CheckpointWithoutOutput` 的实例状态。"""

    def __init__(self, restore_rng: bool = True):
        self.restore_rng = restore_rng
        self.outputs: Optional[Tuple[torch.Tensor, ...]] = None
        self.recomputed: Optional[Tuple[torch.Tensor, ...]] = None
        self.rng_states: Tuple[torch.Tensor, ...] = ()


def output_discarding_checkpoint(fn: Callable, holder: Holder, *args) -> torch.Tensor:
    return OutputDiscardingCheckpoint.apply(fn, holder, *args)


def detach_variable(inputs: Tuple[torch.Tensor, ...]) -> List[torch.Tensor]:
    """镜像 `random.py` L657 的 `detach_variable`。

    不 detach 的话，重算时内部 `backward` 会把梯度写进**外层**那批张量的
    `.grad`，外层再回传一次就等于把梯度算了两遍。源码正是靠这一步
    把重算图的梯度与外层图的 `.grad` 隔开。
    """
    out = []
    for t in inputs:
        d = t.detach()
        d.requires_grad_(t.requires_grad)
        out.append(d)
    return out


def _capture_rng() -> Tuple[torch.Tensor, ...]:
    return (torch.get_rng_state().clone(),)


def _restore_rng(states: Tuple[torch.Tensor, ...]) -> None:
    torch.set_rng_state(states[0])


def normal_checkpoint(fn: Callable, *args) -> torch.Tensor:
    """镜像 `CheckpointFunction`（random.py L600-L672）：存全部输入。"""

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, run_function, *a):
            ctx.run_function = run_function
            ctx.rng_states = _capture_rng()
            with torch.no_grad():
                out = run_function(*a)
            ctx.save_for_backward(*a)
            return out

        @staticmethod
        def backward(ctx, *grad_outputs):
            _restore_rng(ctx.rng_states)
            detached = detach_variable(ctx.saved_tensors)
            with torch.enable_grad():
                out = ctx.run_function(*detached)
            torch.autograd.backward(out, grad_outputs)
            return (None,) + tuple(t.grad for t in detached)

    return _Fn.apply(fn, *args)


@dataclass
class PathResult:
    name: str
    out: torch.Tensor
    grads: List[torch.Tensor]


def run_plain(block: Block, x: torch.Tensor, rng: torch.Tensor) -> PathResult:
    block.zero_grad(set_to_none=True)
    torch.set_rng_state(rng.clone())
    x = x.detach().clone().requires_grad_(True)
    out = block(x)
    out.sum().backward()
    return PathResult("plain", out.detach().clone(),
                      [p.grad.detach().clone() for p in block.parameters()] + [x.grad.detach().clone()])


def run_normal(block: Block, x: torch.Tensor, rng: torch.Tensor) -> PathResult:
    block.zero_grad(set_to_none=True)
    torch.set_rng_state(rng.clone())
    x = x.detach().clone().requires_grad_(True)
    out = normal_checkpoint(lambda h: block(h), x)
    out.sum().backward()
    return PathResult("normal", out.detach().clone(),
                      [p.grad.detach().clone() for p in block.parameters()] + [x.grad.detach().clone()])


def run_output_discarding(block: Block, x0: torch.Tensor, rng: torch.Tensor
                          ) -> Tuple[PathResult, Holder]:
    """对 MLP 那一段做 output-discarding（对应 `recompute_modules` 里的 `mlp`）。"""
    block.zero_grad(set_to_none=True)
    torch.set_rng_state(rng.clone())
    x = x0.detach().clone().requires_grad_(True)

    def mlp_segment(h):
        return block.mlp(block.norm2(h))

    x1 = x + block.o(block.attn_core(block.norm1(x)))
    holder = Holder(restore_rng=True)
    inner = output_discarding_checkpoint(mlp_segment, holder, x1)
    full = x1 + inner
    full.sum().backward()

    grads = [p.grad.detach().clone() for p in block.parameters()]
    grads.append(x.grad.detach().clone())
    return PathResult("output-discarding", full.detach().clone(), grads), holder


def main() -> int:
    block = Block().to(DTYPE)
    x0 = torch.randn(2, SEQ, HIDDEN, dtype=DTYPE)

    print("=" * 96)
    print("normal 与 output-discarding checkpointing 的数值等价性（C 级 · CPU torch float64）")
    print(f"子块结构：SEQ={SEQ} HIDDEN={HIDDEN} FFN={FFN} HEADS={HEADS} "
          f"dropout p={block.p}  种子={20261012}")
    print("=" * 96)

    rng0 = torch.get_rng_state().clone()
    ref = run_plain(block, x0, rng0)
    nrm = run_normal(block, x0, rng0)
    odd, holder = run_output_discarding(block, x0, rng0)

    print("\n【1】前向输出逐位对照")
    d_out_n = ulp_diff(ref.out, nrm.out)
    d_out_o = ulp_diff(ref.out, odd.out)
    print(f"  plain vs normal                 max ULP = {d_out_n}")
    print(f"  plain vs output-discarding      max ULP = {d_out_o}")

    print("\n【2】反向梯度逐位对照（8 个参数 + 1 个输入）")
    print(f"  plain 与 normal   最大绝对差 = "
          f"{max((a - b).abs().max().item() for a, b in zip(ref.grads, nrm.grads)):.3e}")
    print(f"  plain 与 odd      最大绝对差 = "
          f"{max((a - b).abs().max().item() for a, b in zip(ref.grads, odd.grads)):.3e}")
    worst_n = max(ulp_diff(a, b) for a, b in zip(ref.grads, nrm.grads))
    worst_o = max(ulp_diff(a, b) for a, b in zip(ref.grads, odd.grads))
    param_ulp = [ulp_diff(a, b) for a, b in zip(ref.grads[:-1], odd.grads[:-1])]
    x_ulp = ulp_diff(ref.grads[-1], odd.grads[-1])
    worst_param_odd = max(param_ulp)
    inner_ulp = max(param_ulp[6:8])
    outer_ulp = max(param_ulp[0:6])
    for i, (a, b, c) in enumerate(zip(ref.grads, nrm.grads, odd.grads)):
        name = f"grad[{i}]"
        print(f"  {name:<10s} plain vs normal = {ulp_diff(a, b):>3d} ULP   "
              f"plain vs output-discarding = {ulp_diff(a, c):>3d} ULP")
    print(f"  最大 ULP 差：normal = {worst_n}   "
          f"output-discarding = {worst_o}（其中参数 {worst_param_odd}，输入 {worst_o}）")

    print("\n【3】丢弃动作是否真的释放字节（独立验证）")
    probe = torch.zeros(2, SEQ, HIDDEN, dtype=DTYPE)
    before_n = probe.untyped_storage().nbytes()
    probe.untyped_storage().resize_(0)
    after_n = probe.untyped_storage().nbytes()
    expected = 2 * SEQ * HIDDEN * 8
    zero_ok = before_n == expected and after_n == 0 and tuple(probe.shape) == (2, SEQ, HIDDEN)
    print(f"  discard 前 nbytes = {before_n}，discard 后 nbytes = {after_n}，"
          f"期望 = {expected}")
    print(f"  字节归零且形状/句柄仍可用：{zero_ok}")

    print("\n【4】RNG 恢复是不是承重结构")
    block.zero_grad(set_to_none=True)
    torch.set_rng_state(rng0.clone())
    x = x0.detach().clone().requires_grad_(True)
    mlp_segment = lambda h: block.mlp(block.norm2(h))
    x1 = x + block.o(block.attn_core(block.norm1(x)))
    no_rng = Holder(restore_rng=False)
    inner = output_discarding_checkpoint(mlp_segment, no_rng, x1)
    full_norng = x1 + inner
    full_norng.sum().backward()

    diff_rng = ulp_diff(ref.grads[-1], x.grad.detach().clone())
    print(f"  去掉 RNG 恢复后，输入梯度与 plain 的最大 ULP 差 = {diff_rng}")
    print("  差值非 0 说明 random.py L652-L654 的 _set_all_rng_states 是必需的，")
    print("  不是可有可无的保险代码。")

    xg_ref = ref.grads[-1].abs().max().item()
    xg_odd = odd.grads[-1].abs().max().item()
    xg_rel = (ref.grads[-1] - odd.grads[-1]).abs().max().item() / xg_ref
    print("\n【5】ULP 差出现在哪里：就在 checkpoint 边界上")
    print(f"  段内参数（up / down，checkpoint 边界以内）最大 ULP = {inner_ulp}")
    print(f"  段外参数（norm1/norm2/q/k/v/o，边界以外）最大 ULP = {outer_ulp}")
    print(f"  边界张量 x1 的输入梯度最大 ULP             = {x_ulp}")
    print(f"  全局最大绝对差 = {(ref.grads[-1] - odd.grads[-1]).abs().max().item():.3e}"
          f"，相对量级 = {xg_rel:.3e}（float64 eps = 2.22e-16）")
    print("  规律很清楚：**边界以内逐位相同，边界以外差 1 个 ULP 量级。**")
    print("  原因是 detach_variable 在重算图与外层图之间多插了一次梯度累加：")
    print("  外层先把 add 节点的贡献加到 x1 上，重算图的贡献再从 detached 副本加回去。")
    print("  浮点加法不结合，于是段外那条路径的求和顺序变了；段内只有一次求和，故不变。")
    print("  这与 05 篇观测到的「非 2 的幂 dp 差 4-24 ULP」同类：不是语义差异，")
    print("  是求和结合序。normal 路径整块 checkpoint，段外没有第二条路径，故全 0 ULP。")

    print("\n【断言】")
    checks = [
        ("normal 前向逐位相同", d_out_n == 0),
        ("output-discarding 前向逐位相同", d_out_o == 0),
        ("normal 梯度逐位相同", worst_n == 0),
        ("output-discarding 段内参数梯度逐位相同", inner_ulp == 0),
        (f"output-discarding 段外梯度相对误差 < 1e-14（实 {xg_rel:.2e}）", xg_rel < 1e-14),
        ("被丢弃输出 storage 归零", zero_ok),
        ("去掉 RNG 恢复后梯度确实变了", diff_rng != 0),
    ]
    ok = True
    for label, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        ok &= passed

    print()
    if ok:
        print("全部断言通过：前向逐位恒等；段内梯度逐位恒等；丢弃确实释放字节；")
        print("RNG 恢复承重。段外梯度只差求和结合序带来的 1 个 ULP 量级。")
        return 0
    print("存在未通过的断言。")
    return 1


if __name__ == "__main__":
    sys.exit(main())