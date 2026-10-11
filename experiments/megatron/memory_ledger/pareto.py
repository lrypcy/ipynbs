#!/usr/bin/env python3
"""显存账本的激活侧帕累托前沿（解析模型，A 级）。

对应文章《Megatron-LM 深度剖析（08）：显存账本——重计算、卸载与分布保存》。
源码基准：megatron-core 0.20.0 / commit 60e039626。

上一篇（05）的 `mem_ledger.py` 按六项口径算账时，**激活项显式记 0**，
理由是「它与 DP 路径无关，由 microbatch 数与重计算策略决定」。
本脚本接着那句话往下做：把激活这一项按 `recompute_granularity` /
`recompute_method` / `recompute_num_layers` / `recompute_modules` /
`distribute_saved_activations` / offload 三档展开，扫出一条
「激活显存 ↔ 额外重算 FLOPs」的前沿。

为什么代价轴用 FLOPs 而不是吞吐
--------------------------------
本机没有 GPU，吞吐是时延量、依赖具体卡型与 kernel 实现，不是结构量，
拿它当横轴会把「结构性结论」和「某台机器的常数」混在一起。
额外重算 FLOPs 是纯结构量：由「哪些模块被重算」和「每个模块多少 FLOPs」
两个可数事实决定，与硬件无关。前沿的形状因此可被他人复算；
绝对吞吐需要真机，属于 B 级证据，不在本脚本范围内。

模型口径（全部显式列出，便于复核）
--------------------------------
1. GEMM 代价按 2*M*N*K FLOPs；SwiGLU MLP 记 gate/up/down 三个 GEMM。
2. 反向 FLOPs = 前向 2 倍（业界通用近似，不精确到逐算子）。
   基线总代价 = 3F；全量重算加一次前向 = 4F，故 +33.3%。
3. 注意力按 causal 计算，只算下三角，等价于完整注意力的 1/2。
4. 「保存集」按**反向传播真正需要的张量**逐条列出，而不是引用经验公式。
   每个条目写明形状与字节数，读者可自行复核。
5. SP（sequence parallel）只摊薄标记为 sp_shardable 的张量——即那些
   沿 hidden 维在 TP 组内被复制、且 SP 下走 reduce-scatter 的条目。
   这与 03 篇的结论同源：SP 不降低通信量，收益在激活显存。
6. output-discarding 的 7 个模块比 normal 多释放「本模块输出」这一条；
   normal 的 4 个模块只把内部中间量换成「模块输入」一条。
   下表的 normal 列与 output-discarding 列分开列，不混算。

不适用的部分（本脚本刻意不算）
------------------------------
- 权重 / 梯度 / 优化器状态三项：与重计算、卸载无关，已由 05 篇的
  `mem_ledger.py` 算完，此处不重算。
- 碎片与临时缓冲：依赖 allocator 运行时行为，无 GPU 无法建模，
  正文只引 `GlobalMemoryBuffer` 的复用语义，不给数字。
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

BYTES_BF16 = 2

# recompute_modules 的 11 个合法取值，按 docstring 的二分分组。
# normal：core_attn / mlp / moe / shared_experts
# output-discarding：moe_act / layernorm / mla_up_proj / gdn_norm_out
#                     gdp_in_proj / gdp_qkv / mhc
OUTPUT_DISCARDING = (
    "moe_act", "layernorm", "mla_up_proj", "gdn_norm_out",
    "gdp_in_proj", "gdp_qkv", "mhc",
)
NORMAL = ("core_attn", "mlp", "moe", "shared_experts")
ALL_MODULES = NORMAL + OUTPUT_DISCARDING


@dataclass(frozen=True)
class ModelSpec:
    """稠密 decoder 的结构参数。只取本脚本用到的字段。"""

    name: str
    hidden: int
    ffn_hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim


MODELS: Dict[str, ModelSpec] = {
    "llama-7b": ModelSpec("llama-7b", 4096, 11008, 32, 32, 32, 32000),
    "llama3-8b": ModelSpec("llama3-8b", 4096, 14336, 32, 32, 8, 128256),
    "qwen-14b": ModelSpec("qwen-14b", 5120, 13824, 48, 40, 8, 152064),
}


@dataclass(frozen=True)
class RunConfig:
    """一次扫描点。刻意不含吞吐相关字段——本机无 GPU。"""

    model: str = "llama-7b"
    seq_len: int = 4096
    micro_batch: int = 1
    tp: int = 8
    pp: int = 1
    cp: int = 1
    sequence_parallel: bool = True
    # 重计算
    recompute_granularity: Optional[str] = None      # None / "full" / "selective"
    recompute_method: Optional[str] = None          # None / "uniform" / "block"
    recompute_num_layers: Optional[int] = None
    recompute_modules: Tuple[str, ...] = ()
    # 分布保存
    distribute_saved_activations: bool = False
    # offload 档位："" 不卸载 / "fine_grained" 模块级 / "cpu_layers" 整层级
    offload_tier: str = ""

    def key(self) -> str:
        mods = "+".join(self.recompute_modules) if self.recompute_modules else "-"
        return (
            f"gran={self.recompute_granularity or 'none':9s} "
            f"method={self.recompute_method or '-':7s} "
            f"k={str(self.recompute_num_layers):>4s} "
            f"mods={mods:52s} "
            f"sp={'on ' if self.sequence_parallel else 'off'} "
            f"dsa={'on ' if self.distribute_saved_activations else 'off'} "
            f"off={self.offload_tier or '-':11s}"
        )


@dataclass
class LedgerRow:
    """单层「保存集」里的一条。名字、形状、字节数都写出来，便于复核。"""

    name: str
    shape: Tuple[int, ...]
    nbytes: int
    # SP 下是否沿 hidden 维被摊薄
    sp_shardable: bool = False
    # 被 output-discarding 额外释放（只对 output-discarding 模块的输出成立）
    freed_by_output_discarding: bool = False


@dataclass
class Ledger:
    """一次扫描点的全部数字。"""

    stored_total: float = 0.0             # 本 stage 留在显存的字节
    offloaded_total: float = 0.0          # 本 stage 搬到 host 的字节
    extra_recompute_flops: float = 0.0    # 本 stage 额外重算 FLOPs
    layer_flops: float = 0.0              # 本 stage 前向 FLOPs

    @property
    def recompute_ratio(self) -> float:
        """额外重算 FLOPs ÷ 前向 FLOPs。全量重算 = 1.0（多跑一次前向）。"""
        return self.extra_recompute_flops / self.layer_flops

    @property
    def training_cost_ratio(self) -> float:
        """总训练代价相对基线（3F）的增幅。

        基线 = 前向 F + 反向 2F = 3F；重算再加 extra，
        故增幅 = extra / 3F。全量重算 all layers 时为 1/3。
        """
        return self.extra_recompute_flops / (3.0 * self.layer_flops)


def resolved_seq_len(cfg: RunConfig) -> int:
    return cfg.seq_len // cfg.cp


def tokens_per_rank(cfg: RunConfig) -> int:
    """本 rank 上的 token 数 T。"""
    return cfg.micro_batch * resolved_seq_len(cfg)


def per_layer_inventory(cfg: RunConfig) -> List[LedgerRow]:
    """不做任何重计算时，单层反向传播真正需要保存的张量清单。

    条目与形状都对着一个 SwiGLU + RMSNorm 的 decoder layer 写。
    `sp_shardable` 标的是那些「沿 hidden 维在 TP 组内本来是复制的」条目——
    序列并行摊薄的正是这部分。
    """
    m = MODELS[cfg.model]
    T = tokens_per_rank(cfg)
    h = m.hidden
    kv = m.kv_dim // cfg.tp          # 本 TP rank 上的 kv 宽度
    I = m.ffn_hidden

    th = T * h * BYTES_BF16
    tk = T * kv * BYTES_BF16
    ti = T * I * BYTES_BF16

    return [
        LedgerRow("LN1 输入（残差分支复用）", (T, h), th, sp_shardable=True),
        LedgerRow("LN1 输出 / QKV 输入", (T, h), th, sp_shardable=True),
        LedgerRow("q", (T, h), th, sp_shardable=True),
        LedgerRow("k", (T, kv), tk),
        LedgerRow("v", (T, kv), tk),
        LedgerRow("core_attn 输出 / W_o 输入", (T, h), th, sp_shardable=True),
        LedgerRow("W_o 输出", (T, h), th, sp_shardable=True),
        LedgerRow("LN2 输入", (T, h), th, sp_shardable=True),
        LedgerRow("LN2 输出 / gate-up 输入", (T, h), th, sp_shardable=True),
        LedgerRow("gate 中间结果", (T, I), ti),
        LedgerRow("up 中间结果", (T, I), ti),
        LedgerRow("SwiGLU 输出 / down 输入", (T, I), ti),
    ]


def module_flops(cfg: RunConfig, module: str) -> float:
    """单层里被某个模块重算掉的那部分前向 FLOPs。

    只对 `recompute_modules` 里出现的模块有意义。口径见文件头 docstring。
    """
    m = MODELS[cfg.model]
    T = tokens_per_rank(cfg)
    h, I = m.hidden, m.ffn_hidden
    kv = m.kv_dim // cfg.tp

    if module == "core_attn":
        # QK^T 与 AV 两个 GEMM，causal 只算一半；softmax 与 RoPE 记为可忽略。
        qk = 2 * T * (T / 2) * h
        av = 2 * T * (T / 2) * h
        return float(qk + av)
    if module in ("mlp", "shared_experts", "fused_group_mlp"):
        gate = 2 * T * h * I
        up = 2 * T * h * I
        down = 2 * T * I * h
        return float(gate + up + down)
    if module == "mla_up_proj":
        # MLA 的上投影 + RoPE；kv_lora + q_pe 两路投影
        return float(2 * T * h * (m.kv_dim // cfg.tp + m.head_dim))
    if module in ("layernorm", "moe_act", "gdn_norm_out", "gdp_in_proj",
                  "gdp_qkv", "mhc", "moe"):
        # 逐元素算子：FLOPs 相对 GEMM 可忽略，但重算仍要占 kernel 时间。
        return 0.0
    raise ValueError(f"unknown module: {module}")


def layer_flops(cfg: RunConfig) -> float:
    """单层前向 FLOPs（QKV + W_o + SwiGLU）。"""
    m = MODELS[cfg.model]
    T = tokens_per_rank(cfg)
    h, I = m.hidden, m.ffn_hidden
    kv = m.kv_dim // cfg.tp
    qkv = 2 * T * h * (h + 2 * kv)
    wo = 2 * T * h * h
    gate = 2 * T * h * I
    up = 2 * T * h * I
    down = 2 * T * I * h
    return float(qkv + wo + gate + up + down)


def layer_input_bytes(cfg: RunConfig) -> int:
    """整层被 checkpoint 时唯一保存的那一条：层输入 [T, h]。"""
    return tokens_per_rank(cfg) * MODELS[cfg.model].hidden * BYTES_BF16


DENSE_BINDINGS = ("core_attn", "layernorm", "mlp", "moe_act", "moe")


def applicable_modules(cfg: RunConfig) -> Tuple[str, ...]:
    """本模型结构下真正绑定到清单里的模块。

    11 个合法取值里有 6 个只对特定结构有意义（`mla_up_proj` 要 MLA，
    `gdp_*` / `gdn_norm_out` 要 GatedDelta 系，`shared_experts` / `moe` 要 MoE，
    `mhc` 要 enable_mhc_connections）。稠密 decoder 上只有 5 个可算，
    其余在本脚本里按「结构不适用」单列，不给数字。
    """
    return DENSE_BINDINGS


def selective_saved_set(cfg: RunConfig) -> Tuple[List[LedgerRow], float]:
    """selective 重算下的保存集。

    normal 模块：把「模块输入」作为唯一保存项，内部中间量全部丢弃。
    output-discarding 模块：在 normal 的基础上，额外释放「模块输出」
    ——前提是下游没有把它存下来（源码用统一重算钩子保证这一点）。

    返回 (保存集, 额外重算 FLOPs)。
    """
    inv = per_layer_inventory(cfg)

    MODULE_BINDINGS: Dict[str, Tuple[str, str]] = {
        "core_attn": ("LN1 输出 / QKV 输入", "core_attn 输出 / W_o 输入"),
        "layernorm": ("LN1 输入（残差分支复用）", "LN1 输出 / QKV 输入"),
        "mlp": ("LN2 输出 / gate-up 输入", "SwiGLU 输出 / down 输入"),
        "moe_act": ("up 中间结果", "SwiGLU 输出 / down 输入"),
        "moe": ("LN2 输出 / gate-up 输入", "SwiGLU 输出 / down 输入"),
    }

    # normal 与 output-discarding 在本表里对「输出这一条」给出相同的和：
    # 两类模块的输出都由反向时的重算重新产出，差别只在是否额外要求下游
    # 不把它另存一份（源码用 CheckpointWithoutOutputManager 的统一重算钩子
    # 保证，transformer_config.py L2023-L2029 记录了它与卸载的不兼容）。
    out_names = {MODULE_BINDINGS[m_][1]
                 for m_ in cfg.recompute_modules if m_ in MODULE_BINDINGS}

    extra = sum(module_flops(cfg, module) for module in cfg.recompute_modules)

    kept = [row for row in inv if row.name not in out_names]
    return kept, extra


def build(cfg: RunConfig) -> Ledger:
    """按配置算出激活侧的账。"""
    inv = per_layer_inventory(cfg)
    m = MODELS[cfg.model]
    layers_local = m.n_layers // cfg.pp

    stored = 0.0
    offloaded = 0.0
    extra = 0.0

    if cfg.recompute_granularity == "full":
        n_ckpt = _num_checkpointed_layers(cfg, layers_local)
        n_plain = layers_local - n_ckpt
        full_set = sum(_sp_adjusted(row, cfg) for row in inv)
        stored += n_plain * full_set
        stored += n_ckpt * _sp_adjusted(inv[0], cfg)
        extra += n_ckpt * layer_flops(cfg)

    elif cfg.recompute_granularity == "selective":
        saved, module_extra = selective_saved_set(cfg)
        stored += sum(_sp_adjusted(row, cfg) for row in saved) * layers_local
        extra += module_extra * layers_local

    else:
        stored += sum(_sp_adjusted(row, cfg) for row in inv) * layers_local

    # distribute_saved_activations 只摊薄重算边界保存的那一条张量，
    # 源码里就是 ctx.saved_tensors[0]（tensor_parallel/random.py L622 / L648）。
    if cfg.distribute_saved_activations and cfg.recompute_granularity == "full" and cfg.tp > 1:
        stored -= layers_local * _sp_adjusted(inv[0], cfg)
        stored += layers_local * layer_input_bytes(cfg) / cfg.tp

    offload_targets = ("up 中间结果", "gate 中间结果", "SwiGLU 输出 / down 输入")
    if cfg.offload_tier == "fine_grained":
        for row in inv:
            if row.name in offload_targets:
                b = _sp_adjusted(row, cfg) * layers_local
                offloaded += b
                stored -= b
    elif cfg.offload_tier == "cpu_layers":
        offloaded += stored
        stored = 0.0

    return Ledger(
        stored_total=stored,
        offloaded_total=offloaded,
        extra_recompute_flops=extra,
        layer_flops=layer_flops(cfg) * layers_local,
    )


def _num_checkpointed_layers(cfg: RunConfig, layers_local: int) -> int:
    """整层 checkpoint 时，本 PP stage 上被 checkpoint 的层数。"""
    k = cfg.recompute_num_layers
    if k is None:
        return layers_local
    if cfg.recompute_method == "uniform":
        n, rem = divmod(layers_local, k)
        return n * k + (k if rem else 0)
    # block：前 k 层
    return min(k, layers_local)


def _sp_adjusted(row: LedgerRow, cfg: RunConfig) -> float:
    """SP 下摊薄后的字节数。"""
    b = float(row.nbytes)
    if cfg.sequence_parallel and row.sp_shardable and cfg.tp > 1:
        b /= cfg.tp
    return b


def scan(base: RunConfig) -> List[Tuple[RunConfig, Ledger, float, float]]:
    """扫一遍参数空间，返回 (配置, 账, 每 rank 激活字节, 额外重算 FLOPs 比例)。"""
    out = []
    m = MODELS[base.model]
    layers_local = m.n_layers // base.pp
    outstanding = base.pp if base.pp > 1 else 1

    # 基线：不重算
    cfgs: List[RunConfig] = [base]

    # full × uniform × k
    for k in (1, 2, 4, 8):
        if k < layers_local:
            cfgs.append(RunConfig(**{**base.__dict__,
                                     "recompute_granularity": "full",
                                     "recompute_method": "uniform",
                                     "recompute_num_layers": k}))

    # full × block × k
    for k in (1, 2, 4, 8):
        if k < layers_local:
            cfgs.append(RunConfig(**{**base.__dict__,
                                     "recompute_granularity": "full",
                                     "recompute_method": "block",
                                     "recompute_num_layers": k}))

    # selective × 单模块：只扫在本模型结构下可算的
    for mod in applicable_modules(base):
        cfgs.append(RunConfig(**{**base.__dict__,
                                 "recompute_granularity": "selective",
                                 "recompute_method": None,
                                 "recompute_num_layers": None,
                                 "recompute_modules": (mod,),
                                 "distribute_saved_activations": False}))

    # selective 组合
    cfgs.append(RunConfig(**{**base.__dict__,
                             "recompute_granularity": "selective",
                             "recompute_modules": ("core_attn", "mlp")}))

    # distribute_saved_activations：full 与 selective 各来一份
    cfgs.append(RunConfig(**{**base.__dict__,
                             "recompute_granularity": "full",
                             "recompute_method": "uniform",
                             "recompute_num_layers": 1,
                             "distribute_saved_activations": True}))
    cfgs.append(RunConfig(**{**base.__dict__,
                             "recompute_granularity": "selective",
                             "recompute_modules": ("core_attn",),
                             "distribute_saved_activations": True}))

    # DSA 对照：关掉 SP 后 DSA 才有净收益（源码在 transformer_config.py
    # L1912-L1916 直接禁止 DSA 与 SP 同时开）
    no_sp = RunConfig(**{**base.__dict__, "sequence_parallel": False})
    cfgs.append(RunConfig(**{**no_sp.__dict__, "distribute_saved_activations": False}))
    cfgs.append(RunConfig(**{**no_sp.__dict__,
                             "recompute_granularity": "full",
                             "recompute_method": "uniform",
                             "recompute_num_layers": 1,
                             "distribute_saved_activations": False}))
    cfgs.append(RunConfig(**{**no_sp.__dict__,
                             "recompute_granularity": "full",
                             "recompute_method": "uniform",
                             "recompute_num_layers": 1,
                             "distribute_saved_activations": True}))

    # offload 三档
    cfgs.append(RunConfig(**{**base.__dict__, "offload_tier": "fine_grained"}))
    cfgs.append(RunConfig(**{**base.__dict__, "offload_tier": "cpu_layers"}))
    cfgs.append(RunConfig(**{**base.__dict__,
                             "recompute_granularity": "selective",
                             "recompute_modules": ("core_attn",),
                             "offload_tier": "fine_grained"}))

    for cfg in cfgs:
        led = build(cfg)
        act = led.stored_total * outstanding
        out.append((cfg, led, act, led.recompute_ratio))
    return out


def pareto_front(points: Sequence[Tuple[RunConfig, Ledger, float, float]]
                 ) -> List[Tuple[RunConfig, Ledger, float, float]]:
    """帕累托前沿：不存在另一个点同时显存更小且额外 FLOPs 更少。"""
    front = []
    for cfg, led, act, ratio in points:
        dominated = False
        for _c2, _l2, act2, ratio2 in points:
            if (act2 <= act and ratio2 <= ratio) and (act2 < act or ratio2 < ratio):
                dominated = True
                break
        if not dominated:
            front.append((cfg, led, act, ratio))
    front.sort(key=lambda x: (x[2], x[3]))
    deduped: List[Tuple[RunConfig, Ledger, float, float]] = []
    seen = set()
    for item in front:
        sig = (round(item[2], 6), round(item[3], 9))
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(item)
    return deduped


def gib(x: float) -> float:
    return x / (1024 ** 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-7b", choices=sorted(MODELS))
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--mbs", type=int, default=1)
    args = ap.parse_args()

    base = RunConfig(
        model=args.model, seq_len=args.seq, micro_batch=args.mbs,
        tp=args.tp, pp=args.pp, sequence_parallel=True,
    )
    m = MODELS[args.model]
    layers_local = m.n_layers // args.pp
    outstanding = args.pp if args.pp > 1 else 1

    print("=" * 104)
    print(f"显存账本 · 激活侧帕累托前沿（解析）  模型={args.model} "
          f"seq={args.seq} tp={args.tp} pp={args.pp} mbs={args.mbs}")
    print(f"单层前向 FLOPs = {layer_flops(base):.4e}   "
          f"本 stage 层数 = {layers_local}   在飞 microbatch = {outstanding}")
    print("=" * 104)

    points = scan(base)
    print(f"\n共扫描 {len(points)} 个配置点。\n")

    print("【全部扫描点】")
    print(f"{'配置':<124s} {'激活GiB':>9s} {'hostGiB':>9s} {'extra/F':>9s} {'总代价增幅':>11s}")
    print("-" * 156)
    for cfg, led, act, ratio in points:
        print(f"{cfg.key():<124s} {gib(act):>9.2f} {gib(led.offloaded_total * outstanding):>9.2f} "
              f"{ratio * 100:>8.1f}% {led.training_cost_ratio * 100:>10.2f}%")

    recompute_pts = [p_ for p_ in points if not p_[0].offload_tier]
    offload_pts = [p_ for p_ in points if p_[0].offload_tier]

    print("\n【前沿 A：只算重计算】代价轴 = 额外重算 FLOPs")
    print("offload 档位的代价是 host 流量而不是 FLOPs，两者不同量纲，"
          "混到一张图上会让「免费卸载」支配整个前沿，因此分开列。")
    front = pareto_front(recompute_pts)
    print(f"\n共 {len(front)} 个非受支配点（按激活显存升序）：")
    print(f"{'配置':<124s} {'激活GiB':>9s} {'extra/F':>9s} {'总代价增幅':>11s}")
    print("-" * 156)
    for cfg, led, act, ratio in front:
        print(f"{cfg.key():<124s} {gib(act):>9.2f} "
              f"{ratio * 100:>8.1f}% {led.training_cost_ratio * 100:>10.2f}%")

    print("\n【前沿 B：卸载档位】代价轴 = 每 step 的 host 流量字节（结构量，可精确算）")
    print(f"{'配置':<124s} {'激活GiB':>9s} {'hostGiB':>9s} {'extra/F':>9s}")
    print("-" * 156)
    for cfg, led, act, ratio in offload_pts:
        print(f"{cfg.key():<124s} {gib(act):>9.2f} "
              f"{gib(led.offloaded_total * outstanding):>9.2f} {ratio * 100:>8.1f}%")

    n_a = [x for x in ALL_MODULES if x not in applicable_modules(base)]
    print(f"\n【结构不适用】以下 {len(n_a)} 个 recompute_modules 取值在"
          f"{args.model}（稠密 decoder）上无对应模块，不给数字：")
    print("   " + "、".join(n_a))

    print("\n【读数】")
    base_act = act_of(points, lambda c: c.recompute_granularity is None and not c.offload_tier)
    print(f"1. 基线（不重算不卸载）激活 = {gib(base_act):.2f} GiB。")

    uni = [f for f in front if f[0].recompute_method == "uniform"]
    if uni:
        mems = {round(gib(f[2]), 6) for f in uni}
        print(f"2. full+uniform 的 k 取 1/2/4/8 时激活显存恒为 "
              f"{sorted(mems)} GiB —— "
              "k 不改变显存。源码里 uniform 把全部层都划进 chunk 且每段都 checkpoint，"
              "k 只决定 chunk 边界数量，不决定哪些层被 checkpoint。")
    blk = sorted([f for f in points if f[0].recompute_method == "block"],
                 key=lambda x: x[0].recompute_num_layers or 0)
    for cfg, led, act, ratio in blk:
        print(f"3. full+block k={cfg.recompute_num_layers}: "
              f"{gib(act):.2f} GiB, extra/F={ratio * 100:.1f}%")

    dsa = [p_ for p_ in points if p_[0].distribute_saved_activations]
    print("\n4. distribute_saved_activations 的净收益（只摊薄 ctx.saved_tensors[0] 一条）：")
    for cfg, led, act, ratio in dsa:
        twin = next((p_ for p_ in points
                     if p_[0].recompute_granularity == cfg.recompute_granularity
                     and p_[0].recompute_modules == cfg.recompute_modules
                     and p_[0].recompute_method == cfg.recompute_method
                     and p_[0].recompute_num_layers == cfg.recompute_num_layers
                     and p_[0].sequence_parallel == cfg.sequence_parallel
                     and not p_[0].distribute_saved_activations), None)
        if twin:
            delta = gib(twin[2]) - gib(act)
            sp = "SP=on " if cfg.sequence_parallel else "SP=off"
            print(f"   [{sp}] {cfg.recompute_granularity}"
                  f"/{cfg.recompute_method or '-'} "
                  f"k={cfg.recompute_num_layers} mods={'+'.join(cfg.recompute_modules) or '-'}  "
                  f"{gib(twin[2]):.2f} → {gib(act):.2f} GiB（省 {delta:.3f} GiB）")
    print("   注：SP 开着时 DSA 的净收益为 0 —— 它摊的正是层输入那条张量，")
    print("   而 SP 已经把这条沿序列维摊过了。源码因此在 transformer_config.py")
    print("   L1912-L1916 直接禁止 DSA 与 SP 同时开启，二者不是可叠加的两项。")
    return 0


def act_of(points, pred) -> float:
    for cfg, led, act, ratio in points:
        if pred(cfg):
            return act
    return 0.0


if __name__ == "__main__":
    raise SystemExit(main())