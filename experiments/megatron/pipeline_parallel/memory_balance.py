"""A 级解析：均匀切层 vs 非均匀切层的 stage 显存不均。

回答 §4 的问题：--decoder-first-pipeline-num-layers 与
--pipeline-model-parallel-layout 各自解决第一/末 stage 的什么失衡。

本机无 GPU，本脚本只做**参数量解析**，不做字节实测，故属 A 级。
不输出任何 GPU 显存数字（那属 B 级真机实测范围）。

关键不变量：无论怎么切，各 stage 的 decoder 层数之和恒为 L。
不均衡的来源只有两项「附加负载」——
  - 第一 stage：词嵌入 embedding (V*h)
  - 末 stage：final norm + 输出头 lm_head (V*h)
均匀切按层数均分，**附加负载不均分**，所以第一/末 stage 天然更重。

源码基准 megatron-core 0.20.0 / commit 60e039626，纯 Python 无依赖。
对应文章：《Megatron-LM 深度剖析（06）：流水并行调度器》§4。
"""

from __future__ import annotations

import sys
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- 参数量模型


class Config:
    """一个 decoder-only transformer 的形状参数。

    每层参数量按 Megatron 的标准分解（均为解析式，不含 bias 与 dropout）：
      attention : qkv 投影 + 输出投影
      mlp       : fc1 + fc2（GPT-NeoX/SwiGLU 口径为 gate+up+down）
    """

    def __init__(
        self,
        num_layers: int,
        hidden: int,
        ffn_hidden: int,
        num_heads: int,
        num_query_groups: int = 1,
        vocab: int = 51200,
        swiglu: bool = False,
        tie_word_embeddings: bool = True,
    ) -> None:
        self.num_layers = num_layers
        self.hidden = hidden
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.num_query_groups = num_query_groups
        self.vocab = vocab
        self.swiglu = swiglu
        self.tie_word_embeddings = tie_word_embeddings

    @property
    def head_dim(self) -> int:
        return self.hidden // self.num_heads

    @property
    def qkv_dim(self) -> int:
        # GQA：每组一个 kv head，共享 query heads
        return self.hidden + 2 * self.num_query_groups * self.head_dim

    def layer_params(self) -> int:
        """单个 decoder 层的参数量。"""
        attn_qkv = self.hidden * self.qkv_dim
        attn_out = self.hidden * self.hidden
        attn = attn_qkv + attn_out
        if self.swiglu:
            # gate + up 各 h -> ffn，down ffn -> h
            mlp = 2 * self.hidden * self.ffn_hidden + self.ffn_hidden * self.hidden
        else:
            mlp = self.hidden * self.ffn_hidden + self.ffn_hidden * self.hidden
        # 两次 layernorm，各 2h
        norms = 4 * self.hidden
        return attn + mlp + norms

    def embedding_params(self) -> int:
        return self.vocab * self.hidden

    def final_norm_params(self) -> int:
        return 2 * self.hidden

    def lm_head_params(self) -> int:
        if self.tie_word_embeddings:
            return 0
        return self.vocab * self.hidden


# ---------------------------------------------------------------- 切分方案


def split_uniform(cfg: Config, p: int) -> List[int]:
    """均匀切：余数依次分给靠前的 stage（Megatron 的默认行为）。

    这是 Megatron 在 L 不能被 p 整除时的实际做法：靠前 stage 多拿一层。
    """
    base, rem = divmod(cfg.num_layers, p)
    return [base + (1 if i < rem else 0) for i in range(p)]


def split_decoder_first(cfg: Config, p: int, decoder_first: int) -> List[int]:
    """--decoder-first-pipeline-num-layers：手工指定第一 stage 的层数，
    其余 decoder 层在 stage 1..p-1 之间均分。"""
    assert 0 < decoder_first < cfg.num_layers, "decoder_first 必须落在 (0, L)"
    rest = cfg.num_layers - decoder_first
    # 每个 stage 至少 1 层，否则该 stage 只有附加负载、不是合法切分
    assert rest >= p - 1, f"decoder_first={decoder_first} 会让某些 stage 分到 0 层"
    base, rem = divmod(rest, p - 1)
    counts = [decoder_first]
    for i in range(p - 1):
        counts.append(base + (1 if i < rem else 0))
    return counts


def optimal_split(cfg: Config, p: int) -> Tuple[List[int], int]:
    """全局最优分配：minimize max_i stage_params，允许 0 层 stage。

    这是 --pipeline-model-parallel-layout 能达到的下界——它对每个 stage 的
    层数有完全控制权，不像 decoder_first 那样被「余数落在 stage 1」限制。
    水位法：二分最小的容量 T，使得 sum_i floor((T - extra_i)/lp) >= L。
    """
    lp = cfg.layer_params()
    extra = [0] * p
    extra[0] = cfg.embedding_params()
    extra[-1] += cfg.final_norm_params() + cfg.lm_head_params()

    def feasible(T: int) -> bool:
        return sum(max(0, (T - e) // lp) for e in extra) >= cfg.num_layers

    lo, hi = 0, lp * cfg.num_layers + max(extra) + lp
    while lo < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid + 1
    T = lo
    # 从 stage 0 开始按剩余层数贪心填满到 T
    counts = [0] * p
    left = cfg.num_layers
    for i in range(p):
        take = max(0, min((T - extra[i]) // lp, left))
        counts[i] = take
        left -= take
    # 余量补给余量最大的 stage
    i = max(range(p), key=lambda k: (T - extra[k]) // lp - counts[k])
    counts[i] += left
    return counts, T


def stage_params(cfg: Config, layer_counts: List[int]) -> List[int]:
    """按层数分配算各 stage 参数量：第 0 个 stage 带 embedding，
    最后一个带 final norm + lm_head。"""
    lp = cfg.layer_params()
    out: List[int] = []
    last = len(layer_counts) - 1
    for i, n in enumerate(layer_counts):
        total = n * lp
        if i == 0:
            total += cfg.embedding_params()
        if i == last:
            total += cfg.final_norm_params() + cfg.lm_head_params()
        out.append(total)
    return out


def imbalance(layer_counts: List[int], cfg: Config) -> Tuple[float, float, float, List[int]]:
    """返回 (max/min 比, max/mean 比, min/mean 比, 每 stage 参数量)。

    归一化成「相对平均值的倍数」以免不同 config 之间无法比大小。
    """
    sp = stage_params(cfg, layer_counts)
    mean = sum(sp) / len(sp)
    # 允许 0 层 stage（layout 串可以表达纯传参的 stage），此时 min=0，
    # max/min 无意义——瓶颈只看 max/mean。
    lo = min(sp)
    max_min = float("inf") if lo == 0 else max(sp) / lo
    min_mean = 0.0 if lo == 0 else lo / mean
    return max_min, max(sp) / mean, min_mean, sp


# ---------------------------------------------------------------- 方案比较


def compare(cfg: Config, p: int, decoder_first: int) -> None:
    print(f"\n{'=' * 78}")
    print(
        f"config: L={cfg.num_layers} h={cfg.hidden} ffn={cfg.ffn_hidden} "
        f"heads={cfg.num_heads} kv_groups={cfg.num_query_groups} "
        f"V={cfg.vocab} swiglu={cfg.swiglu} tied={cfg.tie_word_embeddings}"
    )
    print(f"pipeline degree p={p}   目标：把 max/mean 压到最接近 1.00")
    print("=" * 78)

    schemes: List[Tuple[str, List[int]]] = [("均匀切（默认）", split_uniform(cfg, p))]
    schemes.append(
        (
            f"--decoder-first-pipeline-num-layers={decoder_first}",
            split_decoder_first(cfg, p, decoder_first),
        )
    )

    print(f"\n{'方案':<44}{'层数分布':<22}{'max/min':>10}{'max/mean':>10}{'min/mean':>10}")
    for name, counts in schemes:
        ratio_maxmin, ratio_maxmean, ratio_minmean, sp = imbalance(counts, cfg)
        counts_s = ",".join(str(c) for c in counts)
        print(
            f"{name:<44}{counts_s:<22}{ratio_maxmin:>9.3f}x"
            f"{ratio_maxmean:>9.3f}x{ratio_minmean:>10.3f}x"
        )

    # 最优 decoder_first：暴力扫一遍
    best_name, best_counts, best_val = "", [], 1e9
    for df in range(1, cfg.num_layers):
        if cfg.num_layers - df < p - 1:
            break
        counts = split_decoder_first(cfg, p, df)
        _, r_maxmean, _, _ = imbalance(counts, cfg)
        if r_maxmean < best_val:
            best_val, best_name, best_counts = r_maxmean, f"decoder_first={df}", counts
    _, base_maxmean, _, _ = imbalance(split_uniform(cfg, p), cfg)
    print(
        f"\n最优 --decoder-first-pipeline-num-layers: {best_name.split('=')[1]}"
        f"  层数 {best_counts}"
        f"  max/mean {best_val:.3f}x  （均匀切 {base_maxmean:.3f}x，改善 {(base_maxmean - best_val) / base_maxmean * 100:.1f}%）"
    )


def check_a4_a5() -> List[str]:
    """对账：A4 层数守恒、A5 比值定义自洽。"""
    failures: List[str] = []
    cfg = Config(num_layers=96, hidden=4096, ffn_hidden=11008, num_heads=32)
    for p in (2, 4, 6, 8, 12, 16):
        u = split_uniform(cfg, p)
        if sum(u) != cfg.num_layers:
            failures.append(f"A4 均匀切 p={p} 层数和不守恒: {sum(u)} != {cfg.num_layers}")
        if max(u) - min(u) > 1:
            failures.append(f"A4 均匀切 p={p} 层数差 >1: {u}")
        df = cfg.num_layers // (2 * p)
        if df >= 1:
            d = split_decoder_first(cfg, p, df)
            if sum(d) != cfg.num_layers:
                failures.append(f"A4 decoder_first p={p} 层数和不守恒: {sum(d)}")
            _, r_maxmean, _, _ = imbalance(d, cfg)
            _, b_maxmean, _, _ = imbalance(u, cfg)
            # 注意：任意 decoder_first 取值可能比均匀切更差（余数落在 stage 1），
            # 那不是不变量。真正的不变量是「族内最优 <= 均匀切」，
            # 因为均匀切本身在该族内（df 取到每 stage 的层数即可复现）。
            best_in_family = min(
                imbalance(split_decoder_first(cfg, p, df), cfg)[1]
                for df in range(1, cfg.num_layers)
                if cfg.num_layers - df >= p - 1
            )
            if best_in_family > b_maxmean + 1e-9:
                failures.append(
                    f"A5 p={p} decoder_first 族内最优 {best_in_family:.4f} 反高于均匀 {b_maxmean:.4f}"
                )
            oc, _ = optimal_split(cfg, p)
            _, og, _, _ = imbalance(oc, cfg)
            if og > b_maxmean + 1e-9:
                failures.append(f"A5 p={p} 全局最优 {og:.4f} 反高于均匀 {b_maxmean:.4f}")
            if sum(oc) != cfg.num_layers:
                failures.append(f"A4 全局最优 p={p} 层数和不守恒: {sum(oc)}")
    return failures


def main() -> int:
    print("=" * 78)
    print("memory_balance.py —— 均匀切 vs 非均匀切层的 stage 不均衡（A 级解析）")
    print("源码基准 megatron-core 0.20.0 / commit 60e039626")
    print("本脚本只算参数量，不输出任何 GPU 显存数字")
    print("=" * 78)

    cfg = Config(
        num_layers=96,
        hidden=8192,
        ffn_hidden=28672,
        num_heads=64,
        vocab=51200,
        swiglu=True,
        tie_word_embeddings=True,
    )
    compare(cfg, 8, 6)
    compare(Config(num_layers=96, hidden=4096, ffn_hidden=11008, num_heads=32), 8, 6)

    print("\n[为什么 p 越大越糟]")
    cfg2 = Config(num_layers=96, hidden=8192, ffn_hidden=28672, num_heads=64,
                  vocab=51200, swiglu=True)
    print(f"{'p':>4}{'层数/stage':>12}{'均匀切':>10}{'decoder_first':>16}{'全局最优':>11}{'  全局最优层数分布':<0}")
    for p in (2, 4, 8, 16, 24, 32):
        u = split_uniform(cfg2, p)
        _, b, _, _ = imbalance(u, cfg2)
        best = 1e9
        for df in range(1, cfg2.num_layers):
            if cfg2.num_layers - df < p - 1:
                break
            _, r, _, _ = imbalance(split_decoder_first(cfg2, p, df), cfg2)
            best = min(best, r)
        oc, _ = optimal_split(cfg2, p)
        _, og, _, _ = imbalance(oc, cfg2)
        dist = ",".join(str(c) for c in oc[:6]) + ("..." if p > 6 else "")
        print(
            f"{p:>4}{cfg2.num_layers // p:>12}{b:>9.3f}x{best:>15.3f}x{og:>10.3f}x   {dist}"
        )

    print("\n[何时非均匀切才有用：判据是 embedding 与单层参数之比]")
    print("  固定 L=96, p=16 (每 stage 6 层), 扫 vocab 直到 embedding > 单层参数")
    print(f"\n{'vocab':>8}{'单层':>10}{'嵌入':>10}{'嵌入/单层':>11}{'均匀切':>10}{'全局最优':>11}{'结论':>10}")
    base = Config(num_layers=96, hidden=8192, ffn_hidden=28672, num_heads=64,
                  vocab=51200, swiglu=True)
    lp = base.layer_params()
    for vocab in (51200, 200000, 400000, 600000, 800000, 1000000):
        c = Config(num_layers=96, hidden=8192, ffn_hidden=28672, num_heads=64,
                   vocab=vocab, swiglu=True)
        p = 16
        _, b, _, _ = imbalance(split_uniform(c, p), c)
        oc, _ = optimal_split(c, p)
        _, og, _, _ = imbalance(oc, c)
        verdict = "非均匀更优" if og < b - 1e-9 else "均匀已最优"
        print(
            f"{vocab:>8}{c.layer_params() / 1e9:>9.2f}B{c.embedding_params() / 1e9:>9.2f}B"
            f"{c.embedding_params() / c.layer_params():>11.2f}{b:>9.3f}x{og:>10.3f}x{verdict:>12}"
        )

    print("\n[对账]")
    failures: List[str] = []
    for name, fn in (("A4/A5", check_a4_a5),):
        f = fn()
        print(f"  {name}: {'ok' if not f else f'!! {len(f)} 处不一致'}")
        failures += f

    print()
    print("=" * 78)
    if failures:
        print(f"对账失败 {len(failures)} 处：")
        for x in failures:
            print("  !! " + x)
    else:
        print("对账通过：层数守恒、均匀切层数差 <=1、非均匀切不劣于均匀切。")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())