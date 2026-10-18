#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""prefix-hit-rate-workload-lab：前缀命中率的工作负载形状依赖

对应文章：《推理引擎（08）：前缀命中率 —— 选型前最便宜的那次测量》

serving-07 的决策树把「前缀命中率」放在负载形状那一级：>60% 选 SGLang，
<20% 选 vLLM，中间先测。这份脚本回答三件事：

  组 1  三种典型负载（多轮对话 / RAG / Agent 循环）的命中率，
        以及**块对齐税**（vLLM 的 16 token 定长块 vs token 级最长前缀）
  组 2  缓存容量（working set）对命中率的塑造：容量曲线与「冷缓存」假象
  组 3  **口径陷阱**：token 加权 vs 请求加权、窗口定义、分层命中，
        为什么两个引擎报出的同一个「cache_hit_rate」不可直接对比

全程纯 Python / CPU、确定性输出（固定 seed）。合成的是 token 序列——
生产里先用自己的 tokenizer 把请求 tokenize 成序列，后面的算法完全相同。

运行：
    python3 run.py
    python3 run.py --selftest
"""

from __future__ import annotations

import json
import random
import sys

SEED = 20261018
BLOCK = 16  # vLLM 的 KV 块大小（token）


# --------------------------------------------------------------------------
# 合成请求序列：每个请求是一个 token id 列表；公共前缀 = 列表的公共头部。
# 长度刻意选成非 16 倍数，让块对齐税可见（真实 system prompt 的 token 数
# 也几乎不可能是 16 的倍数）。
# --------------------------------------------------------------------------


def _round_robin(sessions: list[list[int]], turns: int) -> list[int]:
    """按轮次交错：第 t 轮的所有 session 的第 t 个请求（并发服务的到达形状）。"""
    order = []
    for t in range(turns):
        for s in range(len(sessions)):
            order.append(s * turns + t)
    return order


def make_multiturn(
    rng: random.Random,
    n_sessions: int = 40,
    turns: int = 6,
    system_len: int = 510,
    turn_len: int = 130,
    uniq_base: int = 1_000_000,
) -> list[list[int]]:
    """多轮对话：同一 system prompt，逐轮追加。请求按轮次交错到达。"""
    system = [rng.randrange(uniq_base) for _ in range(system_len)]
    per_session: list[list[int]] = []
    for _ in range(n_sessions):
        toks = list(system)
        seq = []
        for _t in range(turns):
            toks += [rng.randrange(uniq_base) for _ in range(turn_len)]
            seq.append(list(toks))
        per_session.append(seq)
    flat = [r for seq in per_session for r in seq]
    order = _round_robin(list(range(n_sessions)), turns)
    return [flat[i] for i in order]


def make_rag(
    rng: random.Random,
    n_docs: int = 30,
    n_requests: int = 600,
    system_len: int = 250,
    query_len: int = 61,
    doc_len: int = 1022,
    uniq_base: int = 1_000_000,
) -> list[list[int]]:
    """RAG：system + 命中的文档块（Zipf 流行度）+ 用户问题（每次唯一）。"""
    system = [rng.randrange(uniq_base) for _ in range(system_len)]
    docs = [[rng.randrange(uniq_base) for _ in range(doc_len)] for _ in range(n_docs)]
    weights = [1.0 / (i + 1) for i in range(n_docs)]
    reqs = []
    for _ in range(n_requests):
        d = rng.choices(range(n_docs), weights=weights)[0]
        q = [rng.randrange(uniq_base) for _ in range(query_len)]
        reqs.append(system + docs[d] + q)
    return reqs


def make_agent(
    rng: random.Random,
    n_tasks: int = 20,
    steps: int = 15,
    system_len: int = 770,
    tool_len: int = 250,
    step_len: int = 98,
    uniq_base: int = 1_000_000,
) -> list[list[int]]:
    """Agent 循环：system + tool schema 每轮重复，逐步追加观测/动作。"""
    head = [rng.randrange(uniq_base) for _ in range(system_len + tool_len)]
    per_task: list[list[int]] = []
    for _ in range(n_tasks):
        toks = list(head)
        seq = []
        for _s in range(steps):
            toks += [rng.randrange(uniq_base) for _ in range(step_len)]
            seq.append(list(toks))
        per_task.append(seq)
    flat = [r for seq in per_task for r in seq]
    order = _round_robin(list(range(n_tasks)), steps)
    return [flat[i] for i in order]


# --------------------------------------------------------------------------
# 命中率仿真
# --------------------------------------------------------------------------


class _TrieNode:
    __slots__ = ("children",)

    def __init__(self) -> None:
        self.children: dict[int, "_TrieNode"] = {}


def simulate_trie(reqs: list[list[int]]) -> tuple[int, int]:
    """token 级最长前缀匹配（radix 理想形态）：trie 插入全部请求，
    查询沿 trie 走到底。返回 (hit_tokens, query_tokens)。

    query_tokens = 各请求 prompt 总长 —— 与两个引擎的口径一致
    （vLLM PrefixCacheStats.queries 是 token 数，不是请求数）。
    """
    root = _TrieNode()
    hits = 0
    queries = 0
    for toks in reqs:
        queries += len(toks)
        node = root
        depth = 0
        for t in toks:
            nxt = node.children.get(t)
            if nxt is None:
                break
            node = nxt
            depth += 1
        hits += depth
        # 插入整条路径（已存在的部分复用）
        node = root
        for t in toks:
            nxt = node.children.get(t)
            if nxt is None:
                nxt = _TrieNode()
                node.children[t] = nxt
            node = nxt
    return hits, queries


def simulate_block_aligned(
    reqs: list[list[int]], block: int = BLOCK, capacity_blocks: int | None = None
) -> tuple[int, int]:
    """块对齐匹配（vLLM APC 形态）：只有整块相同才算命中，LRU 按块淘汰。

    缓存以块为键；命中 = 请求头部连续的已缓存块数 × block。
    capacity_blocks 给出驻留块数上限（模拟显存预算）。
    """
    from collections import OrderedDict

    block_cache: "OrderedDict[tuple[int, ...], None]" = OrderedDict()
    hits = 0
    queries = 0
    for toks in reqs:
        queries += len(toks)
        nblocks = len(toks) // block
        best = 0
        for b in range(1, nblocks + 1):
            key = tuple(toks[(b - 1) * block : b * block])
            if key not in block_cache:
                break
            best = b
            block_cache.move_to_end(key)  # LRU 触达
        hits += best * block
        for b in range(nblocks):
            key = tuple(toks[b * block : (b + 1) * block])
            if key not in block_cache:
                block_cache[key] = None
        if capacity_blocks is not None:
            while len(block_cache) > capacity_blocks and block_cache:
                block_cache.popitem(last=False)
    return hits, queries


# --------------------------------------------------------------------------
# 组 1：三种负载 × 两种匹配粒度
# --------------------------------------------------------------------------


def group1() -> dict:
    print("=" * 78)
    print("组 1：三种负载的命中率 —— token 级最长前缀 vs 16 token 块对齐")
    print("=" * 78)
    rng = random.Random(SEED)
    traces = {
        "多轮对话（system 510 + 6 轮 x 130）": make_multiturn(rng),
        "RAG（system 250 + doc 1022 + query 61）": make_rag(rng),
        "Agent 循环（system+tools 1020 + 15 步 x 98）": make_agent(rng),
    }
    print(
        f"{'负载':<42}{'请求数':>7}{'token 总量':>12}"
        f"{'全 token':>10}{'块对齐':>10}{'税(pp)':>9}"
    )
    out = {}
    for name, reqs in traces.items():
        q = sum(len(r) for r in reqs)
        h_full, _ = simulate_trie(reqs)
        h_blk, _ = simulate_block_aligned(reqs)
        tax = (h_full - h_blk) / q * 100
        out[name] = {
            "requests": len(reqs),
            "query_tokens": q,
            "avg_prompt_tokens": q / len(reqs),
            "hit_full_token": h_full / q,
            "hit_block16": h_blk / q,
            "tax_pp": tax,
        }
        print(
            f"{name:<42}{len(reqs):>7}{q:>12,}"
            f"{h_full / q:>10.1%}{h_blk / q:>10.1%}{tax:>9.2f}"
        )
    print("\n（税 = token 级与块对齐的命中率差，单位百分点；"
          "理论上限 ≈ 15 token / 平均 prompt 长度）")

    # 自校验 1：块对齐命中不超过全 token（税非负）
    for name, d in out.items():
        assert d["hit_block16"] <= d["hit_full_token"] + 1e-12, name
        assert d["tax_pp"] >= -1e-9, name
    # 自校验 2：税的上界 15/平均prompt（每个请求最多亏一个不完整块）
    for name, d in out.items():
        bound = 15.0 / d["avg_prompt_tokens"] * 100
        assert d["tax_pp"] <= bound + 1e-6, (name, d["tax_pp"], bound)
    print("[自校验] 块对齐 ≤ 全 token；税 ≤ 15 token/平均 prompt（逐负载成立）")

    # 自校验 3：RAG 的闭式上界。共享前缀 = system 250 + doc 1022 = 1272，
    # query 61 唯一 → 稳态命中率上界 1272/1333（忽略冷启动与 Zipf 重叠）
    rag = traces["RAG（system 250 + doc 1022 + query 61）"]
    q_rag = sum(len(r) for r in rag)
    h_rag, _ = simulate_trie(rag)
    bound = 1272 / 1333
    assert h_rag / q_rag <= bound + 1e-9, (h_rag / q_rag, bound)
    print(f"[自校验] RAG 全 token 命中 {h_rag / q_rag:.1%} ≤ 闭式上界 {bound:.1%}"
          f"（system+doc 可共享、query 唯一）")
    print()
    return out


# --------------------------------------------------------------------------
# 组 2：容量曲线与冷缓存
# --------------------------------------------------------------------------


def group2() -> dict:
    print("=" * 78)
    print("组 2：缓存容量（working set）对命中率的塑造")
    print("=" * 78)
    rng = random.Random(SEED)
    reqs = make_rag(rng, n_docs=30, n_requests=400)

    # working set：system + 30 个 doc ≈ 250 + 30*1022 = 30,910 token（79.1 块 -> 80 块）
    working_set_tokens = 250 + 30 * 1022
    working_set_blocks = -(-working_set_tokens // BLOCK)  # 向上取整
    caps_blocks = [64, 128, 256, working_set_blocks + 1, 512, 10**9]
    print(
        f"{'容量 (块)':>12}{'容量 (token)':>14}{'占 working set':>16}"
        f"{'命中率 (块对齐)':>18}"
    )
    out = {
        "working_set_tokens": working_set_tokens,
        "working_set_blocks": working_set_blocks,
        "curve": [],
    }
    for cap in caps_blocks:
        h, q = simulate_block_aligned(reqs, capacity_blocks=cap)
        frac = min(cap / working_set_blocks, 9.99)
        out["curve"].append({"capacity_blocks": cap, "hit_rate": h / q})
        print(f"{cap:>12,}{cap * BLOCK:>14,}{frac:>15.0%}{h / q:>18.1%}")
    print(f"\nworking set ≈ {working_set_tokens:,} token"
          f"（system + 全部 30 个 doc，约 {working_set_blocks} 块）")
    print("→ 容量小于 working set 时命中率被淘汰压住；超过后饱和。")
    print("→ 常见误测：**小显存副本上测出的命中率没有代表性**，"
          "它测的是淘汰曲线不是负载曲线。")

    # 冷缓存假象：前 k 个请求 vs 全窗口
    h_all, q_all = simulate_block_aligned(reqs)
    cold_h, cold_q = simulate_block_aligned(reqs[:20])
    print(f"\n[冷缓存] 前 20 个请求命中率 {cold_h / cold_q:.1%}；"
          f"全窗口 {h_all / q_all:.1%}")
    out["cold_first20"] = cold_h / cold_q
    out["steady_state"] = h_all / q_all
    assert cold_h / cold_q < h_all / q_all, "冷缓存命中应低于稳态"

    # 自校验：容量 >= working set 时与无容量限制一致
    h_cap, _ = simulate_block_aligned(reqs, capacity_blocks=10**9)
    h_inf, _ = simulate_block_aligned(reqs, capacity_blocks=None)
    assert h_cap == h_inf
    print("[自校验] 容量 ≥ working set 时与无限制一致")
    print()
    return out


# --------------------------------------------------------------------------
# 组 3：口径陷阱
# --------------------------------------------------------------------------


def group3() -> dict:
    print("=" * 78)
    print("组 3：口径陷阱 —— 同一个「cache_hit_rate」的三种读法")
    print("=" * 78)

    # 陷阱 1：token 加权 vs 请求加权
    rng = random.Random(SEED)
    reqs = make_multiturn(rng, n_sessions=20, turns=6)
    hits, queries = simulate_trie(reqs)
    token_weighted = hits / queries
    # 逐请求命中比例（请求加权 = 平均每请求命中 token 占比）
    # 复算每请求命中长度：与 simulate_trie 相同的 trie 走查
    root = _TrieNode()
    per_req = []
    for toks in reqs:
        node = root
        depth = 0
        for t in toks:
            nxt = node.children.get(t)
            if nxt is None:
                break
            node = nxt
            depth += 1
        per_req.append(depth / len(toks))
        node = root
        for t in toks:
            nxt = node.children.get(t)
            if nxt is None:
                nxt = _TrieNode()
                node.children[t] = nxt
            node = nxt
    req_weighted = sum(per_req) / len(per_req)
    print(f"多轮对话负载：token 加权 {token_weighted:.1%} vs 请求加权 {req_weighted:.1%}")
    out = {
        "multiturn_token_weighted": token_weighted,
        "multiturn_request_weighted": req_weighted,
    }
    assert abs(token_weighted - req_weighted) > 1e-3

    # 陷阱 2：窗口定义不同（核自源码）
    print("\n窗口口径（核自源码）：")
    print("  vLLM：CachingMetrics 聚合最近 max_recent_requests=1000 个请求")
    print("        （v1/metrics/stats.py:35；指标 vllm:prefix_cache_queries/hits）")
    print("  SGLang：_CacheHitRateWindow 按时间窗，默认 15s")
    print("        （environ.py:604 SGLANG_CACHE_HIT_RATE_WINDOW_SECONDS=15.0；")
    print("         指标 sglang:cache_hit_rate）")
    print("  → 同一负载，请求率变化时两个窗口的加权结果会系统性不同。")

    # 陷阱 3：分层命中与 connector 命中
    print("\n分层口径（核自源码）：")
    print("  SGLang PrefillStats 分 log_device/host/storage_hit_tokens 与")
    print("        reprocessed_*（重算）——命中发生在哪一层分开报")
    print("  vLLM loggers.py:117 另有 connector_prefix_caching_metrics：")
    print("        PD 场景下从 prefill 传来的 KV 与本地 APC 是两本账")
    print("  → 跨引擎对比命中率之前，先确认两边报的是不是同一层。")

    print(f"\n[自校验] token 加权 ≠ 请求加权"
          f"（相差 {abs(token_weighted - req_weighted):.1%}）")
    print()
    return out


# --------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    selftest = "--selftest" in argv
    out = {"group1": group1(), "group2": group2(), "group3": group3()}
    if selftest:
        print("--selftest: 全部断言通过")
        return 0
    print("=" * 78)
    print("结论")
    print("=" * 78)
    print(
        "命中率首先由**负载形状**决定（system prompt 长度、共享文档数、轮数），\n"
        "块对齐税是次要修正（每个请求最多亏一个不完整块，≤15 token；\n"
        "prompt 越短税越重，与 01 篇 companion 实验的结论一致）。\n"
        "容量必须大于 working set，否则测到的是淘汰曲线不是负载曲线。\n"
        "上报口径（token 加权、窗口、分层）两边不同，跨引擎对比前先对齐口径。"
    )
    with open("results/summary.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已写入 results/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
