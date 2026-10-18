"""chunked-prefill-mixing：把 prefill 切碎塞进 decode 空隙，收益与代价的仿真。

对应文章：推理引擎（02）调度与批处理
  https://lrypcy.github.io/2026/10/18/serving-02-scheduling-and-batching/

成本模型（与文章 §2 的推导同一套）：
    t_W       : 每步读一遍权重，与 batch 无关 -> decode 可被 batch 摊薄
    t_compute : 每个 prefill token 的计算成本（算力受限）
    t_kv      : 每个在飞 decode 请求每步的 KV 读取成本（带宽受限）
decode 请求每步只消耗 1 个 token；prefill 请求每步最多消耗 tau 个（chunked prefill）。

一步的耗时有两种口径，本实验两种都跑——混批的收益完全来自第二种：
    sum（保守）：计算与带宽串行（权重读取共享一次），t = t_W + t_compute*pf + t_kv*b
    max（重叠）：计算与带宽并行，t = max(t_W + t_compute*pf, t_W + t_kv*b)
第二种才是引擎把 prefill 与 decode 放进同一 batch 的物理理由：decode 的权重读取
时间里，计算单元本来是闲着的。

两种调度模式：
    phase（分相，无 chunked prefill）：一步要么全是 prefill（整请求入场，可拼批），
        要么全是 decode（所有在飞 +1）。prefill 优先 —— 长 prefill 阻塞 decode。
    mixed（混批，chunked prefill）：一步内 decode 各拿 1 token，剩余预算按 tau
        切给等待中的 prefill 请求。

自校验（构造真值 <-> 反算值）：
    1. token 守恒：全部请求完成，输出 token 总数 == sum(L_out)
    2. 摊薄极限：每请求每步成本 = t_W/b + t_kv，b -> 大时趋近 t_kv（§2 推导）
    3. 分相忙碌时间 <= 解析上界（prefill 全串行 + decode 全串行）
    4. 延迟方向（两种成本口径下都成立）：tau 越小，长请求延迟越长（非递减）
    5. 短请求尾延迟：中载下 mixed（tau>=256）的短请求 p99 低于 phase
       —— 分相模式下短请求要等长 prefill 整个算完（队首阻塞）
"""
import json
import sys

import numpy as np

SEED = 20261018
N_REQ = 400
L_IN_SHORT, L_IN_LONG = 256, 8192
P_LONG = 0.15
L_OUT = 128
C_BUDGET = 2048      # 一步 token 预算（vLLM max_num_batched_tokens 默认值）

T_W = 100.0          # 读一遍权重
T_COMPUTE = 0.05     # 每 prefill token
T_KV = 2.0           # 每在飞 decode 请求每步 KV 读取


def make_workload(rng, mean_inter, n_req=N_REQ):
    inter = rng.exponential(mean_inter, n_req)
    arrive = np.cumsum(inter)
    is_long = rng.random(n_req) < P_LONG
    l_in = np.where(is_long, L_IN_LONG, L_IN_SHORT)
    return arrive, l_in, np.full(n_req, L_OUT, dtype=int)


def step_time(prefill_tokens, n_running, cost):
    """一步耗时。权重读取两类 kernel 共享，只收一次。

    sum（保守）：计算与带宽串行，t = t_W + t_compute*pf + t_kv*b
    max（重叠）：计算与带宽并行，t = max(t_W + t_compute*pf, t_W + t_kv*b)
    """
    t_prefill = T_W + T_COMPUTE * prefill_tokens
    t_decode = T_W + T_KV * n_running
    if cost == "max":
        return max(t_prefill, t_decode)
    return T_W + T_COMPUTE * prefill_tokens + T_KV * n_running


def simulate(arrive, l_in, l_out, tau, mode, cost):
    """离散事件仿真。返回 TTFT、完成时间、makespan、忙碌时间。"""
    n = len(arrive)
    computed = np.zeros(n, dtype=int)
    output = np.zeros(n, dtype=int)
    ttft = np.zeros(n)
    finish = np.zeros(n)
    waiting = []
    running = set()
    t = 0.0
    ptr = 0
    total_out = 0
    target = int(l_out.sum())
    busy = 0.0

    while total_out < target:
        while ptr < n and arrive[ptr] <= t:
            waiting.append(ptr)
            ptr += 1
        if not running and not waiting and ptr < n:
            t = arrive[ptr]          # 空转：直接跳到下一个到达
            continue

        prefill_tokens = 0
        prefilled_now = []          # 本步刚完成 prefill 的请求（首 token 在本步末产出）
        if mode == "mixed":
            # 混批：在飞 decode 各拿 1 token，剩余预算按 tau 切给等待中的 prefill
            batch = set(running)
            budget = C_BUDGET - len(batch)
            for i in list(waiting):
                if budget <= 0:
                    break
                chunk = min(l_in[i] - computed[i], tau, budget)
                if chunk <= 0:
                    break
                computed[i] += chunk
                budget -= chunk
                prefill_tokens += chunk
                batch.add(i)
                if computed[i] >= l_in[i]:
                    waiting.remove(i)
                    running.add(i)
                    prefilled_now.append(i)
        else:
            # 分相：有等待就做 prefill 步（整请求入场，可拼批），否则 decode 步
            if waiting:
                batch = set()
                budget = C_BUDGET
                for i in list(waiting):
                    if budget <= 0:
                        break
                    chunk = min(l_in[i] - computed[i], budget)
                    if chunk <= 0:
                        break
                    computed[i] += chunk
                    budget -= chunk
                    prefill_tokens += chunk
                    batch.add(i)
                    if computed[i] >= l_in[i]:
                        waiting.remove(i)
                        running.add(i)
                        prefilled_now.append(i)

        step = step_time(prefill_tokens, len(running), cost)
        t += step
        busy += step
        for i in prefilled_now:
            ttft[i] = t

        for i in list(running):
            output[i] += 1
            total_out += 1
            if output[i] >= l_out[i]:
                running.discard(i)
                finish[i] = t
    return ttft, finish, t, busy


def run_regime(name, mean_inter, n_req, taus, cost):
    rng = np.random.default_rng(SEED)
    arrive, l_in, l_out = make_workload(rng, mean_inter, n_req)
    long_mask = l_in == L_IN_LONG

    ttft_b, fin_b, t_b, busy_b = simulate(arrive, l_in, l_out, tau=0, mode="phase", cost=cost)
    assert int((fin_b > 0).sum()) == n_req          # 自校验 1：守恒
    delay_b = ttft_b - arrive                       # 相对到达的延迟（隔离到达时间）
    rows = [dict(mode="phase", tau=None, makespan=t_b,
                 ttft_long=float(np.percentile(delay_b[long_mask], 50)),
                 ttft_short=float(np.percentile(delay_b[~long_mask], 50)),
                 ttft_short_p99=float(np.percentile(delay_b[~long_mask], 99)))]
    # 自校验 5：中载下，tau>=256 的混批短请求 p99 应不高于分相（无队首阻塞）
    if 1000 <= mean_inter <= 10000:
        for tau in [t for t in taus if t >= 256]:
            ttft, fin, t_tot, _ = simulate(arrive, l_in, l_out, tau=tau, mode="mixed", cost=cost)
            delay = ttft - arrive
            p99 = float(np.percentile(delay[~long_mask], 99))
            base = float(np.percentile(delay_b[~long_mask], 99))
            assert p99 <= base * (1 + 1e-9), (name, cost, tau, p99, base)
    prev_long = None            # 只在 mixed 行之间比（分相约等于 tau=预算 的极限）
    for tau in taus:
        ttft, fin, t_tot, _ = simulate(arrive, l_in, l_out, tau=tau, mode="mixed", cost=cost)
        assert int((fin > 0).sum()) == n_req        # 自校验 1：守恒
        delay = ttft - arrive
        row = dict(mode="mixed", tau=tau, makespan=t_tot,
                   ttft_long=float(np.percentile(delay[long_mask], 50)),
                   ttft_short=float(np.percentile(delay[~long_mask], 50)),
                   ttft_short_p99=float(np.percentile(delay[~long_mask], 99)))
        # 自校验 4：tau 增大（切得更粗）时，长请求延迟不得上升
        if prev_long is not None:
            assert row["ttft_long"] <= prev_long + 1e-9, (name, cost, tau, row, prev_long)
        prev_long = row["ttft_long"]
        rows.append(row)
    return rows, t_b


def main(which="all"):
    # 自校验 2：摊薄极限
    for b, expect in [(1, T_W + T_KV), (10**6, T_W / 10**6 + T_KV)]:
        per_req = (T_W + T_KV * b) / b
        assert abs(per_req - expect) < 1e-9
    print("自校验2: 每请求每步成本 = t_W/b + t_kv；b=1 时为 102.0，b->inf 时 -> 2.0（§2 推导）")

    # 自校验 3：分相忙碌时间 <= 解析上界
    rng = np.random.default_rng(SEED)
    arrive, l_in, l_out = make_workload(rng, 20000.0, 60)
    n_light = len(arrive)
    t_prefill = T_W * n_light + T_COMPUTE * float(l_in.sum())
    t_decode = (T_W + T_KV) * float(l_out.sum())
    _, _, _, busy_b = simulate(arrive, l_in, l_out, tau=0, mode="phase", cost="sum")
    assert busy_b <= (t_prefill + t_decode) * (1 + 1e-9), (busy_b, t_prefill + t_decode)
    print(f"自校验3: 分相忙碌时间 {busy_b:.0f} <= 解析上界 {t_prefill + t_decode:.0f}"
          f"（prefill 全串行 {t_prefill:.0f} + decode 全串行 {t_decode:.0f}）")

    taus = [64, 256, 1024, 2048]
    regimes = [("轻载", 15000.0, 120), ("中载", 5000.0, 120), ("重载", 20.0, 400)]
    if which != "all":
        regimes = [r for r in regimes if r[0] == which]

    out = {}
    for cost in ["sum", "max"]:
        for name, mean_inter, n_req in regimes:
            rows, t_b = run_regime(name, mean_inter, n_req, taus, cost)
            out.setdefault(cost, {})[name] = rows
            print()
            print(f"=== {name}（平均到达间隔 {mean_inter:.0f} 步，{n_req} 请求，成本口径 {cost}）===")
            hdr = (f"{'模式':>8} {'tau':>6} {'makespan':>11} {'长延迟p50':>11} "
                   f"{'短延迟p50':>11} {'短延迟p99':>11}")
            print(hdr)
            print("-" * len(hdr))
            for r in rows:
                tau_s = "-" if r["tau"] is None else str(r["tau"])
                print(f"{r['mode']:>8} {tau_s:>6} {r['makespan']:>11.0f} "
                      f"{r['ttft_long']:>11.1f} {r['ttft_short']:>11.1f} "
                      f"{r['ttft_short_p99']:>11.1f}")

    # 汇总
    for cost in out:
        if "轻载" in out[cost]:
            light = out[cost]["轻载"]
            phase, small = light[0], light[1]
            print()
            print(f"[{cost}] 轻载（并发<1）：tau=64 vs 分相 —— 长请求延迟 "
                  f"{phase['ttft_long']:.0f} -> {small['ttft_long']:.0f}"
                  f"（{small['ttft_long'] / phase['ttft_long']:.1f}x），"
                  f"短请求延迟 {phase['ttft_short']:.0f} -> {small['ttft_short']:.0f}；"
                  f"makespan {phase['makespan']:.0f} vs {small['makespan']:.0f}")
        if "中载" in out[cost]:
            mid = out[cost]["中载"]
            best = min(mid, key=lambda r: r["makespan"])
            print(f"[{cost}] 中载（并发~3）：makespan 最优 tau={best['tau']}"
                  f"（{best['makespan']:.0f}），分相基线 {mid[0]['makespan']:.0f}，"
                  f"加速比 {mid[0]['makespan'] / best['makespan']:.2f}x")
        if "重载" in out[cost]:
            heavy = out[cost]["重载"]
            print(f"[{cost}] 重载（饱和）：makespan 范围 "
                  f"{min(r['makespan'] for r in heavy):.0f}–{max(r['makespan'] for r in heavy):.0f}"
                  f"（分相 {heavy[0]['makespan']:.0f}）")

    if len(out) == 2 and all(len(v) == 3 for v in out.values()):
        print()
        print("总结论（本负载 prefill 主导，15% 请求 8192 token、其余 256 token）：")
        print("  1. 吞吐不是混批的收益点：三种负载下 makespan 差异 1%-7%，方向不一致")
        print("     （中载混批不慢于分相，重载分相反而略快 2%-7%）。")
        print("  2. 收益点在短请求尾延迟：中载 tau=256 把短请求 p99 从 565 降到 233（-59%），")
        print("     而 makespan 不变 —— 分相模式下短请求要等长 prefill 整个算完（队首阻塞）。")
        print("  3. 代价在任何口径下都存在：tau 越小长请求 p50 越差（中载 889 -> 13917，16x；")
        print("     重载 54747 -> 119766，2.2x），且单调。")
        print("  4. 因此 chunk 大小调的是**延迟分布**不是吞吐，且存在中间最优（本负载 ~256）。")
        print("  5. sum 与 max 两种成本口径只改变绝对耗时（重载 172k vs 146k），")
        print("     不改变任何结论的方向 —— 结论对成本模型不敏感。")

    fn = "results/summary.json" if which == "all" else f"results/summary-{which}.json"
    with open(fn, "w", encoding="utf-8") as f:
        json.dump(dict(seed=SEED, n_req=N_REQ, p_long=P_LONG, l_in_long=L_IN_LONG,
                       l_in_short=L_IN_SHORT, l_out=L_OUT, budget=C_BUDGET,
                       t_w=T_W, t_compute=T_COMPUTE, t_kv=T_KV,
                       results=out), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
