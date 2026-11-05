# pi-07：RECAP 把「用经验自我改进」拆成了哪几件可解析的事

对应文章：π 系列具身模型（07）π\*₀.₆ / RECAP（arXiv:2511.14759v2，Physical Intelligence，
19 Nov 2025）。RECAP = **R**L with **E**xperience and **C**orrections via
**A**dvantage-conditioned **P**olicies。

论文的做法是：先用 offline RL 预训练一个 VLA，再在真机上反复「① 部署收数据 →
② 用全部数据训 value function → ③ 用 value 导出的 binarized advantage 条件化训 VLA」。
它最容易被含混过去的地方不是「RL 能涨点」，而是**为什么把连续的 advantage 砍成
`Advantage: positive/negative` 一个 bit 反而好用，以及 β=1 部署到底等价于对数据做了什么**。

本实验在一个**可解析的标量高斯世界**里把四件事钉死：

1. **二值化不是「丢信息换简单」，是「只保留决策需要的排序信息」。** 连续 advantage
   被压成 1 bit 后丢了 0.6901 的组内方差，但策略质量仍从 -0.0433 提到 -0.0030——
   和调好温度的连续加权最优 -0.0034 几乎持平。因为 `I` 只依赖 A 的**排序**，
   连 value 的仿射标定误差都免疫（平移 δ 从 -0.050 到 0.050，J 恒为 -0.0030）。
2. **β=1 部署 = 对数据按 `I` 做一次硬筛选 + 重新归一化。** 论文 Eq.(2) 的 Bayes 重写
   给出 π̂ ∝ π_ref·p(I∣A)，β=1 时精确等于 π_ref(a∣I)——把支持集截断到
   |a−θ|<τ=0.9520 的正 advantage 区间。保留样本的回报 -0.0030，被丢弃的 -0.0606。
   但**训练**用 Eq.(3) 在全部数据上学无条件 + 条件两项，二者并不矛盾。
3. **on-policy 数据的价值在「覆盖」，interventions 的价值在「指向」。** 离线数据没覆盖
   部署状态时迭代永远不涨（-0.0425 恒定）；只加 autonomous rollouts 也能改进但慢
   （-0.0425 → -0.0028），加 expert corrections 一轮就到 -0.0025（≈解析下界）。
4. **「条件化 positive ≈ 采样多次取最好」只在特定意义下成立。** 二值接受拒绝采样与
   β=1 条件化是同一分布（都是 -0.0030）；但 best-of-N 取最大是 order statistic，
   N 要 8 才在 J 上追平（-0.0022 vs -0.0030），且两者连 |a−θ| 熵都几乎相等
   （6.7946 vs 6.7834）——回报 + 熵都分不开机制，真正的区别在区间边缘尾部质量
   （0.1322 vs 0.2558）与 N 倍采样代价。

**只验机制，不重算论文成绩**：真机 throughput >2×、失败率减半、13 小时 espresso、
两轮 50% / 2× throughput、97% 失败模式移除等全部只是引用，见下方「边界」。

## 合成世界

单步、标量动作 a ∈ ℝ 的 contextual Gaussian bandit，全部量在细网格
（a ∈ [-8, 10]，步长 0.002）上**确定性可算**，无需采样；采样只用于 B 组的迭代与
D 组标定噪声的多 seed 平均。

- **回报 / value**：$V(a)=-(a-\theta)^2/A_{MAX}^2 \in (-1,0]$，θ=0 是成功点（V=0）、
  $A_{MAX}=10$。这是论文 §V-C「value function 预测距成功剩余步数的负值、按最大
  episode 长度归一化到 (-1,0)」的连续化抽象；仿射归一化不改变 advantage 的符号与
  排序，故不影响二值化结论。
- **行为策略 / 数据集**：$\pi_{ref}$ = 两个高斯的混合（论文「demonstrations +
  autonomous attempts 的异构数据」）——「好」模式 $\mathcal N(0.6,1)$ 占 0.45，
  「次优」模式 $\mathcal N(2.4,1)$ 占 0.55。整体偏离成功点（均值 1.5900）。
- **value / advantage**：$V_{ref}=\mathbb E_{\pi_{ref}}[V]$ = -0.0433（解析高斯二阶矩
  ↔ 网格积分 ↔ 201-bin 直方图反算三路一致：-0.0433 / -0.0433 / -0.0435）。
  $A(a)=V(a)-V_{ref}$（behavior policy 的 on-policy MC 估计；单步世界是论文
  post-training $N=50$ lookahead 的退化）。
- **二值化**：$I=\mathbb 1[A>\epsilon_\ell]$，$\epsilon_\ell$ 取使约 30% 数据为正的
  分位（§V-D 的 30% percentile / A-F 的 30% 正 advantage）。因 V 是 $a^2$ 的单调
  仿射，`I=True` 等价于 $|a-\theta|<\tau=0.9520$，$\epsilon_\ell=0.0342$，
  实测正比例 0.3003。文本条件即论文的 `"Advantage: positive"` / `"negative"`。
- **策略提取与指标**：$\pi_{ref}(a\mid I)$ = 把 $\pi_{ref}$ 截断到 $|a-\theta|<\tau$
  再归一化；连续加权用 §III 的 KL 结果 $\hat\pi\propto\pi_{ref}\exp(A/\beta)$。
  策略质量 $J(\pi)=\mathbb E_\pi[V]$（越高越好，上界 0）；另看熵、
  KL$(\hat\pi\|\pi_{ref})$ 作为激进度代理。
- **distributional value / B=201**：把 $V\in[-1,0]$ 分 201 bins，$V=\sum_b p_b v(b)$；
  自校验用 softmax logits 的手写梯度 vs 有限差分（相对误差 4.23e-09）。

## 每个实验组对应论文的哪个机制

| 组 | 对应机制（出处） | 本探针里的操作 |
| --- | --- | --- |
| 自校验 | §IV-A Eq.(1) distributional value / B=201、A-F 的 V 定义 | 梯度 + 三路反算值 |
| A | §III KL 结果、§IV-B Eq.(2)/Eq.(3)、§V-B 文本条件、§V-D 阈值、A-F | 连续 $\exp(A/\beta)$ vs 二值 I；β=1 Bayes 等价；30% 分位 |
| B | §IV Algorithm 1 三阶段、§V-D interventions 强制 I=True / 从 pre-trained checkpoint 微调、§VI-C2 迭代 | 离线 vs autonomous vs corrections，K=0..6 轮 |
| C | §III / §IV-B 改进指示器、A-E CFG、AWR 对照（§VI-C3） | 二值接受拒绝采样 / best-of-N 的解析对照 |
| D | §IV-A value 估计器、§V-D「纠正非最优监督」、§VI-C3 AWR/PPO | 信息损失、标定误差传导、坏 corrections 让迭代变差 |

## 要验证的论断（数字逐位来自 `results/stdout.txt`）

| 组 | 论断 | 实测 |
| --- | --- | --- |
| 自校验 | 手写梯度 ↔ 有限差分 | 相对误差 4.23e-09 |
| 自校验 | V_ref 解析 ↔ 网格 ↔ 201-bin 反算 | -0.0433 / -0.0433 / -0.0435 |
| A | β=1 条件化 = π_ref(a∣I)，把质量拉向成功点 | J -0.0433 → -0.0030；均值 1.5900 → 0.2065 |
| A | 对数据做的事 = 按 I 硬筛选 | 保留 -0.0030 vs 丢弃 -0.0606，q=0.3003 |
| A | 二值 vs 连续加权最优 | 二值 -0.0030 vs 连续 t=0.15 -0.0034 |
| A | β=1 的 Eq.(2) Bayes 逐点等价 | 最大偏差 1.08e-19 |
| B | 只加离线数据不涨 | J1 恒 -0.0425 |
| B | 加 on-policy 纠正规 | -0.0425 → -0.0025（floor -0.0025）；首轮增益 0.0399 |
| B | autonomous 也涨但慢 | -0.0425 → -0.0028 vs corrections -0.0025 |
| B | 迭代收敛、一轮吃到大头 | 增益 0.0399 → 0.0000…；首轮 0.0399 / 全轮 0.0400 |
| C | 拒绝采样 ≡ β=1 条件化 | 都是 -0.0030 |
| C | best-of-N 单调、N→∞ 到 V=0 | -0.0433 → -0.0000；N\*=8 |
| C | 等价点熵却几乎相等 | 6.7946 vs 6.7834；尾部质量 0.1322 vs 0.2558 |
| D | 二值化丢 0.6901 组内方差仍几乎无损 | -0.0433 → -0.0030 |
| D | 仿射标定免疫 / 非线性噪声鲁棒 | 平移恒 -0.0030；σ=0.2 仍 -0.0318 > J_ref |
| D | 坏 corrections 让迭代比不加更差 | c=4 → -0.0652 < 离线基线 -0.0425 |

## 反直觉结论（写正文须照此口径）

- **β=1 条件化 = 硬筛选，但训练仍用全部数据**：Eq.(3) 在全部数据上学无条件项与
  条件项两项，部署只从条件分布采样；「丢弃大量数据」是部署采样的效果，不是训练
  丢弃了数据——这正是论文强调「不像 AWR 那样丢弃 / 大幅降权大量数据」的地方。
- **硬二值指示下 CFG β 退化**：$p(I\mid A)=\mathbb 1/\Pr(I)$ 在区间内是常数，
  $(\cdot)^\beta$ 归一化后 β 被消掉。论文 A-E 的 β>1 锐化依赖 learned conditional
  与 unconditional 之间的**连续似然比**，本探针用硬集合指示，没有建模这一层。
- **二值化对 value 标定误差鲁棒**：`I` 只依赖 A 的排序，仿射误差完全免疫；连与 a
  相关的非线性噪声，σ 放大到 0.2（远超 V 量级）J 仍为 -0.0318、优于 -0.0433。
- **best-of-N ≠ 条件化**：只有在「二值接受 + 保留原密度」的拒绝采样意义下才与
  β=1 条件化同分布；取最大是 order statistic，且回报 + 熵都分不开两种机制。
- **坏 corrections 会反噬**：纠正本身不构成最优监督（论文 §V-D），c=4 时 K 轮后
  比完全不加数据的离线基线还差。

## 运行

```bash
~/Software/miniconda3/bin/python recap_lab.py       # 自校验 + A–D 四组，约 0.5 秒
~/Software/miniconda3/bin/python recap_figures.py   # ≥5 张图（SVG + PNG）
```

复现命令里的解释器必须锁 **`~/Software/miniconda3/bin/python`（numpy 2.1.1 /
matplotlib 3.11.1，CPU）**。`results/stdout.txt` 是用它跑出来的；换成别的
numpy 版本会因为随机数流 / 浮点实现差异让数字漂几个千分位，正文数字就和固化输出
对不上了。纯 numpy / 固定 seed / 多 seed 平均（B 组 40 seed、D 组标定噪声 24 seed）/
每组末尾断言 / 最后打印断言总数（**36 条全绿**）。`recap_figures.py` 的数据图数字
逐位取自 `results/stdout.txt` 的「正文回填表」区。

## 边界

- **只验机制，不重算论文成绩。** 论文里的真机 **throughput more than doubles
  (>2×)**、**失败率减半 / roughly 2×**、**连续 13 小时做 espresso**、两轮迭代
  **50% / 2× throughput 提升**、失败模式移除 **97% 成功率**、各任务 **90%+ 成功率**
  等，全部是**引用**，本探针一个都没有重算；合成世界的 J 是单步标量动作的回报，
  与 successes/hour 不是同一量纲，改进幅度不可与真机直接比。
- **论文 Fig.7–12 的柱状图没有柱顶数值标注**，图内只有纵轴刻度与误差棒（standard
  error）；全文 15 个 `<table>` 都是公式排版，无任何数值结果表。因此**不能目测估数**，
  绝对 throughput / 成功率均不可得，只能引用正文里写明的相对倍数。
- **许可证不同**：RECAP / π\*₀.₆ 是 `arXiv.org perpetual non-exclusive license`
  （与 06 篇 RTC 的 CC BY 4.0 不同）。所有配图按论文**文字重画**，不复用任何官方
  图 / 截图；结构图标注的是论文写明的常量（B=201、N=50、β=1、CFG 1.5–2.5、
  dropout 30%、ε_l 30% 分位），不是从图上量出来的。
- 世界是**单步标量动作**，把论文的多步轨迹（$N=50$ lookahead、每步 reward、失败
  $-C_{fail}$）压成一个 bandit；distributional value 只用来验证 B=201 的梯度与反算，
  没有建模 670M VLM backbone、Gemma 3、action expert、FAST token / flow matching
  与 KI。连续 advantage 加权的温度是无量纲 $t\cdot\mathrm{std}(A)$，与论文的 CFG β
  不是同一个量。
- 「分布偏移」用两个离散状态（demo 覆盖 / 部署 OOD）建模，只测「离线数据覆盖不到的
  区域需要 on-policy 数据」这一半；真机里的图像、可变形物体、多阶段任务未建模。
- 连续优势与二值优势的对照是**同一 reward 下**的对照；结论「二值几乎不输连续加权」
  是这个世界里的结论——reward 是 $a^2$ 的单峰函数，正 advantage 区间恰是成功邻域。
  若 reward 多峰、或高 advantage 区不等于最优区，二值化丢大小信息的代价会更大。
