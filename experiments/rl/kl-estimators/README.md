# KL 估计器 k₁/k₂/k₃ · 可复算实验

配套博客：《KL 散度为什么需要估计：k₁/k₂/k₃ 的偏差、方差与梯度》
（https://lrypcy.github.io/2026/10/05/rl-kl-estimators-deepdive/）

博客正文里每个实测数字都出自本目录的 `kl_estimators_lab.py`。
**所有输出已固化在 `results/stdout.txt`，GitHub 上直接读即可，不需要自己跑。**

约定：样本 $x\sim q$，$r=p(x)/q(x)$，$\delta=\log r=\log p-\log q$，估的是 **reverse KL** $\mathrm{KL}(q\Vert p)=\mathbb E_q[-\log r]$。
在 RLHF 里 $q$ 就是采样策略、$p$ 是参考策略，比值也就是《RL 符号与计算全景》里的 $r^{\rm ref}$。

## 怎么跑

纯 numpy，miniconda base 即可：

```bash
cd <ipynbs 仓库根>
python3 experiments/rl/kl-estimators/kl_estimators_lab.py            # 全部四组
python3 experiments/rl/kl-estimators/kl_estimators_lab.py 1 2        # 分组跑
python3 experiments/rl/kl-estimators/kl_estimators_lab.py --selftest # 只跑自校验
```

## 四个分组各回答什么

| 分组 | 验证什么 | 关键结论 |
|:---|:---|:---|
| 1 | 值估计的偏差/方差随**批大小**怎么变 | 三个都无偏（k2 除外）地收敛；k1 单样本时 45.8% 给出负值，N=256 才降到 12.1% |
| 2 | 梯度的**无偏性**（精确期望，不采样） | $\nabla k_1$、$\nabla k_3$ 无偏（相对偏差 0.000000），$\nabla k_2$ 有偏 0.206 |
| 3 | 梯度方差随**维度**怎么变 | $\mathbb E[(r-1)^2]=\exp(D\mu^2)-1$ 指数增长，$\nabla k_3$ 被这项直接放大 |
| 4 | 梯度/值估计随**序列长度**怎么失效 | $k_3$ 在有限样本下系统性低估，$T\sigma^2$ 大时趋近 $-1$ nat |

## 结论数字（与博客正文逐字一致）

- **值估计（§3）**：regime A（真 KL 0.01104）单样本下 $k_1$ 标准差是真值的 **13.50 倍**、45.8% 的样本给出负值；$k_2/k_3$ 只有 1.43/1.41 倍。批大小涨到 256，三者标准差都落到真值的 0.1 倍量级，但 $k_2$ 的偏差不动——regime B（真 KL 0.62583）下 $k_2$ 恒高估 **+0.223**（约 +36%），$k_1/k_3$ 偏差都在 $10^{-4}$ 量级。
- **梯度无偏性（§5）**：$V=300$ softmax，真梯度取闭式 $p-q$。精确期望下 $\nabla k_1$ 与 $\nabla k_3$ 相对偏差 **0.000000**（无偏），$\nabla k_2$ 相对偏差 **0.206440**（有偏）。无偏的关键恒等式 $\mathbb E_q[r\,\nabla\log p_\theta]=0$ 实测范数 $3.10\times10^{-17}$。批均值梯度的总方差：$k_1$ $1.5483\times10^{-2}$，$k_2$ $1.7325\times10^{-3}$，$k_3$ $1.8240\times10^{-3}$。
- **维度（§6）**：$q=\mathcal N(0,I_D)$、$p=\mathcal N(0.05,I_D)$。解析的 $\mathbb E[(r-1)^2]=\exp(D\mu^2)-1$ 从 $D=1$ 的 $0.0025$ 涨到 $D=1000$ 的 $11.18$。$\mathrm{std}(\nabla k_3)/\mathrm{std}(\nabla k_1)$ 从 $0.07$（$D=1$）翻到 $9.51$（$D=1000$）。
  ⚠️ $\mathrm{std}(\nabla k_3)$ 是重尾量的蒙特卡洛估计，换样本量会明显变化（同一设置试过 161 与 301 两个值）；**正文只引解析的最后一列**，不把这个比值当结论。
- **序列（§6）**：每 token $\log r\sim\mathcal N(-\sigma^2/2,\sigma^2)$，$T=1024,\sigma=0.2$ 时真 KL 20.480，$k_1$ 得 20.466、$k_2$ 得 **229.920**、$k_3$ 得 **19.551**（差 $-0.93$，趋近 $-1$ nat）。$\sigma=0.5$ 时 $T=1024$ 真 KL 128.000，$k_2$ 高达 **8325.156**。

## 几个容易踩的坑

- **无偏不等于梯度无偏**：$k_2$ 两个都有偏；$k_1/k_3$ 两个都无偏，但**只在 $q$ 不随求导参数变化时才成立**（$q$ 必须 detach 或可重参数化）。
- **$\mathbb E_q[r]=1$ 是靠极稀有样本撑起来的**：序列级 $r$ 是连乘，$\mathrm{Var}(r)=\exp(T\sigma^2)-1$，样本量再大也采不到那条尾巴。这就是 $k_3$ 在长序列上系统性低估的原因。
- **$\log r$ 只对 $p$ 求导**：若 $q$ 也依赖参数，$\nabla\log r=\nabla\log p-\nabla\log q$，第二项不能漏。
- **词表维度 vs 序列维度是两回事**：逐 token 的 KL 对词表求和就能算精确（与熵同量级），需要估计是因为每个位置只采到一个样本；而序列级的问题来自连乘。
