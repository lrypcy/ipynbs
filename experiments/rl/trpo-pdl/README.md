# TRPO 理论链 · 可复算实验

配套博客：《从 MDP 到 GRPO（三）：TRPO——信任域，让更新别摔死》
（https://lrypcy.github.io/2026/08/21/mdp-to-grpo-03-trpo-trust-region/）

博客正文里每个实测数字都出自本目录的 `trpo_pdl_lab.py`。表格 MDP 上
$$V, Q, A, \rho^{\pi}, F$$ 全部可以解析算出，因此能直接和真值对账——这是
神经网络上做不到的。**所有输出已固化，GitHub 上直接读即可，不需要自己跑。**

## 怎么跑

纯 numpy，miniconda base 即可：

```bash
cd <ipynbs 仓库根>
python3 experiments/rl/trpo-pdl/trpo_pdl_lab.py 1 2   # 分组跑，参数取 1~4
```

## 四个分组各回答什么

| 分组 | 验证什么 | 关键结论 |
|:---|:---|:---|
| 1 | 性能差异引理（PDL）是恒等式 | 残差 ≤ 5.55e-16；截断部分和误差 ∝ $$\gamma^T$$ |
| 2 | surrogate 一阶精确、二阶漂移 | $$\nabla L = \nabla J$$ 相对误差 1.15e-9；$$\gamma=0.99$$ 反例 L>0 但 $$\Delta J<0$$ |
| 3 | KL 上界定理成立且极保守 | 18 组随机试验无一违反；$$\lvert\text{漂移}\rvert$$ / 定理上界均值 2.8e-4 |
| 4 | Fisher 奇异 / CG / 阻尼 / 泰勒 | $$F$$ 有 $$\lvert\mathcal{S}\rvert$$ 个零特征值；$$\lambda=0.01$$ 使解偏 33%；真实 KL 系统性小于二阶预测 |

## 结论数字（与博客正文逐字一致）

- **PDL（§2）**：6 个随机策略对的残差最大 $$5.55\times10^{-16}$$；截断到 $$T=100$$ 的部分和误差
  $$2.35\times10^{-5}$$，与 $$\gamma^{100}$$ 同量级。
- **surrogate（§3）**：一阶匹配 $$\lVert\nabla J-\nabla L\rVert_\infty=3.95\times10^{-10}$$（相对 1.15e-9）；
  沿梯度方向 $$\alpha=8$$ 时兑现率降到 0.9395；$$\gamma$$ 从 0.80→0.99 绝对漂移放大 25 倍（0.0186→0.4690）。
  反例：平均 KL 仅 0.1283 时，L=+0.1091 而 $$\Delta J=-0.5290$$。
- **KL 上界（§4）**：$$|L-\Delta J|$$ 恒 ≤ $$C\cdot D^{\max}_{\mathrm{KL}}$$，但均值仅占到上界的 $$2.8\times10^{-4}$$
  —— 定理保守 3~4 个数量级，常数 $$C$$ 不进代码。max KL / 平均 KL 实测比值 5.21~7.80。
- **Fisher / 泰勒（§5）**：Fisher 解析与有限差分 Hessian 差 $$8.1\times10^{-13}$$；有 8 个零特征值
  （= $$|\mathcal{S}|$$，softmax 平移冗余）；阻尼 $$\lambda=0.01$$ 时解向量与 $$\lambda\to0$$ 差 33%、
  $$\lambda=0.1$$ 差 80%；CG 迭代 10 次相对误差 $$2.19\times10^{-6}$$；真实 KL 在 $$\delta=10^{-2}$$ 时比
  二阶预测小 0.46%、$$\delta=0.3$$ 时小 9.7%。

## 几个容易踩的坑

- **KL 梯度的符号**：$$\partial\,\mathrm{KL}(\pi_0\|\pi_\theta)/\partial\theta_s = \pi_\theta(s) - \pi_0(s)$$，
  写反会让「解析 $$F$$ vs 有限差分 Hessian」的残差恒为 $$2\lVert F\rVert$$ 量级（本例 7.5e-2），一眼可辨。
- **阻尼只进 CG，不进二阶 KL 预测**：把 $$\lambda I$$ 带进预测值会引入恒定约 25% 的偏差。
- **闭式步长用 $$x^\top F x$$ 不用 $$x^\top g$$**：阻尼下两者差 13%（$$\lambda=0.01$$），正确的写法对应真实的二阶型。
