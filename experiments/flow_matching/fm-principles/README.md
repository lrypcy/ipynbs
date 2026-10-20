# Flow Matching / 扩散模型「原理篇」· 7 组核验实验

配套博文：**《从概率路径到 Flow Matching：用一套数学框架理解现代扩散模型》**
<https://lrypcy.github.io/2026/10/20/flow-matching-00-principles/>

底本：Peter Holderrieth & Ezra Erives, *An Introduction to Flow Matching and
Diffusion Models*, MIT 6.S184 (2026) —— <https://diffusion.csail.mit.edu/>

博文走数学推导；本目录把推导中的关键恒等式（CFM = FM + C、Thm 17 分布不变性、
CFG ≡ 倾斜分布的速度场、四种参数化互为仿射重组）在有限原子混合上逐点验到
机器精度，是本系列的数值核验存档。**输出已固化在 `results/stdout.txt`，
GitHub 上直接读即可，不必自己跑。**

## 文件

| 文件 | 内容 |
| :--- | :--- |
| `fm_principles.ipynb` | **主入口**：逐组带输出，代码与结果并排 |
| `fm_principles_lab.py` | 同一份逻辑的脚本版，支持分组运行（默认的 CLI 入口） |
| `results/stdout.txt` | A–G 七组完整输出（唯一权威数字来源） |

## 怎么跑

纯 numpy、纯 CPU：

```bash
cd <ipynbs 仓库根>/experiments/flow_matching/fm-principles

# 全部（A–G，约 2 分钟）
python3 fm_principles_lab.py

# 分组（沙箱有 CPU 上限，长脚本会被掐断，分组跑是正常用法）
python3 fm_principles_lab.py A B
python3 fm_principles_lab.py E F G
```

## 为什么残差是 $10^{-13}$ 而不是 $10^{-10}$

关键在 $p_{\rm data}$ 的选法：这里取**有限原子的加权混合**，于是 $p_t$、
$\nabla\log p_t$、$u_t$、$\mathrm{div}(u_t)$、$\Delta p_t$ 全部有解析闭式。
偏微分方程的残差因此不经过任何数值差分——差分格式的截断误差被彻底绕开
（旧版 `../fm-modeling/` 用 Richardson 外推只能到 $10^{-10}$，差别全在这里）。

## 七组各回答什么

| 组 | 验证什么 | 结论 |
| :--- | :--- | :--- |
| A | 条件场 ⇔ 条件流反解、Prop 1（$u=a\cdot s+b\cdot x$）、denoiser 反解 Eq.43 | 三条恒等式在 5 条路径上误差 $\le 4.9\times10^{-14}$ |
| B | 连续性方程（Thm 11）；「裸 $u_t$ + 噪声」的 FPE 残差 vs 加修正项后 | CE 残差 $\le 6\times10^{-15}$；裸 $u_t$ 在 $\sigma=0.8$ 时残差 $2.43$，加上 $\frac{\sigma^2}{2}\nabla\log p_t$ 回到 $5\times10^{-14}$ |
| C | Thm 12（$\mathcal L_{\rm FM}=\mathcal L_{\rm CFM}+C$） | 求积下相对跨度 $\le 10^{-12}$；MC 在 $t=0.9$ 时相对散布 $3500\%$～$48000\%$，**不能用 MC 验** |
| D | 四种参数化互为仿射重组；$a_t^2$ 的有效时间权重；$s=v/b$ 的代价 | 换算误差 $\le 2\times10^{-14}$；VP 的 $a_t\equiv-\gamma/2$；丢掉 $b_tx$ 项放大 $21.6\sim903.5$ 倍（$t=0.95	o0.05$） |
| E | Thm 17：同一个 $u_t$ 配任意 $\sigma_t$，终点分布不变 | $\sigma\in[0,1]$ 四档 W1 极差 $0.0196$；$\sigma=2$ 时被 Euler–Maruyama 步长拖垮（$0.1291$） |
| F | Bayes → classifier guidance → CFG；CFG ≡ 倾斜分布的速度场 | 两条恒等式 $\le 8.3\times10^{-14}$；$w$ 从 1 到 8，质量单调向条件类集中（$0.499\to0.956$） |
| G | numpy 手训 velocity 头与 $\epsilon$ 头，换算回 velocity 后对照 | 同解；$t=0.05$ 处 $\epsilon$ 头的 RMSE 是 velocity 头的 **12.13 倍**（$1/\alpha_t$ 放大） |

## 两个容易踩的坑（本目录踩过的）

- **`(N,1)` 与 `(N,)` 的广播**：`p * (u_theta - u_true) ** 2` 里，若前者是一维后者是
  $(N,1)$，会得到 $(N,N)$ 外积而不是逐点积，而且**形状合法、不报错**，只是
  损失随网格点数线性放大。写加权积分前先 `.ravel()` 或统一维度。
- **Thm 12 不能用 MC 验**：判据是残差随网格/样本量是否收敛。
  $t\to1$ 时边际场含 $1/(1-t)$ 因子，常数 $C$ 本身接近 0，
  MC 的散布比被测量还大几个数量级（见 C 组表）。
