# Flow Matching / Diffusion 建模核验 · 14 组可复算实验

配套博客三篇（MIT 6.S184《An Introduction to Flow Matching and Diffusion Models》
讲义的完整实现，纯 numpy、纯 CPU）：

- 《Flow Matching 建模之一：从概率路径到边际向量场》§3–§5
  <https://lrypcy.github.io/2026/08/22/flow-matching-01-algorithm-evolution/>
- 《Flow Matching 建模之二：score、扩散与 SDE 扩展》§3–§9
  <https://lrypcy.github.io/2026/08/22/flow-matching-02-score-diffusion-sde/>
- 《Flow Matching 建模之三：条件生成、离散域与潜空间》§2–§5
  <https://lrypcy.github.io/2026/08/22/flow-matching-03-guidance-discrete-latent/>

三篇正文里每个实测数字都出自本目录。**全部输出已固化在 `results/stdout.txt`，
GitHub 上直接读即可，不需要自己跑。**

## 文件

| 文件 | 内容 |
|---|---|
| `fm_modeling_lab.py` | 连续部分 8 组（Thm 3/9/11/12/17/19、Prop 1） |
| `fm_discrete_lab.py` | 离散状态空间 6 组（Prop 2、Thm 33/36、Ex 35/37/39） |
| `gen_stdout.py` | 逐组跑并拼出 `results/stdout.txt` |
| `results/stdout.txt` | 14 组完整输出（**唯一权威数字来源**） |

## 怎么跑

纯 numpy，miniconda base 即可：

```bash
cd <ipynbs 仓库根>

# 单组（沙箱有 CPU 上限，长脚本会被掐断，分组跑是正常用法）
python3 experiments/flow_matching/fm-modeling/fm_modeling_lab.py 3
python3 experiments/flow_matching/fm-modeling/fm_discrete_lab.py 2

# 快速自校验（只跑不需要长积分的组，几十秒）
python3 experiments/flow_matching/fm-modeling/fm_modeling_lab.py --selftest
python3 experiments/flow_matching/fm-modeling/fm_discrete_lab.py --selftest

# 重新固化输出（14 组全跑，约 3–5 分钟）
python3 experiments/flow_matching/fm-modeling/gen_stdout.py
```

`gen_stdout.py` 也支持分组追加：`gen_stdout.py 1 2 3`（连续）与
`gen_stdout.py d1 d2`（离散，`d` 前缀），配 `--append` 不清空已有内容。

## 14 组各回答什么

### 连续部分（`fm_modeling_lab.py`）

| 组 | 验证什么 | 结论 |
|:---|:---|:---|
| 1 | 连续性方程（Thm 11）逐点残差，7 条路径 | 归一化残差全部 $<10^{-10}$，**与路径选择无关** |
| 2 | 边际化 trick（Thm 9）：ODE 终点是否服从 $p_{\rm data}$ | 一阶矩吻合到 3 位小数；但显式 Euler 把成分间过渡区压成**精确 0** |
| 3 | CFM $\equiv$ FM（Thm 12）：两损失是否只差常数 | MC 样本量放大 100 倍散布**不收敛**（0.7342 → 0.7570）；求积下跨度 $1.8\times10^{-15}$ |
| 4 | score $\leftrightarrow$ velocity 换算（Prop 1） | 条件层与边际层误差都 $<10^{-14}$ |
| 5 | CFM 与 DSM 的精确关系 | 权重是 $a_t^2$；漏掉 $b_tx$ 项会让损失恶化约 10103 倍 |
| 6 | Fokker–Planck（Thm 19）与 SDE 扩展（Thm 17） | 残差 $1.5\times10^{-10}$；三档 $\sigma$ 终点分布极差 0.0013 |
| 7 | Langevin 动力学与 OU 稳态 | 5 模混合无模式坍缩；OU 方差相对误差 1.06% |
| 8 | velocity / score / denoiser / noise 四种参数化 | 互为仿射重参数化，误差 $<10^{-14}$ |

### 离散部分（`fm_discrete_lab.py`）

| 组 | 验证什么 | 结论 |
|:---|:---|:---|
| 1 | Kolmogorov 前向方程（Prop 2） | 讲义系数 $\kappa'/(1-\kappa)$ 残差 $2.7\times10^{-10}$ |
| 2 | 条件速率矩阵（Ex. 37） | 逐位置命中率与理论 $\kappa+(1-\kappa)/V$ 误差 $<10^{-3}$ |
| 3 | 离散边际化 trick（Thm 36） | 用**确定性 KFE 矩阵演化**验证，终点误差 $2.8\times10^{-6}$ |
| 4 | 因子化混合路径（Ex. 35） | 端点与归一性 |
| 5 | 掩码扩散 MDLM（Ex. 39） | unmask 比例精确等于 $\kappa_t$，$t{=}1$ 时 100% 恢复句子 |
| 6 | Thm 38 $\Rightarrow$ 逐位置交叉熵 | NLL 下界等于边缘熵 11.9369 vs 11.9374 |

## 几个容易踩的坑

- **Thm 12 不能用蒙特卡洛验**。$u_t^{\rm marg}$ 含 $1/(1-t)$ 因子，$t\to1$ 时
  实测 $t$ 采样范围放开到 $[0.02,0.98]$ 时边际场 max 绝对值 **47.75**、
  标准差 **2.759**，而条件场只有 **1.825**；范围收到 $[0.10,0.50]$ 后边际场
  降到 3.58 / 0.504，条件场几乎不变（1.826）。有限样本方差被重尾淹没，
  实验 3 的 (0) 段用**同一构造**做 MC 对照：样本量 4000 → 400000 放大 100 倍，
  散布从 0.7342 只变到 0.7570，**完全不收敛**——于是看起来「定理不成立」。
  换成确定性梯形求积后跨度降到 $10^{-15}$。
  **验证「只差一个常数」的恒等式，判据是残差是否随步长/样本量收敛，
  不收敛就是估计量本身不可用。**
- **散度要对 $x$ 求导**。连续性方程右端是 $-\partial_x(p_tu_t)$，不是 $-\partial_t$。
  写错时残差是 $10^{10}$ 量级且**不随步长收敛**——这一点立刻能定位。
- **Richardson 外推的第二个差分项要用 $f(x-h/2)$**。写成两次 $f(x+h/2)$ 会让
  FPE 残差从 $10^{-10}$ 变成 $1.7\times10^4$。
- **相对残差判据在 tails 失效**。$\partial_t p_t\to0$ 处相对误差发散，要用
  「除以该 $t$ 下的最大导数」归一化。
- **离散域的 $p_{\rm data}$ 必须按位置边缘归一化**。不归一化会让 KFE
  假性破坏——我一度据此以为讲义的速率系数有误，归一化后残差立刻降到 $10^{-10}$。
  **怀疑讲义之前先确认自己的约定。**
- **粒子模拟 CTMC 误差太大**。Thm 36 改用确定性 KFE 矩阵演化验证。
- **两个时间方向约定不能混**。本文件用讲义的 $t=0$ 噪声、$t=1$ 数据；
  扩散文献的反向约定（VE/VP）只用于核对恒等式，不能从 $t=0$ 积分做生成。
  两者在 `VALID_PATHS` / `REVERSED_PATHS` 里显式分开。