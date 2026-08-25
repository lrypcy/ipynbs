## 对应文章

- 《PTQ（07）：QuaRot 与 SpinQuant》  
  https://lrypcy.github.io/2026/08/24/ptq-07-quarot-spinquant/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- 手写 Walsh-Hadamard 快速变换（迭代 butterfly），正交性自检 max|HH^T-I|=0
- 离群通道压平：构造 8 条 15x 离群激活通道，对比旋转前后逐通道 max-abs
- 端到端 INT4：直接量化 X/W vs 在线旋转激活 + 离线预旋转权重

## 关键数字（SEED=0 实测）

- 逐通道动态范围收缩 **4.1x**（top-5 通道 45.7~58.4 -> 全体 <= 14.3）
- INT4 端到端输出误差 0.342 -> 0.174（**降低 49.1%**）
- 数学等价性：(xH)(HW)^T = x HH^T W^T = xW，无需转回步骤

## 预期输出

- `results/hadamard_flatten_and_int4_error.png`
- `results/stdout.txt` / `results/results.json`
