## 对应文章

- 《PTQ（12）：Olive 与 AbFloat》  
  https://lrypcy.github.io/2026/08/24/ptq-12-olive-abfloat-hardware/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

本实验为纯 stdout 演示（无图）。两个 Demo：

- **DemoA**：同槽位预算下三种策略的重构误差——全员共享窄网格 /
  victim 置零+饱和 / OVP（细网格+abfloat+victim 保护）
- **DemoB**：abfloat 的可表示范围与离群表示精度

## 关键数字（实测，见 results/stdout.txt）

- 方案1 共享窄网格 0.2215；方案2 victim 置零 0.9319；
  方案3 OVP **0.1329**（好于方案1 40%）
- abfloat(z∈[3,10]) 覆盖 [8, 1792] 且跳过 (0,8) 正常区间；
  离群值(16~64)平均表示误差 abfloat **4.8%** vs 窄格式饱和 91.7%

## 预期输出

- `results/stdout.txt`（无 PNG，控制台实验）
