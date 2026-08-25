## 对应文章

- 《PTQ（09）：SqueezeLLM、VPTQ 与 CLAQ》  
  https://lrypcy.github.io/2026/08/24/ptq-09-squeezellm-vptq-claq/

## 如何运行

```bash
cd <本目录>
/Users/congyuan/Software/miniconda3/bin/python run.py   # 任何带 numpy+matplotlib 的解释器均可
```

纯 numpy + matplotlib，CPU 秒级完成；随机种子固定 SEED=0，结果可复现。
## 实验内容

- **A. 码本学习**：RTN / 普通 k-means / 敏感度加权 k-means（h_j=sum x_tj^2）
- **B. 加权增益的诚实测量**：增益取决于敏感度与幅值分布的耦合程度
- **C. dense-and-sparse 分解**：按显著度 h*w^2 挑极少数坐标保 FP16（SqueezeLLM 杀器）
- 码本大小扫描（2/4/6 bit）

## 关键数字（SEED=0 实测）

- 4-bit 全量化：RTN 3.01e-1 -> k-means 2.73e-2（**-90.9%**）-> 加权 1.69e-2（再降 38.1%）
- dense-and-sparse：0.2% 坐标保 FP16 再降 60.7%；1% 再降 **68.5%**

## 预期输出

- `results/squeezellm_sensitivity_and_decomposition.png`
- `results/codebook_size_sweep.png`
- `results/stdout.txt` / `results/results.json`

注：VPTQ 的向量维度自由度演示见 `../quip_aqlm/`。
