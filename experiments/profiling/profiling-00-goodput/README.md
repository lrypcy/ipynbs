## 对应文章

- 《性能剖析（00）：度量口径——MFU、HFU、Timer 与 goodput，先把尺子做准》§5
  https://lrypcy.github.io/2026/09/29/profiling-00-metrics/

## 如何运行

```bash
# 纯标准库，无第三方依赖，秒级完成
jupyter nbconvert --to notebook --execute profiling-00-goodput.ipynb
```

## 代码来源

逐字摘自 `lrypcy.github.io` 的 `tools/profiling_bench/goodput_calc.py`，
由 `experiments/profiling/_build/build.py` 用 `ast` 按顶层符号抽取。

## 实验内容

- **span 拆分与 goodput**：把一个稳定迭代拆成叶子 span，按 `is_goodput_span` 累加
- **checkpoint 摊薄**：ckpt 间隔从「不存」到 500 步，有效吞吐掉多少、goodput 动多少
- **两种背离形态**：MFU 高 / goodput 低 与 MFU 低 / goodput 高 各指向什么优化动作
- **量化形态一**：把 data_loading 与 communication 漏出压缩到 1/4，goodput 回到多少

## 关键结论

- 存 checkpoint 越勤，**有效吞吐掉得越多，而 goodput 几乎不动**（甚至微升，因为 ckpt 算正事）
- 压缩非计算开销时 **goodput 的分子没动**，只是分母变小 —— 所以 goodput 变好不能推出 MFU 也变好
