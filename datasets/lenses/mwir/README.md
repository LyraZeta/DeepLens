# MWIR 七片候选处方

`mwir_strong_bent7_current_best.json` 是当前可复现实验中最可靠的强弯曲七片
起点，供后续 GPU 多起点和非球面优化使用。

它不是最终验收处方。当前独立固定光线检查约为：

- 质心 RMS 均值/最大值：约 `0.0505 / 0.0756 mm`；
- EFL：约 `561.44 mm`；
- 入瞳直径：`280 mm`；
- Y 向全视场：`9.6°`；
- 波段：`2.7–4.3 µm`；
- 系统 MTF：未通过，最差估计约 `0.03`，目标为 `0.3`；
- 畸变、像高映射、渐晕、F/# 和镜片数量检查已通过。

该文件来自本地结果
`results/mwir-power-bend-focus-60/element_power_optimized.json`。之所以复制到
数据集目录，是因为 `results/` 被 Git 忽略，而换机继续优化需要一个确定、可
版本控制的起点。

推荐先运行：

```powershell
python mwir_power_bend_multistart.py `
  --input-lens datasets/lenses/mwir/mwir_strong_bent7_current_best.json `
  --output results/gpu-power-bend-multistart `
  --device cuda `
  --random-starts 32 `
  --top-starts 8 `
  --iterations 300 `
  --field-count 5 `
  --spp 64 `
  --validation-spp 128 `
  --eval-spp 1024
```

只有输出的 `mwir_metrics.json` 中 `pass.overall == true` 时，才可以把候选称为
最终设计并导出 Zemax 文件。
