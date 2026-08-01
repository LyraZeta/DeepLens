# MWIR 透射望远系统 GPU 换机交接

更新时间：2026-08-01
仓库：`https://github.com/LyraZeta/DeepLens.git`
目标分支：`main`

## 1. 当前任务

使用 DeepLens 设计一个不超过七片的中波红外透射式望远系统，并最终生成可在
Zemax OpticStudio 中继续检查和优化的处方。

当前有效设计指标如下；若早期聊天中的数字与此处冲突，以本表为准。

| 项目 | 当前约束 |
|---|---|
| 系统类型 | 透射式、轴对称、无穷远物距 |
| 波段 | 2.7–4.3 µm |
| 主波长 | 3.5 µm |
| 温度 | 暂定 20 °C |
| Y 向全视场 | 9.6°，即半视场 ±4.8° |
| Y 向半像高 | 47.1454 mm |
| 有效焦距 | 约 561.4396 mm，由像高和视场推导 |
| 入瞳直径 | 280 mm |
| F/# | 约 F/2.005 |
| 镜片数 | 不超过 7 片，可以等于 7 片 |
| 畸变 | 不高于 0.5% |
| 渐晕 | 当前代理目标有效光线比例不低于 0.8，硬下限 0.7 |
| 总长、后焦 | 暂不作硬约束；满足像质后尽量缩短 |
| 探测器 | 尚未确定 |
| 临时像元 | 30 µm 只用于仿真采样，不是已确认探测器约束 |
| 临时奈奎斯特频率 | 16.6633 cy/mm |
| 系统 MTF 阈值 | 临时取 0.3 |

已经取消 `42 µrad` 分辨率约束。早期提到的 `320×256、30 µm` 探测器也不再
作为硬约束。`47.1454 mm` 是从光轴到 Y 向边缘场点的半像高，不是探测器
对角线。

## 2. 聊天过程中的主要结论

1. 最初运行 `4f_system.py` 时出现 GBK 解码错误，项目的 JSON/文本读取已改为
   显式 UTF-8。
2. 项目注释和 Markdown 已基本汉化，同时保留英文版 `README_EN.md`。
3. Git 远端已整理为：
   - `origin`：用户自己的 `LyraZeta/DeepLens`；
   - `upstream`：原项目 `vccimaging/DeepLens`。
4. 卡塞格林原系统的像高、视场和入瞳可以作为透射系统的一阶继承指标，但不能
   同时强行继承未经确认的探测器参数和 42 µrad 约束，否则焦距会互相冲突。
5. 旧的“材料种子”处方在 Zemax 中像玻璃平板不是显示问题：多个表面半径达到
   数千至数万毫米，局部优化又只允许很小的曲率倍率变化，无法从平板拓扑跳出。
6. 内置 `GeoLens.optimize()` 原先只回滚 NaN/Inf、一阶焦距和有效率违规，
   不回滚总 merit 变差的更新；实测会越优化越差。现在 MWIR 流程已增加更新后
   同批光线复算、参数与 Adam 状态回滚、缩步重试。
7. RMS 现在默认以每个场点的光斑质心为参考；像高映射误差由 chief ray 单独
   约束，避免把畸变重复算入光斑 RMS。
8. MTF 代理现在分别计算切向和弧矢方向，并取较差方向，不再把 X/Y 相位错误
   合并成一个标量。

## 3. 当前代码能力

主要文件：

- `mwir_spec.py`：正式规格和一致性检查；
- `mwir_telescope_design.py`：生成母型、应用 MWIR 专用约束、独立验收；
- `mwir_power_bent7_optimize.py`：强弯曲七片球面、非球面和结构阶段优化；
- `mwir_element_power_optimize.py`：按每片净功率和弯曲量优化，可跨过零功率；
- `mwir_power_bend_multistart.py`：power+bend 多起点搜索；
- `mwir_material_seed.py`：材料布局种子；
- `mwir_material_outer_search.py`：材料布局外层搜索；
- `mwir_patent_seed.py`：公开专利拓扑实验种子；
- `mwir_expand_asphere.py`：把已有非球面扩展至高阶偶次项；
- `mwir_lbfgs_probe.py`：实验性 L-BFGS 局部可达性探针；
- `mwir_mtf_curriculum_optimize.py`：保守 MTF 课程实验，支持 CPU/CUDA；
- `deeplens/geolens_pkg/optim.py`：内置优化器的 MWIR 安全门控和回滚增强；
- `test/test_mwir_optimizer_guard.py`：merit 回滚测试；
- `test/test_mwir_curriculum.py`：课程权重和安全默认步长测试。

安全默认学习率已经按有限差分诊断下调：

- 元素功率：`3e-5`；
- 元素弯曲量：`1e-4`；
- 焦面：`1e-3`；
- 相对曲率/结构间隔/圆锥常数：约 `1e-4`；
- 非球面归一化边缘变量：约 `1e-5`。

结构和元素功率优化器支持 curriculum：默认前 25% 只优化基础质心 RMS，随后
50% 用 smoothstep 平滑引入 MTF、焦面、像散、色焦和场曲权重。训练、验证和
post-step guard 使用完全相同的当步权重。

## 4. 当前最可靠候选

换机后从以下受版本控制的文件继续：

```text
datasets/lenses/mwir/mwir_strong_bent7_current_best.json
```

它复制自本机：

```text
results/mwir-power-bend-focus-60/element_power_optimized.json
```

独立固定光线检查约为：

- 质心 RMS 均值/最大值：`0.05048 / 0.07563 mm`；
- EFL、F/#、入瞳、像高映射、畸变、渐晕和镜片数量通过；
- 最差系统 MTF 约 `0.029`，目标 `0.3`，因此 `pass.overall == false`。

这只是当前起点，不是最终处方，也不应直接作为最终 ZMX 交付。

最近的 `results/mwir-mtf-curriculum-100` 实验没有通过独立验收：其训练代理看似
改善，但高采样最差系统 MTF 约 `0.025`，比可靠基线还差。不要把该目录作为
下一轮起点；它说明低采样 MTF 代理仍会过拟合，需要 GPU 上更高采样、多种子和
独立验证。

## 5. 新电脑环境准备

```powershell
git clone https://github.com/LyraZeta/DeepLens.git
cd DeepLens

conda env create -f environment.yml
conda activate deeplens_env
```

随后按照新电脑的显卡驱动和 CUDA 版本，从 PyTorch 官方安装页选择对应的 CUDA
版 PyTorch。不要只依赖环境文件中可能安装到的 CPU 版本。

确认 GPU：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

确认项目和规格：

```powershell
python -m pytest -q
python mwir_spec.py --json
python mwir_telescope_design.py --check-only
```

## 6. 推荐的 GPU 后续流程

### 阶段 A：扩大 power+bend 多起点

不要再从平板材料 seed 开始。直接以版本控制的强弯候选做 32–128 个随机起点，
先低采样排名，再对前 8–16 名做较高采样优化。

```powershell
python mwir_power_bend_multistart.py `
  --input-lens datasets/lenses/mwir/mwir_strong_bent7_current_best.json `
  --output results/gpu-power-bend-multistart `
  --device cuda `
  --random-starts 64 `
  --top-starts 12 `
  --iterations 400 `
  --field-count 7 `
  --spp 96 `
  --validation-spp 256 `
  --ranking-spp 32 `
  --eval-spp 2048
```

如果显存不足，先按 `spp -> validation-spp -> top-starts` 的顺序降低。

### 阶段 B：对多起点冠军做结构与非球面优化

使用 `mwir_power_bent7_optimize.py --stage structural`，打开较高采样和默认
curriculum。先用保守权重；不要一开始就给直接 MTF 很大的权重。

```powershell
python mwir_power_bent7_optimize.py `
  --stage structural `
  --input-lens <阶段A输出的最佳JSON> `
  --output results/gpu-structural `
  --device cuda `
  --iterations 800 `
  --field-count 7 `
  --spp 128 `
  --validation-spp 384 `
  --eval-spp 4096 `
  --mtf-target 0.3 `
  --mtf-surrogate-weight 0.03 `
  --mtf-max-weight 2.0 `
  --curriculum-warmup-fraction 0.25 `
  --curriculum-ramp-fraction 0.50
```

建议至少改变 `--seed` 重复 8 次，不要只信单次运行。

### 阶段 C：独立验收

每个候选都检查：

```text
<output>/mwir_metrics.json
```

最终标准是：

```json
"pass": {
  "overall": true
}
```

在此之前，不生成或交付“最终” Zemax 文件。可先导出候选 ZMX 做 Zemax 二次
优化，但必须明确标注为中间候选。

## 7. 下一轮最值得研究的问题

1. 低采样几何 OTF 会出现相位折叠和种子噪声，需使用更高固定瞳采样或更稳定
   的波前/多频 MTF merit。
2. 当前材料布局 `Si / MgF2 / Si / MgF2 / ZnSe / CaF2 / Si` 比直接照搬专利
   的 `Si / Ge / Ge / Si / Si / Ge / Ge` 更好；专利原系统为 F/3.32 且含折叠
   镜、Dewar 窗和长像方空气段，不能原样缩放。
3. power+bend 多起点应保存每个起点的完整历史，并按独立验证集而非训练 merit
   排名。
4. GPU 广泛搜索后，如果仍无法把 RMS 压到约 10–15 µm 量级，应重新评估
   `F/2、9.6°、2.7–4.3 µm、七片` 组合的可达性，或允许更多镜片/缩小视场/
   放宽临时 MTF。

## 8. Git 约定

之前用户要求“未经明确允许不能提交或推送”；本次用户已经明确授权这一次提交
和推送。后续在新电脑上仍应恢复该约定：除非用户再次明确要求，不要自动
commit/push。
