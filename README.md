<div align="center">
    <img src="assets/logo.png" alt="DeepLens 标志" width="400px" >
</div>

# DeepLens

[English](./README_EN.md)

DeepLens 是一款面向端到端计算成像的可微光学镜头仿真器，支持多种光学模型（例如几何光线追迹、衍射波传播、混合光线—波动模型和代理 PSF 网络）。

DeepLens 可用于：(1) 端到端光学—算法协同设计，(2) 基于梯度的自动化光学设计，以及 (3) 通过图像仿真生成合成数据集。DeepLens 能够帮助研究人员快速构建原型并优化定制光学系统。

<p align="center">
    <a href="https://vccimaging.org/DeepLens/"><img src="https://img.shields.io/badge/Docs-blue?style=flat&logo=readthedocs&logoColor=white" alt="文档"/></a>
    <a href="https://github.com/singer-yang/DeepLens-tutorials"><img src="https://img.shields.io/badge/Tutorials-black?style=flat&logo=github&logoColor=white" alt="教程"/></a>
    <a href="#community"><img src="https://img.shields.io/badge/Community-Slack-4A154B?style=flat&logo=slack&logoColor=white" alt="社区"/></a>
    <a href="https://pypi.org/project/deeplens-core/"><img src="https://img.shields.io/pypi/v/deeplens-core?label=PyPI&color=orange&logo=pypi&logoColor=white" alt="PyPI"/></a>
    <a href="https://deepwiki.com/singer-yang/DeepLens"><img src="https://deepwiki.com/badge.svg" alt="询问 DeepWiki"/></a>
</p>

## 特性

1. **可微光学。** DeepLens 利用可微光学仿真实现准确、高效的梯度计算，从而支持镜头逆向设计。
2. **自动化设计。** DeepLens 通过基于梯度的优化算法和高级优化算法实现全自动光学设计，缩短多种光学系统（例如高阶非球面镜头、超表面和 AR/VR 显示器）的开发周期。
3. **多种光学模型。** DeepLens 除了支持几何光线追迹，还支持混合光线—波动模型、神经网络镜头表征和基于插值的模型。
4. **图像仿真。** DeepLens 可渲染具有空间变化、深度相关像差的照片级真实感图像；与 [End2end-Imaging](https://github.com/vccimaging/End2endImaging) 结合使用时，能够缩小仿真与真实场景之间的差距。

附加特性（可按需定制）：

1. **GPU 内核加速。** 通过适用于 NVIDIA 和 AMD 平台的定制 GPU 内核，实现超过 10 倍的加速和超过 90% 的 GPU 显存占用缩减，使其能够切实部署在本地笔记本电脑上。
2. **偏振光线追迹。** 支持偏振光线追迹，并可通过 [DiffTMM](https://github.com/AI4Optics/DiffTMM) 对薄膜进行逆向设计。
3. **非序列光线追迹。** 支持用于杂散光分析与优化的可微非序列光线追迹模型。
4. **分布式优化。** 支持十亿规模光线追迹和高分辨率（>100k x 100k）衍射传播的分布式仿真与优化。

## 应用

#### 1. 镜头分析与图像仿真

DeepLens 支持全面的镜头分析（点列图、PSF、MTF、畸变等），以及具有空间变化、深度相关像差的照片级真实感图像仿真。

<div align="center">
    <img src="assets/feature.png" alt="镜头分析与图像仿真"/>
</div>

#### 2. 自动化几何镜头设计

借助基于梯度的优化算法和高级优化算法，从零开始实现全自动镜头设计。

> **注意：** 自动化镜头设计目前由 [**AutoLens**](https://github.com/AI4Optics/AutoLens) 项目积极维护。如果你的重点是自动化镜头设计，建议改用 AutoLens 仓库，因为该仓库会针对这一用途持续提供专门的更新与改进。

[![论文](https://img.shields.io/badge/NatComm-2024-orange)](https://www.nature.com/articles/s41467-024-50835-7) [![快速开始](https://img.shields.io/badge/AutoLens-green)](https://github.com/AI4Optics/AutoLens)

<div align="center">
    <img src="assets/autolens1.gif" alt="AutoLens" height="270px"/>
    <img src="assets/autolens2.gif" alt="AutoLens" height="270px"/>
</div>

#### 3. 神经网络镜头 PSF 表征

一种用于高效表征镜头 PSF 的代理网络，支持具有空间变化像差和离焦的快速、准确图像仿真。

[![论文](https://img.shields.io/badge/TPAMI-2023-orange)](https://ieeexplore.ieee.org/document/10209238) [![项目](https://img.shields.io/badge/Project-green)](https://github.com/vccimaging/Aberration-Aware-Depth-from-Focus)

<div align="center">
    <img src="assets/implicit_net.png" alt="神经网络镜头 PSF 表征" height="150px"/>
</div>

#### 4. 混合光线—波动光学模型

用于准确仿真镜头像差和衍射元件的可微光线—波动光学模型，支持端到端折射—衍射混合镜头设计。

[![论文](https://img.shields.io/badge/SiggraphAsia-2024-orange)](https://dl.acm.org/doi/10.1145/3680528.3687640)

<div align="center">
    <img src="assets/hybridlens.png" alt="混合光线—波动光学模型" height="200px"/>
</div>

#### 5. 非序列模型与偏振追迹

通过非序列偏振追迹，准确仿真光线穿过几何波导 AR 显示器时的偏振状态；针对出耦合眼盒响应，对镀膜逆向设计进行端到端优化。

<div align="center">
    <img src="assets/diffgwg.jpg" alt="用于 AR 波导显示器的非序列偏振光线追迹" height="200px"/>
</div>

#### 6. 端到端计算成像

DeepLens 是端到端可微计算成像框架 [**End2endImaging**](https://github.com/vccimaging/End2endImaging) 中的可微光学引擎。End2endImaging 将光学、传感器/ISP 仿真和神经重建网络集成到单个 PyTorch 计算图中，从而实现对整个相机处理流程的联合优化。

<div align="center">
    <img src="assets/end2end.png" alt="End2endImaging" height="200px"/>
</div>

## 安装

克隆此仓库：

```
git clone https://github.com/singer-yang/DeepLens
cd DeepLens
```

创建 conda 环境：

```
conda create -n deeplens_env python=3.12
conda activate deeplens_env

# Linux 和 macOS
pip install torch torchvision
# Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

或

```
conda env create -f environment.yml -n deeplens_env
```

运行演示代码：

```
python 0_hello_geolens.py
```

## 中波红外望远系统设计入口

仓库提供了面向 2.7–4.3 微米透射式系统的一阶规格检查和 GeoLens 初始结构入口：

```powershell
conda activate deeplens_env
python mwir_spec.py
python mwir_spec.py --json
python mwir_telescope_design.py --check-only
python mwir_telescope_design.py --device cpu --iterations 0 --output results\mwir-initial
```

当前默认规格来自 Zemax 系统概要图：Y 方向视场为 -4.8° 到 +4.8°，即 9.6°全视场；
边缘场点半像高为 47.1454 mm，入瞳直径为 280 mm。程序由像高和视场推导出约
561.44 mm 有效焦距和 F/2.005，而不是再用尚未确认的 320x256、30 微米探测器反推
F/0.261。默认 `transmission_baseline` 方案使用圆形等效虚拟焦面完成初始结构设计，
该焦面约为 66.67 × 66.67 mm，其对角线对应 94.2908 mm 的完整 Y 向像高包络；
它只用于数值计算，不是最终探测器规格。

自定义视场、像高和入瞳时，应通过设计入口检查实际命令行参数：

```powershell
python mwir_telescope_design.py --check-only `
  --field-y-deg 9.6 `
  --image-height-mm 47.1454 `
  --entrance-pupil-mm 280
```

### 初始处方、优化与评价

只生成初始处方：

```powershell
python mwir_telescope_design.py --device cpu --iterations 0 `
  --output results\mwir-initial
```

初始结构会先按实测近轴焦距校准组合光焦度，再重新对焦到无穷远。梯度优化使用
100 km 有限物距近似无穷远，正式 MTF 和像高/畸变评价则直接追迹无穷远平行光。
低采样数值验收可用：

```powershell
python mwir_telescope_design.py --device cpu --iterations 0 `
  --evaluate --eval-spp 64 `
  --output results\mwir-initial-eval
```

CPU 最小优化烟雾测试应显式降低采样；`--iterations N` 现在恰好执行 N 次参数更新：

```powershell
python mwir_telescope_design.py --device cpu --iterations 1 `
  --num-ring 2 --num-arm 2 --spp 32 `
  --evaluate --eval-spp 64 `
  --output results\mwir-opt-smoke
```

从已有 JSON 继续时，使用 `--input-lens` 开启一个新阶段。程序只恢复光学处方，
不会重新校准光焦度、重新对焦或恢复旧 Adam 状态；波长、物距、前置光阑、280 mm
入瞳、焦面半径、分辨率和镜片数会重新校验，机械约束也会重新应用。新的输出目录
必须为空且与输入阶段分开，以免覆盖旧 metadata、检查点或最终处方。程序还会读取
伴随的 `mwir_design_metadata.json`，默认拒绝静默改变原视场、像高和目标焦距；若确实
要用旧处方改做另一组目标，必须显式加入 `--allow-retarget`。推荐先稳定场映射，再改善 RMS：

```powershell
# 阶段 1：冻结曲率，优先稳定焦距/像高映射
python mwir_telescope_design.py `
  --input-lens results\mwir-initial\mwir_initial.json `
  --device cpu --iterations 20 `
  --lrs 2e-3 0 2e-4 2e-6 `
  --rms-weight 0.3 `
  --field-weight 1.5 --field-max-weight 2 `
  --num-ring 8 --num-arm 4 --spp 128 `
  --output results\mwir-stage-field --evaluate

# 阶段 2：从上一阶段最终处方继续优化像质
python mwir_telescope_design.py `
  --input-lens results\mwir-stage-field\mwir_final.json `
  --device cpu --iterations 100 `
  --lrs 1e-3 1e-7 5e-4 2e-6 `
  --rms-weight 1 `
  --field-weight 1.5 --field-max-weight 2 `
  --num-ring 8 --num-arm 4 --spp 512 `
  --output results\mwir-stage-rms --evaluate
```

当前第一阶段默认学习率为 `[2e-3, 2e-7, 2e-4, 2e-6]`，顺序是间距、曲率、
圆锥常数和非球面系数。CPU 上建议先用 1–5 步和较小 `spp` 检查趋势，再增加迭代；
若环境确认可用 CUDA，可把 `--device cpu` 改为 `--device cuda`。

MWIR 优化默认只在检查点写入 `optimization/iter*.json`，不会生成耗时的完整分析图；
正式长优化需要这些图时再加 `--checkpoint-analysis`。第一阶段默认关闭曲面形状修正
和口径裁剪，处方稳定后再启用 `--shape-control`，最后才使用 `--prune-surfaces`。
目标像高/场映射损失已从通用正则项中独立出来：在无穷远共轭、两个子午平面和默认
9 个等角场点上，对全部三个训练波长追迹瞄准前置光阑中心的可微主光线，使训练与
正式主光线畸变验收保持一致。默认权重为 1.0，可用
`--field-weight` 调整；最坏场点还会由 `--field-max-weight` 额外约束，通用机械和
曲面正则权重可用 `--regularization-weight` 调整。

当前梯度目标包括光斑 RMS、有效光线比例、目标像高映射和处方正则；MTF 0.3 是
`--evaluate` 的验收阈值，并不是直接参与反向传播的 MTF 损失。优化器会清零局部
NaN/Inf 梯度、按参数组裁剪梯度，并回滚非有限参数或使最低有效率恶化的更新。
大口径 float32 表面若某个高阶非球面基函数超出数值范围，会保留该系数但冻结其优化；
例如本系统的 a18 不参与 Adam，a4–a16 仍正常优化。

### 数值评价定义

`--evaluate` 在 2.7、3.5、4.3 微米和 Y 向 0°、3.36°、4.8°场点上评价。系统
MTF 采用早期工程估计：

```text
几何光线截距 OTF × 理想无中心遮拦圆孔衍射 MTF × 100% 填充率矩形像元 MTF
```

这不是严格的波像差/Huygens MTF，最终处方仍应使用经过验证的物理光学模型复核。
总验收同时要求：EFL 和 F/#误差各不超过 1%，目标像高映射误差和传统畸变各不超过
0.5%，系统 MTF 不低于 0.3，最低有效光线比例不低于 0.7，入瞳误差不超过 1%，
镜片数不超过 7。这里的“渐晕”只是有效光线比例，不含 cos⁴、材料吸收和镀膜损失。

一阶评价明确区分三个容易混淆的量：严格 EFL 先由微小瞳高轴上光线外推高斯近轴
焦面，再在该焦面用正负小视场主光线外推零视场板尺；F/# 和衍射 MTF 使用这个严格
EFL。传统畸变使用每个波长、每个平面在当前传感器面上的局部主光线板尺。任务像高
映射则始终使用规格固定的 561.4396 mm 目标焦距。`lens.foclen/lens.fnum` 仍写入
`mwir_metrics.json` 作为 DeepLens 缓存诊断值，但不再把传感器离开高斯焦面造成的
板尺变化误报成传统畸变，也不会用板尺焦距掩盖真正的 EFL 漂移。两平面验收取最坏值。
当前处方只包含同轴旋转对称面，因此正式场点取 0→+4.8° 并代表 -4.8° 半场；若以后
加入偏心或倾斜面，必须把验收扩展到正负全场。

输出文件按所执行阶段生成：

- 始终生成 `mwir_design_metadata.json` 和 `mwir_initial.json`。
- `--iterations > 0` 额外生成 `mwir_final.json` 和 `optimization/iter*.json`；只有
  `--checkpoint-analysis` 才生成检查点分析图。
- `--evaluate` 生成 `mwir_metrics.json`。
- `--analyze` 在优化前生成初始处方的完整分析图。

### 探测器和历史方案

探测器像元间距确认前，`mwir_metrics.json` 中的奈奎斯特频率和系统 MTF 标记为
临时仿真值；阵列格式可稍后独立确认。当前 47.1454 mm 表示 Y 向半像高，完整高度为
94.2908 mm；它不是探测器半对角。确认矩形探测器宽高比后，应先固定有效高度为
94.2908 mm，再按宽高比计算宽度。水平视场、对角视场和最终探测器型号仍需补齐。

若要复现旧的 42 微弧度方案，必须显式启用；该路径只用于历史对比：

```powershell
python mwir_telescope_design.py --scheme large_fpa `
  --two-pixel-resolution-urad 42 `
  --simulation-pixel-pitch-um 30 `
  --device cpu --iterations 0 `
  --output results\mwir-history-42urad
```

`cassegrain_equivalent` 目前只是“继承卡塞格林一阶指标”的透射基线别名，不会导入
反射镜曲率、间隔、中心遮拦或机械总长，不能理解为自动把卡塞格林处方转换成透射式
处方。20°C 当前也只记录在规格中，尚未包含热折射率、热膨胀、材料吸收、镀膜和公差。

DeepLens 仓库结构：

```
DeepLens/
│
├── deeplens/
│   ├── lens.py             (镜头基类)
│   ├── geolens.py          (折射镜头)
│   ├── hybridlens.py       (折射—衍射混合镜头)
│   ├── diffraclens.py      (衍射镜头)
│   ├── defocuslens.py      (弥散圆模型)
│   ├── psfnetlens.py       (代理镜头 PSF 模型)
│   ├── ...
│   ├── geometric_surface/  (折射与反射表面)
│   ├── diffractive_surface/(衍射表面)
│   ├── phase_surface/      (相位表面)
│   ├── light/              (光线与波动类 Ray、ComplexWave)
│   ├── material/           (玻璃/塑料目录与 refractiveindex.info 数据)
│   ├── imgsim/             (PSF 卷积与蒙特卡洛图像仿真)
│   ├── geolens_pkg/        (eval、optim、vis、io 混入类)
│   └── surrogate/          (MLP、Siren 神经代理模型)
│
├── 0_hello_geolens.py     (入门教程)
├── mwir_spec.py           (中波红外一阶规格检查)
├── mwir_telescope_design.py (中波红外初始结构与优化入口)
├── ...
└── 9_diffractive_surfaces.py (衍射面示例)
```

<a id="community"></a>

## 社区

加入我们的 [Slack](https://join.slack.com/t/deeplens/shared_invite/zt-2wz3x2n3b-plRqN26eDhO2IY4r_gmjOw) 工作区和微信群（singeryang1999），与核心贡献者交流、获取最新行业动态并参与社区。如有任何问题，请联系 Xinge Yang（xinge.yang@kaust.edu.sa）。

## 贡献

我们欢迎所有贡献。若要开始参与，请阅读[贡献指南](./CONTRIBUTING.md)，或查看[开放问题](https://github.com/users/singer-yang/projects/2)。所有项目参与者都应遵守我们的[行为准则](./CODE_OF_CONDUCT.md)。你可以在[贡献者名单](./CONTRIBUTORS.md)及下方查看贡献者：

<a href="https://github.com/singer-yang/DeepLens/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=singer-yang/DeepLens" alt="DeepLens 贡献者" />
</a>

## 引用

如果你在研究中使用了 DeepLens，请引用以下论文。更多信息请参阅 [DeepLens 历史](./CITATION.md)。

```bibtex
@article{yang2024curriculum,
  title={Curriculum learning for ab initio deep learned refractive optics},
  author={Yang, Xinge and Fu, Qiang and Heidrich, Wolfgang},
  journal={Nature communications},
  volume={15},
  number={1},
  pages={6572},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
