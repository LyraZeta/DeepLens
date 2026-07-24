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
