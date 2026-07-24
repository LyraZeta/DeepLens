# `deeplens` 包结构

本文概述 `deeplens` 包的文件结构。

## 顶层文件

-   **`__init__.py`**：包入口。导出 `init_device()`、公共类（`DeepObj`、`Material`、`Ray`、`ComplexWave`）、传播辅助函数以及所有镜头类型。

-   **`base.py`**：定义基类 `DeepObj`，通过张量属性检查提供 `to(device)`、`astype(dtype)` 和 `clone()`。

-   **`config.py`**：光学配置常量（`DEPTH`、`SPP_*`、`PSF_KS`、`WAVE_RGB`、`EPSILON` 等）。

-   **`loss.py`**：用于光学优化的 PSF 相关损失函数。

-   **`utils.py`**：通用工具（图像 I/O、PSNR/SSIM 等批量指标、归一化、视频创建、日志记录、随机种子设置，以及 `interp1d`、`grid_sample_xy`、`foc_dist_balanced`、`diff_float`、`diff_quantize` 等光学辅助函数）。

## 镜头类

-   **`lens.py`**：所有镜头系统的基类 `Lens`。
-   **`geolens.py`**：`GeoLens`——折射镜头系统（可微光线追迹）。
-   **`diffraclens.py`**：`DiffractiveLens`——近轴衍射镜头系统。
-   **`hybridlens.py`**：`HybridLens`——折射—衍射混合系统。
-   **`defocuslens.py`**：`DefocusLens`——散焦（弥散圆）模型。
-   **`psfnetlens.py`**：`PSFNetLens`——用于 PSF 预测的神经网络代理模型。

## 子包

-   **`light/`**：光线追迹与波动光学。
    -   `ray.py`：用于几何光线追迹的 `Ray` 类。
    -   `wave.py`：`ComplexWave` 及传播方法（ASM、Fresnel、Fraunhofer、Rayleigh-Sommerfeld）。

-   **`material/`**：材料属性与色散模型（CDGM、SCHOTT、PLASTIC2022、MISC 目录）。

-   **`geometric_surface/`**：用于折射镜头的几何面（`Spheric`、`Aspheric`、`Aperture`、`Plane`、`Cubic`、`Mirror`、`Prism`、`QTypeFreeform`、`Spiral`、`ThinLens`）。

-   **`diffractive_surface/`**：衍射光学元件与超表面（`Binary2`、`Fresnel`、`Grating`、`Pixel2D`、`Zernike`、`ThinLens`）。

-   **`phase_surface/`**：纯相位面（`Binary2Phase`、`CubicPhase`、`FresnelPhase`、`GratingPhase`、`NURBSPhase`、`PolyPhase`、`VortexPhase`、`ZernikePhase`，以及在 `qphase.py` 中定义但尚未从包入口导出的 `QuarticPhase`）。

-   **`imgsim/`**：图像仿真。
    -   `monte_carlo.py`：用于可微 PSF 累积的 `forward_integral()`。
    -   `psf.py`：多种 PSF 卷积方式（单一、空间变化、深度变化、逐像素）。

-   **`geolens_pkg/`**：`GeoLens` 混入模块（PSF 计算、评估、优化、I/O、2D/3D 可视化）。

-   **`surrogate/`**：神经网络代理模型（`MLP`、`MLPConv`、`Siren`、`ModulateSiren`、`PSFNet_MLPConv`）。
