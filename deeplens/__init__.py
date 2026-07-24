"""DeepLens——可微光学镜头仿真器。"""

import torch


def init_device():
    """初始化并返回默认计算设备（CUDA 或 CPU）。

    GPU 可用时返回 `cuda`，否则返回 `cpu`。有意不自动选择 MPS（Apple Silicon）：
    DeepLens 的波传播/相干光线追迹依赖 float64，而 MPS 后端不支持 float64
    （`Cannot convert a MPS Tensor to float64`），因此自动选择 MPS 会导致所有
    双精度工作流崩溃。Apple Silicon 因而回退到 CPU。仅需在 MPS 上运行
    float32 几何路径的用户仍可显式传入 `device="mps"`。

    返回：
        device (torch.device): 所选计算设备；GPU 可用时为 `cuda`，否则为 `cpu`。
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA: {device_name} for DeepLens")
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        if torch.backends.mps.is_available():
            print(
                "Apple MPS detected but not used (no float64 support); "
                "using CPU for DeepLens. Pass device='mps' to force float32-only MPS."
            )
        else:
            print("Using CPU for DeepLens")
    return device


from .base import DeepObj

from .material import Material
from .light import (
    AngularSpectrumMethod,
    ComplexWave,
    FresnelDiffraction,
    Fresnel_zmin,
    FraunhoferDiffraction,
    Nyquist_ASM_zmax,
    Ray,
    RayleighSommerfeld,
    RayleighSommerfeldIntegral,
    ScalableASM,
)

# 镜头类
from .lens import Lens
from .geolens import GeoLens
from .hybridlens import HybridLens
from .diffraclens import DiffractiveLens
from .defocuslens import DefocusLens
from .psfnetlens import PSFNetLens

# geolens 扩展
from .geolens_pkg import *

# 实用工具
from .utils import *

__all__ = [
    "init_device",
    "DeepObj",
    "Material",
    "Ray",
    "ComplexWave",
    "AngularSpectrumMethod",
    "ScalableASM",
    "FresnelDiffraction",
    "FraunhoferDiffraction",
    "RayleighSommerfeld",
    "RayleighSommerfeldIntegral",
    "Nyquist_ASM_zmax",
    "Fresnel_zmin",
    "Lens",
    "GeoLens",
    "HybridLens",
    "DiffractiveLens",
    "DefocusLens",
    "PSFNetLens",
]
