"""DeepLens HybridLens 类的“你好，世界！”示例。

混合镜头将折射 GeoLens 与位于其后的衍射光学元件（DOE）结合。这里采用可微分的
光线—波动模型：相干光线追迹计算 DOE 平面上的复波前（捕获几何像差），DOE 调制
相位，再由角谱法将光场传播到传感器。

这是一个最简入门示例：加载混合镜头、绘制布局、计算单个轴上 PSF，并将测试图卡
与 RGB PSF 卷积以模拟图像。完整的端到端联合优化循环见 6_hybridlens_design.py。

注意：
    为准确追迹相位，HybridLens 使用 float64。

技术论文：
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu,
    Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design
    with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.
"""

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import HybridLens
from deeplens.config import WAVE_RGB
from deeplens.imgsim import conv_psf

# 为准确追迹相位，HybridLens 要求默认 dtype 为 float64。
torch.set_default_dtype(torch.float64)

# =====================================================================
# 镜头加载
# =====================================================================
# 加载一个混合镜头示例（A489 折射设计 + Binary2 DOE）。
lens = HybridLens(filename="./datasets/lenses/hybridlens/a489_doe.json")
print(f"HybridLens: {len(lens.geolens.surfaces)} refractive surface(s) + "
      f"a {type(lens.doe).__name__} DOE.")

# 将镜头对焦到 1 m 处（深度以 mm 为单位，取负值）。
lens.refocus(foc_dist=-1000.0)

# =====================================================================
# 布局与 PSF 分析
# =====================================================================
save_name = "./hello_hybridlens"

# 绘制镜头布局：折射元件、追迹光线以及 DOE 到传感器的波传播区域。
lens.draw_layout(save_name=f"{save_name}_layout.png")
print(f"Saved lens layout to {save_name}_layout.png")

# 计算单个轴上 PSF。光线—波动模型会一次性捕获所有衍射级次的贡献。相干光线追迹
# 需要 >= 1e6 个采样点。
psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=64, spp=1_000_000)
print(f"On-axis PSF: shape {tuple(psf.shape)}, sum {psf.sum():.3f}")

# =====================================================================
# 图像模拟（PSF 卷积）
# =====================================================================
# 构建 RGB PSF（每个波长一个），并与测试图卡卷积，以模拟混合镜头如何对远处场景
# 成像（轴上 PSF，空间不变）。令传感器匹配输入图像，而不缩放图像。
img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float()[:3] / 255.0
img = img.unsqueeze(0)  # [1, 3, H, W]
lens.geolens.set_sensor_res((img.shape[-1], img.shape[-2]))  # (W, H)；PSF 对 geolens 传感器采样

psf_rgb = torch.stack(
    [lens.psf(points=[0.0, 0.0, -10000.0], ks=128, wvln=w, spp=1_000_000) for w in WAVE_RGB],
    dim=0,
).float()  # [3, ks, ks]，渲染使用 fp32
img = img.to(psf_rgb)  # 与 PSF 的 dtype 和设备保持一致
img_render = conv_psf(img, psf_rgb)
save_image(img_render.clamp(0, 1), f"{save_name}_render.png")
print(f"Saved simulated image to {save_name}_render.png")
