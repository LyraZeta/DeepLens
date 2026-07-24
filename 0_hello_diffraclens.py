"""DeepLens DiffractiveLens 类的“你好，世界！”示例。

本代码从 JSON 配置文件加载一个近轴衍射镜头（传感器前方的单个 Fresnel DOE）。
每个光学元件均建模为相位函数，并使用角谱法（ASM）将波前传播到传感器。随后计算
轴上和轴外点光源（位于无穷远和有限深度）的 PSF，最后将测试图卡与 RGB 点扩散
函数卷积以模拟图像。

注意：
    为保证波传播步骤的数值稳定性，DiffractiveLens 使用 float64。PSF 与 GeoLens
    采用相同的 points 约定（x、y 归一化到 [-1, 1]，z 为以 mm 表示的深度）；
    在近轴范围内支持轴外光源。

技术论文：
    [1] Vincent Sitzmann et al., "End-to-end optimization of optics and image
        processing for achromatic extended depth of field and super-resolution
        imaging," SIGGRAPH 2018.
    [2] Qilin Sun et al., "Learning Rank-1 Diffractive Optics for Single-shot
        High Dynamic Range Imaging," CVPR 2020.
"""

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.imgsim import conv_psf

# =====================================================================
# 镜头加载
# =====================================================================
# 从 JSON 配置文件加载一个最简衍射镜头（单个 Fresnel DOE，聚焦于 f0 = 50 mm，
# 位于传感器前一个焦距处）。使用 float64 运行，以符合文档字符串中关于波传播步骤
# 数值稳定性的说明（否则 DiffractiveLens 默认使用 float32）。
lens = DiffractiveLens(
    filename="./datasets/lenses/diffraclens/fresnel.json", dtype=torch.float64
)

# =====================================================================
# PSF 分析
# =====================================================================
save_name = "./hello_diffraclens"

# 点采用 (x, y, z) 约定：x、y 归一化到 [-1, 1]（传感器半宽/半高），z 为以 mm
# 表示的深度（位于无穷远的物体取 -inf）。

# 无穷远物体（平面波输入）的轴上 PSF。
psf_inf = lens.psf(points=[0.0, 0.0, float("-inf")], wvln=0.55, ks=128)
save_image(psf_inf[None].clamp(min=0), f"{save_name}_psf_inf.png", normalize=True)

# 有限物距（点光源/球面波输入）的轴上 PSF。
psf_near = lens.psf(points=[0.0, 0.0, -500.0], wvln=0.55, ks=128)
save_image(psf_near[None].clamp(min=0), f"{save_name}_psf_near.png", normalize=True)

# 轴外 PSF：归一化视场 x = 0.7 处的准直光源。
psf_off = lens.psf(points=[0.7, 0.0, float("-inf")], wvln=0.55, ks=128)
save_image(psf_off[None].clamp(min=0), f"{save_name}_psf_offaxis.png", normalize=True)

# =====================================================================
# 图像模拟（PSF 卷积）
# =====================================================================
# 模拟镜头如何对无穷远场景成像。令传感器匹配输入图像（而不缩放图像），再将图卡
# 与设计波长 0.55 um 对应的无穷远 PSF 卷积。
img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float()[:3] / 255.0
img = img.unsqueeze(0)  # [1, 3, H, W]
lens.set_sensor_res((img.shape[-1], img.shape[-2]))  # (W, H)

psf_render = lens.psf(points=[0.0, 0.0, float("-inf")], wvln=0.55, ks=64)
psf_rgb = psf_render[None].repeat(3, 1, 1).float()  # [3, ks, ks]，渲染使用 fp32
img = img.to(psf_rgb)
img_render = conv_psf(img, psf_rgb)
save_image(img_render.clamp(0, 1), f"{save_name}_render.png")
