# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""近轴衍射镜头模型。近轴衍射模型中的每个光学元件（镜片、DOE、超表面等）
都建模为相位函数。这种简化光学模型易于使用，但对许多实际应用而言通常精度不足。

参考:
    [1] Vincent Sitzmann*, Steven Diamond*, Yifan Peng*, Xiong Dun, Stephen Boyd, Wolfgang Heidrich, Felix Heide, Gordon Wetzstein, "End-to-end optimization of optics and image processing for achromatic extended depth of field and super-resolution imaging," Siggraph 2018.
    [2] Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide. "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range Imaging," CVPR 2020.
"""

import json
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from .config import DEFAULT_WAVE, DEPTH, EPSILON, PSF_KS, WAVE_RGB
from .lens import Lens
from .diffractive_surface import (
    Binary2,
    DiffractedRotation,
    Fresnel,
    Pixel2D,
    Rank1,
    RotationallySymmetric,
    ThinLens,
    Zernike,
)
from .imgsim import conv_psf
from .utils import diff_float
from .light import ComplexWave


class DiffractiveLens(Lens):
    """将每个元件建模为相位面的近轴衍射镜头。

    每个光学元件（会聚镜片、DOE、超表面等）均表示为作用于入射复波前的相位函数。
    表面之间以及到传感器的自由空间传播由 `ComplexWave.prop_to` 处理；该方法根据
    传播距离选择带限 ASM 或单 FFT Fresnel 衍射。此模型简单且快速，但仅在近轴
    区域内准确（不考虑高阶几何像差）。

    属性:
        surfaces (list): 按顺序排列的衍射/相位面列表。
        d_sensor (torch.Tensor): 从第一表面 (z=0) 到传感器平面的距离 [mm]。

    说明:
        镜头参数默认为 `torch.float32`；如需更高精度的波传播，请传入
        `dtype=torch.float64`。
    """

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float32,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """初始化衍射镜头。

        参数:
            filename (str or None, optional): 镜头配置 JSON 文件的路径。提供时从文件
                加载镜头配置；否则使用空表面列表和默认的 8x8 mm、2000x2000 px
                传感器。默认为 None。
            device (str or None, optional): 计算设备（'cpu' 或 'cuda'）。为 None 时
                由基类 `Lens` 决定。默认为 None。
            dtype (torch.dtype, optional): 镜头参数的数据类型。如需更高精度的波传播，
                请传入 `torch.float64`。默认为 `torch.float32`。
            primary_wvln (float, optional): 主要设计波长 [µm]。调用方法时未显式提供
                `wvln`，则使用此值。默认为 `DEFAULT_WAVE`。
            wvln_rgb (sequence of float, optional): RGB 计算使用的三个波长，按
                [R, G, B] 排列，单位为 µm。默认为 `WAVE_RGB`。
            obj_depth (float, optional): 默认物体深度 [mm]。调用方法时未显式提供
                `depth`，则使用此值。默认为 `DEPTH`。
        """
        super().__init__(
            device=device,
            dtype=dtype,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )

        # 加载镜头文件
        if filename is not None:
            self.read_lens_json(filename)
        else:
            self.surfaces = []
            # 未提供文件时设置默认传感器尺寸和分辨率
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)

        self.astype(self.dtype)

        # 使用总光程长度（从第一元件到传感器）作为焦距
        if hasattr(self, "d_sensor"):
            self.foclen = float(self.d_sensor)
            self.calc_fov()

        # 将所有张量（表面、传感器参数）移动到目标设备。
        self.to(self.device)

    def read_lens_json(self, filename):
        """从 JSON 文件加载镜头配置。

        从指定 JSON 文件读取镜头参数，包括传感器配置和衍射面。若未提供
        sensor_size 或 sensor_res，则分别使用 8mm x 8mm 和 2000x2000 pixels
        的默认值。

        参数:
            filename (str): JSON 配置文件路径。
        """
        assert filename.endswith(".json"), "File must be a .json file."

        with open(filename, "r", encoding="utf-8") as f:
            # 镜头通用信息
            data = json.load(f)
            self.d_sensor = torch.tensor(data["d_sensor"])
            self.lens_info = data.get("info", "None")

            # 读取 sensor_size，缺失时使用默认值
            if "sensor_size" in data:
                sensor_size = tuple(data["sensor_size"])
            else:
                sensor_size = (8.0, 8.0)
                print(
                    f"Sensor_size not found in lens file. Using default: {sensor_size} mm. "
                    "Consider specifying sensor_size in the lens file or using set_sensor()."
                )

            # 读取 sensor_res，缺失时使用默认值
            if "sensor_res" in data:
                sensor_res = tuple(data["sensor_res"])
            else:
                sensor_res = (2000, 2000)
                print(
                    f"Sensor_res not found in lens file. Using default: {sensor_res} pixels. "
                    "Consider specifying sensor_res in the lens file or using set_sensor()."
                )

            # 配置传感器（同时设置 pixel_size 和 r_sensor）。
            self.set_sensor(sensor_size, sensor_res)

            # 加载衍射面/元件
            d = 0.0
            self.surfaces = []
            for surf_dict in data["surfaces"]:
                surf_dict["d"] = d

                if surf_dict["type"].lower() == "binary2":
                    s = Binary2.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "fresnel":
                    s = Fresnel.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "pixel2d":
                    s = Pixel2D.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "thinlens":
                    s = ThinLens.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "zernike":
                    s = Zernike.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "rank1":
                    s = Rank1.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "diffractedrotation":
                    s = DiffractedRotation.init_from_dict(surf_dict)
                elif surf_dict["type"].lower() == "rotationallysymmetric":
                    s = RotationallySymmetric.init_from_dict(surf_dict)
                else:
                    raise ValueError(
                        f"Diffractive surface type {surf_dict['type']} not implemented."
                    )

                self.surfaces.append(s)
                d_next = surf_dict["d_next"]
                d += d_next

    def write_lens_json(self, filename):
        """将镜头配置写入 JSON 文件。

        将包括传感器配置和衍射面数据在内的全部镜头参数保存到指定文件。

        参数:
            filename (str): JSON 文件输出路径。
        """
        assert filename.endswith(".json"), "File must be a .json file."

        # 将镜头保存到文件
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["surfaces"] = []
        data["d_sensor"] = round(self.d_sensor.item(), 3)
        data["sensor_size"] = [
            round(float(self.sensor_size[0]), 3),
            round(float(self.sensor_size[1]), 3),
        ]
        data["sensor_res"] = self.sensor_res

        # 保存衍射面
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i + 1}

            if isinstance(s, Pixel2D):
                surf_data = s.surf_dict(filename.replace(".json", "_pixel2d.pth"))
            elif isinstance(s, (Rank1, RotationallySymmetric)):
                surf_data = s.surf_dict(filename.replace(".json", f"_surf{i + 1}.pth"))
            else:
                surf_data = s.surf_dict()

            surf_dict.update(surf_data)

            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = (
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item()
                )
            else:
                # 最后一个表面：到传感器的距离。read_lens_json 要求每个表面均有
                # d_next，因此文件必须始终包含该字段。
                surf_dict["d_next"] = round(
                    float(self.d_sensor) - self.surfaces[i].d.item(), 3
                )

            data["surfaces"].append(surf_dict)

        # 将数据保存到文件
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # =============================================
    # 实用工具
    # =============================================
    def __call__(self, wave):
        """使波通过镜头系统传播（`forward` 的别名）。

        参数:
            wave (ComplexWave): 进入镜头系统的输入波场。

        返回:
            wave (ComplexWave): 传感器平面处的输出波场。
        """
        return self.forward(wave)

    def forward(self, wave):
        """使波通过衍射镜头系统传播到传感器。

        依次应用各衍射面的相位调制（表面之间进行自由空间传播），再将波传播到
        传感器平面（绝对位置 z = d_sensor [mm]）。自由空间传播委托给
        `ComplexWave.prop_to`，后者根据距离选择带限 ASM 或单 FFT Fresnel 衍射。

        参数:
            wave (ComplexWave): 进入镜头系统的输入波场。

        返回:
            wave (ComplexWave): 传感器平面处的输出波场。
        """
        # 传播到 DOE
        for surf in self.surfaces:
            wave = surf(wave)

        # 传播到传感器
        wave = wave.prop_to(self.d_sensor.item())

        return wave

    # =============================================
    # 图像仿真
    # =============================================
    def render_mono(self, img, wvln=None, ks=None, method="fft"):
        """通过使用点扩散函数对图像做卷积来模拟单色镜头模糊。

        参数:
            img (torch.Tensor): 输入图像，shape (B, 1, H, W)。
            wvln (float, optional): 波长 [µm]。为 None（默认）时使用
                `self.primary_wvln`。
            ks (int, optional): PSF 核尺寸 [pixels]。为 None（默认）时使用完整
                传感器分辨率（`max(self.sensor_res)`）。
            method (str, optional): 传给 `conv_psf` 的卷积后端，可为 ``"conv"``
                或 ``"fft"``。默认为 ``"fft"``，因为默认 `ks`（完整传感器分辨率）
                使直接卷积不切实际。

        返回:
            img_render (torch.Tensor): 应用镜头模糊后的渲染图像，shape (B, 1, H, W)。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        # 无穷远物体的轴上 PSF。psf() 对单点返回 [ks, ks]；添加前置通道维度，
        # 以供 conv_psf 使用 -> (1, ks, ks)。
        psf = self.psf(
            points=[0.0, 0.0, float("-inf")], wvln=wvln, ks=ks
        ).unsqueeze(0)
        img_render = conv_psf(img, psf, method=method)
        return img_render

    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """计算一个或多个点光源的单色 PSF。

        支持离轴点光源。函数签名遵循 `Lens.psf` 和 `GeoLens.psf`。

        参数:
            points (torch.Tensor or list): 点光源坐标，shape [N, 3] 或 [3]。
                x、y 归一化到 [-1, 1]（相对于传感器半宽/半高）；z 为以 mm 表示的
                深度（负值；无穷远物体为 -inf）。
            wvln (float, optional): 波长 [µm]。为 None（默认）时使用
                `self.primary_wvln`。
            ks (int, optional): PSF 核尺寸 [pixels]。传入 `ks=None` 可使用完整
                传感器分辨率（`max(self.sensor_res)`）。默认为 `PSF_KS`。
            **kwargs: 模型特定选项：
                - recenter (bool): ks x ks 核的居中方式（两种选项都使离轴 PSF
                  位于核中心）。为 True 时围绕测得的峰值（传感器平面强度的 argmax）
                  裁剪；为 False（默认）时围绕视场点的透视（针孔）像裁剪。镜头在
                  物理上形成倒像，但结果会翻转，因此按传感器/光源符号约定报告 PSF
                  （+x 光源 -> +x）。
                - upsample_factor (int): 满足 Nyquist 采样约束的场上采样倍数。
                  为 None（默认）时选择使场分辨率接近 4000 x 4000 的倍数。

        返回:
            psf (torch.Tensor): PSF 强度图（归一化至总和为 1）；单点时 shape
                [ks, ks]，批量输入时 shape [N, ks, ks]。

        说明:
            使用单个（非平铺）传播窗口，因此很大的离轴视场可能出现移相/混叠问题；
            参见 "Modeling off-axis diffraction with the least-sampling angular
            spectrum method"。
        """
        recenter = kwargs.get("recenter", False)
        upsample_factor = kwargs.get("upsample_factor", None)
        wvln = self.primary_wvln if wvln is None else wvln
        ks = max(int(self.sensor_res[0]), int(self.sensor_res[1])) if ks is None else ks
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float64)
        single_point = points.dim() == 1
        points = points.reshape(-1, 3)

        # 场平面采样（以高分辨率满足 Nyquist 条件）
        base_res = self.surfaces[0].res
        if upsample_factor is None:
            upsample_factor = max(1, round(4000 / self.surfaces[0].res[0]))

        field_res = [
            base_res[0] * upsample_factor,
            base_res[1] * upsample_factor,
        ]
        field_size = [
            self.surfaces[0].res[0] * self.surfaces[0].ps,
            self.surfaces[0].res[1] * self.surfaces[0].ps,
        ]
        sensor_w, sensor_h = self.sensor_size

        psfs = []
        for pt in points:
            x_norm, y_norm, depth = float(pt[0]), float(pt[1]), float(pt[2])

            # 为此光源（可能离轴）构建入射场。
            if math.isinf(depth):
                # 准直光源：倾斜平面波。将倾斜符号取反，使光源在物理上成像到倒置侧
                #（+x 处的物体聚焦到 -x），与下方有限深度点光源一致；后续翻转会
                # 消除该倒置。
                theta_x = math.atan(-x_norm * sensor_w / 2 / self.foclen)
                theta_y = math.atan(-y_norm * sensor_h / 2 / self.foclen)
                inp_wave = ComplexWave.plane_wave(
                    wvln=wvln,
                    z=0.0,
                    phy_size=field_size,
                    res=field_res,
                    theta_x=theta_x,
                    theta_y=theta_y,
                ).to(self.device)
            else:
                # 有限深度光源：从物点发出的球面波。
                scale = -depth / self.foclen  # 物高/像高
                obj_x = x_norm * scale * sensor_w / 2
                obj_y = y_norm * scale * sensor_h / 2
                inp_wave = ComplexWave.point_wave(
                    point=[obj_x, obj_y, depth],
                    phy_size=field_size,
                    res=field_res,
                    wvln=wvln,
                    z=0.0,
                ).to(self.device)

            # 传播到传感器并计算强度。shape [H, W]。
            output_wave = self.forward(inp_wave)
            intensity = output_wave.u.abs() ** 2

            # 重采样到传感器像素间距。
            factor = output_wave.ps / self.pixel_size
            intensity = F.interpolate(
                intensity,
                scale_factor=(factor, factor),
                mode="bilinear",
                align_corners=False,
            )[0, 0, :, :]

            # 居中裁剪/填充到传感器分辨率。``sensor_res`` 为 (W, H)，而强度张量
            # 按 [H, W] 索引；分别处理各维度，确保非方形传感器正常工作。
            target_h, target_w = int(self.sensor_res[1]), int(self.sensor_res[0])
            intensity_h, intensity_w = intensity.shape[-2:]
            pad_h = max(target_h - intensity_h, 0)
            pad_w = max(target_w - intensity_w, 0)
            if pad_h > 0 or pad_w > 0:
                intensity = F.pad(
                    intensity,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                    mode="constant",
                    value=0,
                )
            intensity_h, intensity_w = intensity.shape[-2:]
            start_h = (intensity_h - target_h) // 2
            start_w = (intensity_w - target_w) // 2
            intensity = intensity[
                start_h : start_h + target_h, start_w : start_w + target_w
            ]

            # 镜头在物理上形成倒像（+x 处的物体聚焦到 -x）。翻转两个轴，以传感器/
            # 光源符号约定（+x 光源 -> +x 传感器位置）报告 PSF，并使准直和有限深度
            # 路径保持一致。
            intensity = torch.flip(intensity, [0, 1])

            # 围绕 PSF 位置裁剪 ks x ks 图块。衍射镜头没有可追迹的主光线，因此当
            # ``recenter`` 为 True 时，裁剪中心是测得的 PSF 峰值（仿真传感器平面
            # 强度的 argmax）；否则裁剪中心是光源视场点的透视（针孔）像。
            if recenter:
                peak = torch.argmax(intensity)
                coord_c_i = int(peak // target_w)
                coord_c_j = int(peak % target_w)
            else:
                # 透视中心：(x_norm, y_norm) 的近轴像。
                # +x 映射到更大的列，+y 映射到更小的行，与未倒置的传感器平面强度一致。
                coord_c_j = int(round(target_w * (1.0 + x_norm) / 2.0))
                coord_c_i = int(round(target_h * (1.0 - y_norm) / 2.0))
            coord_c_i = min(max(coord_c_i, 0), target_h - 1)
            coord_c_j = min(max(coord_c_j, 0), target_w - 1)
            intensity = F.pad(
                intensity,
                [ks // 2, ks // 2, ks // 2, ks // 2],
                mode="constant",
                value=0,
            )
            psf = intensity[coord_c_i : coord_c_i + ks, coord_c_j : coord_c_j + ks]
            psf = psf / (psf.sum() + EPSILON)
            psfs.append(diff_float(psf))

        psf_out = torch.stack(psfs, dim=0)
        return psf_out[0] if single_point else psf_out

    # =============================================
    # 可视化
    # =============================================
    def draw_layout(self, save_name="./doelens.png"):
        """绘制衍射镜头的二维布局图。

        每个衍射面在其轴向位置 `z = surface.d` 处绘制为竖直虚线，传感器则在
        `z = d_sensor` 处绘制为实线矩形。

        参数:
            save_name (str, optional): 图像保存路径。默认为 './doelens.png'。
        """
        fig, ax = plt.subplots(figsize=(12, 4))

        default_l = float(max(self.sensor_size))

        # 将每个衍射面绘制为竖直虚线。
        for i, surf in enumerate(self.surfaces):
            d = float(surf.d)
            surf_l = float(getattr(surf, "w", default_l))
            ax.plot(
                [d, d], [-surf_l / 2, surf_l / 2], "orange", linestyle="--", dashes=[1, 1]
            )
            ax.text(
                d, surf_l / 2 * 1.08, f"{type(surf).__name__}\n(z={d:.1f} mm)",
                ha="center", va="bottom", fontsize=8,
            )

        # 将传感器平面绘制为窄矩形。
        d_sensor = float(self.d_sensor)
        sensor_l = float(self.sensor_size[1])
        width = max(0.01 * d_sensor, 0.2)
        rect = plt.Rectangle(
            (d_sensor - width / 2, -sensor_l / 2), width, sensor_l,
            facecolor="none", edgecolor="black", linewidth=1,
        )
        ax.add_patch(rect)
        ax.text(
            d_sensor, sensor_l / 2 * 1.08, f"Sensor\n(z={d_sensor:.1f} mm)",
            ha="center", va="bottom", fontsize=8,
        )

        # 光轴。
        ax.plot([0, d_sensor], [0, 0], "k-", linewidth=0.5, alpha=0.3)

        ax.set_xlabel("z [mm]")
        ax.set_yticks([])
        ax.margins(x=0.05, y=0.25)
        fig.savefig(save_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def draw_psf(
        self,
        depth=None,
        ks=None,
        save_name="./psf_doelens.png",
        log_scale=True,
        eps=1e-4,
    ):
        """绘制轴上 RGB PSF。

        计算并保存给定深度处 RGB PSF 的可视化结果。

        参数:
            depth (float, optional): 点光源深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            ks (int, optional): PSF 核尺寸 [pixels]。为 None（默认）时使用完整
                传感器分辨率（`max(self.sensor_res)`）。
            save_name (str, optional): PSF 图像保存路径。默认为 './psf_doelens.png'。
            log_scale (bool, optional): 为 True 时以对数尺度显示 PSF。默认为 True。
            eps (float, optional): 避免 log(0) 的对数尺度小量。默认为 1e-4。
        """
        depth = self.obj_depth if depth is None else depth
        psf_rgb = self.psf_rgb(points=[0.0, 0.0, depth], ks=ks)

        if log_scale:
            psf_rgb = torch.log10(psf_rgb + eps)
            psf_rgb = (psf_rgb - psf_rgb.min()) / (psf_rgb.max() - psf_rgb.min())
            save_name = save_name.replace(".png", "_log.png")

        save_image(psf_rgb.unsqueeze(0), save_name, normalize=True)

    # =============================================
    # 优化
    # =============================================
    def get_optimizer(self, lr, optim_surf_ls=None):
        """为可训练衍射面构建 Adam 优化器。

        参数:
            lr (float): 学习率。
            optim_surf_ls (list[int], optional): 要优化的表面索引。为 None 时
                优化所有衍射面。

        返回:
            optimizer (torch.optim.Optimizer): 用于所选表面相位参数的 Adam 优化器。
        """
        if optim_surf_ls is None:
            optim_surf_ls = list(range(len(self.surfaces)))

        params = []
        for idx in optim_surf_ls:
            params += self.surfaces[idx].get_optimizer_params(lr=lr)

        return torch.optim.Adam(params)
