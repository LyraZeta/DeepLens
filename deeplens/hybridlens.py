# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""混合折射-衍射镜头的光线-波动模型。混合镜头由 GeoLens 和位于其后的 DOE
组成。光学仿真采用可微光线-波动模型：先通过相干光线追迹计算 DOE 平面处的复波场，
再使用角谱法将波场传播到传感器平面。该混合镜头模型可模拟：(1) GeoLens 像差，
(2) DOE 相位调制。

技术论文:
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu, Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import json

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_COHERENT,
    WAVE_RGB,
)
from .geolens import GeoLens
from .lens import Lens
from .diffractive_surface import (
    Binary2,
    Fresnel,
    Grating,
    Pixel2D,
    Zernike,
)
from .geometric_surface import Plane
from .imgsim import forward_integral
from .phase_surface import Phase
from .utils import diff_float
from .light import AngularSpectrumMethod


class HybridLens(Lens):
    """使用可微光线-波动模型的混合折射-衍射镜头。

    将 `GeoLens`（折射模块）与置于其后的衍射光学元件（DOE）结合。流程为：
    (1) 通过嵌入的 `GeoLens` 进行相干光线追迹，获得 DOE 平面处包含全部几何像差
    的复波前；(2) 对波前施加 DOE 相位调制；(3) 使用角谱法（ASM）从 DOE 传播到
    传感器平面，生成最终强度 PSF。

    这使梯度能够从图像质量指标端到端回传到折射面参数和 DOE 相位轮廓。默认使用
    `torch.float64`，以保证波传播步骤的数值稳定性。

    属性:
        geolens (GeoLens): 嵌入的折射模块。DOE 平面以 `Plane` 占位表面的形式
            追加到其表面列表。
        doe (Binary2 or Pixel2D or Fresnel or Zernike or Grating): 位于折射组
            后方的衍射光学元件。
        foclen (float): 从嵌入的 `GeoLens` 复制的焦距 [mm]。

    参考:
        Xinge Yang et al., "End-to-End Hybrid Refractive-Diffractive Lens
        Design with Differentiable Ray-Wave Model," SIGGRAPH Asia 2024.
    """

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float64,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """初始化混合折射-衍射镜头。

        参数:
            filename (str, optional): 镜头配置 JSON 文件路径。默认为 None。
            device (str, optional): 计算设备（'cpu' 或 'cuda'）。默认为 None。
            dtype (torch.dtype, optional): 计算数据类型。默认为 `torch.float64`。
            primary_wvln (float, optional): 主要设计波长 [µm]。调用方法时未显式
                提供 `wvln`，则使用此值。默认为 `DEFAULT_WAVE`。
            wvln_rgb (list of float, optional): RGB 计算所用的三个波长 [µm]，
                按 [R, G, B] 排列。默认为 `WAVE_RGB`。
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
            self.geolens = None
            self.doe = None
            # 未提供文件时设置默认传感器尺寸和分辨率
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)
            print(
                f"No lens file provided. Using default sensor_size: {self.sensor_size} mm, "
                f"sensor_res: {self.sensor_res} pixels. Use set_sensor() to change."
            )

        self.double()

    def read_lens_json(self, filename):
        """从 JSON 文件读取镜头配置。

        从指定文件加载 `GeoLens` 及其 DOE。将 `Plane` 表面作为 DOE 平面的占位符
        追加到 GeoLens 表面列表，并与 DOE 孔径（方形或圆形）匹配。同时根据加载的
        GeoLens 设置 `self.foclen` 和传感器尺寸/分辨率。

        支持的 DOE 类型：binary2、pixel2d、fresnel、zernike、grating。

        参数:
            filename (str): JSON 配置文件路径。必须包含带 "type" 字段的 "DOE" 键。

        异常:
            ValueError: 文件中的 DOE 类型不受支持时抛出。
        """
        # 加载 geolens
        geolens = GeoLens(filename=filename, device=self.device)

        # 加载 DOE（衍射面）
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

            doe_dict = data["DOE"]
            doe_param_model = doe_dict["type"].lower()
            if doe_param_model == "binary2":
                doe = Binary2.init_from_dict(doe_dict)
            elif doe_param_model == "pixel2d":
                doe = Pixel2D.init_from_dict(doe_dict)
            elif doe_param_model == "fresnel":
                doe = Fresnel.init_from_dict(doe_dict)
            elif doe_param_model == "zernike":
                doe = Zernike.init_from_dict(doe_dict)
            elif doe_param_model == "grating":
                doe = Grating.init_from_dict(doe_dict)
            else:
                raise ValueError(f"Unsupported DOE parameter model: {doe_param_model}")
            self.doe = doe

        # 向 GeoLens 添加 Plane/Phase 表面（DOE 占位符）。
        # 匹配 DOE 的实际孔径（方形或圆形），使 DOE 区域外的光线在占位表面处
        # 被正确剔除。
        geolens.surfaces.append(
            Plane(d=doe.d.item(), r=doe.r, mat2="air", is_square=doe.is_square)
        )
        # r_doe = float(np.sqrt(doe.w**2 + doe.h**2) / 2)
        # geolens.surfaces.append(Phase(r=r_doe, d=doe.d))
        self.geolens = geolens
        self.foclen = geolens.foclen

        # 更新混合镜头的传感器分辨率和像素尺寸
        self.set_sensor(sensor_size=geolens.sensor_size, sensor_res=geolens.sensor_res)
        self.to(self.device)

    def write_lens_json(self, lens_path):
        """将镜头配置写入 JSON 文件。

        将 `GeoLens` 表面（不含 DOE 占位符）和 DOE 配置序列化到单个 JSON 文件，
        以便使用 `read_lens_json` 重新加载。

        参数:
            lens_path (str): 输出文件路径。
        """
        geolens = self.geolens
        data = {}
        data["info"] = geolens.lens_info if hasattr(geolens, "lens_info") else "None"
        data["foclen"] = round(geolens.foclen, 4)
        data["fnum"] = round(geolens.fnum, 4)
        data["r_sensor"] = round(geolens.r_sensor, 4)
        data["d_sensor"] = round(geolens.d_sensor.item(), 4)
        data["sensor_size"] = [round(i, 4) for i in geolens.sensor_size]
        data["sensor_res"] = geolens.sensor_res

        # 几何镜头
        data["surfaces"] = []
        for i, s in enumerate(geolens.surfaces[:-1]):
            surf_dict = s.surf_dict()

        # 排除最后一个表面（DOE）
            if i < len(geolens.surfaces) - 2:
                surf_dict["d_next"] = round(
                    geolens.surfaces[i + 1].d.item() - geolens.surfaces[i].d.item(), 3
                )
            else:
                surf_dict["d_next"] = round(
                    geolens.d_sensor.item() - geolens.surfaces[i].d.item(), 3
                )

            data["surfaces"].append(surf_dict)

        # 衍射光学元件（DOE）
        data["DOE"] = self.doe.surf_dict()

        with open(lens_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    # =====================================================================
    # 实用工具
    # =====================================================================
    def analysis(self, save_name="./test.png"):
        """对混合镜头执行快速可视化分析。

        生成两幅图：二维镜头布局（保存到 `save_name`）和 DOE 相位图（保存到
        `<save_name>_doe.png`）。

        参数:
            save_name (str, optional): 布局图的基础文件路径。DOE 相位图路径通过
                追加 `_doe.png` 形成。默认为 "./test.png"。
        """
        self.draw_layout(save_name=save_name)
        self.doe.draw_phase_map(save_name=f"{save_name}_doe.png")

    def double(self):
        """将 GeoLens 和 DOE 转换为 `float64` 精度。

        相干光线追迹和 ASM 传播期间的稳定相位累积需要双精度。由 `__init__`
        自动调用。
        """
        self.geolens.astype(torch.float64)
        self.doe.astype(torch.float64)

    def refocus(self, foc_dist):
        """将混合镜头重新对焦到给定物距。

        委托给 `GeoLens.refocus` 调整传感器距离；DOE 相对于折射组保持固定
        （物理上固定在镜筒上）。

        参数:
            foc_dist (float): 目标对焦距离 [mm]（负值，朝向物体）。
        """
        self.geolens.refocus(foc_dist)

    def calc_scale(self, depth):
        """计算物到像的放大缩放因子。

        委托给嵌入的 `GeoLens`。

        参数:
            depth (float): 物距 [mm]（负值，朝向物体）。

        返回:
            scale (float): 放大因子（物高/像高），按
                $-\\text{depth} / \\text{foclen}$ 计算。
        """
        return self.geolens.calc_scale(depth)

    # =====================================================================
    # PSF 相关函数
    # =====================================================================
    def doe_field(self, point, wvln=None, spp=SPP_COHERENT, upsample_factor=None):
        """通过相干光线追迹计算 DOE 平面处的复波场。

        与 `GeoLens.pupil_field` 类似，但在最后一个表面（DOE 平面）而非出瞳处
        计算场。返回的波前编码后续 DOE 调制和 ASM 传播所需的振幅、相位及全部
        衍射级次信息。

        参数:
            point (torch.Tensor): 点光源位置，shape [3] 或 [1, 3]，表示为
                [x, y, z]。x/y 为 [-1, 1] 范围内的归一化传感器坐标；z 为深度 [mm]。
            wvln (float, optional): 波长 [µm]。为 None（默认）时使用
                `self.primary_wvln`。
            spp (int, optional): 采样光线数。为获得准确的相干仿真，必须至少为
                1,000,000。默认为 `SPP_COHERENT`。
            upsample_factor (int or None, optional): 满足 Nyquist 采样约束的场上采样
                倍数。场在 `doe.res * upsample_factor` 网格上采样，间距为
                `doe.ps / upsample_factor`（物理孔径相同，采样更细）。为 None
                （默认）时，选择使场分辨率接近 4000 x 4000 的倍数。

        返回:
            wavefront (torch.Tensor): DOE 平面处的复波前，shape [H, W]，其中
                H = W = `doe.res[0] * upsample_factor`。
            psf_center (list of float): 传感器上估计的 PSF 中心，使用归一化坐标
                [x, y]。

        异常:
            AssertionError: `spp` 小于 1,000,000 或默认 dtype 不是 `float64` 时抛出。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        assert spp >= 1_000_000, (
            "Coherent ray tracing spp is too small, "
            "which may lead to inaccurate simulation."
        )
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase tracing."
        )

        geolens, doe = self.geolens, self.doe

        # 场平面上采样，以满足 ASM Nyquist 约束
        if upsample_factor is None:
            upsample_factor = max(1, round(4000 / doe.res[0]))

        if point.dim() == 1:
            point = point.unsqueeze(0)
        point = point.to(self.device)

        # 计算物方光线原点
        scale = geolens.calc_scale(point[:, 2].item())
        point_obj = point.clone()
        # sensor_size 为 (W, H)：x 按宽度 [0] 缩放，y 按高度 [1] 缩放。
        #（与下方主光线中心及基类 Lens / DiffractiveLens 一致。）
        point_obj[:, 0] = point[:, 0] * scale * geolens.sensor_size[0] / 2
        point_obj[:, 1] = point[:, 1] * scale * geolens.sensor_size[1] / 2

        # 通过主光线确定光线中心
        pointc_chief_ray = geolens.psf_center(point_obj, method="chief_ray")[
            0
        ]  # shape [2]

        # 追迹光线到 DOE 平面
        ray = geolens.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)
        ray.is_coherent = True
        ray, _ = geolens.trace(ray)
        ray = ray.prop_to(doe.d)

        # 计算用于出瞳衍射的全分辨率复场
        wavefront = forward_integral(
            ray.flip_xy(),
            ps=doe.ps / upsample_factor,
            ks=doe.res[0] * upsample_factor,
            pointc=torch.zeros_like(point[:, :2]),
        ).squeeze(0)  # shape [H, W]

        # 根据主光线计算 PSF 中心
        psf_center = [
            pointc_chief_ray[0] / geolens.sensor_size[0] * 2,
            pointc_chief_ray[1] / geolens.sensor_size[1] * 2,
        ]

        return wavefront, psf_center

    def psf(self, points=None, wvln=None, ks=PSF_KS, **kwargs):
        """使用光线-波动模型计算单点单色 PSF。

        返回的 PSF 包含具有物理正确衍射效率的全部衍射级次。流程为：(1) 通过
        `GeoLens` 进行相干光线追迹，获得 DOE 平面处的复波前；(2) 对波前施加
        DOE 相位调制；(3) 使用 ASM 传播到传感器，并计算强度、裁剪及归一化。

        参数:
            points (list or torch.Tensor, optional): [x, y, z] 点光源坐标。
                x、y 为 [-1, 1] 范围内的归一化传感器坐标；z 为深度 [mm]。
                为 None（默认）时使用 [0.0, 0.0, -10000.0]。
            wvln (float, optional): 波长 [µm]。为 None（默认）时使用
                `self.primary_wvln`。
            ks (int or None, optional): 输出 PSF 图块尺寸。为 None 时改为返回场的
                中间一半。默认为 `PSF_KS`。
            **kwargs: 模型特定选项。`spp` (int)：相干光线采样数，默认为
                `SPP_COHERENT`。`upsample_factor` (int)：满足 Nyquist 采样约束
                的场上采样倍数；为 None（默认）时，选择使场分辨率接近
                4000 x 4000 的倍数。

        返回:
            psf (torch.Tensor): 归一化 PSF 图块（总和为 1），shape [ks, ks]
                （`ks` 为 None 时每边约为场的一半）。以 `float32` 精度返回。

        异常:
            ValueError: 默认 dtype 不是 `float64` 时抛出（应先调用 `double`）。
        """
        if points is None:
            points = [0.0, 0.0, -10000.0]
        spp = kwargs.get("spp", SPP_COHERENT)
        upsample_factor = kwargs.get("upsample_factor", None)
        wvln = self.primary_wvln if wvln is None else wvln
        # 检查双精度
        if not torch.get_default_dtype() == torch.float64:
            raise ValueError(
                "Please call HybridLens.double() to set the default dtype to float64 for accurate phase tracing."
            )

        # 检查镜头最后一个表面
        assert isinstance(self.geolens.surfaces[-1], Phase) or isinstance(
            self.geolens.surfaces[-1], Plane
        ), "The last lens surface should be a DOE."
        geolens, doe = self.geolens, self.doe

        # 通过相干光线追迹计算瞳孔场
        if isinstance(points, list):
            point0 = torch.tensor(points)
        elif isinstance(points, torch.Tensor):
            point0 = points
        else:
            raise ValueError("point should be a list or a torch.Tensor.")

        # 场平面上采样，以满足 ASM Nyquist 约束
        if upsample_factor is None:
            upsample_factor = max(1, round(4000 / doe.res[0]))

        wavefront, psfc = self.doe_field(
            point=point0, wvln=wvln, spp=spp, upsample_factor=upsample_factor
        )
        wavefront = wavefront.squeeze(0)  # shape of [H, W]

        # DOE 相位调制。由于波前已经翻转，因此必须翻转相位图。将相位图上采样
        #（使用 nearest，以保留每个平坦 DOE 像素）到场分辨率。
        phase_map = torch.flip(doe.get_phase_map(wvln), [-1, -2])
        if phase_map.shape != wavefront.shape:
            phase_map = F.interpolate(
                phase_map[None, None], size=wavefront.shape, mode="nearest"
            )[0, 0]
        wavefront = wavefront * torch.exp(1j * phase_map)

        # 将波场传播到传感器平面
        h, w = wavefront.shape
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        sensor_field = AngularSpectrumMethod(
            wavefront,
            z=geolens.d_sensor - doe.d,
            wvln=wvln,
            ps=doe.ps / upsample_factor,
            padding=False,
        )

        # 计算 PSF（强度分布）
        psf_inten = sensor_field.abs() ** 2
        psf_inten = (
            F.interpolate(
                psf_inten,
                scale_factor=geolens.sensor_res[0] / h,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )

        # 计算 PSF 中心索引并裁剪有效 PSF 区域（同时考虑插值和填充）
        if ks is not None:
            h, w = psf_inten.shape[-2:]
            psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
            psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

        # 填充以避开无效边缘区域
            psf_inten_pad = F.pad(
                psf_inten,
                [ks // 2, ks // 2, ks // 2, ks // 2],
                mode="constant",
                value=0,
            )
            psf = psf_inten_pad[
                psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
            ]
        else:
            h, w = psf_inten.shape[-2:]
            psf = psf_inten[
                int(h / 2 - h / 4) : int(h / 2 + h / 4),
                int(w / 2 - w / 4) : int(w / 2 + w / 4),
            ]

        # 归一化并转换为 float 精度。
        psf = psf / (psf.sum() + EPSILON)  # shape of [ks, ks] or [h, w]
        return diff_float(psf)

    # =====================================================================
    # 可视化
    # =====================================================================
    @torch.no_grad()
    def draw_layout(
        self,
        save_name="./DOELens.png",
        depth=-10000.0,
        ax=None,
        fig=None,
        dpi=600,
    ):
        """绘制包含光线路径和波传播圆弧的混合镜头布局。

        通过 `GeoLens.draw_lens_2d` 渲染折射元件，在三个视场角（轴上、0.707x、
        0.99x 全视场）处追迹光线，并在 DOE 与传感器之间叠加同心圆弧以表示波传播区域。

        参数:
            save_name (str, optional): 图像保存路径（仅在 `ax` 为 None 时使用）。
                默认为 "./DOELens.png"。
            depth (float, optional): 所追迹光线的物体深度 [mm]。默认为 -10000.0。
            ax (matplotlib.axes.Axes, optional): 用于绘图的现有坐标轴。为 None 时
                创建并保存新图。
            fig (matplotlib.figure.Figure, optional): 现有图形。提供 `ax` 时必需。
            dpi (int, optional): 保存新图时使用的分辨率。默认为 600。

        返回:
            ax (matplotlib.axes.Axes): 坐标轴，仅在提供 `ax` 时返回。`ax` 为 None
                时将图保存到 `save_name`，不返回任何内容。
            fig (matplotlib.figure.Figure): 图形，仅在提供 `ax` 时返回。
        """
        geolens = self.geolens

        # 绘制镜头布局
        if ax is None:
            ax, fig = geolens.draw_lens_2d()
            save_fig = True
        else:
            save_fig = False

        # 将 DOE 绘制为橙色 Fresnel 风格组件
        self.doe.draw_widget(ax, color="orange")

        # 绘制光路
        color_list = ["#CC0000", "#006600", "#0066CC"]
        views = [
            0.0,
            float(np.rad2deg(geolens.rfov) * 0.707),
            float(np.rad2deg(geolens.rfov) * 0.99),
        ]
        arc_radi_list = [0.1, 0.4, 0.7, 1.0, 1.4, 1.8]
        num_rays = 11
        arc_half_angle = 20
        for i, view in enumerate(views):
            # 绘制光线追迹
            ray = geolens.sample_point_source_2D(
                depth=depth,
                fov=view,
                num_rays=num_rays,
                entrance_pupil=True,
                wvln=self.wvln_rgb[2 - i],
            )
            ray.prop_to(-1.0)

            ray, ray_o_record = geolens.trace(ray=ray, record=True)
            ax, fig = geolens.draw_ray_2d(
                ray_o_record, ax=ax, fig=fig, color=color_list[i]
            )

            # 绘制波传播
            # 计算用于波传播可视化的光线中心
            ray_center_doe = (
                ((ray.o * ray.is_valid.unsqueeze(-1)).sum(dim=0) / ray.is_valid.sum())
                .cpu()
                .numpy()
            )  # shape [3]
            ray.prop_to(geolens.d_sensor)  # shape [num_rays, 3]
            ray_center_sensor = (
                ((ray.o * ray.is_valid.unsqueeze(-1)).sum(dim=0) / ray.is_valid.sum())
                .cpu()
                .numpy()
            )  # shape [3]

            arc_radi = ray_center_sensor[2] - ray_center_doe[2]
            chief_theta = np.rad2deg(
                np.arctan2(
                    ray_center_sensor[0] - ray_center_doe[0],
                    ray_center_sensor[2] - ray_center_doe[2],
                )
            )
            theta1 = chief_theta - arc_half_angle
            theta2 = chief_theta + arc_half_angle

            for j in arc_radi_list:
                arc_radi_j = arc_radi * j
                arc = patches.Arc(
                    (ray_center_sensor[2], ray_center_sensor[0]),
                    arc_radi_j,
                    arc_radi_j,
                    angle=180.0,
                    theta1=theta1,
                    theta2=theta2,
                    color=color_list[i],
                )
                ax.add_patch(arc)

        if save_fig:
        # 保存图形
            ax.axis("off")
            ax.set_title("DOE Lens")
            fig.savefig(save_name, bbox_inches="tight", dpi=dpi)
            plt.close()
        else:
            return ax, fig

    # =====================================================================
    # 优化
    # =====================================================================
    def get_optimizer(
        self, doe_lr=1e-4, lens_lr=[1e-4, 1e-4, 1e-2, 1e-5]
    ):
        """为镜头 + DOE 联合设计构建 Adam 优化器。

        将 `GeoLens`（表面厚度、曲率、圆锥常数、非球面系数）和 DOE 相位轮廓的
        可训练参数收集到一个优化器中，并为各参数组设置学习率。

        参数:
            doe_lr (float, optional): DOE 相位参数的学习率。默认为 1e-4。
            lens_lr (list of float, optional): GeoLens 各参数组的学习率，顺序为
                [thickness_d, curvature_c, conic_k, aspheric_a]。默认为
                [1e-4, 1e-4, 1e-2, 1e-5]。

        返回:
            optimizer (torch.optim.Adam): 针对所有可训练参数配置的优化器。
        """
        params = []
        params += self.geolens.get_optimizer_params(lrs=lens_lr)
        params += self.doe.get_optimizer_params(lr=doe_lr)

        optimizer = torch.optim.Adam(params)
        return optimizer
