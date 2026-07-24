# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""几何透镜系统的经典光学性能评估。

本模块提供 ``GeoLensEval`` 混入类，为 ``GeoLens`` 添加与 Zemax 对应的光学评估能力。
所有指标均通过几何光线追迹计算：从物方采样光线，使其经过全部透镜表面（折射与裁剪），
最后在传感器平面上分析结果。

坐标约定（与 DeepLens 其余部分一致）：
    - **z 轴**：光轴，光沿 +z 方向传播。
    - **y 轴**：子午（切向）平面。
    - **x 轴**：弧矢平面。
    - 传感器平面位于 ``z = self.d_sensor``。

从父级 ``GeoLens`` 实例使用的主要依赖：
    - ``self.sample_radial_rays()``、``self.sample_grid_rays()``：光线采样。
    - ``self.trace(ray)``、``self.trace2sensor(ray)``：顺序光线追迹。
    - ``self.psf()``、``self.psf_rgb()``：点扩散函数计算。
    - ``self.render()``：通过光线追迹或 PSF 卷积进行像面渲染。
    - ``self.d_sensor``、``self.sensor_size``、``self.pixel_size``、``self.rfov``、
      ``self.foclen``、``self.device``：透镜几何属性。

主要功能：
    点列图：``spot_points``、``draw_spot_radial``、``draw_spot_map``。
    RMS 点列误差：``rms_map_rgb``、``rms_map``。
    畸变：``calc_distortion_radial``、``draw_distortion_radial``、
        ``calc_distortion_map``、``calc_inv_distortion_map``、
        ``draw_distortion_map``、``distortion_center``。
    MTF（调制传递函数）：``mtf``、``psf2mtf``、``draw_mtf``。
    渐晕：``vignetting``、``draw_vignetting``。
    主光线与光线瞄准：``calc_chief_ray_infinite``。
    综合分析：``analysis_spot``、``analysis_rendering``、``analysis``。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from ..config import (
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
    SPP_RENDER,
)
from ..light import Ray

# 用于波长可视化的 RGB 颜色定义
RGB_RED = "#CC0000"
RGB_GREEN = "#006600"
RGB_BLUE = "#0066CC"
RGB_COLORS = [RGB_RED, RGB_GREEN, RGB_BLUE]
RGB_LABELS = ["R", "G", "B"]


class GeoLensEval:
    """为 ``GeoLens`` 添加经典光学评估方法的混入类。

    本类**不会单独实例化**，而是通过多重继承混入 ``GeoLens``，因此各方法可通过
    ``self`` 直接访问透镜几何属性和光线追迹方法。

    所有评估函数均遵循相同流程：
        1. 从物方采样光线（平行、网格或径向）。
        2. 通过 ``self.trace`` 或 ``self.trace2sensor`` 追迹光线。
        3. 在传感器平面分析光线位置或方向。
        4. 可选生成并保存 matplotlib 图像。

    在相同透镜参数与光线采样密度下，结果精度与 Zemax OpticStudio 对齐。

    通过 ``self`` 使用的 ``GeoLens`` 属性：
        d_sensor (float): 传感器平面的轴向位置，单位 mm。
        sensor_size (tuple[float, float]): 传感器尺寸 ``(width, height)``，单位 mm。
        pixel_size (float): 像素间距，单位 mm。
        sensor_res (tuple[int, int]): 传感器分辨率 ``(W, H)``，单位 pixel。
        rfov (float): 半视场角，单位 rad。
        foclen (float): 等效焦距，单位 mm。
        fnum (float): F 数。
        aper_idx (int): 孔径光阑表面的索引。
        device (torch.device): 计算设备（CPU / CUDA）。
    """

    # ================================================================
    # 点列图
    # ================================================================
    @torch.no_grad()
    def spot_points(self, points, num_rays=SPP_PSF, wvln=None):
        """从物点向传感器追迹光线并返回追迹后的 ``Ray``。

        从每个物理物点朝入瞳采样光线，经过全部透镜表面（折射与裁剪）后，返回位于
        传感器平面的 ``Ray``。这是点列图和 RMS 误差图共用的计算核心。

        算法：
            1. ``self.sample_from_points(points, num_rays, wvln)`` 为每个物点
               生成朝向入瞳的 ``num_rays`` 条光线。
            2. ``self.trace2sensor()`` 使光线经过所有表面并裁剪渐晕光线。

        参数：
            points (torch.Tensor): 物方三维物理坐标，shape 为 ``[..., 3]``，单位 mm。
                支持 ``[3]``、``[N, 3]`` 和 ``[H, W, 3]``。
            num_rays (int): 每个物点的采样光线数，默认为 ``SPP_PSF``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。

        返回：
            ray (Ray): 传感器平面上的追迹光线。位置 shape 为
                ``[..., num_rays, 3]``，有效性掩码 shape 为 ``[..., num_rays]``。
                横向位置使用 ``ray.o[..., :2]``，有效性掩码使用 ``ray.is_valid``，
                加权质心由 ``ray.centroid()`` 给出。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        ray = self.sample_from_points(points=points, num_rays=num_rays, wvln=wvln)
        return self.trace2sensor(ray)

    @torch.no_grad()
    def draw_spot_radial(
        self,
        save_name="./lens_spot_radial.png",
        num_fov=5,
        depth=None,
        num_rays=SPP_PSF,
        wvln_list=None,
        direction="y",
        show=False,
    ):
        """沿指定方向在等间隔视场角处绘制点列图。

        点列图展示给定视场角和深度的点光源在传感器平面上的横向光线交点分布，可反映
        球差、彗差、像散、场曲和色差等像差的综合影响。视场位置按视场角从轴上位置
        0 均匀采样至全视场 ``self.rfov``，因此 ``FoV 1.0`` 对应完整像高。

        对 ``wvln_list`` 中的每个波长，沿指定方向采样 ``num_fov`` 个视场角，追迹至
        传感器，并在对应子图中绘制有效光线的 ``(x, y)``；所有波长以 RGB 颜色叠加。

        参数：
            save_name (str): 输出 PNG 的路径，默认为 ``'./lens_spot_radial.png'``。
            num_fov (int): 从轴上到全视场的采样位置数，默认为 5。
            depth (float): 物距，单位 mm（负值表示实物）；为 ``None`` 或
                ``float('inf')`` 时使用 ``self.obj_depth``。
            num_rays (int): 每个视场、每个波长的光线数，默认为 ``SPP_PSF``。
            wvln_list (list[float]): 波长列表，单位 µm；为 ``None`` 时使用
                ``self.wvln_rgb``。
            direction (str): 采样方向；``"y"`` 为子午方向（默认），``"x"`` 为
                弧矢方向，``"diagonal"`` 为 45° 对角方向。
            show (bool): 为 ``True`` 时交互显示，否则保存到磁盘；默认为 ``False``。
        """
        wvln_list = self.wvln_rgb if wvln_list is None else wvln_list
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        if depth is None or depth == float("inf"):
            depth = self.obj_depth

        # 子图标题使用的视场比例（0 表示轴上，1 表示全视场）
        fov_fracs = torch.linspace(0, 1, num_fov)

        # 准备图像
        fig, axs = plt.subplots(1, num_fov, figsize=(num_fov * 3.5, 3))
        axs = np.atleast_1d(axs)

        # 分别追迹并绘制各波长，再叠加结果
        for wvln_idx, wvln in enumerate(wvln_list):
            # 按视场角（0 .. self.rfov）采样，使 FoV 1.0 达到完整像高并与
            # analysis_spot() 保持一致。
            ray = self.sample_radial_rays(
                num_field=num_fov,
                depth=depth,
                num_rays=num_rays,
                wvln=wvln,
                direction=direction,
            )
            ray = self.trace2sensor(ray)
            ray_o = ray.o[..., :2].cpu().numpy()
            ray_valid_np = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # 在一幅图中绘制多个点列图
            for i in range(num_fov):
                valid = ray_valid_np[i, :]
                xi, yi = ray_o[i, :, 0], ray_o[i, :, 1]

                # 筛选有效光线
                mask = valid > 0
                x_valid, y_valid = xi[mask], yi[mask]

                # 绘制当前波长的光线点和质心
                axs[i].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                axs[i].set_aspect("equal", adjustable="datalim")
                axs[i].tick_params(axis="both", which="major", labelsize=6)
                if wvln_idx == 0:
                    axs[i].set_title(f"FoV {fov_fracs[i].item():.2f}", fontsize=8)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def draw_spot_map(
        self,
        save_name="./lens_spot_map.png",
        num_grid=5,
        depth=None,
        num_rays=SPP_PSF,
        wvln_list=None,
        show=False,
    ):
        """在全视场范围内绘制二维点列图网格。

        与只采样径向切片的 ``draw_spot_radial`` 不同，本方法在 x（弧矢）和 y（子午）
        两个方向采样 ``num_grid × num_grid`` 视场位置，以显示一维径向扫描无法观察到的
        轴外像差。网格按视场角覆盖两轴完整视场，角点达到完整像高。

        对每个波长，``self.sample_grid_rays()`` 在
        ``[-vfov/2, vfov/2] × [-hfov/2, hfov/2]`` 上采样视场角网格，
        ``self.trace2sensor()`` 将其追迹至传感器，再将有效 ``(x, y)`` 位置绘制到
        对应子图；各波长使用 RGB 颜色叠加。

        参数：
            save_name (str): 输出 PNG 的路径，默认为 ``'./lens_spot_map.png'``。
            num_grid (int | tuple[int, int]): 各轴网格点数；子图总数为
                ``grid_w * grid_h``，默认为 5。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            num_rays (int): 每个网格单元、每个波长的光线数，默认为 ``SPP_PSF``。
            wvln_list (list[float]): 波长列表，单位 µm；为 ``None`` 时使用
                ``self.wvln_rgb``。
            show (bool): 为 ``True`` 时交互显示，默认为 ``False``。
        """
        wvln_list = self.wvln_rgb if wvln_list is None else wvln_list
        depth = self.obj_depth if depth is None else depth
        assert isinstance(wvln_list, list), "wvln_list must be a list"
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        grid_w, grid_h = num_grid
        fig, axs = plt.subplots(
            grid_h, grid_w, figsize=(grid_w * 3, grid_h * 3)
        )
        axs = np.atleast_2d(axs)

        # 遍历各波长并叠加散点
        for wvln_idx, wvln in enumerate(wvln_list):
            # 采样覆盖完整视场的视场角网格，使角点达到完整像高；
            # shape 为 [grid_h, grid_w, num_rays, 3]。
            ray = self.sample_grid_rays(
                depth=depth, num_grid=num_grid, num_rays=num_rays, wvln=wvln
            )
            ray = self.trace2sensor(ray)

            # 转为 numpy，shape 为 [grid_h, grid_w, num_rays, 2]
            ray_o = ray.o[..., :2].cpu().numpy()
            ray_valid_np = ray.is_valid.cpu().numpy()

            color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]

            # 按网格单元绘制
            for i in range(grid_h):
                for j in range(grid_w):
                    valid = ray_valid_np[i, j, :]
                    xi, yi = ray_o[i, j, :, 0], ray_o[i, j, :, 1]

                    # 筛选有效光线
                    mask = valid > 0
                    x_valid, y_valid = xi[mask], yi[mask]

                    # 绘制当前波长的光线点
                    axs[i, j].scatter(x_valid, y_valid, 2, color=color, alpha=0.5)
                    axs[i, j].set_aspect("equal", adjustable="datalim")
                    axs[i, j].tick_params(axis="both", which="major", labelsize=6)

        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # RMS 图
    # ================================================================
    @torch.no_grad()
    def rms_map(self, num_grid=32, depth=None, wvln=None, center=None):
        """计算单一波长下各视场位置的 RMS 点列半径。

        每个网格单元追迹 ``SPP_PSF`` 条光线，并计算有效光线交点到参考质心的均方根
        距离。当 ``center`` 为 ``None`` 时，每个单元使用自身质心；提供外部 ``center``
        （例如绿光通道质心）时，RMS 会包含相对该参考点的色移。

        计算公式为 ``RMS = sqrt(mean(||ray_xy - c||^2))``。

        参数：
            num_grid (int | tuple[int, int]): 视场采样网格分辨率，默认为 32。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。
            center (torch.Tensor | None): 外部参考质心，shape 为
                ``[grid_h, grid_w, 2]``；为 ``None`` 时使用各单元自身质心。

        返回：
            rms (torch.Tensor): RMS 点列误差图，shape 为 ``[grid_h, grid_w]``，单位 mm。
            centroid (torch.Tensor): 作为参考的各单元质心，shape 为
                ``[grid_h, grid_w, 2]``，可作为后续调用的 ``center``。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # 生成物理网格点并将光线追迹至传感器
        points = self.point_source_grid(depth=depth, grid=num_grid, normalized=False)
        ray = self.spot_points(points, num_rays=SPP_PSF, wvln=wvln)

        # 复用 Ray.centroid()：shape 为 [grid_h, grid_w, 3]，再切片为 [grid_h, grid_w, 2]
        centroid = ray.centroid()[..., :2]

        # 若提供外部中心则使用它，否则使用自身质心
        ref = center if center is not None else centroid

        # 相对于参考点的 RMS，shape 为 [grid_h, grid_w]
        ray_xy = ray.o[..., :2]
        ray_valid = ray.is_valid
        rms = torch.sqrt(
            (((ray_xy - ref.unsqueeze(-2)) ** 2).sum(-1) * ray_valid).sum(-1)
            / (ray_valid.sum(-1) + EPSILON)
        )

        return rms, centroid

    @torch.no_grad()
    def rms_map_rgb(self, num_grid=32, depth=None):
        """计算 R、G、B 三个波长在各视场位置的 RMS 点列半径。

        对 ``num_grid × num_grid`` 网格中的每个位置，本方法按波长追迹 ``SPP_PSF``
        条光线，并计算有效交点到**共同**参考质心的均方根距离。参考点采用绿光通道质心，
        因而结果包含 R/G/B 质心偏移所产生的横向色差，可作为复色像质指标。

        先调用 ``rms_map(wvln=green)`` 得到绿光 RMS 和质心，再以该质心计算红光与蓝光
        RMS，最后按 ``[R, G, B]`` 堆叠。

        参数：
            num_grid (int | tuple[int, int]): 视场采样网格分辨率，默认为 32。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。

        返回：
            rms_rgb (torch.Tensor): RMS 点列误差图，shape 为
                ``[3, grid_h, grid_w]``，通道顺序为 R、G、B，单位 mm。
        """
        depth = self.obj_depth if depth is None else depth
        # 先计算绿光，以获得共同参考质心
        rms_g, green_centroid = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=self.wvln_rgb[1]
        )

        # 相对于绿光质心计算红光和蓝光
        rms_r, _ = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=self.wvln_rgb[0], center=green_centroid
        )
        rms_b, _ = self.rms_map(
            num_grid=num_grid, depth=depth, wvln=self.wvln_rgb[2], center=green_centroid
        )

        return torch.stack([rms_r, rms_g, rms_b], dim=0)

    # ================================================================
    # 畸变
    # ================================================================
    @torch.no_grad()
    def calc_distortion_radial(
        self,
        num_points=GEO_GRID,
        wvln=None,
        plane="meridional",
        ray_aiming=True,
    ):
        """计算子午方向上等间隔视场角处的相对畸变。

        畸变定义为 ``(h_actual - h_ideal) / h_ideal``，其中
        ``h_ideal = f * tan(theta)`` 为直线投影的理想像高，``h_actual`` 为主光线
        在传感器上的实际像高。正值表示枕形畸变，负值表示桶形畸变。

        本方法从 0 到 ``self.rfov`` 均匀采样 ``num_points`` 个视场角。轴上样本使用
        一个极小正角度以避免 0/0；随后追迹主光线，按弧矢平面的 x 坐标或子午平面的
        y 坐标取得实际像高，并返回相对畸变。

        参数：
            num_points (int): 从轴上到全视场的等间隔样本数，默认为 ``GEO_GRID``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。
            plane (str): ``'meridional'``（y 轴）或 ``'sagittal'``（x 轴），
                默认为 ``'meridional'``。
            ray_aiming (bool): 为 ``True`` 时瞄准主光线使其通过孔径光阑中心，
                对广角透镜更准确；默认为 ``True``。

        返回：
            rfov_samples (np.ndarray): 视场角数组，单位 degree，shape 为
                ``[num_points]``。
            distortions (np.ndarray): 各视场角的无量纲相对畸变，shape 为
                ``[num_points]``；乘以 100 可得到百分比。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        rfov_deg = self.rfov * 180 / torch.pi

        # 从 0 到 rfov_deg 均匀采样视场角。
        # 轴上点（FOV=0）的畸变为 0/0，因此使用极小正角度计算正确极限；
        # 当传感器不在近轴焦点时，该极限可能不为零。
        rfov_samples = torch.linspace(0, rfov_deg, num_points)
        rfov_compute = rfov_samples.clone()
        if rfov_compute[0] == 0:
            # 在单样本情形（num_points == 1）下避免访问 rfov_samples[1]。
            tiny = (
                rfov_samples[1].item() * 0.01
                if len(rfov_samples) > 1
                else min(0.01, float(rfov_deg) * 0.01)
            )
            rfov_compute[0] = min(0.01, tiny)

        # 理想像高：h_ideal = f * tan(theta)
        eff_foclen = float(self.foclen)
        ideal_imgh = eff_foclen * np.tan(rfov_compute.numpy() * np.pi / 180)

        # 将主光线追迹至传感器平面
        chief_ray_o, chief_ray_d = self.calc_chief_ray_infinite(
            rfov=rfov_compute, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        )
        ray = Ray(chief_ray_o, chief_ray_d, wvln=wvln, device=self.device)
        ray, _ = self.trace(ray)
        t = (self.d_sensor - ray.o[..., 2]) / ray.d[..., 2]

        # 从对应横向坐标取得实际像高
        if plane == "sagittal":
            actual_imgh = (ray.o[..., 0] + ray.d[..., 0] * t).abs()
        elif plane == "meridional":
            actual_imgh = (ray.o[..., 1] + ray.d[..., 1] * t).abs()
        else:
            raise ValueError(f"Invalid plane: {plane}")

        actual_imgh = actual_imgh.cpu().numpy()

        # 计算相对畸变，并安全处理轴上奇点
        ideal_imgh = np.asarray(ideal_imgh)
        mask = np.abs(ideal_imgh) < EPSILON
        distortions = np.where(
            mask, 0.0, (actual_imgh - ideal_imgh) / np.where(mask, 1.0, ideal_imgh)
        )

        return rfov_samples.numpy(), distortions

    @torch.no_grad()
    def draw_distortion_radial(
        self,
        save_name=None,
        num_points=GEO_GRID,
        wvln=None,
        plane="meridional",
        ray_aiming=True,
        show=False,
    ):
        """以 Zemax 风格绘制畸变随视场角变化的曲线。

        y 轴为视场角，x 轴为百分比畸变，与 Zemax OpticStudio 的布局约定一致，
        可用于快速判断桶形或枕形畸变。

        参数：
            save_name (str | None): 输出 PNG 的路径；为 ``None`` 时自动生成
                ``'./{plane}_distortion_inf.png'``。
            num_points (int): 视场角样本数，默认为 ``GEO_GRID``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。
            plane (str): ``'meridional'`` 或 ``'sagittal'``，默认为
                ``'meridional'``。
            ray_aiming (bool): 是否在计算主光线时使用光线瞄准，默认为 ``True``。
            show (bool): 为 ``True`` 时交互显示，默认为 ``False``。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        rfov_deg = self.rfov * 180 / torch.pi

        # 计算等间隔视场角处的畸变
        rfov_samples, distortions = self.calc_distortion_radial(
            num_points=num_points, wvln=wvln, plane=plane, ray_aiming=ray_aiming
        )

        # 转为百分比并处理 NaN
        values = np.nan_to_num(distortions * 100, nan=0.0).tolist()

        # 创建图像
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title(f"{plane} Surface Distortion")

        # 绘制畸变曲线
        ax.plot(values, rfov_samples, linestyle="-", color="g", linewidth=1.5)

        # 绘制参考线（竖线）
        ax.axvline(x=0, color="k", linestyle="-", linewidth=0.8)

        # 设置网格
        ax.grid(True, color="gray", linestyle="-", linewidth=0.5, alpha=1)

        # 动态调整 x 轴范围
        value = max(abs(v) for v in values)
        margin = value * 0.2  # 保留 20% 边距
        x_min, x_max = -max(0.2, value + margin), max(0.2, value + margin)

        # 设置刻度
        x_ticks = np.linspace(-value, value, 3)
        y_ticks = np.linspace(0, rfov_deg, 3)

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # 格式化刻度标签
        x_labels = [f"{x:.1f}%" for x in x_ticks]
        y_labels = [f"{y:.1f}" for y in y_ticks]

        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)

        # 设置坐标轴标签
        ax.set_xlabel("Distortion (%)")
        ax.set_ylabel("Field of View (degrees)")

        # 设置坐标轴范围
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, rfov_deg)

        if show:
            plt.show()
        else:
            if save_name is None:
                save_name = f"./{plane}_distortion_inf.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def calc_distortion_map(self, num_grid=16, depth=None, wvln=None):
        """计算从理想像位置映射到实际像位置的二维畸变网格。

        对 ``num_grid × num_grid`` 视场网格中的每个单元，将光线追迹至传感器并计算
        质心，再将质心归一化到 ``[-1, 1]`` 传感器坐标。该映射可与
        ``torch.nn.functional.grid_sample`` 配合，用于图像扭曲或反扭曲。

        参数：
            num_grid (int): 每个轴的网格分辨率，默认为 16。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。

        返回：
            distortion_grid (torch.Tensor): 畸变网格，shape 为
                ``[grid_h, grid_w, 2]``。各 ``(x, y)`` 位于归一化传感器坐标
                ``[-1, 1]`` 中，表示对应理想网格位置的实际质心位置。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # 采样并追迹光线，shape 为 (grid_size, grid_size, num_rays, 3)
        ray = self.sample_grid_rays(depth=depth, num_grid=num_grid, wvln=wvln, uniform_fov=False)
        ray = self.trace2sensor(ray)

        # 计算光线质心，shape 为 (grid_size, grid_size, 2)。
        # 各轴分别除以自身半尺寸，使非方形传感器也能正确映射到 [-1, 1]：
        # x 使用 sensor_size[0]（width, W），y 使用 sensor_size[1]（height, H）。
        # 两个轴均翻转符号以消除成像倒置，并与 ``distortion_center`` 保持一致。
        sensor_w, sensor_h = self.sensor_size
        ray_xy = -ray.centroid()[..., :2]
        x_dist = ray_xy[..., 0] / (sensor_w / 2)
        y_dist = ray_xy[..., 1] / (sensor_h / 2)
        distortion_grid = torch.stack((x_dist, y_dist), dim=-1)
        return distortion_grid

    @torch.no_grad()
    def calc_inv_distortion_map(self, num_grid=16, depth=None, wvln=None):
        """计算供 ``grid_sample`` 应用透镜畸变的网格。

        对畸变传感器网格上的每个点，将光线反向追迹到目标物距平面，并把物方交点转换为
        归一化理想像坐标。将该网格传给 ``torch.nn.functional.grid_sample``，可从无畸变
        图像采样得到畸变图像。

        参数：
            num_grid (int | tuple): 网格分辨率；元组按 ``(grid_w, grid_h)`` 解释。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。

        返回：
            inv_distortion_grid (torch.Tensor): ``grid_sample`` 坐标中的逆畸变网格，
                shape 为 ``[grid_h, grid_w, 2]``。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        grid_w, grid_h = num_grid
        sensor_w, sensor_h = self.sensor_size
        device = self.device

        # 将 grid_sample 输出坐标转换为传感器物理位置。
        # 现有畸变图使用 -sensor_centroid 作为图像坐标。
        x, y = torch.meshgrid(
            torch.linspace(sensor_w / 2, -sensor_w / 2, grid_w, device=device),
            torch.linspace(sensor_h / 2, -sensor_h / 2, grid_h, device=device),
            indexing="xy",
        )
        z = torch.full_like(x, self.d_sensor.item())

        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(grid_h, grid_w, SPP_CALC))
        ray_o = torch.stack((x, y, z), dim=-1).unsqueeze(2).repeat(1, 1, SPP_CALC, 1)
        ray = Ray(ray_o, ray_o2 - ray_o, wvln, device=device)

        ray = self.trace2obj(ray)
        ray = ray.prop_to(depth)
        point_obj = ray.centroid()[..., :2]

        scale = self.calc_scale(depth)
        x_ideal = point_obj[..., 0] / (scale * sensor_w / 2)
        y_ideal = point_obj[..., 1] / (scale * sensor_h / 2)
        inv_distortion_grid = torch.stack((x_ideal, y_ideal), dim=-1)
        return inv_distortion_grid

    def distortion_center(self, points):
        """计算任意归一化物点对应的畸变像质心。

        将归一化物点转换为物方物理位置，从每个点发射光线并追迹通过透镜，最后返回
        传感器上归一化到 ``[-1, 1]`` 的光线质心。这是畸变校正所需的逆映射。

        参数：
            points (torch.Tensor): 归一化点光源位置，shape 为 ``[N, 3]`` 或
                ``[..., 3]``。``x, y`` ∈ [-1, 1] 表示视场位置，
                ``z`` ∈ (-∞, 0] 为物距，单位 mm。

        返回：
            distortion_center (torch.Tensor): 归一化畸变质心，shape 为
                ``[N, 2]`` 或 ``[..., 2]``，其中 ``x, y`` ∈ [-1, 1]。
        """
        sensor_w, sensor_h = self.sensor_size

        # 将归一化点转换为物方坐标
        depth = points[..., 2]
        scale = self.calc_scale(depth)
        points_obj_x = points[..., 0] * scale * sensor_w / 2
        points_obj_y = points[..., 1] * scale * sensor_h / 2
        points_obj = torch.stack([points_obj_x, points_obj_y, depth], dim=-1)

        # 采样光线并追迹至传感器
        ray = self.sample_from_points(points=points_obj)
        ray = self.trace2sensor(ray)

        # 计算质心并归一化到 [-1, 1]
        ray_center = -ray.centroid()  # shape 为 [..., 3]
        distortion_center_x = ray_center[..., 0] / (sensor_w / 2)
        distortion_center_y = ray_center[..., 1] / (sensor_h / 2)
        distortion_center = torch.stack((distortion_center_x, distortion_center_y), dim=-1)
        return distortion_center

    @torch.no_grad()
    def draw_distortion_map(
        self, save_name=None, num_grid=16, depth=None, wvln=None, show=False
    ):
        """绘制畸变网格散点图。

        在归一化传感器坐标 ``[-1, 1]`` 上显示 ``calc_distortion_map()`` 的输出。
        无畸变透镜应形成规则直线网格，偏离情况可显示桶形或枕形畸变。

        参数：
            save_name (str | None): 输出 PNG 的路径；为 ``None`` 时自动生成
                ``'./distortion_{depth}.png'``。
            num_grid (int): 每个轴的网格分辨率，默认为 16。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。
            show (bool): 为 ``True`` 时交互显示，默认为 ``False``。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # 通过光线追迹计算畸变图
        distortion_grid = self.calc_distortion_map(num_grid=num_grid, depth=depth, wvln=wvln)
        # 缩放坐标轴以保持传感器的物理宽高比：长边映射到 ±1，
        # 短边映射到 ±(shorter/longer)。
        sensor_w, sensor_h = self.sensor_size
        max_half = max(sensor_w, sensor_h) / 2
        aspect_x = (sensor_w / 2) / max_half
        aspect_y = (sensor_h / 2) / max_half
        x1 = distortion_grid[..., 0].cpu().numpy() * aspect_x
        y1 = distortion_grid[..., 1].cpu().numpy() * aspect_y

        # 绘制图像
        fig, ax = plt.subplots()
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.scatter(x1, y1, s=20, zorder=3)
        ax.axis("scaled")

        # 按 grid_size 绘制网格线，并逐轴缩放，使叠加网格匹配数据范围
        # （±aspect_x × ±aspect_y）。
        ax.set_xticks(np.linspace(-aspect_x, aspect_x, num_grid))
        ax.set_yticks(np.linspace(-aspect_y, aspect_y, num_grid))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if show:
            plt.show()
        else:
            depth_str = "inf" if depth == float("inf") else f"{-depth}mm"
            if save_name is None:
                save_name = f"./distortion_{depth_str}.png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # 调制传递函数（MTF）
    # ================================================================
    def mtf(self, fov, wvln=None):
        """计算单一视场位置的几何 MTF。

        调制传递函数描述透镜随空间频率保持对比度的能力。本实现采用基于光线的几何方法：
        先通过 ``self.psf()`` 计算指定视场位置的 PSF，再由 ``psf2mtf()`` 投影到切向和
        弧矢方向并计算一维 FFT 幅值。切向与弧矢 MTF 的差异可反映像散。

        参数：
            fov (float): 视场角，单位 rad；内部映射为归一化点
                ``[0, -fov/rfov, self.obj_depth]``。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。

        返回：
            freq (np.ndarray): 空间频率轴，单位 cycles/mm，仅含不包括 DC 的正频率。
            mtf_tan (np.ndarray): 切向（子午）MTF，归一化后低频处趋近 1。
            mtf_sag (np.ndarray): 弧矢 MTF，采用相同归一化方式。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        point = [0, -fov / self.rfov, self.obj_depth]
        psf = self.psf(points=point, recenter=True, wvln=wvln)
        freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size=self.pixel_size)
        return freq, mtf_tan, mtf_sag

    @staticmethod
    def psf2mtf(psf, pixel_size):
        """将二维点扩散函数转换为切向和弧矢 MTF 曲线。

        MTF 是光学传递函数 OTF 的幅值，而 OTF 是 PSF 的傅里叶变换。进行可分离的一维
        分析时，沿 x 轴积分 PSF 得到切向线扩散函数，沿 y 轴积分得到弧矢线扩散函数，
        再计算 ``|FFT(LSF)|`` 并用 DC 分量归一化，使 MTF(0) = 1。按照 Zemax MTF
        图的约定，仅返回不包括 DC 的正频率。

        参数：
            psf (torch.Tensor | np.ndarray): 二维 PSF，shape 为 ``[H, W]``。
                y 轴（行）对应切向（子午）方向，x 轴（列）对应弧矢方向。
            pixel_size (float): 像素间距，单位 mm；频率轴缩放满足
                ``Nyquist = 0.5 / pixel_size`` cycles/mm。

        返回：
            freq (np.ndarray): 空间频率，单位 cycles/mm，为不包括 DC 的正频率，
                长度约为 ``W // 2``。
            mtf_tan (np.ndarray): 切向 MTF，在 DC 处归一化为 1。
            mtf_sag (np.ndarray): 弧矢 MTF，在 DC 处归一化为 1。

        参考资料：
            - https://en.wikipedia.org/wiki/Optical_transfer_function
            - Edmund Optics: Introduction to Modulation Transfer Function.
        """
        # 转为 numpy（支持 torch tensor 和 numpy array）
        try:
            psf_np = psf.detach().cpu().numpy()
        except AttributeError:
            try:
                psf_np = psf.cpu().numpy()
            except AttributeError:
                psf_np = np.asarray(psf)

        # 计算线扩散函数（沿正交轴积分 PSF）
        # y 轴对应切向，x 轴对应弧矢
        lsf_sagittal = psf_np.sum(axis=0)  # x 的函数
        lsf_tangential = psf_np.sum(axis=1)  # y 的函数

        # 单边频谱（适用于实数输入）
        mtf_sag = np.abs(np.fft.rfft(lsf_sagittal))
        mtf_tan = np.abs(np.fft.rfft(lsf_tangential))

        # 使用 DC 分量归一化，确保 MTF(0) == 1
        dc_sag = mtf_sag[0] if mtf_sag.size > 0 else 1.0
        dc_tan = mtf_tan[0] if mtf_tan.size > 0 else 1.0
        if dc_sag != 0:
            mtf_sag = mtf_sag / dc_sag
        if dc_tan != 0:
            mtf_tan = mtf_tan / dc_tan

        # 单边频率轴，单位 cycles/mm
        fx = np.fft.rfftfreq(lsf_sagittal.size, d=pixel_size)
        freq = fx
        positive_freq_idx = freq > 0

        return (
            freq[positive_freq_idx],
            mtf_tan[positive_freq_idx],
            mtf_sag[positive_freq_idx],
        )

    @torch.no_grad()
    def draw_mtf(
        self,
        save_name="./lens_mtf.png",
        relative_fov_list=[0.0, 0.7, 1.0],
        depth_list=None,
        psf_ks=128,
        show=False,
    ):
        """绘制多个深度和视场位置的 MTF 曲线网格。

        生成 ``len(depth_list) × len(relative_fov_list)`` 子图网格。每个子图显示
        R、G、B 波长的切向 MTF（T，实线）和弧矢 MTF（S，虚线），并在传感器
        Nyquist 频率 ``0.5 / pixel_size`` cycles/mm 处绘制竖线。

        参数：
            save_name (str): 输出 PNG 的路径，默认为 ``'./lens_mtf.png'``。
            relative_fov_list (list[float]): ``[0, 1]`` 内的相对视场位置，
                0 表示轴上，1 表示全视场；默认为 ``[0.0, 0.7, 1.0]``。
            depth_list (list[float]): 物距列表，单位 mm；``float('inf')`` 会替换为
                ``self.obj_depth``，为 ``None`` 时使用 ``[self.obj_depth]``。
            psf_ks (int): PSF 核尺寸，单位 pixel，用于控制 MTF 频率分辨率；
                默认为 128。
            show (bool): 为 ``True`` 时交互显示，默认为 ``False``。
        """
        if depth_list is None:
            depth_list = [self.obj_depth]
        pixel_size = self.pixel_size
        nyquist_freq = 0.5 / pixel_size
        num_fovs = len(relative_fov_list)
        if float("inf") in depth_list:
            depth_list = [self.obj_depth if x == float("inf") else x for x in depth_list]
        num_depths = len(depth_list)

        # 创建图像和子图（共 num_depths * num_fovs 个子图）
        fig, axs = plt.subplots(
            num_depths, num_fovs, figsize=(num_fovs * 3, num_depths * 3), squeeze=False
        )

        # 遍历深度和视场
        for depth_idx, depth in enumerate(depth_list):
            for fov_idx, fov_relative in enumerate(relative_fov_list):
                # 计算 RGB PSF
                point = [0, -fov_relative, depth]
                psf_rgb = self.psf_rgb(points=point, ks=psf_ks, recenter=True)

                # 计算 RGB 各波长的 MTF 曲线
                for wvln_idx, wvln in enumerate(self.wvln_rgb):
                    # 从 PSF 计算切向与弧矢 MTF 曲线
                    psf = psf_rgb[wvln_idx]
                    freq, mtf_tan, mtf_sag = self.psf2mtf(psf, pixel_size)

                    # 绘制 MTF 曲线（切向为实线，弧矢为虚线）
                    ax = axs[depth_idx, fov_idx]
                    color = RGB_COLORS[wvln_idx % len(RGB_COLORS)]
                    wvln_label = RGB_LABELS[wvln_idx % len(RGB_LABELS)]
                    wvln_nm = int(wvln * 1000)
                    ax.plot(
                        freq,
                        mtf_tan,
                        color=color,
                        linestyle="-",
                        label=f"{wvln_label}({wvln_nm}nm)-T",
                    )
                    ax.plot(
                        freq,
                        mtf_sag,
                        color=color,
                        linestyle="--",
                        label=f"{wvln_label}({wvln_nm}nm)-S",
                    )

                # 绘制 Nyquist 频率
                ax.axvline(
                    x=nyquist_freq,
                    color="k",
                    linestyle=":",
                    linewidth=1.2,
                    label="Nyquist",
                )

                # 设置子图标题和标签
                fov_deg = round(fov_relative * self.rfov * 180 / np.pi, 1)
                depth_str = "inf" if depth == float("inf") else f"{depth}"
                ax.set_title(f"FOV: {fov_deg}deg, Depth: {depth_str}mm", fontsize=8)
                ax.set_xlabel("Spatial Frequency [cycles/mm]", fontsize=8)
                ax.set_ylabel("MTF", fontsize=8)
                ax.legend(fontsize=6)
                ax.tick_params(axis="both", which="major", labelsize=7)
                ax.grid(True)
                ax.set_ylim(0, 1.05)

        plt.tight_layout()
        if show:
            plt.show()
        else:
            assert save_name.endswith(".png"), "save_name must end with .png"
            plt.savefig(save_name, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # 渐晕
    # ================================================================
    @torch.no_grad()
    def vignetting(self, depth=None, num_grid=32, num_rays=512):
        """计算全视场的相对照度（渐晕）图。

        渐晕衡量各视场位置因光线被透镜孔径或镜筒边缘裁剪而损失的光量。每个网格单元
        的结果为追迹后仍有效的光线数占发射光线总数的比例。1.0 表示所有光线都到达
        传感器，0.0 表示完全遮挡。

        使用 ``uniform_fov=False`` 的 ``self.sample_grid_rays()`` 在像空间均匀采样，
        再由 ``self.trace2sensor()`` 追迹并将被裁剪光线标记为无效；单元透过率为
        ``count(valid) / num_rays``。

        参数：
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            num_grid (int): 每个轴的网格分辨率，默认为 32。
            num_rays (int): 每个网格单元发射的光线数；数值越大，Monte Carlo 噪声
                越低，默认为 512。

        返回：
            vignetting (torch.Tensor): 渐晕图，shape 为 ``[num_grid, num_grid]``，
                数值范围为 ``[0, 1]``。
        """
        depth = self.obj_depth if depth is None else depth
        # 在均匀像空间而非视场角空间采样，以正确映射传感器
        # shape 为 [num_grid, num_grid, num_rays, 3]
        ray = self.sample_grid_rays(
            depth=depth, num_grid=num_grid, num_rays=num_rays, uniform_fov=False
        )

        # 将光线追迹至传感器
        ray = self.trace2sensor(ray)

        # 计算渐晕图
        vignetting = ray.is_valid.sum(-1) / (ray.is_valid.shape[-1])
        return vignetting

    @torch.no_grad()
    def draw_vignetting(self, filename=None, depth=None, resolution=512, show=False):
        """将渐晕图绘制为带色条的灰度图。

        通过 ``self.vignetting()`` 计算渐晕图，再双线性上采样至
        ``resolution × resolution``。白色表示无渐晕，黑色表示完全渐晕。

        参数：
            filename (str | None): 输出 PNG 的路径；为 ``None`` 时自动生成
                ``'./vignetting_{depth}.png'``。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            resolution (int): 方形输出图像的边长，单位 pixel，默认为 512。
            show (bool): 为 ``True`` 时交互显示，默认为 ``False``。
        """
        depth = self.obj_depth if depth is None else depth
        # 计算渐晕图
        vignetting = self.vignetting(depth=depth)

        # 将渐晕图插值到目标分辨率
        vignetting = F.interpolate(
            vignetting.unsqueeze(0).unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        fig, ax = plt.subplots()
        ax.set_title("Relative Illumination (Vignetting)")
        im = ax.imshow(vignetting.cpu().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

        if show:
            plt.show()
        else:
            if filename is None:
                filename = f"./vignetting_{depth}.png"
            plt.savefig(filename, bbox_inches="tight", format="png", dpi=300)
        plt.close(fig)

    # ================================================================
    # 主光线计算与光线瞄准
    # ================================================================
    @torch.no_grad()
    def calc_chief_ray_infinite(
        self,
        rfov,
        depth=0.0,
        wvln=None,
        plane="meridional",
        num_rays=SPP_CALC,
        ray_aiming=True,
    ):
        """计算一个或多个视场角的主光线，并可选执行光线瞄准。

        本方法对多个视场角进行向量化计算。光线瞄准会向入瞳发射一束光线，并选择在
        孔径光阑处最接近光轴的光线。对于近轴近似失效的广角或鱼眼透镜，该过程对准确
        测量畸变十分重要。

        轴上（``rfov = 0``）主光线沿 z 轴；轴外且 ``ray_aiming=False`` 时，主光线
        直接瞄准入瞳中心；启用瞄准时，根据入瞳几何估算物方位置，在其附近生成
        ``num_rays`` 条搜索光线，追迹至孔径光阑并选取最靠近光轴的一条。

        参数：
            rfov (float | torch.Tensor): 视场角，单位 degree。正标量扩展为
                ``[0, rfov]``，非正标量转为单元素 tensor，shape 为 ``[N]`` 的
                tensor 则直接使用。
            depth (float | torch.Tensor): 物方深度，单位 mm，默认为 0.0。
            wvln (float): 波长，单位 µm；为 ``None`` 时使用 ``self.primary_wvln``。
            plane (str): ``'sagittal'`` 或 ``'meridional'``，默认为
                ``'meridional'``。
            num_rays (int): 光线瞄准搜索束的光线数，默认为 ``SPP_CALC``。
            ray_aiming (bool): 为 ``True`` 时执行光线瞄准，默认为 ``True``。

        返回：
            chief_ray_o (torch.Tensor): 光线起点，shape 为 ``[N, 3]``。
            chief_ray_d (torch.Tensor): 单位方向，shape 为 ``[N, 3]``。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        if isinstance(rfov, (int, float)):
            if rfov > 0:
                rfov = torch.linspace(0, rfov, 2, device=self.device)
            else:
                rfov = torch.tensor([float(rfov)], device=self.device)
        else:
            rfov = rfov.to(self.device)

        if not isinstance(depth, torch.Tensor):
            depth = torch.tensor(depth, device=self.device).repeat(len(rfov))

        # 设置主光线
        chief_ray_o = torch.zeros([len(rfov), 3], device=self.device)
        chief_ray_d = torch.zeros([len(rfov), 3], device=self.device)

        # 将 rfov 转换为 rad
        rfov = rfov * torch.pi / 180.0

        if torch.any(rfov == 0):
            chief_ray_o[0, ...] = torch.tensor(
                [0.0, 0.0, depth[0]], device=self.device, dtype=torch.float32
            )
            chief_ray_d[0, ...] = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device, dtype=torch.float32
            )
            if len(rfov) == 1:
                return chief_ray_o, chief_ray_d

        # 提取非零 rfov 条目进行处理
        has_zero = torch.any(rfov == 0)
        if has_zero:
            start_idx = 1
            rfovs = rfov[1:]
            depths = depth[1:]
        else:
            start_idx = 0
            rfovs = rfov
            depths = depth

        if self.aper_idx == 0:
            if plane == "sagittal":
                chief_ray_o[start_idx:, ...] = torch.stack(
                    [depths * torch.tan(rfovs), torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[start_idx:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[start_idx:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), depths * torch.tan(rfovs), depths], dim=-1
                )
                chief_ray_d[start_idx:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

            return chief_ray_o, chief_ray_d

        # 缩放因子
        pupilz, pupilr = self.calc_entrance_pupil()
        y_distance = torch.tan(rfovs) * (abs(depths) + pupilz)

        if ray_aiming:
            scale = 0.05
            min_delta = 0.05 * pupilr  # 基于瞳孔半径设置最小搜索范围
            delta = torch.clamp(scale * y_distance, min=min_delta)

        if not ray_aiming:
            if plane == "sagittal":
                chief_ray_o[start_idx:, ...] = torch.stack(
                    [-y_distance, torch.zeros_like(rfovs), depths], dim=-1
                )
                chief_ray_d[start_idx:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                chief_ray_o[start_idx:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), -y_distance, depths], dim=-1
                )
                chief_ray_d[start_idx:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        else:
            min_y = -y_distance - delta
            max_y = -y_distance + delta
            t = torch.linspace(0, 1, num_rays, device=min_y.device)
            o1_linspace = min_y.unsqueeze(-1) + t * (max_y - min_y).unsqueeze(-1)

            o1 = torch.zeros([len(rfovs), num_rays, 3], device=self.device)
            # 每个视场使用自身深度，而不是全部使用 depths[0]。
            o1[:, :, 2] = depths.unsqueeze(-1)

            o2_linspace = -delta.unsqueeze(-1) + t * (2 * delta).unsqueeze(-1)

            o2 = torch.zeros([len(rfovs), num_rays, 3], device=self.device)
            o2[:, :, 2] = pupilz

            if plane == "sagittal":
                o1[:, :, 0] = o1_linspace
                o2[:, :, 0] = o2_linspace
            else:
                o1[:, :, 1] = o1_linspace
                o2[:, :, 1] = o2_linspace

            # 追迹至孔径光阑
            ray = Ray(o1, o2 - o1, wvln=wvln, device=self.device)
            inc_ray = ray.clone()
            surf_range = range(0, self.aper_idx + 1)
            ray, _ = self.trace(ray, surf_range=surf_range)

            # 查找最接近光轴的光线
            if plane == "sagittal":
                _, center_idx = torch.min(torch.abs(ray.o[..., 0]), dim=1)
                chief_ray_o[start_idx:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[start_idx:, ...] = torch.stack(
                    [torch.sin(rfovs), torch.zeros_like(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )
            else:
                _, center_idx = torch.min(torch.abs(ray.o[..., 1]), dim=1)
                chief_ray_o[start_idx:, ...] = inc_ray.o[
                    torch.arange(len(rfovs)), center_idx.long(), ...
                ]
                chief_ray_d[start_idx:, ...] = torch.stack(
                    [torch.zeros_like(rfovs), torch.sin(rfovs), torch.cos(rfovs)],
                    dim=-1,
                )

        return chief_ray_o, chief_ray_d

    # ====================================================================================
    # 点列、渲染与综合分析
    # ====================================================================================
    @torch.no_grad()
    def analysis_rendering(
        self,
        img_org,
        save_name=None,
        depth=None,
        spp=SPP_RENDER,
        unwarp=False,
        method="ray_tracing",
        show=False,
    ):
        """通过透镜渲染测试图像并报告 PSNR / SSIM。

        模拟把给定图像放在指定物距时传感器捕获的结果。渲染考虑模糊、畸变、渐晕和
        色差等全部几何像差，并可选执行逆畸变校正 ``unwarp``，分别报告原始渲染和
        校正后渲染的质量指标。

        本方法把 ``img_org`` 转为 ``[1, 3, H, W]`` 浮点 tensor，临时调整传感器
        分辨率，调用 ``self.render()``，计算原图与渲染图之间的 PSNR 和 SSIM，必要时
        调用 ``self.unwarp()``，最后恢复原传感器分辨率。

        参数：
            img_org (np.ndarray | torch.Tensor): 源图像，shape 为 ``[H, W, 3]``，
                可为 uint8 ``[0, 255]`` 或 float ``[0, 1]``。
            save_name (str | None): 保存 PNG 的路径前缀；非 ``None`` 时保存
                ``'{save_name}.png'``，若校正畸变还保存
                ``'{save_name}_unwarped.png'``。
            depth (float): 物距，单位 mm；为 ``None`` 时使用 ``self.obj_depth``。
            spp (int): 每个像素的渲染采样数（光线数），默认为 ``SPP_RENDER``。
            unwarp (bool): 为 ``True`` 时在渲染后校正畸变，默认为 ``False``。
            method (str): 渲染后端，可为 ``'ray_tracing'``、``'psf_map'`` 或
                ``'psf_patch'``，默认为 ``'ray_tracing'``。
            show (bool): 为 ``True`` 时使用 matplotlib 显示结果，默认为 ``False``。

        返回：
            img_render (torch.Tensor): 渲染后（并可选反扭曲）的图像，shape 为
                ``[1, 3, H, W]``，浮点值范围为 ``[0, 1]``。
        """
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
        from torchvision.utils import save_image
        depth = self.obj_depth if depth is None else depth
        # 调整传感器分辨率以匹配图像
        sensor_res_original = self.sensor_res
        if isinstance(img_org, np.ndarray):
            img = torch.from_numpy(img_org).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif torch.is_tensor(img_org):
            img = img_org.permute(2, 0, 1).unsqueeze(0).float()
            if img.max() > 1.0:
                img = img / 255.0
        img = img.to(self.device)
        self.set_sensor_res(sensor_res=img.shape[-2:])

        # 图像渲染
        img_render = self.render(img, depth=depth, method=method, spp=spp)

        # 计算 PSNR 和 SSIM
        img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        render_np = img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
        render_psnr = round(peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3)
        render_ssim = round(structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4)
        print(f"Rendered image: PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}")

        # 保存图像
        if save_name is not None:
            save_image(img_render, f"{save_name}.png")

        # 反扭曲以校正几何畸变
        if unwarp:
            img_render = self.unwarp(img_render, depth)

            # 计算 PSNR 和 SSIM
            render_np = img_render.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().detach().numpy()
            render_psnr = round(peak_signal_noise_ratio(img_np, render_np, data_range=1.0), 3)
            render_ssim = round(structural_similarity(img_np, render_np, channel_axis=2, data_range=1.0), 4)
            print(
                f"Rendered image (unwarped): PSNR={render_psnr:.3f}, SSIM={render_ssim:.4f}"
            )

            if save_name is not None:
                save_image(img_render, f"{save_name}_unwarped.png")

        # 恢复传感器分辨率
        self.set_sensor_res(sensor_res=sensor_res_original)

        # 显示图像
        if show:
            plt.imshow(img_render.cpu().squeeze(0).permute(1, 2, 0).numpy())
            plt.title("Rendered image")
            plt.axis("off")
            plt.show()
            plt.close()

        return img_render

    @torch.no_grad()
    def analysis_spot(self, num_field=3, depth=float("inf")):
        """计算 RGB 在多个视场位置的 RMS 与几何点列半径。

        沿子午方向的 ``num_field`` 个等间隔视场位置追迹 R、G、B 三个波长，并以
        **全部波长的联合质心**为参考计算复色 RMS 和几何点列半径，与 Zemax 默认的
        “相对于质心的 RMS 点列半径”一致。

        对每个视场点，汇总三个波长的所有有效光线交点并计算联合质心 ``c``；
        ``RMS = sqrt(mean(||xy - c||²))``，几何半径为
        ``max(||xy - c||)``，最后从 mm 转为 μm（× 1000）。

        参数：
            num_field (int): 从轴上到全视场的采样位置数，默认为 3。
            depth (float): 物距，单位 mm；准直光使用 ``float('inf')``，默认为
                ``float('inf')``。

        返回：
            rms_results (dict[str, dict[str, float]]): 按视场位置字符串索引的点列
                分析结果，例如 ``'fov0.0'``、``'fov0.5'``、``'fov1.0'``。
                ``'rms'`` 为复色 RMS 点列半径，``'radius'`` 为复色几何点列半径，
                两者单位均为 μm。
        """
        # 分别追迹各波长，并按视场汇总跨波长光线
        xy_list = []
        valid_list = []
        for wvln in self.wvln_rgb:
            ray = self.sample_radial_rays(
                num_field=num_field, depth=depth, num_rays=SPP_PSF, wvln=wvln
            )
            ray = self.trace2sensor(ray)
            xy_list.append(ray.o[..., :2])
            valid_list.append(ray.is_valid)

        # 沿波长维汇总，shape 分别为 [num_field, 3*num_rays, 2] 和 [num_field, 3*num_rays]
        xy_all = torch.cat(xy_list, dim=-2)
        valid_all = torch.cat(valid_list, dim=-1)

        # 各视场的联合复色质心，shape 为 [num_field, 1, 2]
        valid_mask = valid_all.unsqueeze(-1)
        center = (xy_all * valid_mask).sum(-2) / (
            valid_all.sum(-1, keepdim=True) + EPSILON
        )
        center = center.unsqueeze(-2)

        # 到联合质心的距离平方，shape 为 [num_field, 3*num_rays]
        dist_sq = ((xy_all - center) ** 2).sum(-1)

        # 各视场的复色 RMS 点列半径，shape 为 [num_field]
        spot_rms = (
            (dist_sq * valid_all).sum(-1) / (valid_all.sum(-1) + EPSILON)
        ).sqrt()
        # 几何点列半径（有效光线中的最大距离）
        dist_masked = torch.where(
            valid_all > 0, dist_sq, torch.full_like(dist_sq, -1.0)
        )
        spot_radius = dist_masked.max(dim=-1).values.clamp(min=0.0).sqrt()

        # 将 mm 转换为 μm
        avg_rms_radius_um = spot_rms * 1000.0
        avg_geo_radius_um = spot_radius * 1000.0

        # 打印结果
        print(f"Ray spot analysis results for depth {depth}:")
        print(
            f"RMS radius: FoV (0.0) {avg_rms_radius_um[0]:.3f} um, FoV (0.5) {avg_rms_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_rms_radius_um[-1]:.3f} um"
        )
        print(
            f"Geo radius: FoV (0.0) {avg_geo_radius_um[0]:.3f} um, FoV (0.5) {avg_geo_radius_um[num_field // 2]:.3f} um, FoV (1.0) {avg_geo_radius_um[-1]:.3f} um"
        )

        # 保存到字典
        rms_results = {}
        fov_ls = torch.linspace(0, 1, num_field)
        for i in range(num_field):
            fov = round(fov_ls[i].item(), 2)
            rms_results[f"fov{fov}"] = {
                "rms": round(avg_rms_radius_um[i].item(), 4),
                "radius": round(avg_geo_radius_um[i].item(), 4),
            }

        return rms_results

    @torch.no_grad()
    def analysis(
        self,
        save_name="./lens",
        depth=float("inf"),
        full_eval=False,
        render=False,
        render_unwarp=False,
        lens_title=None,
        show=False,
    ):
        """运行透镜的综合光学分析流程。

        这是评估透镜设计的主要入口，按顺序串联多个评估步骤，并使用共同的
        ``save_name`` 前缀保存所有图像。始终绘制透镜布局并计算复色点列 RMS/半径；
        ``full_eval=True`` 时还生成点列图、MTF 网格、畸变曲线和渐晕图；
        ``render=True`` 时通过透镜渲染测试图并报告 PSNR/SSIM。

        参数：
            save_name (str): 所有输出文件的路径前缀，各图追加相应后缀，默认为
                ``'./lens'``。
            depth (float): 物距，单位 mm；渲染和渐晕处理中会将 ``float('inf')``
                替换为 ``self.obj_depth``。
            full_eval (bool): 为 ``True`` 时运行全部评估图；否则仅执行布局和点列 RMS，
                默认为 ``False``。
            render (bool): 为 ``True`` 时通过透镜渲染测试图，默认为 ``False``。
            render_unwarp (bool): 当 ``render=True`` 时，为 ``True`` 还会生成反扭曲
                渲染图，默认为 ``False``。
            lens_title (str | None): 布局图标题，默认为 ``None``。
            show (bool): 为 ``True`` 时交互显示所有图，默认为 ``False``。
        """
        # 绘制透镜布局和光线路径
        self.draw_layout(
            filename=f"{save_name}.png",
            lens_title=lens_title,
            depth=depth,
            show=show,
        )

        # 计算 RMS 误差
        self.analysis_spot(depth=depth)

        # 综合光学评估
        if full_eval:
            # 绘制点列图
            self.draw_spot_radial(
                save_name=f"{save_name}_spot.png",
                depth=depth,
                show=show,
            )

            # 绘制 MTF
            if depth == float("inf"):
                self.draw_mtf(
                    depth_list=[self.obj_depth],
                    save_name=f"{save_name}_mtf.png",
                    show=show,
                )
            else:
                self.draw_mtf(
                    depth_list=[depth],
                    save_name=f"{save_name}_mtf.png",
                    show=show,
                )

            # 绘制畸变
            self.draw_distortion_radial(
                save_name=f"{save_name}_distortion.png",
                show=show,
            )

            # 绘制渐晕
            eval_depth = self.obj_depth if depth == float("inf") else depth
            self.draw_vignetting(
                filename=f"{save_name}_vignetting.png",
                depth=eval_depth,
                show=show,
            )

        # 渲染图像并计算 PSNR 和 SSIM
        if render:
            depth = self.obj_depth if depth == float("inf") else depth
            img_org = Image.open("./datasets/charts/NBS_1963_1k.png").convert("RGB")
            img_org = np.array(img_org)
            self.analysis_rendering(
                img_org,
                depth=depth,
                spp=SPP_RENDER,
                unwarp=render_unwarp,
                save_name=f"{save_name}_render",
                show=show,
            )
