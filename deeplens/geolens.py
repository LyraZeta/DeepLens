# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""几何镜头模型。使用可微光线追迹模拟光在几何镜头中的传播，精度与 Zemax 对齐。

技术论文:
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    DEFAULT_WAVE,
    DELTA_PARAXIAL,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_PSF,
    SPP_RENDER,
    WAVE_RGB,
)
from .geolens_pkg.eval import GeoLensEval
from .geolens_pkg.io import GeoLensIO
from .geolens_pkg.optim import GeoLensOptim
from .geolens_pkg.psf_compute import GeoLensPSF
from .geolens_pkg.optim_ops import GeoLensSurfOps
from .geolens_pkg.vis3d import GeoLensVis3D
from .geolens_pkg.vis import GeoLensVis
from .imgsim import backward_integral
from .lens import Lens
from .geometric_surface import Aperture
from .material import Material
from .light import Ray

class GeoLens(
    GeoLensPSF,
    GeoLensEval,
    GeoLensOptim,
    GeoLensSurfOps,
    GeoLensVis,
    GeoLensIO,
    GeoLensVis3D,
    Lens,
):
    """使用向量化光线追迹的可微几何镜头。

    这是 DeepLens 中的主要镜头模型。支持从 JSON、Zemax `.zmx` 或 Code V
    `.seq` 文件加载多元件折射（以及部分反射）系统，精度与 Zemax OpticStudio 对齐。

    使用 mixin 架构：在类定义时组合七个专用 mixin 类，使各项职责相互隔离：
    `GeoLensPSF`（PSF 计算）、`GeoLensEval`（光斑/MTF/畸变/渐晕评估）、
    `GeoLensOptim`（损失和基于梯度的优化）、`GeoLensSurfOps`（表面几何操作）、
    `GeoLensVis`（二维布局/光线可视化）、`GeoLensIO`（JSON/Zemax 读写）和
    `GeoLensVis3D`（三维网格可视化）。

    属性:
        surfaces (list[Surface]): 按顺序排列的光学表面列表。
        materials (list[Material]): 表面之间的光学材料。
        d_sensor (torch.Tensor): 从原点到传感器平面的距离 [mm]。
        foclen (float): 有效焦距 [mm]。
        fnum (float): F-number。
        rfov (float): 实际半对角视场 [radians]。
        sensor_size (tuple): 传感器物理尺寸 (W, H) [mm]。
        sensor_res (tuple): 传感器分辨率 (W, H) [pixels]。
        pixel_size (float): 像素间距 [mm]。

    参考:
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," Nature Communications 2024.
    """

    # GeoLens 默认使用光线追迹渲染（可端到端追迹光线），覆盖基类 `Lens` 的
    # 默认值 ``"psf_patch"``。
    _default_render_method = "ray_tracing"

    def __init__(
        self,
        filename=None,
        device=None,
        dtype=torch.float32,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """初始化折射镜头。

        GeoLens 有两种初始化方式：
            1. 从 .json/.zmx/.seq 文件读取镜头
            2. 不使用镜头文件进行初始化，随后手动添加表面和材料

        参数:
            filename (str, optional): 镜头文件（.json、.zmx 或 .seq）路径。默认为 None。
            device (torch.device, optional): 张量计算设备。默认为 None。
            dtype (torch.dtype, optional): 计算数据类型。默认为 torch.float32。
            primary_wvln (float, optional): 主要设计波长 [µm]。调用方法时未显式
                提供 ``wvln``，则使用此值。默认为 ``DEFAULT_WAVE``。
            wvln_rgb (sequence of float, optional): RGB 计算所用的三个波长，按
                ``[R, G, B]`` 排列，单位为 µm。默认为 ``WAVE_RGB``。
            obj_depth (float, optional): 默认物体深度 [mm]。调用方法时未显式
                提供 ``depth``，则使用此值。默认为 ``DEPTH``。
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
            self.read_lens(filename)
        else:
            self.surfaces = []
            self.materials = []
            # 设置默认传感器尺寸和分辨率
            self.sensor_size = (8.0, 8.0)
            self.sensor_res = (2000, 2000)
            self.to(self.device)

    def read_lens(self, filename):
        """从文件读取 GeoLens。

        支持的文件格式:
            - .json：DeepLens 原生 JSON 格式
            - .zmx：Zemax 镜头文件格式
            - .seq：CODE V 序列文件格式

        参数:
            filename (str): 镜头文件路径。

        说明:
            传感器尺寸和分辨率通常会被文件中的值覆盖。
        """
        # 加载镜头文件
        if filename[-4:] == ".txt":
            raise ValueError("File format .txt has been deprecated.")
        elif filename[-5:] == ".json":
            self.read_lens_json(filename)
        elif filename[-4:] == ".zmx":
            self.read_lens_zmx(filename)
        elif filename[-4:] == ".seq":
            self.read_lens_seq(filename)
        else:
            raise ValueError(f"File format {filename[-4:]} not supported.")

        # 若镜头文件未设置传感器尺寸和分辨率，则补全它们
        if not hasattr(self, "sensor_size"):
            self.sensor_size = (8.0, 8.0)
            print(
                f"Sensor_size not found in lens file. Using default: {self.sensor_size} mm. "
                "Consider specifying sensor_size in the lens file or using set_sensor()."
            )

        if not hasattr(self, "sensor_res"):
            self.sensor_res = (2000, 2000)
            print(
                f"Sensor_res not found in lens file. Using default: {self.sensor_res} pixels. "
                "Consider specifying sensor_res in the lens file or using set_sensor()."
            )
            self.set_sensor_res(self.sensor_res)

        # 加载镜头后计算 foclen、fov 和 fnum
        self.to(self.device)
        self.astype(self.dtype)
        self.post_computation()

    def post_computation(self):
        """加载或修改镜头后计算派生光学属性。

        计算并缓存:
            - 有效焦距（EFL）
            - 入瞳和出瞳的位置及半径
            - 水平、垂直和对角方向的视场（FoV）
            - F-number
            - 镜头设计约束（边缘/中心厚度边界等）

        说明:
            修改镜头几何结构后应调用此方法。
        """
        self.calc_foclen()
        self.calc_pupil()
        self.calc_fov()
        self.init_constraints()

    def __call__(self, ray):
        """追迹光线通过镜头系统（`trace` 的可调用简写）。

        参数:
            ray (Ray): 要追迹的光线对象。

        返回:
            ray_out (Ray): 通过各表面传播后的光线。
            ray_o_record (list or None): 记录的光线位置，或 None。
        """
        return self.trace(ray)

    # ====================================================================================
    # 光线采样
    # ====================================================================================
    @torch.no_grad()
    def sample_grid_rays(
        self,
        depth=float("inf"),
        num_grid=(11, 11),
        num_rays=SPP_PSF,
        wvln=None,
        uniform_fov=True,
        sample_more_off_axis=False,
        scale_pupil=1.0,
    ):
        """从物方采样覆盖视场的光线网格。

        若 `depth` 为无穷大，则在等间隔视场角处采样准直光线；若 `depth` 有限，
        则从物点网格采样发散点光源光线。用于 PSF 图、RMS 误差图和点列图。

        参数:
            depth (float, optional): 物距 [mm]。准直光使用 `float("inf")`。
                默认为 `float("inf")`。
            num_grid (int or tuple, optional): 网格点数，可为 (num_x, num_y)，
                或同时用于两维的单个 int。默认为 (11, 11)。
            num_rays (int, optional): 每个网格点的光线数。默认为 SPP_PSF。
            wvln (float, optional): 波长 [µm]。为 None（默认）时使用
                `self.primary_wvln`。
            uniform_fov (bool, optional): 为 True 时均匀采样 FoV 角；否则采样均匀
                物体网格。默认为 True。
            sample_more_off_axis (bool, optional): 为 True 时将更多网格样本集中于
                离轴视场。默认为 False。
            scale_pupil (float, optional): 瞳孔半径缩放因子。默认为 1.0。

        返回:
            rays (Ray): 采样光线，shape [num_grid[1], num_grid[0], num_rays, 3]。
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # 若 num_grid 为 int，则将其规范化为元组
        if isinstance(num_grid, int):
            num_grid = (num_grid, num_grid)

        # 计算网格光源的视场角。左上视场的 fov_x 为正、fov_y 为负
        x_list = [x for x in np.linspace(1, -1, num_grid[0])]
        y_list = [y for y in np.linspace(-1, 1, num_grid[1])]
        if sample_more_off_axis:
            x_list = [np.sign(x) * np.abs(x) ** 0.5 for x in x_list]
            y_list = [np.sign(y) * np.abs(y) ** 0.5 for y in y_list]

        # 计算 FoV_x 和 FoV_y
        if uniform_fov:
            # 均匀采样 FoV 角
            fov_x_list = [x * self.vfov / 2 for x in x_list]
            fov_y_list = [y * self.hfov / 2 for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]
        else:
            # 采样均匀物体网格
            fov_x_list = [np.arctan(x * np.tan(self.vfov / 2)) for x in x_list]
            fov_y_list = [np.arctan(y * np.tan(self.hfov / 2)) for y in y_list]
            fov_x_list = [float(np.rad2deg(fov_x)) for fov_x in fov_x_list]
            fov_y_list = [float(np.rad2deg(fov_y)) for fov_y in fov_y_list]

        # 采样光线（通过统一 API 采样准直光或点光源）
        rays = self.sample_from_fov(
            fov_x=fov_x_list,
            fov_y=fov_y_list,
            depth=depth,
            num_rays=num_rays,
            wvln=wvln,
            scale_pupil=scale_pupil,
        )
        return rays

    @torch.no_grad()
    def sample_radial_rays(
        self,
        num_field=5,
        depth=float("inf"),
        num_rays=SPP_PSF,
        wvln=None,
        direction="y",
    ):
        """沿选定方向，在等间隔视场角处采样径向光线。

        参数:
            num_field (int): 从轴上到全视场的视场角数量。默认为 5。
            depth (float): 物距 [mm]。准直光使用 ``float('inf')``。
                默认为 ``float('inf')``。
            num_rays (int): 每个视场位置的光线数。默认为 ``SPP_PSF``。
            wvln (float): 波长 [µm]。为 ``None``（默认）时使用
                ``self.primary_wvln``。
            direction (str): 采样方向——``"y"``（子午方向，默认）、
                ``"x"``（弧矢方向）、``"diagonal"``（45°，x = y）。

        返回:
            ray (Ray): shape 为 ``[num_field, num_rays, 3]`` 的光线对象。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        device = self.device
        fov_deg = self.rfov * 180 / torch.pi
        fov_list = torch.linspace(0, fov_deg, num_field, device=device)

        if direction == "y":
            ray = self.sample_from_fov(
                fov_x=0.0, fov_y=fov_list, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "x":
            ray = self.sample_from_fov(
                fov_x=fov_list, fov_y=0.0, depth=depth, num_rays=num_rays, wvln=wvln
            )
        elif direction == "diagonal":
        # sample_from_fov 会创建 meshgrid；成对的对角采样需循环处理
            rays = [
                self.sample_from_fov(
                    fov_x=f.item(), fov_y=f.item(), depth=depth, num_rays=num_rays, wvln=wvln
                )
                for f in fov_list
            ]
            ray_o = torch.stack([r.o for r in rays], dim=0)
            ray_d = torch.stack([r.d for r in rays], dim=0)
            ray = Ray(ray_o, ray_d, wvln, device=device)
        else:
            raise ValueError(f"Invalid direction: {direction!r}. Use 'x', 'y', or 'diagonal'.")
        return ray

    @torch.no_grad()
    def sample_from_points(
        self,
        points=[[0.0, 0.0, -10000.0]],
        num_rays=SPP_PSF,
        wvln=None,
        scale_pupil=1.0,
    ):
        """从物方点光源（绝对物理坐标）采样光线。

        光线从给定物点发出，并向入瞳呈扇形扩散。用于 PSF 和主光线计算。

        参数:
            points (list or torch.Tensor): 物方光线原点 [mm]，shape [3]、[N, 3]
                或 [Nx, Ny, 3]。默认为 [[0.0, 0.0, -10000.0]]。
            num_rays (int): 每个点的光线数。默认为 SPP_PSF。
            wvln (float): 波长 [µm]。为 None（默认）时使用 `self.primary_wvln`。
            scale_pupil (float): 瞳孔半径缩放因子。默认为 1.0。

        返回:
            rays (Ray): 采样光线，shape [*points.shape[:-1], num_rays, 3]。
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # 光线原点已给定
        if not torch.is_tensor(points):
            ray_o = torch.tensor(points, device=self.device)
        else:
            ray_o = points.to(self.device)

        # 在瞳孔上采样点
        pupilz, pupilr = self.get_entrance_pupil()
        pupilr *= scale_pupil
        ray_o2 = self.sample_circle(
            r=pupilr, z=pupilz, shape=(*ray_o.shape[:-1], num_rays)
        )

        # 计算光线方向
        if len(ray_o.shape) == 1:
            # 输入点 shape 为 [3]
            ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)  # shape [num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 2:
            # 输入点 shape 为 [N, 3]
            ray_o = ray_o.unsqueeze(1).repeat(1, num_rays, 1)  # shape [N, num_rays, 3]
            ray_d = ray_o2 - ray_o

        elif len(ray_o.shape) == 3:
            # 输入点 shape 为 [Nx, Ny, 3]
            ray_o = ray_o.unsqueeze(2).repeat(
                1, 1, num_rays, 1
            )  # shape [Nx, Ny, num_rays, 3]
            ray_d = ray_o2 - ray_o

        else:
            raise Exception("The shape of input object positions is not supported.")

        # 计算光线
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        return rays

    @torch.no_grad()
    def sample_from_fov(
        self,
        fov_x=[0.0],
        fov_y=[0.0],
        depth=float("inf"),
        num_rays=SPP_CALC,
        wvln=None,
        entrance_pupil=True,
        scale_pupil=1.0,
    ):
        """在给定视场角处从物方采样光线。

        对无穷远深度，生成准直平行光线：原点分布于入瞳上，同一视场中的所有光线
        共用由 FOV 角决定的方向。

        对有限深度，生成发散点光源光线：点光源位置由 FOV 角和深度决定，光线向
        入瞳呈扇形扩散。

        参数:
            fov_x (float or list): xz 平面内的视场角（degrees）。
            fov_y (float or list): yz 平面内的视场角（degrees）。
            depth (float): 物距 [mm]。准直光线使用 ``float('inf')``，点光源光线
                使用有限值。
            num_rays (int): 每个视场点的光线数。
            wvln (float): 波长 [µm]。为 ``None``（默认）时使用
                ``self.primary_wvln``。
            entrance_pupil (bool): 为 True 时在入瞳上采样，否则在表面 0 上采样。
                默认为 True。
            scale_pupil (float): 瞳孔半径缩放因子。

        返回:
            rays (Ray): shape 为 ``[..., num_rays, 3]`` 的光线；当对应 fov 输入
                为标量时压缩其前导维度。
        """
        wvln = self.primary_wvln if wvln is None else wvln

        # 记录哪些输入为标量，以确定输出 shape
        x_scalar = isinstance(fov_x, (float, int))
        y_scalar = isinstance(fov_y, (float, int))
        if x_scalar:
            fov_x = [float(fov_x)]
        if y_scalar:
            fov_y = [float(fov_y)]

        fov_x_rad = torch.tensor([fx * torch.pi / 180 for fx in fov_x], device=self.device)
        fov_y_rad = torch.tensor([fy * torch.pi / 180 for fy in fov_y], device=self.device)
        fov_x_grid, fov_y_grid = torch.meshgrid(fov_x_rad, fov_y_rad, indexing="xy")

        # 瞳孔位置和半径
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = self.surfaces[0].d.item(), self.surfaces[0].r
        pupilr *= scale_pupil

        if depth == float("inf"):
            # 准直光线：原点位于瞳孔上，每个视场具有统一方向
            ray_o = self.sample_circle(
                r=pupilr, z=pupilz, shape=[len(fov_y), len(fov_x), num_rays]
            )
            dx = torch.tan(fov_x_grid).unsqueeze(-1).expand_as(ray_o[..., 0])
            dy = torch.tan(fov_y_grid).unsqueeze(-1).expand_as(ray_o[..., 1])
            dz = torch.ones_like(ray_o[..., 2])
            ray_d = torch.stack((dx, dy, dz), dim=-1)

            if x_scalar:
                ray_o = ray_o.squeeze(1)
                ray_d = ray_d.squeeze(1)
            if y_scalar:
                ray_o = ray_o.squeeze(0)
                ray_d = ray_d.squeeze(0)

            rays = Ray(ray_o, ray_d, wvln, device=self.device)
            rays.prop_to(-1.0)

        else:
            # 点光源光线：从物点发出，向瞳孔呈扇形扩散
            x = torch.tan(fov_x_grid) * depth
            y = torch.tan(fov_y_grid) * depth
            z = torch.full_like(x, depth)
            points = torch.stack((x, y, z), dim=-1)

            if x_scalar:
                points = points.squeeze(-2)
            if y_scalar:
                points = points.squeeze(0)

            rays = self.sample_from_points(
                points=points, num_rays=num_rays, wvln=wvln, scale_pupil=scale_pupil
            )

        return rays

    @torch.no_grad()
    def sample_sensor(self, spp=64, wvln=None, sub_pixel=False):
        """从传感器像素采样光线（反向光线），用于基于光线追迹的渲染。

        参数:
            spp (int, optional): 每像素采样数。默认为 64。
            wvln (float, optional): 光线波长 [µm]。为 ``None``（默认）时使用
                ``self.primary_wvln``。
            sub_pixel (bool, optional): 是否在像素内部采样多个点。默认为 False。

        返回:
            ray (Ray): 光线对象，shape [H, W, spp, 3]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        w, h = self.sensor_size
        W, H = self.sensor_res
        device = self.device

        # 在传感器平面上采样点
        # 渲染时使用左上点作为参考，因此此处应采样右下点
        x1, y1 = torch.meshgrid(
            torch.linspace(-w / 2, w / 2, W + 1, device=device,)[1:],
            torch.linspace(h / 2, -h / 2, H + 1, device=device,)[1:],
            indexing="xy",
        )
        z1 = torch.full_like(x1, self.d_sensor.item())

        # 在瞳孔上采样第二组点
        # sensor_res 为 (W, H)，但 indexing="xy" 的 meshgrid 会生成 (H, W) 数组
        pupilz, pupilr = self.get_exit_pupil()
        ray_o2 = self.sample_circle(r=pupilr, z=pupilz, shape=(H, W, spp))

        # 构造光线
        ray_o = torch.stack((x1, y1, z1), 2)
        ray_o = ray_o.unsqueeze(2).repeat(1, 1, spp, 1)  # [H, W, spp, 3]

        # 子像素采样，以获得更真实的渲染
        if sub_pixel:
            delta_ox = (
                torch.rand(ray_o.shape[:-1], device=device)
                * self.pixel_size
            )
            delta_oy = (
                -torch.rand(ray_o.shape[:-1], device=device)
                * self.pixel_size
            )
            delta_oz = torch.zeros_like(delta_ox)
            delta_o = torch.stack((delta_ox, delta_oy, delta_oz), -1)
            ray_o = ray_o + delta_o

        # 构造光线
        ray_d = ray_o2 - ray_o  # shape [H, W, spp, 3]
        ray = Ray(ray_o, ray_d, wvln, device=device)
        return ray

    def sample_circle(self, r, z, shape=[16, 16, 512]):
        """在恒定 z 平面的圆内均匀采样点。

        参数:
            r (float): 圆半径 [mm]。
            z (float): 所有采样点共用的 z 坐标 [mm]。
            shape (list): 点网格 shape（不含末尾坐标维度）。默认为 [16, 16, 512]。

        返回:
            points (torch.Tensor): 采样点，shape [*shape, 3]。
        """
        device = self.device

        # 生成随机角度和半径
        theta = torch.rand(*shape, device=device) * 2 * torch.pi
        r2 = torch.rand(*shape, device=device) * r**2
        radius = torch.sqrt(r2)

        # 堆叠成三维点
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        z_tensor = torch.full_like(x, z)
        points = torch.stack((x, y, z_tensor), dim=-1)

        # 手动采样主光线
        # points[..., 0, :2] = 0.0

        return points

    # ====================================================================================
    # 光线追迹
    # ====================================================================================
    def trace(self, ray, surf_range=None, record=False):
        """追迹光线通过镜头。

        根据光线 z 方向的符号自动选择正向或反向追迹。

        参数:
            ray (Ray): 要追迹的光线对象。
            surf_range (range, optional): 要通过的表面索引范围。为 None（默认）时
                追迹全部表面。
            record (bool): 为 True 时记录每个表面处的光线位置。默认为 False。

        返回:
            ray_out (Ray): 通过各表面传播后的光线。
            ray_o_record (list or None): 每个表面处记录的光线位置；record 为 False
                时返回 None。
        """
        if surf_range is None:
            surf_range = range(0, len(self.surfaces))

        if (ray.d[..., 2] > 0).any():
            ray_out, ray_o_rec = self.forward_tracing(ray, surf_range, record=record)
        else:
            ray_out, ray_o_rec = self.backward_tracing(ray, surf_range, record=record)

        return ray_out, ray_o_rec

    def trace2obj(self, ray):
        """追迹光线通过镜头并朝向物方。

        `trace` 的便捷封装，会丢弃位置记录。通常使用传感器侧（反向传播）光线调用。

        参数:
            ray (Ray): 要追迹的光线对象。

        返回:
            ray (Ray): 通过镜头传播后的光线。
        """
        ray, _ = self.trace(ray)
        return ray

    def trace2sensor(self, ray, record=False):
        """正向追迹光线通过镜头并传播到传感器平面。

        参数:
            ray (Ray): 要追迹的光线对象。
            record (bool): 为 True 时记录每个表面处的光线位置。默认为 False。

        返回:
            ray (Ray): 传播到传感器平面的光线。当 record 为 True 时返回元组
                (ray, ray_o_record)，其中 ray_o_record 是各表面处记录的光线位置
                列表（无效点设为 NaN）。
        """
        # 手动将光线传播到较浅深度，以避免数值不稳定
        if ray.o[..., 2].min() < -100.0:
            ray = ray.prop_to(-10.0)

        # 追迹光线
        ray, ray_o_record = self.trace(ray, record=record)
        ray = ray.prop_to(self.d_sensor)

        if record:
            ray_o = ray.o.clone().detach()
            # 设为 NaN，使其在二维布局可视化中被跳过
            ray_o[ray.is_valid == 0] = float("nan")
            ray_o_record.append(ray_o)
            return ray, ray_o_record
        else:
            return ray

    def trace2exit_pupil(self, ray):
        """正向追迹光线通过镜头到达出瞳平面。

        参数:
            ray (Ray): 要追迹的光线对象。

        返回:
            ray (Ray): 传播到出瞳平面的光线对象。
        """
        ray = self.trace2sensor(ray)
        pupil_z, _ = self.get_exit_pupil()
        ray = ray.prop_to(pupil_z)
        return ray

    def forward_tracing(self, ray, surf_range, record):
        """从物方到像方正向追迹光线通过指定范围内的每个表面。

        参数:
            ray (Ray): 要追迹的光线对象。
            surf_range (range): 要通过的表面索引范围。
            record (bool): 为 True 时记录每个表面处的光线位置。

        返回:
            ray_out (Ray): 通过所有表面传播后的光线。
            ray_o_record (list or None): 各表面处的光线位置；record 为 False 时
                返回 None。
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        mat1 = Material("air")
        for i in surf_range:
            n1 = mat1.ior(ray.wvln)
            n2 = self.surfaces[i].mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = self.surfaces[i].mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    def backward_tracing(self, ray, surf_range, record):
        """从像方到物方，按逆序反向追迹光线通过每个表面。

        参数:
            ray (Ray): 要追迹的光线对象。
            surf_range (range): 要通过的表面索引范围。
            record (bool): 为 True 时记录每个表面处的光线位置。

        返回:
            ray_out (Ray): 反向通过所有表面传播后的光线。
            ray_o_record (list or None): 各表面处的光线位置；record 为 False 时
                返回 None。
        """
        if record:
            ray_o_record = []
            ray_o_record.append(ray.o.clone().detach())
        else:
            ray_o_record = None

        surf_indices = list(surf_range)
        mat1 = self.surfaces[surf_indices[-1]].mat2 if surf_indices else Material("air")
        for i in reversed(surf_indices):
            n1 = mat1.ior(ray.wvln)
            mat2 = Material("air") if i == 0 else self.surfaces[i - 1].mat2
            n2 = mat2.ior(ray.wvln)
            ray = self.surfaces[i].ray_reaction(ray, n1, n2)
            mat1 = mat2

            if record:
                ray_out_o = ray.o.clone().detach()
                ray_out_o[ray.is_valid == 0] = float("nan")
                ray_o_record.append(ray_out_o)

        return ray, ray_o_record

    # ====================================================================================
    # 图像仿真
    # ====================================================================================
    def render(self, img_obj, depth=None, method=None, **kwargs):
        """可微图像仿真。

        图像仿真方法：
            [1] PSF 图分块卷积。
            [2] PSF 图块卷积。
            [3] 光线追迹渲染。

        参数:
            img_obj (torch.Tensor): raw 空间中的输入图像对象，shape [N, C, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            method (str, optional): 图像仿真方法，可为 'psf_map'、'psf_patch' 或
                'ray_tracing'。为 None（默认）时使用 `self._default_render_method`
                （`GeoLens` 为 'ray_tracing'）。
            **kwargs: 各方法的附加参数：
                - psf_grid (tuple)：PSF 图方法的网格尺寸。默认为 (10, 10)。
                - psf_ks (int)：PSF 方法的核尺寸。默认为 PSF_KS。
                - psf_spp (int)：PSF 图方法中每个 PSF 的光线数。默认为 SPP_PSF。
                - warp_grid (int)：PSF 图方法的逆畸变网格分辨率。默认为 128。
                - patch_center (tuple)：PSF 图块方法的中心位置。默认为 (0.0, 0.0)。
                - spp (int)：光线追迹的每像素采样数。默认为 SPP_RENDER。

        返回:
            img_render (torch.Tensor): 渲染图像张量，shape [N, C, H, W]。
        """
        method = self._default_render_method if method is None else method
        depth = self.obj_depth if depth is None else depth
        B, C, Himg, Wimg = img_obj.shape
        Wsensor, Hsensor = self.sensor_res

        # 图像仿真
        if method == "psf_map":
            # PSF 渲染——使用 PSF 图渲染图像
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            psf_grid = kwargs.get("psf_grid", (10, 10))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            psf_spp = kwargs.get("psf_spp", SPP_PSF)
            warp_grid = kwargs.get("warp_grid", 128)
            img_obj = self.warp(img_obj, depth=depth, num_grid=warp_grid)
            img_render = self.render_psf_map(
                img_obj,
                depth=depth,
                psf_grid=psf_grid,
                psf_ks=psf_ks,
                psf_spp=psf_spp,
            )

        elif method == "psf_patch":
            # PSF 图块渲染——使用单个 PSF 渲染图像图块
            patch_center = kwargs.get("patch_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_patch(
                img_obj, depth=depth, patch_center=patch_center, psf_ks=psf_ks
            )

        elif method == "ray_tracing":
            # 光线追迹渲染
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            spp = kwargs.get("spp", SPP_RENDER)
            img_render = self.render_raytracing(img_obj, depth=depth, spp=spp)

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_raytracing(self, img, depth=None, spp=SPP_RENDER, vignetting=False):
        """使用光线追迹渲染 RGB 图像。

        参数:
            img (torch.Tensor): RGB 图像张量，shape [N, 3, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            spp (int, optional): 每像素采样数。默认为 SPP_RENDER。
            vignetting (bool, optional): 是否建模渐晕效应。默认为 False。

        返回:
            img_render (torch.Tensor): 渲染后的 RGB 图像张量，shape [N, 3, H, W]。
        """
        depth = self.obj_depth if depth is None else depth
        img_render = torch.zeros_like(img)
        for i in range(3):
            img_render[:, i, :, :] = self.render_raytracing_mono(
                img=img[:, i, :, :],
                wvln=self.wvln_rgb[i],
                depth=depth,
                spp=spp,
                vignetting=vignetting,
            )
        return img_render

    def render_raytracing_mono(self, img, wvln, depth=None, spp=64, vignetting=False):
        """使用光线追迹渲染单一波长的单色图像。

        参数:
            img (torch.Tensor): 单色图像张量，shape [N, 1, H, W] 或 [N, H, W]。
            wvln (float): 波长 [µm]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            spp (int, optional): 每像素采样数。默认为 64。
            vignetting (bool, optional): 是否建模渐晕效应。默认为 False。

        返回:
            img_mono (torch.Tensor): 渲染后的单色图像张量，shape [N, 1, H, W]
                或 [N, H, W]。
        """
        depth = self.obj_depth if depth is None else depth
        img = torch.flip(img, [-2, -1])
        scale = self.calc_scale(depth=depth)
        ray = self.sample_sensor(spp=spp, wvln=wvln)
        ray = self.trace2obj(ray)
        img_mono = self.render_compute_image(
            img, depth=depth, scale=scale, ray=ray, vignetting=vignetting
        )
        return img_mono

    def render_compute_image(self, img, depth, scale, ray, vignetting=False):
        """计算光线与像平面的交点，并将其积分为渲染图像。

        将已追迹光线传播到物平面，使其与缩放后的物体图像相交，并依据渲染方程
        累积辐亮度。反向传播梯度流：image -> w_i -> u -> p -> ray -> surface。

        参数:
            img (torch.Tensor): 物体图像张量，shape [N, C, H, W] 或 [N, H, W]。
            depth (float): 物体深度 [mm]。
            scale (float): 物像缩放因子。
            ray (Ray): 已追迹的传感器光线，shape [H, W, spp, 3]。
            vignetting (bool): 是否建模渐晕效应。默认为 False。

        返回:
            image (torch.Tensor): 渲染图像张量，shape [N, C, H, W] 或 [N, H, W]。
        """
        assert torch.is_tensor(img), "Input image should be Tensor."

        H, W = img.shape[-2:]
        squeeze_channel = False
        if len(img.shape) == 3:
            img = img.unsqueeze(1)
            squeeze_channel = True
        elif len(img.shape) == 4:
            pass
        else:
            raise ValueError("Input image should be [N, C, H, W] or [N, H, W] tensor.")

        # 缩放物体图像的物理尺寸，使其与传感器图像实现 1:1 像素对齐
        ray = ray.prop_to(depth)
        p = ray.o[..., :2]
        pixel_size = scale * self.pixel_size
        ray.is_valid = (
            ray.is_valid
            * (torch.abs(p[..., 0] / pixel_size) < (W / 2 + 1))
            * (torch.abs(p[..., 1] / pixel_size) < (H / 2 + 1))
        )

        image = backward_integral(
            ray=ray,
            img_obj=img,
            ps=pixel_size,
            vignetting=vignetting,
        )
        if squeeze_channel:
            image = image.squeeze(1)

        return image

    def warp(self, img, depth=None, num_grid=128):
        """使用逆畸变映射为图像施加镜头畸变。

        参数:
            img (torch.Tensor): 无畸变图像张量，shape [B, C, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            num_grid (int or tuple): 逆畸变网格的分辨率。

        返回:
            img_warped (torch.Tensor): 畸变图像张量，shape ``[B, C, H, W]``。
        """
        depth = self.obj_depth if depth is None else depth
        inv_distortion_map = self.calc_inv_distortion_map(
            depth=depth, num_grid=num_grid
        )
        inv_distortion_map = inv_distortion_map.permute(2, 0, 1).unsqueeze(0)
        inv_distortion_map = F.interpolate(
            inv_distortion_map, img.shape[-2:], mode="bilinear", align_corners=True
        )
        inv_distortion_map = inv_distortion_map.permute(0, 2, 3, 1).repeat(
            img.shape[0], 1, 1, 1
        )
        img_warped = F.grid_sample(img, inv_distortion_map, align_corners=True)
        return img_warped

    def unwarp(self, img, depth=None, num_grid=128, crop=True, flip=True):
        """使用畸变图对渲染图像进行去变形（移除畸变）。

        参数:
            img (torch.Tensor): 渲染图像张量，shape [N, C, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            num_grid (int, optional): 畸变网格分辨率。默认为 128。
            crop (bool, optional): 是否裁剪图像。默认为 True。
            flip (bool, optional): 是否翻转畸变图。默认为 True。

        返回:
            img_unwarpped (torch.Tensor): 去变形后的图像张量，shape [N, C, H, W]。
        """
        depth = self.obj_depth if depth is None else depth
        # 计算畸变图，shape (num_grid, num_grid, 2)
        distortion_map = self.calc_distortion_map(depth=depth, num_grid=num_grid)

        # 将畸变图插值到图像分辨率
        distortion_map = distortion_map.permute(2, 0, 1).unsqueeze(1)
        # distortion_map = torch.flip(distortion_map, [-2]) if flip else distortion_map
        distortion_map = F.interpolate(
            distortion_map, img.shape[-2:], mode="bilinear", align_corners=True
        )  # shape (B, 2, Himg, Wimg)
        distortion_map = distortion_map.permute(1, 2, 3, 0).repeat(
            img.shape[0], 1, 1, 1
        )  # shape (B, Himg, Wimg, 2)

        # 使用 grid_sample 函数去变形
        img_unwarpped = F.grid_sample(
            img, distortion_map, align_corners=True
        )  # shape (B, C, Himg, Wimg)
        return img_unwarpped

    # ====================================================================================
    # 几何光学计算
    # ====================================================================================

    @torch.no_grad()
    def calc_foclen(self, paraxial_fov=0.01):
        """计算有效焦距（EFL）。

        分两步进行：
        1. 追迹轴上平行光线以找到近轴焦点 z。之所以需要此步骤，是因为传感器
           可能不在焦平面上（例如有限共轭设计或离焦系统）。
        2. 追迹以小角度射向焦点的离轴光线，测量像高，并计算
           EFL = imgh / tan(angle)。

        参数:
            paraxial_fov (float, optional): 离轴光线追迹使用的近轴视场 [radians]。
                默认为 0.01。

        返回:
            eff_foclen (float): 有效焦距 [mm]。

        说明:
            同时缓存 `self.efl`（有效焦距 [mm]）、`self.foclen`（`self.efl` 的别名）
            和 `self.bfl`（后焦距，即从最后一个表面到传感器的距离 [mm]）。

        参考:
            [1] https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/10/Tutorial_MorelSophie.pdf
            [2] https://rafcamera.com/info/imaging-theory/back-focal-length
        """
        # 追迹近轴主光线，shape [1, 1, num_rays, 3]
        paraxial_fov_deg = float(np.rad2deg(paraxial_fov))

        # 1. 追迹轴上平行光线，找到近轴焦点 z（等价于无穷远对焦）
        ray_axis = self.sample_from_fov(
            fov_x=0.0, fov_y=0.0, entrance_pupil=False, scale_pupil=0.2
        )
        ray_axis, _ = self.trace(ray_axis)
        valid_axis = ray_axis.is_valid > 0
        t = -(ray_axis.d[valid_axis, 0] * ray_axis.o[valid_axis, 0]
              + ray_axis.d[valid_axis, 1] * ray_axis.o[valid_axis, 1]) / (
            ray_axis.d[valid_axis, 0] ** 2 + ray_axis.d[valid_axis, 1] ** 2
        )
        focus_z = ray_axis.o[valid_axis, 2] + t * ray_axis.d[valid_axis, 2]
        focus_z = focus_z[~torch.isnan(focus_z) & (focus_z > 0)]
        if focus_z.numel() == 0:
            # 明确报错，避免将 NaN（空集均值）写入 self.foclen，否则会悄然影响
            # 下游的 calc_fov/calc_scale/set_fnum。
            raise ValueError(
                "calc_foclen: no axial rays converged to a positive focus; the "
                "lens may be degenerate or heavily vignetted."
            )
        paraxial_focus_z = float(torch.mean(focus_z))

        # 2. 将离轴近轴光线追迹到近轴焦点，并测量像高
        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=paraxial_fov_deg, entrance_pupil=False, scale_pupil=0.2
        )
        ray, _ = self.trace(ray)
        ray = ray.prop_to(paraxial_focus_z)

        # 计算有效焦距
        valid_sum = ray.is_valid.sum()
        if valid_sum.item() == 0:
            raise ValueError(
                "calc_foclen: no valid off-axis rays reached the paraxial focal "
                "plane; cannot compute the effective focal length."
            )
        paraxial_imgh = (ray.o[:, 1] * ray.is_valid).sum() / valid_sum
        eff_foclen = paraxial_imgh.item() / float(np.tan(paraxial_fov))
        self.efl = eff_foclen
        self.foclen = eff_foclen

        # 计算后焦距
        self.bfl = self.d_sensor.item() - self.surfaces[-1].d.item()

        return eff_foclen

    @torch.no_grad()
    def calc_numerical_aperture(self, n=1.0):
        """计算数值孔径（NA）。

        参数:
            n (float, optional): 折射率。默认为 1.0。

        返回:
            NA (float): 数值孔径。

        参考:
            [1] https://en.wikipedia.org/wiki/Numerical_aperture
        """
        return n * math.sin(math.atan(1 / 2 / self.fnum))
        # return n / (2 * self.fnum)

    @torch.no_grad()
    def calc_focal_plane(self, wvln=None):
        """计算物方对焦距离。光线从传感器中心出发并追迹到物方。

        参数:
            wvln (float, optional): 波长 [µm]。为 ``None``（默认）时使用
                ``self.primary_wvln``。

        返回:
            focal_plane (float): 物方对焦距离 [mm]（负 z，位于镜头前方）。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        device = self.device

        # 从传感器中心采样点光源光线
        o1 = torch.zeros(SPP_CALC, 3, device=device)
        o1[:, 2] = self.d_sensor

        # 将第一表面作为瞳孔进行采样
        # o2 = self.sample_circle(self.surfaces[0].r, z=0.0, shape=[SPP_CALC])
        # o2 *= 0.5  # 缩小采样区域以提高精度
        pupilz, pupilr = self.get_exit_pupil()
        o2 = self.sample_circle(pupilr, pupilz, shape=[SPP_CALC])
        d = o2 - o1
        ray = Ray(o1, d, wvln, device=device)

        # 将光线追迹到物方
        ray = self.trace2obj(ray)

        # 光轴交点
        t = (ray.d[..., 0] * ray.o[..., 0] + ray.d[..., 1] * ray.o[..., 1]) / (
            ray.d[..., 0] ** 2 + ray.d[..., 1] ** 2
        )
        focus_z = (ray.o[..., 2] - ray.d[..., 2] * t)[ray.is_valid > 0].cpu().numpy()
        focus_z = focus_z[~np.isnan(focus_z) & (focus_z < 0)]

        if len(focus_z) > 0:
            focal_plane = float(np.mean(focus_z))
        else:
            raise ValueError(
                "No valid rays found, focal plane in the image space cannot be computed."
            )

        return focal_plane

    @torch.no_grad()
    def calc_sensor_plane(self, depth=float("inf")):
        """计算合焦传感器平面。

        参数:
            depth (float, optional): 物平面深度。默认为 float("inf")。

        返回:
            d_sensor (torch.Tensor): 像方合焦传感器 z 位置 [mm]（标量张量）。
        """
        # 采样并追迹光线，shape [SPP_CALC, 3]
        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=0.0, depth=depth, num_rays=SPP_CALC
        )
        ray = self.trace2sensor(ray)

        # 计算合焦传感器位置
        t = (ray.d[:, 0] * ray.o[:, 0] + ray.d[:, 1] * ray.o[:, 1]) / (
            ray.d[:, 0] ** 2 + ray.d[:, 1] ** 2
        )
        focus_z = ray.o[:, 2] - ray.d[:, 2] * t
        focus_z = focus_z[ray.is_valid > 0]
        focus_z = focus_z[~torch.isnan(focus_z) & (focus_z > 0)]
        d_sensor = torch.mean(focus_z)
        return d_sensor

    @torch.no_grad()
    def calc_fov(self):
        """计算镜头视场（FoV），单位为 radians。

        使用两种方法计算 FoV：
            1. **透视投影**——根据焦距和传感器尺寸计算（忽略畸变的有效 FoV）。
            2. **正向光线追迹**——从物方扫描 FOV 角，追迹到传感器，并寻找质心像高
               与传感器半对角线匹配的角度。这避免了旧反向追迹方法在广角镜头上
               因全视场瞳孔像差导致有效光线为零的问题。

        说明:
            缓存以下属性（所有 FoV 值均以 radians 表示）：`self.vfov`（垂直 FoV）、
            `self.hfov`（水平 FoV）、`self.dfov`（对角 FoV）、`self.rfov_eff`
            （忽略畸变的有效近轴半对角 FoV）、`self.rfov`（光线追迹得到并考虑畸变
            的实际半对角 FoV）、`self.real_dfov`（光线追迹得到的实际对角 FoV）
            以及 `self.eqfl`（35 mm 等效焦距 [mm]）。

        参考:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        # 1. 透视投影（有效 FoV）
        self.vfov = 2 * math.atan(self.sensor_size[0] / 2 / self.foclen)
        self.hfov = 2 * math.atan(self.sensor_size[1] / 2 / self.foclen)
        self.dfov = 2 * math.atan(self.r_sensor / self.foclen)
        self.rfov_eff = self.dfov / 2  # 有效（近轴）半对角 FoV

        # 2. 通过正向光线追迹计算实际 FoV（受畸变影响）
        # 从物方扫描 FOV 角，追迹到传感器，并找到产生与 r_sensor 匹配像高的角度。
        num_fov = 64
        fov_lo = float(np.rad2deg(self.rfov_eff)) * 0.5
        fov_hi = min(float(np.rad2deg(self.rfov_eff)) * 1.8, 89.0)
        fov_samples = torch.linspace(fov_lo, fov_hi, num_fov, device=self.device)

        ray = self.sample_from_fov(
            fov_x=0.0, fov_y=fov_samples.tolist(), num_rays=256
        )
        ray = self.trace2sensor(ray)

        # 每个 FOV 角的质心像高，shape [num_fov]
        valid = ray.is_valid > 0  # [num_fov, num_rays]
        masked_y = ray.o[..., 1] * valid
        n_valid = valid.sum(dim=-1).clamp(min=1)
        imgh = (masked_y.sum(dim=-1) / n_valid).abs()

        # 找到像高最接近 r_sensor 的 FOV 角
        has_valid = valid.sum(dim=-1) > 10
        if has_valid.any():
            imgh[~has_valid] = float("inf")
            diff = (imgh - self.r_sensor).abs()
            best_idx = diff.argmin().item()
            rfov = fov_samples[best_idx].item() * math.pi / 180.0
            self.rfov = rfov
            self.real_dfov = 2 * rfov
        else:
            self.rfov = self.rfov_eff
            self.real_dfov = self.dfov

        # 3. 计算 35mm 等效焦距。35mm 传感器：36mm * 24mm
        self.eqfl = 21.63 / math.tan(self.rfov_eff)

    @torch.no_grad()
    def calc_scale(self, depth):
        """计算缩放因子（物高/像高）。

        使用针孔相机模型计算放大倍率。

        参数:
            depth (float): 物体到镜头的距离（负 z 方向）。

        返回:
            scale (float): 关联物高与像高的缩放因子。
        """
        return -depth / self.foclen

    @torch.no_grad()
    def calc_pupil(self):
        """计算入瞳和出瞳的位置及半径。

        以下情况必须重新计算入瞳和出瞳：
            - 一阶参数变化（例如视场、物高、像高）；
            - 镜头几何结构或材料变化（例如表面曲率、折射率、厚度）；
            - 通常而言，每当镜头配置发生修改时。

        说明:
            缓存 `self.aper_idx`（孔径表面索引）、`self.exit_pupilz`/
            `self.exit_pupilr`（实际出瞳位置和半径 [mm]）、`self.entr_pupilz`/
            `self.entr_pupilr`（实际入瞳位置和半径 [mm]）、
            `self.exit_pupilz_parax`/`self.exit_pupilr_parax` 与
            `self.entr_pupilz_parax`/`self.entr_pupilr_parax`（近轴瞳孔），
            以及 `self.fnum`（根据焦距和入瞳计算的 F-number）。
        """
        # 查找孔径
        self.aper_idx = None
        for i in range(len(self.surfaces)):
            if getattr(self.surfaces[i], "is_aperture", False):
                self.aper_idx = i
                break

        if self.aper_idx is None:
            for i in range(len(self.surfaces)):
                if isinstance(self.surfaces[i], Aperture):
                    self.aper_idx = i
                    break

        if self.aper_idx is None:
            self.aper_idx = np.argmin([s.r for s in self.surfaces])
            print("No aperture found, use the smallest surface as aperture.")

        # 计算入瞳和出瞳
        self.exit_pupilz, self.exit_pupilr = self.calc_exit_pupil(paraxial=False)
        self.entr_pupilz, self.entr_pupilr = self.calc_entrance_pupil(paraxial=False)
        self.exit_pupilz_parax, self.exit_pupilr_parax = self.calc_exit_pupil(
            paraxial=True
        )
        self.entr_pupilz_parax, self.entr_pupilr_parax = self.calc_entrance_pupil(
            paraxial=True
        )

        # 计算 F-number
        self.fnum = self.foclen / (2 * self.entr_pupilr)

    def get_entrance_pupil(self, paraxial=False):
        """获取入瞳位置和半径。

        参数:
            paraxial (bool, optional): 为 True 时返回近轴近似值；为 False 时返回
                实际光线追迹值。默认为 False。

        返回:
            pupilz (float): 入瞳 z 位置 [mm]。
            pupilr (float): 入瞳半径 [mm]。
        """
        if paraxial:
            return self.entr_pupilz_parax, self.entr_pupilr_parax
        else:
            return self.entr_pupilz, self.entr_pupilr

    def get_exit_pupil(self, paraxial=False):
        """获取出瞳位置和半径。

        参数:
            paraxial (bool, optional): 为 True 时返回近轴近似值；为 False 时返回
                实际光线追迹值。默认为 False。

        返回:
            pupilz (float): 出瞳 z 位置 [mm]。
            pupilr (float): 出瞳半径 [mm]。
        """
        if paraxial:
            return self.exit_pupilz_parax, self.exit_pupilr_parax
        else:
            return self.exit_pupilz, self.exit_pupilr

    @torch.no_grad()
    def calc_exit_pupil(self, paraxial=False):
        """计算出瞳位置和半径。

        近轴模式：
            光线从孔径光阑中心附近发出，并靠近光轴。此模式在理想（一阶）光学
            假设下估计出瞳位置和半径，快速且稳定。

        非近轴模式：
            从孔径光阑边缘大量发射光线，并根据这些光线的交点确定出瞳位置和半径。
            此模式更慢，并受孔径相关像差影响。

        除非需要精确的光线瞄准，否则请使用近轴模式。

        参数:
            paraxial (bool): 中心（True）或边缘（False）。

        返回:
            avg_pupilz (float): 出瞳 z 坐标。
            avg_pupilr (float): 出瞳半径。

        参考:
            [1] 出瞳：可从传感器进入物方的光线范围。
            [2] https://en.wikipedia.org/wiki/Exit_pupil
        """
        if self.aper_idx is None or hasattr(self, "aper_idx") is False:
            print("No aperture, use the last surface as exit pupil.")
            return self.surfaces[-1].d.item(), self.surfaces[-1].r

        # 从孔径（边缘或中心）采样光线
        aper_idx = self.aper_idx
        aper_z = self.surfaces[aper_idx].d.item()
        aper_r = self.surfaces[aper_idx].r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]], device=self.device).repeat(32, 1)
            phi_rad = torch.linspace(-0.01, 0.01, 32, device=self.device)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]], device=self.device).repeat(128, 1)  # 瞳孔光线扇尺寸
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi_rad = torch.linspace(-rfov / 2, rfov / 2, 128, device=self.device)

        d = torch.stack(
            (torch.sin(phi_rad), torch.zeros_like(phi_rad), torch.cos(phi_rad)), axis=-1
        )
        ray = Ray(ray_o, d, wvln=self.primary_wvln, device=self.device)

        # 从孔径边缘追迹光线到最后一个表面
        surf_range = range(self.aper_idx + 1, len(self.surfaces))
        ray, _ = self.trace(ray, surf_range=surf_range)

        # 计算交点，求解方程：o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid != 0][:, 0], ray.o[ray.is_valid != 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid != 0][:, 0], ray.d[ray.is_valid != 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # 处理未找到交点或瞳孔过小的情况
        if len(intersection_points) == 0:
            print("No intersection points found, use the last surface as exit pupil.")
            avg_pupilr = self.surfaces[-1].r
            avg_pupilz = self.surfaces[-1].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative exit pupil is detected, use the last surface as pupil."
                )
                avg_pupilr = self.surfaces[-1].r
                avg_pupilz = self.surfaces[-1].d.item()

        return avg_pupilz, avg_pupilr

    @torch.no_grad()
    def calc_entrance_pupil(self, paraxial=False):
        """计算镜头的入瞳。

        入瞳是从光阑前方光学元件观察到的物理孔径光阑的光学像。我们从孔径光阑
        采样反向光线，将其追迹到第一表面，再求这些光线反向延长线的交点。交点
        的平均值定义入瞳位置和半径。

        参数:
            paraxial (bool): 光线采样模式。为 ``True`` 时，光线从孔径光阑中心
                附近发出（快速且近轴稳定）；为 ``False`` 时，从光阑边缘发射更多
                光线（较慢，但会考虑孔径像差）。默认为 ``False``。

        返回:
            avg_pupilz (float): 入瞳 z 位置 [mm]。
            avg_pupilr (float): 入瞳半径 [mm]。

        说明:
            [1] 除非需要精确的光线瞄准，否则请使用近轴模式。
            [2] 此函数仅适用于远距离物体。对显微镜而言，本函数通常返回负的入瞳位置。

        参考:
            [1] 入瞳：可从物方进入传感器的光线范围。
            [2] https://en.wikipedia.org/wiki/Entrance_pupil：“在光学系统中，入瞳是
                从光阑前方的光学元件‘观察’到的物理孔径光阑的光学像。”
            [3] Zemax LLC, *OpticStudio User Manual*, Version 19.4, Document No. 2311, 2019.
        """
        if self.aper_idx is None or not hasattr(self, "aper_idx"):
            print("No aperture stop, use the first surface as entrance pupil.")
            return self.surfaces[0].d.item(), self.surfaces[0].r

        # 从孔径光阑边缘采样光线
        aper_idx = self.aper_idx
        aper_surf = self.surfaces[aper_idx]
        aper_z = aper_surf.d.item()
        if aper_surf.is_square:
            aper_r = float(np.sqrt(2)) * aper_surf.r
        else:
            aper_r = aper_surf.r

        if paraxial:
            ray_o = torch.tensor([[DELTA_PARAXIAL, 0, aper_z]], device=self.device).repeat(32, 1)
            phi = torch.linspace(-0.01, 0.01, 32, device=self.device)
        else:
            ray_o = torch.tensor([[aper_r, 0, aper_z]], device=self.device).repeat(128, 1)  # 瞳孔光线扇尺寸
            rfov = float(np.arctan(self.r_sensor / self.foclen))
            phi = torch.linspace(-rfov / 2, rfov / 2, 128, device=self.device)

        d = torch.stack(
            (torch.sin(phi), torch.zeros_like(phi), -torch.cos(phi)), axis=-1
        )
        ray = Ray(ray_o, d, wvln=self.primary_wvln, device=self.device)

        # 从孔径边缘追迹光线到第一表面
        surf_range = range(0, self.aper_idx)
        ray, _ = self.trace(ray, surf_range=surf_range)

        # 计算交点，求解方程：o1+d1*t1 = o2+d2*t2
        ray_o = torch.stack(
            [ray.o[ray.is_valid > 0][:, 0], ray.o[ray.is_valid > 0][:, 2]], dim=-1
        )
        ray_d = torch.stack(
            [ray.d[ray.is_valid > 0][:, 0], ray.d[ray.is_valid > 0][:, 2]], dim=-1
        )
        intersection_points = self.compute_intersection_points_2d(ray_o, ray_d)

        # 处理未找到交点或入瞳过小的情况
        if len(intersection_points) == 0:
            print(
                "No intersection points found, use the first surface as entrance pupil."
            )
            avg_pupilr = self.surfaces[0].r
            avg_pupilz = self.surfaces[0].d.item()
        else:
            avg_pupilr = torch.mean(intersection_points[:, 0]).item()
            avg_pupilz = torch.mean(intersection_points[:, 1]).item()

            if paraxial:
                avg_pupilr = abs(avg_pupilr / DELTA_PARAXIAL * aper_r)

            if avg_pupilr < EPSILON:
                print(
                    "Zero or negative entrance pupil is detected, use the first surface as entrance pupil."
                )
                avg_pupilr = self.surfaces[0].r
                avg_pupilz = self.surfaces[0].d.item()

        return avg_pupilz, avg_pupilr

    @staticmethod
    def compute_intersection_points_2d(origins, directions):
        """计算二维直线的交点。

        参数:
            origins (torch.Tensor): 直线原点，shape [N, 2]。
            directions (torch.Tensor): 直线方向，shape [N, 2]。

        返回:
            points (torch.Tensor): 交点，shape [N*(N-1)/2, 2]。
        """
        N = origins.shape[0]

        # 创建索引的两两组合
        idx = torch.arange(N)
        idx_i, idx_j = torch.combinations(idx, r=2).unbind(1)

        Oi = origins[idx_i]  # shape [N*(N-1)/2, 2]
        Oj = origins[idx_j]  # shape [N*(N-1)/2, 2]
        Di = directions[idx_i]  # shape [N*(N-1)/2, 2]
        Dj = directions[idx_j]  # shape [N*(N-1)/2, 2]

        # 从 Oi 到 Oj 的向量
        b = Oj - Oi  # shape [N*(N-1)/2, 2]

        # 系数矩阵 A
        A = torch.stack([Di, -Dj], dim=-1)  # shape [N*(N-1)/2, 2, 2]

        # 求解线性方程组 Ax = b
        # 使用最小二乘处理无精确解的情况
        if A.device.type == "mps":
            # 对 MPS 设备在 CPU 上执行 lstsq，再将结果移回
            x, _ = torch.linalg.lstsq(A.cpu(), b.unsqueeze(-1).cpu())[:2]
            x = x.to(A.device)
        else:
            x, _ = torch.linalg.lstsq(A, b.unsqueeze(-1))[:2]
        x = x.squeeze(-1)  # shape [N*(N-1)/2, 2]
        s = x[:, 0]
        t = x[:, 1]

        # 分别使用两条光线计算交点
        P_i = Oi + s.unsqueeze(-1) * Di  # shape [N*(N-1)/2, 2]
        P_j = Oj + t.unsqueeze(-1) * Dj  # shape [N*(N-1)/2, 2]

        # 取平均以减轻数值精度问题
        P = (P_i + P_j) / 2

        return P

    # ====================================================================================
    # 镜头操作
    # ====================================================================================
    @torch.no_grad()
    def refocus(self, foc_dist=float("inf")):
        """通过改变传感器位置，将镜头重新对焦到指定深度。

        参数:
            foc_dist (float, optional): 物体对焦距离 [mm]。无穷远对焦使用
                ``float('inf')``。默认为 ``float('inf')``。

        说明:
            在 DSLR 中，相位检测自动对焦（PDAF）是一种常用且高效的方法；此处通过
            计算绿光的合焦位置来简化该问题。
        """
        # 计算合焦传感器位置
        d_sensor_new = self.calc_sensor_plane(depth=foc_dist)

        # 更新传感器位置
        assert d_sensor_new > 0, "Obtained negative sensor position."
        self.d_sensor = d_sensor_new

        # FoV 会略有变化
        self.post_computation()

    @torch.no_grad()
    def set_fnum(self, fnum):
        """使用二分搜索设置 F-number 和孔径半径。

        参数:
            fnum (float): 目标 F-number。
        """
        target_pupil_r = self.foclen / fnum / 2
        aper_r = self.surfaces[self.aper_idx].r
        lo, hi = 0.1 * aper_r, 5.0 * aper_r

        pupilr = None
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            self.surfaces[self.aper_idx].update_r(float(mid))
            _, pupilr = self.calc_entrance_pupil()
            if abs(pupilr - target_pupil_r) / target_pupil_r < 1e-3:
                break
            if pupilr > target_pupil_r:
                hi = mid
            else:
                lo = mid
        else:
            logging.warning(
                f"set_fnum: did not converge, pupil_r={pupilr:.4f}, target={target_pupil_r:.4f}"
            )

        self.calc_pupil()

    @torch.no_grad()
    def set_target_fov_fnum(self, rfov, fnum):
        """将 FoV、像高和 F-number 设置为设计目标。

        此方法仅用于分配设计目标（它会直接覆盖缓存的一阶量，而不是测量它们）。

        参数:
            rfov (float): 半对角 FoV。按 radians 解释；若值大于 $\\pi$，则视为
                degrees 并转换为 radians。
            fnum (float): 目标 F-number。
        """
        if rfov > math.pi:
            self.rfov_eff = rfov / 180.0 * math.pi
        else:
            self.rfov_eff = rfov

        self.rfov = self.rfov_eff
        self.real_dfov = 2 * self.rfov
        self.foclen = self.r_sensor / math.tan(self.rfov_eff)
        self.eqfl = 21.63 / math.tan(self.rfov_eff)
        self.fnum = fnum
        aper_r = self.foclen / fnum / 2
        self.surfaces[self.aper_idx].update_r(float(aper_r))

        # 设置孔径半径后更新瞳孔
        self.calc_pupil()

    @torch.no_grad()
    def set_fov(self, rfov):
        """将半对角视场设置为设计目标。

        ``calc_fov()`` 会根据焦距和传感器尺寸推导 FoV，而此方法直接为镜头优化
        指定目标 FoV。

        参数:
            rfov (float): 半对角 FoV [radians]。
        """
        self.rfov_eff = rfov
        self.rfov = rfov
        self.real_dfov = 2 * self.rfov
        self.eqfl = 21.63 / math.tan(self.rfov_eff)
