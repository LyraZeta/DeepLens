# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""光学镜头基类。创建新镜头（geolens、diffractivelens 等）时，应继承 Lens 类
并重写核心函数。"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from deeplens import init_device
from .config import (
    DEFAULT_WAVE,
    DEPTH,
    EPSILON,
    PSF_KS,
    SPP_PSF,
    WAVE_RGB,
)
from .base import DeepObj
from .imgsim import (
    conv_psf,
    conv_psf_depth_interp,
    conv_psf_map,
    conv_psf_map_depth_interp,
    splat_psf_per_pixel,
)


class Lens(DeepObj):
    """DeepLens 中所有镜头模型的抽象基类。

    `Lens` 定义由 `GeoLens`、`HybridLens`、`DiffractiveLens`、`PSFNetLens`
    和 `DefocusLens` 共同继承的接口，包括 PSF 计算（`psf`、`psf_rgb`）、图像
    渲染（`render`）、传感器配置和 JSON 文件 I/O。子类使用各自的可微实现覆盖
    核心光学方法（例如 `psf`）。
    """

    # 未提供 ``method`` 时，`render()` 使用的默认图像仿真方法。将其保留为类属性
    #（而非各不相同的签名默认值），使每种镜头类型共用统一的 `render()` 签名；
    # 子类可覆盖此属性以更改默认值（例如 ``GeoLens``）。
    _default_render_method = "psf_patch"

    def __init__(
        self,
        dtype=torch.float32,
        device=None,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """初始化镜头类。

        参数:
            dtype (torch.dtype, optional): 数据类型。默认为 torch.float32。
            device (str, optional): 镜头运行设备。默认为 None。
            primary_wvln (float, optional): 主要设计波长 [µm]。调用方法时未显式
                提供 ``wvln``，则使用此值。默认为 ``DEFAULT_WAVE``（0.587，d-line）。
            wvln_rgb (sequence of float, optional): RGB（多色）计算所用的三个波长，
                按 ``[R, G, B]`` 排列，单位为 µm。默认为 ``WAVE_RGB``。
            obj_depth (float, optional): 默认物体深度 [mm]。调用方法时未显式提供
                ``depth``，则使用此值。应为负值（物体位于镜头前方）。默认为
                ``DEPTH``（−20 000 mm，实际无穷远）。
        """
        # 镜头设备
        if device is None:
            self.device = init_device()
        else:
            self.device = torch.device(device)

        # 镜头默认 dtype
        self.dtype = dtype

        primary_wvln = torch.as_tensor(primary_wvln, dtype=torch.float64)
        wvln_rgb = torch.as_tensor(wvln_rgb, dtype=torch.float64)
        obj_depth = torch.as_tensor(obj_depth, dtype=torch.float64)

        if primary_wvln.numel() != 1:
            raise ValueError("primary_wvln must be a scalar wavelength in [µm].")
        if wvln_rgb.numel() != 3:
            raise ValueError("wvln_rgb must contain exactly three wavelengths in [µm].")
        if obj_depth.numel() != 1:
            raise ValueError("obj_depth must be a scalar depth in [mm].")

        if not (primary_wvln.item() > 0.1 and primary_wvln.item() < 10.0):
            raise ValueError("primary_wvln must be in [µm] and satisfy 0.1 < primary_wvln < 10.")
        if not torch.all((wvln_rgb > 0.1) & (wvln_rgb < 10.0)):
            raise ValueError("wvln_rgb must be in [µm] and every value must satisfy 0.1 < wvln < 10.")
        if not obj_depth.item() < 0.0:
            raise ValueError("obj_depth must be negative [mm], with the object in front of the lens.")

        # 设计波长 [µm]。I/O 可能覆盖这些值。
        self.primary_wvln = float(primary_wvln.item())
        self.wvln_rgb = [float(w) for w in wvln_rgb.tolist()]

        # 默认物体深度 [mm]。
        self.obj_depth = float(obj_depth.item())

    def read_lens_json(self, filename):
        """从 JSON 文件读取镜头。必须由子类覆盖。

        参数:
            filename (str): JSON 镜头文件路径。

        异常:
            NotImplementedError: 此基础实现必须被覆盖。
        """
        raise NotImplementedError

    def write_lens_json(self, filename):
        """将镜头写入 JSON 文件。必须由子类覆盖。

        参数:
            filename (str): JSON 镜头文件的目标路径。

        异常:
            NotImplementedError: 此基础实现必须被覆盖。
        """
        raise NotImplementedError

    def set_sensor(self, sensor_size, sensor_res):
        """设置传感器尺寸和分辨率。

        参数:
            sensor_size (tuple): 传感器尺寸 (w, h) [mm]。
            sensor_res (tuple): 传感器分辨率 (W, H) [pixels]。
        """
        assert sensor_size[0] * sensor_res[1] == sensor_size[1] * sensor_res[0], (
            "Sensor resolution aspect ratio does not match sensor size aspect ratio."
        )
        self.sensor_size = sensor_size
        self.sensor_res = sensor_res
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]
        self.r_sensor = float(np.sqrt(sensor_size[0] ** 2 + sensor_size[1] ** 2)) / 2
        self.calc_fov()

    def set_sensor_res(self, sensor_res):
        """在保持传感器半径不变的情况下设置传感器分辨率（及宽高比）。

        参数:
            sensor_res (tuple): 传感器分辨率 (W, H) [pixels]。
        """
        # 更改传感器分辨率
        self.sensor_res = sensor_res

        # 更改传感器尺寸（r_sensor 保持不变）
        diam_res = float(np.sqrt(self.sensor_res[0] ** 2 + self.sensor_res[1] ** 2))
        self.sensor_size = (
            2 * self.r_sensor * self.sensor_res[0] / diam_res,
            2 * self.r_sensor * self.sensor_res[1] / diam_res,
        )
        self.pixel_size = self.sensor_size[0] / self.sensor_res[0]
        self.calc_fov()

    @torch.no_grad()
    def calc_fov(self):
        """计算镜头的 FoV [radian]。

        参考:
            [1] https://en.wikipedia.org/wiki/Angle_of_view_(photography)
        """
        if not hasattr(self, "foclen"):
            return

        self.vfov = 2 * float(np.arctan(self.sensor_size[0] / 2 / self.foclen))
        self.hfov = 2 * float(np.arctan(self.sensor_size[1] / 2 / self.foclen))
        self.dfov = 2 * float(np.arctan(self.r_sensor / self.foclen))
        self.rfov_eff = self.dfov / 2  # 有效（近轴）半对角 FoV
        self.rfov = self.rfov_eff  # 默认使用有效值；GeoLens 会用光线追迹值覆盖

    # ===========================================
    # PSF 相关函数
    # 1. 点 PSF
    # 2. PSF 图
    # 3. 径向 PSF
    # ===========================================
    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """计算一个或多个点光源的单色 PSF。

        子类必须用可微实现覆盖此方法。实际中常用三种计算模型：几何光线分箱、
        相干光线-波动模型和惠更斯球面波积分。

        参数:
            points (torch.Tensor): 点光源坐标，shape ``[N, 3]`` 或 ``[3]``。
                ``x, y`` 归一化到 ``[-1, 1]``（相对于传感器半对角线）；``z``
                为以 mm 表示的深度（必须为负，即位于镜头前方）。
            wvln (float, optional): 波长 [µm]。为 ``None``（默认）时使用
                ``self.primary_wvln``。
            ks (int, optional): 输出 PSF 核尺寸 [pixels]。默认为 ``PSF_KS``（64）。
            **kwargs: 转发给底层 PSF 计算的附加关键字参数（例如 ``spp``、``model``、
                ``recenter``）。

        返回:
            psf (torch.Tensor): PSF 强度图；单点时 shape ``[ks, ks]``，批量输入时
                shape ``[N, ks, ks]``。

        异常:
            NotImplementedError: 此基础实现必须被覆盖。

        说明:
            此方法对所有可优化镜头参数均可微，因此可直接在训练循环中使用。

        示例:
            ```python
            point = torch.tensor([0.0, 0.0, -10000.0])
            psf = lens.psf(points=point, ks=64, model="geometric")
            print(psf.shape)  # torch.Size([64, 64])
            ```
        """
        raise NotImplementedError

    def psf_rgb(self, points, ks=PSF_KS, **kwargs):
        """堆叠三个波长的调用结果，计算 RGB（三色）PSF。

        对存储在 ``self.wvln_rgb`` 中的 RGB 主波长分别调用三次 `psf`，并沿通道轴
        堆叠结果。

        参数:
            points (torch.Tensor): 点光源坐标，shape ``[N, 3]`` 或 ``[3]``。
                约定与 `psf` 相同。
            ks (int, optional): PSF 核尺寸。默认为 ``PSF_KS``。
            **kwargs: 转发给 `psf`（例如 ``spp``、``model``）。

        返回:
            psf_rgb (torch.Tensor): RGB PSF；单点时 shape ``[3, ks, ks]``，
                批量输入时 shape ``[N, 3, ks, ks]``。
        """
        psfs = []
        for wvln in self.wvln_rgb:
            psfs.append(self.psf(points=points, ks=ks, wvln=wvln, **kwargs))
        psf_rgb = torch.stack(psfs, dim=-3)  # shape [3, ks, ks] or [N, 3, ks, ks]
        return psf_rgb

    def point_source_grid(
        self, depth, grid=(9, 9), normalized=True, quater=False, center=True
    ):
        """生成用于 PSF 计算的点光源网格。

        参数:
            depth (float): 点光源深度（z 坐标）[mm]（负值，位于镜头前方）。
            grid (tuple, optional): 网格尺寸 (grid_w, grid_h)。默认为 (9, 9)，
                即 9x9 网格。
            normalized (bool, optional): 为 True 时返回 [-1, 1] 范围内的归一化物方
                xy 坐标；为 False 时缩放到物理位置 [mm]。默认为 True。
            quater (bool, optional): 为 True 时仅返回网格的四分之一以节省内存。
                默认为 False。
            center (bool, optional): 为 True 时将点置于各图块中心；否则采样到视场
                角点。默认为 True。

        返回:
            point_source (torch.Tensor): 物方光源坐标，shape [grid_h, grid_w, 3]，
                最后一维顺序为 (x, y, z)。`quater` 为 True 时，前两维缩减为返回的
                四分之一区域。
        """
        # 计算点光源网格
        if grid[0] == 1:
            x, y = torch.tensor([[0.0]], device=self.device), torch.tensor([[0.0]], device=self.device)
            assert not quater, "Quater should be False when grid is 1."
        else:
            if center:
            # 使用每个图块的中心
                half_bin_size = 1 / 2 / (grid[0] - 1)
                x, y = torch.meshgrid(
                    torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid[0], device=self.device),
                    torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid[1], device=self.device),
                    indexing="xy",
                )
            else:
            # 使用图像传感器角点
                x, y = torch.meshgrid(
                    torch.linspace(-0.98, 0.98, grid[0], device=self.device),
                    torch.linspace(0.98, -0.98, grid[1], device=self.device),
                    indexing="xy",
                )

        z = torch.full_like(x, depth)
        point_source = torch.stack([x, y, z], dim=-1)

        # 使用传感器平面的四分之一以节省内存
        if quater:
            bound_i = grid[0] // 2 if grid[0] % 2 == 0 else grid[0] // 2 + 1
            bound_j = grid[1] // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        # 将物方光源坐标反归一化到物理坐标
        if not normalized:
            scale = self.calc_scale(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source

    def psf_map(self, grid=(5, 5), wvln=None, depth=None, ks=PSF_KS, **kwargs):
        """计算单色 PSF 图。

        参数:
            grid (tuple): 网格尺寸 (grid_w, grid_h)。默认为 (5, 5)，即 5x5 网格。
            wvln (float): 波长 [µm]。为 ``None``（默认）时使用
                ``self.primary_wvln``。
            depth (float): 物体深度。为 ``None``（默认）时使用 ``self.obj_depth``。
            ks (int): 核尺寸。默认为 PSF_KS。

        返回:
            psf_map (torch.Tensor): 单色 PSF 图，shape
                [grid_h, grid_w, 1, ks, ks]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth

        # PSF 图网格
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3)

        # 计算 PSF 图
        psfs = []
        for i in range(points.shape[0]):
            point = points[i, ...]
            psf = self.psf(points=point, wvln=wvln, ks=ks)
            psfs.append(psf)
        psf_map = torch.stack(psfs).unsqueeze(1)  # shape [grid_h * grid_w, 1, ks, ks]

        # 将 PSF 图从 [grid_h * grid_w, 1, ks, ks] 变形为 [grid_h, grid_w, 1, ks, ks]
        psf_map = psf_map.reshape(grid[1], grid[0], 1, ks, ks)
        return psf_map

    def psf_map_rgb(self, grid=(5, 5), ks=PSF_KS, depth=None, **kwargs):
        """计算 RGB PSF 图。

        参数:
            grid (tuple): 网格尺寸 (grid_w, grid_h)。默认为 (5, 5)，即 5x5 网格。
            ks (int): 核尺寸。默认为 PSF_KS，即 PSF_KS x PSF_KS 核尺寸。
            depth (float): 物体深度。为 ``None``（默认）时使用 ``self.obj_depth``。
            **kwargs: 传给 psf_map() 的附加参数。

        返回:
            psf_map (torch.Tensor): shape [grid_h, grid_w, 3, ks, ks]。
        """
        depth = self.obj_depth if depth is None else depth
        psfs = []
        for wvln in self.wvln_rgb:
            psf_map = self.psf_map(grid=grid, ks=ks, depth=depth, wvln=wvln, **kwargs)
            psfs.append(psf_map)
        psf_map = torch.cat(psfs, dim=2)  # shape [grid_h, grid_w, 3, ks, ks]
        return psf_map

    @torch.no_grad()
    def draw_psf_map(
        self,
        grid=(7, 7),
        ks=PSF_KS,
        depth=None,
        log_scale=False,
        save_name="./psf_map.png",
        show=False,
    ):
        """绘制镜头的 RGB PSF 图并保存（或返回图形）。

        参数:
            grid (tuple, optional): 网格尺寸 (grid_w, grid_h)。默认为 (7, 7)。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            log_scale (bool, optional): 为 True 时以对数尺度归一化每个 PSF，以改善
                可视化效果。默认为 False。
            save_name (str, optional): 输出图像路径。默认为 "./psf_map.png"。
            show (bool, optional): 为 True 时返回 (fig, ax)，而不保存。默认为 False。

        返回:
            result (tuple or None): `show` 为 True 时返回 (fig, ax)，否则返回 None
                （图形保存到 `save_name`）。
        """
        depth = self.obj_depth if depth is None else depth
        # 计算 RGB PSF 图，shape [grid_h, grid_w, 3, ks, ks]
        psf_map = self.psf_map_rgb(depth=depth, grid=grid, ks=ks)

        # 创建网格可视化（vis_map：shape [3, grid_h * ks, grid_w * ks]）
        grid_w, grid_h = grid if isinstance(grid, tuple) else (grid, grid)
        h, w = grid_h * ks, grid_w * ks
        vis_map = torch.zeros((3, h, w), device=psf_map.device, dtype=psf_map.dtype)

        # 将每个 PSF 放入 vis_map
        for i in range(grid_h):
            for j in range(grid_w):
                # 提取此网格位置的 PSF
                psf = psf_map[i, j]  # shape [3, ks, ks]

                # 归一化 PSF
                if log_scale:
                    # 对数尺度归一化，以改善可视化效果
                    psf = torch.log(psf + 1e-4)  # 1e-4 为经验值
                    psf = (psf - psf.min()) / (psf.max() - psf.min() + 1e-8)
                else:
                    # 线性归一化
                    local_max = psf.max()
                    if local_max > 0:
                        psf = psf / local_max

                # 将归一化 PSF 放入可视化图
                y_start, y_end = i * ks, (i + 1) * ks
                x_start, x_end = j * ks, (j + 1) * ks
                vis_map[:, y_start:y_end, x_start:x_end] = psf

        # 创建并显示图形
        fig, ax = plt.subplots(figsize=(10, 10))

        # 转换为 numpy 以便绘图
        vis_map = vis_map.permute(1, 2, 0).cpu().numpy()
        ax.imshow(vis_map)

        # 在左下角附近添加比例尺
        H, W, _ = vis_map.shape
        scale_bar_length = 100
        arrow_length = scale_bar_length / (self.pixel_size * 1e3)
        y_position = H - 20  # 略高于下边缘
        x_start = 20
        x_end = x_start + arrow_length

        ax.annotate(
            "",
            xy=(x_start, y_position),
            xytext=(x_end, y_position),
            arrowprops=dict(arrowstyle="-", color="white"),
            annotation_clip=False,
        )
        ax.text(
            x_end + 5,
            y_position,
            f"{scale_bar_length} μm",
            color="white",
            fontsize=12,
            ha="left",
            va="center",
            clip_on=False,
        )

        # 清理坐标轴并保存
        ax.axis("off")
        plt.tight_layout(pad=0)

        if show:
            return fig, ax
        else:
            plt.savefig(save_name, dpi=300, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

    def point_source_radial(self, depth, grid=9, center=False, direction="diagonal", normalized=True):
        """生成从视场中心到边缘的径向点光源。

        沿选定径向方向（对角、子午或弧矢方向）生成 ``grid`` 个等间隔点，坐标可为
        归一化物方坐标或物理物方坐标。

        参数:
            depth (float): 物体深度（z 坐标）[mm]。
            grid (int): 采样点数。默认为 9。
            center (bool): 为 ``True`` 时将位置偏移到分箱中心。默认为 ``False``。
            direction (str): 采样方向——``"diagonal"``（x = y，45°，默认）、
                ``"y"``（子午方向，x = 0）、``"x"``（弧矢方向，y = 0）。
            normalized (bool): 为 ``True`` 时返回 [0, 1] 范围内的坐标；为 ``False``
                时缩放到物理物方位置 [mm]。默认为 ``True``。

        返回:
            point_source (torch.Tensor): 点光源位置，shape ``[grid, 3]``。
        """
        if grid == 1:
            r = torch.tensor([0.0], device=self.device)
        else:
        # 选择分箱中心以计算 PSF
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                r = torch.linspace(0, 1 - half_bin_size, grid, device=self.device)
            else:
                r = torch.linspace(0, 0.98, grid, device=self.device)

        # 根据方向将径向坐标映射到 (x, y)
        if direction == "diagonal":
            px, py = r, r
        elif direction == "y":
            px, py = torch.zeros_like(r), r
        elif direction == "x":
            px, py = r, torch.zeros_like(r)
        else:
            raise ValueError(f"Invalid direction: {direction!r}. Use 'diagonal', 'x', or 'y'.")

        z = torch.full_like(px, depth)
        point_source = torch.stack([px, py, z], dim=-1)

        if not normalized:
            scale = self.calc_scale(depth)
            point_source[..., 0] = point_source[..., 0] * scale * self.sensor_size[0] / 2
            point_source[..., 1] = point_source[..., 1] * scale * self.sensor_size[1] / 2

        return point_source

    @torch.no_grad()
    def draw_psf_radial(
        self, M=3, depth=None, ks=PSF_KS, log_scale=False, save_name="./psf_radial.png"
    ):
        """绘制并保存径向（45 deg，对角）RGB PSF 序列。

        从视场中心到角点等间隔绘制 M 个 PSF，每个尺寸为 ks x ks，并排列成一行。

        参数:
            M (int, optional): 要绘制的 PSF 数。默认为 3。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            log_scale (bool, optional): 为 True 时以对数尺度归一化每个 PSF，以改善
                可视化效果。默认为 False。
            save_name (str, optional): 输出图像路径。默认为 "./psf_radial.png"。
        """
        from torchvision.utils import make_grid, save_image
        depth = self.obj_depth if depth is None else depth
        x = torch.linspace(0, 1, M)
        y = torch.linspace(0, 1, M)
        z = torch.full_like(x, depth)
        points = torch.stack((x, y, z), dim=-1)

        psfs = []
        for i in range(M):
        # 缩放 PSF 以改善可视化效果
            psf = self.psf_rgb(points=points[i], ks=ks, recenter=True, spp=SPP_PSF)
            psf /= psf.max()

            if log_scale:
                psf = torch.log(psf + EPSILON)
                psf = (psf - psf.min()) / (psf.max() - psf.min())

            psfs.append(psf)

        psf_grid = make_grid(psfs, nrow=M, padding=1, pad_value=0.0)
        save_image(psf_grid, save_name, normalize=True)

    # ===========================================
    # 图像仿真相关函数
    # ===========================================

    # -------------------------------------------
    # 模拟二维场景
    # -------------------------------------------
    def render(self, img_obj, depth=None, method=None, **kwargs):
        """二维（平面）场景的可微图像仿真。

        仅执行图像仿真的光学部分，并且完全可微。

        对非相干成像，使用强度 PSF 与物方图像卷积；对相干成像，先使用复 PSF
        与复物体图像卷积，再取平方得到强度。

        参数:
            img_obj (torch.Tensor): 线性（raw）空间中的输入图像，
                shape ``[B, C, H, W]``。
            depth (float, optional): 物体深度 [mm]（负值）。为 ``None``（默认）时
                使用 ``self.obj_depth``。
            method (str, optional): 渲染方法。为 ``None``（默认）时使用
                ``self._default_render_method``（基类 `Lens` 为 ``"psf_patch"``）。
                可选方法：

                * ``"psf_patch"``——使用在 *patch_center* 处计算的单个 PSF 做卷积。
                * ``"psf_map"``——空间变化 PSF 分块卷积。

            **kwargs: 方法特定的关键字参数：

                * 对 ``"psf_map"``：``psf_grid``（tuple，默认 ``(10, 10)``）、
                  ``psf_ks``（int，默认 ``PSF_KS``）、``psf_spp``（int，默认
                  ``SPP_PSF``）。
                * 对 ``"psf_patch"``：``patch_center``（tuple 或 Tensor，默认
                  ``(0.0, 0.0)``）、``psf_ks``（int）。

        返回:
            img_render (torch.Tensor): 渲染图像，shape ``[B, C, H, W]``。

        异常:
            AssertionError: *method* 为 ``"psf_map"`` 且图像分辨率与传感器
                分辨率不匹配时抛出。
            Exception: 无法识别 *method* 时抛出。

        参考:
            [1] "Optical Aberration Correction in Postprocessing using Imaging Simulation", TOG 2021.
            [2] "Efficient depth- and spatially-varying image simulation for defocus deblur", ICCVW 2025.

        示例:
            ```python
            img_rendered = lens.render(
                img, depth=-10000.0, method="psf_patch",
                patch_center=(0.3, 0.0), psf_ks=64,
            )
            ```
        """
        method = self._default_render_method if method is None else method
        depth = self.obj_depth if depth is None else depth
        # 检查传感器分辨率
        B, C, Himg, Wimg = img_obj.shape
        Wsensor, Hsensor = self.sensor_res

        # 图像仿真（在 RAW 空间中）
        if method == "psf_map":
            # 使用 PSF 图卷积渲染全分辨率图像
            assert Wimg == Wsensor and Himg == Hsensor, (
                f"Sensor resolution {Wsensor}x{Hsensor} must match input image {Wimg}x{Himg}."
            )
            psf_grid = kwargs.get("psf_grid", (10, 10))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            psf_spp = kwargs.get("psf_spp", SPP_PSF)
            img_render = self.render_psf_map(
                img_obj,
                depth=depth,
                psf_grid=psf_grid,
                psf_ks=psf_ks,
                psf_spp=psf_spp,
            )

        elif method == "psf_patch":
            # 使用对应 PSF 渲染图像图块
            patch_center = kwargs.get("patch_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            img_render = self.render_psf_patch(
                img_obj, depth=depth, patch_center=patch_center, psf_ks=psf_ks
            )

        elif method == "psf_pixel":
            raise NotImplementedError(
                "Per-pixel PSF convolution has not been implemented."
            )

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

        return img_render

    def render_psf(self, img_obj, depth=None, patch_center=(0, 0), psf_ks=PSF_KS):
        """使用 PSF 卷积渲染图像图块（已弃用的别名）。

        `render_psf_patch` 的简单封装。为避免混淆，建议直接调用 `render_psf_patch`。

        参数:
            img_obj (torch.Tensor): raw 空间中的输入图像，shape [B, C, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            patch_center (tuple, optional): 归一化物体坐标中的图块中心 (x, y)。
                默认为 (0, 0)。
            psf_ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。

        返回:
            img_render (torch.Tensor): 渲染图像，shape [B, C, H, W]。
        """
        depth = self.obj_depth if depth is None else depth
        return self.render_psf_patch(
            img_obj, depth=depth, patch_center=patch_center, psf_ks=psf_ks
        )

    def render_psf_patch(self, img_obj, depth=None, patch_center=(0, 0), psf_ks=PSF_KS):
        """使用在图块中心计算的单个 PSF 渲染图像图块。

        在 `patch_center` 处计算 RGB PSF，并将其与输入图像卷积。图块内所有像素
        共用同一个 PSF（适用于较小且近似等晕的图块）。

        参数:
            img_obj (torch.Tensor): raw 空间中的输入图像，shape [B, C, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            patch_center (tuple or torch.Tensor): 归一化物体坐标中的图块中心
                (x, y)，shape [2] 或 [B, 2]。
            psf_ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。

        返回:
            img_render (torch.Tensor): 渲染图像，shape [B, C, H, W]。
        """
        depth = self.obj_depth if depth is None else depth
        # 将 patch_center 转换为张量
        if isinstance(patch_center, (list, tuple)):
            points = (patch_center[0], patch_center[1], depth)
            points = torch.tensor(points).unsqueeze(0)
        elif isinstance(patch_center, torch.Tensor):
            depth = torch.full_like(patch_center[..., 0], depth)
            points = torch.stack(
                [patch_center[..., 0], patch_center[..., 1], depth], dim=-1
            )
        else:
            raise Exception(
                f"Patch center must be a list or tuple or tensor, but got {type(patch_center)}."
            )

        # 计算 PSF 并执行 PSF 卷积
        psf = self.psf_rgb(points=points, ks=psf_ks).squeeze(0)
        img_render = conv_psf(img_obj, psf=psf)
        return img_render

    def render_psf_map(
        self,
        img_obj,
        depth=None,
        psf_grid=7,
        psf_ks=PSF_KS,
        psf_spp=SPP_PSF,
    ):
        """使用空间变化 PSF 分块卷积渲染全分辨率图像。

        说明:
            较大的 `psf_grid` 和 `psf_ks` 可提高渲染精度，但速度更慢。

        参数:
            img_obj (torch.Tensor): raw 空间中的输入图像，shape [B, C, H, W]。
            depth (float, optional): 物体深度 [mm]。为 None（默认）时使用
                `self.obj_depth`。
            psf_grid (int or tuple, optional): PSF 网格尺寸。默认为 7。
            psf_ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            psf_spp (int, optional): PSF 计算中每个点的采样数。默认为 SPP_PSF。

        返回:
            img_render (torch.Tensor): 渲染图像，shape [B, C, H, W]。
        """
        depth = self.obj_depth if depth is None else depth
        psf_map = self.psf_map_rgb(grid=psf_grid, ks=psf_ks, depth=depth, spp=psf_spp)
        img_render = conv_psf_map(img_obj, psf_map)
        return img_render

    # -------------------------------------------
    # 模拟三维场景
    # -------------------------------------------
    def _sample_depth_layers(self, depth_min, depth_max, num_layers):
        """在视差空间中采样以焦平面为中心的深度层。

        若镜头具有 `calc_focal_plane` 方法，则在焦平面两侧划分样本，使焦平面
        始终为显式采样点；否则退回到均匀视差采样。

        参数:
            depth_min (float): 最小（最近）深度 [mm]（正值）。
            depth_max (float): 最大（最远）深度 [mm]（正值）。
            num_layers (int): 要采样的深度层数。

        返回:
            disp_ref (torch.Tensor): 采样视差（1/depth），shape [num_layers]。
            depths_ref (torch.Tensor): 用于 PSF 计算的相应深度 [mm]，等于
                -1 / disp_ref（负值），shape [num_layers]。
        """
        # 尝试从镜头获取焦点深度
        if hasattr(self, 'calc_focal_plane'):
            focal_depth = abs(self.calc_focal_plane())  # 正值 [mm]
        else:
            focal_depth = None

        if focal_depth is not None:
            # 扩展范围以包含焦点深度
            depth_min_ext = min(float(depth_min), focal_depth)
            depth_max_ext = max(float(depth_max), focal_depth)

            disp_near = 1.0 / depth_min_ext   # 大视差 = 近处
            disp_far  = 1.0 / depth_max_ext   # 小视差 = 远处
            focal_disp = 1.0 / focal_depth

            # 按两侧范围比例分配样本
            near_range = disp_near - focal_disp
            far_range  = focal_disp - disp_far
            total_range = near_range + far_range

            if total_range < 1e-10:
                disp_ref = torch.full((num_layers,), focal_disp, device=self.device)
            else:
                n_far  = max(1, round((num_layers - 1) * far_range / total_range))
                n_near = num_layers - 1 - n_far

                far_disps  = torch.linspace(disp_far, focal_disp, n_far + 1, device=self.device)        # 包含焦点
                near_disps = torch.linspace(focal_disp, disp_near, n_near + 1, device=self.device)[1:]   # 排除重复焦点
                disp_ref = torch.cat([far_disps, near_disps])
        else:
            # 后备方案：均匀视差采样
            disp_ref = torch.linspace(1.0 / float(depth_max), 1.0 / float(depth_min), num_layers, device=self.device)

        depths_ref = -1.0 / disp_ref
        return disp_ref, depths_ref

    def render_rgbd(self, img_obj, depth_map, method="psf_patch", **kwargs):
        """渲染 RGBD 图像。

        TODO：添加遮挡感知图像仿真。

        参数:
            img_obj (torch.Tensor): 物体图像，shape [B, C, H, W]。
            depth_map (torch.Tensor): 深度图 [mm]，shape [B, 1, H, W]（也接受
                [B, H, W]）。值应为正。
            method (str, optional): 图像仿真方法，可为 "psf_patch"、"psf_map"
                或 "psf_pixel"。默认为 "psf_patch"。
            **kwargs: 方法特定的关键字参数，例如 interp_mode (str)："depth" 或
                "disparity"，默认为 "disparity"；num_layers (int)：深度层数，
                默认为 16。

        返回:
            img_render (torch.Tensor): 渲染图像，shape [B, C, H, W]。

        异常:
            ValueError: depth_map 包含负值时抛出。
            Exception: 无法识别 method 时抛出。

        参考:
            [1] "Aberration-Aware Depth-from-Focus", TPAMI 2023.
            [2] "Efficient Depth- and Spatially-Varying Image Simulation for Defocus Deblur", ICCVW 2025.
        """
        if depth_map.min() < 0:
            raise ValueError("Depth map should be positive.")

        if len(depth_map.shape) == 3:
            # [B, H, W] -> [B, 1, H, W]
            depth_map = depth_map.unsqueeze(1)

        if method == "psf_patch":
            # 渲染图像图块（相同 FoV，不同深度）
            patch_center = kwargs.get("patch_center", (0.0, 0.0))
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            depth_min = kwargs.get("depth_min", depth_map.min())
            depth_max = kwargs.get("depth_max", depth_map.max())
            num_layers = kwargs.get("num_layers", 16)
            interp_mode = kwargs.get("interp_mode", "disparity")

            # 计算不同深度处的 PSF，(num_layers, 3, ks, ks)
            disp_ref, depths_ref = self._sample_depth_layers(depth_min, depth_max, num_layers)

            points = torch.stack(
                [
                    torch.full_like(depths_ref, patch_center[0]),
                    torch.full_like(depths_ref, patch_center[1]),
                    depths_ref,
                ],
                dim=-1,
            )
            psfs = self.psf_rgb(points=points, ks=psf_ks) # (num_layers, 3, ks, ks)

            # 图像仿真
            img_render = conv_psf_depth_interp(img_obj, -depth_map, psfs, depths_ref, interp_mode=interp_mode)
            return img_render

        elif method == "psf_map":
            # 使用 PSF 图卷积渲染全分辨率图像（不同 FoV、不同深度）
            psf_grid = kwargs.get("psf_grid", (8, 8))  # (grid_w, grid_h)
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            depth_min = kwargs.get("depth_min", depth_map.min())
            depth_max = kwargs.get("depth_max", depth_map.max())
            num_layers = kwargs.get("num_layers", 16)
            interp_mode = kwargs.get("interp_mode", "disparity")

            # 计算不同深度处的 PSF 图（转换为负值以进行 PSF 计算）
            disp_ref, depths_ref = self._sample_depth_layers(depth_min, depth_max, num_layers)

            psf_maps = []
            from tqdm import tqdm
            for depth in tqdm(depths_ref):
                psf_map = self.psf_map_rgb(grid=psf_grid, ks=psf_ks, depth=depth)
                psf_maps.append(psf_map)
            psf_map = torch.stack(
                psf_maps, dim=2
            )  # shape [grid_h, grid_w, num_layers, 3, ks, ks]

            # 图像仿真
            img_render = conv_psf_map_depth_interp(
                img_obj, -depth_map, psf_map, depths_ref, interp_mode=interp_mode
            )
            return img_render

        elif method == "psf_pixel":
            # 使用逐像素 PSF splatting 渲染全分辨率图像。此方法计算开销较大。
            psf_ks = kwargs.get("psf_ks", PSF_KS)
            assert img_obj.shape[0] == 1, "Now only support batch size 1"

            # 计算物方中的点
            points_xy = torch.meshgrid(
                torch.linspace(-1, 1, img_obj.shape[-1], device=self.device),
                torch.linspace(1, -1, img_obj.shape[-2], device=self.device),
                indexing="xy",
            )
            points_xy = torch.stack(points_xy, dim=0).unsqueeze(0)
            points = torch.cat([points_xy, -depth_map], dim=1)  # shape [B, 3, H, W]

            # 计算不同像素处的 PSF。此步骤最耗时。
            points = points.permute(0, 2, 3, 1).reshape(-1, 3)  # shape [H*W, 3]
            psfs = self.psf_rgb(points=points, ks=psf_ks)  # shape [H*W, 3, ks, ks]
            psfs = psfs.reshape(
                img_obj.shape[-2], img_obj.shape[-1], 3, psf_ks, psf_ks
            )  # shape [H, W, 3, ks, ks]

            # 图像仿真
            img_render = splat_psf_per_pixel(img_obj, psfs)  # shape [1, C, H, W]
            return img_render

        else:
            raise Exception(f"Image simulation method {method} is not supported.")

    # ===========================================
    # 优化相关函数
    # ===========================================
    def activate_grad(self, activate=True):
        """激活（或停用）每个表面的梯度。

        必须由子类覆盖。

        参数:
            activate (bool, optional): 是否启用梯度。默认为 True。

        异常:
            NotImplementedError: 此基础实现必须被覆盖。
        """
        raise NotImplementedError

    def get_optimizer_params(self, lr=[1e-4, 1e-4, 1e-1, 1e-3]):
        """构建按参数组划分的优化器参数。必须由子类覆盖。

        参数:
            lr (list, optional): 不同镜头参数组的学习率。默认为
                [1e-4, 1e-4, 1e-1, 1e-3]。

        返回:
            params (list): torch 优化器的参数组字典列表。

        异常:
            NotImplementedError: 此基础实现必须被覆盖。
        """
        raise NotImplementedError

    def get_optimizer(self, lr=[1e-4, 1e-4, 0, 1e-3]):
        """为镜头参数组构建 Adam 优化器。

        参数:
            lr (list, optional): 传给 `get_optimizer_params` 的各参数组学习率。
                默认为 [1e-4, 1e-4, 0, 1e-3]。

        返回:
            optimizer (torch.optim.Adam): 配置好的 Adam 优化器。
        """
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)
        return optimizer
