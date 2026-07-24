# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""基于弥散圆（CoC）PSF 的离焦镜头模型。

此镜头通过预先计算弥散圆 PSF 并直接应用它来模拟离焦模糊（景深），而非追迹光线。
CoC PSF 源自近轴光学：给定焦距、F-number 和对焦距离后，各物距处的弥散圆直径
遵循近轴离焦关系，并将所得模糊圆盘用作 PSF。这种 CoC 表述是文献以及 Blender
等软件中离焦仿真的标准方法。它能够描述离焦，但不包含高阶光学像差。

此模型还可通过 `psf_dp`、`psf_rgb_dp`、`psf_map_dp` 和 `render_rgbd_dp`
生成双像素（DP）PSF，即双像素传感器记录的左/右子孔径视图，可用于离焦和深度
估计研究。

参考:
    [1] https://en.wikipedia.org/wiki/Circle_of_confusion
"""

import numpy as np
import torch

from .lens import Lens
from .config import EPSILON, PSF_KS
from .imgsim import conv_psf_depth_interp, conv_psf_occlusion


class DefocusLens(Lens):
    """预先计算弥散圆（CoC）PSF 的离焦镜头。

    此模型不使用光线传递（ABCD）矩阵或薄透镜光线追迹，而是根据焦距、F-number
    和对焦距离推导弥散圆，构建相应 PSF 并直接应用。它能模拟离焦模糊（景深），
    但不包含高阶光学像差，可作为 Blender 及类似工具中常用的快速基线渲染器。

    属性:
        foclen (float): 焦距 [mm]。
        fnum (float): F-number。
        foc_dist (float): 当前对焦距离 [mm]，由 `refocus` 设置（按惯例为负值）。
        sensor_size (tuple): 传感器物理尺寸 (W, H) [mm]。
        sensor_res (tuple): 像素分辨率 (W, H)。
        pixel_size (float): 像素间距 [mm]。
    """

    def __init__(
        self,
        foclen,
        fnum,
        sensor_size=(8.0, 8.0),
        sensor_res=(2000, 2000),
        device=None,
        dtype=torch.float32,
    ):
        """初始化离焦镜头。

        离焦镜头通过与波长无关的弥散圆来建模几何离焦，因此与其他镜头类不同，
        它不接收波长或默认物距参数。

        参数:
            foclen (float): 焦距 [mm]。
            fnum (float): F-number。
            sensor_size (tuple, optional): 传感器物理尺寸 (W, H) [mm]。默认为 (8.0, 8.0)。
            sensor_res (tuple, optional): 传感器分辨率 (W, H) [pixels]。默认为 (2000, 2000)。
            device (str, optional): 计算设备。默认为 None（有 GPU 时自动选择 GPU，否则选择 CPU）。
            dtype (torch.dtype, optional): 计算数据类型。默认为 torch.float32。
        """
        super(DefocusLens, self).__init__(
            device=device,
            dtype=dtype,
        )

        # 镜头参数
        self.foclen = foclen  # 焦距 [mm]
        self.fnum = fnum

        # 配置传感器（设置 sensor_size、sensor_res、pixel_size、r_sensor）。
        self.set_sensor(sensor_size, sensor_res)
        self.astype(self.dtype)

        self.d_far = -20000.0
        self.d_close = -200.0
        self.refocus(foc_dist=-20000)

    def refocus(self, foc_dist):
        """将镜头重新对焦到给定物距。

        参数:
            foc_dist (float): 对焦距离 [mm]，按惯例为负值（物体位于镜头前方）。
                必须小于焦距。

        异常:
            AssertionError: `foc_dist` 不小于 `self.foclen` 时抛出。
        """
        assert foc_dist < self.foclen, "Focus distance is too close."
        self.foc_dist = foc_dist

    # ===========================================
    # PSF 相关函数
    # ===========================================

    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """将离焦 PSF 计算为直径为 CoC 的圆盘。

        PSF 是二维模糊圆盘，其直径等于各物距处的弥散圆；对其施加圆形掩膜并
        归一化，使总和为 1。当 `psf_type="gaussian"` 时，圆盘采用高斯
        衰减填充；当 `psf_type="pillbox"` 时，则为平顶（均匀）圆盘。CoC 模型
        与波长无关，因此仅为保持与其他镜头类型的 API 一致性而接收 `wvln`，
        但不会使用它。

        参数:
            points (torch.Tensor): 物点位置 [mm]，shape [N, 3] 或 [3]；
                深度取 z（第三个）坐标。
            wvln (float or None, optional): 波长 [µm]。此参数会被忽略（CoC 模型
                无色差）。默认为 None。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            **kwargs: 模型特定选项。`psf_type` (str)："gaussian"（默认）或 "pillbox"。

        返回:
            psf (torch.Tensor): 归一化 PSF 核；单点时 shape [ks, ks]，
                N 个点时 shape [N, ks, ks]。
        """
        psf_type = kwargs.get("psf_type", "gaussian")
        points = points.to(self.device)

        # 区分单点与多点输入
        if len(points.shape) == 1:
            points = points.unsqueeze(0)
            single_point = True
        else:
            single_point = False

        # 计算每个点的弥散圆
        depths = points[:, 2]  # shape [N]
        coc_values = self.coc(depths)  # shape [N]

        # 将 CoC 从 mm 转为像素，并设置最小值以保证数值稳定性
        coc_pixel = torch.clamp(
            coc_values / self.pixel_size, min=0.5
        )  # shape [N]，最小为 0.5 pixels
        coc_pixel = coc_pixel.unsqueeze(-1).unsqueeze(-1)  # shape [N, 1, 1]，与 [ks, ks] 广播
        coc_pixel_radius = coc_pixel / 2

        # 创建坐标网格
        x, y = torch.meshgrid(
            torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks, device=self.device),
            torch.linspace(-ks / 2 + 1 / 2, ks / 2 - 1 / 2, ks, device=self.device),
            indexing="xy",
        )
        distance_sq = x**2 + y**2

        # 创建 PSF
        if psf_type == "gaussian":
            # 高斯 PSF
            psf = torch.exp(-distance_sq / (2 * coc_pixel_radius**2)) / (
                2 * np.pi * coc_pixel_radius**2
            )
        elif psf_type == "pillbox":
            # 均匀圆盘 PSF
            psf = torch.ones_like(x)
        else:
            raise ValueError(f"Invalid PSF type: {psf_type}")

        # 应用圆形掩膜
        psf_mask = distance_sq < coc_pixel_radius**2
        psf = psf * psf_mask

        # 归一化 PSF，使其总和为 1
        psf = psf / (psf.sum(dim=(-1, -2), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return psf

    def coc(self, depth):
        """根据近轴离焦关系计算弥散圆（CoC）直径。

        先将深度限制在 `[self.d_far, self.d_close]`，取绝对距离后再计算 CoC。

        参数:
            depth (torch.Tensor): 物体深度 [mm]，shape [B]（或标量）。
                按惯例为负值（物体位于镜头前方）。

        返回:
            coc (torch.Tensor): 弥散圆直径 [mm]，shape 与 `depth` 相同。

        参考:
            [1] https://en.wikipedia.org/wiki/Circle_of_confusion
        """
        depth = torch.as_tensor(depth, device=self.device)
        foc_dist = torch.tensor(
            self.foc_dist, device=self.device, dtype=depth.dtype
        ).abs()
        foclen = self.foclen
        fnum = self.fnum

        depth = torch.clamp(depth, self.d_far, self.d_close)
        depth = torch.abs(depth)

        # 计算弥散圆直径 [mm]
        part1 = torch.abs(depth - foc_dist) / depth
        part2 = foclen**2 / (fnum * (foc_dist - foclen))
        coc = part1 * part2

        return coc

    def dof(self, depth):
        """计算给定物体深度处的景深（DoF）。

        参数:
            depth (torch.Tensor): 物体深度 [mm]，shape [B]（或标量）。
                按惯例为负值（物体位于镜头前方）。

        返回:
            dof (torch.Tensor): 景深 [mm]，shape 与 `depth` 相同。

        参考:
            [1] https://en.wikipedia.org/wiki/Depth_of_field
        """
        depth = torch.as_tensor(depth, device=self.device)
        depth = torch.clamp(depth, self.d_far, self.d_close)
        depth_abs = torch.abs(depth)

        foclen = self.foclen
        fnum = self.fnum

        # 放大倍率
        m = foclen / (depth_abs - foclen)

        # CoC [mm]
        coc = self.coc(depth)

        # 景深 [mm]
        part1 = 2 * fnum * coc * (m + 1)
        part2 = m**2 - (fnum * coc / foclen) ** 2
        dof = part1 / part2

        return dof

    def psf_rgb(self, points, ks=PSF_KS, **kwargs):
        """将单色 PSF 复制到三个通道以计算 RGB PSF。

        离焦模型无色差，因此所有通道共用同一个 PSF。

        参数:
            points (torch.Tensor): 物点位置 [mm]，shape [N, 3]。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            **kwargs: 转发给 `psf`。

        返回:
            psf_rgb (torch.Tensor): RGB PSF，shape [N, 3, ks, ks]。
        """
        psf = self.psf(points, ks=ks, psf_type="gaussian", **kwargs)
        return psf.unsqueeze(1).repeat(1, 3, 1, 1)

    def psf_map(self, grid=(5, 5), ks=PSF_KS, depth=None, **kwargs):
        """计算空间均匀的单色 PSF 图。

        由于离焦模型没有空间变化的像差，每个网格位置都使用相同的轴上 PSF。

        参数:
            grid (tuple, optional): 网格尺寸 (rows, cols)。默认为 (5, 5)。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            depth (float or None, optional): 物体深度 [mm]。为 None（默认）时
                使用 `self.obj_depth`。
            **kwargs: 转发给 `psf`。

        返回:
            psf_map (torch.Tensor): PSF 图，shape [rows, cols, 1, ks, ks]。
        """
        depth = self.obj_depth if depth is None else depth
        points = torch.tensor([[0, 0, depth]], device=self.device)
        psf = self.psf(points=points, ks=ks, psf_type="gaussian", **kwargs)
        psf_map = psf.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        return psf_map

    # =============================================
    # 双像素 PSF
    # =============================================
    def psf_dp(self, points, ks=PSF_KS):
        """通过遮挡基础 PSF 生成左/右双像素 PSF。

        取基础离焦 PSF，并将孔径垂直分为左右两半以模拟双像素传感器。根据物体
        比对焦距离更近还是更远，交换分配给各子孔径的半侧，从而复现随深度变化的
        左右视差；双像素深度估计和自动对焦正是利用该视差。

        参数:
            points (torch.Tensor): 物点位置 [mm]，shape [N, 3]，列为 [x, y, z]；
                深度取自 z。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。

        返回:
            psf_l (torch.Tensor): 左子孔径 PSF，shape [N, ks, ks]。
            psf_r (torch.Tensor): 右子孔径 PSF，shape [N, ks, ks]。
        """
        depth = points[:, 2]

        # 获取基础 PSF
        psf_base = self.psf(points, ks=ks, psf_type="gaussian")
        device = psf_base.device

        # 为双像素仿真创建左右掩膜
        l_mask = torch.ones((ks, ks), device=device)
        r_mask = torch.ones((ks, ks), device=device)

        # 垂直分割孔径（左半侧和右半侧）
        l_pixel, r_pixel = ks // 2, ks // 2 + 1
        l_mask[:, 0:l_pixel] = 0  # 遮挡左 PSF 的右侧
        r_mask[:, r_pixel:] = 0  # 遮挡右 PSF 的左侧

        # 确定对焦位置
        depth = depth.to(device)
        foc_dist = torch.tensor(self.foc_dist, device=device, dtype=depth.dtype)
        near_focus_pos = depth > foc_dist  # shape [N]

        # 根据对焦位置应用掩膜（向量化）
        # 近焦：左 PSF 使用左掩膜，右 PSF 使用右掩膜
        # 远焦：交换掩膜以形成相反的不对称性
        nfp = near_focus_pos.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
        mask_l = torch.where(nfp, l_mask, r_mask)  # [N, ks, ks]
        mask_r = torch.where(nfp, r_mask, l_mask)  # [N, ks, ks]
        psf_l = psf_base * mask_l
        psf_r = psf_base * mask_r

        # 归一化 PSF
        psf_l = psf_l / (psf_l.sum(dim=(-1, -2), keepdim=True) + EPSILON)
        psf_r = psf_r / (psf_r.sum(dim=(-1, -2), keepdim=True) + EPSILON)

        return psf_l, psf_r

    def psf_rgb_dp(self, points, ks=PSF_KS):
        """计算左右子孔径的 RGB 双像素 PSF。

        将单色双像素 PSF 复制到三个颜色通道。

        参数:
            points (torch.Tensor): 物点位置 [mm]，shape [N, 3]。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。

        返回:
            psf_l (torch.Tensor): 左子孔径 RGB PSF，shape [N, 3, ks, ks]。
            psf_r (torch.Tensor): 右子孔径 RGB PSF，shape [N, 3, ks, ks]。
        """
        psf_l, psf_r = self.psf_dp(points, ks=ks)
        psf_l = psf_l.unsqueeze(1).repeat(1, 3, 1, 1)
        psf_r = psf_r.unsqueeze(1).repeat(1, 3, 1, 1)
        return psf_l, psf_r

    def psf_map_dp(self, grid=(5, 5), ks=PSF_KS, depth=None, **kwargs):
        """计算空间均匀的双像素 PSF 图。

        参数:
            grid (tuple, optional): 网格尺寸 (rows, cols)。默认为 (5, 5)。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            depth (float or None, optional): 物体深度 [mm]。为 None（默认）时
                使用 `self.obj_depth`。
            **kwargs: 转发给 `psf_dp`。

        返回:
            psf_map_l (torch.Tensor): 左子孔径 PSF 图，shape [rows, cols, 1, ks, ks]。
            psf_map_r (torch.Tensor): 右子孔径 PSF 图，shape [rows, cols, 1, ks, ks]。
        """
        depth = self.obj_depth if depth is None else depth
        points = torch.tensor([[0, 0, depth]], device=self.device)
        psf_l, psf_r = self.psf_dp(points, ks=ks, **kwargs)
        psf_map_l = psf_l.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        psf_map_r = psf_r.unsqueeze(0).unsqueeze(0).repeat(grid[0], grid[1], 1, 1, 1)
        return psf_map_l, psf_map_r

    # =============================================
    # RGBD 渲染（遮挡感知）
    # =============================================
    def render_rgbd(
        self,
        img_obj,
        depth_map,
        psf_ks=PSF_KS,
        num_layers=16,
    ):
        """面向离焦镜头的遮挡感知 RGBD 渲染。

        使用从后向前的分层合成，防止深度不连续处发生颜色渗漏。由于离焦镜头
        没有空间变化的像差，渲染时使用跨深度层采样的空间不变 PSF。

        参数:
            img_obj (torch.Tensor): 物体图像，shape [B, C, H, W]。
            depth_map (torch.Tensor): 深度图 [mm]，shape [B, 1, H, W]
                （或 [B, H, W]）。值必须为正。
            psf_ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            num_layers (int, optional): 深度层数。默认为 16。

        返回:
            img_render (torch.Tensor): 渲染后的图像，shape [B, C, H, W]。

        参考:
            [1] "Dr.Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering", CVPR 2024.
        """
        if depth_map.min() < 0:
            raise ValueError("Depth map should be positive.")

        if len(depth_map.shape) == 3:
            depth_map = depth_map.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]

        depth_min = depth_map.min()
        depth_max = depth_map.max()

        # 采样深度层
        disp_ref, depths_ref = self._sample_depth_layers(depth_min, depth_max, num_layers)

        # 计算每个深度层的 PSF（空间不变，因此 patch_center=(0,0)）
        points = torch.stack(
            [
                torch.zeros_like(depths_ref),
                torch.zeros_like(depths_ref),
                depths_ref,
            ],
            dim=-1,
        )
        psfs = self.psf_rgb(points=points, ks=psf_ks)  # [num_layers, 3, ks, ks]

        # 遮挡感知渲染
        img_render = conv_psf_occlusion(img_obj, -depth_map, psfs, depths_ref)
        return img_render

    def render_rgbd_dp(
        self,
        rgb_img,
        depth,
        psf_ks=PSF_KS,
        num_layers=16,
    ):
        """根据 RGBD 输入渲染左/右双像素图像。

        在均匀采样的参考深度处计算双像素 PSF，并结合深度插值对图像做卷积，
        生成两个子孔径视图。内部会将正深度取负，使物体位于镜头前方。

        参数:
            rgb_img (torch.Tensor): RGB 物体图像，shape [B, 3, H, W]。
            depth (torch.Tensor): 深度图 [mm]，shape [B, 1, H, W]。
            psf_ks (int, optional): PSF 核尺寸 [pixels]。默认为 PSF_KS。
            num_layers (int, optional): 深度层数。默认为 16。

        返回:
            img_left (torch.Tensor): 左子孔径图像，shape [B, 3, H, W]。
            img_right (torch.Tensor): 右子孔径图像，shape [B, 3, H, W]。
        """
        # 将深度转换为负值
        if (depth > 0).any():
            depth = -depth

        depth_min = depth.min()
        depth_max = depth.max()
        patch_center = (0.0, 0.0)

        # 计算参考深度处的双像素 PSF
        depths_ref = torch.linspace(depth_min, depth_max, num_layers, device=self.device)
        points = torch.stack(
            [
                torch.full_like(depths_ref, patch_center[0]),
                torch.full_like(depths_ref, patch_center[1]),
                depths_ref,
            ],
            dim=-1,
        )
        psfs_left, psfs_right = self.psf_rgb_dp(
            points=points, ks=psf_ks
        )  # shape [num_layers, 3, ks, ks]

        # 使用 PSF 卷积和深度插值渲染双像素图像
        img_left = conv_psf_depth_interp(rgb_img, depth, psfs_left, depths_ref)
        img_right = conv_psf_depth_interp(rgb_img, depth, psfs_right, depths_ref)
        return img_left, img_right


if __name__ == "__main__":
    from torchvision.utils import make_grid, save_image

    lens = DefocusLens(
        foclen=50, fnum=1.8, sensor_size=(20.0, 20.0), sensor_res=(2000, 2000)
    )
    lens.refocus(-1000)
    lens.draw_psf_map(
        save_name="./psf_map_defocus_depth1500_focus1000.png",
        grid=(11, 11),
        ks=PSF_KS,
        depth=-1500,
        log_scale=False,
    )

    # 远处的双像素 PSF
    psf_map_l, psf_map_r = lens.psf_map_dp(grid=(11, 11), ks=128, depth=-1500)
    psf_map_l = psf_map_l.reshape(-1, 1, 128, 128)
    psf_map_r = psf_map_r.reshape(-1, 1, 128, 128)
    save_image(
        make_grid(psf_map_l, nrow=11),
        "./psf_map_dp_left_depth1500_focus1000.png",
        normalize=True,
    )
    save_image(
        make_grid(psf_map_r, nrow=11),
        "./psf_map_dp_right_depth1500_focus1000.png",
        normalize=True,
    )

    # 近处的双像素 PSF
    psf_map_l, psf_map_r = lens.psf_map_dp(grid=(11, 11), ks=128, depth=-800)
    psf_map_l = psf_map_l.reshape(-1, 1, 128, 128)
    psf_map_r = psf_map_r.reshape(-1, 1, 128, 128)
    save_image(
        make_grid(psf_map_l, nrow=11),
        "./psf_map_dp_left_depth800_focus1000.png",
        normalize=True,
    )
    save_image(
        make_grid(psf_map_r, nrow=11),
        "./psf_map_dp_right_depth800_focus1000.png",
        normalize=True,
    )
