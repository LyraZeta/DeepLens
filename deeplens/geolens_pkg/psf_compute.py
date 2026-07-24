# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""几何透镜系统的 PSF 计算方法。

支持三种 PSF 模型：
    1. 几何 PSF（``psf_geometric``）：非相干强度光线追迹——速度快且可微。
    2. 出瞳 PSF（``psf_pupil_prop`` / ``psf_coherent``）：相干追迹到出瞳，
       再以角谱法 (ASM) 进行自由空间传播——精确且可微。
    3. 惠更斯 PSF（``psf_huygens``）：相干追迹到出瞳，再进行惠更斯-菲涅耳
       积分——精确但不可微。

函数：
    - psf()：在几何、相干和惠更斯模型之间分派。
    - psf_geometric()：通过光线分箱计算非相干几何 PSF。
    - psf_coherent()：psf_pupil_prop 的别名。
    - psf_pupil_prop()：通过相干追迹 + ASM 计算出瞳衍射 PSF。
    - pupil_field()：计算出瞳平面的复波前。
    - psf_huygens()：通过次级点光源积分计算惠更斯-菲涅耳 PSF。
    - psf_map()：计算整个视场的几何 PSF 图。
    - psf_center()：通过主光线或针孔投影计算参考 PSF 中心。
"""

import torch
import torch.nn.functional as F

from ..config import (
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
)
from ..imgsim import forward_integral
from ..light import AngularSpectrumMethod
from ..utils import diff_float


class GeoLensPSF:
    """为 `GeoLens` 提供 PSF 计算的混入类。

    通过统一的 `psf` 分派器提供三种 PSF 模型：非相干几何光线追迹、
    相干出瞳衍射（ASM 传播）和惠更斯-菲涅耳积分。几何模型与相干模型
    可微，惠更斯模型不可微。本类不单独实例化，而是混入 `GeoLens`。
    """

    # ====================================================================================
    # 点扩散函数（PSF）
# 支持三种 PSF：
#   1. 几何 PSF（`psf`）：非相干强度光线追迹
#   2. 出瞳 PSF（`psf_pupil_prop` / `psf_coherent`）：相干追迹到出瞳，再使用 ASM 进行自由空间传播
#   3. 惠更斯 PSF（`psf_huygens`）：相干追迹到出瞳，再进行惠更斯-菲涅耳积分
    # ====================================================================================
    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """计算给定点光源的点扩散函数 (PSF)。

        分派到以下三种 PSF 模型之一：
            - geometric：非相干强度光线追迹（快速、可微）。
            - coherent：相干追迹到出瞳 + ASM 传播（精确、可微、单点）。
            - huygens：惠更斯-菲涅耳积分（精确、不可微、单点）。

        参数：
            points (torch.Tensor)：归一化点光源位置。Shape [N, 3]，其中 x、y
                位于 [-1, 1]，z 位于 [-Inf, 0]。coherent 和 huygens 模型
                仅接受单个点（[3] 或 [1, 3]）。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            ks (int, optional)：输出核的像素尺寸。默认值为 PSF_KS。
            **kwargs：模型专用选项：
                spp (int)：每个光源的采样光线数。为 None 时使用对应模型的
                默认值（SPP_PSF / SPP_COHERENT）。
                recenter (bool)：为 True（默认）时以主光线为 PSF 中心，
                否则以针孔投影为中心。
                model (str)：'geometric'（默认）、'coherent'、'huygens' 之一。

        返回：
            psf (torch.Tensor)：总和归一化为 1 的 PSF。单点时 shape 为
                [ks, ks]；几何模型输入 N 个点时为 [N, ks, ks]。

        异常：
            ValueError：`model` 不在支持的名称中时抛出。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        spp = kwargs.get("spp", None)
        recenter = kwargs.get("recenter", True)
        model = kwargs.get("model", "geometric")
        if model == "geometric":
            spp = SPP_PSF if spp is None else spp
            return self.psf_geometric(points, ks, wvln, spp, recenter)
        elif model == "coherent":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_coherent(points, ks, wvln, spp, recenter)
        elif model == "huygens":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_huygens(points, ks, wvln, spp, recenter)
        else:
            raise ValueError(f"Unknown PSF model: {model}")

    def psf_geometric(
        self, points, ks=PSF_KS, wvln=None, spp=SPP_PSF, recenter=True
    ):
        """通过非相干光线分箱计算单波长几何 PSF。

        从各物点采样光线，以非相干方式追迹到传感器，并将命中位置分箱到
        `ks × ks` 强度核中。该模型速度快且可微。

        参数：
            points (torch.Tensor)：归一化点光源位置。Shape [N, 3]，其中 x、y
                位于 [-1, 1]，z 位于 [-Inf, 0]。
            ks (int, optional)：输出核的像素尺寸。默认值为 PSF_KS。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            spp (int, optional)：每个光源的采样光线数。默认值为 SPP_PSF。
            recenter (bool, optional)：为 True（默认）时以主光线为中心，
                否则以针孔投影为中心。

        返回：
            psf (torch.Tensor)：总和归一化为 1 的 PSF。单点时 shape 为
                [ks, ks]，N 个点时为 [N, ks, ks]。

        参考：
            [1] https://optics.ansys.com/hc/en-us/articles/42661723066515-What-is-a-Point-Spread-Function
        """
        wvln = self.primary_wvln if wvln is None else wvln
        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device

        # 点的 shape 为 [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points, device=device)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        else:
            single_point = False

        # 采样光线；通过透视投影确定物方光线位置
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # 将光线以非相干方式追迹到传感器平面
        ray.is_coherent = False
        ray = self.trace2sensor(ray)

        # 计算 PSF 中心，shape [N, 2]
        if recenter:
            pointc = self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = self.psf_center(point_obj, method="pinhole")

        # 蒙特卡洛积分
        psf = forward_integral(ray.flip_xy(), ps=pixel_size, ks=ks, pointc=pointc)

        # 强度归一化
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_coherent(
        self, points, ks=PSF_KS, wvln=None, spp=SPP_COHERENT, recenter=True
    ):
        """计算相干出瞳 PSF（`psf_pupil_prop` 的别名）。

        将相干光线追迹到出瞳，并使用角谱法 (ASM) 将波前传播到传感器。
        完整参数和返回值说明请参阅 `psf_pupil_prop`。

        参数：
            points (torch.Tensor)：单个归一化点光源 [3] 或 [1, 3]，其中 x、y
                位于 [-1, 1]，z 位于 [-Inf, 0]。
            ks (int, optional)：输出核的像素尺寸。默认值为 PSF_KS。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            spp (int, optional)：采样光线数。默认值为 SPP_COHERENT。
            recenter (bool, optional)：为 True（默认）时以主光线为中心。

        返回：
            psf (torch.Tensor)：总和归一化为 1 的 PSF。Shape [ks, ks]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        return self.psf_pupil_prop(points, ks=ks, wvln=wvln, spp=spp, recenter=recenter)

    def psf_pupil_prop(
        self, points, ks=PSF_KS, wvln=None, spp=SPP_COHERENT, recenter=True
    ):
        """通过出瞳衍射模型计算单点单色 PSF。

        步骤：
            1. 通过相干光线追迹计算出瞳平面的复波前。
            2. 使用角谱法 (ASM) 传播到传感器平面，并将强度作为 PSF。
               本函数可微。

        参数：
            points (torch.Tensor or list)：单个归一化点光源 [3] 或 [1, 3]，
                其中 x、y 位于 [-1, 1]，z 位于 [-Inf, 0]。
            ks (int, optional)：输出 PSF 图块的像素尺寸。为 None 时返回未经
                裁剪的完整传播强度场。默认值为 PSF_KS。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            spp (int, optional)：采样光线数。默认值为 SPP_COHERENT。
            recenter (bool, optional)：为 True（默认）时以主光线为中心，
                否则以针孔投影为中心。

        返回：
            psf (torch.Tensor)：总和归一化为 1 的 PSF。指定 `ks` 时 shape
                为 [ks, ks]；`ks` 为 None 时返回 shape 为 [1, 1, 2H, 2H]
                的完整未裁剪强度场（零填充后为出瞳网格的两倍）。

        参考：
            [1] "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model", SIGGRAPH Asia 2024.

        说明：
            与 ZEMAX FFT PSF 类似，但自由空间传播使用角谱法 (ASM)，而非
            单次 FFT。ASM 更精确，因为 FFT 方法假设满足远场条件
            （例如主光线垂直于像面）。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        # 通过相干光线追迹计算瞳面场
        wavefront, psfc = self.pupil_field(
            points=points, wvln=wvln, spp=spp, recenter=recenter
        )

        # 传播到传感器平面并获得强度
        pupilz, pupilr = self.get_exit_pupil()
        h, w = wavefront.shape
        # 手动填充波场
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        # 使用角谱法 (ASM) 进行自由空间传播
        sensor_field = AngularSpectrumMethod(
            wavefront,
            z=self.d_sensor - pupilz,
            wvln=wvln,
            ps=self.pixel_size,
            padding=False,
        )
        # 获取强度
        psf_inten = sensor_field.abs() ** 2

        # 计算 PSF 中心
        h, w = psf_inten.shape[-2:]
            # 同时考虑插值和填充
        psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
        psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

            # 裁剪有效 PSF 区域并归一化
        if ks is not None:
            psf_inten_pad = (
                F.pad(
                    psf_inten,
                    [ks // 2, ks // 2, ks // 2, ks // 2],
                    mode="constant",
                    value=0,
                )
                .squeeze(0)
                .squeeze(0)
            )
            psf = psf_inten_pad[
                psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
            ]
        else:
            psf = psf_inten

        # 强度归一化，shape 为 [ks, ks] 或 [h, w]
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        return diff_float(psf)

    def pupil_field(self, points, wvln=None, spp=SPP_COHERENT, recenter=True):
        """通过相干光线追迹计算出瞳平面的复波前。

        为后续 PSF 计算对波前进行 xy 翻转，并以传感器像素尺寸将其分箱到
        方形 `[H, H]` 网格（H = 传感器像素高度）。本函数可微。

        参数：
            points (torch.Tensor or list)：单个归一化点光源 [3] 或 [1, 3]，
                其中 x、y 位于 [-1, 1]，z 位于 [-Inf, 0]。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            spp (int, optional)：采样光线数。精确相干仿真至少需要
                1,000,000 条光线。默认值为 SPP_COHERENT。
            recenter (bool, optional)：为 True（默认）时以主光线为中心，
                否则以针孔投影为中心。

        返回：
            wavefront (torch.Tensor)：出瞳处按像素尺寸分箱的复波前。Shape [H, H]。
            psf_center (list)：传感器上位于 [-1, 1] 的归一化 PSF 中心 [x, y]。

        说明：
            为准确计算相位，默认 dtype 必须为 torch.float64。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        assert spp >= 1_000_000, (
            f"Ray sampling {spp} is too small for coherent ray tracing, which may lead to inaccurate simulation."
        )
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase calculation."
        )

        sensor_w, sensor_h = self.sensor_size
        device = self.device

        if isinstance(points, list):
            points = torch.tensor(points, device=device).unsqueeze(0)  # [1, 3]
        elif torch.is_tensor(points) and len(points.shape) == 1:
            points = points.unsqueeze(0).to(device)  # [1, 3]
        elif torch.is_tensor(points) and len(points.shape) == 2:
            assert points.shape[0] == 1, (
                f"pupil_field only supports single point input, got shape {points.shape}"
            )
        else:
            raise ValueError(f"Unsupported point type {points.type()}.")

        assert points.shape[0] == 1, (
            "Only one point is supported for pupil field calculation."
        )

        # 物方光线原点
        scale = self.calc_scale(points[:, 2].item())
        point_obj_x = points[:, 0] * scale * sensor_w / 2
        point_obj_y = points[:, 1] * scale * sensor_h / 2
        points_obj = torch.stack([point_obj_x, point_obj_y, points[:, 2]], dim=-1)

        # 由主光线确定光线中心
        # Shape 为 [N, 2]，未归一化物理坐标
        if recenter:
            pointc = self.psf_center(points_obj, method="chief_ray")
        else:
            pointc = self.psf_center(points_obj, method="pinhole")

        # 光线追迹到 exit_pupil
        ray = self.sample_from_points(points=points_obj, num_rays=spp, wvln=wvln)
        ray.is_coherent = True
        ray = self.trace2exit_pupil(ray)

        # 计算复场（物理尺寸和分辨率与传感器相同）
        # 在此翻转复场，以便后续计算 PSF
        pointc_ref = torch.zeros_like(points[:, :2])  # [N, 2]
        wavefront = forward_integral(
            ray.flip_xy(),
            ps=self.pixel_size,
            ks=self.sensor_res[1],
            pointc=pointc_ref,
        )
        wavefront = wavefront.squeeze(0)  # [H, H]

        # PSF 中心（位于传感器平面）。
        pointc = pointc[0, :]
        psf_center = [
            pointc[0] / sensor_w * 2,
            pointc[1] / sensor_h * 2,
        ]

        return wavefront, psf_center

    def psf_huygens(
        self, points, ks=PSF_KS, wvln=None, spp=SPP_COHERENT, recenter=True
    ):
        """通过球面波积分计算单波长惠更斯 PSF。

        由于计算开销很大，本函数不可微。

        步骤：
            1. 将相干光线追迹到出瞳平面。
            2. 将每条光线视为发射球面波的次级点光源，并在 PSF 像素网格上
               相干叠加这些波。每项贡献采用惠更斯-菲涅耳倾斜因子
               $0.5 (1 + \\cos\\theta)$ 和 $1/r$ 球面波振幅衰减。

        参数：
            points (torch.Tensor)：单个归一化点光源 [3] 或 [1, 3]，其中 x、y
                位于 [-1, 1]，z 位于 [-Inf, 0]。
            ks (int, optional)：输出核的像素尺寸。默认值为 PSF_KS。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            spp (int, optional)：采样光线数。默认值为 SPP_COHERENT。
            recenter (bool, optional)：为 True（默认）时以主光线为中心，
                否则以针孔投影为中心。

        返回：
            psf (torch.Tensor)：总和归一化为 1 的 PSF。Shape [ks, ks]。

        参考：
            [1] "Optical Aberrations Correction in Postprocessing Using Imaging Simulation", TOG 2021.

        说明：
            与 ZEMAX 惠更斯 PSF 不同，后者将光线追迹到像面并进行平面波积分。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        assert torch.get_default_dtype() == torch.float64, (
            "Default dtype must be set to float64 for accurate phase calculation."
        )

        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device
        wvln_mm = wvln * 1e-3  # 将波长转换为 mm

        # 点的 shape 为 [N, 3]
        if not torch.is_tensor(points):
            points = torch.tensor(points, device=device)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        elif len(points.shape) == 2 and points.shape[0] == 1:
            single_point = True
        else:
            raise ValueError(
                f"Points must be of shape [3] or [1, 3], got {points.shape}."
            )

        # 从物点采样光线
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # 将光线以相干方式穿过透镜追迹到出瞳
        ray.is_coherent = True
        ray = self.trace2exit_pupil(ray)

        # 计算 PSF 中心（此处不翻转）
        if recenter:
            pointc = -self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = -self.psf_center(point_obj, method="pinhole")

        # 构建 PSF 像素坐标（传感器平面位于 z = d_sensor）
        sensor_z = self.d_sensor.item()
        psf_half_size = (ks / 2) * pixel_size  # PSF 区域的物理半尺寸
        x_coords = torch.linspace(
            -psf_half_size + pixel_size / 2,
            psf_half_size - pixel_size / 2,
            ks,
            device=device,
        )
        y_coords = torch.linspace(
            psf_half_size - pixel_size / 2,
            -psf_half_size + pixel_size / 2,
            ks,
            device=device,
        )
        psf_x, psf_y = torch.meshgrid(
            pointc[0, 0] + x_coords, pointc[0, 1] + y_coords, indexing="xy"
        )  # 各为 [ks, ks]

        # 仅保留有效光线
        valid_mask = ray.is_valid > 0
        valid_pos = ray.o[valid_mask]  # [num_valid, 3]
        valid_dir = ray.d[valid_mask]  # [num_valid, 3]
        valid_opl = ray.opl[valid_mask]  # [num_valid]
        num_valid = valid_pos.shape[0]

        # 惠更斯积分：叠加各次级光源发出的球面波
        psf_complex = torch.zeros(ks, ks, dtype=torch.complex128, device=device)
        opl_min = valid_opl.min()

        # 计算各次级光源到各像素的距离
        batch_size = min(num_valid, 10_000)  # 分批处理光线
        for batch_start in range(0, num_valid, batch_size):
            batch_end = min(batch_start + batch_size, num_valid)

            # 当前批次的光线数据
            batch_pos = valid_pos[batch_start:batch_end]  # [batch, 3]
            batch_dir = valid_dir[batch_start:batch_end]  # [batch, 3]
            batch_opl = valid_opl[batch_start:batch_end].squeeze(-1)  # [batch]

            # 各次级光源到各像素的距离
            # batch_pos: [batch, 3], psf_x: [ks, ks]
            dx = psf_x.unsqueeze(-1) - batch_pos[:, 0]  # [ks, ks, batch]
            dy = psf_y.unsqueeze(-1) - batch_pos[:, 1]  # [ks, ks, batch]
            dz = sensor_z - batch_pos[:, 2]  # [batch]

            # 次级光源到像素的距离 r
            r = torch.sqrt(dx**2 + dy**2 + dz**2)  # [ks, ks, batch]

            # 倾斜因子：cos(theta)，其中 theta 为相对于法线的夹角
            # 使用出瞳处光线方向的 dz 分量
            obliq = torch.abs(batch_dir[:, 2])  # [batch]
            amp = 0.5 * (1.0 + obliq)  # 惠更斯-菲涅耳倾斜因子

            # 总光程 = 穿过透镜的 OPL + 到像素的距离
            total_opl = batch_opl + r  # [ks, ks, batch]

            # 相对于参考的相位
            phase = torch.fmod((total_opl - opl_min) / wvln_mm, 1.0) * (
                2 * torch.pi
            )  # [ks, ks, batch]

            # 复振幅：A * exp(i * phase) / r（球面波衰减）
            # 球面波振幅采用 1/r 衰减
            complex_amp = (amp / r) * torch.exp(1j * phase)  # [ks, ks, batch]

            # 累加当前批次的贡献
            psf_complex += complex_amp.sum(dim=-1)  # [ks, ks]

        # 将复场转换为强度
        psf = psf_complex.abs() ** 2

        # 强度归一化
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        # 翻转 PSF
        psf = torch.flip(psf, [-2, -1])

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_map(
        self,
        depth=None,
        grid=(7, 7),
        ks=PSF_KS,
        spp=SPP_PSF,
        wvln=None,
        recenter=True,
    ):
        """计算给定深度处整个视场的几何 PSF 图。

        覆盖基类 `Lens` 的方法，通过并行追迹所有视场点提高效率。

        参数：
            depth (float, optional)：物面深度 [mm]。为 None（默认）时回退到
                `self.obj_depth`。
            grid (int or tuple, optional)：网格尺寸 (grid_w, grid_h)；int 会扩展
                为方形网格。默认值为 (7, 7)。
            ks (int, optional)：输出核的像素尺寸。默认值为 PSF_KS。
            spp (int, optional)：每个光源的采样光线数。默认值为 SPP_PSF。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。
            recenter (bool, optional)：为 True（默认）时以主光线为中心。

        返回：
            psf_map (torch.Tensor)：PSF 图。Shape [grid_h, grid_w, 1, ks, ks]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        if isinstance(grid, int):
            grid = (grid, grid)
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, recenter=recenter, spp=spp, wvln=wvln
        ).unsqueeze(1)  # [grid_h * grid_w, 1, ks, ks]

        psf_map = psfs.reshape(grid[1], grid[0], 1, ks, ks)
        return psf_map

    @torch.no_grad()
    def psf_center(self, points_obj, method="chief_ray"):
        """计算给定点光源在传感器上的参考 PSF 中心。

        方法为 "chief_ray" 时追迹半孔径光束并取传感器质心的相反数
        （无有效光线时回退到 "pinhole"）；方法为 "pinhole" 时采用无畸变的
        理想透视投影。两种方法返回的中心符号均与原始物点一致。

        参数：
            points_obj (torch.Tensor)：未归一化物面点，shape [..., 3] [mm]，
                范围为 [-Inf, Inf] x [-Inf, Inf] x [-Inf, 0]。
            method (str, optional)："chief_ray" 或 "pinhole"。默认值为 "chief_ray"。

        返回：
            psf_center (torch.Tensor)：传感器平面上的未归一化 PSF 中心 [mm]，
                shape [..., 2]。

        异常：
            ValueError：`method` 既不是 "chief_ray" 也不是 "pinhole" 时抛出。
        """
        if method == "chief_ray":
        # 缩小瞳孔，并将质心光线作为主光线
            ray = self.sample_from_points(points_obj, scale_pupil=0.5, num_rays=SPP_CALC)
            ray = self.trace2sensor(ray)
            if ray.is_valid.any():
                psf_center = ray.centroid()
                psf_center = -psf_center[..., :2]  # shape [..., 2]
            else:
            # 主光线失败时回退到针孔模型（优化期间可能发生）
                return self.psf_center(points_obj, method="pinhole")

        elif method == "pinhole":
            # 针孔相机透视投影，不考虑畸变
            if points_obj[..., 2].min().abs() < 100:
                print(
                    "Point source is too close, pinhole model may be inaccurate for PSF center calculation."
                )
            tan_point_fov_x = -points_obj[..., 0] / points_obj[..., 2]
            tan_point_fov_y = -points_obj[..., 1] / points_obj[..., 2]
            psf_center_x = self.foclen * tan_point_fov_x
            psf_center_y = self.foclen * tan_point_fov_y
            psf_center = torch.stack([psf_center_x, psf_center_y], dim=-1).to(
                self.device
            )

        else:
            raise ValueError(
                f"Unsupported method for PSF center calculation: {method}."
            )

        return psf_center
