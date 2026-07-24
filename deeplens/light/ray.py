# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""光线类。"""

import torch
import torch.nn.functional as F

from ..config import EPSILON
from ..base import DeepObj


class Ray(DeepObj):
    """用于光学仿真的批量光线束。

    保存光线原点、方向、波长、有效性掩码、能量、弯折惩罚，以及相干模式下的
    光程长度。所有张量属性共享批次形状 `(..., num_rays)`，其中原点和方向额外
    带有长度为 3 的末尾空间轴。

    属性：
        o (torch.Tensor): 光线原点，形状为 `(..., num_rays, 3)` [mm]。
        d (torch.Tensor): 单位光线方向，形状为 `(..., num_rays, 3)`。
        wvln (torch.Tensor): 波长标量 [µm]。
        shape (torch.Size): 光线张量共享的批次形状 `(..., num_rays)`。
        is_valid (torch.Tensor): 二值有效性掩码，形状为 `(..., num_rays)`。
        en (torch.Tensor): 能量权重，形状为 `(..., num_rays, 1)`。
        bend_penalty (torch.Tensor): 各表面累计弯折惩罚，形状为
            `(..., num_rays, 1)`。
        opl (torch.Tensor): 光程长度，形状为 `(..., num_rays, 1)` [mm]，仅在
            `is_coherent` 为 True 时累加。
        is_coherent (bool): 是否启用光程长度跟踪。
        device (str): 存放光线张量的计算设备。
    """

    def __init__(self, o, d, wvln, is_coherent=False, device="cpu"):
        """初始化光线对象。

        构造时将方向 `d` 归一化为单位长度。辅助张量（`is_valid`、`en`、
        `bend_penalty`、`opl`）初始化为默认值，并广播至批次形状。

        参数：
            o (torch.Tensor): 光线原点，形状为 `(..., num_rays, 3)` [mm]。
            d (torch.Tensor): 光线方向，形状为 `(..., num_rays, 3)`，内部会
                归一化为单位长度。
            wvln (float): 光线波长 [µm]，必须满足 0.1 < wvln < 10.0。
                该参数必须显式传入；`primary_wvln`/`wvln_rgb` 属于 Lens，
                而不属于 Ray。
            is_coherent (bool, optional): 是否为相干追迹启用光程长度跟踪，
                默认为 False。
            device (str, optional): 计算设备，默认为 "cpu"。
        """
        # 基本光线参数——移动到指定设备
        self.o = (o if torch.is_tensor(o) else torch.tensor(o)).to(device)
        self.d = (d if torch.is_tensor(d) else torch.tensor(d)).to(device)
        self.shape = self.o.shape[:-1]

        # 波长
        assert wvln > 0.1 and wvln < 10.0, "Ray wavelength unit should be [um]"
        self.wvln = torch.tensor(wvln, device=device)

        # 辅助光线参数——直接在指定设备上创建
        self.is_valid = torch.ones(self.shape, device=device)
        self.en = torch.ones((*self.shape, 1), device=device)
        self.bend_penalty = torch.zeros((*self.shape, 1), device=device)

        # 相干光线追迹
        self.is_coherent = is_coherent  # bool
        self.opl = torch.zeros((*self.shape, 1), device=device)

        self.device = device
        self.d = F.normalize(self.d, p=2, dim=-1)

    def prop_to(self, z, n=1.0):
        """将光线原地传播至指定深度平面。

        沿光线方向将每条有效光线的原点移动至轴向坐标为 $z$ 的深度平面。
        对近似平行于该平面的光线（$d_z \\approx 0$）进行截断，以避免产生
        infinite/NaN 参数。在相干模式下，且仅当张量为 float64 时，光程长度
        增加 $n \\cdot t$，其中 $t$ 为传播距离。

        参数：
            z (float): 沿光轴方向的目标深度平面 [mm]。
            n (float, optional): 介质折射率，默认为 1.0。

        返回：
            self (Ray): 更新后的光线，可用于链式调用。
        """
        # 防止光线与目标平面近似平行：d_z ~ 0 会使 t = inf/NaN，并通过下方的
        # torch.where 污染梯度。
        dz = self.d[..., 2]
        dz_safe = torch.where(dz.abs() < EPSILON, torch.full_like(dz, EPSILON), dz)
        t = (z - self.o[..., 2]) / dz_safe
        new_o = self.o + self.d * t.unsqueeze(-1)
        valid_mask = (self.is_valid > 0).unsqueeze(-1)
        self.o = torch.where(valid_mask, new_o, self.o)

        if self.is_coherent:
            if t.dtype != torch.float64:
                raise Warning("Should use float64 in coherent ray tracing.")
            else:
                new_opl = self.opl + n * t.unsqueeze(-1)
                self.opl = torch.where(valid_mask, new_opl, self.opl)

        return self

    def centroid(self):
        """计算有效光线原点的不加权能量质心。

        沿 `num_rays` 轴对光线原点 `o` 求平均，仅计入有效光线（`is_valid`）。

        返回：
            centroid (torch.Tensor): 质心位置，形状为 `(..., 3)` [mm]。
        """
        return (self.o * self.is_valid.unsqueeze(-1)).sum(-2) / self.is_valid.sum(
            -1
        ).add(EPSILON).unsqueeze(-1)

    def rms_error(self, center_ref=None):
        """计算有效光线的平均 RMS 光斑半径。

        对每个批次元素，根据有效光线原点相对 `center_ref` 的平面内 (x, y)
        偏差计算 RMS 半径，随后在批次间求平均得到标量。

        参数：
            center_ref (torch.Tensor, optional): 参考中心，形状为 `(..., 3)` [mm]。
                为 None 时使用各批次质心，默认为 None。

        返回：
            rms_error (torch.Tensor): 标量形式的平均 RMS 光斑半径 [mm]。
        """
        # 计算光线质心作为参考
        if center_ref is None:
            with torch.no_grad():
                center_ref = self.centroid()

        center_ref = center_ref.unsqueeze(-2)

        # 计算各区域的 RMS 误差
        rms_error = ((self.o[..., :2] - center_ref[..., :2]) ** 2).sum(-1)
        rms_error = (rms_error * self.is_valid).sum(-1) / (
            self.is_valid.sum(-1) + EPSILON
        )
        rms_error = rms_error.sqrt()

        # 对 RMS 误差求平均
        return rms_error.mean()

    def flip_xy(self):
        """原地取反光线原点和方向的 x、y 分量。

        用于计算点扩散函数和波前分布。

        返回：
            self (Ray): 更新后的光线，可用于链式调用。
        """
        self.o = torch.cat([-self.o[..., :2], self.o[..., 2:]], dim=-1)
        self.d = torch.cat([-self.d[..., :2], self.d[..., 2:]], dim=-1)
        return self

    def clone(self, device=None):
        """返回光线的深拷贝，并可选择放置到不同设备上。

        适用于将光线保存在 CPU 上，仅在需要时再移动到 GPU。

        参数：
            device (str or None, optional): 克隆对象的目标设备。为 None 时使用
                源光线设备，默认为 None。

        返回：
            ray (Ray): 张量已克隆到目标设备的新光线。
        """
        target_device = self.device if device is None else device

        ray = Ray.__new__(Ray)
        ray.o = self.o.clone().to(target_device)
        ray.d = self.d.clone().to(target_device)
        ray.wvln = self.wvln.clone().to(target_device)
        ray.is_valid = self.is_valid.clone().to(target_device)
        ray.en = self.en.clone().to(target_device)
        ray.bend_penalty = self.bend_penalty.clone().to(target_device)
        ray.opl = self.opl.clone().to(target_device)

        ray.is_coherent = self.is_coherent
        ray.device = target_device
        ray.shape = ray.o.shape[:-1]

        return ray

    def squeeze(self, dim=None):
        """原地压缩所有光线张量的批次维度。

        波长 `wvln` 是标量张量，因此保持不变。

        参数：
            dim (int, optional): 要压缩的维度。为 None 时移除所有大小为 1
                的维度，默认为 None。

        返回：
            self (Ray): 更新后的光线，可用于链式调用。
        """
        self.o = self.o.squeeze(dim)
        self.d = self.d.squeeze(dim)
        # wvln 是单元素张量，无需压缩
        self.is_valid = self.is_valid.squeeze(dim)
        self.en = self.en.squeeze(dim)
        self.opl = self.opl.squeeze(dim)
        self.bend_penalty = self.bend_penalty.squeeze(dim)
        return self

    def unsqueeze(self, dim=None):
        """在所有光线张量中原地插入大小为 1 的批次维度。

        波长 `wvln` 是标量张量，因此保持不变。

        参数：
            dim (int): 插入新维度的位置。实际必须传入 int；默认值 None 不是
                `torch.unsqueeze` 的有效参数。

        返回：
            self (Ray): 更新后的光线，可用于链式调用。
        """
        self.o = self.o.unsqueeze(dim)
        self.d = self.d.unsqueeze(dim)
        # wvln 是单元素张量，无需扩维
        self.is_valid = self.is_valid.unsqueeze(dim)
        self.en = self.en.unsqueeze(dim)
        self.opl = self.opl.unsqueeze(dim)
        self.bend_penalty = self.bend_penalty.unsqueeze(dim)
        return self
