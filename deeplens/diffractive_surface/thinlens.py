# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""不含任何色差的理想薄透镜。"""

import torch
import torch.nn.functional as F
from .diffractive import DiffractiveSurface


class ThinLens(DiffractiveSurface):
    """以衍射表面建模的理想薄透镜。

    施加一个焦距为 `f0`、由所有波长共享的二次（抛物线）透镜相位，因此各波长
    都聚焦到同一点（无色差）。与基类 `DiffractiveSurface` 不同，这里的相位
    不会按材料色散重新缩放。

    属性：
        f0 (torch.Tensor): 以标量张量表示的焦距。[mm]
    """

    def __init__(
        self,
        d,
        f0=None,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """初始化薄透镜。

        参数：
            d (float): 透镜表面沿光轴的位置。[mm]
            f0 (float or None, optional): 初始焦距。[mm] 若为 None，则采样一个
                很大的随机焦距（量级约为 1e6 mm）。默认值为 None。
            res (tuple or int, optional): 透镜分辨率，格式为 (H, W)。[pixel]
                整数会扩展为方形分辨率。默认值为 (2000, 2000)。
            mat (str, optional): 透镜材料。默认值为 "fused_silica"。
            fab_ps (float, optional): 制造像素尺寸。[mm] 默认值为 0.001。
            fab_step (int, optional): 制造量化级数。默认值为 16。
            device (str, optional): 运行透镜的设备。默认值为 "cpu"。
        """
        super().__init__(d=d, res=res, mat=mat, fab_ps=fab_ps, fab_step=fab_step, device=device)

        # 初始焦距
        if f0 is None:
            self.f0 = (
                torch.randn(1, device=self.device) * 1e6
            )  # [mm]，初始为很大的焦距
        else:
            self.f0 = torch.tensor(f0, device=self.device)

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从字典初始化薄透镜。

        参数：
            doe_dict (dict): 表面参数。必须包含键 `d` 和 `res`；可选键为
                `f0`、`mat`、`fab_ps`、`fab_step`。

        返回：
            surface (ThinLens): 构造得到的薄透镜。
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            f0=doe_dict.get("f0", None),
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
        )

    def get_phase_map(self, wvln):
        """计算给定波长下的透镜相位图。

        施加二次薄透镜相位

        $$\\phi(x, y) = -\\frac{\\pi (x^2 + y^2)}{f_0\\, \\lambda}$$

        其中 $\\lambda$ 是以 mm 表示的波长。所有波长都使用同一焦距 `f0`
        （与基类不同，不进行色散缩放）。结果会包裹到 $[0, 2\\pi)$，并重采样
        到 `self.res`。

        参数：
            wvln (float): 波长。[um]

        返回：
            phase_map (torch.Tensor): 包裹后的相位图，shape 为 [H, W]，范围为
                $[0, 2\\pi)$. [rad]
        """

        # 所有波长使用相同焦距
        wvln_mm = wvln * 1e-3
        phase_map = -2 * torch.pi * (self.x**2 + self.y**2) / (2 * self.f0 * wvln_mm)
        self._warn_if_undersampled(phase_map, self.f0, wvln)
        phase_map = torch.remainder(phase_map, 2 * torch.pi)

        # 插值到目标分辨率
        phase_map = (
            F.interpolate(
                phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
            )
            .squeeze(0)
            .squeeze(0)
        )

        return phase_map

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.1):
        """为焦距构建优化器参数组。

        启用 `f0` 的梯度，并将其放入单个参数组。

        参数：
            lr (float, optional): `f0` 的学习率。默认值为 0.1。

        返回：
            optimizer_params (list): 包含一个参数组字典的列表
                `{"params": [f0], "lr": lr}`.
        """
        self.f0.requires_grad = True
        optimizer_params = [{"params": [self.f0], "lr": lr}]
        return optimizer_params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """返回描述该表面的可序列化字典。

        在基础表面字典中加入焦距 `f0`。

        返回：
            surf_dict (dict): 表面参数，包括 `f0`（float，[mm]）。
        """
        surf_dict = super().surf_dict()
        surf_dict["f0"] = self.f0.item()
        return surf_dict
