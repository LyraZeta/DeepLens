# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""菲涅耳 DOE。与折射透镜相比，相位菲涅耳透镜具有反向色散特性。

参考资料：
    [1] https://www.nikonusa.com/learn-and-explore/c/ideas-and-inspiration/phase-fresnel-from-wildlife-photography-to-portraiture
"""

import torch
from .diffractive import DiffractiveSurface


class Fresnel(DiffractiveSurface):
    """相位菲涅耳衍射透镜表面。

    该衍射菲涅耳透镜具有理想的二次（薄透镜）相位分布。与折射透镜相比，
    它呈现反向色散，其唯一自由参数为设计波长下的焦距 `f0`。

    属性：
        f0 (torch.Tensor): 设计波长下的焦距，标量。[mm]
        r2 (torch.Tensor): 缓存的径向坐标平方网格 $x^2 + y^2$，
            shape 为 [H, W]。[mm^2]
    """

    def __init__(
        self,
        d,
        f0=None,
        wvln0=0.55,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """初始化相位菲涅耳衍射透镜。

        该透镜施加由 `f0` 决定的理想薄透镜二次相位。与折射透镜相比，
        它呈现反向色散。

        参数：
            d (float): DOE 表面的轴向位置。[mm]
            f0 (float or None, optional): 设计波长下的焦距。[mm] 若为 None，
                则初始化为接近无穷大的随机值。默认值为 None。
            wvln0 (float, optional): 设计波长。[um] 默认值为 0.55。
            res (tuple or int, optional): DOE 分辨率，[w, h]。[pixel]
                默认值为 (2000, 2000)。
            mat (str, optional): DOE 材料。默认值为 "fused_silica"。
            fab_ps (float, optional): 制造像素尺寸。[mm] 默认值为 0.001。
            fab_step (int, optional): 制造量化级数。默认值为 16。
            device (str, optional): 运行 DOE 的设备。默认值为 "cpu"。
        """
        super().__init__(
            d=d, res=res, wvln0=wvln0, mat=mat, fab_ps=fab_ps, fab_step=fab_step, device=device
        )

        # 初始焦距
        if f0 is None:
            self.f0 = torch.randn(1) * 1e6
        else:
            self.f0 = torch.tensor(f0)

        # 缓存静态 r² 网格（初始化后 x、y 不再变化）
        self.r2 = self.x**2 + self.y**2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从表面参数字典初始化菲涅耳 DOE。

        参数：
            doe_dict (dict): 表面参数。必须包含 "d" 和 "res"；可选键为
                "f0"、"wvln0"、"mat"、"fab_ps"、"fab_step"。

        返回：
            doe (Fresnel): 构造得到的菲涅耳 DOE。
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            f0=doe_dict.get("f0", None),
            wvln0=doe_dict.get("wvln0", 0.55),
            mat=doe_dict.get("mat", "fused_silica"),
        )

    def phase_func(self):
        """计算设计波长下的原始（未包裹）二次相位。

        施加理想薄透镜相位

        $$\\phi(x, y) = -\\frac{\\pi (x^2 + y^2)}{f_0 \\lambda_0}$$

        其中 $\\lambda_0$ 是换算为 mm 的设计波长。若当前网格对相位采样不足，
        则发出一次性警告。

        返回：
            phase (torch.Tensor): 未包裹的原始相位，shape 为 [H, W]。[rad]
        """
        wvln0_mm = self.wvln0 * 1e-3
        phase = -2 * torch.pi * self.r2 / (2 * self.f0 * wvln0_mm)
        self._warn_if_undersampled(phase, self.f0, self.wvln0)
        return phase

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """为焦距 `f0` 构建优化器参数组。

        启用 `f0` 的梯度，并将其作为单个参数组返回。

        参数：
            lr (float, optional): `f0` 的学习率。默认值为 0.001。

        返回：
            optimizer_params (list): 包含一个 `f0` 参数组字典的列表。
        """
        self.f0.requires_grad = True
        optimizer_params = [{"params": [self.f0], "lr": lr}]
        return optimizer_params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """将表面（包括 `f0` 和 `wvln0`）序列化为字典。

        返回：
            surf_dict (dict): 基础表面参数加上 "f0" [mm]，其中 "wvln0" [um]
                会被未舍入的值覆盖。
        """
        surf_dict = super().surf_dict()
        surf_dict["f0"] = self.f0.item()
        surf_dict["wvln0"] = self.wvln0
        return surf_dict
