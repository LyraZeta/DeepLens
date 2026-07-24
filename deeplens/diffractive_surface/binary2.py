# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Binary2 DOE 参数化。"""

import torch
from .diffractive import DiffractiveSurface


class Binary2(DiffractiveSurface):
    """Binary2（Zemax 风格）旋转对称 DOE 表面。

    将设计波长下的相位参数化为径向坐标的偶次多项式
    $\\phi(r) = \\pi \\sum_{i=1}^{5} \\alpha_{2i}\\, r^{2i}$，系数为
    `alpha2`、`alpha4`、`alpha6`、`alpha8`、`alpha10`。径向网格会被缓存，
    因此只需优化这五个标量系数。

    属性：
        alpha2 (torch.Tensor): $r^2$ 的系数。标量张量，shape 为 [1]。
        alpha4 (torch.Tensor): $r^4$ 的系数。标量张量，shape 为 [1]。
        alpha6 (torch.Tensor): $r^6$ 的系数。标量张量，shape 为 [1]。
        alpha8 (torch.Tensor): $r^8$ 的系数。标量张量，shape 为 [1]。
        alpha10 (torch.Tensor): $r^{10}$ 的系数。标量张量，shape 为 [1]。
        x (torch.Tensor): 像素的 x 坐标。[H, W]。[mm]
        y (torch.Tensor): 像素的 y 坐标。[H, W]。[mm]
        r2 (torch.Tensor): 缓存的半径平方 $x^2 + y^2$。[H, W]。[mm^2]
    """

    def __init__(
        self,
        d,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        device="cpu",
    ):
        """使用较小的随机多项式系数初始化 Binary2 DOE。

        参数：
            d (float): DOE 表面的轴向位置。[mm]
            res (tuple or int, optional): 分辨率，格式为 (H, W)；整数会扩展为
                (res, res)。[pixel]。默认值为 (2000, 2000)。
            mat (str, optional): DOE 材料名称。默认值为 "fused_silica"。
            wvln0 (float, optional): 设计波长。[um]。默认值为 0.55。
            fab_ps (float, optional): 制造像素尺寸。[mm]。默认值为 0.001。
            fab_step (int, optional): 制造量化级数。默认值为 16。
            is_square (bool, optional): 孔径是否为方形。默认值为 True。
            device (str, optional): 存放张量的设备。默认值为 "cpu"。
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps, fab_step=fab_step,
            is_square=is_square, device=device,
        )

        # 使用较小的随机值初始化
        self.alpha2 = (torch.rand(1) - 0.5) * 0.02
        self.alpha4 = (torch.rand(1) - 0.5) * 0.002
        self.alpha6 = (torch.rand(1) - 0.5) * 0.0002
        self.alpha8 = (torch.rand(1) - 0.5) * 0.00002
        self.alpha10 = (torch.rand(1) - 0.5) * 0.000002

        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, self.res[1]),
            torch.linspace(self.h / 2, -self.h / 2, self.res[0]),
            indexing="xy",
        )

        # 缓存静态 r² 网格（初始化后 x、y 不再变化）
        self.r2 = self.x**2 + self.y**2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从序列化的表面字典初始化 Binary2 DOE。

        参数：
            doe_dict (dict): 表面字典。必须包含 "d" 和 "res"；可选键 "mat"、
                "wvln0"、"fab_ps"、"fab_step"、"is_square" 缺省时使用构造函数默认值。

        返回：
            doe (Binary2): 构造得到的 Binary2 表面。
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            is_square=doe_dict.get("is_square", True),
        )

    def phase_func(self):
        """计算设计波长下的原始（未包裹）相位。

        在缓存的 $r^2$ 网格上使用 Horner 法计算
        $\\phi(r) = \\pi\\,(\\alpha_2 r^2 + \\alpha_4 r^4 + \\alpha_6 r^6
        + \\alpha_8 r^8 + \\alpha_{10} r^{10})$。

        返回：
            phase (torch.Tensor): 原始相位图。[H, W]。[rad]
        """
        # Horner 法：r2*(a2 + r2*(a4 + r2*(a6 + r2*(a8 + r2*a10))))
        r2 = self.r2
        phase = torch.pi * r2 * (
            self.alpha2
            + r2 * (self.alpha4 + r2 * (self.alpha6 + r2 * (self.alpha8 + r2 * self.alpha10)))
        )
        return phase

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """启用梯度并为各系数构建优化器参数组。

        高阶系数使用逐级增大的学习率（从 `alpha2` 到 `alpha10` 依次为
        `lr`、10x、100x、1000x、10000x），以补偿其较小的量级。

        参数：
            lr (float): `alpha2` 的基础学习率。默认值为 0.001。

        返回：
            optimizer_params (list): 参数组字典列表，每个系数对应一组，
                各组包含键 "params" 和 "lr"。
        """
        self.alpha2.requires_grad = True
        self.alpha4.requires_grad = True
        self.alpha6.requires_grad = True
        self.alpha8.requires_grad = True
        self.alpha10.requires_grad = True

        optimizer_params = [
            {"params": [self.alpha2], "lr": lr},
            {"params": [self.alpha4], "lr": lr * 10},
            {"params": [self.alpha6], "lr": lr * 100},
            {"params": [self.alpha8], "lr": lr * 1000},
            {"params": [self.alpha10], "lr": lr * 10000},
        ]

        return optimizer_params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """将表面及其多项式系数序列化为字典。

        返回：
            surf_dict (dict): 在基础表面字典中加入五个经过舍入的系数
                "alpha2"、"alpha4"、"alpha6"、"alpha8"、"alpha10"。
        """
        surf_dict = super().surf_dict()
        surf_dict["alpha2"] = round(self.alpha2.item(), 6)
        surf_dict["alpha4"] = round(self.alpha4.item(), 6)
        surf_dict["alpha6"] = round(self.alpha6.item(), 6)
        surf_dict["alpha8"] = round(self.alpha8.item(), 6)
        surf_dict["alpha10"] = round(self.alpha10.item(), 6)
        return surf_dict
