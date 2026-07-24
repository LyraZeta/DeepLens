# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""光栅 DOE 参数化。

本模块实现线性光栅衍射光学元件（DOE）。光栅在表面引入线性相位梯度，
使光衍射到多个衍射级次。
"""

import torch
from .diffractive import DiffractiveSurface


class Grating(DiffractiveSurface):
    """线性光栅衍射光学元件。

    光栅在表面引入线性相位梯度，使光衍射到多个衍射级次。相位分布为

    $$\\phi(x, y) = \\alpha \\,\\frac{x \\sin\\theta + y \\cos\\theta}{\\text{norm\\_radii}}$$

    其中 $\\theta$ 是从 y 轴到光栅矢量的夹角，$\\alpha$ 是光栅斜率
    （相位梯度强度），`norm_radii` 用于归一化坐标。

    属性：
        theta (torch.Tensor): 从 y 轴到光栅矢量的夹角。[rad]
        alpha (torch.Tensor): 光栅斜率（相位梯度强度）。[rad]
        norm_radii (float): 坐标归一化半径（DOE 宽度的一半）。[mm]
    """

    def __init__(
        self,
        d,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        theta=0.0,
        alpha=0.0,
        device="cpu",
    ):
        """初始化光栅 DOE。

        参数：
            d (float): DOE 平面的轴向位置。[mm]
            res (tuple or int, optional): DOE 分辨率，格式为 (H, W)；整数会
                扩展为 (res, res)。[pixel]。默认值为 (2000, 2000)。
            mat (str, optional): DOE 的材料名称。默认值为 "fused_silica"。
            wvln0 (float, optional): 设计波长。[um]。默认值为 0.55。
            fab_ps (float, optional): 制造像素尺寸。[mm]。默认值为 0.001。
            fab_step (int, optional): 制造（量化）级数。默认值为 16。
            theta (float, optional): 从 y 轴到光栅矢量的夹角。[rad]。
                默认值为 0.0。
            alpha (float, optional): 光栅斜率（相位梯度强度）。[rad]。
                默认值为 0.0。
            device (str, optional): 放置 DOE 张量的设备。默认值为 "cpu"。
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps, fab_step=fab_step, device=device
        )

        # 光栅参数
        self.theta = torch.tensor(theta)  # 从 y 轴到光栅矢量的夹角
        self.alpha = torch.tensor(alpha)  # 光栅斜率

        # 归一化半径（使用宽度的一半）
        self.norm_radii = self.w / 2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从参数字典初始化光栅 DOE。

        参数：
            doe_dict (dict): DOE 参数字典。必须包含 "d" 和 "res"；"mat"、
                "wvln0"、"fab_ps"、"fab_step"、"theta"、"alpha" 为可选键，
                缺省时使用默认值。

        返回：
            grating (Grating): 构造得到的光栅 DOE 实例。
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            theta=doe_dict.get("theta", 0.0),
            alpha=doe_dict.get("alpha", 0.0),
        )

    def phase_func(self):
        """计算设计波长下的原始光栅相位分布。

        相位是位置的线性函数：

        $$\\phi(x, y) = \\alpha \\,\\frac{x \\sin\\theta + y \\cos\\theta}{\\text{norm\\_radii}}$$

        返回：
            phase (torch.Tensor): 设计波长下未包裹的原始相位分布。
                [H, W]。[rad]
        """
        # 归一化坐标
        x_norm = self.x / self.norm_radii
        y_norm = self.y / self.norm_radii

        # 计算线性相位梯度
        phase = self.alpha * (
            x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
        )

        return phase

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """为光栅参数构建优化器参数组。

        启用 `theta` 和 `alpha` 的梯度。`alpha` 参数组使用相对于 `lr`
        放大 10x 的学习率。

        参数：
            lr (float, optional): 光栅参数的基础学习率。默认值为 0.001。

        返回：
            optimizer_params (list): 优化器参数组字典的列表。
        """
        self.theta.requires_grad = True
        self.alpha.requires_grad = True

        optimizer_params = [
            {"params": [self.theta], "lr": lr},
            {"params": [self.alpha], "lr": lr * 10},
        ]

        return optimizer_params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """返回可序列化的光栅表面参数字典。

        在基础表面字典中加入光栅专用的 `theta`、`alpha` 和 `norm_radii` 项。

        返回：
            surf_dict (dict): 表面参数字典。
        """
        surf_dict = super().surf_dict()
        surf_dict["theta"] = round(self.theta.item(), 6)
        surf_dict["alpha"] = round(self.alpha.item(), 6)
        surf_dict["norm_radii"] = round(self.norm_radii, 6)
        return surf_dict

    def save_ckpt(self, save_path="./grating_doe.pth"):
        """将光栅 DOE 参数保存到检查点文件。

        参数：
            save_path (str, optional): 检查点写入路径。默认值为
                "./grating_doe.pth"。
        """
        torch.save(
            {
                "param_model": "grating",
                "theta": self.theta.clone().detach().cpu(),
                "alpha": self.alpha.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./grating_doe.pth"):
        """从检查点文件加载光栅 DOE 参数。

        将 `theta` 和 `alpha` 恢复到当前设备。

        参数：
            load_path (str, optional): 检查点读取路径。默认值为
                "./grating_doe.pth"。
        """
        ckpt = torch.load(load_path)
        self.theta = ckpt["theta"].to(self.device)
        self.alpha = ckpt["alpha"].to(self.device)
