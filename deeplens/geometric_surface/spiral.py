# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

import torch

from .base import Surface, EPSILON


class Spiral(Surface):
    """螺旋屈光度自由曲面。

    该自由曲面的矢高绕光轴呈螺旋变化，从而产生连续变化的多焦行为。令
    $\\theta = \\mathrm{atan2}(y, x)$，归一化半径平方为
    $\\phi^2 = (x^2 + y^2) / r^2$，则矢高为

    $$
    z(x, y) = \\frac{c_1}{2}\\left(1 + \\cos(N\\theta + \\eta\\phi^2)\\right)
            + \\frac{c_2}{2}\\left(1 - \\cos(N\\theta + \\eta\\phi^2)\\right)
    $$

    其中长度单位均为 [mm]。

    属性：
        c1 (torch.Tensor): 标量矢高振幅项 [mm]。
        c2 (torch.Tensor): 标量矢高振幅项 [mm]。
        N (int): 螺旋臂数量（角频率）。
        eta (float): 控制螺旋紧密程度的径向扭转参数。

    参考文献：
        Spiral diopter: freeform lenses with enhanced multifocal behavior, Optica 2024.
    """

    def __init__(self, r, d, c1, c2, mat2, N=1, eta=5, is_square=False, device="cpu"):
        """初始化 Spiral 表面。

        参数：
            r (float): 表面半径（半孔径）[mm]。
            d (float): 沿光轴到下一表面的距离 [mm]。
            c1 (float): 矢高振幅项 [mm]。
            c2 (float): 矢高振幅项 [mm]。
            mat2 (str): 表面后介质的材料。
            N (int, optional): 螺旋臂数量（角频率）。默认值为 1。
            eta (float, optional): 控制螺旋紧密程度的径向扭转参数。默认值为 5。
            is_square (bool, optional): 孔径是否为方形。默认值为 False。
            device (str, optional): torch 张量使用的设备。默认值为 "cpu"。
        """
        super().__init__(r, d, mat2, is_square=is_square, device=device)
        self.c1 = torch.tensor(c1, dtype=torch.float32, device=device)
        self.c2 = torch.tensor(c2, dtype=torch.float32, device=device)
        self.N = N
        self.eta = eta
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从参数字典初始化 Spiral 表面。

        参数：
            surf_dict (dict): 表面参数。必须包含键 `r`、`d`、`c1`、`c2`、
                `mat2`；可选键为 `N`（默认 1）、`eta`（默认 5）、
                `is_square`（默认 False）。

        返回：
            surface (Spiral): 构造得到的螺旋表面。
        """
        return cls(
            surf_dict["r"],
            surf_dict["d"],
            surf_dict["c1"],
            surf_dict["c2"],
            surf_dict["mat2"],
            surf_dict.get("N", 1),
            surf_dict.get("eta", 5),
            surf_dict.get("is_square", False),
        )

    def _sag(self, x, y):
        """计算螺旋表面的矢高 z(x, y)。

        其中 $\\theta = \\mathrm{atan2}(y, x)$，且
        $\\phi^2 = (x^2 + y^2) / r^2$：

        $$
        z = \\frac{c_1}{2}\\left(1 + \\cos(N\\theta + \\eta\\phi^2)\\right)
          + \\frac{c_2}{2}\\left(1 - \\cos(N\\theta + \\eta\\phi^2)\\right)
        $$

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            sag (torch.Tensor): 表面高度 z [mm]，shape 与 `x` 相同。

        参考文献：
            Spiral diopter: freeform lenses with enhanced multifocal behavior, Optica 2024.
        """
        theta = torch.atan2(y, x)  # [-pi, pi]
        phi_norm_sq = (x**2 + y**2) / self.r**2
        common_cos = torch.cos(self.N * theta + self.eta * phi_norm_sq)
        z1 = self.c1 / 2 * (1 + common_cos)
        z2 = self.c2 / 2 * (1 - common_cos)
        return z1 + z2

    def _dfdxy(self, x, y):
        """计算矢高相对于 x 和 y 的偏导数。

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            dfdx (torch.Tensor): 偏导数 dz/dx [无量纲]，shape 与 `x` 相同。
            dfdy (torch.Tensor): 偏导数 dz/dy [无量纲]，shape 与 `x` 相同。
        """
        phi_sq = x**2 + y**2
        phi_norm_sq = phi_sq / (self.r**2 + EPSILON)
        theta = torch.atan2(y, x)

        # 余弦函数的自变量
        u = self.N * theta + self.eta * phi_norm_sq

        # 公共项：(c2-c1)/2 * sin(u)
        common_term = (self.c1 - self.c2) / 2 * (-torch.sin(u))

        # 避免除零
        inv_phi_sq = 1.0 / (phi_sq + EPSILON)

        # d(u)/dx
        du_dx = -self.N * y * inv_phi_sq + 2 * self.eta * x / self.r**2
        dfdx = common_term * du_dx

        # d(u)/dy
        du_dy = self.N * x * inv_phi_sq + 2 * self.eta * y / self.r**2
        dfdy = common_term * du_dy

        return dfdx, dfdy

    # =========================================
    # 优化
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-4], optim_mat=False):
        """返回该表面的优化器参数组。

        启用表面距离 `d` 以及矢高振幅 `c1`、`c2` 的梯度，并为每个参数分配
        独立学习率。

        参数：
            lrs (list, optional): `[d, c1, c2]` 的学习率。
                默认值为 [1e-4, 1e-4, 1e-4]。
            optim_mat (bool, optional): 是否优化材料参数。螺旋表面不支持。
                默认值为 False。

        返回：
            params (list): torch 优化器的参数组字典列表。

        异常：
            ValueError: 当 `optim_mat` 为 True 时抛出。
        """
        params = []

        # 优化距离
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # 优化 c1
        self.c1.requires_grad_(True)
        params.append({"params": [self.c1], "lr": lrs[1]})

        # 优化 c2
        self.c2.requires_grad_(True)
        params.append({"params": [self.c2], "lr": lrs[2]})

        # 螺旋表面不优化材料参数。
        if optim_mat:
            raise ValueError("Material parameters are not optimized for spiral surface.")

        return params

    # =========================================
    # 输入输出
    # =========================================
    def surf_dict(self):
        """以可序列化字典形式返回表面参数。

        在基础表面字典中加入螺旋矢高振幅。

        返回：
            s_dict (dict): 表面参数，包括 float 类型的 `c1` 和 `c2`。
        """
        s_dict = super().surf_dict()
        s_dict.update(
            {
                "c1": self.c1.item(),
                "c2": self.c2.item(),
            }
        )
        return s_dict
