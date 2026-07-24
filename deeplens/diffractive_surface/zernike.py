# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Zernike DOE 参数化。"""

import math
import torch
from .diffractive import DiffractiveSurface


class Zernike(DiffractiveSurface):
    """由 Zernike 多项式参数化的衍射光学元件。

    DOE 表面相位表示为单位圆盘上前 37 项 Zernike 多项式（OSA/ANSI 顺序）
    的加权和。可学习系数 `z_coeff` 是唯一的优化参数。

    属性：
        zernike_order (int): Zernike 项数（固定为 37）。
        z_coeff (torch.Tensor): Zernike 系数，shape 为 (zernike_order,)。
    """

    def __init__(
        self,
        d,
        z_coeff=None,
        zernike_order=37,
        res=(2000, 2000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        wvln0=0.55,
        device="cpu",
    ):
        """初始化由 Zernike 参数化的 DOE。

        参数：
            d (float): DOE 沿光轴的位置。[mm]
            z_coeff (torch.Tensor or None, optional): Zernike 系数，shape 为
                (zernike_order,)。若为 None，则用缩放 1e-3 的随机值初始化。
                默认值为 None。
            zernike_order (int, optional): Zernike 系数数量。目前仅支持 37。
                默认值为 37。
            res (tuple, optional): DOE 分辨率，以像素表示为 (H, W)。默认值为
                (2000, 2000)。
            mat (str, optional): DOE 基底材料。默认值为 "fused_silica"。
            fab_ps (float, optional): 制造像素尺寸。[mm] 默认值为 0.001。
            fab_step (int, optional): 制造量化级数。默认值为 16。
            wvln0 (float, optional): 设计波长。[um] 默认值为 0.55。
            device (str, optional): 计算设备。默认值为 "cpu"。

        异常：
            AssertionError: 当 zernike_order 不为 37 时抛出。
        """
        super().__init__(
            d=d, res=res, mat=mat, fab_ps=fab_ps, fab_step=fab_step, wvln0=wvln0, device=device
        )

        # 使用随机值初始化 Zernike 系数
        assert zernike_order == 37, "Currently, Zernike DOE only supports 37 orders"
        self.zernike_order = zernike_order
        if z_coeff is None:
            self.z_coeff = torch.randn(zernike_order, device=self.device) * 1e-3
        else:
            self.z_coeff = z_coeff

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从序列化的表面字典初始化 Zernike DOE。

        参数：
            doe_dict (dict): 表面参数。必须包含 "d" 和 "res"；可选键 "mat"、
                "fab_ps"、"fab_step"、"z_coeff"、"zernike_order"、"wvln0"
                缺省时使用默认值。

        返回：
            zernike (Zernike): 构造得到的 Zernike DOE。
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            z_coeff=doe_dict.get("z_coeff", None),
            zernike_order=doe_dict.get("zernike_order", 37),
            wvln0=doe_dict.get("wvln0", 0.55),
        )

    def phase_func(self):
        """计算设计波长下的 DOE 相位图。

        返回：
            phase (torch.Tensor): 根据单位圆盘上的 Zernike 系数计算得到的相位图，
                shape 为 (res[0], res[0])，单位为 rad。
        """
        return calculate_zernike_phase(self.z_coeff, grid=self.res[0])

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """为 Zernike 系数构建优化器参数组。

        同时将 `z_coeff` 设为需要梯度。

        参数：
            lr (float, optional): 系数的学习率。默认值为 0.01。

        返回：
            optimizer_params (list): 单个参数组字典，包含键 "params"
                （`z_coeff` 张量）和 "lr"。
        """
        self.z_coeff.requires_grad = True
        optimizer_params = [{"params": [self.z_coeff], "lr": lr}]
        return optimizer_params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """将 DOE 表面序列化为字典。

        在基础表面字典中加入已移至 CPU 并分离的 Zernike 系数，以及 Zernike 阶数。

        返回：
            surf_dict (dict): 包含 "z_coeff" 和 "zernike_order" 的表面参数。
        """
        surf_dict = super().surf_dict()
        surf_dict["z_coeff"] = self.z_coeff.clone().detach().cpu()
        surf_dict["zernike_order"] = self.zernike_order
        return surf_dict


def calculate_zernike_phase(z_coeff, grid=256):
    """根据 Zernike 多项式的加权和计算相位图。

    在正方形 $[-1, 1]^2$ 的 `grid` x `grid` 采样网格上计算前 37 项归一化
    Zernike 多项式（OSA/ANSI 顺序），并按 `z_coeff` 加权累加。单位圆盘外
    （$r^2 > 1$）的采样点置零。

    参数：
        z_coeff (torch.Tensor): shape 为 (37,) 的 Zernike 系数。
        grid (int, optional): 方形采样网格的像素边长。默认值为 256。

    返回：
        phase (torch.Tensor): shape 为 (grid, grid) 的相位图，并以单位圆盘为掩膜。
    """
    device = z_coeff.device

    # 生成网格
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, grid, device=device),
        torch.linspace(1, -1, grid, device=device),
        indexing="xy",
    )

    # 预计算径向幂（各幂只计算一次，并在不同项间复用）
    r2 = x * x + y * y
    r = torch.sqrt(r2)
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r4 * r2
    r7 = r6 * r
    r8 = r4 * r4

    # 使用和角递推预计算三角函数项
    # 由 atan2 得到 sin(alpha)、cos(alpha)
    alpha = torch.atan2(y, x)
    s1 = torch.sin(alpha)
    c1 = torch.cos(alpha)
    # sin(2a) = 2*sin(a)*cos(a)，cos(2a) = 2*cos²(a) - 1
    s2 = 2 * s1 * c1
    c2 = 2 * c1 * c1 - 1
    # sin(3a) = sin(2a)*cos(a) + cos(2a)*sin(a)，其余项依此类推。
    s3 = s2 * c1 + c2 * s1
    c3 = c2 * c1 - s2 * s1
    s4 = s3 * c1 + c3 * s1
    c4 = c3 * c1 - s3 * s1
    s5 = s4 * c1 + c4 * s1
    c5 = c4 * c1 - s4 * s1
    s6 = s5 * c1 + c5 * s1
    c6 = c5 * c1 - s5 * s1
    s7 = s6 * c1 + c6 * s1
    c7 = c6 * c1 - s6 * s1

    # 预计算共享的径向多项式
    sqrt3 = math.sqrt(3)
    sqrt5 = math.sqrt(5)
    sqrt6 = math.sqrt(6)
    sqrt7 = math.sqrt(7)
    sqrt8 = math.sqrt(8)
    sqrt10 = math.sqrt(10)
    sqrt12 = math.sqrt(12)
    sqrt14 = math.sqrt(14)

    poly_3r3_2r = 3 * r3 - 2 * r
    poly_4r4_3r2 = 4 * r4 - 3 * r2
    poly_10r5_12r3_3r = 10 * r5 - 12 * r3 + 3 * r
    poly_5r5_4r3 = 5 * r5 - 4 * r3
    poly_15r6_20r4_6r2 = 15 * r6 - 20 * r4 + 6 * r2
    poly_6r6_5r4 = 6 * r6 - 5 * r4
    poly_35r7_60r5_30r3 = 35 * r7 - 60 * r5 + 30 * r3
    poly_21r7_30r5_10r3 = 21 * r7 - 30 * r5 + 10 * r3
    poly_7r7_6r5 = 7 * r7 - 6 * r5

    # 直接累加 Zernike 项（避免创建 37 个中间张量）
    c = z_coeff
    ZW = c[0] * 1
    ZW = ZW + c[1] * (2 * r * s1)
    ZW = ZW + c[2] * (2 * r * c1)
    ZW = ZW + c[3] * (sqrt3 * (2 * r2 - 1))
    ZW = ZW + c[4] * (sqrt6 * r2 * s2)
    ZW = ZW + c[5] * (sqrt6 * r2 * c2)
    ZW = ZW + c[6] * (sqrt8 * poly_3r3_2r * s1)
    ZW = ZW + c[7] * (sqrt8 * poly_3r3_2r * c1)
    ZW = ZW + c[8] * (sqrt8 * r3 * s3)
    ZW = ZW + c[9] * (sqrt8 * r3 * c3)
    ZW = ZW + c[10] * (sqrt5 * (6 * r4 - 6 * r2 + 1))
    ZW = ZW + c[11] * (sqrt10 * poly_4r4_3r2 * c2)
    ZW = ZW + c[12] * (sqrt10 * poly_4r4_3r2 * s2)
    ZW = ZW + c[13] * (sqrt10 * r4 * c4)
    ZW = ZW + c[14] * (sqrt10 * r4 * s4)
    ZW = ZW + c[15] * (sqrt12 * poly_10r5_12r3_3r * c1)
    ZW = ZW + c[16] * (sqrt12 * poly_10r5_12r3_3r * s1)
    ZW = ZW + c[17] * (sqrt12 * poly_5r5_4r3 * c3)
    ZW = ZW + c[18] * (sqrt12 * poly_5r5_4r3 * s3)
    ZW = ZW + c[19] * (sqrt12 * r5 * c5)
    ZW = ZW + c[20] * (sqrt12 * r5 * s5)
    ZW = ZW + c[21] * (sqrt7 * (20 * r6 - 30 * r4 + 12 * r2 - 1))
    ZW = ZW + c[22] * (sqrt14 * poly_15r6_20r4_6r2 * s2)
    ZW = ZW + c[23] * (sqrt14 * poly_15r6_20r4_6r2 * c2)
    ZW = ZW + c[24] * (sqrt14 * poly_6r6_5r4 * s4)
    ZW = ZW + c[25] * (sqrt14 * poly_6r6_5r4 * c4)
    ZW = ZW + c[26] * (sqrt14 * r6 * s6)
    ZW = ZW + c[27] * (sqrt14 * r6 * c6)
    ZW = ZW + c[28] * (4 * (poly_35r7_60r5_30r3 - 4) * s1)
    ZW = ZW + c[29] * (4 * (poly_35r7_60r5_30r3 - 4) * c1)
    ZW = ZW + c[30] * (4 * poly_21r7_30r5_10r3 * s3)
    ZW = ZW + c[31] * (4 * poly_21r7_30r5_10r3 * c3)
    ZW = ZW + c[32] * (4 * poly_7r7_6r5 * s5)
    ZW = ZW + c[33] * (4 * poly_7r7_6r5 * c5)
    ZW = ZW + c[34] * (4 * r7 * s7)
    ZW = ZW + c[35] * (4 * r7 * c7)
    ZW = ZW + c[36] * (3 * (70 * r8 - 140 * r6 + 90 * r4 - 20 * r2 + 1))

    # 应用圆形掩膜（复用 r2，避免重新计算 x**2 + y**2）
    ZW = torch.where(r2 <= 1, ZW, torch.zeros(1, device=device))

    return ZW
