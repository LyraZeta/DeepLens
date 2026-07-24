# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Rank-1（低秩）DOE 参数化。

高度图是低秩外积 ``h = h_max * sigmoid(V @ Q.T)``（默认秩为 1）。由于
``h_max`` 对应设计波长下的 2*pi 相移，因此设计波长下的相位为
``2*pi * sigmoid(V @ Q.T)``。

参考文献：
    Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide,
    "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range
    Imaging," CVPR 2020.
"""

import torch

from .diffractive import DiffractiveSurface


class Rank1(DiffractiveSurface):
    """高度图受低秩外积约束的 DOE。

    高度图为低秩乘积 $h = h_{max} \\cdot \\sigma(V Q^T)$，其中 $\\sigma$ 为
    sigmoid，$h_{max}$ 是在设计波长下产生 $2\\pi$ 相移的高度。因此，设计波长
    下的原始相位为 $2\\pi \\cdot \\sigma(V Q^T)$，范围为 $(0, 2\\pi)$。
    默认 `rank` 为 1 时，高度图是单个外积，从而降低 DOE 的制造和优化成本。

    参考文献：
        Qilin Sun, Ethan Tseng, Qiang Fu, Wolfgang Heidrich, Felix Heide,
        "Learning Rank-1 Diffractive Optics for Single-shot High Dynamic Range
        Imaging," CVPR 2020.

    属性：
        rank (int): 高度图的秩。
        V (torch.Tensor): 高度图的左因子。[res[0], rank]
        Q (torch.Tensor): 高度图的右因子。[res[1], rank]
    """

    def __init__(
        self,
        d,
        rank=1,
        V=None,
        Q=None,
        res=(1000, 1000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        device="cpu",
    ):
        """初始化秩为 `rank` 的 DOE。

        参数：
            d (float): DOE 表面的位置。[mm]
            rank (int, optional): 高度图的秩。默认值为 1。
            V (torch.Tensor or None, optional): 左因子，shape 为 [res[0], rank]。
                若为 None，则用较小的随机值初始化。默认值为 None。
            Q (torch.Tensor or None, optional): 右因子，shape 为 [res[1], rank]。
                若为 None，则用较小的随机值初始化。默认值为 None。
            res (tuple or int, optional): DOE 分辨率，格式为 (H, W)；整数会
                扩展为 (res, res)。[pixel] 默认值为 (1000, 1000)。
            mat (str, optional): DOE 材料。默认值为 "fused_silica"。
            wvln0 (float, optional): 设计波长。[um] 默认值为 0.55。
            fab_ps (float, optional): 制造像素尺寸。[mm] 默认值为 0.001。
            fab_step (int, optional): 制造量化级数。默认值为 16。
            is_square (bool, optional): 孔径是否为方形。默认值为 True。
            device (str, optional): 计算设备。默认值为 "cpu"。
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.rank = rank
        self.V = torch.randn(self.res[0], rank) * 1e-3 if V is None else V
        self.Q = torch.randn(self.res[1], rank) * 1e-3 if Q is None else Q
        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从配置字典初始化 Rank1 DOE。

        若 `doe_dict` 包含 "weight_path"，则从该检查点加载 V 和 Q 因子；
        否则随机初始化。

        参数：
            doe_dict (dict): 表面配置。必须包含 "d" 和 "res"；可选键为
                "rank"、"weight_path"、"mat"、"wvln0"、"fab_ps"、"fab_step"、
                "is_square"。

        返回：
            doe (Rank1): 构造得到的 DOE。
        """
        V = Q = None
        weight_path = doe_dict.get("weight_path", None)
        if weight_path is not None:
            w = torch.load(weight_path, weights_only=True)
            V, Q = w["V"], w["Q"]
        return cls(
            d=doe_dict["d"],
            rank=doe_dict.get("rank", 1),
            V=V,
            Q=Q,
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            is_square=doe_dict.get("is_square", True),
        )

    def phase_func(self):
        """计算设计波长下的原始相位图。

        返回未包裹、未量化的相位 $2\\pi \\cdot \\sigma(V Q^T)$，其中
        $\\sigma$ 为 sigmoid。

        返回：
            phase (torch.Tensor): 原始相位图。[H, W]，范围为 $(0, 2\\pi)$。[rad]
        """
        return 2 * torch.pi * torch.sigmoid(self.V @ self.Q.T)

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """为 V 和 Q 因子构建优化器参数组。

        同时启用 `V` 和 `Q` 的梯度。

        参数：
            lr (float, optional): V 和 Q 因子的学习率。默认值为 0.01。

        返回：
            params (list): 单组参数列表 [{"params": [V, Q], "lr": lr}]。
        """
        self.V.requires_grad = True
        self.Q.requires_grad = True
        return [{"params": [self.V, self.Q], "lr": lr}]

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self, weight_path):
        """将表面序列化为字典，并把 V、Q 因子保存到磁盘。

        将包含键 "V" 和 "Q" 的检查点（已分离并位于 CPU 上）写入
        `weight_path`，同时在返回的字典中记录 `rank` 和 `weight_path`。

        参数：
            weight_path (str): V 和 Q 因子的保存路径。

        返回：
            surf_dict (dict): 包含 "rank" 和 "weight_path" 的表面配置。
        """
        surf_dict = super().surf_dict()
        surf_dict["rank"] = self.rank
        surf_dict["weight_path"] = weight_path
        torch.save(
            {"V": self.V.clone().detach().cpu(), "Q": self.Q.clone().detach().cpu()},
            weight_path,
        )
        return surf_dict
