# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""由自由形状一维径向分布参数化的旋转对称 DOE。

相位由包含 ``n_rings`` 个采样点的一维径向向量 ``radial_phase`` 定义，并通过
环之间的可微线性插值，按照 ``r = sqrt(x**2 + y**2)`` 扩展到二维。

参考文献：
    Xiong Dun, Hayato Ikoma, Gordon Wetzstein, Zhanshan Wang, Xinbin Cheng,
    Yifan Peng, "Learned rotationally symmetric diffractive achromat for
    full-spectrum computational imaging," Optica 2020.
"""

import torch

from .diffractive import DiffractiveSurface


class RotationallySymmetric(DiffractiveSurface):
    """由一维径向相位分布定义的旋转对称 DOE。

    相位由自由形状的一维径向向量 `radial_phase` 参数化，该向量包含覆盖内切
    半径的 `n_rings` 个采样点，并在 $r = \\sqrt{x^2 + y^2}$ 上通过可微
    线性插值扩展到二维网格。只优化 `radial_phase`，从结构上保证旋转对称性。

    属性：
        n_rings (int): `radial_phase` 中的径向采样点数。
        circular (bool): 是否将内切圆外的相位置零。
        r_max (float): 内切半径 $\\min(w, h) / 2$。[mm]
        r_grid (torch.Tensor): 各网格点的径向距离。[H, W]。[mm]
        idx0 (torch.Tensor): 插值使用的下环索引。[H, W]。
        idx1 (torch.Tensor): 插值使用的上环索引。[H, W]。
        frac (torch.Tensor): 朝 `idx1` 的 $[0, 1]$ 插值权重。[H, W]。
        radial_phase (torch.Tensor): 一维径向相位分布。[n_rings]。[rad]
    """

    def __init__(
        self,
        d,
        f0=None,
        n_rings=None,
        init="fresnel",
        radial_phase=None,
        res=(1000, 1000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        circular=True,
        device="cpu",
    ):
        """初始化旋转对称 DOE。

        若给定 `radial_phase`，则据此设置一维径向分布；否则由 `init` 决定：
        "fresnel" 构建菲涅耳透镜二次分布
        $\\phi(r) = -\\pi r^2 / (f_0 \\lambda_0)$（需要 `f0`），"flat"
        则初始化接近零的常数分布。

        参数：
            d (float): DOE 平面的轴向位置。[mm]
            f0 (float or None, optional): `init="fresnel"` 使用的焦距。[mm]。
                当 `init="fresnel"` 且 `radial_phase` 为 None 时必须给定。
                默认值为 None。
            n_rings (int or None, optional): 径向采样点数。默认值为 None，
                此时使用 res[0] // 2。
            init (str, optional): 初始化模式，"fresnel" 或 "flat"。若给定
                `radial_phase` 则忽略。默认值为 "fresnel"。
            radial_phase (torch.Tensor or None, optional): 显式指定的一维径向
                相位分布。[n_rings]。[rad]。默认值为 None。
            res (tuple or int, optional): DOE 分辨率，格式为 (H, W)；整数会
                扩展为 (res, res)。[pixel]。默认值为 (1000, 1000)。
            mat (str, optional): DOE 材料名称。默认值为 "fused_silica"。
            wvln0 (float, optional): 设计波长。[um]。默认值为 0.55。
            fab_ps (float, optional): 制造像素尺寸。[mm]。默认值为 0.001。
            fab_step (int, optional): 制造（量化）级数。默认值为 16。
            is_square (bool, optional): 孔径是否为方形。默认值为 True。
            circular (bool, optional): 是否将内切圆外的相位置零。默认值为 True。
            device (str, optional): 放置 DOE 张量的设备。默认值为 "cpu"。

        异常：
            ValueError: 当 `init` 不是 "fresnel" 或 "flat" 时抛出。
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.n_rings = self.res[0] // 2 if n_rings is None else n_rings
        self.circular = circular
        self.r_max = min(self.w, self.h) / 2  # 内切半径 [mm]

        # 缓存径向插值索引和权重（仅为 r 的函数）。
        r = torch.sqrt(self.x**2 + self.y**2)
        t = (r / self.r_max).clamp(0, 1) * (self.n_rings - 1)
        self.idx0 = torch.floor(t).long().clamp(0, self.n_rings - 1)
        self.idx1 = (self.idx0 + 1).clamp(0, self.n_rings - 1)
        self.frac = t - self.idx0.to(t.dtype)
        self.r_grid = r

        # 初始化一维径向相位分布。
        if radial_phase is not None:
            self.radial_phase = radial_phase
        elif init == "fresnel":
            assert f0 is not None, "init='fresnel' requires f0."
            ring_r = torch.linspace(0, self.r_max, self.n_rings)
            wvln0_mm = wvln0 * 1e-3
            self.radial_phase = -torch.pi * ring_r**2 / (float(f0) * wvln0_mm)
        elif init == "flat":
            self.radial_phase = torch.ones(self.n_rings) * 1e-3
        else:
            raise ValueError(f"Unknown init: {init}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从字典初始化 RotationallySymmetric DOE。

        若 `doe_dict` 包含 "weight_path"，则直接加载并使用已保存的一维径向
        相位，跳过 `init` 初始化。

        参数：
            doe_dict (dict): DOE 参数字典。必须包含 "d" 和 "res"；可选键为
                "weight_path"、"f0"、"n_rings"、"init"、"mat"、"wvln0"、
                "fab_ps"、"fab_step"、"circular"。

        返回：
            doe (RotationallySymmetric): 构造得到的 DOE 实例。
        """
        radial_phase = None
        weight_path = doe_dict.get("weight_path", None)
        if weight_path is not None:
            radial_phase = torch.load(weight_path, weights_only=True)
        return cls(
            d=doe_dict["d"],
            f0=doe_dict.get("f0", None),
            n_rings=doe_dict.get("n_rings", None),
            init=doe_dict.get("init", "fresnel"),
            radial_phase=radial_phase,
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            circular=doe_dict.get("circular", True),
        )

    def phase_func(self):
        """计算设计波长下的原始二维相位图。

        在 $r$ 上通过可微线性插值，将一维 `radial_phase` 扩展到二维网格。
        若 `circular` 为 True，则将内切圆外（$r > $ `r_max`）的相位置零。

        返回：
            phase (torch.Tensor): 设计波长下未包裹的原始相位图。
                [H, W]。[rad]
        """
        # 将一维分布可微地线性插值到二维网格。
        phase = (
            self.radial_phase[self.idx0] * (1 - self.frac)
            + self.radial_phase[self.idx1] * self.frac
        )
        if self.circular:
            phase = torch.where(
                self.r_grid <= self.r_max, phase, torch.zeros_like(phase)
            )
        return phase

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """获取一维径向相位分布的优化器参数组。

        启用 `radial_phase` 的梯度，并将其作为单个 Adam 参数组返回。

        参数：
            lr (float, optional): 径向分布的学习率。默认值为 0.01。

        返回：
            params (list): 包含一个优化器参数组字典的列表。
        """
        self.radial_phase.requires_grad = True
        return [{"params": [self.radial_phase], "lr": lr}]

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self, weight_path):
        """返回描述该表面的字典，并保存径向分布。

        在基础表面字典中加入 `n_rings` 和 `weight_path`，并将已分离且位于
        CPU 上的一维 `radial_phase` 保存到 `weight_path`。

        参数：
            weight_path (str): 一维径向相位张量的保存路径。

        返回：
            surf_dict (dict): 描述 DOE 表面的字典。
        """
        surf_dict = super().surf_dict()
        surf_dict["n_rings"] = self.n_rings
        surf_dict["weight_path"] = weight_path
        torch.save(self.radial_phase.clone().detach().cpu(), weight_path)
        return surf_dict
