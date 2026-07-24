# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""用于快照高光谱成像的衍射旋转 DOE。

每个角向楔区都是针对不同“匹配”波长闪耀的菲涅耳透镜，因此聚焦后的 PSF
是各向异性的瓣，其方向随波长单调旋转。

参考文献：
    Daniel S. Jeon, Seung-Hwan Baek, Shinyoung Yi, Qiang Fu, Xiong Dun,
    Wolfgang Heidrich, Min H. Kim, "Compact Snapshot Hyperspectral Imaging
    with Diffracted Rotation," ACM TOG (SIGGRAPH) 2019.
"""

import torch

from .diffractive import DiffractiveSurface


class DiffractedRotation(DiffractiveSurface):
    """各楔区菲涅耳透镜按方位角闪耀的解析螺旋 DOE。

    实现论文公式（12）中的衍射旋转高度图：将孔径划分为 `num_wings` 个角向
    楔区，每个楔区都是针对随方位角线性变化的“匹配”波长闪耀的菲涅耳透镜。
    由此产生具有 `num_wings` 重对称性的各向异性相位分布，其聚焦 PSF 瓣会
    随波长旋转。

    说明：
        论文报告的波长相关 PSF 旋转是在其完整重建流程的焦平面上呈现的。
        在 DeepLens 的近轴角谱 PSF 模型中，轴上焦点实际上小于一个像素，
        因此渲染的 PSF 显示固定的 N 重各向异性结构，而非清晰的旋转。
        此处的相位参数化严格遵循公式（12）；能够分辨旋转瓣的焦平面 PSF
        流程不在本实现范围内。

    属性：
        f0 (torch.Tensor): 焦距，可优化的标量张量。[mm]
        num_wings (int): 角向楔区数量 N。
        wvln_min (float): 最小匹配波长。[um]
        wvln_max (float): 最大匹配波长。[um]
        circular (bool): 若为 True，则将内切圆外的相位置零。
        r2 (torch.Tensor): 径向坐标平方 $x^2+y^2$，shape 为 (H, W)。[mm^2]
        theta (torch.Tensor): [0, 2*pi) 范围内的方位角，shape 为 (H, W)。[rad]
    """

    def __init__(
        self,
        d,
        f0,
        num_wings=3,
        wvln_min=0.42,
        wvln_max=0.66,
        wvln0=None,
        res=(1000, 1000),
        mat="fused_silica",
        fab_ps=0.001,
        fab_step=16,
        is_square=True,
        circular=True,
        device="cpu",
    ):
        """初始化衍射旋转 DOE。

        参数：
            d (float): DOE 表面的轴向位置。[mm]
            f0 (float): 各楔区菲涅耳闪耀使用的焦距。[mm]
            num_wings (int): 角向楔区数量 N。默认值为 3。
            wvln_min (float): 最小匹配波长。[um] 默认值为 0.42。
            wvln_max (float): 最大匹配波长。[um] 默认值为 0.66。
            wvln0 (float or None, optional): 设计波长 [um]。为 None 时默认采用
                `wvln_max`，使包裹后的相位不超过 2*pi。
            res (tuple or int): DOE 分辨率 (H, W)。[pixel] 默认值为 (1000, 1000)。
            mat (str): DOE 材料。默认值为 "fused_silica"。
            fab_ps (float): 制造像素尺寸。[mm] 默认值为 0.001。
            fab_step (int): 相位量化级数。默认值为 16。
            is_square (bool): DOE 孔径是否为方形。默认值为 True。
            circular (bool): 若为 True，则将内切圆外的相位置零。默认值为 True。
            device (str): 计算设备。默认值为 "cpu"。
        """
        if wvln0 is None:
            wvln0 = wvln_max
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps,
            fab_step=fab_step, is_square=is_square, device=device,
        )
        self.f0 = f0 if torch.is_tensor(f0) else torch.tensor(float(f0))
        self.num_wings = num_wings
        self.wvln_min = wvln_min
        self.wvln_max = wvln_max
        self.circular = circular

        # 缓存静态极坐标网格。
        self.r2 = self.x**2 + self.y**2
        self.theta = torch.remainder(torch.atan2(self.y, self.x), 2 * torch.pi)
        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从配置字典初始化 DiffractedRotation DOE。

        参数：
            doe_dict (dict): 表面参数。必须包含 "d"、"f0" 和 "res"；
                其余构造参数缺省时使用默认值。

        返回：
            surf (DiffractedRotation): 构造得到的 DOE 表面。
        """
        return cls(
            d=doe_dict["d"],
            f0=doe_dict["f0"],
            num_wings=doe_dict.get("num_wings", 3),
            wvln_min=doe_dict.get("wvln_min", 0.42),
            wvln_max=doe_dict.get("wvln_max", 0.66),
            wvln0=doe_dict.get("wvln0", None),
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            circular=doe_dict.get("circular", True),
        )

    def phase_func(self):
        """计算设计波长下包裹后的相位图。

        对每个像素，取理想会聚透镜的光程差 $\\sqrt{r^2 + f_0^2} - f_0$，
        按与方位角相关的匹配波长取模进行闪耀，再乘以 $2\\pi / \\lambda_0$。
        设置 `circular` 时，将半径为 $\\min(w, h) / 2$ 的内切圆外相位置零。

        返回：
            phase (torch.Tensor): 包裹后的相位图，shape 为 (H, W)。[rad]
        """
        # 理想会聚透镜的光程差 [mm]。
        opd = torch.sqrt(self.r2 + self.f0**2) - self.f0
        # 各角度的匹配波长（锯齿波，在 2*pi 范围内有 num_wings 个周期）[mm]。
        frac = torch.remainder(self.theta * self.num_wings / (2 * torch.pi), 1.0)
        lam_m_mm = (self.wvln_min + (self.wvln_max - self.wvln_min) * frac) * 1e-3
        wvln0_mm = self.wvln0 * 1e-3
        # 按各自的匹配波长对每个楔区进行闪耀。
        phase = (2 * torch.pi / wvln0_mm) * torch.remainder(opd, lam_m_mm)
        if self.circular:
            r_max = min(self.w, self.h) / 2
            phase = torch.where(
                self.r2 <= r_max**2, phase, torch.zeros_like(phase)
            )
        return phase

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """获取焦距的优化器参数组。

        启用 `f0` 的梯度，并将其作为单个参数组返回。

        参数：
            lr (float): `f0` 的学习率。默认值为 0.001。

        返回：
            params (list): 包含一个 `f0` 参数组字典的列表。
        """
        self.f0.requires_grad = True
        return [{"params": [self.f0], "lr": lr}]

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """将表面参数序列化为字典。

        在父类表面字典中加入衍射旋转参数（`f0`、`num_wings`、`wvln_min`、
        `wvln_max`）。

        返回：
            surf_dict (dict): 适用于 `init_from_dict` 的表面参数。
        """
        surf_dict = super().surf_dict()
        surf_dict["f0"] = round(self.f0.item(), 4)
        surf_dict["num_wings"] = self.num_wings
        surf_dict["wvln_min"] = self.wvln_min
        surf_dict["wvln_max"] = self.wvln_max
        return surf_dict
