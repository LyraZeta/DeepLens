# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""衍射表面（DOE）的基类。"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from ..config import EPSILON
from ..base import DeepObj
from ..material import Material
from ..utils import diff_quantize

logger = logging.getLogger(__name__)


class DiffractiveSurface(DeepObj):
    """衍射光学元件（DOE）的基类。

    衍射表面用于调制入射波场的相位，其光学行为通过波动光学模拟。相位分布
    由子类中的 `phase_func` 定义，并转换为设计波长下经过包裹和量化的相位图。
    默认情况下，DOE 针对 0.55um 设计，即在 0.55um 处具有最高的一阶衍射效率。

    属性：
        d (torch.Tensor): DOE 平面的轴向位置。[mm]
        res (tuple): DOE 分辨率，格式为 (H, W)。[pixel]
        ps (float): 相位图的像素尺寸（若给定设计像素尺寸则使用该值，否则使用
            制造像素尺寸）。[mm]
        w (float): DOE 的物理宽度。[mm]
        h (float): DOE 的物理高度。[mm]
        is_square (bool): 是否将孔径视为方形。
        r (float): 孔径半径（半对角线／外接圆半径）。[mm]
        mat (Material): DOE 材料。
        wvln0 (float): 设计波长。[um]
        n0 (float): 材料在 `wvln0` 处的折射率。
        fab_ps (float): 制造像素尺寸。[mm]
        fab_step (int): 制造（量化）级数。
        x (torch.Tensor): 网格的 x 坐标。[H, W]。[mm]
        y (torch.Tensor): 网格的 y 坐标。[H, W]。[mm]
    """

    def __init__(
        self,
        d,
        res,
        fab_ps=0.001,
        fab_step=16,
        wvln0=0.55,
        mat="fused_silica",
        design_ps=None,
        is_square=True,
        device="cpu",
    ):
        """初始化衍射表面。

        参数：
            d (float): DOE 平面的轴向位置。[mm]
            res (tuple or int): DOE 分辨率，格式为 (H, W)；整数会扩展为
                (res, res)。[pixel]
            fab_ps (float, optional): 制造像素尺寸。[mm]。默认值为 0.001。
            fab_step (int, optional): 制造（量化）级数。默认值为 16。
            wvln0 (float, optional): 设计波长。[um]。默认值为 0.55。
            mat (str, optional): DOE 的材料名称。默认值为 "fused_silica"。
            design_ps (float or None, optional): 设计像素尺寸；若为 None，则使用
                制造像素尺寸作为相位图像素尺寸。[mm]。默认值为 None。
            is_square (bool, optional): 孔径是否为方形。默认值为 True。
            device (str, optional): 放置 DOE 张量的设备。默认值为 "cpu"。
        """
        # 几何参数
        self.d = torch.tensor(d) if not isinstance(d, torch.Tensor) else d
        self.res = (res, res) if isinstance(res, int) else res
        self.ps = fab_ps if design_ps is None else design_ps
        self.w = self.res[0] * self.ps
        self.h = self.res[1] * self.ps
        self.is_square = is_square
        # 表面半径取半对角线（外接圆半径），以便与方形孔径的 Phase / Surface
        # 约定保持一致。
        self.r = float(np.sqrt(self.w**2 + self.h**2) / 2)

        # 相位图
        self.mat = Material(mat)
        self.wvln0 = wvln0  # [um]，设计波长；有时优先采用最大工作波长。
        self.n0 = self.mat.refractive_index(
            self.wvln0
        )  # 设计波长处的折射率

        # DOE 制造参数
        self.fab_ps = fab_ps  # [mm]，制造像素尺寸
        self.fab_step = fab_step

        # x、y 坐标
        self.x, self.y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, self.res[1]),
            torch.linspace(self.h / 2, -self.h / 2, self.res[0]),
            indexing="xy",
        )

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从字典初始化 DOE。

        参数：
            doe_dict (dict): DOE 参数字典。

        返回：
            doe (DiffractiveSurface): 构造得到的 DOE 实例。

        异常：
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError

    def phase_func(self):
        """计算设计波长下的原始相位分布（不包裹、不量化）。

        返回：
            phase (torch.Tensor): 设计波长下未包裹的原始相位分布。
                [H, W]。[rad]

        异常：
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError

    def get_phase_map0(self):
        """计算设计波长下经过包裹和量化的相位图。

        将 `phase_func` 给出的原始相位包裹到 $[0, 2\\pi)$，再量化为
        `fab_step` 个级别。包裹后的相位等效于一幅高度图，其最大高度在设计
        波长处对应 $2\\pi$。

        返回：
            phase0 (torch.Tensor): 设计波长下经过包裹和量化的相位图。
                [H, W]，范围为 $[0, 2\\pi)$。[rad]
        """
        # 设计波长下的原始相位图
        phase0 = self.phase_func()

        # 相位包裹与量化
        phase0 = torch.remainder(phase0, 2 * torch.pi)
        phase0 = diff_quantize(phase0, levels=self.fab_step)
        return phase0

    def get_phase_map(self, wvln):
        """计算给定波长下的相位图。

        首先计算设计波长下的相位图，再结合波长比和材料色散
        $(n - 1) / (n_0 - 1)$ 将其缩放到目标波长；如有需要，最后以最近邻
        方式重采样到 DOE 分辨率。

        参数：
            wvln (float): 波长。[um]

        返回：
            phase_map (torch.Tensor): 给定波长下的相位图。[H, W]。[rad]
        """
        # 设计波长下的相位图
        phase_map0 = self.get_phase_map0()

        # 给定波长下的相位图（隐式转换为高度图）
        n = self.mat.refractive_index(wvln)
        phase_map = phase_map0 * (self.wvln0 / wvln) * (n - 1) / (self.n0 - 1)

        # 插值到目标分辨率（若已匹配则跳过）
        if phase_map.shape[-2:] != (self.res[0], self.res[1]):
            phase_map = (
                F.interpolate(
                    phase_map.unsqueeze(0).unsqueeze(0), size=self.res, mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
            )

        return phase_map

    def _warn_if_undersampled(self, phase, f0, wvln):
        """若逐点二次相位在当前网格上发生混叠，则仅警告一次。

        以逐点乘法施加的透镜相位，只有在相邻像素间的相位步长小于 pi 时才满足
        带限条件；超过该值便会发生混叠，并使 PSF 退化为鬼影晶格伪影。该检查
        针对原始（未包裹）相位执行，因为 ``forward`` 使用的已包裹 [0, 2pi]
        相位无法揭示混叠。此处只发出警告，不改变数值计算。

        参数：
            phase (torch.Tensor): 未包裹的原始相位。[..., H, W]。[rad]
            f0 (float or torch.Tensor): 焦距。[mm]
            wvln (float): 此相位图对应的波长。[um]
        """
        if getattr(self, "_undersample_warned", False):
            return

        with torch.no_grad():
            max_step = torch.maximum(
                torch.diff(phase, dim=-1).abs().max(),
                torch.diff(phase, dim=-2).abs().max(),
            )
        if max_step <= torch.pi:
            return

        self._undersample_warned = True
        f0 = abs(float(f0))
        wvln_mm = wvln * 1e-3
        fnum = f0 / self.w
        fnum_floor = self.ps / wvln_mm
        aperture_max = wvln_mm * f0 / self.ps
        logger.warning(
            f"{self.__class__.__name__}: quadratic phase undersampled at "
            f"wvln={wvln:.3f}um on {self.ps:.4f}mm grid "
            f"(max phase step {float(max_step):.2f} rad/pixel > pi). "
            f"f0={f0:.1f}mm, aperture {self.w:.2f}mm -> f/{fnum:.1f}; "
            f"well-sampled needs f/# > {fnum_floor:.0f} "
            f"(aperture <= {aperture_max:.3f}mm = wvln*f0/ps). "
            f"PSFs may show ghost-lattice aliasing."
        )

    def forward(self, wave):
        """将波场传播到 DOE 平面并施加相位调制。

        输入波场的像素尺寸和物理范围可能与 DOE 不同；先以最近邻方式重采样
        相位图以匹配波场像素尺寸，再通过中心裁剪或零填充匹配波场分辨率，
        最后按 $u \\cdot e^{i\\phi}$ 施加相位。

        参数：
            wave (ComplexWave): 输入复波场，其中场 `u` 的 shape 为
                [B, 1, H, W]。

        返回：
            wave (ComplexWave): 传播和相位调制后的输出复波场，其中场 `u`
                的 shape 为 [B, 1, H, W]。

        参考资料：
            [1] https://github.com/vsitzmann/deepoptics 中的 phaseshifts_from_height_map 函数
        """
        # 传播到 DOE
        wave.prop_to(self.d)

        # 计算波场波长下的相位图，shape 为 [H, W]
        phase_map = self.get_phase_map(wave.wvln)

        # 处理波场与 DOE 之间的像素尺寸差异
        if self.ps != wave.ps:
            scale = self.ps / wave.ps
            phase_map = (
                F.interpolate(
                    phase_map.unsqueeze(0).unsqueeze(0),
                    scale_factor=(scale, scale),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        # 检查场与相位图的分辨率（物理尺寸）是否一致
        wave_h, wave_w = wave.u.shape[-2:]
        phase_h, phase_w = phase_map.shape[-2:]
        if phase_h > wave_h or phase_w > wave_w:
            start_h = (phase_h - wave_h) // 2
            start_w = (phase_w - wave_w) // 2
            phase_map = phase_map[
                ..., start_h : start_h + wave_h, start_w : start_w + wave_w
            ]
        elif phase_h < wave_h or phase_w < wave_w:
            pad_top = (wave_h - phase_h) // 2
            pad_bottom = wave_h - phase_h - pad_top
            pad_left = (wave_w - phase_w) // 2
            pad_right = wave_w - phase_w - pad_left
            phase_map = F.pad(
                phase_map,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )

        wave.u = wave.u * torch.exp(1j * phase_map)
        return wave

    def __call__(self, wave):
        """将 DOE 应用于波场（`forward` 的别名）。

        参数：
            wave (ComplexWave): 输入复波场。

        返回：
            wave (ComplexWave): 输出复波场。
        """
        return self.forward(wave)

    # =======================================
    # 制造相关函数
    # =======================================
    def quantize_phase_map(self, bits=16):
        """将设计波长下的相位图量化到给定级数。

        参数：
            bits (int, optional): 量化级数。默认值为 16。

        返回：
            pmap_q (torch.Tensor): 量化后的相位图。[H, W]，范围为
                $[0, 2\\pi)$。[rad]
        """
        pmap = self.get_phase_map0()
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)
        return pmap_q

    def export_fab_phase_map(self, bits=16, save_path=None):
        """生成制造分辨率的量化相位图并保存检查点。

        通过双线性插值，将相位图从设计像素尺寸上采样到制造像素尺寸，并量化为
        `bits` 个级别。DOE 检查点保存至 `save_path`，DOE 对象本身保持不变。

        参数：
            bits (int, optional): 量化级数。默认值为 16。
            save_path (str or None, optional): 检查点保存路径；若为 None，则生成
                包含制造分辨率、像素尺寸和位深信息的名称。默认值为 None。

        返回：
            pmap_q (torch.Tensor): 制造分辨率下的量化相位图。
                [H_fab, W_fab]，范围为 $[0, 2\\pi)$。[rad]
        """
        # 制造分辨率下的量化相位图
        pmap = self.get_phase_map0()
        fab_res = int(self.ps / self.fab_ps * self.res[0])
        pmap = (
            F.interpolate(
                pmap.unsqueeze(0).unsqueeze(0),
                scale_factor=self.ps / self.fab_ps,
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(0)
            .squeeze(0)
        )
        pmap_q = torch.round(pmap / (2 * torch.pi / bits)) * (2 * torch.pi / bits)

        # 保存相位图
        if save_path is None:
            save_path = f"./doe_fab_{fab_res}x{fab_res}_{int(self.fab_ps * 1000)}um_{bits}bit.pth"
        self.save_ckpt(save_path=save_path)

        return pmap_q

    # =======================================
    # 优化
    # =======================================
    def activate_grad(self, activate=True):
        """启用或禁用相位图参数的梯度。

        参数：
            activate (bool, optional): 是否需要梯度。默认值为 True。

        异常：
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError

    def get_optimizer_params(self, lr=None):
        """为相位图参数构建优化器参数组。

        参数：
            lr (float or None, optional): 学习率。默认值为 None。

        返回：
            params (list): 优化器参数组字典的列表。

        异常：
            NotImplementedError: 必须由子类实现。
        """
        raise NotImplementedError

    def get_optimizer(self, lr=None):
        """为 DOE 相位图参数创建 Adam 优化器。

        参数：
            lr (float or None, optional): 传递给 `get_optimizer_params` 的
                学习率。默认值为 None。

        返回：
            optimizer (torch.optim.Adam): 用于 DOE 相位图参数的优化器。
        """
        params = self.get_optimizer_params(lr)
        optimizer = torch.optim.Adam(params)

        return optimizer

    def loss_quantization(self, bits=16):
        """计算 DOE 的平均相位量化误差。

        返回连续相位图与其 `bits` 级量化结果之间的平均绝对差，用作量化感知
        正则化损失。

        参数：
            bits (int, optional): 量化级数。默认值为 16。

        返回：
            loss (torch.Tensor): 标量平均绝对量化误差。[rad]

        参考文献：
            Quantization-aware Deep Optics for Diffractive Snapshot Hyperspectral Imaging.
        """
        pmap = self.get_phase_map0()
        step = 2 * torch.pi / bits
        pmap_q = torch.round(pmap / step) * step
        loss = torch.mean(torch.abs(pmap - pmap_q))
        return loss

    # =======================================
    # 可视化
    # =======================================
    def draw_phase_map(self, bits=None, save_name="./DOE_phase_map.png"):
        """将设计波长下的相位图保存为归一化图像。

        参数：
            bits (int or None, optional): 量化级数；若给定，则先量化相位图，
                否则使用连续相位图。默认值为 None。
            save_name (str, optional): 图像保存路径。默认值为
                "./DOE_phase_map.png"。
        """
        if bits is not None:
            pmap = self.quantize_phase_map(bits)
        else:
            pmap = self.get_phase_map0()
        save_image(pmap, save_name, normalize=True)

    def draw_phase_map3d(self, bits=None, save_name="./DOE_phase_map3d.png"):
        """保存设计波长相位图的三维散点图。

        参数：
            bits (int or None, optional): 量化级数；若给定，则先量化相位图，
                否则使用连续相位图。默认值为 None。
            save_name (str, optional): 图像保存路径。默认值为
                "./DOE_phase_map3d.png"。
        """
        if bits is not None:
            pmap = self.quantize_phase_map(bits)
        else:
            pmap = self.get_phase_map0()
        
        pmap = pmap / 20.0
        x = np.linspace(-self.w / 2, self.w / 2, self.res[0])
        y = np.linspace(-self.h / 2, self.h / 2, self.res[1])
        X, Y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            X.flatten(),
            Y.flatten(),
            pmap.cpu().numpy().flatten(),
            marker=".",
            s=0.01,
            c=pmap.cpu().numpy().flatten(),
            cmap="viridis",
        )
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_phase_map_fab(self, save_name="./DOE_phase_map.png"):
        """并排保存连续相位图和 16 级量化相位图。

        参数：
            save_name (str, optional): 图像保存路径。默认值为
                "./DOE_phase_map.png"。
        """
        pmap = self.get_phase_map0()
        step = 2 * torch.pi / 16
        pmap_q = torch.round(pmap / step) * step

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * float(np.pi))
        ax[0].set_title(f"Phase map ({self.wvln0}um)", fontsize=10)
        ax[0].grid(False)
        fig.colorbar(ax[0].get_images()[0])

        ax[1].imshow(pmap_q.cpu().numpy(), vmin=0, vmax=2 * float(np.pi))
        ax[1].set_title(f"Quantized phase map ({self.wvln0}um)", fontsize=10)
        ax[1].grid(False)
        fig.colorbar(ax[1].get_images()[0])

        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_cross_section(self, save_name="./DOE_cross_section.png"):
        """保存相位图沿主对角线的曲线图。

        参数：
            save_name (str, optional): 图像保存路径。默认值为
                "./DOE_cross_section.png"。
        """
        pmap = self.get_phase_map0()
        pmap = torch.diag(pmap).cpu().numpy()
        r = np.linspace(
            -self.w / 2 * float(np.sqrt(2)), self.w / 2 * float(np.sqrt(2)), self.res[0]
        )

        fig, ax = plt.subplots()
        ax.plot(r, pmap)
        ax.set_title(f"Phase map ({self.wvln0}um) cross section")
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_widget(self, ax, color="orange", linestyle="-"):
        """在布局图中绘制 DOE 的二维菲涅耳式截面。

        绘制 y=0 处沿 x 轴的截面。对于方形孔径，半范围为半边长（`w/2`）；
        对于圆形孔径，半范围为完整半径 `r`（即半对角线）。

        参数：
            ax (matplotlib.axes.Axes): 用于绘图的坐标轴。
            color (str, optional): 线条颜色。默认值为 "orange"。
            linestyle (str, optional): 线型。默认值为 "-"。
        """
        d = self.d.item()
        max_offset = d / 100
        roc = self.r * 2
        x_half = self.w / 2 if self.is_square else self.r
        x = np.linspace(-x_half, x_half, 256)
        sag = roc * (1 - np.sqrt(1 - x**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color=color, linestyle=linestyle, linewidth=0.75)

    # =======================================
    # 工具函数
    # =======================================
    def surf_dict(self):
        """将 DOE 表面参数序列化为字典。

        返回：
            surf_dict (dict): 适合保存或重建的表面参数，包括类型、尺寸、位置、
                设计波长、分辨率、制造像素尺寸和孔径形状标志。
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "(size)": [round(self.w, 4), round(self.h, 4)],
            "d": round(self.d.item(), 4),
            "wvln0": round(self.wvln0, 4),
            "res": self.res,
            "fab_ps": self.fab_ps,
            "is_square": self.is_square,
        }

        return surf_dict
