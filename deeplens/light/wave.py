# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""用于衍射仿真的复波场类。

本文件包含：
    1. 复波场类
    2. 波场传播函数（ASM、Rayleigh Sommerfeld、Fresnel、Fraunhofer 等）
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.fft import fft2, fftshift, ifft2, ifftshift
from ..config import DELTA, EPSILON
from ..base import DeepObj


# ===================================
# 复波场
# ===================================
class ComplexWave(DeepObj):
    """用于衍射仿真的复标量波场。

    表示均匀矩形网格上的单色相干复振幅。传播方法（带限 ASM、Fresnel）
    以成员函数实现，并使用 `torch.fft` 提高效率。

    属性：
        u (torch.Tensor): 复振幅，形状为 [1, 1, H, W]。
        wvln (float): 波长 [µm]。
        k (float): 波数 $2\\pi / (\\lambda \\times 10^{-3})$ [mm⁻¹]。
        phy_size (tuple): 物理孔径尺寸 (W, H) [mm]。
        ps (float): 像素间距 [mm]，像素为正方形。
        res (tuple): 网格分辨率 (H, W)，单位为像素。
        x (torch.Tensor): x 坐标网格，形状为 [H, W] [mm]。
        y (torch.Tensor): y 坐标网格，形状为 [H, W] [mm]。
        z (torch.Tensor): 轴向位置网格，形状为 [H, W] [mm]。
        plain_asm_dist_max (float): 普通 ASM 的 Nyquist 上限 [mm]，仅供参考。
        fresnel_dist_min (float): 单 FFT Fresnel 达到良好采样所需的最小距离 [mm]。
    """

    def __init__(
        self,
        u=None,
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
    ):
        """初始化复波场。

        参数：
            u (torch.Tensor or None, optional): 初始复振幅。可接受形状为
                [H, W]、[1, H, W] 或 [1, 1, H, W]。为 None 时按给定 res
                创建零场，默认为 None。
            wvln (float, optional): 波长 [µm]，默认为 0.55。
            z (float, optional): 初始轴向位置 [mm]，默认为 0.0。
            phy_size (tuple, optional): 物理孔径 (W, H) [mm]，默认为 (4.0, 4.0)。
            res (tuple, optional): 网格分辨率 (H, W) [pixels]，仅当 u 为 None
                时使用，默认为 (2000, 2000)。

        异常：
            AssertionError: 当像素间距不是正方形，或波长超出 (0.1, 10) µm
                范围时抛出。
        """
        if u is not None:
            if not u.dtype == torch.complex128:
                print(
                    "A complex wave field is created with single precision. " \
                    "In the future, we want to always use double precision."
                )

            self.u = u if torch.is_tensor(u) else torch.from_numpy(u)
            if not self.u.is_complex():
                self.u = self.u.to(torch.complex64)

            # 将 [H, W] 或 [1, H, W] 转换为 [1, 1, H, W]
            if len(u.shape) == 2:
                self.u = u.unsqueeze(0).unsqueeze(0)
            elif len(self.u.shape) == 3:
                self.u = self.u.unsqueeze(0)

            self.res = self.u.shape[-2:]

        else:
            # 初始化零复波场
            amp = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            phi = torch.zeros(res).unsqueeze(0).unsqueeze(0)
            self.u = amp + 1j * phi
            self.res = res

        # 波场参数
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."
        self.wvln = wvln  # [um]，波长
        self.k = 2 * torch.pi / (self.wvln * 1e-3)  # [mm^-1]，波数

        # 物理尺寸和像素尺寸
        self.phy_size = phy_size  # [mm]，物理尺寸
        px = phy_size[0] / self.res[0]
        py = phy_size[1] / self.res[1]
        assert abs(px - py) <= 1e-9 * max(abs(px), abs(py)) + 1e-12, (
            "Pixel size is not square."
        )
        self.ps = phy_size[0] / self.res[0]  # [mm]，像素尺寸

        # 波场网格
        self.x, self.y = self.gen_xy_grid()  # x、y 网格
        self.z = torch.full_like(self.x, z)  # z 网格

        # 缓存参考距离（仅依赖 wvln、ps、phy_size）。
        # plain_asm_dist_max：普通 ASM 的 Nyquist 上限。prop() 使用带限 ASM，
        #   超过该上限仍然有效，因此此值仅供参考。
        # fresnel_dist_min：单 FFT Fresnel 达到良好采样所需的最小距离。
        self.plain_asm_dist_max = Nyquist_ASM_zmax(wvln=self.wvln, ps=self.ps, side_length=self.phy_size[0])
        self.fresnel_dist_min = Fresnel_zmin(wvln=self.wvln, ps=self.ps, side_length=self.phy_size[0])

    @classmethod
    def point_wave(
        cls,
        point=(0.0, 0.0, -1000.0),
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        valid_r=None,
    ):
        """根据点光源在 x0y 平面上创建球面波场。

        相位为 $\\pm k r$，其中 $r$ 是光源到各网格点的距离。对于发散波
        （光源位于平面后方，$z_{src} < z$）取正号，否则取负号。振幅归一化为
        $r_{min} / r$。

        参数：
            point (tuple, optional): 物方点光源位置 (x, y, z) [mm]，默认为
                (0.0, 0.0, -1000.0)。
            wvln (float, optional): 波长 [µm]，默认为 0.55。
            z (float, optional): 波场 z 位置 [mm]，默认为 0.0。
            phy_size (tuple, optional): x0y 平面的物理尺寸 (W, H) [mm]，
                默认为 (4.0, 4.0)。
            res (tuple, optional): 网格分辨率 (H, W) [pixels]，默认为
                (2000, 2000)。
            valid_r (float or None, optional): 设置后，将该半径 [mm] 圆外的
                波场置零，例如模拟镜头孔径。默认为 None。

        返回：
            field (ComplexWave): x0y 平面上的复波场。
        """
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."
        k = 2 * torch.pi / (wvln * 1e-3)  # [mm^-1]，波数

        # 在目标平面上创建网格
        x, y = torch.meshgrid(
            torch.linspace(
                -0.5 * phy_size[0], 0.5 * phy_size[0], res[0], dtype=torch.float64
            ),
            torch.linspace(
                0.5 * phy_size[1], -0.5 * phy_size[1], res[1], dtype=torch.float64
            ),
            indexing="xy",
        )

        # 计算到点光源的距离及球面波相位
        # 在 sqrt 内加入 EPSILON，确保 r 不会恰好为 0；当光源位于平面上的网格
        # 节点时，可避免 1/r 发散以及 r.min()->0。
        r = torch.sqrt(
            (x - point[0]) ** 2 + (y - point[1]) ** 2 + (z - point[2]) ** 2 + EPSILON
        )
        if point[2] < z:
            phi = k * r
        else:
            phi = -k * r
        u = (r.min() / r) * torch.exp(1j * phi)

        # 若提供有效圆半径，则应用该掩码，例如模拟镜头孔径
        if valid_r is not None:
            mask = (x - point[0]) ** 2 + (y - point[1]) ** 2 < valid_r**2
            u = u * mask

        # 创建波场
        return cls(u=u, wvln=wvln, phy_size=phy_size, res=res, z=z)

    @classmethod
    def plane_wave(
        cls,
        wvln=0.55,
        z=0.0,
        phy_size=(4.0, 4.0),
        res=(2000, 2000),
        theta_x=0.0,
        theta_y=0.0,
        valid_r=None,
    ):
        """在 x0y 平面上创建平面波场。

        当 theta_x = theta_y = 0 时，结果是沿 +z 传播的均匀单位振幅平面波。
        非零角度会生成倾斜（斜入射/离轴）平面波，其波矢与光轴形成给定角度；
        这会加入线性相位斜坡
        $\\exp(i k (x \\sin\\theta_x + y \\sin\\theta_y))$，同时振幅保持均匀。

        参数：
            wvln (float, optional): 波长 [µm]，默认为 0.55。
            z (float, optional): 波场 z 位置 [mm]，默认为 0.0。
            phy_size (tuple, optional): 波场物理尺寸 (W, H) [mm]，默认为
                (4.0, 4.0)。
            res (tuple, optional): 网格分辨率 (H, W) [pixels]，默认为
                (2000, 2000)。
            theta_x (float, optional): 波矢在 x-z 平面内的倾角 [rad]，默认为 0.0。
            theta_y (float, optional): 波矢在 y-z 平面内的倾角 [rad]，默认为 0.0。
            valid_r (float or None, optional): 设置后，将该半径 [mm] 圆外的波场
                置零，默认为 None。

        返回：
            field (ComplexWave): 复波场。
        """
        assert wvln > 0.1 and wvln < 10.0, "Wavelength should be in [um]."

        # 创建平面波场
        if theta_x == 0.0 and theta_y == 0.0:
            # 轴上情况：均匀单位振幅场。
            u = torch.ones(res, dtype=torch.float64) + 0j
        else:
            # 离轴情况：倾斜平面波，即线性相位斜坡。
            k = 2 * torch.pi / (wvln * 1e-3)  # [mm^-1]，波数
            x, y = torch.meshgrid(
                torch.linspace(
                    -0.5 * phy_size[0], 0.5 * phy_size[0], res[0], dtype=torch.float64
                ),
                torch.linspace(
                    0.5 * phy_size[1], -0.5 * phy_size[1], res[1], dtype=torch.float64
                ),
                indexing="xy",
            )
            u = torch.exp(1j * k * (x * math.sin(theta_x) + y * math.sin(theta_y)))

        # 若提供有效圆半径，则应用该掩码
        if valid_r is not None:
            x, y = torch.meshgrid(
                torch.linspace(-0.5 * phy_size[0], 0.5 * phy_size[0], res[0]),
                torch.linspace(-0.5 * phy_size[1], 0.5 * phy_size[1], res[1]),
                indexing="xy",
            )
            mask = (x**2 + y**2) < valid_r**2
            u = u * mask

        # 创建波场
        return cls(u=u, phy_size=phy_size, wvln=wvln, res=res, z=z)

    @classmethod
    def image_wave(cls, img, wvln=0.55, z=0.0, phy_size=(4.0, 4.0)):
        """根据图像初始化复波场。

        将图像解释为 [0, 1] 范围内的强度；波场振幅为其平方根，相位为零。

        参数：
            img (torch.Tensor): 输入图像，形状为 [H, W] 或 [B, C, H, W]，
                数据范围为 [0, 1]，dtype 为 float32。
            wvln (float, optional): 波长 [µm]，默认为 0.55。
            z (float, optional): 波场 z 位置 [mm]，默认为 0.0。
            phy_size (tuple, optional): 波场物理尺寸 (W, H) [mm]，默认为
                (4.0, 4.0)。

        返回：
            field (ComplexWave): 复波场。
        """
        assert img.dtype == torch.float32, "Image must be float32."

        amp = torch.sqrt(img)
        phi = torch.zeros_like(amp)
        u = amp + 1j * phi

        return cls(u=u, wvln=wvln, phy_size=phy_size, res=u.shape[-2:], z=z)

    # =============================================
    # 波场传播
    # =============================================
    def prop(self, prop_dist, n=1.0):
        """将波场向前传播 `prop_dist` 并更新 `self`。

        根据传播距离选择衍射方法：距离接近零时不做处理；亚波长距离尚未实现并
        会抛出异常；不超过 `fresnel_dist_min` 时使用带限 ASM；更大距离使用
        单 FFT Fresnel 衍射。轴向网格 `z` 同步增加 `prop_dist`。

        参数：
            prop_dist (float): 传播距离 [mm]。
            n (float, optional): 介质折射率，默认为 1.0。

        返回：
            self (ComplexWave): 传播后的波场，可用于链式调用。

        异常：
            Exception: 当传播距离处于亚波长范围时抛出，因为 FDTD 等全波方法
                尚未实现。

        参考文献：
            [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Table 1.
            [2] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py
            [3] https://spie.org/samples/PM103.pdf
            [4] "Non-approximated Rayleigh Sommerfeld diffraction integral: advantages and disadvantages in the propagation of complex wave fields"
        """
        # 使用缓存边界确定传播方法
        wvln_mm = self.wvln * 1e-3  # [um] 转为 [mm]

        # 波场传播方法
        if prop_dist < DELTA:
            # 零距离：不做处理
            pass

        elif prop_dist < wvln_mm:
            # 亚波长距离：需要全波方法，例如 FDTD
            raise Exception(
                "The propagation distance in sub-wavelength range is not implemented yet. " \
                "Have to use full wave method (e.g., FDTD)."
            )

        elif prop_dist <= self.fresnel_dist_min:
            # 带限 ASM（Matsushima 与 Shimobaba，2009）：采用严格角谱传播，并通过
            # 带宽限制抑制混叠。它在近场和中间场均有效，因此覆盖了原先
            # Nyquist-ASM 与 Fresnel 适用区间之间的空隙。
            self.u = BandLimitedASM(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)

        else:
            # Fresnel 衍射（远场）
            self.u = FresnelDiffraction(self.u, z=prop_dist, wvln=self.wvln, ps=self.ps, n=n)
        
        # 更新 z 网格
        self.z += prop_dist
        return self

    def prop_to(self, z, n=1):
        """将波场传播至绝对平面 `z` 并更新 `self`。

        根据当前轴向位置计算相对距离，并委托给 `prop`。

        参数：
            z (float): 目标平面的 z 坐标 [mm]。
            n (float, optional): 介质折射率，默认为 1。

        返回：
            self (ComplexWave): 传播后的波场，可用于链式调用。
        """
        # 使用 float() 而不是 .item()，以避免 CUDA 张量发生 GPU-CPU 同步
        # （self.z 是完整网格但所有值相同，因此 [0,0] 具有代表性）
        prop_dist = float(z) - float(self.z[0, 0])
        self.prop(prop_dist, n=n)
        return self

    # =============================================
    # 辅助函数
    # =============================================
    def gen_xy_grid(self):
        """生成形状为 [H, W] 的 x、y 坐标网格。

        x 沿宽度方向变化（res[1] 列，范围 phy_size[0]），y 沿高度方向变化
        （res[0] 行，范围 phy_size[1]），与 `point_wave` / `plane_wave` 一致。
        使用 indexing="xy" 时，输出形状为
        (len(y_1d), len(x_1d)) = (H, W)。

        返回：
            x (torch.Tensor): x 坐标网格，形状为 [H, W] [mm]。
            y (torch.Tensor): y 坐标网格，形状为 [H, W] [mm]。
        """
        x, y = torch.meshgrid(
            torch.linspace(-0.5 * self.phy_size[0], 0.5 * self.phy_size[0], self.res[1]),
            torch.linspace(0.5 * self.phy_size[1], -0.5 * self.phy_size[1], self.res[0]),
            indexing="xy",
        )
        return x, y

    def gen_freq_grid(self):
        """生成形状为 [H, W] 的空间频率网格。

        返回：
            fx (torch.Tensor): x 方向频率网格，形状为 [H, W] [mm⁻¹]。
            fy (torch.Tensor): y 方向频率网格，形状为 [H, W] [mm⁻¹]。
        """
        x, y = self.gen_xy_grid()
        fx = x / (self.ps * self.phy_size[0])
        fy = y / (self.ps * self.phy_size[1])
        return fx, fy

    # =============================================
    # 波场输入/输出
    # =============================================
    def load(self, filepath):
        """从文件加载波场，目前仅支持 `.npz`。

        参数：
            filepath (str): 待加载文件的路径。

        异常：
            Exception: 当文件格式不受支持时抛出。
        """
        if filepath.endswith(".npz"):
            self.load_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def load_npz(self, filepath):
        """从 `.npz` 文件加载复波场和网格。

        参数：
            filepath (str): `.npz` 文件路径。
        """
        data = np.load(filepath)
        self.u = torch.from_numpy(data["u"])
        self.x = torch.from_numpy(data["x"])
        self.y = torch.from_numpy(data["y"])
        self.wvln = data["wvln"].item()
        self.phy_size = data["phy_size"].tolist()
        self.res = self.u.shape[-2:]

    def save(self, filepath="./wavefield.npz"):
        """将复波场保存到文件，目前仅支持 `.npz`。

        参数：
            filepath (str, optional): 输出路径，默认为 "./wavefield.npz"。

        异常：
            Exception: 当文件格式不受支持时抛出。
        """
        if filepath.endswith(".npz"):
            self.save_npz(filepath)
        else:
            raise Exception("Unimplemented file format.")

    def save_npz(self, filepath="./wavefield.npz"):
        """将波场保存为 `.npz` 文件及强度/振幅/相位 PNG 图像。

        将 `u`、`x`、`y`、`wvln` 和 `phy_size` 写入 `.npz` 归档，并在同一
        位置额外保存归一化的强度、振幅和相位图像。

        参数：
            filepath (str, optional): 输出 `.npz` 路径，默认为 "./wavefield.npz"。
        """
        from torchvision.utils import save_image
        # 保存数据
        np.savez_compressed(
            filepath,
            u=self.u.cpu().numpy(),
            x=self.x.cpu().numpy(),
            y=self.y.cpu().numpy(),
            wvln=np.array(self.wvln),
            phy_size=np.array(self.phy_size),
        )

        # 保存强度、振幅和相位图像
        u = self.u.cpu()
        save_image(u.abs() ** 2, f"{filepath[:-4]}_intensity.png", normalize=True)
        save_image(u.abs(), f"{filepath[:-4]}_amp.png", normalize=True)
        save_image(u.angle(), f"{filepath[:-4]}_phase.png", normalize=True)

    def save_image(self, save_name=None, data="irr"):
        """将波场渲染为图像，是 `show` 的别名。

        参数：
            save_name (str or None, optional): 输出图像路径；为 None 时改用
                matplotlib 绘图，默认为 None。
            data (str, optional): 要可视化的量，可为 "irr"、"amp"、
                "phi"/"phase"、"real" 或 "imag"，默认为 "irr"。
        """
        return self.show(save_name=save_name, data=data)

    def show(self, save_name=None, data="irr"):
        """将波场渲染为图像，可保存到磁盘或直接绘制。

        参数：
            save_name (str or None, optional): 输出图像路径；为 None 时使用
                matplotlib 显示波场，默认为 None。
            data (str, optional): 要可视化的量："irr"（强度）、"amp"（振幅）、
                "phi"/"phase"、"real" 或 "imag"，默认为 "irr"。

        异常：
            Exception: 当 `data` 无法识别或波场形状不受支持时抛出。
        """
        from torchvision.utils import save_image
        cmap = "gray"
        if data == "irr":
            value = self.u.detach().abs() ** 2
        elif data == "amp":
            value = self.u.detach().abs()
        elif data == "phi" or data == "phase":
            value = torch.angle(self.u).detach()
            cmap = "hsv"
        elif data == "real":
            value = self.u.real.detach()
        elif data == "imag":
            value = self.u.imag.detach()
        else:
            raise Exception(f"Unimplemented visualization: {data}.")

        if len(self.u.shape) == 2:
            raise Exception("Deprecated.")
            if save_name is not None:
                save_image(value, save_name, normalize=True)
            else:
                value = value.cpu().numpy()
                plt.imshow(
                    value,
                    cmap=cmap,
                    extent=[
                        -self.phy_size[0] / 2,
                        self.phy_size[0] / 2,
                        -self.phy_size[1] / 2,
                        self.phy_size[1] / 2,
                    ],
                )

        elif len(self.u.shape) == 4:
            B, C, H, W = self.u.shape
            if B == 1:
                if save_name is not None:
                    save_image(value, save_name, normalize=True)
                else:
                    value = value.cpu().numpy()
                    plt.imshow(
                        value[0, 0, :, :],
                        cmap=cmap,
                        extent=[
                            -self.phy_size[0] / 2,
                            self.phy_size[0] / 2,
                            -self.phy_size[1] / 2,
                            self.phy_size[1] / 2,
                        ],
                    )
            else:
                if save_name is not None:
                    plt.savefig(save_name)
                else:
                    value = value.cpu().numpy()
                    fig, axs = plt.subplots(1, B)
                    for i in range(B):
                        axs[i].imshow(
                            value[i, 0, :, :],
                            cmap=cmap,
                            extent=[
                                -self.phy_size[0] / 2,
                                self.phy_size[0] / 2,
                                -self.phy_size[1] / 2,
                                self.phy_size[1] / 2,
                            ],
                        )
                    fig.show()
        else:
            raise Exception("Unsupported complex field shape.")

    def pad(self, Hpad, Wpad):
        """对波场进行零填充，并相应扩展物理尺寸。

        在上下各填充 `Hpad` 个像素，在左右各填充 `Wpad` 个像素，然后更新
        `res`、`phy_size` 和坐标网格，使像素间距保持不变。该操作原地修改
        `self`。

        参数：
            Hpad (int): 上下两侧各自填充的像素数。
            Wpad (int): 左右两侧各自填充的像素数。
        """
        self.u = F.pad(self.u, (Hpad, Hpad, Wpad, Wpad), mode="constant", value=0)

        Horg, Worg = self.res
        self.res = [Horg + 2 * Hpad, Worg + 2 * Wpad]
        self.phy_size = [
            self.phy_size[0] * self.res[0] / Horg,
            self.phy_size[1] * self.res[1] / Worg,
        ]
        self.x, self.y = self.gen_xy_grid()
        self.z = torch.full_like(self.x, float(self.z[0, 0]))

    def flip(self):
        """沿水平和垂直方向翻转波场及其网格。

        返回：
            self (ComplexWave): 翻转后的波场，可用于链式调用。
        """
        self.u = torch.flip(self.u, [-1, -2])
        self.x = torch.flip(self.x, [-1, -2])
        self.y = torch.flip(self.y, [-1, -2])
        self.z = torch.flip(self.z, [-1, -2])
        return self


# ===================================
# 衍射函数
# ===================================
def AngularSpectrumMethod(u, z, wvln, ps, n=1.0, padding=True):
    """使用普通角谱法传播波场。

    将波场频谱乘以传递函数
    $\\exp(i k z \\sqrt{1 - \\lambda^2 (f_x^2 + f_y^2)})$，再进行逆变换。
    该方法仅在近场有效；超过 Nyquist 上限后会产生混叠，参见 `BandLimitedASM`。

    参数：
        u (torch.Tensor): 复波场，形状为 [H, W] 或 [B, 1, H, W]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        n (float, optional): 折射率，默认为 1.0。
        padding (bool, optional): FFT 前在各侧进行半尺寸零填充，默认为 True。

    返回：
        u (torch.Tensor): 传播后的复波场，形状与输入相同。

    参考资料：
        [1] https://github.com/kaanaksit/odak/blob/master/odak/wave/classical.py#L293
        [2] https://blog.csdn.net/zhenpixiaoyang/article/details/111569495
    """
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3 / n # [um] 转为 [mm]
    k = 2 * torch.pi / wvln_mm  # [mm]-1

    # 形状
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)

    # 填充
    if padding:
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad), mode="constant", value=0)
    else:
        Wimg, Himg = Worg, Horg

    # 使用角谱法传播
    # 通过一维数组的外和计算 fx²+fy²，避免分配 meshgrid
    real_dtype = u.real.dtype
    fx_1d = torch.fft.fftfreq(Wimg, d=ps, device=u.device, dtype=real_dtype)
    fy_1d = torch.fft.fftfreq(Himg, d=ps, device=u.device, dtype=real_dtype)
    f2 = fx_1d.unsqueeze(0) ** 2 + fy_1d.unsqueeze(1) ** 2
    radicand = 1 - wvln_mm**2 * f2
    complex_dtype = torch.complex128 if radicand.dtype == torch.float64 else torch.complex64
    square_root = torch.sqrt(radicand.to(complex_dtype))

    # H 定义在未移位的频率网格上，以匹配 fft2(u)
    H = torch.exp(1j * k * z * square_root)

    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifft2(fft2(u) * H)

    # 移除填充
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def BandLimitedASM(u, z, wvln, ps, n=1.0, padding=True):
    """带限角谱法。

    当传播距离足够大，使传递函数在频域中的振荡快于网格采样能力时，标准 ASM
    会发生混叠并产生幽灵晶格副本。本变体应用 Matsushima 与 Shimobaba 的带宽
    限制：将当前网格无法充分采样其传递函数条纹的频率置零。近场良好采样区域
    保持不变，因此它可直接替代 `AngularSpectrumMethod`，并在中间场仍然有效。

    带宽限制仅抑制传播核 `H` 的混叠，并假设输入波场 `u` 已满足 Nyquist 采样。
    如果 `u` 的局部条纹频率超过 `1 / (2 * ps)`（例如陡峭球面相位、大倾角或
    高 NA 镜头/DOE 相位），它会在传播前就发生混叠，输出将无提示地出错。

    参数：
        u (torch.Tensor): 复波场，形状为 [H, W] 或 [B, 1, H, W]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        n (float, optional): 折射率，默认为 1.0。
        padding (bool, optional): FFT 前在各侧进行半尺寸零填充，默认为 True。

    返回：
        u (torch.Tensor): 传播后的复波场，形状与输入相同。

    参考文献：
        [1] K. Matsushima and T. Shimobaba, "Band-Limited Angular Spectrum
            Method for Numerical Simulation of Free-Space Propagation in Far
            and Near Fields," Optics Express 17(22), 19662-19673, 2009.
    """
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3 / n  # [um] 转为 [mm]
    k = 2 * torch.pi / wvln_mm  # [mm]-1

    # 形状
    if len(u.shape) == 2:
        Horg, Worg = u.shape
    elif len(u.shape) == 4:
        B, C, Horg, Worg = u.shape
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0).unsqueeze(0)

    # 填充
    if padding:
        Wpad, Hpad = Worg // 2, Horg // 2
        Wimg, Himg = Worg + 2 * Wpad, Horg + 2 * Hpad
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad), mode="constant", value=0)
    else:
        Wimg, Himg = Worg, Horg

    # 未移位频率网格上的角谱传递函数。
    real_dtype = u.real.dtype
    fx_1d = torch.fft.fftfreq(Wimg, d=ps, device=u.device, dtype=real_dtype)
    fy_1d = torch.fft.fftfreq(Himg, d=ps, device=u.device, dtype=real_dtype)
    f2 = fx_1d.unsqueeze(0) ** 2 + fy_1d.unsqueeze(1) ** 2
    radicand = 1 - wvln_mm**2 * f2
    complex_dtype = torch.complex128 if radicand.dtype == torch.float64 else torch.complex64
    square_root = torch.sqrt(radicand.to(complex_dtype))
    H = torch.exp(1j * k * z * square_root)

    # 带宽限制（Matsushima 与 Shimobaba，2009）：将传递函数条纹采样不足的频率
    # 置零。限制频率为
    # f_limit = 1 / (lambda * sqrt((2 * df * z)^2 + 1))，其中 df = 1 / (N * ps)
    # 其中 df 是频率采样间隔。在该上限以下窗口全为 1，因此短距离传播与标准
    # ASM 完全一致。
    z_abs = abs(float(z)) if not torch.is_tensor(z) else float(torch.as_tensor(z).abs().max())
    dfx = 1.0 / (Wimg * ps)
    dfy = 1.0 / (Himg * ps)
    fx_limit = 1.0 / (wvln_mm * math.sqrt((2.0 * dfx * z_abs) ** 2 + 1.0))
    fy_limit = 1.0 / (wvln_mm * math.sqrt((2.0 * dfy * z_abs) ** 2 + 1.0))
    window = (fx_1d.abs().unsqueeze(0) < fx_limit) & (fy_1d.abs().unsqueeze(1) < fy_limit)
    H = H * window.to(real_dtype)

    u = ifft2(fft2(u) * H)

    # 移除填充
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def ScalableASM(u, z, wvln, ps, n=1.0, padding=True):
    """可缩放角谱法，尚未实现。

    计划用于目标像素间距与源像素间距不同的传播。目前仅为返回 None 的占位实现。

    参数：
        u (torch.Tensor): 复波场，形状为 [H, W] 或 [B, 1, H, W]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        n (float, optional): 折射率，默认为 1.0。
        padding (bool, optional): FFT 前进行零填充，默认为 True。

    参考文献：
        [1] Scalable angular spectrum propagation. Optica 2023.
    """
    pass


def FresnelDiffraction(u, z, wvln, ps, n=1.0, padding=True, TF=None):
    """使用单 FFT Fresnel 衍射传播波场。

    可采用传递函数（TF）或脉冲响应（IR）形式。当 `TF` 为 None 时自动选择：
    短距离且频域采样良好时使用 TF，否则使用 IR。

    参数：
        u (torch.Tensor): 复波场，形状为 [H, W] 或 [B, C, H, W]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        n (float, optional): 折射率，默认为 1.0。
        padding (bool, optional): FFT 前在各侧进行半尺寸零填充，默认为 True。
        TF (bool or None, optional): 为 True 时使用传递函数形式，为 False 时
            使用脉冲响应形式；为 None 时根据采样条件自动选择。默认为 None。

    返回：
        u (torch.Tensor): 传播后的复波场，形状与输入相同。

    参考资料：
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.1
        [2] https://qiweb.tudelft.nl/aoi/wavefielddiffraction/wavefielddiffraction.html
        [3] https://github.com/nkotsianas/fourier-propagation/blob/master/FTFP.m
    """
    # 填充。对 [H, W] 和 [B, C, H, W] 均将最后两维解包为 (H, W)。
    if padding:
        Horg, Worg = u.shape[-2], u.shape[-1]
        Hpad, Wpad = Horg // 2, Worg // 2
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Hpad = Wpad = 0
    Himg, Wimg = u.shape[-2], u.shape[-1]

    # 介质中的波场参数
    assert wvln > 0.1 and wvln < 10.0, "wvln should be in [um]."
    wvln_mm = wvln / n * 1e-3  # [um] 转为 [mm]
    k = 2 * torch.pi / wvln_mm

    # TF 或 IR 方法
    if TF is None:
        if ps > wvln_mm * abs(z) / (Wimg * ps):
            TF = True
        else:
            TF = False

    if TF:
        # 频率网格：fx 沿宽度 Wimg，fy 沿高度 Himg，得到 [H, W]。
        fx_1d = torch.linspace(-0.5 / ps, 0.5 / ps, Wimg, device=u.device)
        fy_1d = torch.linspace(0.5 / ps, -0.5 / ps, Himg, device=u.device)
        fx, fy = torch.meshgrid(fx_1d, fy_1d, indexing="xy")
        H = torch.exp(-1j * torch.pi * wvln_mm * z * (fx**2 + fy**2))
        H = fftshift(H)
    else:
        # 空间网格：x 沿宽度 Wimg，y 沿高度 Himg，得到 [H, W]。
        x_1d = torch.linspace(-0.5 * Wimg * ps, 0.5 * Wimg * ps, Wimg, device=u.device)
        y_1d = torch.linspace(0.5 * Himg * ps, -0.5 * Himg * ps, Himg, device=u.device)
        x, y = torch.meshgrid(x_1d, y_1d, indexing="xy")
        h_amp = 1 / (1j * wvln_mm * z)
        # exp(i k z) 是 Python 复标量，应使用 math 构造，因为 torch.exp
        # 不接受非张量复数参数。
        h_const_phase = complex(math.cos(k * z), math.sin(k * z))
        h_phase = torch.exp(1j * torch.pi / (wvln_mm * z) * (x**2 + y**2))
        h = h_const_phase * h_amp * h_phase
        H = fft2(fftshift(h)) * ps**2

    # Fourier 变换
    # https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html#torch.fft.fftshift
    u = ifftshift(ifft2(fft2(fftshift(u)) * H))

    # 移除填充：H 轴去除 Hpad，W 轴去除 Wpad
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def FraunhoferDiffraction(u, z, wvln, ps, n=1.0, padding=True):
    """使用单 FFT Fraunhofer（远场）衍射传播波场。

    输出网格的边长为 $L_2 = \\lambda z / ps$，因此输出像素间距与输入不同。

    参数：
        u (torch.Tensor): 复波场，形状为 [H, W] 或 [B, 1, H, W]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        n (float, optional): 折射率，默认为 1.0。
        padding (bool, optional): FFT 前在各侧进行四分之一尺寸的零填充，
            默认为 True。

    返回：
        u (torch.Tensor): 传播后的复波场，形状与输入相同。

    参考资料：
        [1] Computational fourier optics : a MATLAB tutorial. Chapter 5, section 5.5.
    """
    # 填充。对 [H, W] 和 [B, C, H, W] 均将最后两维解包为 (H, W)。
    if padding:
        Horg, Worg = u.shape[-2], u.shape[-1]
        Hpad, Wpad = Horg // 4, Worg // 4
        u = F.pad(u, (Wpad, Wpad, Hpad, Hpad))
    else:
        Hpad = Wpad = 0
    Himg, Wimg = u.shape[-2], u.shape[-1]

    # 边长
    wvln_mm = wvln / n * 1e-3  # [um] 转为 [mm]
    k = 2 * torch.pi / wvln_mm

    # 计算 x、y、fx、fy
    L2 = wvln_mm * z / ps
    x2, y2 = torch.meshgrid(
        torch.linspace(-L2 / 2, L2 / 2, Wimg, device=u.device),
        torch.linspace(-L2 / 2, L2 / 2, Himg, device=u.device),
        indexing="xy",
    )

    # 更短的传播不会影响最终结果。常数相位 exp(i k z) 是 Python 复标量
    # （k、z 为 float），因此使用 math 构造，而不使用拒绝非张量复数参数的
    # torch.exp。
    h_amp = 1 / (1j * wvln_mm * z)
    h_const_phase = complex(math.cos(k * z), math.sin(k * z))
    h_phase = torch.exp(1j * torch.pi / (wvln_mm * z) * (x2**2 + y2**2))
    h = h_amp * h_const_phase * h_phase
    u = h * ps**2 * ifftshift(fft2(fftshift(u)))

    # 移除填充
    if padding:
        u = u[..., Hpad:-Hpad, Wpad:-Wpad]

    return u


def RayleighSommerfeld(u, z, wvln, ps, n=1.0, memory_saving=True):
    """使用 Rayleigh-Sommerfeld 衍射传播波场。

    构建输入平面坐标网格（单元中心采样，x 覆盖 W 范围，y 覆盖 H 范围），并为
    每个输出点在其上积分。该过程可微，但计算代价过高，不适合优化，仅作为
    真值参考。

    参数：
        u (torch.Tensor): 复波场，形状为 [B, 1, H, W]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        n (float, optional): 折射率，默认为 1.0。
        memory_saving (bool, optional): 在较小输出分块中积分，以降低峰值内存，
            默认为 True。

    返回：
        u2 (torch.Tensor): 传播后的复波场，形状与输入相同。
    """
    _, _, H, W = u.shape
    x, y = torch.meshgrid(
        torch.linspace(
            -0.5 * W * ps + 0.5 * ps, 0.5 * W * ps - 0.5 * ps, W, device=u.device
        ),
        # y 轴覆盖 H 范围而不是 W；对非方形波场使用 W 会得到错误的 y。
        torch.linspace(
            0.5 * H * ps - 0.5 * ps, -0.5 * H * ps + 0.5 * ps, H, device=u.device
        ),
        indexing="xy",
    )

    if u.ndim == 2:
        u2 = RayleighSommerfeldIntegral(
            u, x1=x, y1=y, z=z, wvln=wvln, n=n, memory_saving=memory_saving
        )
    elif u.ndim == 4:
        u2 = torch.zeros_like(u)
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                u2[i, j] = RayleighSommerfeldIntegral(
                    u[i, j],
                    x1=x,
                    y1=y,
                    z=z,
                    wvln=wvln,
                    n=n,
                    memory_saving=memory_saving,
                )
    return u2


def RayleighSommerfeldIntegral(
    u1, x1, y1, z, wvln, x2=None, y2=None, n=1.0, memory_saving=False
):
    """计算离散 Rayleigh-Sommerfeld 衍射积分。

    使用不含近轴或远场近似的暴力积分，作为真值参考。若省略输出坐标，则输出
    平面采用与输入平面相同的网格。

    参数：
        u1 (torch.Tensor): 输入波场复振幅，形状为 [H1, W1]。
        x1 (torch.Tensor): 输入波场的 x 坐标 [mm]，形状为 [H1, W1]。
        y1 (torch.Tensor): 输入波场的 y 坐标 [mm]，形状为 [H1, W1]。
        z (float): 传播距离 [mm]。
        wvln (float): 波长 [µm]。
        x2 (torch.Tensor or None, optional): 输出波场的 x 坐标 [mm]，形状为
            [H2, W2]；为 None 时默认为 x1。
        y2 (torch.Tensor or None, optional): 输出波场的 y 坐标 [mm]，形状为
            [H2, W2]；为 None 时默认为 y1。
        n (float, optional): 折射率，默认为 1.0。
        memory_saving (bool, optional): 在较小输出分块中积分，以降低峰值内存，
            默认为 False。

    返回：
        u2 (torch.Tensor): 输出波场复振幅，形状为 [H2, W2]。

    异常：
        AssertionError: 当传播距离低于给定网格的 Nyquist 最小值时抛出。

    参考资料：
        [1] Modeling and propagation of near-field diffraction patterns: A more complete approach. Eq (9).
        [2] https://www.mathworks.com/matlabcentral/fileexchange/75049-complete-rayleigh-sommerfeld-model-version-2
    """
    # 参数
    assert wvln > 0.1 and wvln < 10.0, "wvln unit should be [um]."
    wvln_mm = wvln * 1e-3  # [um] 转为 [mm]
    k = n * 2 * torch.pi / wvln_mm  # 波数 [mm]-1
    if x2 is None:
        x2 = x1.clone()
    if y2 is None:
        y2 = y1.clone()

    # Nyquist 采样准则
    max_side_dist = max(abs(x1.max() - x2.min()), abs(x2.max() - x1.min()))
    ps = (x1.max() - x1.min()) / x1.shape[-1]
    zmin = Fresnel_zmin(
        wvln=wvln, ps=ps.item(), side_length=max_side_dist.item(), n=n
    )
    assert zmin < z, (
        f"Propagation distance is too short, minimum distance is {zmin} mm."
    )

    # Rayleigh-Sommerfeld 衍射积分
    if not memory_saving:
        # 朴素计算

        # 通过 unsqueeze 广播至 [H1, W1, H2, W2]，不复制数据
        x1_b = x1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        y1_b = y1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        u1_b = u1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]

        # Rayleigh-Sommerfeld 衍射积分
        r2 = (x2 - x1_b) ** 2 + (y2 - y1_b) ** 2 + z**2  # 形状为 [H1, W1, H2, W2]
        r = torch.sqrt(r2)
        obliq = z / r

        u2 = torch.sum(
            u1_b * obliq / r * torch.exp(1j * k * r),
            (0, 1),
        )
        u2 = u2 / (1j * wvln_mm)

    else:
        # 分块计算
        u2 = torch.zeros_like(u1) + 0j

        # 通过 unsqueeze 广播至 [H1, W1, 1, 1]，不复制数据
        patch_size = 4
        x1_b = x1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        y1_b = y1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]
        u1_b = u1.unsqueeze(-1).unsqueeze(-1)  # [H1, W1, 1, 1]

        # 分块计算
        from tqdm import tqdm
        for i in tqdm(range(0, x2.shape[0], patch_size)):
            for j in range(0, x2.shape[1], patch_size):
                # 目标分块
                x2_patch = x2[i : i + patch_size, j : j + patch_size]
                y2_patch = y2[i : i + patch_size, j : j + patch_size]
                r2 = (x2_patch - x1_b) ** 2 + (y2_patch - y1_b) ** 2 + z**2
                r = torch.sqrt(r2)
                obliq = z / r

                # 形状为 [patch_size, patch_size]
                u2_patch = torch.sum(
                    u1_b * obliq / r * torch.exp(1j * k * r),
                    (0, 1),
                )

                # 写入输出波场
                u2[i : i + patch_size, j : j + patch_size] = u2_patch

        u2 = u2 / (1j * wvln_mm)

    return u2


# ==============================
# 辅助函数
# ==============================
def Nyquist_ASM_zmax(wvln, ps, side_length, n=1.0):
    """根据 Nyquist 准则计算 ASM 最大传播距离。

    返回 $z_{max} = L \\cdot ps \\cdot n / \\lambda$，即普通角谱传递函数在
    当前网格上无混叠采样时允许的最大距离。

    参数：
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]。
        side_length (float): 波场边长 [mm]。
        n (float, optional): 折射率，默认为 1.0。

    返回：
        zmax (float): ASM 良好采样的最大传播距离 [mm]。
    """
    wvln_mm = wvln * 1e-3
    zmax = side_length * ps * n / wvln_mm
    return zmax

def Fresnel_zmin(wvln, ps, side_length, n=1.0):
    """根据 Nyquist 准则计算 Fresnel 最小传播距离。

    返回 $z_{min} = L \\cdot n / \\lambda$，即单 FFT Fresnel 衍射在当前网格上
    达到良好采样所需的最短距离。`ps` 参数仅为保持接口对称而接收，不会使用。

    参数：
        wvln (float): 波长 [µm]。
        ps (float): 像素尺寸 [mm]，未使用。
        side_length (float): 波场边长 [mm]。
        n (float, optional): 折射率，默认为 1.0。

    返回：
        zmin (float): Fresnel 良好采样的最小传播距离 [mm]。
    """
    wvln_mm = wvln * 1e-3
    zmin = float(np.sqrt(side_length**2) / (wvln_mm / n))
    return zmin
