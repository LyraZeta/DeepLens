# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""使用神经网络表示镜头点扩散函数（PSF）的代理镜头模型。

与传统光线追迹方法相比，该代理模型可以显著加速 PSF 计算。

技术论文：
    Xinge Yang, Qiang Fu, Mohamed Elhoseiny, and Wolfgang Heidrich, "Aberration-Aware Depth-from-Focus" IEEE-TPAMI 2023.
"""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm

from .geolens import GeoLens
from .geolens_pkg.optim import get_cosine_schedule_with_warmup
from .lens import Lens
from .surrogate import MLP
from .surrogate.psfnet_mplconv import PSFNet_MLPConv
from .config import DEFAULT_WAVE, DEPTH, PSF_KS, WAVE_RGB
from .imgsim import rotate_psf, splat_psf_per_pixel


class PSFNetLens(Lens):
    """通过 MLP/MLPConv 网络预测 PSF 的神经代理镜头。

    使用神经网络封装 `GeoLens`，该网络经过训练，可根据
    `(fov, depth, foc_dist)` 输入预测三通道 RGB PSF。训练完成后，PSF
    预测比光线追迹快得多，因此适合实时应用和大规模优化。

    使用 `train_psfnet` 根据光线追迹得到的 PSF 样本训练代理模型，或使用
    `load_net` 加载预训练权重。

    属性：
        lens (GeoLens): 底层折射镜头，用于生成训练数据和提供传感器元数据。
        psfnet (nn.Module): 用于预测 PSF 的神经网络。
        pixel_size (float): 像素间距 [mm]，复制自内嵌镜头。
        foclen (float): 焦距 [mm]，复制自内嵌镜头。
        rfov (float): 实际半对角线视场角 [radians]。
        kernel_size (int): 网络原生 PSF 核的边长 [pixels]。
        d_close (float): 训练时的近物体深度边界 [mm]（负值）。
        d_far (float): 训练时的远物体深度边界 [mm]（负值）。
        foc_d_close (float): 近对焦距离边界 [mm]（负值）。
        foc_d_far (float): 远对焦距离边界 [mm]（负值）。
        foc_dist (float): 当前对焦距离 [mm]（负值）。
    """

    def __init__(
        self,
        lens_path,
        in_chan=3,
        psf_chan=3,
        model_name="mlpconv",
        kernel_size=128,
        dtype=torch.float32,
        primary_wvln=DEFAULT_WAVE,
        wvln_rgb=WAVE_RGB,
        obj_depth=DEPTH,
    ):
        """初始化 PSF 网络镜头。

        加载内嵌的 `GeoLens`，构建 PSF 网络，并将镜头对焦至无穷远。默认设置下，
        网络以 `(fov, depth, foc_dist)` 为输入，输出沿 y 轴的三通道 RGB PSF。

        参数：
            lens_path (str): 镜头文件路径。
            in_chan (int, optional): 输入通道数。默认为 3。
            psf_chan (int, optional): 输出 PSF 通道数。默认为 3。
            model_name (str, optional): 网络架构，可为 "mlp" 或 "mlpconv"。
                默认为 "mlpconv"。
            kernel_size (int, optional): 预测 PSF 核的边长 [pixels]。默认为 128。
            dtype (torch.dtype, optional): 计算所用的数据类型。默认为 torch.float32。
            primary_wvln (float, optional): 主设计波长 [µm]。当调用方法时未显式
                指定 `wvln`，将使用此值作为后备。默认为 DEFAULT_WAVE。
            wvln_rgb (sequence of float, optional): RGB 计算所用的三个波长，
                在 µm 单位下按 [R, G, B] 排列。默认为 WAVE_RGB。
            obj_depth (float, optional): 默认物体深度 [mm]，在调用方法时未显式
                指定深度时使用。默认为 DEPTH。
        """
        super().__init__(
            dtype=dtype,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )

        # 加载镜头（sensor_size 和 sensor_res 从镜头文件中读取）
        self.lens_path = lens_path
        self.lens = GeoLens(
            filename=lens_path,
            device=self.device,
            dtype=dtype,
            primary_wvln=primary_wvln,
            wvln_rgb=wvln_rgb,
            obj_depth=obj_depth,
        )
        self.foclen = self.lens.foclen
        self.rfov = self.lens.rfov

        # 初始化 PSF 网络
        self.in_chan = in_chan
        self.psf_chan = psf_chan
        self.kernel_size = kernel_size
        self.pixel_size = self.lens.pixel_size

        self.psfnet = self.init_net(
            in_chan=in_chan,
            psf_chan=psf_chan,
            kernel_size=kernel_size,
            model_name=model_name,
        )
        self.psfnet.to(self.device)

        # 物体深度范围
        self.d_close = -200
        self.d_far = -20000

        # 对焦距离范围
        # 每个镜头都有最小对焦距离。例如，Canon EF 50mm 镜头只能对焦到 0.5m 及更远处。
        self.foc_d_close = -500
        self.foc_d_far = -20000
        self.refocus(foc_dist=-20000)

    def set_sensor_res(self, sensor_res):
        """同时设置 PSFNetLens 和内嵌 GeoLens 的传感器分辨率。

        同时相应更新像素尺寸。

        参数：
            sensor_res (tuple): 新的传感器分辨率，以 `(W, H)` 表示，单位为 pixels。
        """
        self.lens.set_sensor_res(sensor_res)
        self.pixel_size = self.lens.pixel_size

    # ==================================================
    # 训练函数
    # ==================================================
    def init_net(self, in_chan=2, psf_chan=3, kernel_size=64, model_name="mlpconv"):
        """初始化并返回 PSF 网络。

        网络将 shape 为 [B, in_chan] 的输入（缩放后的
        `(fov, depth, foc_dist)` 特征）映射为 shape 为
        [B, psf_chan, kernel_size, kernel_size] 的 PSF 核。

        参数：
            in_chan (int, optional): 输入通道数。默认为 2。
            psf_chan (int, optional): 输出 PSF 通道数。默认为 3。
            kernel_size (int, optional): PSF 核的边长 [pixels]。默认为 64。
            model_name (str, optional): 网络架构，可为 "mlp" 或 "mlpconv"。
                默认为 "mlpconv"。

        返回：
            psfnet (nn.Module): 构建好的 PSF 网络。

        异常：
            Exception: 当 `model_name` 不是受支持的架构时抛出。
        """
        if model_name == "mlp":
            psfnet = MLP(
                in_features=in_chan,
                out_features=psf_chan * kernel_size**2,
                hidden_features=256,
                hidden_layers=8,
            )
        elif model_name in ("mlpconv", "mlp_conv"):
            psfnet = PSFNet_MLPConv(
                in_chan=in_chan, kernel_size=kernel_size, out_chan=psf_chan
            )
        else:
            raise Exception(f"Unsupported PSF network architecture: {model_name}.")

        return psfnet

    def load_net(self, net_path):
        """从磁盘加载预训练的 PSF 网络权重。

        同时打印检查点中存储的像素尺寸、镜头路径及其当前值，以便发现不匹配，
        随后将权重加载到 `self.psfnet`。

        参数：
            net_path (str): 已保存检查点文件的路径。
        """
        # 检查加载的模型是否正确
        psfnet_dict = torch.load(net_path, map_location="cpu", weights_only=False)
        print(
            f"Pretrained model lens pixel size: {psfnet_dict['pixel_size']*1000.0:.1f} um, "
            f"Current lens pixel size: {self.pixel_size*1000.0:.1f} um"
        )
        print(
            f"Pretrained model lens path: {psfnet_dict['lens_path']}, "
            f"Current lens path: {self.lens_path}"
        )

        # 加载模型权重
        self.psfnet.load_state_dict(psfnet_dict["psfnet_model_weights"])

    def save_psfnet(self, psfnet_path):
        """将 PSF 网络及其元数据保存到磁盘。

        将网络权重与模型名称、通道数、核尺寸、像素尺寸和镜头路径一并存储，
        使检查点包含完整的自描述信息。

        参数：
            psfnet_path (str): 检查点文件的保存路径。
        """
        psfnet_dict = {
            "model_name": self.psfnet.__class__.__name__,
            "in_chan": self.in_chan,
            "pixel_size": self.pixel_size,
            "kernel_size": self.kernel_size,
            "psf_chan": self.psf_chan,
            "lens_path": self.lens_path,
            "psfnet_model_weights": self.psfnet.state_dict(),
        }
        torch.save(psfnet_dict, psfnet_path)

    def train_psfnet(
        self,
        iters=100000,
        bs=128,
        lr=5e-5,
        evaluate_every=500,
        spp=16384,
        concentration_factor=2.0,
        result_dir="./results/psfnet",
    ):
        """训练 PSF 代理网络。

        采样光线追迹得到的 PSF 作为监督信号，使用 L1 损失和 AdamW 在带预热的
        余弦调度下优化网络，并定期将 GT/预测结果对比图和最新检查点保存到
        `result_dir`。

        参数：
            iters (int, optional): 训练迭代次数。默认为 100000。
            bs (int, optional): 批大小。默认为 128。
            lr (float, optional): 学习率。默认为 5e-5。
            evaluate_every (int, optional): 每经过指定次数的迭代执行评估并保存检查点。
                默认为 500。
            spp (int, optional): 每像素采样数（当前未使用）。默认为 16384。
            concentration_factor (float, optional): 控制生成数据时在对焦距离附近
                采样深度的集中程度。默认为 2.0。
            result_dir (str, optional): 保存图像和检查点的目录。
                默认为 "./results/psfnet"。
        """
        # 初始化网络并准备训练
        psfnet = self.psfnet
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.AdamW(psfnet.parameters(), lr=lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=int(iters) // 100, num_training_steps=iters
        )

        # 训练网络
        for i in tqdm(range(iters + 1)):
            # 采样训练数据
            sample_input, sample_psf = self.sample_training_data(
                num_points=bs, concentration_factor=concentration_factor
            )
            sample_input, sample_psf = (
                sample_input.to(self.device),
                sample_psf.to(self.device),
            )

            # 前向传播，pred_psf: [B, 3, ks, ks]
            pred_psf = psfnet(sample_input)

            # 反向传播
            optimizer.zero_grad()
            loss = loss_fn(pred_psf, sample_psf)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # 评估训练结果
            if (i + 1) % evaluate_every == 0:
                # 可视化 PSF
                n_vis = 16
                fig, axs = plt.subplots(n_vis, 2, figsize=(4, n_vis * 2))
                for j in range(n_vis):
                    psf0 = sample_psf[j, ...].detach().clone().cpu()
                    axs[j, 0].imshow(psf0.permute(1, 2, 0) * 255.0)
                    axs[j, 0].axis("off")

                    psf1 = pred_psf[j, ...].detach().clone().cpu()
                    axs[j, 1].imshow(psf1.permute(1, 2, 0) * 255.0)
                    axs[j, 1].axis("off")

                axs[0, 0].set_title("GT")
                axs[0, 1].set_title("Pred")

                fig.suptitle(f"GT/Pred PSFs at iter {i + 1}")
                plt.tight_layout()
                plt.savefig(f"{result_dir}/iter{i + 1}.png", dpi=300)
                plt.close()

                # 保存网络
                self.save_psfnet(f"{result_dir}/PSFNet_last.pth")

    @torch.no_grad()
    def sample_training_data(self, num_points=512, concentration_factor=2.0):
        """为 PSF 代理网络采样一批训练数据。

        每次调用抽取一个对焦距离（Beta 分布偏向近端边界），采样 fov（Beta 分布
        偏向 `rfov`），并在对焦距离附近集中采样深度；随后通过光线追迹计算每个
        采样点的 RGB PSF。返回输入中的深度和对焦距离均缩放 1/1000（即以 m 表示）。

        参数：
            num_points (int, optional): 训练点数量（批大小）。默认为 512。
            concentration_factor (float, optional): 控制在对焦距离附近采样深度的
                集中程度；值越大，采样越集中。默认为 2.0。

        返回：
            sample_input (torch.Tensor): shape 为 [num_points, 3]，各列为
                `(fov, depth/1000, foc_dist/1000)`。fov 位于 [0, rfov] [radians]；
                depth 位于 [d_far, d_close] [mm]；foc_dist 位于
                [foc_d_far, foc_d_close] [mm]。
            sample_psf (torch.Tensor): 光线追迹得到的 RGB PSF，shape 为
                [num_points, 3, kernel_size, kernel_size]。
        """
        d_close = self.d_close
        d_far = self.d_far
        rfov = self.lens.rfov

        # 每次迭代采样一个对焦距离 [mm]，范围为 [foc_d_far, foc_d_close]
        # Beta 分布示例：https://share.google/images/Mrb9c39PdddYx3UHj
        beta_sample = float(np.random.beta(1, 4))  # 偏向 0
        foc_dist = self.foc_d_close + beta_sample * (self.foc_d_far - self.foc_d_close)
        self.lens.refocus(foc_dist)
        foc_dist = torch.full((num_points,), foc_dist)

        # 采样 (fov) [radians]，范围为 [0, rfov]
        beta_values = np.random.beta(4, 1, num_points)  # 偏向 1
        beta_values = torch.from_numpy(beta_values).float()
        fov = beta_values * rfov

        # 采样 (depth) [mm]，范围为 [d_far, d_close]，在对焦距离附近采样更多点
        # std_dev 越小，采样点越集中
        std_dev = -foc_dist / concentration_factor
        depth = foc_dist + torch.randn(num_points) * std_dev
        depth = torch.clamp(depth, d_far, d_close)

        # 创建输入张量
        sample_input = torch.stack([fov, depth / 1000.0, foc_dist / 1000.0], dim=1)
        sample_input = sample_input.to(self.device)

        # 通过光线追迹计算 PSF，shape 为 [B, 3, ks, ks]
        points_x = torch.zeros_like(depth)
        points_y = self.lens.foclen * torch.tan(fov) / self.lens.r_sensor
        points_z = depth
        points = torch.stack((points_x, points_y, points_z), dim=-1)
        sample_psf = self.lens.psf_rgb(
            points=points, ks=self.kernel_size, recenter=True
        )

        return sample_input, sample_psf

    def eval(self):
        """将 PSF 代理网络切换为评估模式。

        禁用内部 `psfnet` 模块的 dropout 和批归一化更新。请在推理前调用。
        """
        self.psfnet.eval()

    def points2input(self, points):
        """将点光源坐标转换为网络输入张量。

        将归一化的传感器平面坐标映射为视场角，并与深度和当前对焦距离组合；
        随后将深度和对焦距离缩放 1/1000（即转换为 m），以匹配训练输入。

        参数：
            points (torch.Tensor): shape 为 [N, 3]。各列依次为 [-1, 1] 范围内
                归一化的 x、y（相对于传感器半尺寸的比例）以及深度 [mm]。

        返回：
            network_inp (torch.Tensor): shape 为 [N, 3]，各列为
                `(fov, depth/1000, foc_dist/1000)`。fov 单位为 [radians]；
                depth 和 foc_dist 的原始单位为 [mm]。
        """
        sensor_h, sensor_w = self.lens.sensor_size
        foclen = self.lens.foclen

        points_x = points[:, 0] * sensor_w / 2
        points_y = points[:, 1] * sensor_h / 2
        points_r = torch.sqrt(points_x**2 + points_y**2)
        fov = torch.atan(points_r / foclen)
        depth = points[:, 2]
        # 使用 float()，避免 shape 为 [1] 的 foc_dist 张量导致 torch.full_like 出错。
        foc_dist = torch.full_like(fov, float(self.foc_dist))
        network_inp = torch.stack((fov, depth / 1000.0, foc_dist / 1000.0), dim=-1)
        return network_inp

    # ==================================================
    # 网络推理
    # ==================================================
    def refocus(self, foc_dist):
        """将镜头重新对焦到指定的物距。

        委托内嵌的 `GeoLens` 完成对焦，并将对焦距离缓存在 `self.foc_dist` 中，
        供后续 PSF 预测使用。

        参数：
            foc_dist (float): 对焦距离 [mm]（负值，指向物方）。
        """
        self.lens.refocus(foc_dist)
        self.foc_dist = foc_dist

    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """通过 RGB 代理网络计算单色 PSF。

        `PSFNetLens` 原生支持 RGB：网络在一次前向传播中预测三通道 PSF，
        因此单色 PSF 返回设计波长（`self.wvln_rgb`）最接近 `wvln` 的 RGB 通道。

        参数：
            points (torch.Tensor): 点光源坐标，shape 为 [N, 3] 或 [3]。
            wvln (float, optional): 波长 [µm]。为 None（默认值）时，使用
                `self.primary_wvln` 作为后备，并映射到最近的 RGB 通道。
            ks (int, optional): 输出核尺寸 [pixels]。默认为 PSF_KS。
            **kwargs: 转发给 `psf_rgb`。

        返回：
            psf (torch.Tensor): PSF；单点时 shape 为 [ks, ks]，批处理时为
                [N, ks, ks]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        points = torch.as_tensor(points, device=self.device)
        single_point = points.dim() == 1
        if single_point:
            points = points.unsqueeze(0)
        # 原生 RGB 网络：选择设计波长最接近请求波长的通道。
        chan = min(
            range(len(self.wvln_rgb)), key=lambda i: abs(self.wvln_rgb[i] - wvln)
        )
        psf = self.psf_rgb(points=points, ks=ks, **kwargs)[:, chan, :, :]
        return psf.squeeze(0) if single_point else psf

    def psf_rgb(self, points, ks=PSF_KS, **kwargs):
        """通过网络计算一批点光源的 RGB PSF。

        网络预测沿 y 轴的 PSF；将每个预测 PSF 旋转 `atan2(x, y)` 至该点的方位角，
        当 `ks` 小于网络原生核尺寸时，再从中心裁剪至 `ks`。

        参数：
            points (torch.Tensor): shape 为 [N, 3]。各列依次为 [-1, 1] 范围内
                归一化的 x、y（相对于传感器半尺寸的比例）以及深度 [mm]。
            ks (int, optional): 输出核尺寸 [pixels]。默认为 PSF_KS。
            **kwargs: 为保持 API 兼容而接收，未使用。

        返回：
            psf (torch.Tensor): RGB PSF，shape 为 [N, 3, ks, ks]。
        """
        # 计算网络输入
        network_inp = self.points2input(points)

        # 使用网络预测沿 y 轴的 PSF
        psf = self.psfnet(network_inp)

        # 后处理 PSF
        # psfnet 使用沿 y 轴的 PSF 进行训练。
        # 需要根据点坐标将 PSF 旋转到正确方向。
        # 从 y 轴正方向逆时针旋转到点 (x, y) 的角度为 atan2(x, y)。
        rot_angle = torch.atan2(points[:, 0], points[:, 1])
        psf = rotate_psf(psf, rot_angle)

        # 将 PSF 裁剪到给定核尺寸
        if ks < self.kernel_size:
            psf = psf[
                :,
                :,
                self.kernel_size // 2 - ks // 2 : self.kernel_size // 2 + ks // 2,
                self.kernel_size // 2 - ks // 2 : self.kernel_size // 2 + ks // 2,
            ]
        return psf

    def psf_map_rgb(self, grid=(11, 11), depth=None, ks=PSF_KS, **kwargs):
        """在视场点网格上计算 RGB PSF 图。

        在给定深度构建点光源网格，并计算每个网格位置的 RGB PSF。

        参数：
            grid (tuple, optional): 网格尺寸，表示为 `(grid_h, grid_w)`。
                默认为 (11, 11)。
            depth (float, optional): 物体深度 [mm]。为 None（默认值）时，使用
                `self.obj_depth` 作为后备。
            ks (int, optional): 核尺寸 [pixels]。默认为 PSF_KS。

        返回：
            psf_map (torch.Tensor): shape 为 [grid_h, grid_w, 3, ks, ks]。
        """
        depth = self.obj_depth if depth is None else depth
        # PSF 图网格
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3).to(self.device)

        # 计算 PSF 图
        psf = self.psf_rgb(points=points, ks=ks)  # [grid*grid, 3, ks, ks]
        psf_map = psf.reshape(grid[0], grid[1], 3, ks, ks)  # [grid, grid, 3, ks, ks]
        return psf_map

    # ==================================================
    # 图像仿真
    # ==================================================
    @torch.no_grad()
    def render_rgbd(self, img, depth, foc_dist, ks=64, high_res=False, chunk_size=256):
        """根据全聚焦图像和深度图渲染离焦图像。

        将镜头重新对焦到 `foc_dist`，根据每个像素的视场位置和深度预测逐像素
        RGB PSF，再将这些 PSF 散布到输入图像上。仅支持批大小为 1。

        参数：
            img (torch.Tensor): 全聚焦图像，shape 为 [1, C, H, W]。
            depth (torch.Tensor): 深度图 [mm]，shape 为 [1, H, W]（深度为负值）。
            foc_dist (torch.Tensor): 对焦距离 [mm]，shape 为 [1]（负值）。
            ks (int, optional): PSF 核尺寸 [pixels]。默认为 64。
            high_res (bool, optional): 若为 True，则分块散布以减少内存占用。
                默认为 False。
            chunk_size (int, optional): `high_res` 为 True 时使用的分块尺寸。
                默认为 256。

        返回：
            render (torch.Tensor): 渲染图像，shape 为 [1, C, H, W]。
        """
        B, C, H, W = img.shape
        assert B == 1, "Only support batch size 1"

        # 将镜头重新对焦到给定距离（把 shape 为 [1] 的张量转换为 Python float，
        # 使 refocus/points2input 能以一致方式处理）。
        self.refocus(float(foc_dist))

        # 逐像素归一化视场坐标，各自的 shape 均为 [H, W]。
        x, y = torch.meshgrid(
            torch.linspace(-1, 1, W, device=self.device),
            torch.linspace(1, -1, H, device=self.device),
            indexing="xy",
        )
        # 每个像素的深度，单位为 mm（points2input 会自行执行 /1000 缩放）。
        depth_hw = depth.reshape(H, W)

        # 每个像素对应一个点光源：[H*W, 3] -> 逐像素 PSF [H*W, 3, ks, ks]。
        points = torch.stack((x, y, depth_hw), dim=-1).reshape(-1, 3).float()
        psf = self.psf_rgb(points=points, ks=ks)
        psf = psf.reshape(H, W, self.psf_chan, ks, ks)

        # 通过逐像素 PSF 散布渲染图像
        if high_res:
            render = splat_psf_per_pixel(img, psf, chunk_size=chunk_size)
        else:
            render = splat_psf_per_pixel(img, psf)

        return render
