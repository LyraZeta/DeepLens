# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF 损失函数。"""

import torch
import torch.nn as nn

class PSFLoss(nn.Module):
    """PSF 紧致性和消色差损失。

    促使点扩散函数在空间上集中，并在各颜色通道间保持相似。总损失是集中项
    （归一化 PSF 的空间方差）与消色差项（通道对之间的均方差）的加权和。

    属性:
        w_achromatic (float): 消色差（通道差异）项的权重。
        w_psf_size (float): 集中（空间方差）项的权重。
    """

    def __init__(self, w_achromatic=1.0, w_psf_size=1.0):
        """初始化 PSF 损失。

        参数:
            w_achromatic (float, optional): 消色差项权重。默认为 1.0。
            w_psf_size (float, optional): 集中项权重。默认为 1.0。
        """
        super(PSFLoss, self).__init__()
        self.w_achromatic = w_achromatic
        self.w_psf_size = w_psf_size

    def forward(self, psf):
        """计算集中性与消色差组合 PSF 损失。

        先将每个通道的 PSF 归一化至总和为 1，再在各维覆盖 $[-1, 1]$ 的归一化
        坐标网格上计算空间方差集中项。消色差项为（未归一化）PSF 所有不同通道对
        之间的均方差。

        参数:
            psf (torch.Tensor): 点扩散函数。接受 shape
                [batch, channels, height, width]、[channels, height, width]
                （会添加 batch 维度）或 [height, width]（扩展并重复为
                [1, 3, height, width]）。

        返回:
            total_loss (torch.Tensor): 标量损失，等于
                `w_psf_size * concentration_loss + w_achromatic * channel_diff`。
        """
        # 确保 psf 的 shape 为 [batch, channels, height, width]
        if psf.dim() == 3:
            psf = psf.unsqueeze(0)  # 添加 batch 维度
        elif psf.dim() == 2:
            psf = (
                psf.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
            )  # 添加 batch 和 channel 维度

        batch, channels, height, width = psf.shape

        # 在空间维度上归一化 PSF
        psf_normalized = psf / psf.view(batch, channels, -1).sum(
            dim=2, keepdim=True
        ).view(batch, channels, 1, 1)

        # 集中性损失：最小化空间方差
        # 计算坐标
        x = torch.linspace(-1, 1, steps=width, device=psf.device, dtype=torch.float32)
        y = torch.linspace(-1, 1, steps=height, device=psf.device, dtype=torch.float32)
        xv, yv = torch.meshgrid(x, y, indexing="ij")
        xv = xv.unsqueeze(0).unsqueeze(0)  # Shape [1, 1, H, W]
        yv = yv.unsqueeze(0).unsqueeze(0)

        # 计算平均位置
        mean_x = (psf_normalized * xv).sum(dim=(2, 3))
        mean_y = (psf_normalized * yv).sum(dim=(2, 3))

        # 计算方差
        var_x = ((xv - mean_x.view(batch, channels, 1, 1)) ** 2 * psf_normalized).sum(
            dim=(2, 3)
        )
        var_y = ((yv - mean_y.view(batch, channels, 1, 1)) ** 2 * psf_normalized).sum(
            dim=(2, 3)
        )
        concentration_loss = var_x + var_y
        concentration_loss = concentration_loss.mean()

        # 消色差损失：最小化通道间差异
        channel_diff = 0
        for i in range(channels):
            for j in range(i + 1, channels):
                channel_diff += torch.mean((psf[:, i, :, :] - psf[:, j, :, :]) ** 2)
        channel_diff = channel_diff / (channels * (channels - 1) / 2)

        total_loss = (
            self.w_psf_size * concentration_loss + self.w_achromatic * channel_diff
        )
        return total_loss

class PSFStrehlLoss(nn.Module):
    """类似 Strehl 的 PSF 锐度评分。

    计算 Strehl 比的代理量：对每个 PSF 按通道进行空间归一化后，取中心像素强度，
    再在通道和 batch 上求平均。值越大表示 PSF 越锐利、越紧致，因此优化时应
    最大化此评分。
    """

    def __init__(self):
        """初始化 Strehl PSF 损失。"""
        super(PSFStrehlLoss, self).__init__()

    def forward(self, psf):
        """计算类似 Strehl 的中心强度评分。

        按样本和通道归一化 PSF，使每个通道在空间维度上的总和为 1；随后读取中心
        像素强度，并在通道和 batch 上求平均。

        参数:
            psf (torch.Tensor): shape [B, 3, ks, ks] 的点扩散函数。也接受 shape
                [3, ks, ks] 的输入，并会添加 batch 维度。

        返回:
            strehl (torch.Tensor): 标量评分，等于通道和 batch 上归一化中心像素
                强度的平均值。

        异常:
            AssertionError: 添加可选 batch 维度后，`psf` 不是含 3 个通道的四维
                张量时抛出。
        """
        # 确保 shape 为 [B, 3, H, W]
        if psf.dim() == 3:
            psf = psf.unsqueeze(0)
        assert psf.dim() == 4 and psf.size(1) == 3, (
            f"Expected psf shape [B, 3, ks, ks], got {tuple(psf.shape)}"
        )

        eps = torch.finfo(psf.dtype).eps
        # 在空间维度上按样本、按通道归一化
        psf_sum = psf.sum(dim=(2, 3), keepdim=True)
        psf_norm = psf / (psf_sum + eps)

        # 中心像素索引
        h, w = psf.shape[-2:]
        cy, cx = h // 2, w // 2

        # 每个样本、每个通道的中心强度
        center_vals = psf_norm[:, :, cy, cx]  # [B, 3]

        # 先在通道上求平均，再在 batch 上求平均
        strehl_per_sample = center_vals.mean(dim=1)  # [B]
        strehl = strehl_per_sample.mean()  # 标量

        return strehl
