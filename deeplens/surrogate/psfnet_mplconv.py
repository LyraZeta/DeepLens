# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""用于表示镜头空间变化 PSF 的 MLP-Conv 网络架构。

MLP 将视场条件 (r, z) 映射为潜向量，卷积解码器再将该潜向量上采样为逐通道
PSF 核，并在空间维度上归一化，使总和为 1。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelwiseNormalization(nn.Module):
    """对各通道进行归一化，使其在空间维度上的总和为 1。

    对每个通道展平后的空间位置应用 softmax，使输出的每个通道都构成有效的
    PSF 能量分布。
    """

    def __init__(self):
        """初始化逐通道归一化模块。"""
        super(ChannelwiseNormalization, self).__init__()

    def forward(self, x):
        """应用逐通道空间 softmax 归一化。

        参数：
            x (torch.Tensor): 形状为 [batch, channels, height, width] 的输入特征图。

        返回：
            out (torch.Tensor): 形状为 [batch, channels, height, width] 的
                归一化特征图，每个通道在空间维度上的总和为 1。
        """
        # x 的形状：[batch, channels, height, width]
        # 重塑为 [batch, channels, -1]，以便在空间维度上应用 softmax
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1)
        # 沿最后一维（空间位置）应用 softmax
        x_softmax = F.softmax(x_flat, dim=2)
        # 重塑回原始形状 [batch, channels, height, width]
        return x_softmax.view(b, c, h, w)


class ResidualBlock(nn.Module):
    """包含两次卷积、批归一化和 ReLU 的残差块。

    两个 conv -> batch-norm 层的结果与捷径分支相加；捷径分支通常为恒等映射，
    当通道数或步幅变化时使用 1x1 卷积投影，最后再经过 ReLU。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        """初始化残差块。

        参数：
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int, optional): 卷积核尺寸，默认为 3。
            padding (int, optional): 卷积填充，默认为 1。
            stride (int, optional): 第一次卷积和捷径分支的步幅，默认为 1。
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二次卷积的步幅应为 1
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 默认保持捷径分支不变
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """应用残差块。

        参数：
            x (torch.Tensor): 形状为 [batch, in_channels, H, W] 的输入特征图。

        返回：
            out (torch.Tensor): 形状为 [batch, out_channels, H', W'] 的输出
                特征图，其中 H' 和 W' 按 `stride` 缩小。
        """
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        return self.relu(out)


class DecoderBlock(nn.Module):
    """先进行残差细化，再通过转置卷积进行 2 倍上采样。

    使用 `ResidualBlock` 细化特征并保持通道数不变，然后通过步幅为 2 的转置
    卷积将空间分辨率扩大一倍，最后应用批归一化和 ReLU。
    """

    def __init__(self, in_channels, out_channels):
        """初始化解码器块。

        参数：
            in_channels (int): 输入通道数。
            out_channels (int): 上采样后的输出通道数。
        """
        super().__init__()
        self.residual = ResidualBlock(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """细化特征图并将其上采样 2 倍。

        参数：
            x (torch.Tensor): 形状为 [batch, in_channels, H, W] 的输入特征图。

        返回：
            out (torch.Tensor): 形状为 [batch, out_channels, 2*H, 2*W] 的
                上采样特征图。
        """
        x = self.residual(x)  # 先进行细化
        x = self.upsample(x)  # 再进行上采样
        x = self.norm(x)
        return self.activation(x)


class MLPConditioner(nn.Module):
    """将视场条件（例如 (r, z)）映射为展平的潜向量。

    通过逐通道可学习仿射变换（缩放和平移）归一化尺度不同的输入，随后使用
    带 ReLU 激活的四层 MLP（in_chan -> 128 -> 512 -> 1024 -> latent_dim）。

    属性：
        scale (torch.Tensor): 可学习的逐输入缩放量，形状为 [in_chan]。
        shift (torch.Tensor): 可学习的逐输入平移量，形状为 [in_chan]。
    """

    def __init__(self, in_chan=2, latent_dim=4096):
        """初始化 MLP 条件网络。

        参数：
            in_chan (int, optional): 输入条件通道数，例如 (r, z) 为 2，默认为 2。
            latent_dim (int, optional): 输出潜向量维度，默认为 4096。
        """
        super(MLPConditioner, self).__init__()
        # 用可学习的缩放和平移参数处理不同的输入范围
        self.scale = nn.Parameter(torch.ones(in_chan))
        self.shift = nn.Parameter(torch.zeros(in_chan))
        self.fc = nn.Sequential(
            nn.Linear(in_chan, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
        )

    def forward(self, x):
        """将输入条件映射为潜向量。

        参数：
            x (torch.Tensor): 形状为 [batch_size, in_chan] 的输入条件。

        返回：
            latent (torch.Tensor): 形状为 [batch_size, latent_dim] 的展平潜向量。
        """
        x = x * self.scale + self.shift
        return self.fc(x)


class ConvDecoder(nn.Module):
    """使用多尺度解码器从潜向量生成 PSF 核。

    将展平潜向量重塑为 [latent_channels, 16, 16]，再通过三个 `DecoderBlock`
    上采样（16 -> 32 -> 64 -> 128）。来自 32x32 和 64x64 层级的两个 1x1
    卷积跳跃连接会插值至完整分辨率并相加，随后应用最终 3x3 卷积和逐通道
    softmax 归一化。

    默认假设 kernel_size=128（= 16 * 2**3）；`__init__` 会结合
    `latent_channels` 校验 `latent_dim` 和 `kernel_size`。

    属性：
        initial_height (int): 上采样前的初始空间尺寸（16）。
        initial_shape (tuple): 重塑目标 [latent_channels, 16, 16]。
    """

    def __init__(
        self, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16
    ):
        """初始化卷积解码器。

        参数：
            kernel_size (int, optional): 输出 PSF 核尺寸，必须等于
                16 * 2**3 = 128，默认为 128。
            out_chan (int, optional): 输出通道数，例如 RGB 为 3，默认为 3。
            latent_dim (int, optional): 输入潜向量维度，必须等于
                latent_channels * 16 * 16，默认为 4096。
            latent_channels (int, optional): 重塑后潜特征图的通道数，默认为 16。

        异常：
            AssertionError: 当 `latent_dim` 不等于 latent_channels * 16 * 16，
                或 `kernel_size` 不为 128 时抛出。
        """
        super(ConvDecoder, self).__init__()
        # 校验潜向量维度是否与重塑目标一致
        self.initial_height = (
            16  # 上采样的初始高/宽（16 -> 32 -> 64 -> 128）
        )
        self.initial_shape = (latent_channels, self.initial_height, self.initial_height)
        expected_dim = latent_channels * self.initial_height * self.initial_height
        assert latent_dim == expected_dim, (
            f"Latent dim must be {expected_dim} for reshape, got {latent_dim}"
        )

        # 如果 kernel_size 改变，需要相应调整上采样层数
        assert kernel_size == self.initial_height * (2**3), (
            f"Adjust upsample layers for kernel_size={kernel_size}"
        )

        # 将解码器块定义为独立模块，以便访问多尺度特征
        self.decoder_block1 = DecoderBlock(latent_channels, 32)  # 16x16 -> 32x32
        self.decoder_block2 = DecoderBlock(32, 16)  # 32x32 -> 64x64
        self.decoder_block3 = DecoderBlock(16, 8)  # 64x64 -> 128x128

        # 多尺度特征的跳跃连接
        self.skip_conv1 = nn.Conv2d(32, 8, 1)  # 来自 32x32 层级
        self.skip_conv2 = nn.Conv2d(16, 8, 1)  # 来自 64x64 层级

        # 最终处理层
        self.final_conv = nn.Conv2d(8, out_chan, kernel_size=3, padding=1)
        self.normalization = ChannelwiseNormalization()

    def forward(self, latent):
        """将潜向量解码为归一化 PSF 核。

        参数：
            latent (torch.Tensor): 形状为 [batch_size, latent_dim] 的展平潜向量。

        返回：
            psf (torch.Tensor): 形状为
                [batch_size, out_chan, kernel_size, kernel_size] 的 PSF 核，
                每个通道在空间维度上归一化为总和 1。
        """
        batch_size = latent.size(0)
        # 将展平潜向量重塑为初始特征图
        x = latent.view(batch_size, *self.initial_shape)

        # 保存中间特征以进行多尺度处理
        x = self.decoder_block1(x)  # 32x32，32 个通道
        skip1 = F.interpolate(
            self.skip_conv1(x), size=128, mode="bilinear", align_corners=False
        )

        x = self.decoder_block2(x)  # 64x64，16 个通道
        skip2 = F.interpolate(
            self.skip_conv2(x), size=128, mode="bilinear", align_corners=False
        )

        x = self.decoder_block3(x)  # 128x128，8 个通道

        # 合并多尺度特征
        x = x + skip1 + skip2

        # 最终处理
        x = self.final_conv(x)
        return self.normalization(x)


class PSFNet_MLPConv(nn.Module):
    """结合 MLP 条件网络和卷积解码器的空间变化 PSF 网络。

    `MLPConditioner` 将视场条件（例如 (r, z)）映射为潜向量，`ConvDecoder`
    再将其上采样为归一化 PSF 核。
    """

    def __init__(
        self,
        in_chan=2,
        kernel_size=128,
        out_chan=3,
        latent_dim=4096,
        latent_channels=16,
    ):
        """初始化 PSF 网络。

        参数：
            in_chan (int, optional): 输入条件通道数，例如 (r, z) 为 2，默认为 2。
            kernel_size (int, optional): 输出 PSF 核尺寸，必须等于 128，默认为 128。
            out_chan (int, optional): 输出通道数，例如 RGB 为 3，默认为 3。
            latent_dim (int, optional): MLP 与解码器共享的潜向量维度，必须等于
                latent_channels * 16 * 16，默认为 4096。
            latent_channels (int, optional): 重塑后潜特征图的通道数，默认为 16。
        """
        super(PSFNet_MLPConv, self).__init__()
        self.mlp = MLPConditioner(in_chan=in_chan, latent_dim=latent_dim)
        self.decoder = ConvDecoder(
            kernel_size=kernel_size,
            out_chan=out_chan,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
        )

    def forward(self, x):
        """为一批视场条件预测 PSF 核。

        参数：
            x (torch.Tensor): 形状为 [batch_size, in_chan] 的输入条件，
                例如 (r, z) 对。

        返回：
            psf (torch.Tensor): 形状为
                [batch_size, out_chan, kernel_size, kernel_size] 的 PSF 核，
                每个通道在空间维度上归一化为总和 1。
        """
        psf = self.decoder(self.mlp(x))
        return psf


# 测试代码
if __name__ == "__main__":
    # 实例化模型
    model = PSFNet_MLPConv(
        in_chan=2, kernel_size=128, out_chan=3, latent_dim=4096, latent_channels=16
    )

    # 虚拟输入：batch_size=2，并给出示例 (r, z) 值
    # r in [-1,1], z in [-10000,0]
    rz = torch.tensor(
        [
            [0.5, -5000.0],  # 示例 1
            [-0.3, -2000.0],  # 示例 2
        ]
    )  # 形状：[2, 2]

    # 前向传播
    with torch.no_grad():  # 测试时不计算梯度
        psf_output = model(rz)

    # 打印形状和示例值
    print(f"Input shape: {rz.shape}")
    print(f"Output shape: {psf_output.shape}")  # 应为 [2, 3, 128, 128]

    # 检查每个通道的输出总和是否约为 1（使用 Softmax 时）
    print(f"Sum per channel (first batch): {psf_output[0].sum(dim=(1, 2))}")
