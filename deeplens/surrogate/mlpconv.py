import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPConv(nn.Module):
    """用于高分辨率 PSF 预测的 MLP 编码器与卷积解码器。

    MLP 编码器将输入特征映射为空间尺寸为 `min(ks, 32)` 的低分辨率特征图，
    随后转置卷积解码器按 2 的幂逐级上采样至目标 PSF 尺寸 `ks`。解码器输出
    始终经过 Sigmoid 激活，并在两个空间维度上进行 L1 归一化，使每个预测
    PSF 的总和为 1。

    参考文献：
        "Differentiable Compound Optics and Processing Pipeline Optimization for
        End-To-end Camera Design".

    属性：
        ks (int): 输出 PSF 的空间尺寸。
        ks_mlp (int): MLP 特征图的空间尺寸，即 `min(ks, 32)`。
        channels (int): 输出通道数。
        encoder (nn.Sequential): 生成特征图的线性编码器。
        decoder (nn.Sequential): 使用转置卷积的上采样解码器。
        activation (nn.Module): 由 `activation` 选择的激活模块。请注意，无论
            该属性为何值，前向传播都会使用 Sigmoid。

    参数：
        in_features (int): 输入特征数，例如视场角与波长。
        ks (int): 输出 PSF 的空间尺寸。大于 32 时必须是 32 的倍数（通过断言
            检查），实际应为 $32 \\cdot 2^n$，以便解码器按 2 的整数次幂上采样。
        channels (int, optional): 输出通道数，默认为 3。
        activation (str, optional): 激活函数名称，可为 `"relu"` 或 `"sigmoid"`。
            该值保存在 `self.activation` 中，但 `forward` 不会使用，默认为 `"relu"`。
    """

    def __init__(self, in_features, ks, channels=3, activation="relu"):
        super(MLPConv, self).__init__()

        self.ks_mlp = min(ks, 32)
        upsample_times = 0  # ks <= 32 时无需上采样，解码器循环执行 0 次
        if ks > 32:
            assert ks % 32 == 0, "ks must be 32n"
            upsample_times = int(math.log(ks / 32, 2))

        linear_output = channels * self.ks_mlp**2
        self.ks = ks
        self.channels = channels

        # MLP 编码器
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, linear_output),
        )

        # 卷积解码器
        conv_layers = []
        conv_layers.append(
            nn.ConvTranspose2d(channels, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        for _ in range(upsample_times):
            conv_layers.append(
                nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
            )
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Upsample(scale_factor=2))

        conv_layers.append(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1)
        )
        conv_layers.append(nn.ReLU())
        conv_layers.append(
            nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1)
        )
        self.decoder = nn.Sequential(*conv_layers)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        """根据输入特征向量预测归一化 PSF。

        将 `x` 编码为形状 `(batch_size, channels, ks_mlp, ks_mlp)` 的特征图，
        经卷积解码器上采样后应用 Sigmoid，并在空间维度上进行 L1 归一化，
        使每个 PSF 的总和为 1。

        参数：
            x (torch.Tensor): 形状为 `(batch_size, in_features)` 的输入张量。

        返回：
            decoded (torch.Tensor): 形状为 `(batch_size, channels, ks, ks)` 的
                归一化 PSF 张量。
        """
        # 使用 MLP 编码输入
        encoded = self.encoder(x)

        # 重塑 MLP 输出以输入 CNN
        decoded_input = encoded.view(
            -1, self.channels, self.ks_mlp, self.ks_mlp
        )  # 重塑为 (batch_size, channels, height, width)

        # 使用 CNN 解码输出
        decoded = self.decoder(decoded_input)

        # 此归一化方式仅适用于 PSF 网络
        decoded = nn.Sigmoid()(decoded)
        decoded = F.normalize(decoded, p=1, dim=[-1, -2])

        return decoded


if __name__ == "__main__":
    # 测试用例
    # 创建具有 4 个输入特征和 64x64 输出的模型
    model = MLPConv(in_features=4, ks=64, channels=3)

    # 创建批大小为 1、包含 4 个特征的虚拟输入张量
    # 形状：[batch_size, in_features]
    input_tensor = torch.randn(1, 4)

    # 获取模型输出
    output_tensor = model(input_tensor)

    # 打印形状
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)

    # 验证输出形状
    # 预期形状：[batch_size, channels, ks, ks]
    assert output_tensor.shape == (1, 3, 64, 64)
    print("Test passed!")


