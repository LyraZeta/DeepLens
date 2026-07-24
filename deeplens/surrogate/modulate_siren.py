import math
from os.path import exists

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModulateSiren(nn.Module):
    """用于潜变量条件图像合成的调制 SIREN。

    将 SIREN 合成器网络（把固定像素坐标网格映射为输出值）与调制器网络结合，
    后者根据条件潜向量缩放各合成器层。该模型用于预测以镜头参数为条件的空间
    变化 PSF。无论 `outermost_linear` / `final_activation` 如何设置，输出始终
    经过 tanh 激活并重塑为图像。

    属性：
        synthesizer (nn.ModuleList): SIREN 正弦层及最终输出层。
        modulator (nn.ModuleList): 各层的 Linear+ReLU 模块，根据潜向量（以及
            上一层调制结果）生成调制向量。
        grid (torch.Tensor): 已注册的坐标缓冲区，形状为
            `(image_height * image_width, dim_in)`，每个坐标轴覆盖 $[-1, 1]$。

    参数：
        dim_in (int): 输入坐标维度，x、y 坐标通常为 2。
        dim_hidden (int): 合成器和调制器的隐藏层宽度。
        dim_out (int): 每个像素的输出维度，例如灰度 PSF 为 1。
        dim_latent (int): 条件潜向量的维度。
        num_layers (int): SIREN 与调制器层数，不含合成器最终输出层。
        image_width (int): 输出图像宽度，单位为像素。
        image_height (int): 输出图像高度，单位为像素。
        w0 (float, optional): 隐藏正弦层的频率乘子，默认为 1.0。
        w0_initial (float, optional): 首个正弦层的频率乘子，默认为 30.0。
        use_bias (bool, optional): 正弦层是否使用偏置，默认为 True。
        final_activation (nn.Module or None, optional): 当 `outermost_linear` 为
            False 时最终 `Siren` 层使用的激活函数，默认为 None（Identity）。
        outermost_linear (bool, optional): 为 True 时，合成器最终层是普通
            `nn.Linear`；否则为 `Siren` 层。默认为 True。
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        dim_latent,
        num_layers,
        image_width,
        image_height,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
        outermost_linear=True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.img_width = image_width
        self.img_height = image_height

        # ==> 合成器
        synthesizer_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            synthesizer_layers.append(
                SineLayer(
                    in_features=layer_dim_in,
                    out_features=dim_hidden,
                    omega_0=layer_w0,
                    bias=use_bias,
                    is_first=is_first,
                )
            )

        if outermost_linear:
            last_layer = nn.Linear(dim_hidden, dim_out)
            with torch.no_grad():
                # w_std = math.sqrt(6 / dim_hidden) / w0
                # self.last_layer.weight.uniform_(- w_std, w_std)
                nn.init.kaiming_normal_(
                    last_layer.weight, a=0.0, nonlinearity="relu", mode="fan_in"
                )
        else:
            final_activation = (
                nn.Identity() if not exists(final_activation) else final_activation
            )
            last_layer = Siren(
                dim_in=dim_hidden,
                dim_out=dim_out,
                w0=w0,
                use_bias=use_bias,
                activation=final_activation,
            )
        synthesizer_layers.append(last_layer)

        self.synthesizer = synthesizer_layers
        # self.synthesizer = nn.Sequential(*synthesizer)

        # ==> 调制器
        modulator_layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_latent if is_first else (dim_hidden + dim_latent)

            modulator_layers.append(
                nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU())
            )

            with torch.no_grad():
                # self.layers[-1][0].weight.uniform_(-1 / dim_hidden, 1 / dim_hidden)
                nn.init.kaiming_normal_(
                    modulator_layers[-1][0].weight,
                    a=0.0,
                    nonlinearity="relu",
                    mode="fan_in",
                )

        self.modulator = modulator_layers
        # self.modulator = nn.Sequential(*modulator_layers)

        # ==> 坐标位置
        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)

    def forward(self, latent):
        """根据条件潜向量合成一批图像。

        将共享坐标网格送入 SIREN 合成器，并用相应调制器输出缩放每一层，
        随后应用 tanh，并重塑为通道优先的图像批次。

        参数：
            latent (torch.Tensor): 形状为 `(batch_size, dim_latent)` 的条件潜向量。

        返回：
            x (torch.Tensor): 形状为 `(batch_size, 1, image_height, image_width)`
                的输出图像张量，取值范围为 $[-1, 1]$。
        """
        x = self.grid.clone().detach().requires_grad_()

        for i in range(self.num_layers):
            if i == 0:
                z = self.modulator[i](latent)
            else:
                z = self.modulator[i](torch.cat((latent, z), dim=-1))

            x = self.synthesizer[i](x)
            x = x * z

        x = self.synthesizer[-1](x)  # 形状为 (h*w, 1)
        x = torch.tanh(x)
        x = x.view(
            -1, self.img_height, self.img_width, 1
        )  # 重塑为 (batch_size, height, width, channels)
        x = x.permute(0, 3, 1, 2)  # 重塑为 (batch_size, channels, height, width)
        return x


class SineLayer(nn.Module):
    """在线性投影后应用正弦非线性的单个 SIREN 层。

    计算 $\\sin(\\omega_0 \\cdot (W x + b))$，并按照 SIREN 方案初始化权重，
    使激活值在不同网络深度下保持稳定分布。

    属性：
        linear (nn.Linear): 正弦运算前应用的仿射投影。
        omega_0 (float): 正弦函数内部的频率乘子。
        is_first (bool): 是否为首层；该值会改变权重初始化方式。

    参数：
        in_features (int): 输入特征维度。
        out_features (int): 输出特征维度。
        bias (bool, optional): 是否包含偏置项，默认为 True。
        is_first (bool, optional): 是否为首个 SIREN 层，默认为 False。
        omega_0 (float, optional): 正弦函数内部的频率乘子，默认为 30。
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """按照 SIREN 方案初始化线性层权重。

        首层从 $[-1/n, 1/n]$ 均匀采样；后续层从
        $[-\\sqrt{6/n}/\\omega_0, \\sqrt{6/n}/\\omega_0]$ 均匀采样，
        其中 $n$ 为 `in_features`。
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        """应用线性投影及带缩放的正弦函数。

        参数：
            input (torch.Tensor): 形状为 `(..., in_features)` 的输入张量。

        返回：
            out (torch.Tensor): 形状为 `(..., out_features)` 的激活后张量。
        """
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):
    """显式保存权重/偏置参数并使用正弦激活的 SIREN 层。

    功能等价于 `SineLayer`，但会把 `weight`/`bias` 保存为原始 `nn.Parameter`
    张量，并允许自定义激活函数（默认为 `Sine`）。当 `outermost_linear` 为
    False 时，该层用作 `ModulateSiren` 的最终合成器层。

    属性：
        weight (nn.Parameter): 形状为 `(dim_out, dim_in)` 的权重张量。
        bias (nn.Parameter or None): 形状为 `(dim_out,)` 的偏置张量，或为 None。
        activation (nn.Module): 在线性投影后应用的非线性函数。

    参数：
        dim_in (int): 输入特征维度。
        dim_out (int): 输出特征维度。
        w0 (float, optional): 传给默认 `Sine` 激活函数并用于权重初始化的频率乘子，
            默认为 1.0。
        c (float, optional): 权重初始化边界 $\\sqrt{c/\\text{dim\\_in}}/w_0$
            中的常数，默认为 6.0。
        is_first (bool, optional): 是否为首个 SIREN 层；该值会改变权重初始化，
            默认为 False。
        use_bias (bool, optional): 是否包含偏置项，默认为 True。
        activation (nn.Module or None, optional): 在线性投影后应用的激活函数。
            默认为 None，即使用频率为 `w0` 的 `Sine`。
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        """按照 SIREN 方案原地初始化权重。

        从 $[-w_{std}, w_{std}]$ 均匀采样。首层的 $w_{std}$ 为
        $1/\\text{dim\\_in}$，其他层为 $\\sqrt{c/\\text{dim\\_in}}/w_0$。
        `bias` 参数会被接收但保持不变，由调用方将其初始化为零。

        参数：
            weight (torch.Tensor): 形状为 `(dim_out, dim_in)` 的权重张量，
                将被原地修改。
            bias (torch.Tensor or None): 偏置张量，未使用。
            c (float): 非首层标准差边界中的常数。
            w0 (float): 非首层标准差边界使用的频率乘子。
        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

    def forward(self, x):
        """应用线性投影及后续激活函数。

        参数：
            x (torch.Tensor): 形状为 `(..., dim_in)` 的输入张量。

        返回：
            out (torch.Tensor): 形状为 `(..., dim_out)` 的激活后张量。
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Sine(nn.Module):
    """计算 $\\sin(w_0 x)$ 的正弦激活模块。

    频率乘子 $w_0$ 是 `Siren` 层默认激活函数使用的参数。

    参数：
        w0 (float, optional): 正弦函数内部的频率乘子，默认为 1.0。
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """应用带缩放的正弦激活。

        参数：
            x (torch.Tensor): 任意形状的输入张量。

        返回：
            out (torch.Tensor): 与输入形状相同、逐元素应用 $\\sin(w_0 x)$ 后的张量。
        """
        return torch.sin(self.w0 * x)
