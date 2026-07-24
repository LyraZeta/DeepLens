import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Siren(nn.Module):
    """单个 SIREN（Sinusoidal Representation Network）层。

    由一个线性层和随后的正弦激活组成，采用论文
    "Implicit Neural Representations with Periodic Activation Functions" 中的初始化方案。

    参数：
        dim_in (int): 输入维度。
        dim_out (int): 输出维度。
        w0 (float): 正弦激活的频率乘子，默认为 1.0。
        c (float): 控制非首层权重初始化尺度的常数，默认为 6.0。
        is_first (bool): 是否为首层；首层使用不同的初始化尺度，默认为 False。
        use_bias (bool): 是否包含偏置项，默认为 True。
        activation (nn.Module or None, optional): 自定义激活模块。默认为 None，
            此时使用 `Sine(w0)`。
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
        """使用 SIREN 方案原地初始化层权重。

        在 $[-w_{std}, w_{std}]$ 上均匀填充 `weight`。首层的尺度为
        $1/\\text{dim}$，其他层为 $\\sqrt{c/\\text{dim}}/w_0$。
        `bias` 参数仅为保持 API 对称而接收，不会被修改，仍保持零初始化值。

        参数：
            weight (torch.Tensor): 形状为 `(dim_out, dim_in)` 的权重张量，将被原地修改。
            bias (torch.Tensor or None): 形状为 `(dim_out,)` 的偏置张量，不会修改。
            c (float): 控制初始化尺度的常数。
            w0 (float): 正弦激活的频率乘子。
        """
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

    def forward(self, x):
        """执行前向传播。

        参数：
            x (torch.Tensor): 形状为 `(..., dim_in)` 的输入张量。

        返回：
            out (torch.Tensor): 形状为 `(..., dim_out)` 的输出张量。
        """
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class Sine(nn.Module):
    """带频率乘子的正弦激活。

    逐元素应用 $\\sin(w_0 x)$，这是 SIREN 网络采用的周期激活函数。

    参数：
        w0 (float): 应用于正弦函数之前的频率乘子，默认为 1.0。
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        """逐元素应用正弦激活。

        参数：
            x (torch.Tensor): 任意形状的输入张量。

        返回：
            out (torch.Tensor): 与输入形状相同、值为 $\\sin(w_0 x)$ 的张量。
        """
        return torch.sin(self.w0 * x)
