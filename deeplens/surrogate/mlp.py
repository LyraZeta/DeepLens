import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """用于低频 PSF 预测的全连接网络。

    使用堆叠线性层、ReLU 激活和 Sigmoid 输出，将 PSF 预测为展平向量。
    输出经过 L1 归一化，使其总和为 1，从而构成有效的 PSF 能量分布。

    参数：
        in_features (int): 输入特征数，例如视场角与波长。
        out_features (int): 输出特征数，即展平后的 PSF 大小。
        hidden_features (int): 隐藏层宽度，默认为 64。
        hidden_layers (int): 隐藏层数量，默认为 3。
    """

    def __init__(self, in_features, out_features, hidden_features=64, hidden_layers=3):
        super(MLP, self).__init__()

        layers = [
            nn.Linear(in_features, hidden_features // 4, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features // 4, hidden_features, bias=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(hidden_layers):
            layers.extend(
                [
                    nn.Linear(hidden_features, hidden_features, bias=True),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.extend(
            [nn.Linear(hidden_features, out_features, bias=True), nn.Sigmoid()]
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """执行前向传播。

        参数：
            x (torch.Tensor): 形状为 `(batch_size, in_features)` 的输入张量。

        返回：
            x (torch.Tensor): 形状为 `(batch_size, out_features)` 的 L1
                归一化输出张量，沿最后一维求和为 1。
        """
        x = self.net(x)
        x = F.normalize(x, p=1, dim=-1)
        return x


if __name__ == "__main__":
    # 测试网络
    mlp = MLP(4, 64, hidden_features=64, hidden_layers=3)
    print(mlp)
    x = torch.rand(100, 4)
    y = mlp(x)
    print(y.size())
