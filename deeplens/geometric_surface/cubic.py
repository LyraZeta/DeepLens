"""三次曲面。"""

import numpy as np
import torch

from .base import Surface


class Cubic(Surface):
    """三次相位板表面。

    该自由曲面的矢高是不具旋转对称性的 $x$、$y$ 奇次多项式：
    $z = b_3 (x^3 + y^3) + b_5 (x^5 + y^5) + b_7 (x^7 + y^7)$。
    有效项数由 `b` 的长度决定（1 至 3 阶）。这类三次相位掩膜用于波前编码／
    扩展景深设计。

    属性：
        b (torch.Tensor): 包含所有三次曲面系数的一维张量，单位依次为
            [1/mm^2]、[1/mm^4]、[1/mm^6]。
        b3 (torch.Tensor): 三次项（$x^3 + y^3$）的标量系数，单位为 [1/mm^2]。
        b5 (torch.Tensor): 五次项（$x^5 + y^5$）的标量系数，单位为 [1/mm^4]。
            仅当 b_degree 至少为 2 时存在。
        b7 (torch.Tensor): 七次项（$x^7 + y^7$）的标量系数，单位为 [1/mm^6]。
            仅当 b_degree 为 3 时存在。
        b_degree (int): 有效多项式项数（1、2 或 3）。
        rotate_angle (float): 表面的面内旋转角，单位为 rad。
    """

    def __init__(
        self,
        r,
        d,
        b,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化三次相位板表面。

        参数：
            r (float): 孔径半径（半直径），单位为 [mm]。
            d (float): 表面沿光轴的轴向距离（位置），单位为 [mm]。
            b (list): 按 $[b_3, b_5, b_7]$ 排列的三次曲面系数。其长度
                （1、2 或 3）决定多项式阶数；单位为 [1/mm^2]、[1/mm^4]、[1/mm^6]。
            mat2 (str or Material): 表面后的材料。
            pos_xy (list, optional): 表面的横向 $(x, y)$ 偏移，单位为 [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list, optional): 局部表面法线（光轴）方向。
                默认值为 [0.0, 0.0, 1.0]。
            is_square (bool, optional): 孔径是否为方形而非圆形。默认值为 False。
            device (str, optional): Torch 设备。默认值为 "cpu"。

        异常：
            ValueError: 当 `b` 的长度不是 1、2 或 3 时抛出。
        """
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.b = torch.tensor(b)

        if len(b) == 1:
            self.b3 = torch.tensor(b[0])
            self.b_degree = 1
        elif len(b) == 2:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b_degree = 2
        elif len(b) == 3:
            self.b3 = torch.tensor(b[0])
            self.b5 = torch.tensor(b[1])
            self.b7 = torch.tensor(b[2])
            self.b_degree = 3
        else:
            raise ValueError("Unsupported cubic degree!")

        self.rotate_angle = 0.0
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从参数字典构造 `Cubic` 表面。

        参数：
            surf_dict (dict): 包含 "r"、"d"、"b" 和 "mat2" 的表面参数。

        返回：
            surf (Cubic): 构造得到的三次曲面。
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["b"], surf_dict["mat2"])

    def _sag(self, x, y):
        """计算三次相位板的表面矢高 $z(x, y)$。

        计算截至有效阶数的
        $z = b_3 (x^3 + y^3) + b_5 (x^5 + y^5) + b_7 (x^7 + y^7)$，
        并可选择先施加面内 `rotate_angle`。

        参数：
            x (torch.Tensor): 局部 x 坐标，单位为 [mm]。
            y (torch.Tensor): 局部 y 坐标，单位为 [mm]，可与 `x` 广播。

        返回：
            z (torch.Tensor): $(x, y)$ 处的表面矢高，单位为 [mm]。
        """
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            z = self.b3 * (x**3 + y**3)
        elif self.b_degree == 2:
            z = self.b3 * (x**3 + y**3) + self.b5 * (x**5 + y**5)
        elif self.b_degree == 3:
            z = (
                self.b3 * (x**3 + y**3)
                + self.b5 * (x**5 + y**5)
                + self.b7 * (x**7 + y**7)
            )
        else:
            raise ValueError("Unsupported cubic degree!")

        if z.dim() == 0:
            z = z.clone().detach().to(self.device)

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return z

    def _dfdxy(self, x, y):
        """计算矢高相对于 $x$ 和 $y$ 的偏导数。

        参数：
            x (torch.Tensor): 局部 x 坐标，单位为 [mm]。
            y (torch.Tensor): 局部 y 坐标，单位为 [mm]，可与 `x` 广播。

        返回：
            dfdx (torch.Tensor): 偏导数 $\\partial z / \\partial x$，无量纲。
            dfdy (torch.Tensor): 偏导数 $\\partial z / \\partial y$，无量纲。
        """
        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) - y * float(
                np.sin(self.rotate_angle)
            )
            y = x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        if self.b_degree == 1:
            dfdx = 3 * self.b3 * x**2
            dfdy = 3 * self.b3 * y**2
        elif self.b_degree == 2:
            dfdx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4
            dfdy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4
        elif self.b_degree == 3:
            dfdx = 3 * self.b3 * x**2 + 5 * self.b5 * x**4 + 7 * self.b7 * x**6
            dfdy = 3 * self.b3 * y**2 + 5 * self.b5 * y**4 + 7 * self.b7 * y**6
        else:
            raise ValueError("Unsupported cubic degree!")

        if self.rotate_angle != 0:
            x = x * float(np.cos(self.rotate_angle)) + y * float(
                np.sin(self.rotate_angle)
            )
            y = -x * float(np.sin(self.rotate_angle)) + y * float(
                np.cos(self.rotate_angle)
            )

        return dfdx, dfdy

    def get_optimizer_params(self, lrs=[1e-4], decay=0.1, optim_mat=False):
        """为该表面的每个参数构建优化器参数组。

        启用轴向距离 `d` 和有效三次曲面系数（以及可选材料）的梯度。若只给定
        一个学习率，则按 `decay` 的幂得到高阶系数的学习率。

        参数：
            lrs (list, optional): 学习率。单元素列表通过 `decay` 扩展到所有系数。
                默认值为 [1e-4]。
            decay (float, optional): 应用于高阶系数学习率的几何衰减因子。
                默认值为 0.1。
            optim_mat (bool, optional): 是否同时优化材料参数。默认值为 False。

        返回：
            params (list): 用于 torch 优化器的参数组字典列表
                （{"params": [...], "lr": ...}）。

        异常：
            ValueError: 当 `b_degree` 不是 1、2 或 3 时抛出。
        """
        # 将学习率扩展到所有三次曲面系数
        if len(lrs) == 1:
            lrs = lrs + [
                lrs[0] * decay ** (b_degree + 1)
                for b_degree in range(self.b_degree - 1)
            ]

        params = []

        # 优化距离
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # 优化三次曲面系数
        if self.b_degree == 1:
            self.b3.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
        elif self.b_degree == 2:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
            params.append({"params": [self.b5], "lr": lrs[2]})
        elif self.b_degree == 3:
            self.b3.requires_grad_(True)
            self.b5.requires_grad_(True)
            self.b7.requires_grad_(True)
            params.append({"params": [self.b3], "lr": lrs[1]})
            params.append({"params": [self.b5], "lr": lrs[2]})
            params.append({"params": [self.b7], "lr": lrs[3]})
        else:
            raise ValueError("Unsupported cubic degree!")

        # 优化材料参数
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # 输入输出
    # =========================================
    def surf_dict(self):
        """将表面参数序列化为字典。

        输出 `init_from_dict` 使用的 `b` 系数列表和 `mat2`。为便于阅读，保留
        标量键 `b3`/`b5`/`b7`；轴向位置写入带括号的 `(d)` 键（仅用于显示，
        加载器根据累积表面间距重建 `d`）。

        返回：
            d (dict): 表面参数，包含 "type"、"b3"、"r"、"(d)"、"b"、
                "mat2"、信息项 "(mat2_n)"/"(mat2_V)"，以及有效时的 "b5"/"b7"。
        """
        b = [self.b3.item()]
        if self.b_degree >= 2:
            b.append(self.b5.item())
        if self.b_degree >= 3:
            b.append(self.b7.item())

        d = {
            "type": "Cubic",
            "b3": self.b3.item(),
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "b": b,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        if self.b_degree >= 2:
            d["b5"] = self.b5.item()
        if self.b_degree >= 3:
            d["b7"] = self.b7.item()
        return d
