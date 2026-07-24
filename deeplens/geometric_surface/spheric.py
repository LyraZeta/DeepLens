# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""球面。"""

import torch

from .base import EPSILON, Surface


class Spheric(Surface):
    """由曲率参数化的球形折射表面。

    半径为 $R = 1/c$、顶点位于光轴上的球面。其矢高（沿 $z$ 的表面高度）为：

    $$
    z(x, y) = \\frac{c \\rho^2}{1 + \\sqrt{1 - c^2 \\rho^2}}, \\quad
    \\rho^2 = x^2 + y^2
    $$

    属性：
        c (torch.Tensor): 表面曲率 $1/R$ [1/mm]，标量张量。
            `get_optimizer_params` 会启用其梯度以进行优化。
    """

    def __init__(
        self,
        c,
        r,
        d,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化球面。

        参数：
            c (float): 表面曲率 $1/R$ [1/mm]。平面使用 0（按平面处理）。
            r (float): 孔径半径 [mm]。
            d (float): 顶点轴向位置 [mm]。
            mat2 (str or Material): 透射侧材料。
            pos_xy (list[float], optional): 横向偏移 `[x, y]` [mm]。
                默认值为 `[0.0, 0.0]`。
            vec_local (list[float], optional): 局部表面法线方向。
                默认值为 `[0.0, 0.0, 1.0]`。
            is_square (bool, optional): 使用方形孔径而非圆形孔径。默认值为 False。
            device (str, optional): 计算设备。默认值为 `"cpu"`。
        """
        super(Spheric, self).__init__(
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.c = torch.tensor(c)
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从参数字典构造 `Spheric` 表面。

        可接受曲率半径 `roc` [mm]（转换为曲率 `c`，其中 `roc == 0` 映射为
        `c = 0`），也可直接接受曲率 `c` [1/mm]。孔径半径 `r` [mm]、顶点位置
        `d` [mm] 和透射材料 `mat2` 从字典中读取。

        参数：
            surf_dict (dict): 表面参数。必须包含 `r`、`d`、`mat2`，以及
                `roc` 或 `c` 之一。

        返回：
            surface (Spheric): 构造得到的球面。
        """
        if "roc" in surf_dict:
            if surf_dict["roc"] != 0:
                c = 1 / surf_dict["roc"]
            else:
                c = 0.0
        else:
            c = surf_dict["c"]

        return cls(
            c=c,
            r=surf_dict["r"],
            d=surf_dict["d"],
            mat2=surf_dict["mat2"],
        )

    def _sag(self, x, y):
        """计算表面矢高 $z = c\\rho^2 / (1 + \\sqrt{1 - c^2\\rho^2})$。

        其中 $\\rho^2 = x^2 + y^2$。将被开方数限制到 `EPSILON`，使其在有效
        半径之外仍保持有限值。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            sag (torch.Tensor): 沿 z 的表面矢高 [mm]，shape 与 `x` 相同。
        """
        c = self.c

        # 计算表面矢高
        r2 = x**2 + y**2
        sag = c * r2 / (1 + torch.sqrt((1 - r2 * c**2).clamp(min=EPSILON)))
        return sag

    def _dfdxy(self, x, y):
        """计算一阶矢高导数 $\\partial z/\\partial x$ 和 $\\partial z/\\partial y$。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            dfdx (torch.Tensor): 矢高关于 x 的导数 [无量纲]，shape 与 `x` 相同。
            dfdy (torch.Tensor): 矢高关于 y 的导数 [无量纲]，shape 与 `x` 相同。
        """
        c = self.c

        # 计算表面矢高导数
        r2 = x**2 + y**2
        sf = torch.sqrt((1 - r2 * c**2).clamp(min=EPSILON))
        dfdr2 = c / (2 * sf)

        dfdx = dfdr2 * 2 * x
        dfdy = dfdr2 * 2 * y

        return dfdx, dfdy

    def _d2fdxy(self, x, y):
        """通过链式法则计算二阶矢高导数。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            d2f_dx2 (torch.Tensor): $\\partial^2 z/\\partial x^2$ [1/mm]，shape 与 `x` 相同。
            d2f_dxdy (torch.Tensor): $\\partial^2 z/\\partial x\\partial y$ [1/mm]，shape 与 `x` 相同。
            d2f_dy2 (torch.Tensor): $\\partial^2 z/\\partial y^2$ [1/mm]，shape 与 `x` 相同。
        """
        c = self.c

        # 计算表面矢高导数
        r2 = x**2 + y**2
        sf = torch.sqrt((1 - r2 * c**2).clamp(min=EPSILON))

        # 一阶导数（df/dr2）
        dfdr2 = c / (2 * sf)

        # 二阶导数（d²f/dr2²）
        d2f_dr2_dr2 = (c**3) / (4 * sf**3)

        # 使用链式法则计算二阶偏导数
        d2f_dx2 = 4 * x**2 * d2f_dr2_dr2 + 2 * dfdr2
        d2f_dxdy = 4 * x * y * d2f_dr2_dr2
        d2f_dy2 = 4 * y**2 * d2f_dr2_dr2 + 2 * dfdr2

        return d2f_dx2, d2f_dxdy, d2f_dy2

    def intersect(self, ray, n=1.0):
        """在局部坐标系中解析求解光线与表面的交点。

        将光线 $p(t) = o + t\\,d$ 代入球面
        $x^2 + y^2 + (z - R)^2 = R^2$（$R = 1/c$），求解关于 $t$ 的二次
        方程，并选取交点最接近 $z = 0$ 处表面顶点的根。平坦表面
        （$|c| < $ `EPSILON`）按平面处理。落在孔径外或不存在实根的光线标记
        为无效。更新 `ray.o`、`ray.is_valid`，并对相干光线更新 `ray.opl`
        （增加 $n\\,t$）。

        参数：
            ray (Ray): 输入光线，将原位修改。
            n (float, optional): 入射介质折射率，用于更新光程。默认值为 1.0。

        返回：
            ray (Ray): 原光线，其中位置、有效性和 opl 已更新。

        异常：
            Exception: 当相干光线以 float32 传播超过 100 mm，导致 OPL 累加
                精度下降时抛出。
        """
        c = self.c

        if torch.abs(c) < EPSILON:
            # 将平坦表面按平面处理
            t = (0.0 - ray.o[..., 2]) / ray.d[..., 2]
            new_o = ray.o + t.unsqueeze(-1) * ray.d
            valid = (new_o[..., 0] ** 2 + new_o[..., 1] ** 2 < self.r**2) & (
                ray.is_valid > 0
            )
        else:
            R = 1.0 / c

            # 从光线起点指向位于 (0, 0, R) 的球心的向量
            oc = ray.o.clone()
            oc[..., 2] = oc[..., 2] - R

            # 二次方程：a*t^2 + b*t + c = 0
            # a = d·d = 1（因为光线方向已归一化）
            # b = 2*(o-center)·d
            # c = (o-center)·(o-center) - R^2

            a = torch.sum(ray.d * ray.d, dim=-1)  # 对归一化光线应为 1
            b = 2.0 * torch.sum(oc * ray.d, dim=-1)
            c_coeff = torch.sum(oc * oc, dim=-1) - R * R

            discriminant = b * b - 4 * a * c_coeff
            valid_intersect = discriminant >= 0

            sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=EPSILON))
            t1 = (-b - sqrt_discriminant) / (2 * a + EPSILON)
            t2 = (-b + sqrt_discriminant) / (2 * a + EPSILON)

            # 选择最接近 z=0（表面顶点）的交点
            z1 = ray.o[..., 2] + t1 * ray.d[..., 2]
            z2 = ray.o[..., 2] + t2 * ray.d[..., 2]
            use_t1 = torch.abs(z1) < torch.abs(z2)
            t = torch.where(use_t1, t1, t2)

            new_o = ray.o + t.unsqueeze(-1) * ray.d

            # 检查孔径
            r_squared = new_o[..., 0] ** 2 + new_o[..., 1] ** 2
            within_aperture = r_squared <= (self.r**2 + EPSILON)

            valid = valid_intersect & within_aperture & (ray.is_valid > 0)

        # 更新光线位置
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            if t.abs().max() > 100 and torch.get_default_dtype() == torch.float32:
                raise Exception(
                    "Using float32 may cause precision problem for OPL calculation."
                )
            new_opl = ray.opl + n * t.unsqueeze(-1)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def is_within_data_range(self, x, y):
        """检查点是否位于矢高定义区域内。

        仅当 $x^2 + y^2 < 1/c^2$ 时点有效，即位于球面矢高为实数的半径内。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            valid (torch.Tensor): 布尔掩膜，shape 与 `x` 相同。
        """
        c = self.c
        valid = (x**2 + y**2) < 1 / c**2
        return valid

    def max_height(self):
        """返回表面的最大有效径向高度。

        等于 $|R| = 1/|c|$ 减去 0.001 mm 的小余量，以保持在矢高定义良好的区域内。

        返回：
            max_height (float): 最大径向高度 [mm]。
        """
        c = self.c
        max_height = torch.sqrt(1 / c**2).item() - 0.001
        return max_height

    # =========================================
    # 优化
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """启用 `c` 和 `d` 的梯度并构建优化器参数组。

        参数：
            lrs (list[float], optional): 顶点位置和曲率的学习率 `[lr_d, lr_c]`。
                默认值为 `[1e-4, 1e-4]`。
            optim_mat (bool, optional): 同时优化透射材料参数（材料为空气时跳过）。
                默认值为 False。

        返回：
            params (list[dict]): 优化器参数组，每组包含 `params` 和 `lr` 键。
        """
        self.c.requires_grad_(True)
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.d], "lr": lrs[0]})
        params.append({"params": [self.c], "lr": lrs[1]})

        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # 输入输出
    # =========================================
    def surf_dict(self):
        """将表面序列化为参数字典。

        返回：
            surf_dict (dict): 表面参数，包含 `type`、`r`、`(c)`、`roc`、
                `(d)`、`mat2`，以及信息项 `(mat2_n)`/`(mat2_V)`。长度单位为
                [mm]，曲率单位为 [1/mm]，数值保留 4 位小数。
        """
        roc = 1 / self.c.item() if self.c.item() != 0 else 0.0
        surf_dict = {
            "type": "Spheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(roc, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """将表面格式化为 Zemax STANDARD 表面块。

        参数：
            surf_idx (int): Zemax 文件中的表面索引。
            d_next (torch.Tensor): 到下一表面的轴向距离 [mm]，标量张量。

        返回：
            zmx_str (str): 多行 Zemax 表面描述。
        """
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    DIAM {self.r} 1 0 0 1 ""
"""
        else:
            zmx_str = f"""SURF {surf_idx} 
    TYPE STANDARD 
    CURV {self.c.item()} 
    DISZ {d_next.item()} 
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str
