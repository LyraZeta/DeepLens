"""平面表面，通常为矩形，可用作红外滤光片、镜头保护玻璃或 DOE 基底。"""

import torch

from .base import Surface


class Plane(Surface):
    """矢高为零的平坦平面表面。

    对红外滤光片、镜头保护玻璃或 DOE 基底等平面光学元件进行建模。孔径默认为
    圆形，设置 `is_square` 后为方形。`Aperture`、`Mirror` 和 `ThinLens`
    均继承此类。

    属性：
        r (float): 孔径半径 [mm]。对于方形孔径，该值为外接圆半径（半对角线）。
        d (torch.Tensor): 顶点轴向位置 [mm]。
        mat2 (Material): 表面透射侧的材料。
        is_square (bool): 孔径是否为方形而非圆形。
        w (float): 方形孔径宽度 [mm]，仅在 `is_square` 时存在。
        h (float): 方形孔径高度 [mm]，仅在 `is_square` 时存在。
    """

    def __init__(
        self,
        r,
        d,
        mat2,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化平坦平面表面。

        参数：
            r (float): 孔径半径 [mm]。对于方形孔径，该值为外接圆半径
                （半对角线），因此边长为 $r\\sqrt{2}$。
            d (float): 表面顶点的轴向位置 [mm]。
            mat2 (str or Material): 透射侧材料（例如 `"N-BK7"`、`"air"`）。
            pos_xy (list[float], optional): 横向偏移 $[x, y]$ [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list[float], optional): 局部法线方向。
                默认值为 [0.0, 0.0, 1.0]（轴上）。
            is_square (bool, optional): 使用方形孔径。默认值为 False。
            device (str, optional): 计算设备。默认值为 "cpu"。
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

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从序列化的表面字典构造 Plane。

        参数：
            surf_dict (dict): 表面参数，包含 "r"（半径 [mm]）、"d"
                （轴向位置 [mm]）和 "mat2"（透射材料）。

        返回：
            plane (Plane): 重建得到的平面表面。
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def intersect(self, ray, n=1.0):
        """在局部坐标系中求解光线与平面的交点并更新光线。

        使用闭式解 $t = -o_z / d_z$（局部坐标系中的平面位于 $z = 0$），
        与使用 Newton 法的基础表面不同。落在孔径外或已无效的光线保留原始
        起点并标记为无效。对于相干光线，光程增加 $n\\,t$。

        参数：
            ray (Ray): 局部坐标系中的入射光线束，起点 `o` 和方向 `d` 的
                shape 为 (..., 3)。
            n (float, optional): 入射介质折射率，用于累加相干光线的光程。
                默认值为 1.0。

        返回：
            ray (Ray): 原光线对象，其中 `o`、`is_valid` 以及相干时的 `opl`
                已原位更新。
        """
        # 求解交点
        t = (0.0 - ray.o[..., 2]) / ray.d[..., 2]
        new_o = ray.o + t.unsqueeze(-1) * ray.d
        
        # 孔径掩膜
        if self.is_square:
            valid = (
                (torch.abs(new_o[..., 0]) < self.w / 2)
                & (torch.abs(new_o[..., 1]) < self.h / 2)
                & (ray.is_valid > 0)
            )
        else:
            valid = (new_o[..., 0] ** 2 + new_o[..., 1] ** 2 < self.r**2) & (
                ray.is_valid > 0
            )

        # 更新光线
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            ray.opl = torch.where(
                valid.unsqueeze(-1), ray.opl + n * t.unsqueeze(-1), ray.opl
            )

        return ray

    def normal_vec(self, ray):
        """返回局部坐标系中交点处的平面法线。

        平面法线恒为 $(0, 0, \\pm 1)$，并会翻转，使其指向光线来向一侧
        （与光线的 z 方向相反）。

        参数：
            ray (Ray): 方向 `d` 的 shape 为 (..., 3) 的光线束。

        返回：
            normal_vec (torch.Tensor): shape 为 (..., 3) 的单位法向量。
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1

        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def _sag(self, x, y):
        """返回表面矢高；对于平面，其恒为零。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]。

        返回：
            sag (torch.Tensor): 与 `x` shape 相同的零值 [mm]。
        """
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        """返回一阶矢高导数；对于平面，两个方向均为零。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]。

        返回：
            dfdx (torch.Tensor): 与 `x` shape 相同的零值 [1]。
            dfdy (torch.Tensor): 与 `x` shape 相同的零值 [1]。
        """
        return torch.zeros_like(x), torch.zeros_like(x)

    def _d2fdxy(self, x, y):
        """返回二阶矢高导数；对于平面，所有导数均为零。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]。

        返回：
            d2fdx2 (torch.Tensor): 与 `x` shape 相同的零值 [1/mm]。
            d2fdxdy (torch.Tensor): 与 `x` shape 相同的零值 [1/mm]。
            d2fdy2 (torch.Tensor): 与 `x` shape 相同的零值 [1/mm]。
        """
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # 优化
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """启用轴向位置 `d` 的梯度并返回优化器参数组。

        参数：
            lrs (list[float], optional): 学习率；`lrs[0]` 用于轴向位置 `d`。
                默认值为 [1e-4]。
            optim_mat (bool, optional): 若为 True，还附加材料的优化器参数
                （材料为空气时跳过）。默认值为 False。

        返回：
            params (list[dict]): 优化器参数组，每组为包含 "params" 和 "lr"
                键的字典。
        """
        params = []

        # 优化 d
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # 优化材料参数
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =========================================
    # 输入输出
    # =========================================
    def surf_dict(self):
        """将平面表面序列化为字典以便保存。

        返回：
            surf_dict (dict): 表面参数，包含 "type"、"r"（半径 [mm]）、
                "(d)"（经舍入的轴向位置 [mm]）、"is_square" 和 "mat2"
                （材料名称）。
        """
        surf_dict = {
            "type": "Plane",
            "r": self.r,
            "(d)": round(self.d.item(), 4),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        return surf_dict
