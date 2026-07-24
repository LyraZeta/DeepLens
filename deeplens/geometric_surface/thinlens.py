"""薄透镜元件，两侧均为空气。"""

import torch
import torch.nn.functional as F

from .plane import Plane


class ThinLens(Plane):
    """两侧均为空气的理想薄透镜。

    该零厚度近轴透镜的焦距为 `f` [mm]，位于轴向位置 `d` [mm]。它将所有
    光线无像差地折向轴上焦点（`f` 大于 0 时为后焦点，`f` 小于 0 时为前方
    虚焦点），并在相干模式下施加菲涅耳透镜的理想二次相位（光程）。
    其平面几何结构继承自 `Plane`。

    属性：
        f (torch.Tensor): 焦距 [mm]，标量张量。
        d (torch.Tensor): 透镜轴向位置 [mm]，标量张量。
        r (float): 孔径半径 [mm]（`is_square` 时为半对角线）。
    """

    def __init__(
        self,
        r,
        d,
        f=100.0,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化薄透镜表面。

        参数：
            r (float): 孔径半径 [mm]（`is_square` 时为半对角线）。
            d (float): 透镜轴向位置 [mm]。
            f (float, optional): 焦距 [mm]；正值会聚，负值发散。默认值为 100.0。
            pos_xy (list[float], optional): 横向偏移 [x, y] [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list[float], optional): 局部表面法线方向；内部会归一化。
                默认值为 [0.0, 0.0, 1.0]（轴上）。
            is_square (bool, optional): 使用方形孔径而非圆形孔径。默认值为 False。
            device (str, optional): 计算设备。默认值为 "cpu"。
        """
        Plane.__init__(
            self,
            r=r,
            d=d,
            mat2="air",
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.f = torch.tensor(f, device=device)

    def set_f(self, f):
        """设置焦距。

        参数：
            f (float): 新焦距 [mm]。
        """
        self.f = torch.tensor(f, device=self.device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从表面字典构造 ThinLens。

        参数：
            surf_dict (dict): 包含 "r"、"d" 和 "f" 的表面参数。

        返回：
            thinlens (ThinLens): 构造得到的薄透镜表面。
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["f"])

    # =========================================
    # 优化
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4, 1e-4], optim_mat=False):
        """启用轴向位置和焦距的梯度，并返回优化器参数组。

        参数：
            lrs (list[float], optional): 学习率；`lrs[0]` 用于轴向位置 `d`，
                `lrs[1]` 用于焦距 `f`。默认值为 [1e-4, 1e-4]。
            optim_mat (bool, optional): 薄透镜两侧均为空气，因此未使用。
                默认值为 False。

        返回：
            params (list[dict]): 优化器参数组，每组为包含 "params" 和 "lr"
                键的字典。
        """
        params = []

        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        self.f.requires_grad_(True)
        params.append({"params": [self.f], "lr": lrs[1]})

        return params

    def refract(self, ray, eta=1.0):
        """使光线束通过理想薄透镜折射。

        方向相同的光线会聚到焦平面上的同一点。将每条光线的方向平移到透镜中心
        以求得会聚点；新方向从光线交点指向该点（发散透镜则背离该点）。在相干
        模式下，光程会加入菲涅耳透镜的理想二次相位；对于正向／反向传播，
        $\\Delta\\,\\text{opl} = \\mp (x^2 + y^2) / (2 f d_z)$。

        参数：
            ray (Ray): 局部坐标系中的入射光线束。起点 `ray.o` 和方向 `ray.d`
                的 shape 为 (..., num_rays, 3) [mm]。
            eta (float, optional): 折射率比；理想薄透镜不使用。默认值为 1.0。

        返回：
            ray (Ray): 原光线束，其中 `d`（以及相干模式下的 `opl`）已原位更新。
        """
        forward = (ray.d * ray.is_valid.unsqueeze(-1))[..., 2].sum() > 0

        # 计算会聚点
        if forward:
            t0 = self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = (
                (self.d + self.f).view(1).expand_as(xy_final[..., 0].unsqueeze(-1))
            )
            o_final = torch.cat([xy_final, z_final], dim=-1)
        else:
            t0 = -self.f / ray.d[..., 2]
            xy_final = ray.d[..., :2] * t0.unsqueeze(-1)
            z_final = (
                (self.d - self.f).view(1).expand_as(xy_final[..., 0].unsqueeze(-1))
            )
            o_final = torch.cat([xy_final, z_final], dim=-1)

        # 新光线方向
        if self.f > 0:
            new_d = o_final - ray.o
        else:
            new_d = ray.o - o_final
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = new_d

        # 光程变化
        if ray.is_coherent:
            valid = ray.is_valid > 0
            if forward:
                new_opl = (
                    ray.opl
                    - (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )
                ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)
            else:
                new_opl = (
                    ray.opl
                    + (ray.o[..., 0] ** 2 + ray.o[..., 1] ** 2)
                    / self.f
                    / 2
                    / ray.d[..., 2]
                )
                ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def _sag(self, x, y):
        """返回表面矢高；对于薄透镜，其恒为零。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]。

        返回：
            sag (torch.Tensor): 零矢高 [mm]，shape 与 `x` 相同。
        """
        return torch.zeros_like(x)

    def _dfdxy(self, x, y):
        """返回矢高梯度；对于薄透镜，两个方向均为零。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]。
            y (torch.Tensor): 局部 y 坐标 [mm]。

        返回：
            dfdx (torch.Tensor): 偏导数 df/dx [无量纲]，shape 与 `x` 相同。
            dfdy (torch.Tensor): 偏导数 df/dy [无量纲]，shape 与 `x` 相同。
        """
        return torch.zeros_like(x), torch.zeros_like(x)

    # =========================================
    # 可视化
    # =========================================
    def draw_widget(self, ax, color="black", linestyle="-"):
        """在 Matplotlib 坐标轴上绘制薄透镜。

        绘制一条跨越孔径的竖线，并添加双向箭头（会聚透镜使用 "<->"，
        发散透镜使用 "]-["）。

        参数：
            ax (matplotlib.axes.Axes): 截面图的目标坐标轴。
            color (str, optional): 线条和箭头颜色。默认值为 "black"。
            linestyle (str, optional): Matplotlib 线型。默认值为 "-"。
        """
        d = self.d.item()
        r = self.r

        # 绘制竖线表示薄透镜
        ax.plot([d, d], [-r, r], color=color, linestyle=linestyle, linewidth=0.75)

        # 绘制箭头表示焦距
        arrowstyle = "<->" if self.f > 0 else "]-["
        ax.annotate(
            "",
            xy=(d, r),
            xytext=(d, -r),
            arrowprops=dict(
                arrowstyle=arrowstyle, color=color, linestyle=linestyle, linewidth=0.75
            ),
        )

    # =========================================
    # 输入输出
    # =========================================
    def surf_dict(self):
        """将表面序列化为参数字典。

        返回：
            surf_dict (dict): 包含 "type"、"f"、"r"、"(d)" 和 "mat2"
                的表面参数。
        """
        surf_dict = {
            "type": "ThinLens",
            "f": round(self.f.item(), 4),
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
        }

        return surf_dict
