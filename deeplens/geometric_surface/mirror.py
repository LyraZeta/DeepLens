"""反射镜表面。"""

from .base import Surface
from .plane import Plane


class Mirror(Plane):
    """平面反射镜表面。

    平坦表面通过镜面反射而非折射来改变入射光线方向，因此介质不变，`mat2`
    默认值为 `"air"`。继承 `Plane` 的平面几何结构，但默认使用方形孔径。

    属性：
        r (float): 孔径半径 [mm]。对于方形孔径，该值为外接圆半径（半对角线）。
        d (torch.Tensor): 反射镜顶点的轴向位置 [mm]。
        mat2 (Material): 反射镜远侧的材料。
        is_square (bool): 孔径是否为方形。
    """

    def __init__(
        self,
        r,
        d,
        mat2="air",
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=True,
        device="cpu",
    ):
        """初始化平面反射镜表面。

        参数：
            r (float): 孔径半径 [mm]。对于方形孔径，该值为外接圆半径
                （半对角线），因此边长为 r * sqrt(2)。
            d (float): 反射镜顶点的轴向位置 [mm]。
            mat2 (str or Material, optional): 反射镜远侧的材料。默认值为 "air"。
            pos_xy (list[float], optional): 横向偏移 [x, y] [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list[float], optional): 局部表面法线方向。
                默认值为 [0.0, 0.0, 1.0]（轴上）。
            is_square (bool, optional): 使用方形孔径。默认值为 True。
            device (str, optional): 计算设备。默认值为 "cpu"。
        """
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            is_square=is_square,
            pos_xy=pos_xy,
            vec_local=vec_local,
            device=device,
        )

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从表面参数字典构造 `Mirror`。

        参数：
            surf_dict (dict): 表面参数；读取键 "r"、"d" 和 "mat2"。

        返回：
            mirror (Mirror): 构造得到的反射镜表面。
        """
        return cls(surf_dict["r"], surf_dict["d"], surf_dict["mat2"])

    def ray_reaction(self, ray, n1=None, n2=None):
        """计算求交和反射后的输出光线。

        将光线变换到反射镜局部坐标系，求解光线与平面的交点，施加镜面反射，
        再变换回全局坐标系。

        参数：
            ray (Ray): 入射光线束。
            n1 (float, optional): 入射介质折射率，仅为兼容基础表面 API 而接收，
                未使用。默认值为 None。
            n2 (float, optional): 透射介质折射率，仅为兼容 API 而接收，
                未使用。默认值为 None。

        返回：
            ray (Ray): 反射后更新的光线束。
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.reflect(ray)
        ray = self.to_global_coord(ray)
        return ray

    # =========================================
    # 输入输出
    # =========================================
    def surf_dict(self):
        """以可序列化字典形式返回反射镜参数。

        返回：
            surf_dict (dict): 包含 "type"、"r"、保留 4 位小数的 "d"、
                材料名称 "mat2"，以及信息项 "(mat2_n)"/"(mat2_V)" 的参数。
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
