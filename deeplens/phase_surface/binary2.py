"""平面基底上的 Binary2 相位面。"""

import torch

from ..config import EPSILON
from .phase import Phase


class Binary2Phase(Phase):
    """平面基底上的 Zemax BINARY_2 相位分布。

    使用归一化半径 $\\rho = r / r_\\text{norm}$ 的偶次径向多项式
    参数化衍射相位：

    $$\\phi(\\rho) = \\sum_{i=1}^{6} a_{2i}\\,\\rho^{2i}$$

    系数 `order2` 至 `order12` 以弧度 [rad] 存储。相位通过霍纳法计算，
    并折返到 $[0, 2\\pi)$。

    属性：
        order2 (torch.Tensor): $\\rho^2$ 的标量系数 [rad]。
        order4 (torch.Tensor): $\\rho^4$ 的标量系数 [rad]。
        order6 (torch.Tensor): $\\rho^6$ 的标量系数 [rad]。
        order8 (torch.Tensor): $\\rho^8$ 的标量系数 [rad]。
        order10 (torch.Tensor): $\\rho^{10}$ 的标量系数 [rad]。
        order12 (torch.Tensor): $\\rho^{12}$ 的标量系数 [rad]。
        param_model (str): 参数化标签，始终为 "binary2"。
        norm_radii (float): 归一化半径 $r_\\text{norm}$ [mm]。
    """

    def __init__(
        self,
        r,
        d,
        order2=0.0,
        order4=0.0,
        order6=0.0,
        order8=0.0,
        order10=0.0,
        order12=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化 Binary2 相位面。

        参数：
            r (float): 孔径半径（半直径）[mm]。
            d (float): 表面在全局坐标中的轴向位置 [mm]。
            order2 (float, optional): $\\rho^2$ 的系数 [rad]，默认为 0.0。
            order4 (float, optional): $\\rho^4$ 的系数 [rad]，默认为 0.0。
            order6 (float, optional): $\\rho^6$ 的系数 [rad]，默认为 0.0。
            order8 (float, optional): $\\rho^8$ 的系数 [rad]，默认为 0.0。
            order10 (float, optional): $\\rho^{10}$ 的系数 [rad]，默认为 0.0。
            order12 (float, optional): $\\rho^{12}$ 的系数 [rad]，默认为 0.0。
            norm_radii (float or None, optional): 多项式归一化半径 [mm]；为 None 时使用 `r`。
            mat2 (str, optional): 表面之后的材料，默认为 "air"。
            pos_xy (tuple, optional): 表面中心的横向 (x, y) 偏移 [mm]，默认为 (0.0, 0.0)。
            vec_local (tuple, optional): 局部表面法线方向，默认为 (0.0, 0.0, 1.0)。
            is_square (bool, optional): 为 True 时使用方形孔径，否则使用圆形孔径；默认为 True。
            device (str, optional): Torch 设备，默认为 "cpu"。
        """
        super().__init__(
            r=r,
            d=d,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        # 初始化多项式系数
        self.order2 = torch.tensor(order2)
        self.order4 = torch.tensor(order4)
        self.order6 = torch.tensor(order6)
        self.order8 = torch.tensor(order8)
        self.order10 = torch.tensor(order10)
        self.order12 = torch.tensor(order12)

        self.param_model = "binary2"
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """根据参数字典构造 Binary2 相位面。

        参数：
            surf_dict (dict): 表面参数。必须包含 "r" 和 "d"，还可包含
                "order2" 至 "order12"、"norm_radii"、"mat2" 和 "is_square"。

        返回：
            obj (Binary2Phase): 构造得到的相位面。
        """
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        is_square = surf_dict.get("is_square", True)
        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            surf_dict.get("order2", 0.0),
            surf_dict.get("order4", 0.0),
            surf_dict.get("order6", 0.0),
            surf_dict.get("order8", 0.0),
            surf_dict.get("order10", 0.0),
            surf_dict.get("order12", 0.0),
            norm_radii,
            mat2,
            is_square=is_square,
        )
        return obj

    def phi(self, x, y):
        """计算设计波长下的参考相位。

        使用霍纳法计算归一化半径的偶次径向多项式，并通过
        `torch.remainder` 将结果折返到 $[0, 2\\pi)$。

        参数：
            x (torch.Tensor): X 坐标 [mm]，形状不限。
            y (torch.Tensor): Y 坐标 [mm]，形状与 `x` 相同。

        返回：
            phi (torch.Tensor): 折返到 $[0, 2\\pi)$ 的相位值 [rad]，形状与 `x` 相同。
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON

        # 霍纳法：r2*(o2 + r2*(o4 + r2*(o6 + r2*(o8 + r2*(o10 + r2*o12)))))
        phi = r2 * (
            self.order2
            + r2 * (self.order4 + r2 * (self.order6 + r2 * (self.order8 + r2 * (self.order10 + r2 * self.order12))))
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """计算给定点处的横向相位梯度。

        对未折返的相位多项式求导，并通过归一化半径应用链式法则。

        参数：
            x (torch.Tensor): X 坐标 [mm]，形状不限。
            y (torch.Tensor): Y 坐标 [mm]，形状与 `x` 相同。

        返回：
            dphidx (torch.Tensor): 偏导数 $\\partial\\phi/\\partial x$ [rad/mm]，形状与 `x` 相同。
            dphidy (torch.Tensor): 偏导数 $\\partial\\phi/\\partial y$ [rad/mm]，形状与 `x` 相同。
        """
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii
        r2 = x_norm * x_norm + y_norm * y_norm + EPSILON

        # 先对多项式求 d/dr2，再用链式法则：dphi/dx = dphi/dr2 * 2*x_norm / norm_radii
        # 霍纳形式：o2 + r2*(2*o4 + r2*(3*o6 + r2*(4*o8 + r2*(5*o10 + r2*6*o12))))
        dphidr2 = (
            self.order2
            + r2 * (2 * self.order4 + r2 * (3 * self.order6 + r2 * (4 * self.order8 + r2 * (5 * self.order10 + r2 * 6 * self.order12))))
        )
        dphidx = dphidr2 * 2 * x_norm / self.norm_radii
        dphidy = dphidr2 * 2 * y_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """构建相位面的优化器参数组。

        为轴向位置 `d` 和六个多项式系数启用梯度；`d` 使用第一个学习率，
        所有系数使用第二个学习率。

        参数：
            lrs (list, optional): 学习率 ``[lr_position, lr_coeffs]``，默认为 [1e-4, 1e-2]。
            optim_mat (bool, optional): 必须为 False；相位面不优化材料，默认为 False。

        返回：
            params (list): 供 torch 优化器使用的参数组字典列表。

        异常：
            AssertionError: `optim_mat` 为 True 时抛出。
        """
        params = []

        # 优化位置
        self.d.requires_grad = True
        params.append({"params": [self.d], "lr": lrs[0]})

        # 优化多项式系数
        self.order2.requires_grad = True
        self.order4.requires_grad = True
        self.order6.requires_grad = True
        self.order8.requires_grad = True
        self.order10.requires_grad = True
        self.order12.requires_grad = True
        params.append({"params": [self.order2], "lr": lrs[1]})
        params.append({"params": [self.order4], "lr": lrs[1]})
        params.append({"params": [self.order6], "lr": lrs[1]})
        params.append({"params": [self.order8], "lr": lrs[1]})
        params.append({"params": [self.order10], "lr": lrs[1]})
        params.append({"params": [self.order12], "lr": lrs[1]})

        # 相位面不优化材料参数。
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./binary2_doe.pth"):
        """将 Binary2 相位系数保存到磁盘。

        参数：
            save_path (str, optional): 输出检查点路径，默认为 "./binary2_doe.pth"。
        """
        torch.save(
            {
                "param_model": self.param_model,
                "order2": self.order2.clone().detach().cpu(),
                "order4": self.order4.clone().detach().cpu(),
                "order6": self.order6.clone().detach().cpu(),
                "order8": self.order8.clone().detach().cpu(),
                "order10": self.order10.clone().detach().cpu(),
                "order12": self.order12.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./binary2_doe.pth"):
        """从磁盘加载 Binary2 相位系数，并放到表面所在设备。

        参数：
            load_path (str, optional): 要加载的检查点路径，默认为 "./binary2_doe.pth"。
        """
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.order2 = ckpt["order2"].to(self.device)
        self.order4 = ckpt["order4"].to(self.device)
        self.order6 = ckpt["order6"].to(self.device)
        self.order8 = ckpt["order8"].to(self.device)
        self.order10 = ckpt["order10"].to(self.device)
        self.order12 = ckpt["order12"].to(self.device)

    def zmx_str(self, surf_idx, d_next):
        """以字符串形式返回 Zemax BINARY_2 表面数据块。

        将 PARM 1-8 设为零（平面基底且无非球面矢高），使 Zemax 将 XDAT
        条目仅解释为相位多项式系数。

        参数：
            surf_idx (int): SURF 头中使用的表面索引。
            d_next (torch.Tensor): 到下一表面的距离 [mm]，通过 `.item()` 读取的标量张量。

        返回：
            zmx_str (str): 多行 Zemax 表面描述。
        """
        coeffs = [
            self.order2.item(),
            self.order4.item(),
            self.order6.item(),
            self.order8.item(),
            self.order10.item(),
            self.order12.item(),
        ]
        n_terms = len(coeffs)

        # 构建 XDAT 数据块：项数、归一化半径，然后是各系数
        xdat_str = f"    XDAT 1 {n_terms} 0 0\n"
        xdat_str += f"    XDAT 2 {self.norm_radii} 0 0\n"
        for j, coeff in enumerate(coeffs, start=3):
            xdat_str += f"    XDAT {j} {coeff} 0 0\n"

        zmx_str = f"""SURF {surf_idx}
    TYPE BINARY_2
    CURV 0.0
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    PARM 1 0
    PARM 2 0
    PARM 3 0
    PARM 4 0
    PARM 5 0
    PARM 6 0
    PARM 7 0
    PARM 8 0
{xdat_str}"""
        return zmx_str

    def surf_dict(self):
        """返回可序列化的表面参数字典。

        返回：
            surf_dict (dict): 表面参数，包括类型、半径 `r` [mm]、舍入后的多项式系数、
                `norm_radii` [mm]、位置 `d` [mm] 和材料名称。
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "order2": round(self.order2.item(), 4),
            "order4": round(self.order4.item(), 4),
            "order6": round(self.order6.item(), 4),
            "order8": round(self.order8.item(), 4),
            "order10": round(self.order10.item(), 4),
            "order12": round(self.order12.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
