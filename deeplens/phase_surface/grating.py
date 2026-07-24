"""平面基底上的光栅相位面。"""

import torch

from .phase import Phase


class GratingPhase(Phase):
    """平面基底上的线性（闪耀）光栅相位分布。"""

    def __init__(
        self,
        r,
        d,
        theta=0.0,
        alpha=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化线性光栅相位面。"""
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

        # 光栅参数
        self.theta = torch.tensor(theta)  # 从 x 轴到光栅矢量的夹角
        self.alpha = torch.tensor(alpha)  # 光栅斜率

        self.param_model = "grating"
        self.to(device)

    @classmethod
    def init_from_dict(cls, param_dict):
        """根据参数字典初始化 `GratingPhase`。"""
        # 提取参数，默认值与 __init__ 签名一致
        r = param_dict.get("r")
        d = param_dict.get("d")
        theta = param_dict.get("theta", 0.0)
        alpha = param_dict.get("alpha", 0.0)
        norm_radii = param_dict.get("norm_radii", None)
        mat2 = param_dict.get("mat2", "air")
        pos_xy = param_dict.get("pos_xy", [0.0, 0.0])
        vec_local = param_dict.get("vec_local", [0.0, 0.0, 1.0])
        is_square = param_dict.get("is_square", True)
        device = param_dict.get("device", "cpu")
        return cls(
            r=r,
            d=d,
            theta=theta,
            alpha=alpha,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

    def phi(self, x, y):
        """计算给定点处折返到 $[0, 2π)$ 的光栅相位。"""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = self.alpha * (
            x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """计算给定点处的相位梯度 `(dphi/dx, dphi/dy)`。"""
        # 将标量导数广播到输入张量形状而不额外分配内存
        dphidx = (self.alpha * torch.sin(self.theta) / self.norm_radii).expand_as(x)
        dphidy = (self.alpha * torch.cos(self.theta) / self.norm_radii).expand_as(y)
        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-3], optim_mat=False):
        """为光栅相位参数构建 Adam 优化器参数组。"""
        params = []

        # 优化光栅参数
        self.theta.requires_grad = True
        self.alpha.requires_grad = True
        params.append({"params": [self.theta], "lr": lrs[0]})
        params.append({"params": [self.alpha], "lr": lrs[1]})

        # 相位面不优化材料参数。
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./grating_doe.pth"):
        """将光栅参数保存到检查点文件。"""
        torch.save(
            {
                "param_model": self.param_model,
                "theta": self.theta.clone().detach().cpu(),
                "alpha": self.alpha.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./grating_doe.pth"):
        """从检查点文件加载光栅参数。"""
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.theta = ckpt["theta"].to(self.device)
        self.alpha = ckpt["alpha"].to(self.device)

    def surf_dict(self):
        """返回可序列化的光栅表面参数字典。"""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "theta": round(self.theta.item(), 4),
            "alpha": round(self.alpha.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
