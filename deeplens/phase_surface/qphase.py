"""平面基底上的四次（Q 型）相位面。"""

import torch

from .phase import Phase


class QuarticPhase(Phase):
    """平面基底上的四次多项式相位分布。"""

    def __init__(
        self,
        r,
        d,
        coeff_x4=0.0,
        coeff_y4=0.0,
        coeff_x3y=0.0,
        coeff_xy3=0.0,
        coeff_x2y2=0.0,
        coeff_x4y=0.0,
        coeff_xy4=0.0,
        coeff_x3y2=0.0,
        coeff_x2y3=0.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化四次多项式相位面。"""
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

        self.coeff_x4 = torch.tensor(coeff_x4)
        self.coeff_y4 = torch.tensor(coeff_y4)
        self.coeff_x3y = torch.tensor(coeff_x3y)
        self.coeff_xy3 = torch.tensor(coeff_xy3)
        self.coeff_x2y2 = torch.tensor(coeff_x2y2)
        self.coeff_x4y = torch.tensor(coeff_x4y)
        self.coeff_xy4 = torch.tensor(coeff_xy4)
        self.coeff_x3y2 = torch.tensor(coeff_x3y2)
        self.coeff_x2y3 = torch.tensor(coeff_x2y3)

        self.param_model = "quartic"
        self.to(device)

    def phi(self, x, y):
        """计算设计波长下折返后的参考相位图。"""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        phi = (
            self.coeff_x4 * x_norm**4
            + self.coeff_y4 * y_norm**4
            + self.coeff_x3y * x_norm**3 * y_norm
            + self.coeff_xy3 * x_norm * y_norm**3
            + self.coeff_x2y2 * x_norm**2 * y_norm**2
            + self.coeff_x4y * x_norm**4 * y_norm
            + self.coeff_xy4 * x_norm * y_norm**4
            + self.coeff_x3y2 * x_norm**3 * y_norm**2
            + self.coeff_x2y3 * x_norm**2 * y_norm**3
        )

        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """计算设计波长下的空间相位梯度。"""
        x_norm = x / self.norm_radii
        y_norm = y / self.norm_radii

        # 对归一化坐标求导
        dphi_dx_norm = (
            4 * self.coeff_x4 * x_norm**3
            + 3 * self.coeff_x3y * x_norm**2 * y_norm
            + self.coeff_xy3 * y_norm**3
            + 2 * self.coeff_x2y2 * x_norm * y_norm**2
            + 4 * self.coeff_x4y * x_norm**3 * y_norm
            + self.coeff_xy4 * y_norm**4
            + 3 * self.coeff_x3y2 * x_norm**2 * y_norm**2
            + 2 * self.coeff_x2y3 * x_norm * y_norm**3
        )

        dphi_dy_norm = (
            4 * self.coeff_y4 * y_norm**3
            + self.coeff_x3y * x_norm**3
            + 3 * self.coeff_xy3 * x_norm * y_norm**2
            + 2 * self.coeff_x2y2 * x_norm**2 * y_norm
            + self.coeff_x4y * x_norm**4
            + 4 * self.coeff_xy4 * x_norm * y_norm**3
            + 2 * self.coeff_x3y2 * x_norm**3 * y_norm
            + 3 * self.coeff_x2y3 * x_norm**2 * y_norm**2
        )

        # 转换回物理坐标
        dphidx = dphi_dx_norm / self.norm_radii
        dphidy = dphi_dy_norm / self.norm_radii

        return dphidx, dphidy

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5], optim_mat=False
    ):
        """为每个四次多项式系数构建优化器参数组。"""
        params = []

        # 使用不同学习率优化四次多项式系数
        self.coeff_x4.requires_grad = True
        self.coeff_y4.requires_grad = True
        self.coeff_x3y.requires_grad = True
        self.coeff_xy3.requires_grad = True
        self.coeff_x2y2.requires_grad = True
        self.coeff_x4y.requires_grad = True
        self.coeff_xy4.requires_grad = True
        self.coeff_x3y2.requires_grad = True
        self.coeff_x2y3.requires_grad = True

        params.append({"params": [self.coeff_x4], "lr": lrs[0]})
        params.append({"params": [self.coeff_y4], "lr": lrs[1]})
        params.append({"params": [self.coeff_x3y], "lr": lrs[2]})
        params.append({"params": [self.coeff_xy3], "lr": lrs[3]})
        params.append({"params": [self.coeff_x2y2], "lr": lrs[4]})
        params.append({"params": [self.coeff_x4y], "lr": lrs[5]})
        params.append({"params": [self.coeff_xy4], "lr": lrs[6]})
        params.append({"params": [self.coeff_x3y2], "lr": lrs[7]})
        params.append({"params": [self.coeff_x2y3], "lr": lrs[8]})

        # 相位面不优化材料参数。
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./quartic_doe.pth"):
        """将四次 DOE 系数保存到检查点文件。"""
        torch.save(
            {
                "param_model": self.param_model,
                "coeff_x4": self.coeff_x4.clone().detach().cpu(),
                "coeff_y4": self.coeff_y4.clone().detach().cpu(),
                "coeff_x3y": self.coeff_x3y.clone().detach().cpu(),
                "coeff_xy3": self.coeff_xy3.clone().detach().cpu(),
                "coeff_x2y2": self.coeff_x2y2.clone().detach().cpu(),
                "coeff_x4y": self.coeff_x4y.clone().detach().cpu(),
                "coeff_xy4": self.coeff_xy4.clone().detach().cpu(),
                "coeff_x3y2": self.coeff_x3y2.clone().detach().cpu(),
                "coeff_x2y3": self.coeff_x2y3.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./quartic_doe.pth"):
        """从检查点文件加载四次 DOE 系数。"""
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.coeff_x4 = ckpt["coeff_x4"].to(self.device)
        self.coeff_y4 = ckpt["coeff_y4"].to(self.device)
        self.coeff_x3y = ckpt["coeff_x3y"].to(self.device)
        self.coeff_xy3 = ckpt["coeff_xy3"].to(self.device)
        self.coeff_x2y2 = ckpt["coeff_x2y2"].to(self.device)
        self.coeff_x4y = ckpt["coeff_x4y"].to(self.device)
        self.coeff_xy4 = ckpt["coeff_xy4"].to(self.device)
        self.coeff_x3y2 = ckpt["coeff_x3y2"].to(self.device)
        self.coeff_x2y3 = ckpt["coeff_x2y3"].to(self.device)

    def surf_dict(self):
        """将表面参数序列化为字典。"""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "coeff_x4": round(self.coeff_x4.item(), 4),
            "coeff_y4": round(self.coeff_y4.item(), 4),
            "coeff_x3y": round(self.coeff_x3y.item(), 4),
            "coeff_xy3": round(self.coeff_xy3.item(), 4),
            "coeff_x2y2": round(self.coeff_x2y2.item(), 4),
            "coeff_x4y": round(self.coeff_x4y.item(), 4),
            "coeff_xy4": round(self.coeff_xy4.item(), 4),
            "coeff_x3y2": round(self.coeff_x3y2.item(), 4),
            "coeff_x2y3": round(self.coeff_x2y3.item(), 4),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
