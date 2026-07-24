"""平面基底上的菲涅耳相位面。"""

import torch

from .phase import Phase


class FresnelPhase(Phase):
    """平面基底上的理想菲涅耳透镜相位分布。"""

    def __init__(
        self,
        r,
        d,
        f0=100.0,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化菲涅耳透镜相位面。"""
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

        # 550 nm 波长下的焦距
        self.f0 = torch.tensor(f0)
        self.param_model = "fresnel"
        self.to(device)

    @classmethod
    def init_from_dict(cls, param_dict):
        """根据参数字典初始化 `FresnelPhase`。"""
        r = param_dict.get("r")
        d = param_dict.get("d")
        f0 = param_dict.get("f0", 100.0)
        norm_radii = param_dict.get("norm_radii", None)
        mat2 = param_dict.get("mat2", "air")
        pos_xy = param_dict.get("pos_xy", [0.0, 0.0])
        vec_local = param_dict.get("vec_local", [0.0, 0.0, 1.0])
        is_square = param_dict.get("is_square", True)
        device = param_dict.get("device", "cpu")
        return cls(
            r=r,
            d=d,
            f0=f0,
            norm_radii=norm_radii,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

    def phi(self, x, y):
        """计算设计波长下折返到 $[0, 2π)$ 的菲涅耳透镜相位。"""
        phi = (
            -2 * torch.pi * torch.fmod((x**2 + y**2) / (2 * 0.55e-3 * self.f0), 1)
        )  # 单位 [mm]
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """计算未折返相位的梯度 `(dphi/dx, dphi/dy)`。"""
        dphidx = -2 * torch.pi * x / (0.55e-3 * self.f0)  # 单位 [mm]
        dphidy = -2 * torch.pi * y / (0.55e-3 * self.f0)
        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """为焦距构建优化器参数组。"""
        params = []

        # 优化焦距
        self.f0.requires_grad = True
        params.append({"params": [self.f0], "lr": lrs[0]})

        # 相位面不优化材料参数。
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./fresnel_doe.pth"):
        """将菲涅耳 DOE 参数保存到检查点文件。"""
        torch.save(
            {
                "param_model": self.param_model,
                "f0": self.f0.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./fresnel_doe.pth"):
        """从检查点文件加载菲涅耳 DOE 参数。"""
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.f0 = ckpt["f0"].to(self.device)

    def surf_dict(self):
        """返回可序列化的表面参数字典。"""
        surf_dict = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "f0": self.f0.item(),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
