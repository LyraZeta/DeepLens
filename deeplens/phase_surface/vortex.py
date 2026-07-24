"""涡旋相位：将螺旋（OAM）相位与可选的菲涅耳透镜组合。"""

import torch

from ..config import EPSILON
from .phase import Phase


class VortexPhase(Phase):
    """组合螺旋相位与可选菲涅耳透镜的涡旋相位面。"""

    def __init__(
        self,
        r,
        d,
        charge=1,
        f0=None,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化涡旋相位面。"""
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

        self.charge = int(charge)
        self.f0 = torch.tensor(float(f0)) if f0 is not None else None
        self.param_model = "vortex"
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """根据参数字典初始化 `VortexPhase`。"""
        f0_raw = surf_dict.get("f0", None)
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            charge=surf_dict.get("charge", 1),
            f0=f0_raw,
            norm_radii=surf_dict.get("norm_radii", None),
            mat2=surf_dict.get("mat2", "air"),
            pos_xy=surf_dict.get("pos_xy", [0.0, 0.0]),
            vec_local=surf_dict.get("vec_local", [0.0, 0.0, 1.0]),
            is_square=surf_dict.get("is_square", True),
            device=surf_dict.get("device", "cpu"),
        )

    # ------------------------------------------------------------------
    # 相位分布
    # ------------------------------------------------------------------
    def phi(self, x, y):
        """计算折返到 $[0, 2π)$ [rad] 的相位图。"""
        phi = self.charge * torch.atan2(y, x)  # 螺旋项，范围为 (-charge·π, charge·π]
        if self.f0 is not None:
            r2 = x * x + y * y
            phi = phi - torch.pi * r2 / self.f0
        phi = torch.remainder(phi, 2 * torch.pi)
        return phi

    def dphi_dxy(self, x, y):
        """计算广义斯涅尔定律所需的解析未折返相位梯度。"""
        r2 = x * x + y * y + EPSILON
        # d/dx [charge·atan2(y,x)] = -charge·y / r²
        # d/dy [charge·atan2(y,x)] =  charge·x / r²
        dphidx = self.charge * (-y / r2)
        dphidy = self.charge * (x / r2)
        if self.f0 is not None:
            scale = torch.pi / self.f0
            dphidx = dphidx - 2.0 * scale * x
            dphidy = dphidy - 2.0 * scale * y
        return dphidx, dphidy

    # ------------------------------------------------------------------
    # 优化
    # ------------------------------------------------------------------
    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """返回优化器参数组。"""
        assert not optim_mat, "Material parameters are not optimized for phase surfaces."
        params = []
        if self.f0 is not None:
            self.f0.requires_grad_(True)
            params.append({"params": [self.f0], "lr": lrs[0]})
        return params

    # ------------------------------------------------------------------
    # 输入输出
    # ------------------------------------------------------------------
    def save_ckpt(self, save_path="./vortex_doe.pth"):
        """将 `VortexPhase` 参数保存到检查点文件。"""
        ckpt = {
            "param_model": self.param_model,
            "charge": self.charge,
            "f0": self.f0.clone().detach().cpu() if self.f0 is not None else None,
        }
        torch.save(ckpt, save_path)

    def load_ckpt(self, load_path="./vortex_doe.pth"):
        """从检查点文件加载 `VortexPhase` 参数。"""
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.charge = int(ckpt["charge"])
        f0 = ckpt.get("f0")
        self.f0 = f0.to(self.device) if f0 is not None else None

    def surf_dict(self):
        """以可序列化字典形式返回表面参数。"""
        d = {
            "type": self.__class__.__name__,
            "r": self.r,
            "is_square": self.is_square,
            "param_model": self.param_model,
            "charge": self.charge,
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        if self.f0 is not None:
            d["f0"] = round(self.f0.item(), 4)
        return d
