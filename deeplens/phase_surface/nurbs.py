"""平面基底上的 NURBS（非均匀有理 B 样条）相位面。"""

import torch

from ..config import EPSILON
from .phase import Phase


class NURBSPhase(Phase):
    """使用 NURBS 曲面参数化的衍射相位面。"""

    def __init__(
        self,
        r,
        d,
        control_points_u=8,
        control_points_v=8,
        degree_u=3,
        degree_v=3,
        control_points=None,
        weights=None,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化 NURBS 相位面。"""
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

        # NURBS 表面参数
        self.control_points_u = control_points_u
        self.control_points_v = control_points_v
        self.degree_u = degree_u
        self.degree_v = degree_v

        # 生成节点向量（夹持 B 样条）
        self.knots_u = self._generate_clamped_knots(control_points_u, degree_u)
        self.knots_v = self._generate_clamped_knots(control_points_v, degree_v)

        # 初始化控制点 (x, y, z)，其中 z 表示相位。
        # 使用默认 dtype（不硬编码为 float32），使 float64 运算保持
        # 双精度。
        if control_points is None:
            # 使用较小的随机相位值初始化
            cp = torch.randn(control_points_u, control_points_v, 3, device=device) * 1e-3
            # 将 x、y 坐标均匀分布在 [-1, 1] 范围内
            u_coords = torch.linspace(0, 1, control_points_u, device=device)
            v_coords = torch.linspace(0, 1, control_points_v, device=device)
            u_grid, v_grid = torch.meshgrid(u_coords, v_coords, indexing='ij')
            cp[..., 0] = u_grid * 2 - 1  # x 坐标
            cp[..., 1] = v_grid * 2 - 1  # y 坐标
        else:
            cp = torch.as_tensor(control_points, dtype=torch.get_default_dtype(), device=device)
            assert cp.shape == (control_points_u, control_points_v, 3), (
                f"control_points must have shape ({control_points_u}, {control_points_v}, 3)"
            )
        self.control_points = cp

        # 初始化有理 B 样条的权重
        if weights is None:
            w = torch.ones(control_points_u, control_points_v, device=device)
        else:
            w = torch.as_tensor(weights, dtype=torch.get_default_dtype(), device=device)
            assert w.shape == (control_points_u, control_points_v), (
                f"weights must have shape ({control_points_u}, {control_points_v})"
            )
        self.weights = w

        self.param_model = "nurbs"
        self.to(device)

    def _generate_clamped_knots(self, n_control_points, degree):
        """生成 B 样条的夹持节点向量。"""
        n_knots = n_control_points + degree + 1
        knots = torch.zeros(n_knots)

        # 夹持节点：首尾各有 degree+1 个零值
        knots[:degree+1] = 0.0
        knots[-degree-1:] = 1.0

        # 内部节点均匀分布
        if n_control_points > degree + 1:
            n_interior = n_control_points - degree - 1
            for i in range(1, n_interior + 1):
                knots[degree + i] = i / (n_interior + 1)

        return knots

    def _find_knot_span(self, knots, degree, u):
        """查找包含参数 `u` 的节点区间索引（Piegl-Tiller FindSpan）。"""
        n = len(knots) - degree - 2  # 控制点数量减 1

        # 处理边界情况
        if u <= knots[degree]:
            return degree
        if u >= knots[n + 1]:
            return n

        # 二分查找节点区间
        low = degree
        high = n + 1
        mid = (low + high) // 2

        while u < knots[mid] or u >= knots[mid + 1]:
            if u < knots[mid]:
                high = mid
            else:
                low = mid
            mid = (low + high) // 2

        return mid

    def _basis_functions(self, knots, degree, u, span):
        """计算参数 `u` 处非零的 B 样条基函数。"""
        N = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)
        left = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)
        right = torch.zeros(degree + 1, dtype=torch.float32, device=knots.device)

        # 初始化零次基函数
        N[0] = 1.0

        # 使用 Cox-de Boor 递推计算基函数
        for j in range(1, degree + 1):
            left[j] = u - knots[span + 1 - j]
            right[j] = knots[span + j] - u
            saved = 0.0

            for r in range(j):
                denom = right[r + 1] + left[j - r]
                if denom != 0:
                    temp = N[r] / denom
                else:
                    temp = 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp

            N[j] = saved

        return N

    def _evaluate_nurbs_surface(self, u, v):
        """在单个参数对 `(u, v)` 处计算 NURBS 曲面点。"""
        # 将参数限制在有效范围内
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        # 查找节点区间
        span_u = self._find_knot_span(self.knots_u, self.degree_u, u)
        span_v = self._find_knot_span(self.knots_v, self.degree_v, v)

        # 计算基函数
        Nu = self._basis_functions(self.knots_u, self.degree_u, u, span_u)
        Nv = self._basis_functions(self.knots_v, self.degree_v, v, span_v)

        # 计算曲面点
        point = torch.zeros(3, dtype=torch.float32, device=self.device)
        weight_sum = 0.0

        for i in range(self.degree_u + 1):
            for j in range(self.degree_v + 1):
                # 控制点索引
                cp_i = span_u - self.degree_u + i
                cp_j = span_v - self.degree_v + j

                # 索引越界时跳过
                if cp_i < 0 or cp_i >= self.control_points_u or cp_j < 0 or cp_j >= self.control_points_v:
                    continue

                # B 样条基函数值
                basis = Nu[i] * Nv[j]

                # 权重
                weight = self.weights[cp_i, cp_j] * basis

                # 累加加权控制点
                point += weight * self.control_points[cp_i, cp_j]
                weight_sum += weight

        # 对有理 B 样条除以权重和
        if weight_sum > 0:
            point = point / weight_sum

        return point

    # ------------------------------------------------------------------
    # 向量化计算（供 phi/dphi_dxy 使用）。它等价于循环调用上述
    # 单点 _evaluate_nurbs_surface，但省去了 Python 的逐点
    # 循环；对于相位图或光线束，该循环可能达到数百万次。
    # ------------------------------------------------------------------
    def _find_knot_span_batch(self, knots, degree, u):
        """批量查找参数的节点区间（向量化 FindSpan）。"""
        n = len(knots) - degree - 2  # 最后一个控制点的索引
        span = torch.searchsorted(knots, u.contiguous(), right=True) - 1
        return torch.clamp(span, degree, n)

    def _basis_functions_batch(self, knots, degree, u, span):
        """批量计算参数对应的 B 样条基函数。"""
        npts = u.shape[0]
        dtype, device = u.dtype, u.device
        Nb = torch.zeros(npts, degree + 1, dtype=dtype, device=device)
        left = torch.zeros(npts, degree + 1, dtype=dtype, device=device)
        right = torch.zeros(npts, degree + 1, dtype=dtype, device=device)
        Nb[:, 0] = 1.0
        for j in range(1, degree + 1):
            left[:, j] = u - knots[span + 1 - j]
            right[:, j] = knots[span + j] - u
            saved = torch.zeros(npts, dtype=dtype, device=device)
            for r in range(j):
                denom = right[:, r + 1] + left[:, j - r]
                safe = torch.where(denom != 0, denom, torch.ones_like(denom))
                temp = torch.where(
                    denom != 0, Nb[:, r] / safe, torch.zeros_like(denom)
                )
                Nb[:, r] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp
            Nb[:, j] = saved
        return Nb

    def _evaluate_z_batch(self, u, v):
        """批量计算 NURBS 相位（z 分量）。"""
        du, dv = self.degree_u, self.degree_v
        u = torch.clamp(u, 0.0, 1.0)
        v = torch.clamp(v, 0.0, 1.0)

        span_u = self._find_knot_span_batch(self.knots_u, du, u)  # [N]
        span_v = self._find_knot_span_batch(self.knots_v, dv, v)  # [N]
        Nu = self._basis_functions_batch(self.knots_u, du, u, span_u)  # [N, du+1]
        Nv = self._basis_functions_batch(self.knots_v, dv, v, span_v)  # [N, dv+1]

        npts = u.shape[0]
        i_off = torch.arange(du + 1, device=u.device)
        j_off = torch.arange(dv + 1, device=u.device)
        cp_i = span_u.unsqueeze(1) - du + i_off  # [N, du+1]
        cp_j = span_v.unsqueeze(1) - dv + j_off  # [N, dv+1]
        cp_i_e = cp_i.unsqueeze(2).expand(npts, du + 1, dv + 1)
        cp_j_e = cp_j.unsqueeze(1).expand(npts, du + 1, dv + 1)

        basis = Nu.unsqueeze(2) * Nv.unsqueeze(1)  # [N, du+1, dv+1]
        w = self.weights[cp_i_e, cp_j_e]  # [N, du+1, dv+1]
        cz = self.control_points[cp_i_e, cp_j_e, 2]  # 相位 (z) [N, du+1, dv+1]
        weight = w * basis
        numer = (weight * cz).sum(dim=(1, 2))  # [N]
        denom = weight.sum(dim=(1, 2))  # [N]
        safe = torch.where(denom > 0, denom, torch.ones_like(denom))
        return torch.where(denom > 0, numer / safe, numer)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """根据参数字典初始化 NURBS 相位面。"""
        mat2 = surf_dict.get("mat2", "air")
        norm_radii = surf_dict.get("norm_radii", None)
        control_points_u = surf_dict.get("control_points_u", 8)
        control_points_v = surf_dict.get("control_points_v", 8)
        degree_u = surf_dict.get("degree_u", 3)
        degree_v = surf_dict.get("degree_v", 3)

        obj = cls(
            surf_dict["r"],
            surf_dict["d"],
            control_points_u=control_points_u,
            control_points_v=control_points_v,
            degree_u=degree_u,
            degree_v=degree_v,
            norm_radii=norm_radii,
            mat2=mat2,
        )

        # 加载控制点和权重
        control_points = surf_dict.get("control_points", None)
        if control_points is not None:
            obj.control_points = torch.as_tensor(control_points, device=obj.device)

        weights = surf_dict.get("weights", None)
        if weights is not None:
            obj.weights = torch.as_tensor(weights, device=obj.device)

        return obj

    def phi(self, x, y):
        """计算设计波长下的参考相位图。"""
        # 将坐标归一化到 NURBS 参数空间的 [0, 1] 范围
        x_norm = (x / self.norm_radii + 1.0) / 2.0  # 将 [-1, 1] 映射到 [0, 1]
        y_norm = (y / self.norm_radii + 1.0) / 2.0  # 将 [-1, 1] 映射到 [0, 1]

        # 对所有点进行向量化 NURBS 计算（z 分量为相位）。
        phi = self._evaluate_z_batch(x_norm.flatten(), y_norm.flatten()).reshape(
            x_norm.shape
        )

        # 应用圆形孔径掩膜（单位圆外的相位置为 0）
        r_squared = (x / self.norm_radii)**2 + (y / self.norm_radii)**2
        mask = r_squared > 1
        phi = torch.where(mask, torch.zeros_like(phi), phi)

        # 确保相位位于 [0, 2π) 范围内
        phi = torch.remainder(phi, 2 * torch.pi)

        return phi

    def dphi_dxy(self, x, y):
        """通过中心差分计算相位导数 `(dphi/dx, dphi/dy)`。"""
        # 为进行数值微分，在略微偏移的位置计算 phi
        eps = 1e-6

        # 计算 dphi/dx
        phi_x_plus = self.phi(x + eps, y)
        phi_x_minus = self.phi(x - eps, y)
        dphidx = (phi_x_plus - phi_x_minus) / (2 * eps)

        # 计算 dphi/dy
        phi_y_plus = self.phi(x, y + eps)
        phi_y_minus = self.phi(x, y - eps)
        dphidy = (phi_y_plus - phi_y_minus) / (2 * eps)

        # 应用圆形掩膜
        r_squared = (x / self.norm_radii)**2 + (y / self.norm_radii)**2
        mask = r_squared > 1
        dphidx = torch.where(mask, torch.zeros_like(dphidx), dphidx)
        dphidy = torch.where(mask, torch.zeros_like(dphidy), dphidy)

        return dphidx, dphidy

    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """为 NURBS 控制点构建优化器参数组。"""
        params = []

        # 为控制点启用梯度（相位仅使用 z 坐标）
        self.control_points.requires_grad = True
        params.append({"params": [self.control_points], "lr": lrs[0]})

        # 可选地优化权重
        if len(lrs) > 1:
            self.weights.requires_grad = True
            params.append({"params": [self.weights], "lr": lrs[1]})

        # 相位面不优化材料参数。
        assert optim_mat is False, (
            "Material parameters are not optimized for phase surface."
        )

        return params

    def save_ckpt(self, save_path="./nurbs_doe.pth"):
        """将 NURBS DOE 参数保存到检查点文件。"""
        torch.save(
            {
                "param_model": "nurbs",
                "control_points": self.control_points.clone().detach().cpu(),
                "weights": self.weights.clone().detach().cpu(),
                "control_points_u": self.control_points_u,
                "control_points_v": self.control_points_v,
                "degree_u": self.degree_u,
                "degree_v": self.degree_v,
                "knots_u": self.knots_u.clone().detach().cpu(),
                "knots_v": self.knots_v.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./nurbs_doe.pth"):
        """从检查点文件加载 NURBS DOE 参数。"""
        ckpt = torch.load(load_path)
        self.param_model = ckpt["param_model"]
        self.control_points_u = ckpt["control_points_u"]
        self.control_points_v = ckpt["control_points_v"]
        self.control_points = ckpt["control_points"].to(self.device)
        self.weights = ckpt["weights"].to(self.device)
        self.degree_u = ckpt["degree_u"]
        self.degree_v = ckpt["degree_v"]
        self.knots_u = ckpt["knots_u"].to(self.device)
        self.knots_v = ckpt["knots_v"].to(self.device)

    def surf_dict(self):
        """以可序列化字典形式返回表面参数。"""
        surf_dict = {
            "type": "Phase",
            "r": self.r,
            "is_square": self.is_square,
            "param_model": "nurbs",
            "control_points": self.control_points.clone().detach().cpu().tolist(),
            "weights": self.weights.clone().detach().cpu().tolist(),
            "control_points_u": self.control_points_u,
            "control_points_v": self.control_points_v,
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "knots_u": self.knots_u.clone().detach().cpu().tolist(),
            "knots_v": self.knots_v.clone().detach().cpu().tolist(),
            "norm_radii": round(self.norm_radii, 4),
            "d": round(self.d.item(), 4),
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }
        return surf_dict
