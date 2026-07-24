"""几何表面的基类。

表面可以折射和反射光线。某些表面还可依据局部光栅近似使光线发生衍射。
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from ..base import DeepObj
from ..config import EPSILON
from ..material import Material


class Surface(DeepObj):
    """所有几何光学表面的基类。

    表面位于全局坐标系的轴向位置 `d` [mm]，孔径半径为 `r` [mm]，并分隔
    两种光学介质。子类通过重写 `_sag` 和 `_dfdxy` 定义表面形状。

    `ray_reaction` 分三个阶段处理光线与表面的相互作用：

    1. 坐标变换：将光线变换到表面局部坐标系。
    2. 求交：使用 Newton 法（`newtons_method`）求解，先进行不可微迭代循环，
       再执行一次可微 Newton 步骤以使梯度能够传播。
    3. 折射／反射：使用矢量 Snell 定律（`refract`）或镜面反射（`reflect`）。

    属性：
        d (torch.Tensor): 表面顶点的轴向位置 [mm]，标量张量。
        r (float): 孔径半径 [mm]。对于方形孔径，该值为外接圆半径（半对角线）。
        mat2 (Material): 透射侧的光学材料。
        pos_x (torch.Tensor): 顶点的横向 x 偏移 [mm]，标量张量。
        pos_y (torch.Tensor): 顶点的横向 y 偏移 [mm]，标量张量。
        vec_local (torch.Tensor): 局部表面法线方向，shape 为 [3]。
        is_square (bool): 为 True 时孔径为方形，否则为圆形。
        w (float): 方形孔径边长 [mm]（仅在 `is_square` 时设置）。
        h (float): 方形孔径边长 [mm]（仅在 `is_square` 时设置）。
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
        """初始化通用光学表面。

        参数：
            r (float): 孔径半径 [mm]。对于方形孔径，该值为外接圆半径
                （半对角线），因此边长为 `r * sqrt(2)`。
            d (float): 表面顶点的轴向位置 [mm]。
            mat2 (str or Material): 透射侧材料（例如 "N-BK7"、"air"）。
            pos_xy (list[float], optional): 横向偏移 [x, y] [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list[float], optional): 局部表面法线方向；内部会归一化。
                默认值为 [0.0, 0.0, 1.0]（轴上）。
            is_square (bool, optional): 使用方形孔径而非圆形孔径。默认值为 False。
            device (str, optional): 计算设备。默认值为 "cpu"。
        """
        super(Surface, self).__init__()

        # 全局方向向量，始终指向 z 轴正方向
        self.vec_global = torch.tensor([0.0, 0.0, 1.0])

        # 表面在全局坐标系中的位置
        self.d = torch.tensor(d)
        self.pos_x = torch.tensor(pos_xy[0])
        self.pos_y = torch.tensor(pos_xy[1])

        # 表面在全局坐标系中的方向向量
        self.vec_local = F.normalize(torch.tensor(vec_local), p=2, dim=-1)

        # 表面后的材料
        self.mat2 = Material(mat2)

        # 表面孔径半径（不可微）。
        # 对于方形孔径，r 为外接圆半径（即半对角线），因此边长为 r * sqrt(2)。
        self.r = float(r)
        self.is_square = is_square
        if is_square:
            self.w = self.r * float(np.sqrt(2))
            self.h = self.r * float(np.sqrt(2))

        # Newton 法参数
        self.newton_maxiter = 8  # [int]，Newton 法最大迭代次数
        self.newton_convergence = 50.0 * 1e-6  # [mm]，Newton 法收敛阈值
        self.newton_step_bound = 5.0  # [mm]，每次迭代的最大步长

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

        # 预计算旋转矩阵（仅依赖静态的 vec_local/vec_global）
        self._cache_rotation_matrices()

    def _cache_rotation_matrices(self):
        """预计算并缓存局部／全局坐标变换的旋转矩阵。

        初始化时调用一次。矩阵仅依赖 `vec_local` 和 `vec_global`，二者在构造后
        保持不变。当表面位于轴上时，两个缓存矩阵均设为 None（无需旋转）。
        """
        needs_rotation = (
            torch.abs(torch.dot(self.vec_local, self.vec_global) - 1.0) > EPSILON
        )
        if needs_rotation:
            self._R_to_local = self._get_rotation_matrix(
                self.vec_local, self.vec_global
            )
            self._R_to_global = self._get_rotation_matrix(
                self.vec_global, self.vec_local
            )
        else:
            self._R_to_local = None
            self._R_to_global = None

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从序列化字典初始化表面。

        参数：
            surf_dict (dict): 表面参数，通常由 `surf_dict` 生成。

        返回：
            surface (Surface): 重建得到的表面实例。

        异常：
            NotImplementedError: 基类始终抛出此异常；由子类重写。
        """
        raise NotImplementedError(
            f"init_from_dict() is not implemented for {cls.__name__}."
        )

    # =====================================================================
    # 光线与表面之间的求交、折射和反射
    # =====================================================================
    def ray_reaction(self, ray, n1, n2, refraction=True):
        """计算求交及折射／反射后的输出光线。

        将光线变换到表面局部坐标系，使用 Newton 法求解交点，应用矢量 Snell
        定律（或镜面反射），再变换回全局坐标系。

        参数：
            ray (Ray): 入射光线束。
            n1 (float): 入射介质的折射率。
            n2 (float): 透射介质的折射率。
            refraction (bool, optional): 为 True 时折射光线；为 False 时反射光线。
                默认值为 True。

        返回：
            ray (Ray): 与表面相互作用后更新的光线束。
        """
        # 将光线变换到局部坐标系
        ray = self.to_local_coord(ray)

        # 求交
        ray = self.intersect(ray, n1)

        if refraction:
            old_d = ray.d.clone()
            ray = self.refract(ray, n1 / n2)
            ray = self.bend_penalty(ray, old_d)
        else:
            # 反射
            ray = self.reflect(ray)

        # 将光线变换到全局坐标系
        ray = self.to_global_coord(ray)

        return ray

    def intersect(self, ray, n=1.0):
        """在局部坐标系中求解光线与表面的交点。

        将每条有效光线的起点移动到表面；对于相干光线，将光程 `n * t`
        累加到 `ray.opl`。

        参数：
            ray (Ray): 局部坐标系中的输入光线束。
            n (float, optional): 光线传播至表面所经过介质的折射率。默认值为 1.0。

        返回：
            ray (Ray): 起点、有效性掩膜以及相干光线的光程均已更新的光线。

        异常：
            Exception: 当相干光线使用 float32 且求交距离超过 100 mm，
                可能导致 OPL 精度问题时抛出。
        """
        # 使用 Newton 法求解光线与表面的求交时间
        t, valid = self.newtons_method(ray)

        # 更新光线
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        ray.o = torch.where(valid.unsqueeze(-1), new_o, ray.o)
        ray.is_valid = ray.is_valid * valid

        if ray.is_coherent:
            # 检查实际张量 dtype（与 ray.py 一致），而不是全局默认值；后者可能
            # 无法反映当前光线的精度。
            if t.abs().max() > 100 and t.dtype != torch.float64:
                raise Exception(
                    "Using float32 may cause precision problem for OPL calculation."
                )
            new_opl = ray.opl + n * t.unsqueeze(-1)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        return ray

    def newtons_method(self, ray):
        """在局部坐标系中使用 Newton 法求解光线与表面的交点。

        先执行 `newton_maxiter - 1` 次不可微迭代，再执行一次可微 Newton 步骤，
        因此梯度只通过最后一步传播。求得的 $t$ 使光线上满足
        `sag(x, y) - z = 0`。

        参数：
            ray (Ray): 局部坐标系中的输入光线束。

        返回：
            t (torch.Tensor): 求交参数（沿光线的距离）[mm]，shape [...] 与光线
                批次匹配。
            valid (torch.Tensor): 已收敛且在范围内的交点布尔掩膜，shape 为 [...]。
        """
        newton_maxiter = self.newton_maxiter
        newton_convergence = self.newton_convergence
        newton_step_bound = self.newton_step_bound

        # 光线方向分量（在各次迭代中复用）
        dxdt, dydt, dzdt = ray.d[..., 0], ray.d[..., 1], ray.d[..., 2]

        # t 的初始猜测（也可使用球面生成初始猜测）
        t = -ray.o[..., 2] / dzdt

        # 1. 使用不可微 Newton 迭代寻找交点
        #    执行 (maxiter - 1) 次迭代；下方可微步骤作为最后一次迭代，
        #    同时允许梯度传播。
        with torch.no_grad():
            for _ in range(newton_maxiter - 1):
                new_o = ray.o + ray.d * t.unsqueeze(-1)
                new_x, new_y = new_o[..., 0], new_o[..., 1]
                valid = self.is_within_data_range(new_x, new_y) & (ray.is_valid > 0)

                x, y = new_x * valid, new_y * valid
                ft = self._sag(x, y) - new_o[..., 2]
                dfdx, dfdy = self._dfdxy(x, y)
                dfdt = dfdx * dxdt + dfdy * dydt - dzdt
                t = t - torch.clamp(
                    ft / (dfdt + EPSILON), -newton_step_bound, newton_step_bound
                )

        # 2. 一次可微 Newton 步骤（最后一次迭代 + 梯度传播）
        new_o = ray.o + ray.d * t.unsqueeze(-1)
        new_x, new_y = new_o[..., 0], new_o[..., 1]
        valid = self.is_valid(new_x, new_y) & (ray.is_valid > 0)

        x, y = new_x * valid, new_y * valid
        ft = self._sag(x, y) - new_o[..., 2]
        dfdx, dfdy = self._dfdxy(x, y)
        dfdt = dfdx * dxdt + dfdy * dydt - dzdt
        t = t - torch.clamp(
            ft / (dfdt + EPSILON), -newton_step_bound, newton_step_bound
        )

        # 3. 确定有效解——复用可微步骤中的 ft 和 valid
        with torch.no_grad():
            valid = valid & (ft.abs() < newton_convergence)

        return t, valid

    def refract(self, ray, eta):
        """在局部坐标系中通过矢量 Snell 定律折射光线。

        表面法线从表面指向光线来向一侧。当 `ray.d` 已归一化时，输出方向保持
        归一化。发生全反射的光线会被标记为无效。

        参数：
            ray (Ray): 入射光线束。
            eta (float): 折射率之比，$\\eta = n_i / n_t$。

        返回：
            ray (Ray): 方向和有效性掩膜均已更新的折射光线。

        参考资料：
            [1] https://registry.khronos.org/OpenGL-Refpages/gl4/html/refract.xhtml
            [2] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # 计算法向量
        normal_vec = self.normal_vec(ray)

        # 按 Snell 定律计算折射，normal_vec * ray_d
        dot_product = (-normal_vec * ray.d).sum(-1).unsqueeze(-1)
        k = 1 - eta**2 * (1 - dot_product**2)

        # 全反射
        valid = (k >= 0).squeeze(-1) & (ray.is_valid > 0)
        k = k * valid.unsqueeze(-1)

        # 更新光线方向
        new_d = eta * ray.d + (eta * dot_product - torch.sqrt(k + EPSILON)) * normal_vec
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        # 更新光线有效性掩膜
        ray.is_valid = ray.is_valid * valid

        return ray

    def bend_penalty(self, ray, old_d):
        """为光线累加逐表面的软弯折惩罚。

        当 `old_d` 与折射后 `ray.d` 之间的弯折角超过 `bend_angle_max`
        （degree，默认 30）时，惩罚会平滑增大；对于较小的折射角，惩罚保持为零。
        该惩罚会加到 `ray.bend_penalty` 中。

        参数：
            ray (Ray): 折射后的光线（`ray.d` 为新方向）。
            old_d (torch.Tensor): 折射前的光线方向，shape 为 [..., 3]，
                与 `ray.d` 相同。

        返回：
            ray (Ray): `bend_penalty`（shape 为 [..., 1]）已更新的光线。
        """
        bend_angle_max = getattr(self, "bend_angle_max", 30.0)
        cos_bend_min = math.cos(math.radians(bend_angle_max))
        cos_bend = torch.sum(ray.d * old_d, dim=-1).unsqueeze(-1)
        per_surf_penalty = F.relu(cos_bend_min - cos_bend)
        valid = ray.is_valid > 0
        ray.bend_penalty = ray.bend_penalty + per_surf_penalty * valid.unsqueeze(-1).float()
        return ray

    def reflect(self, ray):
        """在局部坐标系中使光线在表面发生镜面反射。

        表面法线从表面指向光线来向一侧。反射方向会重新归一化。

        参数：
            ray (Ray): 入射光线束。

        返回：
            ray (Ray): 方向已更新的反射光线。

        参考资料：
            [1] https://registry.khronos.org/OpenGL-Refpages/gl4/html/reflect.xhtml
            [2] https://en.wikipedia.org/wiki/Snell%27s_law, "Vector form" section.
        """
        # 计算表面法向量
        normal_vec = self.normal_vec(ray)

        # 反射
        dot_product = (normal_vec * ray.d).sum(-1).unsqueeze(-1)
        new_d = ray.d - 2 * dot_product * normal_vec
        new_d = F.normalize(new_d, p=2, dim=-1)

        # 更新有效光线
        valid_mask = ray.is_valid > 0
        ray.d = torch.where(valid_mask.unsqueeze(-1), new_d, ray.d)

        return ray

    def normal_vec(self, ray):
        """计算局部坐标系中光线交点处的单位表面法向量。

        法线从表面指向光线来向一侧（会翻转以与正向传播光线相反）。

        参数：
            ray (Ray): 输入光线束，其起点 `ray.o` 位于表面上。

        返回：
            n_vec (torch.Tensor): 单位表面法向量，shape 为 [..., 3]。
        """
        x, y = ray.o[..., 0], ray.o[..., 1]
        nx, ny, nz = self.dfdxyz(x, y)
        n_vec = torch.stack((nx, ny, nz), axis=-1)
        n_vec = F.normalize(n_vec, p=2, dim=-1)

        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        n_vec = torch.where(is_forward, n_vec, -n_vec)
        return n_vec

    def to_local_coord(self, ray):
        """将光线从全局坐标系变换到表面局部坐标系。

        按表面顶点偏移平移光线起点；对于离轴表面，使用缓存的旋转矩阵旋转
        起点和方向。

        参数：
            ray (Ray): 全局坐标系中的输入光线束。

        返回：
            ray (Ray): 以表面局部坐标系表示的光线。
        """
        # 将光线起点平移到表面原点
        offset = torch.stack([self.pos_x, self.pos_y, self.d]).expand_as(ray.o)
        ray.o = ray.o - offset

        # 使用初始化时缓存的矩阵旋转（vec_local/vec_global 为静态），避免在每次
        # 光线与表面相互作用时重新构建。None 表示无需旋转（表面位于轴上）。
        if self._R_to_local is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_local)
            ray.d = self._apply_rotation(ray.d, self._R_to_local)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        return ray

    def to_global_coord(self, ray):
        """将光线从表面局部坐标系变换回全局坐标系。

        这是 `to_local_coord` 的逆变换：对于离轴表面，先使用缓存的逆矩阵旋转，
        再按顶点偏移将起点平移回去。

        参数：
            ray (Ray): 表面局部坐标系中的输入光线束。

        返回：
            ray (Ray): 以全局坐标系表示的光线。
        """
        # 使用缓存的逆矩阵旋转（参见 to_local_coord）。
        if self._R_to_global is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_global)
            ray.d = self._apply_rotation(ray.d, self._R_to_global)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        # 将光线起点平移回全局坐标
        offset = torch.stack([self.pos_x, self.pos_y, self.d]).expand_as(ray.o)
        ray.o = ray.o + offset

        return ray

    def _get_rotation_matrix(self, vec_from, vec_to):
        """计算将 `vec_from` 旋转到 `vec_to` 的旋转矩阵。

        一般情况下使用 Rodrigues 旋转公式，并对同向和反向平行输入进行特殊
        处理。输入会在内部归一化。

        参数：
            vec_from (torch.Tensor): 源方向向量，shape 为 [3]。
            vec_to (torch.Tensor): 目标方向向量，shape 为 [3]。

        返回：
            R (torch.Tensor): 旋转矩阵，shape 为 [3, 3]。
        """
        # 关键步骤：归一化输入向量
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        # 检查向量是否已经同向
        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            # 向量已经同向，返回单位矩阵
            return torch.eye(3, device=self.device)

        if torch.abs(dot_product + 1.0) < EPSILON:
            # 向量方向相反，需要旋转 180 degree
            # 寻找垂直向量
            if torch.abs(vec_from[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=self.device)

            # 通过叉积得到旋转轴
            axis = torch.linalg.cross(vec_from, perp)
            axis = F.normalize(axis, p=2, dim=-1)

            # 180 degree 旋转矩阵
            R = 2.0 * torch.outer(axis, axis) - torch.eye(3, device=self.device)
            return R

        # 一般情况：使用 Rodrigues 旋转公式
        # 对归一化向量：v × u = sin(θ) * k（其中 k 为单位旋转轴），
        # 且 v · u = cos(θ)
        v_cross_u = torch.linalg.cross(vec_from, vec_to)
        cos_angle = dot_product

        # 叉积 v × u 的反对称矩阵（不是归一化旋转轴！）
        # 通过 torch.stack 构建，避免由张量标量复制构造张量
        # （该操作会发出警告并强制主机同步）。
        zero = torch.zeros((), device=self.device, dtype=v_cross_u.dtype)
        K = torch.stack(
            [
                torch.stack([zero, -v_cross_u[2], v_cross_u[1]]),
                torch.stack([v_cross_u[2], zero, -v_cross_u[0]]),
                torch.stack([-v_cross_u[1], v_cross_u[0], zero]),
            ]
        )

        # Rodrigues 公式：R = I + K + K²/(1 + cos(θ))
        # 等价于：R = I + sin(θ)K + (1-cos(θ))K²
        identity = torch.eye(3, device=self.device)
        R = identity + K + torch.mm(K, K) / (1 + cos_angle)

        return R

    def _apply_rotation(self, vectors, R):
        """将旋转矩阵应用于一批向量。

        参数：
            vectors (torch.Tensor): 输入向量，shape 为 [..., 3]。
            R (torch.Tensor): 旋转矩阵，shape 为 [3, 3]。

        返回：
            rotated_vectors (torch.Tensor): 旋转后的向量，shape 为 [..., 3]。
        """
        original_shape = vectors.shape
        # 重塑为 [..., 3] 以执行矩阵乘法
        vectors_flat = vectors.view(-1, 3)
        # 应用旋转：v' = R @ v（批量运算时转置）
        rotated_flat = torch.mm(vectors_flat, R.t())
        # 重塑回原始 shape
        return rotated_flat.view(original_shape)

    # =====================================================================
    # 计算函数
    # =====================================================================
    def sag(self, x, y, valid=None):
        """使用有效性掩膜计算表面矢高 $z = f(x, y)$ [mm]。

        `valid` 掩膜在调用 `_sag` 前将超范围坐标置零，避免球面／非球面在
        $x = y = 0$ 处反向传播时因 $r = \\sqrt{x^2 + y^2}$ 的导数未定义
        （因为 $dr/dx = x/r$）而产生 NaN。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。
            valid (torch.Tensor or None, optional): 有效点的布尔掩膜，shape 与
                `x` 相同。默认值为 None，此时通过 `is_valid` 计算。

        返回：
            z (torch.Tensor): 表面矢高 [mm]，shape 与 `x` 相同。
        """
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        return self._sag(x, y)

    def _sag(self, x, y):
        """计算原始表面矢高 $z = f(x, y)$ [mm]（由子类实现）。

        由 `sag` 调用的子类钩子，要求坐标已应用掩膜。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            z (torch.Tensor): 表面矢高 [mm]，shape 与 `x` 相同。

        异常：
            NotImplementedError: 基类始终抛出此异常；由子类重写。
        """
        raise NotImplementedError(
            "_sag() is not implemented for {}".format(self.__class__.__name__)
        )

    def dfdxyz(self, x, y, valid=None):
        """计算隐式表面函数的梯度。

        表面隐式定义为 $f(x, y, z) = \\mathrm{sag}(x, y) - z = 0$。该梯度用于
        Newton 法和法向量计算。此处的解析实现仅适用于显式表面
        $z = \\mathrm{sag}(x, y)$；对于隐式表面，可改用数值有限差分或 autograd。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。
            valid (torch.Tensor or None, optional): 有效点的布尔掩膜，shape 与
                `x` 相同。默认值为 None，通过 `is_valid` 计算。

        返回：
            dfdx (torch.Tensor): 偏导数 $\\partial f/\\partial x$ [1]，shape 与 `x` 相同。
            dfdy (torch.Tensor): 偏导数 $\\partial f/\\partial y$ [1]，shape 与 `x` 相同。
            dfdz (torch.Tensor): 偏导数 $\\partial f/\\partial z = -1$，shape 与 `x` 相同。
        """
        if valid is None:
            valid = self.is_valid(x, y)

        x, y = x * valid, y * valid
        dx, dy = self._dfdxy(x, y)
        return dx, dy, -torch.ones_like(x)

    def _dfdxy(self, x, y):
        """计算矢高关于 x 和 y 的偏导数（由子类实现）。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            dfdx (torch.Tensor): 偏导数 $\\partial f/\\partial x$，shape 与 `x` 相同。
            dfdy (torch.Tensor): 偏导数 $\\partial f/\\partial y$，shape 与 `x` 相同。

        异常：
            NotImplementedError: 基类始终抛出此异常；由子类重写。
        """
        raise NotImplementedError(
            "_dfdxy() is not implemented for {}".format(self.__class__.__name__)
        )

    def d2fdxyz2(self, x, y, valid=None):
        """计算隐式表面函数的二阶偏导数。

        表面函数为 $f(x, y, z) = \\mathrm{sag}(x, y) - z = 0$，因此所有涉及
        $z$ 的二阶导数均为零。目前仅用于表面约束。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。
            valid (torch.Tensor or None, optional): 有效点的布尔掩膜，shape 与
                `x` 相同。默认值为 None，通过 `is_within_data_range` 计算。

        返回：
            d2f_dx2 (torch.Tensor): $\\partial^2 f/\\partial x^2$，shape 与 `x` 相同。
            d2f_dxdy (torch.Tensor): $\\partial^2 f/\\partial x\\partial y$，shape 与 `x` 相同。
            d2f_dy2 (torch.Tensor): $\\partial^2 f/\\partial y^2$，shape 与 `x` 相同。
            d2f_dxdz (torch.Tensor): $\\partial^2 f/\\partial x\\partial z = 0$，shape 与 `x` 相同。
            d2f_dydz (torch.Tensor): $\\partial^2 f/\\partial y\\partial z = 0$，shape 与 `x` 相同。
            d2f_dz2 (torch.Tensor): $\\partial^2 f/\\partial z^2 = 0$，shape 与 `x` 相同。
        """
        if valid is None:
            valid = self.is_within_data_range(x, y)

        x, y = x * valid, y * valid

        # 计算 sag(x, y) 的二阶导数
        d2f_dx2, d2f_dxdy, d2f_dy2 = self._d2fdxy(x, y)

        # 涉及 z 的混合偏导数为零
        zeros = torch.zeros_like(x)
        d2f_dxdz = zeros  # ∂²f/∂x∂z = 0
        d2f_dydz = zeros  # ∂²f/∂y∂z = 0
        d2f_dz2 = zeros  # ∂²f/∂z² = 0

        return d2f_dx2, d2f_dxdy, d2f_dy2, d2f_dxdz, d2f_dydz, d2f_dz2

    def _d2fdxy(self, x, y):
        """使用中心有限差分计算二阶矢高导数。

        使用 $10^{-6}$ mm 的步长返回 $f''_{xx}$、$f''_{xy}$、$f''_{yy}$。
        仅用于表面约束。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            d2fdx2 (torch.Tensor): $\\partial^2 f/\\partial x^2$，shape 与 `x` 相同。
            d2fdxy (torch.Tensor): $\\partial^2 f/\\partial x\\partial y$，shape 与 `x` 相同。
            d2fdy2 (torch.Tensor): $\\partial^2 f/\\partial y^2$，shape 与 `x` 相同。
        """
        delta_x = 1e-6
        delta_y = 1e-6
        d2fdx2 = (self._dfdxy(x + delta_x, y)[0] - self._dfdxy(x - delta_x, y)[0]) / (
            2 * delta_x
        )
        d2fdy2 = (self._dfdxy(x, y + delta_y)[1] - self._dfdxy(x, y - delta_y)[1]) / (
            2 * delta_y
        )
        d2fdxy = (self._dfdxy(x + delta_x, y)[1] - self._dfdxy(x - delta_x, y)[1]) / (
            2 * delta_x
        )
        return d2fdx2, d2fdxy, d2fdy2

    def is_valid(self, x, y):
        """返回同时位于数据范围和孔径边界内的点掩膜。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            valid (torch.Tensor): 布尔掩膜，shape 与 `x` 相同。
        """
        return self.is_within_data_range(x, y) & self.is_within_boundary(x, y)

    def is_within_boundary(self, x, y):
        """返回孔径边界内的点掩膜。

        对于方形孔径，边界为半边长 `w/2`、`h/2`；否则使用圆形半径 `r`。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            valid (torch.Tensor): 布尔掩膜，shape 与 `x` 相同。
        """
        if self.is_square:
            valid = (torch.abs(x) <= (self.w / 2 + EPSILON)) & (
                torch.abs(y) <= (self.h / 2 + EPSILON)
            )
        else:
            r = self.r
            valid = (x**2 + y**2) <= (r**2 + EPSILON)

        return valid

    def is_within_data_range(self, x, y):
        """返回矢高函数数据区域内的点掩膜。

        基础表面的数据区域无界，因此所有点均有效；子类（如球面）会重写此方法，
        以排除矢高未定义的区域。

        参数：
            x (torch.Tensor): 局部 x 坐标 [mm]，任意 shape。
            y (torch.Tensor): 局部 y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            valid (torch.Tensor): 布尔掩膜，shape 与 `x` 相同（此处均为 True）。
        """
        return torch.ones_like(x, dtype=torch.bool)

    def max_height(self):
        """返回表面的最大有效径向高度 [mm]。

        返回：
            max_height (float): 最大有效高度 [mm]（基础表面为 10e3）。
        """
        return 10e3

    def surface_with_offset(self, x, y, valid_check=True):
        """计算表面在 (x, y) 处的全局 z 坐标。

        在局部矢高上加顶点轴向位置 `d`。用于镜头布局绘制和自相交检测。

        参数：
            x (torch.Tensor or float): 局部 x 坐标 [mm]。
            y (torch.Tensor or float): 局部 y 坐标 [mm]，shape 与 `x` 相同。
            valid_check (bool, optional): 为 True 时通过 `sag` 应用 `is_valid`
                掩膜；为 False 时使用原始 `_sag`。默认值为 True。

        返回：
            z (torch.Tensor): 全局 z 坐标 [mm]，shape 与 `x` 相同。
        """
        x = x if torch.is_tensor(x) else torch.tensor(x, device=self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y, device=self.device)
        if valid_check:
            return self.sag(x, y) + self.d
        else:
            return self._sag(x, y) + self.d

    def surface_sag(self, x, y):
        """以 Python float 形式计算 (x, y) 处的局部表面矢高。

        此函数目前未使用。

        参数：
            x (torch.Tensor or float): 局部 x 坐标 [mm]。
            y (torch.Tensor or float): 局部 y 坐标 [mm]。

        返回：
            sag (float): (x, y) 处的表面矢高 [mm]。
        """
        x = x if torch.is_tensor(x) else torch.tensor(x, device=self.device)
        y = y if torch.is_tensor(y) else torch.tensor(y, device=self.device)
        return self.sag(x, y).item()

    # =====================================================================
    # 优化
    # =====================================================================

    def get_optimizer_params(self, lrs=[1e-4], optim_mat=False):
        """构建逐参数优化器参数组（由子类实现）。

        参数：
            lrs (list[float], optional): 表面可微参数的学习率。默认值为 [1e-4]。
            optim_mat (bool, optional): 是否同时优化材料折射率／色散。
                默认值为 False。

        返回：
            params (list[dict]): Adam 参数组（参数张量和 `lr`）。

        异常：
            NotImplementedError: 基类始终抛出此异常；由子类重写。
        """
        raise NotImplementedError(
            "get_optimizer_params() is not implemented for {}".format(
                self.__class__.__name__
            )
        )

    def get_optimizer(self, lrs=[1e-4], optim_mat=False):
        """为表面可微参数构建 Adam 优化器。

        参数：
            lrs (list[float], optional): 传递给 `get_optimizer_params` 的学习率。
                默认值为 [1e-4]。
            optim_mat (bool, optional): 是否优化材料。默认值为 False。

        返回：
            optimizer (torch.optim.Adam): 表面的 Adam 优化器。
        """
        params = self.get_optimizer_params(lrs, optim_mat=optim_mat)
        return torch.optim.Adam(params)

    def update_r(self, r):
        """更新孔径半径，并将其限制在 `max_height` 以内。

        参数：
            r (float): 请求的孔径半径 [mm]。
        """
        r_max = self.max_height()
        self.r = min(r, r_max)

    # =====================================================================
    # 可视化
    # =====================================================================
    def draw_r(self):
        """返回有效绘制半径 [mm]，并将其限制在 `max_height` 以内。

        返回：
            r_eff (float): 有效绘制半径 [mm]。
        """
        return min(self.r, self.max_height())

    def draw_widget(self, ax, color="black", linestyle="solid"):
        """在 Matplotlib 坐标轴上将表面轮廓绘制为二维曲线。

        绘制在孔径范围内采样的子午（y-z）截面。

        参数：
            ax (matplotlib.axes.Axes): 用于绘制的坐标轴。
            color (str, optional): 线条颜色。默认值为 "black"。
            linestyle (str, optional): Matplotlib 线型。默认值为 "solid"。
        """
        r_eff = self.draw_r()
        r = torch.linspace(-r_eff, r_eff, 128, device=self.device)
        z = self.surface_with_offset(
            r, torch.zeros(len(r), device=self.device), valid_check=False
        )
        ax.plot(
            z.cpu().detach().numpy(),
            r.cpu().detach().numpy(),
            color=color,
            linestyle=linestyle,
            linewidth=0.75,
        )

    def create_mesh(self, n_rings=32, n_arms=128, color=[0.06, 0.3, 0.6]):
        """创建用于三维可视化的表面三角网格。

        设置 `self.vertices`、`self.faces`、`self.rim` 和 `self.mesh_color`。

        参数：
            n_rings (int, optional): 用于径向采样的同心环数量。默认值为 32。
            n_arms (int, optional): 角向分区数量。默认值为 128。
            color (list[float], optional): [0, 1] 范围内的 RGB 网格颜色。
                默认值为 [0.06, 0.3, 0.6]。

        返回：
            self (Surface): 已设置网格数据的表面（用于链式调用）。
        """
        self.vertices = self._create_vertices(n_rings, n_arms)
        self.faces = self._create_faces(n_rings, n_arms)
        self.rim = self._create_rim(n_rings, n_arms)
        self.mesh_color = color
        return self

    def _create_vertices(self, n_rings, n_arms):
        """以径向模式创建用于 PyVista 绘制的网格顶点。

        参数：
            n_rings (int): 同心环数量。
            n_arms (int): 角向分区数量。

        返回：
            vertices (numpy.ndarray): 顶点坐标 [mm]，shape 为
                [n_rings * n_arms + 1, 3]（首个顶点为中心点）。
        """
        n_vertices = n_rings * n_arms + 1
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)

        # 中心顶点
        vertices[0] = [0.0, 0.0, self.surface_with_offset(0.0, 0.0).item()]

        # 创建网格并展平
        rings_mesh, arms_mesh = np.meshgrid(
            np.linspace(1, self.r, n_rings, endpoint=False),
            np.linspace(0, 2 * np.pi, n_arms, endpoint=False),
            indexing="ij",
        )
        rings_flat = rings_mesh.flatten()
        arms_flat = arms_mesh.flatten()

        # 计算 x、y、z 坐标
        x_values = rings_flat * np.cos(arms_flat)
        y_values = rings_flat * np.sin(arms_flat)
        z_values = self.surface_with_offset(x_values, y_values).cpu().numpy()

        # 填充顶点数组
        vertices[1:, 0] = x_values
        vertices[1:, 1] = y_values
        vertices[1:, 2] = z_values

        return vertices

    def _create_faces(self, n_rings, n_arms):
        """创建连接网格顶点的 PyVista 三角面。

        根据透射材料翻转绕序，使外法线方向保持一致。

        参数：
            n_rings (int): 同心环数量。
            n_arms (int): 角向分区数量。

        返回：
            faces (numpy.ndarray): 顶点索引三元组，shape 为
                [n_arms * (2 * n_rings - 1), 3]。
        """
        n_faces = n_arms * (2 * n_rings - 1)
        faces = np.zeros((n_faces, 3), dtype=np.uint32)
        normal_direction = -1 if self.mat2.name != "air" else 1

        # 创建中心三角形
        for j in range(n_arms):
            if normal_direction == 1:
                faces[j] = [0, 1 + j, 1 + (j + 1) % n_arms]
            else:
                # 对相反法线方向翻转绕序
                faces[j] = [0, 1 + (j + 1) % n_arms, 1 + j]

        # 创建径向四边形（每个由 2 个三角形组成）
        face_idx = n_arms

        for i_ring in range(1, n_rings):
            for j_arm in range(n_arms):
                # 获取当前环顶点索引
                a = 1 + (i_ring - 1) * n_arms + j_arm
                b = 1 + (i_ring - 1) * n_arms + (j_arm + 1) % n_arms

                # 获取下一环顶点索引
                c = 1 + i_ring * n_arms + j_arm
                d = 1 + i_ring * n_arms + (j_arm + 1) % n_arms

                # 每个四边形创建两个三角形
                if normal_direction == 1:
                    faces[face_idx] = [a, c, b]
                    faces[face_idx + 1] = [b, c, d]
                else:
                    # 对相反法线方向翻转绕序
                    faces[face_idx] = [a, b, c]
                    faces[face_idx + 1] = [b, d, c]
                face_idx += 2

        return faces

    def _create_rim(self, n_rings, n_arms):
        """创建用于连接相邻表面的边缘（外缘）曲线。

        参数：
            n_rings (int): 同心环数量。
            n_arms (int): 角向分区数量。

        返回：
            rim (RimCurve): 外缘曲线（`n_rings` 为 0 时是单点非闭合曲线）。
        """
        if n_rings == 0:
            return RimCurve(self.vertices[[0]], is_loop=False)

        # 获取外环顶点
        start_idx = 1 + (n_rings - 1) * n_arms
        rim_vertices = self.vertices[start_idx : start_idx + n_arms]
        return RimCurve(rim_vertices, is_loop=True)

    def get_polydata(self):
        """根据缓存的顶点和面构建 PyVista PolyData 对象。

        必须先调用 `create_mesh`。该 PolyData 用于绘制表面并导出为 .obj 文件。

        返回：
            polydata (pyvista.PolyData): 以 PyVista PolyData 对象表示的网格。
        """
        from pyvista import PolyData

        face_vertex_n = 3  # 每个三角形的顶点数
        formatted_faces = np.hstack(
            [
                face_vertex_n * np.ones((self.faces.shape[0], 1), dtype=np.uint32),
                self.faces,
            ]
        )
        return PolyData(self.vertices, formatted_faces)

    # =====================================================================
    # 输入输出
    # =====================================================================
    def surf_dict(self):
        """将表面的通用参数序列化为字典。

        返回：
            surf_dict (dict): 表面参数，包括类型、`r`、`d`、`pos_xy`、
                `vec_local`、`is_square`、`mat2`，以及信息项
                `(mat2_n)`/`(mat2_V)`；数值保留 4 位小数。
        """
        surf_dict = {
            "type": self.__class__.__name__,
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "pos_xy": (round(self.pos_x.item(), 4), round(self.pos_y.item(), 4)),
            "vec_local": (
                round(self.vec_local[0].item(), 4),
                round(self.vec_local[1].item(), 4),
                round(self.vec_local[2].item(), 4),
            ),
            "is_square": self.is_square,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """返回描述该表面的 Zemax（.zmx）文本块。

        参数：
            surf_idx (int): 该表面在 Zemax 表面列表中的索引。
            d_next (float): 到下一表面的轴向距离 [mm]（厚度）。

        返回：
            zmx_str (str): Zemax 格式的表面定义字符串。

        异常：
            NotImplementedError: 基类始终抛出此异常；由子类重写。
        """
        raise NotImplementedError(
            "zmx_str() is not implemented for {}".format(self.__class__.__name__)
        )


class RimCurve:
    """表示表面边缘的简单折线曲线。

    存储表面网格的外缘顶点，并兼容 `LineMesh` 接口，因此可以连接相邻表面的
    边缘，形成闭合镜体以进行三维可视化和导出。

    属性：
        vertices (numpy.ndarray): 边缘顶点坐标 [mm]，shape 为 [N, 3]。
        is_loop (bool): 边缘是否形成闭环。
        n_vertices (int): 边缘顶点数量。
    """

    def __init__(self, vertices, is_loop=False):
        """根据一组顶点初始化边缘曲线。

        参数：
            vertices (numpy.ndarray): 边缘顶点坐标 [mm]，shape 为 [N, 3]。
                若输入支持 `.copy()`，则进行复制。
            is_loop (bool, optional): 边缘是否形成闭环。默认值为 False。
        """
        self.vertices = (
            vertices.copy() if hasattr(vertices, "copy") else np.array(vertices)
        )
        self.is_loop = is_loop
        self.n_vertices = len(vertices)
