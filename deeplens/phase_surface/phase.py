"""Phase 类：承载相位图案的平面基底。"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..config import EPSILON
from ..base import DeepObj
from ..material import Material


class Phase(DeepObj):
    """衍射表面（超表面或 DOE）的基础相位分布。

    表示承载相位图案 $\\phi(x, y)$ 的平坦（零矢高）基底，位于全局坐标系的
    轴向位置 $d$。提供通用光线追迹机制（求交、折射、广义 Snell 衍射、
    局部／全局坐标变换）；相位分布 $\\phi$ 及其梯度由子类定义。

    属性：
        vec_global (torch.Tensor): 全局轴方向 $[0, 0, 1]$，shape 为 [3]。
        d (torch.Tensor): 表面平面的轴向位置 [mm]，标量。
        pos_x (torch.Tensor): 表面 x 偏移 [mm]，标量。
        pos_y (torch.Tensor): 表面 y 偏移 [mm]，标量。
        vec_local (torch.Tensor): 全局坐标中的单位表面法线，shape 为 [3]。
        mat2 (Material): 表面出射侧材料。
        r (float): 表面半径／半孔径 [mm]。
        is_square (bool): 为 True 时孔径是边长 $r\\sqrt{2}$ 的方形；
            否则是半径为 $r$ 的圆形。
        w (float): 方形孔径宽度 $r\\sqrt{2}$ [mm]。
        h (float): 方形孔径高度 $r\\sqrt{2}$ [mm]。
        diffraction_order (int): 广义 Snell 定律使用的衍射级次 $m$。默认值为 1。
        norm_radii (float): 相位多项式坐标归一化所用半径 [mm]。默认值为 `r`。
        device (str or torch.device): 存放张量状态的设备。

    参考资料：
        [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
        [2] https://optics.ansys.com/hc/en-us/articles/360042097313-Small-Scale-Metalens-Field-Propagation
        [3] https://optics.ansys.com/hc/en-us/articles/18254409091987-Large-Scale-Metalens-Ray-Propagation
    """

    def __init__(
        self,
        r,
        d,
        norm_radii=None,
        mat2="air",
        pos_xy=(0.0, 0.0),
        vec_local=(0.0, 0.0, 1.0),
        is_square=True,
        device="cpu",
    ):
        """初始化平坦相位基底。

        参数：
            r (float): 表面半径／半孔径 [mm]。
            d (float): 表面平面的轴向位置 [mm]。
            norm_radii (float or None, optional): 相位多项式坐标归一化所用半径
                [mm]。默认值为 None，此时使用 `r`。
            mat2 (str, optional): 表面出射侧材料。默认值为 "air"。
            pos_xy (tuple, optional): 表面中心的横向 (x, y) 偏移 [mm]。
                默认值为 (0.0, 0.0)。
            vec_local (tuple, optional): 全局坐标中的表面法线方向（不要求已归一化）。
                默认值为 (0.0, 0.0, 1.0)。
            is_square (bool, optional): 为 True 时孔径是边长 $r\\sqrt{2}$ 的方形；
                否则是半径为 $r$ 的圆形。默认值为 True。
            device (str, optional): 张量状态使用的设备。默认值为 "cpu"。
        """
        super().__init__()

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

        # DOE 几何参数
        self.r = float(r)
        self.is_square = is_square
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

        self.diffraction_order = 1
        self.norm_radii = self.r if norm_radii is None else norm_radii

        self.device = device if device is not None else torch.device("cpu")
        self.to(self.device)

        # 预计算旋转矩阵（仅依赖静态的 vec_local/vec_global）
        self._cache_rotation_matrices()

    def _cache_rotation_matrices(self):
        """预计算并缓存局部／全局坐标变换的旋转矩阵。"""
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

    # ==============================
    # 由子类实现的抽象方法
    # ==============================
    def phi(self, x, y):
        """设计波长下的参考相位图，必须由子类实现。"""
        raise NotImplementedError("phi() must be implemented by subclasses")

    def dphi_dxy(self, x, y):
        """计算相位导数，必须由子类实现。"""
        raise NotImplementedError("dphi_dxy() must be implemented by subclasses")

    # ==============================
    # 计算（光线追迹）
    # ==============================
    def ray_reaction(self, ray, n1, n2):
        """追迹光线通过相位表面。

        将光线变换到局部坐标系，与平面求交，依次施加折射和衍射，再变换回
        全局坐标系。

        参数：
            ray (Ray): 全局坐标中的入射光线。
            n1 (float): 表面前介质的折射率。
            n2 (float): 表面后介质的折射率。

        返回：
            ray (Ray): 更新后的全局坐标光线。
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray, n1)
        ray = self.refract(ray, n1 / n2)
        ray = self.diffract(ray, n2=n2)
        ray = self.to_global_coord(ray)
        return ray

    def intersect(self, ray, n=1.0):
        """在局部坐标中求解光线与平面的交点并更新光线。

        将每条光线推进到 $z = 0$ 平面，将落在孔径外的光线标记为无效，并对
        相干光线累加光程。对于几乎平行于平面的光线，避免除以接近零的 z 方向。

        参数：
            ray (Ray): 局部坐标中的光线。
            n (float, optional): 光线传播介质的折射率，用于累加 OPL。
                默认值为 1.0。

        返回：
            ray (Ray): 已推进到表面平面，并更新 `o`、`is_valid` 和 `opl` 的光线。
        """
        # 求解交点。除法前防止 z 方向接近零（光线平行于平面），
        # 与 ray.py 的 prop_to 保持一致。
        dz = ray.d[..., 2]
        dz = torch.where(dz.abs() < EPSILON, torch.full_like(dz, EPSILON), dz)
        t = (0.0 - ray.o[..., 2]) / dz
        new_o = ray.o + t.unsqueeze(-1) * ray.d
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

    def diffract(self, ray, n2=1.0):
        """对光线施加相位表面衍射。

        施加以下两种效应：

        1. 相位 $\\phi$（单位为 rad）按 $\\phi \\cdot \\lambda / (2\\pi)$ 加入光程，
           其中 $\\lambda$ 在内部从 [µm] 转换为 [mm]。
        2. 相位梯度通过广义 Snell 定律弯折光线
           $n_2 \\sin\\theta_2 = n_1 \\sin\\theta_1 + m\\,\\lambda / (2\\pi)\\,\\partial\\phi/\\partial x$.
           由于标准折射已施加，加到单位方向上的剩余偏转为
           $\\Delta l = m\\,\\lambda / (2\\pi n_2)\\,\\partial\\phi/\\partial x$。
           对反向传播光线，偏转符号翻转。

        参数：
            ray (Ray): 包含位置、方向和波长 [µm] 的光线。
            n2 (float, optional): 表面后介质的折射率；偏转按 $1/n_2$ 缩放。
                默认值为 1.0。

        返回：
            ray (Ray): 方向 `d` 以及相干光线的 `opl` 已更新的光线。

        说明：
            此处不建模材料色散。相位分布 $\\phi(x, y)$ 视为与波长无关；只有广义
            Snell 定律中的 $\\lambda$ 缩放和 OPL 累加随波长变化。对于相位分布
            本身通过 $(n(\\lambda) - 1)\\,h$ 随波长变化的物理 DOE，请改用
            `DiffractiveSurface`。

        参考文献：
            [1] https://support.zemax.com/hc/en-us/articles/1500005489061-How-diffractive-surfaces-are-modeled-in-OpticStudio
            [2] Light propagation with phase discontinuities: generalized laws of reflection and refraction. Science 2011.
        """
        forward = (ray.d * ray.is_valid.unsqueeze(-1))[..., 2].sum() > 0
        valid = ray.is_valid > 0

        # 步骤 1：DOE 相位调制
        if ray.is_coherent:
            phi = self.phi(ray.o[..., 0], ray.o[..., 1])
            new_opl = ray.opl + phi.unsqueeze(-1) * (ray.wvln * 1e-3) / (2 * torch.pi)
            ray.opl = torch.where(valid.unsqueeze(-1), new_opl, ray.opl)

        # 步骤 2：通过广义 Snell 定律弯折光线
        # n₂·l₂ = n₁·l₁ + M·λ/(2π)·dφ/dx
        # 折射后：l₂ = l_refracted + M·λ/(2π·n₂)·dφ/dx
        dphidx, dphidy = self.dphi_dxy(ray.o[..., 0], ray.o[..., 1])

        wvln_mm = ray.wvln * 1e-3
        order = self.diffraction_order
        phase_deflection_scale = wvln_mm / (2 * torch.pi * n2)
        if forward:
            new_d_x = ray.d[..., 0] + phase_deflection_scale * dphidx * order
            new_d_y = ray.d[..., 1] + phase_deflection_scale * dphidy * order
        else:
            new_d_x = ray.d[..., 0] - phase_deflection_scale * dphidx * order
            new_d_y = ray.d[..., 1] - phase_deflection_scale * dphidy * order

        new_d = torch.stack([new_d_x, new_d_y, ray.d[..., 2]], dim=-1)
        new_d = F.normalize(new_d, p=2, dim=-1)
        ray.d = torch.where(valid.unsqueeze(-1), new_d, ray.d)

        return ray

    def refract(self, ray, eta):
        """在局部坐标系中根据 Snell 定律计算折射光线。

        参数：
            ray (Ray): 入射光线。
            eta (float): 折射率之比，eta = n_i / n_t

        返回：
            ray (Ray): 折射光线。
        """
        # 计算法向量
        normal_vec = self.normal_vec(ray)

        # 根据 Snell 定律计算折射
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

    def normal_vec(self, ray):
        """计算交点处的表面法向量。

        法线从表面指向光线来向一侧（即翻转为与光线 z 方向相反）。

        参数：
            ray (Ray): 提供传播方向的光线。

        返回：
            normal_vec (torch.Tensor): 单位法向量，shape 与 `ray.d` 相同。
        """
        normal_vec = torch.zeros_like(ray.d)
        normal_vec[..., 2] = -1
        is_forward = ray.d[..., 2].unsqueeze(-1) > 0
        normal_vec = torch.where(is_forward, normal_vec, -normal_vec)
        return normal_vec

    def to_local_coord(self, ray):
        """将光线变换到局部坐标系。

        参数：
            ray (Ray): 全局坐标系中的输入光线。

        返回：
            ray (Ray): 变换到局部坐标系的光线。
        """
        # 将光线起点平移到表面原点
        offset = torch.stack([self.pos_x, self.pos_y, self.d]).expand_as(ray.o)
        ray.o = ray.o - offset

        # 使用初始化时缓存的矩阵旋转，避免每次相互作用时重新构建。
        # None 表示无需旋转（表面位于轴上）。
        if self._R_to_local is not None:
            ray.o = self._apply_rotation(ray.o, self._R_to_local)
            ray.d = self._apply_rotation(ray.d, self._R_to_local)
            ray.d = F.normalize(ray.d, p=2, dim=-1)

        return ray

    def to_global_coord(self, ray):
        """将光线变换到全局坐标系。

        参数：
            ray (Ray): 局部坐标系中的输入光线。

        返回：
            ray (Ray): 变换到全局坐标系的光线。
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
        """计算将 vec_from 旋转到 vec_to 的旋转矩阵。"""
        vec_from = F.normalize(vec_from.to(self.device), p=2, dim=-1)
        vec_to = F.normalize(vec_to.to(self.device), p=2, dim=-1)

        dot_product = torch.dot(vec_from, vec_to)
        if torch.abs(dot_product - 1.0) < EPSILON:
            return torch.eye(3, device=self.device)

        if torch.abs(dot_product + 1.0) < EPSILON:
            if torch.abs(vec_from[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=self.device)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            axis = torch.linalg.cross(vec_from, perp)
            axis = F.normalize(axis, p=2, dim=-1)
            R = 2.0 * torch.outer(axis, axis) - torch.eye(3, device=self.device)
            return R

        v_cross_u = torch.linalg.cross(vec_from, vec_to)
        cos_angle = dot_product

        K = torch.tensor(
            [
                [0, -v_cross_u[2], v_cross_u[1]],
                [v_cross_u[2], 0, -v_cross_u[0]],
                [-v_cross_u[1], v_cross_u[0], 0],
            ],
            device=self.device,
        )

        identity = torch.eye(3, device=self.device)
        R = identity + K + torch.mm(K, K) / (1 + cos_angle)

        return R

    def _apply_rotation(self, vectors, R):
        """将旋转矩阵应用于向量。"""
        original_shape = vectors.shape
        vectors_flat = vectors.view(-1, 3)
        rotated_flat = torch.mm(vectors_flat, R.t())
        return rotated_flat.view(original_shape)

    # ==============================
    # 优化
    # ==============================
    def get_optimizer_params(self, lrs=[1e-4, 1e-2], optim_mat=False):
        """生成优化器参数，必须由子类实现。"""
        raise NotImplementedError(
            "get_optimizer_params() must be implemented by subclasses"
        )

    def get_optimizer(self, lrs):
        """为表面的可学习参数构建 Adam 优化器。

        参数：
            lrs (list or float): 参数组的学习率。单个 float 会包装为单元素列表。

        返回：
            optimizer (torch.optim.Adam): 用于 `get_optimizer_params` 返回参数的
                Adam 优化器。
        """
        if isinstance(lrs, float):
            lrs = [lrs]
        params = self.get_optimizer_params(lrs)
        optimizer = torch.optim.Adam(params)
        return optimizer

    def update_r(self, r):
        """更新表面半径／半孔径及方形孔径范围。

        平坦相位表面没有几何高度约束；由于多项式使用固定 `norm_radii` 归一化，
        相位系数无需重新缩放。

        参数：
            r (float): 新表面半径／半孔径 [mm]。
        """
        self.r = float(r)
        self.w = self.r * float(np.sqrt(2))
        self.h = self.r * float(np.sqrt(2))

    def phase2height_map(self, design_wvln, refractive_idx=1.5, res=512):
        """将相位图转换为用于 DOE 制造的物理高度图。

        根据空气中透射式 DOE 的相位－高度关系推导：
        $\\phi = (2\\pi/\\lambda)(n - 1)h$, giving $h = \\phi\\lambda / (2\\pi(n - 1))$.

        参数：
            design_wvln (float): 设计波长 [µm]。
            refractive_idx (float, optional): DOE 材料在 `design_wvln` 处的折射率。
                默认值为 1.5。
            res (int, optional): 返回方形高度图的像素分辨率。默认值为 512。

        返回：
            height_map (torch.Tensor): shape 为 [res, res] 的高度图，单位与
                `design_wvln` 相同（[µm]）。
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, res),
            torch.linspace(self.h / 2, -self.h / 2, res),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        phi = self.phi(x, y)  # [0, 2π]，shape 为 [res, res]
        height_map = phi * design_wvln / (2 * torch.pi * (refractive_idx - 1))
        return height_map

    # =========================================
    # 可视化
    # =========================================
    def draw_r(self):
        """二维布局绘制使用的有效半径。"""
        return self.r

    def surface_with_offset(self, *args, **kwargs):
        """返回布局绘制所需的表面轴向位置。

        该表面平坦（零矢高），因此无论横向坐标如何都返回平面位置 `d`。
        为兼容 API，可接收任意位置／关键字参数，但会忽略。

        返回：
            d (torch.Tensor): 平面轴向位置 [mm]，标量。
        """
        return self.d

    def draw_phase_map(self, save_name="./DOE_phase_map.png"):
        """绘制截断到 $[0, 2\\pi]$ 的相位图并保存到文件。

        参数：
            save_name (str, optional): 输出图像路径。默认值为
                "./DOE_phase_map.png"。
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.w / 2, self.w / 2, 2000),
            torch.linspace(self.h / 2, -self.h / 2, 2000),
            indexing="xy",
        )
        x, y = x.to(self.device), y.to(self.device)
        pmap = self.phi(x, y)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        im = ax.imshow(pmap.cpu().numpy(), vmin=0, vmax=2 * torch.pi)
        ax.set_title("Phase map 0.55um", fontsize=10)
        ax.grid(False)
        fig.colorbar(im)
        fig.savefig(save_name, dpi=600, bbox_inches="tight")
        plt.close(fig)

    def draw_widget(self, ax, color="black", linestyle="-"):
        """在二维布局坐标轴上将 DOE 绘制为锯齿（闪耀）轮廓。

        参数：
            ax (matplotlib.axes.Axes): 用于绘制的坐标轴。
            color (str, optional): 为保持 API 一致性而接收，但会忽略；轮廓始终
                以橙色绘制。默认值为 "black"。
            linestyle (str, optional): 轮廓的 Matplotlib 线型。默认值为 "-"。
        """
        # 使用不依赖轴向位置的偏移：否则 d=0 的 DOE 会得到 max_offset=0
        # （np.fmod -> NaN，图像空白），负 d 还会产生负偏移。回退到 r 可保证
        # 任意 r>0 的 DOE 都得到严格正值。
        max_offset = max(abs(self.d.item()), self.r) / 100
        d = self.d.item()

        # 绘制 DOE
        roc = self.r * 2
        x = np.linspace(-self.r, self.r, 128)
        y = np.zeros_like(x)
        r = np.sqrt(x**2 + y**2 + EPSILON)
        sag = roc * (1 - np.sqrt(1 - r**2 / roc**2))
        sag = max_offset - np.fmod(sag, max_offset)
        ax.plot(d + sag, x, color="orange", linestyle=linestyle, linewidth=0.75)

    # =========================================
    # 输入输出
    # =========================================
    def save_ckpt(self, save_path="./doe.pth"):
        """保存 DOE 参数，必须由子类实现。"""
        raise NotImplementedError("save_ckpt() must be implemented by subclasses")

    def load_ckpt(self, load_path="./doe.pth"):
        """加载 DOE 参数，必须由子类实现。"""
        raise NotImplementedError("load_ckpt() must be implemented by subclasses")

    def surf_dict(self):
        """返回表面参数，必须由子类实现。"""
        raise NotImplementedError("surf_dict() must be implemented by subclasses")
