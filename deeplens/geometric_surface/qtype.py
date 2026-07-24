# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Q-type（Forbes Q 多项式）自由曲面。

Q-type 多项式是自由光学曲面设计中常用的正交多项式表示。本模块实现 Qbfs
（Q “最佳拟合球面”自由形状矢高）表示。所有长度单位均为 [mm]。

总表面矢高为基础圆锥面与 Q 多项式偏离量之和：

$$
z(x, y) = \\frac{c\\,r^2}{1 + \\sqrt{1 - (1+k)\\,c^2 r^2}}
          + u^4 \\sum_m a_m\\, Q_m^{bfs}(u^2)
$$

其中 $r = \\sqrt{x^2 + y^2}$，曲率为 $c$ [1/mm]，圆锥常数为 $k$，
归一化径向坐标为 $u = r / r_{norm}$（有效范围 $0 \\le u \\le 1$）。

参考文献：
    G. W. Forbes, "Shape specification for axially symmetric optical surfaces,"
    Opt. Express 15, 5218-5226 (2007).
    G. W. Forbes, "Robust, efficient computational methods for axially symmetric
    optical aspheres," Opt. Express 18, 19700-19712 (2010).
    ISO 10110-19:2015 - Optics and photonics - Preparation of drawings for optical
    elements and systems - Part 19: General description of surfaces and components.
"""

import numpy as np
import torch

from .base import EPSILON, Surface


def compute_qbfs_polynomials(u2, n_terms):
    """计算在 $u^2$ 处取值的 Qbfs 多项式 $Q_0, Q_1, \\ldots, Q_{n-1}$。

    Qbfs 多项式通过三项递推由 Jacobi 多项式 $P_m^{(0,4)}(1 - 2u^2)$ 构建，
    再按 Qbfs 正交归一化因子缩放。此处不施加完整 Forbes 定义中的
    $(1 - u^2)^{-5/2}$ 权重（为提高数值稳定性，该权重并入矢高计算），
    因此这里得到的是不带权的归一化多项式。

    参数：
        u2 (torch.Tensor): 归一化径向坐标平方
            $u^2 = r^2 / r_{norm}^2$，任意 shape。
        n_terms (int): 要计算的 Q 多项式项数。

    返回：
        Q (list[torch.Tensor]): 长度为 `n_terms` 的列表，包含
            $Q_0(u^2), Q_1(u^2), \\ldots, Q_{n\\_terms-1}(u^2)$；各项的
            shape 均与 `u2` 相同。`n_terms` 为 0 时返回空列表。
    """
    if n_terms == 0:
        return []

    # 转换为 Jacobi 多项式自变量：x = 1 - 2*u²
    x = 1 - 2 * u2

    # 使用递推计算 Jacobi 多项式 P_m^(0,4)(x)
    # P_0^(0,4)(x) = 1
    # P_1^(0,4)(x) = -2 + 3x
    # 递推式：P_{n+1}^(0,4)(x) = (A_n * x + B_n) * P_n^(0,4)(x) - C_n * P_{n-1}^(0,4)(x)

    P = [torch.ones_like(u2)]  # P_0

    if n_terms > 1:
        P.append(-2 + 3 * x)  # P_1

    alpha, beta = 0, 4
    for n in range(1, n_terms - 1):
        # Jacobi 多项式的递推系数
        an = 2 * n + alpha + beta
        A_n = (
            (2 * n + alpha + beta + 1)
            * (2 * n + alpha + beta + 2)
            / (2 * (n + 1) * (n + alpha + beta + 1))
        )
        B_n = (
            (alpha**2 - beta**2)
            * (2 * n + alpha + beta + 1)
            / (2 * (n + 1) * (n + alpha + beta + 1) * an)
        )
        C_n = (
            (n + alpha)
            * (n + beta)
            * (2 * n + alpha + beta + 2)
            / ((n + 1) * (n + alpha + beta + 1) * an)
        )

        P_next = (A_n * x + B_n) * P[n] - C_n * P[n - 1]
        P.append(P_next)

    # 转换为 Qbfs：Q_m = P_m^(0,4)(1-2u²) * normalization / (1-u²)^(5/2)
    # 归一化因子保证正交性
    # 为提高数值稳定性，此处计算时不包含 (1-u²)^(-5/2) 因子，
    # 而在矢高计算中将其纳入

    # Qbfs 的归一化因子
    # f_m = sqrt((m+1) * (m+5) * (m+2) * (m+4) * (m+3)^2 / (8 * (2m+5)))
    Q = []
    for m in range(n_terms):
        # 归一化因子
        norm = np.sqrt(
            (m + 1) * (m + 5) * (m + 2) * (m + 4) * (m + 3) ** 2 / (8 * (2 * m + 5))
        )
        # Jacobi 多项式在 x=1 处的归一化：P_m^(0,4)(1) = C(m+4, m)
        jacobi_norm = 1.0
        for k in range(1, 5):
            jacobi_norm *= (m + k) / k
        Q.append(P[m] / (jacobi_norm * norm))

    return Q


class QTypeFreeform(Surface):
    """Q-type（Forbes Qbfs 多项式）自由曲面。

    将旋转对称表面表示为基础圆锥面与 Forbes Qbfs 多项式偏离量之和。基函数
    的正交性使系数在基于梯度的优化中具有良好条件。各个系数还会分别存储为
    属性 `q0`、`q1`、……，以便独立优化。所有长度单位均为 [mm]。

    属性：
        c (torch.Tensor): 基础表面曲率（1 / 曲率半径）[1/mm]。
        k (torch.Tensor): 圆锥常数。
        r_norm (float): Q 多项式的归一化半径 [mm]；默认使用孔径半径 `r`。
        qm (torch.Tensor or None): Q 多项式系数
            $[a_0, a_1, \\ldots, a_{n-1}]$；未设置系数时为 None。
        n_qterms (int): Q 多项式项数（`qm` 的长度）。
    """

    def __init__(
        self,
        r,
        d,
        c,
        k,
        qm,
        mat2,
        r_norm=None,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化 Q-type 自由曲面。

        参数：
            r (float): 表面孔径半径（半直径）[mm]。
            d (float): 从原点到表面顶点的轴向距离 [mm]。
            c (float): 基础表面曲率（1 / 曲率半径）[1/mm]。
            k (float): 圆锥常数（k=0 为球面，k=-1 为抛物面）。
            qm (list): Q 多项式系数 $[a_0, a_1, \\ldots, a_{n-1}]$；
                空列表或 None 表示纯圆锥面。
            mat2 (str or Material): 表面出射侧的材料。
            r_norm (float, optional): Q 多项式的归一化半径 [mm]。默认值为
                None，此时使用 `r`。
            pos_xy (list, optional): 表面中心位置 [x, y] [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list, optional): 局部表面法向量。
                默认值为 [0.0, 0.0, 1.0]。
            is_square (bool, optional): 孔径是否为方形。默认值为 False。
            device (str, optional): Torch 设备。默认值为 "cpu"。
        """
        Surface.__init__(
            self,
            r=r,
            d=d,
            mat2=mat2,
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )

        self.c = torch.tensor(c)
        self.k = torch.tensor(k)
        self.r_norm = r_norm if r_norm is not None else r

        # 存储 Q 多项式系数
        if qm is not None and len(qm) > 0:
            self.qm = torch.tensor(qm, dtype=torch.float64)
            self.n_qterms = len(qm)
            # 同时存储各个系数以便优化
            for i, coef in enumerate(qm):
                setattr(self, f"q{i}", torch.tensor(coef))
        else:
            self.qm = None
            self.n_qterms = 0

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从表面规格字典构造 `QTypeFreeform`。

        可接受 `roc`（曲率半径，转换为曲率 $c = 1 / roc$），也可直接接受
        `c`。键 `k`、`qm` 和 `r_norm` 为可选，缺省时使用不含 Q 偏离量的圆锥面。

        参数：
            surf_dict (dict): 表面规格，包含键 `r`、`d`、`mat2`，以及
                `roc` 或 `c` 之一；可选键为 `k`、`qm`、`r_norm`。

        返回：
            surface (QTypeFreeform): 构造得到的表面实例。
        """
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]

        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            k=surf_dict.get("k", 0.0),
            qm=surf_dict.get("qm", []),
            mat2=surf_dict["mat2"],
            r_norm=surf_dict.get("r_norm", None),
        )

    def _sag(self, x, y):
        """计算表面矢高 $z = f(x, y)$。

        矢高为基础圆锥面与 Q 多项式偏离量之和：

        $$
        z = \\frac{c\\,r^2}{1 + \\sqrt{1 - (1+k)\\,c^2 r^2}}
            + u^4 (1 - u^2)^{5/2} \\sum_m a_m\\, Q_m(u^2)
        $$

        其中 $r^2 = x^2 + y^2$，$u^2 = r^2 / r_{norm}^2$。将圆锥面的被开方数
        和 $1 - u^2$ 限制到 `EPSILON`，以避免表面边界外出现 NaN 矢高／梯度。

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            z (torch.Tensor): 表面矢高 [mm]，shape 与 `x` 相同。
        """
        c = self.c
        k = self.k

        # 径向距离平方
        r2 = x**2 + y**2

        # 基础圆锥面矢高。限制被开方数（而非加 EPSILON）：超过圆锥面边界
        # (1+k)c^2 r^2 > 1 后，自变量会变为负数，torch.sqrt 将返回 NaN
        # （梯度也为 NaN）。该处理与 Aspheric._sag 一致。
        sqrt_term = torch.sqrt(torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON))
        z_base = r2 * c / (1 + sqrt_term)

        # Q 多项式偏离量
        if self.n_qterms > 0:
            # 归一化径向坐标
            u2 = r2 / (self.r_norm**2)
            u4 = u2**2

            # 计算 Q 多项式
            Q_polys = compute_qbfs_polynomials(u2, self.n_qterms)

            # 权重因子：(1 - u²)^(5/2)，用于得到正确的 Qbfs 行为
            # 但为保证 u=1 附近的数值稳定性，这里使用软限制
            one_minus_u2 = torch.clamp(1 - u2, min=EPSILON)
            weight = one_minus_u2 ** (5 / 2)

            # 累加 Q 多项式贡献
            z_q = torch.zeros_like(x)
            for m in range(self.n_qterms):
                qm_coef = getattr(self, f"q{m}")
                z_q = z_q + qm_coef * Q_polys[m]

            # 应用 u⁴ 因子和权重
            z_q = u4 * weight * z_q

            return z_base + z_q

        return z_base

    def _dfdxy(self, x, y):
        """计算矢高关于 $x$ 和 $y$ 的一阶导数。

        通过 $r^2$ 应用链式法则：$\\partial z/\\partial x =
        (\\partial z/\\partial r^2)\\,(2x)$。基础圆锥面部分采用解析求导；
        Q 多项式导数 $\\partial Q_m/\\partial u^2$ 使用前向有限差分近似
        （步长为 $10^{-7}$）。

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            dfdx (torch.Tensor): $\\partial z/\\partial x$ [无量纲]，shape 与 `x` 相同。
            dfdy (torch.Tensor): $\\partial z/\\partial y$ [无量纲]，shape 与 `x` 相同。
        """
        c = self.c
        k = self.k

        r2 = x**2 + y**2

        # 基础圆锥面导数 dz_base/dr²。限制被开方数使其非负（参见 _sag）；
        # 加 EPSILON 无法避免 sqrt 产生 NaN。
        sqrt_term = torch.sqrt(torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON))
        dz_base_dr2 = (
            c
            * (1 + sqrt_term + (1 + k) * r2 * c**2 / (2 * sqrt_term))
            / (1 + sqrt_term) ** 2
        )

        # Q 多项式导数
        if self.n_qterms > 0:
            u2 = r2 / (self.r_norm**2)
            u4 = u2**2

            # 计算 Q 多项式及其导数
            Q_polys = compute_qbfs_polynomials(u2, self.n_qterms)

            # 权重因子
            one_minus_u2 = torch.clamp(1 - u2, min=EPSILON)
            weight = one_minus_u2 ** (5 / 2)

            # d(weight)/du² = (5/2) * (1-u²)^(3/2) * (-1) = -(5/2) * (1-u²)^(3/2)
            dweight_du2 = -2.5 * one_minus_u2 ** (3 / 2)

            # 求和及其导数
            Q_sum = torch.zeros_like(x)
            dQ_sum_du2 = torch.zeros_like(x)

            # Q 多项式的导数目前使用有限差分
            delta = 1e-7
            Q_polys_plus = compute_qbfs_polynomials(u2 + delta, self.n_qterms)

            for m in range(self.n_qterms):
                qm_coef = getattr(self, f"q{m}")
                Q_sum = Q_sum + qm_coef * Q_polys[m]
                dQ_du2 = (Q_polys_plus[m] - Q_polys[m]) / delta
                dQ_sum_du2 = dQ_sum_du2 + qm_coef * dQ_du2

            # z_q = u⁴ * weight * Q_sum
            # dz_q/du² = 2u² * weight * Q_sum + u⁴ * dweight/du² * Q_sum + u⁴ * weight * dQ_sum/du²
            dz_q_du2 = (
                2 * u2 * weight * Q_sum
                + u4 * dweight_du2 * Q_sum
                + u4 * weight * dQ_sum_du2
            )

            # 转换 du²/dr² = 1/r_norm²
            dz_q_dr2 = dz_q_du2 / (self.r_norm**2)

            dz_dr2 = dz_base_dr2 + dz_q_dr2
        else:
            dz_dr2 = dz_base_dr2

        # 链式法则：dz/dx = dz/dr² * 2x，dz/dy = dz/dr² * 2y
        return dz_dr2 * 2 * x, dz_dr2 * 2 * y

    def is_within_data_range(self, x, y):
        """检查点是否位于表面的有效数据范围内。

        当点位于圆锥面边界内（仅在 $k > -1$ 且 $c \\ne 0$ 时存在），并且位于
        Q 多项式归一化半径内（$u^2 \\le 1$）时有效。圆锥面测试完全张量化，
        因此可在 `torch.compile` 下安全使用。

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，shape 与 `x` 相同。

        返回：
            mask (torch.Tensor): 布尔张量，点有效处为 True，shape 与 `x` 相同。
        """
        c = self.c
        k = self.k

        r2 = x**2 + y**2

        # 检查圆锥面有效性。该过程完全张量化（不根据张量 k/c 的值进行 Python
        # 分支），因此可在 torch.compile 下安全使用。仅当 k > -1 且 c != 0
        # 时存在实数圆锥边界；否则所有点均视为有效
        # （与 Aspheric.is_within_data_range 一致）。
        has_boundary = (1 + k > 0) & (c.abs() > EPSILON)
        one_plus_k = 1 + k
        c2 = c * c
        # 边界不存在时避免除零或除以负数；无意义的值会被下方的 where 屏蔽。
        denom = torch.where(
            has_boundary, c2 * one_plus_k, torch.ones_like(c2 * one_plus_k)
        )
        limit_sq = 1.0 / denom - EPSILON
        inside = r2 < limit_sq
        valid_conic = torch.where(has_boundary, inside, torch.ones_like(inside))

        # 检查归一化半径（对于 Q 多项式应 <= 1）
        u2 = r2 / (self.r_norm**2)
        valid_qpoly = u2 <= 1 + EPSILON

        return valid_conic & valid_qpoly

    def max_height(self):
        """返回表面的最大有效径向高度。

        取圆锥面边界半径（$k > -1$ 且 $c \\ne 0$ 时；否则使用较大的备用值
        1e4）与 Q 多项式归一化半径 `r_norm` 中的较小值。

        返回：
            height (float): 最大有效径向距离 [mm]。
        """
        c = self.c
        k = self.k

        # 圆锥面限制
        if k > -1 and abs(c) > EPSILON:
            max_conic = np.sqrt(1 / ((k + 1) * c**2)) - 0.001
        else:
            max_conic = 10e3

        # Q 多项式限制（归一化半径）
        max_q = self.r_norm

        return min(max_conic, max_q)

    # =======================================
    # 优化
    # =======================================

    def get_optimizer_params(
        self, lrs=[1e-4, 1e-4, 1e-2, 1e-6], decay=0.1, optim_mat=False
    ):
        """为该表面的每个参数构建优化器参数组。

        启用 `d`、`c`、`k` 及每个 Q 系数的梯度。第 m 个 Q 系数的学习率为
        `lrs[3] * decay**m`，因此高阶项调整得更慢。

        参数：
            lrs (list, optional): [d, c, k, q_coefficients] 的学习率。
                默认值为 [1e-4, 1e-4, 1e-2, 1e-6]。
            decay (float, optional): 应用于 Q 系数学习率的逐阶衰减。
                默认值为 0.1。
            optim_mat (bool, optional): 是否同时优化出射侧材料。默认值为 False。

        返回：
            params (list): 优化器参数组字典列表，每组包含键 "params" 和 "lr"。
        """
        params = []

        # 距离
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # 曲率
        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[1]})

        # 圆锥常数
        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[2]})

        # Q 多项式系数
        if self.n_qterms > 0:
            base_lr = lrs[3] if len(lrs) > 3 else 1e-6
            for m in range(self.n_qterms):
                qm = getattr(self, f"q{m}")
                qm.requires_grad_(True)
                # 衰减高阶项的学习率
                lr = base_lr * (decay**m)
                params.append({"params": [qm], "lr": lr})

        # 材料参数
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # 输入输出
    # =======================================

    def surf_dict(self):
        """将表面序列化为字典。

        包含类型标记、几何参数（`r`、`d`、`c`、`roc`、`k`、`r_norm`）、
        Q 系数列表 `qm` 和出射材料名称。`(c)`、`(q0)` 等仅显示键存储经过
        舍入的标量值。

        返回：
            surf_dict (dict): 表面的字典表示。
        """
        surf_dict = {
            "type": "QTypeFreeform",
            "r": round(self.r, 4),
            "d": round(self.d.item(), 4),
            "(c)": round(self.c.item(), 6),
            "roc": round(1 / self.c.item(), 4)
            if abs(self.c.item()) > EPSILON
            else float("inf"),
            "k": round(self.k.item(), 6),
            "r_norm": round(self.r_norm, 4),
            "qm": [],
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        for m in range(self.n_qterms):
            qm = getattr(self, f"q{m}")
            surf_dict["qm"].append(float(format(qm.item(), ".6e")))
            surf_dict[f"(q{m})"] = float(format(qm.item(), ".6e"))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """返回 Zemax（.zmx）表面描述字符串。

        输出 `QTYPE` 表面，包括曲率、厚度、孔径、圆锥常数、归一化半径
        （PARM 1）和 Q 系数（PARM 2、3、……）。

        参数：
            surf_idx (int): Zemax 文件中的表面索引。
            d_next (torch.Tensor): 到下一表面的距离 [mm]（标量）。

        返回：
            zmx_str (str): 多行 Zemax 表面描述。

        说明：
            Zemax 的 QTYPE 表示与本实现不同，因此导出结果为近似值，
            针对特定版本可能需要调整。
        """
        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx}
    TYPE QTYPE
    CURV {self.c.item()}
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k.item()}
    PARM 1 {self.r_norm}
"""
        else:
            zmx_str = f"""SURF {surf_idx}
    TYPE QTYPE
    CURV {self.c.item()}
    DISZ {d_next.item()}
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k.item()}
    PARM 1 {self.r_norm}
"""

        # 添加 Q 系数
        for m in range(self.n_qterms):
            qm = getattr(self, f"q{m}")
            zmx_str += f"    PARM {m + 2} {qm.item()}\n"

        return zmx_str
