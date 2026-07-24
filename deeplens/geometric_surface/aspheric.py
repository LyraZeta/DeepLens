# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""非球面。

默认情况下，`ai` 系数列表从四阶项（a4）开始。包含二阶项（a2）的旧版 JSON
文件通过 `init_from_dict` 中的 `use_ai2` 标志加载。若存在 `a2`，则单独存储
并计入矢高计算，但**不进行**优化（它会与基础曲率 `c` 竞争）。

参考资料：
    [1] https://en.wikipedia.org/wiki/Aspheric_lens.
"""

import torch

from .base import EPSILON, Surface


class Aspheric(Surface):
    """偶次非球面。

    矢高函数为：

    $$
    z(\\rho) = \\frac{c\\,\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}}
    + \\sum_{i=2}^{n} a_{2i}\\,\\rho^{2i},
    \\quad \\rho^2 = x^2 + y^2
    $$

    多项式从四阶项（a4）开始，因为二阶项会与基础曲率 `c` 竞争。

    所有系数 `c`、`k` 和 `ai` 都是可微 torch 张量，因此可通过梯度下降优化。

    属性：
        c (torch.Tensor): 基础曲率 [1/mm]。
        k (torch.Tensor): 圆锥常数。
        ai2 (torch.Tensor or None): 二阶非球面系数（旧版）。
        ai (torch.Tensor): 偶次非球面系数
            `[a4, a6, a8, ...]`.
    """

    def __init__(
        self,
        r,
        d,
        c,
        k,
        ai,
        mat2,
        ai2=None,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化非球面。

        参数：
            r (float): 孔径半径 [mm]。
            d (float): 顶点轴向位置 [mm]。
            c (float): 基础曲率 `1/R` [1/mm]。
            k (float): 圆锥常数（`0` = 球面，`-1` = 抛物面）。
            ai (list[float] or None): 从四阶项开始的偶次非球面系数：
                `[a4, a6, a8, ...]`。纯圆锥面请传入 `None` 或空列表。
            mat2 (str or Material): 透射侧材料。
            ai2 (float or None, optional): 旧版数据中的二阶非球面系数。
                计入矢高但不优化。默认值为 None。
            pos_xy (list[float], optional): 横向偏移 `[x, y]` [mm]。
                默认值为 `[0.0, 0.0]`。
            vec_local (list[float], optional): 局部法线方向。
                默认值为 `[0.0, 0.0, 1.0]`。
            is_square (bool, optional): 方形孔径标志。默认值为 False。
            device (str, optional): 计算设备。默认值为 `"cpu"`。
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

        # 二阶系数（旧版，不优化）
        if ai2 is not None:
            self.ai2 = torch.tensor(float(ai2))
        else:
            self.ai2 = None

        if ai is not None and len(ai) > 0:
            self.ai = torch.tensor(ai)
            self.ai_degree = len(ai)
            # ai[0] -> ai4，ai[1] -> ai6，ai[2] -> ai8，……
            for i, a in enumerate(ai):
                setattr(self, f"ai{2 * (i + 2)}", torch.tensor(a))
        else:
            self.ai = None
            self.ai_degree = 0

        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从序列化字典创建非球面。

        若存在 `roc`，则从曲率半径 `roc` [mm] 读取基础曲率并转换为
        `c = 1/roc`；否则从 `c` [1/mm] 读取。对于 `use_ai2` 为 True
        （或缺失）的旧版数据，将 `ai` 的首个元素解释为二阶系数 `a2`，
        其余元素解释为 `[a4, a6, a8, ...]`。

        参数：
            surf_dict (dict): 序列化表面，包含键 `r`、`d`、`k`、`mat2`，
                以及 `roc` 或 `c` 之一；可选键为 `ai` 和 `use_ai2`。

        返回：
            surface (Aspheric): 重建得到的非球面。
        """
        if "roc" in surf_dict:
            if surf_dict["roc"] != 0:
                c = 1 / surf_dict["roc"]
            else:
                c = 0.0
        else:
            c = surf_dict["c"]

        ai = surf_dict.get("ai", [])
        ai2_val = None

        # 向后兼容：旧格式将 a2 作为首个元素。
        # 本代码写出的新文件会显式设置 use_ai2。
        if surf_dict.get("use_ai2", True) and len(ai) > 0:
            if "use_ai2" not in surf_dict:
                print(
                    f"Surface dict lacks 'use_ai2'; assuming ai[0]={ai[0]:.4g} is the "
                    "2nd-order coefficient (legacy format)."
                )
            ai2_val = ai[0]  # 提取 a2 系数
            ai = ai[1:]      # 剩余项：[a4, a6, a8, ...]

        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            k=surf_dict["k"],
            ai=ai,
            ai2=ai2_val,
            mat2=surf_dict["mat2"],
        )

    def _get_curvature_params(self):
        """返回基础曲率 `c` [1/mm] 和圆锥常数 `k`。

        返回：
            c (torch.Tensor): 基础曲率 [1/mm]。
            k (torch.Tensor): 圆锥常数。
        """
        return self.c, self.k

    def _sag(self, x, y):
        """计算表面矢高（轴向高度）$z = \\mathrm{sag}(x, y)$。

        偶次非球面矢高为

        $$
        z = \\frac{c\\,\\rho^2}{1 + \\sqrt{1-(1+k)c^2\\rho^2}}
        + a_2\\rho^2 + \\sum_{i\\ge 2} a_{2i}\\,\\rho^{2i},
        \\quad \\rho^2 = x^2 + y^2,
        $$

        仅在存在旧版 `ai2` 系数时才加入 $a_2\\rho^2$ 项。所有长度单位均为 [mm]。

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，可与 `x` 广播。

        返回：
            z (torch.Tensor): 表面矢高 [mm]，shape 与 `x` 和 `y` 的广播结果相同。
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        sf_arg = torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON)
        total_surface = r2 * c / (1 + torch.sqrt(sf_arg))

        # 旧版 a2 项：a2 * r²
        if self.ai2 is not None:
            total_surface = total_surface + self.ai2 * r2

        # 非球面多项式：ai4*r⁴ + ai6*r⁶ + ai8*r⁸ + ……
        r_pow = r2 * r2  # 从 r^4 开始
        for i in range(self.ai_degree):
            total_surface = total_surface + getattr(self, f"ai{2 * (i + 2)}") * r_pow
            r_pow = r_pow * r2

        return total_surface

    def _dfdxy(self, x, y):
        """计算一阶矢高导数 $\\partial z/\\partial x$ 和 $\\partial z/\\partial y$。

        通过 $\\partial z/\\partial x = (\\partial z/\\partial \\rho^2)\\,2x$
        对矢高求导。对于多项式 $\\sum_{i\\ge 2} a_{2i}\\rho^{2i}$，其关于
        $\\rho^2$ 的导数为 $\\sum_{i\\ge 2} i\\,a_{2i}\\,\\rho^{2(i-1)}$，即
        $2a_4\\rho^2 + 3a_6\\rho^4 + \\dots$

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，可与 `x` 广播。

        返回：
            dfdx (torch.Tensor): $\\partial z/\\partial x$ [无量纲]，shape 与输入相同。
            dfdy (torch.Tensor): $\\partial z/\\partial y$ [无量纲]，shape 与输入相同。
        """
        c, k = self._get_curvature_params()

        r2 = x**2 + y**2
        sf_arg = torch.clamp(1 - (1 + k) * r2 * c**2, min=EPSILON)
        sf = torch.sqrt(sf_arg)
        dsdr2 = (1 + sf + (1 + k) * r2 * c**2 / 2 / sf) * c / (1 + sf) ** 2

        # d(a2*r²)/dr² = a2
        if self.ai2 is not None:
            dsdr2 = dsdr2 + self.ai2

        # 非球面多项式关于 r² 的导数：2*ai4*r² + 3*ai6*r⁴ + ……
        r_pow = r2
        for i in range(self.ai_degree):
            order = i + 2  # 2, 3, 4, ...
            dsdr2 = dsdr2 + order * getattr(self, f"ai{2 * order}") * r_pow
            r_pow = r_pow * r2

        return dsdr2 * 2 * x, dsdr2 * 2 * y

    def is_within_data_range(self, x, y):
        """返回圆锥矢高为实数的点掩膜。

        当 $(1+k)c^2\\rho^2 < 1$ 时点有效，即位于圆锥面的实数边界内。该函数
        完全张量化（不根据张量 `k` 的值进行 Python 分支），因此可安全通过
        `torch.compile` 追踪。当 $k \\le -1$ 时，圆锥面没有实数边界，
        因而所有点均视为有效。

        参数：
            x (torch.Tensor): x 坐标 [mm]，任意 shape。
            y (torch.Tensor): y 坐标 [mm]，可与 `x` 广播。

        返回：
            valid (torch.Tensor): 布尔掩膜，shape 与 `x` 和 `y` 的广播结果相同。
        """
        c, k = self._get_curvature_params()
        one_plus_k = 1 + k
        # 计算极限时避免除零或除以负数；无意义的值会被下方的 where 屏蔽。
        safe = torch.where(
            one_plus_k > 0, one_plus_k, torch.ones_like(one_plus_k)
        )
        limit_sq = 1.0 / (c * c * safe)
        inside = (x * x + y * y) < limit_sq
        return torch.where(one_plus_k > 0, inside, torch.ones_like(inside))

    def max_height(self):
        """返回表面的最大有效径向高度。

        对于扁球形／椭球形圆锥面（$k > -1$），矢高仅在
        $\\rho_{max} = \\sqrt{1/((k+1)c^2)}$ 以内为实数，并减去 0.001 mm
        的小余量。对于 $k \\le -1$，不存在边界，因此返回较大值 10000 mm。

        返回：
            max_height (float): 最大有效径向高度 [mm]。
        """
        c, k = self._get_curvature_params()
        if k > -1:
            return torch.sqrt(1 / (k + 1) / (c**2)).item() - 0.001
        return 10e3

    # =======================================
    # 优化
    # =======================================

    def get_optimizer_params(self, lrs=[1e-4, 1e-4, 1e-2, 1e-4], optim_mat=False):
        """获取各类参数的优化器参数。

        每个非球面系数 $a_{2n}$ 的学习率按 $1 / \\max(r, 1)^{2n}$ 缩放，
        从而无论表面半直径如何，每个 Adam 步骤造成的有效矢高扰动都近似恒定
        （约为 lr_base mm）。若不进行这种归一化，梯度会按 $O(r^{2n})$ 缩放，
        对相机尺寸表面可达 $10^5$，并在数十次迭代内产生 NaN。

        参数：
            lrs (list[float], optional): `[d, c, k, ai]` 的学习率。
                默认值为 `[1e-4, 1e-4, 1e-2, 1e-4]`。
            optim_mat (bool, optional): 是否同时优化材料参数。默认值为 False。

        返回：
            params (list[dict]): 可直接传给 torch 优化器的参数组（每组为包含
                `params` 和 `lr` 的字典）。
        """
        params = []

        # 优化距离
        self.d.requires_grad_(True)
        params.append({"params": [self.d], "lr": lrs[0]})

        # 优化曲率
        self.c.requires_grad_(True)
        params.append({"params": [self.c], "lr": lrs[1]})

        # 优化圆锥常数
        self.k.requires_grad_(True)
        params.append({"params": [self.k], "lr": lrs[2]})

        # 使用按 r 归一化的学习率优化非球面系数。
        # 矢高关于 a_{2n} 的梯度按 r^{2n} 缩放。将学习率除以 r^{2n}，
        # 可使每步的有效矢高变化保持约为 lr_base，从而各阶对表面形状演化
        # 具有相同贡献。
        if self.ai is not None:
            if self.ai_degree > 0:
                r_norm = max(self.r, 1.0)
                lr_base = lrs[3] if len(lrs) > 3 else 1e-4
                for i in range(self.ai_degree):
                    p_name = f"ai{2 * (i + 2)}"
                    p = getattr(self, p_name)
                    p.requires_grad_(True)
                    order = 2 * (i + 2)  # 4、6、8、10、……
                    lr_ai = lr_base / r_norm**order
                    params.append({"params": [p], "lr": lr_ai})

        # 优化材料参数
        if optim_mat and self.mat2.get_name() != "air":
            params += self.mat2.get_optimizer_params()

        return params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """将表面序列化为字典。

        非球面系数以 `[a4, a6, a8, ...]` 写入 `ai` 列表；若存在旧版 `ai2`
        系数，则将其前置，使 `ai[0] = a2`，并将 `use_ai2` 设为 True。

        返回：
            surf_dict (dict): 序列化表面（键包括 `type`、`r`、`roc`、`d`、
                `k`、`ai`、`use_ai2`、`mat2`，以及信息项
                `(c)`/`(ai*)`/`(mat2_n)`/`(mat2_V)`）。长度单位为 [mm]，
                `c` 的单位为 [1/mm]。
        """
        has_ai2 = self.ai2 is not None
        surf_dict = {
            "type": "Aspheric",
            "r": round(self.r, 4),
            "(c)": round(self.c.item(), 4),
            "roc": round(1 / self.c.item(), 4),
            "d": round(self.d.item(), 4),
            "k": round(self.k.item(), 4),
            "ai": [],
            "use_ai2": has_ai2,
            "mat2": self.mat2.get_name(),
            "(mat2_n)": round(float(self.mat2.n), 4),
            "(mat2_V)": round(float(self.mat2.V), 4),
        }

        # 若存在 a2，则将其前置到 ai 列表（ai2 键仅供参考；反序列化会在
        # use_ai2=True 时读取 ai[0]）
        if has_ai2:
            surf_dict["ai2"] = float(format(self.ai2.item(), ".6e"))
            surf_dict["ai"].append(float(format(self.ai2.item(), ".6e")))

        for i in range(self.ai_degree):
            order = i + 2
            coeff = getattr(self, f"ai{2 * order}")
            surf_dict[f"(ai{2 * order})"] = float(format(coeff.item(), ".6e"))
            surf_dict["ai"].append(float(format(coeff.item(), ".6e")))

        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """返回该表面的 Zemax（.zmx）文本块。

        输出 `EVENASPH` 表面。PARM 1 存放旧版二阶系数 `a2`，PARM 2 起存放
        `a4, a6, a8, ...`，并用零填充至 PARM 8（a16）。

        参数：
            surf_idx (int): Zemax 文件中的表面索引。
            d_next (torch.Tensor): 到下一表面的轴向距离 [mm]。

        返回：
            zmx_str (str): 多行 Zemax 表面描述。
        """
        assert self.c.item() != 0, (
            "Aperture surface is re-implemented in Aperture class."
        )
        assert self.ai is not None or self.k != 0, (
            "Spheric surface is re-implemented in Spheric class."
        )

        # 收集绝对 ai 值，PARM 1 = a2，PARM 2+ = a4、a6、……
        abs_ai = [self.ai2.item() if self.ai2 is not None else 0.0]
        for i in range(self.ai_degree):
            abs_ai.append(getattr(self, f"ai{2 * (i + 2)}").item())

        # 按 Zemax PARM 格式用零填充（a2–a16 需要 8 个 PARM）
        while len(abs_ai) < 8:
            abs_ai.append(0.0)

        if self.mat2.get_name() == "air":
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {self.c.item()}
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
    PARM 7 {abs_ai[6]}
    PARM 8 {abs_ai[7]}
"""
        else:
            zmx_str = f"""SURF {surf_idx}
    TYPE EVENASPH
    CURV {self.c.item()}
    DISZ {d_next.item()}
    GLAS ___BLANK 1 0 {self.mat2.n} {self.mat2.V}
    DIAM {self.r} 1 0 0 1 ""
    CONI {self.k}
    PARM 1 {abs_ai[0]}
    PARM 2 {abs_ai[1]}
    PARM 3 {abs_ai[2]}
    PARM 4 {abs_ai[3]}
    PARM 5 {abs_ai[4]}
    PARM 6 {abs_ai[5]}
    PARM 7 {abs_ai[6]}
    PARM 8 {abs_ai[7]}
"""
        return zmx_str
