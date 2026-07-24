"""孔径表面。"""

import numpy as np

from .plane import Plane


class Aperture(Plane):
    """孔径光阑表面。

    平坦的圆形（或方形）开口，用于阻挡落在通光孔径之外的光线。继承 `Plane`
    的平面求交逻辑，并且始终位于空气中（不发生折射）。

    属性：
        r (float): 孔径半径（通光半直径），单位为 [mm]。
        d (torch.Tensor): 沿光轴的轴向位置，单位为 [mm]。
        is_square (bool): 若为 True，则孔径为方形而非圆形。
        tolerancing (bool): 是否启用公差扰动。
    """

    def __init__(
        self,
        r,
        d,
        pos_xy=[0.0, 0.0],
        vec_local=[0.0, 0.0, 1.0],
        is_square=False,
        device="cpu",
    ):
        """初始化孔径表面。

        参数：
            r (float): 孔径半径（通光半直径），单位为 [mm]。
            d (float): 沿光轴的轴向位置，单位为 [mm]。
            pos_xy (list, optional): 表面的横向 (x, y) 偏移，单位为 [mm]。
                默认值为 [0.0, 0.0]。
            vec_local (list, optional): 局部表面法线（z 轴）方向。
                默认值为 [0.0, 0.0, 1.0]。
            is_square (bool, optional): 若为 True，则使用方形孔径。默认值为 False。
            device (str, optional): 表面张量使用的 Torch 设备。默认值为 "cpu"。
        """
        Plane.__init__(
            self,
            r=r,
            d=d,
            mat2="air",
            pos_xy=pos_xy,
            vec_local=vec_local,
            is_square=is_square,
            device=device,
        )
        self.tolerancing = False
        self.to(device)

    @classmethod
    def init_from_dict(cls, surf_dict):
        """从表面字典构造 Aperture。

        参数：
            surf_dict (dict): 表面参数。必须包含 "r" 和 "d"；可选键为
                "is_square"、"pos_xy"、"vec_local"、"device"。

        返回：
            aperture (Aperture): 构造得到的孔径表面。
        """
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            is_square=surf_dict["is_square"] if "is_square" in surf_dict else False,
            pos_xy=surf_dict["pos_xy"] if "pos_xy" in surf_dict else [0.0, 0.0],
            vec_local=surf_dict["vec_local"] if "vec_local" in surf_dict else [0.0, 0.0, 1.0],
            device=surf_dict["device"] if "device" in surf_dict else "cpu",
        )

    def ray_reaction(self, ray, n1=1.0, n2=1.0, refraction=False):
        """追迹光线通过孔径。

        将光线变换到局部坐标系，与孔径平面求交（通光孔径外的光线标记为无效），
        再变换回全局坐标系。孔径不发生折射，因此忽略 `n1`、`n2` 和 `refraction`。

        参数：
            ray (Ray): 全局坐标系中的输入光线批次。
            n1 (float, optional): 表面前的折射率（未使用）。默认值为 1.0。
            n2 (float, optional): 表面后的折射率（未使用）。默认值为 1.0。
            refraction (bool, optional): 对孔径忽略。默认值为 False。

        返回：
            ray (Ray): 求交后的全局坐标光线，并已更新 `is_valid`。
        """
        ray = self.to_local_coord(ray)
        ray = self.intersect(ray)
        ray = self.to_global_coord(ray)
        return ray

    # =======================================
    # 可视化
    # =======================================
    def draw_widget(self, ax, color="orange", linestyle="solid"):
        """在二维截面图中将孔径绘制为楔形标记。

        参数：
            ax (matplotlib.axes.Axes): 用于绘制的坐标轴（z-x 截面）。
            color (str, optional): 线条颜色。默认值为 "orange"。
            linestyle (str, optional): Matplotlib 线型。默认值为 "solid"。
        """
        d = self.d.item()
        aper_wedge_l = 0.05 * self.r  # [mm]
        aper_wedge_h = 0.15 * self.r  # [mm]

        # 平行边
        z = np.linspace(d - aper_wedge_l, d + aper_wedge_l, 3)
        x = -self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = self.r * np.ones(3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

        # 垂直边
        z = d * np.ones(3)
        x = np.linspace(self.r, self.r + aper_wedge_h, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)
        x = np.linspace(-self.r - aper_wedge_h, -self.r, 3)
        ax.plot(z, x, color=color, linestyle=linestyle, linewidth=0.8)

    def draw_widget3D(self, ax, color="black"):
        """在三维图中将孔径绘制为边缘圆。

        参数：
            ax (mpl_toolkits.mplot3d.axes3d.Axes3D): 用于绘制的三维坐标轴。
            color (str, optional): 线条颜色。默认值为 "black"。

        返回：
            line (list): `ax.plot` 返回的 Line3D 对象。
        """
        # 绘制边缘圆
        theta = np.linspace(0, 2 * np.pi, 100)
        edge_x = self.r * np.cos(theta)
        edge_y = self.r * np.sin(theta)
        edge_z = np.full_like(edge_x, self.d.item())  # 孔径位置处的 z 为常数

        # 绘制边缘圆
        line = ax.plot(edge_z, edge_x, edge_y, color=color, linewidth=1.5)

        return line

    def create_mesh(self, n_rings=32, n_arms=128, color=[0.0, 0.0, 0.0]):
        """为孔径创建三角剖分表面网格。

        构建顶点、面和边缘，并将它们存储在表面对象上。

        参数：
            n_rings (int, optional): 用于采样的同心环数量。默认值为 32。
            n_arms (int, optional): 角向分区数量。默认值为 128。
            color (list, optional): 网格的 RGB 颜色。默认值为 [0.0, 0.0, 0.0]。

        返回：
            self (Aperture): 已设置 `vertices`、`faces`、`rim` 和 `mesh_color`
                的孔径对象（用于链式调用）。
        """
        self.vertices = self._create_vertices(n_rings, n_arms)
        self.faces = self._create_faces(n_rings, n_arms)
        self.rim = self._create_rim(n_rings, n_arms)
        self.mesh_color = color
        return self

    def _create_vertices(self, n_rings, n_arms):
        """为孔径环带生成网格顶点。

        在孔径位置构建两个共面环：半径为 `r` 的内环，以及半径为
        `1.1 * r` [mm] 的外环。

        参数：
            n_rings (int): 环数（仅使用内环和外环）。
            n_arms (int): 每个环的角向分区数。

        返回：
            vertices (np.ndarray): shape 为 (n_rings * n_arms + 1, 3) 的
                Float32 数组，存储单位为 [mm] 的 (x, y, z) 坐标。
        """
        n_vertices = n_rings * n_arms + 1
        vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        aperture_z = self.d.item()  # 所有顶点均位于孔径位置
        inner_radius = self.r
        outer_radius = 1.1 * self.r

        # 生成内环顶点（前 n_arms 个顶点）
        for j_arm in range(n_arms):
            theta = 2 * np.pi * j_arm / n_arms
            x = inner_radius * np.cos(theta)
            y = inner_radius * np.sin(theta)
            z = aperture_z

            vertices[j_arm] = [x, y, z]

        # 生成外环顶点（第二组 n_arms 个顶点）
        for j_arm in range(n_arms):
            theta = 2 * np.pi * j_arm / n_arms
            x = outer_radius * np.cos(theta)
            y = outer_radius * np.sin(theta)
            z = aperture_z

            vertices[n_arms + j_arm] = [x, y, z]

        return vertices

    def _create_faces(self, n_rings, n_arms):
        """生成连接内环和外环的三角面。

        参数：
            n_rings (int): 环数（用于确定面数组的大小）。
            n_arms (int): 每个环的角向分区数。

        返回：
            faces (np.ndarray): shape 为 (n_arms * (2 * n_rings - 1), 3)
                的 Uint32 顶点索引数组。
        """
        n_faces = n_arms * (2 * n_rings - 1)
        faces = np.zeros((n_faces, 3), dtype=np.uint32)

        # 将内环（索引 0 到 n_arms-1）连接到外环（索引 n_arms 到 2*n_arms-1）
        for j_arm in range(n_arms):
            # 内环顶点
            inner_a = j_arm
            inner_b = (j_arm + 1) % n_arms

            # 外环顶点（偏移 n_arms）
            outer_a = n_arms + j_arm
            outer_b = n_arms + (j_arm + 1) % n_arms

            # 每个四边形创建两个三角形（法线方向为 +z）
            face_idx = j_arm * 2
            faces[face_idx] = [inner_a, outer_a, inner_b]
            faces[face_idx + 1] = [inner_b, outer_a, outer_b]

        return faces

    def _create_rim(self, n_rings, n_arms):
        """为孔径创建边缘（外缘）曲线。

        参数：
            n_rings (int): 环数（未使用，直接选择外环）。
            n_arms (int): 每个环的角向分区数。

        返回：
            rim (RimCurve): 由外环顶点构建的闭合边缘曲线。
        """
        # 从基础模块导入 RimCurve
        from .base import RimCurve

        # 获取外环顶点（顶点数组的后半部分）
        start_idx = n_arms  # 外环起始位置
        rim_vertices = self.vertices[start_idx : start_idx + n_arms]
        return RimCurve(rim_vertices, is_loop=True)

    # =========================================
    # 优化
    # =========================================
    def get_optimizer_params(self, lrs=[1e-4]):
        """启用轴向位置的梯度并构建优化器参数组。

        参数：
            lrs (list, optional): 学习率；`lrs[0]` 用于 `d`。默认值为 [1e-4]。

        返回：
            params (list): 包含一个 `d` 优化器参数组字典的列表。
        """
        self.d.requires_grad_(True)

        params = []
        params.append({"params": [self.d], "lr": lrs[0]})

        return params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self):
        """将孔径参数序列化为字典。

        返回：
            surf_dict (dict): 包含 "type"、"r"、"(d)"、"mat2" 和
                "is_square" 的表面参数。半径和位置单位为 [mm]。
        """
        surf_dict = {
            "type": "Aperture",
            "r": round(self.r, 4),
            "(d)": round(self.d.item(), 4),
            "mat2": "air",
            "is_square": self.is_square,
        }
        return surf_dict

    def zmx_str(self, surf_idx, d_next):
        """将孔径格式化为 Zemax（.zmx）STOP 表面块。

        参数：
            surf_idx (int): Zemax 文件中的表面索引。
            d_next (torch.Tensor): 到下一表面的距离，单位为 [mm]。

        返回：
            zmx_str (str): 此孔径的 Zemax 表面定义字符串。
        """
        zmx_str = f"""SURF {surf_idx}
    STOP
    TYPE STANDARD
    CURV 0.0
    DISZ {d_next.item()}
    DIAM {self.r} 1 0 0 1 ""
"""
        return zmx_str
