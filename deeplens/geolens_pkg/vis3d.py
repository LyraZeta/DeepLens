# Copyright 2026 KAUST Computational Imaging Group, Ziqing Zhao, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""几何镜头系统的 3D 可视化。

GeoLensVis3D 类：
    - create_mesh()：创建所有镜头、连接面、传感器和光阑网格
    - draw_lens_3d()：使用 PyVista 绘制包含光线的镜头 3D 布局
    - save_lens_obj()：将镜头几何结构和光线保存为 .obj 文件
"""

import os
from typing import List, Optional

import numpy as np
import torch

from ..light import Ray
from ..geometric_surface import Aperture


# ==========================================================
# 网格类
# （表面网格在对应的表面类中定义）
# ==========================================================
# PyVista 的本地占位类
class PolyData:
    """`pyvista.PolyData` 的轻量替代实现（不依赖 PyVista）。

    保存顶点数组，以及线连接数组（线网格）或三角形连接数组（面网格）中的一种，
    并可将自身保存为 Wavefront ``.obj`` 文件。`lines` / `faces` 中只能设置一个。
    所有坐标均以毫米 [mm] 为单位。

    属性：
        n_points (int)：顶点数量。
        points (np.ndarray)：顶点坐标，shape (n_points, 3) [mm]。
        lines (np.ndarray or None)：线连接关系，shape (n_lines, 2)，或 None。
        faces (np.ndarray or None)：三角形连接关系，shape (n_faces, 3)，或 None。
        is_linemesh (bool)：若保存的是线网格则为 True。
        is_facemesh (bool)：若保存的是面网格则为 True。
        is_default (bool)：若为空占位对象则为 True（参见 `default`）。
    """

    def __init__(self, vertices, lines, faces):
        """根据顶点以及线连接关系或面连接关系初始化 PolyData。

        参数：
            vertices (np.ndarray)：顶点坐标，shape (n_points, 3) [mm]。
            lines (np.ndarray or None)：线连接关系，shape (n_lines, 2)。面网格传入 None。
            faces (np.ndarray or None)：三角形连接关系，shape (n_faces, 3)。线网格传入 None。

        异常：
            AssertionError：同时提供 `lines` 和 `faces` 时抛出。
        """
        self.n_points = len(vertices)
        self.points = vertices
        self.lines = lines
        self.faces = faces
        self.is_linemesh = False
        self.is_facemesh = False
        self.is_default = False
        if lines is not None:
            self.is_linemesh = True
        if faces is not None:
            self.is_facemesh = True

        assert not (self.is_linemesh and self.is_facemesh), "Invalid polydata"

    def save(self, filename: str):
        """将网格保存为 Wavefront ``.obj`` 文件。

        顶点写为 ``v`` 行，连接关系写为 ``l``（线网格）或 ``f``（面网格）行，
        并将从 0 开始的索引转换为从 1 开始。仅支持输出 ``.obj``。

        参数：
            filename (str)：``.obj`` 文件的输出路径。
        """
        # pyvista.PolyData.save 方法的本地封装
        # 目前仅支持 .obj 格式

        with open(filename, "w") as f:
            mesh_head = "l" if self.is_linemesh else "f"
            v_head = "v"
            if self.is_linemesh:
                for v in self.points:
                    f.write(f"{v_head} {v[0]} {v[1]} {v[2]}\n")
                for l in self.lines:
                    f.write(f"{mesh_head} {l[0] + 1} {l[1] + 1}\n")
            if self.is_facemesh:
                for v in self.points:
                    f.write(f"{v_head} {v[0]} {v[1]} {v[2]}\n")
                for fm in self.faces:
                    f.write(f"{mesh_head} {fm[0] + 1} {fm[1] + 1} {fm[2] + 1}\n")

    # 为占位类实现默认方法
    @staticmethod
    def default():
        """返回 `is_default` 设为 True 的空占位 PolyData。

        用于类型检查和占位初始化。该实例不含顶点和连接关系。

        返回：
            obj (PolyData)：`is_default` 为 True 的空 PolyData。
        """
        obj = PolyData(np.zeros((0, 3)), lines=None, faces=None)
        obj.is_default = True
        return obj


def merge(meshes: List[PolyData]) -> PolyData:
    """将多个 PolyData 网格合并为一个，并偏移连接关系中的索引。

    所有网格必须为同一类型（全部为线网格或全部为面网格）；类型取自首个网格。
    拼接前，每个后续网格的顶点索引都会按当前累计顶点数进行偏移。

    参数：
        meshes (List[PolyData])：待合并的网格，可为空或 None。

    返回：
        merged (PolyData)：合并后的网格；若 `meshes` 为空/None，则返回默认的空 PolyData。
    """
    if meshes is None or len(meshes) == 0:
        return PolyData.default()
    if len(meshes) == 1:
        return meshes[0]
    v_count = meshes[0].n_points
    v_combined = meshes[0].points.copy()
    is_linemesh = meshes[0].is_linemesh
    mesh_combined = meshes[0].lines.copy() if is_linemesh else meshes[0].faces.copy()

    for m in meshes[1:]:
        # 按此前的 v_count 递增顶点编号
        if m.is_linemesh:
            v_combined = np.vstack([v_combined, m.points])
            new_lines = m.lines.copy()
            new_lines += v_count
            mesh_combined = np.vstack([mesh_combined, new_lines])
        elif m.is_facemesh:
            v_combined = np.vstack([v_combined, m.points])
            new_faces = m.faces.copy()
            new_faces += v_count
            mesh_combined = np.vstack([mesh_combined, new_faces])
        v_count += m.n_points
    return (
        PolyData(v_combined, lines=mesh_combined, faces=None)
        if is_linemesh
        else PolyData(v_combined, lines=None, faces=mesh_combined)
    )


class CrossPoly:
    """可生成 `PolyData` 的网格图元基类。

    子类（`LineMesh`、`FaceMesh` 及其变体）构建顶点/边或顶点/面数据，
    并重写 `get_polydata` 以将其公开。
    """

    def __init__(self):
        """初始化空的 CrossPoly 基类实例。"""
        pass

    def get_polydata(self) -> PolyData:
        """以 `PolyData` 形式返回网格。

        返回：
            poly (PolyData)：默认的空 PolyData（由子类重写）。
        """
        return PolyData.default()

    def get_obj_data(self):
        """用于导出原始 ``.obj`` 数据的占位钩子（在基类中不执行操作）。"""
        pass


class LineMesh(CrossPoly):
    """由有序 3D 顶点列表定义的折线网格。

    使用线段连接相邻顶点；若 `is_loop` 为 True，还会将末尾顶点连接回首个顶点。
    坐标单位为 [mm]。

    属性：
        n_vertices (int)：顶点数量。
        is_loop (bool)：折线是否闭合。
        vertices (np.ndarray)：顶点坐标，shape (n_vertices, 3) [mm]。
    """

    def __init__(self, n_vertices, is_loop=False):
        """使用全零顶点初始化线网格。

        参数：
            n_vertices (int)：顶点数量。
            is_loop (bool, optional)：折线是否闭合，默认为 False。
        """
        self.n_vertices = n_vertices
        self.is_loop = is_loop
        self.vertices = np.zeros((n_vertices, 3), dtype=np.float32)
        self.create_data()

    def create_data(self):
        """填充 `vertices`（在 `LineMesh` 基类中不执行操作，由子类重写）。"""
        pass

    def chain(self, other):
        """将另一线网格的顶点原地追加到当前网格。

        参数：
            other (LineMesh)：待追加的线网格，不得为闭环。

        异常：
            ValueError：当前网格或 `other` 为闭环时抛出。
        """
        if self.is_loop or other.is_loop:
            raise ValueError("One of the lines is a loop.")
        self.vertices = np.vstack([self.vertices, other.vertices])
        self.n_vertices = self.vertices.shape[0]
        return None

    def get_polydata(self):
        """构建线连接关系，并以 `PolyData` 形式返回网格。

        返回：
            poly (PolyData)：在线网格的相邻顶点之间包含线段；若 `is_loop` 为 True，
                还包含闭合线段。
        """
        n_line = 0 if self.is_loop else -1
        n_line += self.n_vertices
        line = np.array(
            [[i, (i + 1) % self.n_vertices] for i in range(n_line)], dtype=np.uint32
        )
        return PolyData(self.vertices, lines=line, faces=None)


class Curve(LineMesh):
    """直接由顶点数组构建的开放/闭合折线。

    当前用于表示追迹光线路径。坐标单位为 [mm]。
    """

    def __init__(self, vertices: np.ndarray, is_loop: Optional[bool] = None):
        """根据显式顶点数组初始化曲线。

        参数：
            vertices (np.ndarray)：顶点坐标，shape (n_vertices, 3) [mm]。
            is_loop (bool, optional)：曲线是否闭合，默认为 False。
        """
        if is_loop is None:
            is_loop = False
        n_vertices = vertices.shape[0]
        super().__init__(n_vertices, is_loop)
        self.vertices = vertices


class Circle(LineMesh):
    """由圆心、法向量和半径定义的闭合圆形折线。

    圆位于垂直于 `direction`（其法向量，遵循右手定则）的平面内，圆心为 `origin`。
    坐标单位为 [mm]。当前未使用。

    属性：
        origin (np.ndarray)：圆心，shape (3,) [mm]。
        direction (np.ndarray)：平面法向方向，shape (3,)。
        radius (float)：圆半径 [mm]。
    """

    def __init__(self, n_vertices, origin, direction, radius):
        """初始化圆形网格。

        参数：
            n_vertices (int)：沿圆周采样的点数。
            origin (np.ndarray)：圆心，shape (3,) [mm]。
            direction (np.ndarray)：平面法向方向，shape (3,)。
            radius (float)：圆半径 [mm]。
        """
        self.direction = direction
        self.radius = radius
        self.origin = origin
        super().__init__(n_vertices, is_loop=True)

    def create_data(self):
        """沿圆周均匀采样 `n_vertices` 个点并写入 `vertices`。"""
        # 归一化方向向量
        direction = np.array(self.direction, dtype=np.float32)
        direction = direction / np.linalg.norm(direction)

        # 寻找一个不与该方向平行的向量
        if np.abs(direction[0]) < 0.9:
            v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            v1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        # 使用叉积得到垂直向量
        u = np.cross(direction, v1)
        u = u / np.linalg.norm(u)
        v = np.cross(direction, u)
        v = v / np.linalg.norm(v)

        # 生成圆周上的点
        origin = np.array(self.origin, dtype=np.float32)
        for i in range(self.n_vertices):
            angle = 2 * np.pi * i / self.n_vertices
            x = self.radius * (u[0] * np.cos(angle) + v[0] * np.sin(angle))
            y = self.radius * (u[1] * np.cos(angle) + v[1] * np.sin(angle))
            z = self.radius * (u[2] * np.cos(angle) + v[2] * np.sin(angle))
            self.vertices[i] = origin + np.array([x, y, z])


class FaceMesh(CrossPoly):
    """由顶点和三角形面定义的三角化表面网格。

    用于镜片元件连接面以及传感器/表面网格。可选地携带一条沿边界追踪的 `rim`
    折线。坐标单位为 [mm]。

    属性：
        n_vertices (int)：顶点数量。
        n_faces (int)：三角形面数量。
        vertices (np.ndarray)：顶点坐标，shape (n_vertices, 3) [mm]。
        faces (np.ndarray)：三角形顶点索引，shape (n_faces, 3)。
        rim (LineMesh)：网格的边界折线，或 None。
    """

    def __init__(self, n_vertices: int, n_faces: int):
        """使用全零顶点和面初始化面网格。

        参数：
            n_vertices (int)：顶点数量。
            n_faces (int)：三角形面数量。
        """
        self.n_vertices = n_vertices
        self.n_faces = n_faces
        self.vertices, self.faces = self._create_empty_data()
        self.rim: LineMesh = None  # type: ignore
        self.create_data()
        self.create_rim()

    def _create_empty_data(self):
        """分配全零的顶点数组和面数组。

        返回：
            vertices (np.ndarray)：全零顶点数组，shape (n_vertices, 3)。
            faces (np.ndarray)：全零面索引数组，shape (n_faces, 3)。
        """
        vertices = np.zeros((self.n_vertices, 3), dtype=np.float32)
        faces = np.zeros((self.n_faces, 3), dtype=np.uint32)
        return vertices, faces

    def create_data(self):
        """填充 `vertices` 和 `faces`（在基类中不执行操作，由子类重写）。"""
        pass

    def create_rim(self):
        """填充边界折线 `rim`（在基类中不执行操作，由子类重写）。"""
        pass

    def get_mesh(self):
        """以 `PolyData` 形式返回网格（`get_polydata` 的别名）。

        返回：
            poly (PolyData)：面网格 PolyData。
        """
        return self.get_polydata()

    def get_polydata(self) -> PolyData:
        """以面 `PolyData` 形式返回网格。

        返回：
            poly (PolyData)：由 `vertices` 和 `faces` 构建的面网格 PolyData。
        """
        return PolyData(self.vertices, lines=None, faces=self.faces)


class RectangleMesh(FaceMesh):
    """用于传感器平面的平面矩形网格（两个三角形）。

    矩形以 `center` 为中心，由两个正交单位方向张成；`width` 沿 `direction_w`
    测量，`height` 沿 `direction_h` 测量。坐标单位为 [mm]。

    属性：
        center (np.ndarray)：矩形中心，shape (3,) [mm]。
        direction_w (np.ndarray)：宽度单位方向，shape (3,)。
        direction_h (np.ndarray)：高度单位方向，shape (3,)。
        width (float)：沿 `direction_w` 的范围 [mm]。
        height (float)：沿 `direction_h` 的范围 [mm]。
    """

    def __init__(
        self,
        center: np.ndarray,
        direction_w: np.ndarray,
        direction_h: np.ndarray,
        width: float,
        height: float,
    ):
        """初始化矩形网格。

        参数：
            center (np.ndarray)：矩形中心，shape (3,) [mm]。
            direction_w (np.ndarray)：宽度方向，shape (3,)（内部归一化）。
            direction_h (np.ndarray)：高度方向，shape (3,)（内部归一化）。
            width (float)：沿 `direction_w` 的范围 [mm]。
            height (float)：沿 `direction_h` 的范围 [mm]。

        异常：
            AssertionError：两个方向不正交，或 `width`/`height` 不为正数时抛出。
        """
        # 两个方向应当正交
        assert np.dot(direction_w, direction_h) == 0, "Invalid directions"
        # width 和 height 应当为正数
        assert width > 0 and height > 0, "Invalid width or height"

        self.center = center
        self.direction_w = direction_w / np.linalg.norm(direction_w)
        self.direction_h = direction_h / np.linalg.norm(direction_h)
        self.width = width
        self.height = height
        super().__init__(n_vertices=4, n_faces=2)

    def create_data(self):
        """计算四个角点顶点和两个三角形面。"""
        self.vertices[0] = (
            self.center
            - 0.5 * self.width * self.direction_w
            - 0.5 * self.height * self.direction_h
        )
        self.vertices[1] = (
            self.center
            + 0.5 * self.width * self.direction_w
            - 0.5 * self.height * self.direction_h
        )
        self.vertices[2] = (
            self.center
            + 0.5 * self.width * self.direction_w
            + 0.5 * self.height * self.direction_h
        )
        self.vertices[3] = (
            self.center
            - 0.5 * self.width * self.direction_w
            + 0.5 * self.height * self.direction_h
        )

        self.faces[0] = [0, 1, 2]
        self.faces[1] = [0, 2, 3]


# ====================================================
# 网格工具函数
# ====================================================


def bridge(
    l_a: LineMesh,
    l_b: LineMesh,
) -> FaceMesh:
    """使用三角化面带连接两条折线。

    两条线必须具有相同的顶点数，且均为闭环或均为开放曲线。首先将 `l_b` 的顶点
    与 `l_a` 重新对齐（闭环按最近顶点滚动，开放曲线方向相反时则翻转），随后在
    对应顶点之间生成三角形带。

    参数：
        l_a (LineMesh)：第一条折线。
        l_b (LineMesh)：第二条折线，其顶点数和闭环标志须与 `l_a` 相同。

    返回：
        face_mesh (FaceMesh)：连接两条线的三角化连接面。

    异常：
        ValueError：仅一条线为闭环，或顶点数量不同时抛出。
    """
    # 检查两条线是否均为闭环或均为开放曲线
    if l_a.is_loop ^ l_b.is_loop:
        raise ValueError("Both lines must be either loops or open curves.")

    # 检查两条线是否具有相同的顶点数
    if l_a.n_vertices != l_b.n_vertices:
        raise ValueError("Both lines must have the same number of vertices.")

    n = l_a.n_vertices

    # 将 l_b 的顶点与 l_a 对齐
    if l_a.is_loop:
        # 找到 l_b 中距离 l_a 首个顶点最近的顶点
        distances = np.linalg.norm(l_b.vertices - l_a.vertices[0], axis=1)
        closest_idx = np.argmin(distances)
        # 重排 l_b 的顶点，使其从最近索引开始
        reordered_b = np.roll(l_b.vertices, shift=-closest_idx, axis=0)
    else:
        # 检查 l_b 的起点或终点哪一个更接近 l_a 的起点
        dist_start = np.linalg.norm(l_b.vertices[0] - l_a.vertices[0])
        dist_end = np.linalg.norm(l_b.vertices[-1] - l_a.vertices[0])
        # 若终点更近，则翻转 l_b 的顶点顺序
        if dist_end < dist_start:
            reordered_b = l_b.vertices[::-1]
        else:
            reordered_b = l_b.vertices.copy()

    # 合并 l_a 与重排后的 l_b 的顶点
    vertices = np.vstack([l_a.vertices, reordered_b])

    # 生成面
    faces = []
    if l_a.is_loop:
        for i in range(n):
            j = (i + 1) % n
            a_i = i
            a_j = j
            b_i = i + n
            b_j = j + n
            faces.append([a_i, a_j, b_i])
            faces.append([a_j, b_j, b_i])
    else:
        for i in range(n - 1):
            j = i + 1
            a_i = i
            a_j = j
            b_i = i + n
            b_j = j + n
            faces.append([a_i, a_j, b_i])
            faces.append([a_j, b_j, b_i])

    faces = np.array(faces, dtype=np.uint32)

    # 创建 FaceMesh 实例
    face_mesh = FaceMesh(n_vertices=vertices.shape[0], n_faces=faces.shape[0])
    face_mesh.vertices = vertices
    face_mesh.faces = faces

    return face_mesh


def line_translate(l: LineMesh, dx: float, dy: float, dz: float) -> LineMesh:
    """按固定偏移量平移线网格，并返回新线网格。

    参数：
        l (LineMesh)：待平移的线网格。
        dx (float)：沿 x 方向的平移量 [mm]。
        dy (float)：沿 y 方向的平移量 [mm]。
        dz (float)：沿 z 方向的平移量 [mm]。

    返回：
        new_l (LineMesh)：顶点已平移的新线网格。
    """
    # 创建新线网格
    new_l = LineMesh(l.n_vertices, l.is_loop)
    new_l.vertices = l.vertices.copy()
    new_l.vertices = new_l.vertices + np.array([dx, dy, dz])[None, :]
    return new_l


def surf_to_face_mesh(surf) -> FaceMesh:
    """将 `Surface` 网格转换为 `FaceMesh`。

    将表面预先计算的 `vertices` 和 `faces` 复制到新的 FaceMesh 中。

    参数：
        surf (Surface)：已创建网格的表面（必须公开 `vertices` 和 `faces`）。

    返回：
        face_mesh (FaceMesh)：封装表面几何结构的面网格。
    """
    n_vertices = surf.vertices.shape[0]
    n_faces = surf.faces.shape[0]
    face_mesh = FaceMesh(n_vertices=n_vertices, n_faces=n_faces)
    face_mesh.vertices = surf.vertices
    face_mesh.faces = surf.faces
    return face_mesh


# ====================================================
# 光线可视化
# ====================================================


def curve_list_to_polydata(meshes: List[Curve]) -> List[PolyData]:
    """将 `Curve` 对象列表转换为 `PolyData` 对象列表。

    参数：
        meshes (List[Curve])：待转换的曲线。

    返回：
        polys (List[PolyData])：每条输入曲线对应一个线网格 PolyData。
    """
    return [c.get_polydata() for c in meshes]


def geolens_ray_poly(
    lens,
    fovs: List[float],
    fov_phis: List[float],
    n_rings: int = 3,
    n_arms: int = 4,
) -> List[List[Curve]]:
    """采样并追迹平行光线束，用于绘制镜头布局。

    对 `fovs` 中的每个视场角（视场角非零时，还会遍历 `fov_phis` 中的每个方位角），
    采样 Zemax 风格的环形辐条光瞳图案，并使其穿过镜头进行追迹。零视场角视为单个
    轴上光束（忽略方位角）。

    参数：
        lens (GeoLens)：镜头对象。
        fovs (List[float])：待采样的视场（极）角 [degree]。
        fov_phis (List[float])：待采样的视场方位角 [degree]。
        n_rings (int, optional)：采样的光瞳环数，默认为 3。
        n_arms (int, optional)：采样的光瞳辐条数，默认为 4。

    返回：
        rays_poly (List[List[Curve]])：每个已追迹视场光束对应一项；每项均为该光束的
            `Curve` 光线路径列表。
    """
    rays_poly = []

    R = lens.surfaces[0].r

    for fov in fovs:
        if fov == 0.0:
            center_ray = sample_parallel_3D(lens, R, rings=n_rings, arms=n_arms)
            rays_poly.append(curve_from_trace(lens, center_ray))
        else:
            for fov_phi in fov_phis:
                print(f"fov: {fov}, fov_phi: {fov_phi}")
                # 在该视场上采样光线
                ray = sample_parallel_3D(
                    lens,
                    R,
                    rings=n_rings,
                    arms=n_arms,
                    view_polar=fov,
                    view_azi=fov_phi,
                )
                rays_poly.append(curve_from_trace(lens, ray))
    return rays_poly


def sample_parallel_3D(
    lens,
    R: float,
    wvln=None,
    z=None,
    view_polar: float = 0.0,
    view_azi: float = 0.0,
    rings: int = 3,
    arms: int = 4,
    forward: bool = True,
    entrance_pupil=True,
):
    """在环形辐条光瞳图案上采样平行光线束。

    光线原点位于入瞳（或首个表面）上，且均采用由 `view_polar` / `view_azi` 设置的
    方向。光束包含 $M = rings \\times arms + 1$ 条光线（额外的一条为轴上中心光线）。
    用于绘制镜头设置以及近轴计算（例如重新聚焦到无穷远）。

    参数：
        lens (GeoLens)：镜头对象。
        R (float)：光瞳采样半径 [mm]。当前未使用；采样半径取自入瞳或首表面半径。
        wvln (float, optional)：光线波长 [µm]。为 None 时回退到
            `lens.primary_wvln`，默认为 None。
        z (float, optional)：未使用；采样深度取自光瞳，默认为 None。
        view_polar (float, optional)：入射极角 [degree]，默认为 0.0。
        view_azi (float, optional)：入射方位角 [degree]，默认为 0.0。
        rings (int, optional)：光瞳环数，默认为 3。
        arms (int, optional)：光瞳辐条数，默认为 4。
        forward (bool, optional)：当前未使用，默认为 True。
        entrance_pupil (bool, optional)：为 True 时在计算得到的入瞳上采样；否则在
            首个表面上采样。默认为 True。

    返回：
        ray (Ray)：采样得到的光线束，原点 `o` 和方向 `d` 的 shape (M, 3) [mm]。
    """
    wvln = lens.primary_wvln if wvln is None else wvln
    if entrance_pupil:
        # 在光瞳上采样第二组点
        pupilz, pupilx = lens.calc_entrance_pupil()
    else:
        pupilz, pupilx = 0, lens.surfaces[0].r

    # x2 = torch.linspace(-pupilx, pupilx, M) * 0.99
    rho2 = torch.linspace(0, pupilx, rings + 1) * 0.99
    rho2 = rho2[1:]  # 移除中心点
    phi2 = torch.linspace(0, 2 * np.pi, arms + 1)
    phi2 = phi2[:-1]
    RHO2, PHI2 = torch.meshgrid(rho2, phi2, indexing="ij")
    X2, Y2 = RHO2 * torch.cos(PHI2), RHO2 * torch.sin(PHI2)
    x2, y2 = torch.flatten(X2), torch.flatten(Y2)

    # 重新加入中心点
    x2 = torch.concat((torch.tensor([0]), x2))
    y2 = torch.concat((torch.tensor([0]), y2))

    z2 = torch.full_like(x2, pupilz)
    o2 = torch.stack((x2, y2, z2), dim=-1)  # shape [M, 3]

    view_polar = view_polar / 57.3
    view_azi = view_azi / 57.3
    dx = torch.full_like(x2, np.sin(view_polar) * np.cos(view_azi))
    dy = torch.full_like(x2, np.sin(view_polar) * np.sin(view_azi))
    dz = torch.full_like(x2, np.cos(view_polar))
    d = torch.stack((dx, dy, dz), dim=-1)

    # 将光线原点移动到 z = -0.1 处以进行追迹
    if pupilz > 0:
        o = o2 - d * ((z2 + 0.1) / dz).unsqueeze(-1)
    else:
        o = o2

    return Ray(o, d, wvln, device=lens.device)


def curve_from_trace(lens, ray: Ray, delete_vignetting=True):
    """将光线束追迹至传感器，并返回每条光线的路径曲线。

    记录 `ray` 穿过镜头时的路径，堆叠各表面的交点（shape (n_surf, M, 3) [mm]），
    并将每条光线的路径转换为 `Curve`。

    参数：
        lens (GeoLens)：镜头对象。
        ray (Ray)：待追迹的已采样光线束。
        delete_vignetting (bool, optional)：原本用于丢弃渐晕光线；当前不执行操作，
            因而会保留渐晕光线（坐标为 NaN）。默认为 True。

    返回：
        rays_curve (List[Curve])：每条光线对应一个 `Curve`，记录其穿过各表面到达
            传感器的路径。
    """
    ray, ray_o_records = lens.trace2sensor(ray=ray, record=True)
    rays_curve = []
    # ray_o_records 的 shape 是否为 [n_surf, M, 3]？
    ray_o_records = torch.stack(ray_o_records, dim=0)
    ray_o_records = ray_o_records.permute(1, 0, 2).cpu().numpy()
    if delete_vignetting:
        # 应如何处理渐晕光线？
        # 当前所有带 "nan" 的光线都会传给 poly
        # 此问题需要修复
        pass
    for record in ray_o_records:
        curve = Curve(record, False)
        rays_curve.append(curve)
    return rays_curve


# ====================================================
# PyVista GUI 辅助函数（延迟加载）
# ====================================================


def _wrap_base_poly_to_pyvista(poly: PolyData, pv):
    """将本地 `PolyData` 封装为 `pyvista.PolyData`。

    在连接关系数组前添加 PyVista 要求的每单元顶点数（线段为 2，三角形为 3）。

    参数：
        poly (PolyData)：待封装的本地网格。
        pv (module)：已导入的 `pyvista` 模块（通过参数传入以避免顶层导入）。

    返回：
        pv_poly (pyvista.PolyData)：封装后的 PyVista 网格（若 `poly` 为默认占位对象，
            则为空）。
    """
    if poly.is_default:
        return pv.PolyData()
    else:
        p = poly.points
        m = poly.lines if poly.is_linemesh else poly.faces
        if poly.is_linemesh:
            _add_on = np.ones((m.shape[0], 1), dtype=np.int64)
            _add_on = 2 * _add_on
            new_m = np.hstack([_add_on, m])
        else:
            _add_on = np.ones((m.shape[0], 1), dtype=np.int64)
            _add_on = 3 * _add_on
            new_m = np.hstack([_add_on, m])
        return (
            pv.PolyData(p, lines=new_m)
            if poly.is_linemesh
            else pv.PolyData(p, faces=new_m)
        )


def _draw_mesh_to_plotter(
    plotter, mesh: CrossPoly, color: List[float], opacity: float, pv
):
    """向 PyVista 绘图器添加单个网格。

    参数：
        plotter (pyvista.Plotter)：要在其中绘制的绘图器。
        mesh (CrossPoly)：待绘制的网格图元。
        color (List[float])：RGB 颜色，每个分量位于 [0, 1]。
        opacity (float)：网格不透明度，范围为 [0, 1]。
        pv (module)：已导入的 `pyvista` 模块（通过参数传入以避免顶层导入）。
    """
    poly = _wrap_base_poly_to_pyvista(mesh.get_polydata(), pv)
    plotter.add_mesh(poly, color=color, opacity=opacity)


# ====================================================
# 网格可视化
# ====================================================


class GeoLensVis3D:
    """为 `GeoLens` 提供 3D 网格可视化的混入类。

    将镜头表面、光阑、挡板、传感器和光线路径创建为多边形网格数据，并可选择使用
    PyVista 渲染。所有几何量均以毫米 [mm] 表示，并存储为可保存至 ``.obj`` 文件供
    外部渲染器使用的 `CrossPoly`（顶点/面）对象。

    此类不直接实例化，而是混入 `GeoLens`。
    """

    # # 混入 GeoLens 时用于满足类型检查器的属性存根
    # surfaces: List[Any]
    # d_sensor: Any
    # r_sensor: float

    def create_mesh(
        self,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        is_wrap: bool = False,
    ):
        """为整个镜头构建表面、连接面和传感器网格。

        将表面分组为光学元件（在与空气相邻的表面处分隔）。元件内的相邻表面由连接
        面带相连；启用 `is_wrap` 时，会投影连接面，在半径不同的元件之间形成圆柱形
        镜筒。

        参数：
            mesh_rings (int, optional)：每个表面网格的环数，默认为 32。
            mesh_arms (int, optional)：每个表面网格的辐条数，默认为 128。
            is_wrap (bool, optional)：是否用圆柱形镜筒包裹镜片元件，默认为 False。

        返回：
            surf_meshes_cvt (List[FaceMesh])：各表面的面网格。
            bridge_meshes (List[List[FaceMesh]])：各元件的连接面网格列表（单表面元件
                对应空列表）。
            element_groups (List[List[int]])：表面索引分组，每个光学元件对应一组。
            sensor_mesh (RectangleMesh)：矩形传感器网格。
        """
        surf_meshes = []
        element_group = []
        element_groups = []
        bridge_meshes = []  # 改为用于环绕结构的嵌套列表
        sensor_mesh = None

        # 创建表面网格
        for i, surf in enumerate(self.surfaces):
            # 创建表面网格（Surface 对象列表）
            surf_meshes.append(surf.create_mesh(n_rings=mesh_rings, n_arms=mesh_arms))

            # 将表面加入元件分组
            element_group.append(i)
            if surf.mat2.name == "air":
                element_groups.append(element_group)
                element_group = []

        # 创建连接面网格（FaceMesh 对象列表）
        for i, pair in enumerate(element_groups):
            if len(pair) == 1:
                bridge_meshes.append([])
                continue
            elif len(pair) == 2:
                a_idx, b_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                bridge_mesh_group = []
                if not is_wrap:
                    bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                else:
                    # 创建新 rim 以形成环绕结构
                    # 将较大的 rim 投影到较小 rim 所在平面
                    # 假设元件始终沿 z 轴排序
                    r_a = self.surfaces[a_idx].r
                    r_b = self.surfaces[b_idx].r
                    d_rim_a = np.mean(
                        a.rim.vertices[:, 2], keepdims=False
                    )  # 计算 rim 的平均 z 坐标
                    d_rim_b = np.mean(b.rim.vertices[:, 2], keepdims=False)

                    if r_a > r_b:
                        z = line_translate(a.rim, 0, 0, d_rim_b - d_rim_a)
                        bridge_mesh_wrap = bridge(z, b.rim)
                        bridge_mesh = bridge(a.rim, z)
                        bridge_mesh_group.append(bridge_mesh_wrap)
                    elif r_a < r_b:
                        z = line_translate(b.rim, 0, 0, d_rim_a - d_rim_b)
                        bridge_mesh_wrap = bridge(a.rim, z)
                        bridge_mesh = bridge(z, b.rim)
                        bridge_mesh_group.append(bridge_mesh_wrap)
                    else:
                        bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                bridge_meshes.append(bridge_mesh_group)

            elif len(pair) == 3:
                a_idx, b_idx, c_idx = pair
                a = surf_meshes[a_idx]
                b = surf_meshes[b_idx]
                c = surf_meshes[c_idx]
                bridge_mesh_group = []
                if not is_wrap:
                    bridge_mesh = bridge(a.rim, b.rim)
                    bridge_mesh_group.append(bridge_mesh)
                    bridge_mesh = bridge(b.rim, c.rim)
                    bridge_mesh_group.append(bridge_mesh)
                else:
                    # 创建新 rim 以形成环绕结构
                    # 将较大的 rim 投影到较小 rim 所在平面
                    # 假设元件始终沿 z 轴排序
                    r_a = self.surfaces[a_idx].r
                    r_b = self.surfaces[b_idx].r
                    r_c = self.surfaces[c_idx].r
                    d_rim_a = np.mean(
                        a.rim.vertices[:, 2], keepdims=False
                    )  # 计算 rim 的平均 z 坐标
                    d_rim_b = np.mean(b.rim.vertices[:, 2], keepdims=False)
                    d_rim_c = np.mean(c.rim.vertices[:, 2], keepdims=False)

                    rim_list = [a.rim, b.rim, c.rim]
                    r_list = [r_a, r_b, r_c]
                    d_rim_list = [d_rim_a, d_rim_b, d_rim_c]
                    idx_wrap = r_list.index(max(r_list))
                    r_wrap = r_list[idx_wrap]
                    d_rim_wrap = d_rim_list[idx_wrap]

                    for i in range(3):
                        if i != idx_wrap and r_list[i] != r_wrap:
                            # 使用环绕后的 rim 替换原 rim
                            d_diff = d_rim_list[i] - d_rim_wrap
                            z = line_translate(rim_list[idx_wrap], 0, 0, d_diff)
                            # 在原 rim 与环绕后的 rim 之间添加环绕连接面
                            wrap_mesh = bridge(rim_list[i], z)
                            # 更新 rim
                            rim_list[i] = z
                            bridge_mesh_group.append(wrap_mesh)
                    bridge_mesh = bridge(rim_list[0], rim_list[1])
                    bridge_mesh_group.append(bridge_mesh)
                    bridge_mesh = bridge(rim_list[1], rim_list[2])
                    bridge_mesh_group.append(bridge_mesh)
                bridge_meshes.append(bridge_mesh_group)

            else:
                raise ValueError(f"Invalid bridge group length: {len(pair)}")

        # 创建传感器网格（RectangleMesh 对象）
        sensor_d = self.d_sensor.item()
        sensor_r = self.r_sensor
        h, w = sensor_r * 1.4142, sensor_r * 1.4142
        sensor_mesh = RectangleMesh(
            np.array([0, 0, sensor_d]), np.array([1, 0, 0]), np.array([0, 1, 0]), w, h
        )

        # 将 surf_meshes 转换为 FaceMesh 列表
        surf_meshes_cvt = [surf_to_face_mesh(surf) for surf in surf_meshes]
        return surf_meshes_cvt, bridge_meshes, element_groups, sensor_mesh

    def draw_lens_3d(
        self,
        plotter=None,
        save_dir: Optional[str] = None,
        mesh_rings: int = 32,
        mesh_arms: int = 128,
        surface_color: List[float] = [0.06, 0.3, 0.6],
        draw_rays: bool = True,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        is_wrap: bool = False,
    ):
        """使用 PyVista 渲染 3D 镜头布局（表面、传感器及可选光线）。

        参数：
            plotter (pyvista.Plotter, optional)：用于绘制的现有绘图器。为 None 时创建
                新绘图器，默认为 None。
            save_dir (str, optional)：保存渲染截图 ``lens_layout3d.png`` 的目录。为 None
                时不保存图像，默认为 None。
            mesh_rings (int, optional)：每个表面网格的环数，默认为 32。
            mesh_arms (int, optional)：每个表面网格的辐条数，默认为 128。
            surface_color (List[float], optional)：表面的 RGB 颜色，各分量位于 [0, 1]，
                默认为 [0.06, 0.3, 0.6]。
            draw_rays (bool, optional)：是否追迹并绘制光线，默认为 True。
            fovs (List[float], optional)：待采样的视场角 [degree]，默认为 [0.0]。
            fov_phis (List[float], optional)：待采样的视场方位角 [degree]，默认为 [0.0]。
            ray_rings (int, optional)：待采样的光瞳环数，默认为 6。
            ray_arms (int, optional)：待采样的光瞳辐条数，默认为 8。
            is_wrap (bool, optional)：是否将镜筒包裹为圆柱形，默认为 False。

        返回：
            plotter (pyvista.Plotter)：已添加所有网格的绘图器。

        异常：
            ImportError：未安装 PyVista 时抛出（在此处延迟导入）。

        注意：
            仅在调用此方法时才延迟导入 PyVista。
        """
        # 延迟导入 PyVista
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "PyVista is required for 3D GUI rendering. Install with `pip install pyvista`."
            ) from e

        # 若未提供绘图器，则创建一个
        if plotter is None:
            plotter = pv.Plotter()

        surf_color = surface_color
        sensor_color = [0.5, 0.5, 0.5]

        # 创建网格
        surf_meshes, bridge_meshes, _, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms, is_wrap
        )

        # 绘制网格
        for surf in surf_meshes:
            if not isinstance(surf, Aperture):
                _draw_mesh_to_plotter(
                    plotter, surf, color=surf_color, opacity=0.5, pv=pv
                )

        for bridge_group in bridge_meshes:
            for bridge_mesh in bridge_group:
                _draw_mesh_to_plotter(
                    plotter, bridge_mesh, color=surf_color, opacity=0.5, pv=pv
                )

        _draw_mesh_to_plotter(
            plotter, sensor_mesh, color=sensor_color, opacity=1.0, pv=pv
        )

        # 绘制光线
        if draw_rays:
            rays_curve = geolens_ray_poly(
                self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms
            )

            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            rays_poly_fov = [_wrap_base_poly_to_pyvista(r, pv) for r in rays_poly_fov]
            for r in rays_poly_fov:
                plotter.add_mesh(r)

        # 保存图像
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            plotter.show(screenshot=os.path.join(save_dir, "lens_layout3d.png"))

        return plotter

    def save_lens_obj(
        self,
        save_dir: str,
        mesh_rings: int = 64,
        mesh_arms: int = 128,
        save_rays: bool = False,
        fovs: List[float] = [0.0],
        fov_phis: List[float] = [0.0],
        ray_rings: int = 6,
        ray_arms: int = 8,
        is_wrap: bool = False,
        save_elements: bool = True,
    ):
        """将镜头几何结构、传感器和可选光线保存为 Wavefront ``.obj`` 文件。

        写入 ``lens.obj``（合并所有表面和连接面，不含光阑）与 ``sensor.obj``。
        `save_elements` 为 True 时，还会为每个光学元件写入一个 ``element_{i}.obj``；
        `save_rays` 为 True 时，会为每个已追迹视场光束写入一个
        ``lens_rays_fov_{i}.obj``。

        参数：
            save_dir (str)：写入 ``.obj`` 文件的目录。
            mesh_rings (int, optional)：每个表面网格的环数，默认为 64。
            mesh_arms (int, optional)：每个表面网格的辐条数，默认为 128。
            save_rays (bool, optional)：是否追迹并保存光线，默认为 False。
            fovs (List[float], optional)：待采样的视场角 [degree]，默认为 [0.0]。
            fov_phis (List[float], optional)：待采样的视场方位角 [degree]，默认为 [0.0]。
            ray_rings (int, optional)：待采样的光瞳环数，默认为 6。
            ray_arms (int, optional)：待采样的光瞳辐条数，默认为 8。
            is_wrap (bool, optional)：是否将镜筒包裹为圆柱形，默认为 False。
            save_elements (bool, optional)：是否额外保存各元件的 ``.obj`` 文件，
                默认为 True。

        注意：
            在 Blender 中渲染时，请使用 #F2F7FFFF 作为镜头颜色。此方法直接写入
            ``.obj`` 文件，不需要 PyVista。
        """
        os.makedirs(save_dir, exist_ok=True)

        # 创建表面和连接面网格
        surf_meshes, bridge_meshes, element_groups, sensor_mesh = self.create_mesh(
            mesh_rings, mesh_arms, is_wrap
        )

        # 保存各镜片元件（合并表面和连接面）
        if save_elements:
            for i, pair in enumerate(element_groups):
                print(f"Running in pair {i} with pair length {len(pair)}")
                # 收集表面 PolyData
                surf_polydata_list = [surf_meshes[idx].get_polydata() for idx in pair]

                # 收集可用的连接面 PolyData
                bridge_polydata_list = []
                if i < len(bridge_meshes) and len(bridge_meshes[i]) > 0:
                    print(f"Bridge mesh group number: {len(bridge_meshes[i])}")
                    bridge_polydata_list = [b.get_polydata() for b in bridge_meshes[i]]

                # 合并表面与连接面
                all_polydata = surf_polydata_list + bridge_polydata_list
                if len(all_polydata) == 1:
                    element = all_polydata[0]
                else:
                    element = merge(all_polydata)
                element.save(os.path.join(save_dir, f"element_{i}.obj"))

        # 合并所有表面和连接面，并保存为单个 lens.obj 文件
        surf_polydata = [
            surf.get_polydata()
            for surf in surf_meshes
            if not isinstance(surf, Aperture)
        ]
        bridge_polydata = [
            b.get_polydata() for group in bridge_meshes for b in group
        ]  # 展平嵌套列表
        lens_polydata = surf_polydata + bridge_polydata
        lens_polydata = merge(lens_polydata)
        lens_polydata.save(os.path.join(save_dir, "lens.obj"))

        # 保存传感器
        sensor_polydata = sensor_mesh.get_polydata()
        sensor_polydata.save(os.path.join(save_dir, "sensor.obj"))

        # 保存光线
        if save_rays:
            rays_curve = geolens_ray_poly(
                self, fovs, fov_phis, n_rings=ray_rings, n_arms=ray_arms
            )
            rays_poly_list = [curve_list_to_polydata(r) for r in rays_curve]
            rays_poly_fov = [merge(r) for r in rays_poly_list]
            for i, r in enumerate(rays_poly_fov):
                r.save(os.path.join(save_dir, f"lens_rays_fov_{i}.obj"))
