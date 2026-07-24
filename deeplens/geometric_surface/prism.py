# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""顺序模式下由入射平面、反射镜和出射平面组成的棱镜表面。"""

import numpy as np
import torch

from .base import Surface
from .plane import Plane
from .mirror import Mirror


class Prism(Surface):
    """由入射平面、内部反射镜和出射平面建模的棱镜。

    用于顺序光线追迹的折叠棱镜。光线先经入射平面折射，再由内部反射镜反射，
    最后经出射平面折射出射。棱镜局部坐标系与入射平面的坐标系重合。

    属性：
        mirror_angle (torch.Tensor): 反射镜倾角，单位为 rad（标量），
            由传入 `__init__` 的角度值转换得到。
        plane1 (Plane): 位于轴向位置 $d$ [mm] 的入射平面。
        mirror (Mirror): 位于以下轴向位置的内部反射镜：
            $d + r\\tan(\\text{mirror\\_angle})$ [mm].
        exit_plane (Plane): 出射平面，与反射镜共享轴向位置 [mm]。
        surfaces (list): 按追迹顺序排列的三个子表面
            `[plane1, mirror, exit_plane]`.
    """

    def __init__(self, r, d, mirror_angle=45.0, mat2="air", device="cpu"):
        """根据孔径、位置和反射镜角度初始化棱镜。

        参数：
            r (float): 孔径半径 [mm]。
            d (float): 棱镜入射平面的轴向位置 [mm]。
            mirror_angle (float, optional): 内部反射镜角度，单位为 degree。
                内部以 rad 存储。默认值为 45.0。
            mat2 (str, optional): 棱镜后的材料。默认值为 "air"。
            device (str, optional): 张量计算设备。默认值为 "cpu"。
        """
        Surface.__init__(self, r, d, mat2=mat2, is_square=True, device=device)
        
        self.mirror_angle = torch.tensor(mirror_angle * torch.pi / 180.0)
        self._init_surfaces()
        
    def _init_surfaces(self):
        """构建入射平面、内部反射镜和出射平面子表面。

        入射平面位于轴向位置 $d$ [mm]，反射镜和出射平面位于
        $d + r\\tan(\\text{mirror\\_angle})$ [mm]。该方法设置 `plane1`、
        `mirror`、`exit_plane` 和 `surfaces` 列表。

        棱镜几何结构：
                               ^ 出射光线
                               |
                            _______
                            |    /
              入射光线  ->  |  /
                            |/
        """
        d = self.d.item()
        mat2 = self.mat2.get_name()
        r = self.r
        device = self.device
        mirror_angle = self.mirror_angle.item()
        
        # 棱镜入口处的平面 1
        plane1_d = d
        pos_xy = [0., 0.]
        vec_local = [0., 0., 1.]
        self.plane1 = Plane(r=r, d=plane1_d, pos_xy=pos_xy, vec_local=vec_local, mat2=mat2, device=device)
        
        # 棱镜内部的反射镜
        mirror_d = d + r * float(np.tan(mirror_angle))
        pos_xy = [0., 0.]
        vec_local = [0., -1., 1.]
        self.mirror = Mirror(r=r, d=mirror_d, pos_xy=pos_xy, vec_local=vec_local, device=device)
        
        # 棱镜出口处的平面 2
        plane2_d = mirror_d
        pos_xy = [0., r]
        vec_local = [0., 1., 0.]
        self.exit_plane = Plane(r=r, d=plane2_d, pos_xy=pos_xy, vec_local=vec_local, mat2=mat2, device=device)

        self.surfaces = [self.plane1, self.mirror, self.exit_plane]
    
    @classmethod
    def init_from_dict(cls, surf_dict):
        """从表面字典构造 Prism。

        参数：
            surf_dict (dict): 表面参数。必须包含键 `r` 和 `d`；可选键为
                `mirror_angle`（默认 45.0）、`mat2`（默认 "air"）和
                `device`（默认 "cpu"）。

        返回：
            prism (Prism): 构造得到的棱镜实例。
        """
        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            mirror_angle=surf_dict.get("mirror_angle", 45.0),
            mat2=surf_dict.get("mat2", "air"),
            device=surf_dict.get("device", "cpu"),
        )

    def ray_reaction(self, ray, n1, n2, refraction=True):
        """依次追迹光线束通过棱镜的三个子表面。

        光线在入射平面折射，由内部反射镜反射，再在出射平面折射。每个子表面
        使用自身的默认反应（平面折射、反射镜反射）；折射率 `n1` 和 `n2`
        会转发给平面以进行折射计算。

        参数：
            ray (Ray): 入射光线束。
            n1 (float): 入射介质的折射率。
            n2 (float): 透射介质的折射率。
            refraction (bool, optional): 仅为兼容基础 `Surface.ray_reaction`
                API 而接收；不会转发给子表面，也不起作用。默认值为 True。

        返回：
            ray (Ray): 离开棱镜后更新的光线束。
        """
        for surface in self.surfaces:
            ray = surface.ray_reaction(ray, n1, n2)
        return ray
