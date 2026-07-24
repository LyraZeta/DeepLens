# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""几何透镜系统工具。

函数：
    - create_lens()：使用平面表面创建透镜设计起点
    - create_surface()：根据表面类型创建表面对象
"""

import os
import random

import numpy as np
import torch

from ..geometric_surface import Aperture, Aspheric, Spheric, ThinLens, Plane
from ..material import MATERIAL_data

# 用于随机选择材料的常见光学玻璃
COMMON_GLASSES = [
    "n-bk7", "n-sk16", "h-k9l", "n-lak14", "n-sk2", "bk7", "n-lak7",
    "f2", "n-f2", "n-sf5", "n-sf11", "n-sf1",
    "pmma", "coc", "okp4",
]


# ====================================================================================
# 透镜设计起点生成
# ====================================================================================
def create_lens(
    fov,
    fnum,
    bfl,
    foclen=None,
    imgh=None,
    thickness=None,
    surf_list=None,
    save_dir="./",
):
    """使用平面表面创建透镜设计起点。

    根据元件/光阑规格列表构建 `GeoLens`，随机初始化材料和曲率，
    随后归一化厚度与半孔径、添加占位传感器，并保存透镜 JSON 和分析结果。

    贡献者：Rayengineer

    `foclen` 与 `imgh` 必须且只能提供一个；另一个由下式推导：
    $\\text{imgh} = \\text{foclen} \\cdot \\tan(\\text{fov} / 2)$.

    参数：
        fov (float)：对角线视场角，单位为 degree。
        fnum (float)：目标 F 数（焦距/入瞳直径）；通过
            $\\text{aper\\_r} = \\text{foclen} / \\text{fnum} / 2$ 设置光阑半径。
        bfl (float)：后焦距，即末表面到传感器的距离，单位为 mm。
        foclen (float or None, optional)：焦距，单位为 mm。与 `imgh` 互斥。默认值为 None。
        imgh (float or None, optional)：半对角线像高，单位为 mm（= r_sensor）。与 `foclen` 互斥。默认值为 None。
        thickness (float or None, optional)：总厚度，单位为 mm。默认值为 None，此时使用 `foclen + bfl`。
        surf_list (list or None, optional)：元件规格列表；每个元件为字符串
            ("Aperture") 或表面类型列表。默认值为 None，此时使用
            `[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]]`。
        save_dir (str, optional)：保存透镜 JSON 和分析结果的目录。默认值为 "./"。

    返回：
        lens (GeoLens)：构建完成的透镜，带有占位的 2000x2000 传感器。

    异常：
        ValueError：同时提供或均未提供 `foclen` 与 `imgh` 时抛出。
        Exception：透镜元件规格或表面类型不受支持时抛出。
    """
    from ..geolens import GeoLens

    # 避免使用可变默认参数。
    if surf_list is None:
        surf_list = [["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]]

    # 确定 foclen / imgh
    half_fov = np.deg2rad(fov / 2)
    if foclen is not None and imgh is not None:
        raise ValueError("Specify exactly one of foclen or imgh, not both.")
    elif foclen is not None:
        imgh = round(foclen * float(np.tan(half_fov)), 2)
    elif imgh is not None:
        foclen = round(imgh / float(np.tan(half_fov)), 4)
    else:
        raise ValueError("Specify exactly one of foclen or imgh.")

    # 计算透镜参数
    aper_r = foclen / fnum / 2
    if thickness is None:
        thickness = foclen + bfl
    d_opt = thickness - bfl

    # 材料：使用常见玻璃，而非包含 700 多种材料的完整目录
    mat_names = [m for m in COMMON_GLASSES if m in MATERIAL_data]

    # 创建透镜
    lens = GeoLens()
    surfaces = lens.surfaces

    d_total = 0.0
    for elem_type in surf_list:
        if elem_type == "Aperture":
            d_next = (torch.rand(1) + 0.5).item()
            surfaces.append(Aperture(r=aper_r, d=d_total))
            d_total += d_next

        elif isinstance(elem_type, list):
            if len(elem_type) == 1 and elem_type[0] == "Aperture":
                d_next = (torch.rand(1) + 0.5).item()
                surfaces.append(Aperture(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) == 1 and elem_type[0] == "ThinLens":
                d_next = (torch.rand(1) + 1.0).item()
                surfaces.append(ThinLens(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) in [2, 3]:
                for i, surface_type in enumerate(elem_type):
                    if i == len(elem_type) - 1:
                        mat = "air"
                        d_next = (torch.rand(1) + 0.5).item()
                    else:
                        mat = random.choice(mat_names)
                        d_next = (torch.rand(1) + 1.0).item()

                    surfaces.append(
                        create_surface(surface_type, d_total, aper_r, imgh, mat)
                    )
                    d_total += d_next
            else:
                raise Exception("Lens element type not supported yet.")
        else:
            raise Exception("Lens type format not correct.")

    # 归一化光学部分的总厚度
    d_opt_actual = d_total - d_next
    for s in surfaces:
        s.d = s.d / d_opt_actual * d_opt

    # 根据表面相对于孔径光阑的位置更新表面半孔径。
    # 距光阑较远的表面需要更大的半径，以允许离轴光线通过。
    # r_i = aper_r + |d_i - d_stop| * tan(half_fov)
    d_stop = None
    for s in surfaces:
        if isinstance(s, Aperture):
            d_stop = s.d.item() if hasattr(s.d, "item") else float(s.d)
            break
    if d_stop is not None:
        for s in surfaces:
            if isinstance(s, Aperture):
                continue
            d_i = s.d.item() if hasattr(s.d, "item") else float(s.d)
            s.r = aper_r + abs(d_i - d_stop) * float(np.tan(half_fov))

    # 透镜传感器（占位传感器分辨率）
    lens = lens.to(lens.device)
    lens.d_sensor = torch.tensor(thickness, device=lens.device)
    lens.r_sensor = imgh
    lens.set_sensor_res(sensor_res=(2000, 2000))

    # 透镜计算
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_rfov = False
    lens.post_computation()

    # 保存透镜
    os.makedirs(save_dir, exist_ok=True)
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}"
    lens.write_lens_json(os.path.join(save_dir, f"{filename}.json"))
    lens.analysis(os.path.join(save_dir, f"{filename}"))

    return lens

def create_surface(surface_type, d_total, aper_r, imgh, mat):
    """根据表面类型创建表面对象。

    使用较小的随机曲率初始化 `Spheric`、`Aspheric` 或 `Plane` 表面
    （后续介质为空气时取负值，否则取正值）。

    参数：
        surface_type (str)：表面类型，可为 "Spheric"、"Aspheric" 或 "Plane"。
        d_total (float)：表面沿光轴的轴向位置，单位为 mm。
        aper_r (float)：初始半孔径半径，单位为 mm（稍后在厚度归一化后更新）。
        imgh (float)：半对角线像高，单位为 mm。当前函数未使用。
        mat (str)：表面之后介质的材料名称（"air" 或玻璃名称）。

    返回：
        surface (Surface)：创建的表面对象。

    异常：
        Exception：`surface_type` 不受支持时抛出。
    """
    if mat == "air":
        c = -float(np.random.rand()) * 0.001
    else:
        c = float(np.random.rand()) * 0.001
    # 使用 aper_r 作为初始半径；厚度归一化后将更新该值
    r = aper_r

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)

    elif surface_type == "Aspheric":
        ai = np.random.randn(8).astype(np.float32) * 1e-24
        k = float(np.random.rand()) * 1e-6
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)

    elif surface_type == "Plane":
        return Plane(r=r, d=d_total, mat2=mat)

    else:
        raise Exception("Surface type not supported yet.")


