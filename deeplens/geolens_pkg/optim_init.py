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

from ..config import DEFAULT_WAVE, DEPTH, WAVE_RGB
from ..geometric_surface import Aperture, Aspheric, Spheric, ThinLens, Plane
from ..material import MATERIAL_data, Material

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
    primary_wvln=DEFAULT_WAVE,
    wvln_rgb=WAVE_RGB,
    obj_depth=DEPTH,
    material_names=None,
    sensor_res=(2000, 2000),
    analyze=True,
    curvature_scale=1e-3,
):
    """使用平面表面创建透镜设计起点。

    根据元件/光阑规格列表构建 `GeoLens`，随机初始化材料和曲率，
    随后归一化厚度与半孔径、添加占位传感器，并保存透镜 JSON 和分析结果。

    贡献者：Rayengineer

    `foclen` 与 `imgh` 必须且只能提供一个；另一个由下式推导：
    $\\text{imgh} = \\text{foclen} \\cdot \\tan(\\text{fov} / 2)$.

    参数：
        fov (float)：等效径向全视场角，单位为 degree。对于当前 MWIR
            设计，它对应 Zemax 视场表中的 Y 向全视场。
        fnum (float)：目标 F 数（焦距/入瞳直径）；通过
            $\\text{aper\\_r} = \\text{foclen} / \\text{fnum} / 2$ 设置光阑半径。
        bfl (float)：后焦距，即末表面到传感器的距离，单位为 mm。
        foclen (float or None, optional)：焦距，单位为 mm。与 `imgh` 互斥。默认值为 None。
        imgh (float or None, optional)：最大设计场点的半像高，单位为 mm
            （= r_sensor）。与 `foclen` 互斥。默认值为 None。
        thickness (float or None, optional)：总厚度，单位为 mm。默认值为 None，此时使用 `foclen + bfl`。
        surf_list (list or None, optional)：元件规格列表；每个元件为字符串
            ("Aperture") 或表面类型列表。默认值为 None，此时使用
            `[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]]`。
        save_dir (str, optional)：保存透镜 JSON 和分析结果的目录。默认值为 "./"。
        primary_wvln (float, optional)：主要设计波长 [µm]。默认使用可见光主波长。
        wvln_rgb (sequence of float, optional)：三条代表性波长 [µm]，按
            ``[R, G, B]`` 接口顺序传入。对于 MWIR 可使用 ``[2.7, 3.5, 4.3]``。
        obj_depth (float, optional)：物距 [mm]；负的大数可近似无穷远。
        material_names (sequence of str, optional)：随机初始化使用的材料名称。
            不提供时沿用可见光常用玻璃列表；MWIR 设计应显式提供红外材料。
        sensor_res (tuple[int, int] or None, optional)：占位传感器分辨率。
            传入 None 时保留默认分辨率；创建完成后仍可调用 ``set_sensor``。
        analyze (bool, optional)：是否在生成起点时执行完整分析。大焦平面或 CPU
            设计建议先设为 False，优化结束后再分析。
        curvature_scale (float, optional)：随机初始曲率的尺度 [1/mm]。默认值
            ``1e-3`` 与原有可见光示例一致；长焦系统应按目标焦距使用更小的尺度。

    返回：
        lens (GeoLens)：构建完成的透镜，传感器分辨率由 ``sensor_res`` 指定。

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

    # 材料：默认使用可见光常用玻璃；红外设计通过参数显式传入材料池。
    if material_names is None:
        mat_names = [m for m in COMMON_GLASSES if m in MATERIAL_data]
    else:
        mat_names = list(dict.fromkeys(material_names))
        if not mat_names:
            raise ValueError("material_names 不能为空。")
        # 这里同时检查 AGF、内置自定义表和 refractiveindex.info 回退目录。
        for material_name in mat_names:
            Material(material_name)

    # 创建透镜
    lens = GeoLens(
        primary_wvln=primary_wvln,
        wvln_rgb=list(wvln_rgb),
        obj_depth=obj_depth,
    )
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
                        create_surface(
                            surface_type,
                            d_total,
                            aper_r,
                            imgh,
                            mat,
                            curvature_scale=curvature_scale,
                        )
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
    if sensor_res is not None:
        lens.set_sensor_res(sensor_res=sensor_res)

    # 透镜计算
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_rfov = False
    lens.post_computation()

    # 保存透镜
    os.makedirs(save_dir, exist_ok=True)
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}"
    lens.write_lens_json(os.path.join(save_dir, f"{filename}.json"))
    if analyze:
        lens.analysis(os.path.join(save_dir, f"{filename}"))

    return lens

def create_surface(surface_type, d_total, aper_r, imgh, mat, curvature_scale=1e-3):
    """根据表面类型创建表面对象。

    使用较小的随机曲率初始化 `Spheric`、`Aspheric` 或 `Plane` 表面
    （后续介质为空气时取负值，否则取正值）。

    参数：
        surface_type (str)：表面类型，可为 "Spheric"、"Aspheric" 或 "Plane"。
        d_total (float)：表面沿光轴的轴向位置，单位为 mm。
        aper_r (float)：初始半孔径半径，单位为 mm（稍后在厚度归一化后更新）。
        imgh (float)：最大设计场点的半像高，单位为 mm。当前函数未使用。
        mat (str)：表面之后介质的材料名称（"air" 或玻璃名称）。
        curvature_scale (float)：随机初始曲率尺度 [1/mm]。

    返回：
        surface (Surface)：创建的表面对象。

    异常：
        Exception：`surface_type` 不受支持时抛出。
    """
    if curvature_scale <= 0:
        raise ValueError("curvature_scale 必须为正数。")
    if mat == "air":
        c = -float(np.random.rand()) * curvature_scale
    else:
        c = float(np.random.rand()) * curvature_scale
    # 使用 aper_r 作为初始半径；厚度归一化后将更新该值
    r = aper_r

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)

    elif surface_type == "Aspheric":
        # 非球面系数必须按口径和阶次归一化。固定使用 1e-24 对手机镜头尚可，
        # 但在 140 mm 级半口径上，a18*r^18 会溢出并产生 inf。这里让每一阶
        # 在口径边缘造成的初始矢高扰动约为 1e-6 mm，使不同口径的随机起点
        # 都保持有限且接近平面/球面。
        r_norm = max(float(r), 1.0)
        ai = np.asarray(
            [
                np.random.randn() * 1e-6 / r_norm**order
                for order in range(4, 20, 2)
            ],
            dtype=np.float32,
        )
        k = float(np.random.rand()) * 1e-6
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)

    elif surface_type == "Plane":
        return Plane(r=r, d=d_total, mat2=mat)

    else:
        raise Exception("Surface type not supported yet.")


