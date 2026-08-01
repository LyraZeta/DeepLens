"""中波红外透射式望远系统的 DeepLens 初始设计入口。

本脚本先检查用户规格，再生成一个不超过 7 片透镜的 GeoLens 初始结构。
默认使用 Zemax 图中的 Y 向全视场 9.6°、半像高 47.1454 mm 和 280 mm
入瞳；焦距由像高和视场自动推导为约 561.44 mm。探测器格式尚未确认，
因此初始结构使用圆形等效虚拟仿真焦面，不把 320×256、30 微米写成硬约束。
若要研究旧的 42 微弧度方案，可在命令行显式加入
``--two-pixel-resolution-urad 42``。

示例：

    python mwir_telescope_design.py --check-only
    python mwir_telescope_design.py --device cpu --iterations 0
    python mwir_telescope_design.py --device cuda --iterations 2000

``iterations=0`` 只生成初始镜头和元数据，不启动耗时的梯度优化；建议先用
``--check-only`` 和 ``--iterations 0`` 确认几何规格，再逐步增加迭代次数。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from mwir_spec import MWIRDesignSpec, configure_utf8_console


# 6 个独立透镜元件、1 个光阑；元件数量按透镜而不是光学面计数。
MWIR_SURFACE_LIST = [
    ["Aperture"],
    ["Aspheric", "Spheric"],
    ["Spheric", "Aspheric"],
    ["Spheric", "Spheric"],
    ["Spheric", "Aspheric"],
    ["Aspheric", "Spheric"],
    ["Spheric", "Spheric"],
]

# 独立的六片消色差概念起点。它刻意把每片的功率集中到单面，只适合检查
# 一阶功率和色差配平，不应作为已经完成像差优化的最终处方。
MWIR_BALANCED_SURFACE_LIST = [
    ["Aperture"],
    ["Aspheric", "Spheric"],
    ["Spheric", "Aspheric"],
    ["Aspheric", "Spheric"],
    ["Spheric", "Aspheric"],
    ["Aspheric", "Spheric"],
    ["Spheric", "Aspheric"],
]

MWIR_BALANCED_ACHROMAT_GROUPS = (
    ("si", "mgf2", 0.50),
    ("znse", "caf2", 0.30),
    ("si", "mgf2", 0.20),
)
MWIR_BALANCED_FRONT_POSITIONS_MM = (12.0, 39.0, 130.0, 157.0, 450.0, 477.0)
MWIR_BALANCED_CENTER_THICKNESSES_MM = (22.0, 14.0, 20.0, 14.0, 18.0, 12.0)
MWIR_BALANCED_SEMI_APERTURES_MM = (150.0, 150.0, 155.0, 155.0, 165.0, 165.0)
MWIR_BALANCED_POSITIVE_CONIC = -5.0
MWIR_BALANCED_NEGATIVE_CONIC = 5.0

# 七片强弯曲透射母型。它采用“前正组—中负组—后正组—弯月场平镜”的
# 功率分配，14 个折射面均具有非零曲率。这里的半径和间距只定义一个可优化
# 母型；它们不是最终像质已经达标的处方。
MWIR_POWER_BENT7_SURFACE_LIST = [
    ["Aperture"],
    ["Aspheric", "Spheric"],
    ["Spheric", "Spheric"],
    ["Spheric", "Aspheric"],
    ["Spheric", "Spheric"],
    ["Aspheric", "Aspheric"],
    ["Spheric", "Spheric"],
    ["Aspheric", "Spheric"],
]
MWIR_POWER_BENT7_FRONT_POSITIONS_MM = (12.0, 44.0, 105.0, 139.0, 240.0, 282.0, 413.0)
MWIR_POWER_BENT7_CENTER_THICKNESSES_MM = (24.0, 16.0, 24.0, 26.0, 30.0, 16.0, 12.0)
MWIR_POWER_BENT7_SEMI_APERTURES_MM = (150.0, 150.0, 155.0, 155.0, 160.0, 160.0, 110.0)
MWIR_POWER_BENT7_MATERIALS = ("si", "mgf2", "si", "mgf2", "znse", "caf2", "si")
MWIR_POWER_BENT7_RADII_MM = (
    (502.254, 2176.433),
    (2693.126, 621.491),
    (-1126.607, 1126.607),
    (1397.089, -1397.089),
    (-3539.058, -589.843),
    (-1044.589, -6267.531),
    (-1428.571, -1437.071),
)
# 元素序号从 0 开始，side=0/1 分别表示前/后表面。首轮球面优化保持
# k=A4=A6=...=0；只有进入非球面阶段后才放开这些预留面。
MWIR_POWER_BENT7_ASPHERIC_SIDES = frozenset(
    {(0, 0), (2, 1), (4, 0), (4, 1), (6, 0)}
)

# 这些材料的折射率数据覆盖 2.7–4.3 µm；实际透过率、吸收和镀膜仍需单独核验。
MWIR_MATERIALS = ["znse", "caf2", "mgf2", "ge", "si", "krs5"]

# 第一阶段优先稳定焦距和像高映射；曲率、高阶非球面采用保守步长，随后可通过
# ``--lrs`` 在像质阶段逐步放开。
DEFAULT_MWIR_LRS = (2e-3, 2e-7, 2e-4, 2e-6)


def _mwir_dispersion_number(
    material_name: str,
    wavelengths_um: tuple[float, float, float] = (2.7, 3.5, 4.3),
) -> float:
    """计算用于 2.7–4.3 µm 配对的一阶等效色散数。

    这里采用 ``V_IR = (n_mid - 1) / (n_short - n_long)``。它不是可见光
    d/F/C 线阿贝数，只用于本项目三条 MWIR 设计波长下的薄透镜色差配平。
    """

    from deeplens.material import Material

    if len(wavelengths_um) != 3:
        raise ValueError("MWIR 等效色散数需要短、中、长三个波长。")
    material = Material(material_name)
    n_short, n_primary, n_long = (
        float(material.refractive_index(float(wavelength)))
        for wavelength in wavelengths_um
    )
    dispersion = n_short - n_long
    if not math.isfinite(dispersion) or abs(dispersion) < 1e-12:
        raise ValueError(f"材料 {material_name} 在目标波段的色散过小或无效。")
    value = (n_primary - 1.0) / dispersion
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"材料 {material_name} 的 MWIR 等效色散数无效：{value}。")
    return float(value)


def _balanced_power_design(spec: MWIRDesignSpec) -> dict[str, Any]:
    """求解三组正负薄透镜的目标光焦度及一阶色差配平量。"""

    target_total_power = 1.0 / spec.effective_focal_length_mm
    wavelengths = tuple(float(value) for value in spec.wavelengths_um)
    element_materials: list[str] = []
    element_powers: list[float] = []
    groups: list[dict[str, Any]] = []
    dispersion_numbers: dict[str, float] = {}

    for positive_material, negative_material, net_fraction in (
        MWIR_BALANCED_ACHROMAT_GROUPS
    ):
        positive_v = dispersion_numbers.setdefault(
            positive_material,
            _mwir_dispersion_number(positive_material, wavelengths),
        )
        negative_v = dispersion_numbers.setdefault(
            negative_material,
            _mwir_dispersion_number(negative_material, wavelengths),
        )
        if math.isclose(positive_v, negative_v, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"材料对 {positive_material}/{negative_material} 的等效色散数相同，"
                "无法配平一阶色差。"
            )

        net_power = target_total_power * net_fraction
        positive_power = net_power * positive_v / (positive_v - negative_v)
        negative_power = -net_power * negative_v / (positive_v - negative_v)
        color_sum = positive_power / positive_v + negative_power / negative_v

        element_materials.extend((positive_material, negative_material))
        element_powers.extend((positive_power, negative_power))
        groups.append(
            {
                "positive_material": positive_material,
                "negative_material": negative_material,
                "net_power_fraction": float(net_fraction),
                "net_power_1_per_mm": float(net_power),
                "positive_power_1_per_mm": float(positive_power),
                "negative_power_1_per_mm": float(negative_power),
                "positive_dispersion_number": float(positive_v),
                "negative_dispersion_number": float(negative_v),
                "first_order_color_sum_1_per_mm": float(color_sum),
            }
        )

    return {
        "method": (
            "三组薄透镜分别满足 φ+ + φ- = φ组、"
            "φ+/V+ + φ-/V- = 0；组净功率占总目标光焦度 50%/30%/20%。"
        ),
        "dispersion_number_definition": "(n_3.5um - 1) / (n_2.7um - n_4.3um)",
        "wavelengths_um": list(wavelengths),
        "target_total_power_1_per_mm": float(target_total_power),
        "dispersion_numbers": dispersion_numbers,
        "element_materials": element_materials,
        "element_powers_1_per_mm": element_powers,
        "groups": groups,
        "summed_element_power_1_per_mm": float(sum(element_powers)),
    }


def _build_balanced_transmission_lens(
    spec: MWIRDesignSpec,
    params: dict[str, Any],
    torch_device,
    power_design: dict[str, Any],
):
    """按确定性的六片三组消色差处方构建 ``transmission_balanced``。"""

    import torch

    from deeplens import GeoLens
    from deeplens.geometric_surface import Aperture, Aspheric, Spheric
    from deeplens.material import Material

    materials = list(power_design["element_materials"])
    powers = [float(value) for value in power_design["element_powers_1_per_mm"]]
    if len(materials) != 6 or len(powers) != 6:
        raise ValueError("transmission_balanced 必须包含 6 片确定性透镜。")

    lens = GeoLens(
        primary_wvln=3.5,
        wvln_rgb=list(spec.wavelengths_um),
        obj_depth=spec.object_distance_mm,
    )
    surfaces = [
        Aperture(r=spec.entrance_pupil_diameter_mm / 2.0, d=0.0)
    ]
    primary_wavelength = float(spec.wavelengths_um[1])

    for material_name, target_power, front_z, center_thickness, semi_aperture in zip(
        materials,
        powers,
        MWIR_BALANCED_FRONT_POSITIONS_MM,
        MWIR_BALANCED_CENTER_THICKNESSES_MM,
        MWIR_BALANCED_SEMI_APERTURES_MM,
    ):
        refractive_index = float(
            Material(material_name).refractive_index(primary_wavelength)
        )
        if target_power > 0.0:
            front_curvature = target_power / (refractive_index - 1.0)
            surfaces.append(
                Aspheric(
                    r=semi_aperture,
                    d=front_z,
                    c=front_curvature,
                    k=MWIR_BALANCED_POSITIVE_CONIC,
                    ai=[0.0] * 7,
                    mat2=material_name,
                )
            )
            surfaces.append(
                Spheric(
                    r=semi_aperture,
                    d=front_z + center_thickness,
                    c=0.0,
                    mat2="air",
                )
            )
        else:
            rear_curvature = -target_power / (refractive_index - 1.0)
            surfaces.append(
                Spheric(
                    r=semi_aperture,
                    d=front_z,
                    c=0.0,
                    mat2=material_name,
                )
            )
            surfaces.append(
                Aspheric(
                    r=semi_aperture,
                    d=front_z + center_thickness,
                    c=rear_curvature,
                    k=MWIR_BALANCED_NEGATIVE_CONIC,
                    ai=[0.0] * 7,
                    mat2="air",
                )
            )

    lens.surfaces = surfaces
    lens.lens_info = "MWIR 六片三组正负光焦度消色差透射起点"
    lens.d_sensor = torch.tensor(
        MWIR_BALANCED_FRONT_POSITIONS_MM[-1]
        + MWIR_BALANCED_CENTER_THICKNESSES_MM[-1]
        + params["bfl_mm"]
    )
    lens.r_sensor = float(params["optimization_radial_image_height_mm"])
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_rfov = False
    lens.set_sensor(
        tuple(params["sensor_size_mm"]), tuple(params["sensor_res"])
    )
    lens = lens.to(torch_device)
    lens.post_computation()
    return lens


def _build_power_bent7_lens(
    spec: MWIRDesignSpec,
    params: dict[str, Any],
    torch_device,
):
    """构建双面均有真实光焦度的七片强弯曲 MWIR 母型。

    该母型用确定性的材料、顶点位置、厚度和两面曲率取代旧六片概念处方的
    “单曲面 + 平面”结构。五个曲面预留为偶次非球面，但初值仍严格等价于
    球面，便于先完成球面全局筛选，再逐级放开圆锥常数和低阶非球面系数。
    """

    import torch

    from deeplens import GeoLens
    from deeplens.geometric_surface import Aperture, Aspheric, Spheric

    lengths = {
        len(MWIR_POWER_BENT7_FRONT_POSITIONS_MM),
        len(MWIR_POWER_BENT7_CENTER_THICKNESSES_MM),
        len(MWIR_POWER_BENT7_SEMI_APERTURES_MM),
        len(MWIR_POWER_BENT7_MATERIALS),
        len(MWIR_POWER_BENT7_RADII_MM),
    }
    if lengths != {7}:
        raise ValueError("七片强弯曲母型的材料、位置、厚度、口径和半径数量必须一致。")

    lens = GeoLens(
        primary_wvln=3.5,
        wvln_rgb=list(spec.wavelengths_um),
        obj_depth=spec.object_distance_mm,
    )
    surfaces = [
        Aperture(r=spec.entrance_pupil_diameter_mm / 2.0, d=0.0)
    ]
    for element_index, (
        material_name,
        front_z,
        center_thickness,
        semi_aperture,
        radii,
    ) in enumerate(
        zip(
            MWIR_POWER_BENT7_MATERIALS,
            MWIR_POWER_BENT7_FRONT_POSITIONS_MM,
            MWIR_POWER_BENT7_CENTER_THICKNESSES_MM,
            MWIR_POWER_BENT7_SEMI_APERTURES_MM,
            MWIR_POWER_BENT7_RADII_MM,
        )
    ):
        for side, (radius, z_position, material_after) in enumerate(
            (
                (radii[0], front_z, material_name),
                (radii[1], front_z + center_thickness, "air"),
            )
        ):
            surface_kwargs = {
                "r": semi_aperture,
                "d": z_position,
                "c": 1.0 / radius,
                "mat2": material_after,
            }
            if (element_index, side) in MWIR_POWER_BENT7_ASPHERIC_SIDES:
                surfaces.append(
                    Aspheric(
                        **surface_kwargs,
                        k=0.0,
                        # 只预留 A4/A6/A8/A10；避免在球面起点直接引入高阶自由度。
                        ai=[0.0] * 4,
                    )
                )
            else:
                surfaces.append(Spheric(**surface_kwargs))

    lens.surfaces = surfaces
    lens.lens_info = "MWIR 七片强弯曲正负功率抵消透射母型"
    last_rear_z = (
        MWIR_POWER_BENT7_FRONT_POSITIONS_MM[-1]
        + MWIR_POWER_BENT7_CENTER_THICKNESSES_MM[-1]
    )
    lens.d_sensor = torch.tensor(last_rear_z + params["bfl_mm"])
    lens.r_sensor = float(params["optimization_radial_image_height_mm"])
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_rfov = False
    lens.set_sensor(tuple(params["sensor_size_mm"]), tuple(params["sensor_res"]))
    lens = lens.to(torch_device)
    lens.post_computation()
    return lens


def _scheme_parameters(spec: MWIRDesignSpec, scheme: str) -> dict[str, Any]:
    """根据方案名称返回焦距、Y 向视场、虚拟焦面和 F 数。"""

    if scheme in {
        "transmission_baseline",
        "transmission_balanced",
        "transmission_power_bent7",
        "cassegrain_equivalent",
    }:
        # 这是当前正式设计基线。焦距由 Zemax 的半像高和 Y 向全视场推导，
        # 而不是由一个尚未确认的探测器格式反推。
        focal_length = spec.effective_focal_length_mm
        field_y = spec.full_field_y_deg
        fnum = spec.effective_focal_length_mm / spec.entrance_pupil_diameter_mm
        sensor_res = spec.virtual_sensor_res
        sensor_size = spec.virtual_sensor_size_mm
        image_height = spec.image_height_mm
        sensor_is_virtual = True
        if scheme == "transmission_power_bent7":
            explanation = (
                f"七片强弯曲透射母型：Y 向全视场 {field_y:.4f}°、"
                f"半像高 {image_height:.4f} mm、"
                f"{spec.entrance_pupil_diameter_mm:.3f} mm 入瞳；"
                "采用前正组—中负组—后正组—弯月场平镜，14 面均有非零曲率。"
            )
        elif scheme == "transmission_balanced":
            explanation = (
                f"六片消色差概念起点：Y 向全视场 {field_y:.4f}°、"
                f"半像高 {image_height:.4f} mm、"
                f"{spec.entrance_pupil_diameter_mm:.3f} mm 入瞳；"
                "采用 Si/MgF2、ZnSe/CaF2、Si/MgF2 三组正负光焦度配对；"
                "仅用于一阶检查，不代表像差优化完成。"
            )
        else:
            explanation = (
                f"正式基线：Y 向全视场 {field_y:.4f}°、"
                f"半像高 {image_height:.4f} mm、"
                f"{spec.entrance_pupil_diameter_mm:.3f} mm 入瞳；"
                "探测器使用圆形等效虚拟焦面。"
            )
    elif scheme == "large_fpa":
        if not spec.resolution_constraint_active:
            raise ValueError(
                "large_fpa 是历史大焦面方案，需要显式加入 "
                "--two-pixel-resolution-urad 42。"
            )
        focal_length = spec.required_focal_length_mm
        field_y = spec.full_field_y_deg
        fnum = spec.required_f_number
        image_height = spec.required_image_height_mm
        recommended_res = spec.recommended_detector_res
        if recommended_res is None:
            # 未知探测器纵横比时，使用与目标 Y 半像高相切的圆形等效正方形
            # 焦面；不能复用正式基线的 47.1454 mm 虚拟焦面。
            side_mm = math.sqrt(2.0) * image_height
            count = max(64, int(round(side_mm / spec.pixel_pitch_mm)))
            sensor_res = (count, count)
            sensor_size = (side_mm, side_mm)
            sensor_is_virtual = True
        else:
            # 47.1454/required_image_height_mm 是 Y 半像高，因此矩形探测器
            # 的高度必须覆盖两倍像高，宽度按已知阵列纵横比计算。取整后用
            # 像元数乘像元间距，确保 set_sensor() 的尺寸与分辨率严格同宽高比。
            sensor_res = recommended_res
            sensor_size = (
                sensor_res[0] * spec.pixel_pitch_mm,
                sensor_res[1] * spec.pixel_pitch_mm,
            )
            sensor_is_virtual = not spec.detector_is_known
        explanation = "历史方案：由两像元角采样反推焦距和大焦面，仅用于对比。"
    elif scheme in {"existing_fpa_narrow", "existing_fpa_wide"}:
        if not spec.detector_is_known:
            raise ValueError(
                f"{scheme} 需要同时提供 pixel_pitch_um 和 detector_res；"
                "当前正式设计请使用 transmission_baseline。"
            )
        if scheme == "existing_fpa_narrow":
            focal_length = spec.effective_focal_length_mm
            field_y = spec.current_detector_y_fov_deg
            explanation = "对照方案：固定当前焦距，报告已确认探测器能够覆盖的 Y 向视场。"
        else:
            focal_length = spec.focal_length_for_current_detector_y_fov_mm
            field_y = spec.full_field_y_deg
            explanation = "历史对照方案：用已确认探测器高度覆盖目标 Y 向视场。"
        if focal_length is None or field_y is None:
            raise ValueError("无法从当前探测器参数推导有效焦距或视场。")
        fnum = focal_length / spec.entrance_pupil_diameter_mm
        sensor_res = spec.detector_res
        sensor_size = spec.detector_size_mm
        image_height = sensor_size[1] / 2.0
        sensor_is_virtual = False
    else:
        raise ValueError(f"未知方案：{scheme}")

    # DeepLens 的 rfov/r_sensor 表示径向（半对角）设计场。矩形探测器已知时，
    # 它通常大于用户给定的 Y 向半视场；用传感器半对角和目标焦距推导优化场，
    # 可同时保持 Y 像高、纵横比和有效焦距的一致性。
    radial_image_height = math.hypot(*sensor_size) / 2.0
    optimization_radial_fov = math.degrees(
        2.0 * math.atan(radial_image_height / focal_length)
    )

    # 初始结构的后焦距和厚度只是几何种子，不是总长硬约束；正式优化时可继续压缩。
    return {
        "scheme": scheme,
        "explanation": explanation,
        "focal_length_mm": float(focal_length),
        "field_y_deg": float(field_y),
        "image_height_mm": float(image_height),
        "image_height_y_mm": float(image_height),
        "optimization_radial_image_height_mm": float(radial_image_height),
        "optimization_radial_fov_deg": float(optimization_radial_fov),
        # 保留旧键，避免外部脚本读取元数据时立即失效；它现在表示 Y 向全视场。
        "diagonal_fov_deg": float(field_y),
        "f_number": float(fnum),
        "sensor_res": list(sensor_res),
        "sensor_size_mm": list(sensor_size),
        "sensor_is_virtual": sensor_is_virtual,
        "element_count": 7 if scheme == "transmission_power_bent7" else 6,
        "surface_count": 15 if scheme == "transmission_power_bent7" else 13,
        "bfl_mm": (
            160.0
            if scheme == "transmission_power_bent7"
            else (80.0 if focal_length > 500.0 else 25.0)
        ),
        "thickness_mm": 300.0 if focal_length > 500.0 else 160.0,
        "total_track_constraint_mm": None,
    }


def _make_result_dir(output: str | None) -> Path:
    """创建结果目录。"""

    if output:
        result_dir = Path(output)
    else:
        stamp = datetime.now().strftime("%m%d-%H%M%S")
        result_dir = Path("results") / f"MWIR-Telescope-{stamp}"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


def _apply_mwir_constraints(lens, spec: MWIRDesignSpec) -> None:
    """重新应用不会写入原生 JSON 的 MWIR 机械与畸变约束。"""

    lens.init_constraints(
        {
            "air_center_max": 200.0,
            "air_edge_max": 200.0,
            "thick_center_max": 60.0,
            "thick_edge_max": 60.0,
            "bfl_max": 1_000.0,
            "ttl_max": 1_000_000.0,
            "distortion_max": spec.distortion_limit,
        }
    )


def _count_refractive_elements(lens) -> int:
    """按进入非空气材料的表面数统计独立折射元件。"""

    return len(_element_material_names(lens))


def _element_material_names(lens) -> list[str]:
    """按光路顺序返回各折射元件进入面的材料名称。"""

    names: list[str] = []
    for surface in lens.surfaces:
        material = getattr(surface, "mat2", None)
        name = getattr(material, "name", str(material)).lower()
        if name not in {"air", "vacuum", "occluder", "none"}:
            names.append(name)
    return names


def _find_source_design_metadata(
    input_path: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    """在处方同级或 ``optimization`` 上一级查找设计 metadata。"""

    candidates = [input_path.parent / "mwir_design_metadata.json"]
    if input_path.parent.name.lower() == "optimization":
        candidates.append(input_path.parent.parent / "mwir_design_metadata.json")
    for candidate in candidates:
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as file:
                return candidate, json.load(file)
    return None, None


def _validate_source_design_metadata(
    metadata: dict[str, Any] | None,
    design_params: dict[str, Any],
) -> dict[str, float]:
    """阻止续跑时静默改变原始视场、像高或目标焦距。"""

    if not isinstance(metadata, dict) or not isinstance(metadata.get("design"), dict):
        raise ValueError("输入处方缺少可核验的 mwir_design_metadata.json/design。")
    source_design = metadata["design"]
    fields = (
        "focal_length_mm",
        "field_y_deg",
        "image_height_mm",
        "optimization_radial_fov_deg",
        "optimization_radial_image_height_mm",
        "f_number",
    )
    checked: dict[str, float] = {}
    for name in fields:
        if name not in source_design:
            raise ValueError(f"输入处方 metadata 缺少设计字段：{name}")
        source_value = float(source_design[name])
        target_value = float(design_params[name])
        if not math.isclose(
            source_value, target_value, rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(
                f"输入处方原目标 {name}={source_value}，当前命令要求 "
                f"{target_value}；这不是同规格续跑。"
            )
        checked[name] = source_value
    return checked


def _record_optimization_config(
    result_path: Path, optimization_config: dict[str, Any]
) -> None:
    """把完整阶段参数补写到设计 metadata，便于复现与审计。"""

    metadata_path = result_path / "mwir_design_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    metadata["optimization_config"] = optimization_config
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def _validate_loaded_mwir_lens(
    lens, spec: MWIRDesignSpec, design_params: dict[str, Any]
) -> dict[str, Any]:
    """校验分阶段输入处方仍与当前 MWIR 任务规格一致。"""

    loaded_wavelengths = tuple(float(value) for value in lens.wvln_rgb)
    target_wavelengths = tuple(float(value) for value in spec.wavelengths_um)
    if len(loaded_wavelengths) != len(target_wavelengths) or any(
        not math.isclose(actual, target, rel_tol=0.0, abs_tol=1e-9)
        for actual, target in zip(loaded_wavelengths, target_wavelengths)
    ):
        raise ValueError(
            f"输入处方波长 {loaded_wavelengths} 与当前规格 {target_wavelengths} 不一致。"
        )

    primary_wavelength = float(lens.primary_wvln)
    if not math.isclose(
        primary_wavelength, target_wavelengths[1], rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError(
            f"输入处方主波长 {primary_wavelength} µm 不是当前要求的 "
            f"{target_wavelengths[1]} µm。"
        )

    object_distance = float(lens.obj_depth)
    if not math.isclose(
        object_distance,
        spec.object_distance_mm,
        rel_tol=1e-9,
        abs_tol=1e-3,
    ):
        raise ValueError(
            f"输入处方默认物距 {object_distance} mm 与当前规格 "
            f"{spec.object_distance_mm} mm 不一致。"
        )

    if getattr(lens, "aper_idx", None) != 0:
        raise ValueError("分阶段 MWIR 优化要求第 0 面仍为前置孔径光阑。")

    entrance_pupil_diameter = 2.0 * _detached_float(lens.entr_pupilr)
    if not math.isclose(
        entrance_pupil_diameter,
        spec.entrance_pupil_diameter_mm,
        rel_tol=1e-4,
        abs_tol=1e-3,
    ):
        raise ValueError(
            f"输入处方入瞳直径 {entrance_pupil_diameter:.6f} mm 与当前规格 "
            f"{spec.entrance_pupil_diameter_mm:.6f} mm 不一致。"
        )

    sensor_radius = float(lens.r_sensor)
    target_sensor_radius = float(
        design_params["optimization_radial_image_height_mm"]
    )
    if not math.isclose(
        sensor_radius, target_sensor_radius, rel_tol=1e-4, abs_tol=0.05
    ):
        raise ValueError(
            f"输入处方焦面半径 {sensor_radius:.6f} mm 与当前任务焦面半径 "
            f"{target_sensor_radius:.6f} mm 不一致。"
        )

    sensor_res = tuple(int(value) for value in lens.sensor_res)
    target_sensor_res = tuple(int(value) for value in design_params["sensor_res"])
    if sensor_res != target_sensor_res:
        raise ValueError(
            f"输入处方焦面分辨率 {sensor_res} 与当前任务 {target_sensor_res} 不一致。"
        )

    element_count = _count_refractive_elements(lens)
    if not 1 <= element_count <= spec.max_lenses:
        raise ValueError(
            f"输入处方包含 {element_count} 片折射元件，超出 1–{spec.max_lenses} 片范围。"
        )

    return {
        "primary_wavelength_um": primary_wavelength,
        "wavelengths_um": list(loaded_wavelengths),
        "object_distance_mm": object_distance,
        "entrance_pupil_diameter_mm": entrance_pupil_diameter,
        "sensor_radius_mm": sensor_radius,
        "sensor_res": list(sensor_res),
        "element_count": element_count,
        "front_stop": True,
    }


def _detached_float(value: Any) -> float:
    """将标量张量安全转换为 Python 浮点数，不保留自动微分关系。"""

    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach()
    item = getattr(value, "item", None)
    if item is not None:
        value = item()
    return float(value)


def _calibrate_initial_power(
    lens,
    target_focal_length_mm: float,
    *,
    max_iterations: int = 5,
    logarithmic_tolerance: float = 0.01,
    minimum_factor: float = 0.5,
    maximum_factor: float = 2.0,
) -> float:
    """按实际近轴焦距缩放起点曲率，避免长焦起点退化成短焦或超长焦。

    ``create_lens`` 的随机曲率只是拓扑初始化，不保证组合后的有效焦距。
    对本系统先测量一次实际焦距，再按近轴光焦度反比关系缩放各个曲面曲率，
    让后续梯度优化从正确的数量级开始。返回校准后的实际焦距。
    """

    import math
    import torch

    if max_iterations <= 0:
        raise ValueError("max_iterations 必须为正整数。")
    if not math.isfinite(logarithmic_tolerance) or logarithmic_tolerance <= 0.0:
        raise ValueError("logarithmic_tolerance 必须为正的有限值。")
    if (
        not math.isfinite(minimum_factor)
        or not math.isfinite(maximum_factor)
        or minimum_factor <= 0.0
        or maximum_factor < minimum_factor
    ):
        raise ValueError("曲率校准倍率范围必须满足 0 < minimum_factor <= maximum_factor。")

    for _ in range(max_iterations):
        lens.post_computation()
        actual = abs(float(lens.foclen))
        if not math.isfinite(actual) or actual <= 0.0:
            break
        ratio = actual / target_focal_length_mm
        if abs(math.log(ratio)) < logarithmic_tolerance:
            break
        # 限制单次倍率，避免一次修正把表面推入强非线性区域。
        factor = min(max(ratio, minimum_factor), maximum_factor)
        with torch.no_grad():
            for surface in lens.surfaces:
                if hasattr(surface, "c"):
                    surface.c.mul_(factor)

    lens.post_computation()
    return float(lens.foclen)


def build_initial_lens(
    spec: MWIRDesignSpec,
    scheme: str = "transmission_baseline",
    result_dir: str | os.PathLike[str] = "./results/mwir-initial",
    device: str = "auto",
    analyze: bool = False,
):
    """生成 MWIR GeoLens 初始结构。"""

    # 延迟导入，使 --check-only 在未安装完整图像依赖时仍可运行。
    import torch

    from deeplens import GeoLens
    from deeplens.geolens_pkg import create_lens
    from deeplens.material import Material
    from deeplens.utils import set_logger, set_seed

    params = _scheme_parameters(spec, scheme)
    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    set_seed(0)
    set_logger(str(result_path))
    logging.info("使用设备：%s", torch_device)
    logging.info("%s", params["explanation"])
    scheme_has_aperture_conflict = params["f_number"] < spec.physical_f_number_floor
    if scheme_has_aperture_conflict:
        logging.warning(
            "当前方案的一阶 F/%0.3f 低于空气中理想 F/0.5 下限；"
            "该结果仅用于暴露规格冲突，不应作为可制造处方。",
            params["f_number"],
        )

    balanced_power_design: dict[str, Any] | None = None
    power_bent7_design: dict[str, Any] | None = None
    active_surface_list = MWIR_SURFACE_LIST
    if scheme == "transmission_power_bent7":
        active_surface_list = MWIR_POWER_BENT7_SURFACE_LIST
        lens = _build_power_bent7_lens(spec, params, torch_device)
        curvature_scale = max(
            abs(_detached_float(surface.c))
            for surface in lens.surfaces
            if hasattr(surface, "c")
        )
        power_bent7_design = {
            "architecture": "前正组—中负组—后正组—弯月场平镜",
            "status": "非退化球面优化母型；不是最终验收处方。",
            "front_positions_mm": list(MWIR_POWER_BENT7_FRONT_POSITIONS_MM),
            "center_thicknesses_mm": list(
                MWIR_POWER_BENT7_CENTER_THICKNESSES_MM
            ),
            "semi_apertures_mm": list(MWIR_POWER_BENT7_SEMI_APERTURES_MM),
            "element_materials": list(MWIR_POWER_BENT7_MATERIALS),
            "surface_radii_mm_before_power_calibration": [
                list(pair) for pair in MWIR_POWER_BENT7_RADII_MM
            ],
            "reserved_aspheric_element_sides": sorted(
                [list(value) for value in MWIR_POWER_BENT7_ASPHERIC_SIDES]
            ),
            "reserved_even_asphere_orders": [4, 6, 8, 10],
        }
        logging.info("七片强弯曲母型最大初始曲率：%.3e 1/mm", curvature_scale)
    elif scheme == "transmission_balanced":
        balanced_power_design = _balanced_power_design(spec)
        active_surface_list = MWIR_BALANCED_SURFACE_LIST
        lens = _build_balanced_transmission_lens(
            spec,
            params,
            torch_device,
            balanced_power_design,
        )
        curvature_scale = max(
            abs(_detached_float(surface.c))
            for surface in lens.surfaces
            if hasattr(surface, "c")
        )
        logging.info(
            "六片目标光焦度 [1/mm]：%s",
            [
                f"{value:+.6e}"
                for value in balanced_power_design["element_powers_1_per_mm"]
            ],
        )
        logging.info("最大初始曲率：%.3e 1/mm", curvature_scale)
    else:
        # 让随机起点的总近轴光焦度大致落在目标焦距附近。
        # 原始可见光示例的 1e-3 曲率尺度会把本长焦系统初始化成短焦镜头。
        material_deltas = [
            float(Material(name).refractive_index(3.5)) - 1.0
            for name in MWIR_MATERIALS
        ]
        curvature_scale = 1.0 / (
            params["focal_length_mm"] * max(sum(material_deltas), 1e-6)
        )
        curvature_scale = min(max(curvature_scale, 1e-6), 1e-3)
        logging.info("初始曲率尺度：%.3e 1/mm", curvature_scale)

        lens = create_lens(
            # create_lens/GeoLens 的像高和视场是径向（半对角）定义。显式传入
            # 目标焦距和由焦面半对角推导的径向视场，避免把 Y 半像高误当成
            # 矩形探测器半对角后改变有效焦距。
            foclen=params["focal_length_mm"],
            fov=params["optimization_radial_fov_deg"],
            fnum=params["f_number"],
            bfl=params["bfl_mm"],
            thickness=params["thickness_mm"],
            surf_list=MWIR_SURFACE_LIST,
            save_dir=str(result_path),
            primary_wvln=3.5,
            wvln_rgb=list(spec.wavelengths_um),
            obj_depth=spec.object_distance_mm,
            material_names=MWIR_MATERIALS,
            sensor_res=tuple(params["sensor_res"]),
            analyze=False,
            curvature_scale=curvature_scale,
        )
        lens = lens.to(torch_device)
        lens.set_sensor(tuple(params["sensor_size_mm"]), tuple(params["sensor_res"]))
    # 固定前置光阑半径，使真实入瞳直径等于任务指标。随后按实测近轴焦距
    # 对全部曲率做小步比例校准，让随机拓扑从目标光焦度附近开始；这只校准
    # 一阶光焦度，不代表像差、畸变或 MTF 已经满足要求。
    lens.surfaces[lens.aper_idx].update_r(
        spec.entrance_pupil_diameter_mm / 2.0
    )
    lens.post_computation()
    uncalibrated_focal_length = float(lens.foclen)
    if scheme == "transmission_power_bent7":
        calibrated_focal_length = _calibrate_initial_power(
            lens,
            target_focal_length_mm=params["focal_length_mm"],
            max_iterations=12,
            logarithmic_tolerance=1e-6,
            minimum_factor=0.8,
            maximum_factor=1.2,
        )
    else:
        calibrated_focal_length = _calibrate_initial_power(
            lens, target_focal_length_mm=params["focal_length_mm"]
        )
    logging.info(
        "初始光焦度校准：%.4f mm -> %.4f mm（目标 %.4f mm）。",
        uncalibrated_focal_length,
        calibrated_focal_length,
        params["focal_length_mm"],
    )
    sensor_distance_before_refocus = float(lens.d_sensor)
    lens.refocus(float("inf"))
    lens.post_computation()
    measured_focal_length = float(lens.foclen)
    measured_f_number = float(lens.fnum)
    measured_enpd = float(2.0 * lens.entr_pupilr)
    measured_half_field_deg = float(lens.rfov * 180.0 / math.pi)
    # GeoLens 的默认大镜头约束仍以约 300 mm 总长、20 mm 玻璃厚度为上限。
    # 本任务暂不约束总长，而且 280 mm 口径元件需要更宽松的厚度包络；使用
    # 可持久化覆盖，确保后续 post_computation() 不会把这些值重置。
    _apply_mwir_constraints(lens, spec)
    metadata = {
        "spec": spec.geometry_report(),
        "design": params,
        "wavelengths_um": list(spec.wavelengths_um),
        "material_pool": MWIR_MATERIALS,
        "materials": sorted(set(_element_material_names(lens))),
        "element_materials": _element_material_names(lens),
        "surface_list": active_surface_list,
        "curvature_scale_1_per_mm": curvature_scale,
        "initial_power_calibration": {
            "method": (
                "按实测近轴焦距比例缩放全部可曲面曲率；七片强弯曲母型使用"
                "更严格的 12 次/1e-6 对数误差校准，其他方案保持原快速校准。"
            ),
            "before_focal_length_mm": uncalibrated_focal_length,
            "after_focal_length_mm": measured_focal_length,
            "target_focal_length_mm": params["focal_length_mm"],
            "sensor_distance_before_refocus_mm": sensor_distance_before_refocus,
            "sensor_distance_after_refocus_mm": float(lens.d_sensor),
            "focus_conjugate": "infinity",
            "scope": "仅一阶光焦度校准；不表示像质验收通过。",
        },
        "measured_initial_focal_length_mm": measured_focal_length,
        "measured_initial_f_number": measured_f_number,
        "measured_initial_entrance_pupil_diameter_mm": measured_enpd,
        "measured_initial_half_field_deg": measured_half_field_deg,
        "device": str(torch_device),
        "warning": "当前模型未完整包含材料吸收、镀膜、热光系数、热膨胀和机械公差。",
        "field_definition": (
            "用户指标为 Y 方向全视场；GeoLens 按焦面半对角对应的径向场采样。"
        ),
        "detector_is_hard_constraint": spec.detector_is_known
        and not params["sensor_is_virtual"],
        "optimization_envelope": {
            "total_track_hard_constraint_mm": None,
            "air_gap_max_mm": lens.air_center_max,
            "element_thickness_max_mm": lens.thick_center_max,
            "bfl_max_mm": lens.bfl_max,
        },
        "physical_feasibility_warning": (
            "F 数低于空气中理想 F/0.5 下限；"
            "需要减小有效入瞳或增大焦面后再进行正式设计。"
            if scheme_has_aperture_conflict
            else None
        ),
    }
    if balanced_power_design is not None:
        metadata["balanced_achromat"] = {
            **balanced_power_design,
            "front_positions_mm": list(MWIR_BALANCED_FRONT_POSITIONS_MM),
            "center_thicknesses_mm": list(
                MWIR_BALANCED_CENTER_THICKNESSES_MM
            ),
            "semi_apertures_mm": list(MWIR_BALANCED_SEMI_APERTURES_MM),
            "positive_power_conic": MWIR_BALANCED_POSITIVE_CONIC,
            "negative_power_conic": MWIR_BALANCED_NEGATIVE_CONIC,
            "asphere_orders": [4, 6, 8, 10, 12, 14, 16],
        }
    if power_bent7_design is not None:
        metadata["power_bent7"] = power_bent7_design
    with open(result_path / "mwir_design_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    lens.write_lens_json(str(result_path / "mwir_initial.json"))
    if analyze:
        lens.analysis(
            save_name=str(result_path / "mwir_initial_analysis"),
            depth=spec.object_distance_mm,
            full_eval=True,
        )
    return lens, params, result_path


def load_lens_for_stage(
    spec: MWIRDesignSpec,
    input_lens: str | os.PathLike[str],
    scheme: str = "transmission_baseline",
    result_dir: str | os.PathLike[str] = "./results/mwir-stage",
    device: str = "auto",
    analyze: bool = False,
    allow_retarget: bool = False,
):
    """从已有 JSON 处方开始新的优化阶段，不恢复旧 Adam 状态。"""

    import torch

    from deeplens import GeoLens
    from deeplens.utils import set_logger, set_seed

    input_path = Path(input_lens)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到输入处方：{input_path}")
    if input_path.suffix.lower() != ".json":
        raise ValueError("--input-lens 当前只接受 DeepLens 原生 JSON 处方。")

    result_path = Path(result_dir)
    if result_path.exists() and any(result_path.iterdir()):
        raise ValueError("分阶段续跑要求新的空输出目录，不能覆盖已有阶段结果。")
    result_path.mkdir(parents=True, exist_ok=True)
    initial_copy = result_path / "mwir_initial.json"
    input_resolved = input_path.resolve()
    result_resolved = result_path.resolve()
    try:
        input_resolved.relative_to(result_resolved)
    except ValueError:
        pass
    else:
        raise ValueError("新的结果目录必须与输入处方所在目录分开，避免覆盖原始阶段文件。")

    params = _scheme_parameters(spec, scheme)
    source_metadata_path: Path | None = None
    source_design_checks: dict[str, float] | None = None
    retarget_warning: str | None = None
    try:
        source_metadata_path, source_metadata = _find_source_design_metadata(
            input_path
        )
        source_design_checks = _validate_source_design_metadata(
            source_metadata, params
        )
    except Exception as error:
        if not allow_retarget:
            raise ValueError(
                f"无法确认输入处方与当前设计目标相同：{error} "
                "如确实要改变目标，请显式加入 --allow-retarget。"
            ) from error
        retarget_warning = str(error)

    if device == "auto":
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch_device = torch.device(device)

    set_seed(0)
    set_logger(str(result_path))
    logging.info("从已有处方开始新阶段：%s", input_resolved)
    logging.info("使用设备：%s", torch_device)
    if retarget_warning is not None:
        logging.warning("已显式允许改变原设计目标：%s", retarget_warning)

    lens = GeoLens(filename=str(input_path), device=torch_device)
    checks = _validate_loaded_mwir_lens(lens, spec, params)
    params["element_count"] = checks["element_count"]
    params["surface_count"] = len(lens.surfaces)
    _apply_mwir_constraints(lens, spec)

    metadata = {
        "spec": spec.geometry_report(),
        "design": params,
        "wavelengths_um": list(spec.wavelengths_um),
        "material_pool": MWIR_MATERIALS,
        "materials": sorted(set(_element_material_names(lens))),
        "element_materials": _element_material_names(lens),
        "source_lens_json": str(input_resolved),
        "source_design_metadata": (
            None if source_metadata_path is None else str(source_metadata_path.resolve())
        ),
        "source_design_checks": source_design_checks,
        "allow_retarget": allow_retarget,
        "retarget_warning": retarget_warning,
        "optimizer_reset": True,
        "resume_mode": "只恢复光学处方；重新创建 Adam 与学习率调度器。",
        "power_recalibrated": False,
        "refocused_on_load": False,
        "load_checks": checks,
        "device": str(torch_device),
        "warning": "当前模型未完整包含材料吸收、镀膜、热光系数、热膨胀和机械公差。",
    }
    with open(result_path / "mwir_design_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    lens.write_lens_json(str(initial_copy))
    if analyze:
        lens.analysis(
            save_name=str(result_path / "mwir_initial_analysis"),
            depth=spec.object_distance_mm,
            full_eval=True,
        )
    return lens, params, result_path


def _geometric_mtf_from_intercepts(
    intercepts_xy, frequency_cy_mm: float
) -> tuple[float, float]:
    """由光线截距在单一空间频率处计算几何切向/弧矢 MTF。

    直接计算经验光学传递函数可避免像元尺度 PSF 分箱、窗口裁剪以及频率轴
    插值。Y 截距对应切向（子午）方向，X 截距对应弧矢方向。
    """

    import numpy as np

    try:
        points = intercepts_xy.detach().cpu().numpy()
    except AttributeError:
        points = np.asarray(intercepts_xy)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("计算几何 MTF 至少需要两条有效二维光线截距。")
    if not np.isfinite(points).all():
        raise ValueError("光线截距包含 NaN 或 Inf。")
    if not math.isfinite(frequency_cy_mm) or frequency_cy_mm < 0.0:
        raise ValueError("MTF 空间频率必须为非负有限值。")

    centered = points - points.mean(axis=0, keepdims=True)
    phase_x = -2j * math.pi * frequency_cy_mm * centered[:, 0]
    phase_y = -2j * math.pi * frequency_cy_mm * centered[:, 1]
    mtf_sagittal = float(abs(np.exp(phase_x).mean()))
    mtf_tangential = float(abs(np.exp(phase_y).mean()))
    return mtf_tangential, mtf_sagittal


def _circular_diffraction_mtf(
    frequency_cy_mm: float, wavelength_um: float, f_number: float
) -> float:
    """无中心遮拦圆孔在指定频率处的非相干衍射 MTF。"""

    if frequency_cy_mm < 0.0 or wavelength_um <= 0.0 or f_number <= 0.0:
        raise ValueError("频率、波长和 F 数必须位于物理有效范围。")
    cutoff = 1.0 / (wavelength_um * 1e-3 * f_number)
    normalized = frequency_cy_mm / cutoff
    if normalized >= 1.0:
        return 0.0
    root = math.sqrt(max(0.0, 1.0 - normalized**2))
    return (2.0 / math.pi) * (
        math.acos(normalized) - normalized * root
    )


def _rectangular_pixel_mtf(
    frequency_cy_mm: float, active_width_mm: float
) -> float:
    """矩形像元孔径在单轴频率处的 MTF；默认按 100% 填充率估算。"""

    if frequency_cy_mm < 0.0 or active_width_mm <= 0.0:
        raise ValueError("频率必须非负，像元有效宽度必须为正。")
    argument = math.pi * frequency_cy_mm * active_width_mm
    return 1.0 if abs(argument) < 1e-15 else abs(math.sin(argument) / argument)


def _linear_intercept(x_values, y_values) -> float:
    """以 float64 最小二乘拟合 ``y = intercept + slope * x``。"""

    import numpy as np

    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("线性截距拟合需要至少两组一维等长样本。")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("线性截距拟合样本包含 NaN 或 Inf。")
    design = np.stack([np.ones_like(x), x], axis=-1)
    intercept, _ = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept)


def _paraxial_focus_by_plane(
    lens,
    wavelength_um: float,
    pupil_radius_ratios: tuple[float, ...] = (1e-3, 2e-3, 4e-3),
) -> dict[str, float]:
    """用对称微小瞳高轴上光线外推子午/弧矢高斯焦面。"""

    import numpy as np
    import torch

    from deeplens.light import Ray

    if len(pupil_radius_ratios) < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in pupil_radius_ratios
    ):
        raise ValueError("近轴焦面至少需要两个正的有限瞳高比例。")

    pupil_radius = _detached_float(lens.entr_pupilr)
    if not math.isfinite(pupil_radius) or pupil_radius <= 0.0:
        raise ValueError("实际入瞳半径必须为正的有限值。")
    radii = np.asarray(pupil_radius_ratios, dtype=np.float64) * pupil_radius
    start_z = _detached_float(lens.surfaces[0].d) - 1.0
    dtype = getattr(lens, "dtype", torch.float32)
    focus_by_plane: dict[str, float] = {}

    for plane, coordinate in (("meridional", 1), ("sagittal", 0)):
        origins = torch.zeros((2 * len(radii), 3), device=lens.device, dtype=dtype)
        directions = torch.zeros_like(origins)
        directions[:, 2] = 1.0
        origins[:, 2] = start_z
        for index, radius in enumerate(radii):
            origins[2 * index, coordinate] = -float(radius)
            origins[2 * index + 1, coordinate] = float(radius)

        with torch.no_grad():
            ray = Ray(
                origins,
                directions,
                wvln=wavelength_um,
                device=lens.device,
            )
            ray, _ = lens.trace(ray)
        valid = (ray.is_valid > 0).detach().cpu().numpy()
        if not valid.all():
            raise ValueError(f"{plane} 近轴轴上光线未全部通过系统。")

        ray_o = ray.o.detach().cpu().numpy().astype(np.float64, copy=False)
        ray_d = ray.d.detach().cpu().numpy().astype(np.float64, copy=False)
        slope = ray_d[:, coordinate] / ray_d[:, 2]
        if np.any(np.abs(slope) < 1e-12):
            raise ValueError(f"{plane} 近轴出射斜率过小，无法求高斯焦面。")
        crossings = ray_o[:, 2] - ray_o[:, coordinate] / slope
        paired_crossings = 0.5 * (crossings[0::2] + crossings[1::2])
        focus_z = _linear_intercept(radii**2, paired_crossings)
        if not math.isfinite(focus_z) or focus_z <= _detached_float(
            lens.surfaces[-1].d
        ):
            raise ValueError(f"{plane} 高斯焦面不在最后光学面之后。")
        focus_by_plane[plane] = focus_z

    return focus_by_plane


def _chief_ray_scale_at_plane(
    lens,
    wavelength_um: float,
    image_plane_z_by_plane: dict[str, float],
    field_angles_deg: tuple[float, ...] = (0.05, 0.10, 0.20),
) -> dict[str, float]:
    """由正负小视场主光线外推指定像面上的零视场板尺。"""

    import numpy as np
    import torch

    from deeplens.light import Ray

    if getattr(lens, "aper_idx", None) != 0:
        raise ValueError("当前主光线板尺算法要求第 0 面为前置孔径光阑。")
    if len(field_angles_deg) < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in field_angles_deg
    ):
        raise ValueError("主光线板尺至少需要两个正的有限小视场角。")

    positive_angles = np.asarray(field_angles_deg, dtype=np.float64)
    signed_angles = np.column_stack((-positive_angles, positive_angles)).reshape(-1)
    angle_tensor = torch.as_tensor(
        signed_angles,
        device=lens.device,
        dtype=getattr(lens, "dtype", torch.float32),
    )
    tangent = np.tan(np.deg2rad(positive_angles))
    scale_by_plane: dict[str, float] = {}

    for plane, coordinate in (("meridional", 1), ("sagittal", 0)):
        chief_o, chief_d = lens.calc_chief_ray_infinite(
            rfov=angle_tensor,
            wvln=wavelength_um,
            plane=plane,
            ray_aiming=False,
        )
        with torch.no_grad():
            ray = Ray(chief_o, chief_d, wvln=wavelength_um, device=lens.device)
            ray, _ = lens.trace(ray)
        if not (ray.is_valid > 0).all():
            raise ValueError(f"{plane} 至少一条近轴主光线未通过系统。")

        plane_z = float(image_plane_z_by_plane[plane])
        propagation = (plane_z - ray.o[..., 2]) / ray.d[..., 2]
        coordinate_at_plane = (
            ray.o[..., coordinate] + ray.d[..., coordinate] * propagation
        )
        coordinate_np = (
            coordinate_at_plane.detach().cpu().numpy().astype(np.float64, copy=False)
        )
        odd_height = 0.5 * (coordinate_np[1::2] - coordinate_np[0::2])
        local_scales = odd_height / tangent
        focal_scale = _linear_intercept(tangent**2, local_scales)
        if not math.isfinite(focal_scale) or focal_scale <= 0.0:
            raise ValueError(f"{plane} 主光线板尺必须为正的有限值。")
        scale_by_plane[plane] = focal_scale

    return scale_by_plane


def _chief_ray_effective_focal_length(
    lens, wavelength_um: float = 3.5
) -> dict[str, Any]:
    """在高斯焦面上计算严格 EFL，并返回两平面近轴诊断。"""

    focus_by_plane = _paraxial_focus_by_plane(lens, wavelength_um)
    focal_length_by_plane = _chief_ray_scale_at_plane(
        lens,
        wavelength_um=wavelength_um,
        image_plane_z_by_plane=focus_by_plane,
    )
    return {
        "effective_focal_length_by_plane_mm": focal_length_by_plane,
        "effective_focal_length_mean_mm": sum(focal_length_by_plane.values())
        / len(focal_length_by_plane),
        "paraxial_focus_z_by_plane_mm": focus_by_plane,
    }


def _chief_ray_sensor_plate_scale(
    lens, wavelength_um: float = 3.5
) -> dict[str, float]:
    """计算当前传感器面上的局部主光线板尺；该量不是严格 EFL。"""

    sensor_z = _detached_float(lens.d_sensor)
    return _chief_ray_scale_at_plane(
        lens,
        wavelength_um=wavelength_um,
        image_plane_z_by_plane={
            "meridional": sensor_z,
            "sagittal": sensor_z,
        },
    )


def _chief_ray_image_heights(
    lens,
    half_field_deg: float,
    wavelength_um: float,
    plane: str,
    num_points: int = 9,
):
    """在固定目标角度上追迹主光线并返回实际像高。"""

    import numpy as np
    import torch

    from deeplens.light import Ray

    if num_points < 2:
        raise ValueError("主光线像高评价至少需要两个视场点。")
    field_angles = torch.linspace(
        0.0, half_field_deg, num_points, device=lens.device
    )
    chief_o, chief_d = lens.calc_chief_ray_infinite(
        rfov=field_angles,
        wvln=wavelength_um,
        plane=plane,
        ray_aiming=True,
    )
    ray = Ray(chief_o, chief_d, wvln=wavelength_um, device=lens.device)
    ray, _ = lens.trace(ray)
    if not (ray.is_valid > 0).all():
        raise ValueError("至少一个目标视场的主光线未能到达像面。")
    if not torch.isfinite(ray.d[..., 2]).all() or (
        ray.d[..., 2].abs() < 1e-12
    ).any():
        raise ValueError("至少一条主光线与像面近似平行或具有非有限轴向方向。")
    propagation = (lens.d_sensor - ray.o[..., 2]) / ray.d[..., 2]
    coordinate = 0 if plane == "sagittal" else 1
    actual_height = (
        ray.o[..., coordinate] + ray.d[..., coordinate] * propagation
    ).abs()
    if not torch.isfinite(actual_height).all():
        raise ValueError("主光线像高包含 NaN 或 Inf。")
    return (
        field_angles.detach().cpu().numpy(),
        field_angles.detach().cpu().numpy(),
        actual_height.detach().cpu().numpy(),
    )


def evaluate_lens(
    lens,
    spec: MWIRDesignSpec,
    result_dir: Path,
    psf_spp: int = 512,
    psf_ks: int = 64,
    vignetting_grid: int = 9,
    vignetting_rays: int = 128,
) -> dict[str, Any]:
    """计算 MTF、像高/畸变、渐晕、入瞳和一阶参数验收结果。

    ``psf_ks`` 为旧接口兼容参数；当前实现直接由光线截距计算单频几何 OTF，
    不再生成可能被窗口裁剪或数值分箱低通污染的 PSF 图。
    """

    import numpy as np

    lens.post_computation()
    lens.distortion_max = spec.distortion_limit
    # 探测器未确认时，MTF 评价使用虚拟仿真像元间距；结果只能作为初始结构
    # 的数值检查，不能替代最终探测器奈奎斯特验收。
    nyquist = spec.analysis_nyquist_frequency_cy_mm
    target_focal_length = spec.required_focal_length_mm
    target_f_number = spec.required_f_number
    entrance_pupil_diameter = 2.0 * _detached_float(lens.entr_pupilr)
    cached_focal_length = float(lens.foclen)
    cached_f_number = float(lens.fnum)
    sensor_z = _detached_float(lens.d_sensor)

    first_order_by_wavelength: dict[float, dict[str, Any]] = {}
    first_order_metrics: dict[str, Any] = {}
    first_order_errors: list[str] = []
    for wavelength in spec.wavelengths_um:
        try:
            strict_result = _chief_ray_effective_focal_length(
                lens, wavelength_um=wavelength
            )
            focal_length_by_plane = strict_result[
                "effective_focal_length_by_plane_mm"
            ]
            focus_by_plane = strict_result["paraxial_focus_z_by_plane_mm"]
            plate_scale_by_plane = _chief_ray_sensor_plate_scale(
                lens, wavelength_um=wavelength
            )
            f_number_by_plane = {
                plane: value / entrance_pupil_diameter
                for plane, value in focal_length_by_plane.items()
            }
            result = {
                "effective_focal_length_by_plane_mm": focal_length_by_plane,
                "effective_focal_length_mean_mm": strict_result[
                    "effective_focal_length_mean_mm"
                ],
                "f_number_by_plane": f_number_by_plane,
                "f_number_mean": sum(f_number_by_plane.values())
                / len(f_number_by_plane),
                "paraxial_focus_z_by_plane_mm": focus_by_plane,
                "sensor_minus_paraxial_focus_mm": {
                    plane: sensor_z - value for plane, value in focus_by_plane.items()
                },
                "sensor_plate_scale_by_plane_mm": plate_scale_by_plane,
                "sensor_plate_scale_mean_mm": sum(plate_scale_by_plane.values())
                / len(plate_scale_by_plane),
            }
            first_order_by_wavelength[float(wavelength)] = result
            first_order_metrics[str(wavelength)] = result
        except Exception as error:
            message = f"一阶近轴评价失败：波长 {wavelength} 微米：{error}"
            first_order_errors.append(message)
            first_order_metrics[str(wavelength)] = {"error": str(error)}

    nominal_wavelength = min(
        spec.wavelengths_um, key=lambda value: abs(value - 3.5)
    )
    nominal_first_order = first_order_by_wavelength.get(float(nominal_wavelength))
    if nominal_first_order is None:
        focal_length_by_plane: dict[str, float] = {}
        f_number_by_plane: dict[str, float] = {}
        focal_length = float("nan")
        f_number = float("nan")
        focal_length_error_by_plane: dict[str, float] = {}
        f_number_error_by_plane: dict[str, float] = {}
        focal_length_error = float("inf")
        f_number_error = float("inf")
    else:
        focal_length_by_plane = nominal_first_order[
            "effective_focal_length_by_plane_mm"
        ]
        f_number_by_plane = nominal_first_order["f_number_by_plane"]
        focal_length = nominal_first_order["effective_focal_length_mean_mm"]
        f_number = nominal_first_order["f_number_mean"]
        focal_length_error_by_plane = {
            plane: abs(value / target_focal_length - 1.0)
            for plane, value in focal_length_by_plane.items()
        }
        f_number_error_by_plane = {
            plane: abs(value / target_f_number - 1.0)
            for plane, value in f_number_by_plane.items()
        }
        focal_length_error = max(focal_length_error_by_plane.values())
        f_number_error = max(f_number_error_by_plane.values())

    metrics: dict[str, Any] = {
        "nyquist_frequency_cy_mm": nyquist,
        "nyquist_frequency_is_provisional": not spec.detector_pitch_known,
        "field_definition": "Y 方向全视场",
        "field_y_deg": spec.full_field_y_deg,
        "field_symmetry_assumption": "当前 GeoLens 处方仅含同轴旋转对称面，正半场代表负半场。",
        "evaluation_object_distance": "infinity",
        "focal_length_mm": focal_length,
        "focal_length_by_plane_mm": focal_length_by_plane,
        "focal_length_method": "高斯近轴焦面上的正负小视场主光线零角外推",
        "target_focal_length_mm": target_focal_length,
        "focal_length_relative_error": focal_length_error,
        "focal_length_relative_error_by_plane": focal_length_error_by_plane,
        "f_number": f_number,
        "f_number_by_plane": f_number_by_plane,
        "target_f_number": target_f_number,
        "f_number_relative_error": f_number_error,
        "f_number_relative_error_by_plane": f_number_error_by_plane,
        "pass_uses_worst_plane": True,
        "geolens_cached_focal_length_mm": cached_focal_length,
        "geolens_cached_f_number": cached_f_number,
        "first_order_by_wavelength": first_order_metrics,
        "entrance_pupil_diameter_mm": entrance_pupil_diameter,
        "total_track_length_mm": _detached_float(
            lens.d_sensor - lens.surfaces[0].d
        ),
        "sensor_size_mm": [float(value) for value in lens.sensor_size],
        "sensor_res": list(lens.sensor_res),
        "lens_count": _count_refractive_elements(lens),
        "mtf": {},
        "distortion": {},
        "vignetting": {},
        "errors": first_order_errors,
    }

    mtf_values = []
    expected_mtf_values = len(spec.wavelengths_um) * 3
    for wavelength in spec.wavelengths_um:
        wavelength_result = {}
        for relative_fov in (0.0, 0.7, 1.0):
            try:
                # 按任务给定的物方 Y 视场角直接采样无穷远平行光，避免虚拟
                # 正方形焦面的归一化 Y=1 只覆盖到约 3.4°。
                field_y_deg = relative_fov * spec.half_field_y_deg
                ray = lens.sample_from_fov(
                    fov_x=0.0,
                    fov_y=-field_y_deg,
                    depth=float("inf"),
                    num_rays=psf_spp,
                    wvln=wavelength,
                )
                ray.is_coherent = False
                ray = lens.trace2sensor(ray)
                valid = ray.is_valid > 0
                valid_ratio = float(valid.float().mean().item())
                if valid_ratio < spec.vignetting_floor:
                    raise ValueError(
                        f"有效光线比例 {valid_ratio:.3f} 低于最低要求 "
                        f"{spec.vignetting_floor:.3f}"
                    )
                intercepts = ray.o[..., :2][valid]
                geometric_tan, geometric_sag = _geometric_mtf_from_intercepts(
                    intercepts, nyquist
                )
                wavelength_first_order = first_order_by_wavelength.get(
                    float(wavelength)
                )
                if wavelength_first_order is None:
                    raise ValueError("缺少该波长的严格 EFL/F 数结果。")
                diffraction_tan = _circular_diffraction_mtf(
                    nyquist,
                    wavelength,
                    wavelength_first_order["f_number_by_plane"]["meridional"],
                )
                diffraction_sag = _circular_diffraction_mtf(
                    nyquist,
                    wavelength,
                    wavelength_first_order["f_number_by_plane"]["sagittal"],
                )
                pixel_mtf = _rectangular_pixel_mtf(
                    nyquist, spec.pixel_pitch_mm
                )
                system_tan = geometric_tan * diffraction_tan * pixel_mtf
                system_sag = geometric_sag * diffraction_sag * pixel_mtf
                system_min = min(system_tan, system_sag)
                mtf_values.append(system_min)
                centroid = intercepts.mean(dim=0)
                wavelength_result[str(relative_fov)] = {
                    "field_y_deg": field_y_deg,
                    "valid_ray_ratio": valid_ratio,
                    "centroid_x_mm": float(centroid[0].item()),
                    "centroid_y_mm": float(centroid[1].item()),
                    "geometric_tangential": geometric_tan,
                    "geometric_sagittal": geometric_sag,
                    "ideal_circular_diffraction": min(
                        diffraction_tan, diffraction_sag
                    ),
                    "ideal_circular_diffraction_tangential": diffraction_tan,
                    "ideal_circular_diffraction_sagittal": diffraction_sag,
                    "pixel_aperture_100pct_fill": pixel_mtf,
                    "system_tangential_estimate": system_tan,
                    "system_sagittal_estimate": system_sag,
                    "system_min_estimate": system_min,
                    "method": (
                        "几何光线截距 OTF × 理想圆孔衍射 MTF × "
                        "100% 填充率矩形像元 MTF"
                    ),
                }
            except Exception as error:  # 退化初始结构可能没有足够有效光线。
                message = f"MTF 计算失败：波长 {wavelength} 微米，相对视场 {relative_fov}：{error}"
                metrics["errors"].append(message)
                wavelength_result[str(relative_fov)] = {"error": str(error)}
        metrics["mtf"][str(wavelength)] = wavelength_result

    distortion_values = []
    target_mapping_values = []
    expected_distortion_values = len(spec.wavelengths_um) * 2
    for wavelength in spec.wavelengths_um:
        wavelength_result = {}
        for plane in ("meridional", "sagittal"):
            try:
                field_angles, compute_angles, actual_height = _chief_ray_image_heights(
                    lens,
                    half_field_deg=spec.half_field_y_deg,
                    wavelength_um=wavelength,
                    plane=plane,
                    num_points=9,
                )
                tangent = np.tan(np.deg2rad(compute_angles))
                if not np.isfinite(actual_height).all():
                    raise ValueError("主光线像高包含 NaN 或 Inf。")
                nonzero_field = np.abs(tangent) > 1e-12
                if not nonzero_field.any():
                    raise ValueError("畸变评价没有非零视场点。")
                wavelength_first_order = first_order_by_wavelength.get(
                    float(wavelength)
                )
                if wavelength_first_order is None:
                    raise ValueError("缺少该波长的传感器面主光线板尺。")
                plate_scale = wavelength_first_order[
                    "sensor_plate_scale_by_plane_mm"
                ][plane]
                tangent_nonzero = tangent[nonzero_field]
                actual_nonzero = actual_height[nonzero_field]
                ideal_measured = plate_scale * tangent_nonzero
                ideal_target = target_focal_length * tangent_nonzero
                distortion = (actual_nonzero - ideal_measured) / ideal_measured
                target_mapping_error = (
                    actual_nonzero - ideal_target
                ) / ideal_target
                max_abs = float(np.max(np.abs(distortion)))
                max_target_abs = float(np.max(np.abs(target_mapping_error)))
                distortion_values.append(max_abs)
                target_mapping_values.append(max_target_abs)
                wavelength_result[plane] = {
                    "max_abs_relative": max_abs,
                    "max_abs_percent": max_abs * 100.0,
                    "max_target_mapping_error_relative": max_target_abs,
                    "max_target_mapping_error_percent": max_target_abs * 100.0,
                    "reference_plate_scale_focal_length_mm": plate_scale,
                    "distortion_reference": "同波长、同平面的传感器面近轴主光线板尺",
                    "edge_field_deg": float(field_angles[-1]),
                    "edge_actual_image_height_mm": float(actual_height[-1]),
                    "edge_target_image_height_mm": spec.required_image_height_mm,
                }
            except Exception as error:
                message = f"畸变计算失败：波长 {wavelength} 微米，平面 {plane}：{error}"
                metrics["errors"].append(message)
                wavelength_result[plane] = {"error": str(error)}
        metrics["distortion"][str(wavelength)] = wavelength_result

    try:
        vignetting = lens.vignetting(
            depth=float("inf"),
            num_grid=vignetting_grid,
            num_rays=vignetting_rays,
        )
        vignetting_np = vignetting.detach().cpu().numpy()
        metrics["vignetting"] = {
            "minimum": float(np.nanmin(vignetting_np)),
            "mean": float(np.nanmean(vignetting_np)),
            "corner_minimum": float(
                min(
                    vignetting_np[0, 0],
                    vignetting_np[0, -1],
                    vignetting_np[-1, 0],
                    vignetting_np[-1, -1],
                )
            ),
            "definition": "有效光线比例，不含 cos^4、材料吸收和镀膜损失。",
        }
    except Exception as error:
        metrics["errors"].append(f"渐晕计算失败：{error}")
        metrics["vignetting"] = {"error": str(error)}

    metrics["pass"] = {
        "focal_length": focal_length_error <= spec.focal_length_tolerance,
        "f_number": f_number_error <= spec.f_number_tolerance,
        "target_field_mapping": len(target_mapping_values)
        == expected_distortion_values
        and max(target_mapping_values) <= spec.distortion_limit,
        "system_mtf": len(mtf_values) == expected_mtf_values
        and min(mtf_values) >= spec.system_mtf_threshold,
        "distortion": len(distortion_values) == expected_distortion_values
        and max(distortion_values) <= spec.distortion_limit,
        "vignetting": "minimum" in metrics["vignetting"]
        and metrics["vignetting"]["minimum"] >= spec.vignetting_floor,
        "entrance_pupil": math.isclose(
            metrics["entrance_pupil_diameter_mm"],
            spec.entrance_pupil_diameter_mm,
            rel_tol=0.01,
        ),
        "lens_count": 1 <= metrics["lens_count"] <= spec.max_lenses,
    }
    metrics["pass"]["overall"] = all(metrics["pass"].values())
    with open(result_dir / "mwir_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    return metrics


def optimize_lens(
    lens,
    spec: MWIRDesignSpec,
    result_dir: Path,
    iterations: int,
    design_params: dict[str, Any] | None = None,
    num_ring: int = 8,
    num_arm: int = 4,
    spp: int = 256,
    shape_control: bool = False,
    prune_surfaces: bool = False,
    field_weight: float = 1.0,
    field_max_weight: float = 1.0,
    field_mapping_points: int = 9,
    regularization_weight: float = 0.1,
    rms_weight: float = 1.0,
    mtf_surrogate_weight: float = 0.0,
    mtf_max_weight: float = 1.0,
    ray_resample_interval: int = 1,
    first_order_preferred_error: float = 0.008,
    first_order_hard_error: float = 0.01,
    lrs: tuple[float, float, float, float] = DEFAULT_MWIR_LRS,
    checkpoint_analysis: bool = False,
) -> None:
    """使用 GeoLens 内置 RMS 优化器进行可选的初步优化。

    大口径随机起点的第一阶段默认关闭 ``shape_control``、表面裁剪和检查点
    完整分析，避免在处方尚未稳定时误裁口径或让绘图占据绝大部分 CPU 时间。
    目标像高/场映射使用独立于通用正则项的较高权重，防止 RMS 改善时焦距与
    边缘像高反而漂移。``rms_weight`` 与 ``lrs`` 支持把优化拆成“先稳定板尺、
    后改善像质”的多个阶段；每个阶段都会重新创建 Adam。可选 MTF 代理项在
    正式验收的 0、0.7、1.0 相对 Y 视场上使用固定频率方向方差，避免初始
    MTF 接近有限光线噪声底时直接优化经验 OTF 幅值。训练光线可按固定间隔
    重采样，降低对单一 Monte-Carlo 瞳样本的过拟合。
    """

    if iterations <= 0:
        return
    if len(lrs) != 4 or any(
        not math.isfinite(value) or value < 0.0 for value in lrs
    ):
        raise ValueError("lrs 必须包含 4 个非负有限值：[间距, 曲率, 圆锥常数, 非球面]。")
    if not math.isfinite(rms_weight) or rms_weight < 0.0:
        raise ValueError("rms_weight 必须为非负有限值。")
    if not math.isfinite(mtf_surrogate_weight) or mtf_surrogate_weight < 0.0:
        raise ValueError("mtf_surrogate_weight 必须为非负有限值。")
    if not math.isfinite(mtf_max_weight) or mtf_max_weight < 0.0:
        raise ValueError("mtf_max_weight 必须为非负有限值。")
    if ray_resample_interval < 0:
        raise ValueError("ray_resample_interval 必须为非负整数。")
    if (
        not math.isfinite(first_order_preferred_error)
        or not math.isfinite(first_order_hard_error)
        or first_order_preferred_error <= 0.0
        or first_order_hard_error < first_order_preferred_error
    ):
        raise ValueError(
            "一阶误差门限必须满足 0 < first_order_preferred_error "
            "<= first_order_hard_error。"
        )

    if design_params is None:
        target_focal_length = spec.required_focal_length_mm
        target_rfov = math.radians(spec.half_field_y_deg)
    else:
        target_focal_length = float(design_params["focal_length_mm"])
        target_rfov = math.radians(
            float(design_params["optimization_radial_fov_deg"]) / 2.0
        )
    test_interval = max(1, min(100, iterations, max(10, iterations // 10)))
    mtf_frequency = spec.analysis_nyquist_frequency_cy_mm
    worst_ideal_factor = min(
        _circular_diffraction_mtf(
            mtf_frequency, wavelength, spec.required_f_number
        )
        * _rectangular_pixel_mtf(mtf_frequency, spec.pixel_pitch_mm)
        for wavelength in spec.wavelengths_um
    )
    if worst_ideal_factor <= 0.0:
        raise ValueError("目标频率已超出衍射截止频率，无法设置 MTF 优化目标。")
    geometric_mtf_target = max(
        spec.optical_mtf_target,
        spec.system_mtf_threshold / worst_ideal_factor,
    )
    if geometric_mtf_target >= 1.0:
        raise ValueError("系统 MTF 指标要求的几何 MTF 不小于 1，当前规格不可行。")

    lens.optimize(
        # 曲率本身约为 1e-4 1/mm，沿用手机镜头的 1e-4 学习率会在一步内
        # 改变整个曲面符号；MWIR 大口径起点使用更保守的曲率和非球面步长。
        lrs=list(lrs),
        iterations=iterations,
        test_per_iter=test_interval,
        optim_mat=False,
        shape_control=shape_control,
        sample_more_off_axis=True,
        num_ring=num_ring,
        num_arm=num_arm,
        spp=spp,
        min_valid_ratio=spec.vignetting_floor,
        w_rms=rms_weight,
        w_mtf=mtf_surrogate_weight,
        mtf_frequency_cy_mm=mtf_frequency,
        mtf_target=geometric_mtf_target,
        mtf_max_weight=mtf_max_weight,
        mtf_field_fractions=(0.0, 0.7, 1.0),
        ray_resample_interval=ray_resample_interval,
        target_f_number=spec.required_f_number,
        first_order_preferred_relative_error=first_order_preferred_error,
        first_order_hard_relative_error=first_order_hard_error,
        w_field=field_weight,
        w_reg=regularization_weight,
        field_mapping_all_wavelengths=True,
        field_mapping_max_weight=field_max_weight,
        field_mapping_use_chief_ray=True,
        field_mapping_num_points=field_mapping_points,
        target_focal_length=target_focal_length,
        target_rfov=target_rfov,
        checkpoint_analysis=checkpoint_analysis,
        result_dir=str(result_dir / "optimization"),
    )
    if prune_surfaces:
        lens.prune_surf()
    lens.post_computation()
    lens.write_lens_json(str(result_dir / "mwir_final.json"))


def main() -> None:
    """命令行入口。"""

    configure_utf8_console()
    parser = argparse.ArgumentParser(description="生成 DeepLens 中波红外望远系统初始结构")
    parser.add_argument(
        "--scheme",
        choices=(
            "transmission_baseline",
            "transmission_balanced",
            "transmission_power_bent7",
            "cassegrain_equivalent",
            "large_fpa",
            "existing_fpa_narrow",
            "existing_fpa_wide",
        ),
        default="transmission_baseline",
        help=(
            "选择透射系统方案；transmission_power_bent7 使用七片、14面均有"
            "真实曲率的强弯曲优化母型；transmission_balanced 仅保留为六片"
            "一阶消色差概念对照，默认仍为原 transmission_baseline；"
            "cassegrain_equivalent 只继承一阶指标，不导入反射镜处方。"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto、cpu 或 cuda。")
    parser.add_argument("--iterations", type=int, default=0, help="RMS 优化迭代次数。")
    parser.add_argument(
        "--input-lens",
        default=None,
        help="从已有 DeepLens JSON 处方开始新阶段；只恢复处方，不恢复 Adam。",
    )
    parser.add_argument(
        "--allow-retarget",
        action="store_true",
        help="显式允许输入处方的原视场/焦距目标与当前命令不同；默认拒绝静默改题。",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs=4,
        metavar=("D", "C", "K", "A"),
        default=list(DEFAULT_MWIR_LRS),
        help=(
            "间距、曲率、圆锥常数、非球面系数学习率；默认 "
            f"{list(DEFAULT_MWIR_LRS)}。"
        ),
    )
    parser.add_argument(
        "--rms-weight",
        type=float,
        default=1.0,
        help="RMS 光斑损失权重；场映射阶段可降至 0.2–0.5，默认 1.0。",
    )
    parser.add_argument(
        "--mtf-surrogate-weight",
        type=float,
        default=0.0,
        help=(
            "固定频率 MTF 相位方差代理权重；默认 0（关闭）。像质阶段可从 "
            "0.05–0.2 开始，避免直接 OTF 在噪声底附近产生不稳定梯度。"
        ),
    )
    parser.add_argument(
        "--mtf-max-weight",
        type=float,
        default=1.0,
        help="MTF 代理最坏波长/场/方向相对平均超差的附加权重，默认 1.0。",
    )
    parser.add_argument(
        "--ray-resample-interval",
        type=int,
        default=1,
        help=(
            "每隔多少步重采样训练瞳光线；默认每步重采样，0 表示只在检查点重采样。"
        ),
    )
    parser.add_argument(
        "--first-order-preferred-error",
        type=float,
        default=0.008,
        help="EFL/F 数首选相对误差带，默认 0.008（0.8%%）。",
    )
    parser.add_argument(
        "--first-order-hard-error",
        type=float,
        default=0.01,
        help="EFL/F 数相对误差硬上限，默认 0.01（1%%）。",
    )
    parser.add_argument("--num-ring", type=int, default=8, help="径向视场采样环数。")
    parser.add_argument("--num-arm", type=int, default=4, help="每个采样环的方位臂数。")
    parser.add_argument("--spp", type=int, default=256, help="每视场、每波长的光线数。")
    parser.add_argument(
        "--shape-control",
        action="store_true",
        help="处方初步收敛后启用曲面形状修正；第一阶段默认关闭。",
    )
    parser.add_argument(
        "--prune-surfaces",
        action="store_true",
        help="优化结束后裁剪曲面口径；第一阶段默认关闭。",
    )
    parser.add_argument(
        "--field-weight",
        type=float,
        default=1.0,
        help="目标像高/场映射损失权重，默认 1.0。",
    )
    parser.add_argument(
        "--regularization-weight",
        type=float,
        default=0.1,
        help="机械间隙和曲面形状等通用正则项权重，默认 0.1。",
    )
    parser.add_argument(
        "--field-max-weight",
        type=float,
        default=1.0,
        help="最坏场点像高超差相对平均超差的附加权重，默认 1.0。",
    )
    parser.add_argument(
        "--field-mapping-points",
        type=int,
        default=9,
        help="每个主光线评价平面的等角场点数，默认 9。",
    )
    parser.add_argument(
        "--checkpoint-analysis",
        action="store_true",
        help="在优化检查点生成完整分析图；默认只保存 JSON 以加快 CPU 运行。",
    )
    parser.add_argument("--output", default=None, help="结果目录。")
    parser.add_argument(
        "--field-y-deg",
        type=float,
        default=9.6,
        help="Y 方向全视场，默认 9.6 度。",
    )
    parser.add_argument(
        "--image-height-mm",
        type=float,
        default=47.1454,
        help="Y 向边缘场点半像高，默认 47.1454 mm。",
    )
    parser.add_argument(
        "--entrance-pupil-mm",
        type=float,
        default=280.0,
        help="入瞳直径，默认 280 mm。",
    )
    parser.add_argument(
        "--pixel-pitch-um",
        type=float,
        default=None,
        help="可选：已确认的探测器像元间距；未知时不要填写。",
    )
    parser.add_argument(
        "--detector-res",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="可选：已确认的有效像元数，例如 --detector-res 320 256。",
    )
    parser.add_argument(
        "--simulation-pixel-pitch-um",
        type=float,
        default=30.0,
        help="虚拟焦面数值采样间距，默认 30 微米；不是探测器硬约束。",
    )
    parser.add_argument(
        "--two-pixel-resolution-urad",
        type=float,
        default=None,
        help="可选：重新启用两像元角分辨率约束，例如 42。",
    )
    parser.add_argument("--analyze", action="store_true", help="生成初始结构后执行完整分析。")
    parser.add_argument("--evaluate", action="store_true", help="计算数值 MTF、畸变和渐晕指标。")
    parser.add_argument("--eval-spp", type=int, default=512, help="每个 MTF 点的光线数。")
    parser.add_argument("--check-only", action="store_true", help="只输出规格检查，不导入 DeepLens。")
    args = parser.parse_args()

    spec = MWIRDesignSpec(
        field_y_deg=args.field_y_deg,
        image_height_mm=args.image_height_mm,
        entrance_pupil_diameter_mm=args.entrance_pupil_mm,
        pixel_pitch_um=args.pixel_pitch_um,
        detector_res=(None if args.detector_res is None else tuple(args.detector_res)),
        simulation_pixel_pitch_um=args.simulation_pixel_pitch_um,
        two_pixel_resolution_urad=args.two_pixel_resolution_urad,
    )
    print(json.dumps(spec.geometry_report(), ensure_ascii=False, indent=2))
    if args.check_only:
        return

    result_dir = _make_result_dir(args.output)
    if args.input_lens is None:
        lens, params, result_path = build_initial_lens(
            spec,
            scheme=args.scheme,
            result_dir=result_dir,
            device=args.device,
            analyze=args.analyze,
        )
        source_description = "已生成初始结构"
    else:
        lens, params, result_path = load_lens_for_stage(
            spec,
            input_lens=args.input_lens,
            scheme=args.scheme,
            result_dir=result_dir,
            device=args.device,
            analyze=args.analyze,
            allow_retarget=args.allow_retarget,
        )
        source_description = f"已载入阶段处方 {Path(args.input_lens)}"
    _record_optimization_config(
        result_path,
        {
            "iterations": args.iterations,
            "lrs": list(args.lrs),
            "rms_weight": args.rms_weight,
            "mtf_surrogate_weight": args.mtf_surrogate_weight,
            "mtf_max_weight": args.mtf_max_weight,
            "ray_resample_interval": args.ray_resample_interval,
            "first_order_preferred_error": args.first_order_preferred_error,
            "first_order_hard_error": args.first_order_hard_error,
            "field_weight": args.field_weight,
            "field_max_weight": args.field_max_weight,
            "regularization_weight": args.regularization_weight,
            "field_mapping_points": args.field_mapping_points,
            "num_ring": args.num_ring,
            "num_arm": args.num_arm,
            "rays_per_field_wavelength": args.spp,
            "minimum_valid_ray_ratio": spec.vignetting_floor,
            "shape_control": args.shape_control,
            "prune_surfaces": args.prune_surfaces,
            "checkpoint_analysis": args.checkpoint_analysis,
            "device_requested": args.device,
            "analyze_initial": args.analyze,
            "evaluate_after_stage": args.evaluate,
            "evaluation_rays_per_field": args.eval_spp,
        },
    )
    optimize_lens(
        lens,
        spec,
        result_path,
        args.iterations,
        design_params=params,
        num_ring=args.num_ring,
        num_arm=args.num_arm,
        spp=args.spp,
        shape_control=args.shape_control,
        prune_surfaces=args.prune_surfaces,
        field_weight=args.field_weight,
        field_max_weight=args.field_max_weight,
        field_mapping_points=args.field_mapping_points,
        regularization_weight=args.regularization_weight,
        rms_weight=args.rms_weight,
        mtf_surrogate_weight=args.mtf_surrogate_weight,
        mtf_max_weight=args.mtf_max_weight,
        ray_resample_interval=args.ray_resample_interval,
        first_order_preferred_error=args.first_order_preferred_error,
        first_order_hard_error=args.first_order_hard_error,
        lrs=tuple(args.lrs),
        checkpoint_analysis=args.checkpoint_analysis,
    )
    if args.evaluate:
        metrics = evaluate_lens(
            lens,
            spec,
            result_path,
            psf_spp=args.eval_spp,
        )
        print(json.dumps(metrics["pass"], ensure_ascii=False, indent=2))
    print(
        f"{source_description}：{result_path / 'mwir_initial.json'}\n"
        f"元件数：{params['element_count']}，目标 F/{params['f_number']:.3f}，"
        f"目标 Y 向全视场：{params['field_y_deg']:.4f}°，"
        f"半像高：{params['image_height_mm']:.4f} mm"
    )


if __name__ == "__main__":
    main()
