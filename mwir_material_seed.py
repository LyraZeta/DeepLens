"""按离散材料布局生成 MWIR 七片双面有功率种子。

该脚本只负责外层材料搜索的初始处方，不宣称生成的种子已经满足 MTF。
每组三片正负光焦度先用三波长一阶色差方程配平，再把薄透镜光焦度均分到
前后两个曲面；随后可交给 ``mwir_power_bent7_optimize.py`` 的球面、非球面
和结构阶段继续优化。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from deeplens import GeoLens
from deeplens.geometric_surface import Aperture, Aspheric, Spheric
from deeplens.material import Material
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import (
    _apply_mwir_constraints,
    _calibrate_initial_power,
    _scheme_parameters,
)


def _dispersion_number(material_name: str, wavelengths: tuple[float, ...]) -> float:
    """返回目标三波长下的等效红外色散数。"""

    material = Material(material_name)
    n_short = float(material.refractive_index(wavelengths[0]))
    n_primary = float(material.refractive_index(wavelengths[1]))
    n_long = float(material.refractive_index(wavelengths[2]))
    denominator = n_short - n_long
    if abs(denominator) < 1e-12:
        raise ValueError(f"材料 {material_name} 的波段色散过小。")
    value = (n_primary - 1.0) / denominator
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError(f"材料 {material_name} 的等效色散数无效：{value}。")
    return value


def _pair_power(
    net_power: float,
    positive_material: str,
    negative_material: str,
    wavelengths: tuple[float, ...],
    color_sum: float = 0.0,
) -> tuple[float, float]:
    """在给定净光焦度和色差贡献下求正负元件光焦度。"""

    v_positive = _dispersion_number(positive_material, wavelengths)
    v_negative = _dispersion_number(negative_material, wavelengths)
    denominator = 1.0 / v_positive - 1.0 / v_negative
    if abs(denominator) < 1e-12:
        raise ValueError("正负材料的等效色散数过于接近。")
    positive_power = (color_sum - net_power / v_negative) / denominator
    negative_power = net_power - positive_power
    return positive_power, negative_power


def build_seed(
    spec: MWIRDesignSpec,
    materials: tuple[str, ...],
    device,
    *,
    bend_scale: float = 1.0,
) -> GeoLens:
    """生成一个有真实弯曲的七片消色差搜索种子。

    ``_pair_power`` 只决定每片的净光焦度；如果把两面的曲率严格取成
    ``(+c, -c)``，净光焦度很小的红外配对会退化成几乎平行的玻璃板。
    这里另外引入每片的弯曲量 ``b``，按
    ``c_front = b + phi/(2(n-1))``、
    ``c_rear = b - phi/(2(n-1))`` 分配曲率。
    """

    if len(materials) != 7:
        raise ValueError("七片材料布局必须恰好包含 7 个名称。")
    if not math.isfinite(float(bend_scale)) or float(bend_scale) < 0.0:
        raise ValueError("bend_scale 必须是非负有限数。")
    materials = tuple(value.strip().lower() for value in materials)
    wavelengths = tuple(float(value) for value in spec.wavelengths_um)
    total_power = 1.0 / spec.effective_focal_length_mm
    # 前三对承担 42%/30%/25% 的净光焦度，末片保留 3% 弱场平作用。
    net_fractions = (0.42, 0.30, 0.25)
    weak_power = 0.03 * total_power
    powers = []
    first_pair_color = -weak_power / _dispersion_number(materials[6], wavelengths)
    for pair_index, fraction in enumerate(net_fractions):
        color = first_pair_color if pair_index == 0 else 0.0
        powers.extend(
            _pair_power(
                fraction * total_power,
                materials[2 * pair_index],
                materials[2 * pair_index + 1],
                wavelengths,
                color_sum=color,
            )
        )
    powers.append(weak_power)

    front_positions = (12.0, 60.0, 112.0, 164.0, 240.0, 316.0, 392.0)
    thicknesses = (24.0, 18.0, 24.0, 18.0, 24.0, 18.0, 18.0)
    semi_apertures = (150.0, 150.0, 155.0, 155.0, 160.0, 160.0, 160.0)
    base_bends = (0.00120, 0.00100, 0.0, 0.0, -0.00100, -0.00055, -0.00070)
    lens = GeoLens(
        primary_wvln=3.5,
        wvln_rgb=list(spec.wavelengths_um),
        obj_depth=spec.object_distance_mm,
    )
    surfaces = [Aperture(r=spec.entrance_pupil_diameter_mm / 2.0, d=0.0)]
    for element_index, (material_name, power) in enumerate(zip(materials, powers)):
        n = float(Material(material_name).refractive_index(3.5))
        half_curvature = power / (2.0 * (n - 1.0))
        bend = float(base_bends[element_index]) * float(bend_scale)
        front_curvature = bend + half_curvature
        rear_curvature = bend - half_curvature
        if max(abs(front_curvature), abs(rear_curvature)) < 1e-7:
            front_curvature = 1e-5
            rear_curvature = -1e-5
        front_z = front_positions[element_index]
        rear_z = front_z + thicknesses[element_index]
        # 与 ``mwir_power_bent7_optimize.py`` 保持一致：只预留五个低阶
        # 非球面面，避免材料种子进入非球面阶段时出现自由度数量不匹配。
        # 面编号按七片系统从 0 开始，(元件序号, 面序号) 为前/后表面。
        aspheric_sides = frozenset({(0, 0), (2, 1), (4, 0), (4, 1), (6, 0)})
        asphere_front = (element_index, 0) in aspheric_sides
        asphere_rear = (element_index, 1) in aspheric_sides
        front_cls = Aspheric if asphere_front else Spheric
        rear_cls = Aspheric if asphere_rear else Spheric
        front_kwargs = dict(
            r=semi_apertures[element_index],
            d=front_z,
            c=front_curvature,
            mat2=material_name,
        )
        rear_kwargs = dict(
            r=semi_apertures[element_index],
            d=rear_z,
            c=rear_curvature,
            mat2="air",
        )
        if front_cls is Aspheric:
            front_kwargs.update(k=0.0, ai=[0.0] * 4)
        if rear_cls is Aspheric:
            rear_kwargs.update(k=0.0, ai=[0.0] * 4)
        surfaces.extend((front_cls(**front_kwargs), rear_cls(**rear_kwargs)))

    lens.surfaces = surfaces
    lens.lens_info = "MWIR 七片离散材料三组消色差搜索种子"
    lens.d_sensor = torch.tensor(front_positions[-1] + thicknesses[-1] + 300.0)
    lens.r_sensor = float(spec.image_height_mm)
    lens.float_enpd = True
    lens.float_foclen = False
    lens.float_rfov = False
    lens.set_sensor(tuple(spec.virtual_sensor_size_mm), tuple(spec.virtual_sensor_res))
    lens = lens.to(device)
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)
    _calibrate_initial_power(
        lens,
        spec.effective_focal_length_mm,
        max_iterations=12,
        logarithmic_tolerance=1e-6,
        minimum_factor=0.5,
        maximum_factor=2.0,
    )
    lens.refocus(float("inf"))
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)
    return lens


def main() -> None:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description="生成 MWIR 离散材料七片搜索种子")
    parser.add_argument("--materials", required=True, help="逗号分隔的七片材料布局")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--bend-scale",
        type=float,
        default=1.0,
        help="每片额外弯曲量的倍率；0 恢复旧的对称近似，1 使用强弯曲母型。",
    )
    args = parser.parse_args()
    spec = MWIRDesignSpec()
    materials = tuple(value.strip() for value in args.materials.split(",") if value.strip())
    lens = build_seed(
        spec,
        materials,
        torch.device(args.device),
        bend_scale=args.bend_scale,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lens.write_lens_json(str(output))
    metadata = {
        "materials": list(materials),
        "bend_scale": float(args.bend_scale),
        "spec": spec.geometry_report(),
        "design": _scheme_parameters(spec, "transmission_power_bent7"),
        "warning": "仅为离散材料外层搜索种子，必须经过连续优化和独立高采样验收。",
    }
    with open(output.with_name(output.stem + "_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"材料种子已保存到：{output.resolve()}")


if __name__ == "__main__":
    main()
