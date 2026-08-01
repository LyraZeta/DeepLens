"""从公开 MWIR 七片专利图 14A--14D 生成可优化的透射式起始处方。

该文件只把图中可读的半径、厚度、材料和低阶非球面参数转换成 DeepLens
的连续正向光路。专利原系统包含折叠镜、冷窗和滤光片；这里将折叠段沿光路
展开，并去掉无光焦度的窗口/滤光片，因此它是结构先验，不是专利处方的复现。
所有长度先按英寸读取，再统一换算为毫米并按目标 EFL 缩放。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from deeplens import GeoLens
from deeplens.geometric_surface import Aperture, Aspheric, Spheric
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import _apply_mwir_constraints, _calibrate_initial_power


# 图 14A--14C 的七个有光焦度元件，单位为英寸。
PATENT_MATERIALS = ("si", "ge", "ge", "si", "si", "ge", "ge")
PATENT_RADII_IN = (
    (-10.20901, -44.23198),
    (-58.89313, -9.05168),
    (-39.73968, 126.25564),
    (-19.57203, 61.40640),
    (-3.63682, -4.39844),
    (-4.18099, -2.54807),
    (-9.08177, -13.69442),
)
PATENT_GLASS_THICKNESS_IN = (
    0.984252,
    0.551181,
    0.787402,
    0.787402,
    0.787402,
    0.433071,
    0.472441,
)
# 折叠段与中继段的光程；正值表示展开后的顺序距离。
PATENT_AIR_GAPS_IN = (
    0.0,       # 首片前方由统一的首面位置给出
    1.002346,
    3.978904,
    4.724409 + 6.299213 + 6.299213 + 3.779528,
    0.183018,
    0.329645,
    2.346632,
    0.762753,
)

# 图中四个明确给出非球面参数的面：元件 1 前、元件 2 前、元件 4 前、
# 元件 7 后。A/B/C/D 的单位是英寸对应的 r^4/r^6/r^8/r^10 系数。
PATENT_ASPHERES = {
    (0, 0): (-2.26833, (-6.56599e-4, 2.63473e-5, -2.96207e-8, -8.72988e-10)),
    (1, 0): (-92.97056, (-4.62524e-3, 3.71682e-5, -7.77795e-7, 9.30649e-9)),
    (3, 0): (-8.210351, (1.62033e-4, 1.07944e-5, -3.93774e-7, 3.50634e-9)),
    (6, 1): (-72.644570, (-2.41066e-2, 6.01195e-3, -1.00261e-3, 1.11305e-4)),
}


def _scaled_asphere_coefficients(
    values_in: tuple[float, float, float, float], scale: float
) -> list[float]:
    """把英寸系数换算到毫米并应用整体几何缩放。"""

    # z_mm = 25.4 z_in, r_in = r_mm / 25.4；再将所有尺寸乘 scale。
    # 对 r^m 项，系数按 scale**(1-m) 变换。
    result = []
    for coefficient, order in zip(values_in, (4, 6, 8, 10)):
        result.append(
            coefficient * 25.4 ** (1 - order) * scale ** (1 - order)
        )
    return result


def build_seed(spec, device, *, reverse_curvature: bool = True) -> GeoLens:
    """构建展开后的七片专利结构，并校准到目标 EFL。"""

    target_scale = spec.effective_focal_length_mm / (20.0 * 25.4)
    # 目标入瞳比专利示例 6.0063 in 大；这里先给足口径，后续优化可再收紧。
    semi_apertures = (155.0, 155.0, 160.0, 160.0, 160.0, 150.0, 145.0)
    first_z = 12.0
    mm = 25.4
    lens = GeoLens(
        primary_wvln=3.5,
        wvln_rgb=list(spec.wavelengths_um),
        obj_depth=spec.object_distance_mm,
    )
    surfaces = [Aperture(r=spec.entrance_pupil_diameter_mm / 2.0, d=0.0)]
    z = first_z
    for element_index, material in enumerate(PATENT_MATERIALS):
        front_radius = PATENT_RADII_IN[element_index][0] * mm * target_scale
        rear_radius = PATENT_RADII_IN[element_index][1] * mm * target_scale
        if reverse_curvature:
            front_radius = -front_radius
            rear_radius = -rear_radius
        thickness = PATENT_GLASS_THICKNESS_IN[element_index] * mm * target_scale
        semi = semi_apertures[element_index]
        for side, radius in enumerate((front_radius, rear_radius)):
            key = (element_index, side)
            kwargs = {
                "r": semi,
                "d": z,
                "c": 1.0 / radius,
                "mat2": material if side == 0 else "air",
            }
            if key in PATENT_ASPHERES:
                k, coefficients = PATENT_ASPHERES[key]
                scaled_coefficients = _scaled_asphere_coefficients(
                    coefficients, target_scale
                )
                if reverse_curvature:
                    # 反转专利坐标系的传播方向时，矢高的 z 符号也必须反转；
                    # 圆锥常数保持不变。
                    scaled_coefficients = [-value for value in scaled_coefficients]
                kwargs.update(k=k, ai=scaled_coefficients)
                surfaces.append(Aspheric(**kwargs))
            else:
                surfaces.append(Spheric(**kwargs))
            if side == 0:
                z += thickness
        if element_index < len(PATENT_MATERIALS) - 1:
            gap = PATENT_AIR_GAPS_IN[element_index + 1] * mm * target_scale
            z += gap

    # 该距离包含专利图中的折叠/冷窗/滤光片光程；窗口本身不作为透镜计数。
    bfl = PATENT_AIR_GAPS_IN[-1] * mm * target_scale
    lens.surfaces = surfaces
    lens.lens_info = "公开 MWIR 七片专利图 14 展开结构先验（非专利复现）"
    lens.d_sensor = torch.tensor(z + bfl)
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
        max_iterations=16,
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
    parser = argparse.ArgumentParser(description="生成公开 MWIR 专利结构先验")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--reverse-curvature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否把专利坐标系的曲率符号翻转到正向传播坐标系。",
    )
    args = parser.parse_args()
    spec = MWIRDesignSpec()
    lens = build_seed(spec, torch.device(args.device), reverse_curvature=args.reverse_curvature)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lens.write_lens_json(str(output))
    metadata = {
        "source": "US11960064B2 Fig. 14A-14D",
        "warning": "折叠段已展开、窗口和滤光片已移除；仅为可优化结构先验。",
        "materials": list(PATENT_MATERIALS),
        "scale": spec.effective_focal_length_mm / (20.0 * 25.4),
        "reverse_curvature": bool(args.reverse_curvature),
        "spec": spec.geometry_report(),
    }
    with open(output.with_name(output.stem + "_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"专利结构种子已保存到：{output.resolve()}")


if __name__ == "__main__":
    main()
