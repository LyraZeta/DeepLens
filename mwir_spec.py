"""中波红外透射式望远系统的一阶规格与一致性检查。

当前基线来自 Zemax 系统概要图：视场表在 Y 方向使用 ``-4.8°`` 到
``+4.8°``，最大像高为 ``47.1454 mm``。因此本模块把 9.6°解释为
Y 方向全视场，而不是未经确认的探测器对角视场。

探测器型号、像元间距和阵列格式尚未确认。程序使用一个“虚拟仿真焦面”
完成 DeepLens 初始结构和数值测试，但该焦面不构成最终探测器规格。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Any


def configure_utf8_console() -> None:
    """在 Windows 中文代码页下将终端输出切换为 UTF-8。"""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = (getattr(stream, "encoding", "") or "").lower()
        if reconfigure is not None and encoding in {"", "ascii", "none"}:
            reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class MWIRDesignSpec:
    """中波红外透射式系统的设计规格。

    长度单位为毫米，波长单位为微米，角度单位为度。

    ``diagonal_fov_deg`` 保留为旧代码的兼容字段，但不再表示已确认的
    对角视场；当 ``field_y_deg`` 未提供时，它作为 Y 方向全视场的旧别名。
    ``pixel_pitch_um`` 和 ``detector_res`` 均为可选项，未确认时不参与最终
    探测器验收。``simulation_pixel_pitch_um`` 只用于 DeepLens 的虚拟仿真焦面。
    """

    # 旧接口字段：默认值保留，实际计算通过 full_field_y_deg 读取。
    diagonal_fov_deg: float = 9.6
    two_pixel_resolution_urad: float | None = None
    pixel_pitch_um: float | None = None
    detector_res: tuple[int, int] | None = None
    wavelengths_um: tuple[float, float, float] = (2.7, 3.5, 4.3)
    entrance_pupil_diameter_mm: float = 280.0
    object_distance_mm: float = -100_000_000.0
    max_lenses: int = 7
    temperature_c: float = 20.0
    system_mtf_threshold: float = 0.30
    optical_mtf_target: float = 0.50
    distortion_limit: float = 0.005
    focal_length_tolerance: float = 0.01
    f_number_tolerance: float = 0.01
    vignetting_target: float = 0.80
    vignetting_floor: float = 0.70
    notes: tuple[str, ...] = field(
        default=(
            "优化物距用 -100,000,000 mm（100 km）近似无穷远；正式验收直接使用平行光，有限共轭焦移约 3.2 微米。",
            "9.6°按 Zemax 视场表解释为 Y 方向全视场，半视场角为 4.8°。",
            "47.1454 mm 是从光轴到 Y 向边缘场点的半像高，不是已确认的探测器对角线。",
            "当前版本不把 42 微弧度作为约束；如显式启用，使用仿真像元间距进行历史方案复现。",
            "探测器未知时，MTF 频率采用虚拟仿真焦面的像元间距，仅用于初始数值检查。",
            "渐晕函数给出的是有效光线比例代理量，不含余弦四次方、透过率和镀膜损失。",
        )
    )
    # 新接口字段：不填时使用上面的兼容字段。
    field_y_deg: float | None = None
    image_height_mm: float = 47.1454
    simulation_pixel_pitch_um: float = 30.0
    simulation_sensor_res: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.full_field_y_deg <= 0.0 or self.full_field_y_deg >= 180.0:
            raise ValueError("field_y_deg 必须位于 0 到 180 度之间。")
        if self.image_height_mm <= 0.0:
            raise ValueError("image_height_mm 必须为正数。")
        if self.entrance_pupil_diameter_mm <= 0.0:
            raise ValueError("entrance_pupil_diameter_mm 必须为正数。")
        if self.pixel_pitch_um is not None and self.pixel_pitch_um <= 0.0:
            raise ValueError("pixel_pitch_um 必须为正数或 None。")
        if self.simulation_pixel_pitch_um <= 0.0:
            raise ValueError("simulation_pixel_pitch_um 必须为正数。")
        for name, value in (
            ("distortion_limit", self.distortion_limit),
            ("focal_length_tolerance", self.focal_length_tolerance),
            ("f_number_tolerance", self.f_number_tolerance),
            ("vignetting_target", self.vignetting_target),
            ("vignetting_floor", self.vignetting_floor),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} 必须位于 (0, 1] 范围内。")
        if self.vignetting_floor > self.vignetting_target:
            raise ValueError("vignetting_floor 不能高于 vignetting_target。")
        for name, resolution in (
            ("detector_res", self.detector_res),
            ("simulation_sensor_res", self.simulation_sensor_res),
        ):
            if resolution is not None and (
                len(resolution) != 2
                or min(resolution) <= 0
                or any(int(value) != value for value in resolution)
            ):
                raise ValueError(f"{name} 必须是两个正整数。")

    @property
    def full_field_y_deg(self) -> float:
        """Y 方向全视场 [deg]。"""

        return float(self.diagonal_fov_deg if self.field_y_deg is None else self.field_y_deg)

    @property
    def half_field_y_deg(self) -> float:
        """Y 方向半视场角 [deg]。"""

        return self.full_field_y_deg / 2.0

    @property
    def effective_focal_length_mm(self) -> float:
        """由 Y 向半像高和半视场角反推的有效焦距 [mm]。"""

        return self.image_height_mm / math.tan(math.radians(self.half_field_y_deg))

    @property
    def image_height_full_y_mm(self) -> float:
        """Y 方向完整像面高度 [mm]。"""

        return 2.0 * self.image_height_mm

    @property
    def detector_pitch_known(self) -> bool:
        """是否已经确认探测器像元间距。"""

        return self.pixel_pitch_um is not None

    @property
    def detector_format_known(self) -> bool:
        """是否已经确认探测器有效像元数。"""

        return self.detector_res is not None

    @property
    def detector_is_known(self) -> bool:
        """是否同时确认了像元间距和阵列格式。"""

        return self.detector_pitch_known and self.detector_format_known

    @property
    def analysis_pixel_pitch_um(self) -> float:
        """用于当前数值仿真的像元间距 [µm]。"""

        return float(
            self.simulation_pixel_pitch_um
            if self.pixel_pitch_um is None
            else self.pixel_pitch_um
        )

    @property
    def pixel_pitch_mm(self) -> float:
        """当前分析使用的像元间距 [mm]。"""

        return self.analysis_pixel_pitch_um * 1e-3

    @property
    def single_pixel_ifov_rad(self) -> float | None:
        """单像元瞬时视场角 [rad]；探测器未知时返回 None。"""

        if self.two_pixel_resolution_urad is not None:
            return self.two_pixel_resolution_urad * 1e-6 / 2.0
        if not self.detector_pitch_known:
            return None
        return self.pixel_pitch_mm / self.effective_focal_length_mm

    @property
    def required_focal_length_mm(self) -> float:
        """当前约束下的有效焦距 [mm]。

        默认由像高和 Y 向视场推导。只有显式启用历史两像元约束时，
        才改由像元角采样反推焦距。
        """

        if self.two_pixel_resolution_urad is None:
            return self.effective_focal_length_mm
        return self.pixel_pitch_mm / (self.two_pixel_resolution_urad * 1e-6 / 2.0)

    @property
    def required_image_height_mm(self) -> float:
        """当前焦距和视场对应的半像高 [mm]。"""

        return self.required_focal_length_mm * math.tan(math.radians(self.half_field_y_deg))

    @property
    def required_f_number(self) -> float:
        """按入瞳直径计算的一阶 F 数。"""

        return self.required_focal_length_mm / self.entrance_pupil_diameter_mm

    @property
    def resolution_constraint_active(self) -> bool:
        """是否启用了两像元角分辨率约束。"""

        return self.two_pixel_resolution_urad is not None

    @property
    def detector_size_mm(self) -> tuple[float, float] | None:
        """已确认探测器的物理尺寸 ``(宽, 高)`` [mm]。"""

        if not self.detector_is_known:
            return None
        width_px, height_px = self.detector_res  # type: ignore[misc]
        return width_px * self.pixel_pitch_mm, height_px * self.pixel_pitch_mm

    @property
    def detector_diagonal_mm(self) -> float | None:
        """已确认探测器的对角尺寸 [mm]。"""

        size = self.detector_size_mm
        return None if size is None else math.hypot(*size)

    @property
    def virtual_sensor_res(self) -> tuple[int, int]:
        """供 DeepLens 使用的圆形等效虚拟焦面分辨率。"""

        if self.simulation_sensor_res is not None:
            return self.simulation_sensor_res
        side_mm = math.sqrt(2.0) * self.image_height_mm
        count = max(64, int(round(side_mm / (self.simulation_pixel_pitch_um * 1e-3))))
        return count, count

    @property
    def virtual_sensor_size_mm(self) -> tuple[float, float]:
        """供 DeepLens 使用的圆形等效虚拟焦面尺寸 [mm]。"""

        width_px, height_px = self.virtual_sensor_res
        resolution_diagonal = math.hypot(width_px, height_px)
        image_diameter = 2.0 * self.image_height_mm
        return (
            image_diameter * width_px / resolution_diagonal,
            image_diameter * height_px / resolution_diagonal,
        )

    @property
    def required_full_image_height_y_mm(self) -> float:
        """当前 Y 向全视场所需的完整像面高度 [mm]。"""

        return 2.0 * self.required_image_height_mm

    @property
    def virtual_image_circle_diameter_mm(self) -> float:
        """圆形等效设计包络的像面直径 [mm]。

        探测器未知时，DeepLens 使用旋转对称的圆形等效场；该直径等于
        Y 向完整像面高度，但不代表最终矩形探测器的对角线。
        """

        return self.required_full_image_height_y_mm

    @property
    def required_sensor_size_mm(self) -> tuple[float, float] | None:
        """若已知探测器纵横比，返回覆盖 Y 向全视场的尺寸 [mm]。

        ``image_height_mm`` 是 Y 向半像高，因此探测器高度必须等于
        ``2 * required_image_height_mm``；宽度再按已知像元阵列宽高比计算。
        """

        if self.detector_res is None:
            return None
        width_px, height_px = self.detector_res
        height_mm = self.required_full_image_height_y_mm
        width_mm = height_mm * width_px / height_px
        return width_mm, height_mm

    @property
    def required_sensor_diagonal_mm(self) -> float | None:
        """已知纵横比时，覆盖目标 Y 视场所需的探测器对角尺寸 [mm]。"""

        size = self.required_sensor_size_mm
        return None if size is None else math.hypot(*size)

    @property
    def required_detector_res_float(self) -> tuple[float, float] | None:
        """按当前分析像元间距和已知纵横比反推的理想像元数。"""

        size = self.required_sensor_size_mm
        if size is None:
            return None
        return size[0] / self.pixel_pitch_mm, size[1] / self.pixel_pitch_mm

    @property
    def recommended_detector_res(self) -> tuple[int, int] | None:
        """按已知纵横比取整后的推荐像元数。

        以最简整数宽高比的整数倍取整，避免分别四舍五入后破坏像元阵列纵横比。
        """

        values = self.required_detector_res_float
        if values is None:
            return None
        width_px, height_px = self.detector_res  # type: ignore[misc]
        divisor = math.gcd(int(width_px), int(height_px))
        width_ratio = int(width_px) // divisor
        height_ratio = int(height_px) // divisor
        # 向上取整以保证有效高度不小于目标 Y 像面；量化步长是最简比例中
        # ``height_ratio`` 个像元，而不是单个像元。
        ratio_multiples = max(1, int(math.ceil(values[1] / height_ratio)))
        return width_ratio * ratio_multiples, height_ratio * ratio_multiples

    @property
    def current_detector_x_fov_deg(self) -> float | None:
        """已确认探测器在当前有效焦距下的 X 向全视场 [deg]。"""

        size = self.detector_size_mm
        if size is None:
            return None
        return math.degrees(2.0 * math.atan(size[0] / 2.0 / self.effective_focal_length_mm))

    @property
    def current_detector_y_fov_deg(self) -> float | None:
        """已确认探测器在当前有效焦距下的 Y 向全视场 [deg]。"""

        size = self.detector_size_mm
        if size is None:
            return None
        return math.degrees(2.0 * math.atan(size[1] / 2.0 / self.effective_focal_length_mm))

    @property
    def current_detector_diagonal_fov_deg(self) -> float | None:
        """已确认探测器在当前有效焦距下的对角全视场 [deg]。"""

        diagonal = self.detector_diagonal_mm
        if diagonal is None:
            return None
        return math.degrees(2.0 * math.atan(diagonal / 2.0 / self.effective_focal_length_mm))

    @property
    def focal_length_for_current_detector_mm(self) -> float | None:
        """若把已确认探测器对角线用于 9.6°径向场，所需焦距 [mm]。"""

        diagonal = self.detector_diagonal_mm
        if diagonal is None:
            return None
        return diagonal / 2.0 / math.tan(math.radians(self.half_field_y_deg))

    @property
    def focal_length_for_current_detector_y_fov_mm(self) -> float | None:
        """若把已确认探测器高度用于 9.6° Y 向场，所需焦距 [mm]。"""

        size = self.detector_size_mm
        if size is None:
            return None
        return size[1] / 2.0 / math.tan(math.radians(self.half_field_y_deg))

    @property
    def two_pixel_resolution_with_current_detector_urad(self) -> float | None:
        """已确认像元在当前有效焦距下对应的两像元角采样 [µrad]。"""

        if not self.detector_pitch_known:
            return None
        return 2.0 * self.pixel_pitch_mm / self.effective_focal_length_mm * 1e6

    @property
    def physical_f_number_floor(self) -> float:
        """空气中由 NA≤1 给出的理想 F 数下限。"""

        return 0.5

    @property
    def focal_length_at_physical_f_number_floor_mm(self) -> float:
        """保持 280 mm 入瞳并取理想 F/0.5 时所需焦距 [mm]。"""

        return self.physical_f_number_floor * self.entrance_pupil_diameter_mm

    @property
    def detector_fov_at_physical_f_number_floor_deg(self) -> float | None:
        """已确认焦面在理想 F/0.5 下可覆盖的对角全视场 [deg]。"""

        diagonal = self.detector_diagonal_mm
        if diagonal is None:
            return None
        return math.degrees(
            2.0
            * math.atan(
                diagonal
                / 2.0
                / self.focal_length_at_physical_f_number_floor_mm
            )
        )

    @property
    def current_detector_f_number_for_target_fov(self) -> float | None:
        """已确认焦面高度覆盖当前 Y 向全视场时的一阶 F 数。"""

        focal_length = self.focal_length_for_current_detector_y_fov_mm
        return (
            None
            if focal_length is None
            else focal_length / self.entrance_pupil_diameter_mm
        )

    @property
    def nyquist_frequency_cy_mm(self) -> float | None:
        """已确认探测器的奈奎斯特频率 [cycles/mm]。"""

        if not self.detector_pitch_known:
            return None
        return 1.0 / (2.0 * self.pixel_pitch_mm)

    @property
    def analysis_nyquist_frequency_cy_mm(self) -> float:
        """当前分析使用的奈奎斯特频率 [cycles/mm]。

        已确认像元间距时使用实际探测器频率，否则使用虚拟仿真焦面频率。
        """

        detector_nyquist = self.nyquist_frequency_cy_mm
        if detector_nyquist is not None:
            return detector_nyquist
        pitch_mm = self.virtual_sensor_size_mm[0] / self.virtual_sensor_res[0]
        return 1.0 / (2.0 * pitch_mm)

    def diffraction_cutoff_cy_mm(self, wavelength_um: float) -> float:
        """圆孔径衍射截止频率 [cycles/mm]。"""

        wavelength_mm = wavelength_um * 1e-3
        return 1.0 / (wavelength_mm * self.required_f_number)

    def airy_diameter_um(self, wavelength_um: float) -> float:
        """艾里斑第一暗环直径 [µm]。"""

        return 2.44 * wavelength_um * self.required_f_number

    def diffraction_angle_urad(self, wavelength_um: float) -> float:
        """按 Rayleigh 判据估算的角分辨率 [µrad]。"""

        wavelength_mm = wavelength_um * 1e-3
        return 1.22 * wavelength_mm / self.entrance_pupil_diameter_mm * 1e6

    def ideal_optical_mtf_at_nyquist(self, wavelength_um: float) -> float:
        """在探测器奈奎斯特或虚拟仿真奈奎斯特处估算系统 MTF。"""

        cutoff = self.diffraction_cutoff_cy_mm(wavelength_um)
        nyquist = self.analysis_nyquist_frequency_cy_mm
        normalized_frequency = nyquist / cutoff
        if normalized_frequency >= 1.0:
            optical_mtf = 0.0
        else:
            root = math.sqrt(max(0.0, 1.0 - normalized_frequency**2))
            optical_mtf = (2.0 / math.pi) * (
                math.acos(normalized_frequency) - normalized_frequency * root
            )
        # 方形像元孔径在奈奎斯特频率处的 sinc 值为 2/pi。
        return optical_mtf * (2.0 / math.pi)

    def geometry_report(self) -> dict[str, Any]:
        """返回可序列化的一阶计算结果。"""

        current_size = self.detector_size_mm
        current_diag = self.detector_diagonal_mm
        recommended_size = self.required_sensor_size_mm
        recommended_res = self.recommended_detector_res
        current_diag_fov = self.current_detector_diagonal_fov_deg
        current_x_fov = self.current_detector_x_fov_deg
        current_y_fov = self.current_detector_y_fov_deg
        current_two_pixel = self.two_pixel_resolution_with_current_detector_urad
        current_fnum = self.current_detector_f_number_for_target_fov
        return {
            "field_definition": "Y 方向全视场",
            "field_y_deg": self.full_field_y_deg,
            "half_field_y_deg": self.half_field_y_deg,
            "legacy_diagonal_fov_deg": self.diagonal_fov_deg,
            "image_height_mm": self.image_height_mm,
            "full_image_height_y_mm": self.image_height_full_y_mm,
            "required_full_image_height_y_mm": self.required_full_image_height_y_mm,
            "virtual_image_circle_diameter_mm": self.virtual_image_circle_diameter_mm,
            "required_sensor_diagonal_mm": self.required_sensor_diagonal_mm,
            "two_pixel_resolution_urad_constraint": self.two_pixel_resolution_urad,
            "resolution_constraint_active": self.resolution_constraint_active,
            "detector_pitch_known": self.detector_pitch_known,
            "detector_format_known": self.detector_format_known,
            "detector_is_known": self.detector_is_known,
            "pixel_pitch_um": self.pixel_pitch_um,
            "analysis_pixel_pitch_um": self.analysis_pixel_pitch_um,
            "single_pixel_ifov_urad": (
                None
                if self.single_pixel_ifov_rad is None
                else self.single_pixel_ifov_rad * 1e6
            ),
            "effective_focal_length_mm": self.effective_focal_length_mm,
            "required_focal_length_mm": self.required_focal_length_mm,
            "required_image_height_mm": self.required_image_height_mm,
            "required_f_number": self.required_f_number,
            "entrance_pupil_diameter_mm": self.entrance_pupil_diameter_mm,
            "current_detector_res": (
                None if self.detector_res is None else list(self.detector_res)
            ),
            "current_detector_size_mm": None if current_size is None else list(current_size),
            "current_detector_diagonal_mm": current_diag,
            "current_detector_x_fov_deg": current_x_fov,
            "current_detector_y_fov_deg": current_y_fov,
            "current_detector_diagonal_fov_deg": current_diag_fov,
            "required_sensor_size_mm": (
                None if recommended_size is None else list(recommended_size)
            ),
            "required_detector_res_float": (
                None
                if self.required_detector_res_float is None
                else list(self.required_detector_res_float)
            ),
            "recommended_detector_res": (
                None if recommended_res is None else list(recommended_res)
            ),
            "focal_length_for_current_detector_mm": self.focal_length_for_current_detector_mm,
            "focal_length_for_current_detector_y_fov_mm": self.focal_length_for_current_detector_y_fov_mm,
            "current_detector_two_pixel_resolution_urad": current_two_pixel,
            "current_detector_f_number_for_target_fov": current_fnum,
            "physical_f_number_floor_in_air": self.physical_f_number_floor,
            "nyquist_frequency_cy_mm": self.nyquist_frequency_cy_mm,
            "analysis_nyquist_frequency_cy_mm": self.analysis_nyquist_frequency_cy_mm,
            "mtf_frequency_is_provisional": not self.detector_pitch_known,
            "virtual_sensor_res": list(self.virtual_sensor_res),
            "virtual_sensor_size_mm": list(self.virtual_sensor_size_mm),
            "wavelength_estimates": {
                str(wavelength): {
                    "diffraction_angle_urad": self.diffraction_angle_urad(wavelength),
                    "airy_diameter_um": self.airy_diameter_um(wavelength),
                    "ideal_system_mtf_at_analysis_nyquist": self.ideal_optical_mtf_at_nyquist(
                        wavelength
                    ),
                }
                for wavelength in self.wavelengths_um
            },
            "max_lenses": self.max_lenses,
            "temperature_c": self.temperature_c,
            "system_mtf_threshold": self.system_mtf_threshold,
            "optical_mtf_target": self.optical_mtf_target,
            "distortion_limit": self.distortion_limit,
            "focal_length_tolerance": self.focal_length_tolerance,
            "f_number_tolerance": self.f_number_tolerance,
            "vignetting_target": self.vignetting_target,
            "vignetting_floor": self.vignetting_floor,
            "notes": list(self.notes),
        }

    def has_sampling_conflict(self) -> bool:
        """判断已确认探测器是否与当前场和采样条件冲突。"""

        if not self.detector_is_known:
            return False
        detector_fov = self.current_detector_y_fov_deg
        if detector_fov is None:
            return False
        # 新基线按探测器高度检查 Y 向全视场，不再把 9.6°误当成对角视场。
        fov_error = abs(detector_fov - self.full_field_y_deg)
        if self.two_pixel_resolution_urad is None:
            return fov_error > 1e-3
        current_resolution = self.two_pixel_resolution_with_current_detector_urad
        return current_resolution is not None and (
            fov_error > 1e-3
            or abs(current_resolution - self.two_pixel_resolution_urad) > 1e-3
        )

    def has_physical_aperture_conflict(self) -> bool:
        """判断当前有效焦距和 280 mm 入瞳是否低于空气中理想 F/0.5。"""

        return self.required_f_number < self.physical_f_number_floor


def _format_report(spec: MWIRDesignSpec) -> str:
    """生成适合终端阅读的中文报告。"""

    report = spec.geometry_report()
    lines = [
        "DeepLens 中波红外透射式望远系统一阶规格检查",
        "=" * 52,
        f"视场定义：Y 方向全视场 {report['field_y_deg']:.3f}°（半视场 {report['half_field_y_deg']:.3f}°）",
        f"像高：{report['image_height_mm']:.4f} mm（Y 向半像高）",
        f"Y 向完整像面高度：{report['full_image_height_y_mm']:.4f} mm",
        f"有效焦距：{report['effective_focal_length_mm']:.4f} mm（由像高和视场推导）",
        f"入瞳直径 / 一阶 F 数：{report['entrance_pupil_diameter_mm']:.1f} mm / F{report['required_f_number']:.4f}",
        f"Y 向目标完整像面高度：{report['required_full_image_height_y_mm']:.4f} mm",
        f"圆形等效设计包络直径：{report['virtual_image_circle_diameter_mm']:.4f} mm",
        "",
        f"探测器规格：{'已确认' if spec.detector_is_known else '待确认'}",
        f"仿真焦面：{report['virtual_sensor_res'][0]} x {report['virtual_sensor_res'][1]}，"
        f"{report['virtual_sensor_size_mm'][0]:.4f} x {report['virtual_sensor_size_mm'][1]:.4f} mm",
        f"MTF 评价频率：{report['analysis_nyquist_frequency_cy_mm']:.3f} cycles/mm"
        + ("（虚拟仿真频率）" if report["mtf_frequency_is_provisional"] else "（探测器奈奎斯特频率）"),
    ]
    if spec.detector_is_known:
        lines.extend(
            [
                f"实际探测器尺寸：{report['current_detector_size_mm'][0]:.3f} x "
                f"{report['current_detector_size_mm'][1]:.3f} mm",
                f"实际探测器 X/Y/对角视场：{report['current_detector_x_fov_deg']:.4f}° / "
                f"{report['current_detector_y_fov_deg']:.4f}° / "
                f"{report['current_detector_diagonal_fov_deg']:.4f}°",
            ]
        )
    else:
        lines.append("探测器像元间距、阵列格式和实际奈奎斯特频率尚未作为硬约束。")
    for wavelength, values in report["wavelength_estimates"].items():
        lines.append(
            f"{wavelength} µm：衍射角 {values['diffraction_angle_urad']:.2f} µrad，"
            f"艾里斑 {values['airy_diameter_um']:.2f} µm，"
            f"理想系统 MTF@仿真奈奎斯特 {values['ideal_system_mtf_at_analysis_nyquist']:.3f}"
        )
    lines.extend(
        [
            "",
            "孔径结论：" + (
                "当前 F 数低于空气中理想 F/0.5，下游设计不能直接采用。"
                if spec.has_physical_aperture_conflict()
                else "当前有效焦距与 280 mm 入瞳的一阶 F 数没有数值孔径冲突。"
            ),
            f"MTF 初始验收：系统级不低于 {spec.system_mtf_threshold:.2f}，"
            f"纯光学设计目标不低于 {spec.optical_mtf_target:.2f}。",
            f"畸变初始上限：{spec.distortion_limit:.2%}；"
            f"边缘相对照度目标：{spec.vignetting_target:.0%}，最低 {spec.vignetting_floor:.0%}。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """命令行入口。"""

    configure_utf8_console()
    parser = argparse.ArgumentParser(description="检查 DeepLens 中波红外透射式望远系统一阶规格")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出全部计算结果")
    args = parser.parse_args()

    spec = MWIRDesignSpec()
    if args.json:
        print(json.dumps(spec.geometry_report(), ensure_ascii=False, indent=2))
    else:
        print(_format_report(spec))


if __name__ == "__main__":
    main()
