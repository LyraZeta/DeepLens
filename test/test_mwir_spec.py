"""中波红外透射系统一阶规格检查测试。"""

import math
import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mwir_telescope_design as mwir_design
from mwir_spec import MWIRDesignSpec
from mwir_telescope_design import (
    DEFAULT_MWIR_LRS,
    MWIR_BALANCED_ACHROMAT_GROUPS,
    MWIR_BALANCED_SURFACE_LIST,
    MWIR_POWER_BENT7_MATERIALS,
    MWIR_POWER_BENT7_SURFACE_LIST,
    MWIR_SURFACE_LIST,
    _balanced_power_design,
    _chief_ray_effective_focal_length,
    _chief_ray_sensor_plate_scale,
    _circular_diffraction_mtf,
    _detached_float,
    _geometric_mtf_from_intercepts,
    _linear_intercept,
    _mwir_dispersion_number,
    _rectangular_pixel_mtf,
    _scheme_parameters,
    _validate_loaded_mwir_lens,
    _validate_source_design_metadata,
    build_initial_lens,
    evaluate_lens,
    load_lens_for_stage,
    optimize_lens,
)
from deeplens.geolens_pkg.optim_init import create_surface
from mwir_power_bent7_optimize import (
    _curved_surfaces,
    constrained_curvatures,
    paraxial_state,
)


def test_mwir_default_uses_zemax_y_field_and_image_height():
    """默认焦距应由 4.8°半视场和 47.1454 mm 半像高推导。"""

    spec = MWIRDesignSpec()

    assert math.isclose(spec.full_field_y_deg, 9.6, rel_tol=1e-12)
    assert math.isclose(spec.half_field_y_deg, 4.8, rel_tol=1e-12)
    assert math.isclose(spec.image_height_mm, 47.1454, rel_tol=1e-12)
    assert math.isclose(spec.image_height_full_y_mm, 94.2908, rel_tol=1e-12)
    assert math.isclose(spec.effective_focal_length_mm, 561.439594707126, rel_tol=1e-10)
    assert math.isclose(spec.required_f_number, 2.00514140966831, rel_tol=1e-10)
    assert not spec.has_physical_aperture_conflict()


def test_mwir_default_detector_is_not_a_hard_constraint():
    """探测器未知时不应再生成 320×256、30 µm 的规格冲突。"""

    spec = MWIRDesignSpec()
    report = spec.geometry_report()

    assert not spec.detector_pitch_known
    assert not spec.detector_format_known
    assert not spec.detector_is_known
    assert spec.detector_size_mm is None
    assert spec.current_detector_diagonal_fov_deg is None
    assert report["mtf_frequency_is_provisional"]
    assert report["current_detector_res"] is None


def test_mwir_virtual_sensor_preserves_image_field_radius():
    """虚拟正方形焦面的半对角应接近 47.1454 mm。"""

    spec = MWIRDesignSpec()
    width, height = spec.virtual_sensor_size_mm
    radial_height = math.hypot(width, height) / 2.0

    assert spec.virtual_sensor_res[0] == spec.virtual_sensor_res[1]
    assert math.isclose(radial_height, spec.image_height_mm, abs_tol=0.03)


def test_mwir_legacy_resolution_constraint_is_opt_in():
    """42 µrad 仅在显式传入时启用，便于复现实验。"""

    spec = MWIRDesignSpec(
        two_pixel_resolution_urad=42.0,
        pixel_pitch_um=30.0,
        detector_res=(320, 256),
    )

    assert spec.resolution_constraint_active
    assert math.isclose(spec.required_focal_length_mm, 1428.57142857, rel_tol=1e-9)
    assert math.isclose(spec.required_f_number, 5.102040816, rel_tol=1e-9)
    assert spec.required_image_height_mm > 100.0


def test_mwir_mtf_estimate_is_physical():
    """2.7–4.3 µm 的理想系统 MTF 估计应随波长下降并高于 0.3。"""

    spec = MWIRDesignSpec()
    mtfs = [spec.ideal_optical_mtf_at_nyquist(w) for w in spec.wavelengths_um]

    assert mtfs[0] > mtfs[1] > mtfs[2]
    assert mtfs[-1] > spec.system_mtf_threshold


def test_mwir_default_scheme_uses_six_lenses_and_front_stop():
    """默认透射基线应使用 6 片镜片、前置光阑和正确的一阶参数。"""

    spec = MWIRDesignSpec()
    params = _scheme_parameters(spec, "transmission_baseline")
    lens_elements = [item for item in MWIR_SURFACE_LIST if item != ["Aperture"]]

    assert MWIR_SURFACE_LIST[0] == ["Aperture"]
    assert len(lens_elements) == params["element_count"] == 6
    assert params["element_count"] <= spec.max_lenses
    assert params["sensor_is_virtual"]
    assert math.isclose(params["field_y_deg"], 9.6, rel_tol=1e-12)
    assert math.isclose(params["image_height_mm"], 47.1454, rel_tol=1e-12)
    assert math.isclose(params["focal_length_mm"], 561.439594707126, rel_tol=1e-10)
    assert math.isclose(params["f_number"], 2.00514140966831, rel_tol=1e-10)
    assert params["total_track_constraint_mm"] is None
    assert (
        inspect.signature(build_initial_lens).parameters["scheme"].default
        == "transmission_baseline"
    )
    assert inspect.signature(optimize_lens).parameters["lrs"].default == DEFAULT_MWIR_LRS


def test_balanced_scheme_uses_three_explicit_achromat_pairs():
    """新方案应保持六片限制，并明确三组正负光焦度材料与曲面拓扑。"""

    spec = MWIRDesignSpec()
    params = _scheme_parameters(spec, "transmission_balanced")
    lens_elements = [
        item for item in MWIR_BALANCED_SURFACE_LIST if item != ["Aperture"]
    ]

    assert MWIR_BALANCED_SURFACE_LIST[0] == ["Aperture"]
    assert len(lens_elements) == params["element_count"] == 6
    assert lens_elements == [
        ["Aspheric", "Spheric"],
        ["Spheric", "Aspheric"],
        ["Aspheric", "Spheric"],
        ["Spheric", "Aspheric"],
        ["Aspheric", "Spheric"],
        ["Spheric", "Aspheric"],
    ]
    assert MWIR_BALANCED_ACHROMAT_GROUPS == (
        ("si", "mgf2", 0.50),
        ("znse", "caf2", 0.30),
        ("si", "mgf2", 0.20),
    )
    assert params["focal_length_mm"] == pytest.approx(
        spec.effective_focal_length_mm
    )


def test_power_bent7_scheme_uses_seven_fully_curved_elements():
    """新母型应使用七片、14 个非零曲面，并保留五个低阶非球面位置。"""

    spec = MWIRDesignSpec()
    params = _scheme_parameters(spec, "transmission_power_bent7")
    lens_elements = [
        item for item in MWIR_POWER_BENT7_SURFACE_LIST if item != ["Aperture"]
    ]

    assert MWIR_POWER_BENT7_SURFACE_LIST[0] == ["Aperture"]
    assert len(lens_elements) == params["element_count"] == 7
    assert params["surface_count"] == 15
    assert list(MWIR_POWER_BENT7_MATERIALS) == [
        "si",
        "mgf2",
        "si",
        "mgf2",
        "znse",
        "caf2",
        "si",
    ]
    assert sum(element.count("Aspheric") for element in lens_elements) == 5


def test_power_bent7_paraxial_parameterization_holds_target_efl(tmp_path):
    """相对曲率扰动后，共同倍率校准仍应把近轴 EFL 固定到任务目标。"""

    spec = MWIRDesignSpec()
    lens, params, _ = build_initial_lens(
        spec,
        scheme="transmission_power_bent7",
        result_dir=tmp_path / "power-bent7",
        device="cpu",
        analyze=False,
    )
    surfaces = _curved_surfaces(lens)
    base = torch.stack([surface.c.detach().clone() for surface in surfaces])
    raw = torch.linspace(-0.25, 0.25, len(surfaces), dtype=base.dtype)
    curvatures, state, relative = constrained_curvatures(
        lens, base, raw, spec.effective_focal_length_mm
    )

    assert len(surfaces) == 14
    assert all(abs(_detached_float(surface.c)) > 1e-6 for surface in surfaces)
    assert sum(getattr(surface, "ai_degree", 0) == 4 for surface in surfaces) == 5
    assert float(relative.min()) >= 1.0 / 1.35 - 1e-5
    assert float(relative.max()) <= 1.35 + 1e-5
    assert float(state.effective_focal_length_mm) == pytest.approx(
        spec.effective_focal_length_mm, rel=2e-6
    )
    direct_state = paraxial_state(lens, curvatures)
    assert float(direct_state.effective_focal_length_mm) == pytest.approx(
        float(state.effective_focal_length_mm), rel=1e-7
    )
    assert params["element_count"] == 7


def test_balanced_power_design_contains_negative_power_and_cancels_color():
    """每组都应含负光焦度元件，且薄透镜一阶色差和接近零。"""

    spec = MWIRDesignSpec()
    design = _balanced_power_design(spec)
    powers = design["element_powers_1_per_mm"]

    assert design["element_materials"] == [
        "si",
        "mgf2",
        "znse",
        "caf2",
        "si",
        "mgf2",
    ]
    assert len(powers) == 6
    assert powers[0::2] == pytest.approx(
        [1.009222e-3, 6.374769e-4, 4.036889e-4], rel=2e-5
    )
    assert powers[1::2] == pytest.approx(
        [-1.186543e-4, -1.031362e-4, -4.746173e-5], rel=2e-5
    )
    assert all(value > 0.0 for value in powers[0::2])
    assert all(value < 0.0 for value in powers[1::2])
    assert sum(powers) == pytest.approx(
        1.0 / spec.effective_focal_length_mm, rel=1e-12, abs=1e-15
    )
    for group in design["groups"]:
        assert group["positive_power_1_per_mm"] > 0.0
        assert group["negative_power_1_per_mm"] < 0.0
        assert group["first_order_color_sum_1_per_mm"] == pytest.approx(
            0.0, abs=1e-18
        )

    assert _mwir_dispersion_number("si") == pytest.approx(202.02, rel=2e-4)
    assert _mwir_dispersion_number("mgf2") == pytest.approx(23.75, rel=2e-4)
    assert _mwir_dispersion_number("znse") == pytest.approx(194.45, rel=2e-4)
    assert _mwir_dispersion_number("caf2") == pytest.approx(31.46, rel=2e-4)


def test_balanced_builder_creates_real_cpu_prescription(tmp_path):
    """CPU 实建处方应具有固定材料顺序、正负功率和七个可优化非球面阶次。"""

    spec = MWIRDesignSpec()
    lens, params, result_path = build_initial_lens(
        spec,
        scheme="transmission_balanced",
        result_dir=tmp_path / "balanced",
        device="cpu",
        analyze=False,
    )

    entry_indices = range(1, 13, 2)
    material_names = [lens.surfaces[index].mat2.name for index in entry_indices]
    thin_powers = []
    for index in entry_indices:
        front = lens.surfaces[index]
        rear = lens.surfaces[index + 1]
        n_primary = float(front.mat2.refractive_index(3.5))
        thin_powers.append(
            (n_primary - 1.0)
            * (_detached_float(front.c) - _detached_float(rear.c))
        )

    assert material_names == ["si", "mgf2", "znse", "caf2", "si", "mgf2"]
    target_powers = _balanced_power_design(spec)["element_powers_1_per_mm"]
    calibration_scales = [
        actual / target for actual, target in zip(thin_powers, target_powers)
    ]
    assert calibration_scales == pytest.approx(
        [calibration_scales[0]] * 6, rel=2e-5
    )
    assert 1.0 < calibration_scales[0] < 1.3
    assert all(value > 0.0 for value in thin_powers[0::2])
    assert all(value < 0.0 for value in thin_powers[1::2])
    assert all(
        getattr(lens.surfaces[index], "ai_degree", 0) == 7
        for index in (1, 4, 5, 8, 9, 12)
    )
    assert abs(float(lens.foclen) - params["focal_length_mm"]) / params[
        "focal_length_mm"
    ] < 0.01
    assert (result_path / "mwir_initial.json").is_file()
    assert (result_path / "mwir_design_metadata.json").is_file()


def test_confirmed_detector_is_reported_without_changing_baseline_geometry():
    """确认探测器后可计算采样，但像高/视场基线仍决定焦距。"""

    spec = MWIRDesignSpec(pixel_pitch_um=30.0, detector_res=(320, 256))

    assert spec.detector_is_known
    assert math.isclose(spec.detector_size_mm[0], 9.6, rel_tol=1e-12)
    assert math.isclose(spec.detector_size_mm[1], 7.68, rel_tol=1e-12)
    assert math.isclose(spec.analysis_nyquist_frequency_cy_mm, 16.6666666667, rel_tol=1e-9)
    assert spec.current_detector_y_fov_deg < 1.0
    assert math.isclose(spec.required_focal_length_mm, spec.effective_focal_length_mm)

    finer_pitch = MWIRDesignSpec(pixel_pitch_um=15.0, detector_res=(640, 512))
    assert math.isclose(
        finer_pitch.analysis_nyquist_frequency_cy_mm,
        33.3333333333,
        rel_tol=1e-9,
    )


def test_required_detector_size_uses_full_y_image_height_not_diagonal():
    """已知 5:4 纵横比时，应固定 Y 高度 94.2908 mm，而非固定对角线。"""

    spec = MWIRDesignSpec(pixel_pitch_um=30.0, detector_res=(320, 256))
    width_mm, height_mm = spec.required_sensor_size_mm

    assert math.isclose(height_mm, 94.2908, rel_tol=1e-12)
    assert math.isclose(width_mm, 94.2908 * 320 / 256, rel_tol=1e-12)
    assert math.isclose(
        spec.required_sensor_diagonal_mm,
        math.hypot(width_mm, height_mm),
        rel_tol=1e-12,
    )
    assert spec.recommended_detector_res == (3930, 3144)


def test_large_fpa_preserves_y_height_and_uses_radial_deeplens_field():
    """大焦面方案的物理 Y 高度和 DeepLens 径向场应同时保持焦距一致。"""

    spec = MWIRDesignSpec(
        two_pixel_resolution_urad=42.0,
        pixel_pitch_um=30.0,
        detector_res=(320, 256),
    )
    params = _scheme_parameters(spec, "large_fpa")

    target_height = 2.0 * spec.required_image_height_mm
    assert params["sensor_size_mm"][1] >= target_height
    assert params["sensor_size_mm"][1] - target_height < 4 * spec.pixel_pitch_mm
    assert math.isclose(
        params["sensor_size_mm"][0] / params["sensor_size_mm"][1],
        320 / 256,
        rel_tol=1e-12,
    )
    assert params["optimization_radial_image_height_mm"] > params["image_height_y_mm"]
    assert params["optimization_radial_fov_deg"] > params["field_y_deg"]
    recovered_focal_length = params["optimization_radial_image_height_mm"] / math.tan(
        math.radians(params["optimization_radial_fov_deg"] / 2.0)
    )
    assert math.isclose(
        recovered_focal_length,
        params["focal_length_mm"],
        rel_tol=1e-12,
    )


def test_large_fpa_without_detector_uses_required_virtual_image_radius():
    """探测器未知时，42 µrad 对照方案不能复用正式基线的小虚拟焦面。"""

    spec = MWIRDesignSpec(two_pixel_resolution_urad=42.0)
    params = _scheme_parameters(spec, "large_fpa")

    assert params["sensor_is_virtual"]
    assert math.isclose(
        params["optimization_radial_image_height_mm"],
        spec.required_image_height_mm,
        rel_tol=1e-12,
    )
    assert math.isclose(
        params["optimization_radial_fov_deg"],
        spec.full_field_y_deg,
        rel_tol=1e-12,
    )


def test_large_aperture_asphere_initialization_remains_finite():
    """140 mm 半口径的高阶非球面初值不能因 r^18 溢出而产生 inf。"""

    surface = create_surface(
        "Aspheric",
        d_total=0.0,
        aper_r=140.0,
        imgh=47.1454,
        mat="znse",
        curvature_scale=2e-4,
    )
    radius = torch.linspace(0.0, 140.0, 64)
    zeros = torch.zeros_like(radius)

    assert torch.isfinite(surface.sag(radius, zeros)).all()
    assert torch.isfinite(surface.dfdxyz(radius, zeros)[0]).all()

    optimizer_groups = surface.get_optimizer_params(
        lrs=[1e-3, 1e-6, 1e-3, 1e-5]
    )
    optimized_parameters = [
        parameter
        for group in optimizer_groups
        for parameter in group["params"]
    ]
    assert any(parameter is surface.ai16 for parameter in optimized_parameters)
    assert all(parameter is not surface.ai18 for parameter in optimized_parameters)
    assert not surface.ai18.requires_grad


def test_mtf_factors_are_translation_invariant_and_physically_scaled():
    """几何 OTF 不受平移影响，衍射和像元因子应符合解析结果。"""

    centered = torch.tensor(
        [[-0.001, -0.002], [0.001, 0.002], [-0.001, 0.002], [0.001, -0.002]]
    )
    shifted = centered + torch.tensor([17.0, -9.0])
    frequency = 1.0 / (2.0 * 0.03)

    mtf_centered = _geometric_mtf_from_intercepts(centered, frequency)
    mtf_shifted = _geometric_mtf_from_intercepts(shifted, frequency)

    assert mtf_centered == pytest.approx(mtf_shifted, rel=1e-5, abs=1e-7)
    assert _rectangular_pixel_mtf(frequency, 0.03) == pytest.approx(
        2.0 / math.pi, rel=1e-12
    )
    spec = MWIRDesignSpec()
    diffraction = [
        _circular_diffraction_mtf(frequency, wavelength, spec.required_f_number)
        for wavelength in spec.wavelengths_um
    ]
    assert diffraction == pytest.approx([0.8853, 0.8514, 0.8177], abs=5e-4)


def test_chief_ray_sensor_plate_scale_uses_symmetric_zero_field_extrapolation():
    """正负主光线应消除偏置，并把三次项外推回零视场板尺。"""

    class LensStub:
        device = torch.device("cpu")
        dtype = torch.float32
        aper_idx = 0
        d_sensor = torch.tensor(100.0)

        def calc_chief_ray_infinite(
            self, rfov, wvln, plane, ray_aiming=False
        ):
            tangent = torch.tan(torch.deg2rad(rfov))
            focal_length = 510.0 if plane == "meridional" else 490.0
            cubic = 12_000.0
            offset = 0.37
            height_without_offset = focal_length * tangent + cubic * tangent**3
            coordinate = 1 if plane == "meridional" else 0
            origin = torch.zeros((len(rfov), 3), dtype=torch.float32)
            direction = torch.zeros_like(origin)
            origin[:, coordinate] = offset
            direction[:, coordinate] = height_without_offset / 100.0
            direction[:, 2] = 1.0
            return origin, direction

        def trace(self, ray):
            return ray, None

    result = _chief_ray_sensor_plate_scale(LensStub())

    assert result["meridional"] == pytest.approx(510.0, abs=2e-3)
    assert result["sagittal"] == pytest.approx(490.0, abs=2e-3)
    assert _linear_intercept([0.0, 1.0, 2.0], [7.0, 10.0, 13.0]) == pytest.approx(
        7.0, abs=1e-12
    )


def test_strict_effective_focal_length_extrapolates_paraxial_focus_and_field():
    """严格 EFL 路径应同时外推零瞳高焦面和零视场主光线板尺。"""

    class LensStub:
        device = torch.device("cpu")
        dtype = torch.float32
        aper_idx = 0
        entr_pupilr = torch.tensor(100.0)
        d_sensor = torch.tensor(590.0)
        surfaces = [
            SimpleNamespace(d=torch.tensor(0.0)),
            SimpleNamespace(d=torch.tensor(10.0)),
        ]

        def calc_chief_ray_infinite(
            self, rfov, wvln, plane, ray_aiming=False
        ):
            tangent = torch.tan(torch.deg2rad(rfov))
            origin = torch.zeros((len(rfov), 3), dtype=torch.float32)
            direction = torch.zeros_like(origin)
            coordinate = 1 if plane == "meridional" else 0
            direction[:, coordinate] = tangent
            direction[:, 2] = 1.0
            return origin, direction

        def trace(self, ray):
            focus_z = 600.0
            focal_length = 500.0
            new_origin = torch.zeros_like(ray.o)
            new_direction = torch.zeros_like(ray.d)
            new_origin[:, 2] = 10.0
            new_direction[:, 2] = 1.0
            incoming_x = ray.d[:, 0] / ray.d[:, 2]
            incoming_y = ray.d[:, 1] / ray.d[:, 2]
            if (incoming_x.abs() + incoming_y.abs()).max() > 1e-12:
                coordinate = 0 if incoming_x.abs().max() > 0 else 1
                tangent = incoming_x if coordinate == 0 else incoming_y
                image_height = focal_length * tangent + 8_000.0 * tangent**3
                new_direction[:, coordinate] = image_height / (focus_z - 10.0)
            else:
                coordinate = 0 if ray.o[:, 0].abs().max() > 0 else 1
                pupil_height = ray.o[:, coordinate]
                crossing_z = focus_z + 2.0 * pupil_height**2
                new_origin[:, coordinate] = pupil_height
                new_direction[:, coordinate] = -pupil_height / (
                    crossing_z - 10.0
                )
            ray.o = new_origin
            ray.d = new_direction
            return ray, None

    result = _chief_ray_effective_focal_length(LensStub())

    assert result["paraxial_focus_z_by_plane_mm"] == pytest.approx(
        {"meridional": 600.0, "sagittal": 600.0}, abs=2e-3
    )
    assert result["effective_focal_length_by_plane_mm"] == pytest.approx(
        {"meridional": 500.0, "sagittal": 500.0}, abs=2e-3
    )


def test_loaded_stage_lens_must_match_current_mwir_geometry():
    """续跑处方必须保持波长、入瞳、焦面、前置光阑和镜片数一致。"""

    spec = MWIRDesignSpec()
    params = _scheme_parameters(spec, "transmission_baseline")
    materials = ["air"] + [name for _ in range(6) for name in ("ge", "air")]
    lens = SimpleNamespace(
        wvln_rgb=list(spec.wavelengths_um),
        primary_wvln=3.5,
        obj_depth=spec.object_distance_mm,
        aper_idx=0,
        entr_pupilr=torch.tensor(140.0),
        r_sensor=params["optimization_radial_image_height_mm"],
        sensor_res=tuple(params["sensor_res"]),
        surfaces=[
            SimpleNamespace(mat2=SimpleNamespace(name=name)) for name in materials
        ],
    )

    checks = _validate_loaded_mwir_lens(lens, spec, params)

    assert checks["element_count"] == 6
    assert checks["front_stop"] is True
    lens.sensor_res = (320, 256)
    with pytest.raises(ValueError, match="焦面分辨率"):
        _validate_loaded_mwir_lens(lens, spec, params)


def test_stage_metadata_rejects_silent_field_retargeting():
    """同一焦面处方不能在续跑时静默改成另一组视场/焦距目标。"""

    source_params = _scheme_parameters(MWIRDesignSpec(), "transmission_baseline")
    metadata = {"design": source_params}

    checks = _validate_source_design_metadata(metadata, source_params)
    assert checks["field_y_deg"] == pytest.approx(9.6)

    retargeted_params = _scheme_parameters(
        MWIRDesignSpec(field_y_deg=8.0), "transmission_baseline"
    )
    with pytest.raises(ValueError, match="不是同规格续跑"):
        _validate_source_design_metadata(metadata, retargeted_params)


def test_stage_loader_refuses_nonempty_output_directory(tmp_path):
    """续跑必须写入新的空目录，不能覆盖旧 metadata、检查点或最终处方。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    input_path = source_dir / "mwir_final.json"
    input_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "existing-stage"
    output_dir.mkdir()
    (output_dir / "mwir_final.json").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="新的空输出目录"):
        load_lens_for_stage(
            MWIRDesignSpec(),
            input_lens=input_path,
            result_dir=output_dir,
            device="cpu",
        )


def test_mwir_evaluation_samples_fixed_infinite_y_field_angles(tmp_path, monkeypatch):
    """MTF 评价必须直接采样 0°、3.36°、4.8° 的无穷远 Y 场。"""

    class SurfaceStub:
        d = 0.0

    class RayStub:
        def __init__(self, num_rays):
            coordinate = torch.linspace(-0.002, 0.002, num_rays)
            self.o = torch.stack(
                [coordinate, torch.flip(coordinate, dims=[0]), torch.zeros_like(coordinate)],
                dim=-1,
            )
            self.is_valid = torch.ones(num_rays)
            self.is_coherent = False

    class LensStub:
        foclen = 561.439594707126
        fnum = 2.00514140966831
        entr_pupilr = 140.0
        d_sensor = 600.0
        surfaces = [SurfaceStub()]
        sensor_size = (66.67467, 66.67467)
        sensor_res = (2222, 2222)
        pixel_size = 0.03
        obj_depth = -10_000_000.0
        device = torch.device("cpu")

        def __init__(self):
            self.field_calls = []

        def post_computation(self):
            pass

        def sample_from_fov(self, **kwargs):
            self.field_calls.append((kwargs["fov_y"], kwargs["depth"]))
            return RayStub(kwargs["num_rays"])

        def trace2sensor(self, ray):
            return ray

        def vignetting(self, **kwargs):
            return torch.ones((2, 2))

    lens = LensStub()
    spec = MWIRDesignSpec()

    def fake_chief_heights(lens, half_field_deg, wavelength_um, plane, num_points=9):
        field_angles = np.linspace(0.0, half_field_deg, num_points)
        compute_angles = field_angles.copy()
        actual_height = spec.required_focal_length_mm * np.tan(
            np.deg2rad(compute_angles)
        )
        return field_angles, compute_angles, actual_height

    def fake_strict_focal_length(lens, wavelength_um=3.5):
        by_plane = {
            "meridional": spec.required_focal_length_mm,
            "sagittal": spec.required_focal_length_mm,
        }
        return {
            "effective_focal_length_by_plane_mm": by_plane,
            "effective_focal_length_mean_mm": spec.required_focal_length_mm,
            "paraxial_focus_z_by_plane_mm": {
                "meridional": 610.0,
                "sagittal": 610.0,
            },
        }

    def fake_plate_scale(lens, wavelength_um=3.5):
        return {
            "meridional": spec.required_focal_length_mm,
            "sagittal": spec.required_focal_length_mm,
        }

    monkeypatch.setattr(mwir_design, "_chief_ray_image_heights", fake_chief_heights)
    monkeypatch.setattr(
        mwir_design, "_chief_ray_effective_focal_length", fake_strict_focal_length
    )
    monkeypatch.setattr(
        mwir_design, "_chief_ray_sensor_plate_scale", fake_plate_scale
    )

    evaluate_lens(lens, spec, tmp_path, psf_spp=4, psf_ks=4)

    expected = [
        (-relative_fov * spec.half_field_y_deg, float("inf"))
        for _ in spec.wavelengths_um
        for relative_fov in (0.0, 0.7, 1.0)
    ]
    assert lens.field_calls == expected


def test_evaluation_separates_strict_efl_plate_scale_and_target_mapping(
    tmp_path, monkeypatch
):
    """EFL、传统畸变和固定目标像高必须使用三种不同且明确的参考。"""

    spec = MWIRDesignSpec()
    target = spec.required_focal_length_mm
    plate_scale = 0.98 * target

    class SurfaceStub:
        d = 0.0
        mat2 = SimpleNamespace(name="air")

    class RayStub:
        def __init__(self, num_rays):
            coordinate = torch.linspace(-0.002, 0.002, num_rays)
            self.o = torch.stack(
                [coordinate, torch.flip(coordinate, dims=[0]), torch.zeros_like(coordinate)],
                dim=-1,
            )
            self.is_valid = torch.ones(num_rays)
            self.is_coherent = False

    class LensStub:
        foclen = 1.04 * target
        fnum = foclen / 280.0
        entr_pupilr = 140.0
        d_sensor = 600.0
        surfaces = [SurfaceStub()]
        sensor_size = (66.67467, 66.67467)
        sensor_res = (2222, 2222)
        device = torch.device("cpu")

        def post_computation(self):
            pass

        def sample_from_fov(self, **kwargs):
            return RayStub(kwargs["num_rays"])

        def trace2sensor(self, ray):
            return ray

        def vignetting(self, **kwargs):
            return torch.ones((2, 2))

    def fake_strict_focal_length(lens, wavelength_um=3.5):
        by_plane = {
            "meridional": 0.98 * target,
            "sagittal": 1.02 * target,
        }
        return {
            "effective_focal_length_by_plane_mm": by_plane,
            "effective_focal_length_mean_mm": target,
            "paraxial_focus_z_by_plane_mm": {
                "meridional": 615.0,
                "sagittal": 615.0,
            },
        }

    def fake_plate_scale(lens, wavelength_um=3.5):
        return {"meridional": plate_scale, "sagittal": plate_scale}

    def fake_chief_heights(lens, half_field_deg, wavelength_um, plane, num_points=9):
        field_angles = np.linspace(0.0, half_field_deg, num_points)
        actual_height = plate_scale * np.tan(np.deg2rad(field_angles))
        return field_angles, field_angles.copy(), actual_height

    monkeypatch.setattr(
        mwir_design, "_chief_ray_effective_focal_length", fake_strict_focal_length
    )
    monkeypatch.setattr(
        mwir_design, "_chief_ray_sensor_plate_scale", fake_plate_scale
    )
    monkeypatch.setattr(mwir_design, "_chief_ray_image_heights", fake_chief_heights)

    metrics = evaluate_lens(LensStub(), spec, tmp_path, psf_spp=4, psf_ks=4)

    assert metrics["focal_length_mm"] == pytest.approx(target)
    assert metrics["geolens_cached_focal_length_mm"] == pytest.approx(1.04 * target)
    assert metrics["focal_length_relative_error"] == pytest.approx(0.02)
    assert metrics["pass"]["focal_length"] is False
    assert metrics["pass"]["f_number"] is False
    assert metrics["pass"]["distortion"] is True
    assert metrics["pass"]["target_field_mapping"] is False
    assert metrics["distortion"]["3.5"]["meridional"][
        "reference_plate_scale_focal_length_mm"
    ] == pytest.approx(plate_scale)


def test_detached_float_does_not_warn_for_trainable_scalar():
    """评价输出标量时不应直接把 requires_grad 张量交给 float。"""

    value = torch.tensor(12.5, requires_grad=True)

    assert _detached_float(value) == pytest.approx(12.5)


@pytest.mark.parametrize(
    ("extra_options", "expected_checkpoint_analysis"),
    [({}, False), ({"checkpoint_analysis": True}, True)],
)
def test_mwir_optimization_prioritizes_field_mapping_and_controls_checkpoints(
    tmp_path, extra_options, expected_checkpoint_analysis
):
    """MWIR 第一阶段应提高像高权重，并能控制检查点完整分析。"""

    class LensStub:
        def __init__(self):
            self.kwargs = None

        def optimize(self, **kwargs):
            self.kwargs = kwargs

        def post_computation(self):
            pass

        def write_lens_json(self, path):
            self.final_path = path

    lens = LensStub()
    spec = MWIRDesignSpec()
    params = _scheme_parameters(spec, "transmission_baseline")

    optimize_lens(
        lens,
        spec,
        tmp_path,
        iterations=1,
        design_params=params,
        num_ring=1,
        num_arm=1,
        spp=1,
        rms_weight=0.3,
        lrs=(2e-3, 0.0, 2e-4, 2e-6),
        **extra_options,
    )

    assert lens.kwargs["lrs"] == pytest.approx([2e-3, 0.0, 2e-4, 2e-6])
    assert lens.kwargs["w_rms"] == pytest.approx(0.3)
    assert lens.kwargs["w_mtf"] == pytest.approx(0.0)
    assert lens.kwargs["mtf_frequency_cy_mm"] == pytest.approx(
        spec.analysis_nyquist_frequency_cy_mm
    )
    assert 0.5 < lens.kwargs["mtf_target"] < 1.0
    assert lens.kwargs["mtf_max_weight"] == pytest.approx(1.0)
    assert lens.kwargs["mtf_field_fractions"] == pytest.approx((0.0, 0.7, 1.0))
    assert lens.kwargs["ray_resample_interval"] == 1
    assert lens.kwargs["target_f_number"] == pytest.approx(
        spec.required_f_number
    )
    assert lens.kwargs["first_order_preferred_relative_error"] == pytest.approx(
        0.008
    )
    assert lens.kwargs["first_order_hard_relative_error"] == pytest.approx(0.01)
    assert lens.kwargs["w_field"] == pytest.approx(1.0)
    assert lens.kwargs["w_reg"] == pytest.approx(0.1)
    assert lens.kwargs["field_mapping_all_wavelengths"] is True
    assert lens.kwargs["field_mapping_max_weight"] == pytest.approx(1.0)
    assert lens.kwargs["field_mapping_use_chief_ray"] is True
    assert lens.kwargs["field_mapping_num_points"] == 9
    assert lens.kwargs["checkpoint_analysis"] is expected_checkpoint_analysis
    assert lens.kwargs["target_focal_length"] == pytest.approx(
        spec.required_focal_length_mm
    )
