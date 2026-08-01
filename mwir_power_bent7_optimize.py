"""七片强弯曲 MWIR 透射系统的受约束球面优化器。

旧的 ``transmission_balanced`` 处方把每片功率集中在单面，容易退化成一组
近似玻璃平板。本脚本从 ``transmission_power_bent7`` 母型出发，以相对曲率
而不是绝对曲率作为优化变量，并在每次前向计算中重新标定统一曲率倍率，使
3.5 µm 近轴有效焦距保持在任务目标。光阑和全部顶点位置在首阶段固定，像面
始终跟随近轴焦面，仅优化一个有界的最佳焦点偏移。结构阶段再开放七个玻璃
中心厚度和六个空气间隔，并继续联合优化曲率与低阶非球面。

这一阶段只解决球面母型与真实光焦度分配，不会宣称已经达到最终 MTF。球面
RMS 明显下降后，再把预留的五个面逐级放开圆锥常数和 A4/A6/A8/A10。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from deeplens.utils import set_logger, set_seed
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import (
    _apply_mwir_constraints,
    _build_power_bent7_lens,
    _calibrate_initial_power,
    _detached_float,
    _circular_diffraction_mtf,
    _rectangular_pixel_mtf,
    _scheme_parameters,
    evaluate_lens,
)


# 非球面优化支持从 A4 起的可变阶数。旧处方只有 A4--A10 四项时仍保持
# 原行为；实验处方可以通过在 JSON 中预留 A12/A14/A16 来增加高阶校正自由度。
ASPHERE_ORDERS = (4, 6, 8, 10, 12, 14, 16)
ASPHERE_EDGE_SPANS_MM = (2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)


def _curriculum_scale(
    iteration: int,
    total_iterations: int,
    *,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    """返回辅助像质项在当前迭代使用的平滑权重倍率。

    预热区间内倍率为 0，使优化器先处理质心 RMS 和基础可行性约束；随后用
    smoothstep 从 0 平滑升至 1。两个比例都为 0 时保持旧行为，即从首步起使用
    完整权重。
    """

    if total_iterations <= 0:
        raise ValueError("total_iterations 必须为正整数。")
    if not 0 <= iteration <= total_iterations:
        raise ValueError("iteration 必须位于 [0, total_iterations]。")
    for name, value in (
        ("warmup_fraction", warmup_fraction),
        ("ramp_fraction", ramp_fraction),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须为非负有限值。")
    if warmup_fraction + ramp_fraction > 1.0 + 1e-12:
        raise ValueError("curriculum 的预热比例与渐入比例之和不能超过 1。")
    if warmup_fraction == 0.0 and ramp_fraction == 0.0:
        return 1.0

    progress = iteration / total_iterations
    if progress <= warmup_fraction:
        return 0.0
    if ramp_fraction == 0.0:
        return 1.0
    ramp_progress = min(
        max((progress - warmup_fraction) / ramp_fraction, 0.0),
        1.0,
    )
    return ramp_progress * ramp_progress * (3.0 - 2.0 * ramp_progress)


def _curriculum_ray_weights(
    scale: float,
    *,
    mtf_surrogate_weight: float = 0.0,
    direct_mtf_weight: float = 0.0,
    focus_weight: float = 0.0,
    astigmatism_weight: float = 0.0,
    chromatic_focus_weight: float = 0.0,
    field_curvature_weight: float = 0.0,
) -> dict[str, float]:
    """按同一 curriculum 倍率缩放所有辅助光线 merit 权重。"""

    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("curriculum scale 必须位于 [0, 1]。")
    final_weights = {
        "mtf_surrogate_weight": mtf_surrogate_weight,
        "direct_mtf_weight": direct_mtf_weight,
        "focus_weight": focus_weight,
        "astigmatism_weight": astigmatism_weight,
        "chromatic_focus_weight": chromatic_focus_weight,
        "field_curvature_weight": field_curvature_weight,
    }
    for name, value in final_weights.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} 必须为非负有限值。")
    return {name: scale * value for name, value in final_weights.items()}


@dataclass(frozen=True)
class ParaxialState:
    """空气中系统矩阵对应的一阶量。"""

    effective_focal_length_mm: torch.Tensor
    back_focal_length_mm: torch.Tensor
    focus_z_mm: torch.Tensor


def _curved_surfaces(lens) -> list[Any]:
    """按光路顺序返回具有基础曲率 ``c`` 的折射面。"""

    return [surface for surface in lens.surfaces if hasattr(surface, "c")]


def _surface_indices_3p5(
    lens,
    *,
    dtype: torch.dtype,
    device: torch.device,
    wavelength_um: float = 3.5,
):
    """返回每个曲面前后在指定波长处的折射率常量。

    函数名为兼容既有实验保留；现在不再把 3.5 µm 写死，从而可用同一
    近轴矩阵显式计算 2.7/3.5/4.3 µm 的色焦。
    """

    if not math.isfinite(float(wavelength_um)) or float(wavelength_um) <= 0.0:
        raise ValueError("wavelength_um 必须是正的有限数。")
    wavelength = torch.tensor(float(wavelength_um), dtype=dtype, device=device)
    n_before: list[torch.Tensor] = []
    n_after: list[torch.Tensor] = []
    current = torch.ones((), dtype=dtype, device=device)
    for surface in _curved_surfaces(lens):
        next_index = surface.mat2.ior(wavelength).to(dtype=dtype, device=device)
        n_before.append(current)
        n_after.append(next_index)
        current = next_index
    return n_before, n_after


def paraxial_state(
    lens,
    curvatures: torch.Tensor,
    *,
    wavelength_um: float = 3.5,
) -> ParaxialState:
    """用 ``[y, nθ]`` 光线矩阵计算可微 EFL、BFL 和近轴焦面。

    传播矩阵为 ``[[1, t/n], [0, 1]]``，折射矩阵为
    ``[[1, 0], [-(n2-n1)c, 1]]``。对于最后介质为空气的同轴系统，
    ``EFL=-1/C``、``BFL=-A/C``。该结果与项目的严格小视场主光线评价在
    数值精度内一致，但保留了曲率梯度。
    """

    surfaces = _curved_surfaces(lens)
    if curvatures.ndim != 1 or curvatures.numel() != len(surfaces):
        raise ValueError("curvatures 必须是一维张量，且数量与可曲面一致。")
    dtype = curvatures.dtype
    device = curvatures.device
    n_before, n_after = _surface_indices_3p5(
        lens,
        dtype=dtype,
        device=device,
        wavelength_um=wavelength_um,
    )
    a = torch.ones((), dtype=dtype, device=device)
    b = torch.zeros((), dtype=dtype, device=device)
    c_matrix = torch.zeros((), dtype=dtype, device=device)
    d = torch.ones((), dtype=dtype, device=device)
    previous_z = torch.as_tensor(
        _detached_float(lens.surfaces[0].d), dtype=dtype, device=device
    )

    for surface, curvature, n1, n2 in zip(
        surfaces, curvatures, n_before, n_after
    ):
        surface_z = surface.d.to(dtype=dtype, device=device)
        propagation = (surface_z - previous_z) / n1
        a, b = a + propagation * c_matrix, b + propagation * d
        power = (n2 - n1) * curvature
        c_matrix, d = c_matrix - power * a, d - power * b
        previous_z = surface_z

    if n_after and not torch.isclose(
        n_after[-1], torch.ones_like(n_after[-1]), atol=1e-6, rtol=0.0
    ):
        raise ValueError("最后一个折射面之后必须为空气，才能使用当前 EFL/BFL 定义。")
    effective_focal_length = -1.0 / c_matrix
    back_focal_length = -a / c_matrix
    return ParaxialState(
        effective_focal_length_mm=effective_focal_length,
        back_focal_length_mm=back_focal_length,
        focus_z_mm=previous_z + back_focal_length,
    )


def constrained_curvatures(
    lens,
    base_curvatures: torch.Tensor,
    shape_raw: torch.Tensor,
    target_focal_length_mm: float,
    *,
    relative_log_span: float = math.log(1.35),
    calibration_iterations: int = 12,
) -> tuple[torch.Tensor, ParaxialState, torch.Tensor]:
    """生成保号、有界并自动保持目标 EFL 的曲率。

    每个面的相对倍率位于 ``[1/1.35, 1.35]``。随后对全部曲率施加共同倍率，
    用多次乘法校准把 EFL 拉回目标；共同倍率不改变各面的相对弯曲自由度。
    """

    if base_curvatures.shape != shape_raw.shape:
        raise ValueError("base_curvatures 与 shape_raw 必须具有相同 shape。")
    if calibration_iterations <= 0:
        raise ValueError("calibration_iterations 必须为正整数。")
    relative = torch.exp(relative_log_span * torch.tanh(shape_raw))
    common_scale = torch.ones(
        (), dtype=base_curvatures.dtype, device=base_curvatures.device
    )
    target = torch.as_tensor(
        target_focal_length_mm,
        dtype=base_curvatures.dtype,
        device=base_curvatures.device,
    )
    for _ in range(calibration_iterations):
        trial = base_curvatures * relative * common_scale
        state = paraxial_state(lens, trial)
        correction = (state.effective_focal_length_mm.abs() / target).clamp(
            0.8, 1.2
        )
        common_scale = common_scale * correction
    curvatures = base_curvatures * relative * common_scale
    return curvatures, paraxial_state(lens, curvatures), relative


def _sphere_sag(curvature: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    """计算球面矢高，并在数值边界内保持有限。"""

    argument = (1.0 - curvature.square() * radius.square()).clamp_min(1e-8)
    return curvature * radius.square() / (1.0 + torch.sqrt(argument))


def _clearance_penalty(lens, curvatures: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """惩罚负边缘厚度、空气交叠以及像面碰撞。"""

    surfaces = _curved_surfaces(lens)
    if len(surfaces) % 2:
        raise ValueError("七片母型的可曲面数量必须为偶数。")
    penalties: list[torch.Tensor] = []
    glass_clearances: list[torch.Tensor] = []
    air_clearances: list[torch.Tensor] = []
    for element_index in range(len(surfaces) // 2):
        front = surfaces[2 * element_index]
        rear = surfaces[2 * element_index + 1]
        radius = torch.as_tensor(
            min(float(front.r), float(rear.r)),
            dtype=curvatures.dtype,
            device=curvatures.device,
        )
        zeros = torch.zeros_like(radius)
        front_edge = front.d + front._sag(radius, zeros)
        rear_edge = rear.d + rear._sag(radius, zeros)
        glass = rear_edge - front_edge
        glass_clearances.append(glass)
        penalties.append(torch.relu(3.0 - glass))

        if element_index + 1 < len(surfaces) // 2:
            next_front = surfaces[2 * element_index + 2]
            common_radius = torch.as_tensor(
                min(float(rear.r), float(next_front.r)),
                dtype=curvatures.dtype,
                device=curvatures.device,
            )
            common_zeros = torch.zeros_like(common_radius)
            current_edge = rear.d + rear._sag(common_radius, common_zeros)
            next_edge = next_front.d + next_front._sag(
                common_radius, common_zeros
            )
            air = next_edge - current_edge
            air_clearances.append(air)
            penalties.append(torch.relu(2.0 - air))

    if penalties:
        penalty = torch.stack(penalties).square().mean().sqrt()
    else:
        penalty = torch.zeros((), dtype=curvatures.dtype, device=curvatures.device)
    diagnostics = {
        "minimum_glass_edge_mm": float(
            torch.stack(glass_clearances).min().detach().cpu()
        ),
        "minimum_air_edge_mm": float(
            torch.stack(air_clearances).min().detach().cpu()
        ),
    }
    return penalty, diagnostics


def _sample_fixed_rays(
    lens,
    spec: MWIRDesignSpec,
    *,
    field_count: int,
    spp: int,
    seed: int,
    pupil_scale: float,
):
    if field_count < 3:
        raise ValueError("field_count 至少为 3。")
    if spp < 8:
        raise ValueError("spp 至少为 8。")
    field_degrees = torch.linspace(
        0.0,
        spec.half_field_y_deg,
        field_count,
        device=lens.device,
        dtype=lens.dtype,
    )
    field_values = field_degrees.detach().cpu().tolist()
    batches = []
    chief_batches = []
    for wavelength in spec.wavelengths_um:
        # 各波长使用相同随机种子，使瞳采样差异不会伪装成轴向色差。
        set_seed(seed)
        batches.append(
            lens.sample_from_fov(
                fov_x=0.0,
                fov_y=field_values,
                depth=float("inf"),
                num_rays=spp,
                wvln=wavelength,
                scale_pupil=pupil_scale,
            )
        )
        chief_batches.append(
            lens.sample_from_fov(
                fov_x=0.0,
                fov_y=field_values,
                depth=float("inf"),
                num_rays=1,
                wvln=wavelength,
                scale_pupil=0.0,
            )
        )
    target_y = spec.effective_focal_length_mm * torch.tan(
        field_degrees * math.pi / 180.0
    )
    target_xy = torch.stack([torch.zeros_like(target_y), target_y], dim=-1)
    return batches, chief_batches, target_xy


def _assign_geometry(lens, curvatures: torch.Tensor, sensor_z: torch.Tensor) -> None:
    """把当前可微几何挂到镜头对象上。"""

    for surface, curvature in zip(_curved_surfaces(lens), curvatures):
        surface.c = curvature
    lens.d_sensor = sensor_z


def constrained_surface_positions(
    first_surface_z: torch.Tensor,
    base_gaps: torch.Tensor,
    gap_raw: torch.Tensor,
    *,
    glass_gap_ratio: float = 1.25,
    air_gap_ratio: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """生成正值、有界的七片中心厚度和六个空气间隔。

    十四个折射面之间共有十三个轴向间隔：偶数下标是玻璃中心厚度，奇数
    下标是空气间隔。零参数严格对应输入处方；玻璃厚度限制在基准值的
    ``[1/ratio, ratio]``，空气间隔使用更宽的同类范围。第一折射面的顶点
    位置保持不动，其余顶点由间隔累加得到。
    """

    if base_gaps.ndim != 1 or gap_raw.shape != base_gaps.shape:
        raise ValueError("base_gaps 与 gap_raw 必须是形状相同的一维张量。")
    if base_gaps.numel() < 1:
        raise ValueError("结构至少需要两个折射面，才能定义面间隔。")
    if glass_gap_ratio <= 1.0 or air_gap_ratio <= 1.0:
        raise ValueError("玻璃和空气间隔倍率必须大于 1。")
    if not bool(torch.all(base_gaps > 0.0).detach().cpu()):
        raise ValueError("输入处方的全部中心厚度和空气间隔必须为正。")

    indices = torch.arange(
        base_gaps.numel(), device=base_gaps.device, dtype=torch.long
    )
    glass_span = torch.as_tensor(
        math.log(glass_gap_ratio), dtype=base_gaps.dtype, device=base_gaps.device
    )
    air_span = torch.as_tensor(
        math.log(air_gap_ratio), dtype=base_gaps.dtype, device=base_gaps.device
    )
    spans = torch.where(indices.remainder(2) == 0, glass_span, air_span)
    relative = torch.exp(spans * torch.tanh(gap_raw))
    gaps = base_gaps * relative
    positions = first_surface_z + torch.cat(
        [
            torch.zeros(1, dtype=base_gaps.dtype, device=base_gaps.device),
            torch.cumsum(gaps, dim=0),
        ]
    )
    return positions, gaps, relative


def _assign_surface_positions(lens, positions: torch.Tensor) -> None:
    """把十四个折射面的绝对顶点坐标挂到镜头对象上。"""

    surfaces = _curved_surfaces(lens)
    if positions.shape != (len(surfaces),):
        raise ValueError("positions 的数量与折射面数量不一致。")
    for surface, position in zip(surfaces, positions):
        surface.d = position


def _set_element_materials(lens, material_names: list[str] | tuple[str, ...]) -> None:
    """替换每片透镜的透射材料，并重建相邻介质缓存。"""

    from deeplens.material import Material

    surfaces = _curved_surfaces(lens)
    if len(surfaces) % 2:
        raise ValueError("透镜折射面数量必须为偶数，才能按透镜替换材料。")
    if len(material_names) != len(surfaces) // 2:
        raise ValueError(
            f"材料布局需要 {len(surfaces) // 2} 个名称，实际得到 {len(material_names)} 个。"
        )
    primary_wavelength = torch.tensor(
        3.5, dtype=lens.dtype, device=lens.device
    )
    for element_index, material_name in enumerate(material_names):
        if not isinstance(material_name, str) or not material_name.strip():
            raise ValueError("材料名称必须是非空字符串。")
        front = surfaces[2 * element_index]
        rear = surfaces[2 * element_index + 1]
        old_index = front.mat2.ior(primary_wavelength)
        new_material = Material(material_name.strip().lower())
        new_index = new_material.ior(primary_wavelength)
        if abs(float(new_index) - 1.0) < 1e-8:
            raise ValueError(f"透镜材料不能是空气：{material_name}。")
        # 先按薄透镜光焦度近似缩放该片两面的曲率，避免离散换材后
        # 处方瞬间退化；随后连续优化器仍可重新分配全部曲率。
        power_scale = (old_index - 1.0) / (new_index - 1.0)
        front.c = front.c.detach().clone() * power_scale
        rear.c = rear.c.detach().clone() * power_scale
        front.mat2 = new_material
    lens.post_computation()


def _safe_optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.Tensor],
    *,
    rollback_factor: float = 0.5,
    pre_step_loss: float | None = None,
    post_step_loss_fn=None,
    max_relative_increase: float = 0.02,
    absolute_tolerance: float = 1e-7,
    diagnostics: dict[str, Any] | None = None,
) -> bool:
    """执行一次 Adam 更新，并拒绝明显变差的有限更新。

    仅检查 ``NaN/Inf`` 不足以保护光学优化：Adam 可能产生完全有限、但
    merit 从较小值跳到更大值的更新。调用方可以提供更新前的 merit 和一个
    无梯度的更新后评价函数；若更新后 merit 超过允许的相对增幅，则参数及
    Adam 动量一并恢复，并在同一梯度上最多用两个更小步长重试。无评价函数
    时保持旧的“只检查有限性”行为，便于外部轻量调用。返回值仍表示最终是否
    接受更新；可选的 ``diagnostics`` 字典会记录尝试次数和拒绝原因。
    """

    if not 0.0 < rollback_factor < 1.0:
        raise ValueError("rollback_factor 必须位于 (0, 1) 内。")
    if pre_step_loss is not None and (
        not math.isfinite(float(pre_step_loss)) or float(pre_step_loss) < 0.0
    ):
        raise ValueError("pre_step_loss 必须是非负有限数或 None。")
    if not math.isfinite(max_relative_increase) or max_relative_increase < 0.0:
        raise ValueError("max_relative_increase 必须是非负有限数。")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0.0:
        raise ValueError("absolute_tolerance 必须是非负有限数。")
    pre_step_loss_value = (
        None if pre_step_loss is None else float(pre_step_loss)
    )
    attempt_history: list[dict[str, Any]] = []
    if diagnostics is not None:
        diagnostics.update(
            {
                "accepted": False,
                "attempts": 0,
                "rejection_reason": None,
                "pre_step_loss": pre_step_loss_value,
                "post_step_loss": None,
                "attempt_history": attempt_history,
            }
        )
    gradients_finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in parameters
    )
    if not gradients_finite:
        for group in optimizer.param_groups:
            group["lr"] *= rollback_factor
        optimizer.zero_grad(set_to_none=True)
        if diagnostics is not None:
            diagnostics["rejection_reason"] = "nonfinite_gradient"
        return False

    snapshots = [parameter.detach().clone() for parameter in parameters]
    state_snapshots = {
        parameter: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state.get(parameter, {}).items()
        }
        for parameter in parameters
    }
    def restore_snapshot() -> None:
        """恢复参数及优化器状态，同时保持原梯度供下一次重试。"""

        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)
                state = optimizer.state[parameter]
                state.clear()
                state.update(
                    {
                        key: value.detach().clone() if torch.is_tensor(value) else value
                        for key, value in state_snapshots[parameter].items()
                    }
                )

    final_rejection_reason = None
    final_post_step_loss = None
    for attempt in range(1, 4):
        optimizer.step()
        parameters_finite = all(
            bool(torch.isfinite(parameter).all().item()) for parameter in parameters
        )
        state_finite = all(
            not torch.is_tensor(value)
            or bool(torch.isfinite(value).all().item())
            for parameter in parameters
            for value in optimizer.state.get(parameter, {}).values()
        )
        post_step_loss = None
        rejection_reason = None
        if not parameters_finite:
            rejection_reason = "nonfinite_parameters"
        elif not state_finite:
            rejection_reason = "nonfinite_optimizer_state"
        elif post_step_loss_fn is not None:
            try:
                with torch.no_grad():
                    post_step_loss = float(post_step_loss_fn())
                if not math.isfinite(post_step_loss):
                    rejection_reason = "nonfinite_post_loss"
                elif pre_step_loss_value is not None:
                    allowed = max(
                        pre_step_loss_value + absolute_tolerance,
                        pre_step_loss_value * (1.0 + max_relative_increase),
                    )
                    if post_step_loss > allowed:
                        rejection_reason = "merit_increase"
            except Exception:
                # 光线追迹在坏几何上可能抛出异常；这类更新同样必须回滚。
                rejection_reason = "post_step_evaluation_error"

        attempt_history.append(
            {
                "attempt": attempt,
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
                "post_step_loss": post_step_loss,
                "rejection_reason": rejection_reason,
            }
        )
        final_rejection_reason = rejection_reason
        final_post_step_loss = post_step_loss
        if diagnostics is not None:
            diagnostics["attempts"] = attempt
            diagnostics["post_step_loss"] = post_step_loss

        if rejection_reason is None:
            if diagnostics is not None:
                diagnostics["accepted"] = True
                diagnostics["rejection_reason"] = None
            return True

        restore_snapshot()
        for group in optimizer.param_groups:
            group["lr"] *= rollback_factor

    optimizer.zero_grad(set_to_none=True)
    if diagnostics is not None:
        diagnostics["rejection_reason"] = final_rejection_reason
        diagnostics["post_step_loss"] = final_post_step_loss
    return False


def _aspheric_surfaces(lens) -> list[Any]:
    """返回预留了偶次多项式自由度的非球面。"""

    return [
        surface
        for surface in _curved_surfaces(lens)
        if getattr(surface, "ai_degree", 0) > 0
    ]


def _initial_asphere_raw(lens, *, dtype: torch.dtype, device: torch.device):
    """把已有 k/A4 起的偶次项转为有界归一化变量。

    阶数由输入处方中最大的 ``ai_degree`` 决定，因此旧的四项处方和
    新的 A12/A14/A16 扩展处方可以共用同一优化器。
    """

    surfaces = _aspheric_surfaces(lens)
    if not surfaces:
        raise ValueError("输入处方没有预留可优化非球面。")
    coefficient_count = max(int(surface.ai_degree) for surface in surfaces)
    if coefficient_count > len(ASPHERE_ORDERS):
        raise ValueError(
            f"当前优化器最多支持 {len(ASPHERE_ORDERS)} 个偶次非球面系数，"
            f"输入处方需要 {coefficient_count} 个。"
        )
    edge_spans = torch.tensor(
        ASPHERE_EDGE_SPANS_MM[:coefficient_count], dtype=dtype, device=device
    )
    conic_values = []
    edge_values = []
    for surface in surfaces:
        if surface.ai_degree < 1:
            raise ValueError("非球面阶段要求每个预留面至少包含一个偶次项。")
        normalized_k = (_detached_float(surface.k) + 1.0) / 2.0
        normalized_k = min(max(normalized_k, -0.95), 0.95)
        conic_values.append(math.atanh(normalized_k))
        radius = max(float(surface.r), 1.0)
        normalized_edges = []
        for coefficient_index, order in enumerate(ASPHERE_ORDERS[:coefficient_count]):
            coefficient = _detached_float(getattr(surface, f"ai{order}", 0.0))
            edge_sag = coefficient * radius**order
            fraction = edge_sag / float(edge_spans[coefficient_index])
            normalized_edges.append(math.atanh(min(max(fraction, -0.95), 0.95)))
        edge_values.append(normalized_edges)
    return (
        torch.tensor(conic_values, dtype=dtype, device=device, requires_grad=True),
        torch.tensor(edge_values, dtype=dtype, device=device, requires_grad=True),
        edge_spans,
    )


def _assign_aspheres(
    lens,
    conic_raw: torch.Tensor,
    edge_raw: torch.Tensor,
    edge_spans_mm: torch.Tensor,
):
    """把有界圆锥常数和归一化边缘矢高写入预留非球面。"""

    surfaces = _aspheric_surfaces(lens)
    if conic_raw.shape != (len(surfaces),):
        raise ValueError("conic_raw 的数量与非球面数量不一致。")
    if edge_raw.ndim != 2 or edge_raw.shape[0] != len(surfaces):
        raise ValueError("edge_raw 必须为 [非球面数, 偶次项数]。")
    coefficient_count = edge_raw.shape[1]
    if coefficient_count < 1 or coefficient_count > len(ASPHERE_ORDERS):
        raise ValueError("edge_raw 的偶次项数超出当前优化器支持范围。")
    conics = -1.0 + 2.0 * torch.tanh(conic_raw)
    edge_sags = edge_spans_mm.unsqueeze(0) * torch.tanh(edge_raw)
    coefficients = []
    for surface_index, surface in enumerate(surfaces):
        radius = torch.as_tensor(
            max(float(surface.r), 1.0),
            dtype=edge_raw.dtype,
            device=edge_raw.device,
        )
        surface.k = conics[surface_index]
        surface_coefficients = []
        for coefficient_index, order in enumerate(ASPHERE_ORDERS[:coefficient_count]):
            coefficient = edge_sags[surface_index, coefficient_index] / radius**order
            setattr(surface, f"ai{order}", coefficient)
            surface_coefficients.append(coefficient)
        coefficients.append(torch.stack(surface_coefficients))
    return conics, edge_sags, torch.stack(coefficients)


def _ray_merit(
    lens,
    ray_batches,
    chief_batches,
    target_xy: torch.Tensor,
    *,
    minimum_valid_ratio: float,
    mtf_frequency_cy_mm: float | None = None,
    mtf_target: float = 0.55,
    mtf_surrogate_weight: float = 0.0,
    mtf_max_weight: float = 1.0,
    direct_mtf_weight: float = 0.0,
    direct_mtf_max_weight: float = 1.0,
    focus_weight: float = 0.0,
    astigmatism_weight: float = 0.0,
    chromatic_focus_weight: float = 0.0,
    field_curvature_weight: float = 0.0,
    rms_target_mm: float | None = None,
    rms_target_weight: float = 0.0,
    spot_reference: str = "centroid",
) -> tuple[torch.Tensor, dict[str, Any]]:
    """返回多波长、多视场光斑 merit。

    默认以每个波长/视场的光斑质心为参考计算 RMS，避免把像高映射误差
    重复算入像差 merit；畸变和像高仍由独立的 chief-ray mapping 项约束。
    传入 ``spot_reference="target"`` 可复现旧的理想像点 RMS 定义。
    """

    if spot_reference not in {"centroid", "target"}:
        raise ValueError("spot_reference 必须是 'centroid' 或 'target'。")

    field_weights = torch.linspace(
        1.0,
        1.5,
        target_xy.shape[0],
        dtype=target_xy.dtype,
        device=target_xy.device,
    )
    if mtf_surrogate_weight < 0.0 or not math.isfinite(mtf_surrogate_weight):
        raise ValueError("mtf_surrogate_weight 必须为非负有限值。")
    if direct_mtf_weight < 0.0 or not math.isfinite(direct_mtf_weight):
        raise ValueError("direct_mtf_weight 必须为非负有限值。")
    if direct_mtf_max_weight < 0.0 or not math.isfinite(direct_mtf_max_weight):
        raise ValueError("direct_mtf_max_weight 必须为非负有限值。")
    for name, weight in (
        ("focus_weight", focus_weight),
        ("astigmatism_weight", astigmatism_weight),
        ("chromatic_focus_weight", chromatic_focus_weight),
        ("field_curvature_weight", field_curvature_weight),
    ):
        if weight < 0.0 or not math.isfinite(weight):
            raise ValueError(f"{name} 必须为非负有限值。")
    if rms_target_weight < 0.0 or not math.isfinite(rms_target_weight):
        raise ValueError("rms_target_weight 必须为非负有限值。")
    if rms_target_weight > 0.0:
        if rms_target_mm is None or not math.isfinite(rms_target_mm) or rms_target_mm <= 0.0:
            raise ValueError("启用 RMS 目标惩罚时必须提供正的有限 rms_target_mm。")
    if mtf_max_weight < 0.0 or not math.isfinite(mtf_max_weight):
        raise ValueError("mtf_max_weight 必须为非负有限值。")
    if mtf_surrogate_weight > 0.0 or direct_mtf_weight > 0.0:
        if mtf_frequency_cy_mm is None or not math.isfinite(mtf_frequency_cy_mm):
            raise ValueError("启用 MTF merit 时必须提供正的有限空间频率。")
        if mtf_frequency_cy_mm <= 0.0:
            raise ValueError("MTF merit 空间频率必须大于 0。")
        if not math.isfinite(mtf_target) or not 0.0 < mtf_target < 1.0:
            raise ValueError("mtf_target 必须位于 (0, 1) 内。")

    all_rms = []
    all_rms_to_target = []
    valid_ratios = []
    mapping_errors = []
    mtf_violations = []
    direct_mtf_violations = []
    focus_samples = []
    last_surface_z = lens.surfaces[-1].d
    for sampled, sampled_chief in zip(ray_batches, chief_batches):
        traced = lens.trace2sensor(sampled.clone())
        valid = traced.is_valid.float().clamp(0.0, 1.0)
        valid_count_raw = valid.sum(dim=-1)
        valid_count = valid_count_raw.clamp_min(1.0)
        safe_xy = torch.where(
            valid.unsqueeze(-1) > 0.5,
            traced.o[..., :2],
            torch.zeros_like(traced.o[..., :2]),
        )
        spot_centroid = safe_xy.sum(dim=-2) / valid_count.unsqueeze(-1)
        centered_xy = torch.where(
            valid.unsqueeze(-1) > 0.5,
            safe_xy - spot_centroid.unsqueeze(-2),
            torch.zeros_like(safe_xy),
        )
        centered_mse = (
            centered_xy.square().sum(dim=-1) * valid
        ).sum(dim=-1) / valid_count
        rms_centered = torch.sqrt(centered_mse.clamp_min(1e-12))
        target_error = safe_xy - target_xy.unsqueeze(-2)
        target_mse = (
            target_error.square().sum(dim=-1) * valid
        ).sum(dim=-1) / valid_count
        rms_to_target = torch.sqrt(target_mse.clamp_min(1e-12))
        # 完全挡光的场点不能因零填充得到“完美”光斑；有效率项之外再给出
        # 一个明确的坏光斑代理，避免优化器钻入无效解。
        invalid_spot_rms = torch.full_like(rms_centered, 10.0)
        rms_centered = torch.where(
            valid_count_raw > 0.0, rms_centered, invalid_spot_rms
        )
        rms_to_target = torch.where(
            valid_count_raw > 0.0, rms_to_target, invalid_spot_rms
        )
        rms = rms_centered if spot_reference == "centroid" else rms_to_target
        all_rms.append(rms)
        all_rms_to_target.append(rms_to_target)
        valid_ratios.append(valid.mean(dim=-1))

        chief = lens.trace2sensor(sampled_chief.clone())
        chief_xy = chief.o[..., 0, :2]
        target_height = target_xy.norm(dim=-1)
        relative_mapping = (chief_xy - target_xy).norm(dim=-1)
        relative_mapping = relative_mapping / target_height.clamp_min(1e-9)
        relative_mapping = torch.where(
            target_height > 1e-9,
            relative_mapping,
            torch.zeros_like(relative_mapping),
        )
        mapping_errors.append(relative_mapping)

        if any(
            weight > 0.0
            for weight in (
                focus_weight,
                astigmatism_weight,
                chromatic_focus_weight,
                field_curvature_weight,
            )
        ):
            # 在最后一个折射面处用 q(z)=q0+u(z-z0) 的最小方差解求每个
            # 场点/方向的最佳焦面。该量不依赖传感器面，能显式分离色焦、
            # 场曲和像散，且对曲率、间隔和非球面保持可微。
            last_ray, _ = lens.trace(sampled.clone())
            last_valid = last_ray.is_valid.float().clamp(0.0, 1.0)
            last_count = last_valid.sum(dim=-1).clamp_min(2.0)
            # 光线沿 z 轴通常朝负方向传播；直接 ``clamp_min`` 会把所有
            # 负的 z 方向斜率替换成正的 1e-8，导致焦面/像散项得到完全错误
            # 的梯度。只在分母真正接近零时做带符号的保护。
            z_slope = last_ray.d[..., 2:3]
            z_safe = torch.where(
                z_slope.abs() >= 1e-8,
                z_slope,
                torch.where(z_slope >= 0.0, torch.full_like(z_slope, 1e-8),
                            torch.full_like(z_slope, -1e-8)),
            )
            slope = last_ray.d[..., :2] / z_safe
            # 不同光线与最后曲面的交点 z 可能相差数毫米；先把它们
            # 传播回同一个参考顶点 z，才能用协方差公式求共同最佳焦面。
            q0 = last_ray.o[..., :2] + slope * (
                last_surface_z - last_ray.o[..., 2:3]
            )
            safe_q0 = torch.where(
                last_valid.unsqueeze(-1) > 0.5,
                q0,
                torch.zeros_like(q0),
            )
            safe_slope = torch.where(
                last_valid.unsqueeze(-1) > 0.5,
                slope,
                torch.zeros_like(slope),
            )
            mean_q0 = safe_q0.sum(dim=-2) / last_count.unsqueeze(-1)
            mean_slope = safe_slope.sum(dim=-2) / last_count.unsqueeze(-1)
            centered_q0 = torch.where(
                last_valid.unsqueeze(-1) > 0.5,
                safe_q0 - mean_q0.unsqueeze(-2),
                torch.zeros_like(safe_q0),
            )
            centered_slope = torch.where(
                last_valid.unsqueeze(-1) > 0.5,
                safe_slope - mean_slope.unsqueeze(-2),
                torch.zeros_like(safe_slope),
            )
            covariance = (centered_q0 * centered_slope).sum(dim=-2)
            slope_variance = centered_slope.square().sum(dim=-2).clamp_min(1e-10)
            best_focus_shift = -covariance / slope_variance
            focus_samples.append(best_focus_shift)

        if mtf_surrogate_weight > 0.0:
            centroid = safe_xy.sum(dim=-2) / valid_count.unsqueeze(-1)
            centered = torch.where(
                valid.unsqueeze(-1) > 0.5,
                safe_xy - centroid.unsqueeze(-2),
                torch.zeros_like(safe_xy),
            )
            # 几何 MTF 代理按切向/弧矢两个独立方向计算。旧实现把
            # sigma_x^2+sigma_y^2 当成单轴方差，阈值会额外严格 sqrt(2)
            # 倍，而且会把两个方向混成一个不可解释的标量。
            variance_xy = centered.square().sum(dim=-2) / valid_count.unsqueeze(-1)
            sigma_axis = torch.sqrt(variance_xy.clamp_min(1e-12))
            phase_sigma = 2.0 * math.pi * mtf_frequency_cy_mm * sigma_axis
            target_phase_sigma = math.sqrt(-2.0 * math.log(mtf_target))
            relative_excess = torch.relu(
                phase_sigma / target_phase_sigma - 1.0
            )
            # 每个场点取两个方向的平均超差；外层再单独保留最坏场点约束。
            mtf_violations.append(torch.log1p(relative_excess).mean(dim=-1))

        if direct_mtf_weight > 0.0:
            # 用四个归一化频率同时约束低频和奈奎斯特端，避免单频 OTF
            # 在有限采样下通过相位折叠获得虚假的高分。目标按高斯光斑
            # 的频率平方关系递减，低频项提供更平滑的下降方向。
            direct_fractions = (0.25, 0.50, 0.75, 1.0)
            direct_terms = []
            valid_count_direct = valid.sum(dim=-1).clamp_min(1.0)
            centroid_direct = safe_xy.sum(dim=-2) / valid_count_direct.unsqueeze(-1)
            centered_direct = torch.where(
                valid.unsqueeze(-1) > 0.5,
                safe_xy - centroid_direct.unsqueeze(-2),
                torch.zeros_like(safe_xy),
            )
            for fraction in direct_fractions:
                frequency = mtf_frequency_cy_mm * fraction
                phase = -2.0 * math.pi * frequency * centered_direct
                denominator = valid_count_direct.unsqueeze(-1)
                real = (
                    torch.cos(phase) * valid.unsqueeze(-1)
                ).sum(dim=-2) / denominator
                imag = (
                    torch.sin(phase) * valid.unsqueeze(-1)
                ).sum(dim=-2) / denominator
                mtf_xy = torch.sqrt(
                    (real.square() + imag.square()).clamp_min(1e-12)
                )
                # 取两个方向中较差者，避免把
                # sqrt(MTF_x^2 + MTF_y^2) 错当成一个方向的 MTF。
                mtf_worst_axis = mtf_xy.min(dim=-1).values
                target_fraction = math.exp(
                    math.log(mtf_target) * fraction * fraction
                )
                direct_terms.append(
                    torch.log1p(
                        torch.relu(target_fraction - mtf_worst_axis)
                        / max(target_fraction, 1e-6)
                    )
                )
            direct_mtf_violations.append(torch.stack(direct_terms, dim=0))

    rms_grid = torch.stack(all_rms, dim=0)
    rms_to_target_grid = torch.stack(all_rms_to_target, dim=0)
    valid_grid = torch.stack(valid_ratios, dim=0)
    mapping_grid = torch.stack(mapping_errors, dim=0)
    weighted_mean = (rms_grid * field_weights.unsqueeze(0)).sum()
    weighted_mean = weighted_mean / (
        field_weights.sum() * rms_grid.shape[0]
    )
    spot_merit = weighted_mean + 0.35 * rms_grid.max()
    mapping_violation = torch.relu(mapping_grid - 0.005)
    mapping_merit = mapping_violation.mean() + mapping_violation.max()
    validity_violation = torch.relu(minimum_valid_ratio - valid_grid)
    validity_merit = validity_violation.mean() + validity_violation.max()
    mtf_loss = torch.zeros((), dtype=rms_grid.dtype, device=rms_grid.device)
    mtf_violation_max = torch.zeros_like(mtf_loss)
    if mtf_violations:
        mtf_grid = torch.stack(mtf_violations, dim=0)
        mtf_loss = mtf_grid.mean() + mtf_max_weight * mtf_grid.max()
        mtf_violation_max = mtf_grid.max()
    direct_mtf_loss = torch.zeros_like(mtf_loss)
    direct_mtf_violation_max = torch.zeros_like(mtf_loss)
    direct_mtf_min = torch.ones_like(mtf_loss)
    if direct_mtf_violations:
        direct_grid = torch.stack(direct_mtf_violations, dim=0)
        direct_mtf_loss = (
            direct_grid.mean() + direct_mtf_max_weight * direct_grid.max()
        )
        direct_mtf_violation_max = direct_grid.max()
        # 该诊断只用于监视优化方向，验收仍以高采样独立指标为准。
        direct_mtf_min = torch.exp(-direct_grid[..., -1, :]).min()

    focus_loss = torch.zeros_like(mtf_loss)
    astigmatism_loss = torch.zeros_like(mtf_loss)
    chromatic_focus_loss = torch.zeros_like(mtf_loss)
    field_curvature_loss = torch.zeros_like(mtf_loss)
    focus_residual_max = torch.zeros_like(mtf_loss)
    rms_target_loss = torch.zeros_like(mtf_loss)
    if focus_samples:
        focus_grid = torch.stack(focus_samples, dim=0)
        sensor_offset = lens.d_sensor.to(
            dtype=target_xy.dtype, device=target_xy.device
        ) - last_surface_z
        focus_residual = focus_grid - sensor_offset
        focus_loss = torch.sqrt(focus_residual.square().mean() + 1e-12)
        astigmatism_loss = torch.sqrt(
            (focus_grid[..., 0] - focus_grid[..., 1]).square().mean() + 1e-12
        )
        chromatic_focus_loss = torch.sqrt(
            (focus_grid - focus_grid.mean(dim=0, keepdim=True))
            .square()
            .mean()
            + 1e-12
        )
        field_mean_focus = focus_grid.mean(dim=(0, 2))
        field_curvature_loss = torch.sqrt(
            (field_mean_focus - field_mean_focus.mean()).square().mean() + 1e-12
        )
        focus_residual_max = focus_residual.abs().max()
    if rms_target_weight > 0.0:
        target_rms = torch.as_tensor(
            float(rms_target_mm), dtype=rms_grid.dtype, device=rms_grid.device
        )
        normalized_excess = torch.relu(rms_grid / target_rms - 1.0)
        rms_target_loss = normalized_excess.mean() + normalized_excess.max()
    total = (
        spot_merit
        + 5.0 * mapping_merit
        + 50.0 * validity_merit
        + mtf_surrogate_weight * mtf_loss
        + direct_mtf_weight * direct_mtf_loss
        + focus_weight * focus_loss
        + astigmatism_weight * astigmatism_loss
        + chromatic_focus_weight * chromatic_focus_loss
        + field_curvature_weight * field_curvature_loss
        + rms_target_weight * rms_target_loss
    )
    diagnostics = {
        "rms_mean_mm": float(rms_grid.mean().detach().cpu()),
        "rms_max_mm": float(rms_grid.max().detach().cpu()),
        "rms_to_target_mean_mm": float(rms_to_target_grid.mean().detach().cpu()),
        "rms_to_target_max_mm": float(rms_to_target_grid.max().detach().cpu()),
        "spot_reference": spot_reference,
        "mapping_max_relative": float(mapping_grid.max().detach().cpu()),
        "valid_ratio_min": float(valid_grid.min().detach().cpu()),
        "mtf_surrogate_loss": float(mtf_loss.detach().cpu()),
        "mtf_surrogate_violation_max": float(mtf_violation_max.detach().cpu()),
        "direct_mtf_loss": float(direct_mtf_loss.detach().cpu()),
        "direct_mtf_violation_max": float(
            direct_mtf_violation_max.detach().cpu()
        ),
        "direct_mtf_min_proxy": float(direct_mtf_min.detach().cpu()),
        "focus_loss_mm": float(focus_loss.detach().cpu()),
        "astigmatism_loss_mm": float(astigmatism_loss.detach().cpu()),
        "chromatic_focus_loss_mm": float(chromatic_focus_loss.detach().cpu()),
        "field_curvature_loss_mm": float(field_curvature_loss.detach().cpu()),
        "focus_residual_max_mm": float(focus_residual_max.detach().cpu()),
        "rms_target_loss": float(rms_target_loss.detach().cpu()),
    }
    return total, diagnostics


def _evaluate_parameter_state(
    lens,
    spec: MWIRDesignSpec,
    base_curvatures: torch.Tensor,
    shape_raw: torch.Tensor,
    focus_raw: torch.Tensor,
    ray_batches,
    chief_batches,
    target_xy: torch.Tensor,
    *,
    focus_span_mm: float,
    minimum_valid_ratio: float,
):
    curvatures, first_order, relative = constrained_curvatures(
        lens,
        base_curvatures,
        shape_raw,
        spec.effective_focal_length_mm,
    )
    focus_shift = focus_span_mm * torch.tanh(focus_raw)
    sensor_z = first_order.focus_z_mm + focus_shift
    _assign_geometry(lens, curvatures, sensor_z)
    ray_loss, diagnostics = _ray_merit(
        lens,
        ray_batches,
        chief_batches,
        target_xy,
        minimum_valid_ratio=minimum_valid_ratio,
    )
    clearance_loss, clearance = _clearance_penalty(lens, curvatures)
    # 共同倍率已经把 EFL 约束到目标；保留一个很小的数值残差项用于诊断。
    efl_relative_error = (
        first_order.effective_focal_length_mm / spec.effective_focal_length_mm - 1.0
    ).abs()
    total = ray_loss + 2.0 * clearance_loss + 10.0 * efl_relative_error
    diagnostics.update(clearance)
    diagnostics.update(
        {
            "loss": float(total.detach().cpu()),
            "ray_loss": float(ray_loss.detach().cpu()),
            "clearance_loss": float(clearance_loss.detach().cpu()),
            "effective_focal_length_mm": float(
                first_order.effective_focal_length_mm.detach().cpu()
            ),
            "effective_focal_length_relative_error": float(
                efl_relative_error.detach().cpu()
            ),
            "focus_shift_mm": float(focus_shift.detach().cpu()),
            "sensor_z_mm": float(sensor_z.detach().cpu()),
            "relative_curvature_min": float(relative.min().detach().cpu()),
            "relative_curvature_max": float(relative.max().detach().cpu()),
        }
    )
    return total, diagnostics, curvatures, sensor_z


def _evaluate_aspheric_parameter_state(
    lens,
    spec: MWIRDesignSpec,
    base_curvatures: torch.Tensor,
    shape_raw: torch.Tensor,
    focus_raw: torch.Tensor,
    conic_raw: torch.Tensor,
    edge_raw: torch.Tensor,
    edge_spans_mm: torch.Tensor,
    ray_batches,
    chief_batches,
    target_xy: torch.Tensor,
    *,
    focus_span_mm: float,
    minimum_valid_ratio: float,
    mtf_frequency_cy_mm: float | None = None,
    mtf_target: float = 0.55,
    mtf_surrogate_weight: float = 0.0,
    mtf_max_weight: float = 1.0,
    direct_mtf_weight: float = 0.0,
    direct_mtf_max_weight: float = 1.0,
    focus_weight: float = 0.0,
    astigmatism_weight: float = 0.0,
    chromatic_focus_weight: float = 0.0,
    field_curvature_weight: float = 0.0,
    relative_curvature_ratio: float = 1.25,
    rms_target_mm: float | None = None,
    rms_target_weight: float = 0.0,
):
    """评价曲率、焦点、圆锥常数及 A4–A10 的联合状态。"""

    if relative_curvature_ratio <= 1.0 or not math.isfinite(relative_curvature_ratio):
        raise ValueError("relative_curvature_ratio 必须是大于 1 的有限值。")
    curvatures, first_order, relative = constrained_curvatures(
        lens,
        base_curvatures,
        shape_raw,
        spec.effective_focal_length_mm,
        relative_log_span=math.log(relative_curvature_ratio),
    )
    focus_shift = focus_span_mm * torch.tanh(focus_raw)
    sensor_z = first_order.focus_z_mm + focus_shift
    _assign_geometry(lens, curvatures, sensor_z)
    conics, edge_sags, coefficients = _assign_aspheres(
        lens, conic_raw, edge_raw, edge_spans_mm
    )
    ray_loss, diagnostics = _ray_merit(
        lens,
        ray_batches,
        chief_batches,
        target_xy,
        minimum_valid_ratio=minimum_valid_ratio,
        mtf_frequency_cy_mm=mtf_frequency_cy_mm,
        mtf_target=mtf_target,
        mtf_surrogate_weight=mtf_surrogate_weight,
        mtf_max_weight=mtf_max_weight,
        direct_mtf_weight=direct_mtf_weight,
        direct_mtf_max_weight=direct_mtf_max_weight,
        focus_weight=focus_weight,
        astigmatism_weight=astigmatism_weight,
        chromatic_focus_weight=chromatic_focus_weight,
        field_curvature_weight=field_curvature_weight,
        rms_target_mm=rms_target_mm,
        rms_target_weight=rms_target_weight,
    )
    clearance_loss, clearance = _clearance_penalty(lens, curvatures)
    efl_relative_error = (
        first_order.effective_focal_length_mm / spec.effective_focal_length_mm - 1.0
    ).abs()
    # 很弱的矢高正则只用于在两个同效解之间偏向较温和的非球面，不妨碍
    # 毫米量级边缘修正真正参与像差校正。
    asphere_regularization = 0.002 * torch.sqrt(
        edge_sags.square().mean() + 1e-12
    )
    total = (
        ray_loss
        + 2.0 * clearance_loss
        + 10.0 * efl_relative_error
        + asphere_regularization
    )
    diagnostics.update(clearance)
    diagnostics.update(
        {
            "loss": float(total.detach().cpu()),
            "ray_loss": float(ray_loss.detach().cpu()),
            "clearance_loss": float(clearance_loss.detach().cpu()),
            "asphere_regularization": float(asphere_regularization.detach().cpu()),
            "effective_focal_length_mm": float(
                first_order.effective_focal_length_mm.detach().cpu()
            ),
            "effective_focal_length_relative_error": float(
                efl_relative_error.detach().cpu()
            ),
            "focus_shift_mm": float(focus_shift.detach().cpu()),
            "sensor_z_mm": float(sensor_z.detach().cpu()),
            "relative_curvature_min": float(relative.min().detach().cpu()),
            "relative_curvature_max": float(relative.max().detach().cpu()),
            "conic_min": float(conics.min().detach().cpu()),
            "conic_max": float(conics.max().detach().cpu()),
            "maximum_abs_asphere_edge_contribution_mm": float(
                edge_sags.abs().max().detach().cpu()
            ),
        }
    )
    return (
        total,
        diagnostics,
        curvatures,
        sensor_z,
        conics,
        edge_sags,
        coefficients,
    )


def _evaluate_structural_parameter_state(
    lens,
    spec: MWIRDesignSpec,
    base_curvatures: torch.Tensor,
    first_surface_z: torch.Tensor,
    base_gaps: torch.Tensor,
    shape_raw: torch.Tensor,
    gap_raw: torch.Tensor,
    focus_raw: torch.Tensor,
    conic_raw: torch.Tensor,
    edge_raw: torch.Tensor,
    edge_spans_mm: torch.Tensor,
    ray_batches,
    chief_batches,
    target_xy: torch.Tensor,
    *,
    focus_span_mm: float,
    minimum_valid_ratio: float,
    glass_gap_ratio: float,
    air_gap_ratio: float,
    mtf_frequency_cy_mm: float | None = None,
    mtf_target: float = 0.55,
    mtf_surrogate_weight: float = 0.0,
    mtf_max_weight: float = 1.0,
    direct_mtf_weight: float = 0.0,
    direct_mtf_max_weight: float = 1.0,
    focus_weight: float = 0.0,
    astigmatism_weight: float = 0.0,
    chromatic_focus_weight: float = 0.0,
    field_curvature_weight: float = 0.0,
    relative_curvature_ratio: float = 1.25,
    rms_target_mm: float | None = None,
    rms_target_weight: float = 0.0,
):
    """评价曲率、结构间隔、焦点和低阶非球面的联合状态。"""

    positions, gaps, gap_relative = constrained_surface_positions(
        first_surface_z,
        base_gaps,
        gap_raw,
        glass_gap_ratio=glass_gap_ratio,
        air_gap_ratio=air_gap_ratio,
    )
    _assign_surface_positions(lens, positions)
    result = _evaluate_aspheric_parameter_state(
        lens,
        spec,
        base_curvatures,
        shape_raw,
        focus_raw,
        conic_raw,
        edge_raw,
        edge_spans_mm,
        ray_batches,
        chief_batches,
        target_xy,
        focus_span_mm=focus_span_mm,
        minimum_valid_ratio=minimum_valid_ratio,
        mtf_frequency_cy_mm=mtf_frequency_cy_mm,
        mtf_target=mtf_target,
        mtf_surrogate_weight=mtf_surrogate_weight,
        mtf_max_weight=mtf_max_weight,
        direct_mtf_weight=direct_mtf_weight,
        direct_mtf_max_weight=direct_mtf_max_weight,
        focus_weight=focus_weight,
        astigmatism_weight=astigmatism_weight,
        chromatic_focus_weight=chromatic_focus_weight,
        field_curvature_weight=field_curvature_weight,
        relative_curvature_ratio=relative_curvature_ratio,
        rms_target_mm=rms_target_mm,
        rms_target_weight=rms_target_weight,
    )
    base_total, diagnostics = result[0], result[1]
    # 极弱正则只在像质近似相同时偏向输入结构，不会阻止空气组间距重排。
    spacing_regularization = 0.001 * torch.sqrt(
        torch.log(gap_relative).square().mean() + 1e-12
    )
    total = base_total + spacing_regularization
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "loss": float(total.detach().cpu()),
            "spacing_regularization": float(
                spacing_regularization.detach().cpu()
            ),
            "minimum_center_gap_mm": float(gaps.min().detach().cpu()),
            "surface_span_mm": float((positions[-1] - positions[0]).detach().cpu()),
            "gap_relative_min": float(gap_relative.min().detach().cpu()),
            "gap_relative_max": float(gap_relative.max().detach().cpu()),
        }
    )
    return (
        total,
        diagnostics,
        result[2],
        result[3],
        result[4],
        result[5],
        result[6],
        positions,
        gaps,
        gap_relative,
    )


def optimize_spherical_seed(
    lens,
    spec: MWIRDesignSpec,
    *,
    iterations: int,
    field_count: int,
    spp: int,
    validation_spp: int,
    learning_rate: float,
    focus_learning_rate: float,
    focus_span_mm: float,
    minimum_valid_ratio: float,
    ray_seed: int,
    checkpoint_interval: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """优化球面曲率形状和最佳焦点偏移，并返回固定验证集最佳状态。"""

    if iterations <= 0:
        raise ValueError("iterations 必须为正整数。")
    surfaces = _curved_surfaces(lens)
    base_curvatures = torch.stack(
        [surface.c.detach().clone().to(lens.device) for surface in surfaces]
    )
    base_state = paraxial_state(lens, base_curvatures)
    initial_focus_shift = _detached_float(lens.d_sensor) - float(
        base_state.focus_z_mm.detach().cpu()
    )
    focus_fraction = min(max(initial_focus_shift / focus_span_mm, -0.95), 0.95)
    shape_raw = torch.zeros_like(base_curvatures, requires_grad=True)
    focus_raw = torch.tensor(
        math.atanh(focus_fraction),
        dtype=base_curvatures.dtype,
        device=base_curvatures.device,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [shape_raw], "lr": learning_rate},
            {"params": [focus_raw], "lr": focus_learning_rate},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=0.0
    )
    train_batches, train_chiefs, train_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=field_count,
        spp=spp,
        seed=ray_seed,
        pupil_scale=1.0,
    )
    validation_batches, validation_chiefs, validation_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=max(field_count, 7),
        spp=validation_spp,
        seed=ray_seed + 10_000,
        pupil_scale=1.0,
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        train_loss, train_diag, _, _ = _evaluate_parameter_state(
            lens,
            spec,
            base_curvatures,
            shape_raw,
            focus_raw,
            train_batches,
            train_chiefs,
            train_target,
            focus_span_mm=focus_span_mm,
            minimum_valid_ratio=minimum_valid_ratio,
        )
        is_checkpoint = (
            iteration == 0
            or iteration == iterations
            or iteration % checkpoint_interval == 0
        )
        if is_checkpoint:
            with torch.no_grad():
                _, validation_diag, validation_curvatures, validation_sensor = (
                    _evaluate_parameter_state(
                        lens,
                        spec,
                        base_curvatures,
                        shape_raw,
                        focus_raw,
                        validation_batches,
                        validation_chiefs,
                        validation_target,
                        focus_span_mm=focus_span_mm,
                        minimum_valid_ratio=minimum_valid_ratio,
                    )
                )
            row = {
                "iteration": iteration,
                "train": train_diag,
                "validation": validation_diag,
            }
            history.append(row)
            logging.info(
                "迭代 %d/%d：训练 merit %.6f；验证 RMS 均值/最大 %.6f/%.6f mm；"
                "映射最大 %.4f%%；EFL %.6f mm；最低有效率 %.3f。",
                iteration,
                iterations,
                train_diag["loss"],
                validation_diag["rms_mean_mm"],
                validation_diag["rms_max_mm"],
                100.0 * validation_diag["mapping_max_relative"],
                validation_diag["effective_focal_length_mm"],
                validation_diag["valid_ratio_min"],
            )
            if best is None or validation_diag["loss"] < best["diagnostics"]["loss"]:
                best = {
                    "iteration": iteration,
                    "diagnostics": dict(validation_diag),
                    "shape_raw": shape_raw.detach().clone(),
                    "focus_raw": focus_raw.detach().clone(),
                    "curvatures": validation_curvatures.detach().clone(),
                    "sensor_z": validation_sensor.detach().clone(),
                }

        if iteration == iterations:
            break
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"第 {iteration} 步产生非有限损失。")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_([shape_raw, focus_raw], max_norm=10.0)
        def post_step_merit():
            result = _evaluate_parameter_state(
                lens,
                spec,
                base_curvatures,
                shape_raw,
                focus_raw,
                train_batches,
                train_chiefs,
                train_target,
                focus_span_mm=focus_span_mm,
                minimum_valid_ratio=minimum_valid_ratio,
            )
            return result[0]

        step_diagnostics: dict[str, Any] = {}
        step_ok = _safe_optimizer_step(
            optimizer,
            [shape_raw, focus_raw],
            pre_step_loss=float(train_loss.detach().cpu()),
            post_step_loss_fn=post_step_merit,
            diagnostics=step_diagnostics,
        )
        if not step_ok:
            for group_index, group in enumerate(optimizer.param_groups):
                scheduler.base_lrs[group_index] = group["lr"]
            if step_diagnostics["rejection_reason"] == "merit_increase":
                logging.warning(
                    "第 %d 步 merit 从 %.6f 增至 %.6f，连续 %d 次尝试仍变差，"
                    "已回滚并降低学习率。",
                    iteration,
                    step_diagnostics["pre_step_loss"],
                    step_diagnostics["post_step_loss"],
                    step_diagnostics["attempts"],
                )
            else:
                logging.warning(
                    "第 %d 步检测到 NaN/Inf 或后验评估异常（%s，尝试 %d 次），"
                    "已回滚并降低学习率。",
                    iteration,
                    step_diagnostics["rejection_reason"],
                    step_diagnostics["attempts"],
                )
        else:
            scheduler.step()

    if best is None:
        raise RuntimeError("优化没有生成可用检查点。")
    _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    # 切断训练图，使后续 post_computation/写 JSON 使用普通叶张量。
    for surface in surfaces:
        surface.c = surface.c.detach().clone()
    lens.d_sensor = lens.d_sensor.detach().clone()
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)
    serializable_best = {
        "iteration": best["iteration"],
        "diagnostics": best["diagnostics"],
        "curvatures_1_per_mm": [
            float(value) for value in best["curvatures"].detach().cpu().tolist()
        ],
        "radii_mm": [
            0.0 if abs(float(value)) < 1e-15 else 1.0 / float(value)
            for value in best["curvatures"].detach().cpu().tolist()
        ],
        "sensor_z_mm": float(best["sensor_z"].detach().cpu()),
        "shape_raw": [
            float(value) for value in best["shape_raw"].detach().cpu().tolist()
        ],
        "focus_raw": float(best["focus_raw"].detach().cpu()),
    }
    return serializable_best, history


def optimize_aspheric_seed(
    lens,
    spec: MWIRDesignSpec,
    *,
    iterations: int,
    field_count: int,
    spp: int,
    validation_spp: int,
    learning_rate: float,
    focus_learning_rate: float,
    conic_learning_rate: float,
    asphere_learning_rate: float,
    focus_span_mm: float,
    minimum_valid_ratio: float,
    ray_seed: int,
    checkpoint_interval: int,
    relative_curvature_ratio: float = 1.25,
    rms_target_mm: float | None = None,
    rms_target_weight: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """联合优化曲率、焦点和预留非球面的 k/偶次项。"""

    if iterations <= 0:
        raise ValueError("iterations 必须为正整数。")
    surfaces = _curved_surfaces(lens)
    aspheres = _aspheric_surfaces(lens)
    if len(surfaces) != 14 or not aspheres:
        raise ValueError("非球面阶段要求七片、14 个曲面和至少一个预留非球面。")
    base_curvatures = torch.stack(
        [surface.c.detach().clone().to(lens.device) for surface in surfaces]
    )
    base_state = paraxial_state(lens, base_curvatures)
    initial_focus_shift = _detached_float(lens.d_sensor) - float(
        base_state.focus_z_mm.detach().cpu()
    )
    focus_fraction = min(max(initial_focus_shift / focus_span_mm, -0.95), 0.95)
    shape_raw = torch.zeros_like(base_curvatures, requires_grad=True)
    focus_raw = torch.tensor(
        math.atanh(focus_fraction),
        dtype=base_curvatures.dtype,
        device=base_curvatures.device,
        requires_grad=True,
    )
    conic_raw, edge_raw, edge_spans = _initial_asphere_raw(
        lens, dtype=base_curvatures.dtype, device=base_curvatures.device
    )
    optimizer = torch.optim.Adam(
        [
            {"params": [shape_raw], "lr": learning_rate},
            {"params": [focus_raw], "lr": focus_learning_rate},
            {"params": [conic_raw], "lr": conic_learning_rate},
            {"params": [edge_raw], "lr": asphere_learning_rate},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=0.0
    )
    train_batches, train_chiefs, train_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=field_count,
        spp=spp,
        seed=ray_seed,
        pupil_scale=1.0,
    )
    validation_batches, validation_chiefs, validation_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=max(field_count, 7),
        spp=validation_spp,
        seed=ray_seed + 10_000,
        pupil_scale=1.0,
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        train_result = _evaluate_aspheric_parameter_state(
            lens,
            spec,
            base_curvatures,
            shape_raw,
            focus_raw,
            conic_raw,
            edge_raw,
            edge_spans,
            train_batches,
            train_chiefs,
            train_target,
            focus_span_mm=focus_span_mm,
            minimum_valid_ratio=minimum_valid_ratio,
            relative_curvature_ratio=relative_curvature_ratio,
            rms_target_mm=rms_target_mm,
            rms_target_weight=rms_target_weight,
        )
        train_loss, train_diag = train_result[0], train_result[1]
        is_checkpoint = (
            iteration == 0
            or iteration == iterations
            or iteration % checkpoint_interval == 0
        )
        if is_checkpoint:
            with torch.no_grad():
                validation_result = _evaluate_aspheric_parameter_state(
                    lens,
                    spec,
                    base_curvatures,
                    shape_raw,
                    focus_raw,
                    conic_raw,
                    edge_raw,
                    edge_spans,
                    validation_batches,
                    validation_chiefs,
                    validation_target,
                    focus_span_mm=focus_span_mm,
                    minimum_valid_ratio=minimum_valid_ratio,
                    relative_curvature_ratio=relative_curvature_ratio,
                    rms_target_mm=rms_target_mm,
                    rms_target_weight=rms_target_weight,
                )
            validation_diag = validation_result[1]
            row = {
                "iteration": iteration,
                "train": train_diag,
                "validation": validation_diag,
            }
            history.append(row)
            logging.info(
                "非球面迭代 %d/%d：验证 RMS 均值/最大 %.6f/%.6f mm；"
                "映射最大 %.4f%%；EFL %.6f mm；k范围 %.3f..%.3f；"
                "最大边缘非球面贡献 %.3f mm；最低有效率 %.3f。",
                iteration,
                iterations,
                validation_diag["rms_mean_mm"],
                validation_diag["rms_max_mm"],
                100.0 * validation_diag["mapping_max_relative"],
                validation_diag["effective_focal_length_mm"],
                validation_diag["conic_min"],
                validation_diag["conic_max"],
                validation_diag["maximum_abs_asphere_edge_contribution_mm"],
                validation_diag["valid_ratio_min"],
            )
            if best is None or validation_diag["loss"] < best["diagnostics"]["loss"]:
                best = {
                    "iteration": iteration,
                    "diagnostics": dict(validation_diag),
                    "shape_raw": shape_raw.detach().clone(),
                    "focus_raw": focus_raw.detach().clone(),
                    "conic_raw": conic_raw.detach().clone(),
                    "edge_raw": edge_raw.detach().clone(),
                    "curvatures": validation_result[2].detach().clone(),
                    "sensor_z": validation_result[3].detach().clone(),
                    "conics": validation_result[4].detach().clone(),
                    "edge_sags": validation_result[5].detach().clone(),
                    "coefficients": validation_result[6].detach().clone(),
                }

        if iteration == iterations:
            break
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"非球面第 {iteration} 步产生非有限损失。")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [shape_raw, focus_raw, conic_raw, edge_raw], max_norm=10.0
        )
        def post_step_merit():
            result = _evaluate_aspheric_parameter_state(
                lens,
                spec,
                base_curvatures,
                shape_raw,
                focus_raw,
                conic_raw,
                edge_raw,
                edge_spans,
                train_batches,
                train_chiefs,
                train_target,
                focus_span_mm=focus_span_mm,
                minimum_valid_ratio=minimum_valid_ratio,
                relative_curvature_ratio=relative_curvature_ratio,
                rms_target_mm=rms_target_mm,
                rms_target_weight=rms_target_weight,
            )
            return result[0]

        step_diagnostics: dict[str, Any] = {}
        step_ok = _safe_optimizer_step(
            optimizer,
            [shape_raw, focus_raw, conic_raw, edge_raw],
            pre_step_loss=float(train_loss.detach().cpu()),
            post_step_loss_fn=post_step_merit,
            diagnostics=step_diagnostics,
        )
        if not step_ok:
            for group_index, group in enumerate(optimizer.param_groups):
                scheduler.base_lrs[group_index] = group["lr"]
            if step_diagnostics["rejection_reason"] == "merit_increase":
                logging.warning(
                    "非球面第 %d 步 merit 从 %.6f 增至 %.6f，连续 %d 次尝试仍变差，"
                    "已回滚并降低学习率。",
                    iteration,
                    step_diagnostics["pre_step_loss"],
                    step_diagnostics["post_step_loss"],
                    step_diagnostics["attempts"],
                )
            else:
                logging.warning(
                    "非球面第 %d 步检测到 NaN/Inf 或后验评估异常（%s，尝试 %d 次），"
                    "已回滚并降低学习率。",
                    iteration,
                    step_diagnostics["rejection_reason"],
                    step_diagnostics["attempts"],
                )
        else:
            scheduler.step()

    if best is None:
        raise RuntimeError("非球面优化没有生成可用检查点。")
    _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    _assign_aspheres(lens, best["conic_raw"], best["edge_raw"], edge_spans)
    for surface in surfaces:
        surface.c = surface.c.detach().clone()
    for surface in aspheres:
        surface.k = surface.k.detach().clone()
        coefficient_values = []
        coefficient_count = best["edge_raw"].shape[1]
        for order in ASPHERE_ORDERS[:coefficient_count]:
            coefficient = getattr(surface, f"ai{order}").detach().clone()
            setattr(surface, f"ai{order}", coefficient)
            coefficient_values.append(coefficient)
        surface.ai = torch.stack(coefficient_values).detach().clone()
    lens.d_sensor = lens.d_sensor.detach().clone()
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)

    serializable_best = {
        "iteration": best["iteration"],
        "diagnostics": best["diagnostics"],
        "curvatures_1_per_mm": best["curvatures"].detach().cpu().tolist(),
        "radii_mm": [
            0.0 if abs(float(value)) < 1e-15 else 1.0 / float(value)
            for value in best["curvatures"].detach().cpu().tolist()
        ],
        "sensor_z_mm": float(best["sensor_z"].detach().cpu()),
        "conic_constants": best["conics"].detach().cpu().tolist(),
        "edge_asphere_contributions_mm": best["edge_sags"].detach().cpu().tolist(),
        "asphere_coefficients": (
            best["coefficients"].detach().cpu().tolist()
        ),
        "shape_raw": best["shape_raw"].detach().cpu().tolist(),
        "focus_raw": float(best["focus_raw"].detach().cpu()),
        "conic_raw": best["conic_raw"].detach().cpu().tolist(),
        "edge_raw": best["edge_raw"].detach().cpu().tolist(),
    }
    return serializable_best, history


def optimize_structural_seed(
    lens,
    spec: MWIRDesignSpec,
    *,
    iterations: int,
    field_count: int,
    spp: int,
    validation_spp: int,
    learning_rate: float,
    gap_learning_rate: float,
    focus_learning_rate: float,
    conic_learning_rate: float,
    asphere_learning_rate: float,
    focus_span_mm: float,
    minimum_valid_ratio: float,
    glass_gap_ratio: float,
    air_gap_ratio: float,
    mtf_frequency_cy_mm: float | None,
    mtf_target: float,
    mtf_surrogate_weight: float,
    mtf_max_weight: float,
    direct_mtf_weight: float,
    direct_mtf_max_weight: float,
    focus_weight: float,
    astigmatism_weight: float,
    chromatic_focus_weight: float,
    field_curvature_weight: float,
    ray_seed: int,
    checkpoint_interval: int,
    relative_curvature_ratio: float = 1.25,
    rms_target_mm: float | None = None,
    rms_target_weight: float = 0.0,
    curriculum_warmup_fraction: float = 0.0,
    curriculum_ramp_fraction: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """联合优化十四面曲率、十三个结构间隔、焦点和五个非球面。"""

    if iterations <= 0:
        raise ValueError("iterations 必须为正整数。")
    surfaces = _curved_surfaces(lens)
    aspheres = _aspheric_surfaces(lens)
    if len(surfaces) < 2 or len(surfaces) % 2 or not aspheres:
        raise ValueError("结构阶段要求偶数个折射面以及至少一个非球面。")

    base_curvatures = torch.stack(
        [surface.c.detach().clone().to(lens.device) for surface in surfaces]
    )
    base_positions = torch.stack(
        [surface.d.detach().clone().to(lens.device) for surface in surfaces]
    )
    first_surface_z = base_positions[0].detach().clone()
    base_gaps = torch.diff(base_positions).detach().clone()
    if not bool(torch.all(base_gaps > 0.0).detach().cpu()):
        raise ValueError("结构阶段输入处方的折射面顶点顺序必须严格递增。")

    base_state = paraxial_state(lens, base_curvatures)
    initial_focus_shift = _detached_float(lens.d_sensor) - float(
        base_state.focus_z_mm.detach().cpu()
    )
    focus_fraction = min(max(initial_focus_shift / focus_span_mm, -0.95), 0.95)
    shape_raw = torch.zeros_like(base_curvatures, requires_grad=True)
    gap_raw = torch.zeros_like(base_gaps, requires_grad=True)
    focus_raw = torch.tensor(
        math.atanh(focus_fraction),
        dtype=base_curvatures.dtype,
        device=base_curvatures.device,
        requires_grad=True,
    )
    conic_raw, edge_raw, edge_spans = _initial_asphere_raw(
        lens, dtype=base_curvatures.dtype, device=base_curvatures.device
    )
    parameters = [shape_raw, gap_raw, focus_raw, conic_raw, edge_raw]
    optimizer = torch.optim.Adam(
        [
            {"params": [shape_raw], "lr": learning_rate},
            {"params": [gap_raw], "lr": gap_learning_rate},
            {"params": [focus_raw], "lr": focus_learning_rate},
            {"params": [conic_raw], "lr": conic_learning_rate},
            {"params": [edge_raw], "lr": asphere_learning_rate},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iterations, eta_min=0.0
    )
    train_batches, train_chiefs, train_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=field_count,
        spp=spp,
        seed=ray_seed,
        pupil_scale=1.0,
    )
    validation_batches, validation_chiefs, validation_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=max(field_count, 7),
        spp=validation_spp,
        seed=ray_seed + 10_000,
        pupil_scale=1.0,
    )

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    has_scheduled_weights = any(
        weight > 0.0
        for weight in (
            mtf_surrogate_weight,
            direct_mtf_weight,
            focus_weight,
            astigmatism_weight,
            chromatic_focus_weight,
            field_curvature_weight,
        )
    )
    for iteration in range(iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        curriculum_scale = _curriculum_scale(
            iteration,
            iterations,
            warmup_fraction=curriculum_warmup_fraction,
            ramp_fraction=curriculum_ramp_fraction,
        )
        current_ray_weights = _curriculum_ray_weights(
            curriculum_scale,
            mtf_surrogate_weight=mtf_surrogate_weight,
            direct_mtf_weight=direct_mtf_weight,
            focus_weight=focus_weight,
            astigmatism_weight=astigmatism_weight,
            chromatic_focus_weight=chromatic_focus_weight,
            field_curvature_weight=field_curvature_weight,
        )
        train_result = _evaluate_structural_parameter_state(
            lens,
            spec,
            base_curvatures,
            first_surface_z,
            base_gaps,
            shape_raw,
            gap_raw,
            focus_raw,
            conic_raw,
            edge_raw,
            edge_spans,
            train_batches,
            train_chiefs,
            train_target,
            focus_span_mm=focus_span_mm,
            minimum_valid_ratio=minimum_valid_ratio,
            glass_gap_ratio=glass_gap_ratio,
            air_gap_ratio=air_gap_ratio,
            mtf_frequency_cy_mm=mtf_frequency_cy_mm,
            mtf_target=mtf_target,
            mtf_max_weight=mtf_max_weight,
            direct_mtf_max_weight=direct_mtf_max_weight,
            relative_curvature_ratio=relative_curvature_ratio,
            rms_target_mm=rms_target_mm,
            rms_target_weight=rms_target_weight,
            **current_ray_weights,
        )
        train_loss, train_diag = train_result[0], dict(train_result[1])
        train_diag["curriculum_scale"] = curriculum_scale
        train_diag["active_auxiliary_weights"] = dict(current_ray_weights)
        is_checkpoint = (
            iteration == 0
            or iteration == iterations
            or iteration % checkpoint_interval == 0
        )
        if is_checkpoint:
            with torch.no_grad():
                validation_result = _evaluate_structural_parameter_state(
                    lens,
                    spec,
                    base_curvatures,
                    first_surface_z,
                    base_gaps,
                    shape_raw,
                    gap_raw,
                    focus_raw,
                    conic_raw,
                    edge_raw,
                    edge_spans,
                    validation_batches,
                    validation_chiefs,
                    validation_target,
                    focus_span_mm=focus_span_mm,
                    minimum_valid_ratio=minimum_valid_ratio,
                    glass_gap_ratio=glass_gap_ratio,
                    air_gap_ratio=air_gap_ratio,
                    mtf_frequency_cy_mm=mtf_frequency_cy_mm,
                    mtf_target=mtf_target,
                    mtf_max_weight=mtf_max_weight,
                    direct_mtf_max_weight=direct_mtf_max_weight,
                    relative_curvature_ratio=relative_curvature_ratio,
                    rms_target_mm=rms_target_mm,
                    rms_target_weight=rms_target_weight,
                    **current_ray_weights,
                )
            validation_diag = dict(validation_result[1])
            validation_diag["curriculum_scale"] = curriculum_scale
            validation_diag["active_auxiliary_weights"] = dict(
                current_ray_weights
            )
            history.append(
                {
                    "iteration": iteration,
                    "curriculum_scale": curriculum_scale,
                    "active_auxiliary_weights": dict(current_ray_weights),
                    "train": train_diag,
                    "validation": validation_diag,
                }
            )
            logging.info(
                "结构迭代 %d/%d：验证 RMS 均值/最大 %.6f/%.6f mm；"
                "映射最大 %.4f%%；EFL %.6f mm；面组跨度 %.3f mm；"
                "MTF代理/直接 %.4f/%.4f；焦面/像散 %.4f/%.4f mm；"
                "最小玻璃边厚/空气边隙 %.3f/%.3f mm；curriculum %.3f。",
                iteration,
                iterations,
                validation_diag["rms_mean_mm"],
                validation_diag["rms_max_mm"],
                100.0 * validation_diag["mapping_max_relative"],
                validation_diag["effective_focal_length_mm"],
                validation_diag["surface_span_mm"],
                validation_diag["mtf_surrogate_loss"],
                validation_diag["direct_mtf_loss"],
                validation_diag["focus_loss_mm"],
                validation_diag["astigmatism_loss_mm"],
                validation_diag["minimum_glass_edge_mm"],
                validation_diag["minimum_air_edge_mm"],
                curriculum_scale,
            )
            selection_eligible = (
                not has_scheduled_weights or curriculum_scale >= 1.0 - 1e-12
            )
            if selection_eligible and (
                best is None
                or validation_diag["loss"] < best["diagnostics"]["loss"]
            ):
                best = {
                    "iteration": iteration,
                    "diagnostics": dict(validation_diag),
                    "shape_raw": shape_raw.detach().clone(),
                    "gap_raw": gap_raw.detach().clone(),
                    "focus_raw": focus_raw.detach().clone(),
                    "conic_raw": conic_raw.detach().clone(),
                    "edge_raw": edge_raw.detach().clone(),
                    "curvatures": validation_result[2].detach().clone(),
                    "sensor_z": validation_result[3].detach().clone(),
                    "conics": validation_result[4].detach().clone(),
                    "edge_sags": validation_result[5].detach().clone(),
                    "coefficients": validation_result[6].detach().clone(),
                    "positions": validation_result[7].detach().clone(),
                    "gaps": validation_result[8].detach().clone(),
                    "gap_relative": validation_result[9].detach().clone(),
                }

        if iteration == iterations:
            break
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"结构第 {iteration} 步产生非有限损失。")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
        def post_step_merit():
            result = _evaluate_structural_parameter_state(
                lens,
                spec,
                base_curvatures,
                first_surface_z,
                base_gaps,
                shape_raw,
                gap_raw,
                focus_raw,
                conic_raw,
                edge_raw,
                edge_spans,
                train_batches,
                train_chiefs,
                train_target,
                focus_span_mm=focus_span_mm,
                minimum_valid_ratio=minimum_valid_ratio,
                glass_gap_ratio=glass_gap_ratio,
                air_gap_ratio=air_gap_ratio,
                mtf_frequency_cy_mm=mtf_frequency_cy_mm,
                mtf_target=mtf_target,
                mtf_max_weight=mtf_max_weight,
                direct_mtf_max_weight=direct_mtf_max_weight,
                relative_curvature_ratio=relative_curvature_ratio,
                rms_target_mm=rms_target_mm,
                rms_target_weight=rms_target_weight,
                **current_ray_weights,
            )
            return result[0]

        step_diagnostics: dict[str, Any] = {}
        step_ok = _safe_optimizer_step(
            optimizer,
            parameters,
            pre_step_loss=float(train_loss.detach().cpu()),
            post_step_loss_fn=post_step_merit,
            diagnostics=step_diagnostics,
        )
        if not step_ok:
            for group_index, group in enumerate(optimizer.param_groups):
                scheduler.base_lrs[group_index] = group["lr"]
            if step_diagnostics["rejection_reason"] == "merit_increase":
                logging.warning(
                    "结构第 %d 步 merit 从 %.6f 增至 %.6f，连续 %d 次尝试仍变差，"
                    "已回滚并降低学习率。",
                    iteration,
                    step_diagnostics["pre_step_loss"],
                    step_diagnostics["post_step_loss"],
                    step_diagnostics["attempts"],
                )
            else:
                logging.warning(
                    "结构第 %d 步检测到 NaN/Inf 或后验评估异常（%s，尝试 %d 次），"
                    "已回滚并降低学习率。",
                    iteration,
                    step_diagnostics["rejection_reason"],
                    step_diagnostics["attempts"],
                )
        else:
            scheduler.step()

    if best is None:
        raise RuntimeError("结构优化没有生成可用检查点。")
    _assign_surface_positions(lens, best["positions"])
    _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    _assign_aspheres(lens, best["conic_raw"], best["edge_raw"], edge_spans)
    for surface in surfaces:
        surface.c = surface.c.detach().clone()
        surface.d = surface.d.detach().clone()
    for surface in aspheres:
        surface.k = surface.k.detach().clone()
        coefficient_values = []
        coefficient_count = best["edge_raw"].shape[1]
        for order in ASPHERE_ORDERS[:coefficient_count]:
            coefficient = getattr(surface, f"ai{order}").detach().clone()
            setattr(surface, f"ai{order}", coefficient)
            coefficient_values.append(coefficient)
        if surface.ai_degree <= 4:
            surface.ai = torch.stack(coefficient_values).detach().clone()
        else:
            higher_values = []
            for index in range(4, surface.ai_degree):
                order = 2 * (index + 2)
                higher_values.append(
                    getattr(surface, f"ai{order}").detach().clone()
                )
            surface.ai = torch.stack(coefficient_values + higher_values).detach().clone()
    lens.d_sensor = lens.d_sensor.detach().clone()
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)

    serializable_best = {
        "iteration": best["iteration"],
        "diagnostics": best["diagnostics"],
        "curvatures_1_per_mm": best["curvatures"].detach().cpu().tolist(),
        "radii_mm": [
            0.0 if abs(float(value)) < 1e-15 else 1.0 / float(value)
            for value in best["curvatures"].detach().cpu().tolist()
        ],
        "surface_positions_mm": best["positions"].detach().cpu().tolist(),
        "center_gaps_mm": best["gaps"].detach().cpu().tolist(),
        "gap_relative": best["gap_relative"].detach().cpu().tolist(),
        "sensor_z_mm": float(best["sensor_z"].detach().cpu()),
        "conic_constants": best["conics"].detach().cpu().tolist(),
        "edge_asphere_contributions_mm": best["edge_sags"].detach().cpu().tolist(),
        "asphere_coefficients": (
            best["coefficients"].detach().cpu().tolist()
        ),
        "shape_raw": best["shape_raw"].detach().cpu().tolist(),
        "gap_raw": best["gap_raw"].detach().cpu().tolist(),
        "focus_raw": float(best["focus_raw"].detach().cpu()),
        "conic_raw": best["conic_raw"].detach().cpu().tolist(),
        "edge_raw": best["edge_raw"].detach().cpu().tolist(),
    }
    return serializable_best, history


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)


def run(args: argparse.Namespace) -> Path:
    """构建或载入七片处方，执行指定阶段并保存验证结果。"""

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("输出目录必须不存在或为空，避免覆盖既有设计结果。")
    output.mkdir(parents=True, exist_ok=True)
    set_logger(str(output))
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
    )
    spec = MWIRDesignSpec(
        field_y_deg=args.field_y_deg,
        image_height_mm=args.image_height_mm,
        entrance_pupil_diameter_mm=args.entrance_pupil_mm,
        simulation_pixel_pitch_um=args.simulation_pixel_pitch_um,
    )
    params = _scheme_parameters(spec, "transmission_power_bent7")
    source_lens = None
    material_layout = None
    if args.material_layout:
        material_layout = tuple(
            value.strip().lower()
            for value in args.material_layout.split(",")
            if value.strip()
        )
    if args.stage == "spherical":
        if args.input_lens is not None:
            raise ValueError("球面母型阶段不接受 --input-lens；请直接生成干净母型。")
        lens = _build_power_bent7_lens(spec, params, device)
        _apply_mwir_constraints(lens, spec)
        before_calibration = float(lens.foclen)
        _calibrate_initial_power(
            lens,
            spec.effective_focal_length_mm,
            max_iterations=12,
            logarithmic_tolerance=1e-6,
            minimum_factor=0.8,
            maximum_factor=1.2,
        )
        lens.refocus(float("inf"))
        lens.post_computation()
        _apply_mwir_constraints(lens, spec)
        lens.write_lens_json(str(output / "power_bent7_seed.json"))
        logging.info(
            "七片母型：校准前缓存 EFL %.6f mm；校准/调焦后 EFL %.6f mm，"
            "F/# %.6f，像面 z=%.6f mm。",
            before_calibration,
            float(lens.foclen),
            float(lens.fnum),
            _detached_float(lens.d_sensor),
        )
        best, history = optimize_spherical_seed(
            lens,
            spec,
            iterations=args.iterations,
            field_count=args.field_count,
            spp=args.spp,
            validation_spp=args.validation_spp,
            learning_rate=args.learning_rate,
            focus_learning_rate=args.focus_learning_rate,
            focus_span_mm=args.focus_span_mm,
            minimum_valid_ratio=spec.vignetting_floor,
            ray_seed=args.seed + 101,
            checkpoint_interval=args.checkpoint_interval,
        )
        optimized_filename = "power_bent7_spherical_optimized.json"
        best_filename = "best_spherical_state.json"
        stage_status = "七片强弯曲球面优化阶段"
        fixed_variables = ["前置光阑", "全部曲面顶点位置", "材料", "曲率符号"]
        optimized_variables = ["14 面相对曲率", "相对近轴焦面的最佳焦点偏移"]
    elif args.stage == "aspheric":
        if args.input_lens is None:
            raise ValueError("非球面阶段必须用 --input-lens 指定已收敛的球面 JSON。")
        source_path = Path(args.input_lens)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        from deeplens import GeoLens

        lens = GeoLens(filename=str(source_path), device=device)
        if material_layout is not None:
            _set_element_materials(lens, material_layout)
        _apply_mwir_constraints(lens, spec)
        if (
            len(_curved_surfaces(lens)) < 2
            or len(_curved_surfaces(lens)) % 2
            or not _aspheric_surfaces(lens)
        ):
            raise ValueError("输入处方必须包含偶数个折射面和至少一个非球面。")
        source_lens = str(source_path.resolve())
        lens.write_lens_json(str(output / "power_bent7_asphere_seed.json"))
        best, history = optimize_aspheric_seed(
            lens,
            spec,
            iterations=args.iterations,
            field_count=args.field_count,
            spp=args.spp,
            validation_spp=args.validation_spp,
            learning_rate=args.learning_rate,
            focus_learning_rate=args.focus_learning_rate,
            conic_learning_rate=args.conic_learning_rate,
            asphere_learning_rate=args.asphere_learning_rate,
            focus_span_mm=args.focus_span_mm,
            minimum_valid_ratio=spec.vignetting_floor,
            ray_seed=args.seed + 101,
            checkpoint_interval=args.checkpoint_interval,
            relative_curvature_ratio=args.relative_curvature_ratio,
            rms_target_mm=args.rms_target_mm,
            rms_target_weight=args.rms_target_weight,
        )
        optimized_filename = "power_bent7_aspheric_optimized.json"
        best_filename = "best_aspheric_state.json"
        stage_status = "七片强弯曲低阶非球面优化阶段"
        fixed_variables = ["前置光阑", "全部曲面顶点位置", "材料", "曲率符号"]
        optimized_variables = [
            "14 面相对曲率",
            "相对近轴焦面的最佳焦点偏移",
            "5 个预留面的圆锥常数",
            "5 个预留面的 A4/A6/A8/A10",
        ]
    else:
        if args.input_lens is None:
            raise ValueError("结构阶段必须用 --input-lens 指定已收敛的非球面 JSON。")
        source_path = Path(args.input_lens)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        from deeplens import GeoLens

        lens = GeoLens(filename=str(source_path), device=device)
        if material_layout is not None:
            _set_element_materials(lens, material_layout)
        _apply_mwir_constraints(lens, spec)
        if (
            len(_curved_surfaces(lens)) < 2
            or len(_curved_surfaces(lens)) % 2
            or not _aspheric_surfaces(lens)
        ):
            raise ValueError("输入处方必须包含偶数个折射面和至少一个非球面。")
        source_lens = str(source_path.resolve())
        lens.write_lens_json(str(output / "power_bent7_structural_seed.json"))
        best, history = optimize_structural_seed(
            lens,
            spec,
            iterations=args.iterations,
            field_count=args.field_count,
            spp=args.spp,
            validation_spp=args.validation_spp,
            learning_rate=args.learning_rate,
            gap_learning_rate=args.gap_learning_rate,
            focus_learning_rate=args.focus_learning_rate,
            conic_learning_rate=args.conic_learning_rate,
            asphere_learning_rate=args.asphere_learning_rate,
            focus_span_mm=args.focus_span_mm,
            minimum_valid_ratio=spec.vignetting_floor,
            glass_gap_ratio=args.glass_gap_ratio,
            air_gap_ratio=args.air_gap_ratio,
            mtf_frequency_cy_mm=(
                args.mtf_frequency
                if args.mtf_frequency is not None
                else spec.analysis_nyquist_frequency_cy_mm
            ),
            mtf_target=args.mtf_target,
            mtf_surrogate_weight=args.mtf_surrogate_weight,
            mtf_max_weight=args.mtf_max_weight,
            direct_mtf_weight=args.direct_mtf_weight,
            direct_mtf_max_weight=args.direct_mtf_max_weight,
            focus_weight=args.focus_weight,
            astigmatism_weight=args.astigmatism_weight,
            chromatic_focus_weight=args.chromatic_focus_weight,
            field_curvature_weight=args.field_curvature_weight,
            ray_seed=args.seed + 101,
            checkpoint_interval=args.checkpoint_interval,
            relative_curvature_ratio=args.relative_curvature_ratio,
            rms_target_mm=args.rms_target_mm,
            rms_target_weight=args.rms_target_weight,
            curriculum_warmup_fraction=getattr(
                args, "curriculum_warmup_fraction", 0.0
            ),
            curriculum_ramp_fraction=getattr(
                args, "curriculum_ramp_fraction", 0.0
            ),
        )
        optimized_filename = "power_bent7_structural_optimized.json"
        best_filename = "best_structural_state.json"
        stage_status = "七片强弯曲结构间隔联合优化阶段"
        fixed_variables = ["前置光阑", "第一折射面顶点位置", "材料", "曲率符号"]
        optimized_variables = [
            "14 面相对曲率",
            "7 个玻璃中心厚度",
            "6 个空气间隔",
            "相对近轴焦面的最佳焦点偏移",
            "5 个预留面的圆锥常数",
            "5 个预留面的 A4/A6/A8/A10",
        ]

    lens.write_lens_json(str(output / optimized_filename))
    _json_dump(output / "optimization_history.json", history)
    _json_dump(output / best_filename, best)
    metrics = evaluate_lens(
        lens,
        spec,
        output,
        psf_spp=args.eval_spp,
        vignetting_grid=9,
        vignetting_rays=max(64, min(args.eval_spp, 256)),
    )
    metadata = {
        "status": f"{stage_status}；只有 mwir_metrics.json 全部通过后才能称为最终处方。",
        "spec": spec.geometry_report(),
        "design": params,
        "stage": args.stage,
        "source_lens_json": source_lens,
        "material_layout_override": material_layout,
        "optimizer": {
            "method": (
                "相对曲率有界参数化 + 每步共同倍率 EFL 校准 + 有界焦点偏移 Adam；"
                "非球面阶段额外使用有界 k 与按边缘矢高归一化的 A4–A10；"
                "结构阶段再加入正值有界的玻璃厚度和空气间隔。"
            ),
            "fixed": fixed_variables,
            "optimized": optimized_variables,
            "iterations": args.iterations,
            "field_count": args.field_count,
            "rays_per_field_wavelength": args.spp,
            "validation_rays_per_field_wavelength": args.validation_spp,
            "learning_rate": args.learning_rate,
            "gap_learning_rate": args.gap_learning_rate,
            "focus_learning_rate": args.focus_learning_rate,
            "conic_learning_rate": args.conic_learning_rate,
            "asphere_learning_rate": args.asphere_learning_rate,
            "focus_span_mm": args.focus_span_mm,
            "glass_gap_ratio": args.glass_gap_ratio,
            "air_gap_ratio": args.air_gap_ratio,
            "mtf_frequency_cy_mm": args.mtf_frequency,
            "mtf_target": args.mtf_target,
            "mtf_surrogate_weight": args.mtf_surrogate_weight,
            "mtf_max_weight": args.mtf_max_weight,
            "direct_mtf_weight": args.direct_mtf_weight,
            "direct_mtf_max_weight": args.direct_mtf_max_weight,
            "focus_weight": args.focus_weight,
            "astigmatism_weight": args.astigmatism_weight,
            "chromatic_focus_weight": args.chromatic_focus_weight,
            "field_curvature_weight": args.field_curvature_weight,
            "curriculum_warmup_fraction": getattr(
                args, "curriculum_warmup_fraction", 0.0
            ),
            "curriculum_ramp_fraction": getattr(
                args, "curriculum_ramp_fraction", 0.0
            ),
            "seed": args.seed,
        },
        "best_state": best,
        "pass": metrics.get("pass", {}),
        "next_stage": (
            "若结构阶段后系统 MTF 仍未通过，增加固定验证场并启用多频 OTF/MTF 代理；"
            "不能把仅通过一阶参数和 RMS 下降的处方称为最终设计。"
        ),
    }
    _json_dump(output / "design_run_metadata.json", metadata)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="受约束优化七片强弯曲 MWIR 透射母型"
    )
    parser.add_argument("--output", default="results/mwir-power-bent7-spherical")
    parser.add_argument(
        "--stage",
        choices=("spherical", "aspheric", "structural"),
        default="spherical",
    )
    parser.add_argument("--input-lens", default=None)
    parser.add_argument(
        "--material-layout",
        default=None,
        help="可选的逗号分隔透镜材料布局，例如 ge,mgf2,si,caf2,znse,caf2,ge。",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--spp", type=int, default=48)
    parser.add_argument("--validation-spp", type=int, default=128)
    parser.add_argument("--eval-spp", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gap-learning-rate", type=float, default=1e-4)
    parser.add_argument("--focus-learning-rate", type=float, default=1e-3)
    parser.add_argument("--conic-learning-rate", type=float, default=1e-4)
    parser.add_argument("--asphere-learning-rate", type=float, default=1e-5)
    parser.add_argument("--focus-span-mm", type=float, default=30.0)
    parser.add_argument("--glass-gap-ratio", type=float, default=1.25)
    parser.add_argument("--air-gap-ratio", type=float, default=2.0)
    parser.add_argument(
        "--relative-curvature-ratio",
        type=float,
        default=1.25,
        help="每个曲面相对初始曲率的最大/最小倍率；大于 1 可放宽功率重分配。",
    )
    parser.add_argument(
        "--rms-target-mm",
        type=float,
        default=None,
        help="可选的逐场 RMS 目标（mm），与 --rms-target-weight 联合使用。",
    )
    parser.add_argument(
        "--rms-target-weight",
        type=float,
        default=0.0,
        help="逐场 RMS 目标惩罚权重；默认 0 表示保持原 merit。",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--field-y-deg", type=float, default=9.6)
    parser.add_argument("--image-height-mm", type=float, default=47.1454)
    parser.add_argument("--entrance-pupil-mm", type=float, default=280.0)
    parser.add_argument("--simulation-pixel-pitch-um", type=float, default=30.0)
    parser.add_argument(
        "--mtf-frequency",
        type=float,
        default=None,
        help="MTF 代理频率；省略时使用当前虚拟探测器奈奎斯特频率。",
    )
    parser.add_argument("--mtf-target", type=float, default=0.55)
    parser.add_argument("--mtf-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--mtf-max-weight", type=float, default=1.0)
    parser.add_argument("--direct-mtf-weight", type=float, default=0.0)
    parser.add_argument("--direct-mtf-max-weight", type=float, default=1.0)
    parser.add_argument("--focus-weight", type=float, default=0.0)
    parser.add_argument("--astigmatism-weight", type=float, default=0.0)
    parser.add_argument("--chromatic-focus-weight", type=float, default=0.0)
    parser.add_argument("--field-curvature-weight", type=float, default=0.0)
    parser.add_argument(
        "--curriculum-warmup-fraction",
        type=float,
        default=0.25,
        help="前期仅使用质心 RMS 与基础可行性项的迭代比例。",
    )
    parser.add_argument(
        "--curriculum-ramp-fraction",
        type=float,
        default=0.5,
        help="focus/像散/色焦/场曲/MTF 权重由 0 平滑升至设定值的迭代比例。",
    )
    return parser


def main() -> None:
    configure_utf8_console()
    args = _build_parser().parse_args()
    output = run(args)
    print(f"七片强弯曲 {args.stage} 阶段已保存到：{output.resolve()}")


if __name__ == "__main__":
    main()
