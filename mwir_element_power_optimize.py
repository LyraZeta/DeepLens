"""以七片净光焦度和弯曲量为变量的 MWIR 结构实验。

现有 ``shape_raw`` 直接调 14 个曲率，经过共同 EFL 校准后，元素净光焦度
的梯度可能互相抵消。本脚本把每片透镜写成

``c_front = bend + power / (2(n-1))``
``c_rear  = bend - power / (2(n-1))``

从而直接搜索七片净光焦度与弯曲量。它只用于验证优化自由度是否为当前瓶颈，
默认载入当前最佳连续处方，不宣称输出已经达到 MTF 指标。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch

from deeplens import GeoLens
from deeplens.utils import set_logger, set_seed
from mwir_power_bent7_optimize import (
    _assign_geometry,
    _clearance_penalty,
    _curriculum_ray_weights,
    _curriculum_scale,
    _curved_surfaces,
    _detached_float,
    _ray_merit,
    _sample_fixed_rays,
    _safe_optimizer_step,
    paraxial_state,
)
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import _apply_mwir_constraints, evaluate_lens


def _element_power_state(lens, curvatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """返回每片净光焦度和曲率弯曲量。"""

    surfaces = _curved_surfaces(lens)
    powers = []
    bends = []
    primary = torch.tensor(3.5, dtype=curvatures.dtype, device=curvatures.device)
    for index in range(0, len(surfaces), 2):
        front, rear = surfaces[index], surfaces[index + 1]
        n = front.mat2.ior(primary).to(dtype=curvatures.dtype, device=curvatures.device)
        powers.append((n - 1.0) * (curvatures[index] - curvatures[index + 1]))
        bends.append(0.5 * (curvatures[index] + curvatures[index + 1]))
    return torch.stack(powers), torch.stack(bends)


def _curvatures_from_power(
    lens,
    powers: torch.Tensor,
    bends: torch.Tensor,
) -> torch.Tensor:
    surfaces = _curved_surfaces(lens)
    if powers.shape != bends.shape or powers.numel() * 2 != len(surfaces):
        raise ValueError("净光焦度/弯曲量数量必须等于透镜片数。")
    primary = torch.tensor(3.5, dtype=powers.dtype, device=powers.device)
    values = []
    for index in range(powers.numel()):
        front = surfaces[2 * index]
        n = front.mat2.ior(primary).to(dtype=powers.dtype, device=powers.device)
        half_power = powers[index] / (2.0 * (n - 1.0))
        values.extend((bends[index] + half_power, bends[index] - half_power))
    return torch.stack(values)


def _calibrate_efl(lens, curvatures, target_mm: float, iterations: int = 12):
    scale = torch.ones((), dtype=curvatures.dtype, device=curvatures.device)
    target = torch.as_tensor(target_mm, dtype=curvatures.dtype, device=curvatures.device)
    for _ in range(iterations):
        trial = curvatures * scale
        state = paraxial_state(lens, trial)
        correction = (state.effective_focal_length_mm.abs() / target).clamp(0.6, 1.8)
        scale = scale * correction
    final = curvatures * scale
    return final, paraxial_state(lens, final)


def _paraxial_chromatic_focus_loss(lens, curvatures, wavelengths_um):
    """返回多波长近轴后焦点的标准差（毫米）。"""

    focus_positions = torch.stack(
        [
            paraxial_state(
                lens,
                curvatures,
                wavelength_um=float(wavelength),
            ).focus_z_mm
            for wavelength in wavelengths_um
        ]
    )
    loss = torch.sqrt(
        (focus_positions - focus_positions.mean()).square().mean() + 1e-12
    )
    return loss, focus_positions


def run(args: argparse.Namespace) -> Path:
    configure_utf8_console()
    set_seed(args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录必须为空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    set_logger(str(output))
    device = torch.device(args.device)
    spec = MWIRDesignSpec()
    curriculum_warmup_fraction = float(
        getattr(args, "curriculum_warmup_fraction", 0.0)
    )
    curriculum_ramp_fraction = float(
        getattr(args, "curriculum_ramp_fraction", 0.0)
    )
    mtf_frequency_cy_mm = getattr(args, "mtf_frequency", None)
    if mtf_frequency_cy_mm is None:
        mtf_frequency_cy_mm = spec.analysis_nyquist_frequency_cy_mm
    mtf_target = float(getattr(args, "mtf_target", 0.55))
    mtf_surrogate_weight = float(getattr(args, "mtf_surrogate_weight", 0.0))
    mtf_max_weight = float(getattr(args, "mtf_max_weight", 1.0))
    direct_mtf_weight = float(getattr(args, "direct_mtf_weight", 0.0))
    direct_mtf_max_weight = float(getattr(args, "direct_mtf_max_weight", 1.0))
    paraxial_chromatic_weight = float(args.paraxial_chromatic_weight)
    if (
        not math.isfinite(paraxial_chromatic_weight)
        or paraxial_chromatic_weight < 0.0
    ):
        raise ValueError("paraxial_chromatic_weight 必须为非负有限值。")
    lens = GeoLens(filename=args.input_lens, device=device)
    _apply_mwir_constraints(lens, spec)
    surfaces = _curved_surfaces(lens)
    if len(surfaces) != 14:
        raise ValueError("元素级功率实验要求七片、14 个折射面。")
    base_curvatures = torch.stack([s.c.detach().clone().to(device) for s in surfaces])
    base_powers, base_bends = _element_power_state(lens, base_curvatures)
    base_first_order = paraxial_state(lens, base_curvatures)
    initial_focus_shift = float(
        (lens.d_sensor - base_first_order.focus_z_mm).detach().cpu()
    )
    focus_fraction = min(
        max(initial_focus_shift / args.focus_span_mm, -0.95), 0.95
    )
    # 允许净光焦度跨过零，但限制单片功率的搜索尺度，避免第一步就失效。
    power_span = torch.maximum(base_powers.abs() * args.power_span_factor,
                               torch.full_like(base_powers, args.minimum_power_span))
    bend_span = torch.maximum(base_bends.abs() * args.bend_span_factor,
                              torch.full_like(base_bends, args.minimum_bend_span))
    power_raw = torch.zeros_like(base_powers, requires_grad=True)
    bend_raw = torch.zeros_like(base_bends, requires_grad=True)
    focus_raw = torch.tensor(
        math.atanh(focus_fraction),
        dtype=base_curvatures.dtype,
        device=device,
        requires_grad=True,
    )
    train_batches, train_chiefs, train_target = _sample_fixed_rays(
        lens, spec, field_count=args.field_count, spp=args.spp,
        seed=args.seed + 101, pupil_scale=1.0
    )
    val_batches, val_chiefs, val_target = _sample_fixed_rays(
        lens, spec, field_count=max(args.field_count, 7), spp=args.validation_spp,
        seed=args.seed + 10100, pupil_scale=1.0
    )
    optimizer = torch.optim.Adam([
        {"params": [power_raw], "lr": args.power_learning_rate},
        {"params": [bend_raw], "lr": args.bend_learning_rate},
        {"params": [focus_raw], "lr": args.focus_learning_rate},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iterations)
    best = None
    history = []
    has_scheduled_weights = any(
        weight > 0.0
        for weight in (
            mtf_surrogate_weight,
            direct_mtf_weight,
            args.focus_weight,
            args.astigmatism_weight,
            args.chromatic_focus_weight,
            args.field_curvature_weight,
            paraxial_chromatic_weight,
        )
    )
    for iteration in range(args.iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        curriculum_scale = _curriculum_scale(
            iteration,
            args.iterations,
            warmup_fraction=curriculum_warmup_fraction,
            ramp_fraction=curriculum_ramp_fraction,
        )
        current_ray_weights = _curriculum_ray_weights(
            curriculum_scale,
            mtf_surrogate_weight=mtf_surrogate_weight,
            direct_mtf_weight=direct_mtf_weight,
            focus_weight=args.focus_weight,
            astigmatism_weight=args.astigmatism_weight,
            chromatic_focus_weight=args.chromatic_focus_weight,
            field_curvature_weight=args.field_curvature_weight,
        )
        current_paraxial_chromatic_weight = (
            curriculum_scale * paraxial_chromatic_weight
        )
        active_auxiliary_weights = dict(current_ray_weights)
        active_auxiliary_weights["paraxial_chromatic_weight"] = (
            current_paraxial_chromatic_weight
        )
        powers = base_powers + power_span * torch.tanh(power_raw)
        bends = base_bends + bend_span * torch.tanh(bend_raw)
        raw_curvatures = _curvatures_from_power(lens, powers, bends)
        curvatures, first_order = _calibrate_efl(lens, raw_curvatures, spec.effective_focal_length_mm)
        sensor_z = first_order.focus_z_mm + args.focus_span_mm * torch.tanh(focus_raw)
        _assign_geometry(lens, curvatures, sensor_z)
        train_loss, train_diag = _ray_merit(
            lens, train_batches, train_chiefs, train_target,
            minimum_valid_ratio=spec.vignetting_floor,
            mtf_frequency_cy_mm=mtf_frequency_cy_mm,
            mtf_target=mtf_target,
            mtf_max_weight=mtf_max_weight,
            direct_mtf_max_weight=direct_mtf_max_weight,
            **current_ray_weights,
        )
        clear_loss, clear = _clearance_penalty(lens, curvatures)
        paraxial_color_loss, paraxial_focus_positions = _paraxial_chromatic_focus_loss(
            lens, curvatures, spec.wavelengths_um
        )
        total = (
            train_loss
            + 2.0 * clear_loss
            + current_paraxial_chromatic_weight * paraxial_color_loss
        )
        with torch.no_grad():
            val_curvatures, val_first = _calibrate_efl(lens, raw_curvatures, spec.effective_focal_length_mm)
            val_sensor = val_first.focus_z_mm + args.focus_span_mm * torch.tanh(focus_raw)
            _assign_geometry(lens, val_curvatures, val_sensor)
            val_ray_loss, val_diag = _ray_merit(
                lens, val_batches, val_chiefs, val_target,
                minimum_valid_ratio=spec.vignetting_floor,
                mtf_frequency_cy_mm=mtf_frequency_cy_mm,
                mtf_target=mtf_target,
                mtf_max_weight=mtf_max_weight,
                direct_mtf_max_weight=direct_mtf_max_weight,
                **current_ray_weights,
            )
        val_diag = dict(val_diag)
        val_diag.update(clear)
        validation_color_loss, validation_focus_positions = (
            _paraxial_chromatic_focus_loss(
                lens, val_curvatures, spec.wavelengths_um
            )
        )
        validation_total = (
            val_ray_loss
            + 2.0 * clear_loss
            + current_paraxial_chromatic_weight * validation_color_loss
        )
        val_diag.update({"iteration": iteration,
                         "loss": float(validation_total.detach().cpu()),
                         "curriculum_scale": curriculum_scale,
                         "active_auxiliary_weights": active_auxiliary_weights,
                         "power_min_1_per_mm": float(powers.min().detach().cpu()),
                         "power_max_1_per_mm": float(powers.max().detach().cpu()),
                         "sensor_z_mm": float(val_sensor.detach().cpu()),
                         "paraxial_chromatic_focus_loss_mm": float(
                             validation_color_loss.detach().cpu()
                         ),
                         "paraxial_focus_positions_mm": [
                             float(value)
                             for value in validation_focus_positions.detach().cpu().tolist()
                         ]})
        history.append(val_diag)
        selection_eligible = (
            not has_scheduled_weights or curriculum_scale >= 1.0 - 1e-12
        )
        if selection_eligible and (
            best is None or val_diag["loss"] < best["score"]
        ):
            best = {"score": val_diag["loss"],
                    "diagnostics": val_diag, "powers": powers.detach().clone(),
                    "bends": bends.detach().clone(), "curvatures": val_curvatures.detach().clone(),
                    "sensor_z": val_sensor.detach().clone()}
        if iteration % max(1, args.checkpoint_interval) == 0 or iteration == args.iterations:
            logging.info("元素功率迭代 %d/%d：验证 RMS %.6f/%.6f mm；净功率 %.4g..%.4g 1/mm；像面 %.3f mm；curriculum %.3f",
                         iteration, args.iterations, val_diag["rms_mean_mm"], val_diag["rms_max_mm"],
                         val_diag["power_min_1_per_mm"], val_diag["power_max_1_per_mm"], val_diag["sensor_z_mm"],
                         curriculum_scale)
        if iteration == args.iterations:
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_([power_raw, bend_raw, focus_raw], 10.0)
        def post_step_merit():
            post_powers = base_powers + power_span * torch.tanh(power_raw)
            post_bends = base_bends + bend_span * torch.tanh(bend_raw)
            post_raw_curvatures = _curvatures_from_power(
                lens, post_powers, post_bends
            )
            post_curvatures, post_first_order = _calibrate_efl(
                lens,
                post_raw_curvatures,
                spec.effective_focal_length_mm,
            )
            post_sensor = post_first_order.focus_z_mm + (
                args.focus_span_mm * torch.tanh(focus_raw)
            )
            _assign_geometry(lens, post_curvatures, post_sensor)
            post_ray_loss, _ = _ray_merit(
                lens,
                train_batches,
                train_chiefs,
                train_target,
                minimum_valid_ratio=spec.vignetting_floor,
                mtf_frequency_cy_mm=mtf_frequency_cy_mm,
                mtf_target=mtf_target,
                mtf_max_weight=mtf_max_weight,
                direct_mtf_max_weight=direct_mtf_max_weight,
                **current_ray_weights,
            )
            post_clearance, _ = _clearance_penalty(lens, post_curvatures)
            post_color, _ = _paraxial_chromatic_focus_loss(
                lens, post_curvatures, spec.wavelengths_um
            )
            return (
                post_ray_loss
                + 2.0 * post_clearance
                + current_paraxial_chromatic_weight * post_color
            )

        step_diagnostics: dict[str, object] = {}
        step_ok = _safe_optimizer_step(
            optimizer,
            [power_raw, bend_raw, focus_raw],
            pre_step_loss=float(total.detach().cpu()),
            post_step_loss_fn=post_step_merit,
            diagnostics=step_diagnostics,
        )
        if step_ok:
            if int(step_diagnostics.get("attempts") or 1) > 1:
                for group_index, group in enumerate(optimizer.param_groups):
                    scheduler.base_lrs[group_index] = group["lr"]
            scheduler.step()
        else:
            for group_index, group in enumerate(optimizer.param_groups):
                scheduler.base_lrs[group_index] = group["lr"]
            logging.warning(
                "元素功率第 %d 步被 merit guard 回滚：原因=%s，"
                "merit %.6f -> %s，尝试=%d，学习率=%s。",
                iteration,
                step_diagnostics.get("rejection_reason"),
                float(step_diagnostics.get("pre_step_loss") or 0.0),
                (
                    "None"
                    if step_diagnostics.get("post_step_loss") is None
                    else f"{float(step_diagnostics['post_step_loss']):.6f}"
                ),
                int(step_diagnostics.get("attempts") or 0),
                ",".join(f"{group['lr']:.3g}" for group in optimizer.param_groups),
            )
    if best is None:
        raise RuntimeError("元素功率优化没有生成结果。")
    with torch.no_grad():
        _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    lens.post_computation(); _apply_mwir_constraints(lens, spec)
    lens.write_lens_json(str(output / "element_power_optimized.json"))
    with open(output / "element_power_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(output / "best_element_power_state.json", "w", encoding="utf-8") as f:
        json.dump({"diagnostics": best["diagnostics"],
                   "powers_1_per_mm": best["powers"].cpu().tolist(),
                   "bends_1_per_mm": best["bends"].cpu().tolist(),
                   "curvatures_1_per_mm": best["curvatures"].cpu().tolist(),
                   "sensor_z_mm": float(best["sensor_z"].cpu())}, f, ensure_ascii=False, indent=2)
    evaluate_lens(lens, spec, output, psf_spp=args.eval_spp,
                  vignetting_grid=7, vignetting_rays=min(args.eval_spp, 256))
    return output


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MWIR 七片元素级净光焦度优化实验")
    p.add_argument("--input-lens", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--field-count", type=int, default=5)
    p.add_argument("--spp", type=int, default=32)
    p.add_argument("--validation-spp", type=int, default=64)
    p.add_argument("--eval-spp", type=int, default=256)
    p.add_argument("--power-learning-rate", type=float, default=3e-5)
    p.add_argument("--bend-learning-rate", type=float, default=1e-4)
    p.add_argument("--focus-learning-rate", type=float, default=1e-3)
    p.add_argument("--power-span-factor", type=float, default=2.0)
    p.add_argument("--bend-span-factor", type=float, default=2.0)
    p.add_argument("--minimum-power-span", type=float, default=0.0008)
    p.add_argument("--minimum-bend-span", type=float, default=0.0003)
    p.add_argument("--focus-span-mm", type=float, default=30.0)
    p.add_argument("--focus-weight", type=float, default=0.0)
    p.add_argument("--astigmatism-weight", type=float, default=0.0)
    p.add_argument("--chromatic-focus-weight", type=float, default=0.0)
    p.add_argument("--field-curvature-weight", type=float, default=0.0)
    p.add_argument(
        "--mtf-frequency",
        type=float,
        default=None,
        help="MTF merit 频率；省略时使用虚拟探测器奈奎斯特频率。",
    )
    p.add_argument("--mtf-target", type=float, default=0.55)
    p.add_argument("--mtf-surrogate-weight", type=float, default=0.0)
    p.add_argument("--mtf-max-weight", type=float, default=1.0)
    p.add_argument("--direct-mtf-weight", type=float, default=0.0)
    p.add_argument("--direct-mtf-max-weight", type=float, default=1.0)
    p.add_argument(
        "--paraxial-chromatic-weight",
        type=float,
        default=0.0,
        help="近轴多波长后焦点标准差权重（毫米）；用于早期色焦搜索。",
    )
    p.add_argument(
        "--curriculum-warmup-fraction",
        type=float,
        default=0.25,
        help="前期仅使用质心 RMS 与基础可行性项的迭代比例。",
    )
    p.add_argument(
        "--curriculum-ramp-fraction",
        type=float,
        default=0.5,
        help="focus/像散/色焦/场曲/MTF 权重由 0 平滑升至设定值的迭代比例。",
    )
    p.add_argument("--checkpoint-interval", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260747)
    return p


if __name__ == "__main__":
    run(_parser().parse_args())
