"""MWIR 七片系统的保守 MTF 课程优化实验。

该脚本面向已经具有真实曲率和非球面自由度的七片候选处方。它不会从近似
平板的材料种子开始，而是固定材料与顶点位置，联合微调：

* 14 个折射面的相对曲率；
* 公共像面位置；
* 已预留非球面的圆锥常数；
* A4 起的归一化边缘矢高。

训练前段以质心 RMS 为主，随后平滑引入奈奎斯特频率处的双方向 MTF 代理。
每次更新均复算同一批固定光线；若 merit 变差，则恢复参数和优化器状态并
缩小步长。输出仍是实验候选，最终是否可用只由独立 ``mwir_metrics.json``
决定。
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
    ASPHERE_ORDERS,
    _apply_mwir_constraints,
    _aspheric_surfaces,
    _assign_aspheres,
    _assign_geometry,
    _curved_surfaces,
    _evaluate_aspheric_parameter_state,
    _initial_asphere_raw,
    _safe_optimizer_step,
    _sample_fixed_rays,
    paraxial_state,
)
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import evaluate_lens


def _smooth_ramp(iteration: int, start: int, end: int) -> float:
    """返回从 0 平滑增加到 1 的余弦课程系数。"""

    if end <= start:
        return 1.0 if iteration >= start else 0.0
    fraction = min(max((iteration - start) / (end - start), 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * fraction)


def _objective(
    lens,
    spec: MWIRDesignSpec,
    base_curvatures: torch.Tensor,
    shape_raw: torch.Tensor,
    focus_raw: torch.Tensor,
    conic_raw: torch.Tensor,
    edge_raw: torch.Tensor,
    edge_spans: torch.Tensor,
    ray_batches,
    chief_batches,
    target_xy: torch.Tensor,
    *,
    focus_span_mm: float,
    relative_curvature_ratio: float,
    mtf_weight: float,
    mtf_target: float,
    mtf_max_weight: float,
    focus_weight: float,
    astigmatism_weight: float,
    chromatic_focus_weight: float,
    field_curvature_weight: float,
):
    return _evaluate_aspheric_parameter_state(
        lens,
        spec,
        base_curvatures,
        shape_raw,
        focus_raw,
        conic_raw,
        edge_raw,
        edge_spans,
        ray_batches,
        chief_batches,
        target_xy,
        focus_span_mm=focus_span_mm,
        minimum_valid_ratio=spec.vignetting_floor,
        mtf_frequency_cy_mm=spec.analysis_nyquist_frequency_cy_mm,
        mtf_target=mtf_target,
        mtf_surrogate_weight=mtf_weight,
        mtf_max_weight=mtf_max_weight,
        focus_weight=focus_weight,
        astigmatism_weight=astigmatism_weight,
        chromatic_focus_weight=chromatic_focus_weight,
        field_curvature_weight=field_curvature_weight,
        relative_curvature_ratio=relative_curvature_ratio,
    )


def run(args: argparse.Namespace) -> Path:
    configure_utf8_console()
    set_seed(args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录必须为空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    set_logger(str(output))

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    spec = MWIRDesignSpec()
    lens = GeoLens(filename=args.input_lens, device=device, dtype=dtype)
    _apply_mwir_constraints(lens, spec)
    surfaces = _curved_surfaces(lens)
    aspheres = _aspheric_surfaces(lens)
    if len(surfaces) != 14 or not aspheres:
        raise ValueError("MTF 课程优化要求七片、14 个折射面和非球面自由度。")

    base_curvatures = torch.stack(
        [surface.c.detach().clone().to(device) for surface in surfaces]
    )
    base_first_order = paraxial_state(lens, base_curvatures)
    focus_shift = float(
        (lens.d_sensor - base_first_order.focus_z_mm).detach().cpu()
    )
    focus_fraction = min(
        max(focus_shift / args.focus_span_mm, -0.95), 0.95
    )
    shape_raw = torch.zeros_like(base_curvatures, requires_grad=True)
    focus_raw = torch.tensor(
        math.atanh(focus_fraction),
        dtype=base_curvatures.dtype,
        device=device,
        requires_grad=True,
    )
    conic_raw, edge_raw, edge_spans = _initial_asphere_raw(
        lens, dtype=base_curvatures.dtype, device=device
    )

    parameters = [shape_raw, focus_raw, conic_raw, edge_raw]
    optimizer = torch.optim.SGD(
        [
            {"params": [shape_raw], "lr": args.shape_learning_rate},
            {"params": [focus_raw], "lr": args.focus_learning_rate},
            {"params": [conic_raw], "lr": args.conic_learning_rate},
            {"params": [edge_raw], "lr": args.asphere_learning_rate},
        ]
    )

    train_batches, train_chiefs, train_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=args.field_count,
        spp=args.spp,
        seed=args.seed + 101,
        pupil_scale=1.0,
    )
    validation_batches, validation_chiefs, validation_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=max(args.field_count, 7),
        spp=args.validation_spp,
        seed=args.seed + 10_101,
        pupil_scale=1.0,
    )

    history: list[dict] = []
    best: dict | None = None
    for iteration in range(args.iterations + 1):
        ramp = _smooth_ramp(
            iteration, args.curriculum_start, args.curriculum_end
        )
        mtf_weight = args.mtf_weight * ramp
        focus_weight = args.focus_weight * ramp
        astigmatism_weight = args.astigmatism_weight * ramp
        chromatic_focus_weight = args.chromatic_focus_weight * ramp
        field_curvature_weight = args.field_curvature_weight * ramp

        optimizer.zero_grad(set_to_none=True)
        train_result = _objective(
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
            focus_span_mm=args.focus_span_mm,
            relative_curvature_ratio=args.relative_curvature_ratio,
            mtf_weight=mtf_weight,
            mtf_target=args.mtf_target,
            mtf_max_weight=args.mtf_max_weight,
            focus_weight=focus_weight,
            astigmatism_weight=astigmatism_weight,
            chromatic_focus_weight=chromatic_focus_weight,
            field_curvature_weight=field_curvature_weight,
        )
        train_loss = train_result[0]

        # 验证始终使用最终权重，使不同课程阶段的候选分数可直接比较。
        with torch.no_grad():
            validation_result = _objective(
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
                focus_span_mm=args.focus_span_mm,
                relative_curvature_ratio=args.relative_curvature_ratio,
                mtf_weight=args.mtf_weight,
                mtf_target=args.mtf_target,
                mtf_max_weight=args.mtf_max_weight,
                focus_weight=args.focus_weight,
                astigmatism_weight=args.astigmatism_weight,
                chromatic_focus_weight=args.chromatic_focus_weight,
                field_curvature_weight=args.field_curvature_weight,
            )
        validation_diag = dict(validation_result[1])
        validation_diag.update(
            {
                "iteration": iteration,
                "curriculum_fraction": ramp,
                "training_loss": float(train_loss.detach().cpu()),
                "validation_loss": float(validation_result[0].detach().cpu()),
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
        )
        history.append(validation_diag)
        score = validation_diag["validation_loss"]
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "diagnostics": validation_diag,
                "shape_raw": shape_raw.detach().clone(),
                "focus_raw": focus_raw.detach().clone(),
                "conic_raw": conic_raw.detach().clone(),
                "edge_raw": edge_raw.detach().clone(),
                "curvatures": validation_result[2].detach().clone(),
                "sensor_z": validation_result[3].detach().clone(),
            }

        if (
            iteration == 0
            or iteration == args.iterations
            or iteration % args.checkpoint_interval == 0
        ):
            logging.info(
                "MTF 课程迭代 %d/%d：课程 %.3f；验证 RMS %.6f/%.6f mm；"
                "MTF 代理 %.4f（最坏超差 %.4f）；merit %.6f；lr=%s。",
                iteration,
                args.iterations,
                ramp,
                validation_diag["rms_mean_mm"],
                validation_diag["rms_max_mm"],
                validation_diag["mtf_surrogate_loss"],
                validation_diag["mtf_surrogate_violation_max"],
                score,
                ",".join(
                    f"{float(group['lr']):.3g}"
                    for group in optimizer.param_groups
                ),
            )

        if iteration == args.iterations:
            break
        if not torch.isfinite(train_loss):
            raise RuntimeError(f"第 {iteration} 步训练 merit 非有限。")
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)

        def post_step_merit():
            return _objective(
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
                focus_span_mm=args.focus_span_mm,
                relative_curvature_ratio=args.relative_curvature_ratio,
                mtf_weight=mtf_weight,
                mtf_target=args.mtf_target,
                mtf_max_weight=args.mtf_max_weight,
                focus_weight=focus_weight,
                astigmatism_weight=astigmatism_weight,
                chromatic_focus_weight=chromatic_focus_weight,
                field_curvature_weight=field_curvature_weight,
            )[0]

        step_diag: dict[str, object] = {}
        accepted = _safe_optimizer_step(
            optimizer,
            parameters,
            pre_step_loss=float(train_loss.detach().cpu()),
            post_step_loss_fn=post_step_merit,
            diagnostics=step_diag,
        )
        if not accepted:
            logging.warning(
                "第 %d 步被 merit guard 回滚：%s；lr=%s。",
                iteration,
                step_diag.get("rejection_reason"),
                ",".join(
                    f"{float(group['lr']):.3g}"
                    for group in optimizer.param_groups
                ),
            )

    if best is None:
        raise RuntimeError("MTF 课程优化没有生成候选。")

    _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    _assign_aspheres(lens, best["conic_raw"], best["edge_raw"], edge_spans)
    for surface in surfaces:
        surface.c = surface.c.detach().clone()
    coefficient_count = best["edge_raw"].shape[1]
    for surface in aspheres:
        surface.k = surface.k.detach().clone()
        values = []
        for order in ASPHERE_ORDERS[:coefficient_count]:
            value = getattr(surface, f"ai{order}").detach().clone()
            setattr(surface, f"ai{order}", value)
            values.append(value)
        surface.ai = torch.stack(values).detach().clone()
    lens.d_sensor = lens.d_sensor.detach().clone()
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)
    lens.write_lens_json(str(output / "mwir_mtf_curriculum_optimized.json"))

    with open(output / "optimization_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2, allow_nan=False)
    with open(output / "best_state.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "score": best["score"],
                "diagnostics": best["diagnostics"],
                "curvatures_1_per_mm": best["curvatures"].cpu().tolist(),
                "sensor_z_mm": float(best["sensor_z"].cpu()),
                "shape_raw": best["shape_raw"].cpu().tolist(),
                "focus_raw": float(best["focus_raw"].cpu()),
                "conic_raw": best["conic_raw"].cpu().tolist(),
                "edge_raw": best["edge_raw"].cpu().tolist(),
            },
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    evaluate_lens(
        lens,
        spec,
        output,
        psf_spp=args.eval_spp,
        vignetting_grid=9,
        vignetting_rays=max(64, min(args.eval_spp, 256)),
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MWIR 七片 MTF 课程优化实验")
    parser.add_argument("--input-lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--validation-spp", type=int, default=64)
    parser.add_argument("--eval-spp", type=int, default=512)
    parser.add_argument("--shape-learning-rate", type=float, default=3e-5)
    parser.add_argument("--focus-learning-rate", type=float, default=3e-5)
    parser.add_argument("--conic-learning-rate", type=float, default=3e-5)
    parser.add_argument("--asphere-learning-rate", type=float, default=3e-6)
    parser.add_argument("--relative-curvature-ratio", type=float, default=3.0)
    parser.add_argument("--focus-span-mm", type=float, default=30.0)
    parser.add_argument("--mtf-target", type=float, default=0.3)
    parser.add_argument("--mtf-weight", type=float, default=0.05)
    parser.add_argument("--mtf-max-weight", type=float, default=2.0)
    parser.add_argument("--focus-weight", type=float, default=0.0)
    parser.add_argument("--astigmatism-weight", type=float, default=0.0)
    parser.add_argument("--chromatic-focus-weight", type=float, default=0.0)
    parser.add_argument("--field-curvature-weight", type=float, default=0.0)
    parser.add_argument("--curriculum-start", type=int, default=20)
    parser.add_argument("--curriculum-end", type=int, default=80)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
