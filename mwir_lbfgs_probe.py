"""用固定光线和 L-BFGS 探索七片非球面处方的局部可达像质。

这是一个实验性探针，不把输出直接标记为最终设计。它复用
``mwir_power_bent7_optimize`` 的参数化和 merit，目的是判断 Adam 局部停滞
是否来自步长/动量，而不是再次实现一套光线追迹。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deeplens import GeoLens
from deeplens.utils import set_logger, set_seed
from mwir_power_bent7_optimize import (
    _apply_mwir_constraints,
    _aspheric_surfaces,
    _assign_aspheres,
    _assign_geometry,
    _curved_surfaces,
    _detached_float,
    _evaluate_aspheric_parameter_state,
    _initial_asphere_raw,
    _sample_fixed_rays,
    paraxial_state,
)
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import evaluate_lens


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
    lens = GeoLens(filename=args.input_lens, device=device)
    _apply_mwir_constraints(lens, spec)
    surfaces = _curved_surfaces(lens)
    aspheres = _aspheric_surfaces(lens)
    if len(surfaces) != 14 or not aspheres:
        raise ValueError("L-BFGS 探针要求七片、14 个折射面和非球面自由度。")

    base_curvatures = torch.stack(
        [surface.c.detach().clone().to(device) for surface in surfaces]
    )
    base_state = paraxial_state(lens, base_curvatures)
    initial_focus_shift = _detached_float(lens.d_sensor) - float(
        base_state.focus_z_mm.detach().cpu()
    )
    focus_fraction = min(max(initial_focus_shift / args.focus_span_mm, -0.95), 0.95)
    shape_raw = torch.zeros_like(base_curvatures, requires_grad=True)
    focus_raw = torch.tensor(
        torch.atanh(torch.tensor(focus_fraction, dtype=base_curvatures.dtype)).item(),
        dtype=base_curvatures.dtype,
        device=device,
        requires_grad=True,
    )
    conic_raw, edge_raw, edge_spans = _initial_asphere_raw(
        lens, dtype=base_curvatures.dtype, device=device
    )
    parameters = [shape_raw, focus_raw, conic_raw, edge_raw]
    optimizer = torch.optim.LBFGS(
        parameters,
        lr=args.learning_rate,
        max_iter=args.inner_iterations,
        history_size=args.history_size,
        tolerance_grad=args.tolerance_grad,
        tolerance_change=args.tolerance_change,
        line_search_fn="strong_wolfe",
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

    def evaluate_train() -> torch.Tensor:
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
            focus_span_mm=args.focus_span_mm,
            minimum_valid_ratio=spec.vignetting_floor,
            relative_curvature_ratio=args.relative_curvature_ratio,
            focus_weight=args.focus_weight,
            astigmatism_weight=args.astigmatism_weight,
            chromatic_focus_weight=args.chromatic_focus_weight,
            field_curvature_weight=args.field_curvature_weight,
        )
        return result[0]

    for iteration in range(args.iterations + 1):
        if iteration < args.iterations:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = evaluate_train()
                if not torch.isfinite(loss):
                    raise FloatingPointError("L-BFGS 训练 merit 非有限。")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
                return loss

            optimizer.step(closure)

        with torch.no_grad():
            validation = _evaluate_aspheric_parameter_state(
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
                minimum_valid_ratio=spec.vignetting_floor,
                relative_curvature_ratio=args.relative_curvature_ratio,
                focus_weight=args.focus_weight,
                astigmatism_weight=args.astigmatism_weight,
                chromatic_focus_weight=args.chromatic_focus_weight,
                field_curvature_weight=args.field_curvature_weight,
            )
        diagnostics = dict(validation[1])
        diagnostics["iteration"] = iteration
        history.append(diagnostics)
        score = float(diagnostics["rms_mean_mm"] + 0.35 * diagnostics["rms_max_mm"])
        print(
            f"迭代 {iteration}/{args.iterations}: "
            f"RMS {diagnostics['rms_mean_mm']:.6f}/"
            f"{diagnostics['rms_max_mm']:.6f} mm; score={score:.6f}"
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "diagnostics": diagnostics,
                "shape_raw": shape_raw.detach().clone(),
                "focus_raw": focus_raw.detach().clone(),
                "conic_raw": conic_raw.detach().clone(),
                "edge_raw": edge_raw.detach().clone(),
                "curvatures": validation[2].detach().clone(),
                "sensor_z": validation[3].detach().clone(),
            }

    if best is None:
        raise RuntimeError("L-BFGS 探针没有生成结果。")
    _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    _assign_aspheres(lens, best["conic_raw"], best["edge_raw"], edge_spans)
    for surface in surfaces:
        surface.c = surface.c.detach().clone()
    for surface in aspheres:
        surface.k = surface.k.detach().clone()
        values = [
            getattr(surface, f"ai{order}").detach().clone()
            for order in (4, 6, 8, 10, 12, 14, 16)[: best["edge_raw"].shape[1]]
        ]
        surface.ai = torch.stack(values).detach().clone()
    lens.d_sensor = lens.d_sensor.detach().clone()
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)
    lens.write_lens_json(str(output / "lbfgs_probe_optimized.json"))
    with open(output / "lbfgs_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    with open(output / "lbfgs_best_state.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "score": best["score"],
                "diagnostics": best["diagnostics"],
                "curvatures_1_per_mm": best["curvatures"].cpu().tolist(),
                "sensor_z_mm": float(best["sensor_z"].cpu()),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    evaluate_lens(
        lens,
        spec,
        output,
        psf_spp=args.eval_spp,
        vignetting_grid=7,
        vignetting_rays=min(args.eval_spp, 256),
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MWIR 七片非球面 L-BFGS 探针")
    parser.add_argument("--input-lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument("--field-count", type=int, default=5)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--validation-spp", type=int, default=64)
    parser.add_argument("--eval-spp", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--relative-curvature-ratio", type=float, default=2.0)
    parser.add_argument("--focus-span-mm", type=float, default=30.0)
    parser.add_argument("--focus-weight", type=float, default=0.0)
    parser.add_argument("--astigmatism-weight", type=float, default=0.0)
    parser.add_argument("--chromatic-focus-weight", type=float, default=0.0)
    parser.add_argument("--field-curvature-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--tolerance-grad", type=float, default=1e-7)
    parser.add_argument("--tolerance-change", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=20260754)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
