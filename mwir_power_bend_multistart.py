"""七片 MWIR 的 power+bend 多起点搜索。

该脚本专门处理“相对曲率倍率无法跨过弱曲率/零点”的局部拓扑问题。
每片透镜用净光焦度和弯曲量参数化，先用低采样筛选起点，再对前几名做
固定光线、质心 RMS 的 Adam 优化。输出仍是实验候选，必须经过独立高采样
验收后才能作为处方。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path

import torch

from deeplens import GeoLens
from deeplens.utils import set_logger, set_seed
from mwir_element_power_optimize import (
    _calibrate_efl,
    _curvatures_from_power,
    _element_power_state,
)
from mwir_power_bent7_optimize import (
    _apply_mwir_constraints,
    _assign_geometry,
    _clearance_penalty,
    _curved_surfaces,
    _ray_merit,
    _safe_optimizer_step,
    _sample_fixed_rays,
)
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import evaluate_lens


def _state_from_raw(
    lens,
    spec,
    base_powers,
    base_bends,
    power_span,
    bend_span,
    power_raw,
    bend_raw,
    focus_raw,
    focus_span_mm,
):
    powers = base_powers + power_span * torch.tanh(power_raw)
    bends = base_bends + bend_span * torch.tanh(bend_raw)
    raw_curvatures = _curvatures_from_power(lens, powers, bends)
    curvatures, first_order = _calibrate_efl(
        lens, raw_curvatures, spec.effective_focal_length_mm
    )
    sensor_z = first_order.focus_z_mm + focus_span_mm * torch.tanh(focus_raw)
    _assign_geometry(lens, curvatures, sensor_z)
    return powers, bends, curvatures, first_order, sensor_z


def _score(
    lens,
    batches,
    chiefs,
    target,
    spec,
    curvatures,
):
    ray_loss, diagnostics = _ray_merit(
        lens,
        batches,
        chiefs,
        target,
        minimum_valid_ratio=spec.vignetting_floor,
    )
    clear_loss, clear = _clearance_penalty(lens, curvatures)
    diagnostics = dict(diagnostics)
    diagnostics.update(clear)
    diagnostics["loss"] = float((ray_loss + 2.0 * clear_loss).detach().cpu())
    diagnostics["score"] = float(
        diagnostics["rms_mean_mm"] + 0.35 * diagnostics["rms_max_mm"]
    )
    return ray_loss + 2.0 * clear_loss, diagnostics


def _topology_raw(base, target, span):
    ratio = ((target - base) / span).clamp(-0.98, 0.98)
    return torch.atanh(ratio)


def run(args: argparse.Namespace) -> Path:
    configure_utf8_console()
    set_seed(args.seed)
    random.seed(args.seed)
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
    if len(surfaces) != 14:
        raise ValueError("多起点 power+bend 搜索要求七片、14 个折射面。")
    base_curvatures = torch.stack(
        [surface.c.detach().clone().to(device) for surface in surfaces]
    )
    base_powers, base_bends = _element_power_state(lens, base_curvatures)
    base_state = _calibrate_efl(
        lens, base_curvatures, spec.effective_focal_length_mm
    )[1]
    initial_focus_shift = float(
        (lens.d_sensor - base_state.focus_z_mm).detach().cpu()
    )
    focus_fraction = max(
        -0.95, min(0.95, initial_focus_shift / args.focus_span_mm)
    )
    base_focus_raw = torch.tensor(
        math.atanh(focus_fraction), dtype=base_curvatures.dtype, device=device
    )
    power_span = torch.maximum(
        base_powers.abs() * args.power_span_factor,
        torch.full_like(base_powers, args.minimum_power_span),
    )
    bend_span = torch.maximum(
        base_bends.abs() * args.bend_span_factor,
        torch.full_like(base_bends, args.minimum_bend_span),
    )
    rank_batches, rank_chiefs, rank_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=args.ranking_field_count,
        spp=args.ranking_spp,
        seed=args.seed + 101,
        pupil_scale=1.0,
    )
    train_batches, train_chiefs, train_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=args.field_count,
        spp=args.spp,
        seed=args.seed + 1001,
        pupil_scale=1.0,
    )
    val_batches, val_chiefs, val_target = _sample_fixed_rays(
        lens,
        spec,
        field_count=max(args.field_count, 7),
        spp=args.validation_spp,
        seed=args.seed + 10_001,
        pupil_scale=1.0,
    )

    starts: list[dict] = [
        {
            "name": "baseline",
            "power_raw": torch.zeros_like(base_powers),
            "bend_raw": torch.zeros_like(base_bends),
            "focus_raw": base_focus_raw.clone(),
        }
    ]
    for index in range(max(0, args.random_starts - 1)):
        starts.append(
            {
                "name": f"random_{index:03d}",
                "power_raw": torch.randn_like(base_powers)
                * args.initial_power_noise,
                "bend_raw": torch.randn_like(base_bends)
                * args.initial_bend_noise,
                "focus_raw": base_focus_raw
                + torch.randn((), device=device, dtype=base_focus_raw.dtype)
                * args.initial_focus_noise,
            }
        )
    if args.include_topology_seeds:
        patterns = (
            (1, -1, 1, -1, 1, -1, 1),
            (-1, 1, -1, -1, -1, 1, -1),
            (1, 1, -1, -1, 1, -1, 1),
        )
        for index, signs in enumerate(patterns):
            target_powers = base_powers.abs() * torch.tensor(
                signs, dtype=base_powers.dtype, device=device
            )
            starts.append(
                {
                    "name": f"topology_{index:02d}",
                    "power_raw": _topology_raw(
                        base_powers, target_powers, power_span
                    ),
                    "bend_raw": torch.zeros_like(base_bends),
                    "focus_raw": base_focus_raw.clone(),
                }
            )

    ranking: list[dict] = []
    with torch.no_grad():
        for start in starts:
            _, _, curvatures, _, sensor_z = _state_from_raw(
                lens,
                spec,
                base_powers,
                base_bends,
                power_span,
                bend_span,
                start["power_raw"],
                start["bend_raw"],
                start["focus_raw"],
                args.focus_span_mm,
            )
            try:
                _, diag = _score(
                    lens,
                    rank_batches,
                    rank_chiefs,
                    rank_target,
                    spec,
                    curvatures,
                )
                item = {
                    "name": start["name"],
                    "initial_score": diag["score"],
                    "initial_diagnostics": diag,
                    "sensor_z_mm": float(sensor_z.detach().cpu()),
                }
            except Exception as error:
                item = {
                    "name": start["name"],
                    "initial_score": float("inf"),
                    "error": str(error),
                }
            ranking.append(item)
    ranking.sort(key=lambda item: item.get("initial_score", float("inf")))
    selected_names = {
        item["name"] for item in ranking[: max(1, args.top_starts)]
    }

    optimized: list[dict] = []
    for start in starts:
        if start["name"] not in selected_names:
            continue
        power_raw = start["power_raw"].detach().clone().requires_grad_(True)
        bend_raw = start["bend_raw"].detach().clone().requires_grad_(True)
        focus_raw = start["focus_raw"].detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                {"params": [power_raw], "lr": args.power_learning_rate},
                {"params": [bend_raw], "lr": args.bend_learning_rate},
                {"params": [focus_raw], "lr": args.focus_learning_rate},
            ]
        )
        best: dict | None = None
        history: list[dict] = []
        for iteration in range(args.iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            powers, bends, curvatures, first_order, sensor_z = _state_from_raw(
                lens,
                spec,
                base_powers,
                base_bends,
                power_span,
                bend_span,
                power_raw,
                bend_raw,
                focus_raw,
                args.focus_span_mm,
            )
            train_loss, train_diag = _score(
                lens,
                train_batches,
                train_chiefs,
                train_target,
                spec,
                curvatures,
            )
            with torch.no_grad():
                _, val_curvatures, _, val_sensor = (  # type: ignore[misc]
                    powers,
                    curvatures,
                    first_order,
                    sensor_z,
                )
                _, val_diag = _score(
                    lens,
                    val_batches,
                    val_chiefs,
                    val_target,
                    spec,
                    val_curvatures,
                )
            val_diag = dict(val_diag)
            val_diag["iteration"] = iteration
            val_diag["name"] = start["name"]
            history.append(val_diag)
            score = val_diag["score"]
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "diagnostics": val_diag,
                    "powers": powers.detach().clone(),
                    "bends": bends.detach().clone(),
                    "curvatures": curvatures.detach().clone(),
                    "sensor_z": sensor_z.detach().clone(),
                }
            if iteration % max(1, args.checkpoint_interval) == 0:
                logging.info(
                    "%s 迭代 %d/%d：验证 RMS %.6f/%.6f mm；score %.6f。",
                    start["name"],
                    iteration,
                    args.iterations,
                    val_diag["rms_mean_mm"],
                    val_diag["rms_max_mm"],
                    score,
                )
            if iteration == args.iterations:
                break
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [power_raw, bend_raw, focus_raw], args.max_grad_norm
            )

            def post_step_merit():
                _, _, post_curvatures, _, _ = _state_from_raw(
                    lens,
                    spec,
                    base_powers,
                    base_bends,
                    power_span,
                    bend_span,
                    power_raw,
                    bend_raw,
                    focus_raw,
                    args.focus_span_mm,
                )
                post_loss, _ = _score(
                    lens,
                    train_batches,
                    train_chiefs,
                    train_target,
                    spec,
                    post_curvatures,
                )
                return post_loss

            step_ok = _safe_optimizer_step(
                optimizer,
                [power_raw, bend_raw, focus_raw],
                pre_step_loss=float(train_loss.detach().cpu()),
                post_step_loss_fn=post_step_merit,
            )
            if not step_ok:
                logging.warning(
                    "%s 第 %d 步被 guard 拒绝。",
                    start["name"],
                    iteration,
                )
        if best is not None:
            best["name"] = start["name"]
            best["history"] = history
            optimized.append(best)

    if not optimized:
        raise RuntimeError("多起点搜索没有生成有效优化结果。")
    optimized.sort(key=lambda item: item["score"])
    best = optimized[0]
    _assign_geometry(lens, best["curvatures"], best["sensor_z"])
    lens.post_computation()
    _apply_mwir_constraints(lens, spec)
    lens.write_lens_json(str(output / "power_bend_multistart_optimized.json"))
    for item in optimized:
        item.pop("history", None)
        for key in ("powers", "bends", "curvatures", "sensor_z"):
            value = item.pop(key, None)
            if torch.is_tensor(value):
                item[key] = value.detach().cpu().tolist()
    with open(output / "start_ranking.json", "w", encoding="utf-8") as file:
        json.dump(
            {"initial_ranking": ranking, "optimized": optimized},
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
    parser = argparse.ArgumentParser(description="MWIR 七片 power+bend 多起点搜索")
    parser.add_argument("--input-lens", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--random-starts", type=int, default=4)
    parser.add_argument("--top-starts", type=int, default=2)
    parser.add_argument("--include-topology-seeds", action="store_true")
    parser.add_argument("--initial-power-noise", type=float, default=0.25)
    parser.add_argument("--initial-bend-noise", type=float, default=0.25)
    parser.add_argument("--initial-focus-noise", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--field-count", type=int, default=3)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--validation-spp", type=int, default=32)
    parser.add_argument("--ranking-field-count", type=int, default=3)
    parser.add_argument("--ranking-spp", type=int, default=12)
    parser.add_argument("--eval-spp", type=int, default=256)
    parser.add_argument("--power-learning-rate", type=float, default=3e-5)
    parser.add_argument("--bend-learning-rate", type=float, default=1e-4)
    parser.add_argument("--focus-learning-rate", type=float, default=1e-3)
    parser.add_argument("--power-span-factor", type=float, default=3.0)
    parser.add_argument("--bend-span-factor", type=float, default=3.0)
    parser.add_argument("--minimum-power-span", type=float, default=0.001)
    parser.add_argument("--minimum-bend-span", type=float, default=0.0005)
    parser.add_argument("--focus-span-mm", type=float, default=20.0)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260768)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
