"""MWIR 七片透射系统的材料布局与光焦度外层搜索。

每个候选布局都会调用 :func:`mwir_material_seed.build_seed`，按该布局的
红外等效色散数重新求三组正/负净光焦度，并生成双面有功率的七片种子。
这与仅替换既有处方中的材料不同，能够避免材料折射率变化后系统总功率和
色差配平失效。默认只做快速排序；使用 ``--opt-iterations`` 可对前 N 名
种子继续运行球面连续优化。
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

import torch

from deeplens.utils import set_seed
from mwir_material_seed import _dispersion_number, _pair_power, build_seed
from mwir_power_bent7_optimize import (
    _ray_merit,
    _sample_fixed_rays,
    _curved_surfaces,
    optimize_spherical_seed,
)
from mwir_spec import MWIRDesignSpec, configure_utf8_console
from mwir_telescope_design import evaluate_lens


DEFAULT_PAIRS = (
    ("si", "mgf2"),
    ("ge", "mgf2"),
    ("znse", "caf2"),
    ("si", "caf2"),
    ("ge", "caf2"),
    ("krs5", "mgf2"),
    ("znse", "mgf2"),
    ("ge", "si"),
)
DEFAULT_POOL = ("si", "mgf2", "znse", "caf2", "ge", "krs5")


def _candidate_layouts(max_candidates: int, seed: int) -> list[tuple[str, ...]]:
    """生成去重的布局；三组顺序和第七片弱功率材料均参与搜索。"""

    layouts: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    # 先放覆盖范围较好的异质材料组合，再用固定随机种子补足数量。
    preferred = (
        (DEFAULT_PAIRS[1], DEFAULT_PAIRS[2], DEFAULT_PAIRS[0], "ge"),
        (DEFAULT_PAIRS[0], DEFAULT_PAIRS[2], DEFAULT_PAIRS[1], "ge"),
        (DEFAULT_PAIRS[4], DEFAULT_PAIRS[0], DEFAULT_PAIRS[2], "si"),
        (DEFAULT_PAIRS[2], DEFAULT_PAIRS[4], DEFAULT_PAIRS[0], "ge"),
        (DEFAULT_PAIRS[5], DEFAULT_PAIRS[0], DEFAULT_PAIRS[2], "ge"),
        (DEFAULT_PAIRS[6], DEFAULT_PAIRS[3], DEFAULT_PAIRS[1], "si"),
    )
    for p1, p2, p3, weak in preferred:
        layout = tuple(x for pair in (p1, p2, p3) for x in pair) + (weak,)
        if layout not in seen:
            layouts.append(layout)
            seen.add(layout)
    rng = random.Random(seed)
    all_pairs = list(itertools.product(DEFAULT_PAIRS, repeat=3))
    rng.shuffle(all_pairs)
    for p1, p2, p3 in all_pairs:
        weak = rng.choice(DEFAULT_POOL)
        layout = tuple(x for pair in (p1, p2, p3) for x in pair) + (weak,)
        if layout not in seen:
            layouts.append(layout)
            seen.add(layout)
        if len(layouts) >= max_candidates:
            break
    return layouts[:max_candidates]


def _quick_score(lens, spec: MWIRDesignSpec, *, spp: int, seed: int) -> dict[str, float]:
    """低采样几何 RMS 评分，避免对全部候选运行昂贵的完整验收。"""

    batches, chiefs, target = _sample_fixed_rays(
        lens, spec, field_count=3, spp=spp, seed=seed, pupil_scale=1.0
    )
    with torch.no_grad():
        _, diag = _ray_merit(
            lens,
            batches,
            chiefs,
            target,
            minimum_valid_ratio=spec.vignetting_floor,
        )
    return {
        "rms_mean_mm": float(diag["rms_mean_mm"]),
        "rms_max_mm": float(diag["rms_max_mm"]),
        "mapping_max_relative": float(diag["mapping_max_relative"]),
        "valid_ratio_min": float(diag["valid_ratio_min"]),
        "score": float(diag["rms_mean_mm"] + 0.35 * diag["rms_max_mm"]),
    }


def _power_seed_summary(spec: MWIRDesignSpec, layout: tuple[str, ...]) -> list[float]:
    """返回与 build_seed 一致的七片一阶净光焦度，便于审计候选。"""

    total = 1.0 / spec.effective_focal_length_mm
    weak = 0.03 * total
    wavelengths = tuple(float(value) for value in spec.wavelengths_um)
    first_color = -weak / _dispersion_number(layout[6], wavelengths)
    powers: list[float] = []
    for index, fraction in enumerate((0.42, 0.30, 0.25)):
        color = first_color if index == 0 else 0.0
        powers.extend(
            _pair_power(
                fraction * total,
                layout[2 * index],
                layout[2 * index + 1],
                wavelengths,
                color_sum=color,
            )
        )
    powers.append(weak)
    return [float(value) for value in powers]


def run(args: argparse.Namespace) -> Path:
    configure_utf8_console()
    set_seed(args.seed)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"输出目录必须为空：{output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    spec = MWIRDesignSpec(
        field_y_deg=args.field_y_deg,
        image_height_mm=args.image_height_mm,
        entrance_pupil_diameter_mm=args.entrance_pupil_mm,
    )
    layouts = _candidate_layouts(args.max_candidates, args.seed)
    ranked: list[dict] = []
    for index, layout in enumerate(layouts):
        try:
            lens = build_seed(spec, layout, device)
            quick = _quick_score(lens, spec, spp=args.quick_spp, seed=args.seed + index)
            ranked.append(
                {
                    "index": index,
                    "materials": list(layout),
                    "element_power_1_per_mm": _power_seed_summary(spec, layout),
                    **quick,
                }
            )
            lens.write_lens_json(str(output / f"candidate_{index:03d}_seed.json"))
            print(f"[{index + 1}/{len(layouts)}] {','.join(layout)} score={quick['score']:.6g}")
        except Exception as error:
            ranked.append({"index": index, "materials": list(layout), "error": str(error)})
            print(f"[{index + 1}/{len(layouts)}] {','.join(layout)} 失败：{error}")
    ranked.sort(key=lambda item: item.get("score", math.inf))
    for rank, item in enumerate(ranked):
        item["rank"] = rank + 1
    with open(output / "material_layout_ranking.json", "w", encoding="utf-8") as file:
        json.dump(ranked, file, ensure_ascii=False, indent=2)

    # 可选：只把前 top_k 名送入球面连续优化，避免全组合搜索的巨大成本。
    if args.opt_iterations > 0:
        for rank, item in enumerate(ranked[: args.top_k], start=1):
            if "score" not in item:
                continue
            layout = tuple(item["materials"])
            lens = build_seed(spec, layout, device)
            best, history = optimize_spherical_seed(
                lens,
                spec,
                iterations=args.opt_iterations,
                field_count=args.opt_field_count,
                spp=args.opt_spp,
                validation_spp=args.opt_validation_spp,
                learning_rate=args.learning_rate,
                focus_learning_rate=args.focus_learning_rate,
                focus_span_mm=args.focus_span_mm,
                minimum_valid_ratio=spec.vignetting_floor,
                ray_seed=args.seed + 1000 + rank,
                checkpoint_interval=max(1, args.opt_iterations // 5),
            )
            subdir = output / f"rank_{rank:02d}_{'_'.join(layout)}"
            subdir.mkdir(parents=True, exist_ok=True)
            lens.write_lens_json(str(subdir / "power_bent7_spherical_optimized.json"))
            with open(subdir / "best_spherical_state.json", "w", encoding="utf-8") as file:
                json.dump(best, file, ensure_ascii=False, indent=2, default=str)
            with open(subdir / "optimization_history.json", "w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2, default=str)
            metrics = evaluate_lens(
                lens,
                spec,
                subdir,
                psf_spp=args.eval_spp,
                vignetting_grid=5,
                vignetting_rays=min(args.eval_spp, 128),
            )
            item["optimized_result"] = str(subdir)
            item["optimized_metrics"] = {
                "pass": metrics.get("pass", {}),
                "mtf_min": min(
                    value["system_min_estimate"]
                    for wavelength in metrics.get("mtf", {}).values()
                    for value in wavelength.values()
                    if "system_min_estimate" in value
                ),
            }
    with open(output / "outer_search_metadata.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "spec": spec.geometry_report(),
                "candidate_count": len(layouts),
                "quick_spp": args.quick_spp,
                "opt_iterations": args.opt_iterations,
                "top_k": args.top_k,
                "note": "候选均由材料布局重新求解光焦度后生成；结果需独立高采样复验。",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MWIR 材料布局/光焦度外层搜索")
    parser.add_argument("--output", default="results/mwir-material-outer-search")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--quick-spp", type=int, default=24)
    parser.add_argument("--opt-iterations", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--opt-field-count", type=int, default=3)
    parser.add_argument("--opt-spp", type=int, default=24)
    parser.add_argument("--opt-validation-spp", type=int, default=48)
    parser.add_argument("--eval-spp", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--focus-learning-rate", type=float, default=0.015)
    parser.add_argument("--focus-span-mm", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--field-y-deg", type=float, default=9.6)
    parser.add_argument("--image-height-mm", type=float, default=47.1454)
    parser.add_argument("--entrance-pupil-mm", type=float, default=280.0)
    return parser


if __name__ == "__main__":
    run(_build_parser().parse_args())
