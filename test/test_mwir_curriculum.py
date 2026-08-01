"""MWIR 优化 curriculum 与安全默认步长测试。"""

import ast
import inspect
import textwrap

import pytest

from mwir_element_power_optimize import _parser as _element_parser
from mwir_element_power_optimize import run as run_element_power
from mwir_power_bent7_optimize import (
    _build_parser as _power_bent_parser,
    _curriculum_ray_weights,
    _curriculum_scale,
    optimize_structural_seed,
)


def _expanded_keyword_count(function, called_name: str, keyword_name: str) -> int:
    """统计函数内指定调用展开 ``**keyword_name`` 的次数。"""

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != called_name:
            continue
        if any(
            keyword.arg is None
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == keyword_name
            for keyword in node.keywords
        ):
            count += 1
    return count


def test_curriculum_scale_has_pure_rms_warmup_and_smooth_ramp():
    """默认分段应先保持零权重，再平滑渐入并留出完整权重阶段。"""

    scales = [
        _curriculum_scale(
            iteration,
            100,
            warmup_fraction=0.25,
            ramp_fraction=0.5,
        )
        for iteration in (0, 25, 50, 75, 100)
    ]

    assert scales == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])
    assert scales == sorted(scales)


def test_zero_length_curriculum_preserves_legacy_full_weights():
    """两个比例均为零时应从首步使用完整权重，便于复现旧实验。"""

    assert _curriculum_scale(
        0,
        100,
        warmup_fraction=0.0,
        ramp_fraction=0.0,
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("warmup", "ramp"),
    [(-0.1, 0.5), (0.25, float("nan")), (0.6, 0.5)],
)
def test_curriculum_scale_rejects_invalid_ranges(warmup, ramp):
    with pytest.raises(ValueError):
        _curriculum_scale(
            0,
            100,
            warmup_fraction=warmup,
            ramp_fraction=ramp,
        )


def test_curriculum_scales_all_auxiliary_ray_weights_together():
    weights = _curriculum_ray_weights(
        0.25,
        mtf_surrogate_weight=2.0,
        direct_mtf_weight=4.0,
        focus_weight=6.0,
        astigmatism_weight=8.0,
        chromatic_focus_weight=10.0,
        field_curvature_weight=12.0,
    )

    assert weights == pytest.approx(
        {
            "mtf_surrogate_weight": 0.5,
            "direct_mtf_weight": 1.0,
            "focus_weight": 1.5,
            "astigmatism_weight": 2.0,
            "chromatic_focus_weight": 2.5,
            "field_curvature_weight": 3.0,
        }
    )


def test_train_validation_and_post_step_expand_the_same_current_weights():
    """两个循环的三类 merit 评估都必须复用当步 curriculum 权重。"""

    assert _expanded_keyword_count(
        run_element_power,
        "_ray_merit",
        "current_ray_weights",
    ) == 3
    assert _expanded_keyword_count(
        optimize_structural_seed,
        "_evaluate_structural_parameter_state",
        "current_ray_weights",
    ) == 3


def test_optimizer_parsers_use_calibrated_safe_learning_rates():
    element_args = _element_parser().parse_args(
        ["--input-lens", "seed.json", "--output", "result"]
    )
    power_bent_args = _power_bent_parser().parse_args([])

    assert element_args.power_learning_rate == pytest.approx(3e-5)
    assert element_args.bend_learning_rate == pytest.approx(1e-4)
    assert element_args.focus_learning_rate == pytest.approx(1e-3)
    assert element_args.curriculum_warmup_fraction == pytest.approx(0.25)
    assert element_args.curriculum_ramp_fraction == pytest.approx(0.5)

    assert power_bent_args.learning_rate == pytest.approx(1e-4)
    assert power_bent_args.gap_learning_rate == pytest.approx(1e-4)
    assert power_bent_args.focus_learning_rate == pytest.approx(1e-3)
    assert power_bent_args.conic_learning_rate == pytest.approx(1e-4)
    assert power_bent_args.asphere_learning_rate == pytest.approx(1e-5)
    assert power_bent_args.curriculum_warmup_fraction == pytest.approx(0.25)
    assert power_bent_args.curriculum_ramp_fraction == pytest.approx(0.5)
