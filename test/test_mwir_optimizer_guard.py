"""MWIR 优化器回滚与缩步重试测试。"""

import pytest
import torch

from mwir_power_bent7_optimize import _safe_optimizer_step


def test_safe_optimizer_step_retries_same_adam_gradient_at_half_lr():
    """首步 merit 变差时，应恢复 Adam 状态并用默认半步长重试。"""

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0)
    parameter.grad = torch.tensor(-1.0)
    diagnostics = {}

    accepted = _safe_optimizer_step(
        optimizer,
        [parameter],
        pre_step_loss=0.09,
        post_step_loss_fn=lambda: (parameter - 0.3).square(),
        diagnostics=diagnostics,
    )

    assert accepted
    assert float(parameter.detach()) == pytest.approx(0.5, rel=1e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert float(optimizer.state[parameter]["step"]) == pytest.approx(1.0)
    assert diagnostics["accepted"] is True
    assert diagnostics["attempts"] == 2
    assert diagnostics["rejection_reason"] is None
    assert diagnostics["pre_step_loss"] == pytest.approx(0.09)
    assert diagnostics["post_step_loss"] == pytest.approx(0.04, rel=1e-5)
    assert [item["rejection_reason"] for item in diagnostics["attempt_history"]] == [
        "merit_increase",
        None,
    ]
    attempted_lrs = [
        item["learning_rates"][0] for item in diagnostics["attempt_history"]
    ]
    assert attempted_lrs == pytest.approx([1.0, 0.5])


def test_safe_optimizer_step_stops_after_two_smaller_retries():
    """三个步长都使 merit 变差时，应回到原参数并报告最终拒绝。"""

    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    parameter.grad = torch.tensor(-1.0)
    diagnostics = {}

    accepted = _safe_optimizer_step(
        optimizer,
        [parameter],
        pre_step_loss=0.1,
        post_step_loss_fn=lambda: 1.0,
        diagnostics=diagnostics,
    )

    assert not accepted
    assert float(parameter.detach()) == pytest.approx(0.0)
    assert parameter.grad is None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.125)
    assert diagnostics["accepted"] is False
    assert diagnostics["attempts"] == 3
    assert diagnostics["rejection_reason"] == "merit_increase"
    assert diagnostics["pre_step_loss"] == pytest.approx(0.1)
    assert diagnostics["post_step_loss"] == pytest.approx(1.0)
    attempted_lrs = [
        item["learning_rates"][0] for item in diagnostics["attempt_history"]
    ]
    assert attempted_lrs == pytest.approx([1.0, 0.5, 0.25])


def test_safe_optimizer_step_reports_nonfinite_gradient_without_retry():
    """不可用梯度不能重试，但仍应按默认倍率降低下一步学习率。"""

    parameter = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD([parameter], lr=0.2)
    parameter.grad = torch.tensor(float("inf"))
    diagnostics = {}

    accepted = _safe_optimizer_step(
        optimizer,
        [parameter],
        diagnostics=diagnostics,
    )

    assert not accepted
    assert float(parameter.detach()) == pytest.approx(2.0)
    assert parameter.grad is None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    assert diagnostics == {
        "accepted": False,
        "attempts": 0,
        "rejection_reason": "nonfinite_gradient",
        "pre_step_loss": None,
        "post_step_loss": None,
        "attempt_history": [],
    }
