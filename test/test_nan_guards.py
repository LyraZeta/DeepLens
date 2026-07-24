"""根因 B 的回归测试：NaN / inf 梯度保护。

每项测试都输入退化但合法的数据（重复键、平行于平面的光线、非物理 Abbe 数、超出
圆锥曲面边界的点），并断言前向值和梯度保持有限。过去其中若干情形会产生
NaN/inf，在无提示的情况下破坏基于梯度的设计。
"""

import torch

from deeplens.utils import interp1d
from deeplens.light import Ray
from deeplens.material import Material
from deeplens.geometric_surface.qtype import QTypeFreeform


def test_interp1d_finite_grad_with_duplicate_keys():
    """重复的有序键会使 key_right - key_left == 0；NaN 不得通过失效的
    torch.where 分支反向传播。"""
    # 前导重复值：query 0.5 被限制到索引 1 -> key_left == key_right == 1.0
    # （分母为零），而 query 1.5 是正常插值。混合输入会强制执行除法块，因此在未修复
    # 的代码中，零分母产生的 NaN 会通过 torch.where 分支反向传播。
    key = torch.tensor([[1.0], [1.0], [2.0]])
    value = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    query = torch.tensor([[0.5], [1.5]])

    out = interp1d(query, key, value)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert torch.isfinite(value.grad).all()


def test_ray_prop_to_parallel_is_finite():
    """平行于目标平面的光线满足 d_z ~ 0；t 不得变为 inf/NaN。"""
    o = torch.tensor([[0.0, 0.0, 0.0]])
    d = torch.tensor([[1.0, 0.0, 0.0]])  # 位于平面内：d_z == 0
    ray = Ray(o, d, wvln=0.55)

    ray.prop_to(10.0)
    assert torch.isfinite(ray.o).all()


def test_optimizable_cauchy_finite_at_zero_abbe():
    """可优化的 Abbe 数趋近 0 时，不得使折射率增大到 inf。"""
    mat = Material("1.5/50.0")
    mat.dispersion = "optimizable"
    mat.n = torch.tensor(1.5)
    mat.V = torch.tensor(0.0, requires_grad=True)

    n = mat.ior(torch.tensor(0.55))
    assert torch.isfinite(torch.as_tensor(n)).all()


def test_qtype_sag_finite_beyond_conic_boundary():
    """超出圆锥曲面边界 (1+k)c^2 r^2 > 1 时，sqrt 参数会变为负值；通过 clamp
    （而非 + EPSILON）使 sag/导数保持有限。"""
    # 边界位于 r = 1/c = 1 mm
    surf = QTypeFreeform(r=5.0, d=0.0, c=1.0, k=0.0, qm=None, mat2="air")
    x = torch.tensor([2.0])  # r = 2 mm > 1 mm
    y = torch.tensor([0.0])

    assert torch.isfinite(surf._sag(x, y)).all()
    dx, dy = surf._dfdxy(x, y)
    assert torch.isfinite(dx).all() and torch.isfinite(dy).all()
