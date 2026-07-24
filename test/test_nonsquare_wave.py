"""根因 A 的回归测试：非方形光场/网格。

波动光学代码在多处混淆了 W 轴和 H 轴。由于现有测试均使用方形网格，这些轴交换
问题此前无法显现。这里使用非方形网格（H != W），使转置或不匹配的填充/裁剪能够
暴露出来。
"""

import torch

from deeplens.light.wave import (
    AngularSpectrumMethod,
    BandLimitedASM,
    ComplexWave,
    FraunhoferDiffraction,
    FresnelDiffraction,
)

# 使用方形像素的非方形网格：i=0,1 时 phy_size[i]/res[i] 相等。
H, W = 60, 90
PHY = (4.0, 6.0)  # 4.0/60 == 6.0/90 == ps
RES = (H, W)
PS = PHY[0] / RES[0]


def _field():
    return torch.ones(1, 1, H, W, dtype=torch.complex64)


def test_gen_xy_grid_matches_field_shape_and_orientation():
    field = ComplexWave(res=RES, phy_size=PHY)
    assert field.u.shape[-2:] == (H, W)
    # x/y 必须与 [H, W] 光场对齐（过去输出会转置）。
    assert field.x.shape == (H, W)
    assert field.y.shape == (H, W)
    # x 沿宽度（列）变化，y 沿高度（行）变化。
    assert torch.allclose(field.x[0, :], field.x[-1, :])      # x 沿行向下保持不变
    assert not torch.allclose(field.x[:, 0], field.x[:, -1])  # x 沿列变化
    assert torch.allclose(field.y[:, 0], field.y[:, -1])      # y 沿列保持不变
    assert not torch.allclose(field.y[0, :], field.y[-1, :])  # y 沿行向下变化


def test_fraunhofer_handles_4d_nonsquare():
    u = _field()
    out = FraunhoferDiffraction(u, z=100.0, wvln=0.5, ps=PS)
    assert out.shape == u.shape  # 过去会因四维输入或非方形误裁剪而抛出异常


def test_fresnel_nonsquare_shape_preserved():
    u = _field()
    out = FresnelDiffraction(u, z=50.0, wvln=0.5, ps=PS)
    assert out.shape == u.shape  # 过去不匹配的填充/裁剪会改变 shape


def test_asm_nonsquare_shape_preserved():
    u = _field()
    assert AngularSpectrumMethod(u, z=5.0, wvln=0.5, ps=PS).shape == u.shape
    assert BandLimitedASM(u, z=5.0, wvln=0.5, ps=PS).shape == u.shape
