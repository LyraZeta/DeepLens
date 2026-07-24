# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""统一 ``Lens`` API 的一致性测试。

每种镜头类型（``GeoLens``、``HybridLens``、``DiffractiveLens``、
``DefocusLens``、``PSFNetLens``）都必须公开*相同*的 PSF/渲染 API，使用户无需了解
具体镜头类型即可统一调用 ``psf()`` / ``psf_rgb()`` / ``render()``。

规范接口契约（在 ``Lens`` 上统一定义）：

    psf(points, wvln=None, ks=PSF_KS, **kwargs)   # wvln 位于第 2 位，ks 位于第 3 位
    psf_rgb(points, ks=PSF_KS, **kwargs)
    render(img_obj, depth=None, method=None, **kwargs)   # 统一默认值

不同工作模式专用的选项（``model``、``spp``、``recenter``、``psf_type``、
``upsample_factor``……）必须放在 ``**kwargs`` 中，不能出现在公共签名里。
"""

import inspect
import os

import pytest
import torch

from deeplens import (
    DefocusLens,
    DiffractiveLens,
    GeoLens,
    HybridLens,
    PSFNetLens,
)
from deeplens.lens import Lens
from deeplens.config import PSF_KS

LENS_CLASSES = [GeoLens, HybridLens, DiffractiveLens, DefocusLens, PSFNetLens]

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _public_params(func):
    """不含 ``self`` 的有序参数列表。"""
    return [p for n, p in inspect.signature(func).parameters.items() if n != "self"]


def _has_var_keyword(func):
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(func).parameters.values()
    )


# =============================================================================
# 签名一致性（快速、类级别——无需构造镜头）
# =============================================================================
@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_signature_is_canonical(cls):
    """psf() 必须为 psf(points, wvln=None, ks=PSF_KS, **kwargs)。"""
    sig = inspect.signature(cls.psf)
    names = [p.name for p in _public_params(cls.psf)]
    assert names[:3] == ["points", "wvln", "ks"], (
        f"{cls.__name__}.psf must start with (points, wvln, ks); got {names[:3]}"
    )
    assert sig.parameters["wvln"].default is None, (
        f"{cls.__name__}.psf wvln default must be None"
    )
    assert sig.parameters["ks"].default == PSF_KS, (
        f"{cls.__name__}.psf ks default must be PSF_KS, got {sig.parameters['ks'].default!r}"
    )
    assert _has_var_keyword(cls.psf), f"{cls.__name__}.psf must accept **kwargs"


@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_is_implemented_per_type(cls):
    """每个具体镜头都必须自行提供 psf()，而不能继承基类 stub。"""
    assert cls.psf is not Lens.psf, (
        f"{cls.__name__} does not implement psf(); it inherits Lens.psf (NotImplementedError)"
    )


@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_has_no_mutable_default(cls):
    """psf() 不得使用可变默认参数（如 points=[...]）。"""
    for p in inspect.signature(cls.psf).parameters.values():
        assert not isinstance(p.default, (list, dict, set)), (
            f"{cls.__name__}.psf has a mutable default for {p.name!r}: {p.default!r}"
        )


@pytest.mark.parametrize("cls", LENS_CLASSES, ids=lambda c: c.__name__)
def test_psf_rgb_signature_is_canonical(cls):
    """psf_rgb() 必须接受 **kwargs，并将 ks 默认为 PSF_KS（相同契约）。"""
    sig = inspect.signature(cls.psf_rgb)
    assert sig.parameters["ks"].default == PSF_KS, (
        f"{cls.__name__}.psf_rgb ks default must be PSF_KS, got {sig.parameters['ks'].default!r}"
    )
    assert _has_var_keyword(cls.psf_rgb), f"{cls.__name__}.psf_rgb must accept **kwargs"


def test_render_default_method_is_uniform():
    """各镜头类型的 render() 默认 'method' 不得不同。"""
    defaults = {
        cls.__name__: inspect.signature(cls.render).parameters["method"].default
        for cls in LENS_CLASSES
    }
    assert len(set(defaults.values())) == 1, (
        f"render() default 'method' diverges across lens types: {defaults}"
    )


# =============================================================================
# 行为一致性（使用计算成本较低的镜头类型实际计算 PSF）
# =============================================================================
@pytest.fixture(scope="module")
def defocus_lens():
    return DefocusLens(
        foclen=50.0, fnum=4.0, sensor_size=(8.0, 8.0), sensor_res=(512, 512)
    )


@pytest.fixture(scope="module")
def psfnet_lens():
    lens_path = os.path.join(
        _PROJECT_ROOT, "datasets/lenses/cellphone/cellphone68deg.json"
    )
    return PSFNetLens(lens_path=lens_path)


def test_defocus_psf_positional_wvln_then_ks(defocus_lens):
    """psf(points, 0.55, 32) -> wvln 是第 2 个位置参数，ks 是第 3 个 -> [32, 32]。"""
    pts = torch.tensor([0.0, 0.0, -1000.0])
    psf = defocus_lens.psf(pts, 0.55, 32)
    assert psf.shape == (32, 32)


def test_geolens_psf_positional_wvln_then_ks(sample_singlet_lens):
    """GeoLens psf 也必须遵循 (points, wvln, ks) 的位置顺序。"""
    pts = torch.tensor([0.0, 0.0, -10000.0])
    psf = sample_singlet_lens.psf(pts, 0.55, 32, spp=256)
    assert psf.shape == (32, 32)


def test_diffraclens_default_ks_is_psf_ks(sample_diffraclens):
    """省略 ks 时，DiffractiveLens 必须返回 PSF_KS x PSF_KS 核。"""
    psf = sample_diffraclens.psf(points=[0.0, 0.0, float("-inf")], upsample_factor=1)
    assert psf.shape == (PSF_KS, PSF_KS)


def test_psfnetlens_psf_returns_monochromatic(psfnet_lens):
    """PSFNetLens.psf() 必须存在并返回单色 [N, ks, ks] PSF。"""
    pts = torch.tensor([[0.0, 0.0, -1000.0], [0.3, 0.0, -1000.0]])
    psf = psfnet_lens.psf(pts, ks=32)
    assert psf.shape == (2, 32, 32)


def test_psfnetlens_psf_single_point(psfnet_lens):
    """单个 [3] 点必须折叠 batch 维度 -> [ks, ks]。"""
    pts = torch.tensor([0.0, 0.0, -1000.0])
    psf = psfnet_lens.psf(pts, ks=32)
    assert psf.shape == (32, 32)
