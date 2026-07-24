"""失效或调用即崩溃入口点的回归测试（根因 E）。

每项测试都复现一个导致公共代码路径在首次调用时抛出异常的缺陷（必然崩溃，而非
边界情况）。
"""

import os

import pytest
import torch

from deeplens.geometric_surface import Cubic, Prism
from deeplens.phase_surface import Binary2Phase
from deeplens.surrogate.mlpconv import MLPConv

_CELLPHONE_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets/lenses/cellphone/cellphone68deg.json",
)


def test_prism_init_from_dict_binds_material_and_device():
    """Prism.init_from_dict 过去将位置参数传入错误位置，导致材料名称进入 device
    参数位，而浮点数进入 material 参数位 ->
    AttributeError: 'float' object has no attribute 'lower'。"""
    surf = Prism.init_from_dict(
        {"r": 5.0, "d": 10.0, "mirror_angle": 45.0, "mat2": "N-BK7"}
    )
    # 材料名称必须传给 mat2（而不是 device 参数位）。
    assert surf.mat2.get_name() == "n-bk7"
    assert str(surf.device) == "cpu"


def test_cubic_surf_dict_round_trips():
    """Cubic.surf_dict() 必须输出 init_from_dict 所需的 'b' 列表和 'mat2'；
    过去它只写入标量 b3/b5/b7，且不写入 mat2，因此重新加载时会抛出 KeyError('b')。"""
    surf = Cubic(r=2.0, d=5.0, b=[1e-3, 2e-4, 3e-5], mat2="air")
    sd = surf.surf_dict()
    # JSON 加载器将解析后的轴向位置写入 "d"（io.py）。
    sd["d"] = sd["(d)"]

    surf2 = Cubic.init_from_dict(sd)
    assert surf2.b3.item() == pytest.approx(surf.b3.item())
    assert surf2.b5.item() == pytest.approx(surf.b5.item())
    assert surf2.b7.item() == pytest.approx(surf.b7.item())
    assert surf2.mat2.get_name() == surf.mat2.get_name()


def test_geolens_get_optimizer_params_with_phase_surface(sample_cellphone_lens):
    """GeoLens 中的 Phase 表面过去会使 get_optimizer_params 崩溃并抛出
    IndexError: list index out of range（它在默认的 4 元素 lr 列表上索引 lrs[4]）。"""
    lens = sample_cellphone_lens
    d_last = float(lens.surfaces[-1].d.item())
    lens.surfaces.append(
        Binary2Phase(r=1.0, d=d_last + 1.0, order2=1.0, device=str(lens.device))
    )

    params = lens.get_optimizer_params()  # 默认的 4 元素 lrs
    assert len(params) > 0


@pytest.mark.parametrize("ks", [16, 32, 64])
def test_mlpconv_builds_for_small_kernels(ks):
    """MLPConv 过去仅在 `if ks > 32` 内定义 upsample_times，因此 ks <= 32 时
    会在构造阶段抛出 NameError。"""
    net = MLPConv(in_features=4, ks=ks, channels=3)
    out = net(torch.randn(2, 4))
    assert out.shape == (2, 3, ks, ks)


@pytest.fixture(scope="module")
def psfnet_lens():
    from deeplens.psfnetlens import PSFNetLens

    return PSFNetLens(lens_path=_CELLPHONE_JSON)


def test_psfnetlens_constructs_with_defaults(psfnet_lens):
    """PSFNetLens 的默认 model_name 过去为 'mlp_conv'（init_net 不处理该值），
    且默认 kernel_size=64 与 mlpconv ConvDecoder 不兼容（要求 128）。现在默认构造
    必须成功。"""
    from deeplens.surrogate.psfnet_mplconv import PSFNet_MLPConv

    assert isinstance(psfnet_lens.psfnet, PSFNet_MLPConv)
    assert psfnet_lens.kernel_size == 128


def test_psfnetlens_render_rgbd_shape(psfnet_lens):
    """render_rgbd 过去构建 psf_rgb/points2input 无法处理的四维 points 张量
    [1,H,W,3]，并向需要 [H,W,C,ks,ks] 的 splat_psf_per_pixel 传入
    [N,3,ks,ks] PSF；foc_dist 还是张量，导致 torch.full_like 失效。"""
    H = W = 8
    dev = psfnet_lens.device
    img = torch.rand(1, 3, H, W, device=dev)
    depth = torch.full((1, H, W), -5000.0, device=dev)
    foc_dist = torch.tensor([-2000.0], device=dev)

    out = psfnet_lens.render_rgbd(img, depth=depth, foc_dist=foc_dist, ks=16)
    assert out.shape == img.shape


def test_diffraclens_render_mono(sample_diffraclens):
    """render_mono 过去调用不存在的 self.psf_infinite -> AttributeError。"""
    lens = sample_diffraclens
    img = torch.ones(1, 1, 64, 64, dtype=torch.float32, device=lens.device)
    out = lens.render_mono(img, ks=31)
    assert out.shape == img.shape


def test_create_barrier_runs(sample_cellphone_lens, tmp_path):
    """create_barrier 过去解包返回 None 的 draw_layout() -> TypeError。"""
    import matplotlib

    matplotlib.use("Agg")
    out = str(tmp_path / "barrier.png")
    sample_cellphone_lens.create_barrier(filename=out)
    assert os.path.exists(out)
