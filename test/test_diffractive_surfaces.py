"""deeplens/optics/diffractive_surface/ 测试——Fresnel、Binary2、Pixel2D、Zernike、Grating 和 DiffractiveSurface 基类。"""

import pytest
import torch

from deeplens.diffractive_surface import (
    Binary2,
    DiffractedRotation,
    DiffractiveSurface,
    Fresnel,
    Grating,
    Pixel2D,
    Rank1,
    RotationallySymmetric,
    Zernike,
)


class TestFresnel:
    """测试 Fresnel DOE。"""

    def test_init(self):
        """Fresnel DOE 应使用正确属性初始化。"""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        assert doe.f0.item() == pytest.approx(50.0)
        assert doe.res == (100, 100)

    def test_phase_func_shape(self):
        """phase_func 返回具有 DOE 分辨率的张量。"""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_focal_length_property(self):
        """Fresnel 应具有可优化的 f0。"""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.f0.requires_grad


class TestBinary2:
    """测试 Binary2 DOE。"""

    def test_init(self):
        """Binary2 DOE 应完成初始化。"""
        doe = Binary2(d=0.0, res=100)
        assert doe.res == (100, 100)

    def test_phase_func_shape(self):
        """phase_func 返回具有 DOE 分辨率的张量。"""
        doe = Binary2(d=0.0, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_optimizer_params(self):
        """get_optimizer_params 返回 5 个参数组（alpha2-10）。"""
        doe = Binary2(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 5
        # 所有 alpha 均应需要梯度
        assert doe.alpha2.requires_grad
        assert doe.alpha10.requires_grad

    def test_surf_dict_preserves_is_square(self):
        """surf_dict 往返转换应保留 DOE 光圈形状标志。"""
        doe = Binary2(d=0.0, res=100, is_square=False)

        reloaded = Binary2.init_from_dict(doe.surf_dict())

        assert reloaded.is_square is False


class TestPixel2D:
    """测试 Pixel2D DOE。"""

    def test_init(self):
        """Pixel2D DOE 应使用相位图初始化。"""
        doe = Pixel2D(d=0.0, res=100)
        assert doe.phase_map.shape == (100, 100)

    def test_phase_func_matches_map(self):
        """phase_func 返回已存储的 phase_map。"""
        doe = Pixel2D(d=0.0, res=100)
        phase = doe.phase_func()
        assert torch.equal(phase, doe.phase_map)

    def test_optimizer_params(self):
        """get_optimizer_params 应为 phase_map 启用梯度。"""
        doe = Pixel2D(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.phase_map.requires_grad


class TestZernike:
    """测试 Zernike DOE。"""

    def test_init(self):
        """Zernike DOE 应使用 37 个系数初始化。"""
        doe = Zernike(d=0.0, res=100)
        assert doe.zernike_order == 37
        assert doe.z_coeff.shape == (37,)

    def test_phase_func_shape(self):
        """phase_func 返回具有 DOE 分辨率的张量。"""
        doe = Zernike(d=0.0, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_zero_coeffs_zero_phase(self):
        """全零 Zernike 系数应在各处产生零相位。"""
        doe = Zernike(d=0.0, z_coeff=torch.zeros(37), res=100)
        phase = doe.phase_func()
        assert phase.abs().max().item() < 1e-6

    def test_optimizer_params(self):
        """get_optimizer_params 应为 z_coeff 启用梯度。"""
        doe = Zernike(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.z_coeff.requires_grad


class TestGrating:
    """测试 Grating DOE。"""

    def test_init(self):
        """Grating DOE 应完成初始化。"""
        doe = Grating(d=0.0, res=100, alpha=1.0, theta=0.0)
        assert doe.alpha.item() == pytest.approx(1.0)

    def test_phase_func_shape(self):
        """phase_func 返回具有 DOE 分辨率的张量。"""
        doe = Grating(d=0.0, res=100, alpha=1.0)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_linear_gradient(self):
        """当 theta=0 时，相位应沿 y 线性变化。"""
        doe = Grating(d=0.0, res=100, alpha=10.0, theta=0.0)
        phase = doe.phase_func()
        # 沿一列，相位应线性增大或减小
        col_center = phase[:, 50]
        diffs = col_center[1:] - col_center[:-1]
        # 所有差值应近似相等（线性）
        assert diffs.std().item() < diffs.abs().mean().item() * 0.1 + 1e-6

    def test_optimizer_params(self):
        """get_optimizer_params 返回 2 个参数组（theta、alpha）。"""
        doe = Grating(d=0.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 2


class TestDiffractiveSurfaceBase:
    """测试 DiffractiveSurface 基类功能。"""

    def test_get_phase_map_wrapping(self):
        """get_phase_map0 将相位环绕到 [0, 2*pi]。"""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        pmap = doe.get_phase_map0()
        assert pmap.min().item() >= 0
        assert pmap.max().item() <= 2 * torch.pi + 0.01

    def test_get_phase_map_wavelength(self):
        """不同波长下的 get_phase_map 应缩放相位。"""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        pmap_design = doe.get_phase_map(0.55)
        pmap_other = doe.get_phase_map(0.45)
        # 不同波长的相位图应不同
        assert not torch.allclose(pmap_design, pmap_other)

    def test_forward_applies_phase(self):
        """forward() 应修改波的复光场。"""
        from deeplens.light import ComplexWave

        doe = Fresnel(d=0.0, f0=50.0, res=200, fab_ps=0.02)
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            wave = ComplexWave.plane_wave(
                phy_size=[4.0, 4.0], res=[200, 200], wvln=0.55, z=0.0
            )
            u_before = wave.u.clone()
            wave = doe.forward(wave)
        finally:
            torch.set_default_dtype(old_dtype)
        # 相位调制后的波场应不同
        assert not torch.allclose(wave.u, u_before)

    def test_loss_quantization(self):
        """loss_quantization 返回 >= 0 的标量。"""
        doe = Fresnel(d=0.0, f0=50.0, res=100)
        loss = doe.loss_quantization(bits=16)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_surf_dict_preserves_fab_ps_geometry(self):
        """非默认 fab_ps 下，surf_dict 往返转换应保留物理光圈。

        几何尺寸由 ``w = res * fab_ps`` 推导，因此 ``surf_dict()`` 必须输出
        ``fab_ps``；否则 ``init_from_dict()`` 会将其默认为 0.001，重新加载时光圈会
        在无提示的情况下缩小（4.096mm -> 1.024mm）。
        """
        doe = Fresnel(d=0.0, f0=50.0, res=1024, fab_ps=0.004)
        assert doe.w == pytest.approx(4.096)

        reloaded = Fresnel.init_from_dict(doe.surf_dict())
        assert reloaded.fab_ps == pytest.approx(doe.fab_ps)
        assert reloaded.w == pytest.approx(doe.w)
        assert reloaded.h == pytest.approx(doe.h)


class TestRank1:
    """测试 Rank1 DOE。"""

    def test_init(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        assert doe.res == (100, 100)
        assert doe.V.shape == (100, 1)
        assert doe.Q.shape == (100, 1)

    def test_phase_func_shape(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        phase = doe.phase_func()
        assert phase.shape == (100, 100)

    def test_height_is_low_rank(self):
        """sigmoid 前的高度 logits 应严格满足 rank == `rank`。"""
        doe = Rank1(d=0.0, rank=1, res=100)
        assert torch.linalg.matrix_rank(doe.V @ doe.Q.T) == 1
        doe3 = Rank1(d=0.0, rank=3, res=100)
        assert doe3.V.shape == (100, 3)
        assert torch.linalg.matrix_rank(doe3.V @ doe3.Q.T) == 3

    def test_optimizer_params(self):
        doe = Rank1(d=0.0, rank=1, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.V.requires_grad
        assert doe.Q.requires_grad


class TestDiffractedRotation:
    """测试 DiffractedRotation DOE。"""

    def test_init(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, num_wings=3, res=100)
        assert doe.res == (100, 100)
        assert doe.num_wings == 3
        assert doe.wvln0 == pytest.approx(0.66)  # 默认为 wvln_max

    def test_phase_func_shape(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, res=100)
        assert doe.phase_func().shape == (100, 100)

    def test_phase_is_anisotropic(self):
        """旋转 DOE 不具有转置对称性（不同于径向镜头）。

        ``fab_ps`` 足够大，使镜头 OPD 跨多个匹配波长发生环绕，因此逐角度闪耀结构
        会使相位图随角度变化。
        """
        doe = DiffractedRotation(d=0.0, f0=50.0, num_wings=3, res=128, fab_ps=0.02)
        phase = doe.phase_func()
        assert not torch.allclose(phase, phase.T, atol=1e-3)

    def test_optimizer_params(self):
        doe = DiffractedRotation(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.f0.requires_grad


class TestRotationallySymmetric:
    """测试 RotationallySymmetric DOE。"""

    def test_init(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        assert doe.res == (100, 100)
        assert doe.n_rings == 50
        assert doe.radial_phase.shape == (50,)

    def test_phase_func_shape(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        assert doe.phase_func().shape == (100, 100)

    def test_phase_is_radially_symmetric(self):
        """相位仅取决于半径 => 在方形网格上具有转置对称性。"""
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=128)
        phase = doe.phase_func()
        assert torch.allclose(phase, phase.T, atol=1e-4)

    def test_optimizer_params(self):
        doe = RotationallySymmetric(d=0.0, f0=50.0, res=100)
        params = doe.get_optimizer_params()
        assert len(params) == 1
        assert doe.radial_phase.requires_grad


class TestDiffractiveLensLoad:
    """新表面应通过 DiffractiveLens 从 JSON 加载并生成 PSF。"""

    def test_load_rank1(self, device_auto):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(
            filename="./datasets/lenses/diffraclens/rank1.json", device=device_auto
        )
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()

    def test_load_diffracted_rotation(self, device_auto):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(
            filename="./datasets/lenses/diffraclens/diffracted_rotation.json",
            device=device_auto,
        )
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32, wvln=0.55)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()

    def test_load_rotational_symmetric(self, device_auto):
        from deeplens import DiffractiveLens

        lens = DiffractiveLens(
            filename="./datasets/lenses/diffraclens/rotational_symmetric.json",
            device=device_auto,
        )
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=32)
        assert psf.shape == (32, 32)
        assert torch.isfinite(psf).all()
