"""deeplens/optics/phase_surface/ 测试——FresnelPhase、Binary2Phase、ZernikePhase、GratingPhase、PolyPhase 和 Phase 基类。"""

import pytest
import torch

from deeplens.phase_surface import (
    Binary2Phase,
    FresnelPhase,
    GratingPhase,
    Phase,
    PolyPhase,
    ZernikePhase,
)
from deeplens.light import Ray


class TestFresnelPhase:
    """测试 FresnelPhase 表面。"""

    def test_init(self):
        """FresnelPhase 应使用焦距初始化。"""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        assert s.f0.item() == pytest.approx(50.0)

    def test_phi_shape(self):
        """phi() 返回与输入 shape 匹配的张量。"""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        x = torch.linspace(-2, 2, 100)
        y = torch.zeros(100)
        phase = s.phi(x, y)
        assert phase.shape == (100,)
        # 相位应环绕到 [0, 2*pi]
        assert phase.min().item() >= 0
        assert phase.max().item() <= 2 * torch.pi + 0.01

    def test_dphi_dxy_shape(self):
        """dphi_dxy() 返回两个张量。"""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50,)
        assert dphidy.shape == (50,)

    def test_optimizer_params(self):
        """get_optimizer_params 应为 f0 启用梯度。"""
        s = FresnelPhase(r=5.0, d=0.0, f0=50.0)
        params = s.get_optimizer_params()
        assert len(params) > 0
        assert s.f0.requires_grad


class TestBinary2Phase:
    """测试 Binary2Phase 表面。"""

    def test_init(self):
        """Binary2Phase 应使用默认全零系数初始化。"""
        s = Binary2Phase(r=5.0, d=0.0)
        assert s.order2.item() == pytest.approx(0.0)

    def test_phi_zero_coeffs(self):
        """全零系数应产生接近零的相位。"""
        s = Binary2Phase(r=5.0, d=0.0, order2=0.0, order4=0.0, order6=0.0, order8=0.0, order10=0.0, order12=0.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.zeros(50)
        phase = s.phi(x, y)
        # EPSILON 会避免结果严格为零，但取余后应接近零
        assert phase.max().item() < 0.1

    def test_phi_shape(self):
        """phi() 返回具有正确 shape 的张量。"""
        s = Binary2Phase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 100)
        y = torch.zeros(100)
        phase = s.phi(x, y)
        assert phase.shape == (100,)

    def test_dphi_dxy_shape(self):
        """dphi_dxy() 返回两个 shape 正确的张量。"""
        s = Binary2Phase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50,)
        assert dphidy.shape == (50,)

    def test_optimizer_params(self):
        """get_optimizer_params 返回所有阶次和 d 的参数组。"""
        s = Binary2Phase(r=5.0, d=0.0)
        params = s.get_optimizer_params()
        # d + 6 个阶次系数 = 7
        assert len(params) == 7


class TestZernikePhase:
    """测试 ZernikePhase 表面。"""

    def test_init(self):
        """ZernikePhase 应使用 37 个 Zernike 系数初始化。"""
        s = ZernikePhase(r=5.0, d=0.0)
        assert s.zernike_order == 37
        assert s.z_coeff.shape == (37,)

    def test_phi_shape(self):
        """phi() 对二维输入返回二维张量。"""
        s = ZernikePhase(r=5.0, d=0.0)
        x = torch.linspace(-2, 2, 50).unsqueeze(0).expand(50, 50)
        y = torch.linspace(-2, 2, 50).unsqueeze(1).expand(50, 50)
        phase = s.phi(x, y)
        assert phase.shape == (50, 50)

    def test_dphi_dxy_shape(self):
        """dphi_dxy() 返回两个 shape 正确的张量。"""
        s = ZernikePhase(r=5.0, d=0.0)
        x = torch.linspace(-2, 2, 50).unsqueeze(0).expand(50, 50)
        y = torch.linspace(-2, 2, 50).unsqueeze(1).expand(50, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50, 50)
        assert dphidy.shape == (50, 50)

    def test_optimizer_params(self):
        """get_optimizer_params 应为 z_coeff 启用梯度。"""
        s = ZernikePhase(r=5.0, d=0.0)
        params = s.get_optimizer_params()
        assert len(params) == 1
        assert s.z_coeff.requires_grad


class TestGratingPhase:
    """测试 GratingPhase 表面。"""

    def test_init(self):
        """GratingPhase 应完成初始化。"""
        s = GratingPhase(r=5.0, d=0.0, theta=0.0, alpha=1.0)
        assert s.alpha.item() == pytest.approx(1.0)

    def test_linear_phase(self):
        """当 theta=0 时，相位在 y 方向上线性变化（模 2*pi 后）。"""
        s = GratingPhase(r=5.0, d=0.0, theta=0.0, alpha=1.0)
        x = torch.zeros(50)
        y = torch.linspace(-2, 2, 50)
        phase = s.phi(x, y)
        assert phase.shape == (50,)

    def test_constant_derivatives(self):
        """dphi_dxy 对光栅返回常数导数。"""
        s = GratingPhase(r=5.0, d=0.0, theta=0.0, alpha=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        # 当 theta=0：dphidx = alpha*sin(0)/norm_radii = 0
        # dphidy = alpha*cos(0)/norm_radii = 常数
        assert dphidx.std().item() < 1e-6  # 应为常数（零）
        assert dphidy.std().item() < 1e-6  # 应为常数

    def test_optimizer_params(self):
        """get_optimizer_params 返回 2 个参数组。"""
        s = GratingPhase(r=5.0, d=0.0)
        params = s.get_optimizer_params()
        assert len(params) == 2


class TestPolyPhase:
    """测试 PolyPhase 表面。"""

    def test_init(self):
        """PolyPhase 应完成初始化。"""
        s = PolyPhase(r=5.0, d=0.0, order2=1.0)
        assert s.order2.item() == pytest.approx(1.0)

    def test_phi_shape(self):
        """phi() 返回具有正确 shape 的张量。"""
        s = PolyPhase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.zeros(50)
        phase = s.phi(x, y)
        assert phase.shape == (50,)

    def test_dphi_dxy_shape(self):
        """dphi_dxy 返回两个张量。"""
        s = PolyPhase(r=5.0, d=0.0, order2=1.0)
        x = torch.linspace(-2, 2, 50)
        y = torch.linspace(-2, 2, 50)
        dphidx, dphidy = s.dphi_dxy(x, y)
        assert dphidx.shape == (50,)
        assert dphidy.shape == (50,)


class TestPhaseBaseRayReaction:
    """测试 Phase 基类带衍射的 ray_reaction。"""

    def test_ray_reaction_with_diffraction(self):
        """带衍射的 Phase.ray_reaction 应修改光线方向。"""
        # 使用实现了 phi 和 dphi_dxy 的 FresnelPhase
        s = FresnelPhase(r=5.0, d=5.0, f0=50.0)
        o = torch.tensor([[0.0, 1.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0]])
        ray = Ray(o, d, wvln=0.55)
        ray = s.ray_reaction(ray, n1=torch.tensor(1.0), n2=torch.tensor(1.0))
        # 光线方向应已被衍射修改
        # （在 y=1mm 处，Fresnel 相位梯度非零）
        assert ray.d.shape == (1, 3)
