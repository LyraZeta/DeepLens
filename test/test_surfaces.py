"""
deeplens/optics/geometric_surface/ 测试——几何表面类。
"""

import pytest
import torch
import math

from deeplens.geometric_surface import Spheric, Aspheric, Aperture, Plane
from deeplens.geometric_surface.base import Surface
from deeplens.light import Ray


class TestSphericSurface:
    """测试 Spheric 表面类。"""

    def test_spheric_init(self, device_auto):
        """Spheric 表面应使用曲率初始化。"""
        surf = Spheric(
            c=0.1,  # 曲率 = 1/半径
            r=5.0,  # 光圈半径
            d=0.0,  # 与原点的距离
            mat2="bk7",
            device=device_auto,
        )
        
        assert surf.c.item() == pytest.approx(0.1)
        assert surf.r == 5.0

    def test_spheric_sag_center(self, device_auto):
        """中心处的 sag 应为零。"""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([0.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        z = surf.sag(x, y)
        
        assert torch.allclose(z, torch.tensor([0.0], device=device_auto), atol=1e-6)

    def test_spheric_sag_offaxis(self, device_auto):
        """sag 应随距光轴的距离增大。"""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x1 = torch.tensor([1.0], device=device_auto)
        x2 = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        z1 = surf.sag(x1, y)
        z2 = surf.sag(x2, y)
        
        # 对于正曲率，sag 随半径增大
        assert z2.abs() > z1.abs()

    def test_spheric_sag_symmetry(self, device_auto):
        """sag 应关于光轴对称。"""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x_pos = torch.tensor([2.0], device=device_auto)
        x_neg = torch.tensor([-2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        z_pos = surf.sag(x_pos, y)
        z_neg = surf.sag(x_neg, y)
        
        assert torch.allclose(z_pos, z_neg)

    def test_spheric_intersect(self, device_auto):
        """光线应与 Spheric 表面相交。"""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        # 使用负责处理坐标变换的 ray_reaction
        n1 = 1.0  # 空气
        n2 = surf.mat2.ior(torch.tensor([0.55], device=device_auto)).item()
        ray = surf.ray_reaction(ray, n1, n2)
        
        # 光线应在 z=10 附近击中表面
        assert ray.o[0, 2].item() > 9.0
        assert ray.o[0, 2].item() < 11.0

    def test_spheric_refract(self, device_auto):
        """光线应在 Spheric 表面发生折射。"""
        surf = Spheric(c=0.05, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        # 轴外光线
        o = torch.tensor([[1.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        original_d = ray.d.clone()
        
        # 获取折射率
        n1 = 1.0  # 空气
        n2 = surf.mat2.ior(torch.tensor([0.55], device=device_auto)).item()
        
        ray = surf.ray_reaction(ray, n1, n2)
        
        # 方向应因曲面折射而改变
        assert not torch.allclose(ray.d, original_d, atol=1e-3)

    def test_spheric_init_from_dict(self, device_auto):
        """Spheric 应从字典初始化。"""
        surf_dict = {
            "type": "Spheric",
            "c": 0.05,
            "r": 5.0,
            "d": 10.0,
            "mat2": "bk7",
        }
        
        surf = Spheric.init_from_dict(surf_dict)
        
        assert surf.c.item() == pytest.approx(0.05)
        assert surf.r == 5.0

    def test_spheric_surf_dict(self, device_auto):
        """Spheric 应导出为字典。"""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        d = surf.surf_dict()
        
        assert d["type"] == "Spheric"
        assert "(c)" in d or "c" in d
        assert d["r"] == 5.0


class TestAsphericSurface:
    """测试 Aspheric 表面类。"""

    def test_aspheric_init(self, device_auto):
        """Aspheric 表面应使用系数初始化。"""
        surf = Aspheric(
            c=0.1,
            k=0.0,  # 圆锥常数
            ai=[0.0] * 6,  # 高阶系数
            r=5.0,
            d=0.0,
            mat2="bk7",
            device=device_auto,
        )
        
        assert surf.c.item() == pytest.approx(0.1)
        assert len(surf.ai) == 6

    def test_aspheric_reduces_to_spheric(self, device_auto):
        """k=0 且 ai=0 的 Aspheric 应等同于 Spheric。"""
        c = 0.05
        r = 5.0
        
        asph = Aspheric(c=c, k=0.0, ai=[0.0]*6, r=r, d=0.0, mat2="bk7", device=device_auto)
        sph = Spheric(c=c, r=r, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([1.0, 2.0, 3.0], device=device_auto)
        y = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        
        z_asph = asph.sag(x, y)
        z_sph = sph.sag(x, y)
        
        assert torch.allclose(z_asph, z_sph, atol=1e-5)

    def test_aspheric_conic_parabola(self, device_auto):
        """k=-1 应给出抛物面 sag z = c*r^2 / 2。"""
        c = 0.1
        surf = Aspheric(c=c, k=-1.0, ai=[0.0]*6, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        r_sq = x**2 + y**2
        
        z = surf.sag(x, y)
        expected = c * r_sq / 2  # 抛物面公式
        
        assert torch.allclose(z, expected, atol=1e-5)

    def test_aspheric_higher_order(self, device_auto):
        """高阶系数应影响 sag。"""
        c = 0.0  # 无基础曲率
        ai = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]  # 仅 ai4

        surf = Aspheric(c=c, k=0.0, ai=ai, r=5.0, d=0.0, mat2="bk7", device=device_auto)

        x = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        r_sq = x**2 + y**2

        z = surf.sag(x, y)
        expected = ai[0] * r_sq**2  # ai4 * r^4

        assert torch.allclose(z, expected, atol=1e-5)

    def test_aspheric_init_from_dict(self, device_auto):
        """Aspheric 应从字典初始化。"""
        surf_dict = {
            "type": "Aspheric",
            "c": 0.05,
            "k": -0.5,
            "ai": [0.001, 0.0001, 0.0, 0.0, 0.0, 0.0],
            "r": 5.0,
            "d": 10.0,
            "mat2": "pmma",
        }
        
        surf = Aspheric.init_from_dict(surf_dict)
        
        assert surf.c.item() == pytest.approx(0.05)
        assert surf.k.item() == pytest.approx(-0.5)


class TestApertureSurface:
    """测试 Aperture 表面类。"""

    def test_aperture_init(self, device_auto):
        """Aperture 应使用半径初始化。"""
        aper = Aperture(r=2.0, d=5.0, device=device_auto)
        
        assert aper.r == 2.0
        assert aper.d.item() == pytest.approx(5.0)

    def test_aperture_clips_rays(self, device_auto):
        """Aperture 应使半径外的光线失效。"""
        aper = Aperture(r=2.0, d=5.0, device=device_auto)
        
        # 光圈内的光线
        o_in = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray_in = Ray(o_in, d, wvln=0.55, device=device_auto)
        
        # 光圈外的光线
        o_out = torch.tensor([[5.0, 0.0, 0.0]], device=device_auto)
        ray_out = Ray(o_out, d.clone(), wvln=0.55, device=device_auto)
        
        ray_in = aper.ray_reaction(ray_in)
        ray_out = aper.ray_reaction(ray_out)
        
        assert ray_in.is_valid[0].item() == 1.0
        assert ray_out.is_valid[0].item() == 0.0

    def test_aperture_surf_dict(self, device_auto):
        """Aperture 应导出为字典。"""
        aper = Aperture(r=2.0, d=5.0, device=device_auto)
        
        d = aper.surf_dict()
        
        assert d["type"] == "Aperture"
        assert d["r"] == 2.0


class TestPlaneSurface:
    """测试 Plane 表面类。"""

    def test_plane_init(self, device_auto):
        """Plane 表面应完成初始化。"""
        plane = Plane(r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        assert plane.r == 5.0
        assert plane.d.item() == pytest.approx(10.0)

    def test_plane_sag_zero(self, device_auto):
        """Plane sag 应处处为零。"""
        plane = Plane(r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device_auto)
        y = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device_auto)
        
        z = plane.sag(x, y)
        
        assert torch.allclose(z, torch.zeros_like(z))

    def test_plane_intersect(self, device_auto):
        """光线应在 z=d 处与平面相交。"""
        plane = Plane(r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        # 使用负责处理坐标变换的 ray_reaction
        n1 = 1.0
        n2 = plane.mat2.ior(torch.tensor([0.55], device=device_auto)).item()
        ray = plane.ray_reaction(ray, n1, n2)
        
        assert ray.o[0, 2].item() == pytest.approx(10.0, abs=0.1)


class TestSurfaceBase:
    """测试 Surface 基类功能。"""

    def test_surface_normal_vec(self, device_auto):
        """法向量应指向光源。"""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray = surf.intersect(ray)
        n = surf.normal_vec(ray)
        
        # 在中心处，法向量应指向 -z 方向（朝向光源）
        assert n[0, 2].item() < 0

    def test_surface_reflect(self, device_auto):
        """反射应遵循反射定律。"""
        surf = Spheric(c=0.0, r=5.0, d=10.0, mat2="bk7", device=device_auto)  # 平面镜
        
        # 45 度入射
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[1.0, 0.0, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray = surf.intersect(ray)
        ray = surf.reflect(ray)
        
        # 反射光线应沿 x、-z 方向传播
        assert ray.d[0, 0].item() > 0  # +x
        assert ray.d[0, 2].item() < 0  # -z

    def test_surface_local_coord_transform(self, device_auto):
        """局部坐标变换应可逆。"""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[1.0, 2.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.1, 0.2, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        original_o = ray.o.clone()
        original_d = ray.d.clone()
        
        ray = surf.to_local_coord(ray)
        ray = surf.to_global_coord(ray)
        
        assert torch.allclose(ray.o, original_o, atol=1e-5)
        assert torch.allclose(ray.d, original_d, atol=1e-5)


class TestSurfaceDerivatives:
    """测试表面导数计算。"""

    def test_spheric_dfdxy_center(self, device_auto):
        """中心处的导数应为零。"""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([0.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        dfdx, dfdy = surf._dfdxy(x, y)
        
        assert torch.allclose(dfdx, torch.tensor([0.0], device=device_auto), atol=1e-5)
        assert torch.allclose(dfdy, torch.tensor([0.0], device=device_auto), atol=1e-5)

    def test_spheric_dfdxy_symmetry(self, device_auto):
        """导数应具有适当的对称性。"""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        dfdx1, dfdy1 = surf._dfdxy(x, y)
        dfdx2, dfdy2 = surf._dfdxy(-x, y)
        
        # dfdx 应为反对称
        assert torch.allclose(dfdx1, -dfdx2, atol=1e-5)

    def test_aspheric_dfdxy(self, device_auto):
        """Aspheric 导数应与数值梯度一致。"""
        surf = Aspheric(c=0.05, k=-0.5, ai=[0.001]*6, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([1.5], device=device_auto, requires_grad=True)
        y = torch.tensor([0.5], device=device_auto, requires_grad=True)
        
        z = surf.sag(x, y)
        z.backward()
        
        dfdx_num = x.grad
        dfdy_num = y.grad
        
        x_detach = x.detach()
        y_detach = y.detach()
        dfdx_ana, dfdy_ana = surf._dfdxy(x_detach, y_detach)
        
        assert torch.allclose(dfdx_num, dfdx_ana, atol=1e-4)
        assert torch.allclose(dfdy_num, dfdy_ana, atol=1e-4)
