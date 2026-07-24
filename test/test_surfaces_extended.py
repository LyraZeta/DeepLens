"""测试 test_surfaces.py 未覆盖的几何表面。

覆盖：Cubic、Mirror、ThinLens、QTypeFreeform、Spiral。
"""

import pytest
import torch

from deeplens.geometric_surface import (
    Cubic,
    Mirror,
    QTypeFreeform,
    Spiral,
    ThinLens,
)
from deeplens.light import Ray


class TestCubic:
    """测试 Cubic 表面。"""

    def test_init(self):
        """Cubic 可仅使用 b3 初始化。"""
        s = Cubic(r=5.0, d=0.0, b=[0.01], mat2="bk7")
        assert s.b_degree == 1

    def test_sag_center_zero(self):
        """中心处的 sag 应为零。"""
        s = Cubic(r=5.0, d=0.0, b=[0.01], mat2="bk7")
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        assert z.abs().item() < 1e-6

    def test_sag_nonzero_off_center(self):
        """中心外的 sag 应非零。"""
        s = Cubic(r=5.0, d=0.0, b=[0.1], mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        assert z.abs().item() > 0

    def test_derivatives(self):
        """dfdxyz 返回三个张量（dx、dy、dz）。"""
        s = Cubic(r=5.0, d=0.0, b=[0.1, 0.01], mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)
        sx, sy, sz = s.dfdxyz(x, y)
        assert isinstance(sx, torch.Tensor)
        assert isinstance(sy, torch.Tensor)
        assert isinstance(sz, torch.Tensor)

    def test_surf_dict(self):
        """surf_dict 返回正确类型（要求 3 个 b 项）。"""
        # surf_dict 引用 b3、b5、b7，因此至少需要 3 项
        s = Cubic(r=5.0, d=0.0, b=[0.01, 0.001, 0.0001], mat2="bk7")
        d = s.surf_dict()
        assert d["type"] == "Cubic"


class TestMirror:
    """测试 Mirror 表面。"""

    def test_init(self):
        """Mirror 应能初始化。"""
        m = Mirror(r=10.0, d=0.0)
        assert m.r == 10.0

    def test_ray_reaction_reflects(self):
        """ray_reaction 应反射光线（dz 符号翻转）。"""
        m = Mirror(r=10.0, d=5.0)
        o = torch.tensor([[0.0, 0.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0]])
        ray = Ray(o, d, wvln=0.55)
        ray = m.ray_reaction(ray, n1=1.0, n2=1.0)
        # 经平面镜反射后，z 方向应翻转
        assert ray.d[0, 2].item() < 0

    def test_surf_dict(self):
        """surf_dict 返回正确类型。"""
        m = Mirror(r=10.0, d=0.0)
        d = m.surf_dict()
        assert d["type"] == "Mirror"


class TestThinLens:
    """测试 ThinLens 表面。"""

    def test_init(self):
        """ThinLens 可使用焦距初始化。"""
        tl = ThinLens(r=5.0, d=0.0, f=50.0)
        assert tl.f.item() == pytest.approx(50.0)

    def test_refract_converges(self):
        """薄透镜折射应使平行光线向光轴弯折。"""
        tl = ThinLens(r=10.0, d=0.0, f=50.0)
        # 高度 1mm 处平行于光轴的光线
        o = torch.tensor([[1.0, 0.0, 0.0]])
        d = torch.tensor([[0.0, 0.0, 1.0]])
        ray = Ray(o, d, wvln=0.55)
        ray = tl.ray_reaction(ray, n1=1.0, n2=1.0)
        # 经过薄透镜后，光线应指向光轴
        # dx 应为负值（会聚）
        assert ray.d[0, 0].item() < 0

    def test_sag_is_zero(self):
        """ThinLens sag 应始终为零（平坦）。"""
        tl = ThinLens(r=5.0, d=0.0, f=50.0)
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)
        z = tl.sag(x, y)
        assert z.abs().item() < 1e-10

    def test_surf_dict(self):
        """surf_dict 返回正确类型。"""
        tl = ThinLens(r=5.0, d=0.0, f=50.0)
        d = tl.surf_dict()
        assert d["type"] == "ThinLens"
        assert d["f"] == pytest.approx(50.0)


class TestQTypeFreeform:
    """测试 QTypeFreeform 表面。"""

    def test_init(self):
        """QTypeFreeform 应能初始化。"""
        s = QTypeFreeform(r=5.0, d=0.0, c=0.1, k=0.0, qm=[0.001, 0.0001], mat2="bk7")
        assert s.n_qterms == 2

    def test_sag_center_zero(self):
        """中心处的 sag 应为零。"""
        s = QTypeFreeform(r=5.0, d=0.0, c=0.1, k=0.0, qm=[0.001], mat2="bk7")
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        assert z.abs().item() < 1e-6

    def test_reduces_to_conic_when_qm_zero(self):
        """当 qm=0 时，应与圆锥曲面 sag 匹配。"""
        c = 0.05
        k = -1.0
        s = QTypeFreeform(r=5.0, d=0.0, c=c, k=k, qm=[0.0, 0.0], mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(0.0)
        sag = s.sag(x, y)
        # 圆锥曲面：c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2))
        r2 = 1.0
        expected = c * r2 / (1 + (1 - (1 + k) * c**2 * r2) ** 0.5)
        assert sag.item() == pytest.approx(expected, abs=1e-3)

    def test_surf_dict_roundtrip(self):
        """surf_dict 返回包含 Q 系数的字典。"""
        s = QTypeFreeform(r=5.0, d=0.0, c=0.1, k=0.0, qm=[0.001, 0.0001], mat2="bk7")
        d = s.surf_dict()
        assert d["type"] == "QTypeFreeform"
        assert len(d["qm"]) == 2


class TestSpiral:
    """测试 Spiral 表面。"""

    def test_init(self):
        """Spiral 应能初始化。"""
        s = Spiral(r=5.0, d=0.0, c1=0.1, c2=0.05, mat2="bk7")
        assert s.r == 5.0

    def test_sag_nonzero(self):
        """c1、c2 非零时，sag 应非零。"""
        s = Spiral(r=5.0, d=0.0, c1=0.1, c2=0.05, mat2="bk7")
        x = torch.tensor(1.0)
        y = torch.tensor(1.0)
        z = s.sag(x, y)
        assert isinstance(z, torch.Tensor)
        assert z.abs().item() > 0

    def test_sag_at_origin(self):
        """在原点处，theta=0、phi_norm=0，因此 cos(0)=1。"""
        s = Spiral(r=5.0, d=0.0, c1=0.2, c2=0.1, mat2="bk7")
        x = torch.tensor(0.0)
        y = torch.tensor(0.0)
        z = s.sag(x, y)
        # cos(0) = 1，因此 z = c1/2*(1+1) + c2/2*(1-1) = c1 = 0.2
        assert z.item() == pytest.approx(0.2, abs=1e-4)
