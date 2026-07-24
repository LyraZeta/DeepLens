"""
deeplens/optics/ray.py 测试——Ray 类操作。
"""

import pytest
import torch

from deeplens.light import Ray
from deeplens.config import DEFAULT_WAVE


class TestRayInit:
    """测试 Ray 初始化。"""

    def test_ray_init_basic(self, device_auto):
        """Ray 应使用原点和方向初始化。"""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert ray.o.shape == (1, 3)
        assert ray.d.shape == (1, 3)
        assert ray.wvln.shape == ()  # 零维标量张量

    def test_ray_init_batch(self, device_auto):
        """Ray 应支持批量初始化。"""
        batch_size = 100
        o = torch.zeros(batch_size, 3, device=device_auto)
        d = torch.zeros(batch_size, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert ray.o.shape == (batch_size, 3)
        assert ray.shape == (batch_size,)

    def test_ray_init_multidim(self, device_auto):
        """Ray 应支持多维批量数据。"""
        o = torch.zeros(5, 10, 3, device=device_auto)
        d = torch.zeros(5, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert ray.o.shape == (5, 10, 3)
        assert ray.shape == (5, 10)

    def test_ray_init_normalizes_direction(self, device_auto):
        """Ray 方向应归一化。"""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[3.0, 4.0, 0.0]], device=device_auto)  # 未归一化
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        norm = torch.norm(ray.d, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

    def test_ray_init_wavelength_validation(self, device_auto):
        """Ray 应验证波长以微米表示。"""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        
        # 有效波长（0.55 um = 550 nm）
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        assert torch.isclose(ray.wvln, torch.tensor(0.55, device=device_auto)).item()
        
        # 超出范围的波长应抛出异常
        with pytest.raises(AssertionError):
            Ray(o, d, wvln=550.0, device=device_auto)  # 使用 nm 而非 um

    def test_ray_init_valid_mask(self, device_auto):
        """Ray 应使用全有效掩码初始化。"""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert torch.all(ray.is_valid == 1.0)

    def test_ray_init_opl_zero(self, device_auto):
        """Ray 应使用零光程初始化。"""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert torch.all(ray.opl == 0.0)


class TestRayPropTo:
    """测试光线传播。"""

    def test_ray_prop_to_basic(self, device_auto):
        """光线应传播到 z 平面。"""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.prop_to(z=10.0)
        
        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto))

    def test_ray_prop_to_angled(self, device_auto):
        """光线应以一定角度正确传播。"""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        # xz 平面内 45 度角
        d = torch.tensor([[1.0, 0.0, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.prop_to(z=10.0)
        
        assert torch.allclose(ray.o[0, 0], torch.tensor(10.0, device=device_auto), atol=1e-5)
        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto), atol=1e-5)

    def test_ray_prop_to_backward(self, device_auto):
        """光线应能反向传播。"""
        o = torch.tensor([[0.0, 0.0, 10.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, -1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.prop_to(z=0.0)
        
        assert torch.allclose(ray.o[0, 2], torch.tensor(0.0, device=device_auto))

    def test_ray_prop_to_respects_valid(self, device_auto):
        """传播应遵循有效掩码。"""
        o = torch.zeros(2, 3, device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.is_valid[1] = 0.0  # 将第二条光线设为无效
        original_o = ray.o.clone()
        
        ray.prop_to(z=10.0)
        
        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto))
        assert torch.allclose(ray.o[1], original_o[1])  # 无效光线保持不变

    def test_ray_prop_to_coherent_opl(self, device_auto):
        """相干光线应在传播过程中跟踪 OPL。"""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto, dtype=torch.float64)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto, dtype=torch.float64)
        ray = Ray(o, d, wvln=0.55, is_coherent=True, device=device_auto)
        
        ray.prop_to(z=10.0, n=1.5)
        
        # OPL = n * 距离
        expected_opl = 1.5 * 10.0
        assert torch.allclose(ray.opl[0, 0], torch.tensor(expected_opl, device=device_auto, dtype=torch.float64))


class TestRayCentroid:
    """测试光线质心计算。"""

    def test_ray_centroid_single(self, device_auto):
        """单条光线的质心就是光线位置。"""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        centroid = ray.centroid()
        
        assert torch.allclose(centroid, o.squeeze(0))

    def test_ray_centroid_batch(self, device_auto):
        """质心应为有效光线的均值。"""
        o = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        centroid = ray.centroid()
        
        expected = torch.tensor([1.0, 2.0, 0.0], device=device_auto)
        assert torch.allclose(centroid, expected)

    def test_ray_centroid_respects_valid(self, device_auto):
        """质心应仅考虑有效光线。"""
        o = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.is_valid[1] = 0.0  # 将第二条光线设为无效
        
        centroid = ray.centroid()
        
        expected = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        assert torch.allclose(centroid, expected, atol=1e-5)


class TestRayRmsError:
    """测试 RMS 误差计算。"""

    def test_ray_rms_error_zero(self, device_auto):
        """重合光线的 RMS 误差应为零。"""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        rms = ray.rms_error()
        
        assert torch.allclose(rms, torch.tensor(0.0, device=device_auto), atol=1e-5)

    def test_ray_rms_error_nonzero(self, device_auto):
        """分散光线的 RMS 误差应为正值。"""
        # 形成半径为 1 的圆的光线
        n = 100
        theta = torch.linspace(0, 2 * 3.14159, n, device=device_auto)
        o = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros(n, device=device_auto)], dim=-1)
        d = torch.zeros(n, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        rms = ray.rms_error()
        
        # 单位圆的 RMS 应为 ~1
        assert rms > 0.9 and rms < 1.1

    def test_ray_rms_error_with_reference(self, device_auto):
        """RMS 误差应使用提供的参考中心。"""
        o = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        center_ref = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        rms = ray.rms_error(center_ref=center_ref)
        
        # 相对原点的 RMS：sqrt((1^2 + 3^2) / 2) = sqrt(5)
        expected = torch.sqrt(torch.tensor(5.0, device=device_auto))
        assert torch.allclose(rms, expected, atol=1e-4)


class TestRayClone:
    """测试光线 clone。"""

    def test_ray_clone_creates_copy(self, device_auto):
        """clone 应创建独立副本。"""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        cloned = ray.clone()
        
        # 修改原对象
        ray.o[0, 0] = 999.0
        
        # clone 应保持不变
        assert cloned.o[0, 0] != 999.0

    def test_ray_clone_to_cpu(self, device_auto):
        """clone 应允许指定设备。"""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        cloned = ray.clone(device="cpu")
        
        assert cloned.o.device == torch.device("cpu")

    def test_ray_clone_copies_all_tensor_attributes(self, device_auto):
        """clone 应复制所有张量属性且不共享存储。"""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, is_coherent=True, device=device_auto)

        cloned = ray.clone()

        for attr in ("o", "d", "wvln", "is_valid", "en", "bend_penalty", "opl"):
            src = getattr(ray, attr)
            dst = getattr(cloned, attr)
            assert torch.allclose(src, dst)
            assert src.data_ptr() != dst.data_ptr()

        assert cloned.is_coherent == ray.is_coherent
        assert cloned.device == ray.device
        assert cloned.shape == ray.shape


class TestRaySqueezeUnsqueeze:
    """测试维度操作。"""

    def test_ray_squeeze(self, device_auto):
        """squeeze 应移除单例维度。"""
        o = torch.zeros(1, 10, 3, device=device_auto)
        d = torch.zeros(1, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.squeeze(dim=0)
        
        assert ray.o.shape == (10, 3)
        assert ray.d.shape == (10, 3)

    def test_ray_unsqueeze(self, device_auto):
        """unsqueeze 应添加维度。"""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.unsqueeze(dim=0)
        
        assert ray.o.shape == (1, 10, 3)
        assert ray.d.shape == (1, 10, 3)

    def test_ray_squeeze_unsqueeze_roundtrip(self, device_auto):
        """先 squeeze 再 unsqueeze 应恢复 shape。"""
        o = torch.zeros(1, 10, 3, device=device_auto)
        d = torch.zeros(1, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        original_shape = ray.o.shape
        ray.squeeze(dim=0)
        ray.unsqueeze(dim=0)
        
        assert ray.o.shape == original_shape
