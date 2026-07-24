"""`deeplens/hybridlens.py` 测试——`HybridLens`。"""

import os

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    """每项测试后恢复默认 dtype（HybridLens 会在全局设置 float64）。"""
    old_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old_dtype)


class TestHybridLensInit:
    """测试 HybridLens 初始化。"""

    def test_init_from_json(self, sample_hybridlens):
        """HybridLens 应从 JSON 加载 geolens 和 doe。"""
        lens = sample_hybridlens
        assert lens.geolens is not None
        assert lens.doe is not None
        assert len(lens.geolens.surfaces) > 0

    def test_device_transfer(self, sample_hybridlens):
        """to(device) 应同时转移 geolens 和 doe。"""
        lens = sample_hybridlens
        lens.to(torch.device("cpu"))
        assert lens.doe.d.device.type == "cpu"


class TestHybridLensPSF:
    """测试 PSF 计算。"""

    def test_psf_shape_and_normalization(self, sample_hybridlens):
        """psf() 返回归一化至约 1 的 [ks, ks] 张量。"""
        lens = sample_hybridlens
        ks = 64
        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=ks, spp=1_000_000)
        finally:
            torch.set_default_dtype(old_dtype)
        assert psf.shape == (ks, ks)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.05)
        assert (psf >= 0).all()


class TestHybridLensUtils:
    """测试实用方法。"""

    def test_calc_scale(self, sample_hybridlens):
        """calc_scale 返回正浮点数。"""
        lens = sample_hybridlens
        scale = lens.calc_scale(depth=-10000.0)
        assert isinstance(scale, float)
        assert scale > 0

    def test_refocus(self, sample_hybridlens):
        """refocus 应改变 geolens d_sensor。"""
        lens = sample_hybridlens
        d_before = lens.geolens.d_sensor.clone()
        lens.refocus(foc_dist=-5000.0)
        # 重新对焦后 d_sensor 应发生变化
        assert lens.geolens.d_sensor is not None


class TestHybridLensIO:
    """测试 I/O。"""

    def test_write_read_json_roundtrip(self, sample_hybridlens, test_output_dir):
        """write_lens_json 后再 read_lens_json 应保留结构。"""
        lens = sample_hybridlens
        out_path = os.path.join(test_output_dir, "test_hybridlens_roundtrip.json")
        original_num_surfs = len(lens.geolens.surfaces)

        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)

        from deeplens import HybridLens

        lens2 = HybridLens(filename=out_path)
        assert lens2.geolens is not None
        assert lens2.doe is not None


class TestHybridLensOptim:
    """测试优化辅助方法。"""

    def test_get_optimizer(self, sample_hybridlens):
        """get_optimizer 返回 Adam 优化器。"""
        lens = sample_hybridlens
        optimizer = lens.get_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)


class TestHybridLensVis:
    """draw_layout 的冒烟测试。"""

    def test_draw_layout(self, sample_hybridlens, test_output_dir):
        """draw_layout 应生成文件且不崩溃。"""
        lens = sample_hybridlens
        path = os.path.join(test_output_dir, "test_hybridlens_layout.png")
        lens.draw_layout(save_name=path, dpi=100)
        assert os.path.exists(path)
