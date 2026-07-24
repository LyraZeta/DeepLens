"""
deeplens 核心工具测试——init_device、光学配置常量和 DeepObj。
"""

import pytest
import torch


from deeplens import init_device
from deeplens.base import DeepObj
from deeplens.config import (
    DEPTH,
    DEFAULT_WAVE,
    EPSILON,
    PSF_KS,
    SPP_PSF,
    WAVE_RGB,
)

class TestConstants:
    """测试默认常量是否正确定义。"""

    def test_depth_constant(self):
        """DEPTH 应为表示无穷远的较大负值。"""
        assert DEPTH == -20000.0
        assert DEPTH < 0

    def test_wave_rgb(self):
        """WAVE_RGB 应包含以微米表示的 R、G、B 波长。"""
        assert len(WAVE_RGB) == 3
        assert WAVE_RGB[0] > WAVE_RGB[1] > WAVE_RGB[2]  # R > G > B
        # 所有波长均应位于可见光范围（0.38 - 0.78 um）内
        for wvln in WAVE_RGB:
            assert 0.38 < wvln < 0.78

    def test_default_wave(self):
        """DEFAULT_WAVE 应为绿光波长。"""
        assert 0.5 < DEFAULT_WAVE < 0.6  # 绿光

    def test_spp_psf(self):
        """SPP_PSF 应为 2 的幂。"""
        assert SPP_PSF > 0
        assert (SPP_PSF & (SPP_PSF - 1)) == 0  # 检查是否为 2 的幂

    def test_psf_ks(self):
        """PSF_KS 应为合理的核尺寸。"""
        assert PSF_KS > 0
        assert PSF_KS < 256

    def test_epsilon(self):
        """EPSILON 应为较小的正值。"""
        assert EPSILON > 0
        assert EPSILON < 1e-6


class TestInitDevice:
    """测试设备初始化。"""

    def test_init_device_returns_device(self):
        """init_device 应返回 torch 设备。"""
        device = init_device()
        assert isinstance(device, torch.device)

    def test_init_device_cuda_mps_or_cpu(self):
        """init_device 应返回 cuda、mps 或 cpu。"""
        device = init_device()
        assert device.type in ["cuda", "mps", "cpu"]

    def test_init_device_matches_availability(self):
        """init_device 应在 CUDA 可用时选择 CUDA，否则选择 CPU。

        此处有意不自动选择 MPS：它无法容纳 DeepLens 波动光学使用的 float64 张量，
        因此 Apple Silicon 会回退到 CPU。
        """
        device = init_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"


class TestDeepObj:
    """测试 DeepObj 基类功能。"""

    def test_deep_obj_init(self):
        """DeepObj 应使用默认 dtype 初始化。"""
        obj = DeepObj()
        assert obj.dtype == torch.get_default_dtype()

    def test_deep_obj_init_custom_dtype(self):
        """DeepObj 应接受自定义 dtype。"""
        obj = DeepObj(dtype=torch.float64)
        assert obj.dtype == torch.float64

    def test_deep_obj_str(self):
        """DeepObj 应具有字符串表示。"""
        obj = DeepObj()
        s = str(obj)
        assert "DeepObj" in s

    def test_deep_obj_clone(self):
        """DeepObj clone 应创建独立副本。"""
        obj = DeepObj()
        obj.test_attr = torch.tensor([1.0, 2.0, 3.0])
        cloned = obj.clone()
        
        # 修改原对象
        obj.test_attr[0] = 999.0
        
        # clone 应保持不变
        assert cloned.test_attr[0] != 999.0

    def test_deep_obj_to_device(self, device_auto):
        """DeepObj.to() 应将张量移动到指定设备。"""
        obj = DeepObj()
        obj.tensor_attr = torch.tensor([1.0, 2.0, 3.0])
        
        obj.to(device_auto)
        
        assert obj.device.type == device_auto.type
        assert obj.tensor_attr.device.type == device_auto.type

    def test_deep_obj_to_device_nested(self, device_auto):
        """DeepObj.to() 应能处理嵌套的 DeepObj。"""
        outer = DeepObj()
        inner = DeepObj()
        inner.data = torch.tensor([1.0, 2.0])
        outer.child = inner
        
        outer.to(device_auto)
        
        assert inner.data.device.type == device_auto.type

    def test_deep_obj_to_device_list(self, device_auto):
        """DeepObj.to() 应能处理张量列表。"""
        obj = DeepObj()
        obj.tensor_list = [torch.tensor([1.0]), torch.tensor([2.0])]
        
        obj.to(device_auto)
        
        for t in obj.tensor_list:
            assert t.device.type == device_auto.type

    def test_deep_obj_astype_float32(self):
        """DeepObj.astype() 应转换为 float32。"""
        obj = DeepObj(dtype=torch.float64)
        obj.data = torch.tensor([1.0, 2.0], dtype=torch.float64)
        
        obj.astype(torch.float32)
        
        assert obj.dtype == torch.float32
        assert obj.data.dtype == torch.float32

    def test_deep_obj_astype_float64(self):
        """DeepObj.astype() 应转换为 float64。"""
        obj = DeepObj(dtype=torch.float32)
        obj.data = torch.tensor([1.0, 2.0], dtype=torch.float32)
        
        obj.astype(torch.float64)
        
        assert obj.dtype == torch.float64
        assert obj.data.dtype == torch.float64

    def test_deep_obj_astype_none(self):
        """DeepObj.astype(None) 应不执行任何操作。"""
        obj = DeepObj(dtype=torch.float32)
        original_dtype = obj.dtype
        
        result = obj.astype(None)
        
        assert obj.dtype == original_dtype
        assert result is obj

    def test_deep_obj_astype_invalid(self):
        """DeepObj.astype() 应拒绝无效 dtype。"""
        obj = DeepObj()
        
        with pytest.raises(AssertionError):
            obj.astype(torch.int32)

    def test_deep_obj_call_raises(self):
        """若未实现 forward，DeepObj.__call__() 应抛出异常。"""
        obj = DeepObj()
        
        with pytest.raises(AttributeError):
            obj(torch.tensor([1.0]))
