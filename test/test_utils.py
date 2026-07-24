"""
deeplens/utils.py 测试——实用函数。
"""

import pytest
import torch
import numpy as np

from deeplens.utils import (
    foc_dist_balanced,
    grid_sample_xy,
    img2batch,
    interp1d,
    batch_psnr,
    batch_ssim,
    normalize_ImageNet,
    denormalize_ImageNet,
)


class TestInterp1d:
    """测试一维插值。"""

    def test_interp1d_exact_keys(self, device_auto):
        """在关键点查询应返回精确值。"""
        key = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        value = torch.tensor([0.0, 10.0, 20.0], device=device_auto)
        query = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert torch.allclose(result, value, atol=1e-5)

    def test_interp1d_midpoint(self, device_auto):
        """中点查询应给出中点值。"""
        key = torch.tensor([0.0, 2.0], device=device_auto)
        value = torch.tensor([0.0, 20.0], device=device_auto)
        query = torch.tensor([1.0], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert torch.allclose(result, torch.tensor([10.0], device=device_auto), atol=1e-5)

    def test_interp1d_batch(self, device_auto):
        """应能处理批量值。"""
        key = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        value = torch.tensor([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]], device=device_auto)
        query = torch.tensor([0.5, 1.5], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert result.shape == (2, 2)


class TestGridSampleXY:
    """测试使用 xy 坐标的网格采样。"""

    def test_grid_sample_xy_identity(self, device_auto):
        """恒等网格应返回相同图像。"""
        img = torch.rand(1, 3, 32, 32, device=device_auto)
        
        # 创建恒等网格
        y = torch.linspace(1, -1, 32, device=device_auto)
        x = torch.linspace(-1, 1, 32, device=device_auto)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid_xy = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        
        result = grid_sample_xy(img, grid_xy, align_corners=True)
        
        assert torch.allclose(result, img, atol=1e-4)

    def test_grid_sample_xy_shape(self, device_auto):
        """输出 shape 应与网格 shape 匹配。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # 更小的输出网格
        grid_xy = torch.rand(1, 32, 32, 2, device=device_auto) * 2 - 1
        
        result = grid_sample_xy(img, grid_xy)
        
        assert result.shape == (1, 3, 32, 32)


class TestImg2Batch:
    """测试图像到 batch 的转换。"""

    def test_img2batch_numpy_hwc(self, device_auto):
        """应将 numpy HWC 图像转换为 batch。"""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)
        assert isinstance(batch, torch.Tensor)

    def test_img2batch_tensor_chw(self, device_auto):
        """应将张量 CHW 图像转换为 batch。"""
        img = torch.rand(3, 64, 64, device=device_auto)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)

    def test_img2batch_tensor_hwc(self, device_auto):
        """应将张量 HWC 图像转换为 batch。"""
        img = torch.rand(64, 64, 3, device=device_auto)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)

    def test_img2batch_already_batch(self, device_auto):
        """应能处理已具有 batch 维度的图像。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)


class TestBatchPSNR:
    """测试 batch PSNR 计算。"""

    def test_batch_psnr_identical(self, device_auto):
        """相同图像的 PSNR 应非常高。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        psnr = batch_psnr(img, img)
        
        assert psnr.item() > 40  # 非常高的 PSNR

    def test_batch_psnr_different(self, device_auto):
        """不同图像的 PSNR 应为有限值。"""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = torch.rand(1, 3, 64, 64, device=device_auto)
        
        psnr = batch_psnr(img1, img2)
        
        assert psnr.item() > 0
        assert psnr.item() < 60

    def test_batch_psnr_batch(self, device_auto):
        """应能处理批量输入。"""
        pred = torch.rand(4, 3, 64, 64, device=device_auto)
        target = pred + torch.randn_like(pred) * 0.1
        target = target.clamp(0, 1)
        
        psnr = batch_psnr(pred, target)
        
        assert psnr.shape == (4,)


class TestBatchSSIM:
    """测试 batch SSIM 计算。"""

    def test_batch_ssim_identical(self, device_auto):
        """相同图像的 SSIM 应为 1。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        ssim = batch_ssim(img, img)
        
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_batch_ssim_different(self, device_auto):
        """不同图像的 SSIM 应小于 1。"""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = torch.rand(1, 3, 64, 64, device=device_auto)
        
        ssim = batch_ssim(img1, img2)
        
        assert ssim < 1.0
        assert ssim > -1.0

    def test_batch_ssim_range(self, device_auto):
        """SSIM 应位于 [-1, 1] 范围内。"""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = 1 - img1  # 反相图像
        
        ssim = batch_ssim(img1, img2)
        
        assert -1 <= ssim <= 1


class TestImageNetNormalization:
    """测试 ImageNet 归一化。"""

    def test_normalize_imagenet_shape(self, device_auto):
        """归一化应保留 shape。"""
        batch = torch.rand(4, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        
        assert normalized.shape == batch.shape

    def test_normalize_imagenet_range(self, device_auto):
        """归一化后的值应大致以 0 为中心。"""
        batch = torch.rand(4, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        
        # 均值应接近 0
        assert normalized.mean().abs() < 1.0

    def test_denormalize_imagenet(self, device_auto):
        """反归一化应逆转归一化。"""
        batch = torch.rand(4, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        denormalized = denormalize_ImageNet(normalized)
        
        assert torch.allclose(denormalized, batch, atol=1e-5)


class TestFocDistBalanced:
    """测试对焦距离计算。"""

    def test_foc_dist_balanced_symmetric(self, device_auto):
        """相等距离应给出几何平均对焦距离。"""
        d1 = -1000.0
        d2 = -1000.0
        
        foc = foc_dist_balanced(d1, d2)
        
        assert foc == pytest.approx(d1, abs=1.0)

    def test_foc_dist_balanced_asymmetric(self, device_auto):
        """不对称距离应给出平衡的对焦距离。"""
        d1 = -500.0
        d2 = -2000.0
        
        foc = foc_dist_balanced(d1, d2)
        
        # 结果应位于 d1 和 d2 之间
        assert min(d1, d2) < foc < max(d1, d2)

    def test_foc_dist_balanced_negative(self, device_auto):
        """应能处理负距离（镜头前方）。"""
        d1 = -100.0
        d2 = -10000.0
        
        foc = foc_dist_balanced(d1, d2)
        
        assert foc < 0


class TestUtilsGPU:
    """测试 GPU 上的实用函数。"""

    def test_interp1d_gpu(self, device_auto):
        """插值应能在 GPU 上运行。"""
        key = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        value = torch.tensor([0.0, 10.0, 20.0], device=device_auto)
        query = torch.tensor([0.5, 1.5], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert result.device.type == device_auto.type

    def test_batch_psnr_gpu(self, device_auto):
        """PSNR 应能在 GPU 上运行。"""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = torch.rand(1, 3, 64, 64, device=device_auto)
        
        psnr = batch_psnr(img1, img2)
        
        assert isinstance(psnr, torch.Tensor)

    def test_normalize_imagenet_gpu(self, device_auto):
        """归一化应能在 GPU 上运行。"""
        batch = torch.rand(1, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        
        assert normalized.device.type == device_auto.type
