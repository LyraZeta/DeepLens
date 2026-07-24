"""deeplens/optics/loss.py 测试——PSFLoss 和 PSFStrehlLoss。"""

import pytest
import torch

from deeplens.loss import PSFLoss, PSFStrehlLoss


class TestPSFLoss:
    """测试 PSFLoss。"""

    def test_forward_4d_input(self):
        """PSFLoss 接受 [B, C, H, W] 输入并返回正标量。"""
        loss_fn = PSFLoss()
        psf = torch.rand(1, 3, 64, 64)
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_3d_input(self):
        """PSFLoss 接受 [C, H, W] 输入（自动添加 batch 维度）。"""
        loss_fn = PSFLoss()
        psf = torch.rand(3, 64, 64)
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_forward_2d_input(self):
        """PSFLoss 接受 [H, W] 输入（自动添加 batch 和通道维度）。"""
        loss_fn = PSFLoss()
        psf = torch.rand(64, 64)
        loss = loss_fn(psf)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """PSFLoss 支持梯度反向传播。"""
        loss_fn = PSFLoss()
        psf = torch.rand(1, 3, 32, 32, requires_grad=True)
        loss = loss_fn(psf)
        loss.backward()
        assert psf.grad is not None
        assert psf.grad.shape == psf.shape


class TestPSFStrehlLoss:
    """测试 PSFStrehlLoss。"""

    def test_delta_psf_high_strehl(self):
        """delta（集中）PSF 应产生较高的 Strehl 分数。"""
        loss_fn = PSFStrehlLoss()
        psf = torch.zeros(1, 3, 64, 64)
        psf[:, :, 32, 32] = 1.0  # 所有能量位于中心
        score = loss_fn(psf)
        assert score.item() > 0.5

    def test_uniform_psf_low_strehl(self):
        """均匀（分散）PSF 应产生较低的 Strehl 分数。"""
        loss_fn = PSFStrehlLoss()
        psf = torch.ones(1, 3, 64, 64) / (64 * 64)
        score = loss_fn(psf)
        # 对于均匀分布，中心强度 = 1/(64*64) ≈ 0.00024
        assert score.item() < 0.01

    def test_output_range(self):
        """Strehl 分数应位于 [0, 1]。"""
        loss_fn = PSFStrehlLoss()
        psf = torch.rand(2, 3, 32, 32).abs()
        score = loss_fn(psf)
        assert 0 <= score.item() <= 1.0

    def test_3d_input(self):
        """PSFStrehlLoss 接受 [C, H, W] 输入。"""
        loss_fn = PSFStrehlLoss()
        psf = torch.rand(3, 32, 32).abs()
        score = loss_fn(psf)
        assert score.dim() == 0

    def test_gradient_flow(self):
        """PSFStrehlLoss 支持梯度反向传播。"""
        loss_fn = PSFStrehlLoss()
        psf = (torch.rand(1, 3, 32, 32) + 1e-6).requires_grad_(True)
        score = loss_fn(psf)
        score.backward()
        assert psf.grad is not None
