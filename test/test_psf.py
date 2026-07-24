"""
`deeplens/imgsim/psf.py` 测试——PSF 卷积函数。
"""

import pytest
import torch

from deeplens.imgsim import (
    conv_psf,
    conv_psf_map,
    conv_psf_depth_interp,
    conv_psf_map_depth_interp,
    interp_psf_map,
    rotate_psf,
    splat_psf_per_pixel,
)


class TestConvPSF:
    """测试单个 PSF 卷积。"""

    def test_conv_psf_shape(self, device_auto):
        """输出应与输入具有相同形状。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        psf = torch.rand(3, 11, 11, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)  # 归一化
        
        result = conv_psf(img, psf)
        
        assert result.shape == img.shape

    def test_conv_psf_normalized(self, device_auto):
        """使用归一化 PSF 卷积应保持总能量。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        psf = torch.ones(3, 11, 11, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf(img, psf)
        
        # 总能量应近似保持
        energy_in = img.sum()
        energy_out = result.sum()
        assert torch.allclose(energy_in, energy_out, rtol=0.1)

    def test_conv_psf_delta(self, device_auto):
        """冲激函数 PSF 应返回原始图像。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # 创建冲激 PSF
        psf = torch.zeros(3, 11, 11, device=device_auto)
        psf[:, 5, 5] = 1.0
        
        result = conv_psf(img, psf)
        
        # 结果应与原图非常接近
        assert torch.allclose(result, img, atol=1e-5)

    def test_conv_psf_blur(self, device_auto):
        """方框 PSF 应使图像模糊。"""
        # 创建具有锐利边缘的图像
        img = torch.zeros(1, 3, 64, 64, device=device_auto)
        img[:, :, 20:44, 20:44] = 1.0

        # 方框模糊 PSF
        psf = torch.ones(3, 5, 5, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)

        result = conv_psf(img, psf)

        # 边缘应被平滑
        edge_sharpness_before = (img[:, :, 19, 32] - img[:, :, 20, 32]).abs()
        edge_sharpness_after = (result[:, :, 19, 32] - result[:, :, 20, 32]).abs()
        assert edge_sharpness_after.mean() < edge_sharpness_before.mean()

    @pytest.mark.parametrize("ks", [5, 11, 32])
    def test_conv_psf_fft_matches_conv(self, device_auto, ks):
        """对于奇数和偶数 ks，FFT 后端必须与直接卷积后端匹配。"""
        img = torch.rand(2, 3, 64, 64, device=device_auto)
        psf = torch.rand(3, ks, ks, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)

        result_conv = conv_psf(img, psf, method="conv")
        result_fft = conv_psf(img, psf, method="fft")

        assert result_fft.shape == result_conv.shape == img.shape
        assert torch.allclose(result_fft, result_conv, atol=1e-5)

    def test_conv_psf_fft_delta(self, device_auto):
        """FFT 后端的冲激 PSF 应返回原始图像。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        psf = torch.zeros(3, 11, 11, device=device_auto)
        psf[:, 5, 5] = 1.0

        result = conv_psf(img, psf, method="fft")

        assert torch.allclose(result, img, atol=1e-5)

    def test_conv_psf_unknown_method(self, device_auto):
        """未知方法应抛出 ValueError。"""
        img = torch.rand(1, 3, 16, 16, device=device_auto)
        psf = torch.ones(3, 5, 5, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)

        with pytest.raises(ValueError):
            conv_psf(img, psf, method="bogus")


class TestConvPSFMap:
    """测试空间变化 PSF 卷积。"""

    def test_conv_psf_map_shape(self, device_auto):
        """输出应与输入具有相同 shape。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # PSF 图：[grid_h, grid_w, C, ks, ks]
        psf_map = torch.rand(4, 4, 3, 11, 11, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf_map(img, psf_map)
        
        assert result.shape == img.shape

    def test_conv_psf_map_uniform(self, device_auto):
        """均匀 PSF 图应给出与单个 PSF 相同的结果。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # 创建均匀 PSF（所有网格点处均相同）
        single_psf = torch.rand(3, 11, 11, device=device_auto)
        single_psf = single_psf / single_psf.sum(dim=(-1, -2), keepdim=True)
        
        psf_map = single_psf.unsqueeze(0).unsqueeze(0).expand(4, 4, -1, -1, -1).clone()
        
        result_map = conv_psf_map(img, psf_map)
        result_single = conv_psf(img, single_psf)
        
        # 结果应相近
        assert torch.allclose(result_map, result_single, atol=0.1)


class TestSplatPSFPerPixel:
    """测试逐像素 PSF splatting。"""

    def test_splat_psf_per_pixel_shape(self, device_auto):
        """输出应与输入具有相同 shape。"""
        img = torch.rand(1, 3, 32, 32, device=device_auto)
        
        # 逐像素 PSF：[H, W, C, ks, ks]
        psf = torch.rand(32, 32, 3, 5, 5, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = splat_psf_per_pixel(img, psf)
        
        assert result.shape == img.shape

    @pytest.mark.parametrize("ks", [5, 6])
    def test_splat_psf_per_pixel_chunked_matches_full(self, device_auto, ks):
        """分块渲染应与整幅图像 splat 匹配。"""
        img = torch.rand(1, 3, 31, 29, device=device_auto)

        psf = torch.rand(31, 29, 3, ks, ks, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)

        result_full = splat_psf_per_pixel(img, psf)
        result_chunked = splat_psf_per_pixel(img, psf, chunk_size=8)

        assert torch.allclose(result_chunked, result_full, atol=1e-6)


class TestConvPSFDepthInterp:
    """测试深度插值 PSF 卷积。"""

    def test_conv_psf_depth_interp_shape(self, device_auto):
        """输出应与输入具有相同 shape。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = -torch.rand(1, 1, 64, 64, device=device_auto) - 0.01
        
        # 不同深度处的 PSF 核
        psf_kernels = torch.rand(5, 3, 11, 11, device=device_auto)
        psf_kernels = psf_kernels / psf_kernels.sum(dim=(-1, -2), keepdim=True)
        
        # 每个 PSF 的深度值
        psf_depths = torch.linspace(-2, -0.01, 5, device=device_auto)
        
        result = conv_psf_depth_interp(img, depth, psf_kernels, psf_depths)
        
        assert result.shape == img.shape

    def test_conv_psf_depth_interp_extreme_depths(self, device_auto):
        """应能处理边界处的深度。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # 位于边界值的深度
        depth = torch.full((1, 1, 64, 64), -0.5, device=device_auto)
        
        psf_kernels = torch.rand(5, 3, 11, 11, device=device_auto)
        psf_kernels = psf_kernels / psf_kernels.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.linspace(-2, -0.01, 5, device=device_auto)
        
        result = conv_psf_depth_interp(img, depth, psf_kernels, psf_depths)
        
        assert not torch.isnan(result).any()

    def test_conv_psf_depth_interp_disparity(self, device_auto):
        """应能处理视差插值模式。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = -(torch.rand(1, 1, 64, 64, device=device_auto) + 1.0)  # 使用负深度，避免视差接近零
        
        psf_kernels = torch.rand(5, 3, 11, 11, device=device_auto)
        psf_kernels = psf_kernels / psf_kernels.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.linspace(-3.0, -1.0, 5, device=device_auto)
        
        result = conv_psf_depth_interp(img, depth, psf_kernels, psf_depths, interp_mode="disparity")
        
        assert result.shape == img.shape
        assert not torch.isnan(result).any()

    def test_conv_psf_depth_interp_exact_endpoints(self, device_auto):
        """参考端点处的深度应精确使用端点 PSF。"""
        img = torch.rand(1, 3, 32, 32, device=device_auto)
        psf_kernels = torch.rand(2, 3, 5, 5, device=device_auto)
        psf_kernels = psf_kernels / psf_kernels.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.tensor([-3.0, -1.0], device=device_auto)

        far_depth = torch.full((1, 1, 32, 32), -3.0, device=device_auto)
        near_depth = torch.full((1, 1, 32, 32), -1.0, device=device_auto)

        result_far = conv_psf_depth_interp(img, far_depth, psf_kernels, psf_depths)
        result_near = conv_psf_depth_interp(img, near_depth, psf_kernels, psf_depths)

        assert torch.allclose(result_far, conv_psf(img, psf_kernels[0]), atol=1e-6)
        assert torch.allclose(result_near, conv_psf(img, psf_kernels[1]), atol=1e-6)

    def test_conv_psf_depth_interp_invalid_mode(self, device_auto):
        """无效插值模式应抛出错误。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.rand(1, 1, 64, 64, device=device_auto)
        psf_kernels = torch.rand(5, 3, 11, 11, device=device_auto)
        psf_depths = torch.linspace(0, 1, 5, device=device_auto)
        
        with pytest.raises(AssertionError):
            conv_psf_depth_interp(img, depth, psf_kernels, psf_depths, interp_mode="invalid")


class TestConvPSFMapDepthInterp:
    """测试深度插值 PSF 图卷积。"""

    def test_conv_psf_map_depth_interp_shape(self, device_auto):
        """输出应与输入具有相同 shape。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = -torch.rand(1, 1, 64, 64, device=device_auto) - 0.01
        
        # PSF 图：[grid_h, grid_w, num_depth, C, ks, ks]
        psf_map = torch.rand(4, 4, 5, 3, 11, 11, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.linspace(-2, -0.01, 5, device=device_auto)
        
        result = conv_psf_map_depth_interp(img, depth, psf_map, psf_depths)
        
        assert result.shape == img.shape

    def test_conv_psf_map_depth_interp_disparity(self, device_auto):
        """应能处理视差插值模式。"""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = -(torch.rand(1, 1, 64, 64, device=device_auto) + 1.0)
        
        psf_map = torch.rand(4, 4, 5, 3, 11, 11, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.linspace(-3.0, -1.0, 5, device=device_auto)
        
        result = conv_psf_map_depth_interp(img, depth, psf_map, psf_depths, interp_mode="disparity")
        
        assert result.shape == img.shape
        assert not torch.isnan(result).any()

    def test_conv_psf_map_depth_interp_exact_endpoints(self, device_auto):
        """参考端点处的深度应精确使用端点 PSF 图。"""
        img = torch.rand(1, 3, 32, 32, device=device_auto)
        psf_map = torch.rand(2, 2, 2, 3, 5, 5, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.tensor([-3.0, -1.0], device=device_auto)

        far_depth = torch.full((1, 1, 32, 32), -3.0, device=device_auto)
        near_depth = torch.full((1, 1, 32, 32), -1.0, device=device_auto)

        result_far = conv_psf_map_depth_interp(img, far_depth, psf_map, psf_depths)
        result_near = conv_psf_map_depth_interp(img, near_depth, psf_map, psf_depths)

        assert torch.allclose(result_far, conv_psf_map(img, psf_map[:, :, 0]), atol=1e-6)
        assert torch.allclose(result_near, conv_psf_map(img, psf_map[:, :, 1]), atol=1e-6)


class TestInterpPSFMap:
    """测试 PSF 图插值。"""

    def test_interp_psf_map_upsample(self, device_auto):
        """应上采样 PSF 网格。"""
        grid_old = 3
        grid_new = 6
        ks = 11
        
        psf_map = torch.rand(3, grid_old * ks, grid_old * ks, device=device_auto)
        
        interpolated = interp_psf_map(psf_map, grid_old=grid_old, grid_new=grid_new)
        
        assert interpolated.shape == (3, grid_new * ks, grid_new * ks)

    def test_interp_psf_map_identity(self, device_auto):
        """相同网格尺寸应返回相近的图。"""
        grid = 4
        ks = 11
        
        psf_map = torch.rand(3, grid * ks, grid * ks, device=device_auto)
        
        interpolated = interp_psf_map(psf_map, grid_old=grid, grid_new=grid)
        
        assert torch.allclose(interpolated, psf_map, atol=0.01)


class TestRotatePSF:
    """测试 PSF 旋转。"""

    def test_rotate_psf_shape(self, device_auto):
        """旋转应保留 shape。"""
        psf = torch.rand(4, 3, 21, 21, device=device_auto)
        theta = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device_auto)
        
        rotated = rotate_psf(psf, theta)
        
        assert rotated.shape == psf.shape

    def test_rotate_psf_zero(self, device_auto):
        """旋转角为零时应返回相同 PSF。"""
        psf = torch.rand(1, 3, 21, 21, device=device_auto)
        theta = torch.tensor([0.0], device=device_auto)
        
        rotated = rotate_psf(psf, theta)
        
        assert torch.allclose(rotated, psf, atol=1e-4)

    def test_rotate_psf_symmetric(self, device_auto):
        """对称 PSF 旋转后应保持不变。"""
        # 创建圆对称 PSF（类似高斯分布）
        ks = 21
        center = ks // 2
        y, x = torch.meshgrid(torch.arange(ks), torch.arange(ks), indexing="ij")
        r = torch.sqrt((x - center).float()**2 + (y - center).float()**2)
        psf_single = torch.exp(-r**2 / 10)
        psf_single = psf_single / psf_single.sum()
        
        psf = psf_single.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device_auto)
        theta = torch.tensor([1.57], device=device_auto)  # 90 度
        
        rotated = rotate_psf(psf, theta)
        
        # 由于对称性，结果应近似相同
        assert torch.allclose(rotated, psf, atol=0.05)


class TestPSFGPUPerformance:
    """测试 GPU 上的 PSF 操作。"""

    def test_conv_psf_gpu_batch(self, device_auto):
        """应能在 GPU 上处理批量输入。"""
        batch_size = 4
        img = torch.rand(batch_size, 3, 128, 128, device=device_auto)
        psf = torch.rand(3, 21, 21, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf(img, psf)
        
        assert result.shape == img.shape
        assert result.device.type == device_auto.type

    def test_conv_psf_map_gpu(self, device_auto):
        """PSF 图卷积应能在 GPU 上运行。"""
        img = torch.rand(1, 3, 128, 128, device=device_auto)
        psf_map = torch.rand(8, 8, 3, 15, 15, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf_map(img, psf_map)
        
        assert result.device.type == device_auto.type
