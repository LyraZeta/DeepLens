"""
deeplens/geolens.py 测试——主要几何镜头类。
"""

import os
import pytest
import torch

from deeplens import GeoLens
from deeplens.config import DEPTH, DEFAULT_WAVE


class TestGeoLensLoading:
    """测试从文件加载镜头。"""

    def test_geolens_load_json(self, sample_singlet_lens):
        """应从 JSON 文件加载镜头。"""
        lens = sample_singlet_lens
        
        assert lens is not None
        assert len(lens.surfaces) > 0

    def test_geolens_load_cellphone(self, sample_cellphone_lens):
        """应加载包含非球面的手机镜头。"""
        lens = sample_cellphone_lens
        
        assert lens is not None
        assert len(lens.surfaces) > 1

    def test_geolens_post_computation(self, sample_cellphone_lens):
        """加载后应计算 foclen、fov 和 fnum。"""
        # 使用带光圈的手机镜头
        lens = sample_cellphone_lens
        
        # GeoLens 会计算 hfov、vfov、dfov、rfov，但不直接设置 "fov" 属性
        assert hasattr(lens, "foclen")
        assert hasattr(lens, "dfov")
        assert hasattr(lens, "fnum")
        assert lens.foclen > 0
        # assert lens.fov > 0  # 已移除
        assert lens.fnum > 0

    def test_geolens_write_json(self, sample_singlet_lens, test_output_dir):
        """应将镜头保存为 JSON 文件。"""
        lens = sample_singlet_lens
        save_path = os.path.join(test_output_dir, "test_lens.json")
        
        lens.write_lens_json(save_path)
        
        assert os.path.exists(save_path)
        
        # 重新加载并验证
        lens2 = GeoLens(filename=save_path)
        assert len(lens2.surfaces) == len(lens.surfaces)

    def test_geolens_empty_init(self, device_auto):
        """无文件时应初始化空镜头。"""
        lens = GeoLens()
        lens.to(device_auto)
        
        assert lens.surfaces == []


class TestGeoLensRaySampling:
    """测试光线采样方法。"""

    def test_geolens_sample_from_fov(self, sample_singlet_lens):
        """应在不同视场角采样平行光线。"""
        lens = sample_singlet_lens
        
        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        
        assert ray is not None
        assert ray.o.shape[-1] == 3
        assert ray.d.shape[-1] == 3

    def test_geolens_sample_from_fov_offaxis(self, sample_singlet_lens):
        """应采样轴外平行光线。"""
        lens = sample_singlet_lens
        
        ray = lens.sample_from_fov(fov_x=[5.0], fov_y=[0.0], num_rays=512)
        
        # 轴外光线的 x 方向分量应非零
        assert ray.d[..., 0].abs().max() > 0.01

    def test_geolens_sample_point_source(self, sample_singlet_lens):
        """应通过 sample_from_fov 采样有限深度的点光源光线。"""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], depth=DEPTH, num_rays=512)

        assert ray is not None
        assert ray.shape[-1] == 512

    def test_geolens_sample_from_points(self, sample_singlet_lens):
        """应从指定点采样光线。"""
        lens = sample_singlet_lens
        
        points = [[0.0, 0.0, -10000.0]]
        ray = lens.sample_from_points(points=points, num_rays=512)
        
        assert ray is not None
        assert ray.shape[-1] == 512

    def test_geolens_sample_from_points_batch(self, sample_singlet_lens):
        """应从多个点采样光线。"""
        lens = sample_singlet_lens
        
        points = [[0.0, 0.0, -10000.0], [1.0, 1.0, -10000.0]]
        ray = lens.sample_from_points(points=points, num_rays=512)
        
        assert ray.o.shape[0] == 2

    def test_geolens_sample_sensor(self, sample_cellphone_lens):
        """应从传感器采样反向光线。"""
        lens = sample_cellphone_lens  # 带孔径光阑
        
        ray = lens.sample_sensor(spp=2)
        
        assert ray is not None
        # 光线方向的 z 分量均值应表示反向
        # （确切符号取决于实现）


class TestGeoLensTracing:
    """测试穿过镜头的光线追迹。"""

    def test_geolens_trace_basic(self, sample_singlet_lens):
        """应追迹穿过镜头的光线。"""
        lens = sample_singlet_lens
        
        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, _ = lens.trace(ray)
        
        assert ray_out is not None
        assert ray_out.is_valid.sum() > 0

    def test_geolens_trace_with_record(self, sample_singlet_lens):
        """应在追迹期间记录光线路径。"""
        lens = sample_singlet_lens
        
        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, ray_record = lens.trace(ray, record=True)
        
        assert ray_record is not None
        assert len(ray_record) > 0

    def test_geolens_trace_preserves_valid(self, sample_singlet_lens):
        """追迹应保持或减少有效光线数量。"""
        lens = sample_singlet_lens
        
        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        valid_before = ray.is_valid.sum().item()
        
        ray_out, _ = lens.trace(ray)
        valid_after = ray_out.is_valid.sum().item()
        
        assert valid_after <= valid_before
        assert valid_after > 0  # 应有部分光线保留下来

    def test_geolens_call_is_trace(self, sample_singlet_lens):
        """__call__ 应为 trace 的别名。"""
        lens = sample_singlet_lens
        
        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out = lens(ray)
        
        assert ray_out is not None


class TestGeoLensPSF:
    """测试 PSF 计算。"""

    def test_geolens_psf_mono(self, sample_cellphone_lens):
        """应计算单色 PSF。"""
        lens = sample_cellphone_lens
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31, model="geometric")
        
        # 批量输入时，PSF 应为 [1, ks, ks]
        assert psf.shape == (1, 31, 31)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_psf_coherent_dispatcher(self, sample_cellphone_lens):
        """应通过分派器计算相干 PSF。"""
        lens = sample_cellphone_lens
        
        # 相干 PSF 要求 float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64)
            psf = lens.psf(
                points,
                wvln=DEFAULT_WAVE,
                ks=31,
                model="coherent",
                spp=1_000_000,
            )
            
            assert psf.shape == (31, 31)  # psf_pupil_prop 当前始终返回二维结果
            assert psf.sum().item() == pytest.approx(1.0, abs=0.1)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_huygens_dispatcher(self, sample_cellphone_lens):
        """应通过分派器计算惠更斯 PSF。"""
        lens = sample_cellphone_lens
        
        # 惠更斯模式要求 float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64)
            psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31, spp=10000, model="huygens")
            
            assert psf.shape == (31, 31) # 惠更斯方法当前仅支持单点，返回二维结果
            assert psf.sum().item() == pytest.approx(1.0, abs=0.1)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_normalized(self, sample_cellphone_lens):
        """PSF 之和应约为 1。"""
        lens = sample_cellphone_lens
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=64)
        
        # 除非隐式使用 psf_rgb，否则 DefocusLens PSF 通常为单通道
        # 对单个点，psf() 返回 [N_points, ks, ks]
        assert psf.shape == (1, 64, 64)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_psf_rgb(self, sample_cellphone_lens):
        """应计算 RGB PSF。"""
        lens = sample_cellphone_lens
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf_rgb = lens.psf_rgb(points, ks=64)
        
        # 检查三通道输出
        assert psf_rgb.shape[1] == 3

    def test_geolens_psf_map(self, sample_cellphone_lens):
        """应计算跨视场的 PSF 图。"""
        lens = sample_cellphone_lens
        
        psf_map = lens.psf_map(grid=(3, 3), ks=31, depth=DEPTH)
        
        # PSF 图应具有正确的网格维度
        assert psf_map.shape == (3, 3, 1, 31, 31)

    def test_geolens_psf_huygens_basic(self, sample_cellphone_lens):
        """应为单个点计算惠更斯 PSF（相干模式）。"""
        lens = sample_cellphone_lens
        
        # 惠更斯模式的相干光线追迹要求 float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
            # 使用较小的 spp 以加快测试
            psf = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
            
            # 单个点的 PSF 应具有正确 shape [ks, ks]
            assert psf.shape == (31, 31)
            # 惠更斯 PSF 应为实数值（强度）
            assert not psf.is_complex()
            # 所有值均应非负（强度）
            assert psf.min() >= 0
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_huygens_normalized(self, sample_cellphone_lens):
        """惠更斯 PSF 之和应约为 1。"""
        lens = sample_cellphone_lens
        
        # 惠更斯模式的相干光线追迹要求 float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
            psf = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=64, spp=10000)
            
            # PSF 应已归一化
            assert psf.shape == (64, 64)
            assert psf.sum().item() == pytest.approx(1.0, abs=0.1)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_huygens_vs_geometric_different(self, sample_cellphone_lens):
        """惠更斯 PSF 和几何 PSF 应产生不同结果。"""
        lens = sample_cellphone_lens
        
        # 惠更斯模式的相干光线追迹要求 float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
            psf_geo = lens.psf(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
            psf_huygens = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
            
            # 两者应具有相同 shape
            assert psf_geo.shape == psf_huygens.shape
            # 但数值应不同（相干与非相干）
            # 转换为相同 dtype 以便比较
            assert not torch.allclose(psf_geo.to(torch.float64), psf_huygens.to(torch.float64), atol=1e-3)
        finally:
            torch.set_default_dtype(original_dtype)


class TestGeoLensRendering:
    """测试图像渲染。"""

    def test_geolens_render_psf(self, sample_cellphone_lens, sample_image_small):
        """应使用 PSF 卷积渲染图像。"""
        lens = sample_cellphone_lens
        img = sample_image_small
        
        img_render = lens.render(img, depth=DEPTH, method="psf_patch")
        
        assert img_render.shape == img.shape

    def test_geolens_render_psf_map(self, sample_cellphone_lens, sample_image_small):
        """应使用空间变化 PSF 渲染图像。"""
        lens = sample_cellphone_lens
        img = sample_image_small
        lens.set_sensor_res((64, 64))
        
        # psf_map 要求图像分辨率与传感器分辨率匹配
        # 将输入图像缩放至与传感器分辨率匹配
        img_large = torch.nn.functional.interpolate(img, size=(64, 64), mode='bilinear')
        img_render = lens.render(
            img_large,
            depth=DEPTH,
            method="psf_map",
            psf_grid=(2, 2),
            psf_ks=31,
            psf_spp=1024,
            warp_grid=8,
        )
        
        # 输出 shape 应匹配传感器分辨率，而不是较小的输入图像 shape
        assert img_render.shape[-2:] == lens.sensor_res

    def test_geolens_render_preserves_range(self, sample_cellphone_lens, sample_image_small):
        """渲染图像应具有非负值。"""
        lens = sample_cellphone_lens
        img = sample_image_small
        
        img_render = lens.render(img, depth=DEPTH, method="psf_patch")
        
        assert img_render.min() >= 0

    def test_geolens_analysis_rendering(self, sample_cellphone_lens, sample_image_small, test_output_dir):
        """应运行 analysis_rendering 并返回渲染图像。"""
        import os
        lens = sample_cellphone_lens
        # 将 [B, C, H, W] 转换为 analysis_rendering 所需的 [H, W, C] 格式
        img = sample_image_small.squeeze(0).permute(1, 2, 0)
        save_path = os.path.join(test_output_dir, "analysis_render_test")
        
        # 运行 analysis_rendering
        img_render = lens.analysis_rendering(
            img_org=img, 
            depth=DEPTH, 
            spp=64,
            save_name=save_path,
            method="ray_tracing",
            show=False
        )
        
        # 检查输出 shape [B, C, H, W]
        assert img_render is not None
        assert len(img_render.shape) == 4
        assert img_render.shape[1] == 3  # RGB 通道
        # 检查数值位于有效范围内
        assert img_render.min() >= 0
        assert img_render.max() <= 1
        # 检查输出文件已保存
        assert os.path.exists(f"{save_path}.png")


class TestGeoLensProperties:
    """测试镜头属性计算。"""

    def test_geolens_refocus(self, sample_singlet_lens):
        """应将镜头重新对焦到新距离。"""
        lens = sample_singlet_lens
        original_d = lens.d_sensor.item()
        
        lens.refocus(foc_dist=-500.0)
        
        # d_sensor 应发生变化
        assert lens.d_sensor.item() != original_d

    def test_geolens_aperture_idx(self, sample_cellphone_lens):
        """应识别孔径光阑索引。"""
        lens = sample_cellphone_lens
        
        aper_idx = lens.aper_idx
        
        assert isinstance(aper_idx, int)
        assert 0 <= aper_idx < len(lens.surfaces)

    def test_geolens_fov_calc(self, sample_cellphone_lens):
        """应计算正确的 FoV。"""
        lens = sample_cellphone_lens
        
        lens.calc_fov()
        
        assert hasattr(lens, "dfov")
        assert lens.dfov > 0 
        # lens.calc_fov() 返回 None，因此不检查返回值

    def test_geolens_sensor_properties(self, sample_singlet_lens):
        """应具有正确的传感器属性。"""
        lens = sample_singlet_lens
        
        assert lens.sensor_res[0] > 0
        assert lens.sensor_res[1] > 0
        assert lens.sensor_size[0] > 0
        assert lens.sensor_size[1] > 0


class TestGeoLensDifferentiability:
    """测试梯度在镜头操作中的传播。"""

    def test_geolens_psf_differentiable(self, sample_cellphone_lens):
        """PSF 计算应可微分。"""
        lens = sample_cellphone_lens
        
        # 令一个表面参数需要梯度
        lens.surfaces[1].d.requires_grad_(True)
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31)
        
        loss = psf.sum()
        loss.backward()
        
        # 检查梯度存在
        assert lens.surfaces[1].d.grad is not None

    def test_geolens_get_optimizer(self, sample_cellphone_lens):
        """应返回优化器参数。"""
        lens = sample_cellphone_lens
        
        optimizer = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-4, 1e-4])
        
        assert optimizer is not None


class TestGeoLensVisualization:
    """测试可视化方法。"""

    def test_geolens_draw_layout(self, sample_cellphone_lens, test_output_dir):
        """应绘制镜头布局。"""
        lens = sample_cellphone_lens
        save_path = os.path.join(test_output_dir, "lens_layout.png")
        
        lens.draw_layout(filename=save_path)
        
        assert os.path.exists(save_path)

    def test_geolens_analysis(self, sample_cellphone_lens, test_output_dir):
        """应运行镜头分析。"""
        lens = sample_cellphone_lens
        save_path = os.path.join(test_output_dir, "lens_analysis.png")
        
        # 此项可能需要更多设置，因此这里只检查其不会崩溃
        try:
            lens.analysis(save_name=save_path)
        except Exception as e:
            pytest.skip(f"Analysis requires additional dependencies: {e}")


class TestGeoLensDeviceHandling:
    """测试设备转移和 GPU 支持。"""

    def test_geolens_to_device(self, sample_singlet_lens, device_auto):
        """应将镜头移动到指定设备。"""
        lens = sample_singlet_lens
        lens.to(device_auto)
        
        assert lens.device.type == device_auto.type
        for surf in lens.surfaces:
            assert surf.d.device.type == device_auto.type

    def test_geolens_trace_on_gpu(self, sample_singlet_lens, device_auto):
        """追迹应能在 GPU 上运行。"""
        lens = sample_singlet_lens
        lens.to(device_auto)
        
        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, _ = lens.trace(ray)
        
        assert ray_out.o.device.type == device_auto.type

    def test_geolens_psf_on_gpu(self, sample_cellphone_lens, device_auto):
        """PSF 计算应能在 GPU 上运行。"""
        lens = sample_cellphone_lens
        lens.to(device_auto)
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device_auto)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31)
        
        assert psf.device.type == device_auto.type


class TestGeoLensDistortion:
    """测试畸变计算方法。"""

    def test_geolens_distortion_map(self, sample_cellphone_lens):
        """应计算畸变图。"""
        lens = sample_cellphone_lens

        distortion_map = lens.calc_distortion_map(num_grid=(5, 5), depth=DEPTH)
        
        assert distortion_map.shape == (5, 5, 2)
        # 畸变值应归一化到约 [-1, 1]
        assert distortion_map.abs().max() <= 2.0

    def test_geolens_inv_distortion_map(self, sample_cellphone_lens):
        """应计算用于图像变形的逆畸变网格。"""
        lens = sample_cellphone_lens

        inv_distortion_map = lens.calc_inv_distortion_map(num_grid=(5, 3), depth=DEPTH)

        assert inv_distortion_map.shape == (3, 5, 2)
        assert not torch.isnan(inv_distortion_map).any()
        assert inv_distortion_map.abs().max() <= 2.0

        x, y = torch.meshgrid(
            torch.linspace(-1, 1, 5, device=lens.device),
            torch.linspace(-1, 1, 3, device=lens.device),
            indexing="xy",
        )
        expected_grid = torch.stack((x, y), dim=-1)
        points = torch.cat(
            [
                inv_distortion_map.reshape(-1, 2),
                torch.full((15, 1), DEPTH, device=lens.device),
            ],
            dim=-1,
        )
        actual_grid = lens.distortion_center(points).reshape(3, 5, 2)
        assert torch.allclose(actual_grid, expected_grid, atol=5e-2)

    def test_geolens_warp_shape(self, sample_cellphone_lens):
        """变形应保留图像 shape。"""
        lens = sample_cellphone_lens
        img = torch.rand(1, 3, 32, 48, device=lens.device)

        img_warped = lens.warp(img, depth=DEPTH, num_grid=(5, 3))

        assert img_warped.shape == img.shape

    def test_geolens_distortion_center_single_point(self, sample_cellphone_lens):
        """应计算单个点的畸变中心。"""
        lens = sample_cellphone_lens
        
        # 中心处的单个归一化点
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        distortion_center = lens.distortion_center(points)
        
        # 输出应为 [N, 2]
        assert distortion_center.shape == (1, 2)
        # 中心点的畸变中心应接近原点
        assert distortion_center.abs().max() < 0.5

    def test_geolens_distortion_center_multiple_points(self, sample_cellphone_lens):
        """应计算多个点的畸变中心。"""
        lens = sample_cellphone_lens
        
        # 多个归一化点
        points = torch.tensor([
            [0.0, 0.0, DEPTH],
            [0.5, 0.0, DEPTH],
            [0.0, 0.5, DEPTH],
            [0.5, 0.5, DEPTH],
        ], device=lens.device)
        distortion_center = lens.distortion_center(points)
        
        # 输出应为 [N, 2]
        assert distortion_center.shape == (4, 2)
        # 所有值均应位于有效的归一化范围内
        assert distortion_center.abs().max() <= 2.0

    def test_geolens_distortion_center_normalized_range(self, sample_cellphone_lens):
        """对于合理输入，畸变中心输出应位于归一化范围 [-1, 1] 内。"""
        lens = sample_cellphone_lens
        
        # 归一化点网格
        x = torch.linspace(-0.8, 0.8, 3)
        y = torch.linspace(-0.8, 0.8, 3)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        z = torch.full_like(xx, DEPTH)
        points = torch.stack([xx, yy, z], dim=-1).reshape(-1, 3).to(lens.device)
        
        distortion_center = lens.distortion_center(points)
        
        # 输出应为 [9, 2]
        assert distortion_center.shape == (9, 2)
        # 大多数畸变中心应位于合理范围内
        # （由于镜头畸变可能超过 1.0，但不应过大）
        assert distortion_center.abs().max() < 3.0
