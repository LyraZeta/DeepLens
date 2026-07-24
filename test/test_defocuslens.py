"""
deeplens/defocuslens.py 测试——离焦镜头模型。
"""

import pytest
import torch

from deeplens import DefocusLens
from deeplens.config import DEPTH


class TestDefocusLensInit:
    """测试 DefocusLens 初始化。"""

    def test_paraxial_init(self, device_auto):
        """应使用基本参数完成初始化。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        assert lens.foclen == 50.0
        assert lens.fnum == 1.8

    def test_paraxial_aperture_radius(self, device_auto):
        """检查光圈半径计算。"""
        foclen = 50.0
        fnum = 2.0
        
        lens = DefocusLens(
            foclen=foclen,
            fnum=fnum,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        # DefocusLens 不直接公开 'r'，因此这里只验证参数
        assert lens.foclen == foclen
        assert lens.fnum == fnum


class TestDefocusLensRefocus:
    """测试镜头重新对焦。"""

    def test_paraxial_refocus(self, device_auto):
        """应改变对焦距离。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        original_foc = lens.foc_dist
        lens.refocus(-1000.0)
        
        assert lens.foc_dist != original_foc
        assert lens.foc_dist == -1000.0

    def test_paraxial_refocus_infinity(self, device_auto):
        """应能处理无穷远对焦。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(DEPTH)
        
        assert lens.foc_dist == DEPTH


class TestDefocusLensCoC:
    """测试弥散圆计算。"""

    def test_paraxial_coc_at_focus(self, device_auto):
        """对焦距离处的 CoC 应为零。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        depth = torch.tensor([-1000.0], device=device_auto)
        coc = lens.coc(depth)
        
        assert coc.item() == pytest.approx(0.0, abs=0.01)

    def test_paraxial_coc_out_of_focus(self, device_auto):
        """CoC 应随离焦程度增大。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        
        depth_near = torch.tensor([-500.0], device=device_auto)
        depth_far = torch.tensor([-2000.0], device=device_auto)
        
        coc_near = lens.coc(depth_near)
        coc_far = lens.coc(depth_far)
        
        assert coc_near.abs().item() > 0
        assert coc_far.abs().item() > 0

    def test_paraxial_coc_batch(self, device_auto):
        """应能处理一批深度。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        depths = torch.tensor([-500.0, -1000.0, -2000.0], device=device_auto)
        cocs = lens.coc(depths)
        
        assert cocs.shape == depths.shape


class TestDefocusLensDoF:
    """测试景深计算。"""

    def test_paraxial_dof_exists(self, device_auto):
        """应计算出正值 DoF。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        # DoF 应为正值。注意：在此实现中，CoC=0 的焦点处标准 DoF 可能未定义。
        # 检查轻微离焦距离处的 DoF。
        depth = torch.tensor([-500.0], device=device_auto)
        dof = lens.dof(depth)
        
        assert dof.item() > 0

    def test_paraxial_coc_fnum_dependence(self, device_auto):
        """更大的 f-number（更小的光圈）应产生更小的 CoC。"""
        lens1 = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        lens2 = DefocusLens(
            foclen=50.0,
            fnum=8.0,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens1.refocus(-1000.0)
        lens2.refocus(-1000.0)
        
        depth = torch.tensor([-500.0], device=device_auto)
        coc1 = lens1.coc(depth)
        coc2 = lens2.coc(depth)
        
        assert coc2.item() < coc1.item()


class TestDefocusLensPSF:
    """测试 PSF 生成。"""

    def test_paraxial_psf_gaussian(self, device_auto):
        """应生成高斯 PSF。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[-500.0]], device=device_auto)  # 离焦
        points = torch.cat([torch.zeros(1, 2, device=device_auto), points], dim=-1)
        
        psf = lens.psf(points, ks=31, psf_type="gaussian")
        
        # PSF 为 [N, ks, ks]
        assert psf.shape[-2:] == (31, 31)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_paraxial_psf_pillbox(self, device_auto):
        """应生成均匀圆盘 PSF。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf = lens.psf(points, ks=31, psf_type="pillbox")

        assert psf.shape[-2:] == (31, 31)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_paraxial_psf_in_focus_sharp(self, device_auto):
        """焦点处的 PSF 应较尖锐（较小）。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points_focus = torch.tensor([[0.0, 0.0, -1000.0]], device=device_auto)
        points_defocus = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_focus = lens.psf(points_focus, ks=31, psf_type="gaussian")
        psf_defocus = lens.psf(points_defocus, ks=31, psf_type="gaussian")
        
        # 焦内 PSF 应更集中（峰值更高）
        assert psf_focus.max() > psf_defocus.max()

    def test_paraxial_psf_rgb(self, device_auto):
        """应生成 RGB PSF。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_rgb = lens.psf_rgb(points, ks=31)
        
        # 预期为 [N, 3, ks, ks] 或 [3, ks, ks]
        assert psf_rgb.shape[-3:] == (3, 31, 31)

    def test_paraxial_psf_batch(self, device_auto):
        """应能处理一批点。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([
            [0.0, 0.0, -500.0],
            [0.0, 0.0, -1000.0],
            [0.0, 0.0, -2000.0],
        ], device=device_auto)
        
        psf = lens.psf(points, ks=31, psf_type="gaussian")
        
        # 预期为 [3, ks, ks]
        assert psf.shape[-3:] == (3, 31, 31)


class TestDefocusLensDualPixel:
    """测试双像素 PSF 生成。"""

    def test_paraxial_psf_dp(self, device_auto):
        """应生成双像素 PSF。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_left, psf_right = lens.psf_dp(points, ks=31)
        
        assert psf_left.shape[-2:] == (31, 31)
        assert psf_right.shape[-2:] == (31, 31)

    def test_paraxial_psf_dp_disparity(self, device_auto):
        """左右 PSF 应具有视差。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)  # 离焦
        
        psf_left, psf_right = lens.psf_dp(points, ks=31)
        
        # 左右结果应不同
        diff = (psf_left - psf_right).abs().sum()
        assert diff.item() > 0.01

    def test_paraxial_psf_rgb_dp(self, device_auto):
        """应生成 RGB 双像素 PSF。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_left, psf_right = lens.psf_rgb_dp(points, ks=31)
        
        assert psf_left.shape[-3:] == (3, 31, 31)
        assert psf_right.shape[-3:] == (3, 31, 31)


class TestDefocusLensPSFMap:
    """测试 PSF 图生成。"""

    def test_paraxial_psf_map(self, device_auto):
        """应生成 PSF 图。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        psf_map = lens.psf_map(grid=(3, 3), ks=31, depth=-500.0)
        
        # psf_map：[grid_y, grid_x, 1, ks, ks]
        assert psf_map.shape == (3, 3, 1, 31, 31)

    def test_paraxial_psf_map_dp(self, device_auto):
        """应生成双像素 PSF 图。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        psf_map_left, psf_map_right = lens.psf_map_dp(grid=(3, 3), ks=31, depth=-500.0)
        
        assert psf_map_left.shape == (3, 3, 1, 31, 31)
        assert psf_map_right.shape == (3, 3, 1, 31, 31)


class TestDefocusLensRendering:
    """测试 RGBD 渲染。"""

    def test_paraxial_render_rgbd_dp(self, device_auto):
        """应从 RGBD 渲染双像素图像。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        
        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), -500.0, device=device_auto)
        
        img_left, img_right = lens.render_rgbd_dp(rgb, depth)
        
        assert img_left.shape == rgb.shape
        assert img_right.shape == rgb.shape

    def test_render_rgbd_dp_accepts_render_options(self, device_auto):
        """应接受与 render_rgbd 相同的 PSF 和图层选项。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )

        lens.refocus(-1000.0)

        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), -500.0, device=device_auto)

        img_left, img_right = lens.render_rgbd_dp(
            rgb,
            depth,
            psf_ks=31,
            num_layers=8,
        )

        assert img_left.shape == rgb.shape
        assert img_right.shape == rgb.shape


class TestDefocusLensRenderRGBD:
    """测试考虑遮挡的 RGBD 渲染。"""

    def test_render_rgbd_shape(self, device_auto):
        """应返回正确的输出 shape。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        lens.refocus(-1000.0)

        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), 500.0, device=device_auto)

        result = lens.render_rgbd(rgb, depth)
        assert result.shape == rgb.shape

    def test_render_rgbd_uniform_depth(self, device_auto):
        """深度均匀时，结果应有效且非零。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        lens.refocus(-1000.0)

        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), 500.0, device=device_auto)

        result = lens.render_rgbd(rgb, depth, num_layers=8)
        assert not torch.isnan(result).any()
        assert result.sum() > 0

    def test_render_rgbd_rejects_method_argument(self, device_auto):
        """render_rgbd 应仅公开离焦专用的渲染选项。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        lens.refocus(-1000.0)

        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), 500.0, device=device_auto)

        with pytest.raises(TypeError):
            lens.render_rgbd(rgb, depth, method="psf_patch", num_layers=8)

    def test_render_rgbd_depth_discontinuity(self, device_auto):
        """应能处理深度不连续（遮挡场景）。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        lens.refocus(-1000.0)

        # 创建具有明显深度不连续的场景
        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), 2000.0, device=device_auto)  # 背景
        depth[:, :, 16:48, 16:48] = 500.0  # 前景物体

        result = lens.render_rgbd(rgb, depth, num_layers=16)

        assert result.shape == rgb.shape
        assert not torch.isnan(result).any()
        assert result.min() >= 0

    def test_render_rgbd_3d_depth_input(self, device_auto):
        """应能处理 [B, H, W] 深度输入。"""
        lens = DefocusLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        lens.refocus(-1000.0)

        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth_3d = torch.full((1, 64, 64), 500.0, device=device_auto)  # [B, H, W]

        result = lens.render_rgbd(rgb, depth_3d)
        assert result.shape == rgb.shape
