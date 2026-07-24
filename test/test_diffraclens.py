"""deeplens/optics/diffraclens.py 测试——DiffractiveLens。"""

import builtins
import json
import os

import pytest
import torch


class TestDiffractiveLensInit:
    """测试 DiffractiveLens 初始化。"""

    def test_init_empty(self):
        """无需文件即可创建 DiffractiveLens。"""
        from deeplens import DiffractiveLens

        old_dtype = torch.get_default_dtype()
        lens = DiffractiveLens()
        torch.set_default_dtype(old_dtype)
        assert lens.surfaces == []
        assert lens.sensor_size == (8.0, 8.0)
        assert lens.sensor_res == (2000, 2000)

    def test_init_with_surfaces(self, sample_diffraclens):
        """sample_diffraclens fixture 应创建有效镜头。"""
        lens = sample_diffraclens
        assert len(lens.surfaces) == 1
        assert lens.d_sensor is not None

    def test_init_reads_utf8_json_on_non_utf8_locale(self, tmp_path, monkeypatch):
        """镜头 JSON 解码不得依赖操作系统区域设置。"""
        from deeplens import DiffractiveLens

        lens_path = tmp_path / "utf8_lens.json"
        lens_data = {
            "info": "Fourier-plane filter — UTF-8",
            "d_sensor": 50.0,
            "sensor_size": [1.0, 1.0],
            "sensor_res": [32, 32],
            "surfaces": [
                {
                    "type": "ThinLens",
                    "f0": 50.0,
                    "res": [32, 32],
                    "fab_ps": 0.02,
                    "d_next": 50.0,
                }
            ],
        }
        lens_path.write_text(
            json.dumps(lens_data, ensure_ascii=False), encoding="utf-8"
        )

        real_open = builtins.open

        def open_with_ascii_default(file, mode="r", *args, **kwargs):
            if (
                os.fspath(file) == os.fspath(lens_path)
                and "r" in mode
                and "encoding" not in kwargs
            ):
                kwargs["encoding"] = "ascii"
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", open_with_ascii_default)
        lens = DiffractiveLens(filename=str(lens_path), device="cpu")

        assert lens.lens_info == lens_data["info"]
        assert len(lens.surfaces) == 1


class TestDiffractiveLensPSF:
    """测试 PSF 计算。"""

    def test_psf_shape(self, sample_diffraclens):
        """psf() 返回 [ks, ks] 张量。"""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=ks)
        assert psf.shape == (ks, ks)
        assert (psf >= 0).all()

    def test_psf_finite_depth(self, sample_diffraclens):
        """psf() 应支持有限深度。"""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.0, 0.0, -500.0], ks=ks)
        assert psf.shape == (ks, ks)

    def test_psf_off_axis(self, sample_diffraclens):
        """psf() 应支持轴外点光源。"""
        lens = sample_diffraclens
        ks = 64
        psf = lens.psf(points=[0.3, 0.0, float("-inf")], ks=ks)
        assert psf.shape == (ks, ks)
        assert torch.isfinite(psf).all()
        assert abs(float(psf.sum()) - 1.0) < 1e-3

    def test_psf_batch(self, sample_diffraclens):
        """psf() 应支持一批点 -> [N, ks, ks]。"""
        lens = sample_diffraclens
        ks = 64
        points = [[0.0, 0.0, float("-inf")], [0.3, 0.0, float("-inf")]]
        psf = lens.psf(points=points, ks=ks)
        assert psf.shape == (2, ks, ks)


class TestDiffractiveLensOffAxisCentering:
    """当 recenter=False 时，轴外 PSF 围绕光源的透视（针孔）像进行裁剪，因此
    焦点位于核中心。这还要求采用不反转约定（+x 光源成像到 +x 一侧），否则预测
    中心将与实际峰值不一致。"""

    @staticmethod
    def _cpu_lens():
        from deeplens import DiffractiveLens
        from deeplens.diffractive_surface import Fresnel

        old = torch.get_default_dtype()
        lens = DiffractiveLens(device="cpu")
        lens.surfaces = [Fresnel(f0=50, d=0, res=256, fab_ps=0.008)]
        lens.surfaces[0].to(torch.device("cpu"))
        lens.d_sensor = torch.tensor(50.0, dtype=torch.float64)
        lens.foclen = 50.0
        lens.sensor_size = (2.0, 2.0)
        lens.sensor_res = (256, 256)
        lens.pixel_size = lens.sensor_size[0] / lens.sensor_res[0]
        torch.set_default_dtype(old)
        return lens

    def test_off_axis_x_centered_on_perspective_point(self):
        """recenter=False 时，+x 光源的峰值应位于核中心。"""
        lens = self._cpu_lens()
        ks = 64
        psf = lens.psf(points=[0.7, 0.0, float("-inf")], ks=ks, recenter=False)
        peak = int(torch.argmax(psf))
        row, col = peak // ks, peak % ks
        assert abs(col - ks // 2) <= ks // 8
        assert abs(row - ks // 2) <= ks // 8

    def test_off_axis_y_centered_on_perspective_point(self):
        """recenter=False 时，+y 光源的峰值应位于核中心。"""
        lens = self._cpu_lens()
        ks = 64
        psf = lens.psf(points=[0.0, 0.7, float("-inf")], ks=ks, recenter=False)
        peak = int(torch.argmax(psf))
        row, col = peak // ks, peak % ks
        assert abs(col - ks // 2) <= ks // 8
        assert abs(row - ks // 2) <= ks // 8

    def test_finite_depth_perspective_center_matches_focus(self):
        """对于有限深度的轴外光源，透视中心（recenter=False）必须与真实焦点
        重合，与准直情形相同。因此 recenter=False 的峰值应与 recenter=True 的峰值
        匹配；若两条路径的反转方式不同，前者将错误捕获一个弱旁瓣。"""
        lens = self._cpu_lens()
        ks = 64
        pt = [0.7, 0.0, -5000.0]
        peak_false = float(lens.psf(points=pt, ks=ks, recenter=False).max())
        peak_true = float(lens.psf(points=pt, ks=ks, recenter=True).max())
        assert peak_false >= 0.9 * peak_true


class TestDiffractiveLensDeviceTransfer:
    """测试设备转移。"""

    def test_to_cpu(self, sample_diffraclens):
        """to(cpu) 应将所有张量移动到 CPU。"""
        lens = sample_diffraclens
        lens.to(torch.device("cpu"))
        assert lens.d_sensor.device.type == "cpu"
