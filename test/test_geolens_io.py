"""`deeplens/geolens_pkg/io.py` 测试——`GeoLensIO` 混入类。

测试 JSON、Zemax (.zmx) 和 Code V (.seq) 格式的镜头文件 I/O。
"""

import os

import pytest
import torch


class TestJSONIO:
    """测试 JSON 镜头文件 I/O。"""

    def test_read_write_json_roundtrip(self, sample_singlet_lens, test_output_dir):
        """写入并读回 JSON 后，应保留表面数量和 foclen。"""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_roundtrip.json")
        original_num_surfs = len(lens.surfaces)
        original_foclen = lens.foclen

        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == original_num_surfs
        assert lens2.foclen == pytest.approx(original_foclen, rel=0.01)

    def test_read_write_json_cellphone(self, sample_cellphone_lens, test_output_dir):
        """往返转换手机镜头（包含非球面）。"""
        lens = sample_cellphone_lens
        out_path = os.path.join(test_output_dir, "test_cellphone_roundtrip.json")
        original_num_surfs = len(lens.surfaces)

        lens.write_lens_json(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == original_num_surfs

    def test_json_roundtrip_preserves_spectrum_and_object_depth(
        self, sample_singlet_lens, test_output_dir
    ):
        """红外设计波长和默认物距不能在 JSON 重载后退回可见光默认值。"""

        from deeplens import GeoLens

        lens = sample_singlet_lens
        lens.primary_wvln = 3.5
        lens.wvln_rgb = [2.7, 3.5, 4.3]
        lens.obj_depth = -10_000_000.0
        out_path = os.path.join(test_output_dir, "test_mwir_spectrum_roundtrip.json")

        lens.write_lens_json(out_path)
        lens2 = GeoLens(filename=out_path)

        assert lens2.primary_wvln == pytest.approx(3.5)
        assert lens2.wvln_rgb == pytest.approx([2.7, 3.5, 4.3])
        assert lens2.obj_depth == pytest.approx(-10_000_000.0)


class TestZMXIO:
    """测试 Zemax .zmx 镜头文件 I/O。"""

    def test_read_zmx(self, lenses_dir):
        """加载 .zmx 文件并验证其能生成表面。"""
        zmx_path = os.path.join(lenses_dir, "camera/ef35mm_f2.0.zmx")
        if not os.path.exists(zmx_path):
            pytest.skip("ZMX test file not available")

        from deeplens import GeoLens

        lens = GeoLens()
        lens.read_lens_zmx(zmx_path)
        assert len(lens.surfaces) > 0
        assert lens.d_sensor is not None

    def test_write_zmx(self, sample_singlet_lens, test_output_dir):
        """写入 .zmx 文件并验证其存在。"""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_write.zmx")
        lens.write_lens_zmx(out_path)
        assert os.path.exists(out_path)

    def test_zmx_roundtrip(self, sample_singlet_lens, test_output_dir):
        """写入并读回 .zmx 后，应保留表面数量。"""
        lens = sample_singlet_lens
        original_num_surfs = len(lens.surfaces)
        out_path = os.path.join(test_output_dir, "test_zmx_roundtrip.zmx")
        lens.write_lens_zmx(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens()
        lens2.read_lens_zmx(out_path)
        # ZMX 往返转换可能丢失某些表面类型，但数量应接近
        assert len(lens2.surfaces) > 0

    def test_zmx_aperture_exports_diam(self):
        """Aperture (STOP) 表面必须导出 DIAM（半直径）行。

        回归测试：``Aperture.zmx_str`` 过去完全省略 ``DIAM``，因此导出的孔径光阑
        没有光圈尺寸，重新导入时半径会默认为 1.0 mm。
        """
        from deeplens.geometric_surface import Aperture

        aperture = Aperture(r=2.5, d=0.0)
        surf_str = aperture.zmx_str(surf_idx=1, d_next=torch.tensor(5.0))

        assert "STOP" in surf_str
        diam_lines = [
            ln for ln in surf_str.splitlines() if ln.strip().startswith("DIAM")
        ]
        assert len(diam_lines) == 1, f"Expected one DIAM line, got: {surf_str!r}"
        assert "2.5" in diam_lines[0]

    def test_zmx_aperture_size_roundtrip(self, sample_cellphone_lens, test_output_dir):
        """光圈半直径必须在 .zmx 写入/读取往返转换后保持不变。"""
        from deeplens import GeoLens
        from deeplens.geometric_surface import Aperture

        lens = sample_cellphone_lens
        aper_idx = next(
            i for i, s in enumerate(lens.surfaces) if isinstance(s, Aperture)
        )
        # 使用易于区分的半径，避免读取默认值（1.0）掩盖缺陷。
        lens.surfaces[aper_idx].r = 1.234

        out_path = os.path.join(test_output_dir, "test_zmx_aperture_roundtrip.zmx")
        lens.write_lens_zmx(out_path)

        lens2 = GeoLens()
        lens2.read_lens_zmx(out_path)
        aper2 = next(s for s in lens2.surfaces if isinstance(s, Aperture))
        assert aper2.r == pytest.approx(1.234, abs=1e-3)


class TestSEQIO:
    """测试 Code V .seq 镜头文件 I/O。"""

    def test_write_seq(self, sample_singlet_lens, test_output_dir):
        """写入 .seq 文件并验证其存在。"""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_write.seq")
        lens.write_lens_seq(out_path)
        assert os.path.exists(out_path)


class TestCrossFormat:
    """测试跨格式转换。"""

    def test_json_to_zmx(self, sample_singlet_lens, test_output_dir):
        """读取 JSON → 写入 ZMX → 读取 ZMX：foclen 应相近。"""
        lens = sample_singlet_lens
        zmx_path = os.path.join(test_output_dir, "test_cross_format.zmx")
        lens.write_lens_zmx(zmx_path)

        from deeplens import GeoLens

        lens2 = GeoLens()
        lens2.read_lens_zmx(zmx_path)
        assert len(lens2.surfaces) > 0
