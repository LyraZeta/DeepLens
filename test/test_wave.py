"""
deeplens/optics/wave.py 测试——波动光学与传播。
"""

import pytest
import torch


from deeplens.light import ComplexWave, AngularSpectrumMethod


class TestComplexWaveInit:
    """测试 ComplexWave 初始化。"""

    def test_complex_wave_init_default(self, device_auto):
        """应使用默认参数初始化。"""
        wave = ComplexWave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        assert wave.wvln == 0.55
        assert (wave.z == 0.0).all().item()
        assert wave.phy_size == (4.0, 4.0)
        assert wave.res == (256, 256)

    def test_complex_wave_init_with_field(self, device_auto):
        """应使用自定义光场初始化。"""
        u = torch.ones(256, 256, dtype=torch.complex64, device=device_auto)
        
        wave = ComplexWave(
            u=u,
            wvln=0.55,
            phy_size=(4.0, 4.0),
        )
        
        assert wave.u.shape[-2:] == (256, 256)


class TestComplexWavePointWave:
    """测试点光源球面波。"""

    def test_point_wave_center(self, device_auto):
        """中心处的点波应对称。"""
        wave = ComplexWave.point_wave(
            point=(0, 0, -1000.0),
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        assert wave.u.shape[-2:] == (256, 256)
        # 检查近似对称性
        irr = torch.abs(wave.u)**2
        assert torch.allclose(irr[0, 0, 128, 64], irr[0, 0, 128, 192], rtol=0.1)

    def test_point_wave_intensity(self, device_auto):
        """点波应具有非零强度。"""
        wave = ComplexWave.point_wave(
            point=(0, 0, -1000.0),
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        irr = torch.abs(wave.u)**2
        irr = torch.abs(wave.u)**2
        assert irr.sum().item() > 0


class TestComplexWavePlaneWave:
    """测试平面波初始化。"""

    def test_plane_wave_uniform(self, device_auto):
        """平面波应具有均匀振幅。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        amp = torch.abs(wave.u)
        # 所有振幅应相等（在数值精度范围内）
        # 所有振幅应相等（在数值精度范围内）
        assert torch.allclose(amp, amp[0, 0, 0, 0].expand_as(amp), atol=1e-5)

    def test_plane_wave_with_valid_r(self, device_auto):
        """平面波应遵循有效半径。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
            valid_r=1.0,
        )
        
        # 角点应为零（位于有效半径之外）
        # 角点应为零（位于有效半径之外）
        assert wave.u[0, 0, 0, 0].abs().item() == pytest.approx(0.0, abs=1e-5)


class TestComplexWaveImageWave:
    """测试图像调制波。"""

    def test_image_wave(self, device_auto):
        """应从图像创建波。"""
        img = torch.rand(256, 256, device=device_auto)
        
        wave = ComplexWave.image_wave(
            img=img,
            wvln=0.55,
            phy_size=(4.0, 4.0),
        )
        
        # 振幅应与 sqrt(image) 匹配
        amp = torch.abs(wave.u)
        expected = torch.sqrt(img)
        assert torch.allclose(amp, expected, atol=1e-4)


class TestComplexWavePropagation:
    """测试波传播。"""

    def test_wave_prop_distance(self, device_auto):
        """传播应更新 z 坐标。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        wave.prop(prop_dist=10.0)
        
        assert (wave.z == 10.0).all().item()

    def test_wave_prop_to(self, device_auto):
        """prop_to 应传播到指定 z。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        wave.prop_to(z=10.0)
        
        assert (wave.z == 10.0).all().item()

    def test_wave_prop_energy_conservation(self, device_auto):
        """传播应守恒能量。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        energy_before = (torch.abs(wave.u)**2).sum()
        wave.prop(prop_dist=10.0)
        energy_after = (torch.abs(wave.u)**2).sum()
        
        assert torch.allclose(energy_before, energy_after, rtol=0.1)

    def test_wave_prop_with_refractive_index(self, device_auto):
        """传播应考虑折射率。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        # 在 n=1.5 的介质中传播
        wave.prop(prop_dist=10.0, n=1.5)
        
        assert (wave.z == 10.0).all().item()


class TestAngularSpectrumMethod:
    """测试角谱法传播。"""

    def test_asm_basic(self, device_auto):
        """ASM 应传播光场。"""
        u = torch.ones(256, 256, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=10.0,
            wvln=0.55,
            ps=0.01,  # 像素尺寸 [mm]
        )
        
        assert u_prop.shape == u.shape

    def test_asm_zero_distance(self, device_auto):
        """传播距离为零时应返回相同光场。"""
        u = torch.rand(256, 256, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=0.0,
            wvln=0.55,
            ps=0.01,
        )
        
        assert torch.allclose(u_prop, u, atol=1e-5)

    def test_asm_with_padding(self, device_auto):
        """带填充的 ASM 应避免混叠。"""
        u = torch.ones(128, 128, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=10.0,
            wvln=0.55,
            ps=0.01,
            padding=True,
        )
        
        assert u_prop.shape == u.shape

    def test_asm_batch(self, device_auto):
        """ASM 应支持 batch 维度。"""
        u = torch.ones(1, 1, 256, 256, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=10.0,
            wvln=0.55,
            ps=0.01,
        )
        
        assert u_prop.shape == u.shape


class TestBandLimitedASM:
    """测试带限角谱法（Matsushima & Shimobaba 2009）。"""

    def test_bandlimited_reduces_to_asm_at_short_range(self, device_auto):
        """短距离下，带限窗口覆盖整个频谱，因此带限 ASM 必须与标准 ASM 精确匹配。"""
        from deeplens.light import BandLimitedASM

        u = torch.rand(256, 256, dtype=torch.complex64, device=device_auto)
        kwargs = dict(z=1.0, wvln=0.55, ps=0.01)

        u_bl = BandLimitedASM(u=u, **kwargs)
        u_asm = AngularSpectrumMethod(u=u, **kwargs)

        assert torch.allclose(u_bl, u_asm, atol=1e-5)

    def test_bandlimited_asm_runs_at_long_range(self, device_auto):
        """带限 ASM 应能在标准 ASM 会发生混叠的长距离下运行，并返回 shape 相同的
        有限光场。"""
        from deeplens.light import BandLimitedASM

        u = torch.ones(256, 256, dtype=torch.complex64, device=device_auto)

        u_prop = BandLimitedASM(u=u, z=200.0, wvln=0.55, ps=0.01)

        assert u_prop.shape == u.shape
        assert torch.isfinite(u_prop.real).all()
        assert torch.isfinite(u_prop.imag).all()

    def test_bandlimited_asm_suppresses_aliasing_at_long_range(self, device_auto):
        """在标准 ASM 欠采样的距离下，带限处理必须丢弃容易混叠的高频，因此带限
        ASM 携带的能量不应多于标准 ASM（对锐利光圈应严格更少）。"""
        from deeplens.light import BandLimitedASM

        # 锐利的居中光圈 -> 宽频谱，在长距离下会发生混叠。
        u = torch.zeros(256, 256, dtype=torch.complex64, device=device_auto)
        u[120:136, 120:136] = 1.0
        kwargs = dict(z=200.0, wvln=0.55, ps=0.01, padding=False)

        e_bl = (BandLimitedASM(u=u, **kwargs).abs() ** 2).sum()
        e_asm = (AngularSpectrumMethod(u=u, **kwargs).abs() ** 2).sum()

        assert e_bl > 0
        assert e_bl < e_asm

    def test_prop_handles_intermediate_distance(self, device_auto):
        """prop() 应通过带限 ASM 处理 ASM 与 Fresnel 适用范围之间的距离（过去不支持
        并会抛出异常）。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(2.56, 2.56),  # ps=0.01 -> asm_zmax~46mm, fresnel_zmin~4654mm
            res=(256, 256),
        )

        wave.prop(prop_dist=200.0)  # 200 mm 位于此前的空缺范围内

        assert (wave.z == 200.0).all().item()
        assert torch.isfinite(wave.u.real).all()


class TestComplexWaveGrids:
    """测试网格生成。"""

    def test_gen_xy_grid(self, device_auto):
        """应生成正确的 xy 网格。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        wave.gen_xy_grid()
        
        assert hasattr(wave, "x")
        assert hasattr(wave, "y")
        assert wave.x.shape == (256, 256)
        assert wave.y.shape == (256, 256)

    def test_gen_freq_grid(self, device_auto):
        """应生成频率网格。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        fx, fy = wave.gen_freq_grid()
        
        assert fx is not None
        assert fy is not None


class TestComplexWaveOperations:
    """测试波操作。"""

    def test_wave_pad(self, device_auto):
        """填充应增大分辨率和物理尺寸。"""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        original_size = wave.phy_size
        wave.pad(Hpad=64, Wpad=64)
        
        assert wave.u.shape[-2:] == (256 + 128, 256 + 128)
        assert wave.phy_size[0] > original_size[0]

    def test_wave_flip(self, device_auto):
        """flip 应翻转光场。"""
        wave = ComplexWave(
            u=torch.arange(16).reshape(4, 4).float().to(dtype=torch.complex64),
            wvln=0.55,
            phy_size=(4.0, 4.0),
        )
        
        original_u = wave.u.clone()
        wave.flip()
        
        # 检查角点已交换
        # 检查角点已交换
        assert wave.u[0, 0, 0, 0] == original_u[0, 0, -1, -1]


class TestComplexWaveIO:
    """测试保存/加载功能。"""

    def test_wave_save_load(self, device_auto, test_output_dir):
        """应保存并加载波。"""
        import os
        
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(128, 128),
        )
        
        filepath = os.path.join(test_output_dir, "test_wave.npz")
        wave.save(filepath)
        
        assert os.path.exists(filepath)
        
        # 重新加载
        wave2 = ComplexWave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(128, 128),
        )
        wave2.load(filepath)
        
        assert wave2.u.shape == wave.u.shape


class TestComplexWaveVisualization:
    """测试可视化方法。"""

    def test_wave_show_irradiance(self, device_auto, test_output_dir):
        """应保存辐照度图像。"""
        import os
        
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(128, 128),
        )
        
        save_path = os.path.join(test_output_dir, "wave_irr.png")
        wave.show(save_name=save_path, data="irr")
        
        assert os.path.exists(save_path)
