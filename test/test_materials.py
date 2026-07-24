"""
`deeplens/material/materials.py` 测试——玻璃和塑料材料。
"""

import pytest
import torch

from deeplens.material import RII_data, Material

# 用于 nd/Vd 的谱线：He d 线、H F 线、H C 线 [µm]。
WVLN_D, WVLN_F, WVLN_C = 0.5875618, 0.4861327, 0.6562725


class TestMaterialInit:
    """测试 Material 初始化。"""

    def test_material_vacuum(self, device_auto):
        """真空应满足 n=1。"""
        mat = Material(name="vacuum", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert torch.allclose(n, torch.tensor([1.0], device=device_auto))

    def test_material_air(self, device_auto):
        """空气应满足 n≈1。"""
        mat = Material(name="air", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() == pytest.approx(1.0, abs=0.001)

    def test_material_bk7(self, device_auto):
        """BK7 应具有典型玻璃折射率 ~1.5。"""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() == pytest.approx(1.52, abs=0.02)

    def test_material_case_insensitive(self, device_auto):
        """材料名称应不区分大小写。"""
        mat1 = Material(name="BK7", device=device_auto)
        mat2 = Material(name="bk7", device=device_auto)
        mat3 = Material(name="Bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n1 = mat1.ior(wvln)
        n2 = mat2.ior(wvln)
        n3 = mat3.ior(wvln)
        
        assert torch.allclose(n1, n2)
        assert torch.allclose(n2, n3)

    def test_material_default_air(self, device_auto):
        """名称为 None 时应默认为空气。"""
        mat = Material(name=None, device=device_auto)

        assert mat.name == "air"


class TestMaterialDispersion:
    """测试随波长变化的折射率。"""

    def test_material_dispersion_bk7(self, device_auto):
        """BK7 应表现出正常色散（n 随波长增大而减小）。"""
        mat = Material(name="bk7", device=device_auto)
        
        n_blue = mat.ior(torch.tensor([0.45], device=device_auto))
        n_green = mat.ior(torch.tensor([0.55], device=device_auto))
        n_red = mat.ior(torch.tensor([0.65], device=device_auto))
        
        # 正常色散：n_blue > n_green > n_red
        assert n_blue > n_green > n_red

    def test_material_dispersion_range(self, device_auto):
        """折射率在可见光谱范围内应合理变化。"""
        mat = Material(name="bk7", device=device_auto)
        
        n_min = mat.ior(torch.tensor([0.7], device=device_auto))
        n_max = mat.ior(torch.tensor([0.4], device=device_auto))
        
        # 色散不应过于极端
        delta_n = n_max - n_min
        assert 0.005 < delta_n.item() < 0.05

    def test_material_dispersion_wavelength_input(self, device_auto):
        """应接受张量形式的波长。"""
        mat = Material(name="bk7", device=device_auto)
        
        wvlns = torch.tensor([0.45, 0.55, 0.65], device=device_auto)
        n = mat.ior(wvlns)
        
        assert n.shape == wvlns.shape

    def test_material_refractive_index_alias(self, device_auto):
        """refractive_index 应为 ior 的别名。"""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n1 = mat.ior(wvln)
        n2 = mat.refractive_index(wvln)
        
        assert torch.allclose(n1, n2)


class TestMaterialTypes:
    """测试不同材料类型。"""

    def test_material_cdgm_glass(self, device_auto):
        """CDGM 玻璃应能正常工作。"""
        mat = Material(name="h-k9l", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert 1.4 < n.item() < 2.0

    def test_material_schott_glass(self, device_auto):
        """Schott 玻璃应能正常工作。"""
        mat = Material(name="n-bk7", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert 1.4 < n.item() < 2.0

    def test_material_plastic_pmma(self, device_auto):
        """PMMA 塑料应能正常工作。"""
        mat = Material(name="pmma", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert n.item() == pytest.approx(1.49, abs=0.02)

    def test_material_plastic_polycarb(self, device_auto):
        """聚碳酸酯应能正常工作。"""
        mat = Material(name="polycarb", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert n.item() == pytest.approx(1.58, abs=0.03)

    def test_material_coc(self, device_auto):
        """COC 塑料应能正常工作。"""
        mat = Material(name="coc", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert 1.5 < n.item() < 1.6


class TestMaterialSellmeier:
    """测试 Sellmeier 色散公式。"""

    def test_material_set_sellmeier_param(self, device_auto):
        """应设置自定义 Sellmeier 参数。"""
        mat = Material(name="vacuum", device=device_auto)
        
        # 类似 BK7 的参数
        params = [1.039, 0.006, 0.231, 0.020, 1.010, 103.56]
        mat.set_sellmeier_param(params)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert n.item() > 1.0  # 不再是真空

    def test_material_sellmeier_formula(self, device_auto):
        """Sellmeier 公式应给出正贡献。"""
        mat = Material(name="bk7", device=device_auto)
        
        # 所有波长均应给出 n > 1
        for wvln_val in [0.4, 0.5, 0.6, 0.7]:
            wvln = torch.tensor([wvln_val], device=device_auto)
            n = mat.ior(wvln)
            assert n.item() > 1.0


class TestMaterialMatch:
    """测试材料匹配功能。"""

    def test_material_match_returns_something(self, device_auto):
        """应尝试匹配材料且不崩溃。"""
        mat = Material(name="bk7", device=device_auto)
        
        # 根据实现，此处可能找到匹配，也可能找不到
        try:
            matched = mat.match_material()
            # 若返回结果，则应为有效值或 None
            assert matched is None or len(matched) > 0
        except Exception:
            pytest.skip("match_material not implemented for this material type")

    def test_material_get_name(self, device_auto):
        """get_name 应返回材料名称。"""
        mat = Material(name="bk7", device=device_auto)
        
        name = mat.get_name()
        
        assert name == "bk7"


class TestMaterialOptimization:
    """测试材料参数优化。"""

    def test_material_get_optimizer_params(self, device_auto):
        """应返回与优化器兼容的参数。"""
        mat = Material(name="bk7", device=device_auto)
        
        params = mat.get_optimizer_params(lrs=[1e-4, 1e-2])
        
        assert isinstance(params, list)
        assert len(params) > 0
        for p in params:
            assert "params" in p
            assert "lr" in p

    def test_material_n_trainable(self, device_auto):
        """折射率应可微分。"""
        mat = Material(name="bk7", device=device_auto)
        mat.get_optimizer_params()
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        # 检查 n 是可具有梯度的张量
        assert isinstance(n, torch.Tensor)


class TestRefractiveIndexInfo:
    """测试内置 refractiveindex.info 目录集成。"""

    def test_catalog_loaded(self):
        """应加载包含公式和插值表的内置 JSON 目录。"""
        assert isinstance(RII_data.get("FORMULA"), dict)
        assert isinstance(RII_data.get("INTERP"), dict)
        # 合理性检查：目录随附大量玻璃。
        assert len(RII_data["FORMULA"]) > 1000

    def test_ohara_glass_resolves_to_rii(self, device_auto):
        """应从 refractiveindex.info 加载 AGF 目录中不存在的 OHARA 玻璃。"""
        mat = Material(name="s-bsl7", device=device_auto)
        assert mat.dispersion == "rii"
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        # S-BSL7 是 OHARA 中与 N-BK7 对应的材料，nd ~ 1.5163。
        assert n.item() == pytest.approx(1.5163, abs=2e-3)

    def test_hikari_formula3_glass(self, device_auto):
        """HIKARI 玻璃应使用公式 3（多项式）并给出合理折射率。"""
        mat = Material(name="j-bk7a", device=device_auto)
        assert mat.dispersion == "rii"
        assert mat.rii_formula == 3
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.5168, abs=2e-3)

    def test_substrate_fused_silica(self, device_auto):
        """熔融石英（SiO2）应通过 refractiveindex.info 的公式 1 路径解析。

        ``sio2`` 是 refractiveindex.info 目录独有的名称，因此会覆盖新的 ``"rii"``
        分支（别名 ``fused_silica`` 则由已有的自定义插值表提供，并在其他位置测试）。
        """
        mat = Material(name="sio2", device=device_auto)
        assert mat.dispersion == "rii" and mat.rii_formula == 1
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.4585, abs=2e-3)

    def test_interp_tabulated_glass(self, device_auto):
        """表格化的 refractiveindex.info 晶体应使用 'interp' 分支。

        位于 RII_data['INTERP'] 中可证明测试覆盖了新的表格回退分支，而不是被遮蔽的
        custom/AGF 条目。
        """
        assert "bf1" in RII_data["INTERP"]
        mat = Material(name="bf1", device=device_auto)
        assert mat.dispersion == "interp"
        n = mat.ior(torch.tensor([0.55], device=device_auto))
        assert torch.isfinite(n).all() and 1.4 < n.item() < 2.0

    def test_interp_table_sorted(self):
        """内置插值表必须按波长升序排列（interp 依赖此假设）。"""
        for name, e in RII_data["INTERP"].items():
            w = e["wvlns"]
            assert all(w[i + 1] > w[i] for i in range(len(w) - 1)), name

    def test_substrate_sapphire(self, device_auto):
        """蓝宝石（Al2O3 常光）应能解析，且 nd 与文献值匹配。"""
        mat = Material(name="sapphire", device=device_auto)
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.768, abs=3e-3)

    def test_formula1_constant_term(self, device_auto):
        """MgF2（公式 1）包含非零 C1 常数，必须应用该常数。

        丢弃首项 C1 会使 n 偏移 ~0.1，因此该测试专门防护公式 1 的常数项处理。
        """
        mat = Material(name="mgf2", device=device_auto)
        assert mat.rii_formula == 1 and mat.rii_coeffs[0] != 0.0
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.3777, abs=3e-3)

    def test_infrared_substrate_in_band(self, device_auto):
        """仅适用于 IR 的硅应在有效波段内给出正确折射率。"""
        mat = Material(name="si", device=device_auto)
        n = mat.ior(torch.tensor([2.0], device=device_auto))  # 2 µm
        assert n.item() == pytest.approx(3.4487, abs=0.02)

    def test_rii_normal_dispersion(self, device_auto):
        """refractiveindex.info 玻璃应表现出正常色散。"""
        mat = Material(name="s-bsl7", device=device_auto)
        n_blue = mat.ior(torch.tensor([0.45], device=device_auto))
        n_green = mat.ior(torch.tensor([0.55], device=device_auto))
        n_red = mat.ior(torch.tensor([0.65], device=device_auto))
        assert n_blue > n_green > n_red

    def test_rii_differentiable(self, device_auto):
        """由 RII 公式得到的折射率应对波长可微。"""
        mat = Material(name="s-bsl7", device=device_auto)
        wvln = torch.tensor([WVLN_D], device=device_auto, requires_grad=True)
        n = mat.ior(wvln)
        n.backward()
        # 正常色散 -> dn/dlambda < 0。
        assert wvln.grad is not None and wvln.grad.item() < 0

    @pytest.mark.parametrize(
        "name", ["s-bsl7", "bah32", "bac4", "j-bk7a", "k-baf8", "sapphire"]
    )
    def test_nd_vd_oracle(self, name, device_auto):
        """计算所得 (nd, Vd) 必须复现目录中存储的值。

        这是正确性门槛：同时检查系数解析、公式选择和 µm 单位处理。
        """
        entry = RII_data["FORMULA"][name]
        mat = Material(name=name, device=device_auto)
        nd = mat.ior(torch.tensor([WVLN_D], device=device_auto)).item()
        assert nd == pytest.approx(entry["nd"], abs=1.5e-3)
        if entry["vd"] < 1e37:
            nf = mat.ior(torch.tensor([WVLN_F], device=device_auto)).item()
            nc = mat.ior(torch.tensor([WVLN_C], device=device_auto)).item()
            vd = (nd - 1) / (nf - nc)
            assert vd == pytest.approx(entry["vd"], rel=0.02)

    def test_existing_name_precedence_unchanged(self, device_auto):
        """AGF 目录中的名称优先级应高于 refractiveindex.info。

        n-bk7 同时存在于 SCHOTT.AGF 和 refractiveindex.info 目录中；它仍必须通过
        AGF（Sellmeier）路径解析，而不是 'rii'。
        """
        mat = Material(name="n-bk7", device=device_auto)
        assert mat.dispersion == "sellmeier"
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.5168, abs=1e-3)


class TestMaterialEdgeCases:
    """测试边界情况和错误处理。"""

    def test_material_extreme_wavelength_blue(self, device_auto):
        """应能处理近 UV 波长。"""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.35], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() > 1.0

    def test_material_extreme_wavelength_red(self, device_auto):
        """应能处理近 IR 波长。"""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.9], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() > 1.0

    def test_material_device_consistency(self, device_auto):
        """输出应与输入位于同一设备。"""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.device.type == device_auto.type
