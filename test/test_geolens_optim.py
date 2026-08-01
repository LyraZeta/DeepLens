"""`deeplens/geolens_pkg/optim.py` 测试——`GeoLensOptim` 混入类。

所有方法均通过 `GeoLens` 实例测试（混入类架构）。
"""

import math

import pytest
import torch


class TestOptimizerHelpers:
    """测试优化器参数收集。"""

    def test_get_optimizer_params_returns_list(self, sample_singlet_lens):
        """get_optimizer_params 返回非空的参数字典列表。"""
        lens = sample_singlet_lens
        params = lens.get_optimizer_params()
        assert isinstance(params, list)
        assert len(params) > 0
        for p in params:
            assert "params" in p
            assert "lr" in p

    def test_get_optimizer_returns_adam(self, sample_singlet_lens):
        """get_optimizer 返回 Adam 优化器。"""
        lens = sample_singlet_lens
        optimizer = lens.get_optimizer()
        assert isinstance(optimizer, torch.optim.Adam)

    def test_gradient_clipping_is_independent_between_parameter_groups(
        self, sample_singlet_lens
    ):
        """一个参数组的极大梯度不应缩放其他参数组。"""
        large = torch.nn.Parameter(torch.tensor([0.0]))
        moderate = torch.nn.Parameter(torch.tensor([0.0]))
        optimizer = torch.optim.Adam(
            [{"params": [large]}, {"params": [moderate]}], lr=1e-3
        )
        large.grad = torch.tensor([1000.0])
        moderate.grad = torch.tensor([3.0])

        params, nonfinite = sample_singlet_lens._sanitize_and_clip_gradients(
            optimizer, max_norm=10.0
        )

        assert len(params) == 2
        assert params[0] is large
        assert params[1] is moderate
        assert nonfinite == 0
        assert large.grad.norm().item() == pytest.approx(10.0, rel=1e-5)
        assert moderate.grad.item() == pytest.approx(3.0)

    def test_gradient_sanitizer_replaces_nonfinite_values(self, sample_singlet_lens):
        """NaN/Inf 梯度应在交给 Adam 前替换为零。"""
        parameter = torch.nn.Parameter(torch.zeros(3))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        parameter.grad = torch.tensor([float("nan"), float("inf"), 2.0])

        _, nonfinite = sample_singlet_lens._sanitize_and_clip_gradients(optimizer)

        assert nonfinite == 2
        assert torch.isfinite(parameter.grad).all()
        assert torch.equal(parameter.grad, torch.tensor([0.0, 0.0, 2.0]))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"iterations": 0}, "iterations"),
            ({"iterations": 1, "test_per_iter": 0}, "test_per_iter"),
        ],
    )
    def test_optimize_rejects_nonpositive_loop_counts(
        self, sample_singlet_lens, kwargs, message
    ):
        """训练次数和检查点间隔必须为正，避免隐藏的额外更新或除零。"""

        with pytest.raises(ValueError, match=message):
            sample_singlet_lens.optimize(**kwargs)

    @pytest.mark.parametrize(
        ("iterations", "expected"),
        [(1, 0), (9, 0), (10, 1), (250, 25), (2000, 100)],
    )
    def test_short_optimization_does_not_spend_only_step_in_warmup(
        self, sample_singlet_lens, iterations, expected
    ):
        """短烟雾测试必须有非零学习率，长训练预热最多 100 步。"""

        assert (
            sample_singlet_lens._optimization_warmup_steps(iterations) == expected
        )

    @pytest.mark.parametrize(
        "name",
        [
            "w_rms",
            "w_mtf",
            "w_valid",
            "w_field",
            "w_reg",
            "mtf_max_weight",
            "field_mapping_max_weight",
        ],
    )
    @pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
    def test_optimize_rejects_invalid_loss_weights(
        self, sample_singlet_lens, name, value
    ):
        """各损失权重必须非负且有限，不能让总损失静默污染。"""

        with pytest.raises(ValueError, match=name):
            sample_singlet_lens.optimize(iterations=1, **{name: value})

    def test_optimize_requires_complete_first_order_guard_configuration(
        self, sample_singlet_lens
    ):
        """一阶硬门控的目标和两个误差门限必须成组提供。"""

        with pytest.raises(ValueError, match="同时提供"):
            sample_singlet_lens.optimize(
                iterations=1,
                target_f_number=2.0,
                first_order_preferred_relative_error=0.008,
            )

    def test_optimize_rejects_initial_state_outside_first_order_hard_limit(
        self, sample_singlet_lens, monkeypatch
    ):
        """初始 EFL 或 F 数超过硬上限时不应进入训练循环。"""

        monkeypatch.setattr(
            sample_singlet_lens,
            "_measure_first_order_state",
            lambda: (102.0, 2.0),
        )
        with pytest.raises(ValueError, match="初始处方超出一阶硬上限"):
            sample_singlet_lens.optimize(
                iterations=1,
                target_focal_length=100.0,
                target_f_number=2.0,
                first_order_preferred_relative_error=0.008,
                first_order_hard_relative_error=0.01,
            )

    def test_optimize_requires_frequency_when_mtf_surrogate_is_enabled(
        self, sample_singlet_lens
    ):
        """启用 MTF 代理时不能静默缺失目标频率。"""

        with pytest.raises(ValueError, match="mtf_frequency_cy_mm"):
            sample_singlet_lens.optimize(iterations=1, w_mtf=0.1)

    @pytest.mark.parametrize("interval", [-1, 0.5])
    def test_optimize_rejects_invalid_resample_interval(
        self, sample_singlet_lens, interval
    ):
        """训练光线重采样间隔必须为非负整数。"""

        with pytest.raises(ValueError, match="ray_resample_interval"):
            sample_singlet_lens.optimize(
                iterations=1, ray_resample_interval=interval
            )


class TestConstraints:
    """测试约束初始化。"""

    def test_init_constraints_sets_attrs(self, sample_singlet_lens):
        """init_constraints 在镜头上设置约束属性。"""
        lens = sample_singlet_lens
        lens.init_constraints()
        assert hasattr(lens, "air_edge_min")
        assert hasattr(lens, "thick_center_min")
        assert hasattr(lens, "sag2diam_max")
        assert hasattr(lens, "chief_ray_angle_max")
        assert hasattr(lens, "ttl_min")
        assert hasattr(lens, "surf_angle_max")
        assert hasattr(lens, "bend_angle_max")

    def test_init_constraints_cellphone_vs_camera(
        self, sample_cellphone_lens, sample_camera_lens
    ):
        """手机镜头和相机镜头应获得不同的约束值。"""
        sample_cellphone_lens.init_constraints()
        sample_camera_lens.init_constraints()
        # 手机镜头的约束更严格
        assert sample_cellphone_lens.air_edge_min < sample_camera_lens.air_edge_min

    def test_constraint_overrides_survive_post_computation(self, sample_camera_lens):
        """任务专用约束应在重新计算一阶量后继续生效。"""

        lens = sample_camera_lens
        lens.init_constraints({"ttl_max": 1234.0, "distortion_max": 0.005})
        lens.post_computation()

        assert lens.ttl_max == 1234.0
        assert lens.distortion_max == 0.005

    def test_unknown_constraint_override_is_rejected(self, sample_camera_lens):
        """拼错的约束名不应静默创建无效属性。"""

        with pytest.raises(ValueError, match="未知镜头约束参数"):
            sample_camera_lens.init_constraints({"ttl_mx": 1234.0})


class TestLossFunctions:
    """测试各个损失函数。"""

    def test_loss_reg_returns_tensor_and_dict(self, sample_singlet_lens):
        """loss_reg 返回（标量张量，字典）。"""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss, loss_dict = lens.loss_reg()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert isinstance(loss_dict, dict)
        assert "loss_clearance" in loss_dict
        assert "loss_envelope" in loss_dict
        assert "loss_profile" in loss_dict
        assert "loss_cra" in loss_dict
        assert "loss_ray_bend" in loss_dict

    def test_loss_infocus_scalar(self, sample_singlet_lens):
        """loss_infocus 返回 >= 0 的标量。"""
        lens = sample_singlet_lens
        loss = lens.loss_infocus()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_profile_scalar(self, sample_singlet_lens):
        """loss_profile 返回 >= 0 的标量张量。"""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss = lens.loss_profile()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_bound_returns_tuple(self, sample_singlet_lens):
        """loss_bound 返回 (loss_clearance, loss_envelope)，二者均为 >= 0 的标量。"""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss_clearance, loss_envelope = lens.loss_bound()
        assert isinstance(loss_clearance, torch.Tensor)
        assert isinstance(loss_envelope, torch.Tensor)
        assert loss_clearance.dim() == 0
        assert loss_envelope.dim() == 0
        assert loss_clearance.item() >= 0
        assert loss_envelope.item() >= 0

    def test_loss_cra_scalar(self, sample_singlet_lens):
        """loss_cra 返回 >= 0 的标量张量。"""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss = lens.loss_cra()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_ray_bend_scalar(self, sample_singlet_lens):
        """loss_ray_bend 返回 >= 0 的标量张量。"""
        lens = sample_singlet_lens
        lens.init_constraints()
        loss = lens.loss_ray_bend()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_mat_scalar(self, sample_singlet_lens):
        """loss_mat 返回 >= 0 的标量。"""
        lens = sample_singlet_lens
        loss = lens.loss_mat()
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_rms_scalar(self, sample_singlet_lens):
        """loss_rms 返回 >= 0 的标量张量。"""
        lens = sample_singlet_lens
        loss = lens.loss_rms(num_grid=(2, 2), num_rays=128)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_loss_rms_reports_proxy_when_every_ray_is_invalid(
        self, sample_singlet_lens, monkeypatch
    ):
        """所有光线失效时，公开 RMS 指标也不应返回近零值。"""
        lens = sample_singlet_lens

        def invalidate_all(ray):
            ray.is_valid.zero_()
            return ray

        monkeypatch.setattr(lens, "trace2sensor", invalidate_all)
        loss = lens.loss_rms(num_grid=(2, 2), num_rays=16)

        expected_proxy = max(2.0 * lens.r_sensor, 1.0)
        assert loss.item() == pytest.approx(expected_proxy, rel=1e-5)

    def test_ray_validity_loss_ignores_fields_above_threshold(
        self, sample_singlet_lens
    ):
        """有效率达到阈值时不应产生挡光惩罚。"""
        ray_valid = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])

        loss, valid_ratio = sample_singlet_lens._ray_validity_loss(
            ray_valid, min_valid_ratio=0.5
        )

        assert loss.item() == pytest.approx(0.0)
        assert torch.equal(valid_ratio, torch.tensor([1.0, 0.5]))

    def test_ray_validity_loss_penalizes_partial_and_total_blocking(
        self, sample_singlet_lens
    ):
        """部分挡光和全挡光视场都必须增加总损失。"""
        ray_valid = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0]])

        loss, valid_ratio = sample_singlet_lens._ray_validity_loss(
            ray_valid, min_valid_ratio=0.5
        )

        # 短缺分别为 50% 和 100%，平方后平均：(0.25 + 1) / 2。
        assert loss.item() == pytest.approx(0.625)
        assert torch.equal(valid_ratio, torch.tensor([0.25, 0.0]))

    def test_full_invalid_field_uses_nonzero_rms_proxy(self, sample_singlet_lens):
        """全失效视场的 RMS 代理值不应再接近零。"""
        ray_err = torch.tensor(
            [
                [[3.0, 4.0], [float("inf"), float("inf")]],
                [[float("inf"), 0.0], [0.0, float("inf")]],
            ]
        )
        ray_valid = torch.tensor([[1, 0], [0, 0]])

        mse = sample_singlet_lens._masked_field_mse(
            ray_err, ray_valid, invalid_rms=3.0
        )

        assert torch.allclose(mse, torch.tensor([25.0, 9.0]))
        assert mse[1].sqrt().item() == pytest.approx(3.0)

    def test_fixed_frequency_geometric_mtf_is_translation_invariant_and_directional(
        self, sample_singlet_lens
    ):
        """固定频率几何 MTF 应忽略像点平移，并区分切向与弧矢方向。"""

        intercepts = torch.tensor([[[-0.01, 0.0], [0.01, 0.0]]])
        valid = torch.ones((1, 2))
        shifted = intercepts + torch.tensor([50.0, -30.0])

        mtf = sample_singlet_lens._fixed_frequency_geometric_mtf(
            intercepts, valid, frequency_cy_mm=25.0
        )
        shifted_mtf = sample_singlet_lens._fixed_frequency_geometric_mtf(
            shifted, valid, frequency_cy_mm=25.0
        )

        # Y 截距相同，因此切向 MTF 为 1；X 相位相差 pi，弧矢 MTF 接近 0。
        assert mtf[0, 0].item() == pytest.approx(1.0, rel=1e-6)
        assert mtf[0, 1].item() < 1e-4
        # 50 mm 大像高叠加 10 微米光斑时，float32 坐标量化会留下约 1e-4
        # 的相位幅值误差；质心化后仍应保持在该数值精度量级。
        assert torch.allclose(mtf, shifted_mtf, rtol=5e-4, atol=5e-4)

    def test_fixed_frequency_mtf_surrogate_uses_target_and_has_finite_gradient(
        self, sample_singlet_lens
    ):
        """相位方差代理在目标内为零，超差时提供稳定的恢复梯度。"""

        frequency = 10.0
        target_mtf = 0.5
        target_sigma = (
            (-2.0 * torch.log(torch.tensor(target_mtf))).sqrt()
            / (2.0 * torch.pi * frequency)
        )
        at_target = torch.tensor(
            [
                [-target_sigma, -target_sigma],
                [-target_sigma, target_sigma],
                [target_sigma, -target_sigma],
                [target_sigma, target_sigma],
            ]
        ).unsqueeze(0)
        valid = torch.ones((1, 4))

        inside = sample_singlet_lens._fixed_frequency_mtf_surrogate_violation(
            at_target * 0.9,
            valid,
            frequency_cy_mm=frequency,
            target_mtf=target_mtf,
            invalid_rms=1.0,
        )
        outside_points = (at_target * 2.0).detach().requires_grad_(True)
        outside = sample_singlet_lens._fixed_frequency_mtf_surrogate_violation(
            outside_points,
            valid,
            frequency_cy_mm=frequency,
            target_mtf=target_mtf,
            invalid_rms=1.0,
        )
        loss = outside.mean() + outside.max()
        loss.backward()

        assert torch.equal(inside, torch.zeros_like(inside))
        assert torch.allclose(
            outside,
            torch.full_like(outside, torch.log(torch.tensor(2.0))),
            rtol=1e-5,
            atol=1e-6,
        )
        assert torch.isfinite(outside_points.grad).all()
        assert outside_points.grad.abs().sum().item() > 0.0

    def test_target_field_mapping_loss_ignores_axis_and_uses_tolerance(
        self, sample_singlet_lens
    ):
        """轴上场点应被排除，容差内零损失，超差部分独立计入。"""

        target = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])
        inside = torch.tensor([[[2.0, -3.0]], [[10.04, 0.0]]])
        outside = torch.tensor([[[2.0, -3.0]], [[10.20, 0.0]]])

        inside_loss = sample_singlet_lens._target_field_mapping_loss(
            inside, target, tolerance=0.005
        )
        outside_loss = sample_singlet_lens._target_field_mapping_loss(
            outside, target, tolerance=0.005
        )

        assert inside_loss.item() == pytest.approx(0.0)
        # 2% 相对误差相对 0.5% 容差的超差量为 2%/0.5%-1 = 3。
        assert outside_loss.item() == pytest.approx(3.0, rel=1e-5)

    def test_target_field_mapping_loss_rejects_nonpositive_tolerance(
        self, sample_singlet_lens
    ):
        """像高映射容差必须为正数。"""

        points = torch.zeros((1, 1, 2))
        with pytest.raises(ValueError, match="tolerance"):
            sample_singlet_lens._target_field_mapping_loss(
                points, points, tolerance=0.0
            )

    def test_target_field_mapping_loss_has_finite_gradient(self, sample_singlet_lens):
        """超差像高损失应向质心位置提供有限、非零的恢复梯度。"""

        centroid = torch.tensor([[[0.0, 0.0]], [[10.2, 0.0]]], requires_grad=True)
        target = torch.tensor([[[0.0, 0.0]], [[10.0, 0.0]]])

        loss = sample_singlet_lens._target_field_mapping_loss(
            centroid, target, tolerance=0.005
        )
        loss.backward()

        assert torch.isfinite(centroid.grad).all()
        assert centroid.grad[1].abs().sum().item() > 0.0

    def test_target_field_mapping_loss_can_emphasize_worst_field(
        self, sample_singlet_lens
    ):
        """最坏场附加项应在平均超差之外显式惩罚最大超差。"""

        target = torch.tensor(
            [[[0.0, 0.0]], [[10.0, 0.0]], [[10.0, 0.0]]]
        )
        centroid = torch.tensor(
            [[[0.0, 0.0]], [[10.20, 0.0]], [[10.10, 0.0]]]
        )

        loss = sample_singlet_lens._target_field_mapping_loss(
            centroid,
            target,
            tolerance=0.005,
            max_weight=1.0,
        )

        # 两个离轴场超差分别为 3 和 1：平均 2，再加最坏场 3，合计 5。
        assert loss.item() == pytest.approx(5.0, rel=1e-5)

    def test_checkpoint_can_skip_expensive_analysis(
        self, sample_singlet_lens, monkeypatch, tmp_path
    ):
        """关闭检查点分析时仍应保存 JSON，但不能调用完整 analysis。"""

        calls = []
        monkeypatch.setattr(
            sample_singlet_lens,
            "write_lens_json",
            lambda path: calls.append(("json", path)),
        )
        monkeypatch.setattr(
            sample_singlet_lens,
            "analysis",
            lambda path: calls.append(("analysis", path)),
        )

        sample_singlet_lens._save_optimization_checkpoint(
            tmp_path, iteration=3, run_analysis=False
        )
        assert calls == [("json", f"{tmp_path}/iter3.json")]

        sample_singlet_lens._save_optimization_checkpoint(
            tmp_path, iteration=4, run_analysis=True
        )
        assert calls[-2:] == [
            ("json", f"{tmp_path}/iter4.json"),
            ("analysis", f"{tmp_path}/iter4"),
        ]

    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            (0.80, 0.70, True),
            (0.80, 0.69, False),
            (0.688, 0.700, True),
            (0.688, 0.688, True),
            (0.688, 0.625, False),
            (0.0, 0.0, True),
        ],
    )
    def test_validity_update_guard_uses_temporary_floor_below_target(
        self, sample_singlet_lens, before, after, expected
    ):
        """未达标初始结构可继续改善，但不允许有效率进一步下降。"""
        accepted = sample_singlet_lens._validity_update_is_acceptable(
            before, after, min_valid_ratio=0.7
        )

        assert accepted is expected

    @pytest.mark.parametrize(
        (
            "focal_before",
            "fnum_before",
            "focal_after",
            "fnum_after",
            "expected",
        ),
        [
            (100.6, 2.012, 100.79, 2.0158, True),
            (100.6, 2.012, 100.81, 2.012, False),
            (100.9, 2.018, 100.85, 2.017, True),
            (100.9, 2.018, 100.91, 2.018, False),
            (100.9, 2.018, 101.01, 2.018, False),
            (100.6, 2.012, float("nan"), 2.012, False),
        ],
    )
    def test_first_order_guard_enforces_preferred_band_and_hard_limit(
        self,
        sample_singlet_lens,
        focal_before,
        fnum_before,
        focal_after,
        fnum_after,
        expected,
    ):
        """EFL/F 数在首选带内不得外逃，带外只能改善且不能越过硬上限。"""

        accepted = sample_singlet_lens._first_order_update_is_acceptable(
            focal_length_before=focal_before,
            f_number_before=fnum_before,
            focal_length_after=focal_after,
            f_number_after=fnum_after,
            target_focal_length=100.0,
            target_f_number=2.0,
            preferred_relative_error=0.008,
            hard_relative_error=0.01,
        )

        assert accepted is expected

    @pytest.mark.parametrize(
        ("preferred", "hard"),
        [(0.0, 0.01), (0.01, 0.008), (float("nan"), 0.01)],
    )
    def test_first_order_guard_rejects_invalid_limits(
        self, sample_singlet_lens, preferred, hard
    ):
        """首选门限必须为正且不能大于一阶硬上限。"""

        with pytest.raises(ValueError, match="preferred"):
            sample_singlet_lens._first_order_update_is_acceptable(
                100.0,
                2.0,
                100.0,
                2.0,
                target_focal_length=100.0,
                target_f_number=2.0,
                preferred_relative_error=preferred,
                hard_relative_error=hard,
            )

    def test_first_order_measurement_preserves_training_rng(
        self, sample_singlet_lens
    ):
        """逐步一阶测量不得消耗后续训练光线使用的随机序列。"""

        torch.manual_seed(1234)
        rng_before = torch.random.get_rng_state().clone()

        focal_length, f_number = (
            sample_singlet_lens._measure_first_order_state()
        )

        assert math.isfinite(focal_length)
        assert math.isfinite(f_number)
        assert torch.equal(torch.random.get_rng_state(), rng_before)

    @pytest.mark.parametrize("min_valid_ratio", [0.0, 1.01])
    def test_ray_validity_loss_rejects_invalid_threshold(
        self, sample_singlet_lens, min_valid_ratio
    ):
        """有效率阈值必须位于合法范围。"""
        with pytest.raises(ValueError, match="min_valid_ratio"):
            sample_singlet_lens._ray_validity_loss(
                torch.ones(1, 4), min_valid_ratio=min_valid_ratio
            )


class TestSampleRays:
    """测试 sample_ring_arm_rays。"""

    def test_sample_ring_arm_rays_returns_ray(self, sample_singlet_lens):
        """sample_ring_arm_rays 返回具有正确 shape 的 Ray 对象。"""
        from deeplens.light import Ray

        lens = sample_singlet_lens
        ray = lens.sample_ring_arm_rays(num_ring=4, num_arm=4, spp=64)
        assert isinstance(ray, Ray)
        # shape 应为 [num_ring, num_arm, spp, 3]
        assert ray.o.shape[-1] == 3
        assert ray.d.shape[-1] == 3

    def test_sample_ring_arm_rays_accepts_explicit_target_field(
        self, sample_singlet_lens
    ):
        """显式目标半视场不应依赖镜头当前缓存的 rfov。"""

        target_rfov = 0.08
        ray = sample_singlet_lens.sample_ring_arm_rays(
            num_ring=2,
            num_arm=1,
            spp=8,
            depth=-1000.0,
            sample_more_off_axis=False,
            max_fov_rad=target_rfov,
        )
        edge_origin = ray.o[-1, 0, 0]
        recovered = torch.atan2(
            torch.linalg.vector_norm(edge_origin[:2]), edge_origin[2].abs()
        )

        assert recovered.item() == pytest.approx(target_rfov, rel=1e-5)


class TestGradientFlow:
    """测试损失的梯度反向传播。"""

    def test_loss_rms_backward(self, sample_cellphone_lens):
        """loss_rms 反向传播应在镜头参数上产生梯度。"""
        lens = sample_cellphone_lens
        lens.get_optimizer_params()
        loss = lens.loss_rms(num_grid=(2, 2), num_rays=128)
        loss.backward()
        # 检查至少一个表面参数具有梯度
        has_grad = False
        for s in lens.surfaces:
            if hasattr(s, "c") and isinstance(s.c, torch.Tensor) and s.c.grad is not None:
                has_grad = True
                break
            if hasattr(s, "d") and isinstance(s.d, torch.Tensor) and s.d.grad is not None:
                has_grad = True
                break
        assert has_grad, "No gradients found on lens parameters after backward()"
