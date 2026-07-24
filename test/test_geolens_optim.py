"""deeplens/optics/geolens_pkg/optim.py 测试——GeoLensOptim mixin。

所有方法均通过 GeoLens 实例测试（mixin 架构）。
"""

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
