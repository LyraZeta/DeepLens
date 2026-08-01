# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""GeoLens 的优化与约束函数。

与传统透镜设计相比，可微透镜设计有以下优点：
    1. AutoDiff 梯度计算更快、数值更稳定，这对复杂光学系统尤为重要。
    2. 带动量的一阶优化（如 Adam）通常比二阶优化更稳定，收敛速度也很可观。
    3. 高效定义损失函数可以防止透镜违反约束。

参考：
    Xinge Yang, Qiang Fu, and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.

函数：
    - init_constraints：初始化透镜设计约束
    - loss_reg：透镜设计的经验正则化损失
    - loss_infocus：采样平行光线并计算传感器平面上的 RMS 损失
    - loss_profile：惩罚逐表面的轮廓形状（矢高、斜率）
    - loss_bound：惩罚几何边界违规（间隙与包络）
    - loss_cra：惩罚传感器处超过 chief_ray_angle_max 的主光线角
    - loss_ray_bend：惩罚超过 bend_angle_max 的逐表面累计弯折角
    - loss_rms：带可选质心参考和畸变正则化的 RGB 光斑 RMS
    - sample_ring_arm_rays：使用环臂模式从物方采样光线
    - optimize：通过最小化 RMS 误差优化透镜
"""

import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
from torch.nn.functional import relu
from tqdm import tqdm

from ..config import (
    EPSILON,
    GEO_GRID,
    SPP_CALC,
    SPP_PSF,
)
from ..geometric_surface import Aperture, Aspheric, Plane, Spheric, ThinLens
from ..phase_surface import Phase


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """构建先线性预热、再以半余弦衰减到零的 LR 调度器。

    学习率乘数在预热步内从 0 线性升至 1，随后在剩余步内沿半余弦
    从 1 降至 0。

    参数：
        optimizer (torch.optim.Optimizer)：需要调度学习率的优化器。
        num_warmup_steps (int)：线性预热步数。
        num_training_steps (int)：总训练步数。

    返回：
        scheduler (torch.optim.lr_scheduler.LambdaLR)：应用预热后余弦乘数的调度器。
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class GeoLensOptim:
    """为 ``GeoLens`` 提供可微优化的混入类。

    使用 PyTorch autograd 实现基于梯度的透镜设计：

    * **损失函数**——RMS 光斑误差、焦点、表面规则性、间隙约束和材料有效性。
    * **约束初始化**——边缘厚度和自相交保护。
    * **优化器辅助方法**——按类型设置学习率的参数组和余弦退火调度。
    * **高层 ``optimize()``**——课程学习训练循环。

    本类不单独实例化，而是混入 `GeoLens`。

    参考：
        Xinge Yang et al., "Curriculum learning for ab initio deep learned
        refractive optics," *Nature Communications* 2024.
    """

    # ================================================================
    # 透镜设计约束
    # ================================================================
    def init_constraints(self, constraint_params=None):
        """初始化透镜的几何、光线角度和畸变约束。

        根据传感器半径是否小于 12 mm 选择手机或相机约束预设，设置空气间隔、
        厚度、BFL、TTL、表面形状、CRA、弯折角和畸变限制，并将弯折角限制
        传播到每个表面。

        参数：
            constraint_params (dict, optional)：覆盖预设约束的键值。显式传入时
                会保存在镜头对象中，后续 ``post_computation()`` 调用将继续沿用；
                传入空字典可恢复预设。默认值为 None。
        """
        if constraint_params is None:
            constraint_params = dict(getattr(self, "constraint_params", {}))
        else:
            constraint_params = dict(constraint_params)
            self.constraint_params = constraint_params

        if self.r_sensor < 12.0:
            self.is_cellphone = True

            self.air_edge_min = 0.05
            self.air_edge_max = 5.0
            self.air_center_min = 0.05
            self.air_center_max = 5.0

            self.thick_edge_min = 0.25
            self.thick_edge_max = 5.0
            self.thick_center_min = 0.25
            self.thick_center_max = 5.0

            self.bfl_min = 0.8
            self.bfl_max = 5.0

            self.ttl_min = 0.0
            self.ttl_max = 50.0

            # 表面形状约束
            self.sag2diam_max = 0.5
            self.diam2thick_max = 15.0
            self.tmax2tmin_max = 5.0
            self.surf_angle_max = 45.0  # deg

            # 光线角度约束
            self.chief_ray_angle_max = 45.0  # deg
            self.bend_angle_max = 30.0  # deg

            # 畸变约束
            self.distortion_max = 0.10  # 10 % 相对畸变

        else:
            self.is_cellphone = False

            self.air_edge_min = 0.1
            self.air_edge_max = 100.0  # float("inf")
            self.air_center_min = 0.1
            self.air_center_max = 100.0  # float("inf")

            self.thick_edge_min = 1.0
            self.thick_edge_max = 20.0
            self.thick_center_min = 2.0
            self.thick_center_max = 20.0

            self.bfl_min = 5.0
            self.bfl_max = 100.0  # float("inf")

            self.ttl_min = 0.0  # 默认禁用
            self.ttl_max = 300.0  # float("inf")

            # 表面形状约束
            self.sag2diam_max = 0.5
            self.diam2thick_max = 20.0
            self.tmax2tmin_max = 10.0
            self.surf_angle_max = 45.0  # deg

            # 光线角度约束
            self.chief_ray_angle_max = 45.0  # deg
            self.bend_angle_max = 30.0  # deg

            # 畸变约束
            self.distortion_max = 0.02  # 2 % 相对畸变

        # 应用任务专用覆盖。只允许覆盖已经由预设定义的约束字段，避免拼写错误
        # 静默创建无效属性。
        for name, value in constraint_params.items():
            if not hasattr(self, name):
                raise ValueError(f"未知镜头约束参数：{name}")
            setattr(self, name, value)

        # 将弯折角限制传播到每个表面，供 refract() 读取。
        for s in self.surfaces:
            s.bend_angle_max = self.bend_angle_max

    def loss_reg(
        self,
        w_focus=1.0,
        w_cra=1.0,
        w_ray_bend=1.0,
        w_clearance=1.0,
        w_envelope=1.0,
        w_profile=1.0,
    ):
        """计算透镜设计的组合正则化损失。

        汇总多个约束损失，使透镜在基于梯度的优化期间保持物理有效。

        参数：
            w_focus (float, optional)：聚焦损失权重。默认值为 1.0。
            w_cra (float, optional)：主光线角损失权重。默认值为 1.0。
            w_ray_bend (float, optional)：逐表面弯折惩罚权重。默认值为 1.0。
            w_clearance (float, optional)：间隙惩罚权重（最小空气间隔、最小厚度、
                最小 BFL、最小 TTL）。默认值为 1.0。
            w_envelope (float, optional)：包络惩罚权重（最大空气间隔、最大厚度、
                最大 BFL、最大 TTL）。默认值为 1.0。
            w_profile (float, optional)：逐表面轮廓可行性（矢高、斜率）权重。
                默认值为 1.0。

        返回：
            loss_reg (torch.Tensor)：标量组合正则化损失。
            loss_dict (dict)：用于日志记录的各分量损失值。
        """
        # 正则化损失函数
        # loss_focus = self.loss_infocus()
        loss_cra = self.loss_cra()
        loss_ray_bend = self.loss_ray_bend()
        loss_clearance, loss_envelope = self.loss_bound()
        loss_profile = self.loss_profile()
        # loss_mat = self.loss_mat()
        loss_reg = (
            # w_focus * loss_focus
            +w_clearance * loss_clearance
            + w_envelope * loss_envelope
            + w_profile * loss_profile
            + w_cra * loss_cra
            + w_ray_bend * loss_ray_bend
            # w_mat * loss_mat
        )

        # 返回损失及损失字典
        loss_dict = {
            # "loss_focus": loss_focus.item(),
            "loss_clearance": loss_clearance.item(),
            "loss_envelope": loss_envelope.item(),
            "loss_profile": loss_profile.item(),
            "loss_cra": loss_cra.item(),
            "loss_ray_bend": loss_ray_bend.item(),
            # 'loss_mat': loss_mat.item(),
        }
        return loss_reg, loss_dict

    def loss_infocus(self, target=0.005, wvln=None):
        """采样轴上平行光线，并惩罚传感器平面上的光斑 RMS。

        将零视场光束追迹到传感器，并应用单侧惩罚
        $\\mathrm{relu}(\\text{rms} - \\text{target})$；仅当 RMS 光斑半径
        超过目标时激活。

        参数：
            target (float, optional)：目标轴上 RMS 光斑半径，单位为 mm。
                默认值为 0.005。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.wvln_rgb` 的绿色通道。默认值为 None。

        返回：
            loss (torch.Tensor)：标量聚焦惩罚（至少为 0）。
        """
        if wvln is None:
            wvln = self.wvln_rgb[1]
        loss = torch.tensor(0.0, device=self.device)

        # 光线追迹并计算 RMS 误差
        ray = self.sample_from_fov(fov_x=0.0, fov_y=0.0, wvln=wvln, num_rays=SPP_CALC)
        ray = self.trace2sensor(ray)
        rms_error = ray.rms_error()

        # 平滑惩罚：rms_error 超过 target 时激活
        loss += relu(rms_error - target)

        return loss

    def loss_profile(self):
        """惩罚不可行的逐表面轮廓形状。

        “轮廓”指单个表面的 z(r) 曲线。本损失通过检查以下项目确保各表面
        在物理上可制造：
            1. 矢高与直径之比超过 ``sag2diam_max``。
            2. 最大表面斜率角超过 ``surf_angle_max`` (deg)。

        返回：
            loss (torch.Tensor)：标量轮廓可行性惩罚。
        """
        sag2diam_max = self.sag2diam_max
        grad_max = math.tan(math.radians(self.surf_angle_max))

        loss_grad = torch.tensor(0.0, device=self.device)
        loss_sag2diam = torch.tensor(0.0, device=self.device)
        for i in self.find_diff_surf():
            # 在表面上采样点
            x_ls = torch.linspace(0.0, 1.0, 32, device=self.device) * self.surfaces[i].r
            y_ls = torch.zeros_like(x_ls)

            # 矢高
            sag_ls = self.surfaces[i].sag(x_ls, y_ls)
            sag2diam = sag_ls.abs().max() / self.surfaces[i].r / 2
            loss_sag2diam += relu(
                (sag2diam - sag2diam_max) / sag2diam_max)

            # 一阶导数
            grad_ls = self.surfaces[i].dfdxyz(x_ls, y_ls)[0]
            grad = grad_ls.abs().max()
            loss_grad += relu((grad - grad_max) / grad_max)

        # # 直径与厚度之比，以及 thick_max 与 thick_min 之比
            # if not self.surfaces[i].mat2.name == "air":
            #     surf2 = self.surfaces[i + 1]
            #     surf1 = self.surfaces[i]

        #     # 惩罚直径与厚度之比
            #     diam2thick = 2 * max(surf2.r, surf1.r) / (surf2.d - surf1.d)
            #     loss_diam2thick += torch.nn.functional.relu(diam2thick - diam2thick_max)

        #     # 惩罚 thick_max 与 thick_min 之比。
        #     # 使用 torch.maximum/minimum 实现可微的最大/最小值。
            #     r_edge = min(surf2.r, surf1.r)
            #     thick_center = surf2.d - surf1.d
            #     thick_edge = surf2.surface_with_offset(r_edge, 0.0) - surf1.surface_with_offset(r_edge, 0.0)
            #     thick_max = torch.maximum(thick_center, thick_edge)
            #     thick_min = torch.minimum(thick_center, thick_edge).clamp(min=0.01)
            #     tmax2tmin = thick_max / thick_min

            #     loss_tmax2tmin += torch.nn.functional.relu(tmax2tmin - tmax2tmin_max)

        return loss_sag2diam + loss_grad

    def loss_bound(self):
        """在一次表面采样中惩罚几何边界违规。

        每对表面仅采样一次，其距离同时用于空气间隔、玻璃厚度、BFL 和 TTL
        的间隙（最小值）与包络（最大值）relu 惩罚。

        返回：
            loss_clearance (torch.Tensor)：部件过近/过薄时的标量间隙惩罚。
            loss_envelope (torch.Tensor)：整体组件超过空间预算时的标量包络
                惩罚。两者分开返回，以便调用方独立加权。
        """
        # 最小边界（间隙）
        air_center_min = self.air_center_min
        air_edge_min = self.air_edge_min
        thick_center_min = self.thick_center_min
        thick_edge_min = self.thick_edge_min
        bfl_min = self.bfl_min
        ttl_min = self.ttl_min

        # 最大边界（包络）
        air_center_max = self.air_center_max
        air_edge_max = self.air_edge_max
        thick_center_max = self.thick_center_max
        thick_edge_max = self.thick_edge_max
        bfl_max = self.bfl_max
        ttl_max = self.ttl_max

        loss_clearance = torch.tensor(0.0, device=self.device)
        loss_envelope = torch.tensor(0.0, device=self.device)
        air_c_range = air_center_max - air_center_min
        air_e_range = air_edge_max - air_edge_min
        thick_c_range = thick_center_max - thick_center_min
        thick_e_range = thick_edge_max - thick_edge_min
        bfl_range = bfl_max - bfl_min
        ttl_range = ttl_max - ttl_min

        for i in range(len(self.surfaces) - 1):
            current_surf = self.surfaces[i]
            next_surf = self.surfaces[i + 1]

        # 仅采样一次表面，并同时复用于间隙与包络计算
            r_center = torch.tensor(0.0, device=self.device) * current_surf.r
            z_prev_center = current_surf.surface_with_offset(
                r_center, 0.0, valid_check=False
            )
            z_next_center = next_surf.surface_with_offset(
                r_center, 0.0, valid_check=False
            )

            r_edge = torch.linspace(0.5, 1.0, 16, device=self.device) * current_surf.r
            z_prev_edge = current_surf.surface_with_offset(
                r_edge, 0.0, valid_check=False
            )
            z_next_edge = next_surf.surface_with_offset(r_edge, 0.0, valid_check=False)

            dist_center = z_next_center - z_prev_center
            dist_edges = z_next_edge - z_prev_edge
            dist_edge_lo = torch.min(dist_edges)
            dist_edge_hi = torch.max(dist_edges)

            if current_surf.mat2.name == "air":
                loss_clearance += relu((air_center_min - dist_center) / air_c_range)
                loss_clearance += relu((air_edge_min - dist_edge_lo) / air_e_range)
                loss_envelope += relu((dist_center - air_center_max) / air_c_range)
                loss_envelope += relu((dist_edge_hi - air_edge_max) / air_e_range)
            else:
                loss_clearance += relu((thick_center_min - dist_center) / thick_c_range)
                loss_clearance += relu((thick_edge_min - dist_edge_lo) / thick_e_range)
                loss_envelope += relu((dist_center - thick_center_max) / thick_c_range)
                loss_envelope += relu((dist_edge_hi - thick_edge_max) / thick_e_range)

        # 后焦距
        last_surf = self.surfaces[-1]
        r = torch.linspace(0.0, 1.0, 32, device=self.device) * last_surf.r
        z_last_surf = self.d_sensor - last_surf.surface_with_offset(r, 0.0)
        bfl_lo = torch.min(z_last_surf)
        bfl_hi = torch.max(z_last_surf)
        loss_clearance += relu((bfl_min - bfl_lo) / bfl_range)
        loss_envelope += relu((bfl_hi - bfl_max) / bfl_range)

        # 总轨道长度
        ttl = self.d_sensor - self.surfaces[0].d
        loss_clearance += relu((ttl_min - ttl) / ttl_range)
        loss_envelope += relu((ttl - ttl_max) / ttl_range)

        return loss_clearance, loss_envelope

    def loss_cra(self):
        """惩罚传感器处超过 chief_ray_angle_max 的主光线角。

        在完整 FoV 上使用近轴瞳孔采样（scale_pupil=0.2）。惩罚为
        $\\mathrm{relu}(\\cos\\theta_\\text{ref} - \\cos\\theta_\\text{CRA})$，
        并在有效光线上取平均，其中 $\\cos\\theta = $ `ray.d[..., 2]`。

        返回：
            loss (torch.Tensor)：标量 CRA 惩罚（至少为 0）。
        """
        cos_cra_ref = float(np.cos(np.deg2rad(self.chief_ray_angle_max)))

        ray = self.sample_ring_arm_rays(num_ring=8, num_arm=2, spp=SPP_CALC, scale_pupil=0.2)
        ray = self.trace2sensor(ray)
        cos_cra = ray.d[..., 2]
        valid = ray.is_valid > 0
        penalty_cra = relu(cos_cra_ref - cos_cra)
        return (penalty_cra * valid).sum() / (valid.sum() + EPSILON)

    def loss_ray_bend(self):
        """惩罚超过 bend_angle_max 的逐表面累计弯折角。

        读取 ``ray.bend_penalty``，即在 ``trace2sensor`` 期间收集的逐表面
        relu 贡献之和。各表面独立贡献，因此一个表面的较大弯折不会被另一
        表面的较小弯折掩盖。使用完整瞳孔采样（scale_pupil=1.0）。

        返回：
            loss (torch.Tensor)：标量弯折惩罚（至少为 0）。
        """
        ray = self.sample_ring_arm_rays(num_ring=8, num_arm=2, spp=SPP_CALC, scale_pupil=1.0)
        ray = self.trace2sensor(ray)
        bend_penalty = ray.bend_penalty.squeeze(-1)
        valid = ray.is_valid > 0
        return (bend_penalty * valid).sum() / (valid.sum() + EPSILON)

    def loss_mat(self):
        """惩罚超出可制造范围的材料参数。

        将各非空气表面材料的折射率 *n* 限制在 [1.5, 1.9]，阿贝数 *V*
        限制在 [30, 70]。

        返回：
            loss_mat (torch.Tensor)：标量材料惩罚损失。
        """
        n_max = 1.9
        n_min = 1.5
        V_max = 70
        V_min = 30
        loss_mat = torch.tensor(0.0, device=self.device)
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.name != "air":
                if self.surfaces[i].mat2.n > n_max:
                    loss_mat += (self.surfaces[i].mat2.n - n_max) / (n_max - n_min)
                if self.surfaces[i].mat2.n < n_min:
                    loss_mat += (n_min - self.surfaces[i].mat2.n) / (n_max - n_min)
                if self.surfaces[i].mat2.V > V_max:
                    loss_mat += (self.surfaces[i].mat2.V - V_max) / (V_max - V_min)
                if self.surfaces[i].mat2.V < V_min:
                    loss_mat += (V_min - self.surfaces[i].mat2.V) / (V_max - V_min)

        return loss_mat

    # ================================================================
    # 图像质量损失函数
    # ================================================================
    @staticmethod
    def _ray_validity_loss(ray_valid, min_valid_ratio=0.5):
        """计算逐视场有效光线比例不足的惩罚。

        ``ray_valid`` 的最后一维必须是同一视场内的光线维。惩罚先计算每个
        视场的有效光线比例，再对低于 ``min_valid_ratio`` 的短缺量归一化并
        平方，最后在全部视场上求平均。这样全挡光视场会得到 1，而达到阈值
        的视场不受惩罚。

        需要注意，``is_valid`` 是由求交、孔径和全反射判断产生的硬掩码，
        因此该项主要用于阻止优化目标把挡光误判为低 RMS，而不是平滑的渐晕
        梯度。几何轮廓和弯折损失仍负责提供可微的恢复方向。

        参数：
            ray_valid (torch.Tensor)：有效性掩码，shape 为 ``[..., num_rays]``。
            min_valid_ratio (float, optional)：每个视场允许的最低有效光线比例，
                范围为 ``(0, 1]``。默认值为 0.5。

        返回：
            loss (torch.Tensor)：有效率短缺的标量损失，范围为 ``[0, 1]``。
            valid_ratio (torch.Tensor)：逐视场有效光线比例，shape 为 ``[...]``。
        """
        if not 0.0 < min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio 必须位于 (0, 1] 范围内。")
        if ray_valid.ndim < 1 or ray_valid.shape[-1] < 1:
            raise ValueError("ray_valid 的最后一维必须包含至少一条光线。")

        valid_ratio = ray_valid.float().clamp(0.0, 1.0).mean(dim=-1)
        shortfall = relu(min_valid_ratio - valid_ratio) / min_valid_ratio
        return shortfall.square().mean(), valid_ratio

    @staticmethod
    def _masked_field_mse(ray_err, ray_valid, invalid_rms):
        """计算逐视场 MSE，并为全失效视场给出固定代理值。

        无效光线在平方前置零，避免 ``Inf * 0``。若某个视场没有任何有效
        光线，则使用 ``invalid_rms ** 2``，避免该视场在 RMS 日志和自适应
        权重中被误报为接近零。该代理分支由硬掩码选择，本身不提供梯度；真正
        的挡光约束由更新后的有效率复追迹与回滚负责。
        """
        if invalid_rms <= 0.0:
            raise ValueError("invalid_rms 必须大于 0。")
        if ray_err.shape[:-1] != ray_valid.shape:
            raise ValueError("ray_err 与 ray_valid 的光线维 shape 不匹配。")

        valid_mask = ray_valid.bool()
        safe_err = torch.where(
            valid_mask.unsqueeze(-1), ray_err, torch.zeros_like(ray_err)
        )
        valid_count = ray_valid.float().clamp(0.0, 1.0).sum(dim=-1)
        mse = (safe_err**2).sum(dim=-1).sum(dim=-1) / (valid_count + EPSILON)
        invalid_mse = torch.as_tensor(
            invalid_rms**2, dtype=mse.dtype, device=mse.device
        )
        return torch.where(valid_count > 0, mse, invalid_mse)

    @staticmethod
    def _masked_directional_variance(ray_xy, ray_valid, invalid_rms):
        """计算逐视场、逐方向的光线截距方差。

        与只返回一个径向二阶矩的 RMS 不同，本函数保留 X/Y 两个方向，便于
        对切向和弧矢像质分别施加约束。无效光线在进入求和前被置零；完全无
        有效光线的视场使用 ``invalid_rms ** 2``，避免挡光被误判为优良像质。

        参数：
            ray_xy (torch.Tensor)：像面二维截距，shape 为 ``[..., num_rays, 2]``。
            ray_valid (torch.Tensor)：有效性掩码，shape 为 ``[..., num_rays]``。
            invalid_rms (float)：全失效视场使用的方向 RMS 代理值 [mm]。

        返回：
            variance_xy (torch.Tensor)：X/Y 方向方差，shape 为 ``[..., 2]``。
        """
        if invalid_rms <= 0.0:
            raise ValueError("invalid_rms 必须大于 0。")
        if ray_xy.shape != (*ray_valid.shape, 2):
            raise ValueError("ray_xy 与 ray_valid 的光线维 shape 不匹配。")

        valid_mask = ray_valid.bool()
        valid_weight = ray_valid.float().clamp(0.0, 1.0)
        safe_xy = torch.where(
            valid_mask.unsqueeze(-1), ray_xy, torch.zeros_like(ray_xy)
        )
        valid_count = valid_weight.sum(dim=-1)
        centroid = safe_xy.sum(dim=-2) / (valid_count + EPSILON).unsqueeze(-1)
        centered = torch.where(
            valid_mask.unsqueeze(-1),
            safe_xy - centroid.unsqueeze(-2),
            torch.zeros_like(safe_xy),
        )
        variance_xy = centered.square().sum(dim=-2) / (
            valid_count + EPSILON
        ).unsqueeze(-1)
        invalid_variance = torch.as_tensor(
            invalid_rms**2, dtype=variance_xy.dtype, device=variance_xy.device
        )
        return torch.where(
            (valid_count > 0).unsqueeze(-1), variance_xy, invalid_variance
        )

    @staticmethod
    def _fixed_frequency_geometric_mtf(
        ray_xy, ray_valid, frequency_cy_mm
    ):
        """由可微光线截距计算单一频率的几何切向/弧矢 MTF。

        该实现直接计算经验特征函数 ``|mean(exp(-i 2 pi f x))|``，与 MWIR
        数值验收使用的几何 OTF 定义一致。先减去有效光线质心可降低大像高在
        float32 中造成的相位精度损失；MTF 幅值本身对平移不敏感。

        返回最后一维依次为切向（Y 截距）和弧矢（X 截距）MTF。该量在光斑
        远大于目标周期时会接近有限样本噪声底，因此更适合监控或后期精修；
        早期优化宜使用 ``_fixed_frequency_mtf_surrogate_violation``。
        """
        if not math.isfinite(frequency_cy_mm) or frequency_cy_mm < 0.0:
            raise ValueError("MTF 空间频率必须为非负有限值。")
        if ray_xy.shape != (*ray_valid.shape, 2):
            raise ValueError("ray_xy 与 ray_valid 的光线维 shape 不匹配。")

        valid_mask = ray_valid.bool()
        valid_weight = ray_valid.float().clamp(0.0, 1.0)
        safe_xy = torch.where(
            valid_mask.unsqueeze(-1), ray_xy, torch.zeros_like(ray_xy)
        )
        valid_count = valid_weight.sum(dim=-1)
        centroid = safe_xy.sum(dim=-2) / (valid_count + EPSILON).unsqueeze(-1)
        centered = torch.where(
            valid_mask.unsqueeze(-1),
            safe_xy - centroid.unsqueeze(-2),
            torch.zeros_like(safe_xy),
        )
        phase = -2.0 * math.pi * frequency_cy_mm * centered
        denominator = (valid_count + EPSILON).unsqueeze(-1)
        real = (torch.cos(phase) * valid_weight.unsqueeze(-1)).sum(-2) / denominator
        imag = (torch.sin(phase) * valid_weight.unsqueeze(-1)).sum(-2) / denominator
        mtf_xy = torch.sqrt((real.square() + imag.square()).clamp_min(EPSILON))
        mtf_xy = torch.where(
            (valid_count > 0).unsqueeze(-1), mtf_xy, torch.zeros_like(mtf_xy)
        )
        return torch.stack([mtf_xy[..., 1], mtf_xy[..., 0]], dim=-1)

    @staticmethod
    def _fixed_frequency_mtf_surrogate_violation(
        ray_xy,
        ray_valid,
        frequency_cy_mm,
        target_mtf,
        invalid_rms,
    ):
        """返回固定频率 MTF 的稳健相位方差代理超差量。

        对高斯截距分布，单轴几何 MTF 可写为
        ``exp(-0.5 * (2 pi f sigma) ** 2)``。本函数以此把目标 MTF 转换为
        允许的方向 RMS，再分别约束切向和弧矢方向。相较于直接优化经验 OTF
        幅值，该代理在 MTF 接近有限光线噪声底时仍保持平滑、单调的梯度。

        超差使用 ``log1p`` 压缩，避免毫米级初始光斑相对微米级目标产生过大
        梯度。返回 shape 为 ``[..., 2]``，最后一维依次为切向和弧矢方向。
        """
        if not math.isfinite(frequency_cy_mm) or frequency_cy_mm <= 0.0:
            raise ValueError("MTF 空间频率必须为正的有限值。")
        if not math.isfinite(target_mtf) or not 0.0 < target_mtf < 1.0:
            raise ValueError("target_mtf 必须位于 (0, 1) 范围内。")

        variance_xy = GeoLensOptim._masked_directional_variance(
            ray_xy, ray_valid, invalid_rms=invalid_rms
        )
        sigma_xy = torch.sqrt(variance_xy.clamp_min(EPSILON))
        target_phase_sigma = math.sqrt(-2.0 * math.log(target_mtf))
        phase_sigma_xy = 2.0 * math.pi * frequency_cy_mm * sigma_xy
        relative_excess_xy = relu(phase_sigma_xy / target_phase_sigma - 1.0)
        violation_xy = torch.log1p(relative_excess_xy)
        return torch.stack(
            [violation_xy[..., 1], violation_xy[..., 0]], dim=-1
        )

    @staticmethod
    def _target_field_mapping_loss(
        actual_xy, target_xy, tolerance, max_weight=0.0
    ):
        """惩罚离轴实际像点相对目标针孔像高超出容差的部分。

        轴上场点的目标像高为零，不能用于相对误差，因此会被自动排除。其余
        场点使用二维像点位置误差除以目标像高；容差内损失为零，超出部分按
        容差归一化后线性增长。``max_weight`` 可额外强调全部场点中的最坏超差，
        使训练目标更接近最大畸变/像高验收。该项同时约束目标焦距对应的像高
        尺度和场内映射，不应再混入通用机械/曲面正则项。
        """
        if tolerance <= 0.0:
            raise ValueError("tolerance 必须大于 0。")
        if not math.isfinite(max_weight) or max_weight < 0.0:
            raise ValueError("max_weight 必须为大于或等于 0 的有限值。")
        if actual_xy.shape != target_xy.shape or actual_xy.shape[-1] != 2:
            raise ValueError("actual_xy 与 target_xy 必须具有相同的 [..., 2] shape。")

        target_height = target_xy.norm(dim=-1)
        field_mask = target_height > EPSILON
        relative_error = (actual_xy - target_xy).norm(dim=-1)
        relative_error = relative_error / target_height.clamp_min(EPSILON)
        violation = relu(relative_error / tolerance - 1.0)
        field_count = field_mask.sum().clamp_min(1)
        masked_violation = violation * field_mask.float()
        mean_violation = masked_violation.sum() / field_count
        worst_violation = masked_violation.max()
        return mean_violation + max_weight * worst_violation

    @staticmethod
    def _validity_update_is_acceptable(
        valid_ratio_before, valid_ratio_after, min_valid_ratio
    ):
        """判断参数更新后的最低有效光线比例是否可接受。

        当更新前已经达到目标阈值时，更新后不得跌破阈值；当初始结构尚未
        达标时，当前有效率作为临时底线，只接受持平或改善的更新。该“棘轮”
        规则允许略低于目标的初始处方继续优化，同时阻止优化通过进一步挡光
        降低 RMS。
        """
        if not 0.0 < min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio 必须位于 (0, 1] 范围内。")

        before = float(valid_ratio_before)
        after = float(valid_ratio_after)
        if not math.isfinite(before) or not math.isfinite(after):
            return False
        temporary_floor = min_valid_ratio if before >= min_valid_ratio else before
        return after + EPSILON >= temporary_floor

    @staticmethod
    def _first_order_relative_error(actual, target):
        """返回一阶量相对目标的绝对误差。"""

        actual = float(actual)
        target = float(target)
        if not math.isfinite(target) or target <= 0.0:
            raise ValueError("一阶目标值必须为正的有限值。")
        if not math.isfinite(actual):
            return float("inf")
        return abs(actual / target - 1.0)

    @classmethod
    def _first_order_update_is_acceptable(
        cls,
        focal_length_before,
        f_number_before,
        focal_length_after,
        f_number_after,
        target_focal_length,
        target_f_number,
        preferred_relative_error,
        hard_relative_error,
    ):
        """判断一次更新是否保持 EFL 和 F 数的一阶约束。

        已位于 ``preferred_relative_error`` 内的处方不得越出该范围；位于首选
        范围与 ``hard_relative_error`` 之间的处方只能持平或改善。无论更新前
        状态如何，更新后都不得超过硬上限。该棘轮规则允许从略微超出首选范围
        的已有处方继续收敛，同时阻止优化用焦距漂移换取较低 RMS。
        """

        preferred = float(preferred_relative_error)
        hard = float(hard_relative_error)
        if (
            not math.isfinite(preferred)
            or not math.isfinite(hard)
            or preferred <= 0.0
            or hard < preferred
        ):
            raise ValueError("一阶误差门限必须满足 0 < preferred <= hard。")

        before_errors = (
            cls._first_order_relative_error(
                focal_length_before, target_focal_length
            ),
            cls._first_order_relative_error(f_number_before, target_f_number),
        )
        after_errors = (
            cls._first_order_relative_error(focal_length_after, target_focal_length),
            cls._first_order_relative_error(f_number_after, target_f_number),
        )
        for before, after in zip(before_errors, after_errors):
            if not math.isfinite(before) or not math.isfinite(after):
                return False
            if after > hard + EPSILON:
                return False
            temporary_limit = preferred if before <= preferred + EPSILON else before
            if after > temporary_limit + EPSILON:
                return False
        return True

    def _measure_first_order_state(self):
        """以固定随机样本测量并缓存当前 EFL 与 F 数。

        ``calc_foclen`` 和 ``calc_pupil`` 会在内部采样光线。这里隔离并固定随机
        数状态，使逐步硬门控可复现，也避免门控额外消耗训练光线的随机序列。
        """

        device = torch.device(self.device)
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(0)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(0)
            focal_length = float(self.calc_foclen())
            self.calc_pupil()
            f_number = float(self.fnum)

        if not math.isfinite(focal_length) or not math.isfinite(f_number):
            raise ValueError("一阶参数测量得到非有限 EFL 或 F 数。")
        return focal_length, f_number

    def _trace_min_valid_ratio(self, rays):
        """用既有采样光线快速复追迹，返回所有波长和视场中的最低有效率。"""
        ratios = []
        with torch.no_grad():
            for sampled_ray in rays:
                traced_ray = self.trace2sensor(sampled_ray.clone())
                ratio = traced_ray.is_valid.float().clamp(0.0, 1.0).mean(dim=-1)
                ratios.append(ratio.min())
        return torch.stack(ratios).min()

    def loss_rms(
        self,
        num_grid=GEO_GRID,
        depth=None,
        num_rays=SPP_PSF,
        sample_more_off_axis=False,
    ):
        """在视场点网格上计算 RGB 光斑尺寸 RMS 损失。

        将 R、G、B 光束（绿色优先）追迹到传感器，并相对于绿色针孔中心测量
        光斑半径。绿色光斑误差用于设置分离梯度的逐视场权重掩码，以强调
        更困难的视场。

        参数：
            num_grid (int, optional)：每个轴上的视场网格点数。默认值为 GEO_GRID。
            depth (float, optional)：物面深度，单位为 mm。为 None（默认）时
                回退到 `self.obj_depth`。默认值为 None。
            num_rays (int, optional)：每个视场点的光线数。默认值为 SPP_PSF。
            sample_more_off_axis (bool, optional)：为 True 时将视场样本集中到
                视场边缘。默认值为 False。

        返回：
            avg_rms_error (torch.Tensor)：在 R、G、B 波长上取平均的标量
                RMS 光斑误差，单位为 mm。
        """
        depth = self.obj_depth if depth is None else depth
        invalid_rms_proxy = max(2.0 * float(self.r_sensor), 1.0)
        # 先迭代绿色，使误差自适应权重掩码以参考（绿色）波长为基准。
        loss_rms_ls = []
        w_mask = None
        for i, wvln in enumerate(
            [self.wvln_rgb[1], self.wvln_rgb[0], self.wvln_rgb[2]]
        ):
            ray = self.sample_grid_rays(
                depth=depth,
                num_grid=num_grid,
                num_rays=num_rays,
                wvln=wvln,
                sample_more_off_axis=sample_more_off_axis,
            )

            # 根据绿色主光线（针孔）获得参考中心，并广播到各光线。
            if i == 0:
                with torch.no_grad():
                    center_ref = -self.psf_center(
                        points_obj=ray.o[:, :, 0, :], method="pinhole"
                    )
                center_ref = center_ref.unsqueeze(-2)

            ray = self.trace2sensor(ray)

            # 逐 FoV 将 MSE 转为 RMS；全失效视场使用像面直径级代理值，
            # 避免挡光被误报为近零 RMS。
            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid
            ray_err = ray_xy - center_ref
            mse = self._masked_field_mse(
                ray_err, ray_valid, invalid_rms=invalid_rms_proxy
            )
            l_rms = (mse + EPSILON).sqrt()

            # 第一个波长（绿色）定义分离梯度的权重掩码。
            if w_mask is None:
                w_mask = mse.detach()
                w_mask = w_mask / (w_mask.mean() + EPSILON)

            l_rms_weighted = (l_rms * w_mask).sum() / (w_mask.sum() + EPSILON)
            loss_rms_ls.append(l_rms_weighted)

        avg_rms_error = torch.stack(loss_rms_ls).mean(dim=0)
        return avg_rms_error

    # ================================================================
    # 优化函数示例
    # ================================================================
    def sample_ring_arm_rays(
        self,
        num_ring=8,
        num_arm=2,
        spp=2048,
        depth=None,
        wvln=None,
        scale_pupil=1.0,
        sample_more_off_axis=True,
        max_fov_rad=None,
    ):
        """使用环臂模式从物方采样光线。

        本方法在由视场定义的物面极坐标网格上分布采样点（光束原点），
        用于捕获完整视场内的透镜性能。采样点包括中心，以及 `num_ring`
        个圆环，每个圆环包含 `num_arm` 个点。

        使用 ``self.rfov``（考虑畸变的光追实际 FoV），而非
        ``self.rfov_eff``（近轴针孔 FoV），从而覆盖完整畸变视场。

        参数：
            num_ring (int, optional)：视场中的采样圆环数。默认值为 8。
            num_arm (int, optional)：每个圆环的采样臂（辐条）数。默认值为 2。
            spp (int, optional)：每个视场点的采样光线数。默认值为 2048。
            depth (float, optional)：物面深度，单位为 mm。为 None（默认）时
                回退到 `self.obj_depth`。默认值为 None。
            wvln (float, optional)：波长，单位为 µm。为 None（默认）时回退到
                `self.primary_wvln`。默认值为 None。
            scale_pupil (float, optional)：入瞳半径缩放因子。默认值为 1.0。
            sample_more_off_axis (bool, optional)：为 True 时以平方根曲线扭曲
                圆环视场角，使样本集中到视场边缘。默认值为 True。
            max_fov_rad (float or None, optional)：径向半视场上限 [rad]。为 None
                时使用镜头当前缓存的 ``self.rfov``；从任务指标优化时应显式
                传入目标半视场，避免实际处方 FoV 漂移后连带改变训练目标。

        返回：
            rays (Ray)：视场点按 [num_ring, num_arm] 排列、每点含 `spp`
                条光线的光束。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # 在圆环和采样臂上创建点
        max_fov_rad = self.rfov if max_fov_rad is None else float(max_fov_rad)
        if not math.isfinite(max_fov_rad) or max_fov_rad <= 0.0:
            raise ValueError("max_fov_rad 必须为正的有限弧度值。")
        if sample_more_off_axis:
            beta_values = torch.linspace(0.0, 1.0, num_ring, device=self.device)
            beta_transformed = beta_values**0.5
            ring_fovs = max_fov_rad * beta_transformed
        else:
            ring_fovs = max_fov_rad * torch.linspace(
                0.0, 1.0, num_ring, device=self.device
            )

        arm_angles = torch.linspace(0.0, 2 * torch.pi, num_arm + 1, device=self.device)[
            :-1
        ]
        ring_grid, arm_grid = torch.meshgrid(ring_fovs, arm_angles, indexing="ij")
        x = depth * torch.tan(ring_grid) * torch.cos(arm_grid)
        y = depth * torch.tan(ring_grid) * torch.sin(arm_grid)
        z = torch.full_like(x, depth)
        points = torch.stack([x, y, z], dim=-1)  # shape：[num_ring, num_arm, 3]

        # 采样光线
        rays = self.sample_from_points(
            points=points, num_rays=spp, wvln=wvln, scale_pupil=scale_pupil
        )
        return rays

    def optimize(
        self,
        lrs=[1e-3, 1e-4, 1e-1, 1e-4],
        iterations=5000,
        test_per_iter=100,
        optim_mat=False,
        shape_control=True,
        sample_more_off_axis=False,
        result_dir=None,
        *,
        num_ring=32,
        num_arm=8,
        spp=2048,
        ray_resample_interval=0,
        min_valid_ratio=0.5,
        w_rms=1.0,
        w_mtf=0.0,
        mtf_frequency_cy_mm=None,
        mtf_target=0.5,
        mtf_max_weight=1.0,
        mtf_field_fractions=(0.0, 0.7, 1.0),
        w_valid=2.0,
        target_focal_length=None,
        target_f_number=None,
        first_order_preferred_relative_error=None,
        first_order_hard_relative_error=None,
        target_rfov=None,
        w_field=0.1,
        w_reg=0.1,
        field_mapping_all_wavelengths=False,
        field_mapping_max_weight=0.0,
        field_mapping_use_chief_ray=False,
        field_mapping_num_points=0,
        checkpoint_analysis=True,
    ):
        """通过最小化 RGB RMS 光斑误差优化透镜。

        使用 Adam 优化器和余弦退火运行课程学习训练循环。定期评估透镜、
        保存中间结果，并可选择校正表面形状。

        参数：
            lrs (list, optional)：[d, c, k, a] 参数组的学习率。默认值为
                [1e-3, 1e-4, 1e-1, 1e-4]。
            iterations (int, optional)：总训练迭代数。默认值为 5000。
            test_per_iter (int, optional)：每 N 次迭代评估并保存。默认值为 100。
            optim_mat (bool, optional)：为 True 时在优化中包含材料参数 (n, V)。
                默认值为 False。
            shape_control (bool, optional)：为 True 时在每次评估时调用
                ``correct_shape()``。默认值为 True。
            sample_more_off_axis (bool, optional)：为 True 时将光线样本集中到
                视场边缘，以改善离轴校正。直接传给 ``sample_ring_arm_rays``。
                默认值为 False。
            result_dir (str, optional)：结果保存目录。为 None 时自动生成带
                时间戳的目录。默认值为 None。
            num_ring (int, optional)：径向视场采样环数。默认值为 32。
            num_arm (int, optional)：每个采样环的方位臂数。默认值为 8。
            spp (int, optional)：每个视场、每个波长的光线数。默认值为 2048。
                CPU 冒烟测试可使用较小值，最终优化应逐步提高采样密度。
            ray_resample_interval (int, optional)：训练瞳光线的重采样间隔。
                0 表示保持旧行为、只在检查点重采样；1 表示每步重采样，可降低
                对单一 Monte-Carlo 瞳样本的过拟合。默认值为 0。
            min_valid_ratio (float, optional)：每个视场的最低有效光线比例。
                低于该值会产生独立于 RMS 的挡光惩罚。默认值为 0.5。
            w_rms (float, optional)：RMS 光斑损失权重。默认值为 1.0。
                分阶段优化时可先降低该值，让像高/场映射约束优先稳定，随后
                再恢复为 1.0 改善像质。
            w_mtf (float, optional)：固定频率 MTF 相位方差代理权重。为 0 时
                不额外采样验收场点，保持旧行为。默认值为 0。
            mtf_frequency_cy_mm (float or None, optional)：MTF 代理的空间频率
                [cycles/mm]；启用 ``w_mtf`` 时必须为正有限值。
            mtf_target (float, optional)：几何 MTF 代理目标，必须位于 (0, 1)。
                默认值为 0.5。
            mtf_max_weight (float, optional)：最坏波长、场点和方向相对平均超差
                的附加权重。默认值为 1.0。
            mtf_field_fractions (sequence[float], optional)：相对于目标半视场的
                MTF 代理场点；默认与 MWIR 验收一致，取 0、0.7 和 1.0。
            w_valid (float, optional)：挡光惩罚权重。该无量纲权重会乘以像面
                半径，使惩罚与毫米制 RMS 处于相近量级。默认值为 2.0。
            target_focal_length (float or None, optional)：理想针孔映射使用的目标
                有效焦距 [mm]。为 None 时使用当前 ``self.foclen``。显式传入后，
                优化不会通过修改或伪造镜头的一阶缓存来表达任务目标。
            target_f_number (float or None, optional)：一阶硬门控使用的目标 F 数。
                与两个 ``first_order_*_relative_error`` 参数同时提供时启用门控。
            first_order_preferred_relative_error (float or None, optional)：EFL/F 数
                的首选相对误差带。已进入该范围的处方不允许再次越出；尚未进入
                时只接受持平或改善的一阶更新。
            first_order_hard_relative_error (float or None, optional)：EFL/F 数的
                相对误差硬上限。启用一阶门控时，三个门控参数必须同时提供，且
                满足 ``0 < preferred <= hard``。
            target_rfov (float or None, optional)：训练采样使用的目标径向半视场
                [rad]。为 None 时使用当前 ``self.rfov``。
            w_field (float, optional)：目标焦距针孔映射的像高/场映射约束权重。
                该项独立于通用正则项；默认值 0.1 保持旧版优化量级，任务脚本可
                显式提高。默认值为 0.1。
            w_reg (float, optional)：机械间隙、曲面轮廓、主光线角等通用正则项
                的权重。默认值为 0.1。
            field_mapping_all_wavelengths (bool, optional)：为 True 时使用全部训练
                波长的质心计算目标场映射；为 False 时保持旧行为，只使用主波长。
                默认值为 False。
            field_mapping_max_weight (float, optional)：最坏场映射超差相对平均超差
                的附加权重。默认值为 0，不改变旧版平均损失。
            field_mapping_use_chief_ray (bool, optional)：为 True 时用瞄准入瞳中心
                的单条光线像点约束场映射；适用于前置光阑，并比光束质心更贴近
                主光线畸变验收。为 False 时保持旧版质心语义。默认值为 False。
            field_mapping_num_points (int, optional)：主光线场映射使用的等角场点数，
                包含轴上与边缘。为 0 时复用训练环臂场；启用独立网格时至少为 2。
                默认值为 0。
            checkpoint_analysis (bool, optional)：为 True 时在每个检查点除 JSON
                外还生成完整 ``analysis`` 图。大型 CPU 任务可设为 False，正式
                长优化再开启。默认值为 True，以保持通用接口的既有行为。

        说明：
            调试提示：
                1. 使用较小学习率缓慢优化。
                2. FoV 与厚度应良好匹配。
                3. 将参数范围保持在合理区间。
                4. 更高的非球面阶数效果更好，但也更敏感。
                5. 更多迭代和更大的光线采样量有助于改善收敛。
        """
        # 实验设置
        depth = self.obj_depth
        invalid_rms_proxy = max(2.0 * float(self.r_sensor), 1.0)
        if iterations < 1:
            raise ValueError("iterations 必须为正整数。")
        if test_per_iter < 1:
            raise ValueError("test_per_iter 必须为正整数。")
        if num_ring < 1 or num_arm < 1 or spp < 1:
            raise ValueError("num_ring、num_arm 和 spp 必须为正整数。")
        if (
            not isinstance(ray_resample_interval, int)
            or ray_resample_interval < 0
        ):
            raise ValueError("ray_resample_interval 必须为非负整数。")
        if field_mapping_num_points < 0 or field_mapping_num_points == 1:
            raise ValueError("field_mapping_num_points 必须为 0 或不小于 2 的整数。")
        if not 0.0 < min_valid_ratio <= 1.0:
            raise ValueError("min_valid_ratio 必须位于 (0, 1] 范围内。")
        for name, weight in (
            ("w_rms", w_rms),
            ("w_mtf", w_mtf),
            ("w_valid", w_valid),
            ("w_field", w_field),
            ("w_reg", w_reg),
            ("mtf_max_weight", mtf_max_weight),
            ("field_mapping_max_weight", field_mapping_max_weight),
        ):
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(f"{name} 必须为大于或等于 0 的有限值。")
        if mtf_frequency_cy_mm is not None and (
            not math.isfinite(mtf_frequency_cy_mm) or mtf_frequency_cy_mm <= 0.0
        ):
            raise ValueError("mtf_frequency_cy_mm 必须为正的有限值或 None。")
        if w_mtf > 0.0 and mtf_frequency_cy_mm is None:
            raise ValueError("启用 w_mtf 时必须提供 mtf_frequency_cy_mm。")
        if not math.isfinite(mtf_target) or not 0.0 < mtf_target < 1.0:
            raise ValueError("mtf_target 必须位于 (0, 1) 范围内。")
        mtf_field_fractions = tuple(float(value) for value in mtf_field_fractions)
        if not mtf_field_fractions or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in mtf_field_fractions
        ):
            raise ValueError("mtf_field_fractions 必须包含位于 [0, 1] 的有限值。")
        target_focal_length = (
            float(self.foclen)
            if target_focal_length is None
            else float(target_focal_length)
        )
        first_order_guard_values = (
            target_f_number,
            first_order_preferred_relative_error,
            first_order_hard_relative_error,
        )
        first_order_guard_enabled = any(
            value is not None for value in first_order_guard_values
        )
        if first_order_guard_enabled and any(
            value is None for value in first_order_guard_values
        ):
            raise ValueError(
                "启用一阶硬门控时必须同时提供 target_f_number、"
                "first_order_preferred_relative_error 和 "
                "first_order_hard_relative_error。"
            )
        if first_order_guard_enabled:
            target_f_number = float(target_f_number)
            first_order_preferred_relative_error = float(
                first_order_preferred_relative_error
            )
            first_order_hard_relative_error = float(
                first_order_hard_relative_error
            )
            if not math.isfinite(target_f_number) or target_f_number <= 0.0:
                raise ValueError("target_f_number 必须为正的有限值。")
            if (
                not math.isfinite(first_order_preferred_relative_error)
                or not math.isfinite(first_order_hard_relative_error)
                or first_order_preferred_relative_error <= 0.0
                or first_order_hard_relative_error
                < first_order_preferred_relative_error
            ):
                raise ValueError(
                    "一阶误差门限必须满足 0 < preferred <= hard。"
                )
        target_rfov = self.rfov if target_rfov is None else float(target_rfov)
        if not math.isfinite(target_focal_length) or target_focal_length <= 0.0:
            raise ValueError("target_focal_length 必须为正的有限值。")
        if not math.isfinite(target_rfov) or target_rfov <= 0.0:
            raise ValueError("target_rfov 必须为正的有限弧度值。")

        first_order_before = None
        if first_order_guard_enabled:
            first_order_before = self._measure_first_order_state()
            initial_focal_error = self._first_order_relative_error(
                first_order_before[0], target_focal_length
            )
            initial_f_number_error = self._first_order_relative_error(
                first_order_before[1], target_f_number
            )
            if (
                initial_focal_error > first_order_hard_relative_error + EPSILON
                or initial_f_number_error
                > first_order_hard_relative_error + EPSILON
            ):
                raise ValueError(
                    "初始处方超出一阶硬上限："
                    f"EFL={first_order_before[0]:.6g} mm "
                    f"(误差 {initial_focal_error:.3%})，"
                    f"F/#={first_order_before[1]:.6g} "
                    f"(误差 {initial_f_number_error:.3%})，"
                    f"硬上限为 {first_order_hard_relative_error:.3%}。"
                )

        # 结果目录和日志记录器
        if result_dir is None:
            result_dir = (
                f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-DesignLens"
            )

        os.makedirs(result_dir, exist_ok=True)
        if not logging.getLogger().hasHandlers():
            logger = logging.getLogger()
            logger.setLevel("DEBUG")
            fmt = logging.Formatter(
                "%(asctime)s:%(levelname)s:%(message)s", "%Y-%m-%d %H:%M:%S"
            )
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel("INFO")
            fh = logging.FileHandler(f"{result_dir}/output.log", encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel("INFO")
            logger.addHandler(sh)
            logger.addHandler(fh)
        logging.info(
            f"lr:{lrs}, iterations:{iterations}, num_ring:{num_ring}, num_arm:{num_arm}, rays_per_fov:{spp}."
        )
        logging.info(
            "损失权重：RMS=%g，MTF代理=%g，有效率=%g，场映射=%g，"
            "正则=%g，场映射最坏项=%g，MTF最坏项=%g。",
            w_rms,
            w_mtf,
            w_valid,
            w_field,
            w_reg,
            field_mapping_max_weight,
            mtf_max_weight,
        )
        logging.info(
            "训练光线重采样间隔：%d（0 表示仅检查点）；MTF代理频率=%s，目标=%g。",
            ray_resample_interval,
            "关闭" if mtf_frequency_cy_mm is None else f"{mtf_frequency_cy_mm:g} cy/mm",
            mtf_target,
        )
        if first_order_guard_enabled:
            logging.info(
                "一阶硬门控：目标 EFL=%.6g mm、F/#=%.6g，"
                "首选误差 %.3f%%、硬上限 %.3f%%；"
                "初始 EFL=%.6g mm、F/#=%.6g。",
                target_focal_length,
                target_f_number,
                100.0 * first_order_preferred_relative_error,
                100.0 * first_order_hard_relative_error,
                first_order_before[0],
                first_order_before[1],
            )
        logging.info(
            "If Out-of-Memory, try to reduce num_ring, num_arm, and rays_per_fov."
        )

        # 优化器与调度器
        optimizer = self.get_optimizer(lrs, optim_mat=optim_mat)
        warmup_steps = self._optimization_warmup_steps(iterations)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=iterations,
        )
        logging.info("学习率预热步数：%d。", warmup_steps)

        # 训练循环
        pbar = tqdm(
            total=iterations,
            desc="Progress",
            postfix={"loss_rms": 0},
        )

        def sample_training_batch():
            """采样一批 RMS、场映射和可选 MTF 代理训练光线。"""

            self.calc_pupil()
            sampled_rays = []
            for wavelength in self.wvln_rgb:
                sampled_rays.append(
                    self.sample_ring_arm_rays(
                        num_ring=num_ring,
                        num_arm=num_arm,
                        spp=spp,
                        depth=depth,
                        wvln=wavelength,
                        scale_pupil=1.05,
                        sample_more_off_axis=sample_more_off_axis,
                        max_fov_rad=target_rfov,
                    )
                )

            # 以任务目标焦距的理想针孔投影作为像高/畸变参考。不要调用
            # psf_center(method="pinhole")，因为它读取的是处方实测焦距；
            # 否则焦距漂移会同步移动目标，错误焦距仍可能得到低畸变。
            points_obj = sampled_rays[0].o[:, :, 0, :]
            sampled_pinhole_ref = target_focal_length * (
                points_obj[..., :2] / points_obj[..., 2:].clamp_max(-EPSILON)
            )
            sampled_chief_rays = None
            sampled_chief_target = None
            if field_mapping_use_chief_ray:
                if self.aper_idx != 0:
                    raise ValueError(
                        "field_mapping_use_chief_ray 当前只支持前置光阑（aper_idx=0）。"
                    )
                if field_mapping_num_points >= 2:
                    half_field_deg = target_rfov * 180.0 / math.pi
                    field_angles = torch.linspace(
                        0.0,
                        half_field_deg,
                        field_mapping_num_points,
                        device=self.device,
                    )
                    field_angle_values = field_angles.detach().cpu().tolist()
                    sampled_chief_rays = [
                        (
                            self.sample_from_fov(
                                fov_x=field_angle_values,
                                fov_y=0.0,
                                depth=float("inf"),
                                num_rays=1,
                                wvln=wavelength,
                                scale_pupil=0.0,
                            ),
                            self.sample_from_fov(
                                fov_x=0.0,
                                fov_y=field_angle_values,
                                depth=float("inf"),
                                num_rays=1,
                                wvln=wavelength,
                                scale_pupil=0.0,
                            ),
                        )
                        for wavelength in self.wvln_rgb
                    ]
                    target_height = target_focal_length * torch.tan(
                        field_angles * math.pi / 180.0
                    )
                    zeros = torch.zeros_like(target_height)
                    sampled_chief_target = torch.stack(
                        [
                            torch.stack([target_height, zeros], dim=-1),
                            torch.stack([zeros, target_height], dim=-1),
                        ],
                        dim=0,
                    )
                else:
                    sampled_chief_rays = [
                        self.sample_from_points(
                            points=points_obj,
                            num_rays=1,
                            wvln=wavelength,
                            scale_pupil=0.0,
                        )
                        for wavelength in self.wvln_rgb
                    ]

            sampled_mtf_rays = None
            if w_mtf > 0.0:
                half_field_deg = target_rfov * 180.0 / math.pi
                mtf_field_degrees = [
                    fraction * half_field_deg for fraction in mtf_field_fractions
                ]
                sampled_mtf_rays = [
                    self.sample_from_fov(
                        fov_x=0.0,
                        fov_y=mtf_field_degrees,
                        depth=float("inf"),
                        num_rays=spp,
                        wvln=wavelength,
                        scale_pupil=1.0,
                    )
                    for wavelength in self.wvln_rgb
                ]

            return (
                sampled_rays,
                sampled_pinhole_ref,
                sampled_chief_rays,
                sampled_chief_target,
                sampled_mtf_rays,
            )

        for i in range(iterations):
            # ===> 评估透镜
            is_checkpoint = i % test_per_iter == 0
            if is_checkpoint:
                with torch.no_grad():
                    if shape_control and i > 0:
                        self.correct_shape()

                    self._save_optimization_checkpoint(
                        result_dir,
                        iteration=i,
                        run_analysis=checkpoint_analysis,
                    )

            should_resample = is_checkpoint or (
                ray_resample_interval > 0 and i % ray_resample_interval == 0
            )
            if should_resample:
                with torch.no_grad():
                    (
                        rays_backup,
                        pinhole_ref,
                        chief_rays_backup,
                        chief_mapping_target,
                        mtf_rays_backup,
                    ) = sample_training_batch()

            # ===> 通过最小化 RMS 优化透镜
            # 先追迹绿色：其质心设置 center_ref 并驱动畸变惩罚；
            # 红色和蓝色复用同一个 center_ref。
            loss_rms_ls = []
            loss_valid_ls = []
            valid_ratio_ls = []
            loss_field_mapping = torch.tensor(0.0, device=self.device)
            field_mapping_positions = []
            w_mask = None
            center_ref = None
            wvln_order = [1, 0, 2]  # 绿色、红色、蓝色
            for wv_idx in wvln_order:
                # 将光线追迹到传感器，[num_ring, num_arm, num_rays, 3]
                ray = rays_backup[wv_idx].clone()
                ray = self.trace2sensor(ray)
                centroid_xy = ray.centroid()[..., :2]
                if field_mapping_use_chief_ray:
                    if chief_mapping_target is not None:
                        sagittal_ray, meridional_ray = chief_rays_backup[wv_idx]
                        sagittal_ray = self.trace2sensor(sagittal_ray.clone())
                        meridional_ray = self.trace2sensor(meridional_ray.clone())
                        mapping_xy = torch.stack(
                            [
                                sagittal_ray.o[..., 0, :2].abs(),
                                meridional_ray.o[..., 0, :2].abs(),
                            ],
                            dim=0,
                        )
                    else:
                        chief_ray = self.trace2sensor(chief_rays_backup[wv_idx].clone())
                        mapping_xy = chief_ray.o[..., 0, :2]
                else:
                    mapping_xy = centroid_xy

                if center_ref is None:
                    # 传感器处的绿色质心，shape [num_ring, num_arm, 2]。
                    # 分离梯度，使 RMS 梯度改变光斑形状而非位置；
                    # 光斑位置由目标场映射损失处理。
                    center_ref = centroid_xy.detach().unsqueeze(-2)

                if field_mapping_all_wavelengths or not field_mapping_positions:
                    field_mapping_positions.append(mapping_xy)

                # 光线相对中心的误差及有效掩码
                ray_valid = ray.is_valid
                loss_valid, valid_ratio = self._ray_validity_loss(
                    ray_valid, min_valid_ratio=min_valid_ratio
                )
                loss_valid_ls.append(loss_valid)
                valid_ratio_ls.append(valid_ratio)
                ray_err = ray.o[..., :2] - center_ref

                # 每个视场点的 MSE，shape [num_ring, num_arm]
                mse = self._masked_field_mse(
                    ray_err, ray_valid, invalid_rms=invalid_rms_proxy
                )

                # 权重掩码
                if w_mask is None:
                    w_mask = mse.detach().sqrt().clone()
                    w_mask = w_mask / (w_mask.mean() + EPSILON)
                    w_mask[0, :] = 1.0

                # RMS 与加权损失
                l_rms = torch.clamp(mse, min=EPSILON).sqrt()
                l_rms_weighted = (l_rms * w_mask).sum() / (w_mask.sum() + EPSILON)
                loss_rms_ls.append(l_rms_weighted)

            mapping_positions = torch.stack(field_mapping_positions, dim=0)
            if chief_mapping_target is None:
                mapping_targets = pinhole_ref.unsqueeze(0).expand_as(mapping_positions)
            else:
                mapping_targets = chief_mapping_target.unsqueeze(0).expand_as(
                    mapping_positions
                )
            loss_field_mapping = self._target_field_mapping_loss(
                mapping_positions,
                mapping_targets,
                tolerance=self.distortion_max,
                max_weight=field_mapping_max_weight,
            )

            # 所有波长的 RMS 损失
            loss_rms = sum(loss_rms_ls) / len(loss_rms_ls)

            # 在与验收一致的无穷远 Y 场上，使用方向相位方差作为固定频率
            # MTF 的稳健代理。当前光斑远大于一个 MTF 周期时，直接经验 OTF
            # 会落入有限光线噪声底；方向方差仍能提供单调、低噪声的梯度。
            loss_mtf = torch.tensor(0.0, device=self.device)
            geometric_mtf_min = torch.tensor(float("nan"), device=self.device)
            if w_mtf > 0.0:
                mtf_violations = []
                geometric_mtf_values = []
                for sampled_mtf_ray in mtf_rays_backup:
                    mtf_ray = self.trace2sensor(sampled_mtf_ray.clone())
                    mtf_valid_loss, mtf_valid_ratio = self._ray_validity_loss(
                        mtf_ray.is_valid, min_valid_ratio=min_valid_ratio
                    )
                    loss_valid_ls.append(mtf_valid_loss)
                    valid_ratio_ls.append(mtf_valid_ratio)
                    mtf_violations.append(
                        self._fixed_frequency_mtf_surrogate_violation(
                            mtf_ray.o[..., :2],
                            mtf_ray.is_valid,
                            frequency_cy_mm=mtf_frequency_cy_mm,
                            target_mtf=mtf_target,
                            invalid_rms=invalid_rms_proxy,
                        )
                    )
                    with torch.no_grad():
                        geometric_mtf_values.append(
                            self._fixed_frequency_geometric_mtf(
                                mtf_ray.o[..., :2],
                                mtf_ray.is_valid,
                                frequency_cy_mm=mtf_frequency_cy_mm,
                            )
                        )

                mtf_violation = torch.stack(mtf_violations, dim=0)
                loss_mtf = (
                    mtf_violation.mean()
                    + mtf_max_weight * mtf_violation.max()
                )
                geometric_mtf_min = torch.stack(
                    geometric_mtf_values, dim=0
                ).min()

            loss_valid = sum(loss_valid_ls) / len(loss_valid_ls)
            valid_ratio_min = torch.stack(
                [valid_ratio.min() for valid_ratio in valid_ratio_ls]
            ).min()

            # 总损失
            loss_reg, loss_dict = self.loss_reg()
            valid_penalty_scale = max(float(self.r_sensor), 1.0)
            L_total = (
                w_rms * loss_rms
                + w_mtf * loss_mtf
                + w_valid * valid_penalty_scale * loss_valid
                + w_field * loss_field_mapping
                + w_reg * loss_reg
            )

            # 反向传播
            optimizer.zero_grad()
            if valid_ratio_min <= 0.0:
                logging.warning(
                    "第 %d 次迭代至少有一个视场完全没有有效光线；"
                    "挡光惩罚已加入总损失。",
                    i,
                )
            if not torch.isfinite(L_total):
                logging.warning(
                    "第 %d 次迭代产生非有限总损失，已跳过本次参数更新。",
                    i,
                )
            else:
                L_total.backward()

                # 大口径、多项式非球面起点可能让少量无效光线产生 NaN/Inf
                # 梯度。若直接交给 Adam，一个坏梯度会永久污染动量状态并使
                # 整个处方退化。将这些局部坏梯度置零，并按参数组独立裁剪；
                # 这样高阶非球面系数不会压低曲率、间距等其他参数组的梯度。
                trainable_params, nonfinite_gradients = (
                    self._sanitize_and_clip_gradients(optimizer, max_norm=100.0)
                )
                if nonfinite_gradients:
                    logging.warning(
                        "第 %d 次迭代忽略了 %d 个非有限梯度分量。",
                        i,
                        nonfinite_gradients,
                    )

                # 参数更新前保存快照。若后端数值异常仍产生非有限参数，则恢复
                # 当前步并清空 Adam 状态，避免后续迭代持续传播污染。
                snapshots = [parameter.detach().clone() for parameter in trainable_params]
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in trainable_params):
                    logging.warning(
                        "第 %d 次迭代产生非有限参数，已回滚本次更新。",
                        i,
                    )
                    with torch.no_grad():
                        for parameter, snapshot in zip(trainable_params, snapshots):
                            parameter.copy_(snapshot)
                    optimizer.state.clear()
                else:
                    first_order_after = None
                    first_order_accepted = True
                    if first_order_guard_enabled:
                        try:
                            first_order_after = self._measure_first_order_state()
                        except Exception as error:
                            first_order_accepted = False
                            logging.warning(
                                "第 %d 次迭代的一阶参数复测失败：%s。",
                                i,
                                error,
                            )
                        else:
                            focal_error_after = self._first_order_relative_error(
                                first_order_after[0], target_focal_length
                            )
                            f_number_error_after = self._first_order_relative_error(
                                first_order_after[1], target_f_number
                            )
                            logging.info(
                                "第 %d 次迭代更新后 EFL=%.6g mm（误差 %.3f%%），"
                                "F/#=%.6g（误差 %.3f%%）。",
                                i,
                                first_order_after[0],
                                100.0 * focal_error_after,
                                first_order_after[1],
                                100.0 * f_number_error_after,
                            )
                            first_order_accepted = (
                                self._first_order_update_is_acceptable(
                                    focal_length_before=first_order_before[0],
                                    f_number_before=first_order_before[1],
                                    focal_length_after=first_order_after[0],
                                    f_number_after=first_order_after[1],
                                    target_focal_length=target_focal_length,
                                    target_f_number=target_f_number,
                                    preferred_relative_error=(
                                        first_order_preferred_relative_error
                                    ),
                                    hard_relative_error=(
                                        first_order_hard_relative_error
                                    ),
                                )
                            )

                    if not first_order_accepted:
                        logging.warning(
                            "第 %d 次迭代未通过 EFL/F 数一阶硬门控，"
                            "已回滚本次更新。",
                            i,
                        )
                        with torch.no_grad():
                            for parameter, snapshot in zip(
                                trainable_params, snapshots
                            ):
                                parameter.copy_(snapshot)
                        optimizer.state.clear()
                        try:
                            first_order_before = self._measure_first_order_state()
                        except Exception as error:
                            raise RuntimeError(
                                "一阶门控回滚后无法恢复 EFL/F 数缓存。"
                            ) from error
                    else:
                        # 用同一批采样光线快速复追迹更新后的处方。硬有效性掩码没有
                        # 可用梯度，因此采用接受/回滚规则真正执行最低有效率约束。
                        # 初始有效率低于目标时只禁止继续恶化，不会将处方永久冻结。
                        try:
                            validity_rays = list(rays_backup)
                            if mtf_rays_backup is not None:
                                validity_rays.extend(mtf_rays_backup)
                            valid_ratio_after = self._trace_min_valid_ratio(
                                validity_rays
                            )
                        except Exception as error:
                            valid_ratio_after = torch.tensor(
                                float("nan"), device=self.device
                            )
                            logging.warning(
                                "第 %d 次迭代的更新后有效率复追迹失败：%s",
                                i,
                                error,
                            )

                        if not self._validity_update_is_acceptable(
                            valid_ratio_min,
                            valid_ratio_after,
                            min_valid_ratio=min_valid_ratio,
                        ):
                            logging.warning(
                                "第 %d 次迭代使最低有效率从 %.3f 变为 %.3f，"
                                "已回滚本次更新（目标下限 %.3f）。",
                                i,
                                valid_ratio_min.item(),
                                valid_ratio_after.item(),
                                min_valid_ratio,
                            )
                            with torch.no_grad():
                                for parameter, snapshot in zip(
                                    trainable_params, snapshots
                                ):
                                    parameter.copy_(snapshot)
                            optimizer.state.clear()
                            if first_order_guard_enabled:
                                try:
                                    first_order_before = (
                                        self._measure_first_order_state()
                                    )
                                except Exception as error:
                                    raise RuntimeError(
                                        "有效率门控回滚后无法恢复 EFL/F 数缓存。"
                                    ) from error
                        elif first_order_guard_enabled:
                            first_order_before = first_order_after
            scheduler.step()

            pbar.set_postfix(
                loss_total=L_total.item(),
                loss_rms=loss_rms.item(),
                weighted_rms=(w_rms * loss_rms).item(),
                loss_mtf=loss_mtf.item(),
                weighted_mtf=(w_mtf * loss_mtf).item(),
                geometric_mtf_min=geometric_mtf_min.item(),
                loss_valid=loss_valid.item(),
                valid_min=valid_ratio_min.item(),
                loss_field=loss_field_mapping.item(),
                weighted_field=(w_field * loss_field_mapping).item(),
                weighted_reg=(w_reg * loss_reg).item(),
                **loss_dict,
            )
            pbar.update(1)

        pbar.close()
        # ``iterations`` 表示真实参数更新次数；循环结束后额外保存最终状态，但
        # 不再执行一次隐藏的优化更新。这样 ``iterations=1`` 就确实只更新一次。
        with torch.no_grad():
            self._save_optimization_checkpoint(
                result_dir,
                iteration=iterations,
                run_analysis=checkpoint_analysis,
            )

    # ====================================================================================
    # 优化器辅助方法
    # ====================================================================================
    def _save_optimization_checkpoint(self, result_dir, iteration, run_analysis=True):
        """保存优化检查点，并按需生成耗时的完整分析图。"""

        checkpoint_base = f"{result_dir}/iter{iteration}"
        self.write_lens_json(f"{checkpoint_base}.json")
        if run_analysis:
            self.analysis(checkpoint_base)

    @staticmethod
    def _optimization_warmup_steps(iterations, max_warmup=100):
        """返回约 10% 的预热步数，并保证短烟雾测试具有非零学习率。"""

        if iterations < 1:
            raise ValueError("iterations 必须为正整数。")
        if max_warmup < 0:
            raise ValueError("max_warmup 必须大于或等于 0。")
        return min(max_warmup, iterations // 10)

    @staticmethod
    def _sanitize_and_clip_gradients(optimizer, max_norm=100.0):
        """清理非有限梯度，并对每个优化器参数组独立裁剪范数。

        返回参与本次更新的参数列表和被替换的非有限梯度分量数量。按组裁剪可
        避免单个高阶非球面参数组的极大梯度缩小其他组的有效更新。
        """
        if max_norm <= 0.0:
            raise ValueError("max_norm 必须大于 0。")

        trainable_params = []
        nonfinite_gradients = 0
        for group in optimizer.param_groups:
            group_params = []
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                trainable_params.append(parameter)
                group_params.append(parameter)
                invalid = ~torch.isfinite(parameter.grad)
                if invalid.any():
                    nonfinite_gradients += int(invalid.sum().item())
                    parameter.grad = torch.nan_to_num(
                        parameter.grad, nan=0.0, posinf=0.0, neginf=0.0
                    )

            if group_params:
                torch.nn.utils.clip_grad_norm_(group_params, max_norm=max_norm)

        return trainable_params, nonfinite_gradients

    def find_diff_surf(self):
        """获取可微/可优化的表面索引。

        返回透镜设计期间可优化的表面索引列表，并从优化中排除光阑表面。

        返回：
            diff_surf_range (list or range)：不含光阑的表面索引。
        """
        if self.aper_idx is None:
            diff_surf_range = range(len(self.surfaces))
        else:
            diff_surf_range = list(range(0, self.aper_idx)) + list(
                range(self.aper_idx + 1, len(self.surfaces))
            )
        return diff_surf_range

    def get_optimizer_params(
        self,
        lrs=[1e-4, 1e-4, 1e-2, 1e-4],
        optim_mat=False,
        optim_surf_range=None,
    ):
        """使用按类型设置的学习率构建逐表面 Adam 参数组。

        按表面类型分派，收集每个表面的可训练参数以及传感器距离，
        组成优化器参数组列表。

        建议：
            手机透镜：[d, c, k, a]，[1e-4, 1e-4, 1e-1, 1e-4]。
            相机透镜：[d, c, 0, 0]，[1e-3, 1e-4, 0, 0]。

        参数：
            lrs (list, optional)：[d, c, k, a] 参数组的学习率。默认值为
                [1e-4, 1e-4, 1e-2, 1e-4]。
            optim_mat (bool, optional)：是否优化材料参数。默认值为 False。
            optim_surf_range (list or None, optional)：要优化的表面索引。
                为 None 时使用全部表面。默认值为 None。

        返回：
            params (list)：优化器参数组字典列表。

        异常：
            Exception：某个表面类型不支持优化时抛出。
        """
        # 查找要优化的表面
        if optim_surf_range is None:
            # optim_surf_range = self.find_diff_surf()
            optim_surf_range = range(len(self.surfaces))

        # 优化透镜表面参数
        params = []
        for surf_idx in optim_surf_range:
            surf = self.surfaces[surf_idx]

            if isinstance(surf, Aperture):
                params += surf.get_optimizer_params(lrs=[lrs[0]])

            elif isinstance(surf, Aspheric):
                params += surf.get_optimizer_params(lrs=lrs[:4], optim_mat=optim_mat)

            elif isinstance(surf, Phase):
                # Phase 表面使用 [d_lr, coeff_lr]。若提供第 5 个 lr，则将其
                # 专用于系数；否则回退到最后一个 lr，以避免标准 4 元素
                # lrs 约定触发 IndexError。
                coeff_lr = lrs[4] if len(lrs) > 4 else lrs[-1]
                params += surf.get_optimizer_params(lrs=[lrs[0], coeff_lr])

            # elif isinstance(surf, GaussianRBF):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            # elif isinstance(surf, NURBS):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Plane):
                params += surf.get_optimizer_params(lrs=[lrs[0]], optim_mat=optim_mat)

            # elif isinstance(surf, PolyEven):
            #     params += surf.get_optimizer_params(lrs=lr, optim_mat=optim_mat)

            elif isinstance(surf, Spheric):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            elif isinstance(surf, ThinLens):
                params += surf.get_optimizer_params(
                    lrs=[lrs[0], lrs[1]], optim_mat=optim_mat
                )

            else:
                raise Exception(
                    f"Surface type {surf.__class__.__name__} is not supported for optimization yet."
                )

        # 优化传感器位置
        self.d_sensor.requires_grad = True
        params += [{"params": self.d_sensor, "lr": lrs[0]}]

        return params

    def get_optimizer(
        self,
        lrs=[1e-4, 1e-4, 1e-1, 1e-4],
        optim_surf_range=None,
        optim_mat=False,
    ):
        """为所有可训练透镜参数构建 Adam 优化器。

        参数：
            lrs (list, optional)：[d, c, k, ai] 参数组的学习率。默认值为
                [1e-4, 1e-4, 1e-1, 1e-4]。
            optim_surf_range (list or None, optional)：要优化的表面索引。
                为 None 时包含全部表面。默认值为 None。
            optim_mat (bool, optional)：是否包含材料参数 (n, V)。默认值为 False。

        返回：
            optimizer (torch.optim.Adam)：配置完成的 Adam 优化器。
        """
        # 获取优化器
        params = self.get_optimizer_params(
            lrs=lrs, optim_surf_range=optim_surf_range, optim_mat=optim_mat
        )
        optimizer = torch.optim.Adam(params)
        # optimizer = torch.optim.SGD(params)
        return optimizer
