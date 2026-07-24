# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""前向与反向 Monte-Carlo 积分函数。"""

import torch
import torch.nn.functional as F

from ..config import EPSILON


def forward_integral(ray, ps, ks, pointc=None, interpolate=True):
    """将光线束积分到像素网格上的可微 Monte-Carlo 积分。

    将光线命中位置分配到以 `pointc` 为中心的 `ks` x `ks` 网格中；当 `pointc`
    为 None 时使用有效光线质心。相干模式下累加复振幅
    `sqrt(|dz|) * exp(i * phase)`，非相干模式下累加单位强度。所有 `N` 个视场点
    通过批量 `index_put_(accumulate=True)` 调用散射到各自的输出切片。

    参数：
        ray (Ray): 已追迹光线束，原点 `ray.o` 的形状为 [N, spp, 3]；单个视场点
            时为 [spp, 3]。
        ps (float): 像素尺寸 [mm]。
        ks (int): 方形输出网格的像素尺寸。
        pointc (torch.Tensor or None, optional): 各视场点的参考中心 [mm]，形状为
            [N, 2]。为 None 时使用有效光线质心，默认为 None。
        interpolate (bool, optional): 为 True 时，每条光线通过双线性权重将贡献
            分配给周围四个像素；为 False 时，将光线硬分配至向下取整的像素，
            速度更快但像素内位置没有梯度。默认为 True。

    返回：
        grid (torch.Tensor): 累积波场，形状为 [N, ks, ks]；单个输入点时为
            [ks, ks]。当 `ray.is_coherent` 为 True 时 dtype 为复数，否则为浮点数。
    """
    if ray.o.ndim == 2:
        single_point = True
        ray = ray.unsqueeze(0)
    else:
        single_point = False

    points = ray.o[..., :2]      # [N, spp, 2]
    valid = ray.is_valid         # [N, spp]
    N, spp = valid.shape
    device = valid.device

    # 将网格中心放在 pointc；若未提供，则使用有效光线质心。
    if pointc is None:
        pointc = (points * valid.unsqueeze(-1)).sum(-2) / valid.unsqueeze(-1).sum(
            -2
        ).add(EPSILON)
    points_shift = points - pointc.unsqueeze(-2)    # [N, spp, 2]

    # 剔除落在网格窗口之外的点。
    field_max = (ks / 2 - 0.5) * ps
    in_window = (
        (points_shift[..., 0].abs() < (field_max - 0.001 * ps))
        & (points_shift[..., 1].abs() < (field_max - 0.001 * ps))
    )
    valid = valid * in_window.to(valid.dtype)

    # 每条光线的强度（实数）或复振幅。
    if ray.is_coherent:
        # 加入 EPSILON：sqrt'(0) 为无穷大，会使陡峭光线（dz~0）产生 NaN 梯度。
        amp = torch.sqrt(ray.d[..., 2].abs() + EPSILON)  # sqrt(|dz|)
        opl = ray.opl.squeeze(-1)                       # [N, spp]
        opl_min = opl.min(dim=-1, keepdim=True).values
        wvln_mm = ray.wvln * 1e-3
        phase = torch.fmod((opl - opl_min) / wvln_mm, 1) * (2 * torch.pi)
        value = amp * torch.exp(1j * phase)
    else:
        value = torch.ones_like(valid)

    # 小数像素索引：y 向上对应行向下，x 向右对应列向右。
    # 像素中心位于 [0, ks-1] 范围的整数网格上。
    norm_row = (field_max - points_shift[..., 1]) / (2 * field_max)
    norm_col = (points_shift[..., 0] + field_max) / (2 * field_max)
    pix_row = norm_row * (ks - 1)
    pix_col = norm_col * (ks - 1)
    r_floor = pix_row.floor()
    c_floor = pix_col.floor()

    r0 = r_floor.long().clamp(0, ks - 1)
    c0 = c_floor.long().clamp(0, ks - 1)

    masked_value = valid * value

    # 批量散射：所有 N 个视场点通过支持批次的 ``index_put_`` 同时累加；当索引
    # 元组带有批次维度时，该方法支持逐批次累加。
    batch_idx = torch.arange(N, device=device).unsqueeze(-1).expand(N, spp)
    grid = torch.zeros(N, ks, ks, dtype=value.dtype, device=device)
    if interpolate:
        w_r = pix_row - r_floor
        w_c = pix_col - c_floor
        r1 = (r0 + 1).clamp(0, ks - 1)
        c1 = (c0 + 1).clamp(0, ks - 1)
        grid.index_put_((batch_idx, r0, c0), (1 - w_r) * (1 - w_c) * masked_value, accumulate=True)
        grid.index_put_((batch_idx, r0, c1), (1 - w_r) * w_c * masked_value, accumulate=True)
        grid.index_put_((batch_idx, r1, c0), w_r * (1 - w_c) * masked_value, accumulate=True)
        grid.index_put_((batch_idx, r1, c1), w_r * w_c * masked_value, accumulate=True)
    else:
        grid.index_put_((batch_idx, r0, c0), masked_value, accumulate=True)

    if single_point:
        grid = grid.squeeze(0)
        ray = ray.squeeze(0)    # 恢复调用方的光线形状，因为 unsqueeze 会原地修改

    return grid

def backward_integral(
    ray,
    img_obj,
    ps,
    interpolate=True,
    energy_correction=None,
    vignetting=False,
):
    """用于光线追迹渲染的反向 Monte-Carlo 积分。

    在每条光线的命中位置采样输入图像，并沿每像素采样数（spp）轴求平均以渲染
    输出。输入图像始终在各侧复制填充一个像素，使落在边缘半个像素范围内的
    光线仍可进行双线性采样，而不会被无提示截断。

    参数：
        ray (Ray): 光线对象，`ray.o` 形状为 [h, w, spp, 3]，位置单位为 [mm]。
        img_obj (torch.Tensor): 源图像，形状为 [B, C, H, W]，空间尺寸 H、W
            从该张量读取。
        ps (float): 像素尺寸 [mm]。
        interpolate (bool, optional): 为 True 时对周围四个像素进行双线性采样；
            为 False 时采用最近像素采样。默认为 True。
        energy_correction (torch.Tensor or None, optional): 逐光线权重张量，形状为
            [h, w, spp, 1]，例如 `ray.en`。提供时将其作为重要性权重；默认的非
            渐晕模式下，它同时进入分子和分母，从而得到正确的加权 Monte-Carlo
            均值；渐晕模式下仅缩放分子，分母固定。默认为 None，即均匀权重。
        vignetting (bool, optional): 为 True 时除以固定分母
            `torch.numel(ray.is_valid)`，而不是权重总和；命中光线较少或衰减较强
            的像素会更暗，从而表现机械渐晕。默认为 False。

    返回：
        output (torch.Tensor): 渲染图像，形状为 [B, C, h, w]。

    异常：
        Exception: 当 `ray.is_coherent` 为 True 时抛出，因为不支持相干反向积分。
    """
    assert len(img_obj.shape) == 4
    H, W = img_obj.shape[-2:]
    p = ray.o[..., :2]  # 形状为 [h, w, spp, 2]
    img_obj = F.pad(img_obj, (1, 1, 1, 1), "replicate")

    # 将光线位置转换为 uv 坐标
    u = torch.clamp(W / 2 + p[..., 0] / ps, min=-0.99, max=W - 0.01)
    v = torch.clamp(H / 2 + p[..., 1] / ps, min=0.01, max=H + 0.99)

    # (idx_i, idx_j) 表示左上角参考像素；索引不携带梯度。
    # 因为进行了填充，所以索引需要加 1。
    idx_i = H - v.ceil().long() + 1
    idx_j = u.floor().long() + 1

    # 梯度保存在插值权重参数中
    w_i = v - v.floor().long()
    w_j = u.ceil().long() - u

    if ray.is_coherent:
        raise Exception("Backward coherent integral needs to be checked.")

    # 沿 spp 轴（最后一维）进行 Monte-Carlo 积分。
    if interpolate:
        # 双线性散射
        # img_obj [B, C, H+2, W+2], idx_i/idx_j [h, w, spp] -> out_img [B, C, h, w, spp]
        out_img = img_obj[..., idx_i, idx_j] * w_i * w_j
        out_img += img_obj[..., idx_i + 1, idx_j] * (1 - w_i) * w_j
        out_img += img_obj[..., idx_i, idx_j + 1] * w_i * (1 - w_j)
        out_img += img_obj[..., idx_i + 1, idx_j + 1] * (1 - w_i) * (1 - w_j)
    else:
        out_img = img_obj[..., idx_i, idx_j]

    # 额外的逐光线能量校正因子，例如用于非均匀光线采样。
    weight = ray.is_valid
    if energy_correction is not None:
        weight = weight * energy_correction.squeeze(-1)

    # 以权重总和归一化；渐晕模式使用固定分母，从而得到 Monte-Carlo 均值。
    if vignetting:
        output = torch.sum(out_img * weight, -1) / torch.numel(ray.is_valid)
    else:
        output = torch.sum(out_img * weight, -1) / (torch.sum(weight, -1) + EPSILON)

    return output

def assign_points_to_pixels(
    points,
    mask,
    ks,
    x_range,
    y_range,
    value,
    interpolate=True,
):
    """将点样本散射到 `ks` x `ks` 像素网格上。

    使用 `index_put_(accumulate=True)`，将每个点的 `value`（强度或复振幅）
    分配到 `x_range` x `y_range` 覆盖的网格中。支持非相干和相干光线追迹。
    受高级索引散射方式限制，目前仅处理单个点光源。

    参数：
        points (torch.Tensor): 样本位置 [mm]，以 (x, y) 表示，形状为 [spp, 2]。
        mask (torch.Tensor): 有效性掩码，形状为 [spp]。
        ks (int): 方形输出网格的像素尺寸。
        x_range (tuple): 网格 x 范围 (x_min, x_max) [mm]。
        y_range (tuple): 网格 y 范围 (y_min, y_max) [mm]。
        value (torch.Tensor): 每个点要累加的值（强度或复振幅），形状为 [spp]。
        interpolate (bool, optional): 为 True 时通过双线性权重将每个点分配到
            周围四个像素；为 False 时硬分配到向下取整的像素。默认为 True。

    返回：
        grid (torch.Tensor): 累积强度或复振幅，形状为 [ks, ks]，dtype 与
            `value` 一致。
    """
    # 参数
    device = points.device
    x_min, x_max = x_range
    y_min, y_max = y_range

    # 将点归一化至 [0, 1]，直接计算且不分配中间张量
    norm_0 = (points[:, 1] - y_max) / (y_min - y_max)
    norm_1 = (points[:, 0] - x_min) / (x_max - x_min)

    # 检查点是否处于有效范围内
    valid_points = (norm_0 >= 0) & (norm_0 <= 1) & (norm_1 >= 0) & (norm_1 <= 1)
    mask = mask * valid_points

    if interpolate:
        # 计算浮点像素索引
        pix_0 = norm_0 * (ks - 1)
        pix_1 = norm_1 * (ks - 1)
        pix_0_floor = pix_0.floor()
        pix_1_floor = pix_1.floor()

        # 双线性权重
        w_b = pix_0 - pix_0_floor
        w_r = pix_1 - pix_1_floor
        w_b_1 = 1 - w_b
        w_r_1 = 1 - w_r

        # 四个角点的像素索引，并限制在有效范围内
        r0 = pix_0_floor.long().clamp(0, ks - 1)
        c0 = pix_1_floor.long().clamp(0, ks - 1)
        r1 = (r0 + 1).clamp(0, ks - 1)
        c1 = (c0 + 1).clamp(0, ks - 1)

        # 预先计算一次应用掩码后的值
        masked_value = mask * value

        # 使用高级索引累加各对应像素的计数
        grid = torch.zeros(ks, ks, dtype=value.dtype, device=device)
        grid.index_put_((r0, c0), w_b_1 * w_r_1 * masked_value, accumulate=True)
        grid.index_put_((r0, c1), w_b_1 * w_r * masked_value, accumulate=True)
        grid.index_put_((r1, c0), w_b * w_r_1 * masked_value, accumulate=True)
        grid.index_put_((r1, c1), w_b * w_r * masked_value, accumulate=True)

    else:
        pix_0 = (norm_0 * (ks - 1)).floor().long().clamp(0, ks - 1)
        pix_1 = (norm_1 * (ks - 1)).floor().long().clamp(0, ks - 1)

        grid = torch.zeros(ks, ks, dtype=value.dtype, device=device)
        grid.index_put_((pix_0, pix_1), mask * value, accumulate=True)

    return grid
