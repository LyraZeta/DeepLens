# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF 相关函数。

基于 PSF 的图像仿真在物理上是散射操作：每个输入像素根据其 PSF 将能量分配到
传感器。当 PSF 空间不变时，该散射操作在数学上等价于卷积，因此考虑卷积核方向
后即可用卷积高效实现。

渲染函数：
    空间不变 PSF。
        - conv_psf()：使用一个固定 PSF 核渲染整幅图像。
        - conv_psf_depth_interp()：使用随深度变化但空间不变的 PSF 渲染，
          该 PSF 由参考深度卷积核插值得到。

    空间变化 PSF 图。
        - conv_psf_map()：将图像划分为网格分块，并用对应网格单元的 PSF
          渲染各分块。
        - conv_psf_map_depth_interp()：将图像划分为网格分块，并用该网格单元
          经深度插值得到的 PSF 渲染各分块。

    逐像素 PSF。
        - splat_psf_per_pixel()：使用每个源像素自身的局部 PSF 进行散射。
          该方法支持完整空间变化和离焦，但比基于卷积的近似方法占用更多内存。

    分层深度渲染。
        - conv_psf_occlusion()：采用遮挡感知的从后向前合成进行分层深度渲染。

其他函数：
    - interp_psf_map()：将 PSF 图插值到不同网格尺寸。
    - rotate_psf()：旋转 PSF 核。
"""

import torch
import torch.nn.functional as F

# ================================================
# 用于图像仿真的 PSF 渲染
# ================================================

def conv_psf(img, psf, method="conv"):
    """使用一个空间不变 PSF 渲染图像批次。

    使用反射填充执行逐通道、保持尺寸（"same"）的二维卷积，使输出保持输入的
    空间尺寸。``"conv"`` 和 ``"fft"`` 后端应用相同的反射填充卷积，结果仅有
    FFT 舍入误差，区别只在计算成本。

    参数：
        img (torch.Tensor): 输入图像批次，形状为 ``[B, C, H, W]``。
        psf (torch.Tensor): PSF 核，形状为 ``[C, ks, ks]``；``ks`` 可为奇数
            或偶数。
        method (str, optional): 卷积后端。``"conv"`` 使用直接 ``F.conv2d``，
            每像素成本为 ``~O(ks^2)``；``"fft"`` 使用 FFT 线性卷积，成本近似
            与 ``ks`` 无关。对于衍射镜头的大尺寸色差 PSF（``ks ~= 512-768``），
            直接卷积不可行，应优先使用 ``"fft"``。默认为 ``"conv"``。

    返回：
        img_render (torch.Tensor): 渲染图像，形状为 ``[B, C, H, W]``。

    异常：
        ValueError: 当 ``method`` 不是 ``"conv"`` 或 ``"fft"`` 时抛出。

    示例：
        ```python
        psf = lens.psf_rgb(points=torch.tensor([0.0, 0.0, -10000.0]))
        img_blur = conv_psf(img, psf)
        ```
    """
    B, C, H, W = img.shape
    C_psf, ks, _ = psf.shape
    assert C_psf == C, f"psf channels ({C_psf}) must match image channels ({C})."

    # 总量为 ks - 1 的保持尺寸（"same"）填充。拆分后奇数和偶数卷积核都能使
    # 输出形状与输入相同；对称的 ks // 2 仅能为奇数 ks 保持尺寸，偶数 ks 会
    # 输出 N + 1。
    pad_top  = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left  = (ks - 1) // 2
    pad_right = ks // 2
    img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    if method == "conv":
        # 翻转 PSF，因为 F.conv2d 计算互相关而非卷积。
        psf_k = torch.flip(psf, [-2, -1]).unsqueeze(1)  # 形状为 [C, 1, ks, ks]
        return F.conv2d(img_pad, psf_k, groups=C)

    if method == "fft":
        Hp, Wp = img_pad.shape[-2:]
        # 线性卷积而非循环卷积：将两个操作数至少零填充至 ``Hp + ks - 1``，
        # 使 FFT 乘积等于线性卷积；随后保留偏移 ks - 1 处长度为
        # Hp - ks + 1 == H 的 "valid" 窗口。
        fh, fw = Hp + ks - 1, Wp + ks - 1
        fimg = torch.fft.rfft2(img_pad, s=(fh, fw))
        fpsf = torch.fft.rfft2(psf, s=(fh, fw)).unsqueeze(0)  # [1, C, fh, fw // 2 + 1]
        conv_full = torch.fft.irfft2(fimg * fpsf, s=(fh, fw))  # [B, C, fh, fw]
        return conv_full[..., ks - 1 : ks - 1 + H, ks - 1 : ks - 1 + W]

    raise ValueError(f"Unknown conv_psf method: {method!r} (expected 'conv' or 'fft').")

def conv_psf_map(img, psf_map):
    """使用空间变化 PSF 图渲染图像批次。

    将图像划分为 ``grid_h × grid_w`` 个不重叠分块，并用相应 PSF 核卷积各分块。
    提取分块前先对整幅图像填充，以避免各分块独立填充造成的人为接缝。

    参数：
        img (torch.Tensor): 输入图像批次，形状为 ``[B, C, H, W]``。
        psf_map (torch.Tensor): PSF 图，形状为 ``[grid_h, grid_w, C, ks, ks]``。

    返回：
        img_render (torch.Tensor): 渲染图像，形状为 ``[B, C, H, W]``。
    """
    B, C, H, W = img.shape
    grid_h, grid_w, C_psf, ks, _ = psf_map.shape
    assert C_psf == C, f"PSF map channels ({C_psf}) must match image channels ({C})."
    
    # 填充
    pad_top  = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left  = (ks - 1) // 2
    pad_right = ks // 2
    img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    # 预先一次性翻转整张 PSF 图，避免在循环内逐个翻转 PSF
    psf_map_flipped = torch.flip(psf_map, dims=(-2, -1))

    # 逐分块渲染图像
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w

            # PSF, [C, 1, ks, ks]
            psf = psf_map_flipped[i, j].unsqueeze(1)

            # 考虑重叠区域，以避免边界伪影
            img_pad_patch = img_pad[
                :,
                :,
                h_low : h_high + pad_top + pad_bottom,
                w_low : w_high + pad_left + pad_right,
            ]

            # 卷积，形状为 [B, C, h_high-h_low, w_high-w_low]
            render_patch = F.conv2d(img_pad_patch, psf, groups=C)  
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render

def splat_psf_per_pixel(img, psf, chunk_size=None):
    """使用每个像素自身的 PSF 进行散射，从而渲染图像批次。

    为每个源像素使用不同的 PSF 核，并通过 ``F.fold`` 累积散射贡献。设置
    ``chunk_size`` 时，按图块处理源像素，以降低峰值内存，同时保留跨越图块
    边界的 PSF 贡献。

    参数：
        img (torch.Tensor): 待模糊图像批次，形状为 ``[B, C, H, W]``。
        psf (torch.Tensor): 逐像素局部 PSF，形状为 ``[H, W, C, ks, ks]``；
            ``ks`` 可为奇数或偶数。
        chunk_size (int or None, optional): 节省内存渲染时的源图块尺寸。
            为 ``None`` 时一次渲染整幅图像，默认为 None。

    返回：
        img_render (torch.Tensor): 渲染图像，形状为 ``[B, C, H, W]``。
    """
    B, C, H, W = img.shape
    H_psf, W_psf, C_psf, ks, _ = psf.shape
    assert C == C_psf, ("Image and PSF channels mismatch.")
    assert H == H_psf and W == W_psf, ("Image and PSF size mismatch.")

    pad_top = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left = (ks - 1) // 2
    pad_right = ks // 2

    if chunk_size is None:
        img_expand = img.unsqueeze(-1).unsqueeze(-1)  # [B, C, H, W, 1, 1]
        kernels = psf.permute(2, 0, 1, 3, 4).unsqueeze(0)  # [1, C, H, W, ks, ks]
        img_render = img_expand * kernels  # [B, C, H, W, ks, ks]
        img_render = img_render.permute(0, 1, 4, 5, 2, 3).reshape(
            B, C * ks * ks, H * W
        )
        img_render = F.fold(
            img_render, (H + ks - 1, W + ks - 1), (ks, ks), padding=0
        )
    else:
        assert chunk_size > 0, "chunk_size must be positive."

        img_render = img.new_zeros(
            B,
            C,
            H + pad_top + pad_bottom,
            W + pad_left + pad_right,
        )

        for y0 in range(0, H, chunk_size):
            y1 = min(y0 + chunk_size, H)
            for x0 in range(0, W, chunk_size):
                x1 = min(x0 + chunk_size, W)
                img_patch = img[:, :, y0:y1, x0:x1]
                psf_patch = psf[y0:y1, x0:x1, :, :, :]

                patch_h, patch_w = y1 - y0, x1 - x0
                img_patch = img_patch.unsqueeze(-1).unsqueeze(-1)
                kernels = psf_patch.permute(2, 0, 1, 3, 4).unsqueeze(0)
                render_patch = img_patch * kernels
                render_patch = render_patch.permute(0, 1, 4, 5, 2, 3).reshape(
                    B, C * ks * ks, patch_h * patch_w
                )
                img_render[:, :, y0 : y1 + ks - 1, x0 : x1 + ks - 1] += F.fold(
                    render_patch,
                    (patch_h + ks - 1, patch_w + ks - 1),
                    (ks, ks),
                    padding=0,
                )

    return img_render[
        :,
        :,
        pad_top : pad_top + H,
        pad_left : pad_left + W,
    ]


# ====================================================
# 用于图像仿真的深度变化 PSF 卷积
# ====================================================

def conv_psf_depth_interp(
    img, depth, psf_kernels, psf_depths, interp_mode="depth", padding_mode="reflect"
):
    """对空间均匀但随深度变化的模糊执行深度插值 PSF 卷积。

    先使用各参考深度的 PSF 卷积图像，再利用由 `depth` 得到的逐像素线性插值
    权重混合结果。这样无需为每个像素单独计算 PSF，即可近似单一视场位置在一段
    深度范围内的离焦模糊。

    参数：
        img (torch.Tensor): 图像批次，形状为 ``[B, C, H, W]``，取值在
            ``[0, 1]`` 内。
        depth (torch.Tensor): 深度图，形状为 ``[B, 1, H, W]``，采用负深度约定，
            取值为 ``(-∞, 0)`` mm。
        psf_kernels (torch.Tensor): 参考深度处的 PSF 堆叠，形状为
            ``[num_depth, C, ks, ks]``。
        psf_depths (torch.Tensor): 各 PSF 层深度，形状为 ``[num_depth]``，
            取值为 ``(-∞, 0)`` mm，且必须单调。
        interp_mode (str, optional): 插值空间。``"depth"`` 在线性深度空间插值；
            ``"disparity"`` 在线性 1/depth 空间插值。默认为 ``"depth"``。
        padding_mode (str or None, optional): 卷积前传给 `F.pad` 的填充模式。
            为 ``None`` 时假定 ``img`` 已填充，不再添加填充。默认为 "reflect"。

    返回：
        img_render (torch.Tensor): 模糊图像，形状为 ``[B, C, H, W]``。

    异常：
        AssertionError: 当 `depth` 或 `psf_depths` 含非负值，或 `interp_mode`
            不是 ``"depth"`` / ``"disparity"`` 时抛出。
    """
    assert interp_mode in ["depth", "disparity"], f"interp_mode must be 'depth' or 'disparity', got {interp_mode}"
    assert depth.min() < 0 and depth.max() < 0, f"depth must be negative, got {depth.min()} and {depth.max()}"
    assert psf_depths.min() < 0 and psf_depths.max() < 0, f"psf_depths must be negative, got {psf_depths.min()} and {psf_depths.max()}"
    
    num_depths, C_psf, ks, _ = psf_kernels.shape
    psf_depths = psf_depths.to(device=depth.device, dtype=depth.dtype)

    # =================================
    # 对所有深度执行 PSF 卷积
    # =================================
    B, C, _, _ = img.shape
    assert C_psf == C, f"PSF channels ({C_psf}) must match image channels ({C})."
    assert psf_depths.numel() == num_depths, (
        f"psf_depths length ({psf_depths.numel()}) must match PSF depth count ({num_depths})."
    )
    
    # 准备 PSF 核：[num_depths, C, ks, ks] -> [num_depths*C, 1, ks, ks]
    # 翻转 PSF，因为 F.conv2d 使用互相关
    psf_stacked = torch.flip(psf_kernels, [-2, -1]).reshape(num_depths * C, 1, ks, ks)

    if padding_mode is None:
        img_padded_small = img
    else:
        # 扩展前先填充：先填充 [B, C, H, W]（C 个通道），再扩展到 num_depths*C
        # 这样可将填充工作量降低 num_depths 倍
        pad_top  = (ks - 1) // 2
        pad_bottom = ks // 2
        pad_left  = (ks - 1) // 2
        pad_right = ks // 2
        img_padded_small = F.pad(
            img, (pad_left, pad_right, pad_top, pad_bottom), mode=padding_mode
        )

    # 扩展填充后的图像：[B, C, Hpad, Wpad] -> [B, num_depths*C, Hpad, Wpad]
    img_padded = img_padded_small.repeat(1, num_depths, 1, 1)
    
    # 分组卷积：num_depths*C 个通道分别与自身卷积核进行卷积
    imgs_blur = F.conv2d(img_padded, psf_stacked, groups=num_depths * C)  # [B, num_depths*C, Hout, Wout]
    H, W = imgs_blur.shape[-2:]
    
    # 重塑为 [num_depths, B, C, H, W]
    imgs_blur = imgs_blur.reshape(B, num_depths, C, H, W).permute(1, 0, 2, 3, 4)

    # =================================
    # 深度/视差插值
    # =================================
    B_depth, _, H_depth, W_depth = depth.shape
    assert B_depth == B, f"Depth batch size ({B_depth}) must match image batch size ({B})."
    assert H_depth == H and W_depth == W, (
        f"Depth shape ({H_depth}, {W_depth}) must match rendered shape ({H}, {W})."
    )
    depth_flat = depth.flatten(1)  # 形状为 [B, H*W]
    depth_flat = depth_flat.clamp(psf_depths[0], psf_depths[-1])
    indices = torch.searchsorted(psf_depths, depth_flat, right=True)  # 形状为 [B, H*W]
    indices = indices.clamp(1, num_depths - 1)
    idx0 = indices - 1
    idx1 = indices

    # 计算深度插值权重
    d0 = psf_depths[idx0]  # 形状为 [B, H*W]
    d1 = psf_depths[idx1]
    
    if interp_mode == "depth":
        # 在深度空间中插值
        denom = d1 - d0
        denom[denom == 0] = 1e-6  # 避免除以零
        w1 = (depth_flat - d0) / denom  # 形状为 [B, H*W]
    else:
        # 在视差空间中插值（disparity = 1/depth）
        disp_flat = 1.0 / depth_flat
        disp0 = 1.0 / d0
        disp1 = 1.0 / d1
        denom = disp1 - disp0
        denom[denom == 0] = 1e-6  # 避免除以零
        w1 = (disp_flat - disp0) / denom  # 形状为 [B, H*W]
    
    w0 = 1 - w1

    # 创建权重张量
    weights = torch.zeros(num_depths, B, H * W, device=img.device, dtype=img.dtype)
    weights.scatter_add_(0, idx0.unsqueeze(0).long(), w0.unsqueeze(0))
    weights.scatter_add_(0, idx1.unsqueeze(0).long(), w1.unsqueeze(0))
    weights = weights.view(num_depths, B, 1, H, W)

    # 将权重应用于模糊图像
    img_render = torch.sum(imgs_blur * weights, dim=0)
    return img_render


def conv_psf_map_depth_interp(img, depth, psf_map, psf_depths, interp_mode="depth"):
    """使用空间变化且经深度插值的 PSF 图进行渲染。

    将图像划分为 PSF 图网格单元。对每个单元，使用该单元所有参考深度 PSF
    卷积图像分块，再根据深度图得到的插值权重逐像素混合卷积结果。

    参数：
        img (torch.Tensor): 图像批次，形状为 ``[B, C, H, W]``，取值在
            ``[0, 1]`` 内。
        depth (torch.Tensor): 深度图，形状为 ``[B, 1, H, W]``，采用负深度约定，
            取值在 ``(-inf, 0)`` 内。
        psf_map (torch.Tensor): PSF 图，形状为
            ``[grid_h, grid_w, num_depth, C, ks, ks]``。
        psf_depths (torch.Tensor): 参考深度，形状为 ``[num_depth]``，取值在
            ``(-inf, 0)`` 内，用于插值 ``psf_map``。
        interp_mode (str, optional): ``"depth"`` 表示线性深度插值，
            ``"disparity"`` 表示在 $1/\text{depth}$ 中线性插值，默认为 "depth"。

    返回：
        img_render (torch.Tensor): 渲染图像，形状为 ``[B, C, H, W]``。
    """
    _, _, H, W = img.shape
    grid_h, grid_w, _, _, ks, _ = psf_map.shape

    # 对整幅图像一次性填充，以避免分块接缝处的边界伪影。否则每个分块会独立
    # 填充（在自身边界内反射），并在网格边界产生可见接缝。
    pad_top  = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left  = (ks - 1) // 2
    pad_right = ks // 2
    img_pad = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    # 逐分块渲染图像
    img_render = torch.zeros_like(img)
    for i in range(grid_h):
        h_low  = (i * H) // grid_h
        h_high = ((i + 1) * H) // grid_h

        for j in range(grid_w):
            w_low  = (j * W) // grid_w
            w_high = ((j + 1) * W) // grid_w

            # 从预填充图像中提取重叠分块，无需逐分块填充
            img_pad_patch = img_pad[
                :, :,
                h_low : h_high + pad_top + pad_bottom,
                w_low : w_high + pad_left + pad_right,
            ]
            depth_patch = depth[:, :, h_low:h_high, w_low:w_high]
            render_patch = conv_psf_depth_interp(
                img_pad_patch,
                depth_patch,
                psf_map[i, j],
                psf_depths,
                interp_mode=interp_mode,
                padding_mode=None,
            )
            img_render[:, :, h_low:h_high, w_low:w_high] = render_patch

    return img_render

def conv_psf_occlusion(img, depth, psf_kernels, psf_depths):
    """使用从后向前分层合成的遮挡感知散景渲染。

    将场景离散为多个深度层，并从后方（远处）向前方（近处）合成。每一层使用
    其特定深度 PSF 独立模糊，再通过 over-operator 合成，从而避免深度不连续处
    的颜色渗漏。

    参考文献：
        [1] "Dr.Bokeh: DiffeRentiable Occlusion-aware Bokeh Rendering", CVPR 2024.

    参数：
        img (torch.Tensor): 输入图像，形状为 ``[B, C, H, W]``，取值在
            ``[0, 1]`` 内。
        depth (torch.Tensor): 深度图，形状为 ``[B, 1, H, W]``，采用负深度约定，
            取值为 ``(-inf, 0)`` mm。
        psf_kernels (torch.Tensor): 各深度层的 PSF，形状为
            ``[num_layers, C, ks, ks]``。
        psf_depths (torch.Tensor): 各层深度值，形状为 ``[num_layers]`` mm。
            必须为负数并按从小到大排序，即从远到近，例如 -5000 ... -200。

    返回：
        img_render (torch.Tensor): 渲染图像，形状为 ``[B, C, H, W]``。
    """
    assert depth.min() < 0 and depth.max() < 0, (
        f"depth must be negative, got min={depth.min()} max={depth.max()}"
    )
    assert psf_depths.min() < 0 and psf_depths.max() < 0, (
        f"psf_depths must be negative, got min={psf_depths.min()} max={psf_depths.max()}"
    )

    num_layers, C, ks, _ = psf_kernels.shape
    B, C_img, H, W = img.shape
    assert C == C_img, f"PSF channels ({C}) must match image channels ({C_img})"

    device = img.device
    dtype = img.dtype
    psf_depths = psf_depths.to(device=device, dtype=dtype)

    # 将每个像素分配给最近的深度层
    # depth 和 psf_depths 均为负数；depth_map 形状为 [B, 1, H, W]
    depth_expanded = depth.view(B, 1, H, W).expand(B, num_layers, H, W)
    psf_depths_view = psf_depths.view(1, num_layers, 1, 1)
    dist = torch.abs(depth_expanded - psf_depths_view)  # [B, num_layers, H, W]
    layer_assignment = dist.argmin(dim=1, keepdim=True)  # [B, 1, H, W]

    # 预先计算卷积所需的翻转 PSF 和填充量
    psf_flipped = torch.flip(psf_kernels, [-2, -1])  # [num_layers, C, ks, ks]
    pad_top = (ks - 1) // 2
    pad_bottom = ks // 2
    pad_left = (ks - 1) // 2
    pad_right = ks // 2

    # 从后向前合成：第 0 层最远，第 num_layers-1 层最近
    result = torch.zeros(B, C, H, W, device=device, dtype=dtype)

    for i in range(num_layers):
        # 创建该层的软掩码：属于该层的像素取 1
        mask = (layer_assignment == i).to(dtype=dtype)  # [B, 1, H, W]

        # 无条件执行：全零掩码卷积后仍为零，合成操作等价于不做处理
        # （result = 0 + result * (1 - 0)）。使用 `if mask.sum() == 0` 跳过会在
        # 每次迭代中强制发生 GPU->CPU 同步。

        # 该层 RGB：属于该层的像素保留，其余位置为零
        layer_rgb = img * mask  # [B, C, H, W]

        # 使用该层 PSF 卷积该层 RGB
        psf_i = psf_flipped[i].unsqueeze(1)  # [C, 1, ks, ks]
        layer_rgb_pad = F.pad(layer_rgb, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        blurred_rgb = F.conv2d(layer_rgb_pad, psf_i, groups=C)  # [B, C, H, W]

        # 使用同一 PSF 卷积掩码；由于 PSF 各通道总和均为 1，只使用一个通道
        # 对掩码模糊时跨通道求平均；近轴情况下各通道 PSF 相同
        psf_i_mono = psf_flipped[i, 0:1].unsqueeze(1)  # [1, 1, ks, ks]
        mask_pad = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0)
        blurred_mask = F.conv2d(mask_pad, psf_i_mono, groups=1)  # [B, 1, H, W]
        blurred_mask = blurred_mask.clamp(0, 1)

        # Over 合成（从后向前）：
        # result = blurred_rgb + result * (1 - blurred_mask)
        result = blurred_rgb + result * (1 - blurred_mask)

    return result




def interp_psf_map(psf_map, grid_old, grid_new):
    """将 PSF 图重采样到不同空间网格尺寸。

    支持打包布局 ``[C, grid_old*ks, grid_old*ks]`` 或未打包布局
    ``[grid_old, grid_old, C, ks, ks]``。在 PSF 网格上对每个卷积核采样位置
    进行双线性插值，并以打包布局返回结果。

    参数：
        psf_map (torch.Tensor): 打包或未打包的 PSF 图。
        grid_old (int): 输入网格尺寸。对于未打包输入会忽略该参数，网格尺寸
            直接从 ``psf_map`` 读取。
        grid_new (int): 输出网格尺寸。

    返回：
        psf_map_interp (torch.Tensor): 插值后的打包 PSF 图，形状为
            ``[C, grid_new*ks, grid_new*ks]``。
    """
    if len(psf_map.shape) == 3:
        # [C, grid_old*ks, grid_old*ks]
        C, H, W = psf_map.shape
        assert H % grid_old == 0 and W % grid_old == 0, (
            "PSF map size should be divisible by grid"
        )
        ks = int(H / grid_old)
        assert ks % 2 == 1, "PSF kernel size should be odd"

        # 从 [C, grid*ks, grid*ks] 重塑为 [grid_old, grid_old, C, ks, ks]
        psf_map_interp = psf_map.reshape(C, grid_old, ks, grid_old, ks).permute(
            1, 3, 0, 2, 4
        )  # .reshape(grid_old, grid_old, C, ks, ks)
    elif len(psf_map.shape) == 5:
        # [grid_old, grid_old, C, ks, ks]
        grid_h, grid_w, C, ks_h, ks_w = psf_map.shape
        assert grid_h == grid_w, f"PSF map grid must be square, got {grid_h}x{grid_w}"
        assert ks_h == ks_w, f"PSF kernel must be square, got {ks_h}x{ks_w}"
        grid_old = grid_h
        ks = ks_h
        psf_map_interp = psf_map
    else:
        raise ValueError(
            "PSF map should be [C, grid_old*ks, grid_old*ks] or [grid_old, grid_old, C, ks, ks]"
        )

    # 从 [grid_old, grid_old, C, ks, ks] 重塑为 [ks*ks, C, grid_old, grid_old]
    psf_map_interp = psf_map_interp.permute(3, 4, 2, 0, 1).reshape(
        ks * ks, C, grid_old, grid_old
    )

    # 从 [ks*ks, C, grid_old, grid_old] 插值为 [ks*ks, C, grid_new, grid_new]
    psf_map_interp = F.interpolate(
        psf_map_interp, size=(grid_new, grid_new), mode="bilinear", align_corners=True
    )

    # 从 [ks*ks, C, grid_new, grid_new] 重塑为 [C, grid_new*ks, grid_new*ks]
    psf_map_interp = (
        psf_map_interp.reshape(ks, ks, C, grid_new, grid_new)
        .permute(2, 3, 0, 4, 1)
        .reshape(C, grid_new * ks, grid_new * ks)
    )

    return psf_map_interp


def rotate_psf(psf, theta):
    """逆时针旋转一批 RGB PSF 核。

    使用 ``F.grid_sample`` 围绕每个方形 PSF 核中心进行旋转。

    参数：
        psf (torch.Tensor): PSF 批次，形状为 ``[N, 3, ks, ks]``。
        theta (torch.Tensor): 旋转角，单位为弧度，形状为 ``[N]``。

    返回：
        rotated_psf (torch.Tensor): 旋转后的 PSF，形状为 ``[N, 3, ks, ks]``。
    """
    assert len(psf.shape) == 4, "PSF should be [N, 3, ks, ks]"

    N, _, ks, _ = psf.shape
    assert ks == psf.shape[3], "PSF kernel should be square"

    # 若要逆时针旋转图像，采样网格必须顺时针旋转。
    # 顺时针旋转 theta 的矩阵为：
    # [ cos(theta)  sin(theta) ]
    # [ -sin(theta) cos(theta) ]
    rotation_matrices = torch.zeros(N, 2, 3, device=psf.device, dtype=psf.dtype)
    rotation_matrices[:, 0, 0] = torch.cos(theta)
    rotation_matrices[:, 0, 1] = torch.sin(theta)
    rotation_matrices[:, 1, 0] = -torch.sin(theta)
    rotation_matrices[:, 1, 1] = torch.cos(theta)

    # 旋转 PSF
    grid = F.affine_grid(rotation_matrices, psf.shape, align_corners=True)
    rotated_psf = F.grid_sample(psf, grid, align_corners=True)

    return rotated_psf
