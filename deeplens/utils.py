import logging
import os
import random
from glob import glob

import cv2 as cv
import lpips
import numpy as np
import torch
import torch.nn.functional as F


# ==================================
# 图像输入输出
# ==================================
def img2batch(img):
    """将任意受支持类型的图像转换为归一化张量批次。

    接受布局为 (H, W)、(H, W, C) 或 (C, H, W) 的 numpy 数组或 torch 张量，
    并返回 [0, 1] 范围内的 float32 批次。uint8 输入会缩放 1/255；float32
    输入保持不变。

    参数：
        img (numpy.ndarray or torch.Tensor): 输入图像，shape 为 (H, W)、
            (H, W, C) 或 (C, H, W)，具有 1 或 3 个通道。

    返回：
        img (torch.Tensor): 批处理后的 float32 图像，shape 为 (1, C, H, W)。

    异常：
        ValueError: 当通道数或 dtype 不受支持，或二维输入不是 numpy 数组时抛出。
    """
    # 张量 shape
    if len(img.shape) == 2:
        if isinstance(img, np.ndarray):
            img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        else:
            raise ValueError("Image should be numpy array.")

    elif len(img.shape) == 3:
        if isinstance(img, np.ndarray):
            assert img.shape[-1] in [1, 3], "Image channel should be 1 or 3."
            img = (
                torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
            )  # (H, W, C) -> (1, C, H, W)
        elif torch.is_tensor(img):
            if img.shape[0] in [1, 3]:
                # 假定为 (C, H, W) -> (1, C, H, W)
                img = img.unsqueeze(0)
            elif img.shape[-1] in [1, 3]:
                # 假定为 (H, W, C) -> (1, C, H, W)
                img = img.permute(2, 0, 1).unsqueeze(0)
            else:
                 raise ValueError("Image channel should be 1 or 3.")
        else:
            raise ValueError("Image should be numpy array or torch tensor.")

    # 张量 dtype
    if img.dtype == torch.uint8:
        img = img.to(torch.float32) / 255.0
    elif img.dtype == torch.float32:
        pass
    else:
        raise ValueError("Image type should be uint8 or float32.")

    return img


# ==================================
# 图像批次质量评估
# ==================================
def batch_PSNR(img_clean, img):
    """使用 skimage 计算图像批次的平均 PSNR。

    参数：
        img_clean (torch.Tensor): [0, 1] 范围内的参考图像，shape 为 (B, C, H, W)。
        img (torch.Tensor): [0, 1] 范围内的测试图像，shape 为 (B, C, H, W)。

    返回：
        psnr (float): 整个批次的平均 PSNR [dB]，四舍五入到 4 位小数。
    """
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )
    from skimage.metrics import peak_signal_noise_ratio
    PSNR = 0.0
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Img_clean[i, :, :, :], Img[i, :, :, :])
    return round(PSNR / Img.shape[0], 4)


def batch_psnr(pred, target, max_val=1.0, eps=1e-8):
    """计算两个图像批次中每幅图像的 PSNR（可微）。

    参数：
        pred (torch.Tensor): 预测图像，shape 为 (B, C, H, W)。
        target (torch.Tensor): 目标图像，shape 为 (B, C, H, W)。
        max_val (float, optional): 最大像素值（归一化图像为 1.0，uint8 为 255）。
            默认为 1.0。
        eps (float, optional): 加到 MSE 上以避免 log(0) 的小常数。默认为 1e-8。

    返回：
        psnr (torch.Tensor): 每幅图像的 PSNR [dB]，shape 为 (B,)。

    参考：
        https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    """
    assert pred.shape == target.shape, f"Shape mismatch: {pred.shape} vs {target.shape}"

    # 沿空间维度和通道维度计算 MSE
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])  # shape: [B]

    # 计算 PSNR
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse + eps))

    return psnr


def batch_SSIM(img, img_clean):
    """计算图像批次的平均 SSIM（`batch_ssim` 的别名）。

    参数：
        img (torch.Tensor): [0, 1] 范围内的测试图像，shape 为 (B, C, H, W)。
        img_clean (torch.Tensor): [0, 1] 范围内的参考图像，shape 为 (B, C, H, W)。

    返回：
        ssim (float): 整个批次的平均 SSIM，四舍五入到 4 位小数。
    """
    return batch_ssim(img, img_clean)


def batch_ssim(img, img_clean):
    """使用 skimage 计算图像批次的平均 SSIM。

    评分前将图像转换为 [0, 255] 范围内的 uint8。多通道图像使用
    `channel_axis=0` 评分；单通道图像按二维平面评分。

    参数：
        img (torch.Tensor): [0, 1] 范围内的测试图像，shape 为 (B, C, H, W)。
        img_clean (torch.Tensor): [0, 1] 范围内的参考图像，shape 为 (B, C, H, W)。

    返回：
        ssim (float): 整个批次的平均 SSIM，四舍五入到 4 位小数。
    """
    # 转换为 [0, 255] 范围内的 numpy 数组
    Img = img.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Img_clean = (
        img_clean.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    )

    from skimage.metrics import structural_similarity
    SSIM = 0.0
    for i in range(Img.shape[0]):
        # 根据维数自动检测是否为多通道图像
        if Img.shape[1] > 1:  # 多通道
            SSIM += structural_similarity(
                Img_clean[i, ...], Img[i, ...], channel_axis=0
            )
        else:  # 单通道
            SSIM += structural_similarity(Img_clean[i, 0, ...], Img[i, 0, ...])

    return round(SSIM / Img.shape[0], 4)


def batch_LPIPS(img, img_clean):
    """计算图像批次的平均 LPIPS 感知距离。

    使用带空间图的 VGG 主干网络；返回值为整个批次空间距离图的平均值。

    参数：
        img (torch.Tensor): 测试图像，shape 为 (B, C, H, W)。
        img_clean (torch.Tensor): 参考图像，shape 为 (B, C, H, W)。

    返回：
        lpips (float): 整个批次的平均 LPIPS 距离（越低越好）。
    """
    device = img.device
    loss_fn = lpips.LPIPS(net="vgg", spatial=True)
    loss_fn.to(device)
    dist = loss_fn.forward(img, img_clean)
    return dist.mean().item()


# ==================================
# 图像批次归一化
# ==================================
def normalize_ImageNet(batch):
    """使用 ImageNet 均值和标准差对 RGB 图像批次进行归一化。

    参数：
        batch (torch.Tensor): [0, 1] 范围内的 RGB 图像，shape 为 (B, 3, H, W)。

    返回：
        batch_out (torch.Tensor): 归一化图像，shape 为 (B, 3, H, W)。
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std
    return batch_out


def denormalize_ImageNet(batch):
    """反转 ImageNet 归一化，恢复 [0, 1] 范围内的图像。

    参数：
        batch (torch.Tensor): 经 ImageNet 归一化的 RGB 图像，shape 为 (B, 3, H, W)。

    返回：
        batch_out (torch.Tensor): [0, 1] 范围内的反归一化图像，
            shape 为 (B, 3, H, W)。
    """
    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = batch * std + mean
    return batch_out


# ==================================
# 扩展景深（EDoF）
# ==================================
def foc_dist_balanced(d1, d2):
    """计算使弥散圈（CoC）达到平衡的对焦距离。

    返回调和平均值 $2 d_1 d_2 / (d_1 + d_2)$，即距离为 `d1` 和 `d2` 的
    两个物平面产生相同 CoC 时的对焦距离。

    参数：
        d1 (float or torch.Tensor): 到第一个物平面的距离 [mm]。
        d2 (float or torch.Tensor): 到第二个物平面的距离 [mm]。

    返回：
        foc_dist (float or torch.Tensor): 平衡后的对焦距离 [mm]。

    参考：
        https://en.wikipedia.org/wiki/Circle_of_confusion
    """
    foc_dist = 2 * d1 * d2 / (d1 + d2)
    return foc_dist


# ==================================
# 自动镜头设计（AutoLens）
# ==================================
def create_video_from_images(image_folder, output_video_path, fps=30):
    """根据文件夹中的图像创建视频。

    参数：
        image_folder (str): 包含图像的文件夹路径；递归搜索 "*.png"，
            并按创建时间排序。
        output_video_path (str): 输出 .mp4 视频（mp4v 编解码器）的保存路径。
        fps (int, optional): 输出视频的每秒帧数。默认为 30。
    """
    # 获取 image_folder 及其子文件夹中的所有 .png 文件
    images = glob(os.path.join(image_folder, "**/*.png"), recursive=True)
    # images.sort()  # 按名称排序图像
    images.sort(key=lambda x: os.path.getctime(x))  # 按创建时间排序图像

    if not images:
        print("No PNG images found in the provided directory.")
        return

    # 读取第一幅图像以获取尺寸
    first_image = cv.imread(images[0])
    height, width, layers = first_image.shape

    # 定义编解码器并创建 VideoWriter 对象
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video_writer = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历图像并写入视频
    from tqdm import tqdm
    for image_path in tqdm(images):
        img = cv.imread(image_path)
        video_writer.write(img)

    # 释放视频写入器对象
    video_writer.release()
    print(f"Video saved as {output_video_path}")


# ==================================
# 实验日志记录
# ==================================
def gpu_init(gpu=0):
    """选择计算设备，并将默认浮点 dtype 设置为 float32。

    参数：
        gpu (int, optional): CUDA 可用时所使用的设备索引。默认为 0。

    返回：
        device (torch.device): 所选设备（GPU 可用时为 `cuda:{gpu}`，否则为 `cpu`）。
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("Using: {}".format(device))
    torch.set_default_dtype(torch.float32)
    return device


def set_seed(seed=0):
    """为 Python、NumPy 和 PyTorch 随机数生成器设置种子，以实现可复现运行。

    同时禁用 cuDNN 基准测试和非确定性行为（设置 `deterministic=True`、
    `benchmark=False`、`enabled=False`）。

    参数：
        seed (int, optional): 随机种子。默认为 0。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 用于多 GPU。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def set_logger(dir="./"):
    """配置根日志记录器，使其向控制台输出并写入文件。

    添加 stdout `StreamHandler` 和写入 `{dir}/output.log` 的 `FileHandler`；
    二者均使用 INFO 级别和带时间戳的格式。

    参数：
        dir (str, optional): `output.log` 文件所在目录。默认为 "./"。
    """
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel("INFO")

    fhlr = logging.FileHandler(f"{dir}/output.log", encoding="utf-8")
    fhlr.setFormatter(formatter)
    fhlr.setLevel("INFO")

    # fhlr2 = logging.FileHandler(f"{dir}/error.log", encoding="utf-8")
    # fhlr2.setFormatter(formatter)
    # fhlr2.setLevel('WARNING')

    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    # logger.addHandler(fhlr2)


# ==================================
# 可微插值
# ==================================
def interp1d(query, key, value, mode="linear"):
    """在查询点处对一维关键点上定义的值进行可微插值。

    仅实现 `mode="linear"`：首先对关键点排序，使用 `searchsorted` 定位查询点，
    然后在包围查询点的关键点之间进行线性插值。关键点范围外的查询会被限制到
    两端区间。

    参数：
        query (torch.Tensor): 查询点，shape 为 (N, 1)（展平为 (N,)）。
        key (torch.Tensor): 关键点，shape 为 (M, 1)（展平为 (M,)）。
        value (torch.Tensor): 关键点处的值，shape 为 (M, ...)。
        mode (str, optional): 插值模式；仅支持 "linear"。默认为 "linear"。

    返回：
        query_value (torch.Tensor): 插值后的值，shape 为 (N, ...)。

    异常：
        NotImplementedError: 当 `mode="grid_sample"` 时抛出。
        ValueError: 当 `mode` 不是可识别的值时抛出。

    参考：
        https://github.com/aliutkus/torchinterp1d
    """
    if mode == "linear":
        # 展平 query 和 key 张量以便处理
        query_flat = query.flatten()  # [N]
        key_flat = key.flatten()  # [M]

        # 获取 `value` 的原始形状，以保留额外维度
        value_shape = value.shape  # [M, ...]
        M = value_shape[0]
        extra_dims = value_shape[1:]
        value_reshaped = value.view(M, -1)  # [M, D]，其中 D 为额外维度的乘积

        # 对 key 和 value 排序
        sort_idx = torch.argsort(key_flat)
        key_sorted = key_flat[sort_idx]  # [M]
        value_sorted = value_reshaped[sort_idx]  # [M, D]

        # 查找插值索引
        indices = torch.searchsorted(key_sorted, query_flat, right=False)  # [N]
        indices = torch.clamp(indices, 1, len(key_sorted) - 1)  # [N]

        # 获取左右关键点
        key_left = key_sorted[indices - 1]  # [N]
        key_right = key_sorted[indices]  # [N]
        value_left = value_sorted[indices - 1]  # [N, D]
        value_right = value_sorted[indices]  # [N, D]

        # 线性插值
        result = value_left.clone()  # [N, D]
        mask = key_left != key_right  # [N]
        if mask.any():
            denom = torch.where(mask, key_right - key_left, torch.ones_like(key_left))
            weight = ((query_flat - key_left) / denom).unsqueeze(-1)  # [N, 1]

            # 仅在 mask 为 True 的位置应用插值
            interpolated = value_left + weight * (value_right - value_left)  # [N, D]
            result = torch.where(mask.unsqueeze(-1), interpolated, value_left)  # [N, D]

        # 将结果恢复为 [N, ...]，同时保留额外维度
        result_shape = (query.shape[0],) + extra_dims
        query_value = result.view(result_shape)

    elif mode == "grid_sample":
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        # 这要求关键点之间具有均匀间距。
        raise NotImplementedError("Grid sample is not implemented yet.")

    else:
        raise ValueError(f"Invalid interpolation mode: {mode}")

    return query_value


def grid_sample_xy(
    input, grid_xy, mode="bilinear", padding_mode="zeros", align_corners=False
):
    """在 xy 坐标网格上对输入特征图进行采样。

    封装 `torch.nn.functional.grid_sample`，但接受遵循 xy 约定（y 轴向上）的网格：
    左上角为 (-1, 1)，右下角为 (1, -1)。内部会对 y 分量取负，以匹配 PyTorch
    行方向向下的约定。

    参数：
        input (torch.Tensor): 输入特征图，shape 为 (B, C, H, W)。
        grid_xy (torch.Tensor): 归一化 xy 坐标中的采样网格，shape 为 (B, H, W, 2)。
        mode (str, optional): 插值模式，可为 "bilinear" 或 "nearest"。
            默认为 "bilinear"。
        padding_mode (str, optional): 网格外填充，可为 "zeros"、"border" 或
            "reflection"。默认为 "zeros"。
        align_corners (bool, optional): 是否对齐角点像素。默认为 False。

    返回：
        output (torch.Tensor): 采样后的特征图，shape 为 (B, C, H, W)，
            其中 H 和 W 是 `grid_xy` 的空间维度。
    """
    grid_x = grid_xy[..., 0]
    grid_y = grid_xy[..., 1]
    grid = torch.stack([grid_x, -grid_y], dim=-1)
    return F.grid_sample(
        input,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


# ================================
# 自动微分函数 diff_float
# ================================
class DiffFloat(torch.autograd.Function):
    """在前向传播中将张量转换为 float32，同时保留 float64 梯度。

    前向传播返回 `x.float()`；反向传播将传入梯度向上转换回双精度，
    因而上游计算图保持 float64，而下游运算使用 float32。
    """

    @staticmethod
    def forward(ctx, x):
        """将输入张量转换为 float32。

        参数：
            x (torch.Tensor): 输入张量（通常为 float64）。

        返回：
            out (torch.Tensor): 转换为 float32 的输入。
        """
        ctx.save_for_backward(x)
        return x.float()

    @staticmethod
    def backward(ctx, grad_output):
        """将梯度向上转换回 float64。

        参数：
            grad_output (torch.Tensor): 传入梯度（float32）。

        返回：
            grad_input (torch.Tensor): 转换为 float64 的梯度。
        """
        (x,) = ctx.saved_tensors
        grad_input = grad_output.double()
        return grad_input


def diff_float(input):
    """将张量转换为 float32，同时保留 float64 梯度。

    `DiffFloat` 的便捷封装。

    参数：
        input (torch.Tensor): 输入张量（通常为 float64）。

    返回：
        out (torch.Tensor): 转换为 float32 的输入，可在 float64 下求导。
    """
    return DiffFloat.apply(input)


# ================================
# 自动微分函数 diff_quantize
# ================================
class DiffQuantize(torch.autograd.Function):
    """将张量量化到等间距级别，并使用直通梯度。

    前向传播将每个值舍入到覆盖 `interval` 的 `levels` 个步长中最接近的一档
    （步长为 `interval / levels`）。反向传播保持梯度不变（直通估计器），
    因而不可微的舍入操作不会阻碍优化。
    """

    @staticmethod
    def forward(ctx, x, levels, interval=2 * torch.pi):
        """将输入舍入到最近的量化步长。

        参数：
            x (torch.Tensor): 输入张量。
            levels (int): 量化级别数。
            interval (float, optional): 各级别覆盖的总范围；步长为
                `interval / levels`。默认为 2*pi。

        返回：
            out (torch.Tensor): 量化后的张量，shape 与 `x` 相同。
        """
        step = interval / levels
        return torch.round(x / step) * step

    @staticmethod
    def backward(ctx, grad_output):
        """保持梯度不变地传递（直通估计器）。

        参数：
            grad_output (torch.Tensor): 传入梯度。

        返回：
            grad_input (torch.Tensor): 关于 `x` 的梯度（等于 `grad_output`）。
            grad_levels (None): 始终为 None——`levels` 不可微。
            grad_interval (None): 始终为 None——`interval` 不可微。
        """
        grad_input = grad_output.clone()
        return grad_input, None, None


def diff_quantize(input, levels, interval=2 * torch.pi):
    """将张量量化到等间距级别，并使用直通梯度。

    `DiffQuantize` 的便捷封装。

    参数：
        input (torch.Tensor): 输入张量。
        levels (int): 量化级别数。
        interval (float, optional): 各级别覆盖的总范围；步长为
            `interval / levels`。默认为 2*pi。

    返回：
        out (torch.Tensor): 量化后的张量，shape 与 `input` 相同。
    """
    return DiffQuantize.apply(input, levels, interval)
