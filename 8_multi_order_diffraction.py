"""通过折射—衍射混合镜头的 PSF 可视化多级衍射。

使用光线追迹模型（如 ZEMAX）时，一次只能追迹一个衍射级次。要获得完整结果，必须
多次运行，并为不同级次指定不同的衍射效率。而在光线—波动模型中，波前包含所有
衍射级次的信息，因此可以计算包含所有衍射级次贡献的 PSF。
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplens import HybridLens


def analyze_psf(psf, save_name="./psf"):
    """使用一维中心线剖面和二维图分析并可视化 PSF。

    参数：
        psf: shape 为 [H, W] 的 PSF 张量
        save_name: 保存输出文件的基础名称（默认值："./psf"）
    """
    # 绘制 PSF 沿 Y 方向（中心列）的值
    center_x = psf.shape[-1] // 2

    # 提取中心列剖面（沿 y 方向）
    psf_center = psf[:, center_x].detach().cpu().numpy()

    # 创建线性和对数尺度图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 以像素为单位的 y 轴
    y_pixels = range(len(psf_center))

    # 线性尺度图
    axes[0].plot(y_pixels, psf_center, color="#3498db", alpha=0.8)
    axes[0].set_xlabel("Y Pixel Position")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("PSF Center Profile (Linear)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(
        x=len(psf_center) // 2,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Center",
    )

    # 对数尺度图，以便更清楚地观察高阶衍射峰
    axes[1].semilogy(
        y_pixels,
        psf_center + 1e-10,
        color="#3498db",
        alpha=0.8,
    )
    axes[1].set_xlabel("Y Pixel Position")
    axes[1].set_ylabel("Intensity (log scale)")
    axes[1].set_title("PSF Center Profile (Log)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(
        x=len(psf_center) // 2,
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Center",
    )

    plt.tight_layout()
    plt.savefig(f"{save_name}_center_line.png", dpi=150)
    plt.close()
    print(f"Saved center column profile to {save_name}_center_line.png")

    # 绘制二维 PSF
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 转换为 numpy
    psf_2d = psf.detach().cpu().numpy()

    # 线性尺度二维 PSF
    im0 = axes[0].imshow(psf_2d, cmap="hot")
    axes[0].set_title("PSF (Linear)")
    axes[0].set_xlabel("X (pixels)")
    axes[0].set_ylabel("Y (pixels)")
    plt.colorbar(im0, ax=axes[0], label="Intensity")

    # 对数尺度二维 PSF——显示高阶衍射
    psf_log = np.log10(psf_2d + 1e-10)
    im1 = axes[1].imshow(psf_log, cmap="hot")
    axes[1].set_title("PSF (Log)")
    axes[1].set_xlabel("X (pixels)")
    axes[1].set_ylabel("Y (pixels)")
    plt.colorbar(im1, ax=axes[1], label="log10(Intensity)")

    plt.tight_layout()
    plt.savefig(f"{save_name}_2d.png", dpi=150)
    plt.close()
    print(f"Saved 2D PSF visualization to {save_name}_2d.png")


def main():
    # 加载折射—衍射混合镜头
    # 光栅（DOE）默认针对 0.55um 设计，因此 0.55um PSF 的一阶衍射效率最高。
    lens = HybridLens(
        filename="./datasets/lenses/hybridlens/a489_grating.json", dtype=torch.float64
    )

    # 计算多个波长在指定点处的 PSF
    ks = 1024
    point = [0.0, 0.0, -10000.0]
    wvln_ls = [0.48, 0.55, 0.65]
    for wvln in wvln_ls:
        print(f"Calculating PSF at point {point} for wavelength {wvln}...")
        psf = lens.psf(points=point, ks=ks, wvln=wvln)

        analyze_psf(psf, save_name=f"./psf_{wvln}")


if __name__ == "__main__":
    main()
