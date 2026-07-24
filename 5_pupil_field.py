"""通过相干光线追迹计算空间中点物体对应的镜头光瞳光场（波前）。

注意：波前误差是实际波前与理想球面波前之间的相对误差。在商业软件（如 Zemax）
中，波前误差通过插值计算，因此要求波前像差为低频。而 DeepLens 不依赖插值，
对高频波前也能准确计算。

技术论文：
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

from deeplens import GeoLens


def calculate_wavefield(lens):
    """通过相干光线追迹计算出瞳光场（波前）。"""
    point = torch.tensor([0.0, 0.0, -10000.0])
    wavefront, _ = lens.pupil_field(points=point, spp=20_000_000)
    save_image(wavefront.angle(), "./wavefront_phase.png")
    save_image(torch.abs(wavefront), "./wavefront_amp.png")


def compare_psf(lens):
    """比较三种不同的 PSF，并绘制中心线剖面。

    比较对象：
        1. 几何 PSF（非相干）
        2. 惠更斯 PSF
        3. 出瞳传播 PSF（相干，在数学上等价于惠更斯 PSF）
    """
    point = torch.tensor([0.0, 0.4, -10000.0])
    ks = 64

    # 计算三种不同的 PSF
    psf_coherent = lens.psf_coherent(point, ks=ks)
    save_image(psf_coherent, "./psf_raywave.png", normalize=True)

    psf_incoherent = lens.psf(point, ks=ks)
    save_image(psf_incoherent, "./psf_incoherent.png", normalize=True)

    psf_huygens = lens.psf_huygens(point, ks=ks)
    save_image(psf_huygens, "./psf_huygens.png", normalize=True)

    # ==========================================================
    # 绘制 PSF 沿中心线的值
    # ==========================================================
    center_y = psf_coherent.shape[-2] // 2

    # 提取中心线剖面（若为 RGB，则对通道求平均）
    if psf_coherent.dim() == 3:  # [C, H, W]
        coherent_center = psf_coherent[:, center_y, :].mean(dim=0).cpu().numpy()
        incoherent_center = psf_incoherent[:, center_y, :].mean(dim=0).cpu().numpy()
        huygens_center = psf_huygens[:, center_y, :].mean(dim=0).cpu().numpy()
    else:  # [H, W]
        coherent_center = psf_coherent[center_y, :].cpu().numpy()
        incoherent_center = psf_incoherent[center_y, :].cpu().numpy()
        huygens_center = psf_huygens[center_y, :].cpu().numpy()

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 线性尺度图
    axes[0].plot(coherent_center, label="Ray-wave PSF", alpha=0.8)
    axes[0].plot(incoherent_center, label="Geometric PSF", alpha=0.8)
    axes[0].plot(huygens_center, label="Huygens PSF", alpha=0.8)
    axes[0].set_xlabel("Pixel Position")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("PSF Center Line Compare (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 对数尺度图，以便更清楚地观察峰值和衍射级次
    axes[1].semilogy(coherent_center + 1e-10, label="Ray-wave PSF", alpha=0.8)
    axes[1].semilogy(incoherent_center + 1e-10, label="Geometric PSF", alpha=0.8)
    axes[1].semilogy(huygens_center + 1e-10, label="Huygens PSF", alpha=0.8)
    axes[1].set_xlabel("Pixel Position")
    axes[1].set_ylabel("Intensity (log scale)")
    axes[1].set_title("PSF Center Line Compare (Log Scale)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./psf_center_line_compare.png", dpi=150)
    plt.close()
    print("Saved PSF center line comparison to ./psf_center_line_compare.png")

def main():
    # 最好使用较高的传感器分辨率（4000x4000 基本可接受，但越高越好）
    lens = GeoLens(
        filename="./datasets/lenses/cellphone/cellphone68deg.json",
        dtype=torch.float64,
    )
    lens.set_sensor_res(sensor_res=(8000, 8000))

    # 计算波前
    calculate_wavefield(lens)

    # 比较 PSF
    compare_psf(lens)


if __name__ == "__main__":
    main()
