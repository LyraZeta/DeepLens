"""演示三种基于论文的衍射表面及其 PSF。

对每个表面，保存其设计波长相位图和 PSF，并输出一个定量描述符。若 CUDA 可用
（AutoDL）则在 CUDA 上运行，否则使用 CPU。

  * Rank1（Sun 等，CVPR 2020）：低秩高度图 h = h_max*sigmoid(V@Q.T)。鞍形
    初始化会产生各向异性的条纹状 PSF。用于单次曝光 HDR 的强十字 PSF 来自端到端
    HDR 训练（不在本示例范围内）。
  * DiffractedRotation（Jeon 等，TOG 2019）：按角度划分的闪耀 Fresnel 扇区
    （公式 12）-> N 重“螺旋”相位图。注意：论文中报告的随波长旋转的 PSF 是在其
    重建流程下的焦平面产生的；DeepLens 的近轴 ASM 模型则显示固定的 N 重各向
    异性结构。
  * RotationallySymmetric（Dun 等，Optica 2020）：自由形式一维径向轮廓。
    PSF 在各波长下均为旋转对称（消色差本身需要端到端训练，不在本示例范围内）。

输出写入 ./outputs/diffractive_surfaces/。
"""

import os

import torch
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.diffractive_surface import DiffractedRotation, Rank1

OUT = "./outputs/diffractive_surfaces"
os.makedirs(OUT, exist_ok=True)
# 避免使用 MPS：DeepLens 波传播使用 Apple MPS 不支持的 float64。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def _axis_ratio(psf):
    """PSF 强度协方差特征值之比（1.0 = 各向同性）。"""
    h, w = psf.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=psf.device, dtype=psf.dtype),
        torch.arange(w, device=psf.device, dtype=psf.dtype),
        indexing="ij",
    )
    p = psf / psf.sum()
    cy, cx = (p * yy).sum(), (p * xx).sum()
    yy, xx = yy - cy, xx - cx
    cxx = (p * xx * xx).sum()
    cyy = (p * yy * yy).sum()
    cxy = (p * xx * yy).sum()
    tr = (cxx + cyy).item()
    det = (cxx * cyy - cxy * cxy).item()
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    return (tr / 2 + disc) / max(tr / 2 - disc, 1e-9)


def demo_rank1():
    """鞍形初始化的 rank-1 DOE -> 各向异性条纹状 PSF。"""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/rank1.json", device=DEVICE
    )
    r1 = [s for s in lens.surfaces if isinstance(s, Rank1)][0]
    n0, n1 = r1.res
    # 鞍形初始化：V @ Q.T = outer(ramp, ramp) -> 像散（十字）相位。
    r1.V = torch.linspace(-3, 3, n0, device=lens.device)[:, None]
    r1.Q = torch.linspace(-3, 3, n1, device=lens.device)[:, None]

    r1.draw_phase_map(save_name=f"{OUT}/rank1_phase.png")
    psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=96)
    save_image(psf[None].clamp(min=0), f"{OUT}/rank1_psf.png", normalize=True)
    print(f"[Rank1] PSF axis_ratio={_axis_ratio(psf):.2f}  (anisotropic streak; "
          "strong HDR cross needs end-to-end training)")


def demo_diffracted_rotation():
    """保存螺旋相位图和波长 PSF 拼图（N 重结构）。"""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/diffracted_rotation.json", device=DEVICE
    )
    doe = [s for s in lens.surfaces if isinstance(s, DiffractedRotation)][0]
    doe.draw_phase_map(save_name=f"{OUT}/diffracted_rotation_phase.png")

    frames = []
    for wvln in [0.45, 0.50, 0.55, 0.60, 0.65]:
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128, wvln=wvln)
        frames.append(psf.clamp(min=0))
        print(f"[DiffractedRotation] wvln={wvln:.2f}um  axis_ratio={_axis_ratio(psf):.2f}")
    montage = torch.stack(frames, dim=0)[:, None]
    save_image(montage, f"{OUT}/diffracted_rotation_sweep.png", nrow=len(frames), normalize=True)
    print("[DiffractedRotation] saved spiral phase map + wavelength sweep "
          "(N-fold structure; rotation needs the paper's focal-plane pipeline)")


def demo_rotational_symmetric():
    """多个波长下的旋转对称 PSF。"""
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/rotational_symmetric.json", device=DEVICE
    )
    doe = lens.surfaces[0]
    doe.draw_phase_map(save_name=f"{OUT}/rotational_symmetric_phase.png")

    frames = []
    for wvln in [0.45, 0.55, 0.65]:
        psf = lens.psf(points=[0.0, 0.0, float("-inf")], ks=128, wvln=wvln)
        rot_err = float((psf - torch.rot90(psf, 1)).abs().sum() / psf.abs().sum())
        frames.append(psf.clamp(min=0))
        print(f"[RotationallySymmetric] wvln={wvln:.2f}um  rot90_err={rot_err:.4f}")
    montage = torch.stack(frames, dim=0)[:, None]
    save_image(montage, f"{OUT}/rotational_symmetric_psf.png", nrow=len(frames), normalize=True)
    print("[RotationallySymmetric] saved phase map + multi-wavelength PSF "
          "(rotationally symmetric; achromaticity requires end-to-end training)")


if __name__ == "__main__":
    demo_rank1()
    demo_diffracted_rotation()
    demo_rotational_symmetric()
    print(f"\nDone. Images in {OUT}/")
