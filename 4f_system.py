"""在 Fourier 平面放置衍射表面的 4F 光学系统。

4F 系统通过两个执行 Fourier 变换的镜头，将输入平面中继到输出（传感器）平面。
放置在共用 Fourier（空间频率）平面的衍射表面充当频域滤波器，因此系统 PSF 是
该掩码的逆 Fourier 变换：

    input(z=-f) --f--> ThinLens(f) --f--> Fresnel DOE --f--> ThinLens(f) --f--> sensor
       z=-50              z=0               z=50             z=100            z=150

本脚本从 JSON 加载 4F 系统、绘制布局，并分别在启用 Fourier 平面 DOE 和将其置为
中性（普通 4F 中继）时计算轴上 PSF（输入平面，即镜头 1 前焦平面上的点响应），
从而直观显示滤波器的效果。

PSF 使用 ``ComplexWave.point_wave`` + ``lens.forward``（完整输出光场）直接计算，
而不使用 ``lens.psf``；后者的重新居中/裁剪假设单镜头成像几何，会使 4F 中继偏离
中心。

采样说明：镜头和 DOE 在 0.02mm 网格上逐点施加二次相位，仅当
f/# > ps/lambda (~34) 时才满足带限条件。完整 20mm 光圈（f/2.5）会使相位混叠成
伪影晶格，因此通过 ``point_wave(valid_r=...)`` 将输入点的光圈收缩至
``APERTURE_MM``，以确保各表面均得到充分采样，同时仍能分辨 Airy 光斑和 DOE 模糊。

运行：
    python 4f_system.py            # 默认设备（GPU 机器上使用 CUDA）
    python 4f_system.py cpu        # 强制使用 CPU（本地冒烟测试）
"""

import os
import sys

import torch
from torchvision.utils import save_image

from deeplens import DiffractiveLens
from deeplens.light import ComplexWave

device = sys.argv[1] if len(sys.argv) > 1 else None
save_dir = "./outputs"
os.makedirs(save_dir, exist_ok=True)

# 镜头 1 的前焦距：输入平面位于其前方一个焦距处。
F = 50.0
# 入射光圈直径 [mm]：缩小光圈，使 0.02mm 网格能无混叠地采样镜头/DOE 二次相位
# （要求 f/# > ps/lambda ~ 34）。
APERTURE_MM = 0.3
ZOOM = 64  # 每个 PSF 居中放大视图的半尺寸 [px]

# =====================================================================
# 加载 4F 系统
# =====================================================================
lens = DiffractiveLens(
    filename="./datasets/lenses/diffraclens/4f_doe.json", device=device
)
print(
    f"4F system: {len(lens.surfaces)} surfaces, sensor {lens.sensor_size} mm @ "
    f"{lens.sensor_res} px, d_sensor={float(lens.d_sensor):.1f} mm, "
    f"dtype={lens.dtype}, device={lens.device}"
)
for i, s in enumerate(lens.surfaces):
    print(
        f"  surface {i}: {type(s).__name__:9s} z={float(s.d):6.1f} mm  "
        f"res={tuple(s.res)}  ps={s.ps:.4f} mm  size={float(s.w):.1f} mm"
    )

# 传播范围检查：每个 4F 区段的传播距离均为 F，必须保持在角谱法的 Nyquist 极限
# （size * ps / lambda）内。
ps = lens.surfaces[0].ps
size = lens.surfaces[0].res[0] * ps
wvln = lens.primary_wvln
asm_zmax = size * ps / (wvln * 1e-3)
print(
    f"ASM z_max = {asm_zmax:.1f} mm; per-segment distance = {F:.1f} mm "
    f"-> {'ASM (OK)' if F < asm_zmax else 'OUT OF ASM REGIME!'}"
)

# 带限检查（仅检查传播范围无法发现此问题）：除非 f/# > ps/lambda，否则镜头/DOE
# 二次相位会发生混叠。APERTURE_MM 会缩小光束，以满足该下限要求。
fnum_floor = ps / (wvln * 1e-3)
fnum = F / APERTURE_MM
aperture_max = wvln * 1e-3 * F / ps
print(
    f"Aperture {APERTURE_MM:.2f} mm -> f/{fnum:.0f}; well-sampled needs f/# > "
    f"{fnum_floor:.0f} -> {'OK' if fnum > fnum_floor else 'ALIASED'} "
    f"(max well-sampled aperture {aperture_max:.2f} mm)"
)

# =====================================================================
# 布局
# =====================================================================
lens.draw_layout(save_name=f"{save_dir}/4f_layout.png")
print(f"Saved layout to {save_dir}/4f_layout.png")


# =====================================================================
# PSF（通过完整输出光场获得的输入平面点响应）
# =====================================================================
def psf_full(depth):
    """输入平面点光源在完整传感器平面上的强度。"""
    s0 = lens.surfaces[0]
    field_res = [s0.res[0], s0.res[1]]
    field_size = [s0.res[0] * s0.ps, s0.res[1] * s0.ps]
    inp = ComplexWave.point_wave(
        point=[0.0, 0.0, depth],
        phy_size=field_size,
        res=field_res,
        wvln=wvln,
        z=0.0,
        valid_r=APERTURE_MM / 2,
    ).to(lens.device)
    out = lens.forward(inp)
    return (out.u.abs() ** 2)[0, 0]


def peak_pixel(intensity):
    """强度最大值所在像素 (row, col)，即轴上像点。"""
    W = intensity.shape[1]
    flat = int(torch.argmax(intensity))
    return flat // W, flat % W


def save_psf(intensity, name, center):
    """保存完整视图和居中放大视图（线性与对数），并报告能量集中度。

    裁剪以 ``center``（中继后的轴上像点）而非网格中心为中心：由于 FFT 居中约定，
    4F 中继会将轴上点成像到相对 H//2 固定偏移的像素处，基线和 DOE 的偏移相同。
    """
    I = intensity.detach().float().cpu()
    H, W = I.shape
    ci = max(ZOOM, min(H - ZOOM, center[0]))  # 保证裁剪窗口不越界
    cj = max(ZOOM, min(W - ZOOM, center[1]))

    # 完整传感器视图。
    save_image((I / I.max())[None], f"{save_dir}/{name}_full.png")

    # 居中放大视图（线性 + 对数），便于清晰比较。
    crop = I[ci - ZOOM : ci + ZOOM, cj - ZOOM : cj + ZOOM]
    save_image((crop / crop.max())[None], f"{save_dir}/{name}.png")
    crop_log = torch.log10(crop + 1e-6 * crop.max())
    crop_log = (crop_log - crop_log.min()) / (crop_log.max() - crop_log.min() + 1e-9)
    save_image(crop_log[None], f"{save_dir}/{name}_log.png")

    r = 10
    e20 = float(I[ci - r : ci + r, cj - r : cj + r].sum()) / float(I.sum()) * 100
    print(
        f"  {name}: peak/mean {float(I.max() / I.mean()):.0f}, "
        f"energy in 20px@image-point {e20:.1f}%"
    )


# 首先计算基线：将 Fourier DOE 置为中性（Fresnel f0 -> ~infinity = 平坦相位），
# 使系统退化为普通 4F 中继（点 -> 点）。其尖锐峰值可定位轴上像点，用于将两次
# 裁剪居中。
f0_orig = lens.surfaces[1].f0.clone()
lens.surfaces[1].f0 = torch.full_like(lens.surfaces[1].f0, 1e9)
baseline = psf_full(-F)
ci, cj = peak_pixel(baseline)
H, W = baseline.shape
print(
    f"On-axis image point at pixel ({ci}, {cj}); grid centre ({H // 2}, {W // 2}); "
    f"offset ({ci - H // 2:+d}, {cj - W // 2:+d}) px"
)
save_psf(baseline, "4f_psf_baseline", (ci, cj))
lens.surfaces[1].f0 = f0_orig

# 含 Fourier 平面衍射表面的 PSF，以同一点为中心。
save_psf(psf_full(-F), "4f_psf_doe", (ci, cj))

print(f"Saved PSFs to {save_dir}/4f_psf_doe.png and {save_dir}/4f_psf_baseline.png")
print("Done.")
