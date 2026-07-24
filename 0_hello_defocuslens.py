"""DeepLens DefocusLens 类的“你好，世界！”示例。

本代码构建一个离焦镜头，预先计算弥散圆（CoC）PSF 并直接应用（不进行光线传递或
薄透镜光线追迹）。它能够模拟离焦模糊，但不模拟高阶光学像差，可作为快速的景深
效果基线渲染器，这种方式常用于 Blender 等工具。

示例将重新对焦镜头，检查若干深度处的弥散圆和景深，生成离焦 PSF，并通过渲染
测试图卡来模拟具有深度相关离焦模糊的图像。

注意：
    DefocusLens 会预先计算弥散圆 PSF（不进行光线追迹），因此图像模拟仅基于 PSF
    （考虑遮挡的 PSF 合成）。

参考资料：
    [1] https://en.wikipedia.org/wiki/Circle_of_confusion
"""

import torch
from torchvision.io import read_image
from torchvision.utils import save_image

from deeplens import DefocusLens

# =====================================================================
# 镜头构建与对焦
# =====================================================================
# 配备 20 x 20 mm 传感器的 50 mm f/1.8 镜头。
lens = DefocusLens(
    foclen=50.0,
    fnum=1.8,
    sensor_size=(20.0, 20.0),
    sensor_res=(64, 64),
)

# 将镜头对焦到相机前方 1 m 处（深度以 mm 为单位，取负值）。
lens.refocus(-1000.0)
print(f"DefocusLens: f={lens.foclen} mm, f/{lens.fnum}, focused at {lens.foc_dist} mm.")

# =====================================================================
# 离焦分析：弥散圆（CoC）与景深（DoF）
# =====================================================================
save_name = "./hello_defocuslens"

depths = torch.tensor([-500.0, -1000.0, -2000.0])  # 近处 / 焦内 / 远处
coc = lens.coc(depths)
dof = lens.dof(depths)
for d, c, f in zip(depths.tolist(), coc.tolist(), dof.tolist()):
    print(f"  depth {d:8.1f} mm -> CoC {c:7.4f} mm, DoF {f:8.2f} mm")
# 在对焦距离处 CoC 约为 0，并随离焦深度增大。

# 离焦的轴上点光源会产生圆盘状（pillbox）模糊 PSF。
point = torch.tensor([[0.0, 0.0, -500.0]])
psf = lens.psf(point, ks=31, psf_type="pillbox")
print(f"Defocus PSF: shape {tuple(psf.shape[-2:])}, sum {psf.sum():.3f}")
save_image(psf.clamp(min=0), f"{save_name}_psf.png", normalize=True)

# =====================================================================
# 图像模拟
# =====================================================================
# 通过镜头在均匀的离焦深度处渲染测试图卡。令传感器匹配输入图像，而不缩放图像。
img = read_image("./datasets/charts/Cam_acc_chart_6MP.png").float()[:3] / 255.0
img = img.unsqueeze(0).to(lens.device)  # [1, 3, H, W]
lens.set_sensor_res((img.shape[-1], img.shape[-2]))  # (W, H)
depth_map = torch.full_like(img[:, :1], 2000.0)  # 物体深度 [mm]，取正值
img_render = lens.render_rgbd(img, depth_map, psf_ks=128)
print(f"Rendered chart through lens: shape {tuple(img_render.shape)}")
save_image(img_render.clamp(0, 1), f"{save_name}_render.png")
print(f"Saved outputs to {save_name}_psf.png and {save_name}_render.png")
