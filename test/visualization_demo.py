import argparse
import os

import numpy as np

from deeplens import GeoLens

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default="./visualization")
args = parser.parse_args()
SAVE_DIR = args.save_dir

if not os.path.exists(SAVE_DIR):
    os.mkdir(SAVE_DIR)

# lens_config = os.path.relpath("./lenses/cellphone/cellphone68deg.json")
lens_config = os.path.relpath("./datasets/lenses/camera/ef50mm_f1.8.json")

lens = GeoLens(lens_config)
rfov = lens.rfov

# =============================================================================
# 测试 1：保存镜头 OBJ 文件（不需要 PyVista）
# =============================================================================
print("=" * 60)
print("Test 1: save_lens_obj (no PyVista required)")
print("=" * 60)

lens.save_lens_obj(save_dir=SAVE_DIR, save_elements=True, save_rays=True, is_wrap=True)

print(f"OBJ files saved to {SAVE_DIR}")
print("Test 1 passed!\n")
breakpoint()

# =============================================================================
# 测试 2：绘制镜头三维布局（需要延迟加载的 PyVista）
# =============================================================================
print("=" * 60)
print("Test 2: draw_lens_3d (PyVista required - lazy loaded)")
print("=" * 60)

# 用于无头渲染的 PyVista 设置
os.environ["PYVISTA_OFF_SCREEN"] = "1"  # 强制离屏渲染
os.environ["PYVISTA_JUPYTER_BACKEND"] = "static"  # 避免使用小组件/CDN
# 可选：若此前设置了 DISPLAY，请将其清除，以避免 Qt/GLX 尝试启动
os.environ.pop("DISPLAY", None)

import pyvista as pv

# 若 Xvfb 恰好可用，此设置对许多无头环境有帮助。
# （若未安装 Xvfb，则不执行任何操作。）
try:
    pv.start_xvfb()  # 若存在则启动虚拟 X 服务器
    print("xvfb started")
except Exception as e:
    print("xvfb not started:", e)

plotter = pv.Plotter(off_screen=True, notebook=True)
lens.draw_lens_3d(
    plotter=plotter,
    save_dir=SAVE_DIR,
    fovs=[0.0, rfov * 0.99 * 57.296],
    fov_phis=[45.0, 135.0, 225.0, 315.0],
    draw_rays=True,
)

print(f"3D layout saved to {SAVE_DIR}/lens_layout3d.png")
print("Test 2 passed!\n")

print("=" * 60)
print("All tests completed successfully!")
print("=" * 60)
