"""通过优化轴上 PSF 使其聚焦到传感器，设计 Pixel2D 衍射相位板。

在传感器前一个焦距（50 mm）处放置单个 Pixel2D DOE，其中每个像素都是独立、随机
初始化的相位值。从随机相位开始，最大化轴上 PSF 的峰值（Strehl）强度
（``PSFStrehlLoss``），使准直的轴上光束在传感器平面汇聚为紧致光斑。换言之，
DOE 将从零开始学习类似镜头的 Fresnel 相位轮廓。

技术论文：
    [1] Vincent Sitzmann et al., "End-to-end optimization of optics and image
        processing for achromatic extended depth of field and super-resolution
        imaging," SIGGRAPH 2018.
"""

import logging
import os
import random
import string
from datetime import datetime

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens import DiffractiveLens
from deeplens.loss import PSFStrehlLoss
from deeplens.utils import set_logger, set_seed


def main() -> None:
    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 结果目录
    tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
    result_dir = (
        f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-diffraclens-design-{tag}"
    )
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)
    logging.info(f"Device: {device}")

    # 加载单个随机初始化的 Pixel2D 相位板，传感器位于其后 50 mm 处。
    lens = DiffractiveLens(
        filename="./datasets/lenses/diffraclens/pixel2d.json",
        device=device,
        dtype=torch.float64,
    )
    doe = lens.surfaces[0]
    logging.info(
        f"Pixel2D DOE: {doe.res} px, aperture {doe.w:.2f} x {doe.h:.2f} mm, "
        f"sensor at {float(lens.d_sensor):.1f} mm (f/{lens.foclen / doe.w:.1f}). "
        f"Phase randomly initialized."
    )

    # 轴上准直光源（物体位于无穷远）。
    on_axis_inf = [0.0, 0.0, float("-inf")]

    # 初始（随机相位）状态。
    with torch.no_grad():
        doe.draw_phase_map(save_name=f"{result_dir}/phase_init.png")
        psf_init = lens.psf(points=on_axis_inf, ks=None)
        save_image(
            psf_init[None].clamp(min=0), f"{result_dir}/psf_init.png", normalize=True
        )
    lens.draw_layout(save_name=f"{result_dir}/layout.png")

    # 优化轴上 PSF 使其聚焦（PSFStrehlLoss 最大化中心像素强度，即 Strehl 比）。
    optimizer = lens.get_optimizer(lr=0.1)
    loss_fn = PSFStrehlLoss()

    pbar = tqdm(range(1000 + 1), desc="Designing DOE")
    for i in pbar:
        psf = lens.psf(points=on_axis_inf, ks=None)
        # PSFStrehlLoss 接收 RGB [3, ks, ks] PSF 并返回待最大化的分数；复制单色
        # PSF，并最小化该分数的负值。
        strehl = loss_fn(psf.unsqueeze(0).repeat(3, 1, 1))
        loss = -strehl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"strehl": f"{strehl.item():.4e}"})
        if i % 100 == 0:
            logging.info(f"[iter {i:5d}] strehl = {strehl.item():.6e}")
            with torch.no_grad():
                save_image(
                    psf.detach()[None].clamp(min=0),
                    f"{result_dir}/psf_iter{i}.png",
                    normalize=True,
                )

    # 最终结果。
    with torch.no_grad():
        doe.draw_phase_map(save_name=f"{result_dir}/phase_final.png")
        psf_final = lens.psf(points=on_axis_inf, ks=None)
        save_image(
            psf_final[None].clamp(min=0), f"{result_dir}/psf_final.png", normalize=True
        )
    lens.write_lens_json(f"{result_dir}/final_lens.json")

    logging.info(f"Done. Results in {result_dir}")


if __name__ == "__main__":
    main()
