"""GeoLens 设计示例：优化折射式手机相机镜头。

本实验演示 DeepLens 中的可微分几何镜头优化循环。首先加载一个现有的 80 度手机
镜头并执行初始光学分析，然后使用 Adam 细化镜头，以减小整个视场内的 RGB RMS
光斑尺寸。

优化期间，GeoLens 追迹穿过折射表面的光线、计算 RMS 光斑误差，并将梯度反向传播
到可训练的镜头参数。形状控制使镜头保持物理合理性，材料优化则允许玻璃参数与
表面几何形状一同更新。脚本会将中间分析结果保存到带时间戳的结果文件夹，并将
裁剪后的最终设计写为 JSON。
"""

import logging
import os
import random
import string
from datetime import datetime

import torch

from deeplens import GeoLens
from deeplens.utils import set_logger, set_seed


def main() -> None:
    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 结果目录
    tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
    result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-lens-optim-{tag}"
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)
    logging.info(f"Device: {device}")

    # 加载镜头
    lens = GeoLens(filename="./datasets/lenses/cellphone/cellphone80deg.json")
    lens.analysis(save_name=f"{result_dir}/initial")
    logging.info(f"Loaded lens: FoV={lens.rfov:.4f} rad, F/{lens.fnum:.2f}")

    # 优化
    lens.optimize(
        lrs=[1e-3, 1e-3, 1e-3, 1e-4],
        iterations=10000,
        test_per_iter=100,
        shape_control=True,
        optim_mat=True,
        result_dir=result_dir,
    )

    # 最终结果
    lens.prune_surf()
    lens.post_computation()
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    logging.info(f"Done. Results in {result_dir}")


if __name__ == "__main__":
    main()
