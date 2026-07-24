"""HybridLens 设计示例：优化折射—衍射式相机镜头。

本实验演示 DeepLens 如何使用可微分光线—波动模型联合优化折射光学与衍射光学。
初始系统为混合镜头：在将系统重新对焦到 1 m 后，绿色波长已良好聚焦，但由于色差，
蓝色波长仍处于离焦状态。

目标是使蓝光重新聚焦。每次迭代中，HybridLens 模型计算 489 nm 处的蓝光 PSF，
PSFLoss 促使 PSF 能量集中在理想焦点附近，优化器则通过折射镜头参数和 Binary2
衍射表面进行反向传播。每 100 次迭代，脚本会在带时间戳的结果文件夹中保存当前
镜头 JSON、布局/分析图和 PSF 图像。

技术论文：
    Xinge Yang, Matheus Souza, Kunyi Wang, Praneeth Chakravarthula, Qiang Fu and
    Wolfgang Heidrich, "End-to-End Hybrid Refractive-Diffractive Lens Design with
    Differentiable Ray-Wave Model," Siggraph Asia 2024.
"""

import logging
import os
import random
import string
from datetime import datetime

import torch
import yaml
from torchvision.utils import save_image
from tqdm import tqdm

from deeplens import HybridLens
from deeplens.loss import PSFLoss
from deeplens.utils import set_logger, set_seed


def config():
    # ==> 配置
    args = {"seed": 0, "DEBUG": True}

    # ==> 结果文件夹
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    result_dir = (
        "./results/"
        + datetime.now().strftime("%m%d-%H%M%S")
        + "-HybridLens"
        + "-"
        + random_string
    )
    args["result_dir"] = result_dir
    os.makedirs(result_dir, exist_ok=True)
    print(f"Result folder: {result_dir}")

    if args["seed"] is None:
        seed = random.randint(0, 100)
        args["seed"] = seed
    set_seed(args["seed"])

    # ==> 日志
    set_logger(result_dir)
    if not args["DEBUG"]:
        raise Exception("Add your wandb logging config here.")

    # ==> 设备
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
        args["num_gpus"] = torch.cuda.device_count()
        logging.info(f"Using {args['num_gpus']} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        args["device"] = torch.device("cpu")
        logging.info("Using CPU")

    # ==> 保存配置
    with open(f"{result_dir}/config.yml", "w", encoding="utf-8") as f:
        yaml.dump(args, f)

    with open(
        f"{result_dir}/1_design_hybridlens.py", "w", encoding="utf-8"
    ) as f:
        with open("1_design_hybridlens.py", "r", encoding="utf-8") as code:
            f.write(code.read())

    return args


def main(args):
    # 创建折射—衍射混合镜头
    lens = HybridLens(
        filename="./datasets/lenses/hybridlens/a489_doe.json", dtype=torch.float64
    )
    lens.refocus(foc_dist=-1000.0)

    # 通过 PSF 优化循环使离焦蓝光重新聚焦。
    optimizer = lens.get_optimizer(doe_lr=0.1, lens_lr=[1e-4, 1e-4, 1e-1, 1e-5])
    loss_fn = PSFLoss()
    iterations = 1000
    pbar = tqdm(total=iterations + 1, desc="Progress", postfix={"loss": 0})
    for i in range(iterations + 1):
        psf = lens.psf(points=[0.0, 0.0, -10000.0], ks=128, wvln=0.489)

        optimizer.zero_grad()
        loss = loss_fn(psf)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            lens.write_lens_json(f"{args['result_dir']}/lens_iter{i}.json")
            lens.analysis(save_name=f"{args['result_dir']}/lens_iter{i}.png")
            save_image(
                psf.detach().clone(),
                f"{args['result_dir']}/psf_iter{i}.png",
                normalize=True,
            )

        pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)


if __name__ == "__main__":
    args = config()
    main(args)
