"""
从零开始自动设计镜头。本代码使用 RMS 光斑尺寸进行镜头设计，速度远快于基于图像的镜头设计。

技术论文：
    Xinge Yang, Qiang Fu and Wolfgang Heidrich, "Curriculum learning for ab initio deep learned refractive optics," Nature Communications 2024.
"""

import logging
import math
import os
import random
import string
from datetime import datetime

import torch
import yaml
from tqdm import tqdm

from deeplens import GeoLens
from deeplens.geolens_pkg import create_lens
from deeplens.config import DEPTH, EPSILON, WAVE_RGB
from deeplens.utils import create_video_from_images, set_logger, set_seed


def config():
    """训练配置文件。"""
    # 配置文件
    with open("configs/2_auto_lens_design.yml", encoding="utf-8") as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # 结果目录
    characters = string.ascii_letters + string.digits
    random_string = "".join(random.choice(characters) for i in range(4))
    current_time = datetime.now().strftime("%m%d-%H%M%S")
    exp_name = current_time + "-AutoLens-RMS-" + random_string
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    args["result_dir"] = result_dir

    if args["seed"] is None:
        seed = random.randint(0, 100000)
        args["seed"] = seed
    set_seed(args["seed"])

    # 日志
    set_logger(result_dir)
    logging.info(f"EXP: {args['EXP_NAME']}")

    # 设备
    if torch.cuda.is_available():
        args["device"] = torch.device("cuda")
        args["num_gpus"] = torch.cuda.device_count()
        logging.info(f"Using {args['num_gpus']} {torch.cuda.get_device_name(0)} GPU(s)")
    else:
        args["device"] = torch.device("cpu")
        logging.info("Using CPU")

    # ==> 保存配置和原始代码
    with open(f"{result_dir}/config.yml", "w", encoding="utf-8") as f:
        yaml.dump(args, f)

    with open(f"{result_dir}/2_autolens_rms.py", "w", encoding="utf-8") as f:
        with open("2_autolens_rms.py", "r", encoding="utf-8") as code:
            f.write(code.read())

    return args


def curriculum_design(
    self: GeoLens,
    lrs=[1e-4, 1e-4, 1e-2, 1e-4],
    iterations=5000,
    test_per_iter=100,
    optim_mat=False,
    shape_control=True,
    result_dir="./results",
):
    """通过最小化 RMS 误差来优化镜头。"""
    # 准备
    depth = DEPTH
    num_ring = 16
    num_arm = 4
    spp = 1024

    aper_start = self.surfaces[self.aper_idx].r * 0.25
    aper_final = self.surfaces[self.aper_idx].r

    # 日志
    if not logging.getLogger().hasHandlers():
        set_logger(result_dir)
    logging.info(
        f"lr:{lrs}, iterations:{iterations}, spp:{spp}, num_ring:{num_ring}, num_arm:{num_arm}."
    )

    # 优化器
    optimizer = self.get_optimizer(lrs, optim_mat=optim_mat)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=iterations // 4, T_mult=1
    )

    # 训练循环
    pbar = tqdm(
        total=iterations + 1, desc="Progress", postfix={"loss_rms": 0, "loss_reg": 0}
    )
    for i in range(iterations + 1):
        # =======================================
        # 评估镜头
        # =======================================
        if i % test_per_iter == 0:
            with torch.no_grad():
                # 课程学习：逐步增大光圈尺寸
                progress = 0.5 * (1 + math.cos(math.pi * (1 - i / iterations)))
                aper_r = min(
                    aper_start + (aper_final - aper_start) * progress,
                    aper_final,
                )
                self.surfaces[self.aper_idx].update_r(aper_r)
                self.calc_pupil()

                # 校正镜头形状并评估当前设计
                if i > 0:
                    if shape_control:
                        self.correct_shape()
                        # self.refocus()

                # 保存镜头
                self.write_lens_json(f"{result_dir}/iter{i}.json")
                self.analysis(f"{result_dir}/iter{i}")

                # 采样新光线并计算目标中心
                rays_backup = []
                for wv in WAVE_RGB:
                    ray = self.sample_ring_arm_rays(
                        num_ring=num_ring,
                        num_arm=num_arm,
                        depth=depth,
                        spp=spp,
                        wvln=wv,
                        scale_pupil=1.10,
                    )
                    rays_backup.append(ray)

                center_ref = -self.psf_center(points_obj=ray.o[:, :, 0, :], method="pinhole")
                center_ref = center_ref.unsqueeze(-2).repeat(1, 1, spp, 1)

        # =======================================
        # 通过最小化 RMS 优化镜头
        # =======================================
        loss_rms = []
        for wv_idx, wv in enumerate(WAVE_RGB):
            # 将光线追迹到传感器，[num_grid, num_grid, num_rays, 3]
            ray = rays_backup[wv_idx].clone()
            ray = self.trace2sensor(ray)

            # 光线相对中心的误差与有效掩码
            ray_xy = ray.o[..., :2]
            ray_valid = ray.is_valid
            ray_err = ray_xy - center_ref

            # 权重掩码（不可微分），shape 为 [num_grid, num_grid]
            if wv_idx == 0:
                with torch.no_grad():
                    weight_mask = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
                    weight_mask /= ray_valid.sum(-1) + EPSILON
                    weight_mask /= weight_mask.mean()

                    # 丢弃（权重掩码的 20%）
                    dropout_mask = torch.rand_like(weight_mask) < 0.1
                    weight_mask = weight_mask * (~dropout_mask)

            # RMS 误差损失，shape 为 [num_grid, num_grid]
            l_rms = ((ray_err**2).sum(-1) * ray_valid).sum(-1)
            l_rms /= ray_valid.sum(-1) + EPSILON
            l_rms = (l_rms + EPSILON).sqrt()

            # 加权损失
            l_rms_weighted = (l_rms * weight_mask).sum()
            l_rms_weighted /= weight_mask.sum() + EPSILON
            loss_rms.append(l_rms_weighted)

        # 所有波长的 RMS 损失
        loss_rms = sum(loss_rms) / len(loss_rms)

        # 添加聚焦损失和镜头设计约束
        w_focus = 0.1
        loss_focus = self.loss_infocus()
        loss_reg, loss_dict = self.loss_reg()
        w_reg = 0.05
        L_total = loss_rms + w_focus * loss_focus + w_reg * loss_reg

        # 基于梯度的优化
        optimizer.zero_grad()
        L_total.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss_rms=loss_rms.item(), **loss_dict)
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    args = config()
    result_dir = args["result_dir"]
    device = args["device"]

    # 绑定函数
    GeoLens.curriculum_design = curriculum_design

    # 创建镜头
    lens = create_lens(
        foclen=args["foclen"],
        fov=args["fov"],
        fnum=args["fnum"],
        bfl=args["bfl"],
        thickness=args["thickness"],
        surf_list=args["surf_list"],
        save_dir=result_dir,
    )
    lens.set_target_fov_fnum(
        rfov=args["fov"] / 2 / 57.3,
        fnum=args["fnum"],
    )
    logging.info(
        f"==> Design target: focal length {round(args['foclen'], 2)}, diagonal FoV {args['fov']}deg, F/{args['fnum']}"
    )

    # 使用 RMS 误差进行课程学习
    # 从零开始时优化难度较高且梯度不稳定，课程学习用于寻找优化路径。3000 次迭代是
    # 合理的起始值，增加迭代次数可改善光学性能。此阶段也可选择优化材料。
    lens.curriculum_design(
        lrs=[float(lr) for lr in args["lrs"]],
        iterations=2000,
        test_per_iter=50,
        optim_mat=True,
        shape_control=True,
        result_dir=args["result_dir"],
    )

    # 匹配材料并设置 fnum
    lens.match_materials()
    lens.set_fnum(args["fnum"])
    lens.write_lens_json(f"{result_dir}/curriculum_final.json")

    # 为获得最佳光学性能，通常还需要额外的训练迭代。本代码使用较强的镜头设计约束
    # 和较小的学习率，因此优化较慢，但能稳定改善光学性能。为便于演示，此处仅训练
    # 3000 步。
    lens = GeoLens(filename=f"{result_dir}/curriculum_final.json")
    lens.set_target_fov_fnum(
        rfov=args["fov"] / 2 / 57.3,
        fnum=args["fnum"],
    )
    lens.optimize(
        lrs=[float(lr) for lr in args["lrs"]],
        iterations=5000,
        test_per_iter=100,
        optim_mat=False,
        shape_control=True,
        result_dir=f"{args['result_dir']}/fine-tune",
    )

    # 分析最终结果
    lens.prune_surf()
    lens.post_computation()

    logging.info(
        f"Actual: diagonal FOV {lens.rfov}, r sensor {lens.r_sensor}, F/{lens.fnum}."
    )
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    # 创建视频
    create_video_from_images(f"{result_dir}", f"{result_dir}/autolens.mp4", fps=10)
