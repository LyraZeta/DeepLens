# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Pixel2D DOE 参数化，其中每个像素都是独立参数。"""

import torch
from .diffractive import DiffractiveSurface


class Pixel2D(DiffractiveSurface):
    """使用逐像素直接相位图的 Pixel2D DOE 参数化。

    相位图的每个像素都是独立的可优化参数，因此这是最通用（也是维度最高）的
    DOE 参数化方式。相位图按设计波长 `wvln0` 存储。

    属性：
        phase_map (torch.Tensor): 设计波长下的逐像素相位。
            [H, W]. [rad]
    """

    def __init__(
        self,
        d,
        phase_map_path=None,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        device="cpu",
    ):
        """初始化每个像素均为独立参数的 Pixel2D DOE。

        若 `phase_map_path` 为 None，则用较小的随机值（`torch.randn * 1e-3`）
        初始化相位图；否则从给定路径加载。

        参数：
            d (float): DOE 表面沿光轴的位置。[mm]
            phase_map_path (str or None, optional): 待加载的已保存相位图张量路径。
                若为 None，则随机初始化相位图。默认值为 None。
            res (tuple or int, optional): DOE 分辨率，格式为 (H, W)；整数会
                扩展为 (res, res)。[pixel]。默认值为 (2000, 2000)。
            mat (str, optional): DOE 材料。默认值为 "fused_silica"。
            wvln0 (float, optional): 设计波长。[um]。默认值为 0.55。
            fab_ps (float, optional): 制造像素尺寸。[mm]。默认值为 0.001。
            fab_step (int, optional): 制造量化级数。默认值为 16。
            device (str, optional): 运行 DOE 的设备。默认值为 "cpu"。

        异常：
            ValueError: 当 `phase_map_path` 既不是 None 也不是字符串时抛出。
        """
        super().__init__(d=d, res=res, mat=mat, fab_ps=fab_ps, fab_step=fab_step, wvln0=wvln0, device=device)

        # 使用随机值初始化相位图
        if phase_map_path is None:
            self.phase_map = torch.randn(self.res, device=self.device) * 1e-3
        elif isinstance(phase_map_path, str):
            self.phase_map = torch.load(phase_map_path, map_location=device, weights_only=True)
        else:
            raise ValueError(f"Invalid phase_map_path: {phase_map_path}")

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """从字典初始化 Pixel2D DOE。

        参数：
            doe_dict (dict): 表面字典，必须包含 "d" 和 "res"，可选键为
                "mat"、"fab_ps"、"fab_step"、"phase_map_path"、"wvln0"。

        返回：
            doe (Pixel2D): 构造得到的 Pixel2D DOE。
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            phase_map_path=doe_dict.get("phase_map_path", None),
            wvln0=doe_dict.get("wvln0", 0.55),
        )

    def phase_func(self):
        """返回设计波长下的原始逐像素相位图。

        返回：
            phase_map (torch.Tensor): 设计波长下的逐像素相位。
                [H, W]. [rad]
        """
        return self.phase_map

    # =======================================
    # 优化
    # =======================================
    def get_optimizer_params(self, lr=0.01):
        """获取相位图的优化器参数组。

        启用相位图的梯度，并使用给定学习率将其作为单个 Adam 风格参数组返回。

        参数：
            lr (float, optional): 相位图的学习率。默认值为 0.01。

        返回：
            optimizer_params (list): 包含一个参数组字典的列表
                {"params": [phase_map], "lr": lr}.
        """
        self.phase_map.requires_grad = True
        optimizer_params = [{"params": [self.phase_map], "lr": lr}]
        return optimizer_params

    # =======================================
    # 输入输出
    # =======================================
    def surf_dict(self, phase_map_path):
        """返回可序列化的表面字典，并将相位图保存到磁盘。

        在基础表面字典中加入相位图路径，并将已分离且位于 CPU 上的相位图张量
        写入 `phase_map_path`。

        参数：
            phase_map_path (str): 相位图张量的保存路径，该路径也会记录在
                返回的字典中。

        返回：
            surf_dict (dict): 包含 "phase_map_path" 项的表面字典。
        """
        surf_dict = super().surf_dict()
        surf_dict["phase_map_path"] = phase_map_path
        torch.save(self.phase_map.clone().detach().cpu(), phase_map_path)
        return surf_dict
