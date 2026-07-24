# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""GeoLens 的表面操作混入类。

提供管理光学表面几何形状的方法：
    - 表面裁剪（通光孔径定尺寸）
    - 透镜形状校正
"""

import logging

import torch

from ..geometric_surface import Aperture


class GeoLensSurfOps:
    """为 GeoLens 提供表面几何操作的混入类。

    集中提供设计优化期间修改透镜的方法：通过光线追迹确定通光孔径
    （裁剪），以及校正透镜几何形状。本类应混入 `GeoLens`，
    因而所有方法均可访问宿主上的透镜状态
    （`self.surfaces`、`self.d_sensor`、`self.rfov` 等）。

    主要方法：
        prune_surf：通过光线追迹确定通光孔径尺寸。
        correct_shape：在优化期间修正透镜几何形状。
    """

    # ====================================================================================
# 表面裁剪与形状校正
    # ====================================================================================
    @torch.no_grad()
    def prune_surf(self, mounting_margin=None):
        """裁剪表面半径以使所有有效光线通过，并满足可制造性要求。

        追迹从 0 到完整 FoV 的 16 个子午视场，求出各表面上的最大光线高度
        [mm]，加上安装余量后，再限制候选半径，使其满足边缘矢高上限以及
        与相邻表面之间的边缘间隙（空气间隔/边缘厚度）约束。
        不调整光阑表面的尺寸。最终通过各表面的 `update_r` 写入受限半径。

        参数：
            mounting_margin (float or None, optional)：加到光线追迹所得通光孔径
                半径上的绝对安装余量 [mm]。若为 `None`，则逐表面自动选择：
                光追半径小于 5 mm 时取其 5%，否则取 1 mm。默认值为 None。
        """
        surface_range = self.find_diff_surf()
        num_surfs = len(self.surfaces)

        # ------------------------------------------------------------------
        # 1. 临时移除半径限制，使追迹不被截断
        # ------------------------------------------------------------------
        saved_radii = [self.surfaces[i].r for i in range(num_surfs)]
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].max_height()

        # ------------------------------------------------------------------
        # 2. 在完整 FoV 下追迹光线，求各表面的最大光线高度
        # ------------------------------------------------------------------
        assert self.rfov is not None, "prune_surf() requires self.rfov."
        fov_deg = self.rfov * 180 / torch.pi
        num_fov_samples = 16
        fov_y = torch.linspace(0.0, fov_deg, num_fov_samples, device=self.device)
        ray = self.sample_from_fov(fov_x=[0.0], fov_y=fov_y)
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # 光线记录，shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(
            ray_o_record, nan=0.0, posinf=0.0, neginf=0.0
        )
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # 计算各表面的最大光线高度
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]

        # ------------------------------------------------------------------
        # 3. 生成新半径候选值（尚未写入表面）。
        # ------------------------------------------------------------------
        proposed_r = [float(self.surfaces[i].r) for i in range(num_surfs)]
        for i in surface_range:
            # 光线追迹所需的表面半径
            if surf_r_max[i] > 0:
                base = float(surf_r_max[i].item())
            else:
                base = float(self.surfaces[i].r)

            # 为光追所得半径增加安装余量
            if mounting_margin is None:
                r_expand = 0.05 * base if base < 5.0 else 1.0
            else:
                r_expand = float(mounting_margin)

            # 生成新半径候选值，并将其限制在表面的物理最大高度内
            proposed_r[i] = min(base + r_expand, float(self.surfaces[i].max_height()))

        # ------------------------------------------------------------------
        # 3b. 矢高限制：边缘矢高不得超过 sag_factor * proposed radius。
        # 在 [r_min, proposed_r] 内网格搜索满足约束的最大 r。
        # 网格密度足以处理典型非球面矢高曲线；对非单调极值采用保守处理。
        # ------------------------------------------------------------------
        sag_factor=0.4
        for i in surface_range:
            if not isinstance(self.surfaces[i], Aperture):
                r_prop = proposed_r[i]
                r_cands = torch.linspace(r_prop / 64, r_prop, 64, device=self.device)
                z0 = self.surfaces[i].surface_with_offset(
                    torch.tensor(0.0, device=self.device), 0.0, valid_check=False
                )
                z_cands = self.surfaces[i].surface_with_offset(
                    r_cands, torch.zeros_like(r_cands), valid_check=False
                )
                sag_valid = (z_cands - z0).abs() <= sag_factor * r_cands
                if sag_valid.any():
                    proposed_r[i] = min(r_prop, float(r_cands[sag_valid].max().item()))
                else:
                    proposed_r[i] = float(r_cands[0].item())

        # ------------------------------------------------------------------
        # 4. 边缘间隙检查——预先限制相邻表面对，使写入后的半径不会在边缘
        #    产生自相交。阈值与 loss_bound 一致。限制计算采用相邻表面通光
        #    孔径的公共重叠区，避免依据邻面已被孔径裁掉的区域来裁剪当前表面。
        #    跳过光阑表面；光阑尺寸属于光学规格，不应被裁剪改变。
        #    使用一次向量化网格搜索计算限制值，而非串行二分循环。
        #
        #    每个被裁剪表面都要与前后两个邻面检查。此前实现仅根据 i + 1
        #    限制表面 i，因而表面 i 可能扩展到 i - 1 中，随后导致追迹/优化崩溃。
        # ------------------------------------------------------------------
        min_radius_floor = 0.1  # mm——防止 update_r(0) 使表面失效
        n_cand = 64
        n_edge = 64
        r_frac = torch.linspace(0.5, 1.0, n_edge, device=self.device)
        cand_frac = torch.linspace(1.0 / n_cand, 1.0, n_cand, device=self.device)

        def cap_radius_against_pair(cap_idx, prev_idx, next_idx):
            prev_surf = self.surfaces[prev_idx]
            next_surf = self.surfaces[next_idx]
            if isinstance(prev_surf, Aperture) or isinstance(next_surf, Aperture):
                return
            if isinstance(self.surfaces[cap_idx], Aperture):
                return

            edge_min = 0.1 # mm
            r_check = proposed_r[cap_idx]

            other_idx = next_idx if cap_idx == prev_idx else prev_idx
            other_r = proposed_r[other_idx]

            required_r = max(
                float(surf_r_max[cap_idx].item()),
                min_radius_floor,
            )

                # 向量化限制：一次评估 64 个候选半径的间隙。
            cand_r = cand_frac * r_check
            cand_overlap_r = torch.minimum(
                cand_r, torch.tensor(other_r, device=self.device)
            )
            r_grid = cand_overlap_r.unsqueeze(1) * r_frac.unsqueeze(0)
            z_prev_grid = prev_surf.surface_with_offset(
                r_grid.reshape(-1), 0.0, valid_check=False
            ).reshape(n_cand, n_edge)
            z_next_grid = next_surf.surface_with_offset(
                r_grid.reshape(-1), 0.0, valid_check=False
            ).reshape(n_cand, n_edge)
            per_cand_gap = (z_next_grid - z_prev_grid).min(dim=-1).values
            overlap_ok = per_cand_gap >= edge_min

                # 矢高边界：受限表面在候选 r 处的边缘 z 不得沿轴向越过另一
                # 表面的边缘 z。该检查用于捕获高阶非球面项在设计 r 之外
                # 急剧增大、使其边缘越过邻面，而上述重叠区间隙仍正常的情况。
            cap_surf = self.surfaces[cap_idx]
            other_surf = self.surfaces[other_idx]
            z_other_edge = other_surf.surface_with_offset(
                torch.tensor(other_r, device=self.device),
                torch.tensor(0.0, device=self.device),
                valid_check=False,
            )
            z_cap_at_cand = cap_surf.surface_with_offset(
                cand_r, torch.zeros_like(cand_r), valid_check=False
            )
            if cap_idx > other_idx:
                    # 受限表面在光路中较后——轴向上必须位于另一表面之后
                bracket_ok = z_cap_at_cand > z_other_edge + edge_min
            else:
                    # 受限表面较早——轴向上必须位于另一表面之前
                bracket_ok = z_cap_at_cand < z_other_edge - edge_min

            valid_mask = overlap_ok & bracket_ok
            if not bool(valid_mask.any()):
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): no candidate "
                    f"radius satisfies edge_min {edge_min:.3f} mm at "
                    f"r_check {r_check:.3f} mm (possible sag crossing near "
                    f"axis). Reducing surface {cap_idx} to the ray-required radius "
                    f"{required_r:.3f} mm, but edge clearance may remain "
                    f"violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = float((cand_frac[valid_mask].max() * r_check).item())
            if r_safe < required_r:
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): ray-required "
                    f"radius {required_r:.3f} mm exceeds edge-clearance-safe "
                    f"radius {r_safe:.3f} mm for edge_min {edge_min:.3f} mm. "
                    f"Reducing surface {cap_idx} to the ray-required radius; edge "
                    f"clearance may remain violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = max(r_safe, min_radius_floor)
            if proposed_r[cap_idx] > r_safe:
                proposed_r[cap_idx] = r_safe

        for i in surface_range:
            if i > 0:
                cap_radius_against_pair(i, i - 1, i)
            if i < num_surfs - 1:
                cap_radius_against_pair(i, i, i + 1)

        # ------------------------------------------------------------------
        # 4b. 将受限后的候选半径写入表面。
        # ------------------------------------------------------------------
        for i in surface_range:
            if proposed_r[i] > 0:
                self.surfaces[i].update_r(proposed_r[i])

    @torch.no_grad()
    def correct_shape(self, mounting_margin=None):
        """在透镜设计优化过程中校正无效的透镜形状。

        应用两条校正规则以恢复有效的透镜几何形状：

        1. 平移所有表面（及传感器），使首表面位于
           $z = 0$ mm.
        2. 裁剪所有表面，使全部有效光线通过。

        参数：
            mounting_margin (float or None, optional)：表面裁剪使用的绝对安装
                余量 [mm]，直接传给 `prune_surf`。默认值为 None。
        """
        # 规则 1：将首表面移动到 z = 0.0
        move_dist = self.surfaces[0].d.item()
        for surf in self.surfaces:
            surf.d -= move_dist
        self.d_sensor -= move_dist

        # 规则 2：裁剪所有表面
        self.prune_surf(mounting_margin=mounting_margin)

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """将各表面的材料匹配到玻璃目录中最接近的条目。

        就地将每个表面的 `mat2` 玻璃替换为最接近的真实目录玻璃，
        使理想化设计具备可制造性。

        参数：
            mat_table (str, optional)：玻璃目录名称。支持 'CDGM'（默认目录）
                和 'PLASTIC'。默认值为 'CDGM'。

        异常：
            NotImplementedError：`mat_table` 是无法识别的目录名称时抛出。
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)
