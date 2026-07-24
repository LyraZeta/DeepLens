# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""GeoLens 的可视化函数。

函数：
    光线采样（2D）：
        - sample_parallel_2D()：在物方采样平行光线（2D）
        - sample_point_source_2D()：在物方采样点光源光线（2D）

    2D 布局可视化：
        - draw_layout()：绘制带光线追迹的 2D 透镜布局
        - draw_lens_2d()：在 2D 图中绘制透镜布局
        - draw_ray_2d()：绘制光线路径

    遮挡结构叠加：
        - create_barrier()：在 2D 布局上叠加绘制镜筒
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from ..light import Ray


class GeoLensVis:
    """为 `GeoLens` 提供 2D 透镜布局与光线可视化的混入类。

    生成出版质量的截面图，在子午面或弧矢面中显示透镜表面和追迹光束。

    本类不单独实例化，而是混入 `GeoLens`。
    """

    # ====================================================================================
    # 2D 布局的光线采样函数
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(
        self,
        fov=0.0,
        num_rays=7,
        wvln=None,
        plane="meridional",
        entrance_pupil=True,
        depth=0.0,
    ):
        """在物方采样平行光线（2D）。

        用于：(1) 绘制透镜系统；(2) 2D 几何光学计算，例如重新聚焦到无穷远。

        参数：
            fov (float, optional)：入射角 [degree]。默认值为 0.0。
            num_rays (int, optional)：光线数量。默认值为 7。
            wvln (float or None, optional)：光线波长 [µm]。为 None 时回退到
                `self.primary_wvln`。默认值为 None。
            plane (str, optional)：采样平面，可为 "meridional"（y-z 平面）
                或 "sagittal"（x-z 平面）。默认值为 "meridional"。
            entrance_pupil (bool, optional)：为 True 时在入瞳上采样，否则在
                首表面孔径上采样。默认值为 True。
            depth (float, optional)：光线传播到的采样深度 [mm]。默认值为 0.0。

        返回：
            rays (Ray)：采样光线，其原点/方向张量的 shape 为 [num_rays, 3]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        # 在瞳孔上采样点
        if entrance_pupil:
            pupilz, pupilr = self.get_entrance_pupil()
        else:
            pupilz, pupilr = self.surfaces[0].d.item(), self.surfaces[0].r

        # 采样光线原点，shape [num_rays, 3]
        if plane == "sagittal":
            ray_o = torch.stack(
                (
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), 0),
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_o = torch.stack(
                (
                    torch.full((num_rays,), 0),
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # 采样光线方向，shape [num_rays, 3]
        if plane == "sagittal":
            ray_d = torch.stack(
                (
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_d = torch.stack(
                (
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # 构造光线并传播到目标深度
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        rays.prop_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source_2D(
        self,
        fov=0.0,
        depth=None,
        num_rays=7,
        wvln=None,
        entrance_pupil=True,
    ):
        """在物方采样点光源光线（2D）。

        用于绘制透镜系统。

        参数：
            fov (float, optional)：入射角 [degree]。默认值为 0.0。
            depth (float or None, optional)：物面深度 [mm]。为 None 时回退到
                `self.obj_depth`。默认值为 None。
            num_rays (int, optional)：光线数量。默认值为 7。
            wvln (float or None, optional)：光线波长 [µm]。为 None 时回退到
                `self.primary_wvln`。默认值为 None。
            entrance_pupil (bool, optional)：为 True 时将光线瞄准入瞳，否则
                瞄准首表面孔径。默认值为 True。

        返回：
            ray (Ray)：采样光线，其原点/方向张量的 shape 为 [num_rays, 3]。
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        # 在物面上采样点
        ray_o = torch.tensor([depth * float(np.tan(np.deg2rad(fov))), 0.0, depth])
        ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)

        # 在瞳孔上采样点（第二点）
        if entrance_pupil:
            pupilz, pupilr = self.calc_entrance_pupil()
        else:
            pupilz, pupilr = self.surfaces[0].d.item(), self.surfaces[0].r

        x2 = torch.linspace(-pupilr, pupilr, num_rays) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        ray_o2 = torch.stack((x2, y2, z2), axis=1)

        # 构造光线
        ray_d = ray_o2 - ray_o
        ray = Ray(ray_o, ray_d, wvln, device=self.device)

        # 将光线传播到采样深度
        ray.prop_to(depth)
        return ray

    # ====================================================================================
    # 透镜 2D 布局
    # ====================================================================================
    def draw_layout(
        self,
        filename,
        depth=float("inf"),
        zmx_format=True,
        multi_plot=False,
        lens_title=None,
        show=False,
        return_fig=False,
    ):
        """绘制带光线追迹的 2D 透镜布局。

        ``lens_title`` 为 None 时自动生成标题，其中包含焦距、F 数、FoV、
        IMGH、RGB 波长，以及由 ``analysis_spot()`` 得到的各 FoV RMS
        光斑半径第二行。

        参数：
            filename (str)：输出文件名。
            depth (float, optional)：光线追迹的物距 [mm]。准直输入使用
                ``float('inf')``。默认值为 ``float('inf')``。
            zmx_format (bool, optional)：为 True 时以 Zemax 风格绘制表面。默认值为 True。
            multi_plot (bool, optional)：为 True 时为每个波长创建一个子图。默认值为 False。
            lens_title (str or None, optional)：标题字符串。为 None 时自动生成。默认值为 None。
            show (bool, optional)：为 True 时交互显示图像而非保存。默认值为 False。
            return_fig (bool, optional)：为 True 时直接返回坐标轴和图像，不保存也不关闭，
                供 `create_barrier` 等调用方继续叠加绘制。默认值为 False。

        返回：
            result (tuple or None)：`return_fig` 为 True 时返回 matplotlib
                坐标轴和图像对象 (ax, fig)，否则返回 None。
        """
        num_rays = 11
        num_views = 3

        # 透镜标题
        if lens_title is None:
            eff_foclen = round(self.foclen, 2)
            fov_deg = round(2 * self.rfov * 180 / torch.pi, 1)
            imgh = round(self.r_sensor, 1)
            wvl_nm = [int(round(w * 1000)) for w in self.wvln_rgb]  # µm → nm

            if self.aper_idx is not None:
                _, pupil_r = self.calc_entrance_pupil()
                fnum = round(eff_foclen / pupil_r / 2, 2)
                line1 = (
                    f"FocLen{eff_foclen}mm - F/{fnum} - FoV{fov_deg} - "
                    f"IMGH{imgh}mm - RGB({wvl_nm[0]}/{wvl_nm[1]}/{wvl_nm[2]}nm)"
                )
            else:
                line1 = (
                    f"FocLen{eff_foclen}mm - FoV{fov_deg} - "
                    f"IMGH{imgh}mm - RGB({wvl_nm[0]}/{wvl_nm[1]}/{wvl_nm[2]}nm)"
                )

            spot = self.analysis_spot(num_field=3)
            rms0 = spot["fov0.0"]["rms"]
            rms5 = spot["fov0.5"]["rms"]
            rms10 = spot["fov1.0"]["rms"]
            line2 = f"RMS spot: 0.0FoV={rms0:.2f}\u03bcm  0.5FoV={rms5:.2f}\u03bcm  1.0FoV={rms10:.2f}\u03bcm"
            lens_title = f"{line1}\n{line2}"

        # 绘制透镜布局
        colors_list = ["#CC0000", "#006600", "#0066CC"]
        rfov_deg = float(np.rad2deg(self.rfov))
        fov_ls = np.linspace(0, rfov_deg * 0.99, num=num_views)
        
        if not multi_plot:
            ax, fig = self.draw_lens_2d(zmx_format=zmx_format)
            fig.suptitle(lens_title, fontsize=10, fontfamily="Nimbus Sans")
            for i, fov in enumerate(fov_ls):
            # 采样光线，shape (num_rays, 3)
                if depth == float("inf"):
                    ray = self.sample_parallel_2D(
                        fov=fov,
                        wvln=self.wvln_rgb[2 - i],
                        num_rays=num_rays,
                        depth=-1.0,
                        plane="sagittal",
                    )
                else:
                    ray = self.sample_point_source_2D(
                        fov=fov,
                        depth=depth,
                        num_rays=num_rays,
                        wvln=self.wvln_rgb[2 - i],
                    )
                    ray.prop_to(-1.0)

            # 将光线追迹到传感器并绘制光线路径
                _, ray_o_record = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.draw_ray_2d(
                    ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                )

            ax.axis("off")

        else:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(lens_title, fontsize=10, fontfamily="Nimbus Sans")
            for i, wvln in enumerate(self.wvln_rgb):
                ax = axs[i]
                ax, fig = self.draw_lens_2d(ax=ax, fig=fig, zmx_format=zmx_format)
                for fov in fov_ls:
                # 采样光线，shape (num_rays, 3)
                    if depth == float("inf"):
                        ray = self.sample_parallel_2D(
                            fov=fov,
                            num_rays=num_rays,
                            wvln=wvln,
                            plane="sagittal",
                        )
                    else:
                        ray = self.sample_point_source_2D(
                            fov=fov,
                            depth=depth,
                            num_rays=num_rays,
                            wvln=wvln,
                        )

                # 将光线追迹到传感器并绘制光线路径
                    ray_out, ray_o_record = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.draw_ray_2d(
                        ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                    )
                    ax.axis("off")

        # 允许内部调用方（如 create_barrier）继续在同一坐标轴上绘制，
        # 而不是在此处保存并关闭图像。
        if return_fig:
            return ax, fig

        if show:
            fig.show()
        else:
            fig.savefig(filename, format="png", dpi=300)
        # 关闭当前图像，避免资源泄漏。
            plt.close(fig)

    def draw_lens_2d(
        self,
        ax=None,
        fig=None,
        color="k",
        linestyle="-",
        zmx_format=False,
        fix_bound=False,
    ):
        """在 2D 图中绘制透镜截面布局。

        渲染各表面轮廓，以边缘线连接透镜元件，并绘制传感器平面。

        参数：
            ax (matplotlib.axes.Axes, optional)：用于绘制的现有坐标轴。为 None
                时创建新图像。默认值为 None。
            fig (matplotlib.figure.Figure, optional)：现有图像。默认值为 None。
            color (str, optional)：透镜轮廓线颜色。默认值为 'k'。
            linestyle (str, optional)：线型。默认值为 '-'。
            zmx_format (bool, optional)：为 True 时绘制符合 Zemax 布局风格的
                阶梯式边缘连接。默认值为 False。
            fix_bound (bool, optional)：为 True 时使用固定坐标范围 [-1,7]x[-4,4]。
                默认值为 False。

        返回：
            ax (matplotlib.axes.Axes)：已绘制透镜布局的坐标轴。
            fig (matplotlib.figure.Figure)：图像。
        """
        # 若未提供 ax，则新建一个。
        if ax is None and fig is None:
            # fig, ax = plt.subplots(figsize=(6, 6))
            fig, ax = plt.subplots()

        # 绘制透镜表面
        for i, s in enumerate(self.surfaces):
            s.draw_widget(ax)

                # 连接两个表面
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.n > 1.1:
                s_prev = self.surfaces[i]
                s = self.surfaces[i + 1]

                r_prev = float(s_prev.draw_r())
                r = float(s.draw_r())
                sag_prev = s_prev.surface_with_offset(
                    r_prev, 0.0, valid_check=False
                ).item()
                sag = s.surface_with_offset(
                    r, 0.0, valid_check=False
                ).item()

                if r_prev >= r:
                        # 前表面更宽：在 r_prev 处沿轴向前进，再沿径向向内
                    z = np.array([sag_prev, sag, sag])
                    x = np.array([r_prev, r_prev, r])
                else:
                        # 后表面更宽：在 z_prev 处沿径向向外，再沿轴向前进
                    z = np.array([sag_prev, sag_prev, sag])
                    x = np.array([r_prev, r, r])

                if not zmx_format:
                    # 非 zmx 模式下，直接以对角线连接两个外边缘
                    z = np.array([z[0], z[-1]])
                    x = np.array([x[0], x[-1]])

                ax.plot(z, -x, color, linewidth=0.75)
                ax.plot(z, x, color, linewidth=0.75)
                s_prev = s

        # 绘制传感器
        ax.plot(
            [self.d_sensor.item(), self.d_sensor.item()],
            [-self.r_sensor, self.r_sensor],
            color,
        )

        # 设置图像尺寸
        if fix_bound:
            ax.set_aspect("equal")
            ax.set_xlim(-1, 7)
            ax.set_ylim(-4, 4)
        else:
            ax.set_aspect("equal", adjustable="datalim", anchor="C")
            ax.minorticks_on()
            ax.set_xlim(-0.5, 7.5)
            ax.set_ylim(-4, 4)
            ax.autoscale()

        return ax, fig

    def draw_ray_2d(self, ray_o_record, ax, fig, color="b"):
        """在现有 2D 布局上绘制光线路径。

        每个记录的光线原点都是 [num_rays, 3]（或 [num_view, num_rays, 3]）
        张量；堆叠后得到 [num_view, num_rays, num_path, 3]，末轴存放
        [mm] 下的 (x, y, z)。以折线绘制 z（轴向）和 x（径向）分量。

        参数：
            ray_o_record (list)：光线原点张量列表，每个被追迹表面对应一个，
                shape 为 [num_rays, 3] 或 [num_view, num_rays, 3]。
            ax (matplotlib.axes.Axes)：用于绘制的 Matplotlib 坐标轴。
            fig (matplotlib.figure.Figure)：Matplotlib 图像。
            color (str, optional)：光线路径颜色。默认值为 'b'。

        返回：
            ax (matplotlib.axes.Axes)：已绘制光线路径的坐标轴。
            fig (matplotlib.figure.Figure)：图像。
        """
        # shape (num_view, num_rays, num_path, 2)
        ray_o_record = torch.stack(ray_o_record, dim=-2).cpu().numpy()
        if ray_o_record.ndim == 3:
            ray_o_record = ray_o_record[None, ...]

        for idx_view in range(ray_o_record.shape[0]):
            for idx_ray in range(ray_o_record.shape[1]):
                ax.plot(
                    ray_o_record[idx_view, idx_ray, :, 2],
                    ray_o_record[idx_view, idx_ray, :, 0],
                    color,
                    linewidth=0.8,
                )

                # ax.scatter(
                #     ray_o_record[idx_view, idx_ray, :, 2],
                #     ray_o_record[idx_view, idx_ray, :, 0],
                #     "b",
                #     marker="x",
                # )

        return ax, fig

    # ====================================================================================
    # 透镜 3D 遮挡结构生成
    # ====================================================================================
    def create_barrier(
        self, filename, barrier_thickness=1.0, ring_height=0.5, ring_size=1.0
    ):
        """在 2D 透镜布局上叠加绘制镜筒（遮挡结构）并保存。

        计算跨越各空气间隔的遮挡段（延伸到后续空气间隔的中点，最后一段延伸到
        传感器），以绿色叠加到 `draw_layout` 生成的布局上，并将图像保存为 PNG。

        参数：
            filename (str)：输出 PNG 图像的保存路径。
            barrier_thickness (float, optional)：遮挡结构厚度 [mm]。默认值为 1.0。
            ring_height (float, optional)：环形圈高度 [mm]。当前未使用
                （尚未实现环形圈绘制）。默认值为 0.5。
            ring_size (float, optional)：环形圈尺寸 [mm]。当前未使用
                （尚未实现环形圈绘制）。默认值为 1.0。
        """
        barriers = []
        rings = []

        # 创建遮挡结构
        barrier_z = 0.0
        barrier_r = 0.0
        barrier_length = 0.0
        for i in range(len(self.surfaces)):
            barrier_r = max(self.surfaces[i].r, barrier_r)

            if self.surfaces[i].mat2.get_name() != "air":
            # 更新遮挡结构半径
                # barrier_r = max(geolens.surfaces[i].r, barrier_r)
                pass
            else:
                # 将遮挡结构延伸至当前表面与下一表面之间空气间隔的中点
                max_curr_surf_d = self.surfaces[i].d.item() + max(
                    self.surfaces[i].surface_sag(0.0, self.surfaces[i].r), 0.0
                )
                if i < len(self.surfaces) - 1:
                    min_next_surf_d = self.surfaces[i + 1].d.item() + min(
                        self.surfaces[i + 1].surface_sag(0.0, self.surfaces[i + 1].r),
                        0.0,
                    )
                    extra_space = (min_next_surf_d - max_curr_surf_d) / 2
                else:
                    min_next_surf_d = self.d_sensor.item()
                    extra_space = min_next_surf_d - max_curr_surf_d

                barrier_length = max_curr_surf_d + extra_space - barrier_z

                # 创建遮挡结构
                barrier = {
                    "pos_z": barrier_z,
                    "pos_r": barrier_r,
                    "length": barrier_length,
                    "thickness": barrier_thickness,
                }
                barriers.append(barrier)

                # 重置遮挡结构参数
                barrier_z = barrier_length + barrier_z
                barrier_r = 0.0
                barrier_length = 0.0

        # # 创建环形圈
        # for i in range(len(geolens.surfaces)):
        #     if geolens.surfaces[i].mat2.get_name() != "air":
        #         ring = {
        #             "pos_z": geolens.surfaces[i].d.item(),

        # 绘制透镜布局（保持图像打开，以便叠加遮挡结构）
        ax, fig = self.draw_layout(filename, return_fig=True)

        # 绘制遮挡结构
        barrier_z_ls = []
        barrier_r_ls = []
        for b in barriers:
            barrier_z_ls.append(b["pos_z"])
            barrier_z_ls.append(b["pos_z"] + b["length"])
            barrier_r_ls.append(b["pos_r"])
            barrier_r_ls.append(b["pos_r"])
        ax.plot(barrier_z_ls, barrier_r_ls, "green", linewidth=1.0)
        ax.plot(barrier_z_ls, [-i for i in barrier_r_ls], "green", linewidth=1.0)

        # 绘制环形圈

        fig.savefig(filename, format="png", dpi=300)
        plt.close()

        pass
