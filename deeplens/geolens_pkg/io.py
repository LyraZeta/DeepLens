# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""几何透镜系统的透镜文件 I/O。

提供三种透镜处方格式的读写支持：

- DeepLens 原生 JSON (.json)：`read_lens_json`、`write_lens_json`。
- Zemax 顺序格式 (.zmx)：`read_lens_zmx`、`write_lens_zmx`。
- Code V 顺序格式 (.seq)：`read_lens_seq`、`write_lens_seq`。

除 .zmx/.seq 文件中的视场角使用 degree 外，所有长度均使用
millimetre [mm]，波长均使用 micrometre [µm]。
"""

import json
import math

import torch

from ..geometric_surface import Aperture, Aspheric, Cubic, Plane, Spheric, ThinLens
from ..phase_surface import Binary2Phase, Phase


class GeoLensIO:
    """为 `GeoLens` 提供透镜文件 I/O 的混入类。

    为三种透镜处方格式添加读写方法：DeepLens 原生 JSON、Zemax 顺序格式
    (.zmx) 和 Code V 顺序格式 (.seq)。JSON 是主要且便于阅读的格式，
    带括号的键（如 `"(d_sensor)"`）表示可优化参数。本类不单独实例化，
    而是混入 `GeoLens`；其方法读写宿主透镜的状态（`surfaces`、
    `d_sensor`、`r_sensor`、`enpd`、`rfov_eff` 等）。
    """

    def read_lens_zmx(self, filename="./test.zmx"):
        """从 Zemax .zmx 顺序透镜文件加载透镜。

        解析 STANDARD 和 EVENASPH 表面类型、玻璃材料、视场定义
        （YFLN，单位为 degree）以及入瞳设置（ENPD/FLOA）。填充
        `self.surfaces`、`self.d_sensor` [mm]、`self.r_sensor` [mm]、
        `self.enpd`、`self.float_enpd` 和
        `self.rfov_eff` [rad].

        参数：
            filename (str, optional)：.zmx 文件路径。接受 UTF-8 和
                UTF-16 编码。默认值为 './test.zmx'。

        返回：
            self (GeoLens)：更新后的透镜（便于链式调用）。
        """
        # 读取 .zmx 文件
        try:
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(filename, "r", encoding="utf-16") as file:
                lines = file.readlines()

        # 遍历各行并提取 SURF 字典
        surfs_dict = {}
        current_surf = None
        for line in lines:
            # 去除首尾空白，以保持解析一致
            stripped_line = line.strip()
            
            if stripped_line.startswith("SURF"):
                current_surf = int(stripped_line.split()[1])
                surfs_dict[current_surf] = {}

            elif current_surf is not None and stripped_line != "":
                if len(stripped_line.split(maxsplit=1)) == 1:
                    continue
                else:
                    key, value = stripped_line.split(maxsplit=1)
                    if key == "PARM":
                        new_key = "PARM" + value.split()[0]
                        new_value = value.split()[1]
                        surfs_dict[current_surf][new_key] = new_value
                    else:
                        surfs_dict[current_surf][key] = value

            elif stripped_line.startswith("FLOA") or stripped_line.startswith("ENPD"):
                if stripped_line.startswith("FLOA"):
                    self.float_enpd = True
                    self.enpd = None
                else:
                    self.float_enpd = False
                    self.enpd = float(stripped_line.split()[1])

            elif stripped_line.startswith("YFLN"):
                # 从 YFLN 行解析视场（视场坐标单位为 degree）
                # YFLN 格式：YFLN 0.0 <0.707*rfov_deg> <0.99*rfov_deg>
                parts = stripped_line.split()
                if len(parts) > 1:
                    field_values = [abs(float(x)) for x in parts[1:] if float(x) != 0.0]
                    if field_values:
                        # 最大视场值通常为 0.99 * rfov_deg
                        max_field_deg = max(field_values) / 0.99
                        self.rfov_eff = (
                            max_field_deg * math.pi / 180.0
                        )  # 转换为 radian

        self.float_foclen = False
        self.float_rfov = False
        # 若未从文件解析到 rfov_eff，则设置默认值
        if not hasattr(self, "rfov_eff"):
            self.rfov_eff = None

        # 读取从各 SURF 中提取的数据
        self.surfaces = []
        d = 0.0
        mat1_name = "air"
        for surf_idx, surf_dict in surfs_dict.items():
            if surf_idx > 0 and surf_idx < current_surf:
                # 透镜表面参数
                if "GLAS" in surf_dict:
                    if surf_dict["GLAS"].split()[0] == "___BLANK":
                        mat2_name = f"{surf_dict['GLAS'].split()[3]}/{surf_dict['GLAS'].split()[4]}"
                    else:
                        mat2_name = surf_dict["GLAS"].split()[0].lower()
                else:
                    mat2_name = "air"

                surf_r = (
                    float(surf_dict["DIAM"].split()[0]) if "DIAM" in surf_dict else 1.0
                )
                surf_c = (
                    float(surf_dict["CURV"].split()[0]) if "CURV" in surf_dict else 0.0
                )
                surf_d_next = (
                    float(surf_dict["DISZ"].split()[0]) if "DISZ" in surf_dict else 0.0
                )
                surf_conic = float(surf_dict.get("CONI", 0.0))
                surf_param2 = float(surf_dict.get("PARM2", 0.0))
                surf_param3 = float(surf_dict.get("PARM3", 0.0))
                surf_param4 = float(surf_dict.get("PARM4", 0.0))
                surf_param5 = float(surf_dict.get("PARM5", 0.0))
                surf_param6 = float(surf_dict.get("PARM6", 0.0))
                surf_param7 = float(surf_dict.get("PARM7", 0.0))
                surf_param8 = float(surf_dict.get("PARM8", 0.0))

                # 创建表面对象
                if surf_dict["TYPE"] == "STANDARD":
                    if mat2_name == "air" and mat1_name == "air":
                        # 光阑
                        s = Aperture(r=surf_r, d=d)
                    else:
                        # 球面
                        s = Spheric(c=surf_c, r=surf_r, d=d, mat2=mat2_name)

                elif surf_dict["TYPE"] == "EVENASPH":
                    # 非球面
                    s = Aspheric(
                        c=surf_c,
                        r=surf_r,
                        d=d,
                        ai=[
                            surf_param2,
                            surf_param3,
                            surf_param4,
                            surf_param5,
                            surf_param6,
                            surf_param7,
                            surf_param8,
                        ],
                        k=surf_conic,
                        mat2=mat2_name,
                    )

                else:
                    print(f"Surface type {surf_dict['TYPE']} not implemented.")
                    continue

                self.surfaces.append(s)
                d += surf_d_next
                mat1_name = mat2_name

            elif surf_idx == current_surf:
                # 图像传感器
                self.r_sensor = float(surf_dict["DIAM"].split()[0])

            else:
                pass

        self.d_sensor = torch.tensor(d)
        return self

    def write_lens_zmx(self, filename="./test.zmx"):
        """将透镜写入 Zemax .zmx 顺序透镜文件。

        以 Zemax OpticStudio 格式导出表面（STANDARD 或 EVENASPH）、材料、
        视场定义（有效半 FoV 的 0、0.707 和 0.99 倍处的 YFLN，单位为
        degree）、RGB 波长和入瞳设置，并追加一个图像（传感器）表面。

        参数：
            filename (str, optional)：输出文件路径。默认值为 './test.zmx'。
        """
        lens_zmx_str = ""
        if self.float_enpd:
            enpd_str = "FLOA"
        else:
            enpd_str = f"ENPD {self.enpd}"
        # 文件头字符串。顶层指令写在第 0 列（不随外围 Python 代码块缩进），
        # 从而使生成的 Zemax 文件头不含前导空白；SURF 0 的子关键字保留缩进，
        # 以匹配 ``zmx_str`` 生成的逐表面代码块。
        head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
{enpd_str}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 {0.707 * self.rfov_eff * 57.3} {0.99 * self.rfov_eff * 57.3}
WAVL {self.wvln_rgb[2]:.7f} {self.wvln_rgb[1]:.7f} {self.wvln_rgb[0]:.7f}
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 2
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
    TYPE STANDARD
    CURV 0.0
    DISZ INFINITY
"""
        lens_zmx_str += head_str

        # 表面字符串
        for i, s in enumerate(self.surfaces):
            d_next = (
                self.surfaces[i + 1].d - self.surfaces[i].d
                if i < len(self.surfaces) - 1
                else self.d_sensor - self.surfaces[i].d
            )
            surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
            lens_zmx_str += surf_str

        # 传感器（图像）表面的格式与逐表面 zmx_str 代码块一致：
        # SURF 行位于第 0 列，其子关键字保持缩进。
        sensor_str = f"""SURF {i + 2}
    TYPE STANDARD
    CURV 0.
    DISZ 0.0
    DIAM {self.r_sensor}
"""
        lens_zmx_str += sensor_str

        # 将透镜 zmx 字符串写入文件
        with open(filename, "w") as f:
            f.writelines(lens_zmx_str)
        print(f"Lens written to {filename}")

    # ====================================================================================
    # CODE V 格式 (.seq)
    # ====================================================================================
    def read_lens_seq(self, filename="./test.seq"):
        """从 Code V .seq 顺序文件加载透镜。

        解析标准面与非球面（圆锥常数 K 和多项式系数 A-I，映射到偶次非球面项
        ai[1]-ai[9]）、入瞳直径 (EPD)、视场角（YAN，单位为 degree）、
        孔径光阑 (STO) 和图像面 (SI)。填充 `self.surfaces`、
        `self.d_sensor` [mm]、`self.r_sensor` [mm]、`self.enpd`、`self.hfov`
        [deg] 和 `self.rfov_eff` [rad]。进度会输出到 stdout。

        参数：
            filename (str, optional)：.seq 文件路径。接受 UTF-8 和
                Latin-1 编码。默认值为 './test.seq'。

        返回：
            self (GeoLens)：更新后的透镜（便于链式调用）。
        """
        print(f"\n{'=' * 60}")
        print(f"Start reading CODE V file: {filename}")
        print(f"{'=' * 60}\n")

        # 读取 .seq 文件
        try:
            with open(filename, "r", encoding="utf-8") as file:
                lines = file.readlines()
            print(f"File read successfully (UTF-8)")
        except UnicodeDecodeError:
            try:
                with open(filename, "r", encoding="latin-1") as file:
                    lines = file.readlines()
                print(f"File read successfully (Latin-1)")
            except Exception as e:
                print(f"Failed to read file: {e}")
                return self
        print(f"Total lines: {len(lines)}\n")

        # ============ 步骤 1：解析文件结构 ============
        surfaces = []
        current_surface = {}
        surface_index = 0
        global_diameter = None

        print("Beginning to parse surface data...\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # 跳过无关行
            if not line or line.startswith(
                (
                    "RDM",
                    "TITLE",
                    "UID",
                    "GO",
                    "WL",
                    "XAN",
                    "REF",
                    "WTW",
                    "INI",
                    "WTF",
                    "VUY",
                    "VLY",
                    "DOR",
                    "DIM",
                    "THC",
                )
            ):
                continue
                # 读取入瞳直径
            if line.startswith("EPD"):
                self.enpd = float(line.split()[1])
                self.float_enpd = False
                global_diameter = self.enpd / 2.0
                print(
                    f"[Line {line_num}] EPD={self.enpd} -> default radius={global_diameter}"
                )
                continue
                # 读取视场角
            if line.startswith("YAN"):
                angles = [abs(float(x)) for x in line.split()[1:] if float(x) != 0.0]
                if angles:
                    self.hfov = max(angles)
                    # 同时以 radian 设置 rfov，以便与写入函数保持一致
                    self.rfov_eff = self.hfov * math.pi / 180.0
                    print(f"[Line {line_num}] Max field of view={self.hfov} deg")
                continue
                # 物面
            if line.startswith("SO"):
                parts = line.split()
                thickness = float(parts[2]) if len(parts) > 2 else 1e10

                current_surface = {
                    "type": "OBJECT",
                    "thickness": thickness,
                    "index": surface_index,
                }
                surfaces.append(current_surface)
                print(f"[Line {line_num}] Object surface: T={thickness}")
                surface_index += 1
                current_surface = {}
                continue
                # 标准面
            if line.startswith("S "):
                    # 保存前一个表面
                if current_surface:
                    surfaces.append(current_surface)
                    surface_index += 1

                parts = line.split()
                radius_value = float(parts[1]) if len(parts) > 1 else 0.0
                thickness = float(parts[2]) if len(parts) > 2 else 0.0
                material = parts[3].upper() if len(parts) > 3 else "AIR"

                # 关键：计算曲率 C = 1/R
                if abs(radius_value) > 1e-10:
                    curvature = 1.0 / radius_value
                else:
                    curvature = 0.0

                current_surface = {
                    "type": "STANDARD",
                    "radius": radius_value,
                    "curvature": curvature,
                    "thickness": thickness,
                    "material": material,
                    "index": surface_index,
                    "diameter": global_diameter,
                    "conic": 0.0,
                    "asph_coeffs": {},
                    "is_stop": False,
                }

                print(
                    f"[Line {line_num}] Surface{surface_index}: R={radius_value:.4f} → C={curvature:.6f}, T={thickness}, Mat={material}"
                )
                continue
            # 图像面——暂不追加，等待 CIR
            if line.startswith("SI"):
                if current_surface:
                    surfaces.append(current_surface)
                    surface_index += 1

                parts = line.split()
                thickness = float(parts[1]) if len(parts) > 1 else 0.0

                current_surface = {
                    "type": "IMAGE",
                    "thickness": thickness,
                    "diameter": None,  # 先设为 None，等待 CIR 行更新
                    "index": surface_index,
                }
                print(f"[Line {line_num}] Image surface")
                continue
            # 处理表面属性（CIR、STO、ASP、K、A~J 等）
            if current_surface:
                if line.startswith("CIR"):
                    current_surface["diameter"] = float(
                        line.split()[1].replace(";", "")
                    )
                    print(f"[Line {line_num}]   → CIR={current_surface['diameter']}")

                elif line.startswith("STO"):
                    current_surface["is_stop"] = True
                    print(f"[Line {line_num}]   → Aperture stop flag")

                elif line.startswith("ASP"):
                    current_surface["type"] = "ASPHERIC"
                    print(f"[Line {line_num}]   → Aspheric surface")

                elif line.startswith("K "):
                    current_surface["conic"] = float(line.split()[1].replace(";", ""))
                    print(f"[Line {line_num}]   → K={current_surface['conic']}")

                # 仅提取 A-J 的单字母系数
                elif any(
                    line.startswith(p)
                    for p in [
                        "A ",
                        "B ",
                        "C ",
                        "D ",
                        "E ",
                        "F ",
                        "G ",
                        "H ",
                        "I ",
                        "J ",
                    ]
                ):
                    parts = line.replace(";", "").split()
                    i = 0
                    while i < len(parts) - 1:
                        try:
                            key = parts[i]
                            # 仅接受 A-J 范围内的单个字母
                            if len(key) == 1 and key in [
                                "A",
                                "B",
                                "C",
                                "D",
                                "E",
                                "F",
                                "G",
                                "H",
                                "I",
                                "J",
                            ]:
                                value = float(parts[i + 1])
                                current_surface["asph_coeffs"][key] = value
                                print(f"[Line {line_num}]   → {key}={value}")
                            i += 2
                        except:
                            i += 1

        # 保存最后一个表面
        if current_surface:
            surfaces.append(current_surface)

        print(f"\nParsing complete, total {len(surfaces)} surfaces\n")

        # ============ 步骤 2：创建表面对象 ============
        print(f"{'=' * 60}")
        print("Start creating surface objects:")
        print(f"{'=' * 60}\n")

        self.surfaces = []
        d = 0.0  # 从首个光学表面到当前表面的累计距离
        previous_material = "air"

        for surf in surfaces:
            surf_idx = surf["index"]
            surf_type = surf["type"]

            print(f"{'=' * 50}")
            print(f"Processing surface{surf_idx} ({surf_type}), current d={d:.4f}")

            # 处理物面
            if surf_type == "OBJECT":
                obj_thickness = surf["thickness"]
                if obj_thickness < 1e9:  # 有限物距
                    d += obj_thickness
                    print(
                        f"   Object surface thickness={obj_thickness} → accumulated d={d:.4f}"
                    )
                else:
                    print("   Object surface at infinity")
                previous_material = "air"
                continue

            # 处理图像面
            if surf_type == "IMAGE":
                self.d_sensor = torch.tensor(d)
                # 从 surf 字典读取直径（CIR 值）
                self.r_sensor = (
                    surf.get("diameter") if surf.get("diameter") is not None else 18.0
                )
                print(
                    f"   Image plane position: d_sensor={d:.4f}, r_sensor={self.r_sensor:.4f}"
                )
                break

            # 获取表面参数
            current_material = surf.get("material", "AIR")
            if current_material in ["AIR", "0.0", "", None]:
                current_material = "air"
            else:
                current_material = current_material.lower()

            c = surf.get("curvature", 0.0)
            r = surf.get("diameter", 10.0)
            d_next = surf.get("thickness", 0.0)
            is_stop = surf.get("is_stop", False)

            print(f"   C={c:.6f}, R_aperture={r:.4f}, T={d_next:.4f}")
            print(f"   Material: {previous_material} → {current_material}")
            print(f"   is_stop={is_stop}")

            # 创建表面对象
            try:
                # 情况 1：纯光阑（两侧均为空气，并带 STO 标志）
                if is_stop and current_material == "air" and previous_material == "air":
                    aperture = Aperture(r=r, d=d)
                    self.surfaces.append(aperture)
                    print(f"   Created pure aperture: Aperture(r={r:.4f}, d={d:.4f})")

                # 情况 2：折射面（材料发生变化）
                elif current_material != previous_material:
                    if surf_type == "STANDARD":
                        s = Spheric(c=c, r=r, d=d, mat2=current_material)
                        self.surfaces.append(s)
                        status = " (stop surface)" if is_stop else ""
                        print(
                            f"   Created spherical surface{status}: Spheric(c={c:.6f}, r={r:.4f}, d={d:.4f}, mat2='{current_material}')"
                        )

                    elif surf_type == "ASPHERIC":
                        k = surf.get("conic", 0.0)
                        asph_coeffs = surf.get("asph_coeffs", {})

                        # CODE V 非球面系数映射（向后移一位）：
                        # A → ai[1]（第 2 项，ρ²）
                        # B → ai[2]（第 4 项，ρ⁴）
                        # C → ai[3]（第 6 项，ρ⁶）
                        # D → ai[4]（第 8 项，ρ⁸）
                        # E → ai[5]（第 10 项，ρ¹⁰）
                        # F → ai[6]（第 12 项，ρ¹²）
                        # G → ai[7]（第 14 项，ρ¹⁴）
                        # H → ai[8]（第 16 项，ρ¹⁶）
                        # I → ai[9]（第 18 项，ρ¹⁸）

                        # 初始化 ai 数组（10 个元素）
                        ai = [0.0] * 10
                        ai[0] = 0.0  # ρ⁰ 项（未使用）
                        ai[1] = asph_coeffs.get("A", 0.0)  # ρ²
                        ai[2] = asph_coeffs.get("B", 0.0)  # ρ⁴
                        ai[3] = asph_coeffs.get("C", 0.0)  # ρ⁶
                        ai[4] = asph_coeffs.get("D", 0.0)  # ρ⁸
                        ai[5] = asph_coeffs.get("E", 0.0)  # ρ¹⁰
                        ai[6] = asph_coeffs.get("F", 0.0)  # ρ¹²
                        ai[7] = asph_coeffs.get("G", 0.0)  # ρ¹⁴
                        ai[8] = asph_coeffs.get("H", 0.0)  # ρ¹⁶
                        ai[9] = asph_coeffs.get("I", 0.0)  # ρ¹⁸

                        s = Aspheric(c=c, r=r, d=d, ai=ai, k=k, mat2=current_material)
                        self.surfaces.append(s)
                        status = " (stop surface)" if is_stop else ""
                        print(
                            f"   Created aspheric surface{status}: Aspheric(c={c:.6f}, r={r:.4f}, d={d:.4f}, k={k}, mat2='{current_material}')"
                        )
                        if any(
                            ai[1:]
                        ):  # 若存在非零高阶项（从 ai[1] 开始）
                            print(
                                f"      Aspheric coefficients: A={ai[1]:.2e}, B={ai[2]:.2e}, C={ai[3]:.2e}, D={ai[4]:.2e}"
                            )

                else:
                    print(f"   Skipped (same material on both sides and no stop flag)")

            except Exception as e:
                print(f"   Failed to create surface: {e}")
                import traceback

                traceback.print_exc()

            # 关键：在循环末尾累加距离
            d += d_next
            print(f"   After accumulation: d={d:.4f}")
            previous_material = current_material

        print(f"\n{'=' * 60}")
        print(f"   Done! Created {len(self.surfaces)} objects")
        print(f"   d_sensor={self.d_sensor:.4f}")
        print(f"   r_sensor={self.r_sensor:.4f}")
        print(f"   hfov={self.hfov:.4f}°")
        print(f"{'=' * 60}\n")

        return self

    def write_lens_seq(self, filename="./test.seq"):
        """将透镜写入 Code V .seq 顺序文件。

        以 Code V 格式导出折射面（球面和非球面；跳过纯光阑）、材料、
        视场角（有效半 FoV 的 0、0.707 和 0.99 倍处的 YAN，单位为
        degree）、入瞳直径和图像面。

        参数：
            filename (str, optional)：输出文件路径。默认值为 './test.seq'。

        返回：
            self (GeoLens)：更新后的透镜（便于链式调用）。
        """

        import datetime

        current_date = datetime.datetime.now().strftime("%d-%b-%Y")

        head_str = f"""RDM;LEN       "VERSION: 2023.03       LENS VERSION: 89       Creation Date:  {current_date}"
    TITLE 'Lens Design'
    EPD   {self.enpd}
    DIM   M
    WL    650.0 550.0 480.0
    REF   2
    WTW   1 2 1
    INI   '   '
    XAN   0.0 0.0 0.0
    YAN   0.0  {0.707 * self.rfov_eff * 57.3} {0.99 * self.rfov_eff * 57.3}
    WTF   1.0 1.0 1.0
    VUY   0.0 0.0 0.0
    VLY   0.0 0.0 0.0
    DOR   1.15 1.05
    SO    0.0 0.1e14
    """

        lens_seq_str = head_str
        previous_material = "air"

        for i, surf in enumerate(self.surfaces):
            if i < len(self.surfaces) - 1:
                d_next = self.surfaces[i + 1].d - surf.d
            else:
                d_next = float(self.d_sensor - surf.d)

            current_material = getattr(surf, "mat2", "air")

            if current_material is None or current_material == "air":
                material_str = ""
                material_name = "air"
            elif isinstance(current_material, str):
                material_str = f" {current_material.upper()}"
                material_name = current_material
            else:
                material_name = getattr(current_material, "name", str(current_material))
                material_str = f" {material_name.upper()}"

            is_aperture = surf.__class__.__name__ == "Aperture"

            if is_aperture:
                continue

            is_aspheric = surf.__class__.__name__ == "Aspheric"
            is_stop_surface = getattr(surf, "is_stop", False)

            if is_aspheric:
                if abs(surf.c) > 1e-10:
                    radius = 1.0 / surf.c
                else:
                    radius = 0.0

                k = surf.k if hasattr(surf, "k") else 0.0
                ai = surf.ai if hasattr(surf, "ai") else [0.0] * 10

                surf_str = f"S     {radius} {d_next}{material_str}\n"
                surf_str += f"  CCY 0; THC 0\n"
                surf_str += f"  CIR {surf.r}\n"
                if is_stop_surface:
                    surf_str += f"  STO\n"
                surf_str += f"  ASP\n"
                surf_str += f"  K   {k}\n"

                if len(ai) > 4 and any(ai[1:5]):
                    surf_str += f"  A   {ai[1]:.16e}; B {ai[2]:.16e}; C&\n"
                    surf_str += f"   {ai[3]:.16e}; D {ai[4]:.16e}\n"

                if len(ai) > 8 and any(ai[5:9]):
                    surf_str += f"  E   {ai[5]:.16e}; F {ai[6]:.16e}; G {ai[7]:.16e}; H {ai[8]:.16e}\n"

            else:
                if abs(surf.c) > 1e-10:
                    radius = 1.0 / surf.c
                else:
                    radius = 0.0

                surf_str = f"S     {radius} {d_next}{material_str}\n"
                surf_str += f"  CCY 0; THC 0\n"

                if is_stop_surface:
                    surf_str += f"  STO\n"

                surf_str += f"  CIR {surf.r}\n"

            lens_seq_str += surf_str
            previous_material = material_name

        sensor_str = f"SI    0.0 0.0\n"
        sensor_str += f"  CIR {self.r_sensor}\n"
        lens_seq_str += sensor_str
        lens_seq_str += "GO \n"

        with open(filename, "w") as f:
            f.write(lens_seq_str)

        print(f"Lens written to CODE V file: {filename}")
        return self

    # ====================================================================================
    # JSON 透镜文件 I/O
    # ====================================================================================
    def read_lens_json(self, filename="./test.json"):
        """从 DeepLens 原生 JSON 文件读取透镜。

        加载表面列表、传感器几何形状、入瞳和透镜信息，并依据各表面的
        `type` 字段通过 `init_from_dict` 重建表面。表面位置 `d` [mm]
        由逐表面的 `d_next` 间距累加得到，`self.d_sensor` [mm] 设为总值。
        随后设置 `self.r_sensor` [mm]、`self.enpd` 和 `self.float_enpd`，
        并根据 `sensor_res` 配置传感器分辨率（默认
        2000 x 2000).

        参数：
            filename (str, optional)：JSON 透镜文件路径。默认值为 './test.json'。

        异常：
            Exception：加载器未实现某个表面 `type` 时抛出。

        说明：
            加载后会将透镜移动到 `self.device`。
        """
        self.surfaces = []
        self.materials = []
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 新版原生 JSON 同时保存设计光谱和默认物距。旧文件缺少这些键时
            # 保留构造 GeoLens 时传入的值，确保向后兼容。
            self.primary_wvln = float(
                data.get("primary_wvln", self.primary_wvln)
            )
            self.wvln_rgb = [
                float(value)
                for value in data.get("wvln_rgb", self.wvln_rgb)
            ]
            self.obj_depth = float(data.get("obj_depth", self.obj_depth))
            d = 0.0
            for idx, surf_dict in enumerate(data["surfaces"]):
                surf_dict["d"] = d
                surf_dict["surf_idx"] = idx

                if surf_dict["type"] == "Aperture":
                    s = Aperture.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Aspheric":
                    s = Aspheric.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Cubic":
                    s = Cubic.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "GaussianRBF":
                #     s = GaussianRBF.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "NURBS":
                #     s = NURBS.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Phase":
                    s = Phase.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Binary2Phase":
                    s = Binary2Phase.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Plane":
                    s = Plane.init_from_dict(surf_dict)

                # elif surf_dict["type"] == "PolyEven":
                #     s = PolyEven.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Stop":
                    s = Aperture.init_from_dict(surf_dict)

                elif surf_dict["type"] == "Spheric":
                    s = Spheric.init_from_dict(surf_dict)

                elif surf_dict["type"] == "ThinLens":
                    s = ThinLens.init_from_dict(surf_dict)

                else:
                    raise Exception(
                        f"Surface type {surf_dict['type']} is not implemented in GeoLens.read_lens_json()."
                    )

                s.is_aperture = bool(surf_dict.get("is_aperture", False))
                self.surfaces.append(s)
                d += surf_dict["d_next"]

        self.d_sensor = torch.tensor(d)
        self.lens_info = data.get("info", "None")
        self.enpd = data.get("enpd", None)
        self.float_enpd = True if self.enpd is None else False
        self.float_foclen = False
        self.float_rfov = False
        self.r_sensor = data["r_sensor"]

        self.to(self.device)

        # 设置传感器尺寸和分辨率
        sensor_res = data.get("sensor_res", (2000, 2000))
        self.set_sensor_res(sensor_res=sensor_res)

    def write_lens_json(self, filename="./test.json"):
        """将透镜写入 DeepLens 原生 JSON 文件。

        保存透镜信息、设计波长 [µm]、默认物距 [mm]、焦距 [mm]、F 数、
        入瞳直径、传感器半径/尺寸 [mm] 与分辨率，以及所有表面（各自通过
        `surf_dict`）及其逐表面间距 `d_next` [mm]。几何摘要保留 4 位小数。

        参数：
            filename (str, optional)：输出 JSON 文件路径。默认值为 './test.json'。
        """
        data = {}
        data["info"] = self.lens_info if hasattr(self, "lens_info") else "None"
        data["primary_wvln"] = float(self.primary_wvln)
        data["wvln_rgb"] = [float(value) for value in self.wvln_rgb]
        data["obj_depth"] = float(self.obj_depth)
        data["foclen"] = round(self.foclen, 4)
        data["fnum"] = round(self.fnum, 4)
        if self.float_enpd is False:
            data["enpd"] = round(self.enpd, 4)
        data["r_sensor"] = self.r_sensor
        data["(d_sensor)"] = round(self.d_sensor.item(), 4)
        data["(sensor_size)"] = [round(i, 4) for i in self.sensor_size]
        data["sensor_res"] = list(self.sensor_res)
        data["surfaces"] = []
        for i, s in enumerate(self.surfaces):
            surf_dict = {"idx": i}
            surf_dict.update(s.surf_dict())
            if getattr(s, "is_aperture", False):
                surf_dict["is_aperture"] = True
            if i < len(self.surfaces) - 1:
                surf_dict["d_next"] = round(
                    self.surfaces[i + 1].d.item() - self.surfaces[i].d.item(), 4
                )
            else:
                surf_dict["d_next"] = round(
                    self.d_sensor.item() - self.surfaces[i].d.item(), 4
                )

            data["surfaces"].append(surf_dict)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Lens written to {filename}")
