# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""用于光学镜头的玻璃和塑料材料。"""

import json
import os
import re

import numpy as np
import torch

from ..base import DeepObj


# ===========================================
# 读取 AGF 文件
# ===========================================
def read_agf(file_path):
    """读取 Zemax AGF 玻璃目录并返回材料数据。

    解析 AGF 文件中的 NM（名称/折射率/阿贝数）和 CD（色散系数）记录，
    并按出现顺序将二者配对。

    参数：
        file_path (str): .AGF 目录文件路径。

    返回：
        materials (dict): 从小写材料名到参数字典的映射。字典包含
            calculate_mode、nd、vd 以及六个色散系数
            a_coeff 到 f_coeff（均为浮点数）。

    异常：
        ValueError: 文件无法按 UTF-8 或 UTF-16 解码时抛出。
    """
    encodings = ["utf-8", "utf-16"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                lines = f.readlines()
                break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Error! {file_path} not found.")

    nm_lines = [line for line in lines if re.match(r"^NM\b", line)]
    cd_lines = [line for line in lines if re.match(r"^CD\b", line)]

    materials = {}
    for i in range(len(nm_lines)):
        nm_parts = nm_lines[i].strip().split()
        cd_parts = cd_lines[i].strip().split()

        materials[nm_parts[1].lower()] = {
            "calculate_mode": float(nm_parts[2]),
            "nd": float(nm_parts[4]),
            "vd": float(nm_parts[5]),
            "a_coeff": float(cd_parts[1]),
            "b_coeff": float(cd_parts[2]),
            "c_coeff": float(cd_parts[3]),
            "d_coeff": float(cd_parts[4]),
            "e_coeff": float(cd_parts[5]),
            "f_coeff": float(cd_parts[6]),
        }
    return materials


_dir = os.path.dirname(__file__)
CDGM_data = read_agf(os.path.join(_dir, "CDGM.AGF"))
SCHOTT_data = read_agf(os.path.join(_dir, "SCHOTT.AGF"))
MISC_data = read_agf(os.path.join(_dir, "MISC.AGF"))
PLASTIC_data = read_agf(os.path.join(_dir, "PLASTIC2022.AGF"))
MATERIAL_data = {**MISC_data, **PLASTIC_data, **CDGM_data, **SCHOTT_data}


# ===========================================
# 从 JSON 文件读取自定义材料
# ===========================================
def read_custom_mat(file_path):
    """读取自定义材料 JSON 目录并返回其数据。

    参数：
        file_path (str): 自定义材料 JSON 文件路径。

    返回：
        data (dict): 解析后的 JSON 内容；文件不存在时返回空字典。
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Materials data file not found at {file_path}")
        return {}


CUSTOM_data = read_custom_mat(os.path.join(_dir, "materials_data.json"))

# refractiveindex.info 目录（光学玻璃和基底晶体）由本目录中的
# build_refractiveindex_data.py 离线生成。数据存为 JSON，因此运行时无需依赖
# PyYAML。这些条目仅作回退：AGF 目录或 materials_data.json 中已有的名称仍保持
# 更高优先级（参见 load_dispersion），所以原有行为不变。
RII_data = read_custom_mat(os.path.join(_dir, "refractiveindex_data.json"))
RII_data.setdefault("FORMULA", {})
RII_data.setdefault("INTERP", {})


# ===========================================
# 材料类
# ===========================================
class Material(DeepObj):
    """以波长相关折射率定义的光学材料。

    材料可按名称从随包提供的 CDGM、SCHOTT 或 MISC AGF 目录、自定义 JSON
    目录以及 refractiveindex.info 目录中查找。后者包含来自 SCHOTT、OHARA、
    HOYA、HIKARI、SUMITA、CDGM、LZOS、Crystran 的光学玻璃及常用基底晶体。
    也可以直接用 "n/V" 指定材料（根据阿贝数 V 采用 Cauchy 近似）。
    当多个来源定义同名材料时，按上述顺序解析，因此 refractiveindex.info
    只补充缺失项，不会覆盖已有名称。

    支持的色散模型包括："sellmeier"、"cauchy"、"schott"、"interp"
    （查找表）、"rii"（refractiveindex.info 色散公式）和 "optimizable"
    （n、V 可学习的 Cauchy 模型）。

    属性：
        name (str): 小写材料名。
        device (str): 色散张量所在的计算设备。
        dispersion (str): 当前使用的色散模型（"sellmeier"、"cauchy"、
            "schott"、"interp"、"rii" 或 "optimizable"）。
        n (float or torch.Tensor): d 谱线（587.6 nm）处的折射率；调用
            get_optimizer_params 后会变为可学习张量。
        V (float or torch.Tensor): 阿贝数；在 "optimizable" 模式下同样可学习。
    """

    def __init__(self, name=None, device="cpu"):
        """初始化光学材料。

        参数：
            name (str or None, optional): 材料名（不区分大小写）。可接受形式：

                - 玻璃目录名，例如 "N-BK7"、"H-K9L"
                - "air"（n = 1，无色散）；也接受旧名称 "vacuum" 和
                  "occluder"，并将其规范化为 "air"
                - 内联 Cauchy 表示 "n/V"，例如 "1.5168/64.17"
                - 在 materials_data.json 中注册的自定义名称
                - refractiveindex.info 名称，例如 "s-bsl7"、"sapphire"、
                  "znse"（光学玻璃和基底晶体）

                默认为 None（按 "air" 处理）。
            device (str, optional): 计算设备，默认为 "cpu"。

        异常：
            NotImplementedError: 所有目录中均找不到 name 时抛出。

        示例：
            ```python
            mat = Material("N-BK7")
            n_green = mat.get_ri(0.587)  # 587 nm 处的折射率
            ```
        """
        raw = "air" if name is None else name.lower()
        # 将旧别名规范化为 "air"
        self.name = "air" if raw in ("vacuum", "occluder") else raw
        self.load_dispersion()
        self.device = device

    def get_name(self):
        """返回材料名；若材料可优化，则返回内联 "n/V" 字符串。

        返回：
            name (str): 目录名称；色散模式为 "optimizable" 时，返回根据当前
                (n, V) 格式化的 "{n}/{V}" 字符串。
        """
        if self.dispersion == "optimizable":
            return f"{self.n.item():.4f}/{self.V.item():.2f}"
        else:
            return self.name

    # -------------------------------------------
    # 加载色散方程
    # -------------------------------------------
    def load_dispersion(self):
        """将材料名解析为色散模型及其参数。

        设置 self.dispersion 及对应系数（Sellmeier 的 k 系数和 l 系数、Schott
        的 a 系数、Cauchy 的 A/B 或插值表），同时设置 self.n（d 线折射率）
        和 self.V（阿贝数）。依次在 AGF 目录、内联 "n/V" 形式、自定义 JSON
        表中查找，最后以随包提供的 refractiveindex.info 目录（RII_data）作为
        回退，因而前序来源中的同名材料保持优先。

        异常：
            NotImplementedError: 所有目录中均找不到该材料名时抛出。
            ValueError: 自定义 "interp" 条目的波长表与折射率表长度不一致时抛出。
        """
        # 空气（n=1，无色散）
        if self.name == "air":
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = 0, 0, 0, 0, 0, 0
            self.n, self.V = 1.0, 1e38

        # 在 AGF 文件中找到材料
        elif self.name.lower() in MATERIAL_data:
            self.set_material_param_agf(MATERIAL_data, self.name.lower())

        # 材料由 (n, V) 字符串给出，例如 "1.5168/64.17"
        elif "/" in self.name:
            self.dispersion = "cauchy"
            self.n = float(self.name.split("/")[0])
            self.V = float(self.name.split("/")[1])
            self.A, self.B = self.nV_to_AB(self.n, self.V)

        # 在自定义 JSON 文件中找到材料
        elif self.name in CUSTOM_data["INTERP_TABLE"]:
            self.load_interp_table(CUSTOM_data["INTERP_TABLE"][self.name])

        elif self.name in CUSTOM_data["SELLMEIER_TABLE"]:
            self.dispersion = "sellmeier"
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = CUSTOM_data[
                "SELLMEIER_TABLE"
            ][self.name]
            try:
                self.n = CUSTOM_data["MATERIAL_TABLE"][self.name][0]
                self.V = CUSTOM_data["MATERIAL_TABLE"][self.name][1]
            except KeyError:
                print(f"Warning: {self.name} found in SELLMEIER_TABLE but not in MATERIAL_TABLE.")

        elif self.name in CUSTOM_data["SCHOTT_TABLE"]:
            self.dispersion = "schott"
            self.a0, self.a1, self.a2, self.a3, self.a4, self.a5 = CUSTOM_data[
                "SCHOTT_TABLE"
            ][self.name]
            try:
                self.n = CUSTOM_data["MATERIAL_TABLE"][self.name][0]
                self.V = CUSTOM_data["MATERIAL_TABLE"][self.name][1]
            except KeyError:
                print(f"Warning: {self.name} found in SCHOTT_TABLE but not in MATERIAL_TABLE.")

        elif self.name in CUSTOM_data["MATERIAL_TABLE"]:
            self.dispersion = "cauchy"
            self.n, self.V = CUSTOM_data["MATERIAL_TABLE"][self.name]
            self.A, self.B = self.nV_to_AB(self.n, self.V)

        # refractiveindex.info 目录（仅当上述来源均未定义该名称时才会回退到此处，
        # 因而已有名称保持优先）。
        elif self.name in RII_data["FORMULA"]:
            entry = RII_data["FORMULA"][self.name]
            self.dispersion = "rii"
            self.rii_formula = entry["formula"]
            self.rii_coeffs = entry["coeffs"]
            self.rii_wvln_range = entry.get("wvln_range")
            self.n = entry["nd"]
            self.V = entry["vd"]

        elif self.name in RII_data["INTERP"]:
            self.load_interp_table(RII_data["INTERP"][self.name])

        else:
            raise NotImplementedError(f"Material {self.name} not implemented.")

    def load_interp_table(self, mat_data):
        """根据数据表设置表格折射率（"interp"）材料。

        保存参考波长/折射率数组及其缓存张量，然后从表中采样 self.n
        （d 线折射率）和 self.V（阿贝数）。

        参数：
            mat_data (dict): 包含 "wvlns" 和 "n" 的映射，分别为等长的
                波长 [µm] 与折射率列表。

        异常：
            ValueError: 波长表与折射率表长度不一致时抛出。
        """
        self.dispersion = "interp"
        self.ref_wvlns = mat_data["wvlns"]
        self.ref_n = mat_data["n"]
        if len(self.ref_wvlns) != len(self.ref_n):
            raise ValueError(
                f"Interpolation wavelength and index tables for {self.name} "
                f"have different lengths."
            )
        self._ref_wvlns_t = torch.tensor(self.ref_wvlns)
        self._ref_n_t = torch.tensor(self.ref_n)
        # nd/Vd 定义在可见光 He/H 谱线上。只有表格实际覆盖 F、C 谱线时才给出
        # 二者；否则 np.interp 会钳位到端点，产生无意义的 d 线折射率（例如
        # 仅适用于红外/紫外的晶体）。此时改用波段内参考折射率，并将 Vd 标为
        # 不适用（1e38），与基于公式的处理路径一致。
        wmin, wmax = min(self.ref_wvlns), max(self.ref_wvlns)
        if wmin <= 0.4861 and 0.6563 <= wmax:
            nd = float(np.interp(0.58756, self.ref_wvlns, self.ref_n))
            nF = float(np.interp(0.4861, self.ref_wvlns, self.ref_n))
            nC = float(np.interp(0.6563, self.ref_wvlns, self.ref_n))
            self.n = nd
            self.V = (nd - 1) / (nF - nC) if nF != nC else 1e38
        else:
            self.n = float(np.interp(0.5 * (wmin + wmax), self.ref_wvlns, self.ref_n))
            self.V = 1e38

    def set_material_param_agf(self, material_data, material_name):
        """根据 AGF 目录条目设置色散模型和系数。

        读取 calculate_mode 标志，选择 Schott（模式 1）或 Sellmeier（模式 2）
        模型，填充对应系数，并根据目录中的 nd/vd 字段设置 self.n 和 self.V。

        参数：
            material_data (dict): 解析后的 AGF 目录（名称到参数字典的映射）。
            material_name (str): 要查找的小写材料名。

        异常：
            NotImplementedError: 条目的 calculate_mode 既不是 1 也不是 2
                时抛出。
        """
        if material_name in material_data:
            material = material_data[material_name]

            if material["calculate_mode"] == 1:
                self.dispersion = "schott"
                self.a0 = material["a_coeff"]
                self.a1 = material["b_coeff"]
                self.a2 = material["c_coeff"]
                self.a3 = material["d_coeff"]
                self.a4 = material["e_coeff"]
                self.a5 = material["f_coeff"]
            elif material["calculate_mode"] == 2:
                self.dispersion = "sellmeier"
                self.k1 = material["a_coeff"]
                self.l1 = material["b_coeff"]
                self.k2 = material["c_coeff"]
                self.l2 = material["d_coeff"]
                self.k3 = material["e_coeff"]
                self.l3 = material["f_coeff"]
            else:
                raise NotImplementedError(
                    f"Error: {material_name} calculate_mode {material['calculate_mode']}"
                )

            self.n = material["nd"]
            self.V = material["vd"]
        else:
            print(f"error: not {material_name}")

    def set_sellmeier_param(self, params=None):
        """手动设置自定义材料的六个 Sellmeier 系数。

        将色散模型切换为 "sellmeier"，使后续 ior 调用使用新设置的参数。

        参数：
            params (tuple or list or None, optional): 六个系数
                (k1, l1, k2, l2, k3, l3)，默认为 None（全部置零）。
        """
        # 切换色散模型，使 ior() 使用新设置的参数。
        self.dispersion = "sellmeier"
        if params is None:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        else:
            self.k1, self.l1, self.k2, self.l2, self.k3, self.l3 = params

    # -------------------------------------------
    # 计算折射率
    # -------------------------------------------
    def refractive_index(self, wvln):
        """计算指定波长下的折射率。

        这是对 ior 的轻量封装：输入 Python 浮点数时返回浮点数，否则保持张量
        形式传递。

        参数：
            wvln (float or torch.Tensor): 波长，单位为微米 [µm]。

        返回：
            n (float or torch.Tensor): 折射率。wvln 为浮点数时返回浮点数，
                否则返回与输入形状一致的张量。
        """
        if isinstance(wvln, float):
            wvln = torch.tensor(wvln, device=self.device)
            return self.ior(wvln).item()

        return self.ior(wvln)

    def ior(self, wvln):
        """根据当前色散模型计算折射率。

        根据 self.dispersion 分派到 Sellmeier、Schott、Cauchy、查找表线性插值、
        refractiveindex.info 色散公式（"rii"），或 (n, V) 可学习的
        Cauchy 形式。Cauchy 分支计算 $n = A + B/\\lambda^2$，其中
        $\\lambda$ 的单位为纳米；其他分支直接使用微米单位的 $\\lambda$。

        参数：
            wvln (torch.Tensor): 波长，单位为微米 [µm]，必须位于 (0.1, 10)。

        返回：
            n (torch.Tensor): 折射率，形状与 wvln 相同。

        异常：
            NotImplementedError: self.dispersion 未知时抛出。
        """
        assert wvln.min() > 0.1 and wvln.max() < 10, "Wavelength should be in [um]."

        if self.dispersion == "sellmeier":
            # Sellmeier 方程：https://en.wikipedia.org/wiki/Sellmeier_equation
            n2 = (
                1
                + self.k1 * wvln**2 / (wvln**2 - self.l1)
                + self.k2 * wvln**2 / (wvln**2 - self.l2)
                + self.k3 * wvln**2 / (wvln**2 - self.l3)
            )
            n = torch.sqrt(n2)

        elif self.dispersion == "schott":
            # Schott 方程：https://johnloomis.org/eop501/notes/matlab/sect1/schott.html
            ws = wvln**2
            n2 = (
                self.a0
                + self.a1 * ws
                + (self.a2 + (self.a3 + (self.a4 + self.a5 / ws) / ws) / ws) / ws
            )
            n = torch.sqrt(n2)

        elif self.dispersion == "cauchy":
            # Cauchy 方程：https://en.wikipedia.org/wiki/Cauchy%27s_equation
            n = self.A + self.B / (wvln * 1e3) ** 2

        elif self.dispersion == "interp":
            # 使用缓存张量；必要时移动到正确设备
            if (
                self._ref_wvlns_t.device != wvln.device
                or self._ref_wvlns_t.dtype != wvln.dtype
            ):
                self._ref_wvlns_t = self._ref_wvlns_t.to(
                    device=wvln.device, dtype=wvln.dtype
                )
                self._ref_n_t = self._ref_n_t.to(
                    device=wvln.device, dtype=wvln.dtype
                )
            ref_wvlns = self._ref_wvlns_t
            ref_n = self._ref_n_t

            # 查找上下相邻波长
            i = torch.searchsorted(ref_wvlns, wvln, side="right")
            num_ref_wvlns = len(ref_wvlns)
            idx_low = torch.clamp(i - 1, 0, num_ref_wvlns - 1)
            idx_high = torch.clamp(i, 0, num_ref_wvlns - 1)

            wvln_ref_low = ref_wvlns[idx_low]
            wvln_ref_high = ref_wvlns[idx_high]
            n_ref_low = ref_n[idx_low]
            n_ref_high = ref_n[idx_high]

            # 对 n 进行插值
            denom = wvln_ref_high - wvln_ref_low
            has_interval = denom != 0
            safe_denom = torch.where(has_interval, denom, torch.ones_like(denom))
            weight_high = torch.where(
                has_interval,
                (wvln - wvln_ref_low) / safe_denom,
                torch.zeros_like(wvln),
            )
            weight_low = 1.0 - weight_high
            n = n_ref_low * weight_low + n_ref_high * weight_high

        elif self.dispersion == "rii":
            # refractiveindex.info 色散公式，波长单位为 [µm]。
            # https://refractiveindex.info -> "Dispersion formulas"。系数排列为
            # C1 后接若干（分子，分母/指数）对，与随包提供的
            # refractiveindex_data.json 一致。
            c = self.rii_coeffs
            ws = wvln**2
            if self.rii_formula == 1:
                # Sellmeier（首选）：分母平方。
                n2 = 1.0 + c[0]
                for i in range(1, len(c) - 1, 2):
                    n2 = n2 + c[i] * ws / (ws - c[i + 1] ** 2)
                n = torch.sqrt(n2)
            elif self.rii_formula == 2:
                # Sellmeier-2：分母不平方。
                n2 = 1.0 + c[0]
                for i in range(1, len(c) - 1, 2):
                    n2 = n2 + c[i] * ws / (ws - c[i + 1])
                n = torch.sqrt(n2)
            elif self.rii_formula == 3:
                # 多项式。
                n2 = c[0] + torch.zeros_like(wvln)
                for i in range(1, len(c) - 1, 2):
                    n2 = n2 + c[i] * wvln ** c[i + 1]
                n = torch.sqrt(n2)
            else:
                raise NotImplementedError(
                    f"refractiveindex.info formula {self.rii_formula} not implemented."
                )

        elif self.dispersion == "optimizable":
            # 使用 Cauchy 方程即时计算 (A, B)。除法前将阿贝数钳制在远离零的
            # 范围：无约束的可优化 V 可能趋近于 0，导致 B 和梯度发散
            #（实际材料的阿贝数远大于 1）。
            V_safe = torch.clamp(self.V, min=1.0)
            B = (self.n - 1) / V_safe / (1 / 0.486**2 - 1 / 0.656**2)
            A = self.n - B * 1 / 0.587**2
            n = A + B / wvln**2

        else:
            raise NotImplementedError(f"Error: {self.dispersion} not implemented.")

        return n

    @staticmethod
    def nV_to_AB(n, V):
        """将 (n, V) 转换为 Cauchy 系数 (A, B)。

        给定 d 线折射率和阿贝数，利用 F/d/C 谱线
        （486.1 / 587.6 / 656.3 nm）求解二项 Cauchy 模型
        $n(\\lambda) = A + B/\\lambda^2$ 中的 A 和 B。B 的单位为 nm²，
        与 ior 的 Cauchy 分支一致。

        参数：
            n (float): d 谱线处的折射率。
            V (float): 阿贝数。

        返回：
            A (float): Cauchy 常数项。
            B (float): Cauchy 色散项，单位为 nm²。
        """

        def ivs(a):
            return 1.0 / a**2

        lambdas = [656.3, 587.6, 486.1]
        B = (n - 1) / V / (ivs(lambdas[2]) - ivs(lambdas[0]))
        A = n - B * ivs(lambdas[1])
        return A, B

    # -------------------------------------------
    # 优化并匹配材料
    # -------------------------------------------
    def match_material(self, mat_table=None):
        """将当前材料匹配到目录中最接近的真实玻璃。

        查找归一化 (n, V) 距离最小的目录条目（n 按 0.4 缩放、V 按 40
        缩放），将当前材料重命名为该条目并重新加载其色散。对于空气不执行操作。

        参数：
            mat_table (str or dict or None, optional): 用于匹配的目录。None 或
                "CDGM" 使用 CDGM 常用玻璃，"PLASTIC" 使用塑料表；也可直接
                传入名称到 (n, V) 的字典。默认为 None。

        异常：
            NotImplementedError: mat_table 是无法识别的字符串时抛出。
        """
        if not self.name == "air":
            # 材料匹配表
            if mat_table is None:
                print("No material table provided. Using CDGM common glasses as default.")
                mat_table = CUSTOM_data["CDGM_GLASS"]
            elif mat_table == "CDGM":
                # CDGM 常用玻璃
                mat_table = CUSTOM_data["CDGM_GLASS"]
            elif mat_table == "PLASTIC":
                mat_table = CUSTOM_data["PLASTIC_TABLE"]
            else:
                raise NotImplementedError(f"Material table {mat_table} not implemented.")

            # 查找最接近的材料
            n_range = 0.4 # 折射率范围通常为 [1.5, 1.9]
            V_range = 40.0 # 阿贝数范围通常为 [30, 70]
            n_self = float(self.n) if torch.is_tensor(self.n) else self.n
            V_self = float(self.V) if torch.is_tensor(self.V) else self.V
            self.name = min(
                mat_table,
                key=lambda name: abs(mat_table[name][0] - n_self) / n_range + abs(mat_table[name][1] - V_self) / V_range,
            )

            # 加载新材料参数
            self.load_dispersion()

    def get_optimizer_params(self, lrs=[1e-4, 1e-2]):
        """使 (n, V) 可学习，并返回优化器参数组。

        将 self.n 和 self.V 转换为跟踪梯度的张量，并将色散模型切换为
        "optimizable"。折射率的优化比阿贝数更重要。

        参数：
            lrs (list, optional): n 和 V 的学习率 [lr_n, lr_V]，
                默认为 [1e-4, 1e-2]。

        返回：
            params (list): 两个优化器参数组字典，分别用于 n 和 V，并各自带有
                对应学习率。
        """
        if isinstance(self.n, float):
            self.n = torch.tensor(self.n, device=self.device)
            self.V = torch.tensor(self.V, device=self.device)

        self.n.requires_grad = True
        self.V.requires_grad = True
        self.dispersion = "optimizable"

        params = [
            {"params": [self.n], "lr": lrs[0]},
            {"params": [self.V], "lr": lrs[1]},
        ]
        return params
