# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""为 DeepLens 构建随包分发的 refractiveindex.info 材料目录。

这是一个构建阶段使用的转换器，不属于运行时包。它读取公有领域的
refractiveindex.info 数据库（https://github.com/polyanskiy/
refractiveindex.info-database），提取光学玻璃色散公式以及一组精选的
基底晶体，并用目录自身的 nd/Vd 值验证每个条目，最后生成供
deeplens.material.materials 使用的紧凑 JSON 目录
（deeplens/material/refractiveindex_data.json）。

采用离线转换器后，运行时不依赖 PyYAML；随包数据也仍沿用
pyproject.toml 中已经声明的 material/*.json 打包方式。

用法：
    # 1. 获取上游数据库（记录确切提交以便追溯）：
    #    git clone --depth 1 https://github.com/polyanskiy/refractiveindex.info-database
    # 2. 运行转换器：
    python deeplens/material/build_refractiveindex_data.py \
        --db /path/to/refractiveindex.info-database \
        --out deeplens/material/refractiveindex_data.json

色散公式（refractiveindex.info“Dispersion formulas”，2014-06-29）：
    1 Sellmeier (preferred):  n^2 - 1 = C1 + sum_i  C_{2i} l^2 / (l^2 - C_{2i+1}^2)
    2 Sellmeier-2:            n^2 - 1 = C1 + sum_i  C_{2i} l^2 / (l^2 - C_{2i+1})
    3 Polynomial:             n^2     = C1 + sum_i  C_{2i} l^{C_{2i+1}}
其中波长 l 的单位为微米。这里只处理收录范围内实际出现的公式类型
（1、2、3）以及表格形式的 n；如果遇到其他类型，转换器会明确报错，
以确保数据范围与运行时求值器保持同步。
"""

import argparse
import json
import os
import subprocess

import numpy as np

# yaml（PyYAML）仅在构建阶段依赖，并在 build() 内延迟导入，因此仅仅导入
# 这个模块（目前位于运行时包中）并不要求安装 PyYAML。

# 用于 d 线折射率（nd）和阿贝数（Vd）的谱线。
WVLN_D = 0.5875618  # He d 谱线 [um]
WVLN_F = 0.4861327  # H F 谱线  [um]
WVLN_C = 0.6562725  # H C 谱线  [um]

# 要收录的制造商目录（specs/<maker>/optical/*.yml）。这些是标准光学玻璃
# 目录；crystran 还以表格 n 的形式提供基底晶体数据。
GLASS_MAKERS = ["schott", "ohara", "hoya", "hikari", "sumita", "cdgm", "lzos", "crystran"]

# 同一个不带前缀的材料名出现在多个目录中时的决胜顺序（这种情况很少，
# 主要出现在基底晶体中）。排列靠前者优先。
MAKER_PRIORITY = {m: i for i, m in enumerate(GLASS_MAKERS)}

# 从主目录（main/<book>/nk/<page>.yml）精选的基底晶体。
# 这些条目用红外/紫外光学常用的标准宽波段 Sellmeier/Cauchy 拟合补充制造商
# 玻璃数据。每项都来自权威参考页面；预期 n 是文献在指定波长下的值，
# 用作合理性校验基准（这些晶体不提供 nd/Vd）。
SUBSTRATES = [
    # 名称        主目录路径                    别名                  (wvln_um, n_expected, tol)
    ("sio2",      "SiO2/nk/Malitson.yml",      ["fused_silica"],     (0.5875618, 1.4585, 0.002)),
    ("al2o3",     "Al2O3/nk/Malitson-o.yml",   ["sapphire"],         (0.5875618, 1.7681, 0.003)),
    ("mgf2",      "MgF2/nk/Li-o.yml",          [],                   (0.5875618, 1.3777, 0.003)),
    ("caf2",      "CaF2/nk/Malitson.yml",      ["fluorite"],         (0.5875618, 1.4338, 0.002)),
    ("znse",      "ZnSe/nk/Connolly.yml",      [],                   (0.6328000, 2.5934, 0.02)),
    ("si",        "Si/nk/Salzberg.yml",        ["silicon"],          (2.0000000, 3.4487, 0.02)),
    ("ge",        "Ge/nk/Burnett.yml",         ["germanium"],        (4.0000000, 4.0240, 0.05)),
]

SUPPORTED_FORMULAS = {1, 2, 3}


def _eval_formula(formula, coeffs, wvln):
    """计算 refractiveindex.info 色散公式（NumPy 校验基准）。

    这里用 NumPy 复现运行时 Material.ior 的 rii 分支，使转换器无需 Torch
    即可依据目录中的 nd/Vd 进行自校验。
    """
    w = np.asarray(wvln, dtype=np.float64)
    w2 = w * w
    c = coeffs
    if formula == 1:  # Sellmeier（首选）：分母平方
        n2 = 1.0 + c[0]
        for i in range(1, len(c) - 1, 2):
            n2 = n2 + c[i] * w2 / (w2 - c[i + 1] ** 2)
        return np.sqrt(n2)
    if formula == 2:  # Sellmeier-2：分母不平方
        n2 = 1.0 + c[0]
        for i in range(1, len(c) - 1, 2):
            n2 = n2 + c[i] * w2 / (w2 - c[i + 1])
        return np.sqrt(n2)
    if formula == 3:  # 多项式
        n2 = c[0] + np.zeros_like(w)
        for i in range(1, len(c) - 1, 2):
            n2 = n2 + c[i] * w ** c[i + 1]
        return np.sqrt(n2)
    raise ValueError(f"Unsupported formula {formula}")


def _abbe(formula, coeffs):
    """根据色散公式计算 (nd, Vd)。"""
    nd = float(_eval_formula(formula, coeffs, WVLN_D))
    nf = float(_eval_formula(formula, coeffs, WVLN_F))
    nc = float(_eval_formula(formula, coeffs, WVLN_C))
    vd = (nd - 1.0) / (nf - nc) if nf != nc else float("inf")
    return nd, vd


def _parse_data_block(doc):
    """返回 ('formula', num, coeffs, wrange) 或 ('interp', wvlns, n, wrange)。

    优先选取 DATA 中第一个色散公式条目，否则选取第一个 tabulated n 条目。
    如果两者都不存在（例如文件仅含 tabulated k），则返回 None。
    """
    data = doc.get("DATA")
    if not isinstance(data, list):
        return None

    # 优先使用闭式色散公式。
    for entry in data:
        t = str(entry.get("type", ""))
        if t.startswith("formula"):
            num = int(t.split()[1])
            coeffs = [float(x) for x in str(entry["coefficients"]).split()]
            wr = [float(x) for x in str(entry.get("wavelength_range", "")).split()] or None
            return ("formula", num, coeffs, wr)

    # 如果没有公式，则回退到表格折射率（n 或 nk 中的 n 列）。
    for entry in data:
        t = str(entry.get("type", ""))
        if t in ("tabulated n", "tabulated nk"):
            wvlns, ns = [], []
            for line in str(entry["data"]).splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    wvlns.append(float(parts[0]))
                    ns.append(float(parts[1]))
            if len(wvlns) >= 2:
                # 某些上游表格中的采样点顺序混乱。运行时插值
                #（searchsorted / np.interp）要求波长升序，因此先对
                # (wvln, n) 数据对排序，并删除波长完全重复的项。
                pairs = sorted(zip(wvlns, ns), key=lambda p: p[0])
                wvlns, ns, seen = [], [], set()
                for w, nval in pairs:
                    if w in seen:
                        continue
                    seen.add(w)
                    wvlns.append(w)
                    ns.append(nval)
                wr = [wvlns[0], wvlns[-1]]
                return ("interp", wvlns, ns, wr)
    return None


def _get_props(doc):
    """返回 PROPERTIES 中的 (nd, Vd)，不存在时返回 (None, None)。"""
    props = doc.get("PROPERTIES") or {}
    nd = props.get("nd")
    vd = props.get("Vd")
    return (float(nd) if nd is not None else None,
            float(vd) if vd is not None else None)


def _load_agf_names(material_dir):
    """读取 AGF 目录中已有的材料名并转为小写，用于生成覆盖率报告。"""
    names = set()
    for fn in os.listdir(material_dir):
        if not fn.upper().endswith(".AGF"):
            continue
        path = os.path.join(material_dir, fn)
        for enc in ("utf-8", "utf-16"):
            try:
                with open(path, "r", encoding=enc) as f:
                    for line in f:
                        if line.startswith("NM "):
                            names.add(line.split()[1].lower())
                break
            except (UnicodeDecodeError, IndexError):
                continue
    return names


def build(db_root, material_dir):
    """解析数据库并返回 (catalog_dict, report_dict)。"""
    import yaml  # 仅构建阶段依赖；参见模块说明

    data_root = os.path.join(db_root, "database", "data")
    formula_table, interp_table = {}, {}
    provenance = {}  # 名称 -> 制造商，用于同名冲突决胜
    report = {"parsed": 0, "skipped_no_n": 0, "oracle_fail": [], "collisions": []}

    def _consider(name, maker, kind, payload, priority):
        """插入条目，并按 (priority, MAKER_PRIORITY) 解决同名冲突。"""
        prev = provenance.get(name)
        if prev is not None:
            report["collisions"].append((name, prev["maker"], maker))
            # priority 值越小优先级越高；基底材料的 priority 为 -1。
            if (priority, MAKER_PRIORITY.get(maker, 99)) >= (prev["priority"], MAKER_PRIORITY.get(prev["maker"], 99)):
                return
            formula_table.pop(name, None)
            interp_table.pop(name, None)
        provenance[name] = {"maker": maker, "priority": priority}
        if kind == "formula":
            formula_table[name] = payload
        else:
            interp_table[name] = payload

    # --- 制造商光学玻璃目录 -------------------------------------------------
    for maker in GLASS_MAKERS:
        opt_dir = os.path.join(data_root, "specs", maker, "optical")
        if not os.path.isdir(opt_dir):
            continue
        for fn in sorted(os.listdir(opt_dir)):
            if not fn.endswith(".yml"):
                continue
            name = fn[:-4].lower()
            with open(os.path.join(opt_dir, fn), "r", encoding="utf-8") as f:
                doc = yaml.safe_load(f)
            parsed = _parse_data_block(doc)
            if parsed is None:
                report["skipped_no_n"] += 1
                continue
            report["parsed"] += 1
            nd_cat, vd_cat = _get_props(doc)

            if parsed[0] == "formula":
                _, num, coeffs, wr = parsed
                if num not in SUPPORTED_FORMULAS:
                    raise ValueError(f"{maker}/{fn}: unsupported formula {num} in scope")
                nd_calc, vd_calc = _abbe(num, coeffs)
                # 以 nd/Vd 为基准，校验解析、公式选择及单位处理。
                if nd_cat is not None and abs(nd_calc - nd_cat) > 1.5e-3:
                    report["oracle_fail"].append(
                        f"{maker}/{name}: nd calc={nd_calc:.5f} cat={nd_cat:.5f}")
                    continue
                if vd_cat is not None and vd_cat > 0 and abs(vd_calc - vd_cat) / vd_cat > 0.01:
                    report["oracle_fail"].append(
                        f"{maker}/{name}: Vd calc={vd_calc:.3f} cat={vd_cat:.3f}")
                    continue
                payload = {
                    "formula": num,
                    "coeffs": [float(x) for x in coeffs],
                    "wvln_range": wr,
                    "nd": round(nd_cat if nd_cat is not None else nd_calc, 6),
                    "vd": round(vd_cat if vd_cat is not None else vd_calc, 4),
                    "maker": maker,
                }
                _consider(name, maker, "formula", payload, priority=0)
            else:
                _, wvlns, ns, wr = parsed
                payload = {"wvlns": wvlns, "n": ns, "wvln_range": wr, "maker": maker}
                _consider(name, maker, "interp", payload, priority=0)

    # --- 精选基底晶体（主目录） ---------------------------------------------
    for name, rel, aliases, (w_chk, n_chk, tol) in SUBSTRATES:
        path = os.path.join(data_root, "main", rel)
        if not os.path.isfile(path):
            report["oracle_fail"].append(f"substrate {name}: missing {rel}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        parsed = _parse_data_block(doc)
        if parsed is None or parsed[0] != "formula":
            report["oracle_fail"].append(f"substrate {name}: no formula in {rel}")
            continue
        _, num, coeffs, wr = parsed
        if num not in SUPPORTED_FORMULAS:
            raise ValueError(f"substrate {name}: unsupported formula {num}")
        n_got = float(_eval_formula(num, coeffs, w_chk))
        if abs(n_got - n_chk) > tol:
            report["oracle_fail"].append(
                f"substrate {name}: n({w_chk})={n_got:.4f} expected {n_chk:.4f}")
            continue
        # nd/Vd 定义在可见光 He/H 谱线上。对于仅适用于红外的晶体（如 Si、
        # Ge），这些谱线超出拟合有效范围，计算 d 线值会成为无意义的外推。
        # 只有三条谱线均位于有效波段内时才给出真实 nd/Vd；否则使用文献给出的
        # 波段内参考折射率，并将 Vd 标为不适用（1e38，与空气表示“在这些
        # 谱线处无色散”所用的哨兵值相同）。
        wmin, wmax = (wr[0], wr[1]) if wr else (0.0, float("inf"))
        if wmin <= WVLN_F and WVLN_C <= wmax:
            nd_calc, vd_calc = _abbe(num, coeffs)
        else:
            nd_calc, vd_calc = n_got, 1e38
        payload = {
            "formula": num,
            "coeffs": [float(x) for x in coeffs],
            "wvln_range": wr,
            "nd": round(nd_calc, 6),
            "vd": round(vd_calc, 4) if vd_calc < 1e37 else 1e38,
            "maker": "refractiveindex.info (main)",
            "source_page": rel,
        }
        for nm in [name] + aliases:
            _consider(nm, "refractiveindex.info (main)", "formula", payload, priority=-1)

    # --- 来源与提交信息 -----------------------------------------------------
    commit, commit_date = "unknown", "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "-C", db_root, "rev-parse", "HEAD"], text=True).strip()
        commit_date = subprocess.check_output(
            ["git", "-C", db_root, "log", "-1", "--format=%ci"], text=True).strip()
    except Exception:
        pass

    agf_names = _load_agf_names(material_dir)
    all_names = set(formula_table) | set(interp_table)
    report["total"] = len(all_names)
    report["formula"] = len(formula_table)
    report["interp"] = len(interp_table)
    report["new_vs_agf"] = len(all_names - agf_names)
    report["shadowed_by_agf"] = len(all_names & agf_names)

    catalog = {
        "_meta": {
            "source": "refractiveindex.info database",
            "url": "https://github.com/polyanskiy/refractiveindex.info-database",
            "commit": commit,
            "commit_date": commit_date,
            "license": "CC0 1.0 (public domain)",
            "generated_by": "deeplens/material/build_refractiveindex_data.py",
            "scope": (
                "Manufacturer optical-glass catalogs (specs/<maker>/optical) for "
                + ", ".join(GLASS_MAKERS)
                + "; plus curated substrate crystals from the main shelf."
            ),
            "validation": (
                "Each FORMULA entry reproduces the catalog nd within 1.5e-3 and "
                "Vd within 1%. Substrates checked against literature n."
            ),
            "formulas": {
                "1": "Sellmeier (preferred): n^2-1 = C1 + sum C_2i l^2/(l^2 - C_{2i+1}^2)",
                "2": "Sellmeier-2: n^2-1 = C1 + sum C_2i l^2/(l^2 - C_{2i+1})",
                "3": "Polynomial: n^2 = C1 + sum C_2i l^{C_{2i+1}}",
            },
        },
        "FORMULA": dict(sorted(formula_table.items())),
        "INTERP": dict(sorted(interp_table.items())),
    }
    return catalog, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True, help="Path to refractiveindex.info-database clone")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    material_dir = os.path.dirname(os.path.abspath(args.out))
    catalog, report = build(args.db, material_dir)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(catalog, f, separators=(",", ":"), sort_keys=False)
        f.write("\n")

    print("=" * 60)
    print(f"refractiveindex.info -> {args.out}")
    print(f"  upstream commit : {catalog['_meta']['commit'][:12]} ({catalog['_meta']['commit_date']})")
    print(f"  formula entries : {report['formula']}")
    print(f"  interp entries  : {report['interp']}")
    print(f"  total names     : {report['total']}")
    print(f"  new vs AGF      : {report['new_vs_agf']}")
    print(f"  shadowed by AGF : {report['shadowed_by_agf']} (existing names keep precedence)")
    print(f"  skipped (no n)  : {report['skipped_no_n']}")
    print(f"  collisions      : {len(report['collisions'])}")
    print(f"  oracle failures : {len(report['oracle_fail'])}")
    for msg in report["oracle_fail"][:40]:
        print(f"      ! {msg}")
    sz = os.path.getsize(args.out)
    print(f"  output size     : {sz/1024:.1f} KiB")
    print("=" * 60)


if __name__ == "__main__":
    main()
