"""把七片处方的已有非球面统一扩展到 A4--A16。"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from deeplens import GeoLens
from deeplens.utils import set_seed


def expand(input_lens: str, output_lens: str, device: str = "cpu") -> Path:
    lens = GeoLens(filename=input_lens, device=torch.device(device))
    orders = (4, 6, 8, 10, 12, 14, 16)
    count = 0
    for surface in lens.surfaces:
        if not hasattr(surface, "c") or not hasattr(surface, "k"):
            continue
        existing = [
            getattr(surface, f"ai{order}", torch.zeros_like(surface.c))
            for order in orders
        ]
        surface.ai_degree = len(orders)
        for order, value in zip(orders, existing):
            setattr(surface, f"ai{order}", value.detach().clone())
        surface.ai = torch.stack(
            [getattr(surface, f"ai{order}") for order in orders]
        ).detach().clone()
        count += 1
    if count == 0:
        raise ValueError("输入处方没有可扩展的非球面。")
    lens.post_computation()
    output = Path(output_lens)
    output.parent.mkdir(parents=True, exist_ok=True)
    lens.write_lens_json(str(output))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="扩展七片非球面阶数")
    parser.add_argument("--input-lens", required=True)
    parser.add_argument("--output-lens", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    set_seed(20260766)
    path = expand(args.input_lens, args.output_lens, args.device)
    print(f"已生成高阶非球面处方：{path.resolve()}")


if __name__ == "__main__":
    main()
