import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import importlib
from typing import Tuple

from solvers.r3l.config import cfg
import torch

from tests.lib import prepare, optimize, render, make_scaffold, build_mapping


def pick_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested in ("cuda", "cpu", "mps"):
        return requested
    return "cpu"


def run(
    case_name: str, 
    device: str = "auto", 
    out_dir: str = str(ROOT / "output/tests/test_optimizer"),
    asset_dir: str = str(ROOT / "data/assets"), 
) -> None:
    dev = pick_device(device)

    case_mod = importlib.import_module(f"tests.cases.{case_name}")
    assets = getattr(case_mod, "assets")
    floor = getattr(case_mod, "floor")
    wall = getattr(case_mod, "wall")
    build = getattr(case_mod, "build")

    asset_info, bbox_vec = prepare(assets, dev, asset_dir)
    asset_to_object, object_to_asset = build_mapping(assets, asset_info)
    scaffold = make_scaffold(assets, asset_info, bbox_vec, floor, asset_to_object, object_to_asset, out_dir)
    build(scaffold, dev)
    layout, frames = optimize(assets, scaffold, dev, cfg.solver.base)
    render(layout, frames, assets, asset_info, floor, wall, out_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="studio")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    
    parser.add_argument("--out", default=str(ROOT / "output/tests/test_optimizer"))
    parser.add_argument("--asset", default=str(ROOT / "data/assets"))
    parser.add_argument("--repeat", default=1, type=int)
    args = parser.parse_args()

    for _ in range(args.repeat):
        run(args.case, args.device, args.out, args.asset)


