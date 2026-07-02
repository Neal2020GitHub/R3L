import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from adapters.protocols import RendererSolution
from renderer.blender_render import render_scene
from solvers.r3l.config import BaseStage, cfg
from solvers.r3l.cluster import SceneIndex
from solvers.r3l.compile import compile
from solvers.r3l.constraints import CompiledConstraints
from solvers.r3l.layout import init_layout
from solvers.r3l.optimize import Optimizer
from tests.utils import get_asset_info
from utils.r3l.plot import visualize_process
from utils.r3l.types import AssetInfo, BBoxVec, LossTerm, ParamTable, PoseVec, get_uid


# An empty relational block (the `cluster_internal` / `scene_relational` shape a
# constraint JSON carries when no relational rules apply). Copy it per use, since
# callers mutate their copy.
EMPTY_RELATIONAL = {
    "facing": [],
    "left_of": [],
    "right_of": [],
    "in_front_of": [],
    "behind_of": [],
    "around": [],
    "align": [],
    "angle": [],
}


def compile_scene(
    objects: List[str],
    constr_json: dict,
    bbox_vec: BBoxVec,
    room_size: Tuple[float, float],
) -> CompiledConstraints:
    """Compile a constraint JSON for a scene whose objects are listed in order."""
    object_to_index = {oid: i for i, oid in enumerate(objects)}
    return compile(constr_json, object_to_index, bbox_vec, room_size)


@dataclass
class CaseScaffold:
    """
    A hand-built case scene: the static data a `CompiledConstraints` needs plus a
    mutable `constraints` dict that each case's `build(scaffold)` fills with
    `builders.make_X(...)` terms. `as_compiled` freezes it into the product (no
    clusters, no Var params) the optimizer consumes directly.
    """
    asset_ids: List[str]
    asset_info: Dict[str, AssetInfo]
    bbox_vec: BBoxVec
    room_size: Tuple[float, float]
    object_to_index: Dict[str, int]
    save_dir: str
    constraints: Dict[str, LossTerm] = field(default_factory=dict)

    def as_compiled(self) -> CompiledConstraints:
        n = len(self.asset_ids)
        scene = SceneIndex(
            N=n,
            independent_indices=torch.arange(n, device=cfg.runtime.device, dtype=torch.long),
            cluster_to_index={},
            clusters={},
            object_to_index=dict(self.object_to_index),
            local_pairs=torch.empty(2, 0, dtype=torch.long, device=cfg.runtime.device),
        )
        return CompiledConstraints(
            scene=scene,
            constraints=dict(self.constraints),
            params=ParamTable(names=[], priors=[], kinds=[]),
            bbox_vec=self.bbox_vec,
            room_size=self.room_size,
            save_dir=self.save_dir,
            spec={},
        )


def prepare(
    assets: List[str],
    device: str,
    asset_dir: str,
) -> Tuple[Dict[str, AssetInfo], BBoxVec]:
    unique_ids = list({get_uid(aid) for aid in assets})
    asset_info = get_asset_info(asset_dir, unique_ids)
    bbox_vec = BBoxVec.build(asset_info, assets, device)
    return asset_info, bbox_vec


def make_scaffold(
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
    bbox_vec: BBoxVec,
    room_size: Tuple[float, float],
    asset_to_object: Dict[str, str],
    object_to_asset: Dict[str, str],
    save_dir: str,
) -> CaseScaffold:
    return CaseScaffold(
        asset_ids=assets,
        asset_info=asset_info,
        bbox_vec=bbox_vec,
        room_size=room_size,
        object_to_index={oid: i for i, oid in enumerate(object_to_asset.keys())},
        save_dir=save_dir,
    )


def build_mapping(
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    asset_to_object = {}
    object_to_asset = {}
    idx = {}
    for aid in assets:
        ainfo = asset_info[get_uid(aid)]
        name = ainfo.name.replace(" ", "-")
        i = idx.get(name, -1) + 1
        idx[name] = i
        oid = f"{name}-{i}"
        asset_to_object[aid] = oid
        object_to_asset[oid] = aid
    return asset_to_object, object_to_asset


def optimize(
    assets: List[str],
    scaffold: CaseScaffold,
    device: str,
    stage: BaseStage,
    train_var: bool = False,
) -> Tuple[PoseVec, List[PoseVec]]:
    pose_dict = init_layout(assets, scaffold.room_size)
    pose_vec = PoseVec.from_pose_dict(pose_dict, device)
    optimizer = Optimizer(device=device)
    return optimizer.optimize(
        constraints=scaffold.as_compiled(),
        poses_vec=pose_vec,
        stage=stage,
        train_var=train_var,
    )


def render(
    layout: PoseVec,
    frames: List[PoseVec],
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
    floor: Tuple[float, float],
    wall: float,
    out_dir: str,
) -> None:
    asset_ids = [get_uid(aid) for aid in assets]
    px, py = layout.x.tolist(), layout.y.tolist()
    pz = [asset_info[aid].bbox['z'] * 0.5 for aid in asset_ids]
    positions = [(px[i], py[i], pz[i]) for i in range(len(px))]
    rz = layout.rz.tolist()
    rotations = [(0., 0., float(rz[i] / math.pi * 180)) for i in range(len(rz))]

    floor_vertices = [
        (0.0, 0.0),
        (floor[0], 0.0),
        (floor[0], floor[1]),
        (0.0, floor[1]),
    ]

    render_sol = RendererSolution(
        asset_ids=asset_ids,
        positions=positions,
        rotations=rotations,
        floor_vertices=floor_vertices,
        wall_height=wall,
    )
    render_scene(render_sol, out_dir)
    visualize_process(frames, assets, asset_info, floor, out_dir)
