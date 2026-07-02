import math
from typing import Dict, List, Tuple, TYPE_CHECKING

from utils.r3l.types import Pose, AssetInfo, get_uid
from adapters.protocols import RendererSolution
from renderer.blender_render import render_scene
from renderer.blender_animate import render_animation
from utils.r3l.plot import visualize_process
from .config import cfg

if TYPE_CHECKING:
    from .pipeline import OptResult
    from .constraints import CompiledConstraints


def _poses_to_solution(poses: Dict[str, Pose], asset_info: Dict[str, AssetInfo],
                       room_size: Tuple[float, float]) -> RendererSolution:
    """Lift a pose dict (asset instance id -> Pose) into a Blender-native
    RendererSolution: x/y from the pose, z = half the asset's bbox height (so the
    object sits on the floor), rotation about Z only, unit scales, rectangular floor."""
    asset_ids, positions, rotations = [], [], []
    for aid, pose in poses.items():
        uid = get_uid(aid)
        # asset_ids carries the INSTANCE id (unique per object, e.g. "...-0" /
        # "...-1"), NOT the UID. build_scene keys each imported mesh's handle by
        # this list; keying by UID would collapse same-model instances (two washing
        # machines sharing one UID) into a single handle, orphaning the first import
        # so it never receives per-frame poses. The UID is still needed to look up
        # bbox, so `get_uid` stays — only what we *append* changes.
        asset_ids.append(aid)
        positions.append((pose.x, pose.y, asset_info[uid].bbox["z"] * 0.5))
        rotations.append((0, 0, pose.rz / math.pi * 180))
    W, H = room_size
    return RendererSolution(
        asset_ids=asset_ids, positions=positions, rotations=rotations,
        floor_vertices=[(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)], wall_height=2.5,
    )


def _render_image(save_dir: str, runs: List["OptResult"],
                   asset_info: Dict[str, AssetInfo], room_size: Tuple[float, float]) -> None:
    """Final static stills -> save_dir/image_top.png + image_side.png (render_scene).
    scene.blend is also written here (render_scene save_blender=True)."""
    if not cfg.render.image.enabled:
        return
    sol = _poses_to_solution(runs[-1].pose_map, asset_info, room_size)
    render_scene(sol, save_dir, save_blender=True, add_coordinate_mark=False,
                 merge_by_distance=True)


def _render_2d(save_dir: str, run: "OptResult", asset_ids: List[str],
               asset_info: Dict[str, AssetInfo], room_size: Tuple[float, float],
               asset_to_object: Dict[str, str], spec: dict) -> None:
    """Save one stage's 2D animation -> save_dir/2d_<tag>.gif."""
    v = cfg.render.animation.view_2d
    if not v.enabled:
        return
    visualize_process(
        frames=run.frames, assets=asset_ids, asset_info=asset_info, room_size=room_size,
        save_dir=save_dir, region_name=f"2d_{run.tag}", asset_to_object=asset_to_object,
        constraints_json=spec, palette=v.palette, fill_alpha=v.fill_alpha,
        collision_alpha=v.collision_alpha, existing_color=v.existing_color,
        no_cluster=cfg.modules.decomposition == "none", fps=cfg.render.animation.fps,
    )


def _render_3d(save_dir: str, run: "OptResult", asset_ids: List[str],
               asset_info: Dict[str, AssetInfo], room_size: Tuple[float, float]) -> None:
    """Render one stage's 3D animation via Blender Cycles -> save_dir/3d_<tag>.<fmt>
    (single camera; the camera is chosen by view_3d.camera, not encoded in the
    filename). Each captured frame PoseVec is lifted to a RendererSolution via
    `_poses_to_solution` (the same lift `_render_image` uses for the final pose),
    then handed to the animation engine."""
    v = cfg.render.animation.view_3d
    if not v.enabled or run.tag not in v.stages:
        return
    solutions = [
        _poses_to_solution(f.to_pose_dict(asset_ids), asset_info, room_size) for f in run.frames
    ]
    render_animation(solutions, save_dir, run.tag, v.camera, v.format,
                     cfg.render.animation.fps)


def render(runs: List["OptResult"], constraints: "CompiledConstraints", save_dir: str,
           asset_ids: List[str], asset_info: Dict[str, AssetInfo],
           room_size: Tuple[float, float], asset_to_object: Dict[str, str]) -> None:
    """Orchestrator: animate every stage (2D when view_2d.enabled; 3D per-stage when
    view_3d.enabled and stage in view_3d.stages); render the final stills once when
    image.enabled. All render config is read here from cfg.render — callers need not
    know the render config shape."""
    for run in runs:
        _render_2d(save_dir, run, asset_ids, asset_info, room_size, asset_to_object,
                   constraints.spec)
        _render_3d(save_dir, run, asset_ids, asset_info, room_size)
    _render_image(save_dir, runs, asset_info, room_size)