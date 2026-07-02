"""Blender-Cycles rendering of an optimization trajectory as a single-camera video.

Where ``render_scene`` renders one static layout, this module renders a whole
sequence of poses (the captured optimization frames) by building the scene
*once* and only updating object transforms per frame — so each .glb is imported
exactly once instead of once per frame.
"""
import os
import shutil
from typing import List

from adapters.protocols import RendererSolution
from renderer.blender_render import build_scene, update_transforms, render_view
from utils.fd import suppress_blender_output
from utils.r3l.plot import write_frames


def render_animation(
    solutions: List[RendererSolution],
    save_dir: str,
    tag: str,
    camera: str,
    fmt: str,
    fps: float,
) -> str:
    """Render an optimization trajectory to ``3d_<tag>.<fmt>`` (single camera).

    Builds the scene once from ``solutions[0]`` (importing every .glb once), then
    for each frame applies that frame's poses via ``update_transforms`` and renders
    the requested camera via ``render_view``. Per-frame PNGs are assembled into a
    single GIF or MP4 at ``fps``. Returns the output video path.

    The camera is chosen by the caller (cfg.render.animation.view_3d.camera); it is
    not encoded in the output filename — re-running with a different camera into the
    same save_dir overwrites ``3d_<tag>.<fmt>``.
    """
    assert solutions, "render_animation requires at least one frame"
    os.makedirs(save_dir, exist_ok=True)

    tmp_root = os.path.join(save_dir, f"frames_3d_{tag}")
    os.makedirs(tmp_root, exist_ok=True)

    frame_paths: List[str] = []
    with suppress_blender_output():
        handle = build_scene(solutions[0], views=[camera], merge_by_distance=True)
        for i, sol in enumerate(solutions):
            update_transforms(handle, sol)
            out = os.path.join(tmp_root, f"frame_{i:04d}.png")
            render_view(handle, camera, out)
            frame_paths.append(out)

    out_path = os.path.join(save_dir, f"3d_{tag}.{fmt}")
    write_frames(frame_paths, out_path, fps=fps, fmt=fmt)

    shutil.rmtree(tmp_root, ignore_errors=True)
    return out_path