from typing import List, Tuple, Dict, Optional, Set, TypedDict
from collections import defaultdict
import os
import math

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import torch
from matplotlib.patches import Polygon as MplPolygon
from html2image import Html2Image

from utils.r3l.types import PoseVec, AssetInfo, Pose, get_uid
from utils.log import print_error, print_good
from utils.r3l.geometry import compute_cluster_obb_global


class ClusterInfo(TypedDict):
    anchor: int
    members: List[int]


# ------------------------------
# OBB corners & heading arrows
# ------------------------------
def _rotated_corners(cx: float, cy: float, w: float, d: float, theta_y_up: float) -> np.ndarray:
    """
    Compute 2D rectangle corners centered at (cx, cy) with width=w (x-axis size)
    and depth=d (y-axis size). The orientation theta is given in radians with
    0 pointing to +y.
    Returns corners as (4, 2) in order.
    """
    # Local corners (axis-aligned at zero rotation)
    local = np.array([
        [-w / 2.0, -d / 2.0],
        [ w / 2.0, -d / 2.0],
        [ w / 2.0,  d / 2.0],
        [-w / 2.0,  d / 2.0],
    ], dtype=float)

    phi = theta_y_up  # rectangle generator is axis-aligned at θ=0, so yaw maps to rotation 1:1
    c, s = math.cos(phi), math.sin(phi)
    R = np.array([[c, -s], [s, c]], dtype=float)
    world = local @ R.T
    world[:, 0] += cx
    world[:, 1] += cy
    return world


def _arrow_dxdy(theta_y_up: float, scale: float = 0.5) -> Tuple[float, float]:
    """Arrow vector with 0 pointing to +y and CCW-positive yaw.

    Heading used throughout R3L (e.g., facing_loss) is h = (-sin θ, cos θ).
    We mirror that here to stay consistent with both optimization and Blender.
    """
    return -math.sin(theta_y_up) * scale, math.cos(theta_y_up) * scale


# ------------------------------
# Polygon area & clipping
# ------------------------------
def _polygon_area(poly: List[Tuple[float, float]]) -> float:
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    return 0.5 * abs(area)


def _clip_convex(subject: List[Tuple[float, float]], clip: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sutherland-Hodgman for convex polygons."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def intersect(p1, p2, a, b):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = a
        x4, y4 = b
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0.0:
            return p2
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        return (px, py)

    output = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        input_list = output
        output = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(intersect(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(intersect(s, e, a, b))
            s = e
    return output


def _obb_polygon(cx: float, cy: float, w: float, d: float, theta: float) -> List[Tuple[float, float]]:
    corners = _rotated_corners(cx, cy, w, d, theta)
    return [(float(c[0]), float(c[1])) for c in corners]


def _draw_polygon(ax, poly: List[Tuple[float, float]], *, edgecolor: str, facecolor: Optional[str], alpha: float, zorder: int, linewidth: float = 2.0):
    patch = MplPolygon(poly, closed=True, edgecolor=edgecolor, facecolor=facecolor if facecolor else 'none', linewidth=linewidth, alpha=alpha, zorder=zorder, clip_on=False)
    ax.add_patch(patch)


# ------------------------------
# Visualization
# ------------------------------
def _setup_axes(room_size: Tuple[float, float]):
    fig_size = (6, 6)
    dpi = 160
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    # Room boundary (CW)
    w, h = room_size
    xs = [0.0, w, w, 0.0, 0.0]
    ys = [0.0, 0.0, h, h, 0.0]
    ax.plot(xs, ys, '-', color='black', linewidth=2)
    ax.set_xlim(0.0, w)
    ax.set_ylim(0.0, h)
    ax.set_aspect('equal', 'box')
    # cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


EXISTING_OBJECT_COLOR = '0.75'


def _label_for_instance(aid: str, info: Dict[str, AssetInfo], asset_to_object: Optional[Dict[str, str]]) -> str:
    if asset_to_object is not None:
        return asset_to_object[aid]
    base = get_uid(aid)
    suffix = aid.split('-')[1] if '-' in aid else '0'
    name = info[base].name if base in info else base
    return f"{name}-{suffix}"


def _dims_for_instance(aid: str, info: Dict[str, AssetInfo]) -> Tuple[float, float]:
    base = get_uid(aid)
    assert base in info, f"Asset info for {base} not found"
    bbox = info[base].bbox
    # x is width, y is depth on floor plane (per tests/test_utils mapping)
    return float(bbox['x']), float(bbox['y'])


def _compute_local_center_offset(
    member_local: Dict[str, Tuple[float, float, float]],
    object_to_asset: Dict[str, str],
    asset_info: Dict[str, AssetInfo],
) -> Tuple[float, float]:
    """
    Compute cluster's local AABB center offset from anchor origin (0,0).

    Member locals are anchor-relative. This returns (cx, cy) of the geometric
    center so we can transform from anchor-local to center-local coords.
    """
    x_min = x_max = y_min = y_max = 0.0
    for oid, (lx, ly, lr) in member_local.items():
        aid = object_to_asset.get(oid)
        if aid is None:
            continue
        bx, by = _dims_for_instance(aid, asset_info)
        c, s = abs(math.cos(lr)), abs(math.sin(lr))
        ex = 0.5 * bx * c + 0.5 * by * s
        ey = 0.5 * bx * s + 0.5 * by * c
        x_min = min(x_min, lx - ex)
        x_max = max(x_max, lx + ex)
        y_min = min(y_min, ly - ey)
        y_max = max(y_max, ly + ey)
    return (x_min + x_max) * 0.5, (y_min + y_max) * 0.5


def _parse_clusters(constraints_json: Optional[dict], assets: List[str], asset_to_object: Optional[Dict[str, str]]) -> Dict[str, ClusterInfo]:
    if not constraints_json or "scene_entities" not in constraints_json:
        return {}
    clusters = constraints_json.get("scene_entities", {}).get("clusters", [])
    if not clusters:
        return {}
    asset_index = {aid: i for i, aid in enumerate(assets)}
    object_to_asset = {v: k for k, v in (asset_to_object or {}).items()}
    out: Dict[str, ClusterInfo] = {}
    for clu in clusters:
        cid = clu.get("cluster_id")
        anchor_obj = clu.get("anchor", {}).get("anchor_object_id")
        members_obj = clu.get("members", [])
        if not cid or not anchor_obj:
            continue
        member_indices: List[int] = []
        for obj_id in members_obj:
            aid = object_to_asset.get(obj_id)
            if aid is None or aid not in asset_index:
                continue
            member_indices.append(asset_index[aid])
        aid_anchor = object_to_asset.get(anchor_obj)
        if aid_anchor is None or aid_anchor not in asset_index:
            continue
        out[cid] = {
            "anchor": asset_index[aid_anchor],
            "members": member_indices,
        }
    return out


def _parse_semantic_groups(
    constraints_json: Optional[dict],
    assets: List[str],
    asset_to_object: Optional[Dict[str, str]],
) -> Dict[str, List[int]]:
    """Extract semantic group membership as {group_id: [member_indices]}.

    Unlike clusters, semantic groups have no anchor—purely organizational.
    Used for coloring objects by group without drawing bounding boxes.
    """
    if not constraints_json:
        return {}
    sem_groups = constraints_json.get("semantic_groups", {})
    if not sem_groups:
        return {}
    asset_index = {aid: i for i, aid in enumerate(assets)}
    object_to_asset = {v: k for k, v in (asset_to_object or {}).items()}
    out: Dict[str, List[int]] = {}
    for gid, members in sem_groups.items():
        indices = []
        for obj_id in members:
            aid = object_to_asset.get(obj_id)
            if aid is not None and aid in asset_index:
                indices.append(asset_index[aid])
        if indices:
            out[gid] = indices
    return out


def _draw_label_arrow(ax, cx: float, cy: float, theta: float, label: str, color: str, w: float, d: float, zorder: int):
    ax.text(cx, cy, label, fontsize=9, ha='center', va='center', color=color, zorder=zorder, clip_on=False)
    dx, dy = _arrow_dxdy(theta, scale=min(w, d) * 0.6)
    ax.arrow(cx, cy, dx, dy, head_width=min(w, d) * 0.2, fc=color, ec=color, clip_on=False, length_includes_head=True, zorder=zorder)


def _compute_cluster_obb(
    cluster: ClusterInfo,
    poses: PoseVec,
    bbox_dims: List[Tuple[float, float]],
):
    anchor_idx = cluster["anchor"]
    member_idx = [i for i in cluster["members"] if i != anchor_idx]
    dev = poses.x.device
    empty = torch.tensor([], device=dev)

    mem_x = poses.x[member_idx] if member_idx else empty  # type: ignore[index]
    mem_y = poses.y[member_idx] if member_idx else empty  # type: ignore[index]
    mem_rz = poses.rz[member_idx] if member_idx else empty  # type: ignore[index]
    mem_bx = torch.tensor([bbox_dims[i][0] for i in member_idx], device=dev) if member_idx else empty
    mem_by = torch.tensor([bbox_dims[i][1] for i in member_idx], device=dev) if member_idx else empty

    cx, cy, cr, cbx, cby = compute_cluster_obb_global(
        poses.x[anchor_idx], poses.y[anchor_idx], poses.rz[anchor_idx],
        torch.tensor(bbox_dims[anchor_idx][0], device=dev),
        torch.tensor(bbox_dims[anchor_idx][1], device=dev),
        mem_x, mem_y, mem_rz, mem_bx, mem_by,
    )
    return float(cx), float(cy), float(cbx), float(cby), float(cr)


def write_frames(frame_paths: List[str], out_path: str, fps: float, fmt: str = "gif"):
    """Assemble frame PNGs into an animated GIF or MP4 at the given playback fps.

    GIF: per-frame delay = 1000/fps ms with ``loop=0`` (forever). imageio's
    ``duration`` is in milliseconds — the old ``duration=0.25`` was 0.25 ms,
    truncated to 0, so playback fell back to the viewer's minimum-delay clamp.
    MP4: libx264 via the imageio-ffmpeg backend; ``macro_block_size=1`` drops the
    16-pixel alignment constraint so non-multiple-of-16 frames (e.g. 1080) encode.
    """
    fmt = fmt.lower()
    match fmt:
        # disposal=2 (Restore to Background) clears the canvas to the background
        # — i.e. the transparency index imageio derives from the alpha channel —
        # before each frame is drawn. Without it, frames composite on top of the
        # previous frame's canvas and the transparent regions show the previous
        # frame through, producing trailing/ghosting (残影). Only meaningful for
        # the transparent-background GIF path; mp4/H.264 has no alpha to ghost.
        case "gif": writer_kw = dict(mode='I', fps=fps, loop=0, disposal=2)
        case "mp4": writer_kw = dict(fps=fps, codec='libx264', macro_block_size=1)
        case _:
            raise ValueError(f"unsupported frame format: {fmt!r} (expected 'gif' or 'mp4')")

    with imageio.get_writer(out_path, **writer_kw) as writer:
        for p in frame_paths:
            img = imageio.imread(p)
            if fmt == "mp4":
                # H.264 has no alpha plane — strip it so libx264 gets plain RGB.
                img = img[:, :, :3]
            # GIF: keep RGBA. Blender renders film_transparent PNGs (empty
            # background alpha=0, since set_rendering_settings defaults
            # indoor_camera=False -> film_transparent=True); imageio's GIF writer
            # folds the alpha channel into a palette transparency index, so the
            # background reads as transparent instead of black. GIF transparency
            # is 1-bit, so anti-aliased edges threshold to opaque/transparent —
            # a format limit, not a code issue. MP4/libx264 cannot do this (no
            # alpha); see the mp4 branch above.
            writer.append_data(img)  # type: ignore[attr-defined]


def _cleanup(paths: List[str]):
    for p in paths:
        try:
            os.remove(p)
        except Exception:
            pass


# ------------------------------
# Cluster helpers
# ------------------------------
def _normalize_cluster_member_locals(
    member_local: Dict[str, Tuple[float, float, float]],
    anchor_oid: str,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Rebase cluster-local poses so that the anchor sits at the local origin
    with zero rotation. This keeps the math consistent even if the cogmap
    recorded a non-zero anchor local pose.
    """
    if anchor_oid not in member_local:
        raise ValueError(f"Anchor '{anchor_oid}' missing in cluster members")

    ax, ay, ar = member_local[anchor_oid]
    c = math.cos(-ar)
    s = math.sin(-ar)

    normalized: Dict[str, Tuple[float, float, float]] = {}
    for oid, (x, y, r) in member_local.items():
        dx = x - ax
        dy = y - ay
        nx = c * dx - s * dy
        ny = s * dx + c * dy
        normalized[oid] = (nx, ny, r - ar)
    return normalized


# ------------------------------
# Wireframe rendering
# ------------------------------
def _default_palette(palette: Optional[List[str]]) -> List[str]:
    if palette and isinstance(palette, list) and palette:
        return palette
    return [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]


def _compute_overlap_polys(entities: List[Tuple[str, List[Tuple[float, float]]]], collision_alpha: float, ax):
    n = len(entities)
    drawn = []
    for i in range(n):
        id_i, poly_i = entities[i]
        for j in range(i + 1, n):
            id_j, poly_j = entities[j]
            inter = _clip_convex(poly_i, poly_j)
            if not inter:
                continue
            area = _polygon_area(inter)
            if area < 1e-5:
                continue
            _draw_polygon(ax, inter, edgecolor="none", facecolor="red", alpha=collision_alpha, zorder=3, linewidth=0.0)
            drawn.append((id_i, id_j))
    return drawn


def visualize_frame(
    frame: PoseVec,
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
    room_size: Tuple[float, float],
    existing_poses: Optional[Dict[str, Pose]] = None,
    existing_assets: Optional[List[str]] = None,
    asset_to_object: Optional[Dict[str, str]] = None,
    constraints_json: Optional[dict] = None,
    palette: Optional[List[str]] = None,
    fill_alpha: Optional[float] = None,
    collision_alpha: Optional[float] = None,
    existing_color: Optional[str] = None,
    no_cluster: bool = False,
    semantic_groups: Optional[Dict[str, List[int]]] = None,
    cluster_override: Optional[Dict[str, Tuple[float, float, float, float, float]]] = None,
) -> Image.Image:
    """Wireframe rendering with cluster/group coloring, palette fills, and collision overlays.

    Clusters get bounding boxes; semantic groups only get shared colors (no OBBs).
    """
    fill_alpha = 0.22 if fill_alpha is None else float(fill_alpha)
    collision_alpha = 0.85 if collision_alpha is None else float(collision_alpha)
    palette = _default_palette(palette)
    existing_color = existing_color or EXISTING_OBJECT_COLOR

    bbox_dims = [_dims_for_instance(aid, asset_info) for aid in assets]
    cluster_map = {} if no_cluster else _parse_clusters(constraints_json, assets, asset_to_object)
    # Auto-parse semantic groups from JSON if not explicitly provided
    sem_group_map = semantic_groups if semantic_groups is not None else _parse_semantic_groups(constraints_json, assets, asset_to_object)
    has_clusters = bool(cluster_map)
    has_groups = has_clusters or bool(sem_group_map)

    # Unified color assignment: clusters first, then semantic groups, then independents
    cluster_ids = sorted(cluster_map.keys())
    sem_group_ids = sorted(sem_group_map.keys())
    colors: Dict[str, str] = {}
    palette_idx = 0
    for cid in cluster_ids:
        colors[cid] = palette[palette_idx % len(palette)]
        palette_idx += 1
    for gid in sem_group_ids:
        colors[gid] = palette[palette_idx % len(palette)]
        palette_idx += 1

    # Build reverse mapping: object_index -> group_id
    # Semantic groups first, clusters override (clusters have spatial meaning)
    member_to_group: Dict[int, str] = {}
    for gid, members in sem_group_map.items():
        for m in members:
            member_to_group[m] = gid
    for cid, info in cluster_map.items():
        for m in info["members"]:
            member_to_group[m] = cid

    independent_indices = [i for i in range(len(assets)) if i not in member_to_group]

    indep_colors: Dict[int, str] = {}
    for idx in independent_indices:
        indep_colors[idx] = palette[palette_idx % len(palette)]
        palette_idx += 1

    fig, ax = _setup_axes(room_size)

    # Build main object polygons
    obj_polys: List[Tuple[int, List[Tuple[float, float]]]] = []
    for i, aid in enumerate(assets):
        w, d = bbox_dims[i]
        poly = _obb_polygon(float(frame.x[i]), float(frame.y[i]), w, d, float(frame.rz[i]))
        obj_polys.append((i, poly))

    # Existing objects (treated as independent)
    existing_polys: List[Tuple[str, List[Tuple[float, float]]]] = []
    ex_pose: Optional[PoseVec] = None
    if existing_poses is not None and existing_assets is not None:
        ex_pose = PoseVec.from_pose_dict(existing_poses, device=str(frame.x.device))
        for i, aid in enumerate(existing_assets):
            w, d = _dims_for_instance(aid, asset_info)
            poly = _obb_polygon(float(ex_pose.x[i]), float(ex_pose.y[i]), w, d, float(ex_pose.rz[i]))
            existing_polys.append((aid, poly))

    # Cluster bbox layer
    cluster_polys: List[Tuple[str, List[Tuple[float, float]]]] = []
    if has_clusters:
        for cid in cluster_ids:
            if cluster_override and cid in cluster_override:
                cx, cy, cbx, cby, cr = cluster_override[cid]
            else:
                cx, cy, cbx, cby, cr = _compute_cluster_obb(cluster_map[cid], frame, bbox_dims)
            poly = _obb_polygon(cx, cy, cbx, cby, cr)
            cluster_polys.append((cid, poly))
            _draw_polygon(ax, poly, edgecolor=colors[cid], facecolor=None, alpha=1.0, zorder=1, linewidth=2.0)
            _draw_label_arrow(ax, cx, cy, cr, cid, colors[cid], cbx, cby, zorder=4)

    # Object layer
    for idx, poly in obj_polys:
        aid = assets[idx]
        w, d = bbox_dims[idx]
        is_member = has_groups and idx in member_to_group
        # Cluster members use cluster OBB label; semantic group members get individual labels
        is_cluster_member = is_member and member_to_group[idx] in cluster_map
        color = colors[member_to_group[idx]] if is_member else indep_colors.get(idx, palette[0])
        face = color
        alpha = fill_alpha
        _draw_polygon(ax, poly, edgecolor=color, facecolor=face, alpha=alpha, zorder=2, linewidth=2.0)
        if not is_cluster_member:
            label = _label_for_instance(aid, asset_info, asset_to_object)
            cx, cy = float(frame.x[idx]), float(frame.y[idx])
            _draw_label_arrow(ax, cx, cy, float(frame.rz[idx]), label, color, w, d, zorder=4)

    # Existing layer
    if ex_pose is not None:
        for i, (aid, poly) in enumerate(existing_polys):
            w, d = _dims_for_instance(aid, asset_info)
            _draw_polygon(ax, poly, edgecolor=existing_color, facecolor=existing_color, alpha=fill_alpha, zorder=2, linewidth=2.0)
            label = _label_for_instance(aid, asset_info, asset_to_object)
            cx, cy = float(ex_pose.x[i]), float(ex_pose.y[i])
            _draw_label_arrow(ax, cx, cy, float(ex_pose.rz[i]), label, existing_color, w, d, zorder=4)

    # Collision layer
    # For spatial clusters: use cluster OBBs as collision entities (members grouped)
    # For semantic groups: treat all objects as independent collision entities
    collision_entities: List[Tuple[str, List[Tuple[float, float]]]] = []
    if has_clusters:
        collision_entities.extend(cluster_polys)
        for idx, poly in obj_polys:
            if idx in member_to_group and member_to_group[idx] in cluster_map:
                continue  # Skip cluster members (represented by cluster OBB)
            collision_entities.append((assets[idx], poly))
        collision_entities.extend(existing_polys)
    else:
        collision_entities.extend([(assets[idx], poly) for idx, poly in obj_polys])
        collision_entities.extend(existing_polys)
    _compute_overlap_polys(collision_entities, collision_alpha, ax)

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())  # type: ignore[attr-defined]
    img = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    plt.close(fig)
    return img


def visualize_process(
    frames: List[PoseVec],
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
    room_size: Tuple[float, float],
    save_dir: str,
    region_name: Optional[str] = None,
    existing_poses: Optional[Dict[str, Pose]] = None,
    existing_assets: Optional[List[str]] = None,
    asset_to_object: Optional[Dict[str, str]] = None,
    constraints_json: Optional[dict] = None,
    palette: Optional[List[str]] = None,
    fill_alpha: Optional[float] = None,
    collision_alpha: Optional[float] = None,
    existing_color: Optional[str] = None,
    no_cluster: bool = False,
    semantic_groups: Optional[Dict[str, List[int]]] = None,
    fps: float = 24.0,
):
    assert len(frames) > 0, "frames must be non-empty"
    os.makedirs(save_dir, exist_ok=True)
    tmp_dir = os.path.join(save_dir, 'frames')
    os.makedirs(tmp_dir, exist_ok=True)

    frame_paths: List[str] = []
    frame_gap = max(1, len(frames) // 100)
    for i, fr in enumerate(frames[::frame_gap]):
        out_path = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        img = visualize_frame(
            frame=fr,
            assets=assets,
            asset_info=asset_info,
            room_size=room_size,
            existing_poses=existing_poses,
            existing_assets=existing_assets,
            asset_to_object=asset_to_object,
            constraints_json=constraints_json,
            palette=palette,
            fill_alpha=fill_alpha,
            collision_alpha=collision_alpha,
            existing_color=existing_color,
            no_cluster=no_cluster,
            semantic_groups=semantic_groups,
        )
        img.save(out_path)
        frame_paths.append(out_path)

    file_name = f"{region_name}.gif" if region_name else "optimization.gif"
    gif_path = os.path.join(save_dir, file_name)
    write_frames(frame_paths, gif_path, fps=fps, fmt="gif")
    _cleanup(frame_paths)
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    return gif_path


def cogmap_to_posevec(
    cogmap_dict: dict,
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
    asset_to_object: Dict[str, str],
    constraints_json: dict,
    device: str = "cpu",
) -> PoseVec:
    """
    Convert cognitive map dict to PoseVec in GLOBAL coordinates.

    All objects (independent and cluster members) are returned in world-space
    coordinates. Cluster member local positions from cogmap are transformed
    to global using the cluster's pose.

    Note: This output is suitable for visualization. `Optimizer.optimize`
    converts it into its reparam frame internally via `constraints.scene.localize`.
    """
    
    independents = cogmap_dict.get("independent", {})
    clusters = cogmap_dict.get("clusters", {}) or {}

    anchor_by_cluster: Dict[str, str] = {}
    for clu in constraints_json.get("scene_entities", {}).get("clusters", []):
        cid = clu.get("cluster_id")
        anchor_obj = clu.get("anchor", {}).get("anchor_object_id")
        if cid and anchor_obj:
            anchor_by_cluster[cid] = anchor_obj

    cluster_pose: Dict[str, Tuple[float, float, float]] = {}
    member_to_cluster: Dict[str, str] = {}
    cluster_member_local: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for cid, data in clusters.items():
        if cid not in anchor_by_cluster:
            raise ValueError(f"Anchor for cluster '{cid}' missing in constraints_json")
        anchor_oid = anchor_by_cluster[cid]
        pose = data["pose"]
        cx, cy, rz = float(pose["x"]), float(pose["y"]), float(pose["rz"])
        rz_rad = rz * math.pi / 180.0
        cluster_pose[cid] = (cx, cy, rz_rad)

        members = data.get("members", {})
        member_local = {}
        for oid, mpose in members.items():
            member_local[oid] = (
                float(mpose["x"]),
                float(mpose["y"]),
                float(mpose["rz"]) * math.pi / 180.0,
            )
            member_to_cluster[oid] = cid
        cluster_member_local[cid] = _normalize_cluster_member_locals(member_local, anchor_oid)

    object_to_asset = {v: k for k, v in asset_to_object.items()}

    cluster_offsets: Dict[str, Tuple[float, float]] = {}
    for cid in clusters:
        cluster_offsets[cid] = _compute_local_center_offset(
            cluster_member_local[cid], object_to_asset, asset_info
        )

    n = len(assets)
    px = torch.zeros(n, dtype=torch.float32, device=device)
    py = torch.zeros(n, dtype=torch.float32, device=device)
    pr = torch.zeros(n, dtype=torch.float32, device=device)

    for i, aid in enumerate(assets):
        oid = asset_to_object[aid]
        if oid in independents:
            pose = independents[oid]
            px[i] = float(pose["x"])
            py[i] = float(pose["y"])
            pr[i] = float(pose["rz"]) * math.pi / 180.0
            continue

        cid = member_to_cluster.get(oid)
        if cid is None:
            raise ValueError(f"Object '{oid}' missing in cogmap (neither independent nor cluster member)")

        cx, cy, cr = cluster_pose[cid]
        mx, my, mr = cluster_member_local[cid][oid]

        off_x, off_y = cluster_offsets[cid]
        mx -= off_x
        my -= off_y

        c = math.cos(cr)
        s = math.sin(cr)
        gx = c * mx - s * my + cx
        gy = s * mx + c * my + cy
        gr = cr + mr
        px[i] = gx
        py[i] = gy
        pr[i] = gr

    return PoseVec(x=px, y=py, rz=pr)


def visualize_cogmap(
    cogmap_dict: dict,
    save_dir: str,
    *,
    assets: List[str],
    asset_info: Dict[str, AssetInfo],
    asset_to_object: Dict[str, str],
    constraints_json: Optional[dict] = None,
    out_name: str = "cogmap.png",
) -> Optional[str]:
    """
    Render a single-frame visualization for a cogmap.
    - Uses cogmap cluster AABB size (lx/ly) but re-computes the AABB center
      from member geometry so the box aligns with the drawn members.
    - Reuses visualize_frame() for drawing.
    """
    room = cogmap_dict.get("room", {})
    length = float(room.get("length", 0.0))
    width = float(room.get("width", 0.0))
    assert length > 0.0 and width > 0.0, "Invalid room size in cogmap"

    clusters = cogmap_dict.get("clusters", {}) or {}
    if clusters and constraints_json is None:
        raise ValueError("visualize_cogmap requires constraints_json when clusters are present")

    pose_vec = cogmap_to_posevec(
        cogmap_dict, assets, asset_info, asset_to_object,
        constraints_json if constraints_json else {},
    )

    cluster_override: Dict[str, Tuple[float, float, float, float, float]] = {}
    if clusters:
        cluster_llm_dims: Dict[str, Tuple[float, float]] = {}
        for cid, data in clusters.items():
            aabb = data["aabb"]
            cluster_llm_dims[cid] = (float(aabb["lx"]), float(aabb["ly"]))

        bbox_dims = [_dims_for_instance(aid, asset_info) for aid in assets]
        cluster_map = _parse_clusters(constraints_json, assets, asset_to_object)
        for cid in clusters.keys():
            if cid not in cluster_map:
                raise ValueError(f"Cluster '{cid}' missing in constraints_json cluster list")
            ccx, ccy, _, _, cr = _compute_cluster_obb(cluster_map[cid], pose_vec, bbox_dims)
            llm_lx, llm_ly = cluster_llm_dims[cid]
            cluster_override[cid] = (ccx, ccy, llm_lx, llm_ly, cr)
    img = visualize_frame(
        frame=pose_vec,
        assets=assets,
        asset_info=asset_info,
        room_size=(length, width),
        asset_to_object=asset_to_object,
        constraints_json=constraints_json,
        cluster_override=cluster_override if clusters else None,
    )

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, out_name)
    img.save(path)
    return path


# ------------------------------
# Scene Graph Visualization (Mermaid)
# ------------------------------

# Canonical label ordering for stable, readable edge labels
_LABEL_ORDER = [
    "face", "angle", "left", "right", "front", "behind", "around",
    "wall", "corner", "horiz", "vert",
]


def _label_sort_key(label: str) -> Tuple[int, str]:
    """Return (priority_index, label) for stable sorting."""
    for i, prefix in enumerate(_LABEL_ORDER):
        if label.startswith(prefix):
            return (i, label)
    return (len(_LABEL_ORDER), label)


def _sanitize_id(raw_id: str) -> str:
    """Convert object/cluster id to valid Mermaid node id (replace - with _)."""
    return raw_id.replace("-", "_")


class _EdgeAccumulator:
    """Accumulate edge labels for (src, tar) pairs, then emit merged edges."""

    def __init__(self):
        self._edges: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    def add(self, src: str, tar: str, label: str) -> None:
        self._edges[(_sanitize_id(src), _sanitize_id(tar))].add(label)

    def emit_lines(self, indent: str = "    ") -> List[str]:
        lines = []
        for (src, tar), labels in sorted(self._edges.items()):
            sorted_labels = sorted(labels, key=_label_sort_key)
            label_str = " + ".join(sorted_labels)
            lines.append(f"{indent}{src} --> {tar}: {label_str}")
        return lines


def _resolve_entity_id(
    kind: str,
    raw_id: str,
    cluster_anchor_map: Dict[str, str],
    cluster_id: Optional[str] = None,
) -> str:
    """
    Resolve (kind, raw_id) to actual entity id.
    - kind="object" -> raw_id
    - kind="cluster" -> raw_id (cluster_id)
    - kind="anchor" (inside cluster_internal) -> cluster_anchor_map[cluster_id]
    """
    if kind == "anchor":
        assert cluster_id is not None
        return cluster_anchor_map[cluster_id]
    return raw_id


def _process_relational_edges(
    rel: dict,
    acc: _EdgeAccumulator,
    cluster_anchor_map: Dict[str, str],
    cluster_id: Optional[str] = None,
) -> None:
    """
    Process relational constraints into edges.
    Handles facing, directionals, around, angle.
    """
    resolve = lambda k, i: _resolve_entity_id(k, i, cluster_anchor_map, cluster_id)

    # facing
    for item in rel.get("facing", []):
        src = resolve(item["src_kind"], item["src_id"])
        tar_kind = item["tar_kind"]
        tar_id = item["tar_id"]
        if tar_kind == "wall":
            # Wall facing → anchor edge with face(WALL_ID)
            acc.add(src, "[*]", f"face({tar_id})")
        else:
            tar = resolve(tar_kind, tar_id)
            acc.add(src, tar, "face")
            if item.get("mutual"):
                acc.add(tar, src, "face")

    # directionals: left_of, right_of, in_front_of, behind_of
    dir_map = {"left_of": "left", "right_of": "right", "in_front_of": "front", "behind_of": "behind"}
    for key, label in dir_map.items():
        for item in rel.get(key, []):
            src = resolve(item["src_kind"], item["src_id"])
            tar = resolve(item["tar_kind"], item["tar_id"])
            acc.add(src, tar, label)

    # around
    for item in rel.get("around", []):
        tar = resolve(item["tar_kind"], item["tar_id"])
        for s in item.get("src", []):
            src = resolve(s["src_kind"], s["src_id"])
            acc.add(src, tar, "around")

    # angle constraints (binary src -> tar)
    for item in rel.get("angle", []):
        src = resolve(item["src_kind"], item["src_id"])
        tar = resolve(item["tar_kind"], item["tar_id"])
        acc.add(src, tar, "angle")


def _process_composition_edges(composition: dict, acc: _EdgeAccumulator) -> None:
    """Process compositional constraints into anchor edges (-> [*])."""
    for item in composition.get("against_wall", []):
        acc.add(item["src_id"], "[*]", f"wall({item['wall']})")

    for item in composition.get("corner", []):
        acc.add(item["src_id"], "[*]", f"corner({item['corner']})")

    for item in composition.get("horizontal", []):
        acc.add(item["src_id"], "[*]", "horiz")

    for item in composition.get("vertical", []):
        acc.add(item["src_id"], "[*]", "vert")


def _build_scene_graph_mermaid(scene_json: dict) -> str:
    """
    Build Mermaid stateDiagram-v2 code from constraint JSON.

    Returns the complete mermaid code string.
    """
    entities = scene_json.get("scene_entities", {})
    constraints = scene_json.get("constraints", {})

    clusters = entities.get("clusters", [])
    composition = constraints.get("composition", {})
    cluster_internal = constraints.get("cluster_internal", {})
    scene_relational = constraints.get("scene_relational", {})

    # Build cluster anchor map: cluster_id -> anchor_object_id
    cluster_anchor_map: Dict[str, str] = {}
    for c in clusters:
        cid = c["cluster_id"]
        cluster_anchor_map[cid] = c["anchor"]["anchor_object_id"]

    lines: List[str] = ["stateDiagram-v2"]

    # 1) Cluster blocks (internal edges)
    for c in clusters:
        cid = c["cluster_id"]
        cid_safe = _sanitize_id(cid)
        internal_rel = cluster_internal.get(cid, {})

        # Internal edge accumulator
        internal_acc = _EdgeAccumulator()
        _process_relational_edges(
            internal_rel, internal_acc, cluster_anchor_map, cluster_id=cid
        )

        internal_lines = internal_acc.emit_lines(indent="        ")
        if internal_lines:
            lines.append(f"    state {cid_safe} {{")
            lines.extend(internal_lines)
            lines.append("    }")
        else:
            # Empty cluster state (just declare it)
            lines.append(f"    state {cid_safe} {{}}")

    # 2) Global edges (composition + scene_relational)
    global_acc = _EdgeAccumulator()
    _process_composition_edges(composition, global_acc)
    _process_relational_edges(
        scene_relational, global_acc, cluster_anchor_map, cluster_id=None
    )

    lines.extend(global_acc.emit_lines(indent="    "))

    return "\n".join(lines)


def visualize_scene_graph(
    scene_json: dict,
    save_dir: str,
    format: str = "svg",
    base_name: str = "scene_graph",
) -> Optional[Path]:
    """
    Visualize constraint graph using Mermaid state diagram.

    Generates stateDiagram-v2 code and renders to PNG via mermaid.ink.
    Clusters are wrapped in composite states.

    Args:
        scene_json: Constraint JSON (from code_to_json or loaded from disk).
        save_dir: Directory to save the scene graph.
        format: Output format, either "svg" or "png".
        base_name: Base filename (without extension).

    Returns:
        Path to saved file, or None if rendering failed.
    """
    assert format in ["svg", "png"]
    os.makedirs(save_dir, exist_ok=True)

    mermaid_code = _build_scene_graph_mermaid(scene_json)
    mmd_path = os.path.join(save_dir, f"{base_name}.mmd")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)

    try:
        from mermaid import Mermaid
        from mermaid.graph import Graph
        graph = Graph(title=base_name, script=mermaid_code)
        renderer = Mermaid(graph)
        path: Path = Path(save_dir) / f"{base_name}.svg"
        renderer.to_svg(str(path))
        if format != "svg":
            path = svg_to_png(save_dir, base_name, delete_original=True)
    except Exception as e:
        print_error(f"[scene_graph] Failed to render scene graph: {e}")
        return None

    print_good(f"[scene_graph] Scene graph saved to {path}")
    return path


def svg_to_png(out_dir: str, file_name: str, scale: float = 2.0, delete_original: bool = True) -> Path:
    """
    Read an SVG file, convert it to a high-resolution PNG (default 4x scale), save, and delete the original file.

    Args:
        out_dir: Directory containing the SVG file.
        file_name: Base name of the file (without extension).
        scale: Scaling factor. Default is 2.0 for high clarity.
        delete_original: Whether to delete the original SVG file.

    Returns:
        Path object to the generated PNG file.
    """
    assert 3.0 >= scale > 0.0
    out_path = Path(out_dir)
    svg_path = out_path / f"{file_name}.svg"
    png_path = out_path / f"{file_name}.png"
    hti = Html2Image(
        browser='chrome',
        output_path=str(out_path),
        custom_flags=['--no-sandbox', '--disable-gpu'],
    )
    hti.screenshot(
        other_file=str(svg_path),
        save_as=f"{file_name}.png",
        size=(int(1920 * scale), int(1080 * scale)),
    )
    if delete_original and png_path.exists():
        os.remove(svg_path)
    return png_path