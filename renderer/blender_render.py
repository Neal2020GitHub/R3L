import numpy as np
import bpy
import os
import math
import bmesh
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Tuple, Optional

from mathutils import Vector
from colorama import Fore

from renderer.blender_utils import *
from utils.asset import download_and_process_asset
from utils.fd import suppress_blender_output
from utils.holodeck_v2.constants import ASSET_BASE_DIR
from utils.r3l.types import get_uid
from adapters.protocols import RendererSolution


def _silent(fn):
    """Run `fn` with bpy's C-level stdout/stderr fd-suppressed. bpy.ops emit "Info:"
    reports straight to the file descriptors throughout a render; the user-facing
    render UI is owned by the CLI layer, so the renderer just produces files quietly.
    Errors still surface — exceptions propagate after the fds are restored."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        with suppress_blender_output():
            return fn(*args, **kwargs)
    return wrapper


def get_visual_marks(floor_vertices, scene, cam, interval=1):
    # The world coordinate we want to project
    visual_marks = dict()
    min_vertices = np.min(floor_vertices, axis=0)
    max_vertices = np.max(floor_vertices, axis=0)
    for min_x in range(math.floor(min_vertices[0]), math.ceil(max_vertices[0])+1, interval):
        for min_y in range(math.floor(min_vertices[1]), math.ceil(max_vertices[1])+1, interval):
            world_coord = Vector((min_x, min_y, 0))
            pixel_x, pixel_y = get_pixel_coordinates(scene, cam, world_coord)
            visual_marks[(min_x, min_y)] = (pixel_x, pixel_y)
    return visual_marks


# Function to create an arrow representing an axis
def create_arrow(start, end, radius=0.02, color=(1, 0, 0, 1), name="Arrow"):
    # Create a cylinder (for the shaft)
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=(end - start).length, location=(start + end) / 2)
    shaft = bpy.context.object
    shaft.name = name + "_shaft"

    # Align the shaft to point towards the end
    direction = end - start
    rot_quat = direction.to_track_quat('Z', 'Y')
    shaft.rotation_euler = rot_quat.to_euler()

    # Create a cone (for the tip)
    tip_length = radius * 2
    bpy.ops.mesh.primitive_cone_add(radius1=radius * 2, depth=tip_length, location=end)
    tip = bpy.context.object
    tip.name = name + "_tip"

    # Align the tip to point towards the end
    tip.rotation_euler = rot_quat.to_euler()

    # Create a material for the arrow
    mat = bpy.data.materials.new(name + "_Material")
    mat.diffuse_color = color
    shaft.data.materials.append(mat)
    tip.data.materials.append(mat)

    # Combine shaft and tip into one object
    bpy.ops.object.select_all(action='DESELECT')
    shaft.select_set(True)
    tip.select_set(True)
    bpy.ops.object.join()


# Function to add a coordinate frame at a specific location
def add_coordinate_frame(location=Vector((0, 0, 0.)), scale=1.0):
    # Define the length and color of each axis
    axis_length = scale
    x_color = (1, 0, 0, 1)  # Red
    y_color = (0, 1, 0, 1)  # Green
    z_color = (0, 0, 1, 1)  # Blue

    # Create the X axis arrow
    create_arrow(location, location + Vector((axis_length, 0, 0)), color=x_color, name="X_Axis")

    # Create the Y axis arrow
    create_arrow(location, location + Vector((0, axis_length, 0)), color=y_color, name="Y_Axis")

    # Create the Z axis arrow
    create_arrow(location, location + Vector((0, 0, axis_length)), color=z_color, name="Z_Axis")


@dataclass
class SceneHandle:
    """Live handles to a built Blender scene, for rendering many frames without
    re-importing assets. ``base_rotations`` captures each object's rotation
    *after* the one-time bakes (recenter / -y->+y flip) but *before* any pose is
    applied, so per-frame poses can be added on top instead of overwriting."""
    floor_center_x: float
    floor_center_y: float
    floor_width: float
    wall_height: float
    floor_vertices: object            # np.ndarray of (x, y, 0) floor corners
    cam: object                       # bpy camera object (TRACK_TO the CameraTarget)
    asset_objects: Dict[str, object]  # uid -> imported bpy object
    base_rotations: Dict[str, Tuple[float, float, float]]  # uid -> baked rotation_euler
    viewpoints: Dict[str, Tuple[float, float, float]]      # view name -> cam.location


def build_scene(
    sol: RendererSolution,
    *,
    views: List[str],
    add_hdri: bool = True,
    high_res: bool = True,
    recenter_mesh: bool = True,
    rotate: bool = True,  # rotate from -y -> +y
    merge_by_distance: bool = False,
    merge_distance: float = 0.0001,
    fov_multiplier: float = 1.1,
    add_coordinate_mark: bool = False,
    adjust_top_angle: Optional[float] = None,
) -> SceneHandle:
    """Build the Blender scene once: reset, floor, import all .glb assets (collecting
    object handles + baked base rotations), lights, render settings, and a camera with
    precomputed viewpoints for each requested view. Does NOT apply any pose — call
    ``update_transforms`` for that."""
    reset_blender()
    setup_background()

    # Scene boundary
    floor_vertices = np.array([(v[0], v[1], 0.0) for v in sol.floor_vertices])
    floor_x_values = [p[0] for p in floor_vertices]
    floor_y_values = [p[1] for p in floor_vertices]
    floor_center_x = (max(floor_x_values) + min(floor_x_values)) / 2
    floor_center_y = (max(floor_y_values) + min(floor_y_values)) / 2
    floor_width = max(max(floor_x_values) - min(floor_x_values), max(floor_y_values) - min(floor_y_values))
    wall_height = sol.wall_height

    # Clear existing mesh objects and lights
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    bpy.context.window.scene = bpy.context.scene
    bpy.context.view_layer.update()

    # Floor (built + UV-unwrapped via the shared create_wall_mesh helper)
    floor_obj = create_wall_mesh("floor", floor_vertices, do_uv=True)
    bpy.context.view_layer.update()

    # Import each asset once, bake the one-time transforms, and keep a handle.
    asset_objects: Dict[str, object] = {}
    base_rotations: Dict[str, Tuple[float, float, float]] = {}
    for asset_id, scale in zip(sol.asset_ids, sol.scales):
        # `asset_id` is the INSTANCE id (unique per object, e.g. "...-0"/"...-1").
        # The on-disk asset directory and .glb filename are keyed by the UID, so
        # derive it here. Object handles below are keyed by `asset_id` (instance id)
        # so same-model instances don't collapse into one handle. `get_uid` on an
        # already-bare UID returns it unchanged, so UID-based callers stay supported.
        uid = get_uid(asset_id)
        asset_dir = os.path.join(ASSET_BASE_DIR, uid)
        objects_before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.join(asset_dir, f"{uid}.glb"))
        bpy.context.view_layer.update()
        loaded = bpy.context.view_layer.objects.active
        bpy.ops.object.select_all(action='DESELECT')
        loaded.select_set(True)

        # merge duplicated vertices (enable if objects render black)
        if merge_by_distance:
            imported_meshes = [obj for obj in bpy.context.scene.objects if obj not in objects_before and obj.type == 'MESH']
            merge_imported_doubles(imported_meshes, merge_distance)

        if recenter_mesh:
            bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
        if rotate:  # -y -> +y
            bpy.ops.transform.rotate(value=-math.radians(180), orient_axis='Z')

        bpy.context.object.rotation_mode = "XYZ"
        loaded.scale = scale
        # Snapshot the baked rotation BEFORE any pose is applied (see update_transforms).
        base_rotations[asset_id] = tuple(loaded.rotation_euler)
        asset_objects[asset_id] = loaded

    if add_coordinate_mark:
        if adjust_top_angle is not None:
            add_coordinate_frame(Vector((floor_center_x - floor_width / 2, floor_center_y - floor_width / 2, 0)))
        else:
            add_coordinate_frame()

    if add_hdri:
        load_hdri()

    set_rendering_settings(high_res=high_res)  # picks the Cycles compute device once

    # One camera (TRACK_TO the room center) + a precomputed viewpoint per requested view.
    cam, _ = setup_camera(floor_center_x, floor_center_y, floor_width, wall_height,
                          fov_multiplier=fov_multiplier, use_damped_track=False)
    original_z = cam.location.z
    viewpoints: Dict[str, Tuple[float, float, float]] = {}
    if "top" in views:
        viewpoints["top"] = (cam.location.x, cam.location.y, cam.location.z)
    if "side" in views:
        viewpoints["side"] = side_viewpoint(floor_center_x, floor_center_y, original_z, 3)

    return SceneHandle(floor_center_x, floor_center_y, floor_width, wall_height,
                       floor_vertices, cam, asset_objects, base_rotations, viewpoints)


def update_transforms(handle: SceneHandle, sol: RendererSolution) -> None:
    """Apply per-frame poses to the already-imported objects (no re-import).

    Mirrors render_scene's placement: location is set directly, and the pose
    rotation (XYZ degrees) is *added* to the baked base rotation, so the one-time
    -y->+y flip is preserved across frames."""
    idx = {uid: i for i, uid in enumerate(sol.asset_ids)}
    for uid, obj in handle.asset_objects.items():
        i = idx[uid]
        px, py, pz = sol.positions[i]
        obj.location = [px, py, pz]
        bx, by, bz = handle.base_rotations[uid]
        rx, ry, rz = sol.rotations[i]
        obj.rotation_euler = (bx + np.deg2rad(rx), by + np.deg2rad(ry), bz + np.deg2rad(rz))


def render_view(handle: SceneHandle, view: str, out_path: str) -> str:
    """Render the current scene from one precomputed viewpoint to ``out_path``."""
    bpy.context.scene.camera = handle.cam
    handle.cam.location = handle.viewpoints[view]
    bpy.context.scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return out_path


@_silent
def render_scene(
    renderer_layout: RendererSolution,
    save_dir,
    add_hdri=True,
    high_res=True,
    add_coordinate_mark=False,
    render_top=True,
    recenter_mesh=True,
    rotate=True,  # rotate from -y -> +y
    fov_multiplier=1.1,
    adjust_top_angle=None,
    side_view_phi=45,
    side_view_indices=[3],
    save_blender=False,  # default is False, set to True for debugging
    merge_by_distance=False,
    merge_distance=0.0001,
):
    """Render a static layout to image_top.png + image_side.png.

    Thin caller over the shared ``build_scene`` / ``update_transforms`` primitives;
    the camera setup and per-render coordinate annotation stay inline to preserve
    the exact output filenames and the adjustable top-down tilt."""
    handle = build_scene(
        renderer_layout, views=[], add_hdri=add_hdri, high_res=high_res,
        recenter_mesh=recenter_mesh, rotate=rotate,
        merge_by_distance=merge_by_distance, merge_distance=merge_distance,
        fov_multiplier=fov_multiplier, add_coordinate_mark=add_coordinate_mark,
        adjust_top_angle=adjust_top_angle,
    )
    update_transforms(handle, renderer_layout)

    output_images = []

    if render_top:
        cam, cam_constraint = setup_camera(
            handle.floor_center_x, handle.floor_center_y, handle.floor_width, handle.wall_height,
            fov_multiplier=fov_multiplier, use_damped_track=(adjust_top_angle is not None)
        )
        if adjust_top_angle is not None:
            cam.rotation_euler = (0, 0, 0)
            original_z = cam.location.z
            # idx=0 → theta=0; same side_viewpoint formula shared with the side views.
            cam.location = side_viewpoint(handle.floor_center_x, handle.floor_center_y,
                                          original_z, 0, adjust_top_angle)

        render_path = f"{save_dir}/image_top.png"
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        if add_coordinate_mark:
            visual_marks = get_visual_marks(handle.floor_vertices, bpy.context.scene, cam, interval=1)
            annotate_image_with_coordinates(image_path=render_path, visual_marks=visual_marks, output_path=render_path)

    cam, cam_constraint = setup_camera(
        handle.floor_center_x, handle.floor_center_y, handle.floor_width, handle.wall_height,
        fov_multiplier=fov_multiplier, use_damped_track=False
    )
    original_z = cam.location.z

    for side_view_index in side_view_indices:
        cam.location = side_viewpoint(handle.floor_center_x, handle.floor_center_y, original_z, side_view_index, side_view_phi)
        render_path = f"{save_dir}/image_side.png"
        bpy.context.scene.render.filepath = render_path
        bpy.ops.render.render(write_still=True)
        if add_coordinate_mark:
            visual_marks = get_visual_marks(handle.floor_vertices, bpy.context.scene, cam, interval=1)
            annotate_image_with_coordinates(image_path=render_path, visual_marks=visual_marks, output_path=render_path)
        output_images.append(render_path)

    if save_blender:
        if os.path.exists(f"{save_dir}/scene.blend"):
            os.remove(f"{save_dir}/scene.blend")
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=f"{save_dir}/scene.blend")

    return output_images
