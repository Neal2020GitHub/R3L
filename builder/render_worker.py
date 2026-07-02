#!/usr/bin/env python3
"""
Blender render worker for the Scene Builder.

Runs under the repo's venv (bpy==4.0.0 pip-installed — the same bpy the solver
renderer uses) and imports the shared renderer modules. The builder app
(builder/app.py start_render) launches it as `python render_worker.py
input.json output_dir` only after the cheap per-asset GLB-existence /
conversion-status checks pass, so a misconfigured render fails fast before this
process is spawned.

Usage: python render_worker.py input.json output_dir

Input JSON schema:
{
    "room_size": [width, height],
    "wall_height": 2.5,
    "asset_dir": "/absolute/path/to/assets",
    "items": [
        {"asset_id": "xxx", "x": 1.0, "y": 2.0, "rotation": 0.0, "bbox_y": 0.5}
    ],
    "high_res": false
}
"""
import bpy
import json
import math
import os
import sys

# Launched as `python builder/render_worker.py ...`, sys.path[0] is builder/,
# not the repo root — insert the repo root so the shared modules import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.fd import suppress_blender_output
from renderer.cycles_device import enable_gpu_backend
from renderer.blender_utils import (
    reset_blender,
    setup_background,
    create_wall_mesh,
    setup_camera,
    merge_imported_doubles,
    side_viewpoint,
)


# ============================================================================
# FLOOR MESH CREATION
# ============================================================================

def create_floor_mesh(vertices):
    """Create the floor polygon from vertices and UV-unwrap it. Thin wrapper over
    the shared ``create_wall_mesh(..., do_uv=True)`` — the unwrap is the same
    smart_project pass ``build_scene`` uses for its floor, so it is shared, not
    builder-specific."""
    return create_wall_mesh("floor", vertices, do_uv=True)


# ============================================================================
# LIGHTING SETUP
# Uses area lights instead of HDRI to avoid external file dependency.
# Three-point lighting: key, fill, rim.
# ============================================================================

def setup_lighting(room_center_x, room_center_y, room_size):
    """
    Setup studio-style three-point lighting.

    Positions lights relative to room center for consistent results
    regardless of room size.
    """
    scale = max(room_size) / 5.0  # Normalize to 5m reference room

    # Key light (main light, creates primary shadows)
    bpy.ops.object.light_add(
        type='AREA',
        location=(room_center_x + 5 * scale, room_center_y - 5 * scale, 8 * scale)
    )
    key = bpy.context.object
    key.name = "KeyLight"
    key.data.energy = 500 * (scale ** 2)
    key.data.size = 5 * scale
    key.rotation_euler = (math.radians(45), 0, math.radians(45))

    # Fill light (softens shadows, opposite side)
    bpy.ops.object.light_add(
        type='AREA',
        location=(room_center_x - 5 * scale, room_center_y + 5 * scale, 6 * scale)
    )
    fill = bpy.context.object
    fill.name = "FillLight"
    fill.data.energy = 200 * (scale ** 2)
    fill.data.size = 8 * scale
    fill.rotation_euler = (math.radians(60), 0, math.radians(-135))

    # Rim light (back light, creates edge definition)
    bpy.ops.object.light_add(
        type='AREA',
        location=(room_center_x, room_center_y + 8 * scale, 4 * scale)
    )
    rim = bpy.context.object
    rim.name = "RimLight"
    rim.data.energy = 150 * (scale ** 2)
    rim.data.size = 4 * scale
    rim.rotation_euler = (math.radians(70), 0, math.radians(180))


# ============================================================================
# RENDER SETTINGS
# ============================================================================

def set_render_settings(high_res=False, device_mode="GPU"):
    """
    Configure Cycles render engine.

    Args:
        high_res: If True, use 1080x1080 @ 128 samples. Else 720x720 @ 64 samples.
        device_mode: "GPU"/"CPU" from enable_gpu_backend() — computed by the
            caller OUTSIDE suppress_blender_output so its fallback notice (on
            CPU) is visible, not fd-suppressed.
    """
    render = bpy.context.scene.render
    render.engine = "CYCLES"

    bpy.context.scene.cycles.device = device_mode

    cycles = bpy.context.scene.cycles
    cycles.samples = 128 if high_res else 64
    cycles.use_denoising = True
    cycles.diffuse_bounces = 3
    cycles.glossy_bounces = 3
    cycles.transparent_max_bounces = 5
    cycles.transmission_bounces = 5

    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 1080 if high_res else 720
    render.resolution_y = 1080 if high_res else 720
    render.resolution_percentage = 100
    render.film_transparent = True


# ============================================================================
# MAIN RENDER FUNCTION
# (camera setup uses the shared renderer.blender_utils.setup_camera, which
#  returns (cam, cam_constraint); the call is inlined below.)
# ============================================================================

def render_scene(input_data, output_dir):
    """
    Main render function. Sets up scene and renders top-down + side views.

    Args:
        input_data: Parsed JSON dict with room_size, items, asset_dir, etc.
        output_dir: Directory to write output PNG files
    """
    room_w, room_h = input_data["room_size"]
    wall_height = input_data.get("wall_height", 2.5)
    asset_dir = input_data["asset_dir"]
    items = input_data["items"]
    high_res = input_data.get("high_res", False)

    print(f"[WORKER] Room: {room_w}x{room_h}m, {len(items)} items")

    # ===== Scene Setup =====
    with suppress_blender_output():
        reset_blender()
        setup_background()

    # Clear existing mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    bpy.context.view_layer.update()

    # ===== Create Floor =====
    if not input_data.get("disable_floor_plane", False):
        floor_vertices = [
            (0.0, 0.0, 0.0),
            (float(room_w), 0.0, 0.0),
            (float(room_w), float(room_h), 0.0),
            (0.0, float(room_h), 0.0),
        ]
        with suppress_blender_output():
            create_floor_mesh(floor_vertices)
        bpy.context.view_layer.update()

    # ===== Import and Place Assets =====
    merge_by_distance = input_data.get("merge_by_distance", True)  # Fix black objects
    merge_distance = input_data.get("merge_distance", 0.0001)

    for item in items:
        asset_id = item["asset_id"]
        glb_path = os.path.join(asset_dir, asset_id, f"{asset_id}.glb")

        if not os.path.exists(glb_path):
            print(f"[WORKER] WARNING: GLB not found, skipping: {glb_path}")
            continue

        # Track objects before import to identify newly added meshes
        objects_before = set(bpy.context.scene.objects)

        with suppress_blender_output():
            bpy.ops.import_scene.gltf(filepath=glb_path)

        bpy.context.view_layer.update()
        obj = bpy.context.view_layer.objects.active

        if obj is None:
            print(f"[WORKER] WARNING: No object imported for {asset_id}")
            continue

        # Merge duplicated vertices on all imported mesh objects
        # This fixes "black objects" caused by duplicate vertices
        if merge_by_distance:
            imported_meshes = [o for o in bpy.context.scene.objects if o not in objects_before and o.type == 'MESH']
            merge_imported_doubles(imported_meshes, merge_distance)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)

        # Recenter mesh origin to bounds center
        bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

        # Rotate 180° around Z axis (GLB convention: -Y forward → +Y forward)
        bpy.ops.transform.rotate(value=-math.radians(180), orient_axis='Z')

        # Set transform
        obj.scale = (1.0, 1.0, 1.0)
        obj.rotation_mode = "XYZ"

        # Apply user rotation (input is in radians)
        # CRITICAL: Y-axis flip also requires rotation flip (mirror transformation)
        rz_rad = item.get("rotation", 0)
        obj.rotation_euler[2] += -rz_rad  # Negate rotation due to Y-flip

        # Position: x, y from input; z = half bbox height
        # CRITICAL: Canvas uses screen coordinates (Y increases downward)
        #           Blender uses world coordinates (Y increases upward)
        #           We must flip Y: blender_y = room_height - canvas_y
        bbox_y = item.get("bbox_y", 0.5)
        canvas_y = item["y"]
        blender_y = room_h - canvas_y  # Flip Y axis
        obj.location = [item["x"], blender_y, bbox_y * 0.5]

        print(f"[WORKER] Placed: {asset_id[:8]}... canvas({item['x']:.2f}, {canvas_y:.2f}) → blender({item['x']:.2f}, {blender_y:.2f})")

    # ===== Lighting =====
    floor_center_x = room_w / 2
    floor_center_y = room_h / 2

    with suppress_blender_output():
        setup_lighting(floor_center_x, floor_center_y, (room_w, room_h))

    # ===== Render Settings =====
    # Select the Cycles device OUTSIDE suppress_blender_output so enable_gpu_backend's
    # CPU-fallback notice (if GPU selection fails) is visible, not fd-suppressed.
    device_mode = enable_gpu_backend()
    with suppress_blender_output():
        set_render_settings(high_res=high_res, device_mode=device_mode)

    # ===== Camera Setup =====
    floor_width = max(room_w, room_h)
    cam, _ = setup_camera(floor_center_x, floor_center_y, floor_width, wall_height)

    # ===== Render Top-Down View =====
    top_down_path = os.path.join(output_dir, "top_down_rendering.png")
    bpy.context.scene.render.filepath = top_down_path
    print(f"[WORKER] Rendering top-down → {top_down_path}")

    with suppress_blender_output():
        bpy.ops.render.render(write_still=True)

    # ===== Render Side View (45° elevation) =====
    original_z = cam.location.z
    side_pos = input_data.get("side_view_position", 3)
    cam.location = side_viewpoint(floor_center_x, floor_center_y, original_z, side_pos, 45)

    side_path = os.path.join(output_dir, "side_rendering.png")
    bpy.context.scene.render.filepath = side_path
    print(f"[WORKER] Rendering side → {side_path}")

    with suppress_blender_output():
        bpy.ops.render.render(write_still=True)

    # ===== Export .blend if requested =====
    if input_data.get("export_blend", False):
        blend_path = os.path.join(output_dir, "scene.blend")
        bpy.ops.file.pack_all()  # Embed textures for portability
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"[WORKER] Exported scene → {blend_path}")

    print("[WORKER] Render complete.")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("[WORKER] ========== render_worker.py starting ==========")

    # Launched as `python render_worker.py input.json output_dir` (no blender `--`).
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Usage: python render_worker.py input.json output_dir")
        sys.exit(1)

    input_json_path = argv[0]
    output_dir = argv[1]

    print(f"[WORKER] Input: {input_json_path}")
    print(f"[WORKER] Output: {output_dir}")

    # Read input JSON
    with open(input_json_path, "r") as f:
        input_data = json.load(f)

    # Run render
    render_scene(input_data, output_dir)

    print("[WORKER] ========== render_worker.py finished ==========")
