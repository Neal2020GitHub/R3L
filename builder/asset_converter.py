#!/usr/bin/env python3
"""
Self-contained Blender script for converting msgpack.gz assets to GLB.

Runs under the repo's venv (bpy==4.0.0 pip-installed); no repo imports.

Usage: python asset_converter.py <asset_id> <asset_dir>

The script:
1. Loads {asset_id}.msgpack.gz (raw mesh data: vertices, triangles, UVs)
2. Creates Blender mesh with materials and textures
3. Exports to {asset_id}.glb
4. Cleans up temporary files
"""
import bpy
import gc
import gzip
import json
import math
import msgpack
import os
import sys


def cleanup_blender():
    """Remove orphaned data and reset scene."""
    for collection_name in ["meshes", "materials", "images", "textures", "node_groups"]:
        collection = getattr(bpy.data, collection_name)
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)

    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    gc.collect()


def convert_msgpack_to_glb(asset_dir: str, asset_id: str):
    """
    Convert msgpack.gz asset to GLB format.

    Args:
        asset_dir: Directory containing the asset files
        asset_id: The asset identifier
    """
    print(f"[CONVERTER] Converting {asset_id}")

    # Reset scene
    cleanup_blender()

    msgpack_path = os.path.join(asset_dir, f"{asset_id}.msgpack.gz")
    glb_path = os.path.join(asset_dir, f"{asset_id}.glb")

    if not os.path.exists(msgpack_path):
        raise FileNotFoundError(f"msgpack.gz not found: {msgpack_path}")

    # Load mesh data from msgpack
    print(f"[CONVERTER] Loading {msgpack_path}")
    with gzip.open(msgpack_path, "rb") as f:
        obj_data = msgpack.load(f, raw=False)

    # ===== Create Mesh =====
    # Note: Blender uses Z-up, so we swap Y and Z
    vertices = [[v["x"], v["z"], v["y"]] for v in obj_data["vertices"]]
    triangles = []
    raw_triangles = obj_data["triangles"]
    for i in range(0, len(raw_triangles), 3):
        triangles.append([raw_triangles[i], raw_triangles[i + 1], raw_triangles[i + 2]])

    mesh = bpy.data.meshes.new(name="obj")
    obj = bpy.data.objects.new("obj", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(vertices, [], triangles)
    mesh.update()

    print(f"[CONVERTER] Mesh created: {len(vertices)} vertices, {len(triangles)} triangles")

    # ===== UV Mapping =====
    uvs = [[uv["x"], uv["y"]] for uv in obj_data["uvs"]]
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    uv_layer = mesh.uv_layers["UVMap"]
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            loop = mesh.loops[loop_index]
            uv = uvs[loop.vertex_index]
            uv_layer.data[loop_index].uv = uv

    mesh.update()

    # ===== Materials =====
    material = bpy.data.materials.new(name="AlbedoMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Get or create Principled BSDF
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf is None:
        principled_bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    # Get or create Material Output
    material_output = nodes.get("Material Output")
    if material_output is None:
        material_output = nodes.new("ShaderNodeOutputMaterial")
    links.new(principled_bsdf.outputs["BSDF"], material_output.inputs["Surface"])

    def load_texture(name: str):
        """Load texture image if it exists."""
        path = os.path.join(asset_dir, name)
        if os.path.exists(path):
            return bpy.data.images.load(path)
        return None

    # Albedo (base color)
    albedo = load_texture("albedo.jpg")
    if albedo:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = albedo
        links.new(tex_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

    # Normal map
    normal = load_texture("normal.jpg")
    if normal:
        normal_tex = nodes.new("ShaderNodeTexImage")
        normal_tex.image = normal
        normal_tex.image.colorspace_settings.name = "Non-Color"
        normal_map_node = nodes.new("ShaderNodeNormalMap")
        links.new(normal_tex.outputs["Color"], normal_map_node.inputs["Color"])
        links.new(normal_map_node.outputs["Normal"], principled_bsdf.inputs["Normal"])

    # Emission
    emission = load_texture("emission.jpg")
    if emission:
        emission_tex = nodes.new("ShaderNodeTexImage")
        emission_tex.image = emission
        links.new(emission_tex.outputs["Color"], principled_bsdf.inputs["Emission Color"])
        principled_bsdf.inputs["Emission Strength"].default_value = 1.0

    # Metallic + Roughness (Unity format: G=Smoothness, B=Metallic)
    metallic = load_texture("metallic_smoothness.jpg")
    if metallic:
        metallic_tex = nodes.new("ShaderNodeTexImage")
        metallic_tex.image = metallic
        metallic_tex.image.colorspace_settings.name = "Non-Color"

        # Separate RGB channels
        sep_node = nodes.new("ShaderNodeSeparateRGB")
        links.new(metallic_tex.outputs["Color"], sep_node.inputs["Image"])

        # Roughness = 1 - Smoothness (G channel)
        invert_node = nodes.new("ShaderNodeInvert")
        links.new(sep_node.outputs["G"], invert_node.inputs["Color"])
        links.new(invert_node.outputs["Color"], principled_bsdf.inputs["Roughness"])

        # Metallic = B channel
        links.new(sep_node.outputs["B"], principled_bsdf.inputs["Metallic"])

    # Assign material to object
    obj.data.materials.append(material)
    mesh.update()

    # ===== Apply Rotation =====
    # Apply yRotOffset from asset data (+180 to face -Y as forward in Blender)
    y_rot_offset = obj_data.get("yRotOffset", 0)
    rotation_angle = math.radians(-y_rot_offset + 180)
    obj.rotation_euler = (0, 0, rotation_angle)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # ===== Export GLB =====
    print(f"[CONVERTER] Exporting to {glb_path}")
    bpy.ops.export_scene.gltf(
        filepath=glb_path,
        export_format="GLB",
        use_selection=False
    )

    # ===== Cleanup =====
    # Remove msgpack.gz after successful conversion
    os.remove(msgpack_path)
    print(f"[CONVERTER] Removed {msgpack_path}")

    # Remove success.txt if it exists
    success_txt = os.path.join(asset_dir, "success.txt")
    if os.path.exists(success_txt):
        os.remove(success_txt)

    # Remove thor_renders directory if it exists
    thor_renders = os.path.join(asset_dir, "thor_renders")
    if os.path.exists(thor_renders):
        import shutil
        shutil.rmtree(thor_renders)

    cleanup_blender()

    print(f"[CONVERTER] Conversion complete: {glb_path}")


if __name__ == "__main__":
    print("[CONVERTER] ========== asset_converter.py starting ==========")

    # Launched as `python asset_converter.py <asset_id> <asset_dir>` (no blender `--`).
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Usage: python asset_converter.py <asset_id> <asset_dir>")
        sys.exit(1)

    asset_id = argv[0]
    asset_dir = argv[1]

    print(f"[CONVERTER] Asset ID: {asset_id}")
    print(f"[CONVERTER] Asset dir: {asset_dir}")

    convert_msgpack_to_glb(asset_dir, asset_id)

    print("[CONVERTER] ========== asset_converter.py finished ==========")
