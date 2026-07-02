# pyright: reportAttributeAccessIssue=false
# asset.py is a Blender (bpy) adapter. bpy.data/ops/context are injected by Blender's
# C runtime and are absent from the static stubs, so pyright's attribute checks on the
# bpy module are spurious here. Runtime / integration use guards correctness.
import argparse
import os
import sys
import compress_json
import requests
import json
import gzip
import msgpack
import numpy as np
import bpy
import gc
from typing import Dict, Any, cast


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.holodeck_v2.constants import OBJATHOR_ASSETS_DIR, OBJATHOR_ANNOTATIONS_PATH, BASE_URL, ASSET_BASE_DIR
from utils.log import print_info, print_good, print_error
from utils.fd import suppress_blender_output
                    

def _download_with_asset_id(asset_base_dir, asset_id, max_retries=3):
    tar_path = os.path.join(asset_base_dir, f"{asset_id}.tar")
    url = f"{BASE_URL}/assets/{asset_id}.tar"
    
    for attempt in range(max_retries):
        try:
            print_info(f"Downloading asset from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_length = response.headers.get("content-length")
            content_type = response.headers.get("content-type")
            
            if content_type is not None and content_type.startswith("text/html"):
                raise ValueError(f"Invalid URL: {url}")

            with open(tar_path, "wb") as f:
                if total_length is None:  # no content length header
                    f.write(response.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in response.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
            
            os.system(f"tar -xf {tar_path} -C {asset_base_dir}")
            os.remove(tar_path)
            return
            
        except (requests.exceptions.RequestException, ValueError) as e:
            print_error(f"Download failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt == max_retries - 1:
                raise Exception(f"Download failed, already tried {max_retries} times: {str(e)}")
            else:
                continue

def _load_annotations():
    annotations_path = os.path.join(OBJATHOR_ANNOTATIONS_PATH)
    return compress_json.load(annotations_path)

def _reformat_annotations(asset_base_dir, asset_id):
    asset_dir = os.path.join(asset_base_dir, asset_id)
    
    annotations = compress_json.load(os.path.join(asset_dir, "annotations.json.gz"))
    with open(os.path.join(asset_dir, "thor_metadata.json"), "r") as f:
        thor_metadata = json.load(f)
    annotations["thor_metadata"] = thor_metadata
    
    with open(os.path.join(asset_dir, "annotations.json"), "w") as f:
        json.dump(annotations, f, indent=4)
    
    # remove original json files
    os.remove(os.path.join(asset_dir, "annotations.json.gz"))
    os.remove(os.path.join(asset_dir, "thor_metadata.json"))
    
def _cleanup_blender_data():    
    # Remove all orphaned data blocks
    for collection_name in ["meshes", "materials", "images", "textures", "node_groups"]:
        collection = getattr(bpy.data, collection_name)
        for item in collection:
            if item.users == 0:
                collection.remove(item)
    
    # Force garbage collection in Blender. orphans_purge prints a C-level "Info:"
    # report straight to the stdout fd, so suppress at the fd level.
    with suppress_blender_output():
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    
    # Reset the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Garbage collection to free up memory
    gc.collect()

def _export_msgpack_to_glb(asset_base_dir, asset_id):
    # Reset the scene
    _cleanup_blender_data()
    
    asset_dir = os.path.join(asset_base_dir, asset_id)
    
    # Load the object data from the .msgpack.gz file
    with gzip.open(os.path.join(asset_dir, f"{asset_id}.msgpack.gz"), "rb") as f:
        obj_data = cast(Dict[str, Any], msgpack.load(f, raw=False))
    
    # ----------------- Mesh -----------------
    
    vertices = [[v["x"], v["z"], v["y"]] for v in obj_data["vertices"]]  # Blender Z up
    triangles = np.array(obj_data["triangles"]).reshape((-1, 3))
    
    mesh = bpy.data.meshes.new(name="obj")
    obj = bpy.data.objects.new("obj", mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata(vertices, [], triangles)
    mesh.update()
    
    # ----------------- UV -----------------
    
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
    
    # ----------------- Material -----------------
    
    material = bpy.data.materials.new(name="AlbedoMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Principled BSDF
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf is None:
        principled_bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    # Material Output
    material_output = nodes.get("Material Output")
    if material_output is None:
        material_output = nodes.new("ShaderNodeOutputMaterial")
    links.new(principled_bsdf.outputs["BSDF"], material_output.inputs["Surface"])
    
    def load_image(name):
        path = os.path.abspath(os.path.join(asset_dir, name))
        if os.path.exists(path):
            return bpy.data.images.load(path)
        else:
            print_error(f"Image {name} not found")
            return None

    # Albedo
    albedo = load_image("albedo.jpg")
    if albedo:
        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.image = albedo
        links.new(texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])

    # Normal
    normal = load_image("normal.jpg")
    if normal:
        normal_tex = nodes.new("ShaderNodeTexImage")
        normal_tex.image = normal
        normal_tex.image.colorspace_settings.name = "Non-Color"
        normal_map_node = nodes.new("ShaderNodeNormalMap")
        links.new(normal_tex.outputs["Color"], normal_map_node.inputs["Color"])
        links.new(normal_map_node.outputs["Normal"], principled_bsdf.inputs["Normal"])

    # Emission
    emission = load_image("emission.jpg")
    if emission:
        emission_tex = nodes.new("ShaderNodeTexImage")
        emission_tex.image = emission
        links.new(emission_tex.outputs["Color"], principled_bsdf.inputs["Emission Color"])
        principled_bsdf.inputs["Emission Strength"].default_value = 1.0

    # Metallic + Roughness
    metallic = load_image("metallic_smoothness.jpg")
    if metallic:
        metallic_tex = nodes.new("ShaderNodeTexImage")
        metallic_tex.image = metallic
        metallic_tex.image.colorspace_settings.name = "Non-Color"
        
        # Separate RGB
        sep_node = nodes.new("ShaderNodeSeparateRGB")
        links.new(metallic_tex.outputs["Color"], sep_node.inputs["Image"])
        
        # Unity: G = Smoothness, Roughness = 1 - Smoothness
        invert_node = nodes.new("ShaderNodeInvert")
        links.new(sep_node.outputs["G"], invert_node.inputs["Color"])
        links.new(invert_node.outputs["Color"], principled_bsdf.inputs["Roughness"])

        # B = Metallic
        links.new(sep_node.outputs["B"], principled_bsdf.inputs["Metallic"])
        
    # assign material to object
    obj.data.materials.append(material)
    
    mesh.update()
    
    # ----------------- Rotations -----------------
    
    rotation_angle = np.deg2rad(-obj_data["yRotOffset"] + 180)  # +180 to face -Y as forward in Blender
    obj.rotation_euler = (0, 0, rotation_angle)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    
    # export. The glTF exporter emits Draco / INFO·WARNING lines to the stdout fd
    # (Draco is genuinely absent and we never request compression), so suppress.
    with suppress_blender_output():
        bpy.ops.export_scene.gltf(
            filepath=os.path.join(asset_dir, f"{asset_id}.glb"),
            export_format="GLB",
            use_selection=False
        )
    
    # remove used files
    os.remove(os.path.join(asset_dir, f"{asset_id}.msgpack.gz"))
    # os.remove(os.path.join(asset_dir, "albedo.jpg"))
    # os.remove(os.path.join(asset_dir, "normal.jpg"))
    # os.remove(os.path.join(asset_dir, "emission.jpg"))
    # os.remove(os.path.join(asset_dir, "metallic_smoothness.jpg"))
    os.remove(os.path.join(asset_dir, "success.txt"))
    os.system(f"rm -r {os.path.join(asset_dir, 'thor_renders')}")
    _cleanup_blender_data()

def is_asset_cached(asset_id, asset_base_dir=ASSET_BASE_DIR) -> bool:
    """True if the asset's processed .glb already exists on disk (nothing to fetch)."""
    return os.path.exists(os.path.join(asset_base_dir, asset_id, f"{asset_id}.glb"))


def download_and_process_asset(asset_id, asset_base_dir=ASSET_BASE_DIR):  # entry point
    if is_asset_cached(asset_id, asset_base_dir):
        print_good(f"Asset {asset_id} already exists")
        return

    _download_with_asset_id(asset_base_dir, asset_id)
    _reformat_annotations(asset_base_dir, asset_id)
    _export_msgpack_to_glb(asset_base_dir, asset_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_id", type=str, help="The ID of the asset to download", required=True)
    args = parser.parse_args()
    
    # asset_base_dir = OBJATHOR_ASSETS_DIR
    asset_base_dir = "./data/assets"  # TODO: change to your own path
    if not os.path.exists(asset_base_dir):
        os.makedirs(asset_base_dir, exist_ok=True)
    
    # load annotations
    # annotations = _load_annotations()
    # print(annotations[args.asset_id])
    print_info(f"Downloading asset with ID {args.asset_id}")
    download_and_process_asset(asset_id=args.asset_id, asset_base_dir=asset_base_dir)


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_annotations(obj_data: Dict[str, Any]):
    if "annotations" in obj_data:
        return obj_data["annotations"]
    else:
        # The assert here is just double-checking that a field that should exist does.
        assert "onFloor" in obj_data, f"Can not find annotations in obj_data {obj_data}"

        return obj_data