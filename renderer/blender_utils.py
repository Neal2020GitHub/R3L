import bpy
import bmesh
import os
import math
import numpy as np
from mathutils import Vector
from PIL import Image, ImageDraw, ImageFont
from utils.log import print_good, print_error
from renderer.cycles_device import enable_gpu_backend

def reset_blender():
    bpy.ops.wm.read_factory_settings()
    #for scene in bpy.data.scenes:
    #    for obj in scene.objects:
    #        scene.objects.unlink(obj)
    #for block in bpy.data.orphaned_data:
    #    bpy.data.orphaned_data.remove(block)
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    for scene in bpy.data.scenes:
        if scene.rigidbody_world:
            scene.rigidbody_world.point_cache.frame_start = 1
            bpy.ops.ptcache.free_bake_all()
            
            
def setup_background():
    # Set up a new world or modify the existing one
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    
    # Clear existing nodes
    nodes = world.node_tree.nodes
    nodes.clear()
    
    # Create the nodes for the world shader
    output_node = nodes.new(type='ShaderNodeOutputWorld')
    output_node.location = (200, 0)
    background_node = nodes.new(type='ShaderNodeBackground')
    background_node.location = (0, 0)
    
    # Set background node to emit white light
    background_node.inputs['Color'].default_value = (1, 1, 1, 1)  # Background appears white
    background_node.inputs['Strength'].default_value = 1.0  # Uniform light strength
    
    # Connect the nodes
    world.node_tree.links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    print("White background set up.")
    
    
def create_wall_mesh(name, vertices, do_uv=False):
    """Build a closed-loop polygon mesh (bmesh verts/edges/one face from a
    vertex loop). With ``do_uv=True`` also UV-smart-project it — the floor
    callers (``build_scene`` and the builder worker) both want the unwrap, so it
    is shared here rather than duplicated at each call site."""
    # Create a new mesh
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)

    # Link the object to the scene
    scene = bpy.context.scene
    scene.collection.objects.link(obj)

    # Make the new object the active object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Enter Edit mode to create the wall geometry
    bpy.ops.object.mode_set(mode='EDIT')

    # Create a BMesh
    bm = bmesh.new()
    try:
        # Create the vertices
        for v in vertices:
            bm.verts.new(v)

        # Ensure the lookup table is updated
        bm.verts.ensure_lookup_table()


        # Create the edges between consecutive vertices
        for i in range(len(vertices)-1):
            bm.edges.new([bm.verts[i], bm.verts[i+1]])

        # Create the face (assuming a closed loop)
        bm.faces.new(bm.verts)

        bpy.ops.object.mode_set(mode='OBJECT')


        # Update the mesh with the BMesh data
        bm.to_mesh(mesh)
    finally:
        bm.free()  # always release the BMesh, even if construction raised

    if do_uv:
        # UV unwrap for proper texturing (shared by both floor callers)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project()
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()

    return obj


def load_hdri(hdri_path='./data/HDRIs/studio_small_08_4k.exr', hdri_strength=1, hide=True):  # Replace with the correct path
    # TODO: Search studio_small_08_4k.exr, download the HDRI file and put it in the correct path
    # Check if the file exists
    if not os.path.exists(hdri_path):
        print("HDRI file not found:", hdri_path)
        return
    
    # Get the world
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    # Enable nodes for the world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes (optional, be careful with this)
    # for node in nodes:
    #     nodes.remove(node)

    # Create a new Environment Texture node
    env_texture = nodes.new(type='ShaderNodeTexEnvironment')

    texture_coord = nodes.new(type="ShaderNodeTexCoord")
    # create a new Mapping node
    mapping_node = nodes.new(type='ShaderNodeMapping')

    # Load the HDRI image
    env_texture.image = bpy.data.images.load(hdri_path)

    # Create a Background node
    background = nodes.new(type='ShaderNodeBackground')
    background.location = (-100, 0)  # TODO: Why?
    background.inputs['Strength'].default_value = hdri_strength
    
    # Link the nodes
    links.new(texture_coord.outputs['Generated'], mapping_node.inputs['Vector'])
    links.new(mapping_node.outputs['Vector'], env_texture.inputs['Vector'])
    links.new(env_texture.outputs['Color'], background.inputs['Color'])

    # Create a World Output node if it doesn't exist
    if 'World Output' not in nodes:
        world_output = nodes.new(type='ShaderNodeOutputWorld')
        world_output.location = (100, 0)
    else:
        world_output = nodes['World Output']

    if hide:
        # Prevent HDRI from showing in the background
        # Add a Light Path node and mix shader to control the visibility
        light_path = nodes.new(type='ShaderNodeLightPath')
        mix_shader = nodes.new(type='ShaderNodeMixShader')
        bg_transparent = nodes.new(type='ShaderNodeBackground')
        bg_transparent.inputs['Color'].default_value = (0, 0, 0, 1)  # Black, fully transparent
        
        # Link the nodes to use Light Path for mixing
        links.new(light_path.outputs['Is Camera Ray'], mix_shader.inputs['Fac'])
        links.new(background.outputs['Background'], mix_shader.inputs[1])
        links.new(bg_transparent.outputs['Background'], mix_shader.inputs[2])
        links.new(mix_shader.outputs['Shader'], world_output.inputs['Surface'])
    else:
        links.new(background.outputs['Background'], world_output.inputs['Surface'])
    
    print("HDRI background set successfully.") 
    
    
def merge_imported_doubles(mesh_objects, merge_distance=0.0001):
    """Merge duplicated vertices on each imported mesh (fixes "black objects"
    caused by duplicate verts). Shared by ``renderer.blender_render.build_scene``
    and the builder render worker — both used to inline this block verbatim.
    Per-mesh best-effort: a failure on one mesh is logged and skipped, render
    continues. The BMesh is released in ``finally`` so a construction error
    cannot leak it."""
    for mesh_obj in mesh_objects:
        bm = bmesh.new()
        try:
            mesh = mesh_obj.data
            bm.from_mesh(mesh)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_distance)
            bm.to_mesh(mesh)
            mesh.update()
        except Exception as e:
            print_error(f"Failed to merge vertices for {mesh_obj.name}: {e}")
        finally:
            bm.free()


def side_viewpoint(center_x, center_y, original_z, idx, phi_deg=45):
    """Camera location for the idx-th side view (0..3) at the given pitch.
    Shared by ``build_scene`` (single 'side' view), ``render_scene`` (looped
    ``side_view_indices``), and the builder worker — all three used to inline
    this trigonometry verbatim. ``idx`` numbers the azimuth (theta = idx/4 * 2π);
    ``phi_deg`` is the pitch above the floor."""
    theta = (idx / 4) * math.pi * 2
    phi = math.radians(phi_deg)
    return (
        center_x + original_z * math.sin(phi) * math.cos(theta),
        center_y + original_z * math.sin(phi) * math.sin(theta),
        original_z * math.cos(phi),
    )


def set_rendering_settings(panorama=False, high_res=False, use_cycles=True, indoor_camera=False):
    render = bpy.context.scene.render

    if use_cycles:
        render.engine = "CYCLES"
        bpy.context.scene.cycles.device = enable_gpu_backend()

        cycles_settings = bpy.context.scene.cycles
        cycles_settings.samples = 128
        cycles_settings.diffuse_bounces = 3
        cycles_settings.glossy_bounces = 3
        cycles_settings.transparent_max_bounces = 5
        cycles_settings.transmission_bounces = 5
        cycles_settings.filter_width = 0.01
        cycles_settings.use_denoising = True
    else:
        render.engine = 'BLENDER_EEVEE'
    
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"

    if high_res:
        render.resolution_x = 1080
        render.resolution_y = 1080
        render.resolution_percentage = 100
    else:
        render.resolution_x = 512
        render.resolution_y = 512
        render.resolution_percentage = 100
    
    render.film_transparent = not indoor_camera
    

def setup_camera(center_x, center_y, width, wall_height=.1, wide_lens=False, fov_multiplier=1.1, use_damped_track=False):
    # Check if the camera exists, if not, create one
    cam = bpy.data.objects.get("Camera")
    if cam is None:
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name = "Camera"
    bpy.context.scene.camera = cam

    # setup lens
    if wide_lens:
        cam.data.lens /= 2 # 35/2
    # cam.data.type = 'ORTHO'

    # compute fov
    fov = 2 * math.atan((cam.data.sensor_width / (2 * cam.data.lens)))  # Horizontal FoV calculation
    target_width = abs(fov_multiplier * width)  # Target width to cover
    
    # camera position
    cam.location.x = center_x
    cam.location.y = center_y
    cam.location.z = wall_height + (target_width / 2) / math.tan(fov / 2)  # Z position calculation

    # clear camera constraints
    cam.constraints.clear()
    
    # get or create target Empty
    empty = bpy.data.objects.get("CameraTarget")
    if empty is None:
        empty = bpy.data.objects.new("CameraTarget", None)
        bpy.context.scene.collection.objects.link(empty)
    empty.location = (center_x, center_y, 0)
    
    # add constraint
    if use_damped_track:
        cam_constraint = cam.constraints.new(type="DAMPED_TRACK")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    else:
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
    cam_constraint.target = empty
    
    return cam, cam_constraint


def world_to_camera_view(scene, camera, coord):
    """Convert world coordinates to camera view coordinates"""
    co_local = camera.matrix_world.normalized().inverted() @ coord
    z = -co_local.z

    camera_data = camera.data
    frame = [-v for v in camera_data.view_frame(scene=scene)[:3]]
    if camera_data.type != 'ORTHO':
        frame = [(v / (v.z / z)) for v in frame]

    min_x, max_x = frame[1].x, frame[2].x
    min_y, max_y = frame[0].y, frame[1].y
    x = (co_local.x - min_x) / (max_x - min_x)
    y = (co_local.y - min_y) / (max_y - min_y)
    return Vector((x, y, z))


def get_pixel_coordinates(scene, camera, world_coord):
    """Get pixel coordinates for a given world coordinate"""
    if isinstance(world_coord, np.ndarray) or isinstance(world_coord, list):
        world_coord = Vector(world_coord)
    coord_2d = world_to_camera_view(scene, camera, world_coord)
    return (coord_2d.x, 1 - coord_2d.y)


def annotate_image_with_coordinates(image_path, visual_marks, output_path, format="coordinate", default_font_size=18):
    script_path = os.path.dirname(os.path.realpath(__file__))
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size
    draw = ImageDraw.Draw(img)

    wall_font_size = 18 * img_width / 1000
    # wall_font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), wall_font_size)
    wall_font = ImageFont.truetype("DejaVuSans.ttf", wall_font_size)

    to_draw_list = []
    if format == "coordinate":
        assert type(visual_marks) == dict
        for (x, y), (pixel_x, pixel_y) in visual_marks.items():
            to_draw_list.append({"pixel": (pixel_x, pixel_y), "text": f"({x},{y})"})

        # Define font (you may need to adjust the path to a font file)
        # adjust the font size based on the image size
        default_font_size = 18 * img_width / 1000
        # font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), default_font_size)
        font = ImageFont.truetype("DejaVuSans.ttf", default_font_size)

    elif format == "text":
        assert type(visual_marks) == list
        for visual_mark in visual_marks:
            assert type(visual_mark) == dict
            assert "pixel" in visual_mark and "text" in visual_mark
        to_draw_list = visual_marks

        default_font_size = default_font_size * img_width / 1000
        # font = ImageFont.truetype(os.path.join(script_path, "Arial.ttf"), default_font_size)
        font = ImageFont.truetype("DejaVuSans.ttf", default_font_size)

    else:
        raise ValueError("Invalid format. Choose 'coordinate' or 'text'.")

    # Draw marks and coordinates
    for to_draw_dict in to_draw_list:

        pixel_x, pixel_y = to_draw_dict["pixel"]
        text = to_draw_dict["text"]

        pixel_x = pixel_x * img_width
        pixel_y = pixel_y * img_height

        # Draw an arrow from pixel_x, pixel_y to end_pixel_x, end_pixel_y
        if "end_arrow_pixel" in to_draw_dict:
            end_pixel_x, end_pixel_y = to_draw_dict["end_arrow_pixel"]
            end_pixel_x = end_pixel_x * img_width
            end_pixel_y = end_pixel_y * img_height
            # Calculate arrow properties
            arrow_length = ((end_pixel_x - pixel_x)**2 + (end_pixel_y - pixel_y)**2)**0.5
            angle = math.atan2(end_pixel_y - pixel_y, end_pixel_x - pixel_x)
            # Draw the arrow shaft
            draw.line([pixel_x, pixel_y, end_pixel_x, end_pixel_y], fill="black", width=5)
            # Draw the arrow head
            arrow_head_length = min(15, arrow_length / 3)  # Adjust size as needed
            arrow_head_width = arrow_head_length
            x1 = end_pixel_x - arrow_head_length * math.cos(angle) + arrow_head_width * math.sin(angle)
            y1 = end_pixel_y - arrow_head_length * math.sin(angle) - arrow_head_width * math.cos(angle)
            x2 = end_pixel_x - arrow_head_length * math.cos(angle) - arrow_head_width * math.sin(angle)
            y2 = end_pixel_y - arrow_head_length * math.sin(angle) + arrow_head_width * math.cos(angle)
            draw.polygon([end_pixel_x, end_pixel_y, x1, y1, x2, y2], fill="black")

        # Draw a small red dot
        dot_radius = 3
        draw.ellipse([pixel_x - dot_radius, pixel_y - dot_radius, 
                      pixel_x + dot_radius, pixel_y + dot_radius], 
                     fill="red", outline="red")
        # Draw coordinate text
        text_bbox = draw.textbbox((pixel_x, pixel_y), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # coordinate?
        if text.startswith("("):
            draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2), text, font=font, fill="red")
        else:
            # draw wall variable names
            #if text.startswith("walls["):
            #    draw.text((pixel_x - text_w/2, pixel_y + dot_radius + 2), 
            #            text, font=wall_font, fill="red" if text.startswith("(") else "white")
            # draw asset variable names
            font_color = "black"
            if "end_arrow_pixel" in to_draw_dict:
                if end_pixel_y >= pixel_y:
                    if default_font_size > 80:
                        # that means the arrow is pointing downwards, adjust the text position to be above the dot
                        draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2 - text_h*1.3), text, font=font, fill=font_color)
                    else:
                        draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2 - text_h*1.1), text, font=font, fill=font_color)
                else:
                    draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2), text, font=font, fill=font_color)
            else:
                draw.text((pixel_x - text_w/2, pixel_y + dot_radius - 2), text, font=font, fill=font_color)
    # Save the annotated image
    img.save(output_path)
    print_good(f"Annotated image saved to {output_path}")
