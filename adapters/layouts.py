from .protocols import *


def r3l_to_renderer_layout(input_layout: R3LSolution) -> RendererSolution:
    asset_ids = []
    positions = []
    rotations = []
    asset_names = []

    for object_name, placement in input_layout.layout.items():
        asset_ids.append(placement.asset_id)
        positions.append(placement.position)
        rotations.append(placement.rotation)
        asset_names.append(object_name)

    w, h = input_layout.room_size
    floor_vertices = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]

    return RendererSolution(
        asset_ids=asset_ids,
        positions=positions,
        rotations=rotations,
        floor_vertices=floor_vertices,
        wall_height=2.8,  # default, r3l doesn't use wall_height
        asset_names=asset_names,
    )
