from .protocols import *
from typing import Dict
from utils.log import print_warn


def holodeck_to_r3l_blueprint(blueprint: HolodeckBlueprint) -> R3LBlueprint:
    floor_objs: AssetList = blueprint.objects['floor']
    name2node: Dict[str, str] = {}
    idx: Dict[str, int] = {} # tracks index of assets
    tree: Dict[str, dict] = {}

    for aname, aid in floor_objs:
        j = idx.get(aid, -1) + 1; idx[aid] = j
        node = f"{aid}-{j}"
        tree[node] = {}
        name2node[aname] = node

    small: Dict[str, AssetList] = blueprint.objects['small']
    for parent_name, child_list in small.items():
        if parent_name not in name2node:
            print_warn(f"Skipping non-floor object: {parent_name}")
            continue
        pnode = name2node[parent_name]
        for _, child_aid in child_list:
            k = idx.get(child_aid, -1) + 1; idx[child_aid] = k
            tree[pnode][f"{child_aid}-{k}"] = {}

    room: RoomDict = blueprint.room
    room_length = max(v[0] for v in room['full_vertices'])
    room_width = max(v[1] for v in room['full_vertices'])
    room_size = (room_length, room_width)

    return R3LBlueprint(
        prompt=blueprint.query,
        design=blueprint.design,
        assets=tree,
        room_size=room_size,
    )


def to_r3l_blueprint(scene_config: dict) -> R3LBlueprint:
    return R3LBlueprint(
        prompt=scene_config["prompt"],
        design="",
        assets=scene_config["assets"],
        room_size=(float(scene_config["room_size"][0]), float(scene_config["room_size"][1])),
    )
