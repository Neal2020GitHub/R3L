import os
import json

from utils.r3l.types import AssetInfo, get_uid
from utils.r3l.geometry import calc_bbox_dims

from typing import Dict, List, Tuple


def create_asset_mapping(
    asset_info: Dict[str, AssetInfo],
    asset_ids: List[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build a mapping from asset_id -> object_id and vice versa.
    """
    # Build global object mappings once
    # object_id: "sofa-0", "sofa-1", "dinning-table-0", etc.
    # asset_id: "0a154cfc51c4417485ddd19637982b33-0"
    # object_to_asset: {object_id: asset_id}
    # asset_to_object: {asset_id: object_id}
    idx: Dict[str, int] = {}
    object_to_asset: Dict[str, str] = {}
    asset_to_object: Dict[str, str] = {}

    for asset_id in asset_ids:
        ainfo = asset_info[get_uid(asset_id)]
        name = ainfo.name.replace(" ", "-")
        i = idx.get(name, -1) + 1
        idx[name] = i
        object_id = f"{name}-{i}"
        object_to_asset[object_id] = asset_id
        asset_to_object[asset_id] = object_id

    return object_to_asset, asset_to_object


def create_asset_info(
    asset_ids: List[str],
    asset_dir: str,
) -> Dict[str, AssetInfo]:
    """
    Build a mapping from asset_id -> AssetInfo.
    This is essentially a lookup table for asset detail information
    `asset_ids` contains strings like "<asset_uid>-<index>"; only the base UID is
    used, and duplicates are ignored.
    """

    uniq = {get_uid(aid) for aid in asset_ids}
    asset_info: Dict[str, AssetInfo] = {}

    for asset_id in uniq: 
        path = os.path.join(asset_dir, asset_id, "annotations.json")
        assert os.path.exists(path), f"{path} not found"
        with open(path, "r") as f:
            a = json.load(f) # annotations
            bbox = a['thor_metadata']['assetMetadata']['boundingBox']
            bbox = calc_bbox_dims(bbox['min'], bbox['max'])

            asset_info[asset_id] = AssetInfo(
                name=a['category'],
                desc_short=a['description'], 
                desc_long=a['description_long'],
                bbox={
                    "x": bbox['x'],
                    "y": bbox['z'],  # swap y and z
                    "z": bbox['y']
                }
            )
    return asset_info


def flatten_assets(tree: dict) -> List[str]:
    """Flatten the blueprint's nested asset tree into an ordered asset_id list."""
    out: List[str] = []
    def walk(node: dict) -> None:
        for aid, children in node.items():
            out.append(aid)
            walk(children)
    walk(tree)
    return out

