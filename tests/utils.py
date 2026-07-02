from typing import List, Dict
import os
import json
from utils.r3l.types import AssetInfo
from utils.r3l.geometry import calc_bbox_dims

def get_asset_info(asset_dir: str, asset_ids: List[str]): 
    asset_info: Dict[str, AssetInfo] = {}
    for asset_id in asset_ids: 
        path = os.path.join(asset_dir, asset_id, "annotations.json")
        assert os.path.exists(path), f"{path} not found"
        assert os.path.isfile(path), f"{path} is not a file"
        with open(path, "r") as f:
            a = json.load(f) # annotations
            bbox = a['thor_metadata']['assetMetadata']['boundingBox']
            bbox = calc_bbox_dims(bbox['min'], bbox['max'])

            asset_info[asset_id] = AssetInfo(
                name=a['category'],
                desc_short=a['description'], 
                desc_long=a['description'],
                bbox={
                    "x": bbox['x'],
                    "y": bbox['z'],
                    "z": bbox['y']
                }
            )
    return asset_info