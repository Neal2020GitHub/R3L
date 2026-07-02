# Copyright 2023 Allen Institute for Artificial Intelligence
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from AllenAI Holodeck (https://github.com/allenai/Holodeck;
# Yang et al., CVPR 2024) and Holodeck 2.0 (Bian et al., 2025,
# arXiv:2508.05899). Adapted for the R3L pipeline. See the LICENSE file
# in this directory for the full Apache 2.0 terms.
#
# Modifications (where applicable) are Copyright (c) 2026 Yuqi Wang and
# Zhifeng Gu and licensed under the MIT License (see the repository root
# LICENSE).

from typing import List, TypedDict, Dict, Literal, Union, Any, Optional, NamedTuple, Callable
from typing import Tuple
from torch import nn


class XYZ(TypedDict):
    x: float
    y: float
    z: float


class _RoomDictRequired(TypedDict):
    id: str
    roomType: str
    vertices: List[Tuple[float, float]]
    floorPolygon: List[XYZ]
    full_vertices: List[Tuple[float, float]]


class RoomDict(_RoomDictRequired, total=False):
    pass


class WallSegment(TypedDict):
    id: str
    roomId: str
    polygon: List[XYZ]
    width: float
    height: float
    direction: Optional[str]
    segment: List[List[float]]
    connect_exterior: Optional[str]
    connected_rooms: List[Any]


class WallsDict(TypedDict):
    wall_height: float
    walls: List[WallSegment]


class ClipModelsDict(TypedDict):
    clip_tokenizer: Callable
    clip_model: nn.Module
    clip_preprocess: Callable


AssetList = List[Tuple[str, str]] 
# (object_name, asset_id)


class SmallObjectInfo(TypedDict):
    object_name: str
    quantity: int
    variance_type: str
    importance: Union[int, float]


class ObjectInfo(TypedDict):
    object_name: str
    description: str
    location: Literal["floor", "wall"]
    size: Optional[List[Union[float, int]]]
    quantity: int
    variance_type: Literal["same", "varied"]
    importance: Union[int, float]
    objects_on_top: List[SmallObjectInfo]


class ObjectPlan(Dict[str, ObjectInfo]):
    # maps: object_name -> object details dict
    pass


# ---------- layout placement types ----------

class PlacementPosition(TypedDict):
    x: float
    y: float
    z: float


class PlacementRotation(TypedDict):
    x: float
    y: float
    z: float


class LayoutPlacementDict(TypedDict):
    assetId: str
    id: str
    object_name: str
    position: PlacementPosition
    rotation: PlacementRotation
    roomId: str
    vertices: List[Tuple[float, float]]


# ---------- helper functions ----------

def _normalize_attribute_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    return {key.strip().lower().replace(" ", "_"): value for key, value in d.items()}


def _recursively_normalize_attribute_keys(obj: Any) -> Any:
    if isinstance(obj, Dict):
        return {
            key.strip()
            .lower()
            .replace(" ", "_"): _recursively_normalize_attribute_keys(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, List):
        return [_recursively_normalize_attribute_keys(value) for value in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        print(
            f"Unexpected type {type(obj)} in {obj} while normalizing attribute keys."
            f" Returning the object as is."
        )
        return obj


def objects_on_top_from_dict(obj: Dict[str, Any]) -> Optional[SmallObjectInfo]:
    try:
        _normalize_attribute_keys(obj)
        object_name = obj["object_name"]
        quantity = int(obj["quantity"])
        variance_type = obj.get("variance_type", "same")
        if variance_type not in ["same", "varied"]:
            obj["variance_type"] = (
                "same" if not variance_type.startswith("v") else "varied"
            )
        importance = float(obj.get("importance", 0))
    except (KeyError, ValueError):
        return None

    return {
        "object_name": object_name,
        "quantity": quantity,
        "variance_type": variance_type,
        "importance": importance,
    }


def floor_or_wall_object_from_dict(
    object_name: str,
    obj: Dict[str, Any],
) -> Optional[ObjectInfo]:
    try:
        obj = _normalize_attribute_keys(obj)

        if not object_name:
            return None
        description: str = obj["description"]

        raw_loc = obj.get("location", "floor")
        location: Literal["floor", "wall"] = "wall" if raw_loc == "wall" else "floor"

        raw_size = obj.get("size", None)
        size: Optional[List[Union[float, int]]] = None
        if (
            isinstance(raw_size, list)
            and len(raw_size) == 3
            and all(_is_number(x) for x in raw_size)
        ):
            size = raw_size

        quantity: int = int(obj["quantity"])

        raw_vt = obj.get("variance_type", "same")
        variance_type: Literal["same", "varied"] = (
            "varied" if isinstance(raw_vt, str) and raw_vt.startswith("v") else "same"
        )

        importance: Union[int, float] = float(obj.get("importance", 0))

        children = obj.get("objects_on_top", [])
        parsed = [objects_on_top_from_dict(c) for c in children]
        objects_on_top: List[SmallObjectInfo] = [c for c in parsed if c is not None]

    except (KeyError, ValueError, TypeError):
        return None

    return {
        "object_name": object_name,
        "description": description,
        "location": location,
        "size": size,
        "quantity": quantity,
        "variance_type": variance_type,
        "importance": importance,
        "objects_on_top": objects_on_top,
    }


def object_plan_from_dict(obj: Dict[str, Dict[str, Any]]) -> ObjectPlan:
    opd = {
        key: floor_or_wall_object_from_dict(key, value)
        for key, value in obj.items()
    }
    return ObjectPlan(
        (k, {**v, "object_name": k}) for k, v in opd.items() if v is not None
    )



# ---------- runtime validators ----------

def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def validate_room_dict(room: RoomDict) -> None:
    if not isinstance(room, dict):
        raise TypeError("room must be a dict")
    for k in ("id", "roomType", "vertices", "floorPolygon"):
        if k not in room:
            raise ValueError(f"room missing required key: {k}")
    if not isinstance(room["id"], str) or not isinstance(room["roomType"], str):
        raise TypeError("room.id and room.roomType must be str")
    verts = room["vertices"]
    if not isinstance(verts, list) or any(
        not isinstance(t, (list, tuple)) or len(t) != 2 or not all(_is_number(v) for v in t)
        for t in verts
    ):
        raise TypeError("room.vertices must be List[Tuple[float, float]]")
    fp = room["floorPolygon"]
    if not isinstance(fp, list) or any(
        not isinstance(p, dict) or not all(k in p for k in ("x", "y", "z")) or not all(_is_number(p[k]) for k in ("x", "y", "z"))
        for p in fp
    ):
        raise TypeError("room.floorPolygon must be List[{\'x\',\'y\',\'z\'}]")


def validate_walls_dict(walls: WallsDict) -> None:
    if not isinstance(walls, dict):
        raise TypeError("walls must be a dict")
    if "wall_height" not in walls or "walls" not in walls:
        raise ValueError("walls must contain 'wall_height' and 'walls'")
    if not _is_number(walls["wall_height"]):
        raise TypeError("wall_height must be a number")
    wl = walls["walls"]
    if not isinstance(wl, list):
        raise TypeError("walls['walls'] must be a list")
    for w in wl:
        if not isinstance(w, dict):
            raise TypeError("each wall must be a dict")
        for key in ("id", "roomId", "width", "height", "segment"):
            if key not in w:
                raise ValueError(f"wall missing key: {key}")


def validate_layout_placements(layout: Any) -> None:
    """
    Ensures each placement dict contains required keys and correct types:
      - assetId, id, object_name, roomId: str
      - position: {x,y,z} numbers
      - rotation: {x,y,z} numbers
      - vertices: List of (x,z) numeric pairs (length >= 4 recommended)
    """
    if not isinstance(layout, list):
        raise TypeError("layout must be a list of placement dicts")

    required_keys = (
        "assetId",
        "id",
        "object_name",
        "position",
        "rotation",
        "roomId",
        "vertices",
    )

    for idx, item in enumerate(layout):
        if not isinstance(item, dict):
            raise TypeError(f"layout[{idx}] must be a dict")

        missing = [k for k in required_keys if k not in item]
        if missing:
            raise ValueError(f"layout[{idx}] missing required keys: {missing}")

        # simple string fields
        for key in ("assetId", "id", "object_name", "roomId"):
            if not isinstance(item[key], str):
                raise TypeError(f"layout[{idx}].{key} must be str")

        # position
        pos = item["position"]
        if not isinstance(pos, dict) or not all(k in pos for k in ("x", "y", "z")):
            raise TypeError(f"layout[{idx}].position must be dict with keys 'x','y','z'")
        if not all(_is_number(pos[k]) for k in ("x", "y", "z")):
            raise TypeError(f"layout[{idx}].position values must be numbers")

        # rotation
        rot = item["rotation"]
        if not isinstance(rot, dict) or not all(k in rot for k in ("x", "y", "z")):
            raise TypeError(f"layout[{idx}].rotation must be dict with keys 'x','y','z'")
        if not all(_is_number(rot[k]) for k in ("x", "y", "z")):
            raise TypeError(f"layout[{idx}].rotation values must be numbers")

        # vertices: list of 2D points
        verts = item["vertices"]
        if not isinstance(verts, list):
            raise TypeError(f"layout[{idx}].vertices must be a list")
        if any(
            not isinstance(t, (list, tuple))
            or len(t) != 2
            or not all(_is_number(v) for v in t)
            for t in verts
        ):
            raise TypeError(
                f"layout[{idx}].vertices must be List[Tuple[float, float]]"
            )

def validate_object_plan(plan: Dict[str, Any]) -> None:
    """Validate an ObjectPlan at runtime.

    Requirements for each entry (key -> value):
      - key: str (object name)
      - value: dict with required keys:
          'object_name', 'description', 'location', 'size',
          'quantity', 'variance_type', 'importance', 'objects_on_top'
      - value['object_name'] must equal the key
      - description: str
      - location: one of {'floor', 'wall'}
      - size: None OR List of length 3 with numbers (int or float)
      - quantity: int >= 1
      - variance_type: one of {'same', 'varied'}
      - importance: number (int or float)
      - objects_on_top: list of child dicts, each with:
          'object_name': str, 'quantity': int >= 1,
          'variance_type': {'same','varied'}, 'importance': number
    """
    if not isinstance(plan, dict):
        raise TypeError("object plan must be a dict mapping name -> details")

    required_keys = {
        "object_name",
        "description",
        "location",
        "size",
        "quantity",
        "variance_type",
        "importance",
        "objects_on_top",
    }

    for obj_key, details in plan.items():
        if not isinstance(obj_key, str):
            raise TypeError("object plan keys must be strings (object names)")
        if not isinstance(details, dict):
            raise TypeError(f"object '{obj_key}' details must be a dict")

        missing = [k for k in required_keys if k not in details]
        if missing:
            raise ValueError(f"object '{obj_key}' missing required keys: {missing}")

        # object_name must match key
        if not isinstance(details["object_name"], str):
            raise TypeError(f"object '{obj_key}': object_name must be str")
        if details["object_name"] != obj_key:
            raise ValueError(
                f"object '{obj_key}': object_name '{details['object_name']}' must equal the dict key"
            )

        # description
        if not isinstance(details["description"], str):
            raise TypeError(f"object '{obj_key}': description must be str")

        # location
        loc = details["location"]
        if loc not in ("floor", "wall"):
            raise ValueError(f"object '{obj_key}': location must be 'floor' or 'wall'")

        # size
        size = details["size"]
        if size is not None:
            if not isinstance(size, list) or len(size) != 3 or not all(_is_number(x) for x in size):
                raise TypeError(
                    f"object '{obj_key}': size must be None or List[3] of numbers"
                )

        # quantity
        qty = details["quantity"]
        if not isinstance(qty, int):
            raise TypeError(f"object '{obj_key}': quantity must be int")
        if qty < 1:
            raise ValueError(f"object '{obj_key}': quantity must be >= 1")

        # variance_type
        vt = details["variance_type"]
        if vt not in ("same", "varied"):
            raise ValueError(
                f"object '{obj_key}': variance_type must be 'same' or 'varied'"
            )

        # importance
        if not _is_number(details["importance"]):
            raise TypeError(f"object '{obj_key}': importance must be a number")

        # objects_on_top
        oot = details["objects_on_top"]
        if not isinstance(oot, list):
            raise TypeError(f"object '{obj_key}': objects_on_top must be a list")
        for idx, child in enumerate(oot):
            if not isinstance(child, dict):
                raise TypeError(
                    f"object '{obj_key}' objects_on_top[{idx}] must be a dict"
                )
            for ck in ("object_name", "quantity", "variance_type", "importance"):
                if ck not in child:
                    raise ValueError(
                        f"object '{obj_key}' objects_on_top[{idx}] missing key: {ck}"
                    )
            if not isinstance(child["object_name"], str):
                raise TypeError(
                    f"object '{obj_key}' objects_on_top[{idx}].object_name must be str"
                )
            cqty = child["quantity"]
            if not isinstance(cqty, int):
                raise TypeError(
                    f"object '{obj_key}' objects_on_top[{idx}].quantity must be int"
                )
            if cqty < 1:
                raise ValueError(
                    f"object '{obj_key}' objects_on_top[{idx}].quantity must be >= 1"
                )
            if child["variance_type"] not in ("same", "varied"):
                raise ValueError(
                    f"object '{obj_key}' objects_on_top[{idx}].variance_type must be 'same' or 'varied'"
                )
            if not _is_number(child["importance"]):
                raise TypeError(
                    f"object '{obj_key}' objects_on_top[{idx}].importance must be a number"
                )
