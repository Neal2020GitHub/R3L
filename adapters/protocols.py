from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TypedDict
from itertools import chain

from utils.holodeck_v2.types import (
    WallsDict,
    RoomDict,
    AssetList,
    ObjectPlan,
    validate_room_dict,
    validate_walls_dict,
    validate_object_plan,
)


class HolodeckObjects(TypedDict):
    floor: AssetList
    wall: AssetList
    small: Dict[str, AssetList]


@dataclass(frozen=True)
class HolodeckBlueprint:
    """
    Holodeck object plan & selection.
    1st stage of holodeck.
    """
    query: str
    objects: HolodeckObjects
    walls: WallsDict
    room: RoomDict
    design: str
    plan: ObjectPlan

    def __post_init__(self):
        validate_room_dict(self.room)
        validate_walls_dict(self.walls)
        validate_object_plan(self.plan)
        self._validate_objects()
        if not isinstance(self.query, str):
            raise ValueError(f"`query` must be a string")

    def _validate_objects(self):
        for obj_types in ('floor', 'wall', 'small'):
            if obj_types not in self.objects:
                raise ValueError(f"`{obj_types}` not found in objects")
        if not isinstance(self.objects['floor'], list):
            raise ValueError(f"`objects['floor']` must be a list, got {type(self.objects['floor'])}")
        if not isinstance(self.objects['wall'], list):
            raise ValueError(f"`objects['wall']` must be a list, got {type(self.objects['wall'])}")
        if not isinstance(self.objects['small'], dict):
            raise ValueError(f"`objects['small']` must be a Dict, got {type(self.objects['small'])}")
        for obj_type in self.objects['small']:
            if not isinstance(self.objects['small'][obj_type], list):
                raise ValueError(f"`objects['small'][{obj_type}]` must be a list, got {type(self.objects['small'][obj_type])}")

    def get_asset_ids(self) -> List[str]:
        """
        Return list of all asset-ids
        May contain duplicates
        """
        assets = []
        for _, aid in chain(
            self.objects["floor"],
            self.objects["wall"],
            (item for al in self.objects["small"].values() for item in al),
        ):
            assets.append(aid)
        return assets

    def get_asset_names(self) -> List[str]:
        """
        Return list of all asset-ids
        May contain duplicates
        """
        assets = []
        for name, _ in chain(
            self.objects["floor"],
            self.objects["wall"],
            (item for al in self.objects["small"].values() for item in al),
        ):
            assets.append(name)
        return assets

    def to_scene_dict(self, solution: RendererSolution) -> dict:
        """The rich Holodeck scene.json: the full plan/selection plus the solved
        layout. Distinct from R3LBlueprint's schema; the two are NOT unified."""
        return {
            "query": self.query,
            "objects": self.objects,
            "walls": self.walls,
            "room": self.room,
            "plan": self.plan,
            "layout": solution.to_layout_dict(),
        }


@dataclass
class RendererSolution:
    asset_ids: List[str]
    positions: List[Tuple[float, float, float]]
    rotations: List[Tuple[float, float, float]]
    floor_vertices: List[Tuple[float, float]]
    wall_height: float
    asset_names: List[str] = field(default_factory=list)
    scales: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        if not self.asset_names:
            self.asset_names = [x[:6] for x in self.asset_ids]
        if not self.scales:
            self.scales = [(1., 1., 1.) for _ in self.asset_ids]

        self._validate_length()
        self._validate_floor_vertices()
        self._validate_wall_height()

    def _validate_length(self):
        asset_ids_len = len(self.asset_ids)
        flag: bool = True
        flag = flag and asset_ids_len == len(self.positions)
        flag = flag and asset_ids_len == len(self.rotations)
        flag = flag and (self.asset_names is None or asset_ids_len == len(self.asset_names))
        flag = flag and (self.scales is None or asset_ids_len == len(self.scales))
        if not flag:
            raise ValueError(f"List parameters length mismatch")
        if (self.wall_height is not None and not isinstance(self.wall_height, float)):
            raise ValueError(f"wall_height must be a float")


    def _validate_floor_vertices(self):
        for vertex in self.floor_vertices:
            if not isinstance(vertex, tuple) or len(vertex) != 2 or not all(isinstance(v, float) for v in vertex):
                raise ValueError(f"Floor vertex must be a tuple of (x, y) floats. Got: {vertex}")


    def _validate_wall_height(self):
        assert isinstance(self.wall_height, float)
        if self.wall_height <= 0 or self.wall_height > 5:
            raise ValueError(f"Wall height must be between 0-5")

    def to_layout_dict(self) -> Dict[str, dict]:
        """Serialize the solved placements as {asset_name: {asset_id, position,
        rotation}} — the `layout` block shared by every scene.json schema."""
        return {
            self.asset_names[i]: {
                "asset_id": self.asset_ids[i],
                "position": self.positions[i],
                "rotation": self.rotations[i],
            }
            for i in range(len(self.asset_ids))
        }


@dataclass(frozen=True)
class R3LBlueprint:
    prompt: str
    design: str
    assets: Dict[str, dict] # asset tree
    room_size: Tuple[float, float]

    def get_asset_ids(self) -> List[str]:
        """Unique asset UIDs to download: walk the nested asset tree, strip each
        instance's '-N' suffix, and dedup (order-preserving)."""
        seen: Dict[str, None] = {}
        def walk(node: Dict[str, dict]) -> None:
            for instance_id, children in node.items():
                seen.setdefault(instance_id.split("-")[0], None)
                walk(children)
        walk(self.assets)
        return list(seen)

    def to_scene_dict(self, solution: RendererSolution) -> dict:
        """The round-trippable R3L scene.json: prompt/design/assets/room_size plus
        the solved layout. Distinct from HolodeckBlueprint's schema; not unified."""
        return {
            "prompt": self.prompt,
            "design": self.design,
            "assets": self.assets,
            "room_size": self.room_size,
            "wall_height": 2.5,  # dummy value, r3l doesn't use wall_height
            "layout": solution.to_layout_dict(),
        }


@dataclass
class R3LSolution:
    @dataclass
    class Placement:
        asset_id: str
        position: Tuple[float, float, float]
        rotation: Tuple[float, float, float]

    layout: Dict[str, Placement]
    room_size: Tuple[float, float]

    @classmethod
    def build(cls, data: dict) -> 'R3LSolution':
        layout = {
            oid: cls.Placement(
                asset_id=info['asset_id'],
                position=info['position'],
                rotation=info['rotation'],
            )
            for oid, info in data['layout'].items()
        }
        return cls(layout=layout, room_size=data['room_size'])
