import torch
from dataclasses import dataclass
from typing import Tuple, List, TypedDict, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Only for type checking to avoid runtime circular import: cluster imports
    # types at runtime, types references cluster only under TYPE_CHECKING.
    from solvers.r3l.cluster import AugmentedState


class BBox(TypedDict):
    x: float
    y: float
    z: float


@dataclass
class BBoxVec: 
    x: torch.Tensor # (N,)
    y: torch.Tensor # (N,)
    z: torch.Tensor # (N,)

    def __post_init__(self): 
        assert self.x.shape == self.y.shape == self.z.shape
        assert self.x.ndim == 1

    @staticmethod
    def build(asset_dict: Dict[str, 'AssetInfo'], asset_id_list: List[str], device: str) -> 'BBoxVec': 
        f32 = torch.float32
        return BBoxVec(
            x=torch.tensor([asset_dict[get_uid(aid)].bbox['x'] for aid in asset_id_list], dtype=f32),
            y=torch.tensor([asset_dict[get_uid(aid)].bbox['y'] for aid in asset_id_list], dtype=f32),
            z=torch.tensor([asset_dict[get_uid(aid)].bbox['z'] for aid in asset_id_list], dtype=f32),
        ).to(device)

    def to(self, device: str) -> 'BBoxVec':
        return BBoxVec(
            x=self.x.to(device),
            y=self.y.to(device),
            z=self.z.to(device),
        )
    
    def detach(self) -> 'BBoxVec':
        return BBoxVec(
            x=self.x.detach(),
            y=self.y.detach(),
            z=self.z.detach(),
        )


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    rz: float


@dataclass(frozen=True)
class PoseVec: 
    x: torch.Tensor # x coordinates
    y: torch.Tensor # y coordinates
    rz: torch.Tensor # radian angles

    def __post_init__(self): 
        assert self.x.shape == self.y.shape == self.rz.shape
        assert self.x.ndim == 1

    def to(self, device: str) -> 'PoseVec': 
        return PoseVec(
            x=self.x.to(device),
            y=self.y.to(device),
            rz=self.rz.to(device),
        )
    
    def detach(self) -> 'PoseVec': 
        return PoseVec(
            x=self.x.detach(),
            y=self.y.detach(),
            rz=self.rz.detach(),
        )
    
    def clone(self) -> 'PoseVec': 
        return PoseVec(
            x=self.x.clone(),
            y=self.y.clone(),
            rz=self.rz.clone(),
        )

    def select(self, index: torch.Tensor) -> 'PoseVec':
        return PoseVec(
            x=torch.index_select(self.x, -1, index),
            y=torch.index_select(self.y, -1, index),
            rz=torch.index_select(self.rz, -1, index),
        )

    @classmethod
    def from_pose_dict(cls, pose_dict: Dict[str, Pose], device: str) -> 'PoseVec':
        assert isinstance(pose_dict, dict)
        assert all([isinstance(v, Pose) for v in pose_dict.values()])
        assert all([isinstance(k, str) for k in pose_dict.keys()])
        f32 = torch.float32
        return cls(
            x=torch.tensor([v.x for v in pose_dict.values()], dtype=f32),
            y=torch.tensor([v.y for v in pose_dict.values()], dtype=f32),
            rz=torch.tensor([v.rz for v in pose_dict.values()], dtype=f32),
        ).to(device)

    def to_pose_dict(self, asset_ids: List[str]) -> Dict[str, Pose]:
        x, y, rz = self.x.tolist(), self.y.tolist(), self.rz.tolist()
        return {aid: Pose(x=x[i], y=y[i], rz=rz[i]) for i, aid in enumerate(asset_ids)}


@dataclass(frozen=True)
class ParamVec:
    """
    Vector of optimizable constraint parameters.
    
    Values are raw (unclamped) during optimization.
    Clamping is applied at evaluation time based on param kinds.
    """
    values: torch.Tensor  # (P,) raw parameter values

    def __post_init__(self):
        assert self.values.ndim == 1

    def to(self, device: str) -> 'ParamVec':
        return ParamVec(values=self.values.to(device))

    def detach(self) -> 'ParamVec':
        return ParamVec(values=self.values.detach())

    def select(self, index: torch.Tensor) -> torch.Tensor:
        """Select parameter values by index."""
        return torch.index_select(self.values, -1, index)

    def clone(self) -> 'ParamVec':
        return ParamVec(values=self.values.clone())

    def __len__(self) -> int:
        return self.values.shape[0]

    @classmethod
    def from_priors(cls, priors: List[float], device: str) -> 'ParamVec':
        """Create ParamVec from prior values."""
        f32 = torch.float32
        return cls(values=torch.tensor(priors, dtype=f32).to(device))


@dataclass(frozen=True)
class ParamTable:
    """
    Compile-time schema for a scene's optimizable constraint parameters.

    Holds the three parallel arrays declared by `Var(...)` in the DSL: ordered
    `names`, their numeric `priors`, and their clamp `kinds` ("unit" | "nonneg" |
    "angle_deg"). It is the immutable counterpart to the live `ParamVec`: builders
    bake `kind_of(idx)` into their closures at compile time, and the prior loss
    reads `priors`/`kinds`. `bool(table)` is True iff the scene has any params.
    """
    names: List[str]
    priors: List[float]
    kinds: List[str]

    def __bool__(self) -> bool:
        return len(self.names) > 0

    def kind_of(self, idx: int) -> str:
        """Clamp kind of the param at the given index."""
        return self.kinds[idx]


@dataclass
class AssetInfo:
    name: str
    desc_short: str
    desc_long: str
    bbox: BBox # x, y, z


def get_uid(asset_id: str) -> str:
    """Asset instance id -> its AssetInfo key (drops the -N instance suffix).

    e.g. "1234567890abcdefg-1" -> "1234567890abcdefg".
    """
    return asset_id.split("-")[0]


@dataclass
class LossTerm:
    """
    A single differentiable term in the optimization objective.

    Wraps an evaluate function that computes (loss, nominal_loss) for an
    `AugmentedState` and optional params. Multiple LossTerms are summed to form
    the total constraint loss.
    """
    evaluate_fn: Callable[['AugmentedState', Optional['ParamVec']], Tuple[torch.Tensor, torch.Tensor]]

    def evaluate(self, aug: 'AugmentedState', params: Optional['ParamVec'] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, nominal_loss = self.evaluate_fn(aug, params)
        return loss, nominal_loss
