from typing import List, Tuple
import torch

from solvers.r3l import builders
from tests.lib import CaseScaffold


assets: List[str] = [
    "5bb16d79c1254a20ae9acbd1623f20bf-0",  # sectional sofa
    "9dbe2d62e0ea466ab7788648520b558c-0",  # bed
    "fadc50ddecab4a2eb4d7f8f72b9c6b32-0",  # wardrobe
]

floor: Tuple[float, float] = (5.5, 6.5)
wall: float = 2.7


def build(scaffold: CaseScaffold, device: str):
    scaffold.constraints = {
        'corner_loss': builders.make_corner(
            index=torch.tensor([0, 1], device=device),
            corner_index=torch.tensor([3, 0], device=device),  # TL, BL
            wall_index=torch.tensor([0, 3], device=device),    # L, B
            room_size=scaffold.room_size,
        ),
        'against_wall_loss': builders.make_against_wall(
            index=torch.tensor([2], device=device),
            wall_index=torch.tensor([3], device=device),
            room_size=scaffold.room_size,
        ),
        'facing_loss': builders.make_facing_ortho(
            source_index=torch.tensor([0], device=device),
            target_index=torch.tensor([1], device=device),
        ),
    }



