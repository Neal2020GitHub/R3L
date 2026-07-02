from typing import List, Tuple
import torch

from solvers.r3l import builders
from tests.lib import CaseScaffold

assets: List[str] = [
    "5bb16d79c1254a20ae9acbd1623f20bf-0",  # 0. sectional sofa
    "2bec242684c54e1f8574b579dc10a0e7-0",  # 1. armchair
    "9dbe2d62e0ea466ab7788648520b558c-0",  # 2. bed
    "fadc50ddecab4a2eb4d7f8f72b9c6b32-0",  # 3. wardrobe
    "95819214fa3c429d9cec19b0a44291ed-0",  # 4. gaming desk
    "4112375ac717454d9749d13cf8a04e4d-0",  # 5. gaming chair
    "f9eedc7271614e17b4282384d39a6d29-0",  # 6. table
]

floor: Tuple[float, float] = (6.5, 7.5)
wall: float = 2.7


def build(scaffold: CaseScaffold, device: str):
    scaffold.constraints = {
        'facing_loss': builders.make_facing_ortho(
            source_index=torch.tensor([4, 5], device=device),
            target_index=torch.tensor([5, 4], device=device),
        ),
        # angle_loss (signed yaw offset, positive = CCW)
        "angle_loss": builders.make_angle(
            source_index=torch.tensor([1], device=device),  # table
            target_index=torch.tensor([0], device=device),  # sectional sofa
            angle_deg=torch.tensor([15.0], device=device),  # +15 deg CCW
        ),
        'corner_loss': builders.make_corner(
            index=torch.tensor([0, 2], device=device),
            corner_index=torch.tensor([3, 0], device=device),  # TL, BL
            wall_index=torch.tensor([0, 3], device=device),    # L, B
            room_size=scaffold.room_size,
        ),
        'against_wall_loss': builders.make_against_wall(
            index=torch.tensor([3, 2, 4], device=device),
            wall_index=torch.tensor([3, 3, 1], device=device),
            room_size=scaffold.room_size,
        ),
        'gap_loss': builders.make_gap(
            source_index=torch.tensor([5, 6, 1], device=device),
            target_index=torch.tensor([4, 0, 0], device=device),
            gap=torch.tensor([.2, .3, .3], device=device),
            gamma=1.0,
        ),
        "right_loss": builders.make_right_of(
            source_index=torch.tensor([3, 1], device=device),
            target_index=torch.tensor([2, 0], device=device),
            percentile=torch.tensor([0.5, 0.5], device=device),
        ),
        "infront_loss": builders.make_in_front_of(
            source_index=torch.tensor([6, 5], device=device),
            target_index=torch.tensor([0, 4], device=device),
            percentile=torch.tensor([0.5, 0.5], device=device),
        ),
    }


