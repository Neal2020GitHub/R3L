from typing import List, Tuple
import torch

from solvers.r3l import builders
from tests.lib import CaseScaffold


assets: List[str] = [
    "5bb16d79c1254a20ae9acbd1623f20bf-0",  # sectional sofa
    "2bec242684c54e1f8574b579dc10a0e7-0",  # armchair
    "f9eedc7271614e17b4282384d39a6d29-0",  # table
    "2bec242684c54e1f8574b579dc10a0e7-1",  # chair
    "fadc50ddecab4a2eb4d7f8f72b9c6b32-0",  # wardrobe
]

floor: Tuple[float, float] = (7.0, 8.0)
wall: float = 2.7


def build(scaffold: CaseScaffold, device: str):
    scaffold.constraints = {
        'left_loss': builders.make_left_of(
            source_index=torch.tensor([1], device=device),
            target_index=torch.tensor([0], device=device),
            percentile=torch.tensor([1.0], device=device),
        ),
        'right_loss': builders.make_right_of(
            source_index=torch.tensor([2], device=device),
            target_index=torch.tensor([0], device=device),
            percentile=torch.tensor([1.0], device=device),
        ),
        'infront_loss': builders.make_in_front_of(
            source_index=torch.tensor([3], device=device),
            target_index=torch.tensor([0], device=device),
            percentile=torch.tensor([0.0], device=device),
        ),
        'behind_loss': builders.make_behind_of(
            source_index=torch.tensor([4], device=device),
            target_index=torch.tensor([0], device=device),
            percentile=torch.tensor([1.0], device=device),
        ),
        'gap_loss': builders.make_gap(
            source_index=torch.tensor([1, 2, 3, 4], device=device),
            target_index=torch.tensor([0, 0, 0, 0], device=device),
            gap=torch.tensor([1., 1., 1., 1.], device=device),
        ),
        'facing_wall_loss': builders.make_facing_wall(
            source_index=torch.tensor([0], device=device),
            wall_index=torch.tensor([1], device=device),
        ),
    }


