from typing import List, Tuple
import torch

from solvers.r3l import builders
from tests.lib import CaseScaffold


assets: List[str] = [
    "f9eedc7271614e17b4282384d39a6d29-0",  # table
    "2bec242684c54e1f8574b579dc10a0e7-0",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-1",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-2",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-3",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-4",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-5",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-6",  # chair
    "2bec242684c54e1f8574b579dc10a0e7-7",  # chair
]

floor: Tuple[float, float] = (12.0, 12.0)
wall: float = 2.7


def build(scaffold: CaseScaffold, device: str):
    scaffold.constraints = {
        'facing_loss': builders.make_facing_radial(
            source_index=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device),
            target_index=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], device=device),
        ),
        'face_wall': builders.make_facing_wall(
            source_index=torch.tensor([0], device=device),
            wall_index=torch.tensor([0], device=device),
        ),
        'equidistant_loss': builders.make_distance(
            source_index=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device),
            target_index=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0], device=device),
            distance=torch.tensor([3., 3., 3., 3., 3., 3., 3., 3.], device=device),
        ),
        'around_loss': builders.make_around(
            source_index=torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device),
            target_index=torch.tensor([0], device=device),
            sweep_deg=torch.tensor([180.0], device=device),
            center_rad=torch.tensor([3.14], device=device),
        ),
    }



