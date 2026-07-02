from typing import List, Tuple
import torch

from solvers.r3l import builders
from tests.lib import CaseScaffold

assets: List[str] = [
    "2bec242684c54e1f8574b579dc10a0e7-0",  # armchair
    "2bec242684c54e1f8574b579dc10a0e7-1",  # armchair
]

floor: Tuple[float, float] = (5.0, 6.0)
wall: float = 2.7


def build(scaffold: CaseScaffold, device: str):
    scaffold.constraints = {
        'horizontal_abs_loss': builders.make_horizontal_abs(
            index=torch.tensor([0, 1], device=device),
            x=torch.tensor([0.0, 5.0], device=device),
            room_size=scaffold.room_size,
        ),
    }


