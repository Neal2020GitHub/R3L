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
        'gap_loss': builders.make_gap(
            source_index=torch.tensor([0], device=device),
            target_index=torch.tensor([1], device=device),
            gap=torch.tensor([1.0], device=device),
            gamma=1.0,
        ),
    }


