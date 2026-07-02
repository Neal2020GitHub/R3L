"""
The objective that the optimizer minimizes.

compile (in compile.py) turns a constraint JSON into a CompiledConstraints. It works
much like re.compile, which compiles a pattern into a re.Pattern. Build it once, then
call evaluate on a layout to get its loss.

It is frozen. evaluate never changes it and never touches disk.

Frames are handled elsewhere. scene.localize and scene.globalize move poses between
the global and local frames.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from utils.r3l.types import PoseVec, ParamVec, BBoxVec, LossTerm, ParamTable
from solvers.r3l.cluster import SceneIndex, AugmentedState
from solvers.r3l.physics import physics, regularize


@dataclass(frozen=True)
class CompiledConstraints:
    """
    The compiled objective. compile builds it. Call evaluate to score a layout. It is
    frozen, thus calling `evaluate()` has no side effects.
    """
    scene: SceneIndex                 # the fixed scene structure
    constraints: Dict[str, LossTerm]  # one loss term per named constraint
    params: ParamTable                # the optimizable Var parameters
    bbox_vec: BBoxVec                 # one bounding box per object
    room_size: Tuple[float, float]    # room width and height, in meters
    save_dir: str                     # where artifacts are written
    spec: dict                        # the original constraint JSON, kept for plots

    def evaluate(
        self,
        poses: PoseVec,
        alpha: float,
        params: Optional[ParamVec] = None,
        *,
        reparam: bool = True,
        train_var: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Score a layout. Returns the total loss and a per-term breakdown.

        - poses      must be in the frame given by reparam
        - alpha      weights the physics terms
        - params     the current Var values (only used when train_var is True)
        - train_var  set True to also train the Var parameters
        """
        aug = AugmentedState.build(self.scene, poses, self.bbox_vec, reparam=reparam)

        loss, nominal = physics(aug, self.scene, self.room_size, alpha, reparam=reparam)

        for name, term in self.constraints.items():
            l, n = term.evaluate(aug, params)
            loss = loss + l
            nominal[name] = n

        reg_loss, reg_nominal = regularize(params, self.params, train_var=train_var)
        loss = loss + reg_loss
        nominal.update(reg_nominal)

        assert loss.shape == ()
        return loss, nominal
