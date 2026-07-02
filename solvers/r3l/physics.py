"""
The built-in objective terms: scene physics + the Var-param regularizer.

These are the two loss contributions that exist independently of any compiled
constraint JSON, so they live apart from the `CompiledConstraints` product
(constraints.py) and are imported into its `evaluate`:

- `physics`    -- the frame-aware physical priors over the augmented entity set
                  (collision over the global + per-cluster layers, a wall prior,
                  and the aesthetics snap).
- `regularize` -- pulls each optimizable (Var) param toward its compile-time prior.

This module depends only on the cluster paradigm (`AugmentedState`/`SceneIndex`)
and the leaf loss kernels; it must NOT import constraints.py (the dependency runs
constraints -> physics, never the reverse).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.r3l.types import ParamVec, ParamTable
from utils.r3l.geometry import get_soft_landing
from solvers.r3l.cluster import SceneIndex, AugmentedState
from solvers.r3l.losses import collision_loss, pack_boxes, wall_loss, aesthetics_loss, get_act
from solvers.r3l.config import cfg


def physics(
    aug: AugmentedState,
    scene: SceneIndex,
    room_size: Tuple[float, float],
    alpha: float,
    *,
    reparam: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    The frame-aware physical priors over the augmented entity set.

    Two collision layers (global handles + per-cluster members), a wall prior on
    the global layer, and the aesthetics snap. K=0 collapses to a single global
    layer with `coll_local = 0`, so the total matches the old no-cluster path bit
    for bit (only the nominal key names split: `coll` -> `coll_global`+`coll_local`).

    The single residual `reparam` branch is the aesthetics rotation source: in the
    reparam frame, cluster members are scored at their LOCAL rotation (already
    zeroed against the anchor in `aug.poses`); in the no_localize frame every
    object is scored at its GLOBAL rotation in one shot.
    """
    device = cfg.runtime.device
    poses, bbox = aug.poses, aug.bbox
    gidx = aug.global_indices

    nominal: Dict[str, torch.Tensor] = {}
    method=cfg.constraints.shapes.collision.method
    loss = torch.tensor(0.0, device=device)

    # Global collision: independent objects + cluster handles, all pairs
    if gidx.numel() > 1:
        boxes = pack_boxes(
            position_x=poses.x[gidx], 
            position_y=poses.y[gidx], 
            bbox_x=bbox.x[gidx], 
            bbox_y=bbox.y[gidx], 
            rotation=poses.rz[gidx]
        )
        g = gidx.numel()
        pairs = torch.triu_indices(g, g, 1, device=device)
        boxes1, boxes2 = boxes[pairs[0]], boxes[pairs[1]]
        l_global, l_global_nom = collision_loss(boxes1, boxes2, method, alpha)
        loss = loss + l_global
        nominal['coll_global'] = l_global_nom
    else:
        nominal['coll_global'] = torch.tensor(0.0, device=device)

    # Local collision: within-cluster members, one block-diagonal call
    if scene.local_pairs.shape[1] > 0:
        n = scene.N
        boxes = pack_boxes(
            position_x=poses.x[:n], 
            position_y=poses.y[:n], 
            bbox_x=bbox.x[:n], 
            bbox_y=bbox.y[:n], 
            rotation=poses.rz[:n]
        )
        pairs = scene.local_pairs
        boxes1, boxes2 = boxes[pairs[0]], boxes[pairs[1]]
        l_local, l_local_nom = collision_loss(boxes1, boxes2, method, alpha)
        loss = loss + l_local
        nominal['coll_local'] = l_local_nom
    else:
        nominal['coll_local'] = torch.tensor(0.0, device=device)

    # Wall prior (global layer only)
    if gidx.numel() > 0:
        l_wall, l_wall_nom = wall_loss(
            bbox_x=bbox.x[gidx], bbox_y=bbox.y[gidx],
            position_x=poses.x[gidx], position_y=poses.y[gidx], rotation=poses.rz[gidx],
            room_size=room_size,
            act=get_act(cfg.constraints.shapes.wall.act),
            weight=alpha,
        )
        loss = loss + l_wall
        nominal['wall'] = l_wall_nom
    else:
        nominal['wall'] = torch.tensor(0.0, device=device)

    # Aesthetics snap
    l_aest = torch.tensor(0.0, device=device)
    l_aest_nom = torch.tensor(0.0, device=device)
    if cfg.constraints.shapes.aesthetics.enabled:
        acfg = cfg.constraints.shapes.aesthetics
        angles = acfg.angles_deg
        eps = float(acfg.eps_deg)
        sl = get_soft_landing(acfg.soft_landing)

        def snap(rotation: torch.Tensor) -> None:
            nonlocal l_aest, l_aest_nom
            if rotation.numel() == 0:
                return
            l, n = aesthetics_loss(
                rotation=rotation, angles_deg=angles, eps_deg=eps,
                soft_landing=sl, weight=alpha,
            )
            l_aest = l_aest + l
            l_aest_nom = l_aest_nom + n

        if reparam:
            # Global handles use their (global) rotation; each cluster's non-anchor
            # members are judged at their LOCAL rotation (anchor-relative).
            snap(poses.rz[gidx])
            for cid in scene.sorted_cids:
                snap(poses.rz[scene.clusters[cid].non_anchor_indices])
        else:
            # No_localize: every object's rotation is already global -> one snap.
            snap(poses.rz[: scene.N])
    loss = loss + l_aest
    nominal['aesthetics'] = l_aest_nom

    return loss, nominal


def regularize(
    params: Optional[ParamVec],
    table: ParamTable,
    *,
    train_var: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Regularize each optimizable (Var) param toward its prior, grouped by kind.

    Only contributes when actually training Var params (`train_var=True` with a
    non-None `params` and a non-empty table); otherwise returns a zero with no
    nominal keys.
    """
    device = cfg.runtime.device
    if params is None or not table or not train_var:
        return torch.tensor(0.0, device=device), {}

    lp = cfg.constraints.learnable_priors
    scale_prior = lp.scale_prior
    priors_t = torch.tensor(table.priors, dtype=torch.float32, device=device)

    total = torch.tensor(0.0, device=device)
    nominal: Dict[str, torch.Tensor] = {}
    for kind in ("unit", "nonneg", "angle_deg"):
        kind_cfg = getattr(lp, kind)
        indices = [i for i, k in enumerate(table.kinds) if k == kind]
        if not indices:
            continue
        idx_t = torch.tensor(indices, dtype=torch.long, device=device)
        raw_vals = params.values[idx_t]
        prior_vals = priors_t[idx_t]
        match kind_cfg.loss:
            case "mse": kind_loss = F.mse_loss(raw_vals, prior_vals, reduction='sum')
            case "l1": kind_loss = F.l1_loss(raw_vals, prior_vals, reduction='sum')
            case "smooth_l1": kind_loss = F.smooth_l1_loss(raw_vals, prior_vals, reduction='sum')
            case "circular_deg":
                diff_rad = (raw_vals - prior_vals) * (torch.pi / 180.0)
                kind_loss = (1.0 - torch.cos(diff_rad)).sum()
            case _:
                raise ValueError(f"unknown regularizer loss: {kind_cfg.loss}")
        total = total + kind_cfg.weight * scale_prior * kind_loss
        nominal[f"prior_{kind}"] = kind_loss.detach()

    return total, nominal
