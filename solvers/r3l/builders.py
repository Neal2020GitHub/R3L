"""
Builders turn a loss kernel into a LossTerm.

A kernel (in losses.py) is plain math. A builder binds one constraint's fixed data
onto a kernel and returns a LossTerm. compile calls the builders while it reads the
JSON.

- a normal constraint  fixes its value at compile time
- a Var constraint     reads its value from params each step, then clamps it
"""

from typing import Callable, Optional, Tuple

import torch

from utils.r3l.types import ParamVec, LossTerm
from solvers.r3l.cluster import AugmentedState
from solvers.r3l.config import cfg
from solvers.r3l.losses import (
    select,
    get_act,
    clamp_param,
    facing_radial_loss,
    facing_ortho_loss,
    facing_wall_loss,
    corner_loss,
    horizontal_rel_loss,
    horizontal_abs_loss,
    vertical_rel_loss,
    vertical_abs_loss,
    distance_loss,
    against_wall_loss,
    gap_loss_l2,
    gap_loss_sdf,
    gap_loss_mtv,
    left_of_loss,
    right_of_loss,
    in_front_of_loss,
    behind_of_loss,
    around_loss,
    angle_loss,
)

# Type alias for the value type returned by every kernel.
_LossPair = Tuple[torch.Tensor, torch.Tensor]


def _resolve_param(
    params: Optional[ParamVec],
    param_idx: Optional[torch.Tensor],
    param_kind: Optional[str],
    constant: torch.Tensor,
    constr_type: str,
) -> torch.Tensor:
    """
    Pick the effective value for a (possibly variable) constraint field.

    Variable mode (`param_idx` is not None): read the live raw values from
    `params` and clamp by the compile-time `param_kind`. Constant mode: return
    the value baked in at parse time.
    """
    if param_idx is None:
        return constant
    if params is None:
        raise ValueError(f"{constr_type} constraint requires params in variable mode")
    assert param_kind is not None  # bound together with param_idx at compile time
    raw = params.values[param_idx]
    return clamp_param(raw, param_kind)


def _make_facing(
    kernel: Callable[..., _LossPair],
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> LossTerm:
    """Bind a facing kernel (radial or ortho) onto the shared source/target pose selection."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        return kernel(
            select(layout.rz, source_index),
            select(layout.rz, target_index),
            select(layout.x, source_index),
            select(layout.x, target_index),
            select(layout.y, source_index),
            select(layout.y, target_index),
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_facing_radial(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> LossTerm:
    """Source entities point their front at the target's centroid (radial)."""
    return _make_facing(facing_radial_loss, source_index, target_index)


def make_facing_ortho(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> LossTerm:
    """Source entities face perpendicular to the target's nearest edge."""
    return _make_facing(facing_ortho_loss, source_index, target_index)


def make_facing_wall(
    source_index: torch.Tensor,
    wall_index: torch.Tensor,
) -> LossTerm:
    """Source entities face the named room wall."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        return facing_wall_loss(
            select(aug.poses.rz, source_index),
            wall_index,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_corner(
    index: torch.Tensor,
    corner_index: torch.Tensor,
    wall_index: torch.Tensor,
    room_size: Tuple[float, float],
) -> LossTerm:
    """
    Place each entity's selected edge at a named room corner.

    index: entities (objects or clusters) to constrain.
    corner_index: per-entity corner in {0:BL, 1:BR, 2:TR, 3:TL}.
    wall_index: per-entity wall to align with in {0:L, 1:R, 2:T, 3:B}.
    """
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        ci = corner_index.to(device=cfg.runtime.device, dtype=torch.long)
        wi = wall_index.to(device=cfg.runtime.device, dtype=torch.long)
        return corner_loss(
            select(layout.x, index),
            select(layout.y, index),
            select(layout.rz, index),
            ci,
            wi,
            select(aug.bbox.x, index),
            select(aug.bbox.y, index),
            room_size,
            w_pos=cfg.constraints.shapes.corner.w_pos,
            w_rot=cfg.constraints.shapes.corner.w_rot,
            loss_fn=cfg.constraints.shapes.corner.loss_fn,
            huber_delta=cfg.constraints.shapes.corner.huber_delta,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_horizontal_rel(
    index: torch.Tensor,
    percentile: torch.Tensor,
    room_size: Tuple[float, float],
    percentile_param_idx: Optional[torch.Tensor] = None,
    percentile_param_kind: Optional[str] = None,
) -> LossTerm:
    """Pin each entity to a horizontal room percentile (relative mode)."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_perc = _resolve_param(params, percentile_param_idx, percentile_param_kind, percentile, "horizontal")
        return horizontal_rel_loss(
            select(layout.x, index),
            select(layout.rz, index),
            select(aug.bbox.x, index),
            select(aug.bbox.y, index),
            percentile=eff_perc,
            room_size=room_size,
            loss_fn=cfg.constraints.shapes.horizontal.loss_fn,
            huber_delta=cfg.constraints.shapes.horizontal.huber_delta,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_horizontal_abs(
    index: torch.Tensor,
    x: torch.Tensor,
    room_size: Tuple[float, float],
    x_param_idx: Optional[torch.Tensor] = None,
    x_param_kind: Optional[str] = None,
) -> LossTerm:
    """Pin each entity to an absolute x coordinate."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        eff_x = _resolve_param(params, x_param_idx, x_param_kind, x, "horizontal_abs")
        return horizontal_abs_loss(
            select(aug.poses.x, index),
            x=eff_x,
            room_size=room_size,
            loss_fn=cfg.constraints.shapes.horizontal.loss_fn,
            huber_delta=cfg.constraints.shapes.horizontal.huber_delta,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_vertical_rel(
    index: torch.Tensor,
    percentile: torch.Tensor,
    room_size: Tuple[float, float],
    percentile_param_idx: Optional[torch.Tensor] = None,
    percentile_param_kind: Optional[str] = None,
) -> LossTerm:
    """Pin each entity to a vertical room percentile (relative mode)."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_perc = _resolve_param(params, percentile_param_idx, percentile_param_kind, percentile, "vertical")
        return vertical_rel_loss(
            select(layout.y, index),
            select(layout.rz, index),
            select(aug.bbox.x, index),
            select(aug.bbox.y, index),
            percentile=eff_perc,
            room_size=room_size,
            loss_fn=cfg.constraints.shapes.vertical.loss_fn,
            huber_delta=cfg.constraints.shapes.vertical.huber_delta,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_vertical_abs(
    index: torch.Tensor,
    y: torch.Tensor,
    room_size: Tuple[float, float],
    y_param_idx: Optional[torch.Tensor] = None,
    y_param_kind: Optional[str] = None,
) -> LossTerm:
    """Pin each entity to an absolute y coordinate."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        eff_y = _resolve_param(params, y_param_idx, y_param_kind, y, "vertical_abs")
        return vertical_abs_loss(
            select(aug.poses.y, index),
            y=eff_y,
            room_size=room_size,
            loss_fn=cfg.constraints.shapes.vertical.loss_fn,
            huber_delta=cfg.constraints.shapes.vertical.huber_delta,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_distance(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    distance: torch.Tensor,
    distance_param_idx: Optional[torch.Tensor] = None,
    distance_param_kind: Optional[str] = None,
) -> LossTerm:
    """Hold a target center-to-center distance between source and target."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_dist = _resolve_param(params, distance_param_idx, distance_param_kind, distance, "distance")
        return distance_loss(
            select(layout.x, source_index),
            select(layout.y, source_index),
            select(layout.x, target_index),
            select(layout.y, target_index),
            distance=eff_dist,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_against_wall(
    index: torch.Tensor,
    wall_index: torch.Tensor,
    room_size: Tuple[float, float],
) -> LossTerm:
    """
    Place each entity's selected edge against a named wall.

    wall_index: per-entity wall in {0:left, 1:right, 2:top, 3:bottom}.
    """
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        wi = wall_index.to(device=cfg.runtime.device, dtype=torch.long)
        return against_wall_loss(
            select(layout.x, index),
            select(layout.y, index),
            select(layout.rz, index),
            wi,
            select(aug.bbox.x, index),
            select(aug.bbox.y, index),
            room_size,
            w_pos=cfg.constraints.shapes.against_wall.w_pos,
            w_rot=cfg.constraints.shapes.against_wall.w_rot,
            loss_fn=cfg.constraints.shapes.against_wall.loss_fn,
            huber_delta=cfg.constraints.shapes.against_wall.huber_delta,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_gap(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    gap: torch.Tensor,
    gamma: float = 1.0,
    gap_param_idx: Optional[torch.Tensor] = None,
    gap_param_kind: Optional[str] = None,
) -> LossTerm:
    """Keep a clearance gap between source and target shapes."""
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_gap = _resolve_param(params, gap_param_idx, gap_param_kind, gap, "gap")
        px_s = select(layout.x, source_index)
        py_s = select(layout.y, source_index)
        bbx_s = select(aug.bbox.x, source_index)
        bby_s = select(aug.bbox.y, source_index)
        rz_s = select(layout.rz, source_index)
        px_t = select(layout.x, target_index)
        py_t = select(layout.y, target_index)
        bbx_t = select(aug.bbox.x, target_index)
        bby_t = select(aug.bbox.y, target_index)
        rz_t = select(layout.rz, target_index)
        method = cfg.constraints.shapes.gap.method
        match method:
            case "mtv":
                return gap_loss_mtv(
                    px_s, py_s, bbx_s, bby_s, rz_s,
                    px_t, py_t, bbx_t, bby_t, rz_t,
                    eff_gap,
                )
            case "sdf":
                return gap_loss_sdf(
                    px_s, py_s, bbx_s, bby_s, rz_s,
                    px_t, py_t, bbx_t, bby_t, rz_t,
                    eff_gap, gamma,
                )
            case "l2":
                return gap_loss_l2(
                    px_s, py_s, px_t, py_t, eff_gap
                )
            case _:
                raise ValueError(f"Invalid gap method: {method}")
    return LossTerm(evaluate_fn=evaluate_fn)


def _make_directional(
    kernel,
    constr_type: str,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    percentile: torch.Tensor,
    percentile_param_idx: Optional[torch.Tensor],
    percentile_param_kind: Optional[str],
    w_align_override: Optional[float] = None,
) -> LossTerm:
    """
    Shared body for the directional kernels (left/right/front/behind).

    `w_align_override` pins the lateral-alignment weight (used by the direction-only
    in_front_of, which sets it to 0); when None the cfg default applies. When
    `percentile_param_idx is None` the percentile is the constant baked in here, so
    the term reads no optimizable params.
    """
    act = get_act(cfg.constraints.shapes.directional.act)
    w_align = w_align_override if w_align_override is not None else cfg.constraints.shapes.directional.w_align
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_perc = _resolve_param(params, percentile_param_idx, percentile_param_kind, percentile, constr_type)
        return kernel(
            select(layout.x, source_index),
            select(layout.y, source_index),
            select(layout.x, target_index),
            select(layout.y, target_index),
            eff_perc,
            select(aug.bbox.x, source_index),
            select(aug.bbox.y, source_index),
            select(aug.bbox.x, target_index),
            select(aug.bbox.y, target_index),
            select(layout.rz, source_index),
            select(layout.rz, target_index),
            act=act,
            w_dir=cfg.constraints.shapes.directional.w_dir,
            w_align=w_align,
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_left_of(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    percentile: torch.Tensor,
    percentile_param_idx: Optional[torch.Tensor] = None,
    percentile_param_kind: Optional[str] = None,
) -> LossTerm:
    """Place source to the left of target with lateral percentile alignment."""
    return _make_directional(
        left_of_loss, "left_of", source_index, target_index,
        percentile, percentile_param_idx, percentile_param_kind,
    )


def make_right_of(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    percentile: torch.Tensor,
    percentile_param_idx: Optional[torch.Tensor] = None,
    percentile_param_kind: Optional[str] = None,
) -> LossTerm:
    """Place source to the right of target with lateral percentile alignment."""
    return _make_directional(
        right_of_loss, "right_of", source_index, target_index,
        percentile, percentile_param_idx, percentile_param_kind,
    )


def make_in_front_of(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    percentile: torch.Tensor,
    percentile_param_idx: Optional[torch.Tensor] = None,
    percentile_param_kind: Optional[str] = None,
) -> LossTerm:
    """Place source in front of target with lateral percentile alignment."""
    return _make_directional(
        in_front_of_loss, "in_front_of", source_index, target_index,
        percentile, percentile_param_idx, percentile_param_kind,
    )


def make_behind_of(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    percentile: torch.Tensor,
    percentile_param_idx: Optional[torch.Tensor] = None,
    percentile_param_kind: Optional[str] = None,
) -> LossTerm:
    """Place source behind target with lateral percentile alignment."""
    return _make_directional(
        behind_of_loss, "behind_of", source_index, target_index,
        percentile, percentile_param_idx, percentile_param_kind,
    )


def make_in_front_of_dir_only(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> LossTerm:
    """
    Direction-only in_front_of for mutual facing with a cluster anchor.

    The anchor's rotation is fixed at 0 (facing +y) in cluster-local coordinates,
    so this pushes the non-anchor member into the anchor's +y half-plane to let
    the anchor's fixed orientation naturally "face" the member. It is the regular
    in_front_of with w_align pinned to 0 (no lateral alignment, so the percentile is
    inert) and no gap component or optimizable params.
    """
    percentile = torch.full((source_index.shape[0],), 0.5, device=cfg.runtime.device)  # inert when w_align=0
    return _make_directional(
        in_front_of_loss, "in_front_of", source_index, target_index,
        percentile, None, None, w_align_override=0.0,
    )


def make_around(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    sweep_deg: torch.Tensor,
    center_rad: torch.Tensor,
    sweep_deg_param_idx: Optional[torch.Tensor] = None,
    sweep_deg_param_kind: Optional[str] = None,
) -> LossTerm:
    """Arrange source entities on an arc around the target."""
    N: int = source_index.shape[0]
    zero = torch.tensor(0.0, device=cfg.runtime.device)
    if N <= 1:
        return LossTerm(evaluate_fn=lambda aug, params=None: (zero, zero))

    max_sweep_deg: float = (N - 1) / N * 360.
    # NOTE: in variable mode, clamping happens at evaluate time instead of parse time

    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_sweep = _resolve_param(params, sweep_deg_param_idx, sweep_deg_param_kind, sweep_deg, "around")
        eff_sweep = torch.clamp(eff_sweep, max=max_sweep_deg)
        return around_loss(
            select(layout.x, source_index),
            select(layout.y, source_index),
            select(layout.x, target_index),
            select(layout.y, target_index),
            select(layout.rz, target_index),
            sweep_deg=eff_sweep,
            center_rad=center_rad,
            stable=bool(cfg.constraints.shapes.around.stable),
            eps=float(cfg.constraints.shapes.around.eps),
            w_spacing=float(cfg.constraints.shapes.around.w_spacing),
            w_center=float(cfg.constraints.shapes.around.w_center),
        )
    return LossTerm(evaluate_fn=evaluate_fn)


def make_angle(
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    angle_deg: torch.Tensor,
    angle_param_idx: Optional[torch.Tensor] = None,
    angle_param_kind: Optional[str] = None,
) -> LossTerm:
    """
    Enforce source rotation = target rotation + angle_deg (signed, CCW positive).

    src_rz = tar_rz + angle_deg * (pi/180); target rotation is detached.
    """
    def evaluate_fn(aug: AugmentedState, params: Optional[ParamVec] = None) -> _LossPair:
        layout = aug.poses
        eff_angle = _resolve_param(params, angle_param_idx, angle_param_kind, angle_deg, "angle")
        return angle_loss(
            select(layout.rz, source_index),
            select(layout.rz, target_index),
            eff_angle,
        )
    return LossTerm(evaluate_fn=evaluate_fn)
