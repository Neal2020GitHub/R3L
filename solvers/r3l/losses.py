"""Pure, stateless loss kernels and parameter clamping.

Every function here is a pure tensor->tensor map with zero domain state: it
takes the tensors it needs as arguments and returns a (weighted, nominal) loss
pair. The builders module binds these kernels into LossTerm closures; nothing
here knows about scenes, clusters, or constraint JSON.
"""

import math
import torch
import torch.nn as nn
from typing import Sequence, Tuple

from .activations import Softplus2, ReLU2
from .config import cfg
from utils.r3l.geometry import (
    rot_mat,
    rot_mat_inv,
    compute_sat_separation,
    compute_gap_sdf,
    angle_to_vector,
    SoftLanding,
)
from utils.third_party.Rotated_IoU import losses as rotated_iou

select = lambda t, i: torch.index_select(t, -1, i)

CENTERLINE_RAD = {
    "T": 0.0, "B": math.pi, "L": math.pi / 2, "R": -math.pi / 2,
    "TL": math.pi / 4, "TR": -math.pi / 4, "BL": 3 * math.pi / 4, "BR": -3 * math.pi / 4,
}

# Target rotations facing a wall, indexed by wall_index (0=Left -x, 1=Right +x, 2=Top +y, 3=Bottom -y).
# INWARD : back against the wall, front faces the room interior -> against_wall / corner.
# OUTWARD: front faces the wall itself -> facing_wall. OUTWARD = INWARD rotated 180°.
WALL_INWARD_RAD = torch.tensor([-math.pi / 2, math.pi / 2, math.pi, 0.0], dtype=torch.float32)
WALL_OUTWARD_RAD = torch.tensor([math.pi / 2, -math.pi / 2, 0.0, math.pi], dtype=torch.float32)

get_act = lambda act_name: {
    "relu": nn.ReLU(),
    "relu2": ReLU2(),
    "softplus": nn.Softplus(),
    "softplus2": Softplus2(),
}[act_name]


def clamp_param(raw: torch.Tensor, kind: str) -> torch.Tensor:
    """Clamp a raw parameter value based on its kind."""
    match kind:
        case "unit": return raw  # Unclamped: alignment can exceed [0,1] to resolve collisions
        case "nonneg": return torch.clamp(raw, min=0.0)
        case "angle_deg": return raw  # No clamping for angles
        case _: raise ValueError(f"unknown clamp kind: {kind}")


def pack_boxes(
    position_x: torch.Tensor,
    position_y: torch.Tensor,
    bbox_x: torch.Tensor,
    bbox_y: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Assemble (N, 5) [x, y, bx, by, rz] boxes in Rotated_IoU input order."""
    return torch.stack([
        position_x.to(dtype=torch.float32),
        position_y.to(dtype=torch.float32),
        bbox_x,
        bbox_y,
        rotation.to(dtype=torch.float32),
    ], dim=1)


def collision_loss(
    box1: torch.Tensor,
    box2: torch.Tensor,
    method: str = "diou",
    weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum of pairwise rotated-IoU overlap over ALREADY-PAIRED boxes.

    box1, box2 are (M, 5) [x, y, bx, by, rz]; the m-th pair is box1[m] vs
    box2[m]. The CALLER picks the pairing (all-pairs, block-diagonal, ...); this
    kernel knows nothing about scenes or clusters. Empty (M=0) -> zero.

    Returns (sum * weight * cfg.constraints.weights.collision, sum * weight).
    """
    if box1.shape[0] == 0:
        z = torch.tensor(0.0, device=box1.device)
        return z, z

    match method:
        case "iou": f = rotated_iou.cal_my_iou
        case "diou": f = rotated_iou.cal_my_diou
        case "giou": f = rotated_iou.cal_my_giou
        case _: raise ValueError(f"Invalid method: {method}")

    loss, iou = f(box1.unsqueeze(0), box2.unsqueeze(0))
    loss = loss.sum() * weight

    assert isinstance(loss, torch.Tensor)
    assert isinstance(iou, torch.Tensor)
    return loss * cfg.constraints.weights.collision, loss


def wall_loss(
    bbox_x: torch.Tensor,
    bbox_y: torch.Tensor,
    position_x: torch.Tensor,
    position_y: torch.Tensor,
    rotation: torch.Tensor,
    room_size: Tuple[float, float],
    act: nn.Module = ReLU2(),
    weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]: 
    """
    Loss function to push objects away from room walls.
    Gives high loss when passes wall boundary.
    """

    N = position_x.shape[0]
    assert position_x.shape == (N,)
    assert position_y.shape == (N,)
    assert rotation.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    ex, ey = 0.5 * bbox_x, 0.5 * bbox_y
    px, py = position_x, position_y

    # vertices of the OBBs (N, 2, 4)
    vert = torch.stack([
        torch.stack([-ex, -ey], dim=-1), # BL (N, 2)
        torch.stack([ex, -ey], dim=-1), # BR (N, 2)
        torch.stack([ex, ey], dim=-1), # TR (N, 2)
        torch.stack([-ex, ey], dim=-1), # TL (N, 2)
    ], dim=-1) # (N, 2, 4)
    assert vert.shape == (N, 2, 4)

    # rotate in local space
    rot = rot_mat(rotation).to(cfg.runtime.device) # (N, 2, 2)
    vert = torch.bmm(rot, vert) # (N, 2, 4)
    
    # offset to world coordinates
    pos = torch.stack([px, py], dim=1).unsqueeze(-1) # (N, 2, 1)
    vert = vert + pos # (N, 2, 4)
    assert vert.shape == (N, 2, 4)

    rw, rh = room_size
    # distance of four vertices to four walls
    # Assumes all 4 room vertices are positive num
    dist_l = vert[:, 0, :] # (N, 4)
    dist_r = rw - vert[:, 0, :] # (N, 4)
    dist_t = rh - vert[:, 1, :] # (N, 4)
    dist_b = vert[:, 1, :] # (N, 4)

    # Larger distance -> lower loss. Lower distance -> higher loss.
    # Hence take act(-dist). Note act can be any ReLU-like functions.
    loss = act(-dist_l) + act(-dist_r) + act(-dist_t) + act(-dist_b)
    loss = loss.sum() * weight

    return loss * cfg.constraints.weights.wall, loss


def aesthetics_loss(
    rotation: torch.Tensor,
    angles_deg: Sequence[float],
    eps_deg: float,
    soft_landing: SoftLanding,
    weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Snap rotations toward cardinal angles via a monotonic potential well.

    The loss activates ONLY when |θ - target| < ε:
      • At d=0 (aligned): loss=0
      • At d→ε (boundary): loss→max, but ∂L/∂d→0 (smooth cutoff)
      • At d≥ε (outside): loss=0, gradient=0

    The soft_landing function defines the shape of the potential well:
      - "smoothstep": 3x² - 2x³  (balanced snap at midpoint)
      - "tukey": 1 - (1-x²)²    (crisp, late release)
      - "cosine": 0.5(1-cos(πx)) (extremely smooth)

    Units:
      - rotation: radians
      - angles_deg / eps_deg: degrees (converted internally)
    """
    N = rotation.shape[0]
    if N == 0:
        z = torch.tensor(0.0, device=rotation.device)
        return z, z

    if eps_deg == 0.0:
        raise ValueError("eps_deg must be > 0 (cannot snap with zero window)")

    assert rotation.shape == (N,)
    assert angles_deg

    rot = rotation.to(dtype=torch.float32)
    dev, dtype = rot.device, rot.dtype

    deg2rad = math.pi / 180.0
    angles = torch.tensor(angles_deg, device=dev, dtype=dtype) * deg2rad
    eps = eps_deg * deg2rad

    diff = rot.unsqueeze(1) - angles.unsqueeze(0) # circular dist to each target: shape (N, A)
    diff = torch.atan2(torch.sin(diff), torch.cos(diff))  # normalize to (-π, π]
    d = diff.abs().min(dim=1).values # Distance to CLOSEST target: shape (N,)
    x = (d / eps).clamp(0.0, 1.0)    # Normalize to [0, 1], clamp for numerical safety

    inside = (d < eps).float() # Gate: zero outside window (hard cutoff)
    per_obj = soft_landing(x) * inside # Apply monotonic potential well, gated by inside mask
    loss = per_obj.sum() * weight
    return loss * cfg.constraints.weights.aesthetics, loss


def horizontal_rel_loss(
    position_x: torch.Tensor,
    rotation: torch.Tensor,
    bbox_x: torch.Tensor,
    bbox_y: torch.Tensor,
    percentile: torch.Tensor,
    room_size: Tuple[float, float],
    loss_fn: str = "l2",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Adjust the horizontal position of the object such that it sits at the
    # specified percentile of the room length (x direction span).
    # Percentile is in the range [0, 1].
    # Percentile = 0 means very left edge of the room
    # Percentile = 1 means very right edge of the room
    # Automatically account for rotated bounding box so the object does not exceed the room length.

    N = position_x.shape[0]
    assert position_x.shape == (N,)
    assert rotation.shape == (N,)
    assert bbox_x.shape == (N,)
    assert bbox_y.shape == (N,)
    assert percentile.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    px, theta = position_x, rotation.detach()
    # Calculate horizontal span of the OBB
    # span_x = |bx * cos(theta)| + |by * sin(theta)|
    abs_cos = torch.abs(torch.cos(theta))
    abs_sin = torch.abs(torch.sin(theta))
    w_span = bbox_x * abs_cos + bbox_y * abs_sin

    W = float(room_size[0])

    # Effective length for the center to move
    L_eff = W - w_span

    # Target center position
    # room x range is [0, W]
    # object occupies [px - w_span/2, px + w_span/2]
    # left-most center position: w_span/2
    # right-most center position: W - w_span/2
    # target = left_most + percentile * (right_most - left_most)
    # target = w_span/2 + percentile * (W - w_span)
    x_tgt = 0.5 * w_span + percentile * L_eff

    per = _position_loss_fn(px - x_tgt, loss_fn, huber_delta)
    loss = per.sum()
    return loss * cfg.constraints.weights.horizontal, loss


def horizontal_abs_loss(
    position_x: torch.Tensor,
    x: torch.Tensor,
    room_size: Tuple[float, float],
    loss_fn: str = "l2",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = position_x.shape[0]
    assert position_x.shape == (N,)
    assert x.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    per = _position_loss_fn(position_x - x, loss_fn, huber_delta)
    loss = per.sum()
    return loss * cfg.constraints.weights.horizontal, loss


def vertical_rel_loss(
    position_y: torch.Tensor,
    rotation: torch.Tensor,
    bbox_x: torch.Tensor,
    bbox_y: torch.Tensor,
    percentile: torch.Tensor,
    room_size: Tuple[float, float],
    loss_fn: str = "l2",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Adjust the vertical position of the object such that it sits at the
    # specified percentile of the room width (y direction span).

    N = position_y.shape[0]
    assert position_y.shape == (N,)
    assert rotation.shape == (N,)
    assert bbox_x.shape == (N,)
    assert bbox_y.shape == (N,)
    assert percentile.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    py, theta = position_y, rotation.detach()
    # Calculate vertical span of the OBB
    # span_y = |bx * sin(theta)| + |by * cos(theta)|
    abs_cos = torch.abs(torch.cos(theta))
    abs_sin = torch.abs(torch.sin(theta))
    h_span = bbox_x * abs_sin + bbox_y * abs_cos

    H = float(room_size[1])

    # Effective length for the center to move
    H_eff = H - h_span

    # Target center position
    # room y range is [0, H]
    y_tgt = 0.5 * h_span + percentile * H_eff

    per = _position_loss_fn(py - y_tgt, loss_fn, huber_delta)
    loss = per.sum()
    return loss * cfg.constraints.weights.vertical, loss


def vertical_abs_loss(
    position_y: torch.Tensor,
    y: torch.Tensor,
    room_size: Tuple[float, float],
    loss_fn: str = "l2",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = position_y.shape[0]
    assert position_y.shape == (N,)
    assert y.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    per = _position_loss_fn(position_y - y, loss_fn, huber_delta)
    loss = per.sum()
    return loss * cfg.constraints.weights.vertical, loss




def facing_radial_loss(
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    position_x_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_source: torch.Tensor,
    position_y_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Point-to-point alignment: rotates source to point exactly at target's geometric centroid.

    Assumes:
        - default orientation (0 rad) points +y
        - Positive rotation is CCW rotation
    """

    N = rotation_source.shape[0]
    assert rotation_target.shape == (N,)
    assert position_x_source.shape == (N,)
    assert position_x_target.shape == (N,)
    assert position_y_source.shape == (N,)
    assert position_y_target.shape == (N,)

    rs = rotation_source
    dx = (position_x_target - position_x_source).detach()
    dy = (position_y_target - position_y_source).detach()
    u = torch.stack([dx, dy], dim=-1)  # (N, 2)

    # Normalize u with epsilon protection to prevent NaN gradients when positions coincide
    u_norm = u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    u_safe = u / u_norm  # Unit vector (or near-unit when degenerate)

    v = angle_to_vector(rs)  # (N, 2) - already unit vector by construction
    cos_sim = (u_safe * v).sum(dim=-1)  # Dot product of unit vectors = cosine similarity
    cos_dist = (1 - cos_sim)
    assert cos_dist.shape == (N,)
    loss = cos_dist.sum()
    return loss * cfg.constraints.weights.facing_radial, loss


def facing_ortho_loss(
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    position_x_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_source: torch.Tensor,
    position_y_target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rectilinear alignment: rotate source to face perpendicular to target's nearest edge.

    Logic:
    1. Transform source position to target's local coordinate system
    2. Determine which edge of target the source is closest to (left/right/top/bottom)
    3. Set source rotation to be perpendicular to that edge (i.e., facing that edge)
    """
    N = rotation_source.shape[0]
    assert rotation_target.shape == (N,)
    assert position_x_source.shape == (N,)
    assert position_x_target.shape == (N,)
    assert position_y_source.shape == (N,)
    assert position_y_target.shape == (N,)

    rs, rt = rotation_source, rotation_target.detach()

    # Compute source position relative to target (global frame)
    dx = (position_x_source - position_x_target).detach()
    dy = (position_y_source - position_y_target).detach()

    # Transform to target's local coordinate system
    cos_t, sin_t = torch.cos(rt), torch.sin(rt)
    local_x = dx * cos_t + dy * sin_t
    local_y = -dx * sin_t + dy * cos_t

    # Determine target rotation angle (in target's local frame)
    # Based on which side (left/right/top/bottom) the source is on
    # We want source to face TOWARD target (inward normal), not away from it
    # In target's local frame (default facing +y):
    #   Source on left (local_x < 0) → face +x direction → angle = -π/2
    #   Source on right (local_x > 0) → face -x direction → angle = +π/2
    #   Source below (local_y < 0) → face +y direction → angle = 0
    #   Source above (local_y > 0) → face -y direction → angle = π

    # Use position to determine dominant edge
    target_local_angle = torch.where(
        torch.abs(local_x) > torch.abs(local_y),
        # Left/right is closer
        torch.where(local_x < 0,
                    torch.full_like(rs, -torch.pi / 2),  # Left side → face +x (toward target)
                    torch.full_like(rs, torch.pi / 2)), # Right side → face -x (toward target)
        # Top/bottom is closer
        torch.where(local_y < 0,
                    torch.full_like(rs, 0.0),            # Below → face +y (toward target)
                    torch.full_like(rs, torch.pi))       # Above → face -y (toward target)
    )

    # Transform back to global frame
    target_angle = target_local_angle + rt

    # Compute loss (1 - cos makes loss 0 when angle difference is 0)
    angle_error = rs - target_angle
    loss = 1.0 - torch.cos(angle_error)
    loss = loss.sum()

    return loss * cfg.constraints.weights.facing_ortho, loss


def facing_wall_loss(
    src_rotation: torch.Tensor,
    wall_index: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = src_rotation.shape[0]
    assert src_rotation.shape == (N,)
    assert wall_index.shape == (N,)

    targ_rotation = WALL_OUTWARD_RAD.to(wall_index.device)[wall_index]  # (N,)
    loss = 1.0 - torch.cos(src_rotation - targ_rotation)
    loss = loss.sum()
    return loss * cfg.constraints.weights.facing_wall, loss


def distance_loss(
    src_position_x: torch.Tensor, # (N,)
    src_position_y: torch.Tensor, # (N,)
    tar_position_x: torch.Tensor, # (N,)
    tar_position_y: torch.Tensor, # (N,)
    distance: torch.Tensor, # (N,)
) -> Tuple[torch.Tensor, torch.Tensor]: 
    N = src_position_x.shape[0]
    assert tar_position_x.shape == (N,)
    assert tar_position_y.shape == (N,)
    assert src_position_x.shape == (N,)
    assert src_position_y.shape == (N,)
    assert distance.shape == (N,)

    dx = src_position_x - tar_position_x # (N,)
    dy = src_position_y - tar_position_y # (N,)
    dist = torch.sqrt(dx * dx + dy * dy + 1e-8) # (N,)
    err = dist - distance
    loss = (err * err).sum()
    return loss * cfg.constraints.weights.distance, loss


def _position_loss_fn(error: torch.Tensor, loss_fn: str, huber_delta: float) -> torch.Tensor:
    """Compute position loss from error tensor."""
    if loss_fn == "l2":
        return error ** 2
    elif loss_fn == "l1":
        return error.abs()
    elif loss_fn == "huber":
        abs_e = error.abs()
        return torch.where(
            abs_e < huber_delta,
            0.5 * error ** 2,
            huber_delta * (abs_e - 0.5 * huber_delta)
        )
    else:
        raise ValueError(f"Unknown loss_fn: {loss_fn}")


def against_wall_loss(
    position_x: torch.Tensor,
    position_y: torch.Tensor,
    rotation: torch.Tensor,
    wall_index: torch.Tensor,
    bbox_x: torch.Tensor,
    bbox_y: torch.Tensor,
    room_size: Tuple[float, float],
    w_pos: float = 1.0,
    w_rot: float = 10.0,
    loss_fn: str = "l2",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center-based against-wall loss.

    Constrains ONE axis of object center position based on wall, plus rotation alignment.
    Target position is computed assuming correct rotation, decoupling position and rotation
    into independent convex subproblems. The object's back edge faces the wall, so depth
    along the constrained axis is bbox_y/2 and the object faces away from the wall.

    Args:
        position_x, position_y: Object center coordinates (N,)
        rotation: Object rotation angles in radians (N,)
        wall_index: Target wall per object — 0=L, 1=R, 2=T, 3=B (N,)
        bbox_x, bbox_y: Bounding box dimensions (N,)
        room_size: (width, height) of room
        w_pos: Position regularization weight
        w_rot: Rotation regularization weight
        loss_fn: Position loss type — "l2", "l1", or "huber"
        huber_delta: Delta for Huber loss
    """
    N = position_x.shape[0]
    assert position_x.shape == (N,)
    assert position_y.shape == (N,)
    assert rotation.shape == (N,)
    assert wall_index.shape == (N,)
    assert bbox_x.shape == (N,)
    assert bbox_y.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    dtype = torch.float32
    device = position_x.device
    W, H = float(room_size[0]), float(room_size[1])

    wall_index = wall_index.to(device=device, dtype=torch.long)
    batch_idx = torch.arange(N, device=device)

    # --- Depth from center to the wall-facing back edge ---
    depth_dim = 0.5 * bbox_y  # (N,)

    # --- Compute target position on constrained axis ---
    # L(0): X = depth, R(1): X = W - depth, T(2): Y = H - depth, B(3): Y = depth
    axis_selector = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    sign_tbl = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=dtype, device=device)
    extent_tbl = torch.tensor([0.0, W, H, 0.0], dtype=dtype, device=device)

    axis = axis_selector[wall_index]  # (N,)
    target_pos = extent_tbl[wall_index] + sign_tbl[wall_index] * depth_dim  # (N,)

    pos_stack = torch.stack([position_x, position_y], dim=-1)  # (N, 2)
    actual_pos = pos_stack[batch_idx, axis]  # (N,)

    # --- Position loss (single constrained axis) ---
    pos_error = actual_pos - target_pos
    pos_loss = _position_loss_fn(pos_error, loss_fn, huber_delta)

    # --- Rotation loss (back edge faces wall → object faces wall's inward normal) ---
    rot_target = WALL_INWARD_RAD.to(wall_index.device)[wall_index]
    rot_loss = 1.0 - torch.cos(rotation - rot_target)

    # --- Combine ---
    loss = w_pos * pos_loss + w_rot * rot_loss
    loss = loss.sum()
    return loss * cfg.constraints.weights.against_wall, loss


def corner_loss(
    position_x: torch.Tensor,
    position_y: torch.Tensor,
    rotation: torch.Tensor,
    corner_index: torch.Tensor,
    wall_index: torch.Tensor,
    bbox_x: torch.Tensor,
    bbox_y: torch.Tensor,
    room_size: Tuple[float, float],
    w_pos: float = 1.0,
    w_rot: float = 10.0,
    loss_fn: str = "l2",
    huber_delta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Center-based corner loss.

    Constrains BOTH X and Y axes of object center position, plus rotation alignment.
    Target position is computed assuming correct rotation, decoupling position and rotation
    into independent convex subproblems. The object's back edge faces the aligned wall, so
    the depth half-span is bbox_y/2 and the lateral half-span is bbox_x/2.

    Args:
        position_x, position_y: Object center coordinates (N,)
        rotation: Object rotation angles in radians (N,)
        corner_index: Target corner per object — 0=BL, 1=BR, 2=TR, 3=TL (N,)
        wall_index: Which wall the edge aligns with — 0=L, 1=R, 2=T, 3=B (N,)
        bbox_x, bbox_y: Bounding box dimensions (N,)
        room_size: (width, height) of room
        w_pos: Position regularization weight
        w_rot: Rotation regularization weight
        loss_fn: Position loss type — "l2", "l1", or "huber"
        huber_delta: Delta for Huber loss
    """
    N = position_x.shape[0]
    assert position_x.shape == (N,)
    assert position_y.shape == (N,)
    assert rotation.shape == (N,)
    assert corner_index.shape == (N,)
    assert wall_index.shape == (N,)
    assert bbox_x.shape == (N,)
    assert bbox_y.shape == (N,)
    assert room_size[0] > 0 and room_size[1] > 0

    dtype = torch.float32
    device = position_x.device
    W, H = float(room_size[0]), float(room_size[1])

    wall_index = wall_index.to(device=device, dtype=torch.long)
    corner_index = corner_index.to(device=device, dtype=torch.long)

    # --- Half-spans toward the aligned wall (depth) and along it (lateral) ---
    depth_dim = 0.5 * bbox_y  # (N,) toward the wall the back edge faces
    other_dim = 0.5 * bbox_x  # (N,) along that wall

    # --- Corner coordinates ---
    corner_coord_tbl = torch.tensor([
        [0., 0.],  # BL
        [W, 0.],   # BR
        [W, H],    # TR
        [0., H],   # TL
    ], dtype=dtype, device=device)
    corner_xy = corner_coord_tbl[corner_index]  # (N, 2)

    # --- Sign tables (direction from corner toward room interior) ---
    # BL: +X, +Y | BR: -X, +Y | TR: -X, -Y | TL: +X, -Y
    x_sign_tbl = torch.tensor([1., -1., -1., 1.], dtype=dtype, device=device)
    y_sign_tbl = torch.tensor([1., 1., -1., -1.], dtype=dtype, device=device)

    x_sign = x_sign_tbl[corner_index]  # (N,)
    y_sign = y_sign_tbl[corner_index]  # (N,)

    # --- Determine which axis uses depth vs other ---
    # wall_index 0,1 (L,R) → X uses depth, Y uses other
    # wall_index 2,3 (T,B) → Y uses depth, X uses other
    wall_is_vertical = (wall_index <= 1).float()  # (N,)

    x_offset = wall_is_vertical * depth_dim + (1 - wall_is_vertical) * other_dim
    y_offset = (1 - wall_is_vertical) * depth_dim + wall_is_vertical * other_dim

    target_x = corner_xy[:, 0] + x_sign * x_offset  # (N,)
    target_y = corner_xy[:, 1] + y_sign * y_offset  # (N,)

    # --- Position loss (both axes) ---
    x_error = position_x - target_x
    y_error = position_y - target_y
    pos_loss = _position_loss_fn(x_error, loss_fn, huber_delta) + \
               _position_loss_fn(y_error, loss_fn, huber_delta)

    # --- Rotation loss (back edge faces wall → object faces wall's inward normal) ---
    rot_target = WALL_INWARD_RAD.to(wall_index.device)[wall_index]
    rot_loss = 1.0 - torch.cos(rotation - rot_target)

    # --- Combine ---
    loss = w_pos * pos_loss + w_rot * rot_loss
    loss = loss.sum()
    return loss * cfg.constraints.weights.corner, loss


def gap_loss_l2(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    gap: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = position_x_source.shape[0]
    assert position_x_source.shape == (N,)
    assert position_y_source.shape == (N,)
    assert gap.shape == (N,)

    dx = position_x_source - position_x_target
    dy = position_y_source - position_y_target
    d = torch.sqrt(dx * dx + dy * dy + 1e-8)
    assert d.shape == (N,)
    loss = ((d - gap) ** 2).sum()
    return loss * cfg.constraints.weights.gap, loss


def gap_loss_sdf(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    rotation_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_target: torch.Tensor,
    gap: torch.Tensor,
    gamma: float = 3.0,
    softmin: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]: 
    """
    Encourages the minimum distance (measured from boundary) 
    between two OBBs source[i] and target[i] to approximate 
    gap[i].
    """
    rs = rotation_source.detach()
    rt = rotation_target.detach()
    px_t = position_x_target.detach()
    py_t = position_y_target.detach()
    bbx_t = bbox_x_target.detach()
    bby_t = bbox_y_target.detach()
    
    d_min = compute_gap_sdf(
        position_x_source, position_y_source, bbox_x_source, bbox_y_source, rs,
        px_t, py_t, bbx_t, bby_t, rt,
        softmin=softmin, gamma=gamma,
    )
    
    loss = (d_min - gap).square()
    loss = loss.sum()
    return loss * cfg.constraints.weights.gap, loss


def gap_loss_mtv(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    rotation_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_target: torch.Tensor,
    gap: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SAT/MTV signed-distance gap with target spacing.

    Axes: u_sx=[cosθs,sinθs], u_sy=[-sinθs,cosθs], u_tx=[cosθt,sinθt], u_ty=[-sinθt,cosθt].
    Use carried-sign center projection C(a)=sign(p(a)_det)·p(a) so overlap keeps a
    consistent push. Separation per-axis: S(a)=C(a)-(Rs(a)+Rt(a)). Aggregate with a
    hard max (plus tiny bias) to keep a single, stable push direction.

    This variant fits S_max to the target gap with symmetric L2:
      L = Σ_i (S_max_i - gap_i)^2 · cfg.constraints.weights.gap

    Rotations are detached when building axes, so this loss never torques angles.
    """
    rs = rotation_source.detach()
    rt = rotation_target.detach()
    S_max, _ = compute_sat_separation(
        position_x_source, position_y_source, bbox_x_source, bbox_y_source, rs,
        position_x_target, position_y_target, bbox_x_target, bbox_y_target, rt,
    )
    err = (S_max - gap)
    loss = err.square().sum()
    return loss * cfg.constraints.weights.gap, loss


def _directional_loss(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    percentile: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    act: nn.Module,
    dir_axis: int,
    side: float,
    w_dir: float = 1.0,
    w_align: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Place source on one side of target in target's egocentric frame, with
    percentile alignment along the orthogonal axis.

    `dir_axis` selects the direction axis (0=x for left/right, 1=y for front/behind);
    alignment runs on the other axis. `side` is +1 to push source onto that axis'
    positive half (right/front) and -1 onto the negative half (left/behind):
      • direction:  source's near edge must clear target's far edge →
                    violation = et_dir + R_dir - side·s_dir, penalized by act().
      • alignment:  s_align is pulled toward (2·percentile - 1)·(et_align - R_align),
                    so percentile=0/1 hug the target's two edges and 0.5 centers it.
    """
    N = position_x_source.shape[0]
    assert position_x_source.shape == (N,)
    assert position_y_source.shape == (N,)
    assert position_x_target.shape == (N,)
    assert position_y_target.shape == (N,)
    assert rotation_source.shape == (N,)
    assert rotation_target.shape == (N,)
    assert bbox_x_source.shape == (N,)
    assert bbox_y_source.shape == (N,)
    assert bbox_x_target.shape == (N,)
    assert bbox_y_target.shape == (N,)
    assert percentile.shape == (N,)
    assert dir_axis in (0, 1)

    # Detach target rotation so the directional relation never torques target orientation
    r_inv = rot_mat_inv(rotation_target.detach())  # (N, 2, 2)
    s_local = torch.stack([
        position_x_source - position_x_target,
        position_y_source - position_y_target,
    ], dim=1)
    s_local = torch.bmm(r_inv, s_local.unsqueeze(-1)).squeeze(-1)  # (N, 2)

    et = [0.5 * bbox_x_target, 0.5 * bbox_y_target]
    es_x, es_y = 0.5 * bbox_x_source, 0.5 * bbox_y_source

    # Source half-projection radii onto target's axes (rotations detached)
    phi = (rotation_source.detach() - rotation_target.detach())
    abs_sin = torch.abs(torch.sin(phi))
    abs_cos = torch.abs(torch.cos(phi))
    R = [
        es_x * abs_cos + es_y * abs_sin,  # source half-span on target x-axis
        es_x * abs_sin + es_y * abs_cos,  # source half-span on target y-axis
    ]

    align_axis = 1 - dir_axis

    # DIRECTION TERM (edge-based): source's near edge must clear target's far edge.
    violation = et[dir_axis] + R[dir_axis] - side * s_local[:, dir_axis]
    dir_term = w_dir * act(violation)

    # ALIGNMENT TERM (percentile-based along the orthogonal axis):
    # percentile=0 hugs the -edge, 1 hugs the +edge, 0.5 centers (target offset 0).
    target_align = (2.0 * percentile - 1.0) * (et[align_axis] - R[align_axis])
    align_term = w_align * torch.abs(s_local[:, align_axis] - target_align)

    loss = (dir_term + align_term).sum()
    return loss * cfg.constraints.weights.directional, loss


def left_of_loss(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    percentile: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    act: nn.Module,
    w_dir: float = 1.0,
    w_align: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place source to the LEFT of target (target's -x half), aligned on y."""
    return _directional_loss(
        position_x_source, position_y_source, position_x_target, position_y_target,
        percentile, bbox_x_source, bbox_y_source, bbox_x_target, bbox_y_target,
        rotation_source, rotation_target, act,
        dir_axis=0, side=-1.0, w_dir=w_dir, w_align=w_align,
    )


def right_of_loss(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    percentile: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    act: nn.Module,
    w_dir: float = 1.0,
    w_align: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place source to the RIGHT of target (target's +x half), aligned on y."""
    return _directional_loss(
        position_x_source, position_y_source, position_x_target, position_y_target,
        percentile, bbox_x_source, bbox_y_source, bbox_x_target, bbox_y_target,
        rotation_source, rotation_target, act,
        dir_axis=0, side=+1.0, w_dir=w_dir, w_align=w_align,
    )


def in_front_of_loss(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    percentile: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    act: nn.Module,
    w_dir: float = 1.0,
    w_align: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place source IN FRONT OF target (target's +y half), aligned on x."""
    return _directional_loss(
        position_x_source, position_y_source, position_x_target, position_y_target,
        percentile, bbox_x_source, bbox_y_source, bbox_x_target, bbox_y_target,
        rotation_source, rotation_target, act,
        dir_axis=1, side=+1.0, w_dir=w_dir, w_align=w_align,
    )


def behind_of_loss(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    percentile: torch.Tensor,
    bbox_x_source: torch.Tensor,
    bbox_y_source: torch.Tensor,
    bbox_x_target: torch.Tensor,
    bbox_y_target: torch.Tensor,
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    act: nn.Module,
    w_dir: float = 1.0,
    w_align: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place source BEHIND target (target's -y half), aligned on x."""
    return _directional_loss(
        position_x_source, position_y_source, position_x_target, position_y_target,
        percentile, bbox_x_source, bbox_y_source, bbox_x_target, bbox_y_target,
        rotation_source, rotation_target, act,
        dir_axis=1, side=-1.0, w_dir=w_dir, w_align=w_align,
    )


def around_loss(
    position_x_source: torch.Tensor,
    position_y_source: torch.Tensor,
    position_x_target: torch.Tensor,
    position_y_target: torch.Tensor,
    rotation_target: torch.Tensor,
    sweep_deg: torch.Tensor,
    center_rad: torch.Tensor,
    stable: bool = True,
    eps: float = 1e-6,
    w_spacing: float = 1.0,
    w_center: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Arrange N sources in a sector around target, centered on `center_rad` in target's frame.

    Two loss terms:
      1. Centerline: penalize deviation of arc's circular mean from center_rad
      2. Spacing: penalize uneven gaps between sorted angles (in centerline-relative frame)

    Args:
        position_x_source: (N,) x-coords of sources
        position_y_source: (N,) y-coords of sources
        position_x_target: (1,) x-coord of target
        position_y_target: (1,) y-coord of target
        rotation_target: (1,) target orientation in radians
        sweep_deg: (1,) total angular extent in degrees
        center_rad: (1,) centerline direction in target's local frame (radians)

    Notes on V_target: 
        V_target is the expected center of mass computed
         from the sweep and centerline 
        >>> u = linspace(-sweep/2, +sweep/2, N) # simulate even spacing
        >>> M_target = mean(cos(u)) # expected magnitude
        >>> V_target = M_target * ( sin(center_rad), cos(center_rad) )

        We do not use sinc(s/2), even though
        1/S ∫ cos(u) du = 2sin(sweep/2) / sweep = sinc(sweep/s)
        because when N is small it may introduce systematic bias
        sweep=180, N=2, then M = 0, while sinc(s/2) predicts 0.6366
        So, instead we used the discrete O(1) closed-form solution: 
        M_target = sin(N * delta/2) / (N * sin(delta/2))
    """
    N = position_x_source.shape[0]
    assert N > 1
    assert 360.0 >= sweep_deg > 0.0

    sweep_rad = sweep_deg * (torch.pi / 180.0)
    tar_rz = rotation_target.squeeze()

    # --- 1. Transform to target's local frame ---
    vx = position_x_source - position_x_target
    vy = position_y_source - position_y_target
    vx = vx + torch.randn_like(vx) * eps
    vy = vy + torch.randn_like(vy) * eps
    c, s = torch.cos(tar_rz), torch.sin(tar_rz)
    vx_loc = c * vx + s * vy
    vy_loc = -s * vx + c * vy
    theta_loc = torch.atan2(vx_loc, vy_loc)  # CCW from +y

    # --- 2. Centerline alignment loss ---

    # [Step 2.1] Calculate Actual Mean Vector (First Moment) V_actu
    # Intuition: current center of mass
    # norm(V_actual) -> 1: srcs are concentrated
    # norm(V_actual) -> 0: srcs are dispersed
    v_act_sin = torch.sin(theta_loc).mean() # sin mean (scalar)
    v_act_cos = torch.cos(theta_loc).mean() # cos mean (scalar)


    # [Step 2.2] Calculate Target Expected Norm (m_tar)
    # We need the theoretical modulus of the sum of N unit vectors evenly spaced 
    # in [-sweep/2, +sweep/2].
    # Using the Lagrange trigonometric identity for discrete sums:
    # Sum[cos] = sin(N * delta) / sin(delta), where delta is half the angular spacing.
    # spacing = sweep / (N - 1)  =>  delta = sweep / (2 * (N - 1))

    # Denominator safety: Since sweep_deg non-zero, 
    # and N > 1, the denominator sin(delta) will not be zero.
    delta = sweep_rad / (2 * (N - 1)) # (1,)
    numerator = torch.sin(N * delta) # (1,)
    denominator = torch.sin(delta) # (1,)
    m_tar = (numerator / denominator) / N # (1,), normalize by 1/N

    # [Step 2.3] Construct Target Vector in Local Frame
    # The target centerline is at angle `center_rad` relative to the local frame.
    # We scale this unit direction by the expected shrinkage `m_tar`.
    # Based on atan2(vx, vy) definition above: sin is X component, cos is Y component.
    v_tar_sin = m_tar * torch.sin(center_rad) # (1,)
    v_tar_cos = m_tar * torch.cos(center_rad) # (1,)

    # [Step 2.4] The Loss: Squared Euclidean Distance in Embedding Space
    # L = || V_actual - V_target ||^2
    # Quadratically penalizes both orientation error and spread error.
    centerline_loss = (v_act_sin - v_tar_sin)**2 + (v_act_cos - v_tar_cos)**2
    centerline_loss = centerline_loss.squeeze() # (scalar)

    # --- 3. Spacing loss (in centerline-relative frame) ---
    # Shift so centerline becomes 0, preventing boundary artifacts when arc straddles ±π
    theta_rel = theta_loc - center_rad.squeeze()
    theta_rel = (theta_rel + torch.pi) % (2 * torch.pi) - torch.pi  # normalize to [-π, π]
    theta_sorted = torch.sort(theta_rel, stable=stable)[0]
    gaps = theta_sorted[1:] - theta_sorted[:-1]
    ideal_gap = sweep_rad.squeeze() / (N - 1)
    spacing_loss = ((gaps - ideal_gap) ** 2).mean()

    # --- 4. Combine ---
    loss = w_spacing * spacing_loss + w_center * centerline_loss
    assert loss.shape == ()
    return loss * cfg.constraints.weights.around, loss


def angle_loss(
    rotation_source: torch.Tensor,
    rotation_target: torch.Tensor,
    angle_deg: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Angle offset loss: enforce source rotation = target rotation + angle_deg.

    Following math convention:
    - Positive angle_deg -> CCW rotation relative to target
    - src_rz = tar_rz + angle_deg * (π/180)

    Gradient flows only through rotation_source (target is detached).

    Args:
        rotation_source: (N,) source rotations in radians
        rotation_target: (N,) target rotations in radians
        angle_deg: (N,) signed angle offset in degrees (positive = CCW)
    """
    N = rotation_source.shape[0]
    assert rotation_source.shape == (N,)
    assert rotation_target.shape == (N,)
    assert angle_deg.shape == (N,)

    rt = rotation_target.detach()
    angle_rad = angle_deg * (torch.pi / 180.0)
    target_rz = rt + angle_rad
    loss = 1.0 - torch.cos(rotation_source - target_rz)
    loss = loss.sum()
    return loss * cfg.constraints.weights.angle, loss
