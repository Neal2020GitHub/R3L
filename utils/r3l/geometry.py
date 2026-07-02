from typing import Dict, Tuple
from abc import ABC, abstractmethod
import math
import torch

def calc_bbox_dims(
    bbox_min: Dict[str, float], 
    bbox_max: Dict[str, float]
) -> Dict[str, float]:
    assert len(bbox_min) == len(bbox_max) == 3
    assert all(k in ["x", "y", "z"] for k in bbox_min.keys())
    assert all(k in ["x", "y", "z"] for k in bbox_max.keys())
    return {k: bbox_max[k] - bbox_min[k] for k in ["x", "y", "z"]}

def angle_to_vector(theta: torch.Tensor) -> torch.Tensor:
    """
    Assumes CCW from +y
    Args: theta (theta): (...,)
    Returns: vector: (..., 2)
    """
    s, c = torch.sin(theta), torch.cos(theta)
    return torch.stack((-s, c), dim=-1)


def angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the signed difference between two angles in radians.

    Args: a (radians): (...,)
          b (radians): (...,)
    Returns: difference (radians): (...,)
    """
    d: torch.Tensor = a - b
    return torch.atan2(torch.sin(d), torch.cos(d))


# =============================================================================
# Soft Landing Functions (Monotonic Potential Wells)
# =============================================================================
#
# These functions map normalized distance x ∈ [0, 1] to loss ∈ [0, 1].
# Used by aesthetics_loss to smoothly snap rotations to cardinal angles.
#
# Contract:
#   - L(0) = 0  (perfect alignment → zero cost)
#   - L(1) = 1  (boundary → maximum cost)
#   - L'(x) > 0 for x ∈ (0, 1)  (monotonic → gradient always pulls toward target)
#   - L'(1) → 0  (smooth cutoff → no gradient shock at boundary)
#
# =============================================================================

class SoftLanding(ABC):
    """Base class for soft landing potential well functions."""

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the potential well.

        Args:
            x: Normalized distance in [0, 1]. Values outside this range
               produce undefined behavior (caller must clamp).

        Returns:
            Loss in [0, 1].
        """
        ...


class TukeySoftLanding(SoftLanding):
    """
    Tukey biweight potential: L(x) = 1 - (1 - x²)²

    Borrowed from robust statistics. The "pulling force" peaks at x ≈ 0.577
    (slightly toward the boundary) and decays sharply. Feels "crisp"—holds
    tension through the middle range, then releases quickly at the edge.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 1 - (1 - x.square()).square()


class CosineSoftLanding(SoftLanding):
    """
    Cosine well: L(x) = 0.5 * (1 - cos(πx))

    Half-period of cosine from 0 to π. Force profile is sinusoidal—extremely
    smooth startup and landing. Feels "soft" throughout. Slightly more
    expensive (trig ops) but negligible on GPU.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 - torch.cos(math.pi * x))


class SmoothstepSoftLanding(SoftLanding):
    """
    Smoothstep polynomial: L(x) = 3x² - 2x³

    The classic CG interpolation function. Force peaks exactly at x = 0.5
    and decays symmetrically toward center and edge. Best balance of snap
    (strong pull at midpoint) and smoothness. Pure polynomial—fast.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return 3 * x.square() - 2 * x.pow(3)


def get_soft_landing(name: str) -> SoftLanding:
    return {
        "tukey": TukeySoftLanding,
        "cosine": CosineSoftLanding,
        "smoothstep": SmoothstepSoftLanding,
    }[name]()


def rot_mat(theta: torch.Tensor) -> torch.Tensor: 
    """
    Returns rotation matrix R(theta)

    Args: theta (theta): (...,)
    Returns: rotation matrix: (..., 2, 2)
    """
    s, c = torch.sin(theta), torch.cos(theta)
    r_mat = torch.stack((c, -s, s, c), dim=-1)
    r_mat = r_mat.reshape(theta.shape + (2, 2))
    return r_mat


def rot_mat_inv(theta: torch.Tensor) -> torch.Tensor: 
    """
    Returns inverse rotation matrix R(theta)^-1
    
    Args: theta (theta): (...,)
    Returns: inverse rotation matrix: (..., 2, 2)
    """
    s, c = torch.sin(theta), torch.cos(theta)
    r_mat = torch.stack((c, s, -s, c), dim=-1)
    r_mat = r_mat.reshape(theta.shape + (2, 2))
    return r_mat


def compute_sat_separation(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unified SAT separation computation for OBB pairs.

    Args:
        position_x_source: (N,) source object x positions
        position_y_source: (N,) source object y positions
        bbox_x_source: (N,) source object bbox x dimensions
        bbox_y_source: (N,) source object bbox y dimensions
        rotation_source: (N,) source object rotations (radians)
        position_x_target: (N,) target object x positions
        position_y_target: (N,) target object y positions
        bbox_x_target: (N,) target object bbox x dimensions
        bbox_y_target: (N,) target object bbox y dimensions
        rotation_target: (N,) target object rotations (radians)

    Returns:
        S_max: (N,) max separation (gap detection). Positive = no collision.
        S_min: (N,) min separation (penetration depth). Negative = overlap.
    """
    c_s = torch.stack([position_x_source, position_y_source], dim=1)
    c_t = torch.stack([position_x_target, position_y_target], dim=1)
    d = c_t - c_s

    rs, rt = rotation_source, rotation_target
    cs, ss = torch.cos(rs), torch.sin(rs)
    ct, st = torch.cos(rt), torch.sin(rt)
    u_sx = torch.stack([cs, ss], dim=1)
    u_sy = torch.stack([-ss, cs], dim=1)
    u_tx = torch.stack([ct, st], dim=1)
    u_ty = torch.stack([-st, ct], dim=1)

    A = torch.stack([u_sx, u_sy, u_tx, u_ty], dim=1)

    p = (d.unsqueeze(1) * A).sum(dim=-1)
    sgn = torch.where(p >= 0, p.new_tensor(1.0), p.new_tensor(-1.0))
    C = p * sgn

    e_sx, e_sy = 0.5 * bbox_x_source, 0.5 * bbox_y_source
    e_tx, e_ty = 0.5 * bbox_x_target, 0.5 * bbox_y_target

    def dot_abs(U: torch.Tensor) -> torch.Tensor:
        return torch.abs((A * U.unsqueeze(1)).sum(dim=-1))

    Rs = e_sx.unsqueeze(1) * dot_abs(u_sx) + e_sy.unsqueeze(1) * dot_abs(u_sy)
    Rt = e_tx.unsqueeze(1) * dot_abs(u_tx) + e_ty.unsqueeze(1) * dot_abs(u_ty)

    S = C - (Rs + Rt)

    bias = S.new_tensor([0.0, 1e-6, 2e-6, 3e-6]).unsqueeze(0)
    S_max = (S + bias).max(dim=1).values
    S_min = S.min(dim=1).values

    return S_max, S_min


def compute_gap_sdf(
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
    softmin: bool = False,
    gamma: float = 3.0,
) -> torch.Tensor:
    """
    Compute minimum boundary distance between OBB pairs using SDF (Signed Distance Field) method.
    
    Args:
        position_x_source: (N,) source object x positions
        position_y_source: (N,) source object y positions
        bbox_x_source: (N,) source object bbox x dimensions
        bbox_y_source: (N,) source object bbox y dimensions
        rotation_source: (N,) source object rotations (radians)
        position_x_target: (N,) target object x positions
        position_y_target: (N,) target object y positions
        bbox_x_target: (N,) target object bbox x dimensions
        bbox_y_target: (N,) target object bbox y dimensions
        rotation_target: (N,) target object rotations (radians)
        softmin: if True, use soft-min instead of hard min
        gamma: temperature parameter for softmin
    
    Returns:
        d_min: (N,) minimum signed distance between boundaries
    """
    def rot_translate(lx: torch.Tensor, ly: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        xw = c * lx - s * ly + cx
        yw = s * lx + c * ly + cy
        return torch.stack([xw, yw], dim=-1)  # (N,2)
    
    c_s = torch.cos(rotation_source)
    s_s = torch.sin(rotation_source)
    ex_s, ey_s = 0.5 * bbox_x_source, 0.5 * bbox_y_source
    
    llx_s, lly_s = -ex_s, -ey_s
    lrx_s, lry_s =  ex_s, -ey_s
    urx_s, ury_s =  ex_s,  ey_s
    ulx_s, uly_s = -ex_s,  ey_s
    
    v_ll_s = rot_translate(llx_s, lly_s, position_x_source, position_y_source, c_s, s_s)
    v_lr_s = rot_translate(lrx_s, lry_s, position_x_source, position_y_source, c_s, s_s)
    v_ur_s = rot_translate(urx_s, ury_s, position_x_source, position_y_source, c_s, s_s)
    v_ul_s = rot_translate(ulx_s, uly_s, position_x_source, position_y_source, c_s, s_s)
    V_s = torch.stack([v_ll_s, v_lr_s, v_ur_s, v_ul_s], dim=1)  # (N,4,2)
    
    c_t = torch.cos(rotation_target)
    s_t = torch.sin(rotation_target)
    ex_t, ey_t = 0.5 * bbox_x_target, 0.5 * bbox_y_target
    
    llx_t, lly_t = -ex_t, -ey_t
    lrx_t, lry_t =  ex_t, -ey_t
    urx_t, ury_t =  ex_t,  ey_t
    ulx_t, uly_t = -ex_t,  ey_t
    
    v_ll_t = rot_translate(llx_t, lly_t, position_x_target, position_y_target, c_t, s_t)
    v_lr_t = rot_translate(lrx_t, lry_t, position_x_target, position_y_target, c_t, s_t)
    v_ur_t = rot_translate(urx_t, ury_t, position_x_target, position_y_target, c_t, s_t)
    v_ul_t = rot_translate(ulx_t, uly_t, position_x_target, position_y_target, c_t, s_t)
    V_t = torch.stack([v_ll_t, v_lr_t, v_ur_t, v_ul_t], dim=1)  # (N,4,2)
    
    def sdf_points_to_obb(P: torch.Tensor,
                          cx: torch.Tensor, cy: torch.Tensor,
                          ex: torch.Tensor, ey: torch.Tensor,
                          c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        dx = P[..., 0] - cx.unsqueeze(1)
        dy = P[..., 1] - cy.unsqueeze(1)
        qx =  c.unsqueeze(1) * dx + s.unsqueeze(1) * dy
        qy = -s.unsqueeze(1) * dx + c.unsqueeze(1) * dy
        d_axis_x = qx.abs() - ex.unsqueeze(1)
        d_axis_y = qy.abs() - ey.unsqueeze(1)
        d_axis = torch.stack([d_axis_x, d_axis_y], dim=-1)  # (N,4,2)
        d_outside = d_axis.clamp_min(0).norm(dim=-1)        # (N,4)
        d_inside  = d_axis.amax(dim=-1).clamp_max(0.0)      # (N,4)
        return d_outside + d_inside
    
    d_s2t = sdf_points_to_obb(V_s, position_x_target, position_y_target, ex_t, ey_t, c_t, s_t)  # (N,4)
    d_t2s = sdf_points_to_obb(V_t, position_x_source, position_y_source, ex_s, ey_s, c_s, s_s)  # (N,4)
    
    d_all = torch.cat([d_s2t, d_t2s], dim=1)  # (N,8)
    
    if softmin: 
        beta = 25.0 * float(gamma)
        d_min = -torch.logsumexp(-beta * d_all, dim=1) / beta  # (N,)
    else: 
        d_min = d_all.min(dim=1).values
    
    return d_min


def compute_cluster_aabb(
    local_x: torch.Tensor,     # (M,) local x positions of members (anchor excluded, it's at origin)
    local_y: torch.Tensor,     # (M,) local y positions of members
    local_rz: torch.Tensor,    # (M,) local rotations of members (relative to anchor's facing)
    bbox_x: torch.Tensor,      # (M,) bbox x dimensions of members
    bbox_y: torch.Tensor,      # (M,) bbox y dimensions of members
    anchor_bbox_x: torch.Tensor,  # scalar, anchor's bbox x
    anchor_bbox_y: torch.Tensor,  # scalar, anchor's bbox y
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the AABB of a cluster in its local coordinate frame.
    
    The cluster local frame has:
    - Anchor at origin (0, 0) with rotation 0 (facing +y).
    - Members at (local_x, local_y) with rotation local_rz.
    
    Args:
        local_x: (M,) local x positions of non-anchor members
        local_y: (M,) local y positions of non-anchor members
        local_rz: (M,) local rotations of non-anchor members
        bbox_x: (M,) bbox x dimensions of non-anchor members
        bbox_y: (M,) bbox y dimensions of non-anchor members
        anchor_bbox_x: scalar, anchor's bbox x dimension
        anchor_bbox_y: scalar, anchor's bbox y dimension
    
    Returns:
        cx, cy: Center of the cluster AABB in local coordinates (scalars)
        cbx, cby: Dimensions of the cluster AABB (scalars)
    """
    M = local_x.shape[0]
    
    # Compute AABB extent for each member (rotated box projected onto local axes)
    # For a box with half-extents (ex, ey) rotated by theta:
    # extent_x = ex * |cos(theta)| + ey * |sin(theta)|
    # extent_y = ex * |sin(theta)| + ey * |cos(theta)|
    
    mex = 0.5 * bbox_x  # (M,)
    mey = 0.5 * bbox_y  # (M,)
    
    abs_cos = torch.abs(torch.cos(local_rz))  # (M,)
    abs_sin = torch.abs(torch.sin(local_rz))  # (M,)
    
    # Extent (half-width) of each member in local frame
    extent_x = mex * abs_cos + mey * abs_sin  # (M,)
    extent_y = mex * abs_sin + mey * abs_cos  # (M,)
    
    # Min/Max for members
    min_x_members = local_x - extent_x  # (M,)
    max_x_members = local_x + extent_x  # (M,)
    min_y_members = local_y - extent_y  # (M,)
    max_y_members = local_y + extent_y  # (M,)
    
    # Anchor is at (0, 0) with rotation 0, so its extent is simply half its bbox
    anchor_ex = 0.5 * anchor_bbox_x
    anchor_ey = 0.5 * anchor_bbox_y
    
    # Combine anchor and members for overall AABB
    if M > 0:
        c_min_x = torch.min(torch.min(min_x_members), -anchor_ex)
        c_max_x = torch.max(torch.max(max_x_members), anchor_ex)
        c_min_y = torch.min(torch.min(min_y_members), -anchor_ey)
        c_max_y = torch.max(torch.max(max_y_members), anchor_ey)
    else:
        # Only anchor in cluster (edge case)
        c_min_x = -anchor_ex
        c_max_x = anchor_ex
        c_min_y = -anchor_ey
        c_max_y = anchor_ey
    
    # Cluster dimensions
    cbx = c_max_x - c_min_x
    cby = c_max_y - c_min_y
    
    # Cluster center in local frame
    cx = (c_min_x + c_max_x) * 0.5
    cy = (c_min_y + c_max_y) * 0.5
    
    return cx, cy, cbx, cby


def compute_cluster_obb_global(
    anchor_x: torch.Tensor,
    anchor_y: torch.Tensor,
    anchor_rz: torch.Tensor,
    anchor_bx: torch.Tensor,
    anchor_by: torch.Tensor,
    member_x: torch.Tensor,
    member_y: torch.Tensor,
    member_rz: torch.Tensor,
    member_bx: torch.Tensor,
    member_by: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute cluster OBB from global poses.

    Takes pre-sliced anchor and member data in world coordinates.
    Returns the cluster's oriented bounding box in world coordinates.

    Args:
        anchor_x, anchor_y, anchor_rz: Anchor's global pose (scalars)
        anchor_bx, anchor_by: Anchor's bbox dimensions (scalars)
        member_x, member_y, member_rz: Non-anchor members' global poses (M,)
        member_bx, member_by: Non-anchor members' bbox dimensions (M,)

    Returns:
        cx, cy: Center of cluster OBB in world coordinates (scalars)
        theta: Rotation of cluster OBB, same as anchor (scalar)
        width, height: Dimensions of cluster OBB (scalars)
    """
    # Transform members from global to anchor-local frame
    dx = member_x - anchor_x
    dy = member_y - anchor_y

    c_inv = torch.cos(-anchor_rz)
    s_inv = torch.sin(-anchor_rz)

    local_x = c_inv * dx - s_inv * dy
    local_y = s_inv * dx + c_inv * dy
    local_rz = member_rz - anchor_rz

    # Compute AABB in local frame
    cx_local, cy_local, width, height = compute_cluster_aabb(
        local_x, local_y, local_rz,
        member_bx, member_by,
        anchor_bx, anchor_by,
    )

    # Transform center back to global
    c = torch.cos(anchor_rz)
    s = torch.sin(anchor_rz)
    cx = c * cx_local - s * cy_local + anchor_x
    cy = s * cx_local + c * cy_local + anchor_y

    return cx, cy, anchor_rz, width, height

