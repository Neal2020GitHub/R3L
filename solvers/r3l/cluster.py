"""
Cluster: also referred to as "Unit" in the our paper.

A scene is a flat list of N objects partitioned into independent objects (anchored
to the room) and clusters (groups of member objects rigidly carried by an anchor).
Clusters are modelled as K virtual entities appended after the N objects, so the
augmented index space is [0..N) objects, [N..N+K) clusters.

Two coordinate frames coexist, selected by `reparam`:
- reparam=True  global-to-local reparameterization
- reparam=False global parameterization
- `localize`/`globalize` move whole poses between the local/global frames. 
"""

from dataclasses import dataclass
from typing import Dict, List

import torch

from utils.r3l.types import PoseVec, BBoxVec
from utils.r3l.geometry import compute_cluster_aabb, compute_cluster_obb_global, rot_mat, rot_mat_inv
from solvers.r3l.config import cfg


@dataclass
class ClusterMeta:
    """What one cluster contains. See the fields below."""
    member_indices: torch.Tensor  # (M,) long, includes anchor
    non_anchor_indices: torch.Tensor  # (M-1,) long, excludes anchor
    anchor_index: int


@dataclass(frozen=True)
class SceneIndex:
    """
    The fixed structure of a scene. Built once, never changes while solving.

    Holds:
    - membership   which cluster each object is in
    - index        each entity's slot in the augmented list
    - pairs        the within-cluster box pairs scored by the local collision layer

    Poses and boxes are not here. They change every step, so they live in AugmentedState.
    """
    N: int                                   # number of real objects
    independent_indices: torch.Tensor        # (I,) long, object indices anchored to the room
    cluster_to_index: Dict[str, int]         # cluster_id -> augmented index in [N..N+K)
    clusters: Dict[str, ClusterMeta]         # cluster_id -> membership metadata
    object_to_index: Dict[str, int]          # object_id -> 0..N-1
    local_pairs: torch.Tensor                # (2, M) global object-index pairs, block-diagonal per cluster

    @property
    def sorted_cids(self) -> List[str]:
        """Cluster ids ordered by their augmented index (N, N+1, ...)."""
        return sorted(self.cluster_to_index.keys(), key=lambda cid: self.cluster_to_index[cid])

    def localize(self, poses: PoseVec, *, reparam: bool) -> PoseVec:
        """
        Move cluster members into their anchor's frame, so a cluster moves as one rigid body.

        - members       become relative to the anchor
        - independents  stay global
        - anchors       stay global

        Inverse of globalize. The math is in _transform_members.
        """
        return self._transform_members(poses, inverse=True, reparam=reparam)

    def globalize(self, poses: PoseVec, *, reparam: bool) -> PoseVec:
        """
        Move cluster members back into the global frame, undoing localize.

        - members       return to global coordinates
        - independents  stay global
        - anchors       stay global

        Inverse of localize. The math is in _transform_members.
        """
        return self._transform_members(poses, inverse=False, reparam=reparam)

    def _transform_members(self, poses: PoseVec, *, inverse: bool, reparam: bool) -> PoseVec:
        """
        Move every cluster's non-anchor members between the global and anchor-local
        frames; independent objects and anchors are carried through untouched.

        The two directions are one body that differs only in the rotation applied
        and whether the anchor position is subtracted before or added after it:
        - inverse=True  (localize):   local = R(-ar) @ (member - anchor),  rz -= ar
        - inverse=False (globalize): global = R(+ar) @ local + anchor,     rz += ar

        reparam=False is the no_localize ablation: poses are already global in both
        directions, so the whole transform is a no-op.
        """
        if not reparam or not self.clusters:
            return poses

        out_x = poses.x.clone()
        out_y = poses.y.clone()
        out_rz = poses.rz.clone()

        for meta in self.clusters.values():
            anchor_idx = meta.anchor_index
            non_anchor_idx = meta.non_anchor_indices
            if non_anchor_idx.numel() == 0:
                continue

            ar = poses.rz[anchor_idx]
            mx = poses.x[non_anchor_idx]
            my = poses.y[non_anchor_idx]
            mr = poses.rz[non_anchor_idx]

            if inverse:
                # Subtract anchor, then rotate by -ar into the local frame.
                vx, vy = mx - poses.x[anchor_idx], my - poses.y[anchor_idx]
                r = rot_mat_inv(ar)
                out_x[non_anchor_idx] = r[0, 0] * vx + r[0, 1] * vy
                out_y[non_anchor_idx] = r[1, 0] * vx + r[1, 1] * vy
                out_rz[non_anchor_idx] = mr - ar
            else:
                # Rotate the local vector by +ar, then add the anchor position.
                r = rot_mat(ar)
                out_x[non_anchor_idx] = r[0, 0] * mx + r[0, 1] * my + poses.x[anchor_idx]
                out_y[non_anchor_idx] = r[1, 0] * mx + r[1, 1] * my + poses.y[anchor_idx]
                out_rz[non_anchor_idx] = mr + ar

        return PoseVec(x=out_x, y=out_y, rz=out_rz)


def build_pairs(clusters: Dict[str, ClusterMeta]) -> torch.Tensor:
    """Block-diagonal (2, M) global-index box pairs for the local collision.

    Each cluster contributes its members' upper-triangular (i<j) pairs, mapped to
    global object indices and concatenated in cluster-id order. Singleton/empty
    clusters contribute nothing; cross-cluster pairs are never emitted. Built once
    at compile time -- the fixed pairing the local collision scores.
    """
    device = cfg.runtime.device
    cols: List[torch.Tensor] = []
    for cid in sorted(clusters):
        members = clusters[cid].member_indices
        m = members.numel()
        if m > 1:
            cols.append(members[torch.triu_indices(m, m, 1, device=device)])  # local -> global
    if not cols:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    return torch.cat(cols, dim=1)


@dataclass(frozen=True)
class AugmentedState:
    """
    The poses and boxes for one optimization step. Rebuilt every step from the current poses.

    Holds:
    - pose     where each entity is this step
    - box      each entity's bounding box this step
    - global   the global-layer entities checked for collisions and walls

    The fixed structure is not here. It never changes, so it lives in SceneIndex.
    """
    poses: PoseVec               # (N+K,) augmented poses
    bbox: BBoxVec                # (N+K,) augmented bboxes
    global_indices: torch.Tensor  # global-layer entity indices (independent objects + cluster handles)

    @classmethod
    def build(
        cls,
        scene: SceneIndex,
        poses: PoseVec,
        bbox: BBoxVec,
        *,
        reparam: bool,
    ) -> 'AugmentedState':
        """
        Build this step's augmented state. The reparam flag picks the frame.

        - reparam=True   members are placed relative to their anchor
        - reparam=False  everything stays global
        - no clusters    poses and boxes pass through unchanged
        """
        device = cfg.runtime.device
        N = scene.N
        obj_px, obj_py, obj_rz = poses.x, poses.y, poses.rz
        obj_bx, obj_by = bbox.x, bbox.y

        if not scene.clusters:
            return cls(
                poses=poses,
                bbox=bbox,
                global_indices=torch.arange(N, device=device, dtype=torch.long),
            )

        K = len(scene.clusters)
        sorted_cids = scene.sorted_cids

        aug_px = obj_px.clone()
        aug_py = obj_py.clone()
        aug_rz = obj_rz.clone()
        aug_bx = obj_bx.clone()
        aug_by = obj_by.clone()

        clu_cx_list, clu_cy_list, clu_rz_list, clu_bx_list, clu_by_list = [], [], [], [], []

        for cid in sorted_cids:
            meta = scene.clusters[cid]
            anchor_idx = meta.anchor_index
            non_anchor_idx = meta.non_anchor_indices

            if reparam:
                # Cluster's global pose comes from the anchor's optimizer output
                cluster_gx = obj_px[anchor_idx]
                cluster_gy = obj_py[anchor_idx]
                cluster_gr = obj_rz[anchor_idx]

                # Zero out anchor's pose in the object portion (it's now a fixed local origin)
                aug_px[anchor_idx] = torch.tensor(0.0, device=device)
                aug_py[anchor_idx] = torch.tensor(0.0, device=device)
                aug_rz[anchor_idx] = torch.tensor(0.0, device=device)

                # Compute cluster AABB in local frame; non-anchor members are already local
                if non_anchor_idx.numel() > 0:
                    local_x = obj_px[non_anchor_idx]
                    local_y = obj_py[non_anchor_idx]
                    local_rz = obj_rz[non_anchor_idx]
                    mem_bx = obj_bx[non_anchor_idx]
                    mem_by = obj_by[non_anchor_idx]
                else:
                    local_x = torch.tensor([], device=device)
                    local_y = torch.tensor([], device=device)
                    local_rz = torch.tensor([], device=device)
                    mem_bx = torch.tensor([], device=device)
                    mem_by = torch.tensor([], device=device)

                cx_local, cy_local, cbx, cby = compute_cluster_aabb(
                    local_x, local_y, local_rz,
                    mem_bx, mem_by,
                    obj_bx[anchor_idx], obj_by[anchor_idx],
                )

                # Transform cluster center from local to global by the anchor pose
                c = torch.cos(cluster_gr)
                s = torch.sin(cluster_gr)
                cx = c * cx_local - s * cy_local + cluster_gx
                cy = s * cx_local + c * cy_local + cluster_gy
                cr = cluster_gr
            else:
                # No localization: anchors keep their global pose, OBB built in global frame
                if non_anchor_idx.numel() > 0:
                    cx, cy, cr, cbx, cby = compute_cluster_obb_global(
                        anchor_x=obj_px[anchor_idx],
                        anchor_y=obj_py[anchor_idx],
                        anchor_rz=obj_rz[anchor_idx],
                        anchor_bx=obj_bx[anchor_idx],
                        anchor_by=obj_by[anchor_idx],
                        member_x=obj_px[non_anchor_idx],
                        member_y=obj_py[non_anchor_idx],
                        member_rz=obj_rz[non_anchor_idx],
                        member_bx=obj_bx[non_anchor_idx],
                        member_by=obj_by[non_anchor_idx],
                    )
                else:
                    # Anchor-only cluster: OBB = anchor bbox centered at anchor position
                    cx = obj_px[anchor_idx]
                    cy = obj_py[anchor_idx]
                    cr = obj_rz[anchor_idx]
                    cbx = obj_bx[anchor_idx]
                    cby = obj_by[anchor_idx]

            clu_cx_list.append(cx)
            clu_cy_list.append(cy)
            clu_rz_list.append(cr)
            clu_bx_list.append(cbx)
            clu_by_list.append(cby)

        aug_px = torch.cat([aug_px, torch.stack(clu_cx_list)])
        aug_py = torch.cat([aug_py, torch.stack(clu_cy_list)])
        aug_rz = torch.cat([aug_rz, torch.stack(clu_rz_list)])
        aug_bx = torch.cat([aug_bx, torch.stack(clu_bx_list)])
        aug_by = torch.cat([aug_by, torch.stack(clu_by_list)])
        aug_bz = torch.cat([bbox.z, torch.zeros(K, device=device)])

        # Global entities: independent objects + cluster handles
        cluster_indices = torch.arange(N, N + K, device=device, dtype=torch.long)
        global_indices = (
            torch.cat([scene.independent_indices, cluster_indices])
            if scene.independent_indices.numel() > 0
            else cluster_indices
        )

        return cls(
            poses=PoseVec(x=aug_px, y=aug_py, rz=aug_rz),
            bbox=BBoxVec(x=aug_bx, y=aug_by, z=aug_bz),
            global_indices=global_indices,
        )
