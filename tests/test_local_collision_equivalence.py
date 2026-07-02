"""Guard the batched local (within-cluster) collision against the per-cluster original.

The committed fixtures all have ``coll_local == 0.0`` (no within-cluster
overlap), so ``test_compile_equivalence`` is blind to the batched local
collision. This test exercises the changed path with REAL overlap and pins:

  1. the batched block-diagonal call equals the old per-cluster loop, in both
     value and gradient (the equivalence the refactor must preserve), and
  2. cross-cluster pairs are NEVER scored -- the one bug a batched
     implementation can introduce. Two clusters are placed so members overlap
     WITHIN each cluster AND a cross-cluster pair also overlaps; the
     block-diagonal result must therefore be strictly below the all-pairs
     result over the same boxes.

Scene (boxes are 2x2, so neighbours 0.2-0.6 apart overlap heavily):
  cluster_a = {chair-0 @x=0.0 (anchor), chair-1 @x=0.4}
  cluster_b = {chair-2 @x=0.2 (anchor), chair-3 @x=0.6}
The cross pair (chair-1 @0.4, chair-2 @0.2) overlaps but belongs to no cluster.
"""

import unittest

import torch

from tests.lib import EMPTY_RELATIONAL, compile_scene
from solvers.r3l.cluster import AugmentedState, ClusterMeta, build_pairs
from solvers.r3l.losses import collision_loss, pack_boxes
from utils.r3l.types import BBoxVec, PoseVec

# Global geometry: 4 boxes on the x-axis, with small rotations so the gradient
# check exercises rz too. Indices 0..3 == chair-0..chair-3.
PX = [0.0, 0.4, 0.2, 0.6]
PY = [0.0, 0.0, 0.0, 0.0]
RZ = [0.0, 0.2, 0.0, 0.2]
BX = [2.0, 2.0, 2.0, 2.0]
BY = [2.0, 2.0, 2.0, 2.0]

OBJECTS = ["chair-0", "chair-1", "chair-2", "chair-3"]
ORDER = ["cluster_a", "cluster_b"]  # == sorted(cluster ids) == SceneIndex.sorted_cids
CLUSTERS = {
    "cluster_a": ClusterMeta(
        member_indices=torch.tensor([0, 1]), non_anchor_indices=torch.tensor([1]), anchor_index=0,
    ),
    "cluster_b": ClusterMeta(
        member_indices=torch.tensor([2, 3]), non_anchor_indices=torch.tensor([3]), anchor_index=2,
    ),
}


def _leaves():
    """Fresh (px, py, rz) leaf tensors so two backward passes don't share grad."""
    return (
        torch.tensor(PX, dtype=torch.float32, requires_grad=True),
        torch.tensor(PY, dtype=torch.float32, requires_grad=True),
        torch.tensor(RZ, dtype=torch.float32, requires_grad=True),
    )


def _per_cluster_reference(boxes, clusters, order, weight):
    """The OLD behaviour: loop clusters, all-pairs within each, sum the returns."""
    scaled = torch.zeros((), dtype=boxes.dtype)
    nominal = torch.zeros((), dtype=boxes.dtype)
    for cid in order:
        members = clusters[cid].member_indices
        m = members.numel()
        if m > 1:
            sub = boxes[members]
            idx = torch.triu_indices(m, m, 1)
            l, n = collision_loss(sub[idx[0]], sub[idx[1]], method="iou", weight=weight)
            scaled = scaled + l
            nominal = nominal + n
    return scaled, nominal


class TestLocalCollisionEquivalence(unittest.TestCase):
    def test_batched_equals_per_cluster_value_and_grad(self):
        bx, by = torch.tensor(BX), torch.tensor(BY)
        pairs = build_pairs(CLUSTERS)
        self.assertEqual(
            set(map(tuple, pairs.t().tolist())), {(0, 1), (2, 3)},
            "builder must emit exactly the within-cluster pairs, no cross pairs",
        )

        xb, yb, rzb = _leaves()
        boxes_b = pack_boxes(xb, yb, bx, by, rzb)
        lb, nb = collision_loss(boxes_b[pairs[0]], boxes_b[pairs[1]], method="iou", weight=1.0)

        xr, yr, rzr = _leaves()
        boxes_r = pack_boxes(xr, yr, bx, by, rzr)
        lr, nr = _per_cluster_reference(boxes_r, CLUSTERS, ORDER, weight=1.0)

        # The scene actually overlaps within clusters (otherwise this guards nothing).
        self.assertGreater(nb.item(), 0.1)
        self.assertTrue(torch.allclose(nb, nr, atol=1e-6, rtol=0), f"{nb} vs {nr}")
        self.assertTrue(torch.allclose(lb, lr, atol=1e-6, rtol=0), f"{lb} vs {lr}")

        # Leakage guard: the block-diagonal MUST exclude the overlapping cross
        # pair (chair-1, chair-2), so it is strictly below the all-pairs result.
        full = torch.triu_indices(4, 4, 1)
        _, n_all = collision_loss(boxes_b[full[0]], boxes_b[full[1]], method="iou", weight=1.0)
        self.assertGreater(n_all.item(), nb.item() + 0.1)

        lb.backward()
        lr.backward()
        for got, ref, name in ((xb, xr, "x"), (yb, yr, "y"), (rzb, rzr, "rz")):
            self.assertTrue(
                torch.allclose(got.grad, ref.grad, atol=1e-5, rtol=0),
                f"{name} grad: {got.grad} vs {ref.grad}",
            )

    def test_compiled_scene_coll_local_matches_reference_both_frames(self):
        constr_json = {
            "scene_entities": {
                "independent_objects": [],
                "clusters": [
                    {"cluster_id": "cluster_a",
                     "anchor": {"anchor_kind": "object", "anchor_object_id": "chair-0"},
                     "members": ["chair-0", "chair-1"]},
                    {"cluster_id": "cluster_b",
                     "anchor": {"anchor_kind": "object", "anchor_object_id": "chair-2"},
                     "members": ["chair-2", "chair-3"]},
                ],
            },
            "constraints": {
                "composition": {"horizontal": [], "vertical": [], "against_wall": [], "corner": []},
                "cluster_internal": {
                    "cluster_a": dict(EMPTY_RELATIONAL),
                    "cluster_b": dict(EMPTY_RELATIONAL),
                },
                "scene_relational": dict(EMPTY_RELATIONAL),
            },
            "constraint_params": {"names": [], "priors": [], "kinds": []},
        }
        bbox_vec = BBoxVec(x=torch.tensor(BX), y=torch.tensor(BY), z=torch.ones(4))
        compiled = compile_scene(OBJECTS, constr_json, bbox_vec, (8.0, 8.0))

        # compile built the block-diagonal field, not a flat all-pairs triu.
        self.assertEqual(
            set(map(tuple, compiled.scene.local_pairs.t().tolist())),
            {(0, 1), (2, 3)},
        )

        global_poses = PoseVec(
            x=torch.tensor(PX), y=torch.tensor(PY), rz=torch.tensor(RZ),
        )
        for reparam in (True, False):
            poses = compiled.scene.localize(global_poses.clone(), reparam=reparam)
            _, nominal = compiled.evaluate(poses, alpha=1.0, params=None, reparam=reparam)

            # Independent oracle: same augmented state, old per-cluster loop.
            aug = AugmentedState.build(compiled.scene, poses, bbox_vec, reparam=reparam)
            ref = torch.zeros(())
            for cid in compiled.scene.sorted_cids:
                members = compiled.scene.clusters[cid].member_indices
                if members.numel() > 1:
                    boxes = pack_boxes(
                        aug.poses.x[members], aug.poses.y[members],
                        aug.bbox.x[members], aug.bbox.y[members], aug.poses.rz[members],
                    )
                    m = members.numel()
                    idx = torch.triu_indices(m, m, 1)
                    _, n = collision_loss(boxes[idx[0]], boxes[idx[1]], method="iou", weight=1.0)
                    ref = ref + n

            self.assertGreater(ref.item(), 0.1, f"reparam={reparam}: scene should overlap")
            self.assertTrue(
                torch.allclose(nominal["coll_local"], ref, atol=1e-6, rtol=0),
                f"reparam={reparam}: coll_local {nominal['coll_local']} vs ref {ref}",
            )


if __name__ == "__main__":
    unittest.main()
