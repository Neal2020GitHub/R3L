"""Golden equivalence net for the reparam-frame collapse (Step 1a of the
CompiledConstraints refactor).

The objective is the single ``CompiledConstraints.evaluate(..., reparam: bool)``,
which consumes two DIFFERENT physical pose layouts for the same scene:

  - ``reparam=True``  (reparam frame): cluster anchors are zeroed to a local
    origin and non-anchor members live in anchor-local coordinates. It is fed
    ``localize(global, reparam=True)``.
  - ``reparam=False`` (no_localize ablation frame): every object keeps its raw
    global pose. It is fed the global poses untouched.

This test pins exactly where the two frames must agree and where they must
legitimately diverge, so any "silently flattened both frames" drift cannot pass
green.

Load-bearing caveat: the two frames consume different physical layouts, so we
NEVER reuse one tensor across both -- ``reparam=True`` zeroes anchors in place
via its augmented build, ``reparam=False`` does not. ``eval_both`` builds a fresh
PoseVec for each frame from the same global x/y and the given rotations.

Regimes (alpha=1.0; once with params=None, once with a non-None ParamVec;
``torch.allclose(atol=1e-6, rtol=0)`` on total loss AND every shared nominal key):

  1. aesthetics DISABLED, any anchor rz                  -> EQUAL.
     The two frames differ only by a rigid transform of relative geometry, and
     the live cluster constraints (facing / injected in_front_of) are
     frame-invariant relative quantities.
  2. aesthetics ENABLED + cardinal anchor rz in {0, pi/2} -> EQUAL.
     Aesthetics targets are the cardinals (spaced 90 deg), so shifting a member
     rotation by a cardinal anchor permutes the target set: the local-frame and
     global-frame snap distances coincide.
  3. aesthetics ENABLED + non-cardinal anchor rz~0.4 + member near a cardinal
     -> the 'aesthetics' nominal key DIFFERS by > 1e-3.
     In the reparam frame the member's snap is judged at its LOCAL rotation
     (member - anchor, far from any cardinal -> gated off); reparam=False judges
     the same member at its near-cardinal GLOBAL rotation -> active.

Plus a K=0 (single independent object, no clusters) scene asserting TOTAL loss
equality only: the cluster-vs-no-cluster nominal keys legitimately differ once
collapsed (coll vs coll_global/coll_local), so only the scalar total is pinned.
"""

import math
import unittest
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch

import solvers.r3l.physics as physics_mod
from solvers.r3l.constraints import CompiledConstraints
from tests.lib import EMPTY_RELATIONAL, compile_scene
from utils.r3l.types import BBoxVec, ParamVec, PoseVec

ATOL = 1e-6


@contextmanager
def aesthetics(enabled: bool):
    """Temporarily flip cfg.constraints.shapes.aesthetics.enabled.

    The config is a frozen pydantic model, so the only sound toggle is to build
    a copy with the leaf updated and rebind the module-global ``cfg`` that
    ``physics`` reads for the aesthetics gate + snap angles.
    """
    base = physics_mod.cfg
    shape = base.constraints.shapes.aesthetics.model_copy(update={"enabled": enabled})
    shapes = base.constraints.shapes.model_copy(update={"aesthetics": shape})
    cons = base.constraints.model_copy(update={"shapes": shapes})
    patched = base.model_copy(update={"constraints": cons})
    physics_mod.cfg = patched
    try:
        yield
    finally:
        physics_mod.cfg = base


# A two-chair cluster: anchor=chair-0, members=[chair-0, chair-1], with a mutual
# facing involving the anchor. Parsing this injects a frame-invariant
# in_front_of constraint (see test_mutual_facing_anchor) and a facing (fr_loss),
# so ``self.clusters`` is non-empty and the never-exercised cluster path is hit.
def _cluster_json() -> dict:
    return {
        "scene_entities": {
            "independent_objects": [],
            "clusters": [
                {
                    "cluster_id": "seat_duo",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "chair-0"},
                    "members": ["chair-0", "chair-1"],
                }
            ],
        },
        "constraints": {
            "composition": {"horizontal": [], "vertical": [], "against_wall": [], "corner": []},
            "cluster_internal": {
                "seat_duo": dict(EMPTY_RELATIONAL, facing=[
                    {
                        "src_kind": "object",
                        "src_id": "chair-1",
                        "tar_kind": "anchor",
                        "tar_id": "anchor",
                        "mutual": True,
                        "mode": "radial",
                    }
                ])
            },
            "scene_relational": dict(EMPTY_RELATIONAL),
        },
        "constraint_params": {"names": [], "priors": [], "kinds": []},
    }


def _single_json() -> dict:
    """One independent object, zero clusters -> K=0 path."""
    return {
        "scene_entities": {"independent_objects": ["chair-0"], "clusters": []},
        "constraints": {
            "composition": {"horizontal": [], "vertical": [], "against_wall": [], "corner": []},
            "cluster_internal": {},
            "scene_relational": dict(EMPTY_RELATIONAL),
        },
        "constraint_params": {"names": [], "priors": [], "kinds": []},
    }


def _compile(asset_ids, constr_json: dict) -> CompiledConstraints:
    n = len(asset_ids)
    bbox_vec = BBoxVec(
        x=torch.ones(n),
        y=torch.ones(n),
        z=torch.ones(n),
    )
    return compile_scene(asset_ids, constr_json, bbox_vec, (5.0, 5.0))


def _global_poses(x, y, rz) -> PoseVec:
    return PoseVec(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.float32),
        rz=torch.tensor(rz, dtype=torch.float32),
    )


def _eval_both(
    compiled: CompiledConstraints,
    global_poses: PoseVec,
    params: Optional[ParamVec],
) -> Tuple[Tuple[torch.Tensor, Dict[str, torch.Tensor]], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """Run both frames on the SAME scene from two independent pose tensors.

    reparam frame:  localize(global, reparam=True) -> evaluate(reparam=True)
    no_localize frame:  raw global                 -> evaluate(reparam=False)

    Each frame gets its own freshly cloned global PoseVec, since the two frames
    consume different physical layouts and must never share a tensor.
    """
    reparam_in = compiled.scene.localize(global_poses.clone(), reparam=True)
    reparam_out = compiled.evaluate(reparam_in, alpha=1.0, params=params, reparam=True)

    naive_in = global_poses.clone()
    naive_out = compiled.evaluate(naive_in, alpha=1.0, params=params, reparam=False)
    return reparam_out, naive_out


# Anchor at off-origin, member offset along +x; rotations supplied per regime.
_GLOBAL_XY = ([1.0, 2.0], [1.0, 1.0])
_EMPTY_PARAMS = ParamVec(values=torch.zeros(0, dtype=torch.float32))


class TestFrameEquivalence(unittest.TestCase):
    """Pin reparam=True vs reparam=False on a real cluster."""

    def _assert_total_and_shared_keys_equal(self, reparam_out, naive_out, msg: str):
        l_rep, n_rep = reparam_out
        l_nai, n_nai = naive_out
        self.assertTrue(
            torch.allclose(l_rep, l_nai, atol=ATOL, rtol=0),
            f"{msg}: total loss differs (rep={l_rep.item()} nai={l_nai.item()})",
        )
        shared = set(n_rep) & set(n_nai)
        self.assertTrue(shared, f"{msg}: expected a non-empty shared nominal key set")
        for key in shared:
            self.assertTrue(
                torch.allclose(n_rep[key], n_nai[key], atol=ATOL, rtol=0),
                f"{msg}: nominal['{key}'] differs (rep={float(n_rep[key])} nai={float(n_nai[key])})",
            )

    def _run_cluster(self, anchor_rz: float, member_rz: float, params: Optional[ParamVec]):
        compiled = _compile(["chair-0", "chair-1"], _cluster_json())
        self.assertTrue(compiled.scene.clusters, "cluster scene must have a non-empty cluster set")
        xs, ys = _GLOBAL_XY
        poses = _global_poses(xs, ys, [anchor_rz, member_rz])
        return _eval_both(compiled, poses, params)

    # ----- Regime 1: aesthetics OFF, any anchor rz -> EQUAL -----
    def test_regime1_aesthetics_disabled_equal(self):
        for params, ptag in ((None, "params=None"), (_EMPTY_PARAMS, "params=ParamVec")):
            for anchor_rz in (0.0, 0.4, math.pi / 2):
                with self.subTest(anchor_rz=anchor_rz, params=ptag):
                    with aesthetics(False):
                        rep, nai = self._run_cluster(anchor_rz, member_rz=0.3, params=params)
                    self._assert_total_and_shared_keys_equal(
                        rep, nai, f"regime1 aesthetics-off anchor_rz={anchor_rz} {ptag}"
                    )

    # ----- Regime 2: aesthetics ON, cardinal anchor rz -> EQUAL -----
    def test_regime2_aesthetics_enabled_cardinal_equal(self):
        # member near cardinal (0.02 rad ~ 1.15 deg, inside the 5 deg snap window)
        # so the aesthetics nominal is genuinely NON-zero here, not vacuously equal.
        for params, ptag in ((None, "params=None"), (_EMPTY_PARAMS, "params=ParamVec")):
            for anchor_rz in (0.0, math.pi / 2):
                with self.subTest(anchor_rz=anchor_rz, params=ptag):
                    with aesthetics(True):
                        rep, nai = self._run_cluster(anchor_rz, member_rz=0.02, params=params)
                    self.assertGreater(
                        float(rep[1]["aesthetics"]),
                        0.0,
                        f"regime2 should exercise a non-zero aesthetics snap {ptag}",
                    )
                    self._assert_total_and_shared_keys_equal(
                        rep, nai, f"regime2 aesthetics-on cardinal anchor_rz={anchor_rz} {ptag}"
                    )

    # ----- Regime 3: aesthetics ON, non-cardinal anchor -> aesthetics DIFFERS -----
    def test_regime3_aesthetics_enabled_noncardinal_differs(self):
        for params, ptag in ((None, "params=None"), (_EMPTY_PARAMS, "params=ParamVec")):
            with self.subTest(params=ptag):
                with aesthetics(True):
                    rep, nai = self._run_cluster(anchor_rz=0.4, member_rz=0.02, params=params)
                aest_rep = float(rep[1]["aesthetics"])
                aest_nai = float(nai[1]["aesthetics"])
                self.assertGreater(
                    abs(aest_rep - aest_nai),
                    1e-3,
                    f"regime3 expected aesthetics frames to diverge >1e-3 "
                    f"(rep={aest_rep} nai={aest_nai}) {ptag}",
                )

    # ----- K=0: single independent object -> TOTAL loss equality only -----
    def test_k0_single_object_total_equal(self):
        # Corner placement: the room spans [0, 5] x [0, 5] (wall_loss origin is a
        # room corner, not the center), so a 1x1 object centered at (4.8, 4.8)
        # pokes the +x/+y walls; a near-cardinal rotation also snaps aesthetics.
        # The pinned total is thus non-zero in both aesthetics modes -- not a
        # vacuous 0 == 0.
        for aesthetics_on in (False, True):
            for params, ptag in ((None, "params=None"), (_EMPTY_PARAMS, "params=ParamVec")):
                with self.subTest(aesthetics=aesthetics_on, params=ptag):
                    compiled = _compile(["chair-0"], _single_json())
                    self.assertFalse(compiled.scene.clusters, "K=0 scene must have no clusters")
                    poses = _global_poses([4.8], [4.8], [0.02])
                    with aesthetics(aesthetics_on):
                        (l_rep, _), (l_nai, _) = _eval_both(compiled, poses, params)
                    self.assertTrue(
                        torch.allclose(l_rep, l_nai, atol=ATOL, rtol=0),
                        f"K=0 total loss differs (rep={l_rep.item()} nai={l_nai.item()}) "
                        f"aesthetics={aesthetics_on} {ptag}",
                    )
                    self.assertGreater(
                        l_rep.item(), 0.0, "K=0 scene should pin a non-zero total"
                    )


if __name__ == "__main__":
    unittest.main()
