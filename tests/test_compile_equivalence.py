"""End-to-end compile -> evaluate equivalence on every captured scene.

This is one half of the refactor safety net (the other is
``test_frame_equivalence``). It pins the pinned ``compile`` -> ``evaluate``
behaviour on real scenes so any silent numeric drift through the ``compile`` /
``builders`` / kernel / clamp / prior path is caught by a single scalar+nominal
comparison.

Every ``tests/fixtures/<name>.json`` (paired with ``<name>.llm_output.py``) is a
self-contained scene baked by ``capture_snapshot.py`` -- the gitignored
``output/`` and ``data/`` trees are NOT touched here. Each fixture carries:
  - constr_json : parsed constraint JSON (the dsl-stability target).
  - objects     : ordered object_id list (canonical indices 0..N-1).
  - bbox        : per-object {x, y, z} dims (replaces asset annotations).
  - room_size   : [length, width].
  - pose        : fixed deterministic GLOBAL pose {x, y, rz(rad)}.
  - snapshot    : {loss, nominal{...}} captured from current code.

For each discovered fixture, two assertions run under their own subTest:
  (a) dsl is stable: ``code_to_json(parse_program(<name>.llm_output.py)) ==
      constr_json``.
  (b) compile -> evaluate on the fixed pose is ``torch.allclose`` to the
      snapshot (total loss + every shared nominal key, atol 1e-5). The 1e-5
      bound is set just above the cross-backend float drift (CUDA vs CPU/Mac
      BLAS differ at the ulp level); it still catches any real numeric
      regression through the compile/builders/kernel/clamp/prior path.

If a fixture's snapshot is null (captured in an asset-less environment), its
data-dependent assertion (b) SKIPs with a pointer to the capture script.
"""

import glob
import json
import os
import unittest

import torch

from solvers.r3l.config import cfg
from solvers.r3l.compile import compile
from solvers.r3l.dsl.parse_constraints import parse_program, code_to_json
from utils.r3l.types import BBoxVec, ParamVec, PoseVec

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

_CAPTURE_HINT = (
    "fixture snapshot is null -- run "
    "`python tests/fixtures/capture_snapshot.py <output/scene_dir>` in an "
    "asset-equipped environment to populate bbox + snapshot."
)


def _discover_fixtures() -> list[str]:
    """Return the fixture names (basename without .json) under FIXTURE_DIR."""
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(FIXTURE_DIR, "*.json"))
    )


def _assert_dsl_stable(case: unittest.TestCase, fixture: dict, code: str) -> None:
    """(a) Re-parsing the DSL reproduces the fixture's constr_json verbatim."""
    objects = fixture["objects"]
    var_to_obj_id = {oid.replace("-", "_"): oid for oid in objects}
    program = parse_program(code, var_to_obj_id, hv_absolute=cfg.prompt.hv_absolute)
    case.assertEqual(code_to_json(program), fixture["constr_json"])


def _assert_snapshot_matches(case: unittest.TestCase, fixture: dict) -> None:
    """(b) compile+evaluate on the fixed pose matches the snapshot."""
    snapshot = fixture["snapshot"]
    if snapshot is None:
        case.skipTest(_CAPTURE_HINT)

    device = cfg.runtime.device
    objects = fixture["objects"]
    bbox = fixture["bbox"]
    pose = fixture["pose"]
    room_size = tuple(fixture["room_size"])

    # Self-contained scene rebuild. compile/evaluate read only the
    # object_to_index map and bbox_vec -- never asset_info -- so the object_id
    # doubles as its own asset_id and the bbox comes straight from the fixture
    # arrays (no annotations needed).
    bbox_vec = BBoxVec(
        x=torch.tensor(bbox["x"], dtype=torch.float32, device=device),
        y=torch.tensor(bbox["y"], dtype=torch.float32, device=device),
        z=torch.tensor(bbox["z"], dtype=torch.float32, device=device),
    )
    # Canonical index order matches the bbox/pose row order.
    object_to_index = {oid: i for i, oid in enumerate(objects)}

    compiled = compile(fixture["constr_json"], object_to_index, bbox_vec, room_size)

    params = ParamVec.from_priors(list(compiled.params.priors), device)
    global_pose = PoseVec(
        x=torch.tensor(pose["x"], dtype=torch.float32, device=device),
        y=torch.tensor(pose["y"], dtype=torch.float32, device=device),
        rz=torch.tensor(pose["rz"], dtype=torch.float32, device=device),
    )
    localized = compiled.scene.localize(global_pose, reparam=True)
    with torch.no_grad():
        loss, nominal = compiled.evaluate(
            localized, alpha=1.0, params=params, reparam=True, train_var=True
        )

    # Total loss.
    case.assertTrue(
        torch.allclose(
            loss,
            torch.tensor(snapshot["loss"], dtype=loss.dtype, device=loss.device),
            atol=1e-5,
            rtol=0.0,
        ),
        f"total loss drifted: got {loss.item()!r}, snapshot {snapshot['loss']!r}",
    )

    # Every shared nominal key.
    live = {k: float(v.item()) for k, v in nominal.items()}
    expected = snapshot["nominal"]
    case.assertEqual(
        set(live),
        set(expected),
        f"nominal key set drifted: live-only={set(live) - set(expected)}, "
        f"snapshot-only={set(expected) - set(live)}",
    )
    for key in sorted(set(live) & set(expected)):
        case.assertTrue(
            torch.allclose(
                torch.tensor(live[key]),
                torch.tensor(expected[key]),
                atol=1e-5,
                rtol=0.0,
            ),
            f"nominal[{key!r}] drifted: got {live[key]!r}, "
            f"snapshot {expected[key]!r}",
        )


class TestCompileEquivalence(unittest.TestCase):
    """Pin compile->evaluate across every captured scene fixture."""

    def test_dsl_is_stable(self):
        """(a) Each fixture's DSL re-parses to its committed constr_json."""
        for name in _discover_fixtures():
            with self.subTest(fixture=name):
                with open(os.path.join(FIXTURE_DIR, f"{name}.json")) as f:
                    fixture = json.load(f)
                with open(os.path.join(FIXTURE_DIR, f"{name}.llm_output.py")) as f:
                    code = f.read()
                _assert_dsl_stable(self, fixture, code)

    def test_compiled_evaluate_matches_snapshot(self):
        """(b) Current compile+evaluate matches each fixture's snapshot."""
        for name in _discover_fixtures():
            with self.subTest(fixture=name):
                with open(os.path.join(FIXTURE_DIR, f"{name}.json")) as f:
                    fixture = json.load(f)
                _assert_snapshot_matches(self, fixture)


if __name__ == "__main__":
    unittest.main()
