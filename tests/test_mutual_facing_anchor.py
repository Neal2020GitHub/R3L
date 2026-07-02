"""
Tests for mutual facing + anchor injection logic.

When cluster_internal has facing(mutual=True) involving the anchor, a direction-only
in_front_of constraint is injected to push non-anchor members to anchor's +y front.
This compensates for anchor's fixed (0,0,0) pose in cluster-local coordinates.

Verifies:
1. Injection triggers when mutual facing involves anchor
2. No gap_loss is created by the injection
3. The injected loss produces gradients on member's y position
"""

import unittest
import torch

from utils.r3l.types import BBoxVec, PoseVec
from solvers.r3l.cluster import AugmentedState
from tests.lib import EMPTY_RELATIONAL, compile_scene


def _json(members, facing):
    """A single-cluster scene (anchor=chair-0) carrying one `facing` rule."""
    internal = dict(EMPTY_RELATIONAL, facing=[facing])
    return {
        "scene_entities": {
            "independent_objects": [],
            "clusters": [
                {
                    "cluster_id": "seat_group",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "chair-0"},
                    "members": members,
                }
            ],
        },
        "constraints": {
            "composition": {"horizontal": [], "vertical": [], "against_wall": [], "corner": []},
            "cluster_internal": {"seat_group": internal},
            "scene_relational": dict(EMPTY_RELATIONAL),
        },
        "constraint_params": {"names": [], "priors": [], "kinds": []},
    }


def _compile(members, facing):
    """Compile a cluster scene; return (compiled, bbox_vec)."""
    n = len(members)
    bbox_vec = BBoxVec(
        x=torch.full((n,), 0.5),
        y=torch.full((n,), 0.5),
        z=torch.ones(n),
    )
    return compile_scene(members, _json(members, facing), bbox_vec, (5.0, 5.0)), bbox_vec


# A mutual radial facing between a member and the anchor.
_FACING_MEMBER_ANCHOR = {
    "src_kind": "object", "src_id": "chair-1",
    "tar_kind": "anchor", "tar_id": "anchor",
    "mutual": True, "mode": "radial",
}


class TestMutualFacingAnchorInjection(unittest.TestCase):
    """Test direction-only in_front_of injection for mutual facing with anchor.

    Device comes from the tests/ CPU bootstrap (no per-test cfg mutation)."""

    def test_injection_creates_loss_key(self):
        """Verify that mutual facing with anchor injects in_front_of_loss."""
        compiled, _ = _compile(["chair-0", "chair-1"], _FACING_MEMBER_ANCHOR)
        self.assertIn(
            "in_front_of_loss",
            compiled.constraints,
            "Expected in_front_of_loss to be injected for mutual facing with anchor",
        )

    def test_no_gap_loss_from_injection(self):
        """Verify that the injection does NOT create gap_loss."""
        compiled, _ = _compile(["chair-0", "chair-1"], _FACING_MEMBER_ANCHOR)
        self.assertNotIn(
            "gap_loss",
            compiled.constraints,
            "gap_loss should NOT be created by direction-only in_front_of injection",
        )

    def test_injected_loss_has_gradient_on_member_y(self):
        """Verify the injected loss produces gradient on non-anchor member's y."""
        compiled, bbox_vec = _compile(["chair-0", "chair-1"], _FACING_MEMBER_ANCHOR)

        # Layout: anchor at (0,0), member at (1,0) - NOT in front of anchor (+y).
        # chair-0 is index 0 (anchor), chair-1 is index 1 (member).
        layout = PoseVec(
            x=torch.tensor([0.0, 1.0]),
            y=torch.tensor([0.0, 0.0], requires_grad=True),
            rz=torch.tensor([0.0, 0.0]),
        )
        # K>0 builds a cluster handle; this scene's injected term reads only the
        # object slice, so a global-frame build (reparam=False) keeps the member
        # poses untouched for a direct gradient check.
        aug = AugmentedState.build(compiled.scene, layout, bbox_vec, reparam=False)

        loss_fn = compiled.constraints["in_front_of_loss"]
        loss, _ = loss_fn.evaluate(aug, None)
        loss.backward()

        # Member (index 1) y should have non-zero gradient.
        assert layout.y.grad is not None, "y.grad should not be None after backward"
        member_y_grad = layout.y.grad[1].item()
        self.assertNotEqual(
            member_y_grad,
            0.0,
            f"Expected non-zero gradient on member's y, got {member_y_grad}",
        )

    def test_no_injection_when_not_mutual(self):
        """Verify that injection does NOT happen when mutual=False."""
        facing = dict(_FACING_MEMBER_ANCHOR, mutual=False)
        compiled, _ = _compile(["chair-0", "chair-1"], facing)
        self.assertNotIn(
            "in_front_of_loss",
            compiled.constraints,
            "Should NOT inject when mutual=False",
        )

    def test_no_injection_when_no_anchor_involved(self):
        """Verify that injection does NOT happen when facing doesn't involve anchor."""
        # Mutual facing between two non-anchor members (chair-1 and chair-2).
        facing = {
            "src_kind": "object", "src_id": "chair-1",
            "tar_kind": "object", "tar_id": "chair-2",
            "mutual": True, "mode": "radial",
        }
        compiled, _ = _compile(["chair-0", "chair-1", "chair-2"], facing)
        self.assertNotIn(
            "in_front_of_loss",
            compiled.constraints,
            "Should NOT inject when anchor is not involved in mutual facing",
        )


if __name__ == "__main__":
    unittest.main()
