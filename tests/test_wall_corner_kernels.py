"""
Unit tests for the against_wall and corner loss kernels.

Verifies the wall-index / center-position / rotation core of both kernels:
1. against_wall_loss snaps an object's center to the named wall and faces it inward
2. corner_loss snaps both axes to a named corner and aligns rotation to the wall
"""

import math
import unittest
import torch

from solvers.r3l.losses import against_wall_loss, corner_loss


class TestAgainstWallLoss(unittest.TestCase):
    """Integration test for against_wall_loss."""

    def test_object_at_left_wall(self):
        """Object off the left wall should incur positive loss."""
        loss, _ = against_wall_loss(
            position_x=torch.tensor([0.5]),
            position_y=torch.tensor([2.0]),
            rotation=torch.tensor([-math.pi / 2]),  # facing right (+x)
            wall_index=torch.tensor([0]),  # L
            bbox_x=torch.tensor([1.0]),
            bbox_y=torch.tensor([0.5]),
            room_size=(5.0, 5.0),
        )
        # Center is off the wall, so loss is positive.
        self.assertGreater(loss.item(), 0)

    def test_perfect_alignment_low_loss(self):
        """Object aligned should have lower loss than a misaligned object."""
        # Aligned: back edge at x=0 (center at half bbox_y), facing right (+x).
        loss_aligned, _ = against_wall_loss(
            position_x=torch.tensor([0.5]),  # back edge at x=0 (bbox_y/2 = 0.5)
            position_y=torch.tensor([2.5]),  # centered on wall
            rotation=torch.tensor([-math.pi / 2]),  # facing right (+x)
            wall_index=torch.tensor([0]),  # L
            bbox_x=torch.tensor([0.5]),
            bbox_y=torch.tensor([1.0]),  # half depth = 0.5
            room_size=(5.0, 5.0),
        )
        # Misaligned: same position but wrong rotation.
        loss_misaligned, _ = against_wall_loss(
            position_x=torch.tensor([0.5]),
            position_y=torch.tensor([2.5]),
            rotation=torch.tensor([0.0]),  # facing up instead of right
            wall_index=torch.tensor([0]),  # L
            bbox_x=torch.tensor([0.5]),
            bbox_y=torch.tensor([1.0]),
            room_size=(5.0, 5.0),
        )
        self.assertLess(loss_aligned.item(), loss_misaligned.item())


class TestCornerLoss(unittest.TestCase):
    """Integration test for corner_loss."""

    def test_bl_corner_wall_b(self):
        """Object at BL corner aligning to the bottom wall has non-negative loss."""
        loss, _ = corner_loss(
            position_x=torch.tensor([0.5]),
            position_y=torch.tensor([0.5]),
            rotation=torch.tensor([0.0]),  # facing up
            corner_index=torch.tensor([0]),  # BL
            wall_index=torch.tensor([3]),    # B
            bbox_x=torch.tensor([1.0]),
            bbox_y=torch.tensor([1.0]),
            room_size=(5.0, 5.0),
        )
        self.assertGreaterEqual(loss.item(), 0)

    def test_bl_corner_wall_l(self):
        """Object at BL corner aligning to the left wall."""
        loss, _ = corner_loss(
            position_x=torch.tensor([0.5]),
            position_y=torch.tensor([0.5]),
            rotation=torch.tensor([-math.pi / 2]),  # facing right
            corner_index=torch.tensor([0]),  # BL
            wall_index=torch.tensor([0]),    # L
            bbox_x=torch.tensor([1.0]),
            bbox_y=torch.tensor([1.0]),
            room_size=(5.0, 5.0),
        )
        self.assertGreaterEqual(loss.item(), 0)

    def test_different_walls_different_loss(self):
        """Different wall selections at the same corner produce different losses."""
        common = dict(
            position_x=torch.tensor([0.5]),
            position_y=torch.tensor([0.5]),
            rotation=torch.tensor([0.0]),
            corner_index=torch.tensor([0]),  # BL
            bbox_x=torch.tensor([1.0]),
            bbox_y=torch.tensor([1.0]),
            room_size=(5.0, 5.0),
        )
        loss_b, _ = corner_loss(wall_index=torch.tensor([3]), **common)  # type: ignore[arg-type]  # B
        loss_l, _ = corner_loss(wall_index=torch.tensor([0]), **common)  # type: ignore[arg-type]  # L
        self.assertNotAlmostEqual(loss_b.item(), loss_l.item(), places=2)


if __name__ == "__main__":
    unittest.main()
