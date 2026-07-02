"""Numerical guard for the horizontal/vertical loss-kernel rewrite (§C).

The h/v losses were rewritten to route their per-element error through the
shared ``_position_loss_fn`` (``l2``/``l1``/``huber``) instead of a hard-coded
``margin``-shifted ReLU2. This test pins the drop-in numerical contract at
``loss_fn="l2"``: each function must return a SCALAR equal to
``(error**2).sum() * weight_scale``, where ``weight_scale`` is the constraint's
relative weight from config. A non-scalar (the silent ``backward()`` failure
mode) or a dropped reduction would break here.
"""

import unittest

import torch

from solvers.r3l.config import cfg
from solvers.r3l.losses import (
    horizontal_rel_loss,
    horizontal_abs_loss,
    vertical_rel_loss,
    vertical_abs_loss,
)


def _l2_expected(error: torch.Tensor, scale: float) -> float:
    # Mirrors the kernel exactly: per = error**2; per.sum()*scale.
    return float((error ** 2).sum().item() * scale)


class TestHVKernels(unittest.TestCase):
    def test_horizontal_abs_l2_scalar_and_value(self):
        position_x = torch.tensor([1.0, 4.0])
        x_tgt = torch.tensor([0.0, 5.0])
        scale = cfg.constraints.weights.horizontal

        scaled, nominal = horizontal_abs_loss(
            position_x=position_x,
            x=x_tgt,
            room_size=(5.0, 6.0),
            loss_fn="l2",
            huber_delta=1.0,
        )

        # error = position_x - x_tgt = [1.0, -1.0]
        error = position_x - x_tgt
        self.assertEqual(scaled.dim(), 0)
        self.assertEqual(nominal.dim(), 0)
        self.assertAlmostEqual(scaled.item(), _l2_expected(error, scale), places=5)

    def test_vertical_abs_l2_scalar_and_value(self):
        position_y = torch.tensor([2.0, 1.0])
        y_tgt = torch.tensor([0.0, 3.0])
        scale = cfg.constraints.weights.vertical

        scaled, nominal = vertical_abs_loss(
            position_y=position_y,
            y=y_tgt,
            room_size=(5.0, 6.0),
            loss_fn="l2",
            huber_delta=1.0,
        )

        error = position_y - y_tgt
        self.assertEqual(scaled.dim(), 0)
        self.assertEqual(nominal.dim(), 0)
        self.assertAlmostEqual(scaled.item(), _l2_expected(error, scale), places=5)

    def test_horizontal_rel_l2_scalar_and_value(self):
        # rotation = 0 => horizontal OBB span = bbox_x, so x_tgt is closed-form.
        position_x = torch.tensor([1.0, 4.0])
        rotation = torch.zeros(2)
        bbox_x = torch.tensor([1.0, 2.0])
        bbox_y = torch.tensor([1.0, 1.0])
        percentile = torch.tensor([0.0, 1.0])
        room_size = (5.0, 6.0)
        scale = cfg.constraints.weights.horizontal

        scaled, nominal = horizontal_rel_loss(
            position_x=position_x,
            rotation=rotation,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            percentile=percentile,
            room_size=room_size,
            loss_fn="l2",
            huber_delta=1.0,
        )

        # rotation=0 -> w_span = bbox_x; x_tgt = 0.5*w_span + percentile*(W - w_span)
        W = room_size[0]
        w_span = bbox_x
        x_tgt = 0.5 * w_span + percentile * (W - w_span)
        error = position_x - x_tgt
        self.assertEqual(scaled.dim(), 0)
        self.assertEqual(nominal.dim(), 0)
        self.assertAlmostEqual(scaled.item(), _l2_expected(error, scale), places=5)

    def test_vertical_rel_l2_scalar_and_value(self):
        # rotation = 0 => vertical OBB span = bbox_y, so y_tgt is closed-form.
        position_y = torch.tensor([2.0, 5.0])
        rotation = torch.zeros(2)
        bbox_x = torch.tensor([1.0, 1.0])
        bbox_y = torch.tensor([1.0, 2.0])
        percentile = torch.tensor([0.0, 1.0])
        room_size = (5.0, 6.0)
        scale = cfg.constraints.weights.vertical

        scaled, nominal = vertical_rel_loss(
            position_y=position_y,
            rotation=rotation,
            bbox_x=bbox_x,
            bbox_y=bbox_y,
            percentile=percentile,
            room_size=room_size,
            loss_fn="l2",
            huber_delta=1.0,
        )

        # rotation=0 -> h_span = bbox_y; y_tgt = 0.5*h_span + percentile*(H - h_span)
        H = room_size[1]
        h_span = bbox_y
        y_tgt = 0.5 * h_span + percentile * (H - h_span)
        error = position_y - y_tgt
        self.assertEqual(scaled.dim(), 0)
        self.assertEqual(nominal.dim(), 0)
        self.assertAlmostEqual(scaled.item(), _l2_expected(error, scale), places=5)


if __name__ == "__main__":
    unittest.main()
