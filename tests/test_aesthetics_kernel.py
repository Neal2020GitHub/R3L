"""Numerical guard for the `aesthetics_loss` kernel (snap-to-cardinals).

`aesthetics_loss` is live code (called in CompiledConstraints physics, gated by
`cfg.constraints.shapes.aesthetics.enabled`). The migration changed that gate +
the `acfg` reads, so the kernel needs a guard. The old test toggled the global
flag mid-test, which the frozen config forbids; this is the cfg-free replacement
(à la test_hv_kernels): drive the pure kernel directly on known rotations.

Contract pinned (rotation in radians; angles/eps in degrees, converted inside):
  - aligned to a target angle  -> d=0       -> loss 0
  - outside the +/- eps window -> gated off -> loss 0
  - inside the window          -> soft_landing(d/eps) * weight, scaled by the
    config's relative weight. At d=2deg, eps=5deg: x=0.4, smoothstep(0.4)=
    3*0.4^2 - 2*0.4^3 = 0.352.
A non-scalar return (the silent backward() failure) or a dropped weight/scale
breaks here.
"""

import math
import unittest

import torch

from solvers.r3l.config import cfg
from solvers.r3l.losses import aesthetics_loss
from utils.r3l.geometry import get_soft_landing


class TestAestheticsKernel(unittest.TestCase):
    def setUp(self):
        self.angles = [0, 90, 180, -90]   # cardinals
        self.eps = 5.0                    # degrees
        self.sl = get_soft_landing("smoothstep")
        self.scale = cfg.constraints.weights.aesthetics  # config read (frozen-safe)

    def test_aligned_is_zero(self):
        # rotation exactly at a target (0 rad == 0 deg) -> d=0 -> well bottom -> 0
        scaled, nominal = aesthetics_loss(
            rotation=torch.tensor([0.0]),
            angles_deg=self.angles, eps_deg=self.eps, soft_landing=self.sl,
        )
        self.assertEqual(scaled.dim(), 0)
        self.assertEqual(nominal.dim(), 0)
        self.assertAlmostEqual(nominal.item(), 0.0, places=6)

    def test_outside_window_is_zero(self):
        # 45deg from the nearest cardinal (> eps) -> hard-gated to 0
        scaled, nominal = aesthetics_loss(
            rotation=torch.tensor([math.radians(45.0)]),
            angles_deg=self.angles, eps_deg=self.eps, soft_landing=self.sl,
        )
        self.assertEqual(scaled.dim(), 0)
        self.assertAlmostEqual(nominal.item(), 0.0, places=6)

    def test_inside_window_value_and_scale(self):
        # 2deg from target 0, eps 5 -> x=0.4 -> smoothstep(0.4)=0.352
        w = 2.0
        scaled, nominal = aesthetics_loss(
            rotation=torch.tensor([math.radians(2.0)]),
            angles_deg=self.angles, eps_deg=self.eps, soft_landing=self.sl, weight=w,
        )
        self.assertEqual(scaled.dim(), 0)
        self.assertEqual(nominal.dim(), 0)
        self.assertAlmostEqual(nominal.item(), 0.352 * w, places=5)
        self.assertAlmostEqual(scaled.item(), 0.352 * w * self.scale, places=5)


if __name__ == "__main__":
    unittest.main()
