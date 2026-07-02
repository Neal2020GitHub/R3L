"""Differentiable rotated IoU utilities implemented in pure PyTorch.

Canonical module layout:
- losses: public IoU/GIoU/DIoU APIs for rotated 2D/3D boxes.
- intersection: oriented 2D box intersection geometry.
- enclosing_box: enclosing-box helpers used by GIoU/DIoU.
- vertex_sort: vectorized PyTorch sorting for intersection polygon vertices.

The legacy module aliases remain for existing R3L imports.
"""

import sys

from . import enclosing_box, intersection, losses, vertex_sort

# Backward-compatible module aliases. New code should import the canonical names above.
oriented_iou_loss = losses
box_intersection_2d = intersection
min_enclosing_box = enclosing_box

sys.modules[__name__ + ".oriented_iou_loss"] = losses
sys.modules[__name__ + ".box_intersection_2d"] = intersection
sys.modules[__name__ + ".min_enclosing_box"] = enclosing_box

__all__ = [
    "losses",
    "intersection",
    "enclosing_box",
    "vertex_sort",
    "oriented_iou_loss",
    "box_intersection_2d",
    "min_enclosing_box",
]
