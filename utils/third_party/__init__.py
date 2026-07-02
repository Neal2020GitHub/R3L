import os
import sys

_this_dir = os.path.dirname(__file__)
_rotated_iou_dir = os.path.join(_this_dir, "Rotated_IoU")
if os.path.isdir(_rotated_iou_dir) and _rotated_iou_dir not in sys.path:
    sys.path.append(_rotated_iou_dir)

_cuda_op_dir = os.path.join(_rotated_iou_dir, "cuda_op")
_build_dir = os.path.join(_cuda_op_dir, "build")
_dist_dir = os.path.join(_cuda_op_dir, "dist")

if os.path.isdir(_build_dir):
    for root, dirs, files in os.walk(_build_dir):
        if root not in sys.path and any(name.startswith("sort_vertices") for name in files):
            sys.path.append(root)

if os.path.isdir(_dist_dir):
    for name in os.listdir(_dist_dir):
        if name.endswith('.egg'):
            egg_path = os.path.join(_dist_dir, name)
            if egg_path not in sys.path:
                sys.path.append(egg_path)
