import math
import sys
import unittest

import torch

from utils.third_party.Rotated_IoU import losses as rotated_iou


FIXTURES = [
    [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
    [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0],
    [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0],
    [0.0, 0.0, 2.0, 2.0, 0.0, 5.0, 5.0, 2.0, 2.0, 0.0],
    [0.0, 0.0, 4.0, 4.0, 0.0, 0.5, 0.25, 1.0, 1.5, 0.25],
    [0.0, 0.0, 2.0, 3.0, math.pi / 6.0, 1.0, 1.0, 4.0, 4.0, -math.pi / 4.0],
    [38.0, 120.0, 1.3, 20.0, 50.0, 38.0, 120.0, 1.3, 20.0, 50.0],
]

EXPECTED = {
    "cal_iou": {
        "loss": [[0.0, 1.0, 0.6666666269302368, 1.0, 0.90625, 0.7194992303848267, 0.0]],
        "iou": [[1.0, 0.0, 0.3333333432674408, 0.0, 0.0937500074505806, 0.28050076961517334, 1.0]],
    },
    "cal_my_iou": {
        "loss": [[1.0, 0.0, 0.3333333432674408, 0.0, 0.0937500074505806, 0.28050076961517334, 1.0]],
        "iou": [[1.0, 0.0, 0.3333333432674408, 0.0, 0.0937500074505806, 0.28050076961517334, 1.0]],
    },
    "cal_my_diou": {
        "loss": [[1.0, 0.0, 0.29487180709838867, 0.0, 0.0839843824505806, 0.23903155326843262, 1.0]],
        "iou": [[1.0, 0.0, 0.3333333432674408, 0.0, 0.0937500074505806, 0.28050076961517334, 1.0]],
    },
    "cal_my_giou": {
        "loss": [[1.0, 0.0, 0.3333333432674408, -0.7142857313156128, 0.0937500074505806, 0.18126899003982544, 0.9999998807907104]],
        "iou": [[1.0, 0.0, 0.3333333432674408, 0.0, 0.0937500074505806, 0.28050076961517334, 1.0]],
    },
    "cal_giou": {
        "loss": [[0.0, 1.0, 0.6666666269302368, 1.7142857313156128, 0.90625, 0.8187310099601746, 1.4671910264496546e-07]],
        "iou": [[1.0, 0.0, 0.3333333432674408, 0.0, 0.0937500074505806, 0.28050076961517334, 1.0]],
    },
}

GRAD_FIXTURES = [
    [0.0, 0.0, 2.0, 3.0, math.pi / 6.0, 1.0, 1.0, 4.0, 4.0, -math.pi / 4.0],
    [0.2, -0.1, 2.5, 1.2, 0.4, 0.7, 0.3, 1.7, 2.0, -0.35],
]

EXPECTED_GRAD_BOX1 = [[
    [0.16197623312473297, 0.16197620332241058, 0.06398192793130875, 0.08676546812057495, -0.03224889934062958],
    [0.25426796078681946, 0.21965870261192322, 0.042653314769268036, 0.15435977280139923, 0.03505191206932068],
]]

EXPECTED_GRAD_BOX2 = [[
    [-0.16197621822357178, -0.16197621822357178, -0.06530572474002838, 0.049228765070438385, 0.032248929142951965],
    [-0.25426799058914185, -0.21965867280960083, -0.02252328395843506, -0.0192890465259552, -0.026929795742034912],
]]

BATCH_FIXTURES = [
    [
        [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.0, 2.0, 0.0],
        [0.0, 0.0, 2.0, 3.0, math.pi / 6.0, 1.0, 1.0, 4.0, 4.0, -math.pi / 4.0],
    ],
    [
        [0.0, 0.0, 4.0, 4.0, 0.0, 0.5, 0.25, 1.0, 1.5, 0.25],
        [38.0, 120.0, 1.3, 20.0, 50.0, 38.0, 120.0, 1.3, 20.0, 50.0],
    ],
]

EXPECTED_BATCH_CAL_IOU = {
    "loss": [[0.6666666269302368, 0.7194992303848267], [0.90625, 0.0]],
    "iou": [[0.3333333432674408, 0.28050076961517334], [0.0937500074505806, 1.0]],
}

EXPECTED_HIGH_COORD_SAME_BOX_SORT = [[[0, 1, 2, 0, 1, 2, 0, 8, 8]]]


def _boxes_from_fixture(rows):
    fixtures = torch.tensor(rows, dtype=torch.float32)
    return fixtures[:, :5].unsqueeze(0), fixtures[:, 5:].unsqueeze(0)


class TestRotatedIoUPyTorch(unittest.TestCase):
    def test_import_and_collision_loss_run_on_cpu_without_compiled_extension(self):
        self.assertNotIn("sort_vertices", sys.modules)

        from solvers.r3l.losses import collision_loss, pack_boxes

        bbox_x = torch.tensor([2.0, 2.0], dtype=torch.float32)
        bbox_y = torch.tensor([2.0, 2.0], dtype=torch.float32)
        position_x = torch.tensor([0.0, 1.0], dtype=torch.float32)
        position_y = torch.tensor([0.0, 0.0], dtype=torch.float32)
        rotation = torch.tensor([0.0, 0.0], dtype=torch.float32)

        boxes = pack_boxes(position_x, position_y, bbox_x, bbox_y, rotation)
        idx = torch.triu_indices(2, 2, 1)
        loss, nominal = collision_loss(
            boxes[idx[0]], boxes[idx[1]], method="iou", weight=1.0
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(nominal))
        # Read the UNSCALED nominal (2nd return): scale-independent overlap signal.
        self.assertGreater(nominal.item(), 0.0)

    def test_golden_scalar_outputs(self):
        box1, box2 = _boxes_from_fixture(FIXTURES)

        for fn_name, expected in EXPECTED.items():
            with self.subTest(fn_name=fn_name):
                loss, iou = getattr(rotated_iou, fn_name)(box1, box2)
                torch.testing.assert_close(
                    loss, torch.tensor(expected["loss"], dtype=torch.float32), rtol=1e-5, atol=1e-5
                )
                torch.testing.assert_close(
                    iou, torch.tensor(expected["iou"], dtype=torch.float32), rtol=1e-5, atol=1e-5
                )

    def test_golden_gradients_for_collision_loss_kernel(self):
        box1, box2 = _boxes_from_fixture(GRAD_FIXTURES)
        box1 = box1.clone().detach().requires_grad_(True)
        box2 = box2.clone().detach().requires_grad_(True)
        loss, iou = rotated_iou.cal_my_iou(box1, box2)
        loss.sum().backward()

        self.assertTrue(torch.isfinite(loss).all())
        self.assertTrue(torch.isfinite(iou).all())
        assert box1.grad is not None
        self.assertTrue(torch.isfinite(box1.grad).all())
        torch.testing.assert_close(
            box1.grad, torch.tensor(EXPECTED_GRAD_BOX1, dtype=torch.float32), rtol=2e-5, atol=2e-5
        )
        assert box2.grad is not None
        self.assertTrue(torch.isfinite(box2.grad).all())
        torch.testing.assert_close(
            box2.grad, torch.tensor(EXPECTED_GRAD_BOX2, dtype=torch.float32), rtol=2e-5, atol=2e-5
        )

    def test_golden_batched_outputs(self):
        fixtures = torch.tensor(BATCH_FIXTURES, dtype=torch.float32)
        box1, box2 = fixtures[..., :5], fixtures[..., 5:]
        loss, iou = rotated_iou.cal_iou(box1, box2)
        torch.testing.assert_close(
            loss, torch.tensor(EXPECTED_BATCH_CAL_IOU["loss"], dtype=torch.float32), rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            iou, torch.tensor(EXPECTED_BATCH_CAL_IOU["iou"], dtype=torch.float32), rtol=1e-5, atol=1e-5
        )

    def test_reference_containment_tolerance_and_duplicate_sort_case(self):
        from utils.third_party.Rotated_IoU import intersection

        self.assertEqual(intersection.BOX_IN_BOX_EPSILON, 1e-6)

        box = torch.tensor([[[38.0, 120.0, 1.3, 20.0, 50.0]]], dtype=torch.float32)
        corners = rotated_iou.box2corners_th(box)
        inters, mask_inter = intersection.box_intersection_th(corners, corners)
        c12, c21 = intersection.box_in_box_th(corners, corners)
        vertices, mask = intersection.build_vertices(corners, corners, c12, c21, inters, mask_inter)
        idx = intersection.sort_indices(vertices, mask)
        area, _ = intersection.calculate_area(idx, vertices)

        torch.testing.assert_close(c12.int(), torch.tensor([[[1, 1, 1, 0]]], dtype=torch.int32))
        torch.testing.assert_close(c21.int(), torch.tensor([[[1, 1, 1, 0]]], dtype=torch.int32))
        torch.testing.assert_close(idx, torch.tensor(EXPECTED_HIGH_COORD_SAME_BOX_SORT, dtype=torch.long))
        torch.testing.assert_close(area, torch.tensor([[26.0]], dtype=torch.float32))

        outside = corners.clone()
        outside[:, :, 0, 0] += 1e-4
        self.assertFalse(intersection.box1_in_box2(outside, corners)[0, 0, 0].item())

    def test_r3l_collision_loss_can_optimize_overlap_down(self):
        from solvers.r3l.losses import collision_loss, pack_boxes

        x = torch.tensor([0.0, 0.8, 3.0], dtype=torch.float32, requires_grad=True)
        y = torch.tensor([0.0, 0.2, 0.0], dtype=torch.float32, requires_grad=True)
        rz = torch.tensor([0.0, 0.3, -0.2], dtype=torch.float32, requires_grad=True)
        bbox_x = torch.tensor([2.0, 2.2, 1.0], dtype=torch.float32)
        bbox_y = torch.tensor([2.0, 1.5, 1.0], dtype=torch.float32)

        idx = torch.triu_indices(3, 3, 1)  # all-pairs over the 3 boxes

        def all_pairs_collision():
            boxes = pack_boxes(x, y, bbox_x, bbox_y, rz)
            return collision_loss(boxes[idx[0]], boxes[idx[1]], method="iou", weight=1.0)

        # Drive the optimizer on the scaled loss, but measure overlap with the
        # UNSCALED nominal (2nd return) so the assertion is scale-independent.
        _, initial_nominal = all_pairs_collision()
        initial = initial_nominal.item()

        optimizer = torch.optim.Adam([x, y, rz], lr=0.05)
        for _ in range(80):
            optimizer.zero_grad()
            loss, _ = all_pairs_collision()
            loss.backward()
            optimizer.step()

        _, final_nominal = all_pairs_collision()

        self.assertGreater(initial, 0.1)
        self.assertLess(final_nominal.item(), initial * 0.25)


if __name__ == "__main__":
    unittest.main()
