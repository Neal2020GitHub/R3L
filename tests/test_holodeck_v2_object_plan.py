import unittest

from utils.holodeck_v2.types import object_plan_from_dict


class TestHolodeckV2ObjectPlan(unittest.TestCase):
    def test_prompt_schema_uses_outer_key_as_object_name(self):
        plan = object_plan_from_dict(
            {
                "modern sectional sofa": {
                    "description": "modern sectional light grey sofa",
                    "location": "floor",
                    "size": [200, 100, 80],
                    "quantity": 1,
                    "variance_type": "same",
                    "importance": 10,
                    "objects_on_top": [],
                }
            }
        )

        self.assertEqual(list(plan.keys()), ["modern sectional sofa"])
        self.assertEqual(
            plan["modern sectional sofa"]["object_name"],
            "modern sectional sofa",
        )

    def test_nested_object_name_is_not_required(self):
        plan = object_plan_from_dict(
            {
                "floor lamp": {
                    "description": "warm tripod floor lamp",
                    "location": "floor",
                    "size": [40, 40, 160],
                    "quantity": 2,
                    "variance_type": "same",
                    "importance": 7,
                    "objects_on_top": [],
                }
            }
        )

        self.assertIn("floor lamp", plan)
        self.assertEqual(plan["floor lamp"]["object_name"], "floor lamp")

    def test_outer_key_is_canonical_when_nested_name_disagrees(self):
        plan = object_plan_from_dict(
            {
                "sofa": {
                    "object_name": "chair",
                    "description": "modern sofa",
                    "location": "floor",
                    "size": [200, 100, 80],
                    "quantity": 1,
                    "variance_type": "same",
                    "importance": 10,
                    "objects_on_top": [],
                }
            }
        )

        self.assertEqual(plan["sofa"]["object_name"], "sofa")


if __name__ == "__main__":
    unittest.main()
