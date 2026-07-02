import tempfile
import unittest
from unittest import mock

from PIL import Image

from utils.r3l.plot import visualize_cogmap
from utils.r3l.types import AssetInfo


class TestCogmapAabb(unittest.TestCase):
    def test_visualize_cogmap_uses_geometry_center(self):
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "member-0"]
        asset_info = {
            "anchor": AssetInfo(
                name="anchor",
                desc_short="",
                desc_long="",
                bbox={"x": 2.0, "y": 1.0, "z": 1.0},
            ),
            "member": AssetInfo(
                name="member",
                desc_short="",
                desc_long="",
                bbox={"x": 1.0, "y": 1.0, "z": 1.0},
            ),
        }
        asset_to_object = {"anchor-0": "anchor-0", "member-0": "member-0"}

        constraints_json = {
            "scene_entities": {
                "clusters": [
                    {
                        "cluster_id": "c1",
                        "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                        "members": ["member-0"],
                    }
                ]
            }
        }

        cogmap_dict = {
            "room": {"length": 5.0, "width": 4.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 4.0, "ly": 4.0},
                    "pose": {"x": 0.0, "y": 0.0, "rz": 0.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        # Shift member along +y to move the cluster center away from anchor
                        "member-0": {"x": 0.0, "y": 2.0, "rz": 0.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                out_path = visualize_cogmap(
                    cogmap_dict,
                    save_dir=td,
                    assets=assets,
                    asset_info=asset_info,
                    asset_to_object=asset_to_object,
                    constraints_json=constraints_json,
                    out_name="out.png",
                )

        assert out_path is not None, "out_path should not be None"
        override = captured.get("cluster_override")
        assert override is not None, "cluster_override not captured"
        self.assertIn("c1", override)

        cx, cy, lx, ly, rz = override["c1"]
        self.assertAlmostEqual(lx, 4.0)
        self.assertAlmostEqual(ly, 4.0)
        # cluster_pose IS the geometric center, so computed center should match it
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)
        self.assertAlmostEqual(rz, 0.0, places=5)

    def test_rotated_cluster_applies_offset_before_rotation(self):
        """Offset must be applied in local frame BEFORE cluster rotation."""
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "member-0"]
        asset_info = {
            "anchor": AssetInfo(name="anchor", desc_short="", desc_long="",
                                bbox={"x": 2.0, "y": 1.0, "z": 1.0}),
            "member": AssetInfo(name="member", desc_short="", desc_long="",
                                bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
        }
        asset_to_object = {"anchor-0": "anchor-0", "member-0": "member-0"}
        constraints_json = {
            "scene_entities": {
                "clusters": [{
                    "cluster_id": "c1",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                    "members": ["member-0"],
                }]
            }
        }
        # 90-degree cluster rotation
        cogmap_dict = {
            "room": {"length": 10.0, "width": 10.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 4.0, "ly": 4.0},
                    "pose": {"x": 0.0, "y": 0.0, "rz": 90.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        "member-0": {"x": 0.0, "y": 2.0, "rz": 0.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                visualize_cogmap(cogmap_dict, save_dir=td, assets=assets,
                                 asset_info=asset_info, asset_to_object=asset_to_object,
                                 constraints_json=constraints_json, out_name="out.png")

        cx, cy, _, _, _ = captured["cluster_override"]["c1"]
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_asymmetric_x_offset(self):
        """X-axis asymmetry must produce correct offset."""
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "member-0"]
        asset_info = {
            "anchor": AssetInfo(name="anchor", desc_short="", desc_long="",
                                bbox={"x": 2.0, "y": 1.0, "z": 1.0}),
            "member": AssetInfo(name="member", desc_short="", desc_long="",
                                bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
        }
        asset_to_object = {"anchor-0": "anchor-0", "member-0": "member-0"}
        constraints_json = {
            "scene_entities": {
                "clusters": [{
                    "cluster_id": "c1",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                    "members": ["member-0"],
                }]
            }
        }
        # Member shifted along +X
        cogmap_dict = {
            "room": {"length": 10.0, "width": 10.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 5.0, "ly": 2.0},
                    "pose": {"x": 0.0, "y": 0.0, "rz": 0.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        "member-0": {"x": 3.0, "y": 0.0, "rz": 0.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                visualize_cogmap(cogmap_dict, save_dir=td, assets=assets,
                                 asset_info=asset_info, asset_to_object=asset_to_object,
                                 constraints_json=constraints_json, out_name="out.png")

        cx, cy, _, _, _ = captured["cluster_override"]["c1"]
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_rotated_member_affects_aabb_extent(self):
        """Member rotation changes AABB projection, affecting offset."""
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "member-0"]
        asset_info = {
            "anchor": AssetInfo(name="anchor", desc_short="", desc_long="",
                                bbox={"x": 2.0, "y": 1.0, "z": 1.0}),
            "member": AssetInfo(name="member", desc_short="", desc_long="",
                                bbox={"x": 2.0, "y": 1.0, "z": 1.0}),
        }
        asset_to_object = {"anchor-0": "anchor-0", "member-0": "member-0"}
        constraints_json = {
            "scene_entities": {
                "clusters": [{
                    "cluster_id": "c1",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                    "members": ["member-0"],
                }]
            }
        }
        # Member rotated 45 degrees
        cogmap_dict = {
            "room": {"length": 10.0, "width": 10.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 4.0, "ly": 4.0},
                    "pose": {"x": 0.0, "y": 0.0, "rz": 0.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        "member-0": {"x": 0.0, "y": 2.0, "rz": 45.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                visualize_cogmap(cogmap_dict, save_dir=td, assets=assets,
                                 asset_info=asset_info, asset_to_object=asset_to_object,
                                 constraints_json=constraints_json, out_name="out.png")

        cx, cy, _, _, _ = captured["cluster_override"]["c1"]
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)

    def test_three_member_cluster(self):
        """Symmetric multi-member cluster has zero offset."""
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "memberA-0", "memberB-0"]
        asset_info = {
            "anchor": AssetInfo(name="anchor", desc_short="", desc_long="",
                                bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
            "memberA": AssetInfo(name="memberA", desc_short="", desc_long="",
                                 bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
            "memberB": AssetInfo(name="memberB", desc_short="", desc_long="",
                                 bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
        }
        asset_to_object = {"anchor-0": "anchor-0", "memberA-0": "memberA-0", "memberB-0": "memberB-0"}
        constraints_json = {
            "scene_entities": {
                "clusters": [{
                    "cluster_id": "c1",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                    "members": ["memberA-0", "memberB-0"],
                }]
            }
        }
        # Symmetric: members at (-2, 0) and (2, 0)
        cogmap_dict = {
            "room": {"length": 20.0, "width": 20.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 6.0, "ly": 2.0},
                    "pose": {"x": 5.0, "y": 5.0, "rz": 0.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        "memberA-0": {"x": -2.0, "y": 0.0, "rz": 0.0},
                        "memberB-0": {"x": 2.0, "y": 0.0, "rz": 0.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                visualize_cogmap(cogmap_dict, save_dir=td, assets=assets,
                                 asset_info=asset_info, asset_to_object=asset_to_object,
                                 constraints_json=constraints_json, out_name="out.png")

        cx, cy, _, _, _ = captured["cluster_override"]["c1"]
        self.assertAlmostEqual(cx, 5.0, places=5)
        self.assertAlmostEqual(cy, 5.0, places=5)

    def test_nonzero_world_pose_with_offset(self):
        """Non-origin cluster pose with asymmetric members."""
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "member-0"]
        asset_info = {
            "anchor": AssetInfo(name="anchor", desc_short="", desc_long="",
                                bbox={"x": 2.0, "y": 1.0, "z": 1.0}),
            "member": AssetInfo(name="member", desc_short="", desc_long="",
                                bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
        }
        asset_to_object = {"anchor-0": "anchor-0", "member-0": "member-0"}
        constraints_json = {
            "scene_entities": {
                "clusters": [{
                    "cluster_id": "c1",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                    "members": ["member-0"],
                }]
            }
        }
        # Cluster at (10, 20)
        cogmap_dict = {
            "room": {"length": 50.0, "width": 50.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 4.0, "ly": 4.0},
                    "pose": {"x": 10.0, "y": 20.0, "rz": 0.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        "member-0": {"x": 0.0, "y": 2.0, "rz": 0.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                visualize_cogmap(cogmap_dict, save_dir=td, assets=assets,
                                 asset_info=asset_info, asset_to_object=asset_to_object,
                                 constraints_json=constraints_json, out_name="out.png")

        cx, cy, _, _, _ = captured["cluster_override"]["c1"]
        self.assertAlmostEqual(cx, 10.0, places=5)
        self.assertAlmostEqual(cy, 20.0, places=5)

    def test_negative_local_offset(self):
        """Member behind anchor creates negative local offset."""
        captured = {}

        def fake_visualize_frame(*args, **kwargs):
            captured["cluster_override"] = kwargs.get("cluster_override")
            return Image.new("RGB", (8, 8), color="white")

        assets = ["anchor-0", "member-0"]
        asset_info = {
            "anchor": AssetInfo(name="anchor", desc_short="", desc_long="",
                                bbox={"x": 2.0, "y": 1.0, "z": 1.0}),
            "member": AssetInfo(name="member", desc_short="", desc_long="",
                                bbox={"x": 1.0, "y": 1.0, "z": 1.0}),
        }
        asset_to_object = {"anchor-0": "anchor-0", "member-0": "member-0"}
        constraints_json = {
            "scene_entities": {
                "clusters": [{
                    "cluster_id": "c1",
                    "anchor": {"anchor_kind": "object", "anchor_object_id": "anchor-0"},
                    "members": ["member-0"],
                }]
            }
        }
        # Member at -Y (behind anchor)
        cogmap_dict = {
            "room": {"length": 10.0, "width": 10.0},
            "independent": {},
            "clusters": {
                "c1": {
                    "aabb": {"lx": 4.0, "ly": 4.0},
                    "pose": {"x": 0.0, "y": 0.0, "rz": 0.0},
                    "members": {
                        "anchor-0": {"x": 0.0, "y": 0.0, "rz": 0.0},
                        "member-0": {"x": 0.0, "y": -2.0, "rz": 0.0},
                    },
                }
            },
        }

        with tempfile.TemporaryDirectory() as td:
            with mock.patch("utils.r3l.plot.visualize_frame", side_effect=fake_visualize_frame):
                visualize_cogmap(cogmap_dict, save_dir=td, assets=assets,
                                 asset_info=asset_info, asset_to_object=asset_to_object,
                                 constraints_json=constraints_json, out_name="out.png")

        cx, cy, _, _, _ = captured["cluster_override"]["c1"]
        self.assertAlmostEqual(cx, 0.0, places=5)
        self.assertAlmostEqual(cy, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

