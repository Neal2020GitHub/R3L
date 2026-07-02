"""
Tests for Var-based DSL (optimizable constraint parameters).

Verifies:
1. Var(...) declarations are parsed correctly
2. All numeric fields require Var references (Var-only mode)
3. Alignment labels are mapped to percentile priors correctly
4. Kind inference works based on usage context
5. JSON output includes constraint_params section
6. Error messages are helpful when Var requirements are violated
7. prior loss is added only when training Var params (train_var=True)
"""

import unittest
import torch
from tests.lib import EMPTY_RELATIONAL, compile_scene
from solvers.r3l.dsl.parse_constraints import (
    parse_program,
    code_to_json,
    DslParseError,
    VarDecl,
    VarRef,
    ParamInfo,
)


# Minimal var_to_obj_id for testing
VAR_MAP = {
    "bed_0": "bed-0",
    "nightstand_0": "nightstand-0",
    "nightstand_1": "nightstand-1",
    "chair_0": "chair-0",
    "table_0": "table-0",
    "sofa_0": "sofa-0",
}

def _parse(code: str, *, hv_absolute: bool = True):
    return parse_program(code, VAR_MAP, hv_absolute=hv_absolute)


class TestVarDeclarationParsing(unittest.TestCase):
    """Test that Var(...) declarations are parsed correctly."""

    def test_numeric_var_declaration(self):
        code = """
ns_clear = Var(0.2)
ns_align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        self.assertEqual(len(program.var_decls), 2)
        
        # Check the named declaration
        decl = next((d for d in program.var_decls if d.name == "ns_clear"), None)
        assert decl is not None, "ns_clear declaration not found"
        self.assertEqual(decl.prior, 0.2)

    def test_string_label_var_declaration(self):
        code = """
ns_align = Var("backboard")
ns_clear = Var(0.2)
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        decl = next((d for d in program.var_decls if d.name == "ns_align"), None)
        assert decl is not None, "ns_align declaration not found"
        self.assertEqual(decl.prior, "backboard")
    
    def test_inline_var_declaration(self):
        """Test that inline Var(...) calls are auto-registered."""
        code = """
solver.left_of(source=nightstand_0, target=bed_0, clearance=Var(0.2), alignment=Var("backboard"))
"""
        program = _parse(code)
        # Both inline Vars should be registered with auto-generated names
        self.assertEqual(len(program.var_decls), 2)
        # Check param_infos are generated correctly
        self.assertEqual(len(program.param_infos), 2)

    def test_negative_var_declaration(self):
        code = """
angle_var = Var(-15.0)
solver.align(source=nightstand_0, target=bed_0, angle=angle_var)
"""
        program = _parse(code)
        decl = next((d for d in program.var_decls if d.name == "angle_var"), None)
        assert decl is not None, "angle_var declaration not found"
        self.assertEqual(decl.prior, -15.0)

    def test_duplicate_var_rejected(self):
        code = """
my_var = Var(0.2)
my_var = Var(0.3)
solver.left_of(source=nightstand_0, target=bed_0, clearance=my_var, alignment=Var("center"))
"""
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("duplicate", str(ctx.exception).lower())


class TestVarRequiredEnforcement(unittest.TestCase):
    """Test that Var references are required for all optimizable fields."""

    def test_clearance_requires_var(self):
        code = 'solver.left_of(source=nightstand_0, target=bed_0, clearance=0.2, alignment=Var("backboard"))'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("clearance", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_alignment_requires_var(self):
        code = 'solver.left_of(source=nightstand_0, target=bed_0, clearance=Var(0.2), alignment="backboard")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("alignment", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_x_requires_var(self):
        code = 'solver.horizontal(source=bed_0, x=0.5)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("x", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_percentile_requires_var_rel(self):
        code = 'solver.horizontal(source=bed_0, percentile=0.5)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code, hv_absolute=False)
        self.assertIn("percentile", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_distance_requires_var(self):
        code = 'solver.around(source_list=[chair_0, nightstand_0], target=table_0, distance=1.0, sweep_deg=Var(90.0), centerline="T")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("distance", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_sweep_deg_requires_var(self):
        code = 'solver.around(source_list=[chair_0, nightstand_0], target=table_0, distance=Var(1.0), sweep_deg=90.0, centerline="T")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("sweep_deg", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_angle_requires_var(self):
        code = 'solver.align(source=nightstand_0, target=bed_0, angle=15.0)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("angle", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())


class TestAlignmentLabelMapping(unittest.TestCase):
    """Test that alignment labels are correctly mapped to percentile priors."""

    def test_backboard_to_zero(self):
        code = """
ns_align = Var("backboard")
ns_clear = Var(0.2)
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        # Find param info for ns_align
        param = next((p for p in program.param_infos if p.name == "ns_align"), None)
        assert param is not None, "ns_align param not found"
        self.assertAlmostEqual(param.prior, 0.0)
        self.assertEqual(param.kind, "unit")

    def test_center_to_half(self):
        code = """
ns_align = Var("center")
ns_clear = Var(0.2)
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "ns_align"), None)
        assert param is not None, "ns_align param not found"
        self.assertAlmostEqual(param.prior, 0.5)

    def test_frontboard_to_one(self):
        code = """
ns_align = Var("frontboard")
ns_clear = Var(0.2)
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "ns_align"), None)
        assert param is not None, "ns_align param not found"
        self.assertAlmostEqual(param.prior, 1.0)

    def test_left_right_center_frontal(self):
        code = """
align_left = Var("left")
align_center = Var("center")
align_right = Var("right")
clear = Var(0.5)
solver.in_front_of(source=chair_0, target=table_0, clearance=clear, alignment=align_left)
"""
        program = _parse(code)
        # align_left should be used and mapped to 0.0
        param = next((p for p in program.param_infos if p.name == "align_left"), None)
        assert param is not None, "align_left param not found"
        self.assertAlmostEqual(param.prior, 0.0)


class TestKindInference(unittest.TestCase):
    """Test that param kinds are correctly inferred from usage."""

    def test_clearance_is_nonneg(self):
        code = """
ns_clear = Var(0.2)
ns_align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "ns_clear"), None)
        assert param is not None, "ns_clear param not found"
        self.assertEqual(param.kind, "nonneg")

    def test_alignment_is_unit(self):
        code = """
ns_clear = Var(0.2)
ns_align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "ns_align"), None)
        assert param is not None, "ns_align param not found"
        self.assertEqual(param.kind, "unit")

    def test_x_is_nonneg(self):
        code = """
x_pos = Var(2.0)
solver.horizontal(source=bed_0, x=x_pos)
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "x_pos"), None)
        assert param is not None, "x_pos param not found"
        self.assertEqual(param.kind, "nonneg")

    def test_percentile_is_unit_rel(self):
        code = """
h_pos = Var(0.5)
solver.horizontal(source=bed_0, percentile=h_pos)
"""
        program = _parse(code, hv_absolute=False)
        param = next((p for p in program.param_infos if p.name == "h_pos"), None)
        assert param is not None, "h_pos param not found"
        self.assertEqual(param.kind, "unit")

    def test_angle_is_angle_deg(self):
        code = """
ang = Var(15.0)
solver.align(source=nightstand_0, target=bed_0, angle=ang)
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "ang"), None)
        assert param is not None, "ang param not found"
        self.assertEqual(param.kind, "angle_deg")

    def test_distance_is_nonneg(self):
        code = """
dist = Var(1.0)
sweep = Var(90.0)
solver.around(source_list=[chair_0, nightstand_0], target=table_0, distance=dist, sweep_deg=sweep, centerline="T")
"""
        program = _parse(code)
        param = next((p for p in program.param_infos if p.name == "dist"), None)
        assert param is not None, "dist param not found"
        self.assertEqual(param.kind, "nonneg")


class TestJsonOutput(unittest.TestCase):
    """Test that JSON output includes constraint_params section."""

    def test_constraint_params_in_json(self):
        code = """
ns_clear = Var(0.2)
ns_align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        json_out = code_to_json(program)
        
        self.assertIn("constraint_params", json_out)
        params = json_out["constraint_params"]
        self.assertIn("names", params)
        self.assertIn("priors", params)
        self.assertIn("kinds", params)
        self.assertEqual(len(params["names"]), 2)

    def test_param_field_in_constraint(self):
        code = """
ns_clear = Var(0.2)
ns_align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        json_out = code_to_json(program)
        
        constraint = json_out["constraints"]["scene_relational"]["left_of"][0]
        self.assertIn("clearance_param", constraint)
        self.assertIn("percentile_param", constraint)
        self.assertEqual(constraint["clearance_param"], "ns_clear")
        self.assertEqual(constraint["percentile_param"], "ns_align")


class TestSharedVariables(unittest.TestCase):
    """Test that shared variables work correctly for symmetric constraints."""

    def test_shared_clearance(self):
        code = """
ns_clear = Var(0.2)
ns_align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=ns_clear, alignment=ns_align)
solver.right_of(source=nightstand_1, target=bed_0, clearance=ns_clear, alignment=ns_align)
"""
        program = _parse(code)
        json_out = code_to_json(program)
        
        # Both constraints should reference the same param
        left = json_out["constraints"]["scene_relational"]["left_of"][0]
        right = json_out["constraints"]["scene_relational"]["right_of"][0]
        
        self.assertEqual(left["clearance_param"], "ns_clear")
        self.assertEqual(right["clearance_param"], "ns_clear")
        
        # Only 2 params should exist (not 4)
        self.assertEqual(len(json_out["constraint_params"]["names"]), 2)


class TestAlignAngleNormalization(unittest.TestCase):
    """Test that align() always becomes angle constraint (no align-group)."""

    def test_align_becomes_angle_constraint(self):
        """align() should always produce angle constraint, even with angle=0."""
        code = """
ang = Var(0.0)
solver.align(source=nightstand_0, target=bed_0, angle=ang)
"""
        program = _parse(code)
        json_out = code_to_json(program)
        
        # Should be in angle constraints, not align groups
        self.assertEqual(len(json_out["constraints"]["scene_relational"]["angle"]), 1)
        # align-groups are deprecated but retained for legacy JSON loading
        self.assertEqual(len(json_out["constraints"]["scene_relational"].get("align", [])), 0)

    def test_align_with_nonzero_angle(self):
        code = """
ang = Var(45.0)
solver.align(source=nightstand_0, target=bed_0, angle=ang)
"""
        program = _parse(code)
        json_out = code_to_json(program)
        
        angle_constraint = json_out["constraints"]["scene_relational"]["angle"][0]
        self.assertAlmostEqual(angle_constraint["angle"], 45.0)
        self.assertIn("angle_param", angle_constraint)


def _compile_scene(objects, constraints_json, bbox_vec):
    """Compile a flat (no-cluster) scene from the object list + bbox arrays."""
    return compile_scene(objects, constraints_json, bbox_vec, (10.0, 10.0))


class TestParamVecAffectsGeometryLoss(unittest.TestCase):
    """
    Test that changing ParamVec values actually changes geometry constraint loss,
    not just prior loss. This is the key verification that params_vec is wired correctly.
    """

    def test_horizontal_abs_loss_changes_with_x_param(self):
        """Verify that horizontal_abs_loss changes when x param value changes."""
        import torch
        from utils.r3l.types import BBoxVec, PoseVec, ParamVec
        from solvers.r3l.cluster import AugmentedState

        bbox_vec = BBoxVec(
            x=torch.tensor([1.0]),
            y=torch.tensor([1.0]),
            z=torch.tensor([1.0]),
        )
        constraints_json = {
            "scene_entities": {"independent_objects": ["test-0"], "clusters": []},
            "constraints": {
                "composition": {
                    "horizontal": [
                        {"src_kind": "object", "src_id": "test-0", "x": 2.0, "x_param": "x_pos"}
                    ],
                    "vertical": [], "against_wall": [], "corner": [],
                },
                "cluster_internal": {},
                "scene_relational": dict(EMPTY_RELATIONAL),
            },
            "constraint_params": {"names": ["x_pos"], "priors": [2.0], "kinds": ["nonneg"]},
        }

        compiled = _compile_scene(["test-0"], constraints_json, bbox_vec)
        self.assertIn("horizontal_abs_loss", compiled.constraints)

        # K=0 build leaves object poses untouched; the term reads aug.poses/aug.bbox.
        layout = PoseVec(
            x=torch.tensor([2.0]),
            y=torch.tensor([5.0]),
            rz=torch.tensor([0.0]),
        )
        aug = AugmentedState.build(compiled.scene, layout, bbox_vec, reparam=True)
        term = compiled.constraints["horizontal_abs_loss"]

        # x = 2.0 (match) vs x = 8.0 (far).
        loss_0, _ = term.evaluate(aug, ParamVec(values=torch.tensor([2.0])))
        loss_1, _ = term.evaluate(aug, ParamVec(values=torch.tensor([8.0])))

        self.assertFalse(
            torch.allclose(loss_0, loss_1),
            f"Horizontal abs loss should change with x: loss_0={loss_0.item()}, loss_1={loss_1.item()}"
        )
        self.assertGreater(
            loss_1.item(), loss_0.item(),
            f"Loss for far x should be higher: loss_0={loss_0.item()}, loss_1={loss_1.item()}"
        )

    def test_directional_loss_changes_with_percentile_param(self):
        """Verify that left_of_loss changes when percentile param value changes."""
        import torch
        from utils.r3l.types import BBoxVec, PoseVec, ParamVec
        from solvers.r3l.cluster import AugmentedState

        bbox_vec = BBoxVec(
            x=torch.tensor([2.0, 0.5]),
            y=torch.tensor([1.5, 0.5]),
            z=torch.tensor([0.5, 0.5]),
        )
        constraints_json = {
            "scene_entities": {"independent_objects": ["bed-0", "nightstand-0"], "clusters": []},
            "constraints": {
                "composition": {"horizontal": [], "vertical": [], "against_wall": [], "corner": []},
                "cluster_internal": {},
                "scene_relational": dict(EMPTY_RELATIONAL, left_of=[
                    {
                        "src_kind": "object", "src_id": "nightstand-0",
                        "tar_kind": "object", "tar_id": "bed-0",
                        "clearance": 0.2, "clearance_param": "ns_clear",
                        "percentile": 0.5, "percentile_param": "ns_perc",
                    }
                ]),
            },
            "constraint_params": {"names": ["ns_clear", "ns_perc"], "priors": [0.2, 0.5], "kinds": ["nonneg", "unit"]},
        }

        compiled = _compile_scene(["bed-0", "nightstand-0"], constraints_json, bbox_vec)
        self.assertIn("left_of_loss", compiled.constraints)

        # bed at center facing forward (+y); nightstand to the left and back, so
        # different percentiles yield different alignment losses.
        layout = PoseVec(
            x=torch.tensor([5.0, 3.5]),
            y=torch.tensor([5.0, 4.0]),
            rz=torch.tensor([0.0, 0.0]),
        )
        aug = AugmentedState.build(compiled.scene, layout, bbox_vec, reparam=True)
        term = compiled.constraints["left_of_loss"]

        # percentile = 0.0 (bottom-aligned) vs 1.0 (top-aligned).
        loss_0, _ = term.evaluate(aug, ParamVec(values=torch.tensor([0.2, 0.0])))
        loss_1, _ = term.evaluate(aug, ParamVec(values=torch.tensor([0.2, 1.0])))

        self.assertFalse(
            torch.allclose(loss_0, loss_1),
            f"left_of loss should change with percentile: loss_0={loss_0.item()}, loss_1={loss_1.item()}"
        )


class TestPriorLossTrainVarGate(unittest.TestCase):
    """Test that the prior loss is added only when training Var params (train_var=True)."""

    def test_prior_loss_only_when_training_var(self):
        """Prior loss is absent with train_var=False and present with train_var=True."""
        import torch
        from utils.r3l.types import BBoxVec, PoseVec, ParamVec

        bbox_vec = BBoxVec(
            x=torch.tensor([1.0]),
            y=torch.tensor([1.0]),
            z=torch.tensor([1.0]),
        )
        constraints_json = {
            "scene_entities": {"independent_objects": ["test-0"], "clusters": []},
            "constraints": {
                "composition": {
                    "horizontal": [
                        {"src_kind": "object", "src_id": "test-0", "x": 2.0, "x_param": "x_pos"}
                    ],
                    "vertical": [], "against_wall": [], "corner": [],
                },
                "cluster_internal": {},
                "scene_relational": dict(EMPTY_RELATIONAL),
            },
            "constraint_params": {"names": ["x_pos"], "priors": [2.0], "kinds": ["nonneg"]},
        }

        # The prior gate is the `train_var` arg to evaluate, not any global flag,
        # so one compiled product exercises both cases. Prior is grouped by kind.
        compiled = _compile_scene(["test-0"], constraints_json, bbox_vec)

        layout = PoseVec(
            x=torch.tensor([2.0]),
            y=torch.tensor([5.0]),
            rz=torch.tensor([0.0]),
        )
        # Param value far from prior to make the prior loss clearly non-zero.
        params = ParamVec(values=torch.tensor([8.0]))

        # train_var=False -> prior loss SKIPPED.
        _, nominal_frozen = compiled.evaluate(layout, alpha=1.0, params=params, train_var=False)
        self.assertNotIn("prior_nonneg", nominal_frozen)

        # train_var=True -> prior loss INCLUDED.
        _, nominal_active = compiled.evaluate(layout, alpha=1.0, params=params, train_var=True)
        self.assertIn("prior_nonneg", nominal_active)


class TestCornerConstraintWallParameter(unittest.TestCase):
    """
    Test corner constraint with required wall parameter.

    The corner constraint now requires a 'wall' parameter that specifies
    which wall the entity should face away from. This disambiguates
    the orientation at corners.

    Valid corner/wall combinations:
    - BL (bottom-left): wall must be "B" or "L"
    - BR (bottom-right): wall must be "B" or "R"
    - TL (top-left): wall must be "T" or "L"
    - TR (top-right): wall must be "T" or "R"
    """

    def test_corner_requires_wall_parameter(self):
        """corner() without wall parameter raises error."""
        code = 'solver.corner(source=bed_0, corner="BL")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        # Should mention missing 'wall' argument
        self.assertIn("wall", str(ctx.exception).lower())

    def test_corner_bl_with_wall_b(self):
        """BL corner with wall='B' is valid."""
        code = 'solver.corner(source=bed_0, corner="BL", wall="B")'
        program = _parse(code)
        self.assertEqual(len(program.global_calls), 1)
        call = program.global_calls[0]
        self.assertEqual(call.name, "corner")
        self.assertEqual(call.args["corner"], "BL")
        self.assertEqual(call.args["wall"], "B")

    def test_corner_bl_with_wall_l(self):
        """BL corner with wall='L' is valid."""
        code = 'solver.corner(source=bed_0, corner="BL", wall="L")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "BL")
        self.assertEqual(call.args["wall"], "L")

    def test_corner_br_with_wall_b(self):
        """BR corner with wall='B' is valid."""
        code = 'solver.corner(source=bed_0, corner="BR", wall="B")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "BR")
        self.assertEqual(call.args["wall"], "B")

    def test_corner_br_with_wall_r(self):
        """BR corner with wall='R' is valid."""
        code = 'solver.corner(source=bed_0, corner="BR", wall="R")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "BR")
        self.assertEqual(call.args["wall"], "R")

    def test_corner_tl_with_wall_t(self):
        """TL corner with wall='T' is valid."""
        code = 'solver.corner(source=bed_0, corner="TL", wall="T")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "TL")
        self.assertEqual(call.args["wall"], "T")

    def test_corner_tl_with_wall_l(self):
        """TL corner with wall='L' is valid."""
        code = 'solver.corner(source=bed_0, corner="TL", wall="L")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "TL")
        self.assertEqual(call.args["wall"], "L")

    def test_corner_tr_with_wall_t(self):
        """TR corner with wall='T' is valid."""
        code = 'solver.corner(source=bed_0, corner="TR", wall="T")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "TR")
        self.assertEqual(call.args["wall"], "T")

    def test_corner_tr_with_wall_r(self):
        """TR corner with wall='R' is valid."""
        code = 'solver.corner(source=bed_0, corner="TR", wall="R")'
        program = _parse(code)
        call = program.global_calls[0]
        self.assertEqual(call.args["corner"], "TR")
        self.assertEqual(call.args["wall"], "R")

    def test_corner_bl_with_invalid_wall_t_raises(self):
        """BL corner with wall='T' is invalid (T not part of BL)."""
        code = 'solver.corner(source=bed_0, corner="BL", wall="T")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        # Error should mention the invalid combination
        err = str(ctx.exception)
        self.assertIn("BL", err)
        self.assertIn("T", err)

    def test_corner_bl_with_invalid_wall_r_raises(self):
        """BL corner with wall='R' is invalid (R not part of BL)."""
        code = 'solver.corner(source=bed_0, corner="BL", wall="R")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        err = str(ctx.exception)
        self.assertIn("BL", err)
        self.assertIn("R", err)

    def test_corner_tr_with_invalid_wall_b_raises(self):
        """TR corner with wall='B' is invalid (B not part of TR)."""
        code = 'solver.corner(source=bed_0, corner="TR", wall="B")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        err = str(ctx.exception)
        self.assertIn("TR", err)
        self.assertIn("B", err)

    def test_corner_with_invalid_wall_id_raises(self):
        """Invalid wall id (not T/B/L/R) raises error."""
        code = 'solver.corner(source=bed_0, corner="BL", wall="X")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("X", str(ctx.exception))

    def test_corner_json_output_includes_wall(self):
        """JSON output for corner constraint includes wall field."""
        code = 'solver.corner(source=bed_0, corner="TR", wall="R")'
        program = _parse(code)
        json_out = code_to_json(program)

        corners = json_out["constraints"]["composition"]["corner"]
        self.assertEqual(len(corners), 1)

        corner = corners[0]
        self.assertEqual(corner["src_kind"], "object")
        self.assertEqual(corner["src_id"], "bed-0")
        self.assertEqual(corner["corner"], "TR")
        self.assertEqual(corner["wall"], "R")

    def test_corner_with_cluster_handle(self):
        """corner() works with cluster handles."""
        code = '''
with solver.cluster(cluster_id="sleeping", anchor=bed_0, members=[bed_0, nightstand_0]) as sleeping:
    solver.left_of(source=nightstand_0, target=bed_0, clearance=Var(0.2), alignment=Var("backboard"))

solver.corner(source=sleeping, corner="TL", wall="L")
'''
        program = _parse(code)

        # Find the corner call in global calls
        corner_call = next((c for c in program.global_calls if c.name == "corner"), None)
        assert corner_call is not None, "corner call not found"
        self.assertEqual(corner_call.args["source"].kind, "cluster")
        self.assertEqual(corner_call.args["source"].id, "sleeping")
        self.assertEqual(corner_call.args["corner"], "TL")
        self.assertEqual(corner_call.args["wall"], "L")


class TestClusterHandleValidation(unittest.TestCase):
    """Test cluster handle naming validation."""

    def test_handle_shadowing_asset_rejected(self):
        """Cluster handle that shadows asset variable is rejected."""
        # "cello_0" is both a cluster handle AND an asset variable
        code = '''
with solver.cluster(cluster_id="music", anchor=chair_0, members=[chair_0, table_0]) as table_0:
    solver.in_front_of(source=table_0, target=chair_0, clearance=Var(0.3), alignment=Var("center"))
'''
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        err = str(ctx.exception)
        self.assertIn("shadows", err.lower())
        self.assertIn("table_0", err)

    def test_distinct_handle_name_accepted(self):
        """Cluster with distinct handle name parses successfully."""
        code = '''
with solver.cluster(cluster_id="work_area", anchor=table_0, members=[table_0, chair_0]) as work_cluster:
    solver.in_front_of(source=chair_0, target=table_0, clearance=Var(0.5), alignment=Var("center"))
'''
        program = _parse(code)
        self.assertEqual(len(program.clusters), 1)
        self.assertEqual(program.clusters[0].handle_name, "work_cluster")
        self.assertEqual(program.clusters[0].cluster_id, "work_area")


if __name__ == "__main__":
    unittest.main()
