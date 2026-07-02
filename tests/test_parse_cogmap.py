"""
Tests for parse_cogmap module.

Verifies:
1. Safe expression evaluator works correctly
2. Pose/AABB parsing works with various expression forms
3. Scope tracking correctly identifies cluster vs global assignments
4. Coverage validation catches missing poses/AABBs
5. JSON output has correct structure
"""

import unittest

from solvers.r3l.dsl.parse_constraints import parse_program, ProgramIR
from solvers.r3l.dsl.parse_cogmap import (
    parse_cogmap,
    CogMapParseError,
    _safe_eval,
    _parse_pose_call,
    _parse_aabb_call,
)
import ast

def _parse_program(code: str, var_to_obj_id: dict) -> ProgramIR:
    return parse_program(code, var_to_obj_id, hv_absolute=True)


class TestSafeEval(unittest.TestCase):
    """Test the whitelist expression evaluator."""

    def test_numeric_literal(self):
        """Simple numeric literals should evaluate."""
        node = ast.parse("42.5", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 42.5)
    
    def test_integer_literal(self):
        """Integer literals should convert to float."""
        node = ast.parse("42", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 42.0)
    
    def test_unary_minus(self):
        """Unary minus should work."""
        node = ast.parse("-1.5", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), -1.5)
    
    def test_unary_plus(self):
        """Unary plus should work."""
        node = ast.parse("+3.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 3.0)
    
    def test_binary_add(self):
        """Addition should work."""
        node = ast.parse("1.0 + 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 3.0)
    
    def test_binary_sub(self):
        """Subtraction should work."""
        node = ast.parse("5.0 - 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 3.0)
    
    def test_binary_mul(self):
        """Multiplication should work."""
        node = ast.parse("2.0 * 3.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 6.0)
    
    def test_binary_div(self):
        """Division should work."""
        node = ast.parse("6.0 / 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 3.0)
    
    def test_length_variable(self):
        """'length' variable should resolve to room_length."""
        node = ast.parse("length", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 8.0)
    
    def test_width_variable(self):
        """'width' variable should resolve to room_width."""
        node = ast.parse("width", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 6.0)
    
    def test_room_length_attribute(self):
        """'room.length' should resolve to room_length."""
        node = ast.parse("room.length", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 8.0)
    
    def test_room_width_attribute(self):
        """'room.width' should resolve to room_width."""
        node = ast.parse("room.width", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 6.0)
    
    def test_complex_expression(self):
        """Complex expression should work."""
        node = ast.parse("length / 2.0 - 1.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 3.0)  # 8/2 - 1 = 3
    
    def test_nested_expression(self):
        """Nested expression should work."""
        node = ast.parse("(length + width) / 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 7.0)  # (8+6)/2 = 7
    
    def test_precedence_mul_over_add(self):
        """Multiplication should have higher precedence than addition."""
        node = ast.parse("2.0 + 3.0 * 4.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 14.0)  # 2 + 12 = 14
    
    def test_precedence_div_over_sub(self):
        """Division should have higher precedence than subtraction."""
        node = ast.parse("10.0 - 6.0 / 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 7.0)  # 10 - 3 = 7
    
    def test_left_associativity_sub(self):
        """Subtraction should be left-associative."""
        node = ast.parse("10.0 - 3.0 - 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 5.0)  # (10-3)-2 = 5
    
    def test_left_associativity_div(self):
        """Division should be left-associative."""
        node = ast.parse("24.0 / 4.0 / 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 3.0)  # (24/4)/2 = 3
    
    def test_deeply_nested_parens(self):
        """Deeply nested parentheses should evaluate correctly."""
        node = ast.parse("((1.0 + 2.0) * (3.0 + 4.0))", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 21.0)  # 3 * 7 = 21
    
    def test_mixed_vars_and_literals(self):
        """Mix of room variables and literals with complex nesting."""
        node = ast.parse("(length - 2.0) * (width / 3.0) + 1.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 13.0)  # (8-2)*(6/3)+1 = 6*2+1 = 13
    
    def test_unary_in_nested(self):
        """Unary operators inside nested expressions."""
        node = ast.parse("(-length + width) * -2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 4.0)  # (-8+6)*-2 = -2*-2 = 4
    
    def test_triple_nested(self):
        """Triple-level nesting with all operators."""
        node = ast.parse("((length / 2.0 + 1.0) * 2.0 - width) / 2.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 2.0)  # ((4+1)*2-6)/2 = (10-6)/2 = 2
    
    def test_room_attribute_in_complex_expr(self):
        """room.length/room.width in complex expression."""
        node = ast.parse("room.length / 2.0 - room.width / 3.0", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 2.0)  # 4 - 2 = 2
    
    def test_mixed_name_and_attribute(self):
        """Mix of 'length' name and 'room.width' attribute."""
        node = ast.parse("(length + room.width) / (room.length - width)", mode="eval").body
        self.assertEqual(_safe_eval(node, 8.0, 6.0), 7.0)  # (8+6)/(8-6) = 14/2 = 7
    
    def test_reject_unknown_variable(self):
        """Unknown variables should raise error."""
        node = ast.parse("unknown_var", mode="eval").body
        with self.assertRaises(CogMapParseError):
            _safe_eval(node, 8.0, 6.0)
    
    def test_reject_function_call(self):
        """Function calls should raise error."""
        node = ast.parse("sin(1.0)", mode="eval").body
        with self.assertRaises(CogMapParseError):
            _safe_eval(node, 8.0, 6.0)
    
    def test_reject_boolean(self):
        """Boolean literals should raise error."""
        node = ast.parse("True", mode="eval").body
        with self.assertRaises(CogMapParseError):
            _safe_eval(node, 8.0, 6.0)


class TestParsePoseCall(unittest.TestCase):
    """Test Pose(...) call parsing."""
    
    def _parse_pose(self, expr: str):
        """Helper to parse a Pose(...) expression."""
        call = ast.parse(expr, mode="eval").body
        assert isinstance(call, ast.Call), f"Expected Call node, got {type(call).__name__}"
        return _parse_pose_call(call, 8.0, 6.0)
    
    def test_simple_pose(self):
        """Simple Pose with literal values."""
        pose = self._parse_pose("Pose(x=1.0, y=2.0, rz=90.0)")
        self.assertEqual(pose.x, 1.0)
        self.assertEqual(pose.y, 2.0)
        self.assertEqual(pose.rz, 90.0)
    
    def test_pose_with_expressions(self):
        """Pose with arithmetic expressions."""
        pose = self._parse_pose("Pose(x=length/2, y=width-1, rz=-45.0)")
        self.assertEqual(pose.x, 4.0)  # 8/2
        self.assertEqual(pose.y, 5.0)  # 6-1
        self.assertEqual(pose.rz, -45.0)
    
    def test_pose_reject_positional_args(self):
        """Pose with positional args should fail."""
        with self.assertRaises(CogMapParseError):
            self._parse_pose("Pose(1.0, 2.0, 90.0)")
    
    def test_pose_reject_missing_field(self):
        """Pose missing a field should fail."""
        with self.assertRaises(CogMapParseError):
            self._parse_pose("Pose(x=1.0, y=2.0)")


class TestParseAABBCall(unittest.TestCase):
    """Test AABB(...) call parsing."""
    
    def _parse_aabb(self, expr: str):
        """Helper to parse an AABB(...) expression."""
        call = ast.parse(expr, mode="eval").body
        assert isinstance(call, ast.Call), f"Expected Call node, got {type(call).__name__}"
        return _parse_aabb_call(call, 8.0, 6.0)
    
    def test_simple_aabb(self):
        """Simple AABB with literal values."""
        aabb = self._parse_aabb("AABB(lx=2.0, ly=3.0)")
        self.assertEqual(aabb.lx, 2.0)
        self.assertEqual(aabb.ly, 3.0)
    
    def test_aabb_with_expressions(self):
        """AABB with arithmetic expressions."""
        aabb = self._parse_aabb("AABB(lx=length/4, ly=width/2)")
        self.assertEqual(aabb.lx, 2.0)  # 8/4
        self.assertEqual(aabb.ly, 3.0)  # 6/2
    
    def test_aabb_reject_missing_field(self):
        """AABB missing a field should fail."""
        with self.assertRaises(CogMapParseError):
            self._parse_aabb("AABB(lx=2.0)")


class TestParseCogmapNoCluster(unittest.TestCase):
    """Test cogmap parsing for no-cluster mode (independent objects only)."""
    
    def test_simple_cogmap(self):
        """Parse simple cogmap with independent objects only."""
        code = '''
room = Room(length=8.0, width=6.0)

walls = {}

sofa_0 = Asset(id="sofa-0", description="A sofa", size=(2.0, 1.0))
table_0 = Asset(id="table-0", description="A table", size=(1.0, 1.0))

solver = ConstraintSolver()
solver.against_wall(source=sofa_0, wall="T")

sofa_0.pose = Pose(x=4.0, y=5.5, rz=180.0)
table_0.pose = Pose(x=2.0, y=3.0, rz=0.0)
'''
        var_to_obj_id = {"sofa_0": "sofa-0", "table_0": "table-0"}
        program = _parse_program(code, var_to_obj_id)
        
        cogmap = parse_cogmap(code, program)
        
        self.assertEqual(cogmap["room"]["length"], 8.0)
        self.assertEqual(cogmap["room"]["width"], 6.0)
        self.assertIn("sofa-0", cogmap["independent"])
        self.assertIn("table-0", cogmap["independent"])
        self.assertEqual(cogmap["independent"]["sofa-0"]["x"], 4.0)
        self.assertEqual(cogmap["independent"]["sofa-0"]["rz"], 180.0)
        self.assertNotIn("clusters", cogmap)
    
    def test_missing_pose_raises_error(self):
        """Missing pose should raise coverage error."""
        code = '''
room = Room(length=8.0, width=6.0)

sofa_0 = Asset(id="sofa-0", description="A sofa", size=(2.0, 1.0))
table_0 = Asset(id="table-0", description="A table", size=(1.0, 1.0))

solver = ConstraintSolver()
solver.against_wall(source=sofa_0, wall="T")

sofa_0.pose = Pose(x=4.0, y=5.5, rz=180.0)
# Missing table_0.pose
'''
        var_to_obj_id = {"sofa_0": "sofa-0", "table_0": "table-0"}
        program = _parse_program(code, var_to_obj_id)
        
        with self.assertRaises(CogMapParseError) as ctx:
            parse_cogmap(code, program)
        
        self.assertIn("table_0", str(ctx.exception))


class TestParseCogmapWithCluster(unittest.TestCase):
    """Test cogmap parsing for cluster mode."""
    
    def test_cluster_cogmap(self):
        """Parse cogmap with cluster."""
        code = '''
room = Room(length=8.0, width=6.0)

bed_0 = Asset(id="bed-0", description="A bed", size=(2.0, 2.3))
nightstand_0 = Asset(id="nightstand-0", description="A nightstand", size=(0.6, 0.5))

gap = Var(0.2)
align = Var("backboard")
angle = Var(0.0)

solver = ConstraintSolver()

with solver.cluster(cluster_id="sleeping", anchor=bed_0, members=[bed_0, nightstand_0]) as sleeping:
    solver.left_of(source=nightstand_0, target=bed_0, clearance=gap, alignment=align)
    solver.align(source=nightstand_0, target=bed_0, angle=angle)
    
    bed_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    nightstand_0.pose = Pose(x=-1.5, y=-0.9, rz=0.0)

sleeping.aabb = AABB(lx=3.2, ly=2.3)

solver.against_wall(source=sleeping, wall="T")

sleeping.pose = Pose(x=4.0, y=4.85, rz=180.0)
'''
        var_to_obj_id = {"bed_0": "bed-0", "nightstand_0": "nightstand-0"}
        program = _parse_program(code, var_to_obj_id)
        
        cogmap = parse_cogmap(code, program)
        
        # Room
        self.assertEqual(cogmap["room"]["length"], 8.0)
        self.assertEqual(cogmap["room"]["width"], 6.0)
        
        # No independent objects (all are cluster members)
        self.assertEqual(cogmap["independent"], {})
        
        # Cluster
        self.assertIn("clusters", cogmap)
        self.assertIn("sleeping", cogmap["clusters"])
        
        cluster = cogmap["clusters"]["sleeping"]
        
        # AABB
        self.assertEqual(cluster["aabb"]["lx"], 3.2)
        self.assertEqual(cluster["aabb"]["ly"], 2.3)
        
        # Cluster pose (global)
        self.assertEqual(cluster["pose"]["x"], 4.0)
        self.assertEqual(cluster["pose"]["rz"], 180.0)
        
        # Member poses (local)
        self.assertIn("bed-0", cluster["members"])
        self.assertIn("nightstand-0", cluster["members"])
        self.assertEqual(cluster["members"]["bed-0"]["x"], 0.0)
        self.assertEqual(cluster["members"]["nightstand-0"]["x"], -1.5)
    
    def test_cluster_member_in_wrong_scope_raises(self):
        """Cluster member pose in global scope should raise error."""
        code = '''
room = Room(length=8.0, width=6.0)

bed_0 = Asset(id="bed-0", description="A bed", size=(2.0, 2.3))
nightstand_0 = Asset(id="nightstand-0", description="A nightstand", size=(0.6, 0.5))

gap = Var(0.2)
align = Var("backboard")
angle = Var(0.0)

solver = ConstraintSolver()

with solver.cluster(cluster_id="sleeping", anchor=bed_0, members=[bed_0, nightstand_0]) as sleeping:
    solver.left_of(source=nightstand_0, target=bed_0, clearance=gap, alignment=align)
    solver.align(source=nightstand_0, target=bed_0, angle=angle)
    
    bed_0.pose = Pose(x=0.0, y=0.0, rz=0.0)
    # nightstand_0.pose MISSING inside cluster

sleeping.aabb = AABB(lx=3.2, ly=2.3)
sleeping.pose = Pose(x=4.0, y=4.85, rz=180.0)

# WRONG: nightstand pose in global scope
nightstand_0.pose = Pose(x=-1.5, y=-0.9, rz=0.0)
'''
        var_to_obj_id = {"bed_0": "bed-0", "nightstand_0": "nightstand-0"}
        program = _parse_program(code, var_to_obj_id)
        
        with self.assertRaises(CogMapParseError) as ctx:
            parse_cogmap(code, program)
        
        self.assertIn("nightstand_0", str(ctx.exception))
        self.assertIn("global scope", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()

