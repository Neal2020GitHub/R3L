"""
Lightweight tests for v2 DSL alignment parameter (Var-only mode).

Verifies:
1. alignment Var labels are correctly mapped to percentile values
2. deprecated percentile= keyword is rejected with helpful error
3. numeric 4th positional arg (old percentile) is rejected
4. string literal alignment (without Var) is rejected
"""

import unittest
from solvers.r3l.dsl.parse_constraints import (
    parse_program, 
    code_to_json, 
    DslParseError
)


# Minimal var_to_obj_id for testing
VAR_MAP = {
    "bed_0": "bed-0",
    "nightstand_0": "nightstand-0",
    "chair_0": "chair-0",
    "table_0": "table-0",
}

def _parse(code: str):
    return parse_program(code, VAR_MAP, hv_absolute=True)


class TestAlignmentParsing(unittest.TestCase):
    """Test that alignment Var labels parse correctly to percentile values."""

    def _get_percentile(self, code: str, constraint_type: str) -> float:
        """Helper: parse code and extract percentile from first constraint of given type."""
        program = _parse(code)
        json_out = code_to_json(program)
        constraints = json_out["constraints"]["scene_relational"][constraint_type]
        self.assertEqual(len(constraints), 1)
        return constraints[0]["percentile"]

    def test_left_of_backboard(self):
        code = '''
clr = Var(0.2)
align = Var("backboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "left_of"), 0.0)

    def test_left_of_center(self):
        code = '''
clr = Var(0.2)
align = Var("center")
solver.left_of(source=nightstand_0, target=bed_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "left_of"), 0.5)

    def test_left_of_frontboard(self):
        code = '''
clr = Var(0.2)
align = Var("frontboard")
solver.left_of(source=nightstand_0, target=bed_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "left_of"), 1.0)

    def test_right_of_backboard(self):
        code = '''
clr = Var(0.2)
align = Var("backboard")
solver.right_of(source=nightstand_0, target=bed_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "right_of"), 0.0)

    def test_in_front_of_left(self):
        code = '''
clr = Var(0.5)
align = Var("left")
solver.in_front_of(source=chair_0, target=table_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "in_front_of"), 0.0)

    def test_in_front_of_center(self):
        code = '''
clr = Var(0.5)
align = Var("center")
solver.in_front_of(source=chair_0, target=table_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "in_front_of"), 0.5)

    def test_in_front_of_right(self):
        code = '''
clr = Var(0.5)
align = Var("right")
solver.in_front_of(source=chair_0, target=table_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "in_front_of"), 1.0)

    def test_behind_of_left(self):
        code = '''
clr = Var(0.5)
align = Var("left")
solver.behind_of(source=chair_0, target=table_0, clearance=clr, alignment=align)
'''
        self.assertAlmostEqual(self._get_percentile(code, "behind_of"), 0.0)


class TestPercentileRejected(unittest.TestCase):
    """Test that deprecated percentile syntax is rejected with helpful errors."""

    def test_percentile_keyword_rejected_left_of(self):
        code = 'solver.left_of(source=nightstand_0, target=bed_0, clearance=Var(0.2), percentile=0.0)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("percentile", str(ctx.exception).lower())
        self.assertIn("alignment", str(ctx.exception).lower())
        self.assertIn("backboard", str(ctx.exception))

    def test_percentile_keyword_rejected_right_of(self):
        code = 'solver.right_of(source=nightstand_0, target=bed_0, clearance=Var(0.2), percentile=0.5)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("percentile", str(ctx.exception).lower())

    def test_percentile_keyword_rejected_in_front_of(self):
        code = 'solver.in_front_of(source=chair_0, target=table_0, clearance=Var(0.5), percentile=0.0)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("left", str(ctx.exception))  # hint should show valid labels

    def test_percentile_keyword_rejected_behind_of(self):
        code = 'solver.behind_of(source=chair_0, target=table_0, clearance=Var(0.5), percentile=1.0)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("alignment", str(ctx.exception).lower())

    def test_numeric_positional_rejected_left_of(self):
        # Old syntax: solver.left_of(src, tar, clearance, 0.0) — 4th arg is numeric
        code = 'solver.left_of(nightstand_0, bed_0, Var(0.2), 0.0)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("numeric", str(ctx.exception).lower())
        self.assertIn("alignment", str(ctx.exception).lower())

    def test_numeric_positional_rejected_in_front_of(self):
        code = 'solver.in_front_of(chair_0, table_0, Var(0.5), 0.5)'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("numeric", str(ctx.exception).lower())


class TestStringLiteralAlignmentRejected(unittest.TestCase):
    """Test that string literal alignment (without Var) is rejected."""

    def test_string_literal_alignment_rejected(self):
        """alignment must be a Var reference, not a string literal."""
        code = 'solver.left_of(source=nightstand_0, target=bed_0, clearance=Var(0.2), alignment="backboard")'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("alignment", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())

    def test_numeric_clearance_rejected(self):
        """clearance must be a Var reference, not a numeric literal."""
        code = 'solver.left_of(source=nightstand_0, target=bed_0, clearance=0.2, alignment=Var("backboard"))'
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("clearance", str(ctx.exception))
        self.assertIn("var", str(ctx.exception).lower())


class TestInvalidAlignmentLabel(unittest.TestCase):
    """Test that invalid alignment labels are rejected."""

    def test_wrong_label_for_lateral(self):
        # "left" is valid for frontal, not lateral
        code = '''
clr = Var(0.2)
align = Var("left")
solver.left_of(source=nightstand_0, target=bed_0, clearance=clr, alignment=align)
'''
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("invalid alignment", str(ctx.exception).lower())
        self.assertIn("backboard", str(ctx.exception))

    def test_wrong_label_for_frontal(self):
        # "backboard" is valid for lateral, not frontal
        code = '''
clr = Var(0.5)
align = Var("backboard")
solver.in_front_of(source=chair_0, target=table_0, clearance=clr, alignment=align)
'''
        with self.assertRaises(DslParseError) as ctx:
            _parse(code)
        self.assertIn("invalid alignment", str(ctx.exception).lower())
        self.assertIn("left", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
