"""
Tests for the two-stage (base / finetune) optimization.

Verifies:
1. Optimizer.optimize() runs end-to-end from a per-stage BaseStage.
2. train_var=False excludes the Var `param` role from the optimizer.
3. The Var param tensor is readable regardless of train_var.

(The old TestOptimizationConfig shape-assertions are now guaranteed by the
Pydantic load gate and were dropped. Device comes from the tests/ CPU
bootstrap, so there is no per-test cfg mutation.)
"""

import unittest
import torch
from unittest.mock import MagicMock
import tempfile

from solvers.r3l.config import cfg
from solvers.r3l.optimize import Optimizer
from utils.r3l.types import PoseVec, ParamVec, ParamTable


class TestOptimizerStage(unittest.TestCase):
    """Optimizer.optimize() runs end-to-end from a per-stage BaseStage (device from the tests/ CPU bootstrap)."""

    def _make_mock_constraints(self, has_params: bool = False):
        """Create a mock CompiledConstraints object."""
        mock = MagicMock()

        # `params` is the compile-time ParamTable schema: a single object exposing
        # names/kinds/priors (parallel arrays), `__bool__` (any params?), kind_of.
        params = ParamTable(
            names=["test_param"] if has_params else [],
            priors=[0.5] if has_params else [],
            kinds=["unit"] if has_params else [],
        )
        mock.params = params
        mock.save_dir = tempfile.gettempdir()

        # Pass-through localize/globalize on the scene (reparam keyword is threaded through).
        mock.scene.localize.side_effect = lambda p, *, reparam: p
        mock.scene.globalize.side_effect = lambda p, *, reparam: p

        # evaluate takes a scalar `alpha`, an optional `reparam` flag, and a
        # `train_var` flag (matching CompiledConstraints.evaluate).
        def mock_evaluate(poses, alpha, params=None, reparam=True, train_var=False):
            loss = torch.tensor(0.0, requires_grad=True)
            nominal = {"test": torch.tensor(0.0)}
            return loss, nominal

        mock.evaluate.side_effect = mock_evaluate
        return mock

    def _stage(self):
        # Read the real base stage from config; shrink iterations for speed.
        return cfg.solver.base.model_copy(update={"iterations": 5})

    def test_optimize_runs_from_stage(self):
        """Optimizer.optimize() should run end-to-end given a stage (BaseStage)."""
        optimizer = Optimizer(device="cpu")

        poses = PoseVec(
            x=torch.tensor([1.0]),
            y=torch.tensor([1.0]),
            rz=torch.tensor([0.0]),
        )
        constraints = self._make_mock_constraints(has_params=False)

        # Should not raise.
        result, history = optimizer.optimize(
            constraints=constraints,
            poses_vec=poses,
            stage=self._stage(),
            train_var=True,
        )

        self.assertIsInstance(result, PoseVec)
        self.assertGreater(len(history), 0)


class TestTrainablesGroups(unittest.TestCase):
    """Test the train_var gate on Trainables.groups() and param readability."""

    def test_groups_excludes_param_when_train_var_false(self):
        """groups() should include the param role only when train_var=True."""
        from solvers.r3l.optimize import Trainables, _as_param

        prior = torch.tensor([0.5, 0.3])
        st = Trainables(
            x=_as_param(torch.zeros(1), "cpu"),
            y=_as_param(torch.zeros(1), "cpu"),
            rz=_as_param(torch.zeros(1), "cpu"),
            param=_as_param(prior, "cpu"),
        )

        # train_var=False -> param not handed to the optimizer -> it stays at prior.
        self.assertNotIn("param", st.groups(train_var=False))
        self.assertIn("param", st.groups(train_var=True))
        assert st.param is not None  # narrow Optional for the type checker
        self.assertTrue(torch.allclose(st.param, prior))

    def test_param_vec_available_regardless_of_train_var(self):
        """The Var param tensor exists and is readable independent of train_var."""
        from solvers.r3l.optimize import Trainables, _as_param

        prior = torch.tensor([0.5])
        st = Trainables(
            x=_as_param(torch.zeros(1), "cpu"),
            y=_as_param(torch.zeros(1), "cpu"),
            rz=_as_param(torch.zeros(1), "cpu"),
            param=_as_param(prior, "cpu"),
        )

        params_vec = st.param_vec()
        self.assertIsInstance(params_vec, ParamVec)
        assert params_vec is not None  # narrow Optional for the type checker
        self.assertTrue(torch.allclose(params_vec.values, prior))


if __name__ == "__main__":
    unittest.main()
