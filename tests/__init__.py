"""Test package bootstrap: force the R3L config onto CPU before it loads.

The host is a CUDA box and the kernels read the module-global ``cfg.device``;
left on ``"cuda"`` they allocate CUDA tensors and collide with the CPU tensors
the tests build, raising ``RuntimeError``. There are no test-level CUDA skip
markers, so this bootstrap is the ONLY mechanism keeping the suite on CPU.

This runs when the ``tests`` package is first imported -- i.e. BEFORE any
``solvers.r3l.config`` import inside a test module -- so it must point
``R3L_CONFIG`` at a CPU copy of the default YAML here, unconditionally.

The override is NESTED (``runtime.device``): the config schema is
``extra="forbid"``, so a top-level ``device`` key would be rejected and crash
every test at load.
"""

import os
import tempfile

import yaml

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "solvers", "r3l", "config.yaml"
)

with open(_DEFAULT_CONFIG) as _f:
    _data = yaml.safe_load(_f) or {}

_data["runtime"]["device"] = "cpu"

_cpu_config = tempfile.NamedTemporaryFile(
    mode="w", suffix=".yaml", prefix="r3l_test_config_", delete=False
)
yaml.safe_dump(_data, _cpu_config)
_cpu_config.close()

os.environ["R3L_CONFIG"] = _cpu_config.name
