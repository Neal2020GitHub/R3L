"""Tests for renderer.cycles_device + cli.preflight Cycles row.

Two layers of cycles_device:
- ``select_gpu_backend`` — the pure backend-selection policy (no bpy).
- ``detect_gpu_backend`` / ``enable_gpu_backend`` — the bpy adapters, pinned via
  a FAKE bpy injected into ``sys.modules`` (save/restore isolates the clobber).
  The fix under test: detection enumerates real devices via
  ``prefs.refresh_devices()`` -> ``prefs.devices``, NOT the dynamic
  ``compute_device_type`` RNA enum; ``detect`` does not SELECT a backend (no
  ``compute_device_type`` write); ``enable`` does.

Plus: cli.preflight._check_cycles must use ``detect`` (never ``enable``), and
  _gate_cycles must soft-warn on CPU / hard-fail on missing.

bpy-free at import: importing this module does NOT import bpy (the adapters
  import it lazily inside ``_refresh_cycles_prefs``).
"""
import ast
import glob
import os
import sys
import unittest

import renderer.cycles_device as cycles_device
from renderer.cycles_device import select_gpu_backend


class TestSelectGpuBackend(unittest.TestCase):
    def test_metal_only(self):
        self.assertEqual(select_gpu_backend(["CPU", "METAL"]), "METAL")

    def test_priority_optix_over_metal(self):
        self.assertEqual(select_gpu_backend(["CPU", "METAL", "OPTIX"]), "OPTIX")

    def test_priority_chain(self):
        self.assertEqual(select_gpu_backend(["CPU", "ONEAPI", "METAL"]), "METAL")
        self.assertEqual(select_gpu_backend(["CPU", "HIP", "METAL"]), "HIP")
        self.assertEqual(select_gpu_backend(["CPU", "CUDA", "METAL"]), "CUDA")

    def test_cpu_only_returns_none(self):
        self.assertIsNone(select_gpu_backend(["CPU"]))

    def test_empty_returns_none(self):
        self.assertIsNone(select_gpu_backend([]))

    def test_unknown_types_ignored(self):
        self.assertEqual(select_gpu_backend(["CPU", "BOGUS", "METAL"]), "METAL")

    def test_bpy_not_imported_at_module_level(self):
        # bpy must be lazy (function-local in _refresh_cycles_prefs), never a
        # module-top import, so the module stays importable in non-Blender processes.
        self.assertFalse(hasattr(cycles_device, "bpy"))


# ---------------------------------------------------------------------------
# Fake bpy for the adapter tests. Inject into sys.modules; the adapters do
# `import bpy` inside _refresh_cycles_prefs, which then binds the fake.
# ---------------------------------------------------------------------------

class _FakeDevice:
    def __init__(self, type, name=None):
        self.type = type
        self.name = name or type
        self.use = False


class _FakePrefs:
    """Stand-in for the cycles addon preferences. ``bl_rna`` is a read-spy:
    accessing it records ``bl_rna_accessed`` — the adapters must NEVER touch it
    (that's the old enum_items path). ``refresh_called`` records whether
    ``refresh_devices()`` was invoked — the fix's core mechanism."""
    def __init__(self, devices, *, refresh_raises=False):
        self._devices = devices
        self.compute_device_type = None   # untouched until enable_gpu_backend
        self.refresh_raises = refresh_raises
        self.refresh_called = False
        self.bl_rna_accessed = False

    @property
    def devices(self):
        return self._devices

    def refresh_devices(self):
        self.refresh_called = True
        if self.refresh_raises:
            raise RuntimeError("refresh failed")

    @property
    def bl_rna(self):
        self.bl_rna_accessed = True
        raise AssertionError("adapters must not touch prefs.bl_rna (the old enum_items path)")


class _FakeBpy:
    def __init__(self, prefs):
        ctx = type("Ctx", (), {})()
        prefs_obj = type("P", (), {"addons": {"cycles": type("A", (), {"preferences": prefs})()}})()
        ctx.preferences = prefs_obj
        self.context = ctx


class TestCyclesDeviceAdapter(unittest.TestCase):
    """Pin detect_gpu_backend (no backend selection) and enable_gpu_backend
    (selects + enables) against a fake bpy — no real Blender needed."""

    def setUp(self):
        self._saved_bpy = sys.modules.get("bpy")

    def tearDown(self):
        if self._saved_bpy is None:
            sys.modules.pop("bpy", None)
        else:
            sys.modules["bpy"] = self._saved_bpy

    def _install(self, devices, *, refresh_raises=False):
        prefs = _FakePrefs(devices, refresh_raises=refresh_raises)
        sys.modules["bpy"] = _FakeBpy(prefs)
        return prefs

    # --- enable_gpu_backend (selects + mutates) ---

    def test_enable_metal_sets_prefs_and_use(self):
        cpu, metal = _FakeDevice("CPU"), _FakeDevice("METAL")
        prefs = self._install([cpu, metal])
        from renderer.cycles_device import enable_gpu_backend
        self.assertEqual(enable_gpu_backend(), "GPU")
        self.assertEqual(prefs.compute_device_type, "METAL")
        self.assertTrue(metal.use)
        self.assertFalse(cpu.use)
        self.assertTrue(prefs.refresh_called)  # the fix: enumerate via refresh_devices

    def test_enable_priority_optix_over_metal(self):
        cpu, metal, optix = _FakeDevice("CPU"), _FakeDevice("METAL"), _FakeDevice("OPTIX")
        prefs = self._install([cpu, metal, optix])
        from renderer.cycles_device import enable_gpu_backend
        self.assertEqual(enable_gpu_backend(), "GPU")
        self.assertEqual(prefs.compute_device_type, "OPTIX")
        self.assertTrue(optix.use)
        self.assertFalse(metal.use)

    def test_enable_cpu_only_returns_cpu_untouched(self):
        prefs = self._install([_FakeDevice("CPU")])
        from renderer.cycles_device import enable_gpu_backend
        self.assertEqual(enable_gpu_backend(), "CPU")
        self.assertIsNone(prefs.compute_device_type)  # not mutated

    def test_enable_refresh_failure_falls_back_to_cpu(self):
        prefs = self._install([], refresh_raises=True)
        from renderer.cycles_device import enable_gpu_backend
        self.assertEqual(enable_gpu_backend(), "CPU")  # broad except -> CPU, no crash

    # --- detect_gpu_backend (no backend selection) ---

    def test_detect_metal_without_selecting_backend(self):
        cpu, metal = _FakeDevice("CPU"), _FakeDevice("METAL")
        prefs = self._install([cpu, metal])
        from renderer.cycles_device import detect_gpu_backend
        self.assertEqual(detect_gpu_backend(), "METAL")
        self.assertIsNone(prefs.compute_device_type)   # does NOT select a backend
        self.assertTrue(prefs.refresh_called)          # the fix: refresh_devices invoked

    def test_detect_cpu_only_returns_none(self):
        prefs = self._install([_FakeDevice("CPU")])
        from renderer.cycles_device import detect_gpu_backend
        self.assertIsNone(detect_gpu_backend())

    def test_never_accesses_bl_rna(self):
        # The fix: enumerate prefs.devices, NOT bl_rna.enum_items. Pin both paths.
        cpu, metal = _FakeDevice("CPU"), _FakeDevice("METAL")
        prefs = self._install([cpu, metal])
        from renderer.cycles_device import detect_gpu_backend, enable_gpu_backend
        self.assertEqual(detect_gpu_backend(), "METAL")
        self.assertEqual(enable_gpu_backend(), "GPU")
        self.assertFalse(prefs.bl_rna_accessed)


class TestPreflightCycles(unittest.TestCase):
    """Pin the safety property the refactor exists to enforce at the preflight
    call site: _check_cycles uses the non-selecting detect_gpu_backend, NEVER
    the mutating enable_gpu_backend; and _gate_cycles soft-warns on CPU while
    hard-failing on missing."""

    def setUp(self):
        self._saved_bpy = sys.modules.get("bpy")
        # Track which adapter the preflight call site invokes.
        self._called = {"detect": 0, "enable": 0}
        import renderer.cycles_device as cd
        self._orig_detect = cd.detect_gpu_backend
        self._orig_enable = cd.enable_gpu_backend
        cd.detect_gpu_backend = lambda: (self._called.__setitem__("detect", self._called["detect"] + 1), "METAL")[1]
        cd.enable_gpu_backend = lambda: (self._called.__setitem__("enable", self._called["enable"] + 1), "GPU")[1]

    def tearDown(self):
        import renderer.cycles_device as cd
        cd.detect_gpu_backend = self._orig_detect
        cd.enable_gpu_backend = self._orig_enable
        if self._saved_bpy is None:
            sys.modules.pop("bpy", None)
        else:
            sys.modules["bpy"] = self._saved_bpy

    def _bpy_installed(self):
        sys.modules["bpy"] = type("B", (), {"context": None})()  # any truthy; _check_cycles only does `import bpy`

    def test_check_cycles_uses_detect_not_enable(self):
        self._bpy_installed()
        import cli.preflight as P
        status, backend = P._check_cycles()
        self.assertEqual(status, "gpu")
        self.assertEqual(backend, "METAL")
        self.assertEqual(self._called["detect"], 1)
        self.assertEqual(self._called["enable"], 0)  # must NOT mutate prefs via enable

    def test_check_cycles_missing_when_bpy_absent(self):
        sys.modules.pop("bpy", None)
        # Make `import bpy` raise ImportError by injecting a failing module probe.
        # Simplest: ensure bpy is not importable — remove from sys.modules and
        # block creation via a meta_path finder that raises for 'bpy'.
        import importlib.abc, importlib.machinery
        class _BlockBpy(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "bpy":
                    raise ImportError("blocked for test")
                return None
        finder = _BlockBpy()
        sys.meta_path.insert(0, finder)
        try:
            import cli.preflight as P
            self.assertEqual(P._check_cycles(), ("missing", None))
        finally:
            sys.meta_path.remove(finder)

    def test_gate_cycles_cpu_warns_does_not_exit(self):
        import cli.preflight as P
        P._gate_cycles(("cpu", None))  # must not raise SystemExit

    def test_gate_cycles_missing_fails(self):
        import cli.preflight as P
        with self.assertRaises(SystemExit):
            P._gate_cycles(("missing", None))

    def test_gate_cycles_none_is_noop(self):
        import cli.preflight as P
        P._gate_cycles(None)  # must not raise


class TestNoBpyInTests(unittest.TestCase):
    """Guard against silent regression: venv bpy IS importable, so a top-level
    `import bpy` in a test module would not fail the suite — this AST scan catches it."""

    def test_no_top_level_bpy_import(self):
        offenders = []
        here = os.path.dirname(os.path.abspath(__file__))
        for path in glob.glob(os.path.join(here, "test_*.py")):
            with open(path) as f:
                tree = ast.parse(f.read(), filename=path)
            for node in tree.body:  # top-level only; imports inside functions are fine
                if isinstance(node, ast.Import) and any(a.name == "bpy" for a in node.names):
                    offenders.append(path)
                elif isinstance(node, ast.ImportFrom) and node.module == "bpy":
                    offenders.append(path)
        self.assertEqual(offenders, [], f"top-level bpy import in tests: {offenders}")


if __name__ == "__main__":
    unittest.main()