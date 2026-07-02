"""Cycles compute-backend selection, shared across render entry points.

``bpy`` is imported lazily inside ``_refresh_cycles_prefs`` so the pure policy
(`select_gpu_backend`) and the probe (`detect_gpu_backend`) stay importable in
non-Blender processes — e.g. the unit-test suite, which must not import bpy.

- ``select_gpu_backend``  — pure policy, no bpy; returns a backend name or None.
- ``detect_gpu_backend``  — probe; returns a backend name or None; 
- ``enable_gpu_backend``  — select + enable the best backend;
"""
from typing import Iterable, Optional

# Preference order: highest-throughput backend first.
_GPU_BACKENDS = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI")


def select_gpu_backend(device_types: Iterable[str]) -> Optional[str]:
    """Pure: given the device-type strings present on a host, return the best
    GPU backend by priority, or ``None`` when only CPU is present.

    No bpy — the unit-testable core of the detection policy.
    """
    present = set(device_types) - {"CPU"}
    return next((b for b in _GPU_BACKENDS if b in present), None)


def _refresh_cycles_prefs():
    """Refresh and return the Cycles addon preferences (the live ``prefs.devices``
    list of real devices compiled into this build and present on this host).

    Mutates ``prefs.devices``: ``refresh_devices()`` populates it, and Cycles'
    ``update_device_entries`` sets ``use=True`` on every NEWLY-discovered
    non-CPU device (cycles addon ``properties.py``). So this helper is NOT
    side-effect-free — callers that must not choose a backend still call it
    (they just don't write ``compute_device_type``). Uses ``refresh_devices()``
    — the non-deprecated replacement for ``get_devices()``. ``bpy`` is imported
    lazily here so this module stays importable in a non-Blender process.
    """
    import bpy
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.refresh_devices()  # populates prefs.devices with every real device
    return prefs


def detect_gpu_backend() -> Optional[str]:
    """Probe: the best Cycles GPU backend present on this host, or ``None`` when
    only CPU is present. Does NOT write ``prefs.compute_device_type`` — i.e. it
    does NOT select a backend, so it is safe for preflight (which must not
    choose a backend as a side effect of a check).

    NOT fully read-only: ``_refresh_cycles_prefs`` populates ``prefs.devices``
    and Cycles sets ``use=True`` on newly-discovered GPU devices during that
    refresh. It is "read-only w.r.t. backend SELECTION", not w.r.t. the device
    list — there is no way to enumerate real devices on a cold start without
    refreshing. Enumerates real devices, NOT the ``compute_device_type`` RNA
    enum (that enum is dynamic; ``bl_rna.properties[...].enum_items`` is empty
    on every build and cannot be introspected).
    """
    prefs = _refresh_cycles_prefs()
    return select_gpu_backend(d.type for d in prefs.devices)


def enable_gpu_backend() -> str:
    """Select + enable the best Cycles GPU backend present on this host (set
    ``prefs.compute_device_type`` and the matching devices' ``use`` flag), and
    return the render-device mode ``"GPU"``; fall back to ``"CPU"`` when no GPU
    device is present or selection fails. The verb ``enable`` signals the
    mutation — callers that only need to READ the backend should use
    ``detect_gpu_backend``.

    On selection failure a notice is printed and ``"CPU"`` returned — never let
    device selection crash a render. Note: at the call sites that run inside
    ``suppress_blender_output`` the print is fd-suppressed; preflight surfaces
    the CPU-fallback case BEFORE the run, so the operator is not left blind.
    """
    try:
        prefs = _refresh_cycles_prefs()
        backend = select_gpu_backend(d.type for d in prefs.devices)
        if backend is None:
            return "CPU"
        prefs.compute_device_type = backend
        for d in prefs.devices:
            if d.type == backend:
                d.use = True
        return "GPU"
    except Exception as e:  # never let device selection crash a render; CPU is safe
        print(f"[cycles_device] GPU selection failed ({e!r}); falling back to CPU")
        return "CPU"