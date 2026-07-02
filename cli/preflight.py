"""
Mode-aware preflight: run the real environment checks, show a panel, and fail fast
with a friendly message (never a raw traceback) on any missing required input.

Resolves the objathor retriever paths env-driven WITHOUT importing
utils/holodeck_v2/constants.py (which has import-time side effects), so a missing
key or missing data is reported before any heavy import runs.
"""

import os
import sys

from colorama import Fore

from solvers.r3l.config import cfg
from utils import data, models
from utils.console import INDENT, render_box_table, section
from utils.log import print_error, print_info, print_warn
from utils.models import DOWNLOAD_CMD
from utils.data import DOWNLOAD_CMD as DATA_DOWNLOAD_CMD


SOLVER_MODEL = cfg.llm.heavy            # constraint-gen LLM, both modes
PLANNER_MODEL = "gpt-5"                  # mirrors holodeck_v2.constants.LLM_MODEL_NAME (text only)
BUILDER_MODEL = "gpt-5"                 # Scene Builder room-gen LLM (mirrors builder/app.py; builder mode only)


def _check_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _check_objathor_data():
    """The retriever's objathor data, resolved env-driven via utils.data (which mirrors
    utils.holodeck_v2.constants WITHOUT its import-time side effects). Returns
    (ok, base_dir, missing_files): a green check means every required file is present,
    non-empty, and readable (see utils.data for the precise presence-vs-integrity scope)."""
    base_dir = data.base_dir()
    absent = data.missing()
    missing_files = [p for d in absent for p in data.paths(d) if not os.path.isfile(p)]
    return (len(absent) == 0), base_dir, missing_files


def _check_models():
    """The retriever's CLIP + SBERT checkpoints, resolved purely from the local HF cache
    (zero network). Mirrors _check_objathor_data: returns (ok, missing_models). Uses the
    same resolver the loaders use — a green check means every required file is present,
    non-empty, and readable (see utils.models for the precise presence-vs-integrity scope)."""
    absent = models.missing()
    return (len(absent) == 0), absent


def _check_device() -> bool:
    """Is the configured device actually usable? cuda/mps each need their availability
    check, which imports torch — heavy — so callers run this only after the cheap
    checks pass. cpu is always usable."""
    dev = cfg.runtime.device
    if dev == "cpu":
        return True
    import torch
    if dev == "cuda":
        return torch.cuda.is_available()
    if dev == "mps":
        return torch.backends.mps.is_available()
    return False  # unknown device string (schema should have rejected it already)


def _device_cell(device_ok) -> str:
    """Panel cell for the device: green when the check passed, red 'unavailable' when it
    failed, plain when not checked (a prior required input was missing)."""
    dev = cfg.runtime.device
    if dev == "cpu" or device_ok is None:
        return dev
    return f"{Fore.GREEN}{dev}{Fore.RESET}" if device_ok else f"{Fore.RED}{dev} (unavailable){Fore.RESET}"


def _check_cycles():
    """Detect the Blender Cycles GPU backend without SELECTING one (uses
    ``detect_gpu_backend``, which does not write ``compute_device_type``; never
    ``enable_gpu_backend``, which would choose + enable a backend as a side
    effect of a preflight check).

    Returns ``(status, backend)``: 
    - ``('gpu', name)`` when a GPU backend is present, 
    - ``('cpu', None)`` when only CPU is present
    - ``('missing', None)`` when bpy is not importable (cannot render at all). 
    """
    try:
        import bpy  # noqa: F401 — probe bpy availability only
    except ImportError:
        return ("missing", None)
    from renderer.cycles_device import detect_gpu_backend
    backend = detect_gpu_backend()  # real errors propagate, not masked as 'missing'
    return ("gpu", backend) if backend else ("cpu", None)


def _cycles_cell(cycles_status) -> str:
    """Panel cell for the Blender Cycles row: green backend name when a GPU backend is
    present, yellow 'CPU' when only CPU (working but slower), red 'unavailable' when
    bpy/Cycles is not importable, plain when not checked."""
    if cycles_status is None:
        return "—"
    status, backend = cycles_status
    if status == "gpu":
        return f"{Fore.GREEN}{backend}{Fore.RESET}"
    if status == "cpu":
        return f"{Fore.YELLOW}CPU{Fore.RESET}"
    return f"{Fore.RED}unavailable{Fore.RESET}"


def _gate_cycles(cycles_status) -> None:
    if cycles_status is None:
        return
    status, _ = cycles_status
    if status == "cpu":
        print_warn("Blender Cycles GPU backend unavailable. Renders will use CPU (slower).")
    elif status == "missing":
        _fail("cycles")


def _panel(mode, key_ok, device_ok, *, data_ok=None, scene_ok=None, models_ok=None, cycles_status=None) -> None:
    yes = f"{Fore.GREEN}found{Fore.RESET}"
    no = f"{Fore.RED}MISSING{Fore.RESET}"
    if mode == "builder":
        # The Scene Builder runs gpt-5 (room gen) + the CLIP/SBERT retriever; it
        # never touches the solver, so there is NO solver device row (and no scene
        # file). It DOES render via Blender Cycles (now in-process venv bpy), so it
        # shows the Blender Cycles row. It still needs the key, models, and objathor data.
        rows = [["builder model", BUILDER_MODEL],
                ["retriever", "CLIP + SBERT"],
                ["OPENAI_API_KEY", yes if key_ok else no],
                ["retriever models", yes if models_ok else no],
                ["objathor data", yes if data_ok else no],
                ["Blender Cycles", _cycles_cell(cycles_status)]]
        for line in render_box_table(["preflight", ""], rows).splitlines():
            print(INDENT + line)
        return
    rows = [["solver model", SOLVER_MODEL]]
    if mode == "text":
        rows.append(["planner model", PLANNER_MODEL])
    rows += [["device", _device_cell(device_ok)],
             ["OPENAI_API_KEY", yes if key_ok else no]]
    if mode == "text":
        rows.append(["retriever models", yes if models_ok else no])
        rows.append(["objathor data", yes if data_ok else no])
    else:
        rows.append(["scene file", yes if scene_ok else no])
    rows.append(["Blender Cycles", _cycles_cell(cycles_status)])
    for line in render_box_table(["preflight", ""], rows).splitlines():
        print(INDENT + line)


def _explain_missing_data(consumer, base_dir, missing) -> None:
    """Shared body for the two objathor-data failures (text vs builder): both need the
    same retriever files, only the named consumer differs.

    The remediation branches on two real signals, not on the (post-default) value of
    `base_dir`: whether OBJATHOR_ASSETS_BASE_DIR is set in the environment, and whether the
    resolved directory actually exists on disk. Keying off `base_dir`'s string value can't
    tell an unset var from one explicitly set to the default, so it routes users to the
    wrong fix — e.g. a set-but-wrong path (the dir doesn't exist) used to be told to
    "source env.sh" when what it needs is a download or a path correction.

      env set & dir exists   → an interrupted download; list the missing files, refetch.
      env set & dir missing  → the path is wrong or the data was never downloaded there.
      env unset              → defaulted; the user has not configured a path yet.
    """
    print_error(f"Objathor retriever data not found; {consumer} cannot initialize.")
    env_set = os.environ.get("OBJATHOR_ASSETS_BASE_DIR") is not None
    dir_exists = os.path.isdir(base_dir)

    print_info(f"Fix: {DATA_DOWNLOAD_CMD}")

    if dir_exists:
        # A real directory with missing files — an interrupted download. The specific
        # files are the actionable detail here.
        for path in missing:
            print_error(f"missing required file: {path}")
        print_info("The previous download was interrupted or incomplete; re-running it refetches the gaps.")
    else:
        # The directory itself is absent — listing files under a nonexistent prefix is noise.
        print_error(f"OBJATHOR_ASSETS_BASE_DIR resolves to '{base_dir}', which does not exist on disk.")
        print_info("Either the path is wrong, or the data was never downloaded there.")

    if not env_set:
        # env.example.sh ships a placeholder path; sourcing it unedited exports a bogus
        # value. Tell the user to edit it to a real path first (mirrors the key branch).
        print_info("OBJATHOR_ASSETS_BASE_DIR is unset, so it defaulted to '" + base_dir + "'.")
        print_info("To pin it permanently: cp scripts/env.example.sh scripts/env.sh,")
        print_info("edit OBJATHOR_ASSETS_BASE_DIR in scripts/env.sh to a real path, then: source scripts/env.sh")


def _explain(kind, *, base_dir=None, missing=None, scene_path=None, missing_models=None) -> None:
    match kind:
        case "key":
            print_error("OPENAI_API_KEY is not set in your environment.")
            print_info("Every mode calls the OpenAI API, so this key is required.")
            print_info("Fix: cp scripts/env.example.sh scripts/env.sh")
            print_info("then edit it and run: source scripts/env.sh (set OPENAI_API_KEY=sk-...).")
        case "data":
            _explain_missing_data("Holodeck Retriever", base_dir, missing)
        case "data_builder":
            _explain_missing_data("the Scene Builder's CLIP/SBERT retriever", base_dir, missing)
        case "models":
            print_error("Retriever models (CLIP + SBERT) are not in the local cache.")
            for m in (missing_models or models.MODELS):
                print_error(f"missing model: {m.repo}")
            print_info("These are a one-time download. Fetch them once with:")
            print_info(f"run: {DOWNLOAD_CMD}")
        case "scene":
            print_error(f"scene file does not exist: {scene_path}")
            print_info("Select a valid scene file. Or create one from scratch via the Scene Builder.")
        case "device":
            dev = cfg.runtime.device
            print_error(f"runtime.device is '{dev}' but {dev.upper()} is not available on this machine.")
            print_info(f"Fix: set runtime.device to 'cpu' in solvers/r3l/config.yaml (slower, but always works),")
            print_info(f"or to a GPU backend this machine supports ('cuda' / 'mps').")
        case "cycles":
            print_error("Blender Cycles is not available: bpy could not be imported or the cycles addon is missing.")
            print_info("Every render path needs the pinned bpy wheel; CPU fallback is unavailable if bpy itself is absent.")
            print_info("Fix: uv sync (or pip install -e .) to install the pinned bpy==4.0.* wheel.")
        case _:
            raise ValueError(f"unknown preflight failure kind: {kind!r}")


def _fail(kind, **ctx) -> None:
    print()
    _explain(kind, **ctx)
    sys.exit(1)


def preflight(mode, args) -> None:
    """Run the selected mode's checks, show the panel, and exit(1) with a friendly screen on a
    missing input. Cheap checks (key, retriever models, data/scene) gate first; the heavier
    device check (which imports torch for cuda) runs only once they pass, so a misconfigured
    run still fails fast without loading torch. text and builder also gate on the retriever's
    CLIP/SBERT models (resolved from the local cache); builder skips the device gate — it
    never runs the solver."""
    key_ok = _check_openai_key()
    section("Preflight")
    if mode == "text":
        data_ok, base_dir, missing = _check_objathor_data()
        models_ok, missing_models = _check_models()
        device_ok = _check_device() if (key_ok and data_ok and models_ok) else None
        cycles_status = _check_cycles() if (key_ok and data_ok and models_ok and device_ok is not False) else None
        _panel(mode, key_ok, device_ok, data_ok=data_ok, models_ok=models_ok, cycles_status=cycles_status)
        if not key_ok: _fail("key")
        if not models_ok: _fail("models", missing_models=missing_models)
        if not data_ok: _fail("data", base_dir=base_dir, missing=missing)
    elif mode == "builder":
        # The Scene Builder's CLIP/SBERT retriever needs the same objathor data, but it
        # never runs the solver — so no device/cuda gate, and no scene file. It DOES
        # render via Blender Cycles (in-process venv bpy), so it gets the Cycles check.
        data_ok, base_dir, missing  = _check_objathor_data()
        models_ok, missing_models   = _check_models()
        cycles_status               = _check_cycles() if (key_ok and data_ok and models_ok) else None
        _panel("builder", key_ok, None, data_ok=data_ok, models_ok=models_ok, cycles_status=cycles_status)
        if not key_ok: _fail("key")
        if not models_ok: _fail("models", missing_models=missing_models)
        if not data_ok: _fail("data_builder", base_dir=base_dir, missing=missing)
        _gate_cycles(cycles_status)
        return
    else:
        scene_ok = os.path.isfile(args.scene_json)
        device_ok = _check_device() if (key_ok and scene_ok) else None
        cycles_status = _check_cycles() if (key_ok and scene_ok) else None
        _panel(mode, key_ok, device_ok, scene_ok=scene_ok, cycles_status=cycles_status)
        if not key_ok: _fail("key")
        if not scene_ok: _fail("scene", scene_path=args.scene_json)
    if device_ok is False:
        _fail("device")
    _gate_cycles(cycles_status)
