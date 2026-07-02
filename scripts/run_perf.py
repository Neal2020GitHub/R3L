"""Benchmark the R3L optimization stage end-to-end (the Axis-2 wall-clock harness).

Sibling of ``run_tests.py`` (verify correctness) and ``run_cases.py`` (run
demo+render cases); ``run_perf.py`` runs *timing* benchmarks. It measures the
real ``Optimizer.optimize`` -- the same call ``Pipeline.optimize`` runs in
production -- over the committed scene fixtures, and reports milliseconds per
optimization step so any future kernel/loop optimization can be tracked against
a reproducible baseline.

Why time the REAL optimizer (recorder/dashboard included) instead of a stripped
inner loop: the per-step telemetry sync in the recorder is itself one of the
optimization targets, so a faithful baseline must include it -- when that cost
is later removed, this harness shows the improvement instead of hiding it.

What it loads: each ``tests/fixtures/<name>.json`` minted by
``capture_snapshot.py`` is a self-contained scene (constr_json + per-object bbox
+ room_size + a deterministic global pose). The benchmark rebuilds the exact
``CompiledConstraints`` the way ``test_compile_equivalence`` does -- no
``data/assets`` and no ``output/`` access -- then feeds the global pose straight
into ``Optimizer.optimize`` (which localizes internally).

Device note: the loss kernels read the module-global ``cfg.runtime.device``, so
selecting cuda/cpu means rewriting the config BEFORE ``solvers`` is imported
(the same temp-YAML mechanism ``tests/__init__`` uses to force CPU). This script
therefore must NOT ``import tests`` -- it needs the production cuda config.

Correct CUDA timing is load-bearing on this launch-bound workload: warm up to
absorb cudaMalloc/autotune/first-touch, ``torch.cuda.synchronize()`` on both
ends of the timed region (async launches would otherwise report ~0), and report
the MEDIAN over repeats (robust to one-off allocator/GC stalls). Metrics are
reported per fixture and never aggregated across N -- the whole story is how
cost scales with object count N and cluster count K.

Usage:
    python scripts/run_perf.py                         # all fixtures, both stages, cuda
    python scripts/run_perf.py --fixture bedroom_0 --stage base
    python scripts/run_perf.py --iters 150 --repeat 2  # quick smoke
    python scripts/run_perf.py --json out/perf.json
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE_DIR = ROOT / "tests" / "fixtures"
DEFAULT_CONFIG = ROOT / "solvers" / "r3l" / "config.yaml"

# stage selector -> (cfg.solver attribute, whether the stage trains Var params).
# Mirrors Pipeline.optimize: base freezes Var params, finetune trains
# them (train_var=True), warm-started from base.
STAGES = {
    "base": ("base", False),
    "finetune": ("finetune", True),
}


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch.cuda.is_available() is False")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("--device mps requested but torch.backends.mps.is_available() is False")
    return requested


def _bootstrap_config(device: str) -> None:
    """Point R3L_CONFIG at a temp copy of the config with the chosen device.

    Must run BEFORE any ``solvers.r3l`` import: the kernels read the frozen
    module-global ``cfg.runtime.device`` at allocation time, so device choice is
    a config-load concern, not a per-call argument. Renders/verbose are forced
    off (benchmark wants no Blender/console side effects); frame_every and
    loss_every keep their config values so the per-step recorder/frame cost the
    benchmark measures matches a real run.
    """
    base = os.environ.get("R3L_CONFIG") or str(DEFAULT_CONFIG)
    with open(base) as f:
        data = yaml.safe_load(f) or {}
    data["runtime"]["device"] = device
    data["runtime"]["render"] = False
    data["runtime"]["verbose"] = False
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="r3l_perf_config_", delete=False
    )
    yaml.safe_dump(data, tmp)
    tmp.close()
    os.environ["R3L_CONFIG"] = tmp.name


def _discover_fixtures() -> list[str]:
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in FIXTURE_DIR.glob("*.json")
    )


def _silent(fn, *args, **kwargs):
    """Run a noisy optimizer call with its dashboard/recorder prints muted.

    The dashboard render and constraint-param report are kept (they are part of
    the real per-step cost) -- only their console output is swallowed so the
    benchmark report stays clean.
    """
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _timed_optimize(optimizer, compiled, pose, stage, train_var, device) -> float:
    """One full optimize() call, wall-clocked with device sync on both ends."""
    _sync(device)
    t0 = time.perf_counter()
    _silent(
        optimizer.optimize,
        constraints=compiled, poses_vec=pose, stage=stage,
        train_var=train_var, tag=None,  # tag=None -> skip end-of-loop CSV/PNG IO
    )
    _sync(device)
    return time.perf_counter() - t0


def run(
    fixtures: list[str],
    stages: list[str],
    device: str,
    warmup: int,
    repeat: int,
    iters_override: int | None,
    seed: int | None,
) -> list[dict]:
    # Imported here, AFTER _bootstrap_config has set R3L_CONFIG.
    from solvers.r3l.config import cfg
    from solvers.r3l.compile import compile as compile_spec
    from solvers.r3l.optimize import Optimizer
    from utils.r3l.types import BBoxVec, PoseVec

    if seed is not None:
        torch.manual_seed(seed)

    scratch = tempfile.mkdtemp(prefix="r3l_perf_artifacts_")
    results: list[dict] = []

    for name in fixtures:
        with open(FIXTURE_DIR / f"{name}.json") as f:
            fx = json.load(f)

        bbox = fx["bbox"]
        bbox_vec = BBoxVec(
            x=torch.tensor(bbox["x"], dtype=torch.float32, device=device),
            y=torch.tensor(bbox["y"], dtype=torch.float32, device=device),
            z=torch.tensor(bbox["z"], dtype=torch.float32, device=device),
        )
        object_to_index = {oid: i for i, oid in enumerate(fx["objects"])}
        compiled = compile_spec(
            fx["constr_json"], object_to_index, bbox_vec, tuple(fx["room_size"]), scratch
        )
        pose = fx["pose"]
        global_pose = PoseVec(
            x=torch.tensor(pose["x"], dtype=torch.float32, device=device),
            y=torch.tensor(pose["y"], dtype=torch.float32, device=device),
            rz=torch.tensor(pose["rz"], dtype=torch.float32, device=device),
        )

        N = compiled.scene.N
        K = len(compiled.scene.clusters)
        optimizer = Optimizer(device=device)

        for stage_name in stages:
            attr, train_var = STAGES[stage_name]
            stage = getattr(cfg.solver, attr)
            if iters_override is not None:
                stage = stage.model_copy(update={"iterations": iters_override})
            iters = stage.iterations

            # Warmup (discarded): a short call absorbs allocator/autotune/first-touch.
            warm_stage = stage.model_copy(update={"iterations": min(warmup, iters)})
            _silent(
                optimizer.optimize,
                constraints=compiled, poses_vec=global_pose, stage=warm_stage,
                train_var=train_var, tag=None,
            )

            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            times = [
                _timed_optimize(optimizer, compiled, global_pose, stage, train_var, device)
                for _ in range(repeat)
            ]
            median_s = statistics.median(times)
            peak_mb = (
                torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else None
            )

            row = {
                "fixture": name, "N": N, "K": K, "device": device,
                "stage": stage_name, "iters": iters, "repeat": repeat,
                "ms_per_step": median_s / iters * 1e3,
                "steps_per_s": iters / median_s,
                "total_ms": median_s * 1e3,
                "peak_mem_mb": peak_mb,
            }
            results.append(row)
            _print_row(row)

    return results


def _print_row(r: dict) -> None:
    mem = f"{r['peak_mem_mb']:8.1f}" if r["peak_mem_mb"] is not None else "     n/a"
    print(
        f"  {r['fixture']:24s} N={r['N']:>3} K={r['K']:>2} | {r['stage']:8s} "
        f"iters={r['iters']:>4} | {r['ms_per_step']:7.3f} ms/step | "
        f"{r['steps_per_s']:8.1f} step/s | total {r['total_ms']:9.1f} ms | "
        f"peak {mem} MB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the R3L optimization stage (ms/step).")
    p.add_argument("--fixture", action="append", default=None,
                   help="fixture name (repeatable); default = all under tests/fixtures/")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    p.add_argument("--stage", default="both", choices=["base", "finetune", "both"])
    p.add_argument("--warmup", type=int, default=25, help="warmup iterations to discard")
    p.add_argument("--repeat", type=int, default=3, help="timed passes; median is reported")
    p.add_argument("--iters", type=int, default=None,
                   help="override stage iterations (quick smoke runs); default = config")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--json", default=None, help="write machine-readable results to this path")
    args = p.parse_args()

    device = _resolve_device(args.device)
    _bootstrap_config(device)  # MUST precede the solver import inside run()

    fixtures = args.fixture or _discover_fixtures()
    if not fixtures:
        raise SystemExit(f"no fixtures found under {FIXTURE_DIR}")
    missing = [n for n in fixtures if not (FIXTURE_DIR / f"{n}.json").exists()]
    if missing:
        raise SystemExit(f"fixture(s) not found: {missing}")
    stages = ["base", "finetune"] if args.stage == "both" else [args.stage]

    dev_label = device
    if device == "cuda":
        dev_label = f"cuda ({torch.cuda.get_device_name(0)})"
    elif device == "mps":
        dev_label = "mps (Apple Silicon)"
    print(
        f"R3L optimize benchmark | device={dev_label} | "
        f"warmup={args.warmup} repeat={args.repeat} "
        f"iters={args.iters or 'config'} | fixtures={len(fixtures)}"
    )

    results = run(
        fixtures=fixtures, stages=stages, device=device,
        warmup=args.warmup, repeat=args.repeat,
        iters_override=args.iters, seed=args.seed,
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {len(results)} rows -> {args.json}")


if __name__ == "__main__":
    main()
