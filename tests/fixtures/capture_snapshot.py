"""Capture a self-contained fixture for ``test_compile_equivalence``.

Run this ONCE per scene in an asset-equipped environment (where ``data/assets/
<uid>/annotations.json`` exists for every asset in the scene). It freezes the
*current* ``compile`` -> ``evaluate`` behaviour into ``tests/fixtures/<name>.json`` so
the equivalence test never has to touch the gitignored ``output/`` or ``data/``
trees again.

Usage:
    python tests/fixtures/capture_snapshot.py output/kitchen_12x9_detailed

The fixture name is derived from the output directory's basename. The run
subdir is located automatically: it is ``<dir>/r3l/`` (current pipeline) or
``<dir>/rndnet/`` (legacy pipeline) -- whichever exists. From that subdir this
reads ``llm_output.py`` (the DSL source of truth) + ``scene.json`` (assets,
room_size, and the solved layout used as the deterministic GLOBAL pose).

What it writes:
  - ``<name>.llm_output.py`` : a verbatim copy of the run's DSL.
  - ``<name>.json`` carrying, all keyed in canonical object order (index 0..N-1):
      - constr_json : parsed constraint JSON. Cross-checked against the run's
                      on-disk ``constraints.json`` -- the capture refuses to
                      write a fixture whose re-parsed DSL has drifted from it.
      - objects     : ordered object_id list defining indices 0..N-1.
      - bbox        : {x, y, z} dim arrays per object (from asset annotations).
      - room_size   : [length, width].
      - pose        : {x, y, rz} fixed deterministic GLOBAL pose (rz in radians),
                      taken from the scene's solved layout in ``scene.json``.
      - snapshot    : {loss, nominal{...}} captured from the CURRENT code via
                      compile -> localize(pose) -> evaluate(alpha=1.0,
                      params=ParamVec(priors), reparam=True, train_var=True).

The CPU/fp32 bootstrap in ``tests/__init__.py`` is applied by importing the
``tests`` package first, so the captured numbers match the test environment.
"""

import argparse
import json
import math
import os

# Force the CPU/fp32 config exactly as the test suite does, BEFORE importing cfg.
import tests  # noqa: F401  (side effect: sets R3L_CONFIG to a CPU copy)

import torch

from solvers.r3l.compile import compile
from solvers.r3l.dsl.parse_constraints import parse_program, code_to_json
from solvers.r3l.config import cfg
from solvers.r3l.assets import create_asset_info, create_asset_mapping
from utils.r3l.types import BBoxVec, PoseVec, ParamVec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET_DIR = os.path.join(REPO, "data", "assets")
FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# The pipeline writes its artifacts into exactly one of these run subdirs.
_RUN_SUBDIRS = ("r3l", "rndnet")


def _locate_run_dir(scene_dir: str) -> str:
    """Return the single ``r3l``/``rndnet`` run subdir of ``scene_dir``."""
    present = [
        os.path.join(scene_dir, sub)
        for sub in _RUN_SUBDIRS
        if os.path.isdir(os.path.join(scene_dir, sub))
    ]
    if not present:
        raise FileNotFoundError(
            f"No run subdir found under {scene_dir}; expected one of {_RUN_SUBDIRS}."
        )
    if len(present) > 1:
        raise RuntimeError(
            f"Ambiguous run subdirs under {scene_dir}: {present}; expected exactly one."
        )
    return present[0]


def capture(scene_dir: str) -> None:
    scene_dir = os.path.abspath(scene_dir)
    name = os.path.basename(scene_dir.rstrip(os.sep))
    run_dir = _locate_run_dir(scene_dir)
    device = cfg.runtime.device  # "cpu" via tests bootstrap

    dsl_out = os.path.join(FIXTURE_DIR, f"{name}.llm_output.py")
    json_out = os.path.join(FIXTURE_DIR, f"{name}.json")

    # 1. Scene inputs from the source artifacts -----------------------------
    with open(os.path.join(run_dir, "scene.json")) as f:
        scene = json.load(f)
    asset_ids = list(scene["assets"].keys())
    room_size = (float(scene["room_size"][0]), float(scene["room_size"][1]))

    # 2. Asset info + bbox + canonical object order (production helpers) -----
    asset_info = create_asset_info(asset_ids, ASSET_DIR)
    object_to_asset, asset_to_object = create_asset_mapping(asset_info, asset_ids)
    # Canonical index 0..N-1: the order `Pipeline.compile` feeds to `compile`
    # is object_to_asset.keys(), which create_asset_mapping fills in asset_ids
    # order -> identical to [asset_to_object[aid] for aid in asset_ids].
    objects = [asset_to_object[aid] for aid in asset_ids]
    bbox_vec = BBoxVec.build(asset_info, asset_ids, device)

    # 3. constr_json: re-parse the DSL (the dsl-stability source of truth) ---
    with open(os.path.join(run_dir, "llm_output.py")) as f:
        code = f.read()
    var_to_obj_id = {oid.replace("-", "_"): oid for oid in objects}
    program = parse_program(code, var_to_obj_id, hv_absolute=cfg.prompt.hv_absolute)
    constr_json = code_to_json(program)

    # Cross-check against the on-disk constraints.json (defensive, not relied
    # upon by the test): the test pins constr_json itself.
    with open(os.path.join(run_dir, "constraints.json")) as f:
        disk_json = json.load(f)
    assert constr_json == disk_json, (
        "Re-parsed DSL diverged from on-disk constraints.json; refusing to "
        "capture a fixture that would not reflect the committed DSL."
    )

    # 4. Fixed deterministic GLOBAL pose from the solved layout --------------
    layout = scene["layout"]
    assert list(layout.keys()) == objects, (
        "scene.json layout order does not match canonical object order; "
        f"layout={list(layout.keys())[:3]}... objects={objects[:3]}..."
    )
    px = [float(layout[o]["position"][0]) for o in objects]
    py = [float(layout[o]["position"][1]) for o in objects]
    prz = [math.radians(float(layout[o]["rotation"][2])) for o in objects]
    pose = PoseVec(
        x=torch.tensor(px, dtype=torch.float32, device=device),
        y=torch.tensor(py, dtype=torch.float32, device=device),
        rz=torch.tensor(prz, dtype=torch.float32, device=device),
    )

    # 5. Capture the snapshot from CURRENT code ------------------------------
    object_to_index = {oid: i for i, oid in enumerate(object_to_asset.keys())}
    compiled = compile(constr_json, object_to_index, bbox_vec, room_size, run_dir)

    params = ParamVec.from_priors(list(compiled.params.priors), device)
    localized = compiled.scene.localize(pose, reparam=True)
    with torch.no_grad():
        loss, nominal = compiled.evaluate(
            localized, alpha=1.0, params=params, reparam=True, train_var=True
        )

    snapshot = {
        "loss": float(loss.item()),
        "nominal": {k: float(v.item()) for k, v in nominal.items()},
    }

    # 6. Write the self-contained fixture ------------------------------------
    with open(dsl_out, "w") as f:
        f.write(code)

    fixture = {
        "constr_json": constr_json,
        "objects": objects,
        "bbox": {
            "x": bbox_vec.x.tolist(),
            "y": bbox_vec.y.tolist(),
            "z": bbox_vec.z.tolist(),
        },
        "room_size": [room_size[0], room_size[1]],
        "pose": {"x": px, "y": py, "rz": prz},
        "snapshot": snapshot,
    }
    with open(json_out, "w") as f:
        json.dump(fixture, f, indent=2)

    print(f"[capture] scene = {name} (run dir: {os.path.relpath(run_dir, REPO)})")
    print(f"[capture] wrote {dsl_out}")
    print(f"[capture] wrote {json_out}")
    print(f"[capture] N objects = {len(objects)}, params = {len(params)}")
    print(f"[capture] loss = {snapshot['loss']:.8f}")
    print(f"[capture] nominal keys = {sorted(snapshot['nominal'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scene_dir",
        help="Scene output directory (e.g. output/kitchen_12x9_detailed). "
        "Its basename becomes the fixture name; its r3l/ or rndnet/ subdir "
        "supplies llm_output.py + scene.json.",
    )
    args = parser.parse_args()
    capture(args.scene_dir)


if __name__ == "__main__":
    main()
