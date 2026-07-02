"""
The staged synthesis flow: build the blueprint, then download → solve → render →
save, reporting each stage with the console chrome.

Heavy pipeline imports (r3l / holodeck / adapters / utils.asset) are LAZY, inside
the flow functions, on purpose: it keeps importing this module cheap and lets
preflight run (and fail friendly) before torch/Blender ever load.
"""

import contextlib
import io
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING

from colorama import Fore, Style
from tqdm import tqdm

from solvers.r3l.config import cfg
from utils.console import INDENT, info, ok, render_box_table, section, spinner, term_width

if TYPE_CHECKING:
    from adapters.protocols import HolodeckBlueprint, R3LBlueprint, RendererSolution
    Blueprint = HolodeckBlueprint | R3LBlueprint  # either mode's blueprint


def _artifacts() -> list[str]:
    """Build the per-run artifact list from the render config, so enabled-but-
    config-skipped outputs aren't falsely reported missing in the summary."""
    out = ["scene.json"]
    if cfg.render.image.enabled:
        out += ["image_top.png", "image_side.png", "scene.blend"]
    if cfg.render.animation.view_2d.enabled:
        out += [f"2d_{s}.gif" for s in ("base", "finetune")]
    if cfg.render.animation.view_3d.enabled:
        fmt = cfg.render.animation.view_3d.format
        out += [f"3d_{s}.{fmt}" for s in cfg.render.animation.view_3d.stages]
    out += [f"loss_curve_{s}.csv" for s in ("base", "finetune")]
    out += [f"loss_curve_{s}.png" for s in ("base", "finetune")]
    return out


# =============================================================================
# Shared driver: download → staged solve → renderer layout
# =============================================================================

def _get_sub_save_dir(base_dir: str, session_id: str) -> str:
    save_dir = os.path.join(base_dir, session_id, "r3l")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def _download_assets(asset_ids) -> None:
    """Fetch + process any not-yet-cached asset into ASSET_BASE_DIR behind one clean
    tqdm bar. When everything is already cached there is nothing to fetch, so the bar
    is skipped entirely (the caller still reports the count). bpy serializes
    processing, so this can't be parallelized; per-asset status prints are captured so
    they don't smear the bar (bpy's own stdout is fd-suppressed in utils.asset)."""
    from utils.asset import download_and_process_asset, is_asset_cached
    from utils.holodeck_v2.constants import ASSET_BASE_DIR

    os.makedirs(ASSET_BASE_DIR, exist_ok=True)
    pending = [a for a in asset_ids if not is_asset_cached(a, ASSET_BASE_DIR)]
    if not pending:
        return  # all assets already on disk — no bar, nothing to download
    ncols = min(term_width() - len(INDENT), 72)
    for aid in tqdm(pending, desc=f"{INDENT}downloading assets",
                    dynamic_ncols=False, ncols=ncols, unit="asset", leave=True):
        with contextlib.redirect_stdout(io.StringIO()):
            download_and_process_asset(asset_id=aid, asset_base_dir=ASSET_BASE_DIR)


def _drive(blueprint: "Blueprint", save_dir: str, asset_dir: str, total: int) -> "RendererSolution":
    """Download → the three solver stages (Generate, Optimize, Render) → renderer-ready
    layout. `total` is the mode's stage count; Download precedes the three solver stages."""
    from r3l import create_r3l_layout, Stage
    from adapters.layouts import r3l_to_renderer_layout

    download_idx = total - 3  # the 3 solver stages (Generate/Optimize/Render) follow Download

    section("Downloading assets", download_idx, total)
    # HolodeckBlueprint.get_asset_ids may repeat ids; dedup so each asset downloads once.
    asset_ids = list(dict.fromkeys(blueprint.get_asset_ids()))
    _download_assets(asset_ids)
    ok(f"{len(asset_ids)} assets ready")

    stage_title = {
        Stage.GENERATE: ("Generating spatial constraints", download_idx + 1),
        Stage.OPTIMIZE: ("Optimizing layout", download_idx + 2),
        Stage.RENDER: ("Rendering", download_idx + 3),
    }

    def on_stage(stage):
        title, idx = stage_title[stage]
        section(title, idx, total)
        if stage is Stage.RENDER and cfg.render.image.enabled:
            info("rendering static stills via Blender")

    solution = create_r3l_layout(blueprint, save_dir, asset_dir, on_stage)
    if cfg.render.image.enabled:
        ok("renders written")
    return r3l_to_renderer_layout(solution)


def _save_scene(blueprint: "Blueprint", solution: "RendererSolution", save_dir: str) -> None:
    """Persist the mode-specific scene.json (each blueprint owns its own schema)."""
    with open(os.path.join(save_dir, "scene.json"), "w") as f:
        json.dump(blueprint.to_scene_dict(solution), f, indent=4)


def _final_summary(out_dir: str) -> None:
    out_dir = os.path.normpath(out_dir)  # drop the leading "./" so the terminal links the path
    section("Done")
    print(f"{INDENT}{Fore.GREEN}{Style.BRIGHT}✓ scene ready{Style.RESET_ALL}  "
          f"{Style.DIM}→{Style.RESET_ALL} {out_dir}")
    print()
    rows = []
    for name in _artifacts():
        ready = os.path.isfile(os.path.join(out_dir, name))
        status = f"{Fore.GREEN}ready{Fore.RESET}" if ready else f"{Style.DIM}—{Style.RESET_ALL}"
        rows.append([name, status])
    for line in render_box_table(["artifact", "status"], rows).splitlines():
        print(INDENT + line)


# =============================================================================
# Mode flows
# =============================================================================

def _session_id(query: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    slug = query.replace(" ", "_").replace("'", "")[:30]
    return f"{slug}-{timestamp}"


def _floor_plan_line(room, walls) -> str:
    verts = room["full_vertices"]
    length = max(v[0] for v in verts)
    width = max(v[1] for v in verts)
    return (f"floor plan: {room['roomType']} · {length:.1f} × {width:.1f} m"
            f" · wall height {walls['wall_height']:.1f} m")


def _holodeck_summary(blueprint: "HolodeckBlueprint") -> None:
    if blueprint.design:
        info(f"design: {' '.join(blueprint.design.split())}")
    floor = blueprint.objects["floor"]
    print(f"{INDENT}{Style.DIM}objects{Style.RESET_ALL}")
    for i, (object_id, _aid) in enumerate(floor, 1):
        name = object_id.rsplit("-", 1)[0]
        desc = blueprint.plan.get(name, {}).get("description", "")
        line = f"{INDENT}{INDENT}{Style.DIM}{i}{Style.RESET_ALL} {name}"
        if desc:
            line += f"  {Style.DIM}{desc}{Style.RESET_ALL}"
        print(line)
    ok(f"selected {len(floor)} objects")


def _quiet_step(label: str, work):
    """Run a blocking planning sub-step. In normal mode, animate a spinner and capture
    the vendored planner's chatter (its pure-dialogue prints belong to the planner, not
    the UI). In verbose mode, show that raw dialogue for debugging — no spinner, so the
    two never fight over the line. Mirrors the verbose gate in optimize/generate."""
    if cfg.runtime.verbose:
        return work()
    with spinner(label), contextlib.redirect_stdout(io.StringIO()):
        return work()


def run_text(args) -> None:
    """Text mode (5 stages): Holodeck plans + selects, then the shared solver drive,
    presented as a spinner + clean recap per phase (raw dialogue only in verbose)."""
    from holodeck import init_holodeck, create_floor_plan, create_object_selection

    total = 5  # Planning, Download, Generate, Optimize, Render
    info(f'room description: "{args.query}"')

    section("Planning with Holodeck", 1, total)
    _quiet_step("loading Holodeck (CLIP/SBERT + LLM)",
                lambda: init_holodeck(os.environ["OPENAI_API_KEY"]))
    room, walls = _quiet_step("generating floor plan (gpt-5)",
                              lambda: create_floor_plan(args.query))
    info(_floor_plan_line(room, walls))
    blueprint = _quiet_step(
        "selecting & retrieving furniture (gpt-5 + CLIP/SBERT)",
        lambda: create_object_selection(
            args.query, room, walls, include_floor=True, include_wall=False, include_small=False))
    _holodeck_summary(blueprint)

    save_dir = _get_sub_save_dir(args.save_dir, _session_id(args.query))
    renderer_layout = _drive(blueprint, save_dir, args.asset_dir, total)
    _save_scene(blueprint, renderer_layout, save_dir)
    _final_summary(save_dir)


def run_scene(args) -> None:
    """Scene mode (4 stages): build the R3L blueprint from a scene JSON, then drive."""
    from adapters.blueprints import to_r3l_blueprint

    total = 4  # Download, Generate, Optimize, Render
    with open(args.scene_json) as f:
        scene_config = json.load(f)
    blueprint = to_r3l_blueprint(scene_config)

    session_id = os.path.basename(args.scene_json).split(".")[0]
    save_dir = _get_sub_save_dir(args.save_dir, session_id)
    renderer_layout = _drive(blueprint, save_dir, args.asset_dir, total)
    _save_scene(blueprint, renderer_layout, save_dir)
    _final_summary(save_dir)
