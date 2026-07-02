"""
R³L single-entry CLI — text (Holodeck) or scene-file layout synthesis.

  TEXT  (5 stages): Planning(Holodeck) → Download → Generate constraints
                    → Optimize → Render
  SCENE (4 stages): Download → Generate constraints → Optimize → Render
  BUILDER         : launch the Scene Builder web app to author a scene file (an
                    authoring tool — no solver/render; its output feeds SCENE mode)

This file is the thin coordinator: parse args, resolve the mode, gather the mode's
input, preflight, dispatch. The work lives in cli/ (selectors, preflight, flow, builder);
the shared visual toolkit in utils.console.

Run it with (the `-q` keeps uv's cold-start install but silences the reinstall
churn from Blender's mislabeled bpy wheel):

  uv run -q python main.py                                   # interactive
  uv run -q python main.py --query "a cozy living room"      # text mode
  uv run -q python main.py --scene_json benchmark/bedroom/bedroom_0.json
  uv run -q python main.py --mode builder                    # Scene Builder web app
"""

import argparse
import atexit
import os
import sys

from cli.builder import run_builder
from cli.flow import run_scene, run_text
from cli.preflight import preflight
from cli.selectors import pick_scene, prompt_query, select_mode
from utils.console import print_banner
from utils.log import print_warn


def _silence_exit_noise() -> None:
    """Blender prints "Not freed memory blocks" to fd 1 at interpreter shutdown: its
    core heap is freed only by WM_exit, which CPython skips when a non-main (torch/CUDA)
    thread is still alive. The detector printf's straight to stdout, so we flush our own
    streams then redirect fd 1 to /dev/null. Registered first in __main__ so — atexit
    being LIFO — it runs last, just before C-level teardown."""
    def _drop_stdout() -> None:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    atexit.register(_drop_stdout)


def _resolve_mode(args) -> str | None:
    """The explicit --mode wins; otherwise infer from the given input; otherwise ask."""
    if args.mode is not None: return args.mode
    if args.query is not None: return "text"
    if args.scene_json is not None: return "scene"
    return select_mode()


def main() -> None:
    ap = argparse.ArgumentParser(description="R³L single-entry CLI.")
    ap.add_argument("--mode", choices=["text", "scene", "builder"], help="skip the menu; run this mode")
    ap.add_argument("--query", default=None, help="text-mode room description (skips the prompt)")
    ap.add_argument("--scene_json", default=None, help="scene-mode scene JSON path (skips the picker)")
    ap.add_argument("--save_dir", default="./output/", help="output root directory")
    ap.add_argument("--asset_dir", default="./data/assets/", help="downloaded-asset directory")
    args = ap.parse_args()

    print_banner()

    mode = _resolve_mode(args)
    if mode is None:
        sys.exit(0)

    if mode == "text":
        if args.query is None:
            args.query = prompt_query("a cozy living room")
    elif mode == "scene":
        if args.scene_json is None:
            args.scene_json = pick_scene()
        if args.scene_json is None:
            sys.exit(0)
    # builder needs no input — it just launches a server.

    preflight(mode, args)

    match mode:
        case "text": run_text(args)
        case "scene": run_scene(args)
        case "builder": run_builder(args)
        case _: raise RuntimeError(f"unreachable: mode={mode}")


if __name__ == "__main__":
    _silence_exit_noise()
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_warn("interrupted")
        sys.exit(130)
