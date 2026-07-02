"""
Interactive input selectors (prompt_toolkit, inline).

Gather the user's choices for an interactive run: the mode menu, the room-description
prompt, and the fuzzy scene-file picker. Pure UI over leaf deps (prompt_toolkit +
utils.console/log); no pipeline imports, so importing this stays cheap.
"""

import glob
import os
import sys

from prompt_toolkit import prompt as ptk_prompt
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from utils.console import INDENT, ok, section
from utils.log import print_error, print_warn


_MODE_OPTIONS = [
    ("text", "Start from a few words", "Holodeck plans the scene"),
    ("scene", "Start from a scene file", "Ready made from the builder"),
    ("builder", "Start from a blank canvas", "Create a scene file from scratch"),
]


def prompt_query(default: str) -> str:
    """Read the room description, styled to match the rest of the UI: its own
    heavy section, indented to the body, a cyan ❯ prompt, default pre-filled and
    editable. Falls back to the default off a TTY."""
    if not sys.stdin.isatty():
        return default
    section("Describe your room")
    style = PTStyle.from_dict({"qmark": "ansicyan bold", "": "ansiwhite"})
    message = FormattedText([("class:qmark", f"{INDENT}❯ ")])
    try:
        q = ptk_prompt(message, default=default, style=style)
    except (EOFError, KeyboardInterrupt):
        q = default
    print()
    return q.strip() or default


def select_mode():
    state = {"idx": 0}
    label_w = max(len(label) for _, label, _ in _MODE_OPTIONS)

    def get_text():
        frags = [("", "How do you want to start?\n")]
        for i, (_, label, hint) in enumerate(_MODE_OPTIONS):
            selected = i == state["idx"]
            marker = "  ❯ " if selected else "    "
            row_style = "fg:cyan bold" if selected else ""
            frags.append((row_style, f"{marker}{label.ljust(label_w)}"))
            frags.append(("fg:ansibrightblack", f"   {hint}"))
            frags.append(("", "\n"))
        return frags

    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        state["idx"] = (state["idx"] - 1) % len(_MODE_OPTIONS)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        state["idx"] = (state["idx"] + 1) % len(_MODE_OPTIONS)

    @kb.add("enter")
    def _accept(event):
        event.app.exit(result=_MODE_OPTIONS[state["idx"]][0])

    @kb.add("c-c")
    @kb.add("q")
    def _abort(event):
        event.app.exit(result=None)

    body = Window(
        content=FormattedTextControl(get_text, focusable=True),
        height=len(_MODE_OPTIONS) + 1, 
        always_hide_cursor=True
    )
    app = Application(
        layout=Layout(body), 
        key_bindings=kb, 
        full_screen=False,
        erase_when_done=True, 
        mouse_support=False
    )

    result = app.run()
    match result:
        case "text": ok("mode: start from a few words")
        case "scene": ok("mode: start from a scene file")
        case "builder": ok("mode: start from a blank canvas")
        case None: print_warn("Mode selection aborted")
    return result


def _pick_scene_file(files):
    """Fuzzy arrow-key picker over the benchmark scene files; returns the chosen
    relative path, or None if aborted."""
    prompt_label = "Pick a scene file: "
    state = {"idx": 0}
    query = Buffer(multiline=False)

    def filtered():
        q = query.text.strip().lower()
        return list(files) if not q else [f for f in files if q in f.lower()]

    def get_list_text():
        items = filtered()
        if not items:
            return [("fg:ansired", "  (no matches)")]
        if state["idx"] >= len(items):
            state["idx"] = len(items) - 1
        frags = []
        for i, path in enumerate(items):
            selected = i == state["idx"]
            marker = "❯ " if selected else "  "
            style = "fg:cyan bold" if selected else ""
            frags.append((style, f"{marker}{path}\n"))
        return frags

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        state["idx"] = max(0, state["idx"] - 1)

    @kb.add("down")
    def _down(event):
        state["idx"] = min(len(filtered()) - 1, state["idx"] + 1)

    @kb.add("enter")
    def _accept(event):
        items = filtered()
        if items:
            event.app.exit(result=items[state["idx"]])

    @kb.add("c-c")
    def _abort(event):
        event.app.exit(result=None)

    query.on_text_changed += lambda _: state.__setitem__("idx", 0)

    input_row = VSplit([
        Window(FormattedTextControl(lambda: [("bold", prompt_label)]), width=len(prompt_label)),
        Window(BufferControl(buffer=query), height=1),
    ])
    list_win = Window(FormattedTextControl(get_list_text), height=Dimension(min=1, max=len(files)))
    app = Application(layout=Layout(HSplit([input_row, list_win]), focused_element=query),
                      key_bindings=kb, full_screen=False, erase_when_done=True,
                      mouse_support=False)
    result = app.run()
    if result: ok(f"scene: {result}")
    else: print_warn("Scene selection aborted")
    return result


def pick_scene():
    """Glob the benchmark scenes and let the user pick one; exits if none exist."""
    found = sorted(glob.glob(os.path.join("benchmark", "**", "*.json"), recursive=True))
    if not found:
        print_error("no scene files found under benchmark/, consider creating using the Builder.")
        sys.exit(1)
    return _pick_scene_file(found)
