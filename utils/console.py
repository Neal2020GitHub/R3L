"""
The R³L terminal-UI toolkit (a leaf: colorama + stdlib only).

One home for everything the curated CLI draws: text measurement (ANSI-aware
width, padding), table/bar rendering, the structural chrome (banner, the D1
section divider, rules, ok/info lines), a threaded spinner for blocking calls,
and the live reasoning window. Terse bracket-prefixed status lines for library
internals live separately in `utils.log`; file-descriptor control in `utils.fd`.

Visual system, deliberately restrained:
  dim       = structure (rules, table chrome, secondary body text)
  cyan      = hierarchy anchors (banner, the section chip, stage numbers)
  green/red = SEMANTICS only (success/failure, loss down/up)
"""

import itertools
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager

from colorama import Back, Fore, Style


# =============================================================================
# Text measurement + rendering primitives (ANSI-aware)
# =============================================================================

_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(s: str) -> str:
    """Remove ANSI escape codes from a string."""
    return _ANSI_ESCAPE.sub('', s)


def vis_len(s: str) -> int:
    """Visible length of a string (ignoring ANSI codes)."""
    return len(strip_ansi(s))


def pad_cell(s: str, width: int, align: str = 'l') -> str:
    """Pad to width by visible length. align: 'l'=left, 'r'=right."""
    gap = width - vis_len(s)
    if gap <= 0:
        return s
    return ' ' * gap + s if align == 'r' else s + ' ' * gap


def vol_bar(vol: float, width: int = 12, color: bool = True) -> str:
    """A filled/empty block bar for vol in [0,1], threshold-colored green/yellow/red."""
    vol = max(0.0, min(1.0, vol))
    filled = int(round(vol * width))
    bar = '█' * filled + '░' * (width - filled)
    if not color:
        return bar
    if vol < 0.33:
        return f"{Fore.GREEN}{bar}{Fore.RESET}"
    if vol < 0.67:
        return f"{Fore.YELLOW}{bar}{Fore.RESET}"
    return f"{Fore.RED}{bar}{Fore.RESET}"


def render_box_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Unicode box table, column-width adaptive, strictly aligned."""
    ncols = len(headers)
    widths = [vis_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < ncols:
                widths[i] = max(widths[i], vis_len(cell))

    TL, TR, BL, BR = '┌', '┐', '└', '┘'
    HZ, VT = '─', '│'
    LT, RT, TT, BT, CR = '├', '┤', '┬', '┴', '┼'

    def hz_line(left: str, mid: str, right: str) -> str:
        return left + mid.join(HZ * (w + 2) for w in widths) + right

    def data_line(cells: list[str]) -> str:
        parts = []
        for i, cell in enumerate(cells):
            w = widths[i] if i < len(widths) else 0
            parts.append(' ' + pad_cell(cell, w) + ' ')
        return VT + VT.join(parts) + VT

    lines = [hz_line(TL, TT, TR), data_line(headers), hz_line(LT, CR, RT)]
    for row in rows:
        lines.append(data_line(list(row) + [''] * (ncols - len(row))))
    lines.append(hz_line(BL, BT, BR))
    return '\n'.join(lines)


# =============================================================================
# Structural chrome (banner, section divider, rules, body lines)
# =============================================================================

ACCENT = Fore.CYAN
INDENT = "  "
RULE_W = 64

# Static pyfiglet `slant` rendering of "R3L", embedded as a constant so the
# banner carries no runtime figlet dependency.
BANNER = r"""
    ____ _____ __
   / __ \__  // /
  / /_/ //_ </ /
 / _, _/__/ / /___
/_/ |_/____/_____/
"""


def term_width(default: int = 80) -> int:
    return shutil.get_terminal_size((default, 24)).columns


def rule_width() -> int:
    return min(term_width(), RULE_W)


def section(title: str, idx: int | None = None, total: int | None = None) -> None:
    """The D1 section divider: a FILLED chip badge (the heaviest element) + a bold
    light-cyan title + a dim heavy `━` rule that recedes. Numbered stages put
    `idx/total` in the chip; unnumbered sections put the title in the chip itself.
    One blank line above and below."""
    print()
    if idx is not None:
        chip = f"{Style.BRIGHT}{Back.LIGHTCYAN_EX}{Fore.BLACK} {idx}/{total} {Style.RESET_ALL}"
        head = f"{chip} {Style.BRIGHT}{Fore.LIGHTCYAN_EX}{title}{Style.RESET_ALL} "
    else:
        chip = f"{Style.BRIGHT}{Back.LIGHTCYAN_EX}{Fore.BLACK} {title} {Style.RESET_ALL}"
        head = f"{chip} "
    rule = f"{Style.DIM}{ACCENT}{'━' * max(0, rule_width() - vis_len(head))}{Style.RESET_ALL}"
    print(head + rule)
    print()


def rule() -> None:
    """A dim heavy `━` rule spanning the body width (drawn under the banner)."""
    print(f"{Style.DIM}{ACCENT}{'━' * rule_width()}{Style.RESET_ALL}")


def inter_stage_rule() -> None:
    """A very thin `┈` separator (no padding) between the two optimization stages.
    Dim DEFAULT-grey (body-text color), deliberately NOT accent-cyan: it recedes as
    a sub-stage boundary rather than reading as structure like the `━` rules do."""
    print(f"{INDENT}{Style.DIM}{'┈' * (rule_width() - len(INDENT))}{Style.RESET_ALL}")


def print_banner() -> None:
    print(f"{ACCENT}{BANNER}{Fore.RESET}")
    rule()


def ok(msg: str) -> None:
    """A green success tick, indented to the body."""
    print(f"{INDENT}{Fore.GREEN}✓{Fore.RESET} {msg}")


def info(msg: str) -> None:
    """A dim secondary line, indented to the body."""
    print(f"{INDENT}{Style.DIM}{msg}{Style.RESET_ALL}")


# =============================================================================
# Spinner for a blocking, silent call (daemon thread)
# =============================================================================

@contextmanager
def spinner(label: str):
    """Animate a braille spinner on a daemon thread while a blocking call runs in
    the with-block, clearing its line on exit. It writes to stderr (captured at
    entry) so an outer `contextlib.redirect_stdout` silencing the call's chatter
    does not swallow the animation. The wrapped call MUST be silenced (redirect its
    stdout) — any leaked print smears the spinner's `\\r` line."""
    stream = sys.stderr
    stop = threading.Event()

    def spin() -> None:
        frames = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
        while not stop.is_set():
            stream.write(f"\r{INDENT}{ACCENT}{next(frames)}{Fore.RESET} {Style.DIM}{label}{Style.RESET_ALL}")
            stream.flush()
            time.sleep(0.08)
        stream.write("\r\x1b[K")
        stream.flush()

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


# =============================================================================
# Live reasoning-summary window (bounded to HEIGHT lines, page-flips on overflow)
# =============================================================================

def _wrap(text: str, width: int) -> list[str]:
    """Greedy word-wrap by visible width (ANSI-aware)."""
    lines, cur = [], ""
    for w in text.split():
        cand = w if not cur else cur + " " + w
        if vis_len(cand) > width:
            if cur:
                lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


class ReasoningWindow:
    """A live, bounded display of an LLM's streamed reasoning summary: a header plus
    at most HEIGHT lines. Fed text deltas as they stream in, it re-wraps the current
    page and page-flips (clears and restarts) when the page would overflow HEIGHT
    lines, so it never grows past HEIGHT. `close()` collapses (erases) the window.

    If no delta ever arrives (the model streamed no summary), nothing is drawn and
    `saw_any` stays False — the caller then prints the static fallback line instead."""

    HEIGHT = 3

    def __init__(self):
        self.saw_any = False
        self._page = ""
        self._drawn = False
        self._width = max(20, term_width() - len(INDENT) - 1)

    def feed(self, delta: str) -> None:
        self.saw_any = True
        cand = self._page + delta
        self._page = delta if len(_wrap(cand, self._width)) > self.HEIGHT else cand
        self._render()

    def _render(self) -> None:
        lines = _wrap(self._page, self._width)[: self.HEIGHT]
        lines += [""] * (self.HEIGHT - len(lines))
        block = [f"{INDENT}{Style.DIM}… reasoning{Style.RESET_ALL}"]
        block += [f"{INDENT}{Style.DIM}{ln}{Style.RESET_ALL}" if ln else "" for ln in lines]
        if self._drawn:
            sys.stdout.write(f"\x1b[{self.HEIGHT + 1}A")
        for b in block:
            sys.stdout.write("\r\x1b[K" + b + "\n")
        self._drawn = True
        sys.stdout.flush()

    def close(self) -> None:
        """Collapse the window (erase it) if anything was drawn; a no-op otherwise."""
        if self._drawn:
            sys.stdout.write(f"\x1b[{self.HEIGHT + 1}A\x1b[J")
            sys.stdout.flush()
            self._drawn = False


def reasoning_unavailable() -> None:
    """The single static line shown when the model streams no reasoning summary —
    no timer, no spinner, just a marker that spatial reasoning is underway."""
    print(f"{INDENT}{Style.DIM}… spatial reasoning in progress (summary unavailable){Style.RESET_ALL}")
