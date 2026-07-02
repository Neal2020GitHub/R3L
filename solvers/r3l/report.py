"""
Solver-layer reporting: the optimizer's console + on-disk loss reporting.

Lives in the solver package (not utils/) because it is solver-private — its only
consumer is solvers/r3l/optimize.py — and it depends on solvers.r3l.losses
(clamp_param). Keeping it here makes that a same-package import and removes the
utils→solvers import inversion this module used to introduce; its presentation
primitives come DOWN from utils.console.

  - LossDashboard            fixed-height in-place loss grid (console view)
  - emit_constraint_param_report  the optimized-Var-param box table (console + file)
  - LossCurveRecorder        per-iter loss history -> loss_curve.csv / .png
"""

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from colorama import Fore, Style

from .losses import clamp_param
from utils.console import INDENT, pad_cell, render_box_table, term_width, vis_len, vol_bar

# =============================================================================
# Optimization loss dashboard (fixed-height in-place region)
# =============================================================================

def _fmt_loss(v: float) -> str:
    """Magnitude-adaptive loss formatting (fewer decimals as the value grows)."""
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _progress_bar(frac: float, width: int) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return f"{Fore.GREEN}{'█' * filled}{Style.DIM}{'░' * (width - filled)}{Style.RESET_ALL}"


class LossDashboard:
    """Fixed-height, in-place console region that replaces the flat per-step loss
    log. ROWS are loss components ("total" first and distinct); COLUMNS are the
    most-recent logged steps that fit, scrolling left as new ones arrive. Only the
    value cells carry color (green = improved vs the previous column, red = rose).
    `max_rows` None shows every component; an int shows "total" + the (max_rows-1)
    largest others.

    This is purely the console view — the full per-component history still goes to
    loss_curve.csv / .png via LossCurveRecorder, unchanged.
    """

    LABEL_GAP = 2
    COL_W = 8
    BAR_W = 12

    def __init__(self, stage_name: str, total_iters: int, max_rows: Optional[int] = None):
        self.stage_name = stage_name
        self.total_iters = total_iters
        self.max_rows = max_rows
        self._hist: List[Tuple[int, Dict[str, float]]] = []
        self._order: List[str] = []
        self._label_w = 0
        self._best = math.inf
        self._height = 0
        self._drawn = False

    @staticmethod
    def _label(key: str) -> str:
        """Console label for a nominal key: strip the `_loss` suffix the dynamic
        constraint terms carry (against_wall_loss -> against_wall); the built-in
        coll_*/wall/aesthetics/prior_* keys have no suffix and pass through."""
        return key.split("_loss")[0]

    def _row(self, total: float, nominal: Dict[str, torch.Tensor]) -> Dict[str, float]:
        row: Dict[str, float] = {"total": total}
        for k, v in nominal.items():
            row[self._label(k)] = float(v)
        return row

    def _ensure_order(self, row: Dict[str, float]) -> None:
        if self._order:
            return
        rest = [k for k in row.keys() if k != "total"]
        self._order = (["total"] if "total" in row else []) + rest
        self._label_w = max(max(vis_len(k) for k in self._order), vis_len("step")) + self.LABEL_GAP

    def _visible_rows(self, latest: Dict[str, float]) -> List[str]:
        if self.max_rows is None:
            return self._order
        others = [k for k in self._order if k != "total"]
        others.sort(key=lambda k: abs(latest.get(k, 0.0)), reverse=True)
        return (["total"] if "total" in self._order else []) + others[: max(0, self.max_rows - 1)]

    def _max_cols(self) -> int:
        return max(1, (term_width() - self._label_w - len(INDENT)) // self.COL_W)

    def _color(self, comp: str, j: int) -> str:
        if j <= 0:
            return ""
        cur = self._hist[j][1].get(comp)
        prev = self._hist[j - 1][1].get(comp)
        if cur is None or prev is None:
            return ""
        return Fore.GREEN if cur <= prev else Fore.RED

    def _render_lines(self) -> List[str]:
        latest = self._hist[-1][1]
        rows = self._visible_rows(latest)
        vis_idx = range(len(self._hist))[-self._max_cols():]

        head = f"{Style.DIM}{pad_cell('step', self._label_w, 'l')}"
        for j in vis_idx:
            head += pad_cell(str(self._hist[j][0]), self.COL_W, 'r')
        head += Style.RESET_ALL
        lines = [INDENT + head]

        for comp in rows:
            label_style = Style.BRIGHT if comp == "total" else Style.DIM
            label = f"{label_style}{pad_cell(comp, self._label_w, 'l')}{Style.RESET_ALL}"
            cells = ""
            for j in vis_idx:
                val = self._hist[j][1].get(comp)
                text = _fmt_loss(val) if val is not None else "-"
                cells += f"{self._color(comp, j)}{pad_cell(text, self.COL_W, 'r')}{Style.RESET_ALL}"
            lines.append(INDENT + label + cells)

        step = self._hist[-1][0]
        bar = _progress_bar(step / self.total_iters if self.total_iters else 1.0, self.BAR_W)
        bottom = (f"{INDENT}{Style.DIM}{pad_cell(self.stage_name + '  ', self._label_w, 'l')}{Style.RESET_ALL}"
                  f"{bar}  {Style.DIM}{step}/{self.total_iters}{Style.RESET_ALL}"
                  f"   {Style.DIM}best{Style.RESET_ALL} {Fore.GREEN}{_fmt_loss(self._best)}{Fore.RESET}")
        lines.append(bottom)
        return lines

    def update(self, step: int, total: float, nominal: Dict[str, torch.Tensor]) -> None:
        """Append one logged step and redraw the region in place."""
        row = self._row(total, nominal)
        self._ensure_order(row)
        self._hist.append((step, row))
        self._best = min(self._best, total)

        lines = self._render_lines()
        out = sys.stdout
        if self._drawn:
            out.write(f"\x1b[{self._height}A")
        for ln in lines:
            out.write("\r\x1b[K" + ln + "\n")
        self._height = len(lines)
        self._drawn = True
        out.flush()

    def finish(self) -> None:
        """Release the in-place region so later lines print below the final frame."""
        self._drawn = False


# =============================================================================
# Constraint parameter report (table + volatility bar)
# =============================================================================

# Natural scale for each parameter kind - defines what "full volatility" means
_KIND_SCALE = {
    "unit": 1.0,        # [0, 1] range
    "nonneg": 2.5,     # typical spatial extent
    "angle_deg": 25.0, # full rotation range
}


def _compute_volatility(eff_prior: float, eff_final: float, kind: str) -> float:
    """Volatility as absolute change normalized by kind's natural scale."""
    delta = abs(eff_final - eff_prior)
    scale = _KIND_SCALE.get(kind, 1.0)
    return min(1.0, delta / scale)


def _fmt_value(val: float, kind: str) -> str:
    """Format a param value based on its kind."""
    if kind == "angle_deg":
        return f"{val:.1f}°"
    return f"{val:.4f}"


def _kind_label(kind: str) -> str:
    """Human-readable kind label."""
    labels = {"unit": "unit", "nonneg": "nonneg", "angle_deg": "angle(deg)"}
    return labels.get(kind, kind)


def format_constraint_param_report(
    names: List[str],
    kinds: List[str],
    prior_raw: List[float],
    final_raw: List[float],
    report_tag: Optional[str] = None,
    color: bool = True,
) -> str:
    """
    Format a constraint parameter report as a Unicode box table.

    Args:
        names: Parameter names.
        kinds: Parameter kinds (unit/nonneg/angle_deg).
        prior_raw: Raw prior values.
        final_raw: Raw final values (after optimization).
        report_tag: Optional tag (e.g., attempt number) for header.
        color: If True, include ANSI colors in output.
    
    Returns:
        Multi-line string of the formatted report.
    """
    n = len(names)
    if n == 0:
        return ""

    # Build header line
    tag_str = f"|attempt={report_tag}" if report_tag is not None else ""
    header_line = f"[params{tag_str}] optimized constraint params (P={n})"

    # Table headers
    headers = ["param", "kind", "eff(prior→final)", "raw(prior→final)", "note", "volatility"]

    # Build rows
    rows: list[list[str]] = []
    for i in range(n):
        name = names[i]
        kind = kinds[i]
        raw_p = prior_raw[i]
        raw_f = final_raw[i]
        eff_p = float(clamp_param(torch.as_tensor(raw_p), kind))
        eff_f = float(clamp_param(torch.as_tensor(raw_f), kind))

        # Check if clamp was applied
        clamped = (raw_f != eff_f)
        note = "clamped" if clamped else ""

        # Format values
        eff_str = f"{_fmt_value(eff_p, kind)} → {_fmt_value(eff_f, kind)}"
        raw_str = f"{_fmt_value(raw_p, kind)} → {_fmt_value(raw_f, kind)}"

        # Volatility
        vol = _compute_volatility(eff_p, eff_f, kind)
        bar = vol_bar(vol, width=12, color=color)

        rows.append([name, _kind_label(kind), eff_str, raw_str, note, bar])

    table = render_box_table(headers, rows)
    return f"{header_line}\n\n{table}"


def emit_constraint_param_report(
    names: List[str],
    kinds: List[str],
    prior_raw: List[float],
    final_raw: List[float],
    save_dir: str,
    report_tag: Optional[str] = None,
) -> Optional[str]:
    """
    Emit constraint parameter report: print to stdout (with color) and append to file (no color).

    Args:
        names: Parameter names.
        kinds: Parameter kinds.
        prior_raw: Raw prior values.
        final_raw: Raw final values.
        save_dir: Directory to write report file.
        report_tag: Optional tag for header.
    
    Returns:
        Path to the written file, or None if no params.
    """
    if not names:
        return None

    # Print to stdout with color
    report_color = format_constraint_param_report(
        names, kinds, prior_raw, final_raw,
        report_tag=report_tag, color=True
    )
    print(report_color)

    # Write to file without color
    report_plain = format_constraint_param_report(
        names, kinds, prior_raw, final_raw,
        report_tag=report_tag, color=False
    )

    file_path = os.path.join(save_dir, "constraint_params_report.txt")
    os.makedirs(save_dir, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(report_plain)
        f.write("\n\n")

    return file_path


# =============================================================================
# Loss Curve Recording (CSV + Plot)
# =============================================================================

class LossCurveRecorder:
    """Records loss history for CSV export and visualization."""

    def __init__(self, save_dir: Optional[Path | str] = None):
        self.history: List[Dict[str, float]] = []
        self.save_dir = Path(save_dir) if save_dir else None

    def record(self, iter: int, total: float, nominal: Dict[str, torch.Tensor]) -> None:
        """Record a single iteration's loss values."""
        entry: Dict[str, float] = {"iter": iter, "total": total}
        for k, v in nominal.items():
            entry[k] = v.item()
        self.history.append(entry)

    def save_csv(self, filename: str = "loss_curve.csv") -> Optional[Path]:
        """Save loss history to CSV file."""
        if not self.save_dir or not self.history:
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
        return path

    def save_plot(self, filename: str = "loss_curve.png") -> Optional[Path]:
        """Save loss curve visualization."""
        if not self.save_dir or not self.history:
            return None
        # Lazy import matplotlib (heavy dependency)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename

        iters = [h["iter"] for h in self.history]
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot total loss (bold black)
        ax.plot(iters, [h["total"] for h in self.history],
                label="total", linewidth=2, color="black")

        # Plot component losses
        keys = [k for k in self.history[0].keys() if k not in ("iter", "total")]
        for k in keys:
            ax.plot(iters, [h[k] for h in self.history], label=k, alpha=0.7)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("Optimization Loss Curve")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path