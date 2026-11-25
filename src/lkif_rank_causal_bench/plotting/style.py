"""Colours, measure definitions and the small drawing helpers shared by every figure."""

# colour = the measure; line style = the direction (forward solid with filled
# markers, reverse dashed with open markers; bars: forward filled, reverse open);
# band = +-1 sd over the grid points sharing the plotted parameter value.

from pathlib import Path
from typing import Any, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8982", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, BROWN, PINK = "#2a78d6", "#eb6834", "#1baf7a", "#6a3d9a", "#8c564b", "#d6609b"
C_LKIF, C_RAW, C_GC, C_TE, C_CCC, C_PCMCI, C_CCM = AQUA, MUTED, BLUE, VIOLET, ORANGE, BROWN, PINK

FIG_W = 7.2  # inches; the report's text column is 7.27 in
AXES_LABELS = ("damping $\\beta$", "frequency ratio $K$", "coupling $\\mu$")


class Measure(NamedTuple):
    """An estimator as drawn in the figures: label, result keys, colour, y-axis unit."""

    label: str
    key: str
    reverse_key: str | None
    colour: str
    unit: str


ESTIMATORS = (
    Measure("GC (conditional)", "gc_conditional", "gc_reverse", C_GC, "nats"),
    Measure("TE (Kraskov)", "te_conditional", "te_conditional_reverse", C_TE, "nats"),
    Measure("CCC", "ccc", "ccc_reverse", C_CCC, "CCC"),
    Measure("PCMCI", "pcmci", "pcmci_reverse", C_PCMCI, "|partial corr.|"),
    Measure("CCM", "ccm", "ccm_reverse", C_CCM, "cross-map skill"),
)
TRUTHS = (
    Measure("Normalised LKIF", "truth", "truth_null", C_LKIF, "$\\tau$"),
    # the raw reverse entry is the exactly-zero (1, 2) entry of lkif_raw_full
    Measure("Raw LKIF", "truth_raw", None, C_RAW, "$T$"),
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": GRID,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.linewidth": 0.8,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
    }
)


def tidy(ax: Axes, ygrid: bool = True) -> None:
    """Remove top/right spines and draw a recessive horizontal grid."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def save(fig: Figure, path: Path) -> None:
    """Write a figure tightly cropped and close it."""
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def reverse_of(results: dict[str, Any], m: Measure) -> np.ndarray:
    """A measure's reverse-direction array (the raw LKIF null from lkif_raw_full)."""
    return results["lkif_raw_full"][..., 1, 2] if m.reverse_key is None else results[m.reverse_key]


def mean_sd(grid: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd over the other two grid axes, as a function of one axis."""
    other = tuple(a for a in range(3) if a != axis)
    return grid.mean(axis=other), grid.std(axis=other)


def draw_line(
    ax: Axes,
    x: np.ndarray,
    m: np.ndarray,
    s: np.ndarray | None,
    colour: str,
    forward: bool = True,
    label: str | None = None,
) -> None:
    """One direction of one measure: mean line with an optional +-1 sd band."""
    if s is not None:
        ax.fill_between(x, m - s, m + s, color=colour, alpha=0.16, linewidth=0, zorder=2)
    ax.plot(
        x,
        m,
        color=colour,
        linewidth=2.0,
        linestyle="solid" if forward else (0, (4, 2)),
        marker="o",
        markersize=6.5,
        markeredgewidth=1.4,
        markerfacecolor=colour if forward else SURFACE,
        markeredgecolor=SURFACE if forward else colour,
        zorder=3,
        label=label,
    )


def direction_legend(fig: Figure, y: float, band: bool = True) -> None:
    """Figure legend explaining forward/reverse line styles and the sd band."""
    handles: list[Any] = [
        Line2D(
            [],
            [],
            color=INK_2,
            linewidth=2,
            marker="o",
            markersize=6,
            markerfacecolor=INK_2,
            markeredgecolor=INK_2,
            label="forward  $x_2 \\to v_1$",
        ),
        Line2D(
            [],
            [],
            color=INK_2,
            linewidth=2,
            linestyle=(0, (4, 2)),
            marker="o",
            markersize=6,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_2,
            label="reverse  $v_1 \\to x_2$",
        ),
    ]
    if band:
        handles.append(Patch(facecolor=INK_2, alpha=0.16, label="$\\pm 1$ sd over grid points"))
    fig.legend(handles=handles, loc="lower center", ncols=len(handles), fontsize=8.5, bbox_to_anchor=(0.5, y))


def measure_legend(fig: Figure, y: float, extra: tuple[Any, ...] = ()) -> None:
    """Figure legend with one colour patch per estimator, plus extra handles."""
    handles = [Patch(facecolor=m.colour, label=m.label) for m in ESTIMATORS] + list(extra)
    fig.legend(handles=handles, loc="lower center", ncols=len(handles), fontsize=8.5, bbox_to_anchor=(0.5, y))
