"""Summary figures drawn from the report numbers: heatmaps, significance bars, confidence intervals."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from lkif_rank_causal_bench.plotting.style import (
    BLUE,
    C_CCC,
    C_GC,
    C_TE,
    FIG_W,
    GRID,
    INK,
    INK_2,
    MUTED,
    ORANGE,
    SURFACE,
    save,
    tidy,
)

Numbers = dict[str, Any]

# rows of the overview heatmaps: (report key, label)
ROWS = (
    ("gc_conditional", "GC, conditional"),
    ("gc_bivariate", "GC, bivariate"),
    ("te_kraskov", "TE (Kraskov)"),
    ("ccc", "CCC"),
    ("pcmci", "PCMCI"),
    ("ccm", "CCM"),
)
# columns: (dataset key, label)
SETUPS = (
    ("wide_short", "Wide\nshort"),
    ("wide_long", "Wide\nlong"),
    ("box", "Default"),
    ("stationary", "Stationary"),
    ("outside", "Outside\nbox"),
)
# diverging: orange = opposite order to LKIF, grey = no agreement, blue = same order
TAU_CMAP = LinearSegmentedColormap.from_list("tau", [ORANGE, "#f6c9b3", "#f1f0ec", "#b9d2f1", BLUE])
# sequential, neutral: how often the null direction wins
ERR_CMAP = LinearSegmentedColormap.from_list("err", ["#f4f3ef", "#b9b7af", "#52514e"])


def _value(nums: Numbers, setup: str, row: str, field: str, index: int | None = None) -> float:
    """One number from report_numbers, NaN when that estimator or run is missing."""
    entry = (nums["datasets"].get(setup) or {}).get(row)
    if entry is None or entry.get(field) is None:
        return np.nan
    v = entry[field] if index is None else entry[field][index]
    return np.nan if v is None else float(v)


def _annotated_heatmap(
    ax: Axes,
    values: np.ndarray,
    cmap: LinearSegmentedColormap,
    vmin: float,
    vmax: float,
    fmt: str,
    xlabels: Sequence[str],
    ylabels: Sequence[str],
) -> None:
    """Grid of coloured cells with each value written in it; missing cells are marked n/a."""
    n_rows, n_cols = values.shape
    ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=vmin, vmax=vmax, edgecolors=SURFACE, linewidth=2.5)
    for (i, j), v in np.ndenumerate(values):
        if np.isnan(v):
            ax.text(j + 0.5, i + 0.5, "n/a", ha="center", va="center", fontsize=7, color=MUTED)
            continue
        shade = (v - vmin) / (vmax - vmin)
        dark = shade > 0.9 or (cmap is TAU_CMAP and shade < 0.1)
        ax.text(
            j + 0.5, i + 0.5, format(v, fmt), ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK
        )
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks(np.arange(n_cols) + 0.5, list(xlabels), fontsize=7.5)
    ax.set_yticks(np.arange(n_rows) + 0.5, list(ylabels), fontsize=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def overview_figure(nums: Numbers, path: Path) -> None:
    """Global Kendall tau and the share of points where reverse > forward, every estimator on every grid."""
    tau = np.array([[_value(nums, s, r, "kendall") for s, _ in SETUPS] for r, _ in ROWS])
    err = np.array([[_value(nums, s, r, "rev_gt_fwd") for s, _ in SETUPS] for r, _ in ROWS]) * 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.8), dpi=200, gridspec_kw={"wspace": 0.06})
    _annotated_heatmap(a1, tau, TAU_CMAP, -1.0, 1.0, "+.2f", [name for _, name in SETUPS], [name for _, name in ROWS])
    a1.set_title("Kendall $\\tau$ against LKIF\n(higher is better)", fontsize=8.5, color=INK, pad=6)
    _annotated_heatmap(a2, err, ERR_CMAP, 0.0, 100.0, ".0f", [name for _, name in SETUPS], [""] * len(ROWS))
    a2.set_title("% of points where reverse > forward\n(lower is better)", fontsize=8.5, color=INK, pad=6)
    save(fig, path)


def axis_overview_figure(nums: Numbers, path: Path) -> None:
    """Mean per-axis Kendall tau (along beta, K, mu) for every estimator and grid."""
    fig, axes = plt.subplots(1, len(SETUPS), figsize=(FIG_W, 2.5), dpi=200, gridspec_kw={"wspace": 0.05})
    for k, (ax, (setup, label)) in enumerate(zip(axes, SETUPS, strict=True)):
        vals = np.array([[_value(nums, setup, r, "axis_mean", i) for i in range(3)] for r, _ in ROWS])
        _annotated_heatmap(
            ax,
            vals,
            TAU_CMAP,
            -1.0,
            1.0,
            "+.2f",
            ["$\\beta$", "$K$", "$\\mu$"],
            [name for _, name in ROWS] if k == 0 else [""] * len(ROWS),
        )
        ax.set_title(label.replace("\n", " "), fontsize=8, color=INK, pad=5)
    save(fig, path)


def significance_figure(nums: Numbers, path: Path) -> None:
    """Share of grid points significant at 0.05 in each direction, per test."""
    fp = nums.get("false_positives") or {}
    tests = [
        ("GC, $\\chi^2$ test", "gc_p_analytic", "gc_rev_p_analytic", C_GC),
        ("GC, trial-shuffle surrogates", "gc_p_surrogate", "gc_rev_p_surrogate", C_GC),
        ("TE, trial-shuffle surrogates", "te_p_surrogate", "te_rev_p_surrogate", C_TE),
        ("CCC, t-test against 0", "ccc_p", "ccc_rev_p", C_CCC),
        ("CCC, paired t-test fwd vs rev", "ccc_paired_p", "ccc_rev_paired_p", C_CCC),
    ]
    y = np.arange(len(tests))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(FIG_W * 0.85, 2.6), dpi=200)
    for yi, (_label, fk, rk, colour) in zip(y, tests, strict=True):
        f, r = 100 * (fp.get(fk) or 0.0), 100 * (fp.get(rk) or 0.0)
        ax.barh(yi + h / 2, f, h, color=colour, edgecolor=colour, linewidth=1.2, zorder=3)
        ax.barh(yi - h / 2, r, h, color=SURFACE, edgecolor=colour, linewidth=1.2, zorder=3)
        ax.text(f + 1.5, yi + h / 2, f"{f:.0f} %", va="center", fontsize=7.5, color=INK_2)
        ax.text(r + 1.5, yi - h / 2, f"{r:.0f} %", va="center", fontsize=7.5, color=INK_2)
    ax.set_yticks(y, [t[0] for t in tests], fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of grid points significant at 0.05")
    ax.tick_params(axis="y", length=0)
    tidy(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.legend(
        handles=[
            Patch(facecolor=INK_2, edgecolor=INK_2, label="forward (true detection)"),
            Patch(facecolor=SURFACE, edgecolor=INK_2, label="reverse (false positive)"),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncols=2,
        fontsize=8,
    )
    save(fig, path)


def confidence_figure(nums: Numbers, path: Path) -> None:
    """Forest plot: point estimate and 2.5-97.5 % range of the global and per-axis tau for GC, TE, CCC."""
    ci = nums.get("bootstrap") or {}
    estimators = [("gc", "GC", C_GC), ("te", "TE (20 of 40 realisations)", C_TE), ("ccc", "CCC", C_CCC)]
    measures = ["global", "along $\\beta$", "along $K$", "along $\\mu$"]
    rows: list[tuple[str, str, str, float, float, float]] = []  # (measure, estimator, colour, centre, lo, hi)
    for i, measure in enumerate(measures):
        for key, label, colour in estimators:
            c = ci.get(key)
            if c:
                centre = c["mean"][i] if key == "te" and "mean" in c else c["point"][i]
                rows.append((measure, label, colour, centre, c["lo"][i], c["hi"][i]))
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6), dpi=200)
    y = np.arange(len(rows))[::-1].astype(float)
    y -= np.repeat(np.arange(len(measures)), len(estimators))[: len(rows)] * 1.2  # gap between measures
    for yi, (_, _label, colour, centre, lo, hi) in zip(y, rows, strict=True):
        ax.plot([lo, hi], [yi, yi], color=colour, linewidth=3.5, solid_capstyle="round", zorder=3)
        ax.plot(centre, yi, "o", color=colour, markersize=6, markeredgecolor=SURFACE, zorder=4)
        ax.text(1.12, yi, f"{centre:+.2f}  [{lo:+.2f}, {hi:+.2f}]", va="center", fontsize=7.5, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=1.0, zorder=2)
    ax.set_yticks(y, [label for _, label, *_ in rows], fontsize=7.5)
    for i, measure in enumerate(measures):
        top = y[i * len(estimators)]
        ax.text(-1.17, top + 0.55, measure, fontsize=8.5, color=INK, ha="left", va="bottom", fontweight="bold")
    ax.set_xlim(-1.0, 1.05)
    ax.set_ylim(y.min() - 0.7, y.max() + 1.2)
    ax.set_xlabel("Kendall $\\tau$ against LKIF")
    ax.tick_params(axis="y", length=0)
    tidy(ax, ygrid=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    save(fig, path)
