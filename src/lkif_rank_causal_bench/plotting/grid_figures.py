"""The five figures drawn for every grid: sorted by truth, three trends, direction."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from lkif_rank_causal_bench.analysis.agreement import kendall
from lkif_rank_causal_bench.oscillator.grid import axis_values
from lkif_rank_causal_bench.plotting.style import (
    AXES_LABELS,
    C_LKIF,
    ESTIMATORS,
    FIG_W,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    TRUTHS,
    direction_legend,
    draw_line,
    mean_sd,
    measure_legend,
    reverse_of,
    save,
    tidy,
)

Results = dict[str, Any]


class Run(NamedTuple):
    """One run drawn in the direction figure, with its marker style."""

    label: str
    results: Results
    marker: str
    filled: bool


def _unit(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Rescale values so that lo maps to 0 and hi to 1."""
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


def sorted_by_truth_figure(r: Results, path: Path) -> None:
    """Grid points ordered by the LKIF truth, with each estimator's forward and reverse values overlaid."""
    truth = r["truth"].ravel()
    order = np.argsort(truth, kind="stable")
    n = truth.size
    x = np.arange(n)
    window = max(5, n // 12)
    fig, axes = plt.subplots(1, len(ESTIMATORS), figsize=(FIG_W, 2.05), dpi=200, sharey=True)
    for ax, m in zip(axes, ESTIMATORS, strict=True):
        fwd = r[m.key].ravel()[order]
        rev = reverse_of(r, m).ravel()[order]
        lo, hi = float(np.min(fwd)), float(np.max(fwd))  # forward range sets the scale for both directions
        ax.plot(x, _unit(truth[order], float(truth.min()), float(truth.max())), color=C_LKIF, linewidth=1.8, zorder=4)
        ax.scatter(x, _unit(rev, lo, hi), s=4, color=MUTED, alpha=0.6, linewidths=0, zorder=2)
        ax.scatter(x, _unit(fwd, lo, hi), s=5, color=m.colour, alpha=0.55, linewidths=0, zorder=3)
        smooth = np.convolve(_unit(fwd, lo, hi), np.ones(window) / window, mode="valid")
        ax.plot(x[window // 2 : window // 2 + smooth.size], smooth, color=m.colour, linewidth=1.8, zorder=5)
        ax.set_title(f"{m.label}\n$\\tau$ = {kendall(truth, r[m.key].ravel()):+.2f}", fontsize=8, color=INK, pad=4)
        ax.set_xlabel("points by LKIF rank", fontsize=7.5)
        ax.set_xticks([0, n - 1], ["low", "high"], fontsize=7)
        ax.tick_params(labelsize=7)
        tidy(ax)
    axes[0].set_ylabel("scaled value", fontsize=7.5)
    axes[0].set_ylim(-0.25, 1.15)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.4)
    handles = [
        Line2D([], [], color=C_LKIF, linewidth=1.8, label="LKIF truth"),
        Line2D([], [], color=INK_2, marker="o", linewidth=0, markersize=3.5, label="forward, each point"),
        Line2D([], [], color=INK_2, linewidth=1.8, label="forward, moving average"),
        Line2D([], [], color=MUTED, marker="o", linewidth=0, markersize=3.5, label="reverse, each point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    save(fig, path)


def trend_limits(r: Results) -> list[tuple[float, float]]:
    """y-limits per panel shared by the three trend figures of one grid."""
    lims = []
    for m in TRUTHS + ESTIMATORS:
        lo, hi = np.inf, -np.inf
        for grid in (r[m.key], reverse_of(r, m)):
            for i in range(3):
                mean, sd = mean_sd(grid, i)
                lo, hi = min(lo, float((mean - sd).min())), max(hi, float((mean + sd).max()))
        pad = 0.06 * (hi - lo)
        lims.append((lo - pad, hi + pad))
    return lims


def trend_figure(r: Results, axis: int, lims: Sequence[tuple[float, float]], path: Path) -> None:
    """Truths and estimators, both directions, against one parameter (mean +- sd over the others)."""
    x = axis_values(r["params_tensor"], axis)
    fig, grid = plt.subplots(2, 4, figsize=(FIG_W, 3.7), dpi=200)
    axes = grid.ravel()
    axes[-1].set_visible(False)
    for ax, m, lim in zip(axes, TRUTHS + ESTIMATORS, lims, strict=False):  # 8 axes, 7 panels
        draw_line(ax, x, *mean_sd(reverse_of(r, m), axis), m.colour, forward=False)
        draw_line(ax, x, *mean_sd(r[m.key], axis), m.colour, forward=True)
        ax.set_title(m.label, fontsize=8.5, color=INK, pad=4)
        ax.set_xlabel(AXES_LABELS[axis], fontsize=8.5)
        ax.set_ylabel(m.unit, fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(*lim)
        tidy(ax)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6, h_pad=0.8)
    direction_legend(fig, -0.01)
    save(fig, path)


def reverse_fraction_figure(runs: Sequence[Run], path: Path) -> None:
    """Fraction of grid points where reverse > forward, along each axis, per estimator and run."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.3), dpi=200, sharey=True)
    for i, ax in enumerate(axes):
        x = axis_values(runs[0].results["params_tensor"], i)
        other = tuple(a for a in range(3) if a != i)
        for m in ESTIMATORS:
            for run in runs:
                frac = (reverse_of(run.results, m) > run.results[m.key]).mean(axis=other)
                ax.plot(
                    x,
                    frac,
                    color=m.colour,
                    linewidth=1.8,
                    marker=run.marker,
                    markersize=6,
                    markeredgewidth=1.3,
                    markerfacecolor=m.colour if run.filled else SURFACE,
                    markeredgecolor=m.colour,
                    zorder=3,
                )
        ax.set_xlabel(AXES_LABELS[i])
        ax.set_ylim(-0.05, 1.05)
        tidy(ax)
    axes[0].set_ylabel("fraction of grid points\nwith reverse > forward")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    extra: tuple[Any, ...] = ()
    if len(runs) > 1:
        extra = tuple(
            Line2D(
                [],
                [],
                color=INK_2,
                linewidth=0,
                marker=run.marker,
                markersize=6,
                markerfacecolor=INK_2 if run.filled else SURFACE,
                markeredgecolor=INK_2,
                label=run.label,
            )
            for run in runs
        )
    measure_legend(fig, -0.03, extra)
    save(fig, path)


def has_all_estimators(r: Results) -> bool:
    """Whether a results dict holds both directions of every estimator."""
    return all(k in r for m in ESTIMATORS for k in (m.key, m.reverse_key))


def grid_figures(prefix: str, r: Results, runs: Sequence[Run], out_dir: Path) -> list[Path]:
    """Write the five figures of one grid as <prefix>_*.png."""
    lims = trend_limits(r)
    paths = {
        name: out_dir / f"{prefix}_{name}.png"
        for name in ("sorted_by_truth", "trend_beta", "trend_K", "trend_mu", "reverse_fraction")
    }
    sorted_by_truth_figure(r, paths["sorted_by_truth"])
    for axis, name in enumerate(("trend_beta", "trend_K", "trend_mu")):
        trend_figure(r, axis, lims, paths[name])
    reverse_fraction_figure(runs, paths["reverse_fraction"])
    return list(paths.values())
