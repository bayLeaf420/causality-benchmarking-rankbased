"""Figures of the follow-up experiments: CCC settings, GC lag order, sensitivity sweeps."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from lkif_rank_causal_bench.analysis.agreement import kendall
from lkif_rank_causal_bench.plotting.style import C_CCC, C_GC, C_TE, FIG_W, INK, INK_2, SURFACE, save, tidy

# From claude_tests/logs/ccc_window_sweep.log: one realisation at beta = 0.5,
# K = 1.3, mu = 2.0 (time span 60, 600 samples). (setting, forward, reverse)
CCC_SETTINGS_SWEEP = (
    ("90/90/90/16", 0.0384, 0.0273),
    ("150/15/40/8", 0.0988, -0.0123),
    ("150/15/40/4", 0.0554, -0.0120),
    ("200/20/50/4", 0.0682, -0.0027),
    ("250/25/60/4", 0.0501, 0.0021),
    ("300/30/60/2", 0.0182, -0.0112),
    ("250/25/60/2", -0.0014, -0.0109),
)


def ccc_settings_figure(path: Path) -> None:
    """Forward (filled) and reverse (open) CCC bars for each window/bin setting."""
    x = np.arange(len(CCC_SETTINGS_SWEEP))
    w = 0.36
    fig, ax = plt.subplots(figsize=(FIG_W * 0.85, 2.6), dpi=200)
    ax.bar(
        x - 0.5 * (w + 0.04),
        [s[1] for s in CCC_SETTINGS_SWEEP],
        w,
        color=C_CCC,
        edgecolor=C_CCC,
        linewidth=1.2,
        zorder=3,
        label="forward  $x_2 \\to v_1$",
    )
    ax.bar(
        x + 0.5 * (w + 0.04),
        [s[2] for s in CCC_SETTINGS_SWEEP],
        w,
        color=SURFACE,
        edgecolor=C_CCC,
        linewidth=1.2,
        zorder=3,
        label="reverse  $v_1 \\to x_2$",
    )
    ax.axhline(0, color=INK_2, linewidth=1.0, zorder=4)
    ticks = [s[0].rsplit("/", 1) for s in CCC_SETTINGS_SWEEP]
    ax.set_xticks(x, [f"{a}\n{b} bins" for a, b in ticks], fontsize=7.5)
    ax.set_xlabel("past / present / jump, number of bins")
    ax.set_ylabel("CCC")
    ax.legend(loc="upper right", fontsize=8.5)
    tidy(ax)
    save(fig, path)


def lag_order_figure(d: dict[str, Any], path: Path) -> None:
    """Left: tau of GC per lag order against the exact GC; right: estimate / exact GC (log)."""
    lags, truth, pop = d["lags"], d["truth"], d["pop_cond"]
    tau = [kendall(truth, d[f"gc_p{p}"]) for p in lags]
    tau_pop = kendall(truth, pop)
    logr = np.array([np.log(d[f"gc_p{p}"] / pop).ravel() for p in lags])
    m, s = logr.mean(1), logr.std(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(FIG_W, 2.5), dpi=200)
    a1.plot(lags, tau, color=C_GC, linewidth=2, marker="o", markersize=6.5, markeredgecolor=SURFACE, zorder=3)
    a1.axhline(tau_pop, color=INK_2, linewidth=1.0, zorder=2)
    a1.annotate(
        f"exact GC: $\\tau$ = {tau_pop:+.2f}",
        (lags[-1], tau_pop),
        textcoords="offset points",
        xytext=(0, -11),
        ha="right",
        fontsize=8,
        color=INK_2,
    )
    a1.set_xlabel("GC lag order $p$")
    a1.set_ylabel("Kendall $\\tau$ vs LKIF")
    a1.set_ylim(0, 0.6)
    a1.set_title("Rank agreement", fontsize=9, color=INK, pad=5)
    a2.fill_between(lags, np.exp(m - s), np.exp(m + s), color=C_GC, alpha=0.16, linewidth=0)
    a2.plot(lags, np.exp(m), color=C_GC, linewidth=2, marker="o", markersize=6.5, markeredgecolor=SURFACE, zorder=3)
    a2.axhline(1, color=INK_2, linewidth=1.0, zorder=2)
    a2.set_yscale("log")
    a2.set_xlabel("GC lag order $p$")
    a2.set_ylabel("estimate / exact GC")
    a2.set_title("Size of the estimate", fontsize=9, color=INK, pad=5)
    for a in (a1, a2):
        a.set_xticks(lags)
        tidy(a)
    fig.tight_layout()
    save(fig, path)


def sweep_taus(
    sweep: dict[str, Any], base: dict[str, Any] | None, base_label: str | None, keys: tuple[str, ...]
) -> tuple[list[float], dict[str, list[float]]]:
    """Global tau per stored 'label/key' sweep value; base supplies base_label if given."""
    labels = {k.split("/")[0] for k in sweep if "/" in k}
    if base is not None and base_label is not None:
        labels.add(base_label)
    ordered = sorted(labels, key=float)

    def tau_at(label: str, key: str) -> float:
        """Tau for one sweep label (from base when it is the base label)."""
        if base is not None and label == base_label:
            return kendall(base["truth"], base[key])
        truth = sweep[f"{label}/truth"] if f"{label}/truth" in sweep else sweep["truth"]
        return kendall(truth, sweep[f"{label}/{key}"])

    return [float(x) for x in ordered], {key: [tau_at(x, key) for x in ordered] for key in keys}


def sensitivity_figure(
    noise: dict[str, Any], dt: dict[str, Any], realisations: dict[str, Any], base: dict[str, Any], path: Path
) -> None:
    """Global tau against noise power, sampling interval and number of realisations."""
    keys = ("gc_conditional", "te_conditional")
    colours = {"gc_conditional": C_GC, "te_conditional": C_TE}
    panels = [
        (sweep_taus(noise, base, "0.1", keys), "noise power $q = q_1 = q_2$"),
        (sweep_taus(dt, base, "0.1", keys), "sampling interval $\\Delta t$"),
        (
            sweep_taus(
                {k: v for k, v in realisations.items() if not k.endswith(("/ccc", "/ccc_reverse"))}, None, None, keys
            ),
            "realisations $R$",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.4), dpi=200, sharey=True)
    for ax, ((xs, taus), xlabel) in zip(axes, panels, strict=True):
        for key, vals in taus.items():
            ax.plot(
                xs, vals, color=colours[key], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, zorder=3
            )
        ax.axhline(0, color=INK_2, linewidth=1.0, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{x:g}" for x in xs])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        tidy(ax)
    ccc_rs = sorted({int(k.split("/")[0]) for k in realisations if k.endswith("/ccc")})
    if ccc_rs:
        axes[2].plot(
            ccc_rs,
            [kendall(realisations["ccc_truth"], realisations[f"{r}/ccc"]) for r in ccc_rs],
            color=C_CCC,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            zorder=3,
        )
    axes[0].set_ylabel("Kendall $\\tau$ vs LKIF")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    handles = [
        Patch(facecolor=C_GC, label="GC (conditional)"),
        Patch(facecolor=C_TE, label="TE (Kraskov)"),
        Patch(facecolor=C_CCC, label="CCC"),
    ]
    fig.legend(handles=handles, loc="lower center", ncols=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    save(fig, path)
