"""Rank agreement (Kendall tau-b, Spearman, Pearson) between an estimator and the LKIF truth."""

# The benchmark asks whether a heuristic orders the grid as LKIF does, so
# Kendall tau-b (a rank statistic that handles ties) is the primary score.
# A global tau over a 3-D grid hides axis-wise inversions, hence per_axis_agreement.

from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from lkif_rank_causal_bench.oscillator.grid import AXIS_NAMES


def stat_p(result: Any) -> tuple[float, float]:
    """(statistic, p-value) of a scipy.stats test result."""
    return float(result.statistic), float(result.pvalue)


def kendall(truth: np.ndarray, estimate: np.ndarray) -> float:
    """Kendall tau-b between two arrays of the same size."""
    return stat_p(kendalltau(np.ravel(truth), np.ravel(estimate), variant="b"))[0]


def rank_agreement(truth: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    """Global Kendall, Spearman and Pearson statistics with p-values, over finite entries."""
    t = np.asarray(truth).ravel()
    h = np.asarray(estimate).ravel()
    if t.shape != h.shape:
        raise ValueError(f"shape mismatch: truth {t.shape} vs heuristic {h.shape}")
    ok = np.isfinite(t) & np.isfinite(h)
    n_finite = int(ok.sum())
    if n_finite < 3:
        return dict(
            kendall_tau=np.nan,
            kendall_p=np.nan,
            spearman_rho=np.nan,
            spearman_p=np.nan,
            pearson_r=np.nan,
            pearson_p=np.nan,
            n=int(t.size),
            n_finite=n_finite,
        )
    t, h = t[ok], h[ok]
    kt = stat_p(kendalltau(t, h, variant="b"))
    sp = stat_p(spearmanr(t, h))
    if np.ptp(t) == 0 or np.ptp(h) == 0:  # pearsonr needs variance in both inputs
        pr_stat, pr_p = np.nan, np.nan
    else:
        pr_stat, pr_p = stat_p(pearsonr(t, h))
    return dict(
        kendall_tau=kt[0],
        kendall_p=kt[1],
        spearman_rho=sp[0],
        spearman_p=sp[1],
        pearson_r=pr_stat,
        pearson_p=pr_p,
        n=int(ok.size),
        n_finite=n_finite,
    )


def per_axis_agreement(truth: np.ndarray, estimate: np.ndarray, axis: int) -> np.ndarray:
    """Kendall tau along one grid axis with the other two fixed; NaN for short or constant slices."""
    t = np.asarray(truth)
    h = np.asarray(estimate)
    if t.shape != h.shape or t.ndim != 3:
        raise ValueError(f"need two matching 3-D grids, got {t.shape}, {h.shape}")
    t_moved = np.moveaxis(t, axis, -1)
    h_moved = np.moveaxis(h, axis, -1)
    out = np.full(t_moved.shape[:-1], np.nan)
    for idx in np.ndindex(*t_moved.shape[:-1]):
        a, b = t_moved[idx], h_moved[idx]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 3 and np.ptp(a[ok]) > 0 and np.ptp(b[ok]) > 0:
            out[idx] = stat_p(kendalltau(a[ok], b[ok], variant="b"))[0]
    return out


def summarise(truth: np.ndarray, estimate: np.ndarray, name: str = "heuristic") -> dict[str, Any]:
    """Global statistics plus median, worst and fraction-positive per-axis tau."""
    per_axis = {}
    for ax, ax_name in enumerate(AXIS_NAMES):
        taus = per_axis_agreement(truth, estimate, ax)
        finite = taus[np.isfinite(taus)]
        per_axis[ax_name] = {
            "taus": taus,
            "median": float(np.median(finite)) if finite.size else np.nan,
            "min": float(np.min(finite)) if finite.size else np.nan,
            "frac_positive": float(np.mean(finite > 0)) if finite.size else np.nan,
        }
    return {"name": name, "global": rank_agreement(truth, estimate), "per_axis": per_axis}


def format_report(report: dict[str, Any]) -> str:
    """Plain-text rendering of a summarise() report."""
    g = report["global"]
    lines = [
        f"{report['name']}:",
        f"  global   Kendall tau-b = {g['kendall_tau']:+.4f}  (p = {g['kendall_p']:.2e})",
        f"           Spearman rho  = {g['spearman_rho']:+.4f}",
        f"           Pearson r     = {g['pearson_r']:+.4f}   [linear, not a rank statistic]",
        f"           n = {g['n_finite']}/{g['n']} finite",
        "  per-axis Kendall tau (other two axes fixed):",
    ]
    for ax_name, d in report["per_axis"].items():
        lines.append(
            f"           along {ax_name:5s}: median {d['median']:+.4f}  "
            f"worst {d['min']:+.4f}  frac(tau>0) {d['frac_positive']:.2f}"
        )
    return "\n".join(lines)


def summary_line(label: str, truth: np.ndarray, forward: np.ndarray, reverse: np.ndarray | None) -> str:
    """One log line: global tau, mean per-axis tau, reverse > forward fraction, median forward."""
    axes = " ".join(f"{np.nanmean(per_axis_agreement(truth, forward, i)):+.2f}" for i in range(3))
    rev = "" if reverse is None else f"  rev>fwd {(reverse > forward).mean():.3f}"
    return (
        f"{label:28s} tau {rank_agreement(truth, forward)['kendall_tau']:+.3f}  "
        f"per-axis(b,K,mu) {axes}{rev}  median fwd {np.median(forward):.4f}"
    )
