"""Every number the results report quotes, computed from results/ into docs/report_numbers.json."""

# docs/make_results_docx.js reads the JSON, so the report tables always match
# the result files. Missing results come out as null.

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from lkif_rank_causal_bench.analysis.agreement import per_axis_agreement, stat_p
from lkif_rank_causal_bench.paths import DOCS, RESULTS
from lkif_rank_causal_bench.results import load_results, merge_files

# (report key, forward array, reverse array or None)
ESTIMATORS: tuple[tuple[str, str, str | None], ...] = (
    ("gc_conditional", "gc_conditional", "gc_reverse"),
    ("gc_bivariate", "gc_bivariate", None),
    ("te_kraskov", "te_conditional", "te_conditional_reverse"),
    ("te_kraskov_bivariate", "te_bivariate", "te_bivariate_reverse"),
    ("ccc", "ccc", "ccc_reverse"),
    ("pcmci", "pcmci", "pcmci_reverse"),
    ("ccm", "ccm", "ccm_reverse"),
)
SWEEP_ESTIMATORS = (ESTIMATORS[0], ESTIMATORS[2])  # run in every sensitivity sweep
DATASETS = {
    "wide_short": "benchmark_ccc.npz",
    "wide_long": "benchmark_ccc_long.npz",
    "box": "default_grid.npz",
    "stationary": "exp_stationary.npz",
    "outside": "exp_outside_box.npz",
}
OTHER_KEYS = ("pcmci", "pcmci_p", "pcmci_reverse", "pcmci_reverse_p", "ccm", "ccm_reverse")
Numbers = dict[str, Any]


def r3(x: Any) -> float | None:
    """Round to 3 decimals; None for missing or non-finite values."""
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def load(results_dir: Path, name: str) -> dict[str, Any] | None:
    """A result file as a dict, or None if it does not exist."""
    path = results_dir / name
    return load_results(path) if path.exists() else None


def scores(truth: np.ndarray, fwd: np.ndarray, rev: np.ndarray | None) -> Numbers:
    """Global and per-axis agreement of one estimator, plus its direction statistics."""
    t, h = np.ravel(truth), np.ravel(fwd)
    axes = [per_axis_agreement(truth, fwd, i) for i in range(3)]
    out: Numbers = {
        "kendall": r3(stat_p(kendalltau(t, h, variant="b"))[0]),
        "kendall_p": float(stat_p(kendalltau(t, h, variant="b"))[1]),
        "spearman": r3(stat_p(spearmanr(t, h))[0]),
        "pearson": r3(stat_p(pearsonr(t, h))[0]),
        "axis_mean": [r3(np.nanmean(a)) for a in axes],
        "axis_sd": [r3(np.nanstd(a)) for a in axes],
        "median_fwd": float(np.median(fwd)),
    }
    if rev is not None:
        out["rev_gt_fwd"] = r3((rev > fwd).mean())
        out["median_rev"] = float(np.median(rev))
        out["rev_gt_fwd_by_axis"] = [
            [r3(v) for v in (rev > fwd).mean(axis=tuple(a for a in range(3) if a != i))] for i in range(3)
        ]
    return out


def dataset_numbers(d: dict[str, Any]) -> Numbers:
    """Scores of every estimator present in one grid run."""
    out: Numbers = {"n_points": int(d["truth"].size), "truth_range": [r3(d["truth"].min()), r3(d["truth"].max())]}
    for key, fk, rk in ESTIMATORS:
        ok = fk in d and (rk is None or rk in d)
        out[key] = scores(d["truth"], d[fk], d[rk] if rk else None) if ok else None
    if "truth_raw" in d:
        out["vs_raw"] = {
            key: r3(stat_p(kendalltau(np.ravel(d["truth_raw"]), np.ravel(d[fk]), variant="b"))[0])
            for key, fk, _ in ESTIMATORS
            if fk in d
        }
    return out


def sweep_numbers(d: dict[str, Any] | None, base: dict[str, Any], base_x: float) -> Numbers | None:
    """Scores for each value of a setting sweep; the base run supplies base_x."""
    if d is None:
        return None
    xs = sorted({float(k.split("/")[0]) for k in d if "/" in k} | {base_x})
    out: Numbers = {}
    for x in xs:
        src = base if x == base_x else {k.split("/", 1)[1]: v for k, v in d.items() if k.startswith(f"{x}/")}
        entry: Numbers = {
            key: scores(src["truth"], src[fk], src[rk])
            for key, fk, rk in SWEEP_ESTIMATORS
            if rk is not None and fk in src and rk in src
        }
        entry["truth_range"] = [r3(src["truth"].min()), r3(src["truth"].max())]
        out[f"{x:g}"] = entry
    return out


def realisation_numbers(d: dict[str, Any] | None) -> Numbers | None:
    """Scores per realisation count, GC and TE from the sweep, CCC from default-run subsets."""
    if d is None:
        return None
    rs = sorted({int(k.split("/")[0]) for k in d if "/gc_conditional" in k})
    out: Numbers = {
        str(r): {
            key: scores(d["truth"], d[f"{r}/{fk}"], d[f"{r}/{rk}"])
            for key, fk, rk in SWEEP_ESTIMATORS
            if f"{r}/{fk}" in d
        }
        for r in rs
    }
    for r in (10, 20, 40):
        if f"{r}/ccc" in d:
            out.setdefault(str(r), {})["ccc"] = scores(d["ccc_truth"], d[f"{r}/ccc"], d[f"{r}/ccc_reverse"])
    return out


def lag_numbers(d: dict[str, Any] | None) -> tuple[Numbers | None, Numbers | None]:
    """Per-lag GC agreement and the exact-GC summary."""
    if d is None:
        return None, None
    t = d["truth"]

    def tau(a: np.ndarray, b: np.ndarray) -> float | None:
        """Rounded Kendall tau of two grids."""
        return r3(stat_p(kendalltau(a.ravel(), b.ravel(), variant="b"))[0])

    lag = {
        str(int(p)): {
            "kendall": tau(t, d[f"gc_p{p}"]),
            "axis_mean": [r3(np.nanmean(per_axis_agreement(t, d[f"gc_p{p}"], i))) for i in range(3)],
            "median": float(np.median(d[f"gc_p{p}"])),
            "ratio_median": r3(np.median(d[f"gc_p{p}"] / d["pop_cond"])),
            "kendall_vs_exact": tau(d["pop_cond"], d[f"gc_p{p}"]),
        }
        for p in d["lags"]
    }
    exact = {
        "kendall": tau(t, d["pop_cond"]),
        "axis_mean": [r3(np.nanmean(per_axis_agreement(t, d["pop_cond"], i))) for i in range(3)],
        "median": float(np.median(d["pop_cond"])),
        "bivariate_kendall": tau(t, d["pop_biv"]),
        "reverse_max_abs": float(np.abs(d["pop_cond_rev"]).max()),
    }
    return lag, exact


def pcmci_significance(d: dict[str, Any] | None, alpha: float = 0.05) -> Numbers | None:
    """PCMCI rates from its stored p-values (smallest over 2 lags): raw, and Bonferroni-corrected by 2."""
    if d is None or "pcmci_p" not in d:
        return None
    f, r = np.ravel(d["pcmci_p"]), np.ravel(d["pcmci_reverse_p"])
    return {
        "pcmci_p_raw": r3((f <= alpha).mean()),
        "pcmci_rev_p_raw": r3((r <= alpha).mean()),
        "pcmci_p_analytic": r3((2 * f <= alpha).mean()),
        "pcmci_rev_p_analytic": r3((2 * r <= alpha).mean()),
    }


def false_positive_numbers(
    d: dict[str, Any] | None,
    te: dict[str, Any] | None,
    ccm: dict[str, Any] | None = None,
    pcmci: dict[str, Any] | None = None,
) -> Numbers | None:
    """Fraction of points significant per test (rank rule p <= alpha for the surrogate tests)."""
    out: Numbers = {}
    if d is not None:
        a = float(d["alpha"])

        def rate(k: str) -> float | None:
            """Fraction of points with p <= alpha for one test."""
            return r3((d[k] <= a).mean())

        out = {
            k: rate(k)
            for k in (
                "gc_p_analytic",
                "gc_rev_p_analytic",
                "gc_p_surrogate",
                "gc_rev_p_surrogate",
                "ccc_p",
                "ccc_rev_p",
                "ccc_paired_p",
                "ccc_rev_paired_p",
            )
        }
    if te is not None:
        out.update(
            te_p_surrogate=r3((te["fwd_p"] <= 0.05).mean()),
            te_rev_p_surrogate=r3((te["rev_p"] <= 0.05).mean()),
            te_n_points=int(len(te["fwd_p"])),
        )
    if ccm is not None:
        a = float(ccm["alpha"])
        for k in ("ccm_paired_p", "ccm_rev_paired_p", "ccm_p_surrogate", "ccm_rev_p_surrogate"):
            out[k] = r3((ccm[k] <= a).mean())
    out.update(pcmci_significance(pcmci) or {})
    return out or None


def bootstrap_numbers(
    d: dict[str, Any] | None, te: dict[str, Any] | None, *others: dict[str, Any] | None
) -> Numbers | None:
    """Point estimate and 2.5-97.5 % range of tau for GC, CCC, TE, CCM and PCMCI."""
    ci: Numbers = {}
    for src, keys in ((d, ("gc", "ccc")), *((o, ("ccm", "pcmci")) for o in others)):
        if src is None:
            continue
        for k in keys:
            if f"{k}_taus" not in src or not len(src[f"{k}_taus"]):
                continue
            lo, hi = np.percentile(src[f"{k}_taus"], [2.5, 97.5], axis=0)
            ci[k] = {
                "point": [r3(v) for v in src[f"{k}_point"]],
                "lo": [r3(v) for v in lo],
                "hi": [r3(v) for v in hi],
                "B": int(len(src[f"{k}_taus"])),
            }
    if te is not None and len(te["te_taus"]):
        lo, hi = np.percentile(te["te_taus"], [2.5, 97.5], axis=0)
        ci["te"] = {
            "point": [r3(v) for v in te["te_point"]],
            "lo": [r3(v) for v in lo],
            "hi": [r3(v) for v in hi],
            "B": int(len(te["te_taus"])),
            "mean": [r3(v) for v in te["te_taus"].mean(0)],
            "method": str(te["method"]) if "method" in te else "with replacement (invalid)",
        }
    return ci or None


def report_numbers(results_dir: Path = RESULTS) -> Numbers:
    """All report numbers as one nested dictionary."""
    data = {k: load(results_dir, v) for k, v in DATASETS.items()}
    box = data["box"]
    if box is not None:
        merge_files(box, results_dir / "exp_other_estimators.npz", OTHER_KEYS)
    nums: Numbers = {"datasets": {k: dataset_numbers(d) if d is not None else None for k, d in data.items()}}
    if box is not None:
        nums["noise"] = sweep_numbers(load(results_dir, "exp_noise.npz"), box, 0.1)
        nums["dt"] = sweep_numbers(load(results_dir, "exp_dt.npz"), box, 0.1)
    nums["realisations"] = realisation_numbers(load(results_dir, "exp_realisations.npz"))
    nums["lag"], nums["exact_gc"] = lag_numbers(load(results_dir, "exp_gc_lag_statespace.npz"))
    nums["false_positives"] = false_positive_numbers(
        load(results_dir, "exp_false_positives.npz"),
        load(results_dir, "exp_false_positives_te.npz"),
        load(results_dir, "exp_false_positives_ccm.npz"),
        box,
    )
    nums["pcmci_significance"] = {k: pcmci_significance(d) for k, d in data.items()}
    nums["bootstrap"] = bootstrap_numbers(
        load(results_dir, "exp_bootstrap.npz"),
        load(results_dir, "exp_bootstrap_te.npz"),
        load(results_dir, "exp_bootstrap_ccm.npz"),
        load(results_dir, "exp_bootstrap_pcmci.npz"),
    )
    return {k: v for k, v in nums.items() if v is not None or k in ("bootstrap",)}


def write_report_numbers(results_dir: Path = RESULTS, out: Path = DOCS / "report_numbers.json") -> Path:
    """Compute the report numbers and write them as JSON."""
    out.write_text(json.dumps(report_numbers(results_dir), indent=1))
    return out
