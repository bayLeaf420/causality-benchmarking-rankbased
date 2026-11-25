"""Significance tests in both directions; a significant reverse result is a false positive."""

# Trial-shuffle surrogates take the source from another realisation (cyclic
# shift by s = 1..S): each series keeps its dynamics, the coupling is broken.
# With S = 19 the smallest attainable p is 0.05, so the rank rule is p <= alpha.

import itertools
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import chi2, ttest_1samp, ttest_rel

from lkif_rank_causal_bench.benchmark import Benchmark
from lkif_rank_causal_bench.causal_algorithms import build_algorithm
from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm
from lkif_rank_causal_bench.config import RunConfig, algorithm_from_entry
from lkif_rank_causal_bench.oscillator.model import TRUTH_ENTRY
from lkif_rank_causal_bench.results import load_results, save_results


def p_values(result: Any) -> np.ndarray:
    """p-values of a vectorised scipy.stats test."""
    return np.asarray(result.pvalue)


def surrogate_p(observed: np.ndarray, surrogates: np.ndarray) -> np.ndarray:
    """Rank p-value (1 + #surrogates >= observed) / (S + 1), per point."""
    return (1 + (surrogates >= observed).sum(0)) / (surrogates.shape[0] + 1)


def shuffled(X: np.ndarray, variable: int, shift: int, realisation_axis: int) -> np.ndarray:
    """Copy of X with one variable taken from another realisation (cyclic shift)."""
    Xs = X.copy()
    if realisation_axis == 1:
        Xs[:, :, variable] = np.roll(X[:, :, variable], shift, axis=1)
    else:
        Xs[:, variable] = np.roll(X[:, variable], shift, axis=0)
    return Xs


def grid_surrogates(algorithm: CausalAlgorithm, X: np.ndarray, source: int, target: int, n: int) -> np.ndarray:
    """Algorithm values on n trial-shuffled copies of the whole grid, shape (n, n_points)."""
    idx = np.arange(X.shape[0])
    return np.array([algorithm.estimate(shuffled(X, source, s, 1), source, target, idx).values
                     for s in range(1, n + 1)])


def run_significance(cfg: RunConfig) -> Path:
    """GC (analytic chi-square and surrogates) and CCC (t-tests) at every grid point."""
    opts = cfg.options
    alpha, S = opts.get("alpha", 0.05), opts.get("n_surrogates", 19)
    obs = load_results(opts["observed_from"])
    shape = obs["truth"].shape
    bench = Benchmark(cfg.system)
    X = bench.samples()
    src, tgt = TRUTH_ENTRY
    spec = algorithm_from_entry(opts["gc"])
    gc = build_algorithm(spec.algorithm, spec.params)
    lag = spec.params.get("lag", 2)
    N = X.shape[1] * (X.shape[3] - lag)
    out: dict[str, np.ndarray] = {
        # the likelihood-ratio statistic N * GC is chi-square(lag) under the null
        "gc_p_analytic": chi2.sf(N * obs["gc_conditional"], lag),
        "gc_rev_p_analytic": chi2.sf(N * obs["gc_reverse"], lag),
    }
    surr_f = grid_surrogates(gc, X, src, tgt, S)
    surr_r = grid_surrogates(gc, X, tgt, src, S)
    out["gc_p_surrogate"] = surrogate_p(obs["gc_conditional"].ravel(), surr_f).reshape(shape)
    out["gc_rev_p_surrogate"] = surrogate_p(obs["gc_reverse"].ravel(), surr_r).reshape(shape)
    out["gc_surrogates"] = surr_f.reshape((S, *shape))
    out["gc_rev_surrogates"] = surr_r.reshape((S, *shape))

    fwd, rev = obs["ccc_per_realisation"], obs["ccc_reverse_per_realisation"]
    out["ccc_p"] = p_values(ttest_1samp(fwd, 0, axis=1, alternative="greater")).reshape(shape)
    out["ccc_rev_p"] = p_values(ttest_1samp(rev, 0, axis=1, alternative="greater")).reshape(shape)
    out["ccc_paired_p"] = p_values(ttest_rel(fwd, rev, axis=1, alternative="greater")).reshape(shape)
    out["ccc_rev_paired_p"] = p_values(ttest_rel(rev, fwd, axis=1, alternative="greater")).reshape(shape)

    for k in ("gc_p_analytic", "gc_rev_p_analytic", "gc_p_surrogate", "gc_rev_p_surrogate",
              "ccc_p", "ccc_rev_p", "ccc_paired_p", "ccc_rev_paired_p"):
        print(f"{k:22s} significant at {alpha}: {(out[k] <= alpha).mean():.3f}")
    return save_results(cfg.output, {"alpha": np.array(alpha), "params_tensor": bench.params(), **out})


def run_significance_subgrid(cfg: RunConfig) -> Path:
    """Surrogate test for one algorithm at the sub-grid points options.index_values^3."""
    opts = cfg.options
    S = opts.get("n_surrogates", 19)
    spec = algorithm_from_entry(opts["algorithm"])
    algorithm = build_algorithm(spec.algorithm, spec.params)
    bench = Benchmark(cfg.system)
    X = bench.samples()
    params = bench.params()
    src, tgt = TRUTH_ENTRY
    res: dict[str, list[float]] = {k: [] for k in ("fwd", "rev", "fwd_p", "rev_p")}
    points = []
    for ib, ik, im in itertools.product(*(opts["index_values"],) * 3):
        g = int(np.ravel_multi_index((ib, ik, im), bench.shape))
        Xp = X[g : g + 1]
        for lab, s_, t_ in (("fwd", src, tgt), ("rev", tgt, src)):
            observed = algorithm.estimate(Xp, s_, t_, np.array([g])).values[0]
            surr = [algorithm.estimate(shuffled(Xp, s_, s, 1), s_, t_, np.array([g])).values[0]
                    for s in range(1, S + 1)]
            res[lab].append(observed)
            res[lab + "_p"].append(float(surrogate_p(np.array(observed), np.array(surr))))
        points.append(params[ib, ik, im])
        print(f"point {len(points)}: fwd p={res['fwd_p'][-1]:.2f}  rev p={res['rev_p'][-1]:.2f}", flush=True)
    return save_results(cfg.output, {"points": np.array(points), **{k: np.array(v) for k, v in res.items()}})
