"""Spread of Kendall tau over resampled realisations: bootstrap or subsampling."""

# Resampling WITH replacement is invalid for the Kraskov TE estimator: duplicated
# realisations put exact copies of samples at zero distance and bias every
# replicate. TE therefore uses subsampling without replacement (method: subsample).

import time
from pathlib import Path

import numpy as np

from lkif_rank_causal_bench.analysis.agreement import kendall, per_axis_agreement
from lkif_rank_causal_bench.benchmark import Benchmark
from lkif_rank_causal_bench.causal_algorithms import build_algorithm
from lkif_rank_causal_bench.config import RunConfig, algorithm_from_entry
from lkif_rank_causal_bench.oscillator.model import TRUTH_ENTRY
from lkif_rank_causal_bench.results import load_results, save_results


def tau_profile(truth: np.ndarray, estimate: np.ndarray) -> list[float]:
    """[global tau, mean tau along beta, along K, along mu]."""
    return [kendall(truth, estimate)] + [float(np.nanmean(per_axis_agreement(truth, estimate, i))) for i in range(3)]


def run_bootstrap(cfg: RunConfig) -> Path:
    """Each step in options.steps resamples realisations, in order, from one shared random generator."""
    opts = cfg.options
    rng = np.random.default_rng(opts.get("seed", 0))
    point_from = load_results(opts["point_from"])
    truth = point_from["truth"]
    shape = truth.shape
    bench = Benchmark(cfg.system)
    out: dict[str, np.ndarray] = {}
    for step in opts["steps"]:
        name, B = step["name"], step["B"]
        out[f"{name}_point"] = np.array(tau_profile(truth, point_from[step["point_key"]]))
        taus: list[list[float]] = []
        if "per_realisation_from" in step:
            # cheap path: resample stored per-realisation values (CCC)
            d = load_results(step["per_realisation_from"])
            fwd, rev = d[f"{step['point_key']}_per_realisation"], d[f"{step['point_key']}_reverse_per_realisation"]
            R = fwd.shape[1]
            rev_frac = []
            for _ in range(B):
                idx = rng.integers(0, R, R)
                f_, r_ = fwd[:, idx].mean(1).reshape(shape), rev[:, idx].mean(1).reshape(shape)
                taus.append(tau_profile(truth, f_))
                rev_frac.append((r_ > f_).mean())
            out[f"{name}_rev_frac"] = np.array(rev_frac)
        else:
            spec = algorithm_from_entry(step["algorithm"])
            algorithm = build_algorithm(spec.algorithm, spec.params)
            X = bench.samples()
            R = X.shape[1]
            src, tgt = TRUTH_ENTRY
            t0 = time.time()
            for b in range(B):
                if time.time() - t0 > step.get("stop_after", np.inf):
                    break
                if step.get("method", "replacement") == "subsample":
                    idx = np.sort(rng.choice(R, step["size"], replace=False))
                else:
                    idx = rng.integers(0, R, R)
                est = algorithm.estimate(X[:, idx], src, tgt, np.arange(X.shape[0])).values.reshape(shape)
                taus.append(tau_profile(truth, est))
                print(f"{name} replicate {b + 1}: tau {taus[-1][0]:+.3f}", flush=True)
            out["method"] = np.array(f"subsample {step['size']} of {R}" if step.get("method") == "subsample"
                                     else "replacement", dtype=object)
        out[f"{name}_taus"] = np.array(taus)
        lo, hi = np.percentile(np.array(taus), [2.5, 97.5], axis=0)
        print(f"{name}: global tau {out[f'{name}_point'][0]:+.3f}, 2.5-97.5 % [{lo[0]:+.3f}, {hi[0]:+.3f}] (B = {len(taus)})")
    return save_results(cfg.output, out)
