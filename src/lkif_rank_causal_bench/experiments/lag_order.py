"""GC at several lag orders against the exact (population) GC of the model."""

from pathlib import Path

import numpy as np

from lkif_rank_causal_bench.analysis.agreement import kendall
from lkif_rank_causal_bench.benchmark import Benchmark
from lkif_rank_causal_bench.causal_algorithms.granger import GrangerCausality
from lkif_rank_causal_bench.causal_algorithms.granger_exact import exact_gc_grid
from lkif_rank_causal_bench.config import RunConfig
from lkif_rank_causal_bench.oscillator.model import TRUTH_ENTRY
from lkif_rank_causal_bench.results import save_results


def run_lag_order(cfg: RunConfig) -> Path:
    """Conditional GC (both directions) for each lag in options.lags, plus exact GC."""
    lags: list[int] = cfg.options["lags"]
    bench = Benchmark(cfg.system)
    truth = bench.truth()["truth"]
    X = bench.samples()
    idx = np.arange(X.shape[0])
    src, tgt = TRUTH_ENTRY
    out: dict[str, np.ndarray] = {}
    for p in lags:
        gc = GrangerCausality(lag=p)
        out[f"gc_p{p}"] = gc.estimate(X, src, tgt, idx).values.reshape(bench.shape)
        out[f"gc_rev_p{p}"] = gc.estimate(X, tgt, src, idx).values.reshape(bench.shape)
        print(f"lag {p}: tau {kendall(truth, out[f'gc_p{p}']):+.3f}", flush=True)
    out["pop_cond"] = exact_gc_grid(cfg.system, src, tgt, p=2)
    out["pop_cond_rev"] = exact_gc_grid(cfg.system, tgt, src, p=2)
    out["pop_biv"] = exact_gc_grid(cfg.system, src, tgt, p=2, cond=())
    print(f"exact GC: tau {kendall(truth, out['pop_cond']):+.3f}, "
          f"max |reverse| {np.abs(out['pop_cond_rev']).max():.2e}")
    return save_results(cfg.output, {"lags": np.array(lags), "truth": truth,
                                     "params_tensor": bench.params(), **out})
