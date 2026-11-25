"""Non-parametric transfer entropy with IDTxl's Kraskov (KSG) estimator."""

# TE(s -> y) = I( y(t) ; s(t-1..t-p) | y(t-1..t-p), c(t-1..t-p) for c in cond ),
# on the same ensemble-demeaned, pooled samples GC uses (TE = GC / 2 if Gaussian).
# IDTxl's JIDT estimators need pkg_resources, which current setuptools lacks, so
# this uses IDTxl's pure-Python PythonKraskovCMI.

import os
from collections.abc import Iterable

import numpy as np

from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm, Estimate


def te_inputs(
    X: np.ndarray,
    source: int,
    target: int,
    lag: int = 2,
    cond: Iterable[int] | None = None,
    ensemble_demean: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Target present (N, 1), source past (N, lag) and conditioning pasts for one point."""
    X = np.asarray(X, dtype=np.float64)
    R, V, T = X.shape
    if ensemble_demean:
        X = X - X.mean(axis=0, keepdims=True)
    if cond is None:
        cond = [v for v in range(V) if v not in (source, target)]
    cond = [v for v in cond if v not in (source, target)]

    def past(v: int) -> np.ndarray:
        """Lags 1..lag of variable v, shape (R * (T - lag), lag)."""
        cols = [X[:, v, lag - k : T - k] for k in range(1, lag + 1)]
        return np.stack(cols, axis=-1).reshape(-1, lag)

    present = X[:, target, lag:].reshape(-1, 1)
    conditional = np.concatenate([past(target)] + [past(v) for v in cond], axis=1)
    return present, past(source), conditional


def kraskov_te(
    present: np.ndarray,
    source_past: np.ndarray,
    conditional: np.ndarray,
    k: int = 4,
    threads: int = 1,
    seed: int = 0,
) -> float:
    """Kraskov conditional mutual information in nats (can be slightly negative from bias)."""
    from idtxl.estimators_python import PythonKraskovCMI

    est = PythonKraskovCMI({"kraskov_k": k, "normalise": True, "num_threads": threads, "rng_seed": seed})
    return float(est.estimate(present, source_past, conditional))


class KraskovTE(CausalAlgorithm):
    """Conditional (or bivariate) Kraskov TE at a fixed lag, one estimate per grid point."""

    name = "te_kraskov"

    def __init__(self, lag: int = 2, k: int = 4, conditional: bool = True, threads: int | None = None) -> None:
        """Set the lag, neighbour count, conditioning and k-d tree threads (None = all CPUs)."""
        self.lag = lag
        self.k = k
        self.conditional = conditional
        self.threads = threads or os.cpu_count() or 1

    def estimate(self, X: np.ndarray, source: int, target: int, indices: np.ndarray) -> Estimate:
        """TE at every point; each point's noise seed is its global grid index."""
        cond = None if self.conditional else ()
        values = np.array(
            [
                kraskov_te(
                    *te_inputs(np.asarray(X[i], dtype=np.float64), source, target, lag=self.lag, cond=cond),
                    k=self.k,
                    threads=self.threads,
                    seed=int(indices[i]),
                )
                for i in range(X.shape[0])
            ]
        )
        return Estimate(values=values)
