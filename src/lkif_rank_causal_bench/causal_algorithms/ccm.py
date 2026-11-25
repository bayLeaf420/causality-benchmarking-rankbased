"""Convergent cross mapping (Sugihara et al. 2012): does target's embedding predict source?"""

import numpy as np
from scipy.spatial import KDTree

from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm, Estimate
from lkif_rank_causal_bench.causal_algorithms.parallel import map_in_pool, physical_cores
from lkif_rank_causal_bench.causal_algorithms.pcmci import demean


def delay_embed(x: np.ndarray, E: int, tau: int) -> np.ndarray:
    """Rows (x_t, x_{t-tau}, ..., x_{t-(E-1)tau}) for t = (E-1)tau .. T-1."""
    T = x.shape[0]
    start = (E - 1) * tau
    return np.stack([x[start - i * tau : T - i * tau] for i in range(E)], axis=1)


def ccm_skill(source: np.ndarray, target: np.ndarray, E: int = 4, tau: int = 4, theiler: int = 10) -> float:
    """Correlation between source and its simplex estimate from target's delay embedding."""
    M = delay_embed(target, E, tau)
    y = source[(E - 1) * tau :]
    n = M.shape[0]
    k = E + 1
    # extra neighbours so k remain after dropping the Theiler window
    d, idx = KDTree(M).query(M, k=k + 2 * theiler + 1)
    keep = np.abs(idx - np.arange(n)[:, None]) > theiler
    order = np.argsort(~keep, axis=1, kind="stable")[:, :k]  # first k kept, nearest first
    di = np.take_along_axis(d, order, axis=1)
    ii = np.take_along_axis(idx, order, axis=1)
    w = np.exp(-di / np.maximum(di[:, :1], 1e-12))
    pred = (w * y[ii]).sum(axis=1) / w.sum(axis=1)
    return float(np.corrcoef(pred, y)[0, 1])


def ccm_link(
    X: np.ndarray, source: int, target: int, E: int = 4, tau: int = 4, theiler: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Skill per realisation: (target predicts source, source predicts target), each shape (R,)."""
    Xd = demean(X)
    fwd = [ccm_skill(Xd[r, source], Xd[r, target], E, tau, theiler) for r in range(Xd.shape[0])]
    rev = [ccm_skill(Xd[r, target], Xd[r, source], E, tau, theiler) for r in range(Xd.shape[0])]
    return np.array(fwd), np.array(rev)


def _ccm_task(task: tuple[np.ndarray, int, int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """One grid point in a worker process."""
    X, source, target, E, tau, theiler = task
    return ccm_link(X, source, target, E, tau, theiler)


class CCM(CausalAlgorithm):
    """Cross-map skill in both directions: the mean over realisations, and each realisation's value."""

    name = "ccm"

    def __init__(self, E: int = 4, tau: int = 4, theiler: int = 10, workers: int | None = None) -> None:
        """Set the embedding dimension, delay, Theiler window and worker processes (None = per core)."""
        self.E = E
        self.tau = tau
        self.theiler = theiler
        self.workers = workers or physical_cores()

    def estimate(self, X: np.ndarray, source: int, target: int, indices: np.ndarray) -> Estimate:
        """Forward direction only."""
        forward, _ = self.estimate_both(X, source, target, indices, reverse=False)
        return forward

    def estimate_both(
        self, X: np.ndarray, source: int, target: int, indices: np.ndarray, reverse: bool = True
    ) -> tuple[Estimate, Estimate | None]:
        """Both directions from one pass per point."""
        tasks = [
            (np.asarray(X[i], dtype=np.float64), source, target, self.E, self.tau, self.theiler)
            for i in range(X.shape[0])
        ]
        out = np.array(map_in_pool(_ccm_task, tasks, self.workers))  # (n_points, 2, R)
        # the mean of each 1-D row, exactly as the stored results were computed
        fwd, rev = (np.array([float(np.mean(row)) for row in out[:, d]]) for d in (0, 1))
        forward = Estimate(values=fwd, per_realisation=out[:, 0])
        return forward, (Estimate(values=rev, per_realisation=out[:, 1]) if reverse else None)
