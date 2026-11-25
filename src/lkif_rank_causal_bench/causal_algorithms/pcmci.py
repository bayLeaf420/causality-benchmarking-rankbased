"""PCMCI with the partial-correlation test (tigramite), realisations as multiple datasets."""

import numpy as np

from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm, Estimate
from lkif_rank_causal_bench.causal_algorithms.parallel import map_in_pool, physical_cores


def demean(X: np.ndarray) -> np.ndarray:
    """Remove the across-realisation mean at each time point, as GC and TE do."""
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=0, keepdims=True)


def pcmci_link(
    X: np.ndarray, source: int, target: int, tau_max: int = 2, pc_alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """(forward value, forward p, reverse value, reverse p): max |MCI partial corr| over lags 1..tau_max."""
    import tigramite.pcmci as tigramite_pcmci
    from tigramite.data_processing import DataFrame
    from tigramite.independence_tests.parcorr import ParCorr

    data = np.transpose(demean(X), (0, 2, 1))  # (R, T, N)
    df = DataFrame(data, analysis_mode="multiple")
    runner = tigramite_pcmci.PCMCI(dataframe=df, cond_ind_test=ParCorr(significance="analytic"), verbosity=0)
    res = runner.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    val, pv = res["val_matrix"], res["p_matrix"]
    lags = slice(1, tau_max + 1)
    return (
        float(np.abs(val[source, target, lags]).max()),
        float(pv[source, target, lags].min()),
        float(np.abs(val[target, source, lags]).max()),
        float(pv[target, source, lags].min()),
    )


def _pcmci_task(task: tuple[np.ndarray, int, int, int, float]) -> tuple[float, float, float, float]:
    """One grid point in a worker process."""
    X, source, target, tau_max, pc_alpha = task
    return pcmci_link(X, source, target, tau_max, pc_alpha)


class PCMCI(CausalAlgorithm):
    """PCMCI-ParCorr link strength and p-value, both directions in one pass per point."""

    name = "pcmci"

    def __init__(self, tau_max: int = 2, pc_alpha: float = 0.05, workers: int | None = None) -> None:
        """Set the largest lag, the PC-stage significance level and worker processes (None = per core)."""
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.workers = workers or physical_cores()

    def estimate(self, X: np.ndarray, source: int, target: int, indices: np.ndarray) -> Estimate:
        """Forward direction only."""
        forward, _ = self.estimate_both(X, source, target, indices, reverse=False)
        return forward

    def estimate_both(
        self, X: np.ndarray, source: int, target: int, indices: np.ndarray, reverse: bool = True
    ) -> tuple[Estimate, Estimate | None]:
        """Both directions from one PCMCI run per point."""
        tasks = [
            (np.asarray(X[i], dtype=np.float64), source, target, self.tau_max, self.pc_alpha) for i in range(X.shape[0])
        ]
        out = np.array(map_in_pool(_pcmci_task, tasks, self.workers))
        forward = Estimate(values=out[:, 0], p_values=out[:, 1])
        backward = Estimate(values=out[:, 2], p_values=out[:, 3]) if reverse else None
        return forward, backward
