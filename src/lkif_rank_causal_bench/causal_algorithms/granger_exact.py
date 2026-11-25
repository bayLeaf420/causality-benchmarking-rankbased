"""Exact (population) Granger causality of the known VAR(1) model, by Kalman filtering."""

# The state-space method of MVGC2 (Barnett & Seth 2015) in Python: each
# conditional variance is the innovation variance of a noise-free Kalman filter
# that observes only some variables. The filter is time-varying from the initial
# covariance, which handles the transient; variances are averaged over the same
# indices k = p .. T-1 that the regression estimator pools.

from collections.abc import Sequence

import numpy as np

from lkif_rank_causal_bench.config import SystemConfig
from lkif_rank_causal_bench.oscillator.grid import params_tensor
from lkif_rank_causal_bench.oscillator.model import make_system_matrix, noise_covariance
from lkif_rank_causal_bench.oscillator.sampler import propagator


def innovation_variances(
    P: np.ndarray, Q: np.ndarray, Sigma0: np.ndarray, observed: Sequence[int], target: int, n_time: int
) -> np.ndarray:
    """var(target_k - E[target_k | observed_0..k-1]) for k = 0 .. n_time-1."""
    observed = list(observed)
    C = np.eye(P.shape[0])[observed]
    j = observed.index(target)
    prior = np.array(Sigma0, dtype=float)
    out = np.empty(n_time)
    for k in range(n_time):
        S = C @ prior @ C.T
        out[k] = S[j, j]
        gain = prior @ C.T @ np.linalg.pinv(S)  # condition on y_k = C x_k
        post = prior - gain @ C @ prior
        prior = P @ post @ P.T + Q
        prior = 0.5 * (prior + prior.T)
    return out


def population_gc(
    P: np.ndarray,
    Q: np.ndarray,
    Sigma0: np.ndarray,
    source: int,
    target: int,
    n_time: int,
    p: int = 2,
    cond: Sequence[int] | None = None,
) -> float:
    """Exact GC source -> target in nats, pooled over k = p .. n_time-1; cond None = all others."""
    n = P.shape[0]
    if cond is None:
        cond = [v for v in range(n) if v not in (source, target)]
    full = sorted({target, source, *cond})
    reduced = sorted({target, *cond})
    v_full = innovation_variances(P, Q, Sigma0, full, target, n_time)[p:]
    v_red = innovation_variances(P, Q, Sigma0, reduced, target, n_time)[p:]
    return float(np.log(v_red.mean() / v_full.mean()))


def exact_gc_grid(
    system: SystemConfig, source: int, target: int, p: int = 2, cond: Sequence[int] | None = None
) -> np.ndarray:
    """Exact GC at every grid point of a system, shape (n_beta, n_K, n_mu)."""
    sim = system.simulation
    params = np.asarray(params_tensor(system.grid))
    G = noise_covariance(sim.q1, sim.q2)
    Sigma0 = np.diag(np.asarray(sim.sigma_init) ** 2)
    out = np.zeros(params.shape[:-1])
    for idx in np.ndindex(*params.shape[:-1]):
        beta, K, mu = params[idx]
        P, Q = (np.asarray(m) for m in propagator(make_system_matrix(beta, K, mu, sim.beta2), G, sim.dt))
        out[idx] = population_gc(P, Q, Sigma0, source, target, sim.tau[2], p=p, cond=cond)
    return out
