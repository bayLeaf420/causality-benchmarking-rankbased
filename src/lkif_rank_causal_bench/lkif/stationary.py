"""Normalised LKIF of the stationary process (damped driver), from the Lyapunov equation."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_continuous_lyapunov

from lkif_rank_causal_bench.lkif.flow import normalise_flow, raw_flow
from lkif_rank_causal_bench.oscillator.model import make_system_matrix


def stationary_covariance(A: jax.Array | np.ndarray, G: jax.Array | np.ndarray) -> np.ndarray:
    """Solve A Sigma + Sigma A^T + G = 0; raises if A is not strictly stable."""
    A_np = np.asarray(A, dtype=float)
    G_np = np.asarray(G, dtype=float)
    lam = np.linalg.eigvals(A_np)
    if np.max(lam.real) >= -1e-12:
        raise ValueError(
            "A is not strictly stable (max Re(eig) = "
            f"{np.max(lam.real):.3e}), so no stationary covariance exists. "
            "With beta2 = 0 the driving oscillator is undamped; set beta2 > 0 "
            "to use the stationary route, or use the time-integral route instead."
        )
    return solve_continuous_lyapunov(A_np, -G_np)


def stationary_point(beta: float, K: float, mu: float, G: np.ndarray, beta2: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalised and raw LKIF matrices at one point of the stationary process."""
    A = make_system_matrix(beta, K, mu, beta2)
    Sigma = jnp.asarray(stationary_covariance(A, G))
    flow = raw_flow(A, Sigma)
    return np.asarray(normalise_flow(flow, Sigma, jnp.asarray(G))), np.asarray(flow)
