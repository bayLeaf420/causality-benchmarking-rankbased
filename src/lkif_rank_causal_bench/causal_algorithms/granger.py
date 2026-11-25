"""Linear Granger causality from two least-squares regressions, pooled over realisations."""

# GC(s -> y | S) = ln( var(resid of y on lags of S) / var(resid of y on lags of S + s) ).
# For a Gaussian process TE = GC / 2. Omitting a variable from a VAR gives a VARMA,
# so the reduced model is biased at finite lag; see granger_exact for the exact value.

from collections.abc import Iterable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm, Estimate


def design_matrix(X: jax.Array, variables: Sequence[int], p: int) -> jax.Array:
    """Stack lags 1..p of the chosen variables, shape (R * (T - p), len(variables) * p)."""
    T = X.shape[2]
    # the predictor of y_t at lag l is X[:, v, t - l]; t runs over p .. T-1
    cols = [X[:, v, p - lag : T - lag] for v in variables for lag in range(1, p + 1)]
    Z = jnp.stack(cols, axis=1)  # (R, k, T-p)
    return jnp.transpose(Z, (0, 2, 1)).reshape(-1, len(cols))


def residual_variance(Z: jax.Array, y: jax.Array, ridge: float, dof_correction: bool) -> jax.Array:
    """Least-squares residual variance via ridged normal equations (jit- and vmap-friendly)."""
    N, k = Z.shape
    ZtZ = Z.T @ Z
    # ridge scaled by mean(diag(Z'Z)) so it is dimensionless and inert when well-conditioned
    scale = jnp.mean(jnp.diagonal(ZtZ)) + 1e-300
    coef = jnp.linalg.solve(ZtZ + ridge * scale * jnp.eye(k), Z.T @ y)
    resid = y - Z @ coef
    denom = (N - k) if dof_correction else N
    return jnp.sum(resid**2) / denom


def gc_from_trials(
    X: jax.Array,
    target: int,
    source: int,
    p: int,
    cond: Iterable[int] | None = None,
    ensemble_demean: bool = True,
    ridge: float = 1e-10,
    dof_correction: bool = True,
) -> jax.Array:
    """GC source -> target in nats for one point, X of shape (R, n_vars, T); cond None = all others."""
    R, V, T = X.shape
    if ensemble_demean and R < 2:
        raise ValueError(
            "ensemble_demean needs n_realisations > 1 (it would zero a single "
            "realisation). Pass ensemble_demean=False for single-trial data."
        )
    if p < 1 or p >= T:
        raise ValueError(f"lag order p={p} must satisfy 1 <= p < n_time={T}")
    if target == source:
        raise ValueError("target and source must differ")

    # the process is non-stationary: without removing the ensemble mean, GC
    # measures agreement with the deterministic mean trajectory, not the dynamics
    if ensemble_demean:
        X = X - jnp.mean(X, axis=0, keepdims=True)

    if cond is None:
        others = [v for v in range(V) if v not in (target, source)]
    else:
        others = [v for v in cond if v not in (target, source)]
    full_vars = sorted({target, source, *others})
    reduced_vars = sorted({target, *others})

    y = X[:, target, p:].reshape(-1)
    var_full = residual_variance(design_matrix(X, full_vars, p), y, ridge, dof_correction)
    var_reduced = residual_variance(design_matrix(X, reduced_vars, p), y, ridge, dof_correction)
    return jnp.log(var_reduced / var_full)


def gc_grid(
    Xg: jax.Array,
    target: int,
    source: int,
    p: int,
    cond: Iterable[int] | None = None,
    ensemble_demean: bool = True,
    ridge: float = 1e-10,
    dof_correction: bool = True,
) -> jax.Array:
    """gc_from_trials vmapped over the leading grid axis of Xg (n_points, R, n_vars, T)."""
    cond_t = None if cond is None else tuple(cond)

    def one(X: jax.Array) -> jax.Array:
        """GC at one point."""
        return gc_from_trials(X, target, source, p, cond_t, ensemble_demean, ridge, dof_correction)

    return jax.jit(jax.vmap(one))(Xg)


def var_order_bic(
    X: jax.Array, max_p: int, ensemble_demean: bool = True, ridge: float = 1e-10
) -> tuple[int, jax.Array]:
    """VAR order minimising BIC on the full model, and the BIC for p = 1 .. max_p."""
    # pick the order once and hold it fixed across the grid; a per-point order
    # would make part of the measured trend an artefact of order selection
    R, V, T = X.shape
    if ensemble_demean:
        X = X - jnp.mean(X, axis=0, keepdims=True)
    scores = []
    for p in range(1, max_p + 1):
        Z = design_matrix(X, list(range(V)), p)
        N = Z.shape[0]
        logdet = jnp.asarray(0.0)
        for t in range(V):
            y = X[:, t, p:].reshape(-1)
            logdet = logdet + jnp.log(residual_variance(Z, y, ridge, dof_correction=False))
        scores.append(logdet + V * V * p * jnp.log(N) / N)
    bic = jnp.stack(scores)
    return int(jnp.argmin(bic)) + 1, bic


class GrangerCausality(CausalAlgorithm):
    """Conditional or bivariate linear GC at a fixed lag order."""

    name = "gc"

    def __init__(
        self, lag: int = 2, conditional: bool = True, chunk: int = 4, gaussian_te_key: str | None = None
    ) -> None:
        """Set the lag order, conditioning, grid points per vmapped call and optional TE = GC/2 key."""
        self.lag = lag
        self.conditional = conditional
        # one vmapped call over a whole 216 x 40 x 1500 grid needs several GB; the
        # chunk also fixes the batching, which changes results at the 1e-16 level
        self.chunk = chunk
        self.gaussian_te_key = gaussian_te_key

    def estimate(self, X: np.ndarray, source: int, target: int, indices: np.ndarray) -> Estimate:
        """GC at every point, computed chunk by chunk in float64."""
        cond = None if self.conditional else ()
        out = [
            np.asarray(
                gc_grid(
                    jnp.asarray(np.asarray(X[i : i + self.chunk], dtype=np.float64)),
                    target=target,
                    source=source,
                    p=self.lag,
                    cond=cond,
                )
            )
            for i in range(0, X.shape[0], self.chunk)
        ]
        return Estimate(values=np.concatenate(out))

    def derived(self, forward: Estimate) -> dict[str, np.ndarray]:
        """Gaussian TE = GC / 2 when a key for it is configured."""
        return {self.gaussian_te_key: forward.values / 2.0} if self.gaussian_te_key else {}
