"""Exact realisations of the linear oscillator system as a VAR(1) process."""

# The system is linear, so the exactly discretised process is
#     x_{k+1} = P x_k + eps_k,  P = exp(A dt),  eps_k ~ N(0, Q),
# with Q from the Van Loan block-matrix identity: no integrator error at any dt,
# and the ensemble marginals are exactly the N(m(t), Sigma(t)) the LKIF truth uses.

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from lkif_rank_causal_bench.config import ParameterGrid, SimulationConfig
from lkif_rank_causal_bench.oscillator.grid import params_tensor
from lkif_rank_causal_bench.oscillator.model import make_system_matrix, noise_covariance


def propagator(A: jax.Array, G: jax.Array, dt: float) -> tuple[jax.Array, jax.Array]:
    """One-step propagator P = exp(A dt) and noise covariance Q (Van Loan)."""
    n = A.shape[0]
    M = jnp.block([[A, G], [jnp.zeros((n, n)), -A.T]]) * dt
    E = jax.scipy.linalg.expm(M)
    P = E[:n, :n]
    Q = E[:n, n:] @ P.T
    return P, 0.5 * (Q + Q.T)  # symmetrise away round-off


def psd_sqrt(M: jax.Array) -> jax.Array:
    """Symmetric square root of a PSD matrix, safe when it is rank deficient."""
    # Q is ill-conditioned (cond ~ 1e4 at dt = 0.025) and exactly 0 when G = 0;
    # an eigendecomposition handles both where Cholesky can fail on round-off.
    w, V = jnp.linalg.eigh(0.5 * (M + M.T))
    return V * jnp.sqrt(jnp.clip(w, 0.0, None))[None, :]


def discrete_lyapunov(P: jax.Array, Q: jax.Array, n_doublings: int = 32) -> jax.Array:
    """Solve S = P S P^T + Q by repeated squaring (needs a stable P)."""

    def body(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], None]:
        """One doubling step of the Lyapunov iteration."""
        S, M = carry
        return (M @ S @ M.T + S, M @ M), None

    (S, _), _ = jax.lax.scan(body, (Q, P), None, length=n_doublings)
    return 0.5 * (S + S.T)


def point_matrices(
    params: jax.Array, G: jax.Array, beta2: float, dt: float, stationary_init: bool
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Per-point P, sqrt(Q) and sqrt(stationary covariance), shared by all realisations."""
    # computed once per grid point, not once per realisation
    A = make_system_matrix(params[0], params[1], params[2], beta2)
    P, Q = propagator(A, G, dt)
    L = psd_sqrt(Q)
    L0 = psd_sqrt(discrete_lyapunov(P, Q)) if stationary_init else jnp.zeros((4, 4))
    return P, L, L0


def sample_trajectory(
    P: jax.Array,
    L: jax.Array,
    L0: jax.Array,
    key: jax.Array,
    tau_num: int,
    mean_init: tuple[float, ...],
    sigma_init: tuple[float, ...],
    stationary_init: bool,
    burn_in: int,
) -> jax.Array:
    """One realisation of shape (tau_num, 4) from precomputed point matrices."""
    key_init, key_noise = jax.random.split(key)

    if stationary_init:
        x0 = L0 @ jax.random.normal(key_init, (4,))
    else:
        x0 = jnp.asarray(mean_init) + jnp.asarray(sigma_init) * jax.random.normal(key_init, (4,))

    noise = jax.random.normal(key_noise, (burn_in + tau_num - 1, 4))

    def step(x: jax.Array, w: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Advance one step and keep the new state."""
        x_next = P @ x + L @ w
        return x_next, x_next

    def step_discard(x: jax.Array, w: jax.Array) -> tuple[jax.Array, None]:
        """Advance one step without stacking the state (burn-in)."""
        return P @ x + L @ w, None

    # a no-op when burn_in == 0: the carry comes straight back out as x0
    x0, _ = jax.lax.scan(step_discard, x0, noise[:burn_in])
    _, xs = jax.lax.scan(step, x0, noise[burn_in:])
    return jnp.concatenate([x0[None, :], xs], axis=0)


@partial(jax.jit, static_argnums=(2, 3, 4, 6, 7, 8))
def sample_block(
    params: jax.Array,
    keys: jax.Array,
    tau_init: tuple[float, float, int],
    mean_init: tuple[float, ...],
    sigma_init: tuple[float, ...],
    G: jax.Array,
    beta2: float,
    stationary_init: bool,
    burn_in: int,
) -> jax.Array:
    """Realisations for one block of keys, shape (n_keys, n_beta, n_K, n_mu, tau_num, 4)."""
    tau_start, tau_end, tau_num = tau_init
    dt = (tau_end - tau_start) / (tau_num - 1)

    def mats(p: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Point matrices for one (beta, K, mu)."""
        return point_matrices(p, G, beta2, dt, stationary_init)

    P, L, L0 = jax.vmap(jax.vmap(jax.vmap(mats)))(params)

    def one(P_: jax.Array, L_: jax.Array, L0_: jax.Array, k: jax.Array) -> jax.Array:
        """One trajectory at one point."""
        return sample_trajectory(P_, L_, L0_, k, tau_num, mean_init, sigma_init, stationary_init, burn_in)

    f = jax.vmap(one, in_axes=(0, 0, 0, 0))  # over n_mu
    f = jax.vmap(f, in_axes=(0, 0, 0, 0))  # over n_K
    f = jax.vmap(f, in_axes=(0, 0, 0, 0))  # over n_beta
    f = jax.vmap(f, in_axes=(None, None, None, 0))  # over realisations
    return f(P, L, L0, keys)


def sample_oscillator_tensor(
    n_realisations: int,
    params: jax.Array,
    tau_init: tuple[float, float, int],
    mean_init: tuple[float, ...],
    sigma_init: tuple[float, ...],
    key: jax.Array,
    G: jax.Array | None = None,
    beta2: float = 0.0,
    stationary_init: bool = False,
    burn_in: int = 0,
    realisation_block: int = 8,
) -> jax.Array:
    """Realisations over an explicit params tensor, shape (R, n_beta, n_K, n_mu, tau_num, 4)."""
    if stationary_init and beta2 <= 0.0:
        raise ValueError(
            "stationary_init requires beta2 > 0: with an undamped driving "
            "oscillator the system is only marginally stable and has no "
            "stationary distribution. Use burn_in, or the time-integral "
            "ground truth, instead."
        )
    G_eff = jnp.zeros((4, 4)) if G is None else jnp.asarray(G)
    keys = jax.random.split(key, (n_realisations, *params.shape[:-1]))
    args = (
        tuple(tau_init),
        tuple(float(m) for m in mean_init),
        tuple(float(s) for s in sigma_init),
        G_eff,
        float(beta2),
        bool(stationary_init),
        int(burn_in),
    )
    block = max(1, int(realisation_block))
    if n_realisations <= block:
        return sample_block(params, keys, *args)
    parts = [np.asarray(sample_block(params, keys[i : i + block], *args)) for i in range(0, n_realisations, block)]
    return jnp.asarray(np.concatenate(parts, axis=0))


def to_estimator_layout(samples: jax.Array) -> jax.Array:
    """Reshape (R, b, K, m, T, V) samples to the estimator layout (b*K*m, R, V, T)."""
    X = jnp.transpose(samples, (1, 2, 3, 0, 5, 4))
    n_b, n_K, n_mu, R, V, T = X.shape
    return X.reshape(n_b * n_K * n_mu, R, V, T)


def sample_grid(grid: ParameterGrid, sim: SimulationConfig) -> np.ndarray:
    """All realisations at every grid point, shape (n_grid, R, 4, n_time), in sim.sample_dtype."""
    if sim.stationary and sim.beta2 <= 0.0:
        raise ValueError("stationary sampling needs beta2 > 0: an undamped driver has no stationary law")
    params = params_tensor(grid)
    R = sim.n_realisations
    # every key is split up front and sliced per block, so the draws do not
    # depend on the block size; only peak memory does
    keys = jax.random.split(jax.random.key(sim.seed), (R, *params.shape[:-1]))
    G = jnp.asarray(noise_covariance(sim.q1, sim.q2))
    static = (
        tuple(sim.tau),
        tuple(float(m) for m in sim.mean_init),
        tuple(float(s) for s in sim.sigma_init),
    )
    parts = []
    for i in range(0, R, sim.realisation_block):
        block = sample_block(
            params,
            keys[i : i + sim.realisation_block],
            *static,
            G,
            float(sim.beta2),
            bool(sim.stationary),
            int(sim.burn_in),
        )
        parts.append(np.asarray(to_estimator_layout(block), dtype=sim.sample_dtype))
        del block
    return np.concatenate(parts, axis=1)
