"""
lkif_ode.py — normalised Liang–Kleeman information flow via ODE integration.

Replaces the symbolic matrix-exponential pipeline with a diffrax solve of the
moment ODEs, plus an augmented state that accumulates the time-integral of the
normalised LKIF. No SymPy, no caching, GPU/vmap/grad friendly.
"""
from typing import Tuple

import jax
import jax.numpy as jnp
import diffrax

# ---------------------------------------------------------------- system

def make_system_matrix(beta, K, mu) -> jax.Array:
    """Builds the 4x4 system matrix A of the de-dimensionalised oscillator.

    Args:
        beta: damping parameter.
        K: frequency ratio of the two oscillators.
        mu: coupling strength (omega_0 = 1).

    Returns:
        A: system matrix of shape (4, 4).
    """
    return jnp.array([
        [0.0,   1.0,        0.0,        0.0],
        [-1.0, -beta,      -mu * K**2,  0.0],
        [0.0,   0.0,        0.0,        1.0],
        [0.0,   0.0,       -K**2,       0.0],
    ])


# ---------------------------------------------------------------- flow

def raw_flow(A: jax.Array, Sigma: jax.Array) -> jax.Array:
    """Computes the raw LKIF matrix, T(i->j) = A[j,i] * Sigma[j,i] / Sigma[j,j].

    Args:
        A: system matrix, shape (4, 4).
        Sigma: covariance matrix at the current tau, shape (4, 4).

    Returns:
        flow: matrix of raw T(i->j), shape (4, 4).
    """
    diag = jnp.diagonal(Sigma)                       # Sigma[j,j], indexed by column j
    # Guard: variance Sigma[j,j] can pass through 0 for the undamped oscillator.
    # Sub 1.0; numerator Sigma[j,i] vanishes alongside it so the ratio is 0 there.
    safe_diag = jnp.where(diag == 0.0, 1.0, diag)
    # Sigma symmetric => Sigma[j,i] = Sigma[i,j]; broadcast 1/Sigma[j,j] over cols j.
    ratio = Sigma / safe_diag[None, :]               # entry (i, j) = Sigma[i,j]/Sigma[j,j]
    return A.T * ratio                               # entry (i, j) = A[j,i] * ratio[i,j]


def normalise_flow(flow: jax.Array) -> jax.Array:
    """Computes normalised LKIF, |flow[i,j]| divided by the column-j sum of |flow|.

    Note: the self term |A[j,j]| equals flow[j,j] for this linear system, so the
    Liang normaliser is exactly the full column sum of absolute flows.

    Args:
        flow: raw LKIF matrix, shape (4, 4).

    Returns:
        normalised_flow: matrix of normalised LKIF, shape (4, 4).
    """
    abs_flow = jnp.abs(flow)
    denom = jnp.sum(abs_flow, axis=0)                # column sums, indexed by j
    # Guard: a column summing to 0 (no self-dynamics, no inflow now) gives 0/0.
    # Sub 1.0; numerator |flow[i,j]| is 0 in that column anyway so entries stay 0.
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    return abs_flow / safe_denom[None, :]


# ---------------------------------------------------------------- ODE

def _vector_field(t, y, args):
    """Augmented ODE right-hand side for the moment and integral state.

    Args:
        t: current time tau (unused; the system is autonomous).
        y: state tuple (Sigma, running_integral), each of shape (4, 4).
        args: system matrix A, shape (4, 4).

    Returns:
        dy: tuple (dSigma, d_integral), each of shape (4, 4).
    """
    Sigma, _ = y
    A = args
    dSigma = A @ Sigma + Sigma @ A.T
    d_integral = normalise_flow(raw_flow(A, Sigma))  # integrate the *normalised* flow
    return (dSigma, d_integral)


def _integrate_single(
    params: jax.Array,
    tau_init: Tuple[float, float, int],
    sigma_init: Tuple[float, float, float, float],
    rtol: float,
    atol: float,
    max_steps: int,
) -> jax.Array:
    """Integrates the normalised LKIF over one parameter point.

    Args:
        params: single point (beta, K, mu), shape (3,).
        tau_init: contains (tau_start, tau_end, tau_num); tau_num unused here.
        sigma_init: diagonal of the initial covariance, (s11, s22, s33, s44).
        rtol: relative solver tolerance.
        atol: absolute solver tolerance.
        max_steps: solver step cap.

    Returns:
        integral: time-integrated normalised LKIF, shape (4, 4).
    """
    beta, K, mu = params[0], params[1], params[2]
    A = make_system_matrix(beta, K, mu)

    tau_start, tau_end = tau_init[0], tau_init[1]
    Sigma0 = jnp.diag(jnp.asarray(sigma_init) ** 2)  # initial covariance (variances on diag)
    integral0 = jnp.zeros((4, 4))
    y0 = (Sigma0, integral0)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_vector_field),
        diffrax.Tsit5(),                             # adaptive 5th-order, good for smooth linear ODEs
        t0=tau_start,
        t1=tau_end,
        dt0=(tau_end - tau_start) / 100.0,           # initial step; controller adapts from here
        y0=y0,
        args=A,
        saveat=diffrax.SaveAt(t1=True),              # only need the endpoint (the integral)
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    _, integral = sol.ys
    return integral[-1]                              # strip the length-1 time axis


# ---------------------------------------------------------------- grid driver

def normalised_lkif_grid(
    params_tensor: jax.Array,
    tau_init: Tuple[float, float, int],
    sigma_init: Tuple[float, float, float, float],
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
) -> jax.Array:
    """Computes time-integrated normalised LKIF across the whole parameter grid.

    Args:
        params_tensor: grid of shape (n_beta, n_K, n_mu, 3).
        tau_init: contains (tau_start, tau_end, tau_num).
        sigma_init: diagonal of the initial covariance, (s11, s22, s33, s44).
        rtol: relative solver tolerance.
        atol: absolute solver tolerance.
        max_steps: solver step cap.

    Returns:
        nlkif: integrated normalised LKIF, shape (n_beta, n_K, n_mu, 4, 4).
    """
    solve = lambda p: _integrate_single(p, tau_init, sigma_init, rtol, atol, max_steps)
    # vmap the three grid axes; the trailing length-3 axis is the (beta,K,mu) vector.
    solve = jax.vmap(jax.vmap(jax.vmap(solve)))
    return jax.jit(solve)(params_tensor)