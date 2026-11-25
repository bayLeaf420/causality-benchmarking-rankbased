"""Time-averaged normalised LKIF of the transient process, by integrating the moment ODE."""

from typing import Any

import diffrax
import jax
import jax.numpy as jnp

from lkif_rank_causal_bench.lkif.flow import normalise_flow, raw_flow
from lkif_rank_causal_bench.oscillator.model import make_system_matrix

_FLOAT32_EPS = 1.2e-7


def check_precision(rtol: float, atol: float) -> None:
    """Raise if the solver tolerances cannot be met in the active JAX precision."""
    # in float32 a tolerance below eps makes the step controller shrink the step
    # until max_steps, and diffrax only reports "maximum steps reached"
    if jax.config.read("jax_enable_x64"):
        return
    if min(rtol, atol) < _FLOAT32_EPS:
        raise RuntimeError(
            f"rtol={rtol:g}, atol={atol:g} are below float32 epsilon "
            f"({_FLOAT32_EPS:g}) but JAX is running in single precision, so "
            "the solver cannot converge and will exhaust max_steps. Call "
            "jax.config.update('jax_enable_x64', True) before creating any "
            "array, or pass looser tolerances."
        )


def vector_field(
    t: Any, y: tuple[jax.Array, jax.Array, jax.Array], args: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Right-hand side for (Sigma, integral of normalised flow, integral of raw flow)."""
    Sigma, _, _ = y
    A, G = args
    dSigma = A @ Sigma + Sigma @ A.T + G
    flow = raw_flow(A, Sigma)
    # the raw integral is kept so trends can be compared without normalisation
    return (dSigma, normalise_flow(flow, Sigma, G), flow)


def integrate_point(
    params: jax.Array,
    tau_init: tuple[float, float, int],
    sigma_init: tuple[float, float, float, float],
    G: jax.Array | None = None,
    beta2: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
    raw: bool = False,
) -> jax.Array:
    """Time integral of the normalised (or raw) LKIF matrix at one (beta, K, mu)."""
    beta, K, mu = params[0], params[1], params[2]
    A = make_system_matrix(beta, K, mu, beta2)
    G_eff = jnp.zeros((4, 4)) if G is None else G

    tau_start, tau_end = tau_init[0], tau_init[1]
    Sigma0 = jnp.diag(jnp.asarray(sigma_init) ** 2)  # sigma_init holds standard deviations
    integral0 = jnp.zeros((4, 4))
    y0 = (Sigma0, integral0, integral0)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=tau_start,
        t1=tau_end,
        dt0=(tau_end - tau_start) / 100.0 if tau_end > tau_start else None,
        y0=y0,
        args=(A, G_eff),
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    assert sol.ys is not None
    _, normalised, raw_integral = sol.ys
    chosen = raw_integral if raw else normalised
    return chosen[-1]  # strip the length-1 time axis


def lkif_integral_grid(
    params: jax.Array,
    tau_init: tuple[float, float, int],
    sigma_init: tuple[float, float, float, float],
    G: jax.Array | None = None,
    beta2: float = 0.0,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 100_000,
    raw: bool = False,
) -> jax.Array:
    """Time-integrated LKIF over a params tensor, shape (n_beta, n_K, n_mu, 4, 4)."""
    check_precision(rtol, atol)

    def solve(p: jax.Array) -> jax.Array:
        """Integral at one point."""
        return integrate_point(p, tau_init, sigma_init, G, beta2, rtol, atol, max_steps, raw)

    return jax.jit(jax.vmap(jax.vmap(jax.vmap(solve))))(params)


def time_average(integral: jax.Array, tau_init: tuple[float, float, int]) -> jax.Array:
    """Turn a time integral of the flow into a time average over the window."""
    return integral / (tau_init[1] - tau_init[0])
