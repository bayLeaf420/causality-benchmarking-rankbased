"""System matrix and noise covariance of the de-dimensionalised coupled oscillators."""

import jax
import jax.numpy as jnp

# State order used everywhere: (x1, v1, x2, v2).
X1, V1, X2, V2 = 0, 1, 2, 3

# x2 enters v1's equation, so T(x2 -> v1) is the one informative LKIF entry.
TRUTH_ENTRY: tuple[int, int] = (X2, V1)
# The reverse direction is identically zero: a built-in false-positive null.
NULL_ENTRY: tuple[int, int] = (V1, X2)


def make_system_matrix(
    beta: float | jax.Array, K: float | jax.Array, mu: float | jax.Array, beta2: float | jax.Array = 0.0
) -> jax.Array:
    """Drift matrix A of the driven oscillator (omega_0 = 1) and its driver (frequency K)."""
    return jnp.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, -beta, -mu * K**2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -(K**2), -beta2],
        ]
    )


def noise_covariance(q1: float = 0.0, q2: float = 0.0) -> jax.Array:
    """Diffusion matrix G = B B^T with white noise on the two velocities only."""
    # a position equation is the kinematic identity dx = v dt and carries no force
    return jnp.diag(jnp.asarray([0.0, q1, 0.0, q2]))
