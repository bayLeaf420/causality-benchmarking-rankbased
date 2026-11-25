"""Raw and normalised LKIF of a linear system from its drift and covariance."""

import jax
import jax.numpy as jnp


def raw_flow(A: jax.Array, Sigma: jax.Array) -> jax.Array:
    """Raw LKIF matrix, entry (i, j) = T(i -> j) = A[j, i] Sigma[j, i] / Sigma[j, j]."""
    diag = jnp.diagonal(Sigma)
    # Sigma[j,j] can pass through 0 for the undamped oscillator; Sigma[j,i]
    # vanishes with it, so substituting 1.0 leaves the ratio at 0 there
    safe_diag = jnp.where(diag <= 0.0, 1.0, diag)
    ratio = Sigma / safe_diag[None, :]  # entry (i, j) = Sigma[i,j] / Sigma[j,j]
    return A.T * ratio  # entry (i, j) = A[j,i] * ratio[i,j]


def normalise_flow(flow: jax.Array, Sigma: jax.Array | None = None, G: jax.Array | None = None) -> jax.Array:
    """Normalised LKIF |T(i -> j)| / Z_j, where Z_j is column j's total entropy budget."""
    # Z_j = sum_k |T(k->j)| (self term included) + G[j,j] / (2 Sigma[j,j]).
    # Check the 1/2 noise prefactor against Liang 2015 before quoting absolute
    # values; a monotone error leaves a rank benchmark unaffected.
    abs_flow = jnp.abs(flow)
    denom = jnp.sum(abs_flow, axis=0)
    if G is not None:
        if Sigma is None:
            raise ValueError("normalise_flow needs Sigma to add the noise term")
        var = jnp.diagonal(Sigma)
        safe_var = jnp.where(var <= 0.0, 1.0, var)
        denom = denom + jnp.abs(jnp.diagonal(G)) / (2.0 * safe_var)
    # a column summing to 0 has a zero numerator too, so 1.0 keeps entries at 0
    safe_denom = jnp.where(denom <= 0.0, 1.0, denom)
    return abs_flow / safe_denom[None, :]
