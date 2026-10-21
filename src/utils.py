import jax.numpy as jnp
import jax
import sympy as sp
from typing import Dict, Tuple, Callable

from algorithms.tc import _symbolic_information_flow

def build_params_tensor_jax(
    params_dict: Dict[str, Tuple[float, float, int]],
) -> jax.Array:
    """Helper function to quickly get the params_tensor in jax array form.

    Args:
        params_dict: Dictionary where keys are 'beta', 'K', and 'mu'.
                    The values are Tuples with [start, stop, num].

    Returns:
        params_tensor: jax.Array of shape (num_beta, num_K, num_mu, 3)
    """

    beta_builder = params_dict["beta"]
    K_builder = params_dict["K"]
    mu_builder = params_dict["mu"]

    # '*' operator "unpacks" the tuple values into the function call.
    beta_arr = jnp.linspace(*beta_builder)
    K_arr = jnp.linspace(*K_builder)
    mu_arr = jnp.linspace(*mu_builder)

    params_tensor = jnp.stack(
        jnp.meshgrid(beta_arr, K_arr, mu_arr, indexing="ij"), axis=-1
    )

    return params_tensor


def _build_lambda_fns(
    tau_init: Tuple[float, float, int], sigma_init: Tuple[float, ...]
) -> Tuple[Tuple[Tuple[Callable, ...], ...], Tuple[Tuple[Callable, ...], ...]]:
    """ Builds python callables from symbolic expressions of LKIF.

    Args:
        tau_init: consists of (tau_start, tau_end, N)
        sigma_init: consists of s11, s22, s33, s44

    Returns:
        lambda_information_flow consists of all T(i->j) as lambda functions
    """
    information_flow, _, _, symbols, _ = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols

    # Fix tau_0 and initial variances to numeric value
    information_flow = information_flow.subs(
        [
            (tau_0, float(tau_init[0])),
            (s11, float(sigma_init[0])),
            (s22, float(sigma_init[1])),
            (s33, float(sigma_init[2])),
            (s44, float(sigma_init[3])),
        ]
    )

    req_symbols = (tau, beta, K, mu)
    lambda_information_flow = tuple(
        tuple(
            sp.lambdify(req_symbols, information_flow[i, j], modules="jax", cse=True)
            for j in range(information_flow.cols)
        )
        for i in range(information_flow.rows)
    )

    # Whole-matrix numeric normaliser, then split into scalar per-entry closures.
    norm_matrix_fn = build_norm_lambda(tau_init, sigma_init)

    def _make_entry(i, j):
        return lambda tau_v, beta_v, K_v, mu_v: norm_matrix_fn(
            tau_v, beta_v, K_v, mu_v
        )[i, j]

    n = information_flow.rows
    lambda_norm_info = tuple(
        tuple(_make_entry(i, j) for j in range(information_flow.cols))
        for i in range(n)
    )

    return lambda_information_flow, lambda_norm_info


def build_norm_lambda(
    tau_init: Tuple[float, float, int],
    sigma_init: Tuple[float, ...],
) -> Callable:
    """Numerically-normalised LKIF, built from the *raw* flow lambda.

    Avoids constructing the huge symbolic normalised matrix (Abs blocks
    cancellation, so lambdifying it directly is slow). Instead we lambdify the
    cheap flow matrix once and do the normalisation as array arithmetic.

    Returns a callable (tau, beta, K, mu) -> (4, 4) jax array of normalised LKIF.
    """
    information_flow, _, _, symbols, _ = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols

    subs = {
        tau_0: float(tau_init[0]),
        s11: float(sigma_init[0]), s22: float(sigma_init[1]),
        s33: float(sigma_init[2]), s44: float(sigma_init[3]),
    }
    flow_lambda = sp.lambdify(
        (tau, beta, K, mu), information_flow.subs(subs), modules="jax", cse=True
    )

    # Structural mask: T(i->j) is nonzero iff A[j, i] != 0. For this system the
    # nonzero (i, j) pairs are fixed, so we can hard-code the mask.
    mask = jnp.array([
        [0, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=bool)

    def norm_lambda(tau_v, beta_v, K_v, mu_v):
        flow = jnp.asarray(flow_lambda(tau_v, beta_v, K_v, mu_v))   # (4, 4)
        abs_flow = jnp.abs(flow)

        # |A[j, j]| self-term per column j. Only A[1,1] = -beta is nonzero.
        self_term = jnp.array([0.0, jnp.abs(beta_v), 0.0, 0.0])

        # Denominator for column j: |A[j,j]| + sum_{k != j} |flow[k, j]|.
        # sum over k of |flow[k,j]| minus the diagonal |flow[j,j]| gives k != j.
        col_offdiag = jnp.sum(abs_flow, axis=0) - jnp.diagonal(abs_flow)
        denom = self_term + col_offdiag                             # (4,)

        # Guard against 0/0 in columns with no self-term and no inflow at this
        # point (e.g. the undamped x3/x4 oscillator). Masked entries are zeroed
        # anyway; unmasked entries in a zero-denominator column are genuinely 0.
        safe_denom = jnp.where(denom == 0.0, 1.0, denom)
        normalised = abs_flow / safe_denom[None, :]                      # broadcast over cols
        return jnp.where(mask, normalised, 0.0)

    return norm_lambda