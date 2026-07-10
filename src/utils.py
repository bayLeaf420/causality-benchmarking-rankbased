import jax.numpy as jnp
import jax
import sympy as sp
from typing import Dict, Tuple, Callable

from algorithms.tc import _symbolic_information_flow, _normalised_information_flow

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
) -> Tuple[Tuple[Tuple[Callable], ...], ...]:
    """ Builds python callables from symbolic expressions of LKIF.

    Args:
        tau_init: consists of (tau_start, tau_end, N)
        sigma_init: consists of s11, s22, s33, s44

    Returns:
        lambda_information_flow consists of all T(i->j) as lambda functions
    """
    information_flow, _, _, symbols, _ = _symbolic_information_flow()
    norm_info_flow = _normalised_information_flow()
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

    norm_info_flow = norm_info_flow.subs(
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
            sp.lambdify(req_symbols, information_flow[i, j], modules="jax")
            for j in range(information_flow.cols)
        )
        for i in range(information_flow.rows)
    )
    lambda_norm_info = tuple(
        tuple(
            sp.lambdify(req_symbols, norm_info_flow[i, j], modules="jax")
            for j in range(norm_info_flow.cols)
        )
        for i in range(norm_info_flow.rows)
    )

    return lambda_information_flow, lambda_norm_info
