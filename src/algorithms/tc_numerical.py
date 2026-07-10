# type: ignore
import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable
from quadax import quadgk
from utils import _build_lambda_fns

# First vmap is (beta_n, K_n, mu_n, 3) -> (K_n, mu_n, 3)
# Second is (K_n, mu_n, 3) -> (mu_n, 3)
# Third is (mu_n, 3) -> (3,)
def _single_time_integrals(
    params_tensor: jax.Array,
    tau_init: Tuple[float, float, int],
    lambda_information_flow: List[List[Callable]],
    lambda_norm_info: List[List[Callable]],
) -> tuple[jax.Array, jax.Array]:
    """Returns numeric time integrals of LKIF and normalised LKIF. 

    Args:
        params_tensor: Shape [beta, K, mu, 3]
        tau_init: tuple of (tau_start, tau_end, tau_nums)
        lambda_information_flow: Information flow matrix (List of List) of lambda functions.Build in
                                implementation script
    Returns:
        integrals_tensor: Matrix of integrals, shape is [beta, K, mu]
        norm_integral_tensor: Matrix of integrals of normalised LKIF, shape is [beta, K, mu]
    """
    beta = params_tensor[0]
    K = params_tensor[1]
    mu = params_tensor[2]

    tau_start, tau_end = tau_init[0], tau_init[1]

    # substituted_lambda_fns = [
    #    [fn(tau_array, beta, K, mu) for fn in row]
    #     for row in lambda_information_flow
    # ]
    time_interval = jnp.asarray([tau_start, tau_end])
    rows1 = [
        jnp.stack([quadgk(fn, time_interval, (beta, K, mu))[0] for fn in row])
        for row in lambda_information_flow
    ]
    integrals_tensor = jnp.stack(rows1)

    rows2 = [
        jnp.stack([quadgk(fn, time_interval, (beta, K, mu))[0] for fn in row])
        for row in lambda_norm_info
    ]
    norm_integral_tensor = jnp.stack(rows2)

    return integrals_tensor, norm_integral_tensor


def time_integrals(
    params_tensor: jax.Array,
    tau_init: Tuple[float, float, int],
    sigma_init: Tuple[float, ...], # 4
) -> tuple[jax.Array, jax.Array]:
    """Wrapper to call vmapped version. 
    """
    get_time_integrals = jax.vmap(
        jax.vmap(
            jax.vmap(
                _single_time_integrals,
                in_axes=(0, None, None, None),
            ),
            in_axes=(0, None, None, None),
        ),
        in_axes=(0, None, None, None),
    )
    get_time_integrals = jax.jit(get_time_integrals, static_argnums=(1, 2, 3))
    lambda_lkif, lambda_n_lkif = _build_lambda_fns(tau_init, sigma_init)
    return get_time_integrals(params_tensor, tau_init, lambda_lkif, lambda_n_lkif)