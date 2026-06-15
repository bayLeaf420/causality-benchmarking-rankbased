import sympy as sp
import jax
import jax.numpy as jnp
from typing import List, Tuple, Callable
from quadax import quadgk
from functools import partial

from algorithms.tc import _symbolic_information_flow

# Will have a function to build lambdified function of beta, K, mu and tau. 
# tau_0 and covariances will remain fixed. 
# Integration along tau

def _build_lambda_fns(
        tau_init: Tuple[float, float, int], sigma_init: Tuple[float]
    )->List[List[Callable]]:
    """
    Inputs:
    1. tau_init: consists of (tau_start, tau_end, N)
    2. sigma_init: consists of s11, s22, s33, s44

    Outputs:
    1. lambda_information_flow consists of all T(i->j) as lambda functions
    """
    information_flow, _, _, symbols, _ = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols

    # Fix tau_0 and initial variances to numeric value
    information_flow = information_flow.subs([
        (tau_0, float(tau_init[0])), 
        (s11, float(sigma_init[0])),
        (s22, float(sigma_init[1])),
        (s33, float(sigma_init[2])),
        (s44, float(sigma_init[3])),
        ])
    
    req_symbols = (tau, beta, K, mu)
    lambda_information_flow = [
        [sp.lambdify(req_symbols, information_flow[i, j], modules='jax') for j in range(information_flow.cols)]
        for i in range(information_flow.rows)
    ]

    return lambda_information_flow
    
# First vmap is (beta_n, K_n, mu_n, 3) -> (K_n, mu_n, 3)
# Second is (K_n, mu_n, 3) -> (mu_n, 3)
# Third is (mu_n, 3) -> (3,)
@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(0, None, None))
def time_integrals(
        params_tensor: jax.Array,
        tau_init: Tuple[float, float, int],
        lambda_information_flow: List[List[Callable]]

) -> jax.Array:
    """
    Inputs:
    1. params_tensor.
    2. tau_init: tuple of (tau_start, tau_end, tau_nums) 
    3. lambda_information_flow: Information flow matrix (List of List) of lambda functions.Build in
                                implementation script
    Outputs:
    1. integrals_tensor: matrix of integrals. Same shape as params_tensor

    Points:
    Input into any element of lambda_information_flow is of form (tau, beta, K, mu). 
    """
    beta = params_tensor[0]
    K = params_tensor[1]
    mu = params_tensor[2]

    tau_start, tau_end = tau_init[0], tau_init[1]

    # substituted_lambda_fns = [
    #    [fn(tau_array, beta, K, mu) for fn in row]
    #     for row in lambda_information_flow
    # ]

    rows = [
        jnp.stack([quadgk(fn, [tau_start, tau_end], (beta, K, mu)) for fn in row])
        for row in lambda_information_flow
    ]
    integrals_tensor = jnp.stack(rows)

    return integrals_tensor