import sympy as sp
# import jax
# import jax.numpy as jnp
from typing import Tuple, Callable

from algorithms.tc import _symbolic_information_flow

# Will have a function to build lambdified function of beta, K, mu and tau. 
# tau_0 and covariances will remain fixed. 
# Integration along tau

def _build_lambda_fn(
        tau_init: Tuple[float, float, int], sigma_init: Tuple[float]
    )->Callable:
    """
    Inputs:
    1. tau_init: consists of (tau_start, tau_end, N)
    2. sigma_inti: consists of s11, s22, s33, s44

    Outputs
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
    
    free_symbols = (beta, K, mu)
    lambda_information_flow = [
        [sp.lambdify(free_symbols, information_flow[i, j], modules='jax') for j in range(information_flow.cols)]
        for i in range(information_flow.rows)
    ]

    return lambda_information_flow
    
