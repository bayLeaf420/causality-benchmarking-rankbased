import sympy as sp
import jax
import jax.numpy as jnp

from algorithms.tc import _symbolic_information_flow

# Will have a function to build lambdified function of beta, K, mu and tau. 
# tau_0 and covariances will remain fixed. 
# Integration along tau

def _build_lambda_fn(tau_start: float, sigma_vec: jax.Arrray):
    information_flow, _, symbols = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols

    # Fix tau_0 and initial variances to numeric value
    information_flow = information_flow.subs([
        (tau_0, float(tau_start)), 
        (s11, float(sigma_vec[0])),
        (s22, float(sigma_vec[1])),
        (s33, float(sigma_vec[2])),
        (s44, float(sigma_vec[3])),
        ])
    
    
