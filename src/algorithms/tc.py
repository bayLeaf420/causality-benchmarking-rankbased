import sympy as sp
import jax
from typing import Tuple
# import jax.numpy as jnp

# This library tells theoretical causality between coupled oscillators
# Note: tau_vec will always be a jnp.array. 
# IN experiments script only call build_time_integral_fn and then the tc_eval function.

def _symbolic_information_flow()-> Tuple[sp.Matrix, sp.Matrix, ]:
    """Inputs: None


    """
    # Define time symbols (real and positive so that sympy can simplify the math)
    tau, tau_0 = sp.symbols('tau tau_0', real=True, positive=True)
    # Define oscillator parameters, assuming omega_0 = 1 for the expression itself.
    beta, K, mu = sp.symbols('beta K mu', real=True, positive=True)
    # Covariances = To be used for calculating flow
    s11, s22, s33, s44 = sp.symbols('sigma11 sigma22 sigma33 sigma44', real=True)

    #Define system matrix A
    A = sp.Matrix([
        [0, 1, 0, 0],
        [-1, -beta, mu*(K**2), 0],
        [0, 0, 0, 1],
        [0, 0, -K**2, 0],
    ])

    # Define initial covariance matrix
    Sigma_0 = sp.Matrix([
        [s11**2, 0, 0, 0],
        [0, s22**2, 0, 0],
        [0, 0, s33**2, 0],
        [0, 0, 0, s44**2],
    ])

    exp_At = (A * (tau - tau_0)).exp()

    # Sigma_t is covariance matrix as a function of time
    Sigma_t = exp_At * Sigma_0 * exp_At.T

    information_flow = sp.Matrix(A.rows, A.cols, lambda i, j: A[j, i] * (Sigma_t[j, i] / Sigma_t[j, j]))

    return information_flow, Sigma_t, (tau, tau_0, beta, K, mu, s11, s22, s33, s44)



def build_time_integral_fn(tau_start: float, tau_end: float, sigma_vec: jax.Array) -> sp.Matrix:
    """
    Inputs:
    1. tau_vec: list, contains tau_start, tau_end, tau_num
    2. sigma_vec: representing the diagonal of covar matrix. Denotes [s11, s22, s33, s44]
    Returns time integral of information_flow matrix

    Outputs:
    JAX compatible lambda function for definite integral from tau_start to tau_end
    """
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
    
    # Simplify each entry before integrating to help SymPy do its job
    information_flow = information_flow.applyfunc(sp.cancel)

    # Integrate the function to obtain definite time integral expression
    integrated = information_flow.applyfunc(
        lambda expr: sp.integrate(expr, (tau, float(tau_start), float(tau_end)))
    ) 

    # Warn about unevaluated entries
    unevaluated = [
        (i, j) for i in range(integrated.rows)
                for j in range(integrated.cols)
                if integrated[i, j].has(sp.Integral)
    ]
    if unevaluated:
        print(f"WARNING: unevaluated integrals at positions {unevaluated}. "
              "Consider additional simplification or numerical fallback.")

    free_symbols = (beta, K, mu)
    lambda_info_integral = [
        [sp.lambdify(free_symbols, integrated[i, j], modules='jax') for j in range(integrated.cols)]
        for i in range(integrated.rows)
    ]

    return lambda_info_integral


def _tc_eval_inner(params_tensor: jax.Array, time_integral_fn) -> jax.Array:
    """Inputs:
    1. params_tensor: consists of axes (beta, mu, K). We vmap over this tensor. Shape is (beta_num, K_num, mu_num, 3) 
    2. time_integral_fn: Built in experiment script using _build_time_integral_fn. Takes in (beta, K, mu)_val
    
    Outputs:
    1. Causal matrix of shape (4, 4), denotes adjacency matrix of weighted causal graph.

    The function calculates time integral of each T(j->i) expression outputted by symbolic_information_flow() 
    """
    beta_val = params_tensor[0]
    K_val = params_tensor[1]
    mu_val = params_tensor[2]

    return time_integral_fn(beta_val, K_val, mu_val)
    
tc_eval = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(
                _tc_eval_inner,
                in_axes=(0, None),
            ),
            in_axes=(1, None),
        ),
        in_axes=(2, None),
    ),
    static_argnums=(1,)
)



