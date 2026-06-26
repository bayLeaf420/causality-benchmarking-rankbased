import sympy as sp
import jax
import jax.numpy as jnp
from typing import Tuple, Callable
import pickle
from pathlib import Path
# import jax.numpy as jnp

# This library tells theoretical causality between coupled oscillators
# Note: tau_vec will always be a jnp.array. 
# IN experiments script only call build_time_integral_fn and then the tc_eval function.

CACHE_PATH_SIF = Path("results/symbolic_flow_cache.pkl")
CACHE_PATH_NSIF = Path("results/normalised_flow_cache.pkl")

def _symbolic_information_flow()-> Tuple[sp.Matrix, sp.Matrix, sp.Matrix, Tuple, Tuple]:
    """Inputs: None
    """
    # Expensive to compute -- just cache output 1st time
    if CACHE_PATH_SIF.exists():
        with open(CACHE_PATH_SIF, "rb") as f:
            return pickle.load(f)
        
    # breakpoint()
    # Define time symbols (real and positive so that sympy can simplify the math)
    tau, tau_0 = sp.symbols('tau tau_0', real=True)
    # Define oscillator parameters, assuming omega_0 = 1 for the expression itself.
    beta, K, mu = sp.symbols('beta K mu', real=True)
    # Covariances = To be used for calculating flow
    s11, s22, s33, s44 = sp.symbols('sigma11 sigma22 sigma33 sigma44', real=True)
    # Means: Used later in sampling
    mean1, mean2, mean3, mean4 = sp.symbols('mean1 mean2 mean3 mean4', real=True)

    # Define system matrix A
    A = sp.Matrix([
        [0, 1, 0, 0],
        [-1, -beta, -mu*(K**2), 0],
        [0, 0, 0, 1],
        [0, 0, -K**2, 0],
    ])

    # Define initial mean vector
    mean_0 = sp.Matrix([mean1, mean2, mean3, mean4])

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
    Sigma_t = Sigma_t.applyfunc(lambda e: sp.re(sp.expand_complex(e)))

    # Sigma_t = Sigma_t.applyfunc(lambda e: sp.expand_complex(e).simplify())

    # mean_t is mean vector as a function of time
    mean_t = exp_At * mean_0
    mean_t = mean_t.applyfunc(lambda e: sp.re(sp.expand_complex(e)))

    information_flow = sp.Matrix(A.rows, A.cols, lambda i, j: A[j, i] * (Sigma_t[j, i] / Sigma_t[j, j]))

    symbols = (tau, tau_0, beta, K, mu, s11, s22, s33, s44)
    mean_symbols = (mean1, mean2, mean3, mean4)
    result = information_flow, Sigma_t, mean_t, symbols, mean_symbols 

    CACHE_PATH_SIF.parent.mkdir(exist_ok=True)
    with open(CACHE_PATH_SIF, "wb") as f:
        pickle.dump(result, f)
    return result

def _normalised_information_flow():

    if CACHE_PATH_NSIF.exists():
        with open(CACHE_PATH_NSIF, "rb") as f:
            return pickle.loaf(f)

    information_flow, sigma_t, mean_t, symbols, mean_symbols = _symbolic_information_flow()
    (tau, tau_0, beta, K, mu, s11, s22, s33, s44) = symbols 
    A = sp.Matrix([
        [0, 1, 0, 0],
        [-1, -beta, -mu*(K**2), 0],
        [0, 0, 0, 1],
        [0, 0, -K**2, 0],
    ])

    normalised_information_flow = sp.Matrix(
        information_flow.rows,
        information_flow.cols,
        lambda i, j: 0 if A[i, j] == 0 else (
            sp.Abs(information_flow[i, j]/(sp.Abs(A[j, j]) + sp.Abs(sum(information_flow[:, j]))))
        )
    )

    with open(CACHE_PATH_NSIF, "wb") as f:
        pickle.dump(normalised_information_flow, f)

    return normalised_information_flow


def build_time_integral_fn(tau_init: jax.Array, sigma_init: Tuple[float]) -> sp.Matrix:
    """
    Inputs:
    1. tau_init: list, contains tau_start, tau_end, tau_num
    2. sigma_vec: representing the diagonal of covar matrix. Denotes (s11, s22, s33, s44). Tuple of 4 floats.
    Returns time integral of information_flow matrix

    Outputs:
    JAX compatible lambda function for definite integral from tau_start to tau_end
    """
    # breakpoint()
    information_flow, _, _, symbols, _ = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols
    tau_start, tau_end = tau_init[0], tau_init[1]
    # Fix tau_0 and initial variances to numeric value
    information_flow = information_flow.subs([
        (tau_0, float(tau_start)), 
        (s11, float(sigma_init[0])),
        (s22, float(sigma_init[1])),
        (s33, float(sigma_init[2])),
        (s44, float(sigma_init[3])),
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
        [sp.lambdify(free_symbols, integrated[i, j], modules='jax', cse=True) for j in range(integrated.cols)]
        for i in range(integrated.rows)
    ]

    return lambda_info_integral


def _tc_integral_inner(params_tensor: jax.Array, time_integral_fn_list: Callable) -> jax.Array:
    """Inputs:
    1. params_tensor: consists of axes (beta, mu, K). We vmap over this tensor. Shape is (beta_num, K_num, mu_num, 3) 
    2. time_integral_fn_list: Built in experiment script using _build_time_integral_fn. Takes in (beta, K, mu)_val
    
    Outputs:
    1. Causal matrix of shape (4, 4), denotes adjacency matrix of weighted causal graph.

    The function calculates time integral of each T(j->i) expression outputted by symbolic_information_flow() 
    """
    beta_val = params_tensor[0]
    K_val = params_tensor[1]
    mu_val = params_tensor[2]

    rows = [
        jnp.stack([fn(beta_val, K_val, mu_val) for fn in row])
        for row in time_integral_fn_list
    ]
    return jnp.stack(rows)
    
tc_integral = jax.vmap(
    jax.vmap(
        jax.vmap(
            _tc_integral_inner,
            in_axes=(0, None),
        ),
        in_axes=(1, None),
    ),
    in_axes=(2, None),
),
    


"""
function tc_integral:
Inputs:
1. params_tensor
2. time_integral_fn

Outputs:
1. time_integral_tensor
"""


