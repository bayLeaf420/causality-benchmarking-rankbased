import sympy as sp
from typing import Tuple
import pickle
from pathlib import Path
import os
# import jax.numpy as jnp

# This library tells theoretical causality between coupled oscillators
# Note: tau_vec will always be a jnp.array.
# IN experiments script only call build_time_integral_fn and then the tc_eval function.
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./results")).resolve()
CACHE_PATH_SIF = CACHE_DIR / "symbolic_flow_cache.pkl"
CACHE_PATH_NSIF = CACHE_DIR / "normalised_flow_cache.pkl"


def _symbolic_information_flow() -> Tuple[
    sp.Matrix, sp.Matrix, sp.Matrix, Tuple, Tuple
]:
    """Function to calculate symbolic expression for Liang Kleeman information flow 
    between each pair of variables in a de-dimensionalised damped driven oscillator 
    defined with 4 variables. 

    Args:
        None

    Returns:
        information_flow: Matrix of expressions for each T(i -> j)
        Sigma_t: Matrix of expressions for each covariance (i, j)
        mean_t: Column matrix for each mean. 
        symbols: Symbols to be used for evaluating the expression
        mean_symbols: Symbols for mean values. 
    """
    # Expensive to compute -- just cache output 1st time
    if CACHE_PATH_SIF.exists():
        with open(CACHE_PATH_SIF, "rb") as f:
            return pickle.load(f)

    # breakpoint()
    # Define time symbols (real and positive so that sympy can simplify the math)
    tau, tau_0 = sp.symbols("tau tau_0", real=True)
    # Define oscillator parameters, assuming omega_0 = 1 for the expression itself.
    beta, K, mu = sp.symbols("beta K mu", real=True)
    # Covariances = To be used for calculating flow
    s11, s22, s33, s44 = sp.symbols("sigma11 sigma22 sigma33 sigma44", real=True)
    # Means: Used later in sampling
    mean1, mean2, mean3, mean4 = sp.symbols("mean1 mean2 mean3 mean4", real=True)

    # Define system matrix A
    A = sp.Matrix(
        [
            [0, 1, 0, 0],
            [-1, -beta, -mu * (K**2), 0],
            [0, 0, 0, 1],
            [0, 0, -(K**2), 0],
        ]
    )

    # Define initial mean vector
    mean_0 = sp.Matrix([mean1, mean2, mean3, mean4])

    # Define initial covariance matrix
    Sigma_0 = sp.Matrix(
        [
            [s11**2, 0, 0, 0],
            [0, s22**2, 0, 0],
            [0, 0, s33**2, 0],
            [0, 0, 0, s44**2],
        ]
    )

    exp_At = (A * (tau - tau_0)).exp()

    # Sigma_t is covariance matrix as a function of time
    Sigma_t = exp_At * Sigma_0 * exp_At.T
    Sigma_t = Sigma_t.applyfunc(lambda e: sp.re(sp.expand_complex(e)))

    # Sigma_t = Sigma_t.applyfunc(lambda e: sp.expand_complex(e).simplify())

    # mean_t is mean vector as a function of time
    mean_t = exp_At * mean_0
    mean_t = mean_t.applyfunc(lambda e: sp.re(sp.expand_complex(e)))

    information_flow = sp.Matrix(
        A.rows, A.cols, lambda i, j: A[j, i] * (Sigma_t[j, i] / Sigma_t[j, j])
    )

    symbols = (tau, tau_0, beta, K, mu, s11, s22, s33, s44)
    mean_symbols = (mean1, mean2, mean3, mean4)
    result = information_flow, Sigma_t, mean_t, symbols, mean_symbols

    CACHE_PATH_SIF.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH_SIF, "wb") as f:
        pickle.dump(result, f)
    return result


def _normalised_information_flow():

    if CACHE_PATH_NSIF.exists():
        with open(CACHE_PATH_NSIF, "rb") as f:
            return pickle.load(f)

    information_flow, sigma_t, mean_t, symbols, mean_symbols = (
        _symbolic_information_flow()
    )
    (tau, tau_0, beta, K, mu, s11, s22, s33, s44) = symbols
    A = sp.Matrix(
        [
            [0, 1, 0, 0],
            [-1, -beta, -mu * (K**2), 0],
            [0, 0, 0, 1],
            [0, 0, -(K**2), 0],
        ]
    )
    n = information_flow.rows
    def _entry(i, j):
        if A[j, i] == 0:
            return sp.Integer(0)
        denom = sp.Abs(A[j, j]) + sum( # type: ignore
            (sp.Abs(information_flow[k, j]) for k in range(n) if k != j), # type: ignore
            sp.Integer(0),
        )
        return sp.Abs(information_flow[i, j])/denom

    normalised_information_flow = sp.Matrix(
        information_flow.rows,
        information_flow.cols,
        _entry,
    )

    CACHE_PATH_NSIF.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH_NSIF, "wb") as f:
        pickle.dump(normalised_information_flow, f)

    return normalised_information_flow


