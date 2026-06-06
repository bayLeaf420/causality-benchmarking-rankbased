#!/usr/bin/env python3
"""Compute Liang–Kleeman information flow on all parameter grid combinations, batched with JAX."""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import sympy as sp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theoretical_causality.sym_information import symbolic_information_flow
from runs.TC_run_functions import create_params_grid


def main():
    # Parameter ranges
    k_dict      = {"start": 0.1, "stop": 2.0, "num": 5}
    b_dict      = {"start": 0.1, "stop": 2.0, "num": 5}
    m_dict      = {"start": 0.5, "stop": 1.5, "num": 3}
    omega_dict  = {"start": 0.5, "stop": 2.5, "num": 4}

    # Initial covariance diagonals
    var = [1.0, 1.0, 1.0, 1.0]

    # Time domain
    time_dict = {"start": 0.0, "stop": 10.0, "num": 1000}

    # Direction
    direction = "3->2"

    # Build parameter grid (k, b, m, omega)
    param_tensor = create_params_grid(k_dict, b_dict, m_dict, omega_dict)
    N_k, N_b, N_m, N_omega, _ = param_tensor.shape
    print("Parameter tensor shape (k, b, m, omega):", param_tensor.shape)

    # Time vector
    t_start = time_dict["start"]
    t_vec = jnp.linspace(t_start, time_dict["stop"], time_dict["num"])

    # Load symbolic expression and lock t0
    flow_dict, _ = symbolic_information_flow()
    t0_sym = sp.symbols('t0', real=True)
    expr = flow_dict[direction].subs(t0_sym, t_start)

    # Lambdify to JAX
    t_sym, k_sym, b_sym, m_sym, omega_sym, s11_sym, s22_sym, s33_sym, s44_sym = \
        sp.symbols('t k b m omega sigma11 sigma22 sigma33 sigma44', real=True)
    j_func = sp.lambdify(
        (t_sym, k_sym, b_sym, m_sym, omega_sym, s11_sym, s22_sym, s33_sym, s44_sym),
        expr, modules='jax'
    )

    s11_v, s22_v, s33_v, s44_v = var

    # JIT‑compiled evaluator for a single parameter tuple
    @jax.jit
    def evaluate_one(p_vec):
        k_val, b_val, m_val, omega_val = p_vec
        return jnp.atleast_1d(j_func(t_vec, k_val, b_val, m_val, omega_val,
                                     s11_v, s22_v, s33_v, s44_v))

    # Flatten grid and vectorise
    flat_params = jnp.array(param_tensor.reshape(-1, 4))
    flow_flat = jax.vmap(evaluate_one, in_axes=(0,))(flat_params)

    # Restore grid shape
    flow = flow_flat.reshape(N_k, N_b, N_m, N_omega, len(t_vec))

    print("Information‑flow array shape (params × time):", flow.shape)
    print("Time‑points array shape:", (len(t_vec),))

    # Save
    np.savez("liang_kleeman_results.npz",
             flow=np.asarray(flow),
             time=np.asarray(t_vec))
    print("Saved -> liang_kleeman_results.npz")


if __name__ == "__main__":
    main()
