#!/usr/bin/env python3
"""Run Liang‑Kleeman information flow for all parameter combinations
   produced by create_params_grid(), batched with JAX."""

import jax
import jax.numpy as jnp
import numpy as np

# The existing functions we need (adjust the import paths if the repo structure differs)
from theoretical_causality.eval_TC_JAX import evaluate_theoretical_causality_JAX
from runs.TC_run_functions import create_params_grid   # used to build the grid (optional)


def main():
    # =========================================================================
    # 1  Define the parameter grids (the same dicts that create_params_grid uses)
    #    Adjust the ranges and resolutions to your needs.
    # =========================================================================
    k_dict      = {"start": 0.1, "stop": 2.0, "num": 5}   # spring constant
    b_dict      = {"start": 0.1, "stop": 2.0, "num": 5}   # damping
    m_dict      = {"start": 0.5, "stop": 1.5, "num": 3}   # mass
    omega_dict  = {"start": 0.5, "stop": 2.5, "num": 4}   # driving frequency

    # =========================================================================
    # 2  Initial covariances [var_{11}, var_{22}, var_{33}, var_{44}]
    #    (off‑diagonal entries start at zero).
    # =========================================================================
    var = [1.0, 1.0, 0.0, 0.0]

    # =========================================================================
    # 3  Time points at which we evaluate the information flow.
    # =========================================================================
    time_dict = {"start": 0.0, "stop": 10.0, "num": 200}

    # =========================================================================
    # 4  Direction of the information flow ('2->1' or '1->2').
    # =========================================================================
    direction = "2->1"

    # -------------------------------------------------------------------------
    # Optional: create the param tensor (just to see its shape / usage)
    param_tensor = create_params_grid(k_dict, b_dict, m_dict, omega_dict)
    print("Parameter tensor shape (k, b, m, omega) :", param_tensor.shape)
    # -------------------------------------------------------------------------

    # =========================================================================
    # 5  Batched evaluation  (uses JAX vmap internally)
    # =========================================================================
    info_flow, time_points = evaluate_theoretical_causality_JAX(
        k_dict, b_dict, m_dict, omega_dict,
        var, time_dict, direction
    )

    print("Information‑flow array shape (params × time):", info_flow.shape)
    print("Time‑points array shape:", time_points.shape)

    # =========================================================================
    # 6  Save the results to disk for later plotting / analysis.
    # =========================================================================
    np.savez(
        "liang_kleeman_results.npz",
        flow=np.asarray(info_flow),
        time=np.asarray(time_points),
    )
    print("Saved → liang_kleeman_results.npz")


if __name__ == "__main__":
    main()
