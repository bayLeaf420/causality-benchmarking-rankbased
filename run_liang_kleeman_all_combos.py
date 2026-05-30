#!/usr/bin/env python3
"""Compute Liang–Kleeman information flow on all parameter grid combinations
   by re‑using the existing evaluate_theoretical_causality function."""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from theoretical_causality.eval_theoretical_causality import evaluate_theoretical_causality
from runs.TC_run_functions import create_params_grid


def main():
    # Parameter ranges
    k_dict      = {"start": 0.1, "stop": 2.0, "num": 5}
    b_dict      = {"start": 0.1, "stop": 2.0, "num": 5}
    m_dict      = {"start": 0.5, "stop": 1.5, "num": 3}
    omega_dict  = {"start": 0.5, "stop": 2.5, "num": 4}

    # Initial covariance diagonals (variances σ²)
    var = [1.0, 1.0, 1.0, 1.0]

    # Time domain
    time_dict = {"start": 0.0, "stop": 10.0, "num": 1000}

    # Build parameter grid (k, b, m, ω)
    param_tensor = create_params_grid(k_dict, b_dict, m_dict, omega_dict)
    N_k, N_b, N_m, N_omega, _ = param_tensor.shape
    print("Parameter tensor shape (k, b, m, ω):", param_tensor.shape)

    # Time vector for output shape
    t_vec = np.linspace(time_dict["start"], time_dict["stop"], time_dict["num"])
    N_t = len(t_vec)

    # Allocate array for T(3→2)
    flow32 = np.empty((N_k, N_b, N_m, N_omega, N_t))

    total = N_k * N_b * N_m * N_omega
    cnt = 0
    # Loop over every combination
    for ik in range(N_k):
        for ib in range(N_b):
            for im in range(N_m):
                for iw in range(N_omega):
                    params = param_tensor[ik, ib, im, iw]
                    # evaluate_theoretical_causality(var, parameters, time_dict)
                    result32, _, _ = evaluate_theoretical_causality(
                        var, list(params), time_dict
                    )
                    flow32[ik, ib, im, iw, :] = result32
                    cnt += 1
                    if cnt % 10 == 0:
                        print(f"Processed {cnt}/{total}")

    print("Information‑flow array shape (params × time):", flow32.shape)
    print("Time‑points array shape:", (N_t,))

    # Save
    np.savez("liang_kleeman_results.npz",
             flow=flow32,
             time=t_vec)
    print("Saved -> liang_kleeman_results.npz")


if __name__ == "__main__":
    main()
