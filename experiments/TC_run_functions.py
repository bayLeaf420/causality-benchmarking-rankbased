import os
import sys
import numpy as np
# import sympy as sp
# import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath('theoretical_causality'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Going to project parent folder
# from theoretical_causality.eval_theoretical_causality import plot_theoretical_causality
# from oscillators.oscillators import damped_driven_oscillator_numpy
from theoretical_causality.eval_theoretical_causality import plot_theoretical_causality

"""
STEPS:
----------
 1. Make a function to create parameter tensor of axes (k, b, m, omega)
 2. Make a function which takes in parameter tensors and runs theoretical causality, outputting 
    a 2 tensors of dims (k, b, m, omega, t), first for T_(3->2) and second for T_(2->1)
"""

def create_params_grid(
        k_dict: dict, # Dict containing keys {'start', 'stop', 'num'} for linspace
        b_dict: dict, # Same keys
        m_dict: dict,
        omega_dict: dict,
) -> np.array:
    """
    Creates a 4-dimensional tensor grid of system parameters.
    
    Output shape: (N_k, N_b, N_m, N_omega, 4)
    where the final axis index contains the values: [k, b, m, omega]

    While output is a np.array, it can be converted later to a JAX array and used in a batched manner
    with vmap. 
    """
    # 1. Create individual vectors
    k_vec = np.linspace(k_dict['start'], k_dict['stop'], k_dict['num'])
    b_vec = np.linspace(b_dict['start'], b_dict['stop'], b_dict['num'])
    m_vec = np.linspace(m_dict['start'], m_dict['stop'], m_dict['num'])
    omega_vec = np.linspace(omega_dict['start'], omega_dict['stop'], omega_dict['num'])

    # 2. Build the N-dimensional grid structures
    # indexing='ij' ensures matrix-style cartesian ordering: (dim0, dim1, dim2, dim3)
    K, B, M, OMEGA = np.meshgrid(k_vec, b_vec, m_vec, omega_vec, indexing='ij')
    
    # 3. Stack along the final axis to create a cohesive parameter coordinate space
    grid_tensor = np.stack([K, B, M, OMEGA], axis=-1)
    
    return grid_tensor

"""We run theoretical causality here"""

def plot_random_TC(
    param_tensor: np.ndarray, 
    mu: list, 
    var: list, 
    time_dict: dict
):
    """
    Selects a random parameter combination coordinate node out of a 4D parameter grid,
    unpacks its physical values, and renders its information flow profile.
    """
    # 1. Identify the boundaries of your parameter grid axes
    N_k, N_b, N_m, N_omega, _ = param_tensor.shape
    
    # 2. Draw a random coordinate index for each axis
    idx_k = np.random.randint(0, N_k)
    idx_b = np.random.randint(0, N_b)
    idx_m = np.random.randint(0, N_m)
    idx_w = np.random.randint(0, N_omega)
    
    # 3. Extract the single [k, b, m, omega] parameter array at those coordinates
    chosen_params = param_tensor[idx_k, idx_b, idx_m, idx_w]
    k_val, b_val, m_val, w_val = chosen_params
    
    # 4. Diagnostics printout to console for execution tracking
    print("=" * 60)
    print(f"[RANDOM SAMPLING] Selected Tensor Node Coordinates: ({idx_k}, {idx_b}, {idx_m}, {idx_w})")
    print(f"-> Physical Parameters: k={k_val:.3f}, b={b_val:.3f}, m={m_val:.3f}, omega={w_val:.3f}")
    print("=" * 60)
    
    # 5. Fire up your plotting routine
    # converting to list to ensure clean standalone variable unpacking inside the visualizer
    plot_theoretical_causality(mu, var, list(chosen_params), time_dict)

if __name__ == "__main__":
    # Define varied counts to verify dimensions are tracking properly
    k_cfg = {'start': 1.0, 'stop': 10.0, 'num': 5}       # N = 5
    b_cfg = {'start': 0.1, 'stop': 1.0, 'num': 4}        # N = 4
    m_cfg = {'start': 1.0, 'stop': 5.0, 'num': 3}        # N = 3
    w_cfg = {'start': 0.5, 'stop': 2.5, 'num': 6}        # N = 6

    param_grid = create_params_grid(k_cfg, b_cfg, m_cfg, w_cfg)
    
    print(f"Generated Grid Tensor Shape: {param_grid.shape}\n")
    print(f"Grid Tensor first value: \n{param_grid[0, 0, 0, 0]}")

    mu = [0.4, 4.0, 1.9, 5.0]
    var = [3.25, 3.25, 3.25, 3.25]
    time_dict = {'start': 0.0, 'stop': 10.0, 'num': 1000}

    plot_random_TC(param_grid, mu, var, time_dict)



    