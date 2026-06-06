import os
import sys
# import numpy as np
import sympy as sp
import jax 
import jax.numpy as jnp
# import matplotlib.pyplot as plt

# sys.path.append(os.path.abspath('theoretical_causality'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Going to project parent folder
from sym_information import symbolic_information_flow
# from oscillators.oscillators import damped_driven_oscillator_numpy
from runs.TC_run_functions import create_params_grid

def evaluate_theoretical_causality_JAX(
        k_dict: dict,
        b_dict: dict,
        m_dict: dict,
        omega_dict: dict,
        var: list, # Initial variance [s11, s22, s33, s44] eq to diag(Covar_Matrix)
        time_dict: dict, # Keys 'start', 'stop' and 'num
        direction: str, # String '2->1' will return causality for '2->1' direction
) -> tuple[jax.Array, jax.Array]:
    """Accelerates theoretical info flow evaluations across params 
    grid using JAX JIT compilation"""

    # 1. Build parameters tensor
    param_tensor = create_params_grid(k_dict, b_dict, m_dict, omega_dict)
    N_k, N_b, N_m, N_omega, _ = param_tensor.shape

    # 2. Create time array
    t_start = time_dict['start']
    t_vec = jnp.linspace(t_start, time_dict['stop'], time_dict['num'])
    N_t = len(t_vec)

    # 3. Pull symbolic expressions and lock down t0 to init start time
    flow_dict, _ = symbolic_information_flow()
    expr = flow_dict[direction].subs(sp.symbols('t0'), t_start)
    # expr21 = flow_dict['2->1'].subs(sp.symbols('t0'), t_start)

    # 4. Compile expressions to JAX modules
    t, k, b, m, omega = sp.symbols('t k b m omega', real=True)
    s11, s22, s33, s44 = sp.symbols('sigma11 sigma22 sigma33 sigma44', real=True)
    
    j_func = sp.lambdify((t, k, b, m, omega, s11, s22, s33, s44), expr, modules='jax')
    # j_func21 = sp.lambdify((t, k, b, m, omega, s11, s22, s33, s44), expr21, modules='jax')

    # Unpack variances
    s11_v, s22_v, s33_v, s44_v = var

    @jax.jit
    def evaluate_single_config(p_vec):
        """Calculate the full time series array for one param node"""
        k_val, b_val, m_val, omega_val = p_vec

        val = j_func(t_vec, k_val, b_val, m_val, omega_val, s11_v, s22_v, s33_v, s44_v)
        # val21 = j_func21(t_vec, k_val, b_val, m_val, omega_val, s11_v, s22_v, s33_v, s44_v)
        
        # Ensure outputs behave as solid 1D time arrays rather than dynamic shapes
        return jnp.atleast_1d(val) # , jnp.atleast_1d(val21)
    
    # 6. Flatten grid space into 2D matrix struc for easy vmap
    flat_params = jnp.array(param_tensor.reshape(-1, 4))

    # 7. Apply vectorization batch across flattened space
    vmap_evaluator = jax.vmap(evaluate_single_config, in_axes=(0,))
    flat_t32, flat_t21 = vmap_evaluator(flat_params)

    # 8. Reconstruct structural dimensions back to matching grid outputs
    t32_tensor = flat_t32.reshape(N_k, N_b, N_m, N_omega, N_t)
    t21_tensor = flat_t21.reshape(N_k, N_b, N_m, N_omega, N_t)

    return t32_tensor, t21_tensor
    

