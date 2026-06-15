from algorithms.tc import _symbolic_information_flow, build_time_integral_fn, tc_integral
from utils import build_params_tensor_jax
import time
from itertools import product
import pickle
from pathlib import Path
import jax.numpy as jnp
import sympy as sp
import pdb

def test_symb_info_flow():
    max_chars = 80
    # Use *_ to correctly unpack and ignore the trailing metadata tuples
    expr_matrix, *_ = _symbolic_information_flow()

    output_path = Path("./results/test_results/symbolic_info_flow_matrix.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(expr_matrix, f)
    print(f"\n[INFO] Successfully saved sp.Matrix to {output_path.resolve()}")
    
    # Check all 16 entries of the 4x4 matrix
    for i, j in product(range(4), range(4)):
        # Convert the SymPy expression to a string before checking length
        expr_str = str(expr_matrix[i, j])
        
        if len(expr_str) > max_chars:
            print(f"Matrix[{i},{j}]: " + expr_str[:max_chars] + " ... [TRUNCATED]")
        else:
            print(f"Matrix[{i},{j}]: " + expr_str)
            
    return None

def test_tc_integral():

    tau_init = (0, 10, 100)
    sigma_init = (1, 1, 1, 1)
    time_integral_fn_list = build_time_integral_fn(tau_init, sigma_init)

    params_dict = {
        'beta': (0.2, 3.0, 3),
        'K': (0.2, 2.2, 2),
        'mu': (0.3, 1.3, 3),
    }
    params_tensor = build_params_tensor_jax(params_dict)
    time_integrals_tc = tc_integral(params_tensor, time_integral_fn_list)
    print(f"Shape of integrals: {time_integrals_tc.shape}")

    output_path = Path("./results/test_results/tc_integral_vals.npz")
    jnp.savez(output_path, time_integrals_tc)

    return None

def test_integral_build():
    tau_init = (0, 10, 100)
    sigma_init = (1, 1, 1, 1)
    time_integral_fn_list = build_time_integral_fn(tau_init, sigma_init)

    output_path = Path("./results/time_integrals_fn_list.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(time_integral_fn_list, f)
    print(f"\n[INFO] Successfully saved sp.Matrix to {output_path.resolve()}")

    return None

def test_integration_time():
    breakpoint()
    information_flow, _, _, symbols, _ = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols
    tau_start = 0.0
    tau_end = 10.0
    sigma_init = (1, 1, 1, 1)
    information_flow = information_flow.subs([
        (tau_0, float(tau_start)), 
        (s11, float(sigma_init[0])),
        (s22, float(sigma_init[1])),
        (s33, float(sigma_init[2])),
        (s44, float(sigma_init[3])),
        ])
    information_flow = information_flow.applyfunc(sp.cancel)
    integral_of_3_2 = sp.integrate(information_flow[3, 2], (tau, float(tau_start), float(tau_end)))
    output_path = Path("./results/integral_of_3_2.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(integral_of_3_2, f)
    print(f"\n[INFO] Successfully saved sp.Matrix to {output_path.resolve()}")
    return None

# IMPNOTE: We have determined now that analytical integration with sympy takes too long. 

# def test_numerical_integration():
#     params_dict = {
#         'beta': (0.1)
#     }

if __name__=="__main__":
    
    start_time = time.perf_counter()
    test_symb_info_flow()
    # test_tc_integral()
    # test_integral_build()
    # pdb.runcall(test_integration_time)
    end_time = time.perf_counter()

    # Calculate and print total execution time
    execution_time = end_time - start_time
    print(f"Function took {execution_time:.8f} seconds to complete.")
    
