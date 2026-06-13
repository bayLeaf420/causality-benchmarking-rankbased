from algorithms.tc import _symbolic_information_flow
import time
from itertools import product
import pickle
from pathlib import Path

def test_symb_info_flow():
    max_chars = 80
    # Use *_ to correctly unpack and ignore the trailing metadata tuples
    expr_matrix, *_ = _symbolic_information_flow()

    output_path = Path("symbolic_info_flow_matrix.pkl")
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

if __name__=="__main__":
    
    start_time = time.perf_counter()
    test_symb_info_flow()
    end_time = time.perf_counter()

    # Calculate and print total execution time
    execution_time = end_time - start_time
    print(f"Function took {execution_time:.8f} seconds to complete.")
    
