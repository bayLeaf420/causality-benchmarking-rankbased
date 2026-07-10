# type: ignore
import sympy as sp
import jax


from utils import build_params_tensor_jax
from algorithms.tc_numerical import time_integrals
from algorithms.tc import _symbolic_information_flow, _normalised_information_flow
from utils import build_params_tensor_jax


params_dict = {
        'beta': (0.0, 10.0, 2),
        'K': (0.0, 9.0, 2),
        'mu': (0.0, 5.0, 2),
    }
tau_init = (0.0, 10.0, 400)
sigma_init = (0.0, )


def test_build_tensor():
    tensor = build_params_tensor_jax(params_dict)
    assert tensor.shape == (100, 24, 22, 3)


def test_syminf():
    syminf, *_ = _symbolic_information_flow()
    n_syminf, *_ = _normalised_information_flow()
    is_zero = sp.Matrix(
        [
            [True, False, True, True],
            [False, False, False, True],
            [True, True, True, False],
            [True, True, False, True],
        ]
    )
    # bool_matrix_syminf = sp.Matrix(syminf.rows, syminf.cols, lambda i, j: syminf[i, j].equals(0))
    # assert bool_matrix_syminf == is_zero 
    bool_matrix_n_syminf = sp.Matrix(
        n_syminf.rows, n_syminf.cols, lambda i, j: n_syminf[i, j].equals(0)
    )
    assert bool_matrix_n_syminf == is_zero
    
if __name__=="__main__":
    test_syminf()