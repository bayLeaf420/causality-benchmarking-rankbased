import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from sampling import sample_oscillator_tensor, _lambdify_oscillator
from algorithms.tc import _symbolic_information_flow
import numpy as np
import sympy as sp
import time

def test_simple_plot():
    n_r = 1
    params_dict = {
        'beta': (1.0, 1.1, 1),
        'K': (1.4, 1.5, 1),
        'mu': (0.6, 0.7, 1),
    }
    tau_init = (0, 15, 200)
    key = jax.random.key(29)

    mean_init = (1.0, 0.0, 0.0, 1.53)
    sigma_init = (0.05, 0.05, 0.05, 0.05)

    func_inputs = (n_r, params_dict, tau_init, mean_init, sigma_init, key)

    sample_arr =  sample_oscillator_tensor(*func_inputs)
    # sample_arr.reshape(4, 2000)
    sample_arr = sample_arr.reshape(4, 2000)
    print(f"Shape of sampled array: {sample_arr.shape}\n")
    print(sample_arr)
    print(jnp.isnan(sample_arr))
    tau_vec = jnp.linspace(*tau_init)

    plt.plot(tau_vec, sample_arr[1,:])
    plt.show()

def test_check_eigvals():
    beta, K, mu = 1.2, 1.1, 0.6

    A = np.array([
        [0, 1, 0, 0],
        [-1, -beta, -mu*K**2, 0],
        [0, 0, 0, 1],
        [0, 0, -K**2, 0],
    ])
    print(np.linalg.eigvals(A))

def test_lambdify():
    tau_init = (0, 15, 100)
    mean_init = (1.0, 0.0, 0.0, 1.53)
    sigma_init = (1.0, 1.0, 1.0, 1.0)
    mean_lambda, sigma_lambda = _lambdify_oscillator(tau_init, mean_init, sigma_init)

    beta, K, mu = 1.0, 1.4, 0.6
    print(f"beta, K, mu = {beta, K, mu}")
    for t in [0.0, 0.005, 0.1, 1.0, 5.0, 10.0, 15.0]:
        m = mean_lambda(t, beta, K, mu)
        s = sigma_lambda(t, beta, K, mu)
        print(f"tau={t}: mean={m}")
        print(f"tau={t}: sigma=\n{s}\n")

def test_eigvals():
    information_flow, Sigma_t, mean_t, symbols, mean_symbols = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 = symbols

    expr = Sigma_t[0,0].subs([(tau_0, 0), (s11,1),(s22,1),(s33,1),(s44,1), (beta, sp.Rational(13,10))])
    print("Raw:", expr)
    print("Contains I:", expr.has(sp.I))
    print("Free symbols:", expr.free_symbols)

    # Try evaluating numerically at tau=0.1 symbolically (not via lambdify)
    val = expr.subs(tau, sp.Rational(1,10))
    print("Symbolic value at tau=0.1:", val)
    print("Numeric (.evalf()):", val.evalf())


if __name__=="__main__":
    st = time.perf_counter()

    test_simple_plot()
    # test_check_eigvals()
    # test_lambdify()
    # test_eigvals()

    et = time.perf_counter()

    tt = et - st
    print(f"\nTime taken by function to execute: {tt}\n")