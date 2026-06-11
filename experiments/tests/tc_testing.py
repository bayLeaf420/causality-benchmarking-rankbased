import jax.numpy as jnp
from algorithms.tc import build_time_integral_fn, tc_eval

sigma_vec = jnp.array([1.0, 1.0, 1.0, 1.0])

def check_shape(result):
    assert result.shape == (4, 4), f"Expected (4,4), got {result.shape}"

# --- General cases ---

def test_general_1():
    """Typical mid-range parameters"""
    fn = build_time_integral_fn(0.0, 1.0, sigma_vec)
    params = jnp.array([[[1.0]],[[1.0]],[[1.0]]])  # beta=K=mu=1
    result = tc_eval(params, fn)
    check_shape(result[0,0,0])
    assert jnp.all(jnp.isfinite(result)), "Non-finite values in output"
    print("PASS test_general_1")

def test_general_2():
    """Grid of parameters, checks output shape across batch"""
    fn = build_time_integral_fn(0.0, 2.0, sigma_vec)
    beta = jnp.linspace(0.5, 2.0, 3)
    K    = jnp.linspace(0.5, 2.0, 3)
    mu   = jnp.linspace(0.5, 2.0, 3)
    params = jnp.stack(jnp.meshgrid(beta, K, mu, indexing='ij'), axis=-1)  # (3,3,3,3)
    result = tc_eval(params, fn)
    assert result.shape == (3, 3, 3, 4, 4), f"Expected (3,3,3,4,4), got {result.shape}"
    assert jnp.all(jnp.isfinite(result)), "Non-finite values in output"
    print("PASS test_general_2")

# --- Edge cases ---

def test_edge_zero_interval():
    """tau_start == tau_end: integral should be zero everywhere"""
    fn = build_time_integral_fn(1.0, 1.0, sigma_vec)
    params = jnp.array([[[1.0]],[[1.0]],[[1.0]]])
    result = tc_eval(params, fn)
    assert jnp.allclose(result, 0.0, atol=1e-6), f"Expected zeros, got {result}"
    print("PASS test_edge_zero_interval")

def test_edge_large_params():
    """Very large beta/K/mu — checks for inf/nan"""
    fn = build_time_integral_fn(0.0, 1.0, sigma_vec)
    params = jnp.array([[[1e6]],[[1e6]],[[1e6]]])
    result = tc_eval(params, fn)
    assert jnp.all(jnp.isfinite(result)), "Got non-finite values with large params"
    print("PASS test_edge_large_params")

def test_edge_small_params():
    """Very small beta/K/mu — checks for division by zero or underflow"""
    fn = build_time_integral_fn(0.0, 1.0, sigma_vec)
    params = jnp.array([[[1e-6]],[[1e-6]],[[1e-6]]])
    result = tc_eval(params, fn)
    assert jnp.all(jnp.isfinite(result)), "Got non-finite values with small params"
    print("PASS test_edge_small_params")

if __name__ == "__main__":
    test_general_1()
    test_general_2()
    test_edge_zero_interval()
    test_edge_large_params()
    test_edge_small_params()