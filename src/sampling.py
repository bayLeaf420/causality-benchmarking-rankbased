import sympy as sp
import jax
import jax.numpy as jnp
from typing import Tuple, Callable, Dict
from algorithms.tc import _symbolic_information_flow
from functools import partial# , lru_cache

# @lru_cache(maxsize=None)
def _lambdify_oscillator(
        tau_init: Tuple[float, float, int], 
        mean_init: Tuple[float], # 4 floats 
        sigma_init: Tuple[float], # 4 floats
    ) -> Tuple[Callable, Callable]:

    _, sigma_expr, mean_expr, symbols, mean_symbols = _symbolic_information_flow()
    tau, tau_0, beta, K, mu, s11, s22, s33, s44 =symbols
    mean1, mean2, mean3, mean4 = mean_symbols
    tau_start = tau_init[0]

    # Mean and covariance are completely independent of each other we don't need put the initial
    # means or covariances only into their own expressions. 
    mean_expr = mean_expr.subs([ 
        (tau_0, float(tau_start)), 
        (mean1, float(mean_init[0])),
        (mean2, float(mean_init[1])),
        (mean3, float(mean_init[2])),
        (mean4, float(mean_init[3])),
        ])
    
    sigma_expr = sigma_expr.subs([
        (tau_0, float(tau_start)), 
        (s11, float(sigma_init[0])),
        (s22, float(sigma_init[1])),
        (s33, float(sigma_init[2])),
        (s44, float(sigma_init[3])),
    ])

    # Use lambdify to convert into a JAX-compatible lambda function. 
    mean_expr_flat = [mean_expr[i, 0] for i in range(4)] # So that output is of shape (4,)
    mean_lambda = sp.lambdify((tau, beta, K, mu), mean_expr_flat, modules='jax', cse=True)
    sigma_lambda = sp.lambdify((tau, beta, K, mu), sigma_expr, modules='jax', cse=True)

    return mean_lambda, sigma_lambda

@partial(jax.jit, static_argnums=(2, 3))
@partial(jax.vmap, in_axes=(None, None, None, None, 0)) # Map over keys n_realisations shaped dimension
@partial(jax.vmap, in_axes = (0, None, None, None, 0)) # Map over params_tensor and keys n_beta shaped dimension
@partial(jax.vmap, in_axes = (0, None, None, None, 0)) # Map over params_tensor and keys n_K shaped dimension
@partial(jax.vmap, in_axes = (0, None, None, None, 0)) # Map over params_tensor and keys n_mu shaped dimension
def _sample_single(
        params_tensor: jax.Array, 
        tau_vec: jax.Array, 
        mean_lambda: Callable, 
        sigma_lambda: Callable, 
        keys_grid: jax.Array,
        )->jax.Array:
    """
    Inputs:
    1. params_tensor: assume shape (3,). Vmap over first 3 dims in (n_beta, n_K, n_mu, 3)
    2. tau_vec: shape (N, )
    3. mean_vec: shape(N, 4)
    4. sigma_vec: shape(N, 4, 4)
    5. keys_grid: assume shape (1,). Vmap over all 3 dims in (n_beta, n_K, n_mu,)

    Outputs:
    1. x: Time series of shape (4, n)
    """
    # Assume params_tensor is single values. Because _sample_single itself is gonna be vmapped
    beta = params_tensor[0]
    K = params_tensor[1]
    mu = params_tensor[2]

    keys = jax.random.split(keys_grid, num=tau_vec.shape[0]) # Split keys so that 

    return jax.vmap(
        lambda tau, key, beta, K, mu: jax.random.multivariate_normal(
            key, jnp.array(mean_lambda(tau, beta, K, mu)), sigma_lambda(tau, beta, K, mu), 
        ),
        in_axes=(0, 0, None, None, None), # Vmap over tau and key dimensions. 
    )(tau_vec, keys, beta, K, mu) 


def sample_oscillator_tensor(
        n_realisations: int,
        params_dict: Dict[str, Tuple[float, float, int]], 
        tau_init:Tuple[float, float, int], 
        mean_init: Tuple[float], # 4 floats
        sigma_init: Tuple[float], # 4 floats
        key: jax.Array
    )->jax.Array:
    """
    Inputs:
    0. n_realisations: Number of realisations of the tensor needed. 
    1. params_dict: disctionary with keys ['beta', 'K', 'mu']
    2. tau_init: contians tau_start, tau_end, tau_num
    2. mean_vec: vec with mean1, mean2, mean3, mean4
    3. sigma_vec: vec with s11, s22, s33, s44

    Outputs:
    oscillator_tensor: shape(n_realisations, n_beta, n_K, n_mu, N)

    Point of this function is to take in the base input forms and output a tensor representing 1 sample drawn over param space
    """

    mean_lambda, sigma_lambda = _lambdify_oscillator(tau_init, mean_init, sigma_init)
    tau_vec = jnp.linspace(tau_init[0], tau_init[1], tau_init[2])

    beta_vec = jnp.linspace(params_dict['beta'][0], params_dict['beta'][1], params_dict['beta'][2]) # Shape (N_beta,)
    K_vec = jnp.linspace(params_dict['K'][0], params_dict['K'][1], params_dict['K'][2])
    mu_vec = jnp.linspace(params_dict['mu'][0], params_dict['mu'][1], params_dict['mu'][2])

    params_tensor = jnp.stack(jnp.meshgrid(beta_vec, K_vec, mu_vec, indexing='ij'), axis=-1)

    key_num = beta_vec.shape[0] * K_vec.shape[0] * mu_vec.shape[0] * n_realisations
    keys = jax.random.split(key, num=key_num)
    keys = keys.reshape(n_realisations, beta_vec.shape[0], K_vec.shape[0], mu_vec.shape[0], *key.shape)

    return _sample_single(params_tensor, tau_vec, mean_lambda, sigma_lambda, keys)
