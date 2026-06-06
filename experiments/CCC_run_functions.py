import os 
import sys
import jax 
import jax.numpy as jnp # To jit and batch the functions
import matplotlib.pyplot as plt # To help with plotting functions

# Linking parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from CCC_functions.ETC_and_causality import CCC_calculation_2vec # For CCC calculation
from runs.TC_run_functions import create_params_grid # For creating params tensor
from oscillators.oscillators import damped_driven_oscillator, driving_force

def add_samples_to_params(
        k_dict: dict,      # Dict for k, with keys 'start', 'stop', and 'num'.
        b_dict: dict,      # Same
        m_dict: dict,      # Same
        omega_dict: dict,  # Same
        n_samples: int,    # Number of samples taken from normal distribution
        mu: jnp.array,     # Vector consisting of 4 mu values 
        var: jnp.array,    # Vector denoting diagonal of covariance matrix. 
        key: jax.Array,    # JAX PRNGKey for reproducible random sampling
) -> jax.Array:
    """
    Function to increase dimensions of parameters tensor by appending
    n_samples of random initial conditions to every parameter node.
    
    Output shape: (N_k, N_b, N_m, N_omega, n_samples, 8)
    where the final axis holds: [k, b, m, omega, x1_0, v1_0, x2_0, v2_0]
    """
    # 1. Build the base 4D parameter grid and convert to a JAX array
    param_tensor = jnp.array(create_params_grid(k_dict, b_dict, m_dict, omega_dict))
    N_k, N_b, N_m, N_omega, _ = param_tensor.shape
    
    # 2. Draw random initial condition vectors from the normal distribution using the passed key
    covariance_matrix = jnp.diag(var)
    ic_samples = jax.random.multivariate_normal(key, mean=mu, cov=covariance_matrix, shape=(n_samples,))
    
    # 3. Expand axes to prepare both tensors for broadcasting
    # Shape changes from (N_k, N_b, N_m, N_omega, 4) -> (N_k, N_b, N_m, N_omega, 1, 4)
    param_expanded = param_tensor[:, :, :, :, jnp.newaxis, :]
    
    # Shape changes from (n_samples, 4) -> (1, 1, 1, 1, n_samples, 4)
    ic_expanded = ic_samples[jnp.newaxis, jnp.newaxis, jnp.newaxis, jnp.newaxis, :, :]
    
    # 4. Broadcast both arrays to the unified target matrix shape
    target_grid_shape = (N_k, N_b, N_m, N_omega, n_samples, 4)
    param_broadcasted = jnp.broadcast_to(param_expanded, target_grid_shape)
    ic_broadcasted = jnp.broadcast_to(ic_expanded, target_grid_shape)
    
    # 5. Concatenate along the final axis to create the unified 8-element vector
    mega_grid = jnp.concatenate([param_broadcasted, ic_broadcasted], axis=-1)
    
    return mega_grid



    

def eval_CCC_temporal(
        k_dict: dict, # Dict for k, with keys 'start', 'stop', and 'num'.
        b_dict: dict, # Same
        m_dict: dict, # Same
        omega_dict: dict, # Same
        cause_bins: int, # Number of Bins for causal time series. 
        effect_bins: int, # Number of Bins for effect time series. 
        INFO: dict, # Dict for info required by CCC with keys 'past', 'present', 'jump'.
        n_samples: int, # Number of samples taken from normal distribution
        mu: jnp.array, # Vector consisting of 4 mu values s
        var: jnp.array, # Vector denoting diagonal of covariance matrix. 
)-> jax.Array: 
    """
    Evaluate the CCC values between cause and effect oscillators, over the params grid, and across
    'n_samples' samples. 

    Output:
    --------------
    CCC_tensor: jax array of shape (N_k, N_b, N_m, N_omega, N_samples)

    Algorithm:
    --------------
    1. Create required tensors: Create the parameters grid, along with random samples. 
    """
    