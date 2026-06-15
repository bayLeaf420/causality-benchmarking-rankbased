import jax.numpy as jnp
import jax
from typing import Dict, Tuple

def build_params_tensor_jax(
        params_dict: Dict[str, Tuple[float, float, int]],
    ) -> jax.Array:
    """
    Inputs:
    1. params_dict: Dictionary where keys are 'beta', 'K', and 'mu'. 
                    The values are Tuples with [start, stop, num].
    Outputs:
    1. params_tensor: jax.Array of shape (num_beta, num_K, num_mu)

    Helper function to quickly get the params_tensor in jax array form.
    """

    beta_builder = params_dict['beta']
    K_builder = params_dict['K']
    mu_builder = params_dict['mu']

    # '*' operator "unpacks" the tuple values into the function call. 
    beta_arr = jnp.linspace(*beta_builder)
    K_arr = jnp.linspace(*K_builder)
    mu_arr = jnp.linspace(*mu_builder)

    params_tensor = jnp.stack(jnp.meshgrid(beta_arr, K_arr, mu_arr, indexing='ij'), axis=-1)

    return params_tensor

# def sym_inf_flow_path = 