from algorithms.tc_numerical import time_integrals, _build_lambda_fns
from sampling import sample_oscillator_tensor
from algorithms.ccc.ETC_and_CCC import CCC_calculation
import jax
import jax.numpy as jnp

def main():
    # ----Generate Samples over grid----
    n_realisations = 1
    params_dict = {
        'beta': (0.05, 20.0, 5),
        'K': (0.05, 20.0, 5),
        'mu': (0.05, 20.0, 5),
    }
    tau_init = (0.0, 10.0, 500)
    mean_init = (1.0, 0.0, -1.0, 0.0)
    sigma_init = (0.1, 0.1, 0.1, 0.1)
    key = jax.random.key(99)

    dynamical_system = sample_oscillator_tensor(
        n_realisations,
        params_dict,
        tau_init,
        mean_init,
        sigma_init,
        key,
    )
    # ---- v2->x1 pair ---
    idx = jnp.asarray([3, 0])
    input_data_ccc = dynamical_system[idx, :]

    # ---- Run CCC over grid ----
    CCC_grid = jax.vmap(
        jax.vmap(
            jax.vmap(
                CCC_calculation,
                in_axes=(0, None, None, None)
            ),
            in_axes=(0, None, None, None)
        ),
        in_axes = (0, None, None, None)
    )(input_data_ccc, (90, 90, 90), (16, 16), 1e-6)

    # ---- Get TC over grid ---
    lambda_info, lambda_norm_info = _build_lambda_fns(tau_init, sigma_init)

    beta_vec = jnp.linspace(
        params_dict["beta"][0], params_dict["beta"][1], params_dict["beta"][2]
    )  # Shape (N_beta,)
    K_vec = jnp.linspace(params_dict["K"][0], params_dict["K"][1], params_dict["K"][2])
    mu_vec = jnp.linspace(
        params_dict["mu"][0], params_dict["mu"][1], params_dict["mu"][2]
    )

    params_tensor = jnp.stack(
        jnp.meshgrid(beta_vec, K_vec, mu_vec, indexing="ij"), axis=-1
    )
    TC_grid = time_integrals(
        params_tensor,
        tau_init,
        lambda_info,
        lambda_norm_info
    )
