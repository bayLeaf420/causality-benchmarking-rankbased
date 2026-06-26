import jax.numpy as jnp
import numpy as np
import jax

"""Defining Oscillators"""


def damped_driven_oscillator(t: jnp.array, params: jnp.array):
    k = params[0]
    b = params[1]
    m = params[2]
    x0 = params[3]
    v0 = params[4]
    F0 = params[5]
    omega = params[6]

    # t is a numpy array of linspace form
    alpha = b / (2 * m)
    # beta = ((4 * k * m - b**2) ** 0.5) / (2 * m)
    omega_0 = (k / m) ** 0.5
    C = F0 / ((omega_0**2 - omega**2) ** 2 + ((b * omega) / m) ** 2) ** 0.5
    phi = jnp.arctan2(b * omega, m * (omega_0**2 - omega**2))
    D = (v0 + alpha * x0 - alpha * C * jnp.exp(1j * phi)) / (1j * omega)
    A = (x0 - C * jnp.exp(1j * phi) + D) / 2
    B = A - D

    is_critical = jnp.isclose(b**2, 4 * k * m)

    x = jax.lax.cond(
        is_critical,
        lambda _: jnp.real((A * t + B) * jnp.exp(-omega_0 * t)),
        lambda _: jnp.real(
            jnp.exp(-alpha * t)
            * (A * jnp.exp(1j * omega_0 * t) + B * jnp.exp(-1j * omega_0 * t))
            + C * jnp.exp(1j * phi) * jnp.exp(1j * omega * t)
        ),
        operand=None,
    )

    return x, A, B, C, alpha

def damped_driven_oscillator_numpy(t, params):
    k = params[0]
    b = params[1]
    m = params[2]
    x0 = params[3]
    v0 = params[4]
    F0 = params[5]
    omega = params[6]

    alpha = b / (2 * m)
    omega_0 = (k / m) ** 0.5
    
    # Amplitude and phase phase shift of the steady-state driving force response
    C = F0 / ((omega_0**2 - omega**2) ** 2 + ((b * omega) / m) ** 2) ** 0.5
    phi = np.arctan2(b * omega, m * (omega_0**2 - omega**2))
    
    # Integration constants determined by initial conditions
    D = (v0 + alpha * x0 - alpha * C * np.exp(1j * phi)) / (1j * omega)
    A = (x0 - C * np.exp(1j * phi) + D) / 2
    B = A - D

    # Check for critical damping condition
    if np.isclose(b**2, 4 * k * m):
        x = np.real((A * t + B) * np.exp(-omega_0 * t))
    else:
        # Underdamped / Overdamped general analytical form
        x = np.real(
            np.exp(-alpha * t) * (A * np.exp(1j * omega_0 * t) + B * np.exp(-1j * omega_0 * t))
            + C * np.exp(1j * phi) * np.exp(1j * omega * t)
        )

    return x, A, B, C, alpha


def driving_force(t, params):
    F0 = params[5]
    omega = params[6]
    return F0 * jnp.cos(omega * t)


def noisy_dd_oscillator(
    t: jnp.array,
    params: jnp.array,
    noise_factor: float,
    key: jax.random.PRNGKey,
) -> tuple:

    x, A, B, C, alpha = damped_driven_oscillator(t, params)
    noise_std_dev = noise_factor  # * x.std()
    noise = jax.random.normal(key, x.shape, dtype=jnp.float32) * noise_std_dev

    return x + noise, A, B, C, alpha


def noisy_driving_force(
    t: jnp.array,
    params: jnp.array,
    noise_factor: float,
    key: jax.random.PRNGKey,
) -> tuple:

    F = driving_force(t, params)
    noise_std_dev = noise_factor  # * F.std()
    noise = jax.random.normal(key, F.shape, dtype=jnp.float32) * noise_std_dev
    return F + noise
