"""The (beta, K, mu) parameter tensor and helpers to index it."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from lkif_rank_causal_bench.config import ParameterGrid

AXIS_NAMES: tuple[str, str, str] = ("beta", "K", "mu")


def params_tensor(grid: ParameterGrid) -> jax.Array:
    """Grid of (beta, K, mu) points, shape (n_beta, n_K, n_mu, 3), ij indexing."""
    beta = jnp.linspace(*grid.beta)
    K = jnp.linspace(*grid.K)
    mu = jnp.linspace(*grid.mu)
    return jnp.stack(jnp.meshgrid(beta, K, mu, indexing="ij"), axis=-1)


def params_tensor_from_ranges(ranges: Mapping[str, tuple[float, float, int]]) -> jax.Array:
    """Params tensor from a {'beta': (start, stop, num), 'K': ..., 'mu': ...} mapping."""
    return params_tensor(ParameterGrid(beta=ranges["beta"], K=ranges["K"], mu=ranges["mu"]))


def axis_values(params: np.ndarray, axis: int) -> np.ndarray:
    """The values of one parameter along its axis of a params tensor."""
    idx: list[int | slice] = [0, 0, 0]
    idx[axis] = slice(None)
    return params[tuple(idx) + (axis,)]


def flatten_grid(arr: jax.Array) -> jax.Array:
    """Collapse the three leading grid axes of an array into one."""
    return arr.reshape(-1, *arr.shape[3:])
