"""Benchmark of heuristic causality measures against Liang-Kleeman information flow."""

import jax

# The LKIF moment ODE runs at atol = 1e-9, below float32 epsilon, so the whole
# package works in float64. This must run before any JAX array is created.
jax.config.update("jax_enable_x64", True)
