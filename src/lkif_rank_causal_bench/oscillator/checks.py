"""Warnings about sampling choices that make GC or TE ill-conditioned."""

import numpy as np

from lkif_rank_causal_bench.config import ParameterGrid, SimulationConfig


def sampling_warnings(grid: ParameterGrid, sim: SimulationConfig) -> list[str]:
    """Human-readable warnings for oversampling, aliasing, short windows and degenerate noise."""
    out = []
    dt = sim.dt
    span = sim.tau[1] - sim.tau[0]
    fastest_period = 2 * np.pi  # omega_0 = 1 for oscillator 1
    slowest_period = 2 * np.pi / max(grid.K[0], 1e-12)

    per_period = fastest_period / dt
    if per_period > 100:
        out.append(
            f"dt = {dt:.4g} gives {per_period:.0f} samples per period of the "
            "driven oscillator. That is heavily oversampled: the VAR approaches "
            "a unit root and GC estimates collapse. Aim for 10-60."
        )
    if per_period < 6:
        out.append(f"dt = {dt:.4g} gives only {per_period:.1f} samples per period; the dynamics are aliased.")
    n_slow = span / slowest_period
    if n_slow < 3:
        out.append(
            f"tau window {span:.4g} covers only {n_slow:.1f} periods of the "
            f"slowest driver (K_min = {grid.K[0]:.3g}, period {slowest_period:.3g}). "
            "Lengthen tau_end or raise K_min."
        )
    if sim.q1 <= 0.0:
        out.append(
            "q1 = 0 makes the process deterministic given x0; Granger causality is degenerate there. Set q1 > 0."
        )
    if sim.n_realisations < 2:
        out.append("n_realisations must exceed 1 for across-trial demeaning.")
    if sim.stationary and sim.beta2 <= 0.0:
        out.append("stationary = True requires beta2 > 0.")
    return out
