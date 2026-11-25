"""The LKIF ground-truth arrays for a system: transient or stationary."""

import numpy as np

from lkif_rank_causal_bench.config import SystemConfig
from lkif_rank_causal_bench.lkif.stationary import stationary_point
from lkif_rank_causal_bench.lkif.transient import lkif_integral_grid, time_average
from lkif_rank_causal_bench.oscillator.grid import params_tensor
from lkif_rank_causal_bench.oscillator.model import NULL_ENTRY, TRUTH_ENTRY, noise_covariance


def _truth_dict(norm: np.ndarray, raw: np.ndarray) -> dict[str, np.ndarray]:
    """Package full normalised and raw flow grids into the stored result keys."""
    s, t = TRUTH_ENTRY
    return {
        "lkif_full": norm,
        "truth": norm[..., s, t],
        "truth_null": norm[..., NULL_ENTRY[0], NULL_ENTRY[1]],
        "truth_raw": np.abs(raw[..., s, t]),
        "lkif_raw_full": raw,
    }


def lkif_truth(system: SystemConfig) -> dict[str, np.ndarray]:
    """Truth arrays (truth, truth_null, truth_raw, lkif_full, lkif_raw_full) over the grid."""
    sim = system.simulation
    params = params_tensor(system.grid)
    G = noise_covariance(sim.q1, sim.q2)
    if sim.stationary:
        # samples come from the stationary law, so the truth is the flow at the
        # stationary covariance, not a time average of the transient
        p = np.asarray(params)
        norm = np.zeros(p.shape[:-1] + (4, 4))
        raw = np.zeros(p.shape[:-1] + (4, 4))
        for idx in np.ndindex(*p.shape[:-1]):
            norm[idx], raw[idx] = stationary_point(*p[idx], G=np.asarray(G), beta2=sim.beta2)
        return _truth_dict(norm, raw)

    norm = lkif_integral_grid(params, sim.tau, sim.sigma_init, G=G, beta2=sim.beta2)
    raw = lkif_integral_grid(params, sim.tau, sim.sigma_init, G=G, beta2=sim.beta2, raw=True)
    return _truth_dict(np.asarray(time_average(norm, sim.tau)), np.asarray(time_average(raw, sim.tau)))
