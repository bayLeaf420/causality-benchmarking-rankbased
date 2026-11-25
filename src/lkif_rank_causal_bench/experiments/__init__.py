"""Experiment types, selected by the `experiment:` field of a run config."""

from collections.abc import Callable
from pathlib import Path

from lkif_rank_causal_bench.config import RunConfig
from lkif_rank_causal_bench.experiments.bootstrap import run_bootstrap
from lkif_rank_causal_bench.experiments.grid import run_grid
from lkif_rank_causal_bench.experiments.lag_order import run_lag_order
from lkif_rank_causal_bench.experiments.significance import run_significance, run_significance_subgrid
from lkif_rank_causal_bench.experiments.sweeps import run_realisation_sweep, run_setting_sweep

EXPERIMENTS: dict[str, Callable[[RunConfig], Path]] = {
    "grid": run_grid,
    "setting_sweep": run_setting_sweep,
    "realisation_sweep": run_realisation_sweep,
    "lag_order": run_lag_order,
    "bootstrap": run_bootstrap,
    "significance": run_significance,
    "significance_subgrid": run_significance_subgrid,
}


def run_experiment(cfg: RunConfig) -> Path:
    """Run the experiment a config names and return its output file."""
    if cfg.experiment not in EXPERIMENTS:
        raise KeyError(f"unknown experiment '{cfg.experiment}'; choose from {sorted(EXPERIMENTS)}")
    return EXPERIMENTS[cfg.experiment](cfg)
