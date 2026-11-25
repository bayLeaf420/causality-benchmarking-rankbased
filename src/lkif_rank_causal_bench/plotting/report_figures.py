"""Write every report figure from the result files in results/ into docs/figures/."""

from pathlib import Path

from lkif_rank_causal_bench.analysis.report_numbers import report_numbers
from lkif_rank_causal_bench.paths import FIGURES, RESULTS
from lkif_rank_causal_bench.plotting.experiment_figures import (
    ccc_settings_figure,
    lag_order_figure,
    sensitivity_figure,
)
from lkif_rank_causal_bench.plotting.grid_figures import Run, grid_figures, has_all_estimators
from lkif_rank_causal_bench.plotting.overview_figures import (
    axis_overview_figure,
    confidence_figure,
    overview_figure,
    significance_figure,
)
from lkif_rank_causal_bench.results import load_results, merge_files

# PCMCI and CCM of the default grid were saved in their own file
OTHER_KEYS = ("pcmci", "pcmci_p", "pcmci_reverse", "pcmci_reverse_p", "ccm", "ccm_reverse")


def default_grid_results(results_dir: Path = RESULTS) -> dict[str, object]:
    """The default-grid results with PCMCI and CCM merged in from their separate file."""
    return merge_files(
        load_results(results_dir / "default_grid.npz"), results_dir / "exp_other_estimators.npz", OTHER_KEYS
    )


def make_all_figures(results_dir: Path = RESULTS, out_dir: Path = FIGURES) -> list[Path]:
    """Draw the grid figure sets and the experiment figures that have complete inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    long, short = load_results(results_dir / "benchmark_ccc_long.npz"), load_results(results_dir / "benchmark_ccc.npz")
    if has_all_estimators(long) and has_all_estimators(short):
        runs = [Run("long run", long, "o", True), Run("short run", short, "s", False)]
        written += grid_figures("wide", long, runs, out_dir)
    box = default_grid_results(results_dir)
    if has_all_estimators(box):
        written += grid_figures("box", box, [Run("default grid", box, "o", True)], out_dir)
    sweeps = [results_dir / f"{n}.npz" for n in ("exp_noise", "exp_dt", "exp_realisations")]
    if all(p.exists() for p in sweeps):
        noise, dt, real = (load_results(p) for p in sweeps)
        sensitivity_figure(noise, dt, real, box, out_dir / "sensitivity.png")
        written.append(out_dir / "sensitivity.png")
    if (results_dir / "exp_gc_lag_statespace.npz").exists():
        lag_order_figure(load_results(results_dir / "exp_gc_lag_statespace.npz"), out_dir / "gc_lag_statespace.png")
        written.append(out_dir / "gc_lag_statespace.png")
    nums = report_numbers(results_dir)
    for name, fn in (
        ("overview", overview_figure),
        ("overview_axes", axis_overview_figure),
        ("significance", significance_figure),
        ("confidence", confidence_figure),
    ):
        fn(nums, out_dir / f"{name}.png")
        written.append(out_dir / f"{name}.png")
    ccc_settings_figure(out_dir / "ccc_window_sweep.png")
    written.append(out_dir / "ccc_window_sweep.png")
    for p in written:
        print(f"wrote {p}")
    return written
