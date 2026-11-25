# lkif-rank-causal-bench

A benchmark that tests whether heuristic causality measures rank systems in the
same order as a known ground truth.

The test system is a damped oscillator (x₁, v₁) driven in one direction by a
second oscillator (x₂, v₂). The system is linear, so the true information flow
from x₂ to v₁ is known exactly: it is the Liang–Kleeman information flow (LKIF).
The benchmark computes the LKIF over a grid of system parameters, samples noisy
trajectories at every grid point, and scores each causality measure by how well
it reproduces the LKIF ranking of the grid (Kendall's τ). The reverse direction,
v₁ → x₂, has zero true flow, so it measures false positives.

Measures included:

| Measure | Implementation |
|---|---|
| Granger causality (conditional and bivariate) | least-squares regression, in JAX |
| Exact Granger causality of the model | Kalman filter on the known VAR(1) |
| Transfer entropy (Kraskov estimator) | [IDTxl](https://github.com/pwollstadt/IDTxl) |
| Compression-Complexity Causality (CCC) | JAX |
| PCMCI (partial-correlation test) | [tigramite](https://github.com/jakobrunge/tigramite) |
| Convergent cross mapping (CCM) | NumPy / SciPy |

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/bayLeaf420/causality-benchmarking-rankbased.git
cd causality-benchmarking-rankbased
uv sync
```

`uv sync` installs the package with its dependencies, including IDTxl (from its
GitHub repository, as it is not on PyPI), tigramite, pytest, ruff and pyright.

## Usage

Every run is described by a YAML file in `configs/`:

```bash
python -m lkif_rank_causal_bench run configs/default_grid.yaml
python -m lkif_rank_causal_bench run configs/noise_sweep.yaml configs/dt_sweep.yaml   # several, in order
```

Results are written as `.npz` files in `results/`. A grid run saves after each
causal algorithm and skips algorithms that are already in its output file, so an
interrupted run continues where it stopped.

The package can also be used from Python:

```python
from lkif_rank_causal_bench.benchmark import Benchmark
from lkif_rank_causal_bench.config import algorithm_from_entry, load_run_config

system = load_run_config("configs/default_grid.yaml").system
results = Benchmark(system).run([algorithm_from_entry("gc"), algorithm_from_entry("te_kraskov")])
results["truth"], results["gc_conditional"], results["te_conditional"]   # arrays shaped like the grid
```

## Configuration

There are three kinds of YAML file:

* **`configs/base.yaml`** defines the default system:
  - the parameter grid: damping β, frequency ratio K and coupling μ, each as `[start, stop, number]`;
  - the time window;
  - the noise power;
  - the number of realisations;
  - the random seed.
* **`configs/<run>.yaml`** defines one run:
  - a name, an experiment type, an output file;
  - the algorithms to use;
  - any changes to the system.

  A run file inherits from another file with `extends:`, and only the fields it
  sets are overridden.
* **`configs/algorithms/<name>.yaml`** holds the hyperparameters of one causal
  algorithm, such as the window lengths and number of bins of CCC, or the lag
  order of Granger causality. It also sets the keys under which results are saved.

A run file lists algorithms by file name, optionally with overrides:

```yaml
extends: base.yaml
name: example
experiment: grid
output: results/example.npz
system:
  simulation: {q1: 0.3, q2: 0.3}       # change the noise power
algorithms:
  - gc                                  # configs/algorithms/gc.yaml as it is
  - use: ccc                            # configs/algorithms/ccc.yaml with two changes
    params: {n_realisations: 10, workers: 4, parts_dir: results/example_ccc}
```

Experiment types:

| `experiment:` | What it does | Example config |
|---|---|---|
| `grid` | LKIF truth and every listed algorithm over one grid | `default_grid.yaml`, `wide_long.yaml`, `stationary.yaml`, `outside_box.yaml` |
| `setting_sweep` | repeats the grid run for several values of a simulation setting | `noise_sweep.yaml`, `dt_sweep.yaml` |
| `realisation_sweep` | scores the algorithms on the first R realisations, for several R | `realisations.yaml` |
| `lag_order` | Granger causality at several lag orders, and the exact GC | `lag_order.yaml` |
| `bootstrap` | spread of τ over resampled realisations | `bootstrap.yaml`, `bootstrap_te.yaml`, `bootstrap_ccm.yaml`, `bootstrap_pcmci.yaml` |
| `significance`, `significance_subgrid` | significance tests in both directions (false-positive rates) | `false_positives.yaml`, `false_positives_te.yaml` |
| `significance_algorithm` | one algorithm on the whole grid: paired t-tests of its per-realisation values and trial-shuffle surrogates | `false_positives_ccm.yaml` |

## Project structure

```
configs/                         YAML run configurations and algorithm hyperparameters
src/lkif_rank_causal_bench/      the benchmark package
  config.py                      typed configuration dataclasses and the YAML loader
  oscillator/                    the test system
    model.py                     drift matrix A and noise covariance G
    grid.py                      the (beta, K, mu) parameter tensor
    sampler.py                   exact VAR(1) sampling of trajectories
    checks.py                    warnings about unsuitable sampling choices
  lkif/                          the ground truth
    flow.py                      raw and normalised LKIF from A and the covariance
    transient.py                 time-averaged LKIF from the moment ODE (diffrax)
    stationary.py                LKIF of the stationary process (Lyapunov equation)
    truth.py                     the truth arrays for a whole grid
  causal_algorithms/             the causality measures, one class each
    base.py                      the CausalAlgorithm interface and the Estimate result
    granger.py, granger_exact.py, transfer_entropy.py, ccc.py, pcmci.py, ccm.py
    parallel.py                  process pool for the slow per-point measures
  benchmark.py                   Benchmark: system + truth + samples + any algorithm
  experiments/                   the experiment types in the table above
  analysis/                      rank-agreement statistics and the report numbers
  plotting/                      the report figures
  results.py, paths.py, cli.py   result files, default paths, command line
src/algorithms/ccc/              CCC and Effort-To-Compress implementation (used by causal_algorithms/ccc.py)
src/algorithms/tc.py             symbolic LKIF (SymPy), kept as an independent check
tests/                           pytest suite, one file per package
claude_tests/                    diagnostic scripts and their logs (index in claude_tests/INDEX.md)
docs/                            report builder, figures and report
results/                         result files (not tracked by git)
```

## How the code fits together

```
configs/*.yaml ──▶ RunConfig ──▶ experiments/<type>.py
                                        │
                                 Benchmark(system)
                    ┌───────────────────┼─────────────────────┐
                    ▼                   ▼                     ▼
          oscillator/sampler.py   lkif/truth.py     causal_algorithms/*
          samples (n_grid, R, 4, T)  truth arrays   one Estimate per point
                    └───────────────────┴──────────▶ results.py (.npz)
```

Every causality measure implements one method:

```python
estimate(X, source, target, indices) -> Estimate
```

`X` holds the samples of some grid points, with shape `(n_points, realisations,
4, time)`, in the state order (x₁, v₁, x₂, v₂). `indices` gives their positions in
the grid. Measures that get both directions in one pass (PCMCI, CCM, CCC) also
override `estimate_both`. CCC and CCM also return one value per realisation,
which the paired t-tests and the cheap bootstrap use.

### Adding a causality measure

1. Write a subclass of `CausalAlgorithm` in `causal_algorithms/`, with a `name`
   and an `estimate` method.
2. Add it to `REGISTRY` in `causal_algorithms/__init__.py`.
3. Add `configs/algorithms/<name>.yaml` with its hyperparameters and result keys.
4. List it under `algorithms:` in a run config.

## Reading the code

For a first read, follow the data from the configuration to the figures. Each
module uses only what the earlier ones define.

1. `config.py`, then `configs/base.yaml` and `configs/algorithms/`
2. `oscillator/`: `model.py`, `grid.py`, `sampler.py`, `checks.py`
3. `lkif/`: `flow.py`, `transient.py`, `stationary.py`, `truth.py`
4. `causal_algorithms/`:
   - `base.py`, then the estimators: `granger.py`, `granger_exact.py`,
     `transfer_entropy.py`, `ccc.py` (with `src/algorithms/ccc/`), `pcmci.py`, `ccm.py`;
   - then `parallel.py` and `__init__.py`.
5. `benchmark.py`
6. `results.py`, then `experiments/`: `grid.py`, `sweeps.py`, `lag_order.py`,
   `bootstrap.py`, `significance.py`
7. `analysis/`: `agreement.py`, `report_numbers.py`
8. `plotting/`: `style.py`, `grid_figures.py`, `experiment_figures.py`, `report_figures.py`
9. `cli.py`

The tests in `tests/` follow the same order and state what each part must do.

## Reproducibility

* **Seeds.** Every run is seeded from its config. The same config gives the same
  samples, and so the same results.
* **GC batching.** Granger causality is computed in batches of grid points
  (`chunk` in `configs/algorithms/gc.yaml`). A different batch size changes
  results at the 1e-16 level.
* **XLA threading.** JAX results depend on XLA's threading settings. The parallel
  CCC workers run single-threaded, and CCC is not affected by this.
* **Equivalence check.** `claude_tests/refactor_equivalence.py` recomputes a
  sample of stored results and checks that they are bitwise identical.
