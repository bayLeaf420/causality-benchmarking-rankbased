"""Typed run configuration, loaded from YAML files in configs/."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from lkif_rank_causal_bench.paths import CONFIGS

ALGORITHM_DIR = CONFIGS / "algorithms"


@dataclass(frozen=True)
class ParameterGrid:
    """The (beta, K, mu) grid, each axis given as (start, stop, num) for linspace."""

    beta: tuple[float, float, int] = (0.25, 2.25, 6)
    K: tuple[float, float, int] = (1.0, 2.6, 6)
    mu: tuple[float, float, int] = (0.5, 2.5, 6)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of points along beta, K and mu."""
        return (int(self.beta[2]), int(self.K[2]), int(self.mu[2]))

    @property
    def size(self) -> int:
        """Total number of grid points."""
        n_b, n_k, n_m = self.shape
        return n_b * n_k * n_m

    def ranges(self) -> dict[str, tuple[float, float, int]]:
        """The three axis ranges keyed by parameter name."""
        return {"beta": self.beta, "K": self.K, "mu": self.mu}


@dataclass(frozen=True)
class SimulationConfig:
    """How trajectories are generated at every grid point."""

    tau: tuple[float, float, int] = (0.0, 150.0, 1500)
    mean_init: tuple[float, float, float, float] = (1.0, 0.0, -1.3, 0.0)
    sigma_init: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    q1: float = 0.1
    q2: float = 0.1
    # 0.0 keeps the driver a pure SHO (intended physics, non-stationary process)
    beta2: float = 0.0
    # True samples the stationary law (needs beta2 > 0) and scores the stationary LKIF
    stationary: bool = False
    burn_in: int = 0
    n_realisations: int = 40
    seed: int = 0
    # float64 reproduces the default-grid and wide-grid runs, float32 the experiment runs
    sample_dtype: str = "float32"
    # realisations drawn per compiled call; only peak memory depends on it
    realisation_block: int = 8

    @property
    def dt(self) -> float:
        """Sampling interval of the time grid."""
        return (self.tau[1] - self.tau[0]) / (self.tau[2] - 1)


@dataclass(frozen=True)
class SystemConfig:
    """Grid plus simulation: everything that defines the data and the LKIF truth."""

    grid: ParameterGrid = field(default_factory=ParameterGrid)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


@dataclass(frozen=True)
class AlgorithmSpec:
    """One causal algorithm, its hyperparameters and the result keys it writes."""

    algorithm: str
    key: str
    reverse_key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """A complete run: system, algorithms, experiment type and output file."""

    name: str
    experiment: str
    output: str
    system: SystemConfig = field(default_factory=SystemConfig)
    algorithms: tuple[AlgorithmSpec, ...] = ()
    save_samples: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two config dictionaries; override wins."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file and resolve its chain of `extends:` parents."""
    path = Path(path)
    data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    parent = data.pop("extends", None)
    if parent is None:
        return data
    return _merge(read_yaml(path.parent / parent), data)


def _tuples(d: dict[str, Any], cls: type) -> dict[str, Any]:
    """Convert YAML lists to tuples for the tuple-typed fields of a dataclass."""
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}


def system_from_dict(d: dict[str, Any]) -> SystemConfig:
    """Build a SystemConfig from its YAML dictionary."""
    grid = ParameterGrid(**_tuples(d.get("grid", {}), ParameterGrid))
    simulation = SimulationConfig(**_tuples(d.get("simulation", {}), SimulationConfig))
    return SystemConfig(grid=grid, simulation=simulation)


def algorithm_from_entry(entry: str | dict[str, Any]) -> AlgorithmSpec:
    """Resolve a run's algorithm entry: a configs/algorithms name, optionally with overrides."""
    if isinstance(entry, str):
        entry = {"use": entry}
    entry = dict(entry)
    base: dict[str, Any] = read_yaml(ALGORITHM_DIR / f"{entry.pop('use')}.yaml") if "use" in entry else {}
    merged = _merge(base, entry)
    return AlgorithmSpec(
        algorithm=merged["algorithm"],
        key=merged["key"],
        reverse_key=merged.get("reverse_key"),
        params=dict(merged.get("params", {})),
    )


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run configuration from a YAML file."""
    d = read_yaml(path)
    return RunConfig(
        name=d["name"],
        experiment=d["experiment"],
        output=d["output"],
        system=system_from_dict(d.get("system", {})),
        algorithms=tuple(algorithm_from_entry(e) for e in d.get("algorithms", [])),
        save_samples=d.get("save_samples"),
        options=dict(d.get("options", {})),
    )


def with_simulation(system: SystemConfig, **changes: Any) -> SystemConfig:
    """Copy of a SystemConfig with some simulation fields changed."""
    return replace(system, simulation=replace(system.simulation, **changes))


def system_as_dict(system: SystemConfig) -> dict[str, Any]:
    """Plain-dictionary form of a SystemConfig, for storing next to results."""
    return asdict(system)
