"""The Benchmark: one oscillator system, its LKIF truth, its samples, and any causal algorithm."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lkif_rank_causal_bench.causal_algorithms import CausalAlgorithm, build_algorithm
from lkif_rank_causal_bench.causal_algorithms.base import Estimate
from lkif_rank_causal_bench.config import AlgorithmSpec, SystemConfig
from lkif_rank_causal_bench.lkif.truth import lkif_truth
from lkif_rank_causal_bench.oscillator.checks import sampling_warnings
from lkif_rank_causal_bench.oscillator.grid import params_tensor
from lkif_rank_causal_bench.oscillator.model import TRUTH_ENTRY
from lkif_rank_causal_bench.oscillator.sampler import sample_grid


def estimate_arrays(
    spec: AlgorithmSpec, forward: Estimate, backward: Estimate | None, shape: tuple[int, ...]
) -> dict[str, np.ndarray]:
    """Result arrays for one algorithm under its configured keys (grid-shaped values)."""
    out: dict[str, np.ndarray] = {spec.key: forward.values.reshape(shape)}
    pairs: list[tuple[str, Estimate]] = [(spec.key, forward)]
    if backward is not None and spec.reverse_key:
        out[spec.reverse_key] = backward.values.reshape(shape)
        pairs.append((spec.reverse_key, backward))
    for key, est in pairs:
        if est.p_values is not None:
            out[f"{key}_p"] = est.p_values.reshape(shape)
        if est.per_realisation is not None:
            pr = est.per_realisation
            out[f"{key}_per_realisation"] = pr
            out[f"{key}_sem"] = (pr.std(axis=1, ddof=1) / np.sqrt(pr.shape[1])).reshape(shape)
    return out


class Benchmark:
    """Truth and estimates for one system; swap the algorithm, keep everything else fixed."""

    def __init__(self, system: SystemConfig) -> None:
        """Hold the system; truth and samples are computed lazily and cached."""
        self.system = system
        self._truth: dict[str, np.ndarray] | None = None
        self._samples: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        """Grid shape (n_beta, n_K, n_mu)."""
        return self.system.grid.shape

    def params(self) -> np.ndarray:
        """The (beta, K, mu) tensor of the grid."""
        return np.asarray(params_tensor(self.system.grid))

    def warnings(self) -> list[str]:
        """Sampling-rate and coverage warnings for this system."""
        return sampling_warnings(self.system.grid, self.system.simulation)

    def truth(self) -> dict[str, np.ndarray]:
        """LKIF truth arrays, computed once."""
        if self._truth is None:
            self._truth = lkif_truth(self.system)
        return self._truth

    def samples(self) -> np.ndarray:
        """Samples in estimator layout (n_grid, R, 4, n_time), computed once."""
        if self._samples is None:
            self._samples = sample_grid(self.system.grid, self.system.simulation)
        return self._samples

    def use_samples(self, X: np.ndarray) -> None:
        """Replace the cached samples, e.g. a realisation subset or surrogate data."""
        self._samples = X

    def estimate(
        self, algorithm: CausalAlgorithm, spec: AlgorithmSpec, points: Sequence[int] | None = None
    ) -> dict[str, np.ndarray]:
        """Run one algorithm source x2 -> target v1 (and reverse), on all or selected points."""
        X = self.samples()
        indices = np.arange(X.shape[0]) if points is None else np.asarray(points)
        source, target = TRUTH_ENTRY
        forward, backward = algorithm.estimate_both(
            X[indices], source, target, indices, reverse=spec.reverse_key is not None
        )
        shape = self.shape if points is None else (len(indices),)
        out = estimate_arrays(spec, forward, backward, shape)
        out.update({k: v.reshape(shape) for k, v in algorithm.derived(forward).items()})
        return out

    def run(self, specs: Sequence[AlgorithmSpec]) -> dict[str, np.ndarray]:
        """Truth plus every algorithm's results over the whole grid."""
        results: dict[str, np.ndarray] = {"params_tensor": self.params(), **self.truth()}
        for spec in specs:
            print(f"  {spec.key}: {spec.algorithm} {spec.params}", flush=True)
            results.update(self.estimate(build_algorithm(spec.algorithm, spec.params), spec))
        return results

    def save_samples(self, path: str | Path) -> None:
        """Write the samples to a .npy file (used by resumable CCC and later runs)."""
        np.save(path, self.samples())
