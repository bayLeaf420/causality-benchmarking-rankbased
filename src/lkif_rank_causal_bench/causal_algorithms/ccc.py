"""Adapter around the CCC implementation in src/algorithms/ccc (called unchanged)."""

# CCC is computed per realisation and averaged. Work is split into chunks of
# (grid point, realisation) pairs; with workers > 0 each chunk is saved on its
# own, so an interrupted run resumes where it stopped.

import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path

import numpy as np

from lkif_rank_causal_bench.causal_algorithms.base import CausalAlgorithm, Estimate
from lkif_rank_causal_bench.causal_algorithms.parallel import SINGLE_THREAD_ENV, environment, pool
from lkif_rank_causal_bench.config import ParameterGrid, SimulationConfig

_compiled: dict[tuple[int, int, int, int, float], object] = {}


class IncompleteRunError(RuntimeError):
    """Raised when a resumable run stopped on its time limit; run it again to continue."""


def ccc_chunk(pairs: np.ndarray, settings: tuple[int, int, int, int, float]) -> np.ndarray:
    """CCC of each (cause, effect) row pair, pairs of shape (n, 2, T) in float32."""
    import jax
    import jax.numpy as jnp

    from algorithms.ccc.ETC_and_CCC import CCC_calculation

    if settings not in _compiled:
        past, present, jump, bins, tol = settings
        _compiled[settings] = jax.jit(
            jax.vmap(lambda d: CCC_calculation(d, (past, present, jump), (bins, bins), tol)[0])
        )
    fn = _compiled[settings]
    return np.asarray(fn(jnp.asarray(pairs)))  # type: ignore[operator]


def _worker(task: tuple[str, str, str, int, int, int, tuple[int, int, int, int, float]]) -> float:
    """Compute and save one chunk in a worker process; returns its wall time."""
    pairs_path, parts, direction, start, count, n_real, settings = task
    pairs = np.load(pairs_path, mmap_mode="r")  # (n_points, n_real, 2, T): rows (source, target)
    order = [0, 1] if direction == "fwd" else [1, 0]  # cause first
    g, r = np.divmod(np.arange(start, start + count), n_real)
    data = np.stack([pairs[gi, ri][order] for gi, ri in zip(g, r, strict=True)]).astype(np.float32)
    t0 = time.time()
    out = ccc_chunk(data, settings)
    path = Path(parts) / f"{direction}_{start:06d}.npy"
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, out)
    tmp.rename(path)
    return time.time() - t0


class CCC(CausalAlgorithm):
    """Compression-Complexity Causality, per realisation, averaged over realisations."""

    name = "ccc"

    def __init__(
        self,
        past: int = 640,
        present: int = 64,
        jump: int = 200,
        bins: int = 8,
        tolerance: float = 1e-6,
        n_realisations: int | None = None,
        chunk: int = 32,
        workers: int = 0,
        parts_dir: str | None = None,
        stop_after: float | None = None,
    ) -> None:
        """Set the CCC windows and bins, realisations used, chunking and parallel execution."""
        self.settings = (past, present, jump, bins, tolerance)
        self.n_realisations = n_realisations
        self.chunk = chunk
        self.workers = workers
        self.parts_dir = parts_dir
        self.stop_after = stop_after

    def estimate(self, X: np.ndarray, source: int, target: int, indices: np.ndarray) -> Estimate:
        """Single direction; estimate_both is the efficient path."""
        forward, _ = self.estimate_both(X, source, target, indices, reverse=False)
        return forward

    def estimate_both(
        self, X: np.ndarray, source: int, target: int, indices: np.ndarray, reverse: bool = True
    ) -> tuple[Estimate, Estimate | None]:
        """Per-realisation CCC in both directions, in-process or in resumable worker processes."""
        R = X.shape[1] if self.n_realisations is None else min(self.n_realisations, X.shape[1])
        pairs = np.asarray(X[:, :R][:, :, [source, target]], dtype=np.float32)
        directions = ["fwd", "rev"] if reverse else ["fwd"]
        if self.workers > 0:
            results = self._parallel(pairs, directions)
        else:
            results = {d: self._in_process(pairs, d) for d in directions}
        estimates = {d: Estimate(values=v.mean(axis=1), per_realisation=v) for d, v in results.items()}
        return estimates["fwd"], estimates.get("rev")

    def _in_process(self, pairs: np.ndarray, direction: str) -> np.ndarray:
        """All chunks of one direction in this process, shape (n_points, R)."""
        n, R, _, T = pairs.shape
        order = [0, 1] if direction == "fwd" else [1, 0]
        flat = pairs[:, :, order].reshape(n * R, 2, T)
        out = [ccc_chunk(flat[s : s + self.chunk], self.settings) for s in range(0, n * R, self.chunk)]
        return np.concatenate(out).reshape(n, R)

    def _parallel(self, pairs: np.ndarray, directions: list[str]) -> dict[str, np.ndarray]:
        """All chunks in single-threaded workers, saved per chunk and resumable."""
        if self.parts_dir is None:
            raise ValueError("parallel CCC needs parts_dir for its per-chunk files")
        parts = Path(self.parts_dir)
        parts.mkdir(parents=True, exist_ok=True)
        for t in parts.glob("*.tmp.npy"):
            t.unlink()
        pairs_path = parts / "pairs.npy"
        if not pairs_path.exists():
            np.save(pairs_path, pairs)
        n, R = pairs.shape[:2]
        total = n * R
        tasks = [(d, s) for d in directions for s in range(0, total, self.chunk)]

        def done(d: str, s: int) -> bool:
            """Whether a chunk's result file exists."""
            return (parts / f"{d}_{s:06d}.npy").exists()

        queue = [t for t in tasks if not done(*t)]
        print(f"CCC {self.settings}: {len(tasks)} chunks, {len(queue)} to do, {self.workers} workers", flush=True)
        t_start = time.time()
        with environment(SINGLE_THREAD_ENV), pool(self.workers) as ex:
            running: set[Future[float]] = set()
            while queue or running:
                while (
                    queue
                    and len(running) < self.workers
                    and (self.stop_after is None or time.time() - t_start < self.stop_after)
                ):
                    d, s = queue.pop(0)
                    running.add(
                        ex.submit(
                            _worker, (str(pairs_path), str(parts), d, s, min(self.chunk, total - s), R, self.settings)
                        )
                    )
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    f.result()
        if not all(done(*t) for t in tasks):
            raise IncompleteRunError(f"CCC stopped after {self.stop_after}s; run again to continue")
        return {
            d: np.concatenate([np.load(parts / f"{d}_{s:06d}.npy") for s in range(0, total, self.chunk)]).reshape(n, R)
            for d in directions
        }

    def window_warnings(self, grid: ParameterGrid, sim: SimulationConfig) -> list[str]:
        """Warnings about CCC window geometry and the regions where CCC mis-ranks the direction."""
        past, present, jump, bins, _ = self.settings
        out = []
        if past + present > sim.tau[2]:
            out.append(
                f"ccc_past + ccc_present = {past + present} exceeds n_time = {sim.tau[2]}; CCC_calculation will raise."
            )
        elif present > past / 2:
            out.append(
                f"ccc_present = {present} is not much smaller than ccc_past = "
                f"{past}. CCC measures the complexity of a short present given a "
                "long past; equal windows lose the directional separation."
            )
        n_wins = (sim.tau[2] - (past + present)) // jump + 1
        if n_wins < 4:
            out.append(
                f"only {n_wins} CCC window(s) fit in n_time = {sim.tau[2]}; lower ccc_jump or lengthen the series."
            )
        if bins**2 > past / 2:
            out.append(
                f"ccc_bins^2 = {bins**2} joint symbols over a past of {past} samples is too fine; ETC saturates."
            )
        # measured: CCC's directional accuracy is governed by K and mu, not by
        # how many driver periods fit in the window (logs/ccc_window_length_comparison.log)
        if grid.K[0] < 1.0:
            out.append(
                f"K_min = {grid.K[0]:.3g} < 1: CCC ranks the reverse direction "
                "above the forward one at essentially every point with K < 1 "
                "(measured: 100% of points at K = 0.4, 70% at K = 0.85), and "
                "lengthening ccc_past does not fix it. Treat CCC results below "
                "K ~ 1 as unreliable, or restrict the grid to K >= 1."
            )
        if grid.mu[0] < 0.5:
            out.append(
                f"mu_min = {grid.mu[0]:.3g}: CCC also mis-ranks the direction at "
                "weak coupling (measured: 100% of points at mu = 0.25 in the long "
                "run). The forward signal there is within its own realisation scatter."
            )
        slowest_period = 2 * np.pi / max(grid.K[0], 1e-12)
        periods_in_window = (past * sim.dt) / slowest_period
        if periods_in_window < 2.0:
            needed = int(np.ceil(2.0 * slowest_period / sim.dt))
            out.append(
                f"ccc_past = {past} samples holds only {periods_in_window:.1f} "
                f"periods of the slowest driver (K_min = {grid.K[0]:.3g}), which "
                "drives the grid-median CCC negative. Raise ccc_past to about "
                f"{needed} (needing n_time > {needed + present})."
            )
        return out
