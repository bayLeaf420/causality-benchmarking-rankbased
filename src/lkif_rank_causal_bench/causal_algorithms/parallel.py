"""Process pool with a controlled environment, shared by the slow per-point algorithms."""

import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager

# One thread per worker: XLA for CCC, BLAS for PCMCI. Measured faster per call
# for the sequential ETC loop (claude_tests/logs/ccc_cpu_timing_long*.log).
SINGLE_THREAD_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def physical_cores() -> int:
    """Half the logical CPU count: one worker per physical core on this machine."""
    return max(1, (os.cpu_count() or 2) // 2)


@contextmanager
def environment(env: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables (inherited by spawned workers)."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def pool(workers: int) -> ProcessPoolExecutor:
    """A spawn-context process pool (fresh interpreters pick up the environment)."""
    return ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"))


def map_in_pool[T, R](fn: Callable[[T], R], tasks: Iterable[T], workers: int) -> list[R]:
    """Run fn over tasks in single-threaded worker processes, keeping task order."""
    with environment(SINGLE_THREAD_ENV), pool(workers) as ex:
        return list(ex.map(fn, tasks))
