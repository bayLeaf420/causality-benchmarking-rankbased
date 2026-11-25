"""Grid run: LKIF truth plus every configured causal algorithm over one system."""

from pathlib import Path

from lkif_rank_causal_bench.analysis.agreement import summary_line
from lkif_rank_causal_bench.benchmark import Benchmark
from lkif_rank_causal_bench.causal_algorithms import build_algorithm
from lkif_rank_causal_bench.config import RunConfig, system_as_dict
from lkif_rank_causal_bench.results import load_results, save_results, update_results


def run_grid(cfg: RunConfig) -> Path:
    """Compute truth and algorithms, saving after each one; algorithms already saved are skipped."""
    bench = Benchmark(cfg.system)
    for w in bench.warnings():
        print(f"[warning] {w}")
    out = Path(cfg.output)
    done = set(load_results(out)) if out.exists() else set()
    if "truth" not in done:
        save_results(out, {"params_tensor": bench.params(), **bench.truth()}, system_as_dict(cfg.system))
    if cfg.save_samples and not Path(cfg.save_samples).exists():
        bench.save_samples(cfg.save_samples)
    for spec in cfg.algorithms:
        if spec.key in done:
            print(f"  {spec.key}: already saved, skipped")
            continue
        print(f"  {spec.key}: {spec.algorithm} {spec.params}", flush=True)
        update_results(out, bench.estimate(build_algorithm(spec.algorithm, spec.params), spec))
    res = load_results(out)
    for spec in cfg.algorithms:
        rev = res.get(spec.reverse_key) if spec.reverse_key else None
        print(summary_line(spec.key, res["truth"], res[spec.key], rev))
    return out
