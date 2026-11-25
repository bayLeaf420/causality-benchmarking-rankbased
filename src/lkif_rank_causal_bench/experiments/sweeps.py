"""Sensitivity runs: change one simulation setting, or the number of realisations, and rescore."""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from lkif_rank_causal_bench.analysis.agreement import summary_line
from lkif_rank_causal_bench.benchmark import Benchmark
from lkif_rank_causal_bench.causal_algorithms import build_algorithm
from lkif_rank_causal_bench.config import RunConfig, with_simulation
from lkif_rank_causal_bench.results import load_results, save_results


def _saved(path: Path) -> dict[str, np.ndarray]:
    """The 'label/key' arrays already in a sweep file (empty if none)."""
    return {k: v for k, v in load_results(path).items() if "/" in k} if path.exists() else {}


def run_setting_sweep(cfg: RunConfig) -> Path:
    """For each value of options.fields (e.g. q1 and q2, or tau), rerun truth and algorithms."""
    opts = cfg.options
    fields: list[str] = opts["fields"]
    values: list[Any] = opts["values"]
    labels = [str(x) for x in opts.get("labels", values)]
    out = Path(cfg.output)
    saved = _saved(out)
    for value, label in zip(values, labels):
        if any(k.startswith(f"{label}/") for k in saved):
            print(f"{label}: already saved, skipped")
            continue
        changes = {f: tuple(value) if isinstance(value, list) else value for f in fields}
        bench = Benchmark(with_simulation(cfg.system, **changes))
        res = bench.truth() | {}
        for spec in cfg.algorithms:
            res.update(bench.estimate(build_algorithm(spec.algorithm, spec.params), spec))
        saved.update({f"{label}/{k}": v for k, v in res.items()})
        for spec in cfg.algorithms:
            rev = res.get(spec.reverse_key) if spec.reverse_key else None
            print(f"{label}: " + summary_line(spec.key, res["truth"], res[spec.key], rev), flush=True)
        save_results(out, {opts.get("index_key", "values"): np.array(labels, dtype=float), **saved})
    return out


def run_realisation_sweep(cfg: RunConfig) -> Path:
    """Score the algorithms on the first R realisations of one sample, for each R in options.subsets."""
    opts = cfg.options
    bench = Benchmark(cfg.system)
    truth = bench.truth()["truth"]
    full = bench.samples()
    out: dict[str, np.ndarray] = {"truth": truth}
    for R in opts["subsets"]:
        bench.use_samples(full[:, :R])
        for spec in cfg.algorithms:
            res = bench.estimate(build_algorithm(spec.algorithm, spec.params), spec)
            out.update({f"{R}/{k}": v for k, v in res.items()})
            rev = res.get(spec.reverse_key) if spec.reverse_key else None
            print(f"R = {R}: " + summary_line(spec.key, truth, res[spec.key], rev), flush=True)
        save_results(cfg.output, {"rs": np.array(opts["subsets"]), **out})
    ccc_from = opts.get("ccc_from")
    if ccc_from and Path(ccc_from).exists():
        # CCC realisation subsets come from a stored per-realisation run (an
        # 80-realisation CCC run would cost about two hours)
        d = load_results(ccc_from)
        shape = d["truth"].shape
        for R in opts.get("ccc_subsets", []):
            fwd = d["ccc_per_realisation"][:, :R].mean(1).reshape(shape)
            rev = d["ccc_reverse_per_realisation"][:, :R].mean(1).reshape(shape)
            out[f"{R}/ccc"], out[f"{R}/ccc_reverse"] = fwd, rev
            print(summary_line(f"CCC, R = {R}", d["truth"], fwd, rev))
        out["ccc_truth"] = d["truth"]
    return save_results(cfg.output, {"rs": np.array(opts["subsets"]), **out})


def system_for_label(cfg: RunConfig, **changes: Any) -> RunConfig:
    """Copy of a run config with simulation fields changed (helper for custom sweeps)."""
    return replace(cfg, system=with_simulation(cfg.system, **changes))
