"""Command line: run a config, make the report figures, or compute the report numbers."""

import argparse
import time
from collections.abc import Sequence

from lkif_rank_causal_bench.causal_algorithms.ccc import IncompleteRunError
from lkif_rank_causal_bench.config import load_run_config


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the run, figures and numbers sub-commands."""
    parser = argparse.ArgumentParser(prog="python -m lkif_rank_causal_bench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one or more YAML configs in order")
    run.add_argument("configs", nargs="+", help="paths to configs/*.yaml")
    sub.add_parser("figures", help="write docs/figures/*.png from results/")
    sub.add_parser("numbers", help="write docs/report_numbers.json from results/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a sub-command; returns the process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "run":
        from lkif_rank_causal_bench.experiments import run_experiment

        for path in args.configs:
            cfg = load_run_config(path)
            t0 = time.time()
            print(f"=== {cfg.name} ({cfg.experiment}) -> {cfg.output}", flush=True)
            try:
                out = run_experiment(cfg)
            except IncompleteRunError as e:
                print(f"INCOMPLETE: {e}")
                return 0
            print(f"=== {cfg.name}: wrote {out} in {(time.time() - t0) / 60:.1f} min", flush=True)
    elif args.command == "figures":
        from lkif_rank_causal_bench.plotting.report_figures import make_all_figures

        make_all_figures()
    elif args.command == "numbers":
        from lkif_rank_causal_bench.analysis.report_numbers import write_report_numbers

        print(f"wrote {write_report_numbers()}")
    return 0
