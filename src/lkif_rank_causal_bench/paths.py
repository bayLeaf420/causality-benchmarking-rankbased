"""Repository locations used as defaults for configs, results and report files."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
RESULTS = REPO / "results"
DOCS = REPO / "docs"
FIGURES = DOCS / "figures"
