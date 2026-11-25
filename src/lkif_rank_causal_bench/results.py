"""Reading and writing result files (.npz with arrays plus the run configuration)."""

from pathlib import Path
from typing import Any

import numpy as np


def load_results(path: str | Path) -> dict[str, Any]:
    """Every array in a result file; 'config' and 'warnings' come back as Python objects."""
    with np.load(path, allow_pickle=True) as f:
        out: dict[str, Any] = {k: f[k] for k in f.files}
    for key in ("config", "method"):
        if key in out and out[key].dtype == object:
            out[key] = out[key].item()
    if "warnings" in out:
        out["warnings"] = list(out["warnings"])
    return out


def save_results(path: str | Path, arrays: dict[str, Any], config: dict[str, Any] | None = None) -> Path:
    """Write arrays (and optionally the run configuration) to one compressed .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = dict(arrays)
    if config is not None:
        payload["config"] = np.array(config, dtype=object)
    np.savez_compressed(path, **payload)
    return path


def update_results(path: str | Path, arrays: dict[str, Any]) -> Path:
    """Add or replace arrays in an existing result file (created if missing)."""
    current = load_results(path) if Path(path).exists() else {}
    config = current.pop("config", None)
    current.update(arrays)
    return save_results(path, current, config)


def merge_files(base: dict[str, Any], path: str | Path, keys: tuple[str, ...]) -> dict[str, Any]:
    """Copy selected arrays from another result file into a results dict, if present."""
    if Path(path).exists():
        other = load_results(path)
        if "done" not in other or bool(np.all(other["done"])):
            base.update({k: other[k] for k in keys if k in other})
    return base
