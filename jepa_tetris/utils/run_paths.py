"""Resolve per-run output directories under results/."""
from __future__ import annotations

import time
from pathlib import Path

RESULTS_ROOT = Path("results")


def run_dir(run_name: str | None, results_root: Path | str = RESULTS_ROOT) -> Path:
    """Return results/<YYYYMMDD-HHMMSS>[_<run_name>]/, creating it on disk."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    folder = ts if not run_name else f"{ts}_{run_name}"
    path = Path(results_root) / folder
    path.mkdir(parents=True, exist_ok=True)
    return path
