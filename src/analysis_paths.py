# analysis_paths.py
#
# Central path resolution for the mirror-manifold pipeline.
#
# Raw-data location is resolved as:
#   1) the DATA_ROOT environment variable, if set;
#   2) <repo_root>/data/raw/HDF5 otherwise.
# Result tables live in <repo_root>/results. Nothing is hard-coded to a
# specific user's machine.
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root(script_path: str | Path) -> Path:
    """Repo root, given a script that lives one level down (pipeline/, figures/, src/)."""
    return Path(script_path).resolve().parent.parent


def data_root(repo_root: Path) -> Path:
    """Raw HDF5 root: DATA_ROOT env var, else <repo>/data/raw/HDF5."""
    env = os.environ.get("DATA_ROOT")
    return Path(env) if env else (repo_root / "data" / "raw" / "HDF5")


@dataclass(frozen=True)
class AnalysisPaths:
    project_dir: Path
    base_dir: Path
    go_nogo_dir: Path
    results_dir: Path
    final_dir: Path
    session_results_csv: Path
    time_resolved_csv: Path
    object_generalization_csv: Path


def infer_default_paths(script_path: str | Path) -> AnalysisPaths:
    """Resolve sensible, machine-independent defaults for a pipeline/figure script."""
    root = _repo_root(script_path)
    results_dir = root / "results"
    final_dir = results_dir / "final_analysis"
    base_dir = data_root(root)

    return AnalysisPaths(
        project_dir=root,
        base_dir=base_dir,
        # Inventory CSVs and result tables both live under results/ in the clean repo.
        go_nogo_dir=results_dir,
        results_dir=results_dir,
        final_dir=final_dir,
        session_results_csv=results_dir / "session_object_event_results.csv",
        time_resolved_csv=results_dir / "time_resolved_results.csv",
        object_generalization_csv=results_dir / "object_generalization_results.csv",
    )


def resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    return Path(value)
