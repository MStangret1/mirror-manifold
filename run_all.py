#!/usr/bin/env python3
"""
run_all.py - conservative driver for the mirror-manifold analysis pipeline.

It runs the main numbered pipeline scripts in order, but ONLY the ones that
actually exist. Missing scripts are skipped with a warning instead of failing.

Raw HDF5 data are required only for the scripts that actually read them
(the batch/decoding/dPCA steps). If you only want to reproduce summaries and
figures from the shipped CSVs, you do not need the raw data at all.

Usage
-----
  python run_all.py --data-root path/to/HDF5        # full run (needs raw data)
  python run_all.py --dry-run                       # list steps, run nothing
  python run_all.py --only 04_meta_summary          # run a single step

The raw-data location is resolved in this order:
  1. --data-root argument
  2. DATA_ROOT environment variable
  3. data/raw/HDF5   (default, relative to the repo root)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
PIPELINE_DIR = REPO_ROOT / "pipeline"
SRC_DIR = REPO_ROOT / "src"

# Ordered list of pipeline steps. `needs_data` marks steps that read raw HDF5.
STEPS = [
    ("00_inventory.py",                      False),
    ("01_batch_session_object_analysis.py",  True),
    ("02_batch_time_resolved.py",            True),
    ("03_object_generalization.py",          True),
    ("04_meta_summary.py",                   False),
    ("05_mixed_effects.py",                  False),
    ("06_dpca_analysis.py",                  True),
    # Additional steps present in this repo are run if found:
    ("07_shuffle_null_model.py",             True),
    ("08_divergence_onset.py",               False),
    ("09_cross_event_generalization.py",     True),
    ("10_neuron_count_sensitivity.py",       False),
    ("11_supplementary_stats.py",            False),
]


def resolve_data_root(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)
    env = os.environ.get("DATA_ROOT")
    if env:
        return Path(env)
    return REPO_ROOT / "data" / "raw" / "HDF5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="Root HDF5 directory (Spikes/ and Events/).")
    ap.add_argument("--dry-run", action="store_true", help="List the steps that would run; run nothing.")
    ap.add_argument("--only", default=None, help="Run only the step whose filename starts with this string.")
    ap.add_argument("--continue-on-error", action="store_true", help="Keep going if a step fails.")
    args = ap.parse_args()

    data_root = resolve_data_root(args.data_root)
    data_available = data_root.exists()

    print("=" * 70)
    print("mirror-manifold :: run_all.py")
    print(f"  repo root : {REPO_ROOT}")
    print(f"  data root : {data_root}  ({'found' if data_available else 'NOT found'})")
    print("=" * 70)

    # Make src/ importable by the pipeline scripts (from core_utils import ...).
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC_DIR), str(PIPELINE_DIR), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["DATA_ROOT"] = str(data_root)

    failures = []
    for name, needs_data in STEPS:
        if args.only and not name.startswith(args.only):
            continue
        script = PIPELINE_DIR / name
        if not script.exists():
            print(f"[SKIP ] {name:<42} (not present in pipeline/)")
            continue
        if needs_data and not data_available:
            print(f"[SKIP ] {name:<42} (needs raw data; --data-root not found)")
            continue

        cmd = [sys.executable, str(script)]
        if needs_data:
            cmd += ["--base-dir", str(data_root)]

        print(f"[RUN  ] {name}")
        if args.dry_run:
            print(f"         would run: {' '.join(cmd)}")
            continue

        result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            print(f"[FAIL ] {name} (exit {result.returncode})")
            failures.append(name)
            if not args.continue_on_error:
                print("\nStopping (use --continue-on-error to keep going).")
                return 1
        else:
            print(f"[OK   ] {name}")

    print("=" * 70)
    if failures:
        print(f"Completed with {len(failures)} failing step(s): {', '.join(failures)}")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
