from __future__ import annotations

import argparse
import os
import pandas as pd

from core_utils import ensure_dir, iter_analysis_units, load_inventory, analyze_unit


def main():
    ap = argparse.ArgumentParser(description="Run session-object-event analyses across the full dataset.")
    ap.add_argument("--base-dir", required=True, help="Root HDF5 directory containing Events/ and Spikes/")
    ap.add_argument("--go-nogo-dir", required=True, help="Directory with inventory CSVs")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--min-neurons", type=int, default=3)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--gauss-smooth", type=float, default=None)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    inv = load_inventory(args.go_nogo_dir)

    rows = []
    for unit in iter_analysis_units(inv["summary"], events=("Event1", "Event3")):
        print(f"Processing {unit.area} | {unit.session} | {unit.object_} | {unit.event}")
        row = analyze_unit(
            unit,
            base_dir=args.base_dir,
            min_neurons=args.min_neurons,
            n_perm=args.n_perm,
            n_boot=args.n_boot,
            gauss_smooth=args.gauss_smooth,
        )
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "session_object_event_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    main()
