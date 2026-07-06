from __future__ import annotations

import argparse
import os
import pandas as pd

from core_utils import (
    EVENT_WINDOWS,
    ensure_dir,
    build_tensor,
    iter_analysis_units,
    load_inventory,
    time_resolved_decode,
)


def main():
    ap = argparse.ArgumentParser(description="Time-resolved decoding across all session/object/event units.")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--go-nogo-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-neurons", type=int, default=3)
    ap.add_argument("--window-bins", type=int, default=5)
    ap.add_argument("--step-bins", type=int, default=1)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    inv = load_inventory(args.go_nogo_dir)
    rows = []

    for unit in iter_analysis_units(inv["summary"], events=("Event1", "Event3")):
        obs_go = build_tensor(args.base_dir, unit.area, unit.session, "Context2", "Condition1", unit.object_, unit.event, unit.interval, min_neurons=args.min_neurons)
        obs_nogo = build_tensor(args.base_dir, unit.area, unit.session, "Context2", "Condition2", unit.object_, unit.event, unit.interval, min_neurons=args.min_neurons)
        if obs_go is None or obs_nogo is None:
            continue
        df = time_resolved_decode(obs_go, obs_nogo, unit.interval, window_bins=args.window_bins, step_bins=args.step_bins)
        df.insert(0, "event", unit.event)
        df.insert(0, "object", unit.object_)
        df.insert(0, "session", unit.session)
        df.insert(0, "area", unit.area)
        rows.append(df)
        print(f"Done {unit.area} | {unit.session} | {unit.object_} | {unit.event}")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_csv = os.path.join(args.out_dir, "time_resolved_results.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
