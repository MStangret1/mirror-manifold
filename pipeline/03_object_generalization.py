from __future__ import annotations

import argparse
import os
import pandas as pd

from core_utils import ensure_dir, build_tensor, load_inventory, object_generalization_matrix


EVENTS = ("Event1", "Event3")
OBJECTS = ("Object1", "Object2", "Object3")


def main():
    ap = argparse.ArgumentParser(description="Train/test across objects for OBS Go vs OBS No-Go.")
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--go-nogo-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-neurons", type=int, default=3)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    inv = load_inventory(args.go_nogo_dir)
    summary = inv["summary"]
    rows = []

    for (area, session), g in summary.groupby(["area", "session"]):
        combos = {(r.context, r.condition) for r in g.itertuples()}
        if not {("Context2", "Condition1"), ("Context2", "Condition2")}.issubset(combos):
            continue
        for event in EVENTS:
            interval = {"Event1": (-0.2, 0.8), "Event3": (-0.8, 0.6)}[event]
            tensors_by_object = {}
            for object_ in OBJECTS:
                go = build_tensor(args.base_dir, area, session, "Context2", "Condition1", object_, event, interval, min_neurons=args.min_neurons)
                nogo = build_tensor(args.base_dir, area, session, "Context2", "Condition2", object_, event, interval, min_neurons=args.min_neurons)
                if go is not None and nogo is not None:
                    tensors_by_object[object_] = (go, nogo)
            if len(tensors_by_object) < 2:
                continue
            df = object_generalization_matrix(tensors_by_object)
            df.insert(0, "event", event)
            df.insert(0, "session", session)
            df.insert(0, "area", area)
            rows.append(df)
            print(f"Done {area} | {session} | {event}")

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_csv = os.path.join(args.out_dir, "object_generalization_results.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")


if __name__ == "__main__":
    main()
