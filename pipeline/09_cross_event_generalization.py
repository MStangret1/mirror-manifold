# 10_cross_event_generalization.py
from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from analysis_paths import infer_default_paths, resolve_path
from core_utils import EVENT_WINDOWS, bootstrap_mean_ci, build_tensor, ensure_dir, load_inventory
from core_utils_ext import cross_event_decode

OBJECTS = ("Object1", "Object2", "Object3")
EVENTS = ("Event1", "Event3")


def main() -> None:
    defaults = infer_default_paths(__file__)
    ap = argparse.ArgumentParser(
        description="Cross-event Go/NoGo generalization with optional permutation null."
    )
    ap.add_argument("--base-dir", default=str(defaults.base_dir))
    ap.add_argument("--go-nogo-dir", default=str(defaults.go_nogo_dir))
    ap.add_argument("--out-dir", default=str(defaults.final_dir / "10_cross_event_generalization"))
    ap.add_argument("--min-neurons", type=int, default=3)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-repeats", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--gauss-smooth", type=float, default=None)
    args = ap.parse_args()

    base_dir = resolve_path(args.base_dir, defaults.base_dir)
    go_nogo_dir = resolve_path(args.go_nogo_dir, defaults.go_nogo_dir)
    out_dir = resolve_path(args.out_dir, defaults.final_dir / "10_cross_event_generalization")
    ensure_dir(str(out_dir))

    inv = load_inventory(str(go_nogo_dir))
    summary = inv["summary"]

    unit_rows: List[Dict] = []
    for (area, session), g in summary.groupby(["area", "session"]):
        combos = {(r.context, r.condition) for r in g.itertuples()}
        if not {("Context2", "Condition1"), ("Context2", "Condition2")}.issubset(combos):
            continue
        for obj in OBJECTS:
            tensors: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
            for event in EVENTS:
                interval = EVENT_WINDOWS[event]
                go = build_tensor(
                    str(base_dir),
                    area,
                    session,
                    "Context2",
                    "Condition1",
                    obj,
                    event,
                    interval,
                    gauss_smooth=args.gauss_smooth,
                    min_neurons=args.min_neurons,
                )
                nogo = build_tensor(
                    str(base_dir),
                    area,
                    session,
                    "Context2",
                    "Condition2",
                    obj,
                    event,
                    interval,
                    gauss_smooth=args.gauss_smooth,
                    min_neurons=args.min_neurons,
                )
                tensors[event] = {"go": go, "nogo": nogo}
            if any(tensors[e][c] is None for e in EVENTS for c in ("go", "nogo")):
                continue

            res = cross_event_decode(
                tensorA_go=tensors["Event1"]["go"],
                tensorA_nogo=tensors["Event1"]["nogo"],
                tensorB_go=tensors["Event3"]["go"],
                tensorB_nogo=tensors["Event3"]["nogo"],
                n_folds=args.n_folds,
                n_repeats=args.n_repeats,
                n_perm=args.n_perm,
            )
            within_mean = (res["auc_within_A"] + res["auc_within_B"]) / 2.0
            cross_mean = (res["auc_A_to_B"] + res["auc_B_to_A"]) / 2.0
            ceiling = max(within_mean - 0.5, np.finfo(float).eps)
            normalized_transfer = (cross_mean - 0.5) / ceiling
            row = {
                "area": area,
                "session": session,
                "object": obj,
                "auc_within_Event1": res["auc_within_A"],
                "auc_within_Event3": res["auc_within_B"],
                "auc_Event1_to_Event3": res["auc_A_to_B"],
                "auc_Event3_to_Event1": res["auc_B_to_A"],
                "auc_within_Event1_std": res["auc_within_A_std"],
                "auc_within_Event3_std": res["auc_within_B_std"],
                "auc_Event1_to_Event3_std": res["auc_A_to_B_std"],
                "auc_Event3_to_Event1_std": res["auc_B_to_A_std"],
                "p_Event1_to_Event3": res["p_A_to_B"],
                "p_Event3_to_Event1": res["p_B_to_A"],
                "n_perm": res["n_perm"],
                "n_trials_go_shared": res["n_trials_go_shared"],
                "n_trials_nogo_shared": res["n_trials_nogo_shared"],
                "truncated_to_shared_min": res["truncated_to_shared_min"],
                "gti": cross_mean - 0.5,
                "normalized_transfer": normalized_transfer,
                "n_folds_run": res["n_folds_run"],
            }
            unit_rows.append(row)

    df_units = pd.DataFrame(unit_rows)
    df_units.to_csv(out_dir / "cross_event_generalization_per_unit.csv", index=False)

    summary_rows: List[Dict] = []
    for area, g in df_units.groupby("area"):
        row = {"area": area, "n_units": int(len(g))}
        for col in [
            "auc_within_Event1",
            "auc_within_Event3",
            "auc_Event1_to_Event3",
            "auc_Event3_to_Event1",
            "gti",
            "normalized_transfer",
        ]:
            vals = g[col].dropna().to_numpy()
            m, lo, hi = bootstrap_mean_ci(vals)
            row[f"{col}_mean"] = m
            row[f"{col}_ci95_low"] = lo
            row[f"{col}_ci95_high"] = hi
            try:
                _, p = wilcoxon(
                    vals - (0.5 if col != "normalized_transfer" else 0.0),
                    alternative="greater",
                    zero_method="wilcox",
                )
            except Exception:
                p = np.nan
            row[f"{col}_p_gt_null"] = float(p)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        out_dir / "cross_event_generalization_summary.csv", index=False
    )
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()