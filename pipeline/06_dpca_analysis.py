# 09_dpca.py
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis_paths import infer_default_paths, resolve_path
from core_utils import EVENT_WINDOWS, bootstrap_mean_ci, ensure_dir, load_inventory
from core_utils_ext import anova_marginalize, build_mean_tensor_decision_object, variance_fractions

OBJECTS = ("Object1", "Object2", "Object3")
EVENTS = ("Event1", "Event3")
MARG_LABELS = ("t", "d", "o", "dt", "ot", "do", "dot")
_CONDITION_LABELS = ("Go", "NoGo")   # axis-1 order in X = (n,d,o,t)

try:
    from dPCA.dPCA import dPCA as _dPCA

    HAS_DPCA = True
except Exception:
    HAS_DPCA = False


def _dpca_components_long(X: np.ndarray, n_components: int) -> pd.DataFrame:
    """
    X shape: (n_neurons, n_decision=2, n_objects=3, n_timebins).
    After fit_transform, Z[marg] has shape (n_components, 2, 3, n_timebins).
    FIX: keep the decision axis separate (Go / NoGo) instead of averaging over it,
    so decision-containing marginalizations have non-zero scores.
    """
    dpca = _dPCA(labels="dot", n_components=n_components, regularizer="auto")
    Z = dpca.fit_transform(X)
    rows: List[Dict] = []
    for marg, arr in Z.items():
        arr = np.asarray(arr)
        if arr.ndim < 2:
            continue
        n_comps    = arr.shape[0]
        n_timebins = arr.shape[-1]
        for cond_idx, cond_label in enumerate(_CONDITION_LABELS):
            if cond_idx >= arr.shape[1]:
                break  # guard: fewer levels than expected
            if arr.ndim == 4:
                # shape (n_comps, n_decision, n_objects, n_timebins)
                # average over object axis only
                traj = arr[:, cond_idx, :, :].mean(axis=1)
            elif arr.ndim == 3:
                # shape (n_comps, n_decision, n_timebins)
                traj = arr[:, cond_idx, :]
            else:
                traj = arr.reshape(n_comps, n_timebins)
            for comp_idx in range(traj.shape[0]):
                for time_idx in range(traj.shape[1]):
                    rows.append(
                        {
                            "marginalization": marg,
                            "component": int(comp_idx + 1),
                            "condition": cond_label,
                            "time_index": int(time_idx),
                            "score": float(traj[comp_idx, time_idx]),
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    defaults = infer_default_paths(__file__)
    ap = argparse.ArgumentParser(
        description="ANOVA variance decomposition with optional dPCA component export."
    )
    ap.add_argument("--base-dir", default=str(defaults.base_dir))
    ap.add_argument("--go-nogo-dir", default=str(defaults.go_nogo_dir))
    ap.add_argument("--out-dir", default=str(defaults.final_dir / "09_dpca"))
    ap.add_argument("--min-neurons", type=int, default=3)
    ap.add_argument("--gauss-smooth", type=float, default=None)
    ap.add_argument("--n-dpca-components", type=int, default=3)
    args = ap.parse_args()

    base_dir = resolve_path(args.base_dir, defaults.base_dir)
    go_nogo_dir = resolve_path(args.go_nogo_dir, defaults.go_nogo_dir)
    out_dir = resolve_path(args.out_dir, defaults.final_dir / "09_dpca")
    ensure_dir(str(out_dir))

    inv = load_inventory(str(go_nogo_dir))
    summary = inv["summary"]

    unit_rows: List[Dict] = []
    traj_rows: List[pd.DataFrame] = []
    for (area, session), g in summary.groupby(["area", "session"]):
        combos = {(r.context, r.condition) for r in g.itertuples()}
        if not {("Context2", "Condition1"), ("Context2", "Condition2")}.issubset(combos):
            continue
        for event in EVENTS:
            X = build_mean_tensor_decision_object(
                str(base_dir),
                area,
                session,
                event,
                EVENT_WINDOWS[event],
                objects=OBJECTS,
                gauss_smooth=args.gauss_smooth,
                min_neurons=args.min_neurons,
            )
            if X is None:
                continue
            vf = variance_fractions(X)
            row: Dict = {
                "area": area,
                "session": session,
                "event": event,
                "n_neurons": int(X.shape[0]),
                "n_bins": int(X.shape[-1]),
                "dpca_available": HAS_DPCA,
            }
            row.update({f"var_{k}": vf[k] for k in MARG_LABELS})
            row["total_var"] = vf["total_var"]
            unit_rows.append(row)

            comps = anova_marginalize(X)
            # X shape: (n_neurons, n_decision=2, n_objects=3, n_timebins)
            # FIX: keep decision axis separate so 'd'/'dt' marginalizations
            # store non-zero Go and NoGo scores rather than their mean (=0).
            for marg, arr in comps.items():
                for cond_idx, cond_label in enumerate(_CONDITION_LABELS):
                    # Average over the object axis only; keep decision separate.
                    mean_traj = arr[:, cond_idx, :, :].mean(axis=1)
                    for neuron_idx in range(mean_traj.shape[0]):
                        for time_idx in range(mean_traj.shape[1]):
                            traj_rows.append(
                                pd.DataFrame(
                                    [
                                        {
                                            "area": area,
                                            "session": session,
                                            "event": event,
                                            "source": "anova",
                                            "marginalization": marg,
                                            "condition": cond_label,
                                            "component": int(neuron_idx + 1),
                                            "time_index": int(time_idx),
                                            "score": float(mean_traj[neuron_idx, time_idx]),
                                        }
                                    ]
                                )
                            )

            if HAS_DPCA:
                try:
                    df_comp = _dpca_components_long(X, args.n_dpca_components)
                    if len(df_comp):
                        df_comp.insert(0, "event", event)
                        df_comp.insert(0, "session", session)
                        df_comp.insert(0, "area", area)
                        df_comp.insert(3, "source", "dpca")
                        traj_rows.append(df_comp)
                except Exception as exc:
                    row["dpca_error"] = str(exc)

    if not unit_rows:
        raise RuntimeError("No units processed. Check paths and inventory completeness.")

    df_units = pd.DataFrame(unit_rows)
    df_units.to_csv(out_dir / "dpca_variance_per_unit.csv", index=False)

    summary_rows: List[Dict] = []
    for (area, event), g in df_units.groupby(["area", "event"]):
        row = {"area": area, "event": event, "n_units": int(len(g))}
        for marg in MARG_LABELS:
            vals = g[f"var_{marg}"].dropna().to_numpy()
            m, lo, hi = bootstrap_mean_ci(vals)
            row[f"var_{marg}_mean"] = m
            row[f"var_{marg}_ci95_low"] = lo
            row[f"var_{marg}_ci95_high"] = hi
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "dpca_variance_summary.csv", index=False)

    if traj_rows:
        pd.concat(traj_rows, ignore_index=True).to_csv(
            out_dir / "dpca_component_trajectories_long.csv", index=False
        )
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()