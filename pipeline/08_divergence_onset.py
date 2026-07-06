# 07_divergence_onset.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis_paths import infer_default_paths, resolve_path
from core_utils import ensure_dir
from core_utils_ext import bh_qvalues, compute_divergence_onset_from_pvalues


def main() -> None:
    defaults = infer_default_paths(__file__)
    ap = argparse.ArgumentParser(description="Null-based divergence onset from time-resolved decoding.")
    ap.add_argument("--time-csv", default=str(defaults.time_resolved_csv))
    ap.add_argument(
        "--null-time-csv",
        default=str(defaults.final_dir / "06_shuffle_null" / "shuffle_null_time_resolved.csv"),
    )
    ap.add_argument("--out-dir", default=str(defaults.final_dir / "07_divergence_onset"))
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-consecutive", type=int, default=2)
    args = ap.parse_args()

    time_csv = resolve_path(args.time_csv, defaults.time_resolved_csv)
    null_time_csv = resolve_path(
        args.null_time_csv,
        defaults.final_dir / "06_shuffle_null" / "shuffle_null_time_resolved.csv",
    )
    out_dir = resolve_path(args.out_dir, defaults.final_dir / "07_divergence_onset")
    ensure_dir(str(out_dir))

    if not Path(null_time_csv).exists():
        raise FileNotFoundError(
            f"Required null-time CSV not found: {null_time_csv}. Run 06_shuffle_null_model.py first."
        )

    df_time = pd.read_csv(time_csv)
    df_null = pd.read_csv(null_time_csv)
    merge_cols = ["area", "session", "object", "event", "start_bin", "stop_bin", "center_time_s"]
    df = df_time.merge(
        df_null[merge_cols + ["p_auc", "null_auc_mean", "null_auc_ci95_high"]],
        on=merge_cols,
        how="inner",
        validate="one_to_one",
    )

    unit_rows: List[Dict] = []
    q_rows: List[pd.DataFrame] = []
    for keys, g in df.groupby(["area", "session", "object", "event"]):
        g = g.sort_values("center_time_s").copy()
        g["q_auc"] = bh_qvalues(g["p_auc"].to_numpy())
        onset = compute_divergence_onset_from_pvalues(
            center_times=g["center_time_s"].to_numpy(),
            p_values=g["q_auc"].to_numpy(),
            effect_values=g["auc"].to_numpy(),
            alpha=args.alpha,
            min_consecutive=args.min_consecutive,
        )
        row = {col: val for col, val in zip(["area", "session", "object", "event"], keys)}
        row["onset_s"] = onset
        row["onset_ms"] = onset * 1000.0 if np.isfinite(onset) else np.nan
        row["n_time_bins"] = int(len(g))
        row["n_significant_post_bins"] = int(
            ((g["center_time_s"] >= 0) & (g["q_auc"] <= args.alpha) & (g["auc"] > 0.5)).sum()
        )
        unit_rows.append(row)
        q_rows.append(g)

    df_units = pd.DataFrame(unit_rows)
    df_q = pd.concat(q_rows, ignore_index=True)

    summary_rows: List[Dict] = []
    for (area, event), g in df_units.groupby(["area", "event"]):
        vals = g["onset_ms"].dropna().to_numpy()
        summary_rows.append(
            {
                "area": area,
                "event": event,
                "n_units": int(len(g)),
                "n_onset_found": int(np.isfinite(g["onset_ms"]).sum()),
                "onset_ms_median": float(np.median(vals)) if len(vals) else np.nan,
                "onset_ms_q25": float(np.percentile(vals, 25)) if len(vals) else np.nan,
                "onset_ms_q75": float(np.percentile(vals, 75)) if len(vals) else np.nan,
                "onset_ms_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "onset_ms_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
            }
        )
    df_summary = pd.DataFrame(summary_rows)

    cmp_rows: List[Dict] = []
    for area, g in df_units.groupby("area"):
        e1 = g.loc[g["event"] == "Event1", "onset_ms"].dropna().to_numpy()
        e3 = g.loc[g["event"] == "Event3", "onset_ms"].dropna().to_numpy()
        row: Dict[str, float | int | str] = {"area": area, "comparison": "Event3_vs_Event1"}
        row["n_event1"] = int(len(e1))
        row["n_event3"] = int(len(e3))
        if len(e1) >= 3 and len(e3) >= 3:
            U, p = mannwhitneyu(e3, e1, alternative="less")
            row.update(
                {
                    "U": float(U),
                    "p_event3_earlier": float(p),
                    "median_event1_ms": float(np.median(e1)),
                    "median_event3_ms": float(np.median(e3)),
                }
            )
        else:
            row.update(
                {
                    "U": np.nan,
                    "p_event3_earlier": np.nan,
                    "median_event1_ms": np.nan,
                    "median_event3_ms": np.nan,
                }
            )
        cmp_rows.append(row)
    df_cmp = pd.DataFrame(cmp_rows)

    df_units.to_csv(out_dir / "divergence_onset_per_unit.csv", index=False)
    df_summary.to_csv(out_dir / "divergence_onset_summary.csv", index=False)
    df_cmp.to_csv(out_dir / "divergence_onset_comparisons.csv", index=False)
    df_q.to_csv(out_dir / "divergence_onset_timewise_qvalues.csv", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()