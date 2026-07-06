"""
11_neuron_count_sensitivity.py
-------------------------------
Robustness check: does the Event 3 observational Go/No-Go effect depend on
how well-populated the recording sessions are?

Splits Event 3 units at the median of n_neurons into high-N and low-N
subsets, recomputes the post-event decoding AUC within each subset, and
returns stratified bootstrap 95% CIs. Also reports a per-area breakdown.

Inputs:  session_object_event_results.csv
Outputs: neuron_count_sensitivity.csv
         neuron_count_sensitivity_by_area.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

_RESULTS   = Path(__file__).resolve().parent.parent / "results"
INPUT_CSV  = str(_RESULTS / "session_object_event_results.csv")
OUT_POOLED = str(_RESULTS / "neuron_count_sensitivity.csv")
OUT_BYAREA = str(_RESULTS / "neuron_count_sensitivity_by_area.csv")

EVENT    = "Event3"
METRIC   = "dec_obs_go_nogo_auc_post"
N_BOOT   = 1000
RNG_SEED = 20260411  # reproducible

def stratified_bootstrap_mean(values, strata, n_boot, rng):
    """
    Stratified bootstrap of the mean: resample with replacement *within*
    each stratum, then take the overall mean. Matches the procedure used
    in the meta-summary so CIs are comparable.
    """
    values = np.asarray(values, dtype=float)
    strata = np.asarray(strata)
    boot_means = np.empty(n_boot, dtype=float)
    strata_unique = np.unique(strata)
    idx_by_stratum = {s: np.where(strata == s)[0] for s in strata_unique}

    for b in range(n_boot):
        resampled = []
        for s in strata_unique:
            idx = idx_by_stratum[s]
            pick = rng.choice(idx, size=len(idx), replace=True)
            resampled.append(values[pick])
        boot_means[b] = np.concatenate(resampled).mean()

    return boot_means

def summarise(values, strata, n_boot, rng):
    boot = stratified_bootstrap_mean(values, strata, n_boot, rng)
    return {
        "n_units":      int(len(values)),
        "mean":         float(np.mean(values)),
        "ci95_low":     float(np.percentile(boot, 2.5)),
        "ci95_high":    float(np.percentile(boot, 97.5)),
        "boot_mean":    float(boot.mean()),
    }

def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["event"] == EVENT].copy()
    if df.empty:
        raise RuntimeError(f"No rows with event == {EVENT} in {INPUT_CSV}")

    median_n = float(df["n_neurons"].median())
    df["subset"] = np.where(df["n_neurons"] >= median_n, "high_N", "low_N")

    print(f"Event:                 {EVENT}")
    print(f"Total Event {EVENT[-1]} units: {len(df)}")
    print(f"Median n_neurons:      {median_n:.1f}")
    print(f"high_N units (>= med): {(df['subset']=='high_N').sum()}")
    print(f"low_N  units (<  med): {(df['subset']=='low_N').sum()}")
    print()

    rng = np.random.default_rng(RNG_SEED)

    # ----- Pooled across areas, stratified by area -----
    pooled_rows = []
    for sub_name, sub_df in df.groupby("subset"):
        s = summarise(
            values=sub_df[METRIC].values,
            strata=sub_df["area"].values,
            n_boot=N_BOOT, rng=rng,
        )
        s["subset"]            = sub_name
        s["median_n_neurons"]  = median_n
        s["min_n_neurons"]     = int(sub_df["n_neurons"].min())
        s["max_n_neurons"]     = int(sub_df["n_neurons"].max())
        pooled_rows.append(s)

    pooled_df = pd.DataFrame(pooled_rows)[
        ["subset", "n_units", "median_n_neurons",
         "min_n_neurons", "max_n_neurons",
         "mean", "ci95_low", "ci95_high", "boot_mean"]
    ].sort_values("subset")

    print("=== Pooled across areas (stratified bootstrap by area) ===")
    print(pooled_df.to_string(index=False))
    print()

    # ----- By area -----
    byarea_rows = []
    for (sub_name, area), g in df.groupby(["subset", "area"]):
        s = summarise(
            values=g[METRIC].values,
            strata=np.zeros(len(g)),  # single stratum within (subset, area)
            n_boot=N_BOOT, rng=rng,
        )
        s["subset"] = sub_name
        s["area"]   = area
        byarea_rows.append(s)

    byarea_df = pd.DataFrame(byarea_rows)[
        ["area", "subset", "n_units",
         "mean", "ci95_low", "ci95_high", "boot_mean"]
    ].sort_values(["area", "subset"])

    print("=== Per area ===")
    print(byarea_df.to_string(index=False))
    print()

    # ----- Save -----
    pooled_df.to_csv(OUT_POOLED, index=False)
    byarea_df.to_csv(OUT_BYAREA, index=False)
    print(f"Saved: {OUT_POOLED}")
    print(f"Saved: {OUT_BYAREA}")

if __name__ == "__main__":
    main()
