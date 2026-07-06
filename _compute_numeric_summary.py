"""
Compute key thesis numbers from the derived CSV result files.

Every number written here is derived directly from a CSV in the results tree;
nothing is hand-entered. Output:
  - results/repo_numeric_summary.csv  (long/tidy: metric, group, value, n, source)
The human-readable report (NUMERIC_CONSISTENCY_REPORT.md) is written separately.
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve the go-nogo results tree relative to the original project layout.
HERE = Path(__file__).resolve().parent               # .../mirror-manifold
PROJECT = HERE.parent                                # .../bachelor_data_cogsci
RES = PROJECT / "go-nogo" / "reworked_results"
FINAL = RES / "final_analysis"
INV = PROJECT / "go-nogo"

rows = []  # each: dict(metric, group, value, n, source)


def add(metric, group, value, n, source):
    rows.append(
        {
            "metric": metric,
            "group": group,
            "value": (round(float(value), 6) if value is not None and not (isinstance(value, float) and np.isnan(value)) else ""),
            "n": ("" if n is None else int(n)),
            "source": source,
        }
    )


def load(p):
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------- 1. session-level
soe = load(RES / "session_object_event_results.csv")
src_soe = "go-nogo/reworked_results/session_object_event_results.csv"
if soe is not None:
    add("n_analysis_units", "overall", len(soe), len(soe), src_soe)
    for col, lbl in [
        ("dec_obs_go_nogo_auc_full", "OBS Go-vs-NoGo AUC full"),
        ("dec_obs_go_nogo_auc_pre", "OBS Go-vs-NoGo AUC pre"),
        ("dec_obs_go_nogo_auc_post", "OBS Go-vs-NoGo AUC post"),
        ("dec_exe_obs_auc_full", "EXE-vs-OBS AUC full"),
    ]:
        if col in soe:
            add(f"mean_{col}", "overall", soe[col].mean(), soe[col].notna().sum(), src_soe)

    # pre vs post by event
    for ev, g in soe.groupby("event"):
        add("mean_auc_pre", f"event={ev}", g["dec_obs_go_nogo_auc_pre"].mean(), len(g), src_soe)
        add("mean_auc_post", f"event={ev}", g["dec_obs_go_nogo_auc_post"].mean(), len(g), src_soe)
    # pre vs post by area x event
    for (ar, ev), g in soe.groupby(["area", "event"]):
        add("mean_auc_pre", f"area={ar},event={ev}", g["dec_obs_go_nogo_auc_pre"].mean(), len(g), src_soe)
        add("mean_auc_post", f"area={ar},event={ev}", g["dec_obs_go_nogo_auc_post"].mean(), len(g), src_soe)
    # units by area
    for ar, g in soe.groupby("area"):
        add("n_units", f"area={ar}", len(g), len(g), src_soe)

    # permutation significance
    if "perm_obs_go_nogo_auc_p" in soe:
        p = soe["perm_obs_go_nogo_auc_p"].dropna()
        add("n_perm_tests_with_p", "overall", len(p), len(p), src_soe)
        for thr in (0.05, 0.01):
            add(f"n_perm_sig_p<{thr}", "overall", int((p < thr).sum()), len(p), src_soe)
            add(f"frac_perm_sig_p<{thr}", "overall", float((p < thr).mean()), len(p), src_soe)
        for ar, g in soe.groupby("area"):
            pg = g["perm_obs_go_nogo_auc_p"].dropna()
            add("n_perm_sig_p<0.05", f"area={ar}", int((pg < 0.05).sum()), len(pg), src_soe)

# ---------------------------------------------------------------- 2. time-resolved
tr = load(RES / "time_resolved_results.csv")
src_tr = "go-nogo/reworked_results/time_resolved_results.csv"
if tr is not None:
    # peak of the across-unit mean AUC curve, per area x event
    m = tr.groupby(["area", "event", "center_time_s"])["auc"].mean().reset_index()
    for (ar, ev), g in m.groupby(["area", "event"]):
        i = g["auc"].idxmax()
        add("time_resolved_peak_auc", f"area={ar},event={ev}", g.loc[i, "auc"], len(tr[(tr.area==ar)&(tr.event==ev)]), src_tr)
        add("time_resolved_peak_time_s", f"area={ar},event={ev}", g.loc[i, "center_time_s"], None, src_tr)

# ---------------------------------------------------------------- 3. divergence onset
dv = load(FINAL / "07_divergence_onset" / "divergence_onset_summary.csv")
src_dv = "go-nogo/reworked_results/final_analysis/07_divergence_onset/divergence_onset_summary.csv"
if dv is not None:
    for _, r in dv.iterrows():
        grp = f"area={r['area']},event={r['event']}"
        add("divergence_onset_ms_median", grp, r.get("onset_ms_median"), r.get("n_units"), src_dv)
        add("divergence_onset_ms_mean", grp, r.get("onset_ms_mean"), r.get("n_units"), src_dv)
        add("divergence_n_onset_found", grp, r.get("n_onset_found"), r.get("n_units"), src_dv)

# ---------------------------------------------------------------- 4. cross-object generalization
og = load(RES / "object_generalization_results.csv")
src_og = "go-nogo/reworked_results/object_generalization_results.csv"
if og is not None:
    # same-object vs cross-object
    for iscross, g in og.groupby("is_cross_object"):
        lbl = "cross_object" if bool(iscross) else "same_object"
        add(f"mean_auc_{lbl}", "overall", g["auc"].mean(), len(g), src_og)
    # by event x cross/same
    for (ev, iscross), g in og.groupby(["event", "is_cross_object"]):
        lbl = "cross_object" if bool(iscross) else "same_object"
        add(f"mean_auc_{lbl}", f"event={ev}", g["auc"].mean(), len(g), src_og)

# cross-event generalization (Event1<->Event3)
ceg = load(FINAL / "10_cross_event_generalization" / "cross_event_generalization_summary.csv")
src_ceg = "go-nogo/reworked_results/final_analysis/10_cross_event_generalization/cross_event_generalization_summary.csv"
if ceg is not None:
    for _, r in ceg.iterrows():
        grp = f"area={r['area']}"
        # normalized_transfer is a ratio that blows up numerically -> excluded as unstable
        for col in ["auc_within_Event1_mean", "auc_within_Event3_mean",
                    "auc_Event1_to_Event3_mean", "auc_Event3_to_Event1_mean",
                    "gti_mean"]:
            if col in ceg:
                add(col, grp, r[col], r.get("n_units"), src_ceg)

# ---------------------------------------------------------------- 5. dPCA
dps = load(FINAL / "09_dpca" / "dpca_variance_summary.csv")
src_dps = "go-nogo/reworked_results/final_analysis/09_dpca/dpca_variance_summary.csv"
if dps is not None:
    marg = {"var_t": "time", "var_d": "decision", "var_o": "object",
            "var_dt": "decision x time", "var_ot": "object x time",
            "var_do": "decision x object", "var_dot": "decision x object x time"}
    for _, r in dps.iterrows():
        grp = f"area={r['area']},event={r['event']}"
        for col, lbl in marg.items():
            mcol = f"{col}_mean"
            if mcol in dps:
                add(f"dpca_var_{lbl}", grp, r[mcol], r.get("n_units"), src_dps)

dpu = load(FINAL / "09_dpca" / "dpca_variance_per_unit.csv")
src_dpu = "go-nogo/reworked_results/final_analysis/09_dpca/dpca_variance_per_unit.csv"
if dpu is not None:
    add("dpca_n_units_total", "session x event", len(dpu), len(dpu), src_dpu)
    if "dpca_available" in dpu:
        flag = dpu["dpca_available"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
        # NOTE: if 0, the dPCA package was absent at run time -> variance via fallback demixing
        add("dpca_n_units_pkg_available", "overall", int(flag.sum()), len(dpu), src_dpu)
    # unit counts by area/event come from the aggregated summary n_units column
    if dps is not None:
        for _, r in dps.iterrows():
            add("dpca_n_units", f"area={r['area']},event={r['event']}", r.get("n_units"), r.get("n_units"), src_dps)

# ---------------------------------------------------------------- 6. inventory
inv_neu = load(INV / "inventory_neurons_by_session.csv")
if inv_neu is not None and "n_neurons" in inv_neu:
    add("total_neurons", "sum_over_sessions", inv_neu["n_neurons"].sum(), len(inv_neu), "go-nogo/inventory_neurons_by_session.csv")
    for ar, g in inv_neu.groupby("area"):
        add("total_neurons", f"area={ar}", g["n_neurons"].sum(), len(g), "go-nogo/inventory_neurons_by_session.csv")

# ---------------------------------------------------------------- write
out = pd.DataFrame(rows, columns=["metric", "group", "value", "n", "source"])
outdir = HERE / "results"
outdir.mkdir(parents=True, exist_ok=True)
outpath = outdir / "repo_numeric_summary.csv"
out.to_csv(outpath, index=False)
print(f"Wrote {outpath} ({len(out)} rows)")
# Echo a compact view for the report
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 200)
print(out.to_string(index=False))
