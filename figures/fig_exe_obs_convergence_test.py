"""
fig_exe_obs_convergence_test.py
-------------------------------
Tests whether Execution-Go and Observation-Go trajectories converge over time
in AIP but not F5/F6, beyond what's visible in the 2-D PCA projection used in
fig_exe_obs_benchmark_v4.py.

For each area:
    - iterate every (session, object) pair with both contexts complete and
      >= 4 trials in each,
    - fit a shared 8-PC PCA on the concatenated Exe-Go + Obs-Go tensors,
    - per time bin, compute
        (1) cross-validated logistic-regression AUC for Exe vs Obs
            using all 8 PCs as features,
        (2) Euclidean centroid distance between the two conditions in 8-D.

Outputs:
    reworked_results/fig_exe_obs_convergence_test.{pdf,png}
    reworked_results/exe_obs_convergence_per_session.csv   (long format)
    reworked_results/exe_obs_convergence_summary.csv       (per-area summary)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

import core_utils as cu

# ---------- CONFIG (machine-independent) -------------------------------------
_REPO_ROOT  = Path(__file__).resolve().parent.parent
RESULTS_DIR = _REPO_ROOT / "results"
GO_NOGO_DIR = RESULTS_DIR                      # inventory CSVs live under results/
HDF5_DIR    = Path(os.environ.get("DATA_ROOT", str(_REPO_ROOT / "data" / "raw" / "HDF5")))
GEN_DIR     = _REPO_ROOT / "figures" / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

BASE_DIR     = HDF5_DIR
EVENT        = "Event3"
INTERVAL     = (-1.0, 1.0)
BINSIZE      = 0.05
GAUSS_SMOOTH = 0.05
N_PCS        = 8

EXE_CONTEXT, EXE_CONDITION = "Context1", "Condition1"
OBS_CONTEXT, OBS_CONDITION = "Context2", "Condition1"

EARLY_WINDOW = (-1.0, -0.5)
LATE_WINDOW  = ( 0.5,  1.0)

OUT_PDF      = GEN_DIR / "fig_exe_obs_convergence_test.pdf"
OUT_LONG_CSV = RESULTS_DIR / "exe_obs_convergence_per_session.csv"
OUT_SUM_CSV  = RESULTS_DIR / "exe_obs_convergence_summary.csv"

# ---------- shared rcParams (matches fig_exe_obs_benchmark_v4) ---------------
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75, "ytick.major.width": 0.75,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.dpi": 600,
})

MM = 1 / 25.4
FIG_W, FIG_H = 183 * MM, 110 * MM
AREA_COLORS = {"AIP": "#d97706", "F5": "#1f4e8c", "F6": "#15803d"}
AREAS = ["AIP", "F5", "F6"]


def find_best_session_for_area(combos_df, area):
    """All (session, object) in this area with both contexts complete,
    sorted by neuron count descending. Identical to v4 benchmark helper."""
    have_exe = combos_df[
        (combos_df.area == area)
        & (combos_df.context == EXE_CONTEXT)
        & (combos_df.condition == EXE_CONDITION)
        & (combos_df.is_complete == True)
    ][["session", "object", "n_neurons"]]
    have_obs = combos_df[
        (combos_df.area == area)
        & (combos_df.context == OBS_CONTEXT)
        & (combos_df.condition == OBS_CONDITION)
        & (combos_df.is_complete == True)
    ][["session", "object"]]
    merged = have_exe.merge(have_obs, on=["session", "object"], how="inner")
    if merged.empty:
        return None
    return merged.sort_values("n_neurons", ascending=False).reset_index(drop=True)


def time_resolved_auc_and_distance(Z_exe, Z_obs):
    """
    Z_exe: (n_trials_exe, n_bins, n_pcs)
    Z_obs: (n_trials_obs, n_bins, n_pcs)
    Returns (aucs, dists) both shape (n_bins,).
    """
    n_bins = Z_exe.shape[1]
    aucs  = np.full(n_bins, np.nan)
    dists = np.zeros(n_bins)

    y = np.concatenate([np.zeros(Z_exe.shape[0]),
                        np.ones(Z_obs.shape[0])]).astype(int)
    min_class = int(np.bincount(y).min())
    n_splits = max(2, min(5, min_class))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=0)

    for t in range(n_bins):
        X = np.vstack([Z_exe[:, t, :], Z_obs[:, t, :]])
        try:
            scores = cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0),
                X, y, cv=cv, scoring="roc_auc",
            )
            aucs[t] = float(np.mean(scores))
        except Exception as e:
            print(f"    AUC failed at bin {t}: {e}")

        mu_exe_t = Z_exe[:, t, :].mean(axis=0)
        mu_obs_t = Z_obs[:, t, :].mean(axis=0)
        dists[t] = float(np.linalg.norm(mu_exe_t - mu_obs_t))

    return aucs, dists


# =============================================================================
# RUN
# =============================================================================
combos = pd.read_csv(GO_NOGO_DIR / "inventory_event_combos_with_completeness.csv")

per_session_records = []   # for the long CSV
per_area_curves = {area: [] for area in AREAS}  # for plotting

for area in AREAS:
    cands = find_best_session_for_area(combos, area)
    if cands is None or cands.empty:
        print(f"{area}: no candidate sessions")
        continue

    print(f"\n=== {area}: {len(cands)} candidate (session, object) pairs ===")
    n_used = 0
    for _, row in cands.iterrows():
        s, o, n = row["session"], row["object"], int(row["n_neurons"])
        T_exe = cu.build_tensor(BASE_DIR, area, s, EXE_CONTEXT, EXE_CONDITION, o,
                                EVENT, INTERVAL, BINSIZE, gauss_smooth=GAUSS_SMOOTH)
        T_obs = cu.build_tensor(BASE_DIR, area, s, OBS_CONTEXT, OBS_CONDITION, o,
                                EVENT, INTERVAL, BINSIZE, gauss_smooth=GAUSS_SMOOTH)
        if T_exe is None or T_obs is None:
            continue
        if T_exe.shape[0] < 4 or T_obs.shape[0] < 4:
            continue

        try:
            scaler, pca, (Z_exe, Z_obs) = cu.fit_shared_pca(
                T_exe, T_obs, n_components=N_PCS,
            )
        except ValueError as e:
            print(f"  skip {s} obj={o}: {e}")
            continue

        aucs, dists = time_resolved_auc_and_distance(Z_exe, Z_obs)
        n_bins = Z_exe.shape[1]
        time_axis = np.linspace(INTERVAL[0], INTERVAL[1], n_bins)

        per_area_curves[area].append({
            "session": s, "object": o, "n_neurons": n,
            "time_axis": time_axis, "aucs": aucs, "dists": dists,
        })

        for t_i in range(n_bins):
            per_session_records.append({
                "area": area, "session": s, "object": o,
                "n_neurons": n,
                "n_trials_exe": int(T_exe.shape[0]),
                "n_trials_obs": int(T_obs.shape[0]),
                "n_pcs_used": int(pca.n_components_),
                "time_s": float(time_axis[t_i]),
                "auc": float(aucs[t_i]),
                "centroid_distance": float(dists[t_i]),
            })

        n_used += 1
        print(f"  used {s} obj={o} n={n} (trials: exe={T_exe.shape[0]}, obs={T_obs.shape[0]})")

    print(f"{area}: {n_used} sessions usable")


# =============================================================================
# SAVE LONG CSV
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
long_df = pd.DataFrame(per_session_records)
long_df.to_csv(OUT_LONG_CSV, index=False)
print(f"\nSaved per-session long CSV: {OUT_LONG_CSV}")


# =============================================================================
# PER-AREA SUMMARY
# =============================================================================
summary_rows = []
for area in AREAS:
    df_a = long_df[long_df.area == area]
    if df_a.empty:
        summary_rows.append({
            "area": area, "n_sessions": 0,
            "auc_early": np.nan, "auc_late": np.nan,
            "dist_early": np.nan, "dist_late": np.nan,
        })
        continue
    n_sessions = df_a.groupby(["session", "object"]).ngroups
    early = df_a[(df_a.time_s >= EARLY_WINDOW[0]) & (df_a.time_s <= EARLY_WINDOW[1])]
    late  = df_a[(df_a.time_s >= LATE_WINDOW[0])  & (df_a.time_s <= LATE_WINDOW[1])]
    summary_rows.append({
        "area": area,
        "n_sessions": int(n_sessions),
        "auc_early":  float(early["auc"].mean()),
        "auc_late":   float(late["auc"].mean()),
        "dist_early": float(early["centroid_distance"].mean()),
        "dist_late":  float(late["centroid_distance"].mean()),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT_SUM_CSV, index=False)

print("\n=== Per-area summary ===")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nSaved summary CSV: {OUT_SUM_CSV}")


# =============================================================================
# FIGURE: 2 rows x 3 cols
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H),
                         gridspec_kw={"wspace": 0.32, "hspace": 0.32})

# Common distance y-limit so areas are visually comparable
all_dists = np.concatenate([
    np.concatenate([c["dists"] for c in per_area_curves[a]])
    for a in AREAS if per_area_curves[a]
]) if any(per_area_curves[a] for a in AREAS) else np.array([0.0, 1.0])
dist_ymax = float(np.nanmax(all_dists)) * 1.05
dist_ymin = 0.0

for col, area in enumerate(AREAS):
    ax_auc  = axes[0, col]
    ax_dist = axes[1, col]
    color = AREA_COLORS[area]
    curves = per_area_curves[area]

    if not curves:
        for ax in (ax_auc, ax_dist):
            ax.text(0.5, 0.5, f"no usable\n{area} sessions",
                    ha="center", va="center", transform=ax.transAxes)
        continue

    # Per-session thin lines
    for c in curves:
        ax_auc.plot(c["time_axis"], c["aucs"], color=color, alpha=0.30, lw=0.7)
        ax_dist.plot(c["time_axis"], c["dists"], color=color, alpha=0.30, lw=0.7)

    # Mean across sessions (interpolate to a shared grid if needed; here all
    # share the same grid because INTERVAL/BINSIZE is fixed)
    time_axis = curves[0]["time_axis"]
    auc_stack  = np.vstack([c["aucs"]  for c in curves])
    dist_stack = np.vstack([c["dists"] for c in curves])
    mean_auc  = np.nanmean(auc_stack,  axis=0)
    mean_dist = np.nanmean(dist_stack, axis=0)

    ax_auc.plot(time_axis,  mean_auc,  color=color, lw=2.2, zorder=5)
    ax_dist.plot(time_axis, mean_dist, color=color, lw=2.2, zorder=5)

    # Reference lines
    ax_auc.axhline(0.5, color="#6b7280", lw=0.6, ls="--", zorder=1)
    for ax in (ax_auc, ax_dist):
        ax.axvline(0.0, color="#6b7280", lw=0.6, ls="--", zorder=1)

    ax_auc.set_ylim(0.4, 1.05)
    ax_dist.set_ylim(dist_ymin, dist_ymax)

    ax_auc.set_title(area, loc="left", fontweight="bold", color=color, pad=4)
    n_used = len(curves)
    ax_auc.text(0.97, 0.05, f"n = {n_used} sessions",
                transform=ax_auc.transAxes, ha="right", va="bottom",
                fontsize=6.0, color="#6b7280")

    ax_dist.set_xlabel("Time from Event 3 (s)")
    if col == 0:
        ax_auc.set_ylabel("Exe vs Obs AUC")
        ax_dist.set_ylabel("Centroid distance (8D)")

# Legend in top-left panel
handles = [
    Line2D([0], [0], color="#374151", lw=0.8, alpha=0.4, label="per session"),
    Line2D([0], [0], color="#374151", lw=2.2, label="across-session mean"),
]
axes[0, 0].legend(handles=handles, loc="lower left", frameon=False,
                  handletextpad=0.4, borderpad=0.2, labelspacing=0.3,
                  fontsize=5.8)

fig.suptitle("Execution vs Observation: time-resolved convergence in full 8-PC space",
             fontsize=8.5, fontweight="bold", y=1.00, x=0.06, ha="left")

fig.savefig(OUT_PDF)
fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300)
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PDF.with_suffix('.png')}")
