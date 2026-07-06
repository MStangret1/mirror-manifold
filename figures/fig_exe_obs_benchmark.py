"""
fig_exe_obs_benchmark_v4.py
---------------------------
Figure 1: three-area population state-space view of EXE vs OBS.

One subpanel per area (AIP, F5, F6). Each shows the highest-N session that
has both Execute-Go and Observe-Go available, projected into a SHARED PCA
space within that session, smoothed for visual clarity. Mean trajectories
have time encoded as line saturation; single-trial post-event clouds and
1-SD covariance ellipses confirm trial-level separation; AUC for that area
is annotated in the corner.

Label mapping (from core_utils.py):
    Execute-Go = (Context1, Condition1)
    Observe-Go = (Context2, Condition1)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

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
GAUSS_SMOOTH = 0.05            # 50 ms Gaussian smoothing on firing rates
N_PCS        = 6
OUT_PDF      = GEN_DIR / "fig_exe_obs_benchmark.pdf"

EXE_CONTEXT, EXE_CONDITION = "Context1", "Condition1"
OBS_CONTEXT, OBS_CONDITION = "Context2", "Condition1"

# ---------- shared rcParams --------------------------------------------------
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
FIG_W, FIG_H = 183 * MM, 65 * MM
AREA_COLORS = {"AIP": "#d97706", "F5": "#1f4e8c", "F6": "#15803d"}
COLOR_EXE = "#b91c1c"
COLOR_OBS = "#1f4e8c"
AREAS = ["AIP", "F5", "F6"]


def gradient_line(ax, xs, ys, base_color, lw=2.0, zorder=3):
    pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    alphas = np.linspace(0.25, 1.0, n)
    rgb = to_rgba(base_color)[:3]
    colors = np.array([(*rgb, a) for a in alphas])
    lc = LineCollection(segs, colors=colors, linewidths=lw, zorder=zorder,
                        capstyle="round", joinstyle="round")
    ax.add_collection(lc)


def cov_ellipse(ax, xs, ys, color, n_std=1.0, alpha=0.13, zorder=2):
    if len(xs) < 3:
        return
    cov = np.cov(xs, ys)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(xy=(np.mean(xs), np.mean(ys)),
                         width=w, height=h, angle=angle,
                         facecolor=color, edgecolor=color,
                         alpha=alpha, lw=0.6, zorder=zorder))


def find_best_session_for_area(combos_df, area):
    """Highest-N (session, object) in this area with both contexts complete."""
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


# =============================================================================
# LOAD INPUTS
# =============================================================================
combos = pd.read_csv(GO_NOGO_DIR / "inventory_event_combos_with_completeness.csv")
df = pd.read_csv(RESULTS_DIR / "session_object_event_results.csv")
d3 = df[df["event"] == EVENT].copy()

# AUC per area for the corner annotation (mean over Event-3 units)
auc_by_area = (
    d3.groupby("area")["dec_exe_obs_auc_full"].mean().to_dict()
)

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                         gridspec_kw={"wspace": 0.32})

for ax, area in zip(axes, AREAS):
    cands = find_best_session_for_area(combos, area)
    if cands is None:
        ax.text(0.5, 0.5, f"no complete\n{area} session",
                ha="center", va="center", transform=ax.transAxes)
        continue

    # Walk candidates until build_tensor succeeds for both contexts
    T_exe = T_obs = None
    chosen = None
    for _, row in cands.iterrows():
        s, o, n = row["session"], row["object"], int(row["n_neurons"])
        te = cu.build_tensor(BASE_DIR, area, s, EXE_CONTEXT, EXE_CONDITION, o,
                             EVENT, INTERVAL, BINSIZE, gauss_smooth=GAUSS_SMOOTH)
        to = cu.build_tensor(BASE_DIR, area, s, OBS_CONTEXT, OBS_CONDITION, o,
                             EVENT, INTERVAL, BINSIZE, gauss_smooth=GAUSS_SMOOTH)
        if te is not None and to is not None and te.shape[0] >= 4 and to.shape[0] >= 4:
            T_exe, T_obs, chosen = te, to, (s, o, n)
            break

    if T_exe is None:
        ax.text(0.5, 0.5, f"no valid {area} tensor", ha="center", va="center",
                transform=ax.transAxes)
        continue

    session, obj, n_neur = chosen
    print(f"{area}: {session} obj={obj} n={n_neur}  T_exe={T_exe.shape}  T_obs={T_obs.shape}")

    scaler, pca, (Z_exe, Z_obs) = cu.fit_shared_pca(T_exe, T_obs, n_components=N_PCS)
    mu_exe = cu.mean_traj(Z_exe)
    mu_obs = cu.mean_traj(Z_obs)
    var_pc12 = pca.explained_variance_ratio_[:2].sum() * 100

    # Mean trajectories
    gradient_line(ax, mu_exe[:, 0], mu_exe[:, 1], COLOR_EXE, lw=2.0)
    gradient_line(ax, mu_obs[:, 0], mu_obs[:, 1], COLOR_OBS, lw=2.0)

    # Start (open circle) / end (filled triangle) markers
    ax.scatter(mu_exe[0, 0], mu_exe[0, 1], s=26, facecolor="white",
               edgecolor=COLOR_EXE, lw=1.2, zorder=6)
    ax.scatter(mu_obs[0, 0], mu_obs[0, 1], s=26, facecolor="white",
               edgecolor=COLOR_OBS, lw=1.2, zorder=6)
    ax.scatter(mu_exe[-1, 0], mu_exe[-1, 1], s=38, marker=">",
               color=COLOR_EXE, zorder=6, edgecolor="white", lw=0.5)
    ax.scatter(mu_obs[-1, 0], mu_obs[-1, 1], s=38, marker=">",
               color=COLOR_OBS, zorder=6, edgecolor="white", lw=0.5)

    # Single-trial post-event clouds + ellipses
    post = slice(int(0.75 * Z_exe.shape[1]), Z_exe.shape[1])
    exe_pts = Z_exe[:, post, :2].mean(axis=1)
    obs_pts = Z_obs[:, post, :2].mean(axis=1)
    ax.scatter(exe_pts[:, 0], exe_pts[:, 1], s=8, color=COLOR_EXE,
               alpha=0.35, lw=0, zorder=4)
    ax.scatter(obs_pts[:, 0], obs_pts[:, 1], s=8, color=COLOR_OBS,
               alpha=0.35, lw=0, zorder=4)
    cov_ellipse(ax, exe_pts[:, 0], exe_pts[:, 1], COLOR_EXE)
    cov_ellipse(ax, obs_pts[:, 0], obs_pts[:, 1], COLOR_OBS)

    # AUC annotation in upper right corner
    auc = auc_by_area.get(area, np.nan)
    ax.text(0.97, 0.97, f"AUC = {auc:.3f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=6.5, color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    # Session info in lower right corner
    ax.text(0.97, 0.03, f"{session}\nn={n_neur}, {var_pc12:.0f}% var",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.8, color="#6b7280")

    ax.set_xlabel("PC1")
    if area == "AIP":
        ax.set_ylabel("PC2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{area}", loc="left", fontweight="bold",
                 color=AREA_COLORS[area], pad=4)

# Single legend in the first subpanel
handles = [
    Line2D([0], [0], color=COLOR_EXE, lw=2.0, label="Execution-Go"),
    Line2D([0], [0], color=COLOR_OBS, lw=2.0, label="Observation-Go"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#374151", markersize=5, lw=0, label="start"),
    Line2D([0], [0], marker=">", color="#374151", markersize=6, lw=0, label="end"),
]
axes[0].legend(handles=handles, loc="lower left", frameon=False,
               handletextpad=0.3, borderpad=0.2, labelspacing=0.25,
               fontsize=5.8)

fig.suptitle("Population state separation: Execution vs Observation at Event 3",
             fontsize=8.5, fontweight="bold", y=1.04, x=0.06, ha="left")

# =============================================================================
# SAVE
# =============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PDF.with_suffix(".png"), dpi=300)
print(f"\nSaved: {OUT_PDF}")
print(f"Saved: {OUT_PDF.with_suffix('.png')}")
