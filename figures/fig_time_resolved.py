"""
fig_time_resolved.py
---------------------
Publication-quality figure for the time-resolved Go/No-Go decoding result.

Layout: 3-panel row, double Nature column width (183 mm).
  One panel per area (AIP, F5, F6). Each panel shows two AUC traces
  (Event 1 in grey, Event 3 in the area's signature colour) aligned to
  event onset, with a thin SEM ribbon around each mean trace.
  Vertical dashed line at t = 0 marks event onset; horizontal dashed
  line at AUC = 0.5 marks chance.

Note on the shuffle null: the per-unit shuffle null in
shuffle_null_time_resolved.csv was computed independently for each
(session x object) unit. Averaging its upper-CI bounds across units
produces an overly wide band that reflects per-unit noise rather than
group-level noise. The statistically correct comparison for a group-mean
trace is a group-level shuffle null (averaging across units within each
shuffle iteration), which would require re-running the shuffle pipeline.
The per-unit null is therefore reported only as a consistency check in
the shuffle-null subsection of Results, and not drawn here. Chance is
shown as the horizontal dashed line, and statistical significance of
the rise is established in the pre/post and mixed-effects analyses.

Inputs : time_resolved_results.csv
Output : fig_time_resolved.pdf, fig_time_resolved.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Publication rcParams (identical to fig_main_result.py for consistency)
# ---------------------------------------------------------------------------
rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          7,
    "axes.titlesize":     8,
    "axes.labelsize":     7,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "legend.fontsize":    6.5,
    "axes.linewidth":     0.75,
    "xtick.major.width":  0.75,
    "ytick.major.width":  0.75,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "savefig.bbox":       "tight",
    "savefig.dpi":        600,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MM = 1 / 25.4
FIG_W, FIG_H = 183 * MM, 62 * MM

COLOR_E1 = "#9ca3af"                          # neutral grey, matches "pre" colour
AREA_COLORS_E3 = {"AIP": "#d97706",           # same palette as fig_main_result
                  "F5":  "#1f4e8c",
                  "F6":  "#15803d"}

X_LIMS = (-0.80, 0.62)
Y_LIMS = (0.40, 0.85)

# ---------------------------------------------------------------------------
# Load and aggregate
# ---------------------------------------------------------------------------
from pathlib import Path
_RESULTS = Path(__file__).resolve().parent.parent / "results"
tr = pd.read_csv(_RESULTS / "time_resolved_results.csv")

def aggregate_obs(df, area, event):
    """Mean AUC and SEM across (session, object) units at each time bin."""
    sub = df[(df["area"] == area) & (df["event"] == event)]
    g = sub.groupby("center_time_s")["auc"].agg(["mean", "sem"]).reset_index()
    g = g.sort_values("center_time_s")
    return g["center_time_s"].values, g["mean"].values, g["sem"].values

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                         sharey=True, gridspec_kw={"wspace": 0.18})

for ax, area in zip(axes, ["AIP", "F5", "F6"]):
    e3_color = AREA_COLORS_E3[area]

    # ----- Event 1 (grey) -----
    t1, m1, s1 = aggregate_obs(tr, area, "Event1")
    ax.fill_between(t1, m1 - s1, m1 + s1, color=COLOR_E1, alpha=0.30, lw=0, zorder=2)
    ax.plot(t1, m1, color=COLOR_E1, lw=1.2, zorder=3, label="Event 1 (cue)")

    # ----- Event 3 (area colour) -----
    t3, m3, s3 = aggregate_obs(tr, area, "Event3")
    ax.fill_between(t3, m3 - s3, m3 + s3, color=e3_color, alpha=0.25, lw=0, zorder=2)
    ax.plot(t3, m3, color=e3_color, lw=1.5, zorder=4, label="Event 3 (Go/No-Go)")

    # ----- Reference lines -----
    ax.axhline(0.5, color="black", lw=0.55, ls=(0, (3, 3)), zorder=0)
    ax.axvline(0.0, color="black", lw=0.55, ls=(0, (2, 2)), zorder=0)

    # ----- Cosmetics -----
    ax.set_xlim(*X_LIMS)
    ax.set_ylim(*Y_LIMS)
    ax.set_xticks([-0.6, -0.3, 0.0, 0.3, 0.6])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("Time from event onset (s)")
    ax.set_title(area, pad=3, loc="left", fontweight="bold", color=e3_color)

axes[0].set_ylabel("Decoding AUC")
axes[0].legend(loc="upper left", frameon=False, handletextpad=0.4,
               borderpad=0.2, labelspacing=0.3)

# Panel labels
for ax, lab in zip(axes, ["a", "b", "c"]):
    ax.text(-0.16, 1.06, lab, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")


out_dir = Path(__file__).resolve().parent / "generated"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_time_resolved.pdf")
fig.savefig(out_dir / "fig_time_resolved.png", dpi=600)
tr.to_csv(out_dir / "fig_time_resolved.csv", index=False) # backup raster for slides

print("Saved fig_time_resolved.pdf and fig_time_resolved.png")
