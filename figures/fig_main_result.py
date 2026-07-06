"""
fig_main_result.py
-------------------
Publication-quality figure for the main pre/post Go/No-Go decoding result.

Layout: 2-panel figure, single Nature column width (89 mm).
  Left panel : pooled across all areas, pre vs post AUC at Event 1 vs Event 3
  Right panel: same dissociation broken down by area (AIP, F5, F6)

Per-unit dots are shown faintly with paired pre->post lines; group means are
overlaid as solid markers with bootstrap 95% CIs. Chance is marked at 0.5.

Inputs:  session_object_event_results.csv, meta_summary_by_area_event.csv
Output:  fig_main_result.pdf (vector, ready for \\includegraphics)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Publication rcParams. Set once, inherited by every axis. This block is the
# single biggest difference between a default matplotlib figure and one that
# looks like it belongs in a journal. Reuse it across all thesis figures.
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
    "pdf.fonttype":       42,    # editable text in Illustrator/Inkscape
    "ps.fonttype":        42,
    "savefig.bbox":       "tight",
    "savefig.dpi":        600,
})

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MM = 1 / 25.4                                # mm -> inches helper
FIG_W, FIG_H = 183 * MM, 70 * MM              # double-column wide, ~70 mm tall
COLOR_PRE  = "#9ca3af"                       # neutral grey
COLOR_POST = "#1f4e8c"                       # deep blue
AREA_COLORS = {"AIP": "#d97706", "F5": "#1f4e8c", "F6": "#15803d"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def boot_ci(x, n_boot=2000, ci=95, rng=None):
    """Percentile bootstrap CI for the mean."""
    rng = rng or np.random.default_rng(20260411)
    x = np.asarray(x, dtype=float)
    bs = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(x.mean()), float(lo), float(hi)

def plot_paired(ax, df, color_post, x_offset=0.0, dot_alpha=0.18, jitter=0.05):
    """
    Plot pre/post AUC for one (event-)group on a single axis position.
    Faded paired lines for individual units, solid mean+CI markers on top.
    """
    rng = np.random.default_rng(7)
    pre  = df["dec_obs_go_nogo_auc_pre"].values
    post = df["dec_obs_go_nogo_auc_post"].values

    x_pre  = x_offset - 0.18 + rng.uniform(-jitter, jitter, size=len(pre))
    x_post = x_offset + 0.18 + rng.uniform(-jitter, jitter, size=len(pre))

    # Paired faint lines
    for xp, xq, yp, yq in zip(x_pre, x_post, pre, post):
        ax.plot([xp, xq], [yp, yq], color="#cbd5e1", lw=0.4, alpha=dot_alpha * 3, zorder=1)
    # Faint dots
    ax.scatter(x_pre,  pre,  s=6, color=COLOR_PRE,  alpha=dot_alpha, lw=0, zorder=2)
    ax.scatter(x_post, post, s=6, color=color_post, alpha=dot_alpha, lw=0, zorder=2)

    # Group mean + bootstrap CI
    for x_centre, vals, c in [(x_offset - 0.18, pre, COLOR_PRE),
                              (x_offset + 0.18, post, color_post)]:
        m, lo, hi = boot_ci(vals)
        ax.errorbar(x_centre, m, yerr=[[m - lo], [hi - m]],
                    fmt="o", color=c, ms=4.5, lw=1.1,
                    capsize=2.5, capthick=0.9, zorder=5,
                    markerfacecolor=c, markeredgecolor="white", mew=0.5)

def style_auc_axis(ax, ylabel=True):
    ax.axhline(0.5, color="black", lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.set_ylim(0.30, 1.02)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    if ylabel:
        ax.set_ylabel("Decoding AUC")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
from pathlib import Path
_RESULTS = Path(__file__).resolve().parent.parent / "results"
df = pd.read_csv(_RESULTS / "session_object_event_results.csv")
# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(FIG_W, FIG_H),
    gridspec_kw={"width_ratios": [1, 2.1], "wspace": 0.35},
)

# --- Panel A: pooled across areas -----------------------------------------
for i, ev in enumerate(["Event1", "Event3"]):
    plot_paired(ax1, df[df["event"] == ev], color_post=COLOR_POST, x_offset=i)

ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Event 1\n(cue)", "Event 3\n(Go/No-Go)"])
ax1.set_xlim(-0.55, 1.55)
style_auc_axis(ax1, ylabel=True)
ax1.set_title("Pooled across areas", pad=4, loc="left")

# Pre / post legend, manual so it matches the dot colours
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PRE,
           markersize=4.5, label="Pre-event"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_POST,
           markersize=4.5, label="Post-event"),
]
ax1.legend(handles=legend_handles, loc="upper left", frameon=False,
           handletextpad=0.4, borderpad=0.2, labelspacing=0.3)

# --- Panel B: by area ------------------------------------------------------
positions = []
labels    = []
xpos = 0
for area in ["AIP", "F5", "F6"]:
    for ev in ["Event1", "Event3"]:
        sub = df[(df["area"] == area) & (df["event"] == ev)]
        plot_paired(ax2, sub, color_post=AREA_COLORS[area], x_offset=xpos)
        positions.append(xpos)
        labels.append("E1" if ev == "Event1" else "E3")
        xpos += 1
    xpos += 0.6  # gap between areas

ax2.set_xticks(positions)
ax2.set_xticklabels(labels)
ax2.set_xlim(-0.55, xpos - 0.05)
style_auc_axis(ax2, ylabel=False)
ax2.set_title("By area", pad=4, loc="left")

# Area labels above each pair
group_centres = [0.5, 0.5 + 2.6, 0.5 + 5.2]
for area, xc in zip(["AIP", "F5", "F6"], group_centres):
    ax2.text(xc, 1.06, area, ha="center", va="bottom",
             fontsize=7.5, fontweight="bold",
             color=AREA_COLORS[area], transform=ax2.transData)

# Panel labels (Nature style: bold, upper left, outside axis)
for ax, lab in [(ax1, "a"), (ax2, "b")]:
    ax.text(-0.18, 1.08, lab, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")

out_dir = Path(__file__).resolve().parent / "generated"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_main_result.pdf")
fig.savefig(out_dir / "fig_main_result.png", dpi=600) 
df.to_csv(out_dir / "fig_main_result_data.csv", index=False) # backup raster for slides
print("Saved fig_main_result.pdf and fig_main_result.png")
