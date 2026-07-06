"""
fig_area_robustness.py  (v3)
-----------------------------
Robustness of the Event-3 decoding effect to dataset slicing.

Layout: 2-panel row, double Nature column width (183 mm).
  Panel a: forest plot of post-Event-3 AUC by area (AIP, F5, F6) with
           bootstrap 95% CIs. Wilcoxon p-values are listed in a fixed
           right-hand column in small dark-grey text (visually subordinate
           to the data markers).
  Panel b: forest plot of post-Event-3 AUC for high-N vs low-N subsets
           (pooled, F5, F6 -- AIP not splittable). A subtitle directly
           under the panel title states the split threshold and the
           neuron-count ranges, so the reader knows what 'high' and
           'low' mean without consulting the caption.

Both panels share an x-axis (Decoding AUC) with a vertical dashed chance
line at 0.5, and use the same area colour palette as fig_main_result.py.

v3 changes (relative to v2):
  - Removed delta=0.000 annotation from panel b (rhetorical, not data).
  - Removed coloured per-row p-values from panel a; replaced with a
    quiet right-hand column of dark-grey p-value text aligned to a
    single x-position so they form a column rather than floating.
  - Shortened panel b row labels: "F5 high" instead of "F5 (high-N)".
  - Removed bold/normal weight distinction between high and low rows
    in panel b (position and colour already carry the grouping).
  - Added an in-panel subtitle to panel b stating the median split
    threshold and the neuron-count ranges.
  - Widened wspace between panels to prevent any cross-panel rendering
    overlap of right-hand text.

Inputs : meta_summary_by_area_event.csv,
         neuron_count_sensitivity.csv,
         neuron_count_sensitivity_by_area.csv
Output : fig_area_robustness.pdf, fig_area_robustness.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# Publication rcParams (identical to other thesis figures)
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
FIG_W, FIG_H = 183 * MM, 80 * MM

AREA_COLORS = {"AIP": "#d97706",
               "F5":  "#1f4e8c",
               "F6":  "#15803d"}

AREA_COLORS_LIGHT = {"AIP": "#fdba74",
                     "F5":  "#93c5fd",
                     "F6":  "#86efac"}

X_LIMS = (0.45, 0.90)
P_COLUMN_X = 0.93   # fixed column for p-value text in panel a (data coords)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent / "results"

ma = pd.read_csv(BASE_DIR / "meta_summary_by_area_event.csv")
ma_e3 = ma[ma["event"] == "Event3"].set_index("area")

WILCOX_P = {"AIP": 0.026, "F5": 2.2e-8, "F6": 3.1e-7}

ma  = pd.read_csv(BASE_DIR / "meta_summary_by_area_event.csv")
ns  = pd.read_csv(BASE_DIR / "neuron_count_sensitivity.csv")
nsa = pd.read_csv(BASE_DIR / "neuron_count_sensitivity_by_area.csv")

ma_e3 = ma[ma["event"] == "Event3"].set_index("area")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def draw_forest_row(ax, y, mean, lo, hi, color, ms=5.5):
    ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                fmt="o", color=color, ms=ms, lw=1.1,
                capsize=2.5, capthick=0.9,
                markerfacecolor=color, markeredgecolor="white", mew=0.5,
                zorder=4)

def style_forest_axis(ax, row_positions, x_max=None):
    ax.axvline(0.5, color="black", lw=0.55, ls=(0, (3, 3)), zorder=0)
    ax.set_xlim(X_LIMS[0], x_max if x_max is not None else X_LIMS[1])
    ax.set_ylim(min(row_positions) - 0.7, max(row_positions) + 0.7)
    ax.set_yticks([])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xlabel("Decoding AUC (post Event 3)")
    ax.spines["left"].set_visible(False)

def fmt_p(p):
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        mant = p / 10**exp
        return f"p = {mant:.0f}\u00d710$^{{{exp}}}$"
    return f"p = {p:.3f}"

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(FIG_W, FIG_H),
    gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.45},
)

# --- Panel A: area comparison ---------------------------------------------
areas_top_to_bot = ["F5", "F6", "AIP"]
y_positions_a = list(range(len(areas_top_to_bot) - 1, -1, -1))

for area, y in zip(areas_top_to_bot, y_positions_a):
    row = ma_e3.loc[area]
    n_units = int(row["n_units"])
    color = AREA_COLORS[area]

    draw_forest_row(
        axA, y,
        mean=row["dec_obs_go_nogo_auc_post_mean"],
        lo  =row["dec_obs_go_nogo_auc_post_ci95_low"],
        hi  =row["dec_obs_go_nogo_auc_post_ci95_high"],
        color=color,
    )
    # Left-side row label, area-coloured.
    axA.text(X_LIMS[0] - 0.015, y, f"{area}  (n = {n_units})",
             ha="right", va="center", fontsize=7, color=color,
             fontweight="bold")

style_forest_axis(axA, row_positions=y_positions_a)
axA.set_title("Area comparison", pad=4, loc="left", fontweight="bold")

# --- Panel B: high-N vs low-N sensitivity ---------------------------------
b_entries = []  # (group, sub, mean, lo, hi, color, label)

# Pooled
hN = ns[ns["subset"] == "high_N"].iloc[0]
lN = ns[ns["subset"] == "low_N"].iloc[0]
b_entries.append(("Pooled", "high", hN["mean"], hN["ci95_low"],
                  hN["ci95_high"], "#374151", "Pooled high"))
b_entries.append(("Pooled", "low",  lN["mean"], lN["ci95_low"],
                  lN["ci95_high"], "#9ca3af", "Pooled low"))
# F5, F6
for area in ["F5", "F6"]:
    for sub_name in ["high_N", "low_N"]:
        r = nsa[(nsa["area"] == area) & (nsa["subset"] == sub_name)].iloc[0]
        color = AREA_COLORS[area] if sub_name == "high_N" else AREA_COLORS_LIGHT[area]
        suffix = "high" if sub_name == "high_N" else "low"
        b_entries.append((area, suffix, r["mean"], r["ci95_low"],
                          r["ci95_high"], color, f"{area} {suffix}"))
# AIP (high only)
r = nsa[(nsa["area"] == "AIP") & (nsa["subset"] == "high_N")].iloc[0]
b_entries.append(("AIP", "high", r["mean"], r["ci95_low"], r["ci95_high"],
                  AREA_COLORS["AIP"], "AIP high only"))

# y-positions: tight pairs, gap between groups
y_positions_b = []
y = 0.0
prev_group = None
for group, *_ in b_entries:
    if prev_group is not None and group != prev_group:
        y += 0.55
    y_positions_b.append(y)
    y += 1.0
    prev_group = group
y_positions_b = [max(y_positions_b) - p for p in y_positions_b]

for (group, sub, m, lo, hi, color, label), yp in zip(b_entries, y_positions_b):
    draw_forest_row(axB, yp, m, lo, hi, color=color)
    axB.text(X_LIMS[0] - 0.015, yp, label, ha="right", va="center",
             fontsize=7, color=color)

style_forest_axis(axB, row_positions=y_positions_b)
axB.set_title("Neuron-count sensitivity", pad=4, loc="left", fontweight="bold")

# Panel labels
for ax, lab in [(axA, "a"), (axB, "b")]:
    ax.text(-0.30, 1.06, lab, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top", ha="left")
out_dir = Path(__file__).resolve().parent / "generated"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig_area_robustness.pdf")
fig.savefig(out_dir / "fig_area_robustness.png", dpi=600)
ns.to_csv(out_dir / "fig_area_robustness_neuron_count_sensitivity.csv", index=False)
nsa.to_csv(out_dir / "fig_area_robustness_neuron_count_sensitivity_by_area.csv", index=False)
ma_e3.reset_index().to_csv(out_dir / "fig_area_robustness_area_comparison.csv", index=False)

print("Saved fig_area_robustness.pdf and fig_area_robustness.png")
