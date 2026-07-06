"""
15_thesis_figures.py
====================
Generates six publication-quality thesis figures.

  fig1_main_result.png      -- AUC by area x event (core finding)
  fig2_time_resolved.png    -- Temporal decoding dynamics per area
  fig3_prepost_event3.png   -- Pre vs post AUC for Event3 (paired)
  fig4_dpca_variance.png    -- dPCA variance decomposition
  fig5_trajectory_dist.png  -- Neural trajectory distance geometry
  fig6_exe_obs_scatter.png  -- Exe-Obs vs Obs-GoNogo (motor coding test)

Run from the go-nogo root directory:
  python reworked_pipeline/15_thesis_figures.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "figure.dpi":        150,
})

AREA_COLORS = {"AIP": "#4C72B0", "F5": "#DD8452", "F6": "#55A868"}
EVENT_COLORS = {"Event1": "#BBBBBB", "Event3": "#E05A2B"}
EVENT_LABELS = {"Event1": "Event 1 (baseline)", "Event3": "Event 3 (movement)"}
AREA_ORDER = ["AIP", "F5", "F6"]

CHANCE = 0.5


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "n.s."


def _wilcoxon_greater(a, b):
    if len(a) < 3:
        return float("nan")
    try:
        _, p = wilcoxon(a, b, alternative="greater")
        return p
    except Exception:
        return float("nan")


def _add_chance(ax, xmin=None, xmax=None, **kw):
    kw.setdefault("color", "gray")
    kw.setdefault("linewidth", 1.0)
    kw.setdefault("linestyle", "--")
    kw.setdefault("zorder", 1)
    if xmin is not None:
        ax.axhline(CHANCE, xmin=xmin, xmax=xmax, **kw)
    else:
        ax.axhline(CHANCE, **kw)


def _strip_jitter(n: int, center: float = 0.0, width: float = 0.18, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(center - width, center + width, size=n)


# ---------------------------------------------------------------------------
# Fig 1 — Main result: AUC by area x event
# ---------------------------------------------------------------------------

def fig1_main_result(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    auc_col = "dec_obs_go_nogo_auc_full"
    events  = ["Event1", "Event3"]
    n_areas = len(AREA_ORDER)
    group_w = 0.9   # total width per area
    bar_w   = group_w / 2 - 0.04

    xticks, xlabels = [], []

    for ai, area in enumerate(AREA_ORDER):
        cx = ai * (group_w + 0.4)       # centre of area group

        for ei, event in enumerate(events):
            sub  = df[(df["area"] == area) & (df["event"] == event)][auc_col].dropna().values
            x0   = cx + (ei - 0.5) * (bar_w + 0.04)
            col  = EVENT_COLORS[event]
            jx   = _strip_jitter(len(sub), center=x0, width=0.11, seed=ai*10+ei)

            # box (manual percentiles)
            q25, q50, q75 = np.percentile(sub, [25, 50, 75])
            iqr = q75 - q25
            wlo = max(sub[sub >= q25 - 1.5*iqr].min(), sub.min())
            whi = min(sub[sub <= q75 + 1.5*iqr].max(), sub.max())

            # whisker + box
            ax.plot([x0, x0], [wlo, q25], color=col, lw=1.2, zorder=2)
            ax.plot([x0, x0], [q75, whi], color=col, lw=1.2, zorder=2)
            rect = plt.Rectangle((x0 - bar_w/2, q25), bar_w, iqr,
                                  facecolor=col, alpha=0.30, edgecolor=col,
                                  linewidth=1.2, zorder=2)
            ax.add_patch(rect)
            ax.plot([x0 - bar_w/2, x0 + bar_w/2], [q50, q50],
                    color=col, lw=2.2, zorder=3)

            # strip
            ax.scatter(jx, sub, color=col, s=22, alpha=0.70,
                       edgecolors="white", linewidths=0.4, zorder=4)

        # significance annotation: Event3 vs Event1
        sub3 = df[(df["area"] == area) & (df["event"] == "Event3")][auc_col].dropna().values
        sub1 = df[(df["area"] == area) & (df["event"] == "Event1")][auc_col].dropna().values
        p = _wilcoxon_greater(sub3, sub1)
        stars = _sig_stars(p)
        top   = max(sub3.max(), sub1.max()) + 0.04
        x1_pos = cx - (bar_w/2 + 0.02)
        x2_pos = cx + (bar_w/2 + 0.02)
        ax.plot([x1_pos, x1_pos, x2_pos, x2_pos],
                [top - 0.01, top + 0.005, top + 0.005, top - 0.01],
                lw=1.0, color="#333333", zorder=5)
        ax.text(cx, top + 0.015, stars, ha="center", va="bottom",
                fontsize=9, color="#333333", zorder=5)

        xticks.append(cx)
        xlabels.append(area)

    _add_chance(ax)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=11)
    ax.set_ylabel("AUC  (Go vs No-Go decoding)", fontsize=11)
    ax.set_ylim(0.12, 1.02)
    ax.set_xlim(-0.55, (n_areas - 1) * 1.3 + 0.55)

    # Legend
    patch1 = mpatches.Patch(facecolor=EVENT_COLORS["Event1"], alpha=0.55,
                             edgecolor=EVENT_COLORS["Event1"], label=EVENT_LABELS["Event1"])
    patch3 = mpatches.Patch(facecolor=EVENT_COLORS["Event3"], alpha=0.55,
                             edgecolor=EVENT_COLORS["Event3"], label=EVENT_LABELS["Event3"])
    chance_line = mlines.Line2D([], [], color="gray", linestyle="--", lw=1.0, label="Chance (0.5)")
    ax.legend(handles=[patch1, patch3, chance_line],
              loc="upper left", fontsize=9, framealpha=0.85)

    ax.set_title("Go/No-Go Observation Decoding  —  Full Window AUC", fontsize=11)
    fig.tight_layout()
    p = out / "fig1_main_result.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Fig 2 — Time-resolved decoding
# ---------------------------------------------------------------------------

def fig2_time_resolved(tr: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)

    for ax, area in zip(axes, AREA_ORDER):
        for event in ["Event1", "Event3"]:
            sub = tr[(tr["area"] == area) & (tr["event"] == event)]
            grp = sub.groupby("center_time_s")["auc"]
            t   = np.array(sorted(grp.groups.keys()))
            mn  = grp.mean().loc[t].values
            se  = grp.sem().loc[t].values

            col = EVENT_COLORS[event]
            ax.plot(t, mn, color=col, lw=2.0, zorder=3,
                    label=EVENT_LABELS[event])
            ax.fill_between(t, mn - se, mn + se,
                            color=col, alpha=0.18, zorder=2)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
        _add_chance(ax)
        ax.set_title(area, fontsize=12, color=AREA_COLORS[area], fontweight="bold")
        ax.set_xlabel("Time relative to event onset (s)", fontsize=10)
        ax.spines["left"].set_visible(True)

    axes[0].set_ylabel("Mean AUC  (Go vs No-Go)", fontsize=10)
    axes[0].set_ylim(0.35, 0.88)

    # Shared legend on rightmost panel
    patch1 = mpatches.Patch(facecolor=EVENT_COLORS["Event1"], alpha=0.60,
                             label=EVENT_LABELS["Event1"])
    patch3 = mpatches.Patch(facecolor=EVENT_COLORS["Event3"], alpha=0.60,
                             label=EVENT_LABELS["Event3"])
    ev0_line = mlines.Line2D([], [], color="black", linestyle="--", lw=0.8, label="Event onset (t=0)")
    chance_line = mlines.Line2D([], [], color="gray", linestyle="--", lw=1.0, label="Chance (0.5)")
    axes[2].legend(handles=[patch1, patch3, ev0_line, chance_line],
                   fontsize=8.5, framealpha=0.85, loc="upper left")

    fig.suptitle("Time-Resolved Go/No-Go Decoding  (sliding 5-bin window, 20 ms/bin)",
                 fontsize=10.5, y=1.01)
    fig.tight_layout()
    p = out / "fig2_time_resolved.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Fig 3 — Pre / Post AUC for Event 3 (paired)
# ---------------------------------------------------------------------------

def fig3_prepost(df: pd.DataFrame, out: Path) -> None:
    e3 = df[df["event"] == "Event3"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)

    for ax, area in zip(axes, AREA_ORDER):
        sub   = e3[e3["area"] == area][["dec_obs_go_nogo_auc_pre",
                                         "dec_obs_go_nogo_auc_post"]].dropna()
        pre   = sub["dec_obs_go_nogo_auc_pre"].values
        post  = sub["dec_obs_go_nogo_auc_post"].values
        n     = len(pre)
        col   = AREA_COLORS[area]

        # Paired lines (light gray)
        jx_pre  = _strip_jitter(n, center=0.0, width=0.10, seed=hash(area) % 99)
        jx_post = _strip_jitter(n, center=1.0, width=0.10, seed=hash(area) % 99 + 1)
        for xp, xq, vp, vq in zip(jx_pre, jx_post, pre, post):
            ax.plot([xp, xq], [vp, vq], color=col, alpha=0.22, lw=0.9, zorder=2)

        # Strip dots
        ax.scatter(jx_pre,  pre,  color=col, s=28, alpha=0.80,
                   edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter(jx_post, post, color=col, s=28, alpha=0.80,
                   edgecolors="white", linewidths=0.4, zorder=4)

        # Mean ± SEM bars
        for xi, vals in zip([0, 1], [pre, post]):
            mn = vals.mean()
            se = vals.std(ddof=1) / np.sqrt(len(vals))
            ax.errorbar(xi, mn, yerr=se, fmt="o", color="black",
                        markersize=6, capsize=4, lw=2.0, zorder=5)

        # Significance bracket
        _, p = wilcoxon(post, pre, alternative="greater")
        top = max(post.max(), pre.max()) + 0.06
        ax.plot([0, 0, 1, 1], [top - 0.02, top, top, top - 0.02],
                lw=1.0, color="#333333")
        ax.text(0.5, top + 0.01, _sig_stars(p), ha="center", fontsize=9)

        _add_chance(ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre\n(t < 0 s)", "Post\n(t >= 0 s)"], fontsize=10)
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(area, fontsize=12, color=col, fontweight="bold")

    axes[0].set_ylabel("AUC  (Go vs No-Go decoding)", fontsize=10)
    axes[0].set_ylim(0.10, 1.06)
    fig.suptitle("Pre- vs Post-Onset Decoding  —  Event 3", fontsize=11, y=1.01)
    fig.tight_layout()
    p = out / "fig3_prepost_event3.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Fig 4 — dPCA variance decomposition
# ---------------------------------------------------------------------------

def fig4_dpca(dpca: pd.DataFrame, out: Path) -> None:
    # Columns: var_t, var_d, var_o, var_dt, var_ot, var_do, var_dot
    components = ["var_t_mean",  "var_d_mean",  "var_o_mean",
                  "var_dt_mean", "var_ot_mean", "var_do_mean", "var_dot_mean"]
    labels_c   = ["Time",   "Decision\n(Go/No-Go)", "Object",
                  "Time x\nDecision", "Time x\nObject",
                  "Dec x\nObject",    "Three-way"]
    cmap = matplotlib.colormaps["tab10"]
    colors_c = [cmap(i) for i in range(len(components))]

    # Build per-row label and values
    rows = []
    for _, r in dpca.iterrows():
        rows.append({
            "label": f"{r['area']}  {r['event'].replace('Event', 'E')}",
            "area":  r["area"],
            "event": r["event"],
            "vals":  [r[c] for c in components],
        })
    # Sort: AIP/F5/F6 x E1/E3
    order = [(a, e) for a in AREA_ORDER for e in ["Event1", "Event3"]]
    rows.sort(key=lambda r: order.index((r["area"], r["event"])))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bar_h = 0.65
    yticks, ylabels = [], []

    for ri, row in enumerate(rows):
        y  = ri * (bar_h + 0.25)
        left = 0.0
        for ci, (val, col) in enumerate(zip(row["vals"], colors_c)):
            ax.barh(y, val, left=left, height=bar_h,
                    color=col, edgecolor="white", linewidth=0.4)
            if val > 0.025:
                ax.text(left + val / 2, y, f"{val:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white", fontweight="bold")
            left += val
        yticks.append(y)
        # Color area label
        ec = EVENT_COLORS[row["event"]]
        ylabels.append(row["label"])

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9.5)

    # Color the y-tick labels by event
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(EVENT_COLORS[row["event"]])

    ax.set_xlabel("Fraction of total variance explained", fontsize=10)
    ax.set_xlim(0, 1.02)
    ax.set_title("dPCA Variance Decomposition  (mean per area x event)", fontsize=11)

    # Legend
    patches = [mpatches.Patch(facecolor=colors_c[i], label=labels_c[i])
               for i in range(len(components))]
    ax.legend(handles=patches, loc="lower right",
              ncol=2, fontsize=8, framealpha=0.85)

    # Event legend (y-label colors)
    p1 = mpatches.Patch(facecolor=EVENT_COLORS["Event1"], label=EVENT_LABELS["Event1"])
    p3 = mpatches.Patch(facecolor=EVENT_COLORS["Event3"], label=EVENT_LABELS["Event3"])
    ax.legend(handles=[p1, p3] + patches,
              loc="lower right", ncol=2, fontsize=8, framealpha=0.85)

    fig.tight_layout()
    p = out / "fig4_dpca_variance.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Fig 5 — Proportion of session-objects with reliable Go/No-Go coding
# ---------------------------------------------------------------------------

def fig5_proportion_sig(df: pd.DataFrame, out: Path) -> None:
    """
    Two-metric reliability figure per area:
      Left  y-axis: proportion with permutation-test p < 0.05
      Right y-axis: (same scale) proportion with bootstrap 95% CI excluding chance

    A grouped bar chart with two pairs of bars (E1 grey / E3 coloured) per area.
    The figure answers: "In how many session-objects does the coding reliably
    exceed chance, and does this change between Event 1 and Event 3?"
    """
    metrics = [
        ("perm_sig",   "Permutation p < 0.05",     "solid"),
        ("boot_excl",  "Bootstrap 95% CI\nexcludes chance (0.5)", "//"),
    ]
    # Derive boolean columns
    df = df.copy()
    df["perm_sig"]  = df["perm_obs_go_nogo_auc_p"] < 0.05
    df["boot_excl"] = df["boot_obs_go_nogo_auc_ci95_low"] > CHANCE

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bar_w   = 0.18
    gap     = 0.06
    metric_spacing = bar_w + gap
    event_spacing  = len(metrics) * metric_spacing + 0.25
    area_spacing   = len(["Event1", "Event3"]) * event_spacing + 0.45

    xtick_pos, xtick_lbl = [], []

    for ai, area in enumerate(AREA_ORDER):
        ax_base = ai * area_spacing

        for ei, event in enumerate(["Event1", "Event3"]):
            sub   = df[(df["area"] == area) & (df["event"] == event)]
            n     = len(sub)
            ev_base = ax_base + ei * event_spacing
            ecol  = EVENT_COLORS[event]

            for mi, (col_name, _, hatch) in enumerate(metrics):
                prop = sub[col_name].mean()        # fraction True
                xi   = ev_base + mi * metric_spacing
                h    = "////" if hatch != "solid" else ""
                ax.bar(xi, prop * 100, width=bar_w,
                       color=ecol,
                       alpha=0.80 if event == "Event3" else 0.40,
                       hatch=h, edgecolor=ecol if hatch == "solid" else "#555555",
                       linewidth=0.8, zorder=3)
                # Annotate the bar value
                ax.text(xi, prop * 100 + 1.5,
                        f"{prop*100:.0f}%",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#333333")

            # x-tick at centre of this event's two bars
            ev_centre = ev_base + (len(metrics) - 1) * metric_spacing / 2
            xtick_pos.append(ev_centre)
            ev_short = "E1" if event == "Event1" else "E3"
            xtick_lbl.append(f"{ev_short}")

        # Area label at centre of area group (in axes coords y, data coords x)
        area_centre = ax_base + (len(["Event1","Event3"]) * event_spacing - gap) / 2
        ax.text(area_centre, -9,
                area, ha="center", va="top",
                fontsize=12, color=AREA_COLORS[area],
                fontweight="bold", clip_on=False)

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=9)
    ax.set_ylabel("Session-objects showing reliable coding (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.axhline(5, color="gray", lw=0.8, linestyle=":", zorder=1,
               label="5% false-positive rate")
    ax.set_xlim(-0.30, (len(AREA_ORDER) - 1) * area_spacing + event_spacing)

    # Legend: event colours + hatch for metric
    leg_handles = [
        mpatches.Patch(facecolor=EVENT_COLORS["Event1"], alpha=0.50,
                       label=EVENT_LABELS["Event1"]),
        mpatches.Patch(facecolor=EVENT_COLORS["Event3"], alpha=0.85,
                       label=EVENT_LABELS["Event3"]),
        mpatches.Patch(facecolor="white", edgecolor="#555555", hatch="////",
                       label="Bootstrap 95% CI excl. chance"),
        mpatches.Patch(facecolor="white", edgecolor="#555555", hatch="",
                       label="Permutation p < 0.05"),
        mlines.Line2D([], [], color="gray", linestyle=":", lw=0.8,
                      label="5% FPR reference"),
    ]
    ax.legend(handles=leg_handles, fontsize=8.5, framealpha=0.85,
              loc="upper left", ncol=2)

    ax.set_title(
        "Reliability of Go/No-Go Observation Coding\n"
        "Proportion of session-objects with above-chance decoding",
        fontsize=11,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)
    p = out / "fig5_proportion_sig.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Fig 6 — Bootstrap CI forest plot: area × event mean AUC
# ---------------------------------------------------------------------------

def fig6_bootstrap_forest(df: pd.DataFrame, out: Path) -> None:
    """
    Forest plot of bootstrapped mean AUC (Go vs No-Go, Observation) with
    95% confidence intervals, broken down by area and event.

    One row per area × event combination.  CIs that exclude chance (0.5)
    are drawn in the area colour; those that include chance are drawn grey.
    Individual session-object AUCs are shown as translucent strip on the right.

    This figure shows:
      1. Whether the group-level estimate is reliably above chance.
      2. The spread of individual session-objects within each condition.
      3. The Event1 vs Event3 contrast at a glance.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    auc_col  = "dec_obs_go_nogo_auc_full"
    ci_lo    = "boot_obs_go_nogo_auc_ci95_low"
    ci_hi    = "boot_obs_go_nogo_auc_ci95_high"

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    row_h  = 1.0          # vertical spacing per row
    events = ["Event1", "Event3"]
    rows: list[dict] = []

    for area in reversed(AREA_ORDER):         # bottom = AIP, top = F6
        for event in reversed(events):        # bottom = E1, top = E3
            sub = df[(df["area"] == area) & (df["event"] == event)]
            vals = sub[auc_col].dropna().values
            lo   = sub[ci_lo].mean()
            hi   = sub[ci_hi].mean()
            mn   = sub[auc_col].mean()
            rows.append(dict(area=area, event=event, vals=vals,
                             lo=lo, hi=hi, mn=mn))

    for ri, row in enumerate(rows):
        y     = ri * row_h
        col   = AREA_COLORS[row["area"]]
        ci_excl_chance = row["lo"] > CHANCE   # CI excludes chance?
        line_col = col if ci_excl_chance else "#AAAAAA"
        dot_col  = col if ci_excl_chance else "#AAAAAA"

        # CI line
        ax.plot([row["lo"], row["hi"]], [y, y],
                color=line_col, lw=2.5, solid_capstyle="round", zorder=3)
        # Mean dot
        ax.scatter([row["mn"]], [y], color=dot_col, s=60,
                   zorder=4, edgecolors="white", linewidths=0.8)
        # CI end ticks
        for xv in [row["lo"], row["hi"]]:
            ax.plot([xv, xv], [y - 0.12, y + 0.12],
                    color=line_col, lw=1.5, zorder=3)

        # Individual session-object jitter (very small, stacked above/below row)
        jy = y + _strip_jitter(len(row["vals"]), center=0.0, width=0.28,
                               seed=ri * 17)
        ax.scatter(row["vals"], jy, color=col, s=10, alpha=0.35,
                   edgecolors="none", zorder=2)

        # Row label
        ev_short = "E1" if row["event"] == "Event1" else "E3"
        lbl_col  = EVENT_COLORS[row["event"]]
        ax.text(-0.005, y,
                f"{row['area']}  {ev_short}",
                ha="right", va="center", fontsize=9,
                color=lbl_col, fontweight="bold",
                transform=ax.get_yaxis_transform())

    # Chance line
    ax.axvline(CHANCE, color="gray", lw=1.0, linestyle="--", zorder=1)
    ax.text(CHANCE + 0.005, len(rows) * row_h - 0.2,
            "Chance\n(0.5)", color="gray", fontsize=8, va="top")

    ax.set_yticks([])
    ax.set_xlabel("AUC  (Go vs No-Go, Observation)", fontsize=11)
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(-0.7, len(rows) * row_h - 0.3)
    ax.spines["left"].set_visible(False)

    # Legend
    filled_ci  = mlines.Line2D([], [], color=AREA_COLORS["AIP"], lw=2.5,
                                marker="o", markersize=5, markerfacecolor=AREA_COLORS["AIP"],
                                label="95% bootstrap CI (excludes chance)")
    grey_ci    = mlines.Line2D([], [], color="#AAAAAA", lw=2.5,
                                marker="o", markersize=5, markerfacecolor="#AAAAAA",
                                label="95% bootstrap CI (includes chance)")
    ind_dots   = mlines.Line2D([], [], color="gray", marker="o", lw=0,
                                markersize=5, alpha=0.45,
                                label="Individual session-objects")
    chance_ln  = mlines.Line2D([], [], color="gray", linestyle="--", lw=1.0,
                                label="Chance (AUC = 0.5)")
    ax.legend(handles=[filled_ci, grey_ci, ind_dots, chance_ln],
              fontsize=8.5, framealpha=0.88, loc="lower right")

    ax.set_title(
        "Bootstrap 95% CIs  —  Go/No-Go Observation Decoding\n"
        "by Area and Event  (colour = CI excludes chance)",
        fontsize=11,
    )
    fig.tight_layout()
    p = out / "fig6_bootstrap_forest.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent.parent / "results"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "generated"),
    )
    args = ap.parse_args()

    res = Path(args.results_dir)
    out = Path(args.out_dir)  # generated figures (git-ignored)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df   = pd.read_csv(str(res / "session_object_event_results.csv"))
    tr   = pd.read_csv(str(res / "time_resolved_results.csv"))
    dpca = pd.read_csv(str(res / "final_analysis" / "09_dpca" / "dpca_variance_summary.csv"))

    print("\nGenerating figures...")
    fig1_main_result(df, out)
    fig2_time_resolved(tr, out)
    fig3_prepost(df, out)
    fig4_dpca(dpca, out)
    fig5_proportion_sig(df, out)
    fig6_bootstrap_forest(df, out)

    print("\nAll figures saved to:", out)


if __name__ == "__main__":
    main()
