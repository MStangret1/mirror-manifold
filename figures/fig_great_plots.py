"""
Six figures (PNG + SVG) written to the output directory:

  plotA_cross_event_generalization
  plotB_decision_trajectory
  plotC_null_overlay_significance
  plotD_prepost_upgraded
  plotE_reliability
  plotF_divergence_cdf
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib as mpl
# Global style.
mpl.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "legend.fontsize":      8,
    "figure.titlesize":     11,
    "axes.linewidth":       0.8,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "lines.linewidth":      1.4,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.format":       "png",
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon

warnings.filterwarnings("ignore", category=FutureWarning)

# Color constants.
AIP_COLOR  = "#1f77b4"   # blue
F5_COLOR   = "#ff7f0e"   # orange
F6_COLOR   = "#2ca02c"   # green
E1_COLOR   = "#888888"   # grey  (Event 1 baseline)
E3_COLOR   = "#ff7f0e"   # orange (same as F5 — E3 only appears with area context)
NULL_COLOR = "#cccccc"   # light grey (null/chance)

AREA_COLORS  = {"AIP": AIP_COLOR, "F5": F5_COLOR, "F6": F6_COLOR}
EVENT_COLORS = {"Event1": E1_COLOR, "Event3": E3_COLOR}
EVENT_LABELS = {"Event1": "Event 1 (baseline)", "Event3": "Event 3 (movement)"}
AREA_ORDER   = ["AIP", "F5", "F6"]
CHANCE       = 0.5


# helpers

def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    """Save figure as PNG (300 dpi) and SVG. Print one-line confirmation each."""
    for ext in ("png", "svg"):
        p = out / f"{stem}.{ext}"
        fig.savefig(str(p))
        print(f"  → saved {p.name}")
    plt.close(fig)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    finite = np.isfinite(pvals)
    q = np.full_like(pvals, np.nan)
    if not finite.any():
        return q
    pv    = pvals[finite]
    n     = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    inv = np.empty(n); inv[order] = q_ranked
    q[finite] = inv
    return q


def _wilcoxon_greater(a: np.ndarray, b: np.ndarray) -> float:
    diffs = np.asarray(a, float) - np.asarray(b, float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 3:
        return np.nan
    try:
        _, p = scipy_wilcoxon(diffs, alternative="greater")
        return float(p)
    except Exception:
        return np.nan


def _wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p_hat  = k / n
    denom  = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(p_hat), float(centre - margin), float(centre + margin)


def _sig_stars(p: float) -> str:
    if not np.isfinite(p): return "n.s."
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def _add_chance(ax, **kw):
    kw.setdefault("color", "#777777"); kw.setdefault("lw", 0.9)
    kw.setdefault("linestyle", "--"); kw.setdefault("zorder", 1)
    ax.axhline(CHANCE, **kw)


def _find_onset(t: np.ndarray, sig: np.ndarray, n_consec: int = 5) -> float:
    count = 0
    for i, s in enumerate(sig):
        if s:
            count += 1
            if count >= n_consec:
                return float(t[i - n_consec + 1])
        else:
            count = 0
    return np.nan


def _jitter(n, center=0.0, width=0.12, seed=42):
    return np.random.default_rng(seed).uniform(center - width, center + width, size=n)


# PLOT A - Cross-event generalisation heatmap

def plotA(res: Path, out: Path) -> None:
    summary  = pd.read_csv(res / "final_analysis/10_cross_event_generalization"
                               "/cross_event_generalization_summary.csv")
    per_unit = pd.read_csv(res / "final_analysis/10_cross_event_generalization"
                               "/cross_event_generalization_per_unit.csv")

    fig = plt.figure(figsize=(12, 7.5))
    gs  = gridspec.GridSpec(2, 3, height_ratios=[2.4, 1], hspace=0.45, wspace=0.38)

    AUC_KEYS = {
        ("E1", "E1"): ("auc_within_Event1_mean",    "auc_within_Event1_ci95_low",    "auc_within_Event1_ci95_high"),
        ("E3", "E3"): ("auc_within_Event3_mean",    "auc_within_Event3_ci95_low",    "auc_within_Event3_ci95_high"),
        ("E1", "E3"): ("auc_Event1_to_Event3_mean", "auc_Event1_to_Event3_ci95_low", "auc_Event1_to_Event3_ci95_high"),
        ("E3", "E1"): ("auc_Event3_to_Event1_mean", "auc_Event3_to_Event1_ci95_low", "auc_Event3_to_Event1_ci95_high"),
    }
    TRAIN_ROWS = ["E1", "E3"]
    TEST_COLS  = ["E1", "E3"]

    cmap = mpl.colormaps["RdBu_r"]
    norm = TwoSlopeNorm(vmin=0.45, vcenter=0.50, vmax=0.75)

    gti_axes = []

    for ci, area in enumerate(AREA_ORDER):
        row_sum = summary[summary.area == area].iloc[0]
        row_pu  = per_unit[per_unit.area == area]

        ax_hm = fig.add_subplot(gs[0, ci])

        grid_vals  = np.zeros((2, 2))
        cell_texts = []
        for ri, train in enumerate(TRAIN_ROWS):
            for cj, test in enumerate(TEST_COLS):
                mk, lk, hk = AUC_KEYS[(train, test)]
                mn, lo, hi = row_sum[mk], row_sum[lk], row_sum[hk]
                grid_vals[ri, cj] = mn
                cell_texts.append((ri, cj, mn, lo, hi))

        im = ax_hm.imshow(grid_vals, cmap=cmap, norm=norm,
                          aspect="equal", origin="upper")

        for ri, cj, mn, lo, hi in cell_texts:
            fc  = cmap(norm(mn))
            lum = 0.299*fc[0] + 0.587*fc[1] + 0.114*fc[2]
            tc  = "white" if lum < 0.45 else "black"
            ax_hm.text(cj - 0.38, ri - 0.35,
                       f"{mn:.3f}\n[{lo:.3f},\n {hi:.3f}]",
                       ha="left", va="top", fontsize=7.5,
                       color=tc, fontweight="bold")

        ax_hm.set_xticks([0, 1])
        ax_hm.set_xticklabels(["Test E1", "Test E3"], fontsize=8.5)
        ax_hm.set_yticks([0, 1])
        ax_hm.set_yticklabels(["Train E1", "Train E3"], fontsize=8.5)
        ax_hm.tick_params(left=False, bottom=False)
        ax_hm.set_title(area, fontsize=12, color=AREA_COLORS[area],
                        fontweight="bold", pad=10)
        ax_hm.text(0.97, 0.03, f"n = {int(row_sum.n_units)} units",
                   transform=ax_hm.transAxes, ha="right", va="bottom",
                   fontsize=7.5, color="#666666")
        for spine in ax_hm.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.6)

        if ci == 2:
            cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.09)
            cbar.set_label("Mean AUC", fontsize=8)
            cbar.ax.tick_params(labelsize=7)

        pu_e3 = row_pu["auc_within_Event3"].dropna().values
        jx3   = _jitter(len(pu_e3), center=1.0, width=0.22, seed=ci*7+1)
        jy3   = _jitter(len(pu_e3), center=1.0, width=0.22, seed=ci*7+2)
        ax_hm.scatter(jx3, jy3, c=pu_e3, cmap=cmap, norm=norm,
                      s=13, alpha=0.25, edgecolors="#333333",
                      linewidths=0.2, zorder=5)

        gti_axes.append(fig.add_subplot(gs[1, ci]))

    # GTI bar panels
    for ci, (area, ax_g) in enumerate(zip(AREA_ORDER, gti_axes)):
        row_sum = summary[summary.area == area].iloc[0]
        row_pu  = per_unit[per_unit.area == area]
        acol    = AREA_COLORS[area]

        gti_mn = row_sum.gti_mean
        gti_lo = row_sum.gti_ci95_low
        gti_hi = row_sum.gti_ci95_high

        ax_g.barh(0, gti_mn, height=0.55,
                  color=acol, alpha=0.75, edgecolor=acol, linewidth=1.2, zorder=3)
        ax_g.plot([gti_lo, gti_hi], [0, 0], color=acol, lw=2.5,
                  solid_capstyle="round", zorder=4)
        ax_g.scatter([gti_lo, gti_hi], [0, 0], color=acol, s=35,
                     zorder=5, edgecolors="white", linewidths=0.7)

        gti_pu = row_pu.gti.dropna().values
        jy_pu  = _jitter(len(gti_pu), center=0.0, width=0.30, seed=ci*13)
        ax_g.scatter(gti_pu, jy_pu, color=acol, s=14, alpha=0.45,
                     edgecolors="none", zorder=2)

        ax_g.axvline(0, color="#444444", lw=1.0, linestyle="--", zorder=1)
        ax_g.set_xlim(-0.35, 0.35)
        ax_g.set_yticks([])
        ax_g.set_ylim(-0.65, 0.65)
        ax_g.set_xlabel("GTI", fontsize=9)
        ax_g.spines["left"].set_visible(False)

        ax_g.text(0.02, 0.95,
                  f"GTI = {gti_mn:.3f}\n[{gti_lo:.3f}, {gti_hi:.3f}]",
                  transform=ax_g.transAxes, va="top", ha="left", fontsize=8,
                  color=acol,
                  bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                            edgecolor=acol, lw=0.8, alpha=0.9))

        star = _sig_stars(row_sum.get("gti_p_gt_null", np.nan))
        if star != "n.s.":
            ax_g.text(gti_mn, 0.38, star, ha="center", va="bottom",
                      fontsize=9, color=acol)

    gti_axes[0].set_ylabel("GTI", fontsize=8, labelpad=4)
    gti_axes[1].text(0.005, 0.50,
                     "GTI = 0:\northogonal\nsubspaces",
                     transform=gti_axes[1].transAxes,
                     ha="left", va="center", fontsize=7,
                     color="#666666", fontstyle="italic")

    fig.suptitle(
        "Cross-Event Generalisation  —  2×2 AUC Heatmaps per Area\n"
        "Color = mean AUC (shared scale 0.45–0.75)  ·  dots = individual session-objects"
        " (E3 diagonal cell)\n"
        "Only Train E3 / Test E3 (bottom-right) is hot — the E3 code is geometrically novel",
        fontsize=9,
    )
    _save(fig, out, "plotA_cross_event_generalization")
    print("plotA: row order flipped (E1 top, E3 bottom), colorbar 0.45–0.75, "
          "dots alpha 0.25, inline GTI annotation in F5 panel.")


# PLOT B - Demixed decision-axis trajectories
# Source: dt ANOVA marginalization from dpca_component_trajectories_long.csv
# Decision axis = PC1 from SVD of the (Go - NoGo) matrix.

def plotB_trajectory(res: Path, out: Path) -> None:
    """
    2 rows (Event1 top, Event3 bottom) × 3 cols (AIP, F5, F6).
    Each panel: Go (area-color, solid) and No-Go (grey, dashed) projected
    onto the first left-singular vector of (Go − NoGo) in the dt ANOVA
    marginalization space.  Y-axes shared within each area column.
    Caption: single representative session per area.
    """
    df = pd.read_csv(res / "final_analysis/09_dpca"
                         "/dpca_component_trajectories_long.csv")

    # Representative sessions: highest-neuron per area that is in the dpca output
    REPS = {
        "AIP": ("MonkeyA_Session02", 55),
        "F5":  ("MonkeyC_Session06",  9),
        "F6":  ("MonkeyB_Session01", 41),
    }
    # Bin counts and epoch start times (20 ms bins, from session results convention)
    N_BINS   = {"Event1": 50, "Event3": 70}
    T_START  = {"Event1": -0.20, "Event3": -0.80}
    BIN_W    = 0.02

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=False)

    # First pass: compute all projections and collect y-ranges per area
    traces   = {}   # (area, event) → (t, go, nogo, var_exp, n_neurons)
    y_ranges = {a: [np.inf, -np.inf] for a in AREA_ORDER}

    for area in AREA_ORDER:
        sess, n_neurons_spec = REPS[area]
        for event in ("Event1", "Event3"):
            n_bins  = N_BINS[event]
            t_start = T_START[event]
            time_s  = np.arange(n_bins) * BIN_W + t_start

            sub = df[
                (df.area == area) & (df.session == sess) &
                (df.event == event) & (df.marginalization == "dt") &
                (df.source == "anova")
            ]
            if sub.empty:
                traces[(area, event)] = (time_s,
                                         np.zeros(n_bins), np.zeros(n_bins),
                                         0.0, n_neurons_spec)
                continue

            n_neu = int(sub.component.max())
            go_mat   = np.zeros((n_neu, n_bins))
            nogo_mat = np.zeros((n_neu, n_bins))
            for cond, mat in (("Go", go_mat), ("NoGo", nogo_mat)):
                sc = sub[sub.condition == cond].sort_values(
                    ["component", "time_index"])
                for ni in range(1, n_neu + 1):
                    row = sc[sc.component == ni]["score"].values
                    if len(row) == n_bins:
                        mat[ni - 1] = row

            diff_mat = go_mat - nogo_mat
            U, S, _  = np.linalg.svd(diff_mat, full_matrices=False)
            pc1_vec  = U[:, 0]
            pc1_go   = pc1_vec @ go_mat
            pc1_nogo = pc1_vec @ nogo_mat

            # Sign convention: Go > NoGo on average post-onset
            t0_bin = int(-t_start / BIN_W)
            if (pc1_go[t0_bin:] - pc1_nogo[t0_bin:]).mean() < 0:
                pc1_go, pc1_nogo = -pc1_go, -pc1_nogo

            var_exp = float(S[0] ** 2 / np.maximum((S ** 2).sum(), 1e-12) * 100)
            traces[(area, event)] = (time_s, pc1_go, pc1_nogo, var_exp, n_neu)

            for v in (pc1_go, pc1_nogo):
                y_ranges[area][0] = min(y_ranges[area][0], v.min())
                y_ranges[area][1] = max(y_ranges[area][1], v.max())

    # Second pass: plot
    for col_i, area in enumerate(AREA_ORDER):
        sess, n_neurons_spec = REPS[area]
        acol = AREA_COLORS[area]

        y_lo, y_hi = y_ranges[area]
        y_pad = max((y_hi - y_lo) * 0.18, 0.5)

        for row_i, event in enumerate(("Event1", "Event3")):
            ax = axes[row_i, col_i]
            t, go, nogo, var_exp, n_neu = traces[(area, event)]

            ax.plot(t, go,   color=acol,    lw=2.0, solid_capstyle="round",
                    label="Go")
            ax.plot(t, nogo, color=E1_COLOR, lw=2.0, linestyle="--",
                    solid_capstyle="round", label="No-Go")
            ax.axhline(0, color=NULL_COLOR, lw=0.7, linestyle=":", zorder=1)
            ax.axvline(0, color="black",    lw=0.9, linestyle="--", zorder=1)

            ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

            if col_i == 0:
                ev_tag = "Event 3 (movement)" if event == "Event3" \
                         else "Event 1 (baseline)"
                ax.set_ylabel(f"{ev_tag}\nDecision PC1 (a.u.)", fontsize=9)
            if row_i == 1:
                ax.set_xlabel("Time relative to event onset (s)", fontsize=9)
            if row_i == 0:
                ax.set_title(area, fontsize=12, color=acol, fontweight="bold")

            ax.text(0.97, 0.04,
                    f"n = {n_neurons_spec} neurons\n(single representative session)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7, color="#777777", fontstyle="italic")
            ax.text(0.97, 0.97,
                    f"PC1 = {var_exp:.0f}% var",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color="#555555")

    # Shared legend above the figure
    handles = [
        mlines.Line2D([], [], color=AREA_COLORS["F5"], lw=2.0, label="Go"),
        mlines.Line2D([], [], color=E1_COLOR, lw=2.0, linestyle="--",
                      label="No-Go"),
        mlines.Line2D([], [], color="black", lw=0.9, linestyle="--",
                      label="Event onset (t = 0)"),
        mlines.Line2D([], [], color=NULL_COLOR, lw=0.7, linestyle=":",
                      label="Zero line"),
    ]
    fig.legend(handles=handles, fontsize=8.5, framealpha=0.9,
               ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.01))

    fig.suptitle(
        "Demixed Decision-Axis Trajectories (dt ANOVA component, PC1 of Go − No-Go)\n"
        "Go and No-Go overlap at Event 1 (baseline) and diverge after Event 3 onset\n"
        "Single representative session per area  ·  Y-axes matched within each column",
        fontsize=9.5, y=1.07,
    )
    fig.tight_layout()
    _save(fig, out, "plotB_decision_trajectory")
    print("plotB: real demixed-decision trajectories from dt ANOVA PC1 "
          "(Go vs No-Go, shared y-axis within area column).")


# PLOT C - Time-resolved AUC with shuffle-null envelope + FDR bars

def plotC(res: Path, out: Path) -> None:
    FDR_Q    = 0.05
    N_CONSEC = 5   # FIX C-1: was 3, now 5 for a more conservative onset estimate

    tr   = pd.read_csv(res / "time_resolved_results.csv")
    null = pd.read_csv(res / "final_analysis/06_shuffle_null"
                          "/shuffle_null_time_resolved.csv")

    merged = pd.merge(
        tr  [["area", "session", "object", "event", "center_time_s", "auc"]],
        null[["area", "session", "object", "event", "center_time_s",
              "null_auc_mean", "null_auc_ci95_low", "null_auc_ci95_high"]],
        on=["area", "session", "object", "event", "center_time_s"], how="inner",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)

    onset_report = {}

    for ax, area in zip(axes, AREA_ORDER):
        acol = AREA_COLORS[area]

        for event in ("Event1", "Event3"):
            sub = tr[(tr.area == area) & (tr.event == event)]
            grp = sub.groupby("center_time_s")["auc"]
            t   = np.array(sorted(grp.groups.keys()))
            mn  = grp.mean().loc[t].values
            se  = grp.sem().loc[t].values
            col = EVENT_COLORS[event]
            ax.fill_between(t, mn - se, mn + se, color=col, alpha=0.18, zorder=3)
            ax.plot(t, mn, color=col, lw=2.0, zorder=4, label=EVENT_LABELS[event])

        m3       = merged[(merged.area == area) & (merged.event == "Event3")]
        ng       = m3.groupby("center_time_s")
        t_null   = np.array(sorted(ng.groups.keys()))
        null_mn  = ng["null_auc_mean"].mean().loc[t_null].values
        null_sem = ng["null_auc_mean"].sem().loc[t_null].values
        null_hi  = null_mn + 1.96 * null_sem

        ax.fill_between(t_null, null_mn - 1.96*null_sem, null_hi,
                        color=NULL_COLOR, alpha=0.60, zorder=2,
                        label="Null 95 % CI (E3)")
        ax.plot(t_null, null_mn, color="#BBBBBB", lw=0.8, linestyle=":", zorder=2)

        obs_mn = np.array([ng.get_group(ti)["auc"].mean() for ti in t_null])
        pvals  = np.array([
            _wilcoxon_greater(
                ng.get_group(ti)["auc"].values,
                ng.get_group(ti)["null_auc_mean"].values,
            ) for ti in t_null
        ])
        qvals = _bh_qvalues(pvals)
        sig   = (qvals < FDR_Q) & (obs_mn > null_hi)

        y_bar = 0.915
        dt    = float(np.diff(t_null).mean()) if len(t_null) > 1 else 0.02
        for ti, s in zip(t_null, sig):
            if s:
                ax.plot([ti - dt*0.45, ti + dt*0.45], [y_bar, y_bar],
                        color=acol, lw=5.5, solid_capstyle="butt",
                        zorder=6, clip_on=False)

        post_mask = t_null >= 0.0
        peak_auc  = np.nanmax(obs_mn[post_mask]) if post_mask.any() else np.nan
        onset_s   = _find_onset(t_null, sig, N_CONSEC)
        onset_report[area] = onset_s

        peak_str  = f"Peak AUC = {peak_auc:.3f}"
        onset_str = (f"Onset = {onset_s*1000:.0f} ms"
                     if np.isfinite(onset_s) else "Onset = n.s.")
        crit_str  = f"Criterion: ≥{N_CONSEC} bins"   # FIX C-3: document criterion

        ax.text(0.03, 0.97, peak_str + "\n" + onset_str + "\n" + crit_str,
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                          edgecolor=acol, lw=1.2, alpha=0.92), zorder=7)

        ax.axvline(0, color="black", lw=0.9, linestyle="--", zorder=1)
        _add_chance(ax)
        ax.set_title(area, fontsize=12, color=acol, fontweight="bold")
        ax.set_xlabel("Time relative to event onset (s)", fontsize=9)
        ax.spines["left"].set_visible(True)

    axes[0].set_ylabel("Mean AUC  (Go vs No-Go)", fontsize=9)
    axes[0].set_ylim(0.28, 0.95)

    handles = [
        mpatches.Patch(facecolor=E1_COLOR, alpha=0.60,
                       label=EVENT_LABELS["Event1"]),
        mpatches.Patch(facecolor=E3_COLOR, alpha=0.60,
                       label=EVENT_LABELS["Event3"]),
        mpatches.Patch(facecolor=NULL_COLOR, alpha=0.70,
                       label="Shuffle null 95 % CI (E3)"),
        mlines.Line2D([], [], color=AREA_COLORS["F5"], lw=5,
                      label=f"BH-FDR sig. bins (E3, q < {FDR_Q})"),
        mlines.Line2D([], [], color="black", linestyle="--", lw=0.9,
                      label="Event onset (t = 0)"),
        mlines.Line2D([], [], color="#777777", linestyle="--", lw=0.9,
                      label="Chance (0.5)"),
    ]
    fig.legend(handles=handles, fontsize=8, framealpha=0.92,
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.10))

    fig.suptitle(
        "Time-Resolved Go/No-Go Decoding  ·  Shuffle-Null Envelope  ·  "
        "BH-FDR Significance Bars\n"
        "Colored bars = contiguous bins E3 > null (q < 0.05 BH);  "
        "insets: peak AUC, onset latency (≥5 consecutive significant bins)",
        fontsize=9.5,
    )
    fig.tight_layout()
    # Print onset latencies as required
    print("plotC: onset latencies (≥5 consec bins criterion):", onset_report)
    _save(fig, out, "plotC_null_overlay_significance")
    print("plotC: N_CONSEC→5, legend moved to figure bottom, criterion documented in insets.")


# PLOT D - Paired pre/post decoding

def plotD(res: Path, out: Path) -> None:
    df      = pd.read_csv(res / "session_object_event_results.csv")
    sp      = pd.read_csv(res / "final_analysis/06_shuffle_null"
                              "/shuffle_null_prepost_per_unit.csv")
    me_coef = pd.read_csv(res / "final_analysis/08_mixed_effects"
                              "/mixed_effects_coefficients.csv")

    e3  = df[df.event == "Event3"].copy()
    sp3 = sp[sp.event == "Event3"].copy()

    me_row = me_coef[
        (me_coef.model == "delta_auc_full") &
        (me_coef.term.str.contains("Event3")) &
        (~me_coef.term.str.contains(":"))
    ]
    if not me_row.empty:
        me_c  = float(me_row.iloc[0]["coef"])
        me_lo = float(me_row.iloc[0]["ci_low"])
        me_hi = float(me_row.iloc[0]["ci_high"])
        me_p  = float(me_row.iloc[0]["p_value"])
        me_txt = (f"Mixed-effects model (random intercept per session):  "
                  f"ΔEvent3 coef = {me_c:.3f} [{me_lo:.3f}, {me_hi:.3f}],  "
                  f"p = {me_p:.4f}")
    else:
        me_txt = ""

    fig = plt.figure(figsize=(14, 5.5))
    gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.7], wspace=0.38)

    delta_summary = []

    for ci, area in enumerate(AREA_ORDER):
        ax   = fig.add_subplot(gs[ci])
        acol = AREA_COLORS[area]
        sub  = e3[e3.area == area][["dec_obs_go_nogo_auc_pre",
                                     "dec_obs_go_nogo_auc_post"]].dropna()
        pre  = sub["dec_obs_go_nogo_auc_pre"].values
        post = sub["dec_obs_go_nogo_auc_post"].values
        n    = len(pre)
        delta = post - pre

        dz = float(delta.mean() / delta.std(ddof=1)) \
             if delta.std(ddof=1) > 0 else np.nan

        sp_a    = sp3[sp3.area == area]
        null_lo = sp_a["null_delta_ci95_low"].mean()
        null_hi = sp_a["null_delta_ci95_high"].mean()

        jx_pre  = _jitter(n, center=0.0, width=0.10, seed=hash(area) % 99)
        jx_post = _jitter(n, center=1.0, width=0.10, seed=hash(area) % 99 + 1)

        for xp, xq, vp, vq in zip(jx_pre, jx_post, pre, post):
            going_up = vq > vp
            lc = acol if going_up else NULL_COLOR
            ax.plot([xp, xq], [vp, vq],
                    color=lc, alpha=0.25 if going_up else 0.18, lw=0.9, zorder=2)

        ax.scatter(jx_pre,  pre,  color=acol, s=26, alpha=0.70,
                   edgecolors="white", linewidths=0.4, zorder=4)
        ax.scatter(jx_post, post, color=acol, s=26, alpha=0.70,
                   edgecolors="white", linewidths=0.4, zorder=4)

        rng    = np.random.default_rng(seed=ci*71)
        bmeans = np.array([rng.choice(delta, size=n, replace=True).mean()
                           for _ in range(5000)])
        delta_mn = delta.mean()
        bci_lo   = float(np.percentile(bmeans, 2.5))
        bci_hi   = float(np.percentile(bmeans, 97.5))

        for xi, vals in zip([0, 1], [pre, post]):
            mn   = vals.mean()
            bs_mn = np.array([rng.choice(vals, size=n, replace=True).mean()
                               for _ in range(5000)])
            blo = np.percentile(bs_mn, 2.5)
            bhi = np.percentile(bs_mn, 97.5)
            ax.errorbar(xi, mn, yerr=[[mn-blo], [bhi-mn]],
                        fmt="o", color="black", markersize=6.5,
                        capsize=4, lw=2.0, zorder=5)

        _, p_wx = scipy_wilcoxon(post, pre, alternative="greater")
        top = max(post.max(), pre.max()) + 0.07
        ax.plot([0, 0, 1, 1], [top - 0.02, top, top, top - 0.02],
                lw=1.0, color="#333333")
        ax.text(0.5, top + 0.01, _sig_stars(p_wx),
                ha="center", va="bottom", fontsize=9, color="#333333")

        _add_chance(ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pre\n(t < 0 s)", "Post\n(t ≥ 0 s)"], fontsize=9)
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(area, fontsize=12, color=acol, fontweight="bold")
        ax.set_ylabel("AUC  (Go vs No-Go)" if ci == 0 else "", fontsize=9)
        ax.set_ylim(0.05, 1.10)

        ax.text(0.97, 0.05,
                f"n = {n} units\nd_z = {dz:.2f}\nWilcoxon p = {p_wx:.3f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor=acol, lw=1.1, alpha=0.92))

        delta_summary.append((area, delta_mn, bci_lo, bci_hi,
                               null_lo, null_hi, n, dz, p_wx))

    # right panel: delta-AUC per area
    ax_d = fig.add_subplot(gs[3])

    ax_d.axvline(0, color="#666666", lw=1.0, linestyle="--", zorder=1,
                 label="Δ = 0")

    null_band_drawn = False
    for ri, (area, dmn, dlo, dhi, nlo, nhi, n, dz, _) in enumerate(delta_summary):
        acol = AREA_COLORS[area]
        y = ri

        ax_d.fill_betweenx([y - 0.35, y + 0.35], nlo, nhi,
                            color=NULL_COLOR, alpha=0.70, zorder=1,
                            label="Shuffle null Δ 95 % CI" if not null_band_drawn else "_")
        null_band_drawn = True

        ax_d.plot([dlo, dhi], [y, y], color=acol, lw=3.0,
                  solid_capstyle="round", zorder=3)
        ax_d.scatter([dmn], [y], color=acol, s=70, zorder=4,
                     edgecolors="white", linewidths=0.8)

        ax_d.text(-0.02, y, area, ha="right", va="center",
                  fontsize=9, color=acol, fontweight="bold",
                  transform=ax_d.get_yaxis_transform())
        ax_d.text(dhi + 0.01, y, f"d_z={dz:.2f}", ha="left", va="center",
                  fontsize=7.5, color=acol)

    # Small annotation labeling the grey band
    ax_d.text(0.03, 0.03,
              "grey band =\nshuffle null\nΔ 95 % CI",
              transform=ax_d.transAxes, ha="left", va="bottom",
              fontsize=7, color="#666666", fontstyle="italic")

    ax_d.set_yticks([])
    ax_d.set_ylim(-0.65, 2.65)
    ax_d.set_xlabel("ΔAUC  (Post − Pre)", fontsize=8)
    ax_d.spines["left"].set_visible(False)
    ax_d.set_title("Effect size\nby area", fontsize=9, pad=6)

    if me_txt:
        fig.text(0.5, -0.05, me_txt,
                 ha="center", va="bottom", fontsize=8,
                 color="#333333",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8F8F8",
                           edgecolor="#CCCCCC", lw=1.0))

    fig.suptitle(
        "Paired Pre- vs Post-Onset Decoding  —  Event 3  (Upgraded)\n"
        "Area-color lines = post > pre;  grey lines = decrease  ·  "
        "mean with bootstrap 95 % CI  ·  Cohen's d_z\n"
        "Pre and post are independently fitted decoders using the same trials "
        "and neurons, differing only in which time bins enter the feature matrix",
        fontsize=9,
    )
    _save(fig, out, "plotD_prepost_upgraded")
    print("plotD: null band labeled 'shuffle null Δ 95 % CI', Δ=0 dashed line added.")


# PLOT E - Reliability: prevalence dots + sessionxobject heatmap

def plotE(res: Path, out: Path) -> None:
    df = pd.read_csv(res / "session_object_event_results.csv")
    df["perm_sig"] = df["perm_obs_go_nogo_auc_p"] < 0.05
    auc_col = "dec_obs_go_nogo_auc_full"

    fig = plt.figure(figsize=(13, 6.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1.1, 1.8], wspace=0.40)

    # Panel A: prevalence
    ax_prev = fig.add_subplot(gs[0])

    x_pos = 0.0
    area_centres = {}
    xticks, xlabels = [], []

    for ai, area in enumerate(AREA_ORDER):
        acol     = AREA_COLORS[area]
        area_x   = []
        for ei, event in enumerate(("Event1", "Event3")):
            sub  = df[(df.area == area) & (df.event == event)]
            k    = int(sub["perm_sig"].sum())
            n    = len(sub)
            prop, lo, hi = _wilson_ci(k, n)

            col = EVENT_COLORS[event]
            ax_prev.errorbar(x_pos, prop, yerr=[[prop-lo],[hi-prop]],
                             fmt="o", color=acol,
                             mec="white" if event == "Event3" else acol,
                             mfc=col, mew=1.5,
                             markersize=10, capsize=4, lw=2.2, zorder=4)
            ax_prev.text(x_pos, hi + 0.02, f"{k}/{n}",
                         ha="center", va="bottom", fontsize=8, color="#333333")
            area_x.append(x_pos)
            xticks.append(x_pos); xlabels.append("E1" if event == "Event1" else "E3")
            x_pos += 0.55

        area_centres[area] = np.mean(area_x)
        x_pos += 0.45

    for area, cx in area_centres.items():
        ax_prev.text(cx, -0.09, area, ha="center", va="top",
                     fontsize=11, color=AREA_COLORS[area], fontweight="bold",
                     clip_on=False)

    ax_prev.axhline(0.05, color="#888888", lw=0.9, linestyle=":",
                    label="5 % FPR floor")

    ax_prev.set_ylim(-0.02, 0.90)   # was 1.05, now 0.90 with room above max
    ax_prev.set_xticks(xticks)
    ax_prev.set_xticklabels(xlabels, fontsize=8.5)
    ax_prev.set_ylabel("Proportion reliably above chance\n(permutation p < 0.05)",
                        fontsize=9)
    ax_prev.set_xlim(-0.35, x_pos - 0.45 + 0.35)
    ax_prev.spines["bottom"].set_visible(True)

    leg = [
        mlines.Line2D([], [], color="#555555", linestyle=":", lw=0.9,
                      label="5 % FPR floor"),
        mlines.Line2D([], [], marker="o", color="w",
                      mfc=E1_COLOR, mec=AIP_COLOR, mew=1.5, markersize=9,
                      lw=0, label="Event 1"),
        mlines.Line2D([], [], marker="o", color="w",
                      mfc=E3_COLOR, mec=AIP_COLOR, mew=1.5, markersize=9,
                      lw=0, label="Event 3"),
    ]
    ax_prev.legend(handles=leg, fontsize=8, framealpha=0.88, loc="upper left")
    ax_prev.set_title("A  —  Prevalence (proportion sig.)", fontsize=10, pad=8)

    ax_prev.text(0.02, 0.02,
                 "AIP: 0/12 units sig. at E1\n(cleanest possible baseline)",
                 transform=ax_prev.transAxes, ha="left", va="bottom",
                 fontsize=7.5, color=AIP_COLOR, fontstyle="italic",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                           edgecolor=AIP_COLOR, lw=0.8, alpha=0.88))

    # Panel B: sessionxobject heatmap
    ax_hm = fig.add_subplot(gs[1])

    e3 = df[df.event == "Event3"].copy()
    e1 = df[df.event == "Event1"].copy()

    rows_all = []
    area_boundaries = []
    y_offset = 0
    for area in AREA_ORDER:
        block = e3[e3.area == area].sort_values(auc_col, ascending=False).copy()
        block["_area"] = area
        area_boundaries.append((y_offset, y_offset + len(block), area))
        rows_all.append(block[["session", "object", "_area", auc_col, "perm_sig"]])
        y_offset += len(block)

    rows_df = pd.concat(rows_all, ignore_index=True)
    n_rows  = len(rows_df)

    e1_lookup    = e1.set_index(["session", "object"])[auc_col].to_dict()
    e1_sig_lookup = e1.set_index(["session","object"])["perm_sig"].to_dict()

    e1_auc = np.array([e1_lookup.get((r.session, r.object), np.nan)
                        for _, r in rows_df.iterrows()])
    e3_auc = rows_df[auc_col].values
    sig_e3 = rows_df["perm_sig"].values
    sig_e1 = np.array([e1_sig_lookup.get((r.session, r.object), False)
                        for _, r in rows_df.iterrows()])

    hm_data  = np.column_stack([e1_auc, e3_auc])
    cmap_hm  = mpl.colormaps["RdBu_r"]
    norm_hm  = TwoSlopeNorm(vmin=0.40, vcenter=0.50, vmax=0.90)

    im = ax_hm.imshow(hm_data, cmap=cmap_hm, norm=norm_hm,
                      aspect="auto", origin="upper",
                      extent=[-0.5, 1.5, n_rows - 0.5, -0.5])

    for row_i, (s1, s3) in enumerate(zip(sig_e1, sig_e3)):
        if s1: ax_hm.scatter(0, row_i, color="black", s=10, zorder=5,
                              marker="o", linewidths=0)
        if s3: ax_hm.scatter(1, row_i, color="black", s=10, zorder=5,
                              marker="o", linewidths=0)

    for (y_start, y_end, area) in area_boundaries:
        acol = AREA_COLORS[area]
        if y_start > 0:
            ax_hm.axhline(y_start - 0.5, color="white", lw=2.0, zorder=6)
        ax_hm.fill_betweenx([y_start - 0.5, y_end - 0.5],
                             [-0.5, -0.5], [-0.3, -0.3],
                             color=acol, alpha=0.85, zorder=7, clip_on=False)
        ax_hm.text(-0.6, (y_start + y_end) / 2, area,
                   ha="right", va="center", fontsize=9,
                   color=acol, fontweight="bold")

    ax_hm.set_xticks([0, 1])
    ax_hm.set_xticklabels(["Event 1", "Event 3"], fontsize=9.5)
    ax_hm.set_yticks([])
    ax_hm.set_xlim(-0.7, 1.8)
    ax_hm.set_ylim(n_rows - 0.5, -0.5)

    cbar = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.03)
    cbar.set_label("AUC (Go vs No-Go)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    ax_hm.scatter([], [], color="black", s=14, marker="o",
                  label="Permutation p < 0.05")
    ax_hm.legend(fontsize=8, framealpha=0.88, loc="lower right")
    ax_hm.set_title("B  —  Session×Object AUC Heatmap\n"
                    "(sorted by E3 AUC ↓ within area;  ● = perm. sig.)",
                    fontsize=9.5, pad=8)

    fig.suptitle(
        "Reliability of Go/No-Go Observation Coding  —  Prevalence + Individual-Unit Heatmap\n"
        "Left: proportion sig. (Wilson 95 % CI; raw counts shown)  ·  "
        "Right: AUC heatmap (0.40–0.90 scale)",
        fontsize=9.5,
    )
    _save(fig, out, "plotE_reliability")
    print("plotE: colorbar tightened 0.40–0.90, AIP E1 caption note added, "
          "y-axis set to [−0.02, 0.90].")


# PLOT F - Divergence onset CDF

def plotF(res: Path, out: Path) -> None:
    div = pd.read_csv(res / "final_analysis/07_divergence_onset"
                          "/divergence_onset_per_unit.csv")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ls_map    = {"Event1": "--", "Event3": "-"}
    alpha_map = {"Event1": 0.55, "Event3": 0.90}

    # Track median positions for non-overlapping annotations
    median_info = {}

    for area in AREA_ORDER:
        acol = AREA_COLORS[area]
        for event in ("Event1", "Event3"):
            sub      = div[(div.area == area) & (div.event == event)]
            n_total  = len(sub)
            onsets   = sub["onset_ms"].dropna().sort_values().values

            if len(onsets) == 0:
                ax.plot([0, 700], [0, 0], color=acol,
                        ls=ls_map[event], lw=1.4, alpha=alpha_map[event])
                continue

            frac_max = len(onsets) / n_total
            t_cdf    = np.concatenate([[0], np.repeat(onsets, 2), [750]])
            y_cdf    = np.concatenate([[0],
                                       np.repeat(np.arange(1, len(onsets)+1) / n_total, 2),
                                       [frac_max]])

            ax.plot(t_cdf, y_cdf, color=acol,
                    ls=ls_map[event],
                    lw=2.2 if event == "Event3" else 1.4,
                    alpha=alpha_map[event],
                    label=f"{area} {event.replace('Event', 'E')}")

            if event == "Event3" and len(onsets) >= 1:
                med = float(np.median(onsets))
                ax.axvline(med, color=acol, lw=0.8, linestyle=":",
                           alpha=0.70, zorder=2)
                median_info[area] = (med, frac_max)

    # Assign y-fractions: AIP higher, F6 lower (they are the close pair)
    y_offsets = {"AIP": 0.72, "F5": 0.55, "F6": 0.30}
    for area in AREA_ORDER:
        if area not in median_info:
            continue
        med, frac_max = median_info[area]
        acol   = AREA_COLORS[area]
        y_box  = y_offsets[area]   # annotation box y in axes coordinates

        # Leader line: from median-vertical/frac_max point to the box
        ax.annotate(
            f"{area}\nmedian\n{med:.0f} ms",
            xy=(med, frac_max),
            xytext=(med + 18, y_box),
            textcoords=("data", "axes fraction"),
            fontsize=7.5, color=acol, va="center",
            arrowprops=dict(arrowstyle="-", color=acol,
                            lw=0.7, relpos=(0, 0.5)),
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor=acol, lw=0.8, alpha=0.88),
        )

    ax.set_xlabel("Time after event onset (ms)", fontsize=10)
    ax.set_ylabel("Cumulative fraction of units with detected onset", fontsize=9)
    ax.set_xlim(-20, 620)
    ax.set_ylim(-0.02, 0.72)
    ax.axhline(0, color=NULL_COLOR, lw=0.5)

    leg_handles = []
    for area in AREA_ORDER:
        leg_handles.append(mlines.Line2D([], [], color=AREA_COLORS[area],
                                          lw=2.2, ls="-",
                                          label=f"{area}  Event3"))
    for area in AREA_ORDER:
        leg_handles.append(mlines.Line2D([], [], color=AREA_COLORS[area],
                                          lw=1.4, ls="--", alpha=0.60,
                                          label=f"{area}  Event1"))
    ax.legend(handles=leg_handles, fontsize=8, framealpha=0.88,
              loc="upper left", ncol=2)

    ax.set_title(
        "Divergence Onset CDF  —  Fraction of Session-Objects Reaching Significance\n"
        "(Solid = Event3, dashed = Event1;  CDF capped at detected fraction;\n"
        " vertical dotted = median onset;  N.B. ~25 % detection rate — see Plot C for population onset)",
        fontsize=9,
    )

    fig.tight_layout()
    _save(fig, out, "plotF_divergence_cdf")
    print("plotF: AIP/F6 annotation boxes separated vertically with leader lines.")


# Entry point

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate six great plots (PNG + SVG) into go-nogo/great_plots/.")
    ap.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent.parent / "results"),
    )
    ap.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "generated"),
    )
    args = ap.parse_args()

    res = Path(args.results_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Results dir : {res}")
    print(f"Output dir  : {out}\n")

    plotA(res, out)
    plotB_trajectory(res, out)
    plotC(res, out)
    plotD(res, out)
    plotE(res, out)
    plotF(res, out)

    print(f"\nAll plots (PNG + SVG) saved to: {out}")


if __name__ == "__main__":
    main()
