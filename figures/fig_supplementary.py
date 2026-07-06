"""
16_extra_figures.py
===================
Extra supplementary figures — do NOT overwrite existing figures.

  fig2_extra1_null_overlay.png   --  Fig 2 + shuffle-null shading +
                                     BH-FDR significance bars + peak/onset insets
  figS1_per_object_time.png      --  Time-resolved decoding split by object

Run from the go-nogo root directory:
  python reworked_pipeline/16_extra_figures.py
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
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon as scipy_wilcoxon

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Global style  (mirror 15_thesis_figures.py)
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

AREA_COLORS  = {"AIP": "#4C72B0", "F5": "#DD8452", "F6": "#55A868"}
EVENT_COLORS = {"Event1": "#BBBBBB", "Event3": "#E05A2B"}
EVENT_LABELS = {"Event1": "Event 1 (baseline)", "Event3": "Event 3 (movement)"}
AREA_ORDER   = ["AIP", "F5", "F6"]
CHANCE       = 0.5

OBJECT_COLORS = {
    "Object1": {"color": "#1f77b4", "ls": "-"},
    "Object2": {"color": "#e15759", "ls": "--"},
    "Object3": {"color": "#59a14f", "ls": ":"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction. Returns q-values."""
    pvals = np.asarray(pvals, dtype=float)
    finite = np.isfinite(pvals)
    q = np.full_like(pvals, np.nan)
    if not finite.any():
        return q
    pv = pvals[finite]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    inv = np.empty(n)
    inv[order] = q_ranked
    q[finite] = inv
    return q


def _wilcoxon_greater_pval(a: np.ndarray, b: np.ndarray) -> float:
    """One-sided Wilcoxon signed-rank test  H1: a > b.  Returns p-value."""
    diffs = np.asarray(a, float) - np.asarray(b, float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 3:
        return np.nan
    try:
        _, p = scipy_wilcoxon(diffs, alternative="greater")
        return float(p)
    except Exception:
        return np.nan


def _find_onset(t: np.ndarray, sig: np.ndarray, n_consec: int = 3) -> float:
    """Return first time where sig is True for n_consec consecutive bins, else NaN."""
    count = 0
    for i, s in enumerate(sig):
        if s:
            count += 1
            if count >= n_consec:
                return float(t[i - n_consec + 1])
        else:
            count = 0
    return np.nan


def _add_chance(ax, **kw):
    kw.setdefault("color", "gray")
    kw.setdefault("linewidth", 1.0)
    kw.setdefault("linestyle", "--")
    kw.setdefault("zorder", 1)
    ax.axhline(CHANCE, **kw)


# ---------------------------------------------------------------------------
# Figure 1 — Null overlay + significance bar + insets
# ---------------------------------------------------------------------------

def fig_null_overlay(tr: pd.DataFrame, null: pd.DataFrame, out: Path) -> None:
    """
    Three-panel layout (one per area) reproducing fig2 with three additions:

      1. Light grey shaded band = mean null AUC ± 1.96 × SEM across
         session-objects.  Represents the 95 % CI of the population-mean
         AUC under the shuffle null for Event 3.

      2. Coloured horizontal bar above x-axis marks contiguous time bins
         where the Event-3 mean AUC exceeds the upper null bound AND
         passes BH-FDR correction (q < 0.05) across all time bins.
         Significance is assessed with a per-bin Wilcoxon signed-rank test
         (obs_auc vs null_auc_mean, paired by session-object).

      3. Text inset per panel: peak AUC (post-onset maximum) and
         onset latency (first bin in the longest BH-significant run ≥ 3
         consecutive bins).
    """
    FDR_Q    = 0.05
    N_CONSEC = 3

    # Merge observed and null on the keys that identify a unique (unit, time-bin)
    merged = pd.merge(
        tr[["area", "session", "object", "event", "center_time_s", "auc"]],
        null[["area", "session", "object", "event", "center_time_s",
              "null_auc_mean", "null_auc_ci95_low", "null_auc_ci95_high"]],
        on=["area", "session", "object", "event", "center_time_s"],
        how="inner",
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)

    for ax, area in zip(axes, AREA_ORDER):

        # ── observed traces (both events) ──────────────────────────────────
        for event in ["Event1", "Event3"]:
            sub = tr[(tr["area"] == area) & (tr["event"] == event)]
            grp = sub.groupby("center_time_s")["auc"]
            t   = np.array(sorted(grp.groups.keys()))
            mn  = grp.mean().loc[t].values
            se  = grp.sem().loc[t].values
            col = EVENT_COLORS[event]
            ax.fill_between(t, mn - se, mn + se,
                            color=col, alpha=0.18, zorder=3)
            ax.plot(t, mn, color=col, lw=2.0, zorder=4,
                    label=EVENT_LABELS[event])

        # ── null envelope for Event 3 ───────────────────────────────────────
        m3 = merged[(merged["area"] == area) & (merged["event"] == "Event3")]
        null_grp = m3.groupby("center_time_s")

        t_null    = np.array(sorted(null_grp.groups.keys()))
        null_mn   = null_grp["null_auc_mean"].mean().loc[t_null].values
        null_sem  = null_grp["null_auc_mean"].sem().loc[t_null].values

        null_lo = null_mn - 1.96 * null_sem
        null_hi = null_mn + 1.96 * null_sem

        ax.fill_between(t_null, null_lo, null_hi,
                        color="#CCCCCC", alpha=0.55, zorder=2,
                        label="Shuffle null 95 % CI (E3)")
        ax.plot(t_null, null_mn,
                color="#999999", lw=0.9, linestyle=":", zorder=2)

        # ── per-bin Wilcoxon test: obs_auc > null_auc_mean ──────────────────
        pvals = np.array([
            _wilcoxon_greater_pval(
                null_grp.get_group(t_i)["auc"].values,
                null_grp.get_group(t_i)["null_auc_mean"].values,
            )
            for t_i in t_null
        ])
        qvals = _bh_qvalues(pvals)

        # Observed mean AUC at null time points
        obs_grp = m3.groupby("center_time_s")["auc"].mean()
        obs_mn  = np.array([obs_grp.get(t_i, np.nan) for t_i in t_null])

        # Significant = q < FDR_Q  AND  obs mean > null upper bound
        sig = (qvals < FDR_Q) & (obs_mn > null_hi)

        # Draw significance bar (horizontal coloured ticks above plot)
        y_bar = axes[0].get_ylim()[1] if hasattr(axes[0], '_sig_ybar') else 0.895
        # Use a fixed y position slightly above the top of the data range
        y_bar = 0.895
        if sig.any():
            dt = np.diff(t_null).mean() if len(t_null) > 1 else 0.02
            for t_i, s in zip(t_null, sig):
                if s:
                    ax.plot([t_i - dt * 0.45, t_i + dt * 0.45],
                            [y_bar, y_bar],
                            color=AREA_COLORS[area], lw=5,
                            solid_capstyle="butt", zorder=6, clip_on=False)

        # ── inset: peak AUC and onset latency ─────────────────────────────
        post_mask = t_null >= 0.0
        peak_auc  = np.nanmax(obs_mn[post_mask]) if post_mask.any() else np.nan
        onset_s   = _find_onset(t_null, sig, N_CONSEC)

        peak_str  = f"Peak AUC = {peak_auc:.3f}" if np.isfinite(peak_auc) else "Peak AUC = n/a"
        onset_str = (f"Onset = {onset_s * 1000:.0f} ms"
                     if np.isfinite(onset_s) else "Onset = n.s.")

        ax.text(
            0.03, 0.97, peak_str + "\n" + onset_str,
            transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=AREA_COLORS[area], linewidth=1.2, alpha=0.90),
            zorder=7,
        )

        ax.axvline(0, color="black", lw=0.8, linestyle="--", zorder=1)
        _add_chance(ax)
        ax.set_title(area, fontsize=12, color=AREA_COLORS[area], fontweight="bold")
        ax.set_xlabel("Time relative to event onset (s)", fontsize=10)
        ax.spines["left"].set_visible(True)

    axes[0].set_ylabel("Mean AUC  (Go vs No-Go)", fontsize=10)
    axes[0].set_ylim(0.30, 0.93)

    # Shared legend on rightmost panel
    handles = [
        mpatches.Patch(facecolor=EVENT_COLORS["Event1"], alpha=0.55,
                       label=EVENT_LABELS["Event1"]),
        mpatches.Patch(facecolor=EVENT_COLORS["Event3"], alpha=0.55,
                       label=EVENT_LABELS["Event3"]),
        mpatches.Patch(facecolor="#CCCCCC", alpha=0.70,
                       label="Shuffle null 95 % CI (E3, mean ± 1.96 SEM)"),
        mlines.Line2D([], [], color="black", linestyle="--", lw=0.8,
                      label="Event onset (t = 0)"),
        mlines.Line2D([], [], color="gray",  linestyle="--", lw=1.0,
                      label="Chance (0.5)"),
        mlines.Line2D([], [], color=AREA_COLORS["F5"], lw=5,
                      label=f"BH-FDR sig. bins (E3, q < {FDR_Q})"),
    ]
    axes[2].legend(handles=handles, fontsize=8, framealpha=0.88,
                   loc="upper left")

    fig.suptitle(
        "Time-Resolved Go/No-Go Decoding  ·  Shuffle-Null Envelope  ·  BH-FDR Significance\n"
        "(bar = contiguous bins where E3 > null upper bound, q < 0.05;"
        "  inset: peak AUC & onset latency)",
        fontsize=9.5, y=1.03,
    )
    fig.tight_layout()
    p = out / "fig2_extra1_null_overlay.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Figure 2 — Per-object time courses
# ---------------------------------------------------------------------------

def fig_per_object(tr: pd.DataFrame, out: Path) -> None:
    """
    6-panel grid (2 rows × 3 cols): rows = Event3 / Event1, cols = areas.
    Each panel shows one trace per grasped object ± SEM, plus the
    collapsed mean in light grey.  Tests whether the decoding time course
    is object-invariant (supports a grasp-general decision code) or
    object-specific (biologically interesting).
    """
    objects = sorted(tr["object"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.0),
                             sharey=True, sharex=False)

    for col_i, area in enumerate(AREA_ORDER):
        for row_i, event in enumerate(["Event3", "Event1"]):
            ax = axes[row_i, col_i]

            # Grey mean across all objects
            sub_all = tr[(tr["area"] == area) & (tr["event"] == event)]
            grp_all = sub_all.groupby("center_time_s")["auc"]
            t = np.array(sorted(grp_all.groups.keys()))
            mn_all = grp_all.mean().loc[t].values
            ax.plot(t, mn_all, color="#CCCCCC", lw=3.5, zorder=1,
                    label="All objects (mean)")

            # Per-object traces
            for obj in objects:
                sub = tr[(tr["area"] == area) &
                         (tr["event"] == event) &
                         (tr["object"] == obj)]
                if sub.empty:
                    continue
                grp = sub.groupby("center_time_s")["auc"]
                t_o = np.array(sorted(grp.groups.keys()))
                mn_o = grp.mean().loc[t_o].values
                se_o = grp.sem().loc[t_o].values

                style = OBJECT_COLORS.get(obj, {"color": "black", "ls": "-"})
                ax.plot(t_o, mn_o, color=style["color"],
                        ls=style["ls"], lw=1.8, zorder=3, label=obj)
                ax.fill_between(t_o, mn_o - se_o, mn_o + se_o,
                                color=style["color"], alpha=0.12, zorder=2)

            ax.axvline(0, color="black", lw=0.8, linestyle="--", zorder=1)
            _add_chance(ax)

            ev_tag = "E3 (movement)" if event == "Event3" else "E1 (baseline)"
            ax.set_title(f"{area}  —  {ev_tag}",
                         fontsize=10, color=AREA_COLORS[area], fontweight="bold")
            if col_i == 0:
                ax.set_ylabel("Mean AUC  (Go vs No-Go)", fontsize=9)
            if row_i == 1:
                ax.set_xlabel("Time relative to event onset (s)", fontsize=9)

    # Shared legend on top-right panel
    handle_mean = mlines.Line2D([], [], color="#CCCCCC", lw=3.5,
                                label="All objects (mean)")
    handles = [handle_mean]
    for obj in objects:
        style = OBJECT_COLORS.get(obj, {"color": "black", "ls": "-"})
        handles.append(mlines.Line2D([], [], color=style["color"],
                                     ls=style["ls"], lw=1.8, label=obj))
    handles += [
        mlines.Line2D([], [], color="black", linestyle="--", lw=0.8,
                      label="Event onset (t = 0)"),
        mlines.Line2D([], [], color="gray",  linestyle="--", lw=1.0,
                      label="Chance (0.5)"),
    ]
    axes[0, 2].legend(handles=handles, fontsize=8.5, framealpha=0.88,
                      loc="upper left")

    axes[0, 0].set_ylim(0.30, 0.93)

    fig.suptitle(
        "Time-Resolved Go/No-Go Decoding  —  Split by Grasped Object\n"
        "(one trace per object ± SEM across sessions;"
        "  grey = collapsed mean)",
        fontsize=10.5, y=1.02,
    )
    fig.tight_layout()
    p = out / "figS1_per_object_time_resolved.png"
    fig.savefig(str(p), bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate extra supplementary figures (does not overwrite existing ones)."
    )
    ap.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Path to results directory.",
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "generated"),
    )
    args = ap.parse_args()

    res  = Path(args.results_dir)
    out  = Path(args.out_dir)  # generated figures (git-ignored)
    out.mkdir(parents=True, exist_ok=True)
    null_path = res / "final_analysis" / "06_shuffle_null" / "shuffle_null_time_resolved.csv"
    tr_path   = res / "time_resolved_results.csv"

    for path, name in [(tr_path, "time_resolved_results.csv"),
                       (null_path, "shuffle_null_time_resolved.csv")]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print("Loading data...")
    tr   = pd.read_csv(str(tr_path))
    null = pd.read_csv(str(null_path))
    print(f"  time_resolved  : {len(tr):,} rows  ({tr.groupby(['area','session','object','event']).ngroups} units)")
    print(f"  shuffle_null   : {len(null):,} rows  ({null.groupby(['area','session','object','event']).ngroups} units)")

    print("\nGenerating extra figures...")
    fig_null_overlay(tr, null, out)
    fig_per_object(tr, out)

    print(f"\nAll extra figures saved to: {out}")


if __name__ == "__main__":
    main()
