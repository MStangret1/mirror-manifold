"""
13_single_trial_pca_scatter.py
==============================
Single-trial PCA scatter for MonkeyA Session02, AIP, Event3.

Shows that Go and No-Go trials are separable in PC1-PC2 even WITHOUT
trial-averaging — i.e., the decoding result is not a trial-averaging artefact.

Design choices:
  - Load the pre-saved tensors (trials x bins x neurons) for OBS Go and OBS No-Go.
  - Mean-firing-rate over the POST-EVENT window (t >= 0) for each trial =>
    one point per trial in neuron space.
  - StandardScaler + PCA fitted on all trials jointly.
  - Each trial plotted as a dot in PC1-PC2. Go = orange, No-Go = green.
  - Object shown as marker shape (circle / triangle / square).
  - 95% confidence ellipse drawn per condition (Pearson-style: 2-sigma ellipse
    of the 2D Gaussian fit).

Output: saved to reworked_results/single_trial_pca_scatter.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_condition(artifacts_dir: Path, condition: str, session: str = "MonkeyA_Session02",
                   area: str = "AIP", event: str = "Event3",
                   objects: tuple[str, ...] = ("Object1", "Object2", "Object3")
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate tensors for a condition across objects.

    Returns:
        X: (total_trials, post_bins, n_neurons)
        obj_ids: (total_trials,) int array, 0/1/2 for Object1/2/3
    """
    parts, ids = [], []
    for oid, obj in enumerate(objects):
        fname = artifacts_dir / f"tensor_{condition}_{area}_{session}_{obj}_{event}.npy"
        if not fname.exists():
            print(f"  WARNING: {fname.name} not found, skipping")
            continue
        t = np.load(str(fname))  # (trials, bins, neurons)
        parts.append(t)
        ids.append(np.full(t.shape[0], oid, dtype=int))
    return np.concatenate(parts, axis=0), np.concatenate(ids)


def post_event_mean(tensor: np.ndarray, n_bins: int, post_fraction: float = 0.5) -> np.ndarray:
    """
    Average over the second half of the time window (post-event period).
    Returns: (trials, neurons)
    """
    start_bin = int(n_bins * (1 - post_fraction))
    return tensor[:, start_bin:, :].mean(axis=1)


def confidence_ellipse_2d(x: np.ndarray, y: np.ndarray, ax: plt.Axes,
                           n_std: float = 2.0, **kwargs) -> None:
    """Draw a covariance-based confidence ellipse for 2D data."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    e = Ellipse(xy=(np.mean(x), np.mean(y)), width=width, height=height,
                angle=angle, **kwargs)
    ax.add_patch(e)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts-dir",
        default=str(Path(__file__).resolve().parent.parent / "artifacts"),
    )
    ap.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "generated"),
    )
    # NOTE: the session/area/event below are an ILLUSTRATIVE single-unit example,
    # not the full analysis. Override via CLI to render other units.
    ap.add_argument("--session", default="MonkeyA_Session02")
    ap.add_argument("--area", default="AIP")
    ap.add_argument("--event", default="Event3")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    objects = ("Object1", "Object2", "Object3")
    obj_markers = ["o", "^", "s"]  # circle, triangle, square
    obj_labels  = ["Object 1", "Object 2", "Object 3"]

    # --- load tensors --------------------------------------------------------
    print("Loading OBS Go tensors...")
    go_tensor, go_obj = load_condition(art, "obs_go", args.session, args.area, args.event, objects)
    print(f"  OBS Go:   {go_tensor.shape}  (trials x bins x neurons)")

    print("Loading OBS No-Go tensors...")
    nogo_tensor, nogo_obj = load_condition(art, "obs_nogo", args.session, args.area, args.event, objects)
    print(f"  OBS NoGo: {nogo_tensor.shape}")

    n_bins = go_tensor.shape[1]

    # --- per-trial features: mean over post-event window --------------------
    # Event3 spans -0.8 to +0.6 s; post-event = t >= 0 ~= last 43% of 70 bins
    # Using last 43 bins (t=0 to +0.6 s) out of 70
    n_post_bins = round(n_bins * 0.6 / 1.4)  # 0.6s out of 1.4s total window
    go_feat   = go_tensor[:,   -n_post_bins:, :].mean(axis=1)   # (n_go,   n_neurons)
    nogo_feat = nogo_tensor[:, -n_post_bins:, :].mean(axis=1)   # (n_nogo, n_neurons)

    print(f"  Using last {n_post_bins}/{n_bins} bins (post-event window)")
    print(f"  Feature matrix: go={go_feat.shape}, nogo={nogo_feat.shape}")

    # --- PCA on all trials jointly ------------------------------------------
    X_all = np.concatenate([go_feat, nogo_feat], axis=0)
    labels = np.array([0] * len(go_feat) + [1] * len(nogo_feat))   # 0=Go, 1=NoGo
    obj_all = np.concatenate([go_obj, nogo_obj])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X_scaled)  # (n_trials_total, 2)

    var_exp = pca.explained_variance_ratio_ * 100
    print(f"  PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%")

    Z_go   = Z[labels == 0]
    Z_nogo = Z[labels == 1]
    obj_go   = obj_all[labels == 0]
    obj_nogo = obj_all[labels == 1]

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))

    GO_COLOR   = "#E07B00"   # warm orange
    NOGO_COLOR = "#2A8C3F"   # forest green
    ALPHA_DOT  = 0.85
    DOT_SIZE   = 70

    for oid, (marker, olabel) in enumerate(zip(obj_markers, obj_labels)):
        # Go dots
        mask_go = obj_go == oid
        ax.scatter(Z_go[mask_go, 0], Z_go[mask_go, 1],
                   c=GO_COLOR, marker=marker, s=DOT_SIZE,
                   alpha=ALPHA_DOT, edgecolors="white", linewidths=0.5,
                   zorder=3)
        # NoGo dots
        mask_nogo = obj_nogo == oid
        ax.scatter(Z_nogo[mask_nogo, 0], Z_nogo[mask_nogo, 1],
                   c=NOGO_COLOR, marker=marker, s=DOT_SIZE,
                   alpha=ALPHA_DOT, edgecolors="white", linewidths=0.5,
                   zorder=3)

    # 95% confidence ellipses (2-sigma)
    confidence_ellipse_2d(Z_go[:, 0], Z_go[:, 1], ax,
                          n_std=2.0, facecolor=GO_COLOR, alpha=0.12,
                          edgecolor=GO_COLOR, linewidth=1.5, linestyle="--", zorder=2)
    confidence_ellipse_2d(Z_nogo[:, 0], Z_nogo[:, 1], ax,
                          n_std=2.0, facecolor=NOGO_COLOR, alpha=0.12,
                          edgecolor=NOGO_COLOR, linewidth=1.5, linestyle="--", zorder=2)

    # --- legend: condition (color) + object (shape) -------------------------
    leg_go   = mpatches.Patch(facecolor=GO_COLOR,   label="Go (Condition 1)",    alpha=0.85)
    leg_nogo = mpatches.Patch(facecolor=NOGO_COLOR, label="No-Go (Condition 2)", alpha=0.85)

    obj_handles = [
        plt.scatter([], [], c="gray", marker=m, s=50, label=lbl)
        for m, lbl in zip(obj_markers, obj_labels)
    ]
    legend1 = ax.legend(handles=[leg_go, leg_nogo],
                        loc="upper left", fontsize=9, title="Condition",
                        title_fontsize=9, framealpha=0.85)
    ax.legend(handles=obj_handles,
              loc="lower right", fontsize=9, title="Object",
              title_fontsize=9, framealpha=0.85)
    ax.add_artist(legend1)

    # --- labels & style -------------------------------------------------------
    ax.set_xlabel(f"PC 1  ({var_exp[0]:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC 2  ({var_exp[1]:.1f}% variance)", fontsize=11)
    ax.set_title(
        f"Single-trial PCA  |  {args.session}  {args.area}  {args.event}\n"
        f"OBS Go vs No-Go  —  post-event window (t = 0 to +0.6 s)",
        fontsize=10,
    )
    ax.axhline(0, color="gray", linewidth=0.4, linestyle=":")
    ax.axvline(0, color="gray", linewidth=0.4, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    out_path = out / "single_trial_pca_scatter.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved figure: {out_path}")

    # --- quick separability check -------------------------------------------
    from scipy.stats import mannwhitneyu as mwu
    stat, p = mwu(Z_go[:, 0], Z_nogo[:, 0], alternative="two-sided")
    print(f"  PC1 Go vs NoGo:  U={stat:.0f}, p={p:.4e}  "
          f"(Go mean={Z_go[:,0].mean():+.2f}, NoGo mean={Z_nogo[:,0].mean():+.2f})")
    stat2, p2 = mwu(Z_go[:, 1], Z_nogo[:, 1], alternative="two-sided")
    print(f"  PC2 Go vs NoGo:  U={stat2:.0f}, p={p2:.4e}  "
          f"(Go mean={Z_go[:,1].mean():+.2f}, NoGo mean={Z_nogo[:,1].mean():+.2f})")


if __name__ == "__main__":
    main()
