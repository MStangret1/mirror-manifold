from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binomtest, wilcoxon
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Dataset defaults
# ---------------------------
EVENT_WINDOWS = {
    "Event1": (-0.2, 0.8),  # cue onset / early planning
    "Event3": (-0.8, 0.6),  # sound offset / go-no-go signal
    "Event4": (-0.5, 0.8),  # movement onset (Go only)
}

CONTRASTS = {
    "exe_go_vs_obs_go": (("Context1", "Condition1"), ("Context2", "Condition1")),
    "obs_go_vs_obs_nogo": (("Context2", "Condition1"), ("Context2", "Condition2")),
}

DEFAULT_BINSIZE = 0.02
DEFAULT_N_PCS = 8
DEFAULT_N_SPLITS = 5
DEFAULT_N_REPEATS = 10
DEFAULT_RANDOM_STATE = 0


# ---------------------------
# IO helpers
# ---------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_inventory(go_nogo_dir: str) -> Dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(os.path.join(go_nogo_dir, "inventory_completeness_summary.csv")),
        "combos": pd.read_csv(os.path.join(go_nogo_dir, "inventory_event_combos_with_completeness.csv")),
        "neurons": pd.read_csv(os.path.join(go_nogo_dir, "inventory_neurons_by_session.csv")),
    }


def find_event_file(base_dir: str, area: str, session: str, context: str, condition: str, object_: str, event: str) -> Optional[str]:
    pattern = os.path.join(base_dir, "Events", area, f"{session}_{context}_{condition}_{object_}_{event}.h5")
    matches = glob.glob(pattern)
    return matches[0] if len(matches) == 1 else None


def list_spike_files(base_dir: str, area: str, session: str) -> List[str]:
    pattern = os.path.join(base_dir, "Spikes", area, f"{session}_Spk_*.h5")
    return sorted(glob.glob(pattern))


# ---------------------------
# Safe FR extraction
# ---------------------------
def neuron_firing_rate_hdf5_safe(
    spikes_file: str,
    events_file: str,
    interval: Tuple[float, float],
    binsize: float,
    gauss_smooth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Safe replacement for Neuron_firing_rate_HDF5.
    Never plots unless explicitly extended by user.
    Returns:
        firing_rate: (n_bins,)
        firing_rate_trial: (n_trials, n_bins)
        raster: (n_trials, n_raster_bins) bool
    """
    with h5py.File(spikes_file, "r") as f:
        spikes = np.array(f["/spike_data"]).flatten()
    with h5py.File(events_file, "r") as f:
        events = np.array(f["/events_data"]).flatten()

    n_bins = int(round((interval[1] - interval[0]) / binsize))
    raster_binsize = 0.0001
    n_raster_bins = int(round((interval[1] - interval[0]) / raster_binsize))

    firing_rate = np.zeros(n_bins, dtype=float)
    firing_rate_trial = np.zeros((len(events), n_bins), dtype=float)
    raster = np.zeros((len(events), n_raster_bins), dtype=bool)

    for j, ev in enumerate(events):
        edges = np.linspace(ev + interval[0], ev + interval[1], n_bins + 1)
        counts = np.histogram(spikes, bins=edges)[0]
        firing_rate_trial[j, :] = counts / binsize
        firing_rate += counts

        edges_raster = np.linspace(ev + interval[0], ev + interval[1], n_raster_bins + 1)
        raster[j, :] = np.histogram(spikes, bins=edges_raster)[0].astype(bool)

    firing_rate = firing_rate / (binsize * max(len(events), 1))
    if gauss_smooth is not None:
        firing_rate = gaussian_filter1d(firing_rate, gauss_smooth / np.sqrt(2))
    return firing_rate, firing_rate_trial, raster


# ---------------------------
# Tensor builders
# ---------------------------
def build_tensor(
    base_dir: str,
    area: str,
    session: str,
    context: str,
    condition: str,
    object_: str,
    event: str,
    interval: Tuple[float, float],
    binsize: float = DEFAULT_BINSIZE,
    gauss_smooth: Optional[float] = None,
    min_neurons: int = 1,
) -> Optional[np.ndarray]:
    evt = find_event_file(base_dir, area, session, context, condition, object_, event)
    if evt is None:
        return None

    spike_files = list_spike_files(base_dir, area, session)
    if len(spike_files) < min_neurons:
        return None

    fr_trials_list: List[np.ndarray] = []
    shape_ref: Optional[Tuple[int, int]] = None

    for spk in spike_files:
        _, fr_trial, _ = neuron_firing_rate_hdf5_safe(spk, evt, interval, binsize, gauss_smooth=gauss_smooth)
        if shape_ref is None:
            shape_ref = fr_trial.shape
        elif fr_trial.shape != shape_ref:
            return None
        fr_trials_list.append(fr_trial)

    if not fr_trials_list:
        return None
    return np.stack(fr_trials_list, axis=2)  # (trials, bins, neurons)


# ---------------------------
# Features / PCA / decoding
# ---------------------------
def make_features_mean_over_time(tensor: np.ndarray, time_slice: Optional[slice] = None) -> np.ndarray:
    if time_slice is None:
        time_slice = slice(0, tensor.shape[1])
    return tensor[:, time_slice, :].mean(axis=1)


def make_features_flatten_time(tensor: np.ndarray, time_slice: Optional[slice] = None) -> np.ndarray:
    if time_slice is None:
        time_slice = slice(0, tensor.shape[1])
    X = tensor[:, time_slice, :]
    return X.reshape(X.shape[0], -1)


def mean_center_conditions(*tensors: np.ndarray) -> List[np.ndarray]:
    """
    Remove condition-independent signal across conditions at each time bin.
    Assumes same bins x neurons for all tensors; can differ in trial count.
    """
    stacked_means = np.stack([ten.mean(axis=0) for ten in tensors], axis=0)  # (conditions, bins, neurons)
    global_mean = stacked_means.mean(axis=0)
    return [ten - global_mean[None, :, :] for ten in tensors]


def fit_shared_pca(*tensors: np.ndarray, n_components: int = DEFAULT_N_PCS, random_state: int = DEFAULT_RANDOM_STATE):
    X = np.concatenate(tensors, axis=0)
    T, B, N = X.shape
    X2 = X.reshape(T * B, N)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X2)
    n_comp = min(n_components, Xs.shape[0], Xs.shape[1])
    if n_comp < 2:
        raise ValueError(f"Too few dimensions for PCA: n_samples={Xs.shape[0]}, n_features={Xs.shape[1]}")
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z2 = pca.fit_transform(Xs)
    Z3 = Z2.reshape(T, B, n_comp)
    out = []
    start = 0
    for ten in tensors:
        ntr = ten.shape[0]
        out.append(Z3[start:start+ntr])
        start += ntr
    return scaler, pca, out


def whiten_Z(Z: np.ndarray, pca: PCA, eps: float = 1e-12) -> np.ndarray:
    evals = pca.explained_variance_[: Z.shape[2]]
    return Z / np.sqrt(evals + eps)


def mean_traj(Z: np.ndarray) -> np.ndarray:
    return Z.mean(axis=0)


def traj_distance(muA: np.ndarray, muB: np.ndarray) -> np.ndarray:
    return np.linalg.norm(muA - muB, axis=1)


def velocity(mu: np.ndarray) -> np.ndarray:
    return np.diff(mu, axis=0)


def cosine_sim_time(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n1 = np.linalg.norm(v1, axis=1) + eps
    n2 = np.linalg.norm(v2, axis=1) + eps
    return np.sum(v1 * v2, axis=1) / (n1 * n2)


def pca_subspace(tensor: np.ndarray, n_components: int = 5, random_state: int = DEFAULT_RANDOM_STATE) -> Optional[np.ndarray]:
    X = tensor.reshape(-1, tensor.shape[2])
    Xs = StandardScaler().fit_transform(X)
    k = min(n_components, Xs.shape[0], Xs.shape[1])
    if k < 2:
        return None
    p = PCA(n_components=k, random_state=random_state).fit(Xs)
    return p.components_.T


def min_principal_angle(tensorA: np.ndarray, tensorB: np.ndarray, n_components: int = 5) -> float:
    UA = pca_subspace(tensorA, n_components=n_components)
    UB = pca_subspace(tensorB, n_components=n_components)
    if UA is None or UB is None:
        return np.nan
    return float(np.min(subspace_angles(UA, UB)))


def build_decoder_pipeline(n_pcs: int = DEFAULT_N_PCS, C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("pca", PCA(n_components=n_pcs, random_state=DEFAULT_RANDOM_STATE)),
        ("clf", LogisticRegression(penalty="l2", C=C, solver="liblinear", max_iter=5000)),
    ])


def make_cv(n_splits: int = DEFAULT_N_SPLITS, n_repeats: int = DEFAULT_N_REPEATS, random_state: int = DEFAULT_RANDOM_STATE):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)


def _safe_n_pcs(X: np.ndarray, requested: int) -> int:
    n_samples, n_features = X.shape
    # keep one dimension margin if close to sample limit
    k = min(requested, n_features, n_samples - 1)
    return max(2, k)


def cv_score_accuracy(model: Pipeline, X: np.ndarray, y: np.ndarray, cv) -> Tuple[float, np.ndarray]:
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean()), scores


def cv_score_auc(model: Pipeline, X: np.ndarray, y: np.ndarray, cv) -> Tuple[float, np.ndarray]:
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        proba = m.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    aucs = np.asarray(aucs)
    return float(aucs.mean()), aucs


def decode_two_tensors(
    tensorA: np.ndarray,
    tensorB: np.ndarray,
    time_slice: Optional[slice] = None,
    feature_mode: str = "mean",
    n_pcs: int = DEFAULT_N_PCS,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, float]:
    if feature_mode == "mean":
        XA = make_features_mean_over_time(tensorA, time_slice=time_slice)
        XB = make_features_mean_over_time(tensorB, time_slice=time_slice)
    elif feature_mode == "flatten":
        XA = make_features_flatten_time(tensorA, time_slice=time_slice)
        XB = make_features_flatten_time(tensorB, time_slice=time_slice)
    else:
        raise ValueError("feature_mode must be 'mean' or 'flatten'")

    X = np.vstack([XA, XB])
    y = np.array([1] * len(XA) + [0] * len(XB))
    k = _safe_n_pcs(X, n_pcs)
    model = build_decoder_pipeline(n_pcs=k, C=1.0)
    cv = make_cv(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    acc, acc_all = cv_score_accuracy(model, X, y, cv)
    auc, auc_all = cv_score_auc(model, X, y, cv)
    return {
        "acc": float(acc),
        "auc": float(auc),
        "acc_std": float(np.std(acc_all, ddof=1)) if len(acc_all) > 1 else 0.0,
        "auc_std": float(np.std(auc_all, ddof=1)) if len(auc_all) > 1 else 0.0,
        "n_features": int(X.shape[1]),
        "n_pcs_used": int(k),
    }


def permutation_test_cv(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    n_perm: int = 1000,
    metric: str = "auc",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    if metric == "accuracy":
        obs, _ = cv_score_accuracy(model, X, y, cv)
    elif metric == "auc":
        obs, _ = cv_score_auc(model, X, y, cv)
    else:
        raise ValueError("metric must be 'accuracy' or 'auc'")

    null_scores = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        y_perm = rng.permutation(y)
        if metric == "accuracy":
            s, _ = cv_score_accuracy(model, X, y_perm, cv)
        else:
            s, _ = cv_score_auc(model, X, y_perm, cv)
        null_scores[i] = s
    p = (np.sum(null_scores >= obs) + 1) / (n_perm + 1)
    return float(obs), null_scores, float(p)


def stratified_trial_bootstrap(
    XA: np.ndarray,
    XB: np.ndarray,
    model: Pipeline,
    cv,
    B: int = 1000,
    metric: str = "auc",
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    nA, nB = len(XA), len(XB)
    out = np.empty(B, dtype=float)
    for b in range(B):
        ia = rng.integers(0, nA, size=nA)
        ib = rng.integers(0, nB, size=nB)
        Xb = np.vstack([XA[ia], XB[ib]])
        yb = np.array([1] * nA + [0] * nB)
        if metric == "accuracy":
            s, _ = cv_score_accuracy(model, Xb, yb, cv)
        else:
            s, _ = cv_score_auc(model, Xb, yb, cv)
        out[b] = s
    return out


def summarize_bootstrap(scores: np.ndarray, obs: float) -> Dict[str, float]:
    return {
        "obs": float(obs),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "ci95_low": float(np.percentile(scores, 2.5)),
        "ci95_high": float(np.percentile(scores, 97.5)),
        "bias": float(np.mean(scores) - obs),
    }


# ---------------------------
# Analysis units
# ---------------------------
@dataclass
class AnalysisUnit:
    area: str
    session: str
    object_: str
    event: str
    interval: Tuple[float, float]



def iter_analysis_units(
    inventory_summary: pd.DataFrame,
    events: Sequence[str] = ("Event1", "Event3"),
    areas: Optional[Sequence[str]] = None,
    objects: Sequence[str] = ("Object1", "Object2", "Object3"),
) -> Iterable[AnalysisUnit]:
    df = inventory_summary.copy()
    if areas is not None:
        df = df[df["area"].isin(areas)]

    # Use sessions present in both EXE Go and OBS Go; OBS No-Go exists only in Context2/Condition2.
    by_keys = ["area", "session"]
    avail = df.groupby(by_keys).apply(
        lambda g: {
            (row["context"], row["condition"])
            for _, row in g.iterrows()
        }
    )

    for (area, session), combos in avail.items():
        if ("Context1", "Condition1") not in combos:
            continue
        if ("Context2", "Condition1") not in combos:
            continue
        if ("Context2", "Condition2") not in combos:
            continue
        for object_ in objects:
            for event in events:
                yield AnalysisUnit(area=area, session=session, object_=object_, event=event, interval=EVENT_WINDOWS[event])


# ---------------------------
# Session-object level batch analyses
# ---------------------------
def analyze_unit(
    unit: AnalysisUnit,
    base_dir: str,
    binsize: float = DEFAULT_BINSIZE,
    gauss_smooth: Optional[float] = None,
    min_neurons: int = 3,
    decode_repeats: int = DEFAULT_N_REPEATS,
    n_perm: int = 500,
    n_boot: int = 500,
) -> Optional[Dict[str, float]]:
    exe_go = build_tensor(base_dir, unit.area, unit.session, "Context1", "Condition1", unit.object_, unit.event, unit.interval, binsize=binsize, gauss_smooth=gauss_smooth, min_neurons=min_neurons)
    obs_go = build_tensor(base_dir, unit.area, unit.session, "Context2", "Condition1", unit.object_, unit.event, unit.interval, binsize=binsize, gauss_smooth=gauss_smooth, min_neurons=min_neurons)
    obs_nogo = build_tensor(base_dir, unit.area, unit.session, "Context2", "Condition2", unit.object_, unit.event, unit.interval, binsize=binsize, gauss_smooth=gauss_smooth, min_neurons=min_neurons)

    if exe_go is None or obs_go is None or obs_nogo is None:
        return None

    n_trials_exe, n_bins, n_neurons = exe_go.shape
    n_trials_obs_go = obs_go.shape[0]
    n_trials_obs_nogo = obs_nogo.shape[0]

    # Shared manifold on raw and centered data
    centered_exe, centered_obs_go, centered_obs_nogo = mean_center_conditions(exe_go, obs_go, obs_nogo)
    _, pca_raw, (Z_exe, Z_obs_go, Z_obs_nogo) = fit_shared_pca(exe_go, obs_go, obs_nogo)
    _, pca_ctr, (Zc_exe, Zc_obs_go, Zc_obs_nogo) = fit_shared_pca(centered_exe, centered_obs_go, centered_obs_nogo)

    Zw_exe, Zw_go, Zw_nogo = whiten_Z(Z_exe, pca_raw), whiten_Z(Z_obs_go, pca_raw), whiten_Z(Z_obs_nogo, pca_raw)
    Zcw_exe, Zcw_go, Zcw_nogo = whiten_Z(Zc_exe, pca_ctr), whiten_Z(Zc_obs_go, pca_ctr), whiten_Z(Zc_obs_nogo, pca_ctr)

    mu_exe, mu_go, mu_nogo = mean_traj(Zw_exe), mean_traj(Zw_go), mean_traj(Zw_nogo)
    muce, mucg, mucn = mean_traj(Zcw_exe), mean_traj(Zcw_go), mean_traj(Zcw_nogo)

    d_exe_obs = traj_distance(mu_exe, mu_go)
    d_go_nogo = traj_distance(mu_go, mu_nogo)
    d_exe_nogo = traj_distance(mu_exe, mu_nogo)
    d_go_nogo_centered = traj_distance(mucg, mucn)

    v_exe, v_go, v_nogo = velocity(mu_exe), velocity(mu_go), velocity(mu_nogo)
    v_cos_exe_obs = cosine_sim_time(v_exe, v_go)
    v_cos_go_nogo = cosine_sim_time(v_go, v_nogo)

    bin_centers = unit.interval[0] + (np.arange(n_bins) + 0.5) * binsize
    pre_idx = np.where(bin_centers < 0.0)[0]
    post_idx = np.where(bin_centers >= 0.0)[0]
    if len(pre_idx) == 0 or len(post_idx) == 0:
        raise ValueError(
            f"Cannot infer pre/post slices for {unit.event} with interval={unit.interval}, n_bins={n_bins}, binsize={binsize}"
        )
    pre_slice = slice(int(pre_idx[0]), int(pre_idx[-1]) + 1)
    post_slice = slice(int(post_idx[0]), int(post_idx[-1]) + 1)

    decode_full = decode_two_tensors(obs_go, obs_nogo, time_slice=slice(0, n_bins), n_repeats=decode_repeats)
    decode_pre = decode_two_tensors(obs_go, obs_nogo, time_slice=pre_slice, n_repeats=decode_repeats)
    decode_post = decode_two_tensors(obs_go, obs_nogo, time_slice=post_slice, n_repeats=decode_repeats)

    decode_exe_obs = decode_two_tensors(exe_go, obs_go, time_slice=slice(0, n_bins), n_repeats=decode_repeats)

    # permutation + bootstrap on the main decision contrast
    XA = make_features_mean_over_time(obs_go)
    XB = make_features_mean_over_time(obs_nogo)
    X = np.vstack([XA, XB])
    y = np.array([1] * len(XA) + [0] * len(XB))
    n_pcs_used = _safe_n_pcs(X, DEFAULT_N_PCS)
    model = build_decoder_pipeline(n_pcs=n_pcs_used, C=1.0)
    cv = make_cv(n_splits=DEFAULT_N_SPLITS, n_repeats=decode_repeats, random_state=DEFAULT_RANDOM_STATE)

    obs_auc, null_auc, p_auc = permutation_test_cv(model, X, y, cv, n_perm=n_perm, metric="auc")
    obs_acc, null_acc, p_acc = permutation_test_cv(model, X, y, cv, n_perm=n_perm, metric="accuracy")

    boot_auc = stratified_trial_bootstrap(XA, XB, model, cv, B=n_boot, metric="auc")
    boot_acc = stratified_trial_bootstrap(XA, XB, model, cv, B=n_boot, metric="accuracy")
    boot_auc_sum = summarize_bootstrap(boot_auc, obs_auc)
    boot_acc_sum = summarize_bootstrap(boot_acc, obs_acc)

    return {
        "area": unit.area,
        "session": unit.session,
        "object": unit.object_,
        "event": unit.event,
        "interval_start": float(unit.interval[0]),
        "interval_stop": float(unit.interval[1]),
        "n_neurons": int(n_neurons),
        "n_bins": int(n_bins),
        "n_trials_exe_go": int(n_trials_exe),
        "n_trials_obs_go": int(n_trials_obs_go),
        "n_trials_obs_nogo": int(n_trials_obs_nogo),
        "n_pcs_raw": int(pca_raw.n_components_),
        "n_pcs_centered": int(pca_ctr.n_components_),
        "dist_exe_obs_mean": float(np.mean(d_exe_obs)),
        "dist_go_nogo_mean": float(np.mean(d_go_nogo)),
        "dist_exe_nogo_mean": float(np.mean(d_exe_nogo)),
        "dist_go_nogo_centered_mean": float(np.mean(d_go_nogo_centered)),
        "vel_cos_exe_obs_mean": float(np.mean(v_cos_exe_obs)),
        "vel_cos_go_nogo_mean": float(np.mean(v_cos_go_nogo)),
        "angle_exe_obs_min": float(min_principal_angle(exe_go, obs_go, n_components=5)),
        "angle_go_nogo_min": float(min_principal_angle(obs_go, obs_nogo, n_components=5)),
        "angle_exe_nogo_min": float(min_principal_angle(exe_go, obs_nogo, n_components=5)),
        "dec_obs_go_nogo_acc_full": decode_full["acc"],
        "dec_obs_go_nogo_auc_full": decode_full["auc"],
        "dec_obs_go_nogo_acc_pre": decode_pre["acc"],
        "dec_obs_go_nogo_auc_pre": decode_pre["auc"],
        "dec_obs_go_nogo_acc_post": decode_post["acc"],
        "dec_obs_go_nogo_auc_post": decode_post["auc"],
        "dec_exe_obs_acc_full": decode_exe_obs["acc"],
        "dec_exe_obs_auc_full": decode_exe_obs["auc"],
        "perm_obs_go_nogo_auc": obs_auc,
        "perm_obs_go_nogo_auc_p": p_auc,
        "perm_obs_go_nogo_acc": obs_acc,
        "perm_obs_go_nogo_acc_p": p_acc,
        "boot_obs_go_nogo_auc_mean": boot_auc_sum["mean"],
        "boot_obs_go_nogo_auc_ci95_low": boot_auc_sum["ci95_low"],
        "boot_obs_go_nogo_auc_ci95_high": boot_auc_sum["ci95_high"],
        "boot_obs_go_nogo_acc_mean": boot_acc_sum["mean"],
        "boot_obs_go_nogo_acc_ci95_low": boot_acc_sum["ci95_low"],
        "boot_obs_go_nogo_acc_ci95_high": boot_acc_sum["ci95_high"],
    }


# ---------------------------
# Time-resolved and object generalization
# ---------------------------
def time_resolved_decode(
    tensorA: np.ndarray,
    tensorB: np.ndarray,
    interval: Tuple[float, float],
    binsize: float = DEFAULT_BINSIZE,
    window_bins: int = 5,
    step_bins: int = 1,
    n_repeats: int = DEFAULT_N_REPEATS,
) -> pd.DataFrame:
    n_bins = tensorA.shape[1]
    rows = []
    for start in range(0, n_bins - window_bins + 1, step_bins):
        stop = start + window_bins
        dec = decode_two_tensors(tensorA, tensorB, time_slice=slice(start, stop), n_repeats=n_repeats)
        center_bin = (start + stop - 1) / 2
        center_time = interval[0] + (center_bin + 0.5) * binsize
        rows.append({
            "start_bin": start,
            "stop_bin": stop,
            "center_time_s": center_time,
            "acc": dec["acc"],
            "auc": dec["auc"],
            "n_pcs_used": dec["n_pcs_used"],
        })
    return pd.DataFrame(rows)


def object_generalization_matrix(
    tensors_by_object: Dict[str, Tuple[np.ndarray, np.ndarray]],
    feature_mode: str = "mean",
    n_repeats: int = DEFAULT_N_REPEATS,
) -> pd.DataFrame:
    """
    Train on object i, test on object j for OBS Go vs OBS No-Go.
    Off-diagonal entries are true cross-object transfer.
    Diagonal entries use within-object cross-validation (not train=test on the same data).
    tensors_by_object maps object -> (obs_go_tensor, obs_nogo_tensor)
    """
    rows = []
    objects = sorted(tensors_by_object)
    for train_obj in objects:
        go_train, nogo_train = tensors_by_object[train_obj]
        Xg_tr = make_features_mean_over_time(go_train) if feature_mode == "mean" else make_features_flatten_time(go_train)
        Xn_tr = make_features_mean_over_time(nogo_train) if feature_mode == "mean" else make_features_flatten_time(nogo_train)
        X_train = np.vstack([Xg_tr, Xn_tr])
        y_train = np.array([1] * len(Xg_tr) + [0] * len(Xn_tr))
        k = _safe_n_pcs(X_train, DEFAULT_N_PCS)
        model = build_decoder_pipeline(n_pcs=k)
        cv = make_cv(n_splits=DEFAULT_N_SPLITS, n_repeats=n_repeats, random_state=DEFAULT_RANDOM_STATE)

        for test_obj in objects:
            go_test, nogo_test = tensors_by_object[test_obj]
            Xg_te = make_features_mean_over_time(go_test) if feature_mode == "mean" else make_features_flatten_time(go_test)
            Xn_te = make_features_mean_over_time(nogo_test) if feature_mode == "mean" else make_features_flatten_time(nogo_test)
            X_test = np.vstack([Xg_te, Xn_te])
            y_test = np.array([1] * len(Xg_te) + [0] * len(Xn_te))

            if train_obj == test_obj:
                acc, _ = cv_score_accuracy(model, X_train, y_train, cv)
                auc, _ = cv_score_auc(model, X_train, y_train, cv)
                rows.append({
                    "train_object": train_obj,
                    "test_object": test_obj,
                    "acc": float(acc),
                    "auc": float(auc),
                    "evaluation": "within_object_cv",
                    "is_cross_object": False,
                })
            else:
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                proba = model.predict_proba(X_test)[:, 1]
                rows.append({
                    "train_object": train_obj,
                    "test_object": test_obj,
                    "acc": float(np.mean(pred == y_test)),
                    "auc": float(roc_auc_score(y_test, proba)),
                    "evaluation": "cross_object_holdout",
                    "is_cross_object": True,
                })
    return pd.DataFrame(rows)


# ---------------------------
# Meta-analysis summaries
# ---------------------------
def bootstrap_mean_ci(values: np.ndarray, B: int = 5000, random_state: int = DEFAULT_RANDOM_STATE) -> Tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    boots = np.empty(B, dtype=float)
    for b in range(B):
        idx = rng.integers(0, len(values), size=len(values))
        boots[b] = np.mean(values[idx])
    return float(np.mean(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarize_results(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = [
        "dist_exe_obs_mean",
        "dist_go_nogo_mean",
        "dist_go_nogo_centered_mean",
        "dec_obs_go_nogo_auc_full",
        "dec_obs_go_nogo_auc_pre",
        "dec_obs_go_nogo_auc_post",
        "dec_exe_obs_auc_full",
        "perm_obs_go_nogo_auc_p",
        "n_neurons",
    ]
    rows = []
    for keys, g in df.groupby(list(group_cols)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["n_units"] = int(len(g))
        for metric in metrics:
            mean, lo, hi = bootstrap_mean_ci(g[metric].to_numpy())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = lo
            row[f"{metric}_ci95_high"] = hi
        # sign / wilcoxon for pre vs post AUC
        valid = g[["dec_obs_go_nogo_auc_pre", "dec_obs_go_nogo_auc_post"]].dropna()
        if len(valid) >= 3:
            try:
                _, p = wilcoxon(valid["dec_obs_go_nogo_auc_post"], valid["dec_obs_go_nogo_auc_pre"], alternative="greater", zero_method="wilcox")
            except ValueError:
                p = np.nan
            n_pos = int((valid["dec_obs_go_nogo_auc_post"] - valid["dec_obs_go_nogo_auc_pre"] > 0).sum())
            n_nonzero = int((valid["dec_obs_go_nogo_auc_post"] - valid["dec_obs_go_nogo_auc_pre"] != 0).sum())
            p_sign = binomtest(n_pos, n_nonzero, p=0.5, alternative="greater").pvalue if n_nonzero > 0 else np.nan
        else:
            p, p_sign = np.nan, np.nan
        row["wilcoxon_post_gt_pre_auc_p"] = p
        row["sign_post_gt_pre_auc_p"] = p_sign
        rows.append(row)
    return pd.DataFrame(rows)
