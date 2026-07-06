# core_utils_ext.py
"""
High-quality extensions to core_utils.py for the final analysis pass.

Important principles:
- no silent thresholding based on arbitrary fixed numbers when a data-driven
  alternative is available;
- explicit validation of inputs and shape assumptions;
- all null-model-based significance should be derived from the same feature
  extraction pipeline as the primary metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from core_utils import (
    DEFAULT_BINSIZE,
    DEFAULT_N_PCS,
    DEFAULT_N_REPEATS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    _safe_n_pcs,
    build_decoder_pipeline,
    build_tensor,
    cv_score_auc,
    make_cv,
    make_features_mean_over_time,
)


@dataclass(frozen=True)
class PrePostSlices:
    pre_slice: slice
    post_slice: slice
    bin_centers: np.ndarray


def bin_centers_from_interval(
    interval: Tuple[float, float],
    n_bins: int,
    binsize: float = DEFAULT_BINSIZE,
) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    start = float(interval[0])
    return start + (np.arange(n_bins) + 0.5) * float(binsize)


def infer_pre_post_slices(
    interval: Tuple[float, float],
    n_bins: int,
    binsize: float = DEFAULT_BINSIZE,
) -> PrePostSlices:
    centers = bin_centers_from_interval(interval=interval, n_bins=n_bins, binsize=binsize)
    pre_idx = np.where(centers < 0.0)[0]
    post_idx = np.where(centers >= 0.0)[0]
    if len(pre_idx) == 0 or len(post_idx) == 0:
        raise ValueError(
            f"Cannot infer pre/post slices from interval={interval}, n_bins={n_bins}, binsize={binsize}."
        )
    return PrePostSlices(
        pre_slice=slice(int(pre_idx[0]), int(pre_idx[-1]) + 1),
        post_slice=slice(int(post_idx[0]), int(post_idx[-1]) + 1),
        bin_centers=centers,
    )


def _two_sided_upper_pvalue(null: np.ndarray, obs: float) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if len(null) == 0 or not np.isfinite(obs):
        return np.nan
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))


def _summarize_null(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_ci95_low": np.nan,
            f"{prefix}_ci95_high": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_ci95_low": float(np.percentile(arr, 2.5)),
        f"{prefix}_ci95_high": float(np.percentile(arr, 97.5)),
    }


def _safe_cv_auc(model, X, y, cv):
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) < 2:
            continue
        if len(np.unique(y_test)) < 2:
            continue

        model.fit(X[train_idx], y_train)
        proba = model.predict_proba(X[test_idx])[:, 1]
        scores.append(roc_auc_score(y_test, proba))

    if len(scores) == 0:
        return np.nan, np.nan
    if len(scores) == 1:
        return float(scores[0]), 0.0
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


def bh_qvalues(pvals: Iterable[float]) -> np.ndarray:
    pvals_arr = np.asarray(list(pvals), dtype=float)
    q = np.full_like(pvals_arr, np.nan, dtype=float)

    valid = np.isfinite(pvals_arr)
    if not valid.any():
        return q

    pv = pvals_arr[valid]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]

    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_valid = np.empty_like(q_ranked)
    q_valid[order] = q_ranked
    q[valid] = q_valid
    return q


def shuffle_pre_post_null(
    obs_go: np.ndarray,
    obs_nogo: np.ndarray,
    interval: Tuple[float, float],
    n_shuffle: int = 500,
    binsize: float = DEFAULT_BINSIZE,
    n_splits: int = DEFAULT_N_SPLITS,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, float]:
    """
    Null model for the pre->post AUC jump using real event-relative pre/post bins.
    """
    if obs_go.ndim != 3 or obs_nogo.ndim != 3:
        raise ValueError("obs_go and obs_nogo must have shape (n_trials, n_bins, n_neurons)")
    if obs_go.shape[1] != obs_nogo.shape[1] or obs_go.shape[2] != obs_nogo.shape[2]:
        raise ValueError("obs_go and obs_nogo must have matching n_bins and n_neurons")

    slices = infer_pre_post_slices(interval=interval, n_bins=obs_go.shape[1], binsize=binsize)

    XA_pre = make_features_mean_over_time(obs_go, time_slice=slices.pre_slice)
    XB_pre = make_features_mean_over_time(obs_nogo, time_slice=slices.pre_slice)
    XA_post = make_features_mean_over_time(obs_go, time_slice=slices.post_slice)
    XB_post = make_features_mean_over_time(obs_nogo, time_slice=slices.post_slice)

    X_pre = np.vstack([XA_pre, XB_pre])
    X_post = np.vstack([XA_post, XB_post])
    y = np.array([1] * len(XA_pre) + [0] * len(XB_pre), dtype=int)

    k = _safe_n_pcs(
        np.vstack([make_features_mean_over_time(obs_go), make_features_mean_over_time(obs_nogo)]),
        DEFAULT_N_PCS,
    )
    model = build_decoder_pipeline(n_pcs=k)
    cv = make_cv(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    obs_auc_pre, _ = _safe_cv_auc(model, X_pre, y, cv)
    obs_auc_post, _ = _safe_cv_auc(model, X_post, y, cv)
    obs_delta = (
        obs_auc_post - obs_auc_pre
        if np.isfinite(obs_auc_pre) and np.isfinite(obs_auc_post)
        else np.nan
    )

    rng = np.random.default_rng(random_state)
    null_pre = np.empty(n_shuffle, dtype=float)
    null_post = np.empty(n_shuffle, dtype=float)
    null_delta = np.empty(n_shuffle, dtype=float)

    for i in range(n_shuffle):
        y_perm = rng.permutation(y)
        s_pre, _ = _safe_cv_auc(model, X_pre, y_perm, cv)
        s_post, _ = _safe_cv_auc(model, X_post, y_perm, cv)

        if np.isnan(s_pre) or np.isnan(s_post):
            null_pre[i] = np.nan
            null_post[i] = np.nan
            null_delta[i] = np.nan
        else:
            null_pre[i] = s_pre
            null_post[i] = s_post
            null_delta[i] = s_post - s_pre

    out = {
        "obs_auc_pre": float(obs_auc_pre) if np.isfinite(obs_auc_pre) else np.nan,
        "obs_auc_post": float(obs_auc_post) if np.isfinite(obs_auc_post) else np.nan,
        "obs_delta_auc": float(obs_delta) if np.isfinite(obs_delta) else np.nan,
        "p_pre": _two_sided_upper_pvalue(null_pre, obs_auc_pre),
        "p_post": _two_sided_upper_pvalue(null_post, obs_auc_post),
        "p_delta": _two_sided_upper_pvalue(null_delta, obs_delta),
        "n_shuffle": int(n_shuffle),
        "pre_bin_start": int(slices.pre_slice.start),
        "pre_bin_stop_exclusive": int(slices.pre_slice.stop),
        "post_bin_start": int(slices.post_slice.start),
        "post_bin_stop_exclusive": int(slices.post_slice.stop),
    }
    out.update(_summarize_null(null_pre, "null_pre"))
    out.update(_summarize_null(null_post, "null_post"))
    out.update(_summarize_null(null_delta, "null_delta"))
    return out


def _window_feature_matrix(tensor: np.ndarray, start_bin: int, stop_bin: int) -> np.ndarray:
    if stop_bin <= start_bin:
        raise ValueError("stop_bin must be greater than start_bin")
    return make_features_mean_over_time(tensor, time_slice=slice(int(start_bin), int(stop_bin)))


def time_resolved_shuffle_null(
    obs_go: np.ndarray,
    obs_nogo: np.ndarray,
    windows: pd.DataFrame,
    n_shuffle: int = 500,
    n_splits: int = 3,
    n_repeats: int = 1,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Faster time-resolved null model.

    Important:
    - this is intentionally lighter than the main decoding pipeline,
      because the full repeated-CV null across all windows is too expensive.
    - observed AUC and null AUC are computed with the same CV scheme here,
      so p-values remain internally consistent within this procedure.
    """
    required = {"start_bin", "stop_bin", "center_time_s"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"windows missing required columns: {sorted(missing)}")

    XA_full = make_features_mean_over_time(obs_go)
    XB_full = make_features_mean_over_time(obs_nogo)
    y = np.array([1] * len(XA_full) + [0] * len(XB_full), dtype=int)

    k = _safe_n_pcs(np.vstack([XA_full, XB_full]), DEFAULT_N_PCS)
    model = build_decoder_pipeline(n_pcs=k)

    # lighter CV only for time-null
    cv = make_cv(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    rng = np.random.default_rng(random_state)

    rows: List[Dict[str, float]] = []
    windows_sorted = (
        windows.sort_values(["start_bin", "stop_bin", "center_time_s"])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    for _, w in windows_sorted.iterrows():
        start_bin = int(w["start_bin"])
        stop_bin = int(w["stop_bin"])
        center = float(w["center_time_s"])

        X_go = _window_feature_matrix(obs_go, start_bin, stop_bin)
        X_nogo = _window_feature_matrix(obs_nogo, start_bin, stop_bin)
        X = np.vstack([X_go, X_nogo])

        obs_auc, _ = _safe_cv_auc(model, X, y, cv)

        null_auc = np.empty(n_shuffle, dtype=float)
        for i in range(n_shuffle):
            y_perm = rng.permutation(y)
            s, _ = _safe_cv_auc(model, X, y_perm, cv)
            null_auc[i] = s

        rows.append(
            {
                "start_bin": start_bin,
                "stop_bin": stop_bin,
                "center_time_s": center,
                "obs_auc": float(obs_auc) if np.isfinite(obs_auc) else np.nan,
                "null_auc_mean": float(np.nanmean(null_auc)),
                "null_auc_ci95_low": float(np.nanpercentile(null_auc, 2.5)),
                "null_auc_ci95_high": float(np.nanpercentile(null_auc, 97.5)),
                "p_auc": _two_sided_upper_pvalue(null_auc, obs_auc),
                "n_shuffle": int(n_shuffle),
                "cv_n_splits": int(n_splits),
                "cv_n_repeats": int(n_repeats),
            }
        )

    return pd.DataFrame(rows)


def compute_divergence_onset_from_pvalues(
    center_times: np.ndarray,
    p_values: np.ndarray,
    effect_values: np.ndarray,
    alpha: float,
    min_consecutive: int,
) -> float:
    if len(center_times) != len(p_values) or len(center_times) != len(effect_values):
        raise ValueError("center_times, p_values and effect_values must have equal length")

    order = np.argsort(center_times)
    t = np.asarray(center_times)[order]
    p = np.asarray(p_values)[order]
    eff = np.asarray(effect_values)[order]

    post_mask = t >= 0.0
    if post_mask.sum() == 0:
        return np.nan

    run = 0
    post_t = t[post_mask]
    post_p = p[post_mask]
    post_eff = eff[post_mask]

    for i, (ti, pi, ei) in enumerate(zip(post_t, post_p, post_eff)):
        if np.isfinite(pi) and pi <= alpha and np.isfinite(ei) and ei > 0.5:
            run += 1
            if run >= min_consecutive:
                return float(post_t[i - min_consecutive + 1])
        else:
            run = 0

    return np.nan


def anova_marginalize(X: np.ndarray) -> Dict[str, np.ndarray]:
    mu = X.mean(axis=(1, 2, 3), keepdims=True)
    mu_t = X.mean(axis=(1, 2), keepdims=True)
    mu_d = X.mean(axis=(2, 3), keepdims=True)
    mu_o = X.mean(axis=(1, 3), keepdims=True)
    mu_dt = X.mean(axis=2, keepdims=True)
    mu_ot = X.mean(axis=1, keepdims=True)
    mu_do = X.mean(axis=3, keepdims=True)
    ones = np.ones_like(X)

    A_t = (mu_t - mu) * ones
    A_d = (mu_d - mu) * ones
    A_o = (mu_o - mu) * ones
    A_dt = (mu_dt - mu_t - mu_d + mu) * ones
    A_ot = (mu_ot - mu_t - mu_o + mu) * ones
    A_do = (mu_do - mu_d - mu_o + mu) * ones
    A_dot = X - mu_dt - mu_ot - mu_do + mu_t + mu_d + mu_o - mu

    return {"t": A_t, "d": A_d, "o": A_o, "dt": A_dt, "ot": A_ot, "do": A_do, "dot": A_dot}


def variance_fractions(X: np.ndarray) -> Dict[str, float]:
    mu = X.mean(axis=(1, 2, 3), keepdims=True)
    X_c = X - mu
    total_var = float(np.sum(X_c ** 2))
    if total_var == 0.0:
        out = {k: np.nan for k in ("t", "d", "o", "dt", "ot", "do", "dot")}
        out["total_var"] = 0.0
        return out

    comps = anova_marginalize(X)
    out: Dict[str, float] = {}
    for label, A in comps.items():
        out[label] = float(np.sum(A ** 2) / total_var)
    out["total_var"] = total_var
    return out


def build_mean_tensor_decision_object(
    base_dir: str,
    area: str,
    session: str,
    event: str,
    interval: Tuple[float, float],
    objects: Sequence[str] = ("Object1", "Object2", "Object3"),
    binsize: float = DEFAULT_BINSIZE,
    gauss_smooth: Optional[float] = None,
    min_neurons: int = 3,
) -> Optional[np.ndarray]:
    means: List[List[np.ndarray]] = []
    contexts_conditions = [("Context2", "Condition1"), ("Context2", "Condition2")]

    for ctx, cond in contexts_conditions:
        row: List[np.ndarray] = []
        for obj in objects:
            T = build_tensor(
                base_dir,
                area,
                session,
                ctx,
                cond,
                obj,
                event,
                interval,
                binsize=binsize,
                gauss_smooth=gauss_smooth,
                min_neurons=min_neurons,
            )
            if T is None:
                return None
            row.append(T.mean(axis=0))
        means.append(row)

    n_neurons = means[0][0].shape[1]
    for d_row in means:
        for m in d_row:
            if m.shape[1] != n_neurons:
                return None

    X = np.stack([np.stack([m.T for m in d_row], axis=1) for d_row in means], axis=1)
    return X


def cross_event_decode(
    tensorA_go: np.ndarray,
    tensorA_nogo: np.ndarray,
    tensorB_go: np.ndarray,
    tensorB_nogo: np.ndarray,
    n_folds: int = 5,
    n_repeats: int = DEFAULT_N_REPEATS,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_perm: int = 0,
) -> Dict[str, float]:
    n_go = min(tensorA_go.shape[0], tensorB_go.shape[0])
    n_nogo = min(tensorA_nogo.shape[0], tensorB_nogo.shape[0])

    truncated = int(
        tensorA_go.shape[0] != tensorB_go.shape[0]
        or tensorA_nogo.shape[0] != tensorB_nogo.shape[0]
    )

    Xfa = np.vstack(
        [
            make_features_mean_over_time(tensorA_go[:n_go]),
            make_features_mean_over_time(tensorA_nogo[:n_nogo]),
        ]
    )
    Xfb = np.vstack(
        [
            make_features_mean_over_time(tensorB_go[:n_go]),
            make_features_mean_over_time(tensorB_nogo[:n_nogo]),
        ]
    )
    y = np.array([1] * n_go + [0] * n_nogo, dtype=int)

    k = _safe_n_pcs(Xfa, DEFAULT_N_PCS)
    rng = np.random.default_rng(random_state)

    def _run_once(y_labels: np.ndarray) -> Dict[str, List[float]]:
        out = {kname: [] for kname in ["A_to_B", "B_to_A", "within_A", "within_B"]}
        for _ in range(n_repeats):
            seed = int(rng.integers(0, 2**31 - 1))
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for tr_idx, te_idx in skf.split(Xfa, y_labels):
                y_train = y_labels[tr_idx]
                y_test = y_labels[te_idx]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                m_ab = build_decoder_pipeline(n_pcs=k)
                m_ab.fit(Xfa[tr_idx], y_train)
                out["A_to_B"].append(
                    float(roc_auc_score(y_test, m_ab.predict_proba(Xfb[te_idx])[:, 1]))
                )

                m_ba = build_decoder_pipeline(n_pcs=k)
                m_ba.fit(Xfb[tr_idx], y_train)
                out["B_to_A"].append(
                    float(roc_auc_score(y_test, m_ba.predict_proba(Xfa[te_idx])[:, 1]))
                )

                m_aa = build_decoder_pipeline(n_pcs=k)
                m_aa.fit(Xfa[tr_idx], y_train)
                out["within_A"].append(
                    float(roc_auc_score(y_test, m_aa.predict_proba(Xfa[te_idx])[:, 1]))
                )

                m_bb = build_decoder_pipeline(n_pcs=k)
                m_bb.fit(Xfb[tr_idx], y_train)
                out["within_B"].append(
                    float(roc_auc_score(y_test, m_bb.predict_proba(Xfb[te_idx])[:, 1]))
                )

        return out

    obs = _run_once(y)

    result = {
        "auc_A_to_B": float(np.mean(obs["A_to_B"])) if len(obs["A_to_B"]) else np.nan,
        "auc_B_to_A": float(np.mean(obs["B_to_A"])) if len(obs["B_to_A"]) else np.nan,
        "auc_within_A": float(np.mean(obs["within_A"])) if len(obs["within_A"]) else np.nan,
        "auc_within_B": float(np.mean(obs["within_B"])) if len(obs["within_B"]) else np.nan,
        "auc_A_to_B_std": float(np.std(obs["A_to_B"], ddof=1)) if len(obs["A_to_B"]) > 1 else 0.0,
        "auc_B_to_A_std": float(np.std(obs["B_to_A"], ddof=1)) if len(obs["B_to_A"]) > 1 else 0.0,
        "auc_within_A_std": float(np.std(obs["within_A"], ddof=1)) if len(obs["within_A"]) > 1 else 0.0,
        "auc_within_B_std": float(np.std(obs["within_B"], ddof=1)) if len(obs["within_B"]) > 1 else 0.0,
        "n_folds_run": int(len(obs["A_to_B"])),
        "n_trials_go_shared": int(n_go),
        "n_trials_nogo_shared": int(n_nogo),
        "truncated_to_shared_min": truncated,
    }

    if n_perm > 0:
        null_A_to_B = np.empty(n_perm, dtype=float)
        null_B_to_A = np.empty(n_perm, dtype=float)

        for i in range(n_perm):
            y_perm = rng.permutation(y)
            perm = _run_once(y_perm)
            null_A_to_B[i] = float(np.mean(perm["A_to_B"])) if len(perm["A_to_B"]) else np.nan
            null_B_to_A[i] = float(np.mean(perm["B_to_A"])) if len(perm["B_to_A"]) else np.nan

        result.update(
            {
                "p_A_to_B": _two_sided_upper_pvalue(null_A_to_B, result["auc_A_to_B"]),
                "p_B_to_A": _two_sided_upper_pvalue(null_B_to_A, result["auc_B_to_A"]),
                "null_A_to_B_mean": float(np.nanmean(null_A_to_B)),
                "null_B_to_A_mean": float(np.nanmean(null_B_to_A)),
                "n_perm": int(n_perm),
            }
        )
    else:
        result.update(
            {
                "p_A_to_B": np.nan,
                "p_B_to_A": np.nan,
                "null_A_to_B_mean": np.nan,
                "null_B_to_A_mean": np.nan,
                "n_perm": 0,
            }
        )

    return result