# 08_mixed_effects.py
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import warnings

from analysis_paths import infer_default_paths, resolve_path
from core_utils import ensure_dir

warnings.filterwarnings("ignore", category=ConvergenceWarning)

OPTIMIZERS = ["lbfgs", "bfgs", "powell", "cg", "nm"]


def logit_clip(x: pd.Series, eps: float = 1e-6) -> pd.Series:
    z = x.clip(eps, 1.0 - eps)
    return np.log(z / (1.0 - z))


def fit_mixedlm_robust(
    formula: str,
    data: pd.DataFrame,
    group_col: str,
    reml: bool,
) -> Tuple[object | None, str | None, str | None]:
    md = smf.mixedlm(formula, data, groups=data[group_col])
    last_error = None
    for method in OPTIMIZERS:
        try:
            res = md.fit(reml=reml, method=method, maxiter=2000, disp=False)
            return res, method, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    return None, None, last_error


def fit_fallback_ols(formula: str, data: pd.DataFrame, group_col: str):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data[group_col]})


def coef_table_from_result(
    result,
    label: str,
    formula: str,
    model_type: str,
    optimizer: str | None,
) -> pd.DataFrame:
    if hasattr(result, "fe_params"):
        params = result.fe_params
        conf = result.conf_int().loc[params.index]
        pvals = result.pvalues.loc[params.index]
        ses = result.bse.loc[params.index]
    else:
        params = result.params
        conf = result.conf_int().loc[params.index]
        pvals = result.pvalues.loc[params.index]
        ses = result.bse.loc[params.index]
    rows = []
    for term in params.index:
        rows.append(
            {
                "model": label,
                "formula": formula,
                "model_type": model_type,
                "optimizer": optimizer,
                "term": term,
                "coef": float(params.loc[term]),
                "se": float(ses.loc[term]),
                "p_value": float(pvals.loc[term]),
                "ci_low": float(conf.loc[term, 0]),
                "ci_high": float(conf.loc[term, 1]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    defaults = infer_default_paths(__file__)
    ap = argparse.ArgumentParser(
        description="High-quality mixed-effects summary for session-level decoding outcomes."
    )
    ap.add_argument("--results-csv", default=str(defaults.session_results_csv))
    ap.add_argument("--out-dir", default=str(defaults.final_dir / "08_mixed_effects"))
    args = ap.parse_args()

    results_csv = resolve_path(args.results_csv, defaults.session_results_csv)
    out_dir = resolve_path(args.out_dir, defaults.final_dir / "08_mixed_effects")
    ensure_dir(str(out_dir))

    df = pd.read_csv(results_csv).copy()
    df["delta_auc"] = df["dec_obs_go_nogo_auc_post"] - df["dec_obs_go_nogo_auc_pre"]
    df["log_n_neurons"] = np.log1p(df["n_neurons"]) if "n_neurons" in df.columns else np.nan
    df["event"] = pd.Categorical(df["event"], categories=["Event1", "Event3"])
    df["area"] = pd.Categorical(df["area"], categories=["AIP", "F5", "F6"])
    df["auc_post_logit"] = logit_clip(df["dec_obs_go_nogo_auc_post"])
    df["auc_full_logit"] = logit_clip(df["dec_obs_go_nogo_auc_full"])

    model_specs = [
        ("delta_auc", "delta_auc"),
        ("auc_post_logit", "dec_obs_go_nogo_auc_post"),
        ("auc_full_logit", "dec_obs_go_nogo_auc_full"),
    ]

    coef_tables: List[pd.DataFrame] = []
    fit_rows: List[Dict] = []
    re_rows: List[Dict] = []
    compare_rows: List[Dict] = []

    for outcome, raw_source in model_specs:
        full_formula = (
            f"{outcome} ~ C(event, Treatment('Event1')) * C(area, Treatment('AIP')) + log_n_neurons"
        )
        reduced_formula = f"{outcome} ~ C(area, Treatment('AIP')) + log_n_neurons"

        full_ml, opt_full, err_full = fit_mixedlm_robust(
            full_formula, df, group_col="session", reml=False
        )
        red_ml, opt_red, err_red = fit_mixedlm_robust(
            reduced_formula, df, group_col="session", reml=False
        )
        full_reml, opt_full_reml, err_full_reml = fit_mixedlm_robust(
            full_formula, df, group_col="session", reml=True
        )

        if full_reml is not None:
            coef_tables.append(
                coef_table_from_result(
                    full_reml,
                    f"{outcome}_full",
                    full_formula,
                    "MixedLM",
                    opt_full_reml,
                )
            )
            if hasattr(full_reml, "random_effects"):
                for sess, v in full_reml.random_effects.items():
                    if isinstance(v, dict):
                        val = float(next(iter(v.values())))
                    else:
                        arr = np.asarray(v).ravel()
                        val = float(arr[0]) if len(arr) else np.nan
                    re_rows.append(
                        {"model": f"{outcome}_full", "session": sess, "random_intercept": val}
                    )
            fit_rows.append(
                {
                    "model": f"{outcome}_full",
                    "outcome": outcome,
                    "source_metric": raw_source,
                    "formula": full_formula,
                    "model_type": "MixedLM",
                    "optimizer": opt_full_reml,
                    "converged": bool(getattr(full_reml, "converged", False)),
                    "llf": float(getattr(full_reml, "llf", np.nan)),
                    "aic": float(getattr(full_reml, "aic", np.nan)),
                    "bic": float(getattr(full_reml, "bic", np.nan)),
                    "n_obs": int(getattr(full_reml, "nobs", len(df))),
                    "n_groups": int(df["session"].nunique()),
                    "error": None,
                }
            )
        else:
            ols = fit_fallback_ols(full_formula, df, group_col="session")
            coef_tables.append(
                coef_table_from_result(ols, f"{outcome}_full", full_formula, "OLS_clustered", None)
            )
            fit_rows.append(
                {
                    "model": f"{outcome}_full",
                    "outcome": outcome,
                    "source_metric": raw_source,
                    "formula": full_formula,
                    "model_type": "OLS_clustered",
                    "optimizer": None,
                    "converged": True,
                    "llf": float(ols.llf),
                    "aic": float(ols.aic),
                    "bic": float(ols.bic),
                    "n_obs": int(ols.nobs),
                    "n_groups": int(df["session"].nunique()),
                    "error": err_full_reml,
                }
            )

        if full_ml is not None and red_ml is not None:
            lr_stat = 2.0 * (float(full_ml.llf) - float(red_ml.llf))
            df_diff = int(len(full_ml.fe_params) - len(red_ml.fe_params))
            p_lr = float(chi2.sf(lr_stat, df=df_diff)) if df_diff > 0 else np.nan
        else:
            lr_stat = np.nan
            df_diff = np.nan
            p_lr = np.nan
        compare_rows.append(
            {
                "outcome": outcome,
                "full_formula": full_formula,
                "reduced_formula": reduced_formula,
                "full_optimizer": opt_full,
                "reduced_optimizer": opt_red,
                "full_error": err_full,
                "reduced_error": err_red,
                "lr_stat": lr_stat,
                "df_diff": df_diff,
                "p_event_family": p_lr,
            }
        )

    pd.concat(coef_tables, ignore_index=True).to_csv(
        out_dir / "mixed_effects_coefficients.csv", index=False
    )
    pd.DataFrame(fit_rows).to_csv(out_dir / "mixed_effects_model_fit.csv", index=False)
    pd.DataFrame(compare_rows).to_csv(
        out_dir / "mixed_effects_model_comparisons.csv", index=False
    )
    pd.DataFrame(re_rows).to_csv(out_dir / "mixed_effects_random_effects.csv", index=False)
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()