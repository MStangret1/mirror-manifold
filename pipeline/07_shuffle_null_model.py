from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from analysis_paths import infer_default_paths, resolve_path
from core_utils import (
    DEFAULT_BINSIZE,
    build_tensor,
    ensure_dir,
    iter_analysis_units,
    load_inventory,
)
from core_utils_ext import shuffle_pre_post_null, time_resolved_shuffle_null

UnitKey = Tuple[str, str, str, str]


def _unit_key(area: str, session: str, object_: str, event: str) -> UnitKey:
    return (str(area), str(session), str(object_), str(event))


def _keys_from_df(df: pd.DataFrame, cols=("area", "session", "object", "event")) -> Set[UnitKey]:
    if df is None or len(df) == 0:
        return set()
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    return set(zip(df[cols[0]].astype(str), df[cols[1]].astype(str), df[cols[2]].astype(str), df[cols[3]].astype(str)))


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _save_prepost(rows: List[Dict], out_dir: Path) -> None:
    if not rows:
        return
    df_rows = pd.DataFrame(rows).drop_duplicates(
        subset=["area", "session", "object", "event"],
        keep="last",
    )
    df_rows.to_csv(out_dir / "shuffle_null_prepost_per_unit.csv", index=False)

    summary = (
        df_rows.groupby(["area", "event"])[
            ["obs_auc_pre", "obs_auc_post", "obs_delta_auc", "p_delta", "p_post"]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(out_dir / "shuffle_null_prepost_summary.csv", index=False)


def _save_time_rows(time_rows: List[pd.DataFrame], out_dir: Path) -> None:
    if not time_rows:
        return
    df_time = pd.concat(time_rows, ignore_index=True).drop_duplicates(
        subset=["area", "session", "object", "event", "start_bin", "stop_bin", "center_time_s"],
        keep="last",
    )
    df_time.to_csv(out_dir / "shuffle_null_time_resolved.csv", index=False)


def main() -> None:
    defaults = infer_default_paths(__file__)
    ap = argparse.ArgumentParser(
        description="Resume-safe shuffle null model for pre/post and time-resolved Go/NoGo decoding."
    )
    ap.add_argument("--base-dir", default=str(defaults.base_dir))
    ap.add_argument("--go-nogo-dir", default=str(defaults.go_nogo_dir))
    ap.add_argument("--results-csv", default=str(defaults.session_results_csv))
    ap.add_argument(
        "--time-csv",
        default=str(defaults.time_resolved_csv),
        help="Time-resolved decoding CSV used to recover exact sliding windows.",
    )
    ap.add_argument("--out-dir", default=str(defaults.final_dir / "06_shuffle_null"))
    ap.add_argument("--n-shuffle", type=int, default=500)
    ap.add_argument("--min-neurons", type=int, default=3)
    ap.add_argument("--gauss-smooth", type=float, default=None)
    ap.add_argument("--binsize", type=float, default=DEFAULT_BINSIZE)

    ap.add_argument(
        "--skip-time-null",
        action="store_true",
        help="Only compute pre/post null, skip per-window null.",
    )
    ap.add_argument(
        "--time-null-only",
        action="store_true",
        help="Compute only time-resolved null; do not recompute pre/post null.",
    )
    ap.add_argument(
        "--force-recompute-time-null",
        action="store_true",
        help="Ignore existing time-null results and recompute all time-null units from scratch.",
    )
    ap.add_argument(
        "--force-recompute-prepost",
        action="store_true",
        help="Ignore existing pre/post results and recompute all pre/post units from scratch.",
    )

    args = ap.parse_args()

    if args.skip_time_null and args.time_null_only:
        raise ValueError("Cannot use --skip-time-null together with --time-null-only.")

    base_dir = resolve_path(args.base_dir, defaults.base_dir)
    go_nogo_dir = resolve_path(args.go_nogo_dir, defaults.go_nogo_dir)
    results_csv = resolve_path(args.results_csv, defaults.session_results_csv)
    time_csv = resolve_path(args.time_csv, defaults.time_resolved_csv)
    out_dir = resolve_path(args.out_dir, defaults.final_dir / "06_shuffle_null")
    ensure_dir(str(out_dir))

    prepost_path = out_dir / "shuffle_null_prepost_per_unit.csv"
    prepost_summary_path = out_dir / "shuffle_null_prepost_summary.csv"
    time_path = out_dir / "shuffle_null_time_resolved.csv"

    prev = pd.read_csv(results_csv)
    passed = _keys_from_df(prev)

    windows_df = None
    if not args.skip_time_null:
        if not Path(time_csv).exists():
            raise FileNotFoundError(f"time-csv not found: {time_csv}")
        windows_df = pd.read_csv(time_csv)

    # Expected unit universe = units that passed the main analysis
    inv = load_inventory(str(go_nogo_dir))
    all_units_raw = list(iter_analysis_units(inv["summary"], events=("Event1", "Event3")))
    all_units = [u for u in all_units_raw if _unit_key(u.area, u.session, u.object_, u.event) in passed]
    expected_keys = {_unit_key(u.area, u.session, u.object_, u.event) for u in all_units}

    # Load existing outputs for resume
    existing_prepost_df = None if args.force_recompute_prepost else _safe_read_csv(prepost_path)
    existing_time_df = None if args.force_recompute_time_null else _safe_read_csv(time_path)

    rows: List[Dict] = []
    if existing_prepost_df is not None and len(existing_prepost_df):
        rows = existing_prepost_df.to_dict(orient="records")

    time_rows: List[pd.DataFrame] = []
    if existing_time_df is not None and len(existing_time_df):
        time_rows = [existing_time_df]

    existing_prepost_keys = set() if args.force_recompute_prepost else _keys_from_df(existing_prepost_df) if existing_prepost_df is not None else set()
    existing_time_keys = set() if args.force_recompute_time_null else _keys_from_df(existing_time_df) if existing_time_df is not None else set()

    # Determine what remains
    if args.time_null_only:
        todo_prepost_keys = set()
    else:
        todo_prepost_keys = expected_keys - existing_prepost_keys

    if args.skip_time_null:
        todo_time_keys = set()
    else:
        todo_time_keys = expected_keys - existing_time_keys

    print(f"Expected eligible units: {len(expected_keys)}", flush=True)
    print(f"Existing pre/post units: {len(existing_prepost_keys)}", flush=True)
    print(f"Existing time-null units: {len(existing_time_keys)}", flush=True)
    print(f"Remaining pre/post units: {len(todo_prepost_keys)}", flush=True)
    print(f"Remaining time-null units: {len(todo_time_keys)}", flush=True)

    if not todo_prepost_keys and not todo_time_keys:
        print("Nothing to do. All requested units already computed.", flush=True)
        print(f"Saved outputs to {out_dir}", flush=True)
        return

    done_counter = 0
    total_counter = len(all_units)

    for i, unit in enumerate(all_units, start=1):
        key = _unit_key(unit.area, unit.session, unit.object_, unit.event)
        need_prepost = key in todo_prepost_keys
        need_time = key in todo_time_keys

        if not need_prepost and not need_time:
            continue

        print(
            f"[{i}/{total_counter}] START {unit.area} | {unit.session} | {unit.object_} | {unit.event}"
            f" | need_prepost={need_prepost} | need_time={need_time}",
            flush=True,
        )

        try:
            obs_go = build_tensor(
                str(base_dir),
                unit.area,
                unit.session,
                "Context2",
                "Condition1",
                unit.object_,
                unit.event,
                unit.interval,
                gauss_smooth=args.gauss_smooth,
                min_neurons=args.min_neurons,
            )
            obs_nogo = build_tensor(
                str(base_dir),
                unit.area,
                unit.session,
                "Context2",
                "Condition2",
                unit.object_,
                unit.event,
                unit.interval,
                gauss_smooth=args.gauss_smooth,
                min_neurons=args.min_neurons,
            )

            if obs_go is None or obs_nogo is None:
                print(
                    f"[{i}/{total_counter}] SKIP missing tensor for {unit.area} | {unit.session} | {unit.object_} | {unit.event}",
                    flush=True,
                )
                continue

            if need_prepost:
                res = shuffle_pre_post_null(
                    obs_go=obs_go,
                    obs_nogo=obs_nogo,
                    interval=unit.interval,
                    binsize=args.binsize,
                    n_shuffle=args.n_shuffle,
                )
                rows.append(
                    {
                        "area": unit.area,
                        "session": unit.session,
                        "object": unit.object_,
                        "event": unit.event,
                        **res,
                    }
                )
                _save_prepost(rows, out_dir)
                print(
                    f"[{i}/{total_counter}] DONE PREPOST {unit.area} | {unit.session} | {unit.object_} | {unit.event}"
                    f" | delta_auc={res['obs_delta_auc']:.4f} | p_delta={res['p_delta']:.4f}",
                    flush=True,
                )

            if need_time and windows_df is not None:
                unit_windows = (
                    windows_df.loc[
                        (windows_df["area"].astype(str) == str(unit.area))
                        & (windows_df["session"].astype(str) == str(unit.session))
                        & (windows_df["object"].astype(str) == str(unit.object_))
                        & (windows_df["event"].astype(str) == str(unit.event)),
                        ["start_bin", "stop_bin", "center_time_s"],
                    ]
                    .drop_duplicates()
                )

                if len(unit_windows) == 0:
                    print(
                        f"[{i}/{total_counter}] SKIP TIME-NULL no windows found for {unit.area} | {unit.session} | {unit.object_} | {unit.event}",
                        flush=True,
                    )
                else:
                    print(
                        f"[{i}/{total_counter}] TIME-NULL {unit.area} | {unit.session} | {unit.object_} | {unit.event}"
                        f" | n_windows={len(unit_windows)}",
                        flush=True,
                    )

                    time_df = time_resolved_shuffle_null(
                        obs_go=obs_go,
                        obs_nogo=obs_nogo,
                        windows=unit_windows,
                        n_shuffle=args.n_shuffle,
                    )
                    time_df.insert(0, "event", unit.event)
                    time_df.insert(0, "object", unit.object_)
                    time_df.insert(0, "session", unit.session)
                    time_df.insert(0, "area", unit.area)
                    time_rows.append(time_df)
                    _save_time_rows(time_rows, out_dir)

                    print(
                        f"[{i}/{total_counter}] DONE TIME-NULL {unit.area} | {unit.session} | {unit.object_} | {unit.event}",
                        flush=True,
                    )

            done_counter += 1

        except KeyboardInterrupt:
            print(
                f"[{i}/{total_counter}] INTERRUPTED by user at {unit.area} | {unit.session} | {unit.object_} | {unit.event}",
                flush=True,
            )
            print("Partial results were saved. You can rerun the same command to resume.", flush=True)
            raise

        except Exception as exc:
            print(
                f"[{i}/{total_counter}] ERROR {unit.area} | {unit.session} | {unit.object_} | {unit.event}"
                f" | {type(exc).__name__}: {exc}",
                flush=True,
            )
            continue

    # Final consistency report
    final_prepost_df = _safe_read_csv(prepost_path)
    final_time_df = _safe_read_csv(time_path)

    final_prepost_keys = _keys_from_df(final_prepost_df) if final_prepost_df is not None else set()
    final_time_keys = _keys_from_df(final_time_df) if final_time_df is not None else set()

    print("---- FINAL STATUS ----", flush=True)
    if not args.time_null_only:
        print(f"Pre/post completed: {len(final_prepost_keys)} / {len(expected_keys)}", flush=True)
        missing_pre = expected_keys - final_prepost_keys
        print(f"Pre/post missing: {len(missing_pre)}", flush=True)

    if not args.skip_time_null:
        print(f"Time-null completed: {len(final_time_keys)} / {len(expected_keys)}", flush=True)
        missing_time = expected_keys - final_time_keys
        print(f"Time-null missing: {len(missing_time)}", flush=True)

    print(f"Saved outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()