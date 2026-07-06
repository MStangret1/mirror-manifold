from __future__ import annotations

import argparse
import os
import pandas as pd

from core_utils import ensure_dir, summarize_results
from analysis_paths import infer_default_paths


def main():
    defaults = infer_default_paths(__file__)
    ap = argparse.ArgumentParser(description="Aggregate session-object-event results into meta summaries.")
    ap.add_argument("--results-csv", default=str(defaults.session_results_csv))
    ap.add_argument("--out-dir", default=str(defaults.results_dir))
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.results_csv)

    summary_area_event = summarize_results(df, ["area", "event"])
    summary_event = summarize_results(df, ["event"])
    summary_area = summarize_results(df, ["area"])

    summary_area_event.to_csv(os.path.join(args.out_dir, "meta_summary_by_area_event.csv"), index=False)
    summary_event.to_csv(os.path.join(args.out_dir, "meta_summary_by_event.csv"), index=False)
    summary_area.to_csv(os.path.join(args.out_dir, "meta_summary_by_area.csv"), index=False)
    print("Saved meta summaries.")


if __name__ == "__main__":
    main()
