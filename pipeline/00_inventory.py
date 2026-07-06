import os
import re
import glob
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

# ====== Paths (machine-independent) ======
# Raw HDF5 root: DATA_ROOT env var, else <repo>/data/raw/HDF5
_REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = os.environ.get("DATA_ROOT", str(_REPO_ROOT / "data" / "raw" / "HDF5"))
OUT_DIR = str(_REPO_ROOT / "results")
# =========================================

AREAS = ["AIP", "F5", "F6"]

# Filename patterns: ..._Context#_Condition#_Object#_Event#.h5 and ..._Spk_*.h5
re_event = re.compile(
    r'^(?P<session>Monkey[A-Z]_Session\d+)_'
    r'(?P<context>Context\d+)_'
    r'(?P<condition>Condition\d+)_'
    r'(?P<object>Object\d+)_'
    r'(?P<event>Event\d+)\.h5$'
)

re_spk = re.compile(
    r'^(?P<session>Monkey[A-Z]_Session\d+)_Spk_(?P<unit>.+)\.h5$'
)

def scan_events(base_dir):
    rows = []
    for area in AREAS:
        events_dir = os.path.join(base_dir, "Events", area)
        if not os.path.isdir(events_dir):
            continue
        for fp in glob.glob(os.path.join(events_dir, "*.h5")):
            bn = os.path.basename(fp)
            m = re_event.match(bn)
            if m:
                d = m.groupdict()
                d["area"] = area
                d["path"] = os.path.relpath(fp, base_dir)
                rows.append(d)
            else:
                # event file with unexpected name
                rows.append({"area": area, "session": None, "context": None, "condition": None,
                             "object": None, "event": None, "path": os.path.relpath(fp, base_dir)})
    return pd.DataFrame(rows)

def scan_spikes(base_dir):
    rows = []
    for area in AREAS:
        spk_dir = os.path.join(base_dir, "Spikes", area)
        if not os.path.isdir(spk_dir):
            continue
        for fp in glob.glob(os.path.join(spk_dir, "*.h5")):
            bn = os.path.basename(fp)
            m = re_spk.match(bn)
            if m:
                d = m.groupdict()
                d["area"] = area
                d["path"] = os.path.relpath(fp, base_dir)
                rows.append(d)
            else:
                rows.append({"area": area, "session": None, "unit": None, "path": os.path.relpath(fp, base_dir)})
    return pd.DataFrame(rows)

events_df = scan_events(BASE_DIR)
spikes_df = scan_spikes(BASE_DIR)

print("Events parsed rows:", len(events_df), " | unmatched:", int(events_df["session"].isna().sum()))
print("Spikes parsed rows:", len(spikes_df), " | unmatched:", int(spikes_df["session"].isna().sum()))


# Drop unparseable rows before tallying.
events_ok = events_df.dropna(subset=["session","context","condition","object","event"]).copy()
spikes_ok = spikes_df.dropna(subset=["session"]).copy()

# neuron counts: area+session
neurons = (spikes_ok.groupby(["area","session"])
           .agg(n_neurons=("path","count"))
           .reset_index()
           .sort_values(["area","session"]))

# event counts: area+session+context+condition+object
events_counts = (events_ok.groupby(["area","session","context","condition","object"])
                 .agg(n_files=("path","count"),
                      events_list=("event", lambda x: sorted(set(x))))
                 .reset_index())

print("\nNeuron counts (first 15):")
print(neurons.head(15).to_string(index=False))

print("\nEvent combinations (first 15):")
print(events_counts.head(15).to_string(index=False))


# Required event sets per condition, used to score completeness.
GO_REQUIRED = {"Event1","Event2","Event3","Event4","Event5","Event6"}
NOGO_REQUIRED = {"Event1","Event2","Event3","Event6"}

def completeness(required_set, events_list):
    s = set(events_list)
    missing = sorted(list(required_set - s))
    ok = (len(missing) == 0)
    return ok, missing

events_counts["events_set"] = events_counts["events_list"].apply(lambda lst: set(lst))

# Score completeness per combination.
oks = []
missings = []
for _, row in events_counts.iterrows():
    req = GO_REQUIRED if row["condition"] == "Condition1" else (NOGO_REQUIRED if row["condition"] == "Condition2" else set())
    ok, missing = completeness(req, row["events_list"])
    oks.append(ok)
    missings.append(",".join(missing))
events_counts["is_complete"] = oks
events_counts["missing_events"] = missings

events_counts = events_counts.merge(neurons, on=["area","session"], how="left")
events_counts["n_neurons"] = events_counts["n_neurons"].fillna(0).astype(int)

complete_summary = (events_counts.groupby(["area","session","context","condition"])
                    .agg(n_objects=("object","nunique"),
                         n_complete=("is_complete","sum"),
                         n_total=("is_complete","count"),
                         n_neurons=("n_neurons","max"))
                    .reset_index()
                    .sort_values(["area","session","context","condition"]))

print("\nComplete summary (first 30 rows):")
print(complete_summary.head(30).to_string(index=False))


# Rank sessions by completeness; restrict to Context1 (EXE).
exe = complete_summary[complete_summary["context"]=="Context1"].copy()

# Rank by No-Go (Condition2) completeness.
rank_nogo = (exe[exe["condition"]=="Condition2"]
             .assign(frac_complete=lambda d: d["n_complete"]/d["n_total"])
             .sort_values(["frac_complete","n_neurons","n_objects"], ascending=[False, False, False]))

print("\nTop sessions for EXE No-Go (Condition2) completeness:")
print(rank_nogo.head(30).to_string(index=False))

# Same for Go (Condition1).
rank_go = (exe[exe["condition"]=="Condition1"]
           .assign(frac_complete=lambda d: d["n_complete"]/d["n_total"])
           .sort_values(["frac_complete","n_neurons","n_objects"], ascending=[False, False, False]))

print("\nTop sessions for EXE Go (Condition1) completeness:")
print(rank_go.head(30).to_string(index=False))

os.makedirs(OUT_DIR, exist_ok=True)

events_df.to_csv(os.path.join(OUT_DIR, "inventory_events_raw.csv"), index=False)
spikes_df.to_csv(os.path.join(OUT_DIR, "inventory_spikes_raw.csv"), index=False)
neurons.to_csv(os.path.join(OUT_DIR, "inventory_neurons_by_session.csv"), index=False)
events_counts.to_csv(os.path.join(OUT_DIR, "inventory_event_combos_with_completeness.csv"), index=False)
complete_summary.to_csv(os.path.join(OUT_DIR, "inventory_completeness_summary.csv"), index=False)

print("\nSaved CSVs to:", OUT_DIR)
