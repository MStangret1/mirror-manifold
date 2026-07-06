# Neural Manifold Analysis of Mirror Neuron Dynamics

Analysis code for a bachelor thesis studying decision-related (Go vs No-Go)
population dynamics in monkey frontal and parietal cortex during observation.

## Description

Using single-unit recordings from macaque areas **AIP**, **F5**, and **F6**, this
project asks how observed Go/No-Go decisions are represented in the geometry of
neural population activity. The pipeline builds per-session population tensors,
applies dimensionality reduction (PCA / demixed-PCA-style marginalization) and
cross-validated linear decoding, and quantifies **when** and **where** Go and
No-Go trajectories diverge. Analyses are run per area × event (focusing on
**Event1** and **Event3**), with permutation and shuffle-null significance
testing, bootstrap stability, mixed-effects modelling, and cross-object /
cross-event generalization. All headline numbers are aggregated into
[`results/`](results) and cross-checked in
[`NUMERIC_CONSISTENCY_REPORT.md`](NUMERIC_CONSISTENCY_REPORT.md).

## Repository structure

```
mirror-manifold/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── run_all.py                    # conservative pipeline driver
├── _compute_numeric_summary.py   # rebuilds results/repo_numeric_summary.csv
├── FIGURE_PIPELINE_MAP.md        # script -> input CSV -> output figure map
├── NUMERIC_CONSISTENCY_REPORT.md # key numbers, recomputed from results/
├── src/                          # reusable functions (I/O, decoding, dPCA, paths)
├── pipeline/                     # numbered analysis scripts (00–11)
│   ├── 00_inventory.py
│   ├── 01_batch_session_object_analysis.py
│   ├── 02_batch_time_resolved.py
│   ├── 03_object_generalization.py
│   ├── 04_meta_summary.py
│   ├── 05_mixed_effects.py
│   ├── 06_dpca_analysis.py
│   ├── 07_shuffle_null_model.py
│   ├── 08_divergence_onset.py
│   ├── 09_cross_event_generalization.py
│   ├── 10_neuron_count_sensitivity.py
│   └── 11_supplementary_stats.py
├── figures/                      # figure-generation scripts (figures/fig_*.py)
├── results/                      # small derived CSV result tables (committed)
└── data/
    └── README.md                 # data instructions only (raw data NOT committed)
```

## Data availability

The raw neural recordings are **not** included in this repository and are **not
redistributed**. This project uses the public dataset:

> Tili et al. 2025, *"Mirror Neurons in Monkey Frontal and Parietal Areas."*

Download it from the original public source and place it under
`data/raw/HDF5/` (or point `DATA_ROOT` at an existing copy). Full instructions,
including the expected folder layout and file-naming convention, are in
[`data/README.md`](data/README.md). In short:

```
data/raw/HDF5/Spikes/    # per-area spike files
data/raw/HDF5/Events/    # per-area event files
```

or set an environment variable:

```bash
export DATA_ROOT=/absolute/path/to/HDF5      # PowerShell: $env:DATA_ROOT = "..."
```

## Installation

Requires **Python 3.13** (tested). Create an environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`dPCA` is optional; if it is not installed, the dPCA step falls back to an
internal demixing routine (and records `dpca_available=False`). The dPCA
variance-share tables shipped under `results/` were generated with the package
absent (fallback demixing, `dpca_available=False` for all rows). Treat them as
fallback marginalized-variance outputs, not package-based dPCA results, unless
you rerun step 06 (`pipeline/06_dpca_analysis.py`) with `dPCA` installed.

## Reproducing the analysis

`run_all.py` is the recommended entry point — it puts `src/` on `PYTHONPATH` and
sets `DATA_ROOT` for you. If you call a script **directly**, add `src/` to the
path first (`export PYTHONPATH=src` / PowerShell `$env:PYTHONPATH="src"`), and
run from the repo root.

**Without raw data** (from the shipped result tables) — regenerate summaries and
figures:

```bash
export PYTHONPATH=src                        # PowerShell: $env:PYTHONPATH="src"
python pipeline/04_meta_summary.py           # aggregates → results/meta_summary_*.csv
python figures/fig_main_result.py            # reads results/, writes figures/generated/
```

**With raw data** — full pipeline from HDF5:

```bash
python run_all.py --data-root /absolute/path/to/HDF5
# or, step by step (PYTHONPATH=src set as above):
python pipeline/00_inventory.py
python pipeline/01_batch_session_object_analysis.py --base-dir /absolute/path/to/HDF5
...
```

`run_all.py --dry-run` lists the steps without running them, and skips any step
whose raw data or script file is missing. See
[`FIGURE_PIPELINE_MAP.md`](FIGURE_PIPELINE_MAP.md) for which script produces
which figure/table.

## Outputs / results

Derived tables live in [`results/`](results), including:

- `session_object_event_results.csv` — per session × object × event decoding & geometry (204 analysis units)
- `time_resolved_results.csv` — time-resolved decoding AUC
- `object_generalization_results.csv` — cross-object generalization
- `meta_summary_by_area.csv`, `..._by_event.csv`, `..._by_area_event.csv` — aggregates
- `final_analysis/` — shuffle-null, divergence onset, mixed effects, dPCA variance, cross-event generalization
- `repo_numeric_summary.csv` — the key thesis numbers, recomputed from the above

A human-readable digest of the key numbers is in
[`NUMERIC_CONSISTENCY_REPORT.md`](NUMERIC_CONSISTENCY_REPORT.md). Generated
figures are written to `figures/generated/` (git-ignored).

## Citation

Repository: https://github.com/MStangret1/mirror-manifold

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff), and cite
the Tili et al. (2025) dataset separately.

## License

Source code is released under the [MIT License](LICENSE). The license covers the
code only — **not** the neural dataset or any third-party publications, which
retain their own terms.
