# Repository audit report

Audit of `bachelor_data_cogsci/` in preparation for a private → public GitHub
repository (`mirror-manifold`). Generated 2026-07-02.

**Scope of scan:** all `.py`, `.md`, `.ipynb`, `.txt`, `.yaml`, `.cff` files
(excluding the raw `Data HDF5/` tree).

---

## 1. Summary of findings

| Category | Result |
|---|---|
| Absolute Windows paths (`C:\Users\<USER>\...`) | **Found in ~40 files** — must be de-hardcoded |
| Personal username / folder (`<USER>`, `Documents\UW`) | Present inside those same absolute paths |
| Email addresses | **None found** in code/text |
| API keys / tokens / passwords / secrets | **None found** |
| Raw HDF5 paths hard-coded | Yes (in `00_inventory`, `analysis_paths.py`, several fig scripts) |
| Hard-coded output/data paths | Yes (see §2) |
| Temp-folder references | None found in code |
| Copyrighted material | **`literatura/` (10 PDFs of published papers)** — must NOT be published |
| Large binaries | See §5 |

**No credentials or personal contact data are present.** The only privacy issue
is the recurring absolute path `C:\Users\<USER>\Documents\UW\bachelor_data_cogsci\...`,
which leaks the local username and folder layout.

---

## 2. Files containing private / local paths

### 2a. Files that WILL go into the clean repo — **must be fixed**

| File (original) | Issue | Suggested replacement |
|---|---|---|
| `go-nogo/00_inventory` | `BASE_DIR`, `OUT_DIR` hard-coded to `C:\Users\<USER>\...` | `BASE_DIR = os.environ.get("DATA_ROOT", "data/raw/HDF5")`; write outputs relative to repo (`results/`) |
| `go-nogo/reworked_pipeline/analysis_paths.py` | `WINDOWS_PROJECT_DIR/BASE_DIR/RESULTS_DIR` hard-coded | Prefer `DATA_ROOT` env var, then script-relative fallback (see §3 patch) |
| `.../11_neuron_count_sensitivity.py` | `ROOT`, `INPUT_CSV`, `OUT_*` absolute | derive from `analysis_paths` / `--results-dir` |
| `.../fig_main_result.py` | `pd.read_csv(r"C:\Users\<USER>\...session_object_event_results.csv")` | `results_dir / "session_object_event_results.csv"` |
| `.../fig_time_resolved.py` | absolute `read_csv(...time_resolved_results.csv)` | `results_dir / "time_resolved_results.csv"` |
| `.../fig_area_robustness.py` | absolute `read_csv(...meta_summary_by_area_event.csv)` | `BASE_DIR / "meta_summary_by_area_event.csv"` (already uses relative for the others) |
| `.../fig_exe_obs_benchmark_v4.py` | `GO_NOGO_DIR = Path(r"C:\Users\<USER>\...")` | resolve via `analysis_paths` |
| `.../fig_exe_obs_convergence_test.py` | `GO_NOGO_DIR` absolute | resolve via `analysis_paths` |
| `go-nogo/reworked_pipeline/README_analysis_plan.md` | mentions the absolute path in prose | replace with `<repo>/` or `data/raw/HDF5/` |

### 2b. Files with the same issue but **excluded** from the clean repo

These are older / exploratory and are **not** proposed for the clean repo, so
their paths are harmless once excluded — listed for completeness:

`debug_trajectory_analysis.py`, `independent_test.py`,
`go-nogo/01_konfiguracja.py`, `..._v2.py`, `02_permutation_test_decoding.py`,
`03_bootstrap_stability.py`, `04_manifold_across_sessions.py`,
`08_time_resolved_decoding.py`, `go-nogo/fig_manifold_*.py` (10 files),
`go-nogo/Neuron_firing_rate_HDF5.py`, all of `listopad/*.py` and `*.ipynb`,
`Script HDF5/Script/CreateSingleRaster.py`, `wykresy_październik/import os.py`,
`literatura/2203.11874v2.md`, and the two files under `.ipynb_checkpoints/`.

---

## 3. Suggested standard replacement

Use this pattern anywhere a raw-data root is needed:

```python
import os
from pathlib import Path
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/raw/HDF5"))
```

or expose it as a CLI argument (the batch scripts 01–03/06/10 already do this):

```python
ap.add_argument("--data-root", "--base-dir", dest="base_dir",
                default=os.environ.get("DATA_ROOT", "data/raw/HDF5"))
```

For `analysis_paths.py`, the recommended patch replaces the Windows-first
default with an env-var-first, script-relative fallback:

```python
_ENV = os.environ.get("DATA_ROOT")
WINDOWS_BASE_DIR = Path(_ENV) if _ENV else (Path(__file__).resolve().parents[1] / "data" / "raw" / "HDF5")
```

Result CSVs should be written **relative to the repo** (`results/…`), never to an
absolute user path.

---

## 4. Files that are safe (no changes needed)

- All CSVs under `go-nogo/reworked_results/` and `go-nogo/inventory_*.csv`
  (numeric data only, no paths/PII).
- `go-nogo/reworked_pipeline/core_utils.py`, `core_utils_ext.py` — no absolute
  paths; explicit random seeds; pure functions.
- Batch scripts `01–04`, `06_*_FIXED`, `07`, `08`, `09`, `10`, `12` — take
  `--base-dir` and resolve defaults through `analysis_paths` (only the default
  needs the §3 patch).

---

## 5. Files that should NOT be committed

| Path | Reason | Handling |
|---|---|---|
| `Data HDF5/`, `Data HDF5.zip` (79 MB / 40 MB) | Raw dataset, not ours to redistribute | Exclude; `.gitignore` covers `*.h5`, `*.zip`, `data/raw/` |
| `Script HDF5.zip` (3.9 MB) | Zip of loader scripts | Exclude (`*.zip`) |
| `literatura/` (83 MB, 10 published-paper PDFs) | **Copyrighted** third-party papers | **Do not publish.** Keep local only |
| `listopad/analysis_results.pkl` (108 MB) | Large pickle intermediate | Exclude (`*.pkl`) |
| `listopad/firing_rates_all_sessions.npy/.pkl` | Large intermediates | Exclude (`*.npy`, `*.pkl`) |
| `listopad/data_overview.spydata` | Spyder session dump | Exclude (`*.spydata`) |
| `go-nogo/artifacts/*.npy` | Cached tensors (regenerable) | Exclude (`*.npy`) |
| `.ipynb_checkpoints/`, `__pycache__/`, `*.pyc` | Editor/interpreter cruft | Exclude |
| `.claude/settings.local.json` | Local tool state | Exclude (`.claude/`) |

## 5b. Suspicious / large files inside the proposed repo

| File | Size | Note |
|---|---|---|
| `results/.../09_dpca/dpca_component_trajectories_long.csv` | **36 MB** | Regenerable; git-ignored by default. Only `17_great_plots.py` needs it. |
| `results/.../07_divergence_onset/divergence_onset_timewise_qvalues.csv` | 1.7 MB | OK to commit (borderline) |
| `results/.../06_shuffle_null/shuffle_null_time_resolved.csv` | 1.6 MB | OK to commit |
| PNG/SVG figures (several 1–1.6 MB) | | Prefer keeping only in `figures/generated/` (git-ignored) and regenerate |

---

## 6. Duplicated / versioned files to resolve (pick one)

The project contains several parallel versions. Recommended canonical choice → keep:

- `06_shuffle_null_model.py` vs `06_shuffle_null_model_FIXED.py` → **keep FIXED**
- `core_utils.py` vs `core_utils_FIXED_v2.py` → confirm which is current (imports
  point at `core_utils`, so **keep `core_utils.py`**; verify `_FIXED_v2` isn't newer)
- `fig_exe_obs_benchmark_v2.py` vs `v4.py` → **keep v4**
- `11_neuron_count_sensitivity.py` vs `11_run_final_analysis.py` → different roles;
  keep both but renumber to avoid the duplicate `11_` prefix
- `01_konfiguracja.py` / `_v2.py`, `manifold_results_across_sessions(.csv/_v2.csv)` →
  old pipeline, **excluded** from clean repo

---

## 7. Pipeline consistency

Findings from inspecting the numbered scripts destined for `pipeline/`:

- **Runnable from repo root:** batch scripts use `argparse` and resolve defaults
  via `analysis_paths.py`. ✔ (after the §3 default patch)
- **Consistent folder paths:** they read/write through `analysis_paths` (results
  → `reworked_results/…` → clean-repo `results/…`). ✔ once the absolute defaults
  are removed.
- **Outputs to `results/`:** analysis scripts write to the results tree. ✔
  Figure scripts read from `results/` and write images. ✔ (fig scripts currently
  write next to results; retarget to `figures/generated/`).
- **No local Windows dependency at runtime:** only the *defaults* in
  `analysis_paths.py` / a few fig scripts hard-code the Windows path — patch per §3.
- **No accidental single-unit runs:** the clean batch scripts iterate **all**
  units via `iter_analysis_units(..., events=("Event1","Event3"))`. The
  single-unit tensors (`..._Session02_Object1_Event3.npy`) belong to *debug*
  scripts (`debug_trajectory_analysis.py`) that are **excluded**. ✔
- **No raw HDF5 required to commit:** raw files are git-ignored; summaries/figs
  reproduce from shipped CSVs. ✔
- **Random seeds explicit:** `core_utils.DEFAULT_RANDOM_STATE = 0`,
  `PCA(random_state=…)`, `RepeatedStratifiedKFold(random_state=…)`,
  `11_neuron_count_sensitivity.RNG_SEED = 20260411`, and figure jitter seeds are
  all fixed. ✔ (Minor: figures use `seed=hash(area) % 99`; `hash()` of a string
  is salted per-process unless `PYTHONHASHSEED` is set — cosmetic only, affects
  jitter positions, not results.)

**Action item:** set `PYTHONHASHSEED=0` if you want pixel-identical jitter across
runs, or replace `hash(area)` with a fixed lookup.

---

## 8. Proposed file mapping (OLD → NEW)

> Nothing below has been moved yet. **Copy** (not move) is recommended so the
> original working tree stays intact. Confirm before I proceed.

### src/ (reusable functions)
```
go-nogo/reworked_pipeline/core_utils.py          → src/core_utils.py
go-nogo/reworked_pipeline/core_utils_ext.py      → src/core_utils_ext.py
go-nogo/reworked_pipeline/analysis_paths.py      → src/analysis_paths.py   (+ DATA_ROOT patch)
go-nogo/reworked_pipeline/05_patch_safe_firing_rate.py → src/patch_safe_firing_rate.py
```

### pipeline/ (numbered analysis scripts)
```
go-nogo/00_inventory                             → pipeline/00_inventory.py  (+ .py, + DATA_ROOT patch)
reworked_pipeline/01_batch_session_object_analysis.py → pipeline/01_batch_session_object_analysis.py
reworked_pipeline/02_batch_time_resolved.py      → pipeline/02_batch_time_resolved.py
reworked_pipeline/03_object_generalization.py    → pipeline/03_object_generalization.py
reworked_pipeline/04_meta_summary.py             → pipeline/04_meta_summary.py
reworked_pipeline/08_mixed_effects.py            → pipeline/05_mixed_effects.py    (renumbered)
reworked_pipeline/09_dpca.py                     → pipeline/06_dpca_analysis.py    (renumbered)
reworked_pipeline/06_shuffle_null_model_FIXED.py → pipeline/07_shuffle_null_model.py
reworked_pipeline/07_divergence_onset.py         → pipeline/08_divergence_onset.py
reworked_pipeline/10_cross_event_generalization.py → pipeline/09_cross_event_generalization.py
reworked_pipeline/11_neuron_count_sensitivity.py → pipeline/10_neuron_count_sensitivity.py
reworked_pipeline/12_supplementary_stats.py      → pipeline/11_supplementary_stats.py
```
*(Numbering can instead be kept 1:1 with the originals if you prefer — say which.)*

### figures/
```
reworked_pipeline/fig_main_result.py             → figures/fig_main_result.py
reworked_pipeline/fig_time_resolved.py           → figures/fig_time_resolved.py
reworked_pipeline/fig_area_robustness.py         → figures/fig_area_robustness.py
reworked_pipeline/fig_exe_obs_benchmark_v4.py    → figures/fig_exe_obs_benchmark.py
reworked_pipeline/fig_exe_obs_convergence_test.py → figures/fig_exe_obs_convergence_test.py
reworked_pipeline/13_single_trial_pca_scatter.py → figures/fig_single_trial_pca_scatter.py
reworked_pipeline/15_thesis_figures.py           → figures/fig_thesis_set.py
reworked_pipeline/16_extra_figures.py            → figures/fig_supplementary.py
reworked_pipeline/17_great_plots.py              → figures/fig_great_plots.py
```

### results/ (small derived CSVs)
```
go-nogo/reworked_results/*.csv                   → results/*.csv
go-nogo/reworked_results/final_analysis/**/*.csv → results/final_analysis/**/*.csv
                                                    (except dpca_component_trajectories_long.csv → git-ignored)
go-nogo/inventory_*.csv                          → results/inventory/*.csv
```

### data/
```
(new)                                            → data/README.md   [created]
```

### Excluded from the clean repo (kept local only)
```
Data HDF5/, Data HDF5.zip, Script HDF5.zip, literatura/, listopad/, outputs/,
results/ (top-level old PNGs), wykresy_październik/, go-nogo/artifacts/,
go-nogo/fig_manifold_*.py, go-nogo/0X old numbered scripts, debug_*.py,
independent_test.py, .ipynb_checkpoints/, __pycache__/
```
