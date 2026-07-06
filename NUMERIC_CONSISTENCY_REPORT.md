# Numeric consistency report

All numbers below are **recomputed directly from the CSV result files** by
`_compute_numeric_summary.py` and stored tidily in
[`results/repo_numeric_summary.csv`](results/repo_numeric_summary.csv).
Nothing here is hand-entered. Rounding is to 3 decimals for display.

Source tree: `go-nogo/reworked_results/` (→ `results/` in the clean repo).

---

## A. Analysis units and neuron counts

| Quantity | Value | Source |
|---|---|---|
| Analysis units (session × object × event) | **204** | `session_object_event_results.csv` |
| — AIP | 24 | " |
| — F5 | 102 | " |
| — F6 | 78 | " |
| dPCA units (session × event) | 68 total | `dpca_variance_per_unit.csv` |
| — with dPCA **package** available | **0** (fallback demixing used) | " |
| Total neurons (Σ over sessions, inventory) | 355 (AIP 86, F5 106, F6 163) | `inventory_neurons_by_session.csv` |

> ⚠ **dPCA note:** `dpca_available = False` for all 68 rows — the `dPCA` package
> was not installed when this ran, so variance shares come from the internal
> fallback, not the published dPCA implementation. State this in the thesis, or
> re-run step 06 with `dPCA` installed.

---

## B. OBS Go vs OBS No-Go decoding AUC (overall)

| Window | Mean AUC | n |
|---|---|---|
| full | **0.605** | 204 |
| pre | **0.535** | 204 |
| post | **0.645** | 204 |

For reference, EXE vs OBS AUC (full) = **0.946** (n=204) — the execution/
observation distinction decodes near-ceiling, as expected.

## C. Mean AUC pre vs post, by event

| Event | Pre | Post | n |
|---|---|---|---|
| Event1 | 0.513 | 0.538 | 102 |
| Event3 | 0.557 | **0.751** | 102 |

The post-event Go/No-Go signal is carried almost entirely by **Event3**.

## D. Mean AUC pre vs post, by area × event

| Area | Event | Pre | Post | n |
|---|---|---|---|---|
| AIP | Event1 | 0.533 | 0.475 | 12 |
| AIP | Event3 | 0.546 | 0.731 | 12 |
| F5 | Event1 | 0.515 | 0.517 | 51 |
| F5 | Event3 | 0.555 | **0.770** | 51 |
| F6 | Event1 | 0.503 | 0.586 | 39 |
| F6 | Event3 | 0.562 | 0.733 | 39 |

## E. Permutation significance (OBS Go/No-Go AUC)

| Group | Significant | n tested | fraction |
|---|---|---|---|
| p < 0.05 overall | **47** | 204 | 0.230 |
| p < 0.01 overall | 29 | 204 | 0.142 |
| p < 0.05, AIP | 6 | 24 | 0.250 |
| p < 0.05, F5 | 26 | 102 | 0.255 |
| p < 0.05, F6 | 15 | 78 | 0.192 |

## F. Time-resolved peak AUC (across-unit mean curve)

| Area | Event | Peak AUC | Peak time (s) |
|---|---|---|---|
| AIP | Event1 | 0.564 | 0.15 |
| AIP | Event3 | **0.795** | 0.55 |
| F5 | Event1 | 0.551 | 0.69 |
| F5 | Event3 | 0.763 | 0.55 |
| F6 | Event1 | 0.575 | 0.51 |
| F6 | Event3 | 0.718 | 0.51 |

## G. Divergence onset (Go vs No-Go), from null-based onset

| Area | Event | Median onset (ms) | n units with onset / n |
|---|---|---|---|
| AIP | Event1 | n/a | 0 / 12 |
| AIP | Event3 | 450 | 2 / 12 |
| F5 | Event1 | 10* | 1 / 51 |
| F5 | Event3 | **340** | 18 / 51 |
| F6 | Event1 | 270* | 1 / 39 |
| F6 | Event3 | 440 | 6 / 39 |

\* Event1 onsets rest on 0–1 units and are **not reliable** — treat as "no
robust divergence in Event1." The interpretable onset is **Event3 in F5 ≈ 340 ms**
(the only cell with a healthy count, 18/51 units).

## H. Cross-object generalization (OBS Go vs No-Go)

| Event | Same-object AUC | Cross-object AUC |
|---|---|---|
| overall | 0.605 (n=204) | 0.605 (n=408) |
| Event1 | 0.533 | 0.519 |
| Event3 | **0.676** | **0.692** |

Cross-object ≈ same-object → the Go/No-Go code **generalizes across objects**
(the decision signal is not object-specific), strongest in Event3.

## I. Cross-event generalization (Event1 ↔ Event3), by area

| Area | Within E1 | Within E3 | E1→E3 | E3→E1 | GTI |
|---|---|---|---|---|---|
| AIP | 0.500 | 0.683 | 0.537 | 0.496 | 0.017 |
| F5 | 0.522 | 0.689 | 0.530 | 0.504 | 0.017 |
| F6 | 0.553 | 0.658 | 0.524 | 0.539 | 0.031 |

Within-Event3 decoding is strong, but **transfer across events is near chance**
(E1↔E3 ≈ 0.50–0.54) → the Event1 and Event3 decision codes are largely distinct.
(The `normalized_transfer` column is numerically degenerate — values ~1e13 — and
is **excluded**; do not report it.)

## J. dPCA marginalized variance shares (mean per area × event)

Fractions of explained variance by marginalization (t=time, d=decision,
o=object). Fallback-computed (see §A note).

| Area | Event | time | decision | object | dec×time | obj×time | dec×obj | dec×obj×time |
|---|---|---|---|---|---|---|---|---|
| AIP | E1 | 0.361 | 0.004 | 0.006 | 0.125 | 0.244 | 0.010 | 0.250 |
| AIP | E3 | 0.196 | 0.011 | 0.012 | 0.171 | 0.296 | 0.016 | 0.299 |
| F5 | E1 | 0.276 | 0.007 | 0.012 | 0.139 | 0.278 | 0.015 | 0.273 |
| F5 | E3 | 0.268 | 0.039 | 0.010 | 0.180 | 0.252 | 0.012 | 0.239 |
| F6 | E1 | 0.325 | 0.008 | 0.009 | 0.130 | 0.256 | 0.011 | 0.263 |
| F6 | E3 | 0.248 | 0.023 | 0.007 | 0.154 | 0.280 | 0.007 | 0.282 |

Pure **decision** variance is small (≤4%); decision information lives mostly in
the **decision×time** interaction — consistent with a transient, dynamic signal.

dPCA units per area × event: **AIP 4, F5 17, F6 13** (both events).

---

## K. Comparison with thesis text

**No thesis document (`.tex`, `.md`, `.docx`, `.pdf` draft) was found in the
project.** The only prose file is `go-nogo/reworked_pipeline/README_analysis_plan.md`
(a plan, not results). Therefore every claim below is **Cannot verify** until you
provide the thesis file — at which point this table can be filled in
automatically.

| Claim in thesis | Source file | Value in thesis | Value recomputed from CSV | Status |
|---|---|---|---|---|
| _(no thesis file found)_ | — | — | see §A–§J | **Cannot verify** |

**How to complete this step:** drop the thesis `.tex`/`.md`/`.docx` into the
project and re-run; I will extract reported AUCs, p-values, %s, neuron/unit
counts, Event1/Event3 claims, dPCA unit counts and object-generalization values,
and diff them against §A–§J with OK / Minor-rounding / Mismatch / Cannot-verify
statuses. **No thesis text will be edited** — only suggested corrections reported.

---

## Watch-outs to carry into the writing (do not overclaim)

1. **Event1 shows no reliable Go/No-Go signal** (post AUC ≈ 0.51–0.59; divergence
   onset rests on ≤1 unit). Frame the positive result as **Event3-specific**.
2. **dPCA package was absent** (fallback used) — either re-run with `dPCA`
   installed or say so explicitly.
3. **AIP is small** (24 units / 4 dPCA units) — area comparisons involving AIP
   are underpowered; report CIs.
4. **`normalized_transfer` and Event1 divergence onsets are numerically
   unstable** — exclude from the thesis.
