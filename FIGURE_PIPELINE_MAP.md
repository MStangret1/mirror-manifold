# Figure / script map

Which figure script reads which result CSV, what image it writes, and the
likely thesis figure it corresponds to (inferred from filenames — **verify
against your thesis**, since no thesis document was found in the project to
confirm the mapping automatically).

All input CSVs live under `results/` (originally `go-nogo/reworked_results/`).
Generated images are written to `figures/generated/` in the clean repo.

| Figure script | Reads (CSV) | Writes (figure) | Likely thesis figure |
|---|---|---|---|
| `fig_main_result.py` | `session_object_event_results.csv`, `meta_summary_by_area_event.csv` | `fig_main_result.pdf/.png`, `fig_main_result_data.csv` | Main result: OBS Go vs No-Go decoding by area/event |
| `fig_time_resolved.py` | `time_resolved_results.csv`, `final_analysis/06_shuffle_null/shuffle_null_time_resolved.csv` | `fig_time_resolved.pdf/.png`, `fig_time_resolved.csv` | Time-resolved decoding AUC with null band |
| `fig_area_robustness.py` | `meta_summary_by_area_event.csv`, `neuron_count_sensitivity.csv`, `neuron_count_sensitivity_by_area.csv` | `fig_area_robustness.pdf/.png` + 3 derived CSVs | Robustness / neuron-count control |
| `fig_exe_obs_benchmark_v4.py` | `inventory_event_combos_with_completeness.csv`, `session_object_event_results.csv` | `fig_exe_obs_benchmark_v4.pdf/.png` | EXE vs OBS decoding benchmark |
| `fig_exe_obs_convergence_test.py` | `exe_obs_convergence_per_session.csv`, `exe_obs_convergence_summary.csv`, `inventory_event_combos_with_completeness.csv` | `fig_exe_obs_convergence_test.pdf/.png` | EXE/OBS convergence control |
| `13_single_trial_pca_scatter.py` | (raw tensors / PCA) | `single_trial_pca_scatter.png` | Single-trial PCA scatter |
| `15_thesis_figures.py` | multiple result CSVs | `fig1_main_result.png`, `fig2_time_resolved.png`, `fig3_prepost_event3.png`, `fig4_dpca_variance.png`, `fig5_trajectory_dist.png`, `fig6_exe_obs_scatter.png` | Thesis figure set 1–6 |
| `16_extra_figures.py` | `time_resolved_results.csv`, `shuffle_null_time_resolved.csv` | `fig2_extra1_null_overlay.png`, `figS1_per_object_time_resolved.png` | Supplementary S1 + null overlay |
| `17_great_plots.py` | `cross_event_generalization_summary.csv` & `_per_unit.csv`, `dpca_component_trajectories_long.csv`, `time_resolved_results.csv`, `shuffle_null_time_resolved.csv`, `session_object_event_results.csv` | `plotA_cross_event_generalization`, `plotB_decision_trajectory`, `plotC_null_overlay_significance`, `plotD_prepost_upgraded`, `plotE_reliability`, `plotF_divergence_cdf`, `plotG_3d_pca_trajectories` (.png/.svg) | "Great plots" panel set |

## Notes

- `17_great_plots.py` depends on `dpca_component_trajectories_long.csv` (~36 MB).
  That file is **git-ignored** by default (regenerable from `pipeline/09_dpca_analysis.py`).
  If you want plot B/G to work from a clean checkout, either regenerate it or
  remove that line from `.gitignore`.
- Superseded figure scripts found in the original project (kept out of the clean
  repo): `fig_exe_obs_benchmark_v2.py`, `fig_manifold_*.py` (old exploratory
  manifold renders under `go-nogo/` and `outputs/`), `fig_exe_obs_benchmark_v4`
  supersedes `v2/v3`.
