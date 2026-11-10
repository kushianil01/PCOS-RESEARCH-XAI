# Research Evidence Map

Each important result -> the file holding the numbers -> the script that produced it. Preserved inputs: raw data hash (`data/documentation/MANIFEST_raw_sha256.txt`), split IDs (`results/metrics/split_definition.json`), configurations (`src/pcos/config.py`, grids in `src/pcos/modeling.py`, chosen parameters in `models/model_metadata.json`), predictions (`results/predictions/`), logs (`results/logs/`), package versions (`requirements.txt`, `models/model_metadata.json`).

| Result | Value (see RESULTS_SUMMARY for context) | Evidence file | Script |
|---|---|---|---|
| Dataset size, prevalence, missingness | 541 rows, 177 positive, 3 missing cells | `results/metrics/data_quality_summary.json` | `experiments/01_ingest_and_quality.py` |
| Cleaning actions, flagged values | 7 actions, 19 flags | `results/metrics/cleaning_log.csv`, `quality_flags.csv` | `01` |
| Cycle-code distribution | 2:390, 4:150, 5:1 | `data_quality_summary.json` | `01` |
| Split | 432 / 109 (141 / 36 positive) | `results/metrics/split_definition.json` | `02` (`src/pcos/split.py`) |
| Redundancy study, VIF, input-set decision | BMI set; all paired CIs include 0 | `results/metrics/feature_redundancy_cv.csv`, `feature_redundancy_paired_diffs.csv`, `feature_set_decision.json` | `02` |
| CV model comparison | LR 0.876, tree 0.823, RF 0.880, GBM 0.872 | `results/metrics/final_evaluation.json` (`cv_summary_train`), `cv_grid_results.csv` | `03` |
| Final model and threshold | logistic regression, C=0.1, thr 0.241 | `final_evaluation.json`, `models/model_metadata.json` | `03` |
| Held-out test metrics and CIs | AUC 0.895 [0.825, 0.957], sens 0.861, spec 0.712 | `final_evaluation.json` (`test_metrics`), `results/predictions/test_predictions.csv` | `03`; recomputed by `tests/test_model_consistency.py` |
| Baselines (dummy, rules) | R4 sens 0.778 / spec 0.863 | `final_evaluation.json` | `03` |
| Calibration | slope 1.52 (test), ECE 0.081 | `final_evaluation.json` (`calibration_*`), `fig08` | `03` |
| Ablation / lab / ultrasound / cycle-length comparison | see section 6 | `results/metrics/ablation_comparison_nested_cv.csv` | `04` |
| Label circularity | proxy >= 2 criteria: accuracy 0.839 | `results/metrics/label_circularity.json` | `05` |
| Odds ratios and bootstrap stability | skin darkening 2.40, ... | `results/metrics/explain_logreg_coefficients.csv` | `06` |
| Importance (permutation, SHAP), rank agreement | top-2 share 32% | `explain_importance_table.csv`, `explainability_summary.json` | `06` |
| Paired test-AUC differences | LR - symptom count +0.025 [-0.003, 0.059] | `explainability_summary.json` | `06` |
| Subgroup performance | age/BMI AUCs | `results/metrics/subgroup_evaluation.csv` | `06` |
| Error analysis incl. withheld follicle counts | FN 74% >= 12 follicles | `error_analysis.csv`, `explainability_summary.json` | `06` |
| Robustness | 10% flipped answers: AUC 0.864 | `results/metrics/robustness_test.csv` | `06` |
| Test-suite outcome | 75 passed | `results/test_report.xml`, `docs/TEST_REPORT.md` | `pytest` |
| Dataset license / provenance | CC BY-NC-SA 4.0 (Kaggle API metadata, 2026-10-17) | `docs/DATASET_CARD.md`, `docs/DATASET_SELECTION_REPORT.md` | Phase 1 manual verification (no script output stored for the Kaggle API call) |
| Phase 1 feasibility probe (preliminary only) | non-invasive AUC ~0.88 on all data | `results/audit/feasibility_probe_output.txt` | `experiments/phase1/feasibility_probe.py` |
