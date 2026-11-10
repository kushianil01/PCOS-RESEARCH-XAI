# Final Quality Audit

PASS = verified with evidence; FAIL = problem found; NOT VERIFIED = could not verify. Audit performed by the same agent that built the project (not independent of the author); evidence is listed so it can be re-checked.

## Dataset audit
| Item | Status | Evidence |
|---|---|---|
| Original source traceable | PASS (with caveat) | Kaggle URL, uploader, SHA-256 in `docs/DATASET_CARD.md`; **underlying hospital provenance NOT VERIFIED** (no descriptor) |
| License and permitted use documented | PASS | CC BY-NC-SA 4.0 from Kaggle API metadata 2026-10-17; restrictions in `docs/DATASET_CARD.md`. Uploader's right to license NOT VERIFIED |
| Target definition understood | FAIL (inherent) | Label construction undocumented; only inferred consistency with Rotterdam-style criteria (`label_circularity.json`). Documented as limitation |
| Preprocessing decisions justified | PASS | `docs/DATASET_CARD.md`, `src/pcos/data.py`, cleaning log, tests |
| Leakage risks investigated | PASS | Row-ID AUC 0.538; imputation/scaling inside CV pipelines; excluded variables; label-circularity analysis; follicle counts withheld |
| Limitations explicit | PASS | `docs/LIMITATIONS_AND_ETHICS.md` |
| Cycle-code and variable definitions | NOT VERIFIED | Cycle coding inferred from secondary papers; cycle length unusable; symptom definitions absent. Handled by documentation and exclusion |

## Experiment audit
| Item | Status | Evidence |
|---|---|---|
| Analysis reproducible | PASS (as run once) | Scripts, seeds, logs; LR test probabilities identical on rerun (`EXPERIMENT_LOG` 03c). Full clean-environment rerun on another machine NOT VERIFIED |
| Splitting strategy appropriate | PASS (limited) | Stratified random split; no group/time field exists, repeated patients cannot be ruled out |
| Final test set separate from selection | PASS | Steps 02 and 03a use `train` only (code review: `02_feature_redundancy.py`, `03_train_evaluate.py`); threshold from train OOF. Caveat: test numbers were visible to the author after 03b; no decision changed afterwards (log) |
| Metrics match saved predictions | PASS | `tests/test_model_consistency.py::test_reported_metrics_recompute_from_saved_predictions` |
| Baselines meaningful | PASS | Dummy and four pre-declared rules; R4 rule is competitive and reported |
| Statistical claims supported | PASS (limited) | Only bootstrap CIs and approximate corrected paired intervals; no significance claims; test set small |
| Results and configurations preserved | PASS | `results/`, `models/model_metadata.json` |
| External validation | NOT VERIFIED / not performed | No second labelled open dataset exists; not claimed |
| Calibration acceptable | FAIL (partial) | Slope 1.52 on test (under-dispersed probabilities); documented, not corrected |

## Explainability audit
| Item | Status | Evidence |
|---|---|---|
| Method appropriate | PASS | Exact linear SHAP + odds ratios for a linear model; permutation importance as model-agnostic check |
| Correlation/attribution limits considered | PASS | Grouped permutation, bootstrap rank instability, phi correlations (`RESULTS_SUMMARY` section 8) |
| Causal claims avoided | PASS | Wording in app, results and limitations |
| Conclusions proportional | PASS (reviewer-judged) | Claims limited to label concordance; small-sample caveats stated |
| Explanation stability | PASS (partial) | Sign stability high for main features; top-1 rank unstable, disclosed |

## Application audit
| Item | Status | Evidence |
|---|---|---|
| App loads the evaluated model | PASS | `pcos.predict.load_bundle` hash check; consistency test |
| Preprocessing identical in evaluation and inference | PASS | Same pipeline object; BMI formula test; 109-row reproduction |
| Tests pass | PASS | 75 passed (`docs/TEST_REPORT.md`) |
| Missing/invalid values handled | PASS | 47 validation tests; UI tests |
| Outputs communicated responsibly | PASS (reviewer-judged) | Banner on every page, threshold explanation, calibration context, extrapolation warnings. No user study of comprehension performed (NOT VERIFIED) |
| No invasive/lab inputs | PASS | Unknown fields rejected; UI test; feature-list test |

## Documentation audit
| Item | Status | Evidence |
|---|---|---|
| Setup instructions complete | PASS (as written) | `README.md`, `REPRODUCIBILITY.md`; followed only on the author's machine |
| Another person can reproduce | NOT VERIFIED | No independent replication performed |
| Dependencies documented | PASS | `requirements.txt`, `pyproject.toml` |
| License and source recorded | PASS | `docs/DATASET_CARD.md`, manifest |
| Technical claims supported by outputs | PASS | `docs/EVIDENCE_MAP.md`; numbers in RESULTS_SUMMARY were cross-checked against result files by hand once |
| Requested items not produced | NOTE | No FastAPI layer (by design); architecture diagram is Mermaid source plus embedded diagram in markdown, not a rendered image |
