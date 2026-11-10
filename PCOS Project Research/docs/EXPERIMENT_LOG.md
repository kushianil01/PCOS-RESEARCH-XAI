# Experiment Log (what was actually run, in order)

Date of work: 2026-10-17. Machine: Windows 11, Python 3.13.15 (uv-managed venv at `~/venvs/pcos`; created outside OneDrive because OneDrive blocked package-file removal inside the project folder on the first attempt).

| # | Experiment / action | Config | Outcome | Decision taken |
|---|---|---|---|---|
| P1 | Candidate search and download (Kaggle, Figshare, Zenodo, UCI, NHANES codebooks) | see `docs/DATASET_SELECTION_REPORT.md` | Kerala PCOS usable; Figshare file has no label; NHANES RHQ cycles 2015-16, 2017-18, 2021-23 have no PCOS item; UCI CDC diabetes zip URL returned 404 (not downloaded, empty folder removed) | Kerala selected, approved by user with conditions |
| P2 | Exploratory audit and feasibility probe (`experiments/phase1/`) | untuned default LR/RF, 5x5 CV on **all** 541 rows | Non-invasive AUC ~0.88 (LR), ultrasound-only ~0.89, lab-only ~0.65-0.71, all ~0.94-0.96 | Screening only; **not reported as results**. Showed non-invasive signal is symptom-driven, motivating the circularity analysis |
| 01 | Ingest, validate, clean, EDA (`experiments/01`) | raw zip SHA-256 asserted | 541 rows, 7 logged cleaning actions, 19 quality flags, no rows removed | Cycle coding 2/4/5 recoded as inferred; BMI re-derived |
| 02 | Frozen split + feature redundancy (train only) | LR C=1, RF leaf 3; 5x5 CV | No larger set beat BMI set by the pre-declared rule | Primary inputs fixed (10 model features) |
| 03a | Baselines + tuning + selection (train only) | grids in `src/pcos/modeling.py`; 5x5 CV | RF best (0.880); LR within 1 corrected SE (0.876) | LR selected by pre-declared rule |
| 03b | Held-out test, one shot | threshold 0.241 from train OOF | AUC 0.895, sens 0.861, spec 0.712 | Reported as is. The test numbers were visible to me from this point on; I changed no model, feature, threshold or selection rule afterwards |
| 03c | **Re-run of step 03** | added `transformed_train_means` to model metadata (needed for exact contributions in the app) | Logistic-regression test probabilities bit-identical; RF differs by 3e-16; model file re-serialised, new SHA-256 recorded | None; documented here for transparency |
| 04 | Nested-CV ablation, lab/ultrasound comparison, cycle-length sensitivity | outer 5x3 (seed 43), inner 5-fold, small grids | See RESULTS_SUMMARY section 6; run took roughly 30 minutes with other jobs sharing the CPU | None to the model; findings feed limitations |
| 05 | Label circularity | proxy criteria defined in script header | Proxy >= 2 criteria matches label 83.9% | Limitation text strengthened |
| 06 | Explainability, subgroup, errors, robustness | 1000 bootstraps, 200 permutations | First run warned that SHAP subsampled the background to 100 rows; **fixed to the full 432-row background and re-run**. Values changed in the 3rd decimal only | None |
| 07 | Ablation figures | n/a | fig15, fig16 | None |
| T | pytest | `tests/` | First run stopped at 1 failure because a result file did not exist yet (step 04 was still running); non-UI tests then passed (65). Second full run: 1 failure from a test selecting "No" in the cycle selectbox, which only offers Regular/Irregular (test bug, fixed). Final run **75 passed** | Test bug fixed, no application change |

## Failed or abandoned items
- First attempt to create the venv inside the OneDrive folder failed (access denied); moved.
- A shell heredoc containing documentation text failed to parse; the files were created with the file-writer tool instead (no data effect).
- UCI CDC Diabetes zip: HTTP 404, abandoned (fallback dataset not needed).
- No experiment was discarded for poor results. Not run: SVM (omitted by design), external validation (no data), subgroup fairness beyond age/BMI (no data).
