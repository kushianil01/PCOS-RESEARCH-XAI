# Automated Test Report

Command: `.venv/Scripts/python -m pytest --junitxml=results/test_report.xml` (raw: `results/test_report.xml`). Final run on 2026-10-17: **75 passed, 0 failed, 0 skipped** (11 s).

| File | Tests | Covers |
|---|---|---|
| `tests/test_validation.py` | 47 | valid input; each of the 11 fields missing; None/NaN as missing; wrong types (strings, lists, bool for numbers, non-0/1 for yes/no); out-of-range age/height/weight incl. inf and negative; legitimate zeros accepted and not treated as missing; unknown/invasive fields (`lh`, `amh`) rejected; multiple errors reported together; implausible BMI; out-of-training-range warning; `predict` refusing invalid input |
| `tests/test_model_consistency.py` | 11 | model loads and SHA-256 matches metadata; feature lists identical across config, metadata and decision file and exclude lab/ultrasound/excluded variables; processed data and split unchanged since training; split disjoint and stratified; **app inference path reproduces the 109 saved test probabilities (1e-9)** (rows with a missing input excluded, as the app refuses to impute); decisions equal saved threshold; **headline test metrics recomputed from `test_predictions.csv`**; labels in predictions match data; contributions equal `shap.LinearExplainer` and sum to the model log-odds; input-dict order irrelevant; output contract |
| `tests/test_data_pipeline.py` | 7 | raw archive hash; schema valid and violations detected; cleaning keeps all rows and logs coercions; target/primary values untouched; processed file equals fresh cleaning; few missing primary values |
| `tests/test_app.py` | 10 | all six pages render without exceptions; empty form blocked with listed missing fields; valid submission shows score and threshold; out-of-range value rejected; no laboratory/ultrasound/other excluded input widgets exist |

## Outstanding issues
- No API layer exists (not implemented by design), hence no API tests.
- UI tests are headless smoke tests (Streamlit AppTest); visual layout was checked manually once in a browser (overview and prediction form), not exhaustively.
- Tests depend on generated artefacts; run the experiments first (`REPRODUCIBILITY.md`).
- Two earlier failing runs during development are described in `docs/EXPERIMENT_LOG.md`; both were fixed before the final run.
