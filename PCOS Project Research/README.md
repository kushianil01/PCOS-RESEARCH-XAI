# Explainable Non-Invasive PCOS Label-Concordance Screening Support (research prototype)

**Research question.** Using a single documented Kerala PCOS cohort, how well can a reproducible, explainable model built only on non-invasive, self-reportable inputs reproduce the dataset's PCOS label, how much does it lose against laboratory/ultrasound-based models, and how stable, calibrated and shortcut-prone are its predictions and explanations?

> Research prototype. Not a diagnostic or treatment tool, not clinically validated, no external validation. The target is the dataset's own PCOS label (label concordance). Data license CC BY-NC-SA 4.0: non-commercial use only.

## What is in the project
- Reproducible pipeline: hash-verified ingestion, schema validation, conservative cleaning, frozen stratified split, feature-redundancy study, tuning/selection on training data only, one-shot held-out evaluation, nested-CV ablation, label-circularity analysis, explainability, subgroup/error/robustness analyses.
- Streamlit application (`app/streamlit_app.py`) with overview, data-quality dashboard, evaluation dashboard (CV vs held-out clearly separated), prediction form with strict validation, explanations, and responsible-use section. It loads the exact saved model (SHA-256 checked) and reads all metrics from result files.
- 75 automated tests (`tests/`), including evaluation-to-deployment consistency.
- Documentation in `docs/` and this folder (see index below).

## Headline results (details and sources: `docs/RESULTS_SUMMARY.md`)
Final model: logistic regression, 10 non-invasive features. Held-out test (n=109): AUC 0.895 [0.825, 0.957], sensitivity 0.861, specificity 0.712 at a threshold fixed from training data. Simple symptom counts perform close to it; the label agrees strongly with a crude Rotterdam-style proxy, so high concordance partly reflects how the label was made.

## Architecture
`docs/SYSTEM_ARCHITECTURE.md` (Mermaid source `docs/architecture_diagram.mmd`).

## Install
Python 3.11+ (tested on 3.13.15).
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```
(Windows PowerShell: `.venv\Scripts\python`; macOS/Linux: `.venv/bin/python`. Keep the virtual environment outside cloud-synced folders such as OneDrive.)

## Run the application
```bash
.venv/Scripts/python -m streamlit run app/streamlit_app.py
```

## Re-run the experiments and tests
```bash
.venv/Scripts/python experiments/run_all.py
.venv/Scripts/python -m pytest
```
Details: `REPRODUCIBILITY.md`. The raw data must be placed as described there.

## Documentation index
`REPRODUCIBILITY.md`, `docs/DATASET_SELECTION_REPORT.md`, `docs/DATASET_CARD.md`, `docs/METHODOLOGY.md`, `docs/SYSTEM_ARCHITECTURE.md`, `docs/RESULTS_SUMMARY.md`, `docs/LIMITATIONS_AND_ETHICS.md`, `docs/EXPERIMENT_LOG.md`, `docs/FIGURE_INDEX.md`, `docs/EVIDENCE_MAP.md`, `docs/TEST_REPORT.md`, `docs/QUALITY_AUDIT.md`.

## Known limitations (short list; full list in `docs/LIMITATIONS_AND_ETHICS.md`)
Single regional cohort; undocumented label and several variable definitions (cycle coding inferred); label circularity; small test set; no external validation; no ethics documentation for the source data; non-commercial license; explanations are associations, not causes.
