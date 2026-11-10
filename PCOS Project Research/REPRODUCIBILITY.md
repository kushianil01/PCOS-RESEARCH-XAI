# Reproducibility

## Environment (as run)
Windows 11, Python 3.13.15, packages pinned in `requirements.txt` (pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1, shap 0.54.1, statsmodels 0.15.0, streamlit 1.66.0, pytest 9.1.1, matplotlib 3.11.2, joblib 1.6.0, openpyxl 3.1.5). Exact versions used for the saved model are in `models/model_metadata.json`.

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e .
```
Place the virtual environment outside cloud-synced folders.

## Dataset acquisition
The repository keeps the original download unchanged at `data/raw/kerala_kaggle/pcos.zip` and an extracted copy in `data/raw/kerala_kaggle/extracted/`. To re-acquire (CC BY-NC-SA 4.0; verify the current license on the Kaggle page first):
```bash
mkdir -p data/raw/kerala_kaggle/extracted
curl -L -o data/raw/kerala_kaggle/pcos.zip https://www.kaggle.com/api/v1/datasets/download/prasoonkottarathil/polycystic-ovary-syndrome-pcos
unzip data/raw/kerala_kaggle/pcos.zip -d data/raw/kerala_kaggle/extracted
```
Expected SHA-256 of the zip: `380ce7ff504cb8aa89349a251c64841a09aadbfe410627df94e6a62e67a23891` (step 01 aborts on a mismatch). If Kaggle changes its download policy, authenticate with the Kaggle CLI instead. Other raw files from the dataset search are listed with hashes in `data/documentation/MANIFEST_raw_sha256.txt` (not needed to reproduce the study).

## Run
```bash
.venv/Scripts/python experiments/run_all.py          # about 35-60 min; step 04 dominates (use --skip-ablation to omit 04 and 07)
.venv/Scripts/python -m pytest --junitxml=results/test_report.xml
.venv/Scripts/python -m streamlit run app/streamlit_app.py
```
Steps individually: `experiments/01_ingest_and_quality.py`, `02_feature_redundancy.py`, `03_train_evaluate.py`, `04_ablation_and_comparison.py`, `05_label_circularity.py`, `06_explain_and_errors.py`, `07_ablation_figures.py` (order matters; each writes a log to `results/logs/`).

## Seeds and configuration
Split `random_state=42`, 20% test, stratified on the label (`results/metrics/split_definition.json` stores the row IDs). CV seed 42 (5 folds x 5 repeats); nested CV outer seed 43 (5 x 3), inner seed 42; bootstrap seed 42 (2000 resamples for test CIs, 1000 for coefficients); all estimators `random_state=42`. Constants live in `src/pcos/config.py`.

## Expected outputs
- `data/processed/kerala_pcos_clean.csv`
- `models/pcos_primary_model.joblib`, `models/model_metadata.json`
- `results/metrics/`: `final_evaluation.json`, `feature_set_decision.json`, `ablation_comparison_nested_cv.csv`, `label_circularity.json`, `explainability_summary.json`, subgroup/error/robustness CSVs, cleaning and quality logs, split definition
- `results/predictions/test_predictions.csv`, `train_oof_predictions.csv`
- `results/figures/fig01..fig16` (PNG 300 dpi + PDF)
- `results/logs/*.log`, `results/test_report.xml`

## What should and should not match exactly
Logistic-regression test probabilities were bit-identical across two runs. Random-forest/gradient-boosting probabilities may differ in the 15th decimal place across runs (thread summation order). Bootstrap and permutation results are seed-fixed. Different package versions or operating systems may change last-digit results; metrics in `docs/RESULTS_SUMMARY.md` were produced with the versions listed above. Step 03 overwrites the model and its metadata; steps 05 and 06 and the tests depend on them, so rerun in order.

## Verification
`tests/test_model_consistency.py` checks that the application path reproduces the saved test predictions to 1e-9 and recomputes every headline test metric from `results/predictions/test_predictions.csv`.
