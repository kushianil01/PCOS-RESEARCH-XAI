# Methodology

## Research question (refined after the dataset audit and approved)
Using a single documented Kerala PCOS cohort, how well can a reproducible, explainable model built only on non-invasive, self-reportable inputs reproduce the dataset's PCOS label, how much does it lose against laboratory/ultrasound-based models, and how stable, calibrated and shortcut-prone are its predictions and explanations?

Scope conditions (from the approval): deployed system strictly non-invasive; target = existing label (label concordance, not diagnosis); symptom-ablation and label-circularity analyses mandatory; no external-validation, clinical, causal or generalisability claims.

## Dataset rationale
Kerala PCOS (541 records) is the only labelled open tabular PCOS dataset found that contains non-invasive variables (`docs/DATASET_SELECTION_REPORT.md`). Its weaknesses (undocumented label and codes, single region, license NC-SA) are carried into the limitations rather than hidden.

## Design
1. **Ingestion and cleaning** (`experiments/01`): archive hash check, schema validation, conservative cleaning, quality flags. No row deleted, no value changed to help results.
2. **Frozen split**: stratified 80/20 (`random_state=42`) -> 432 train (141 positive) / 109 test (36 positive). Row IDs saved. No group or time variable exists, so a stratified random split is the only option; this cannot protect against repeated patients across hospitals (undetectable).
3. **Feature redundancy** (`experiments/02`, training split only, rule declared before running): candidates BMI / height+weight / height+weight+BMI / +waist / +waist+hip, logistic regression and random forest with fixed hyper-parameters, 5x5 repeated stratified CV, paired fold differences with Nadeau-Bengio corrected variance, VIF, Spearman pairs. Move to a larger set only if AUC rises >= 0.01 for both models with a CI excluding 0.
4. **Baselines** (`experiments/03`): prior-probability dummy; four pre-declared transparent rules (irregular cycle; irregular AND androgen sign; irregular OR hair growth; >= 3 of 5 symptoms). The rule with the best balanced accuracy on the training split is the "simple-rule baseline".
5. **Models**: logistic regression (L2), decision tree, random forest, gradient boosting. SVM omitted: no calibrated probabilities without extra machinery and no interpretability benefit at n=432.
6. **Tuning and selection** (training split only): grid search with 5x5 repeated stratified CV, ROC-AUC. Final model = simplest family (logreg < tree < rf < gbm) within one Nadeau-Bengio-corrected standard error of the best family's mean CV AUC (rule declared before running).
7. **Pipelines**: median imputation + standardisation for continuous inputs, mode imputation for binary inputs; fitted inside each training fold; the same fitted pipeline is deployed.
8. **Operating threshold**: highest threshold with >= 85% sensitivity on training out-of-fold predictions (screening orientation). Fixed before touching test data.
9. **Held-out evaluation**: final model and all comparison models are scored on the test split once, after all decisions. Percentile bootstrap CIs (2000 resamples of test records; reflects test-sample variability, not training variability).
10. **Calibration**: calibration slope, calibration-in-the-large, 5-bin ECE, Brier, reliability plot, on training OOF and test. No recalibration applied.
11. **Ablation and comparison** (`experiments/04`): nested CV on all 541 records (outer 5x3, inner 5-fold grid, small grids for logreg and RF). Primary-set ablations; separate comparison sets with ultrasound, laboratory and combined inputs (never deployable); one sensitivity run with the ambiguous cycle-length variable.
12. **Label circularity** (`experiments/05`): crude Rotterdam-style proxy from available columns (our approximation), proxy-vs-label agreement, relation of the non-invasive score to withheld follicle counts, AUC within follicle strata.
13. **Explainability** (`experiments/06`): logistic odds ratios with 1000-bootstrap intervals and sign stability; permutation importance (AUC drop) on test (200 repeats) and via CV on train; grouped permutation for related inputs; exact linear SHAP (interventional, full training background); rank agreement between methods; paired bootstrap of test AUC differences (logreg vs rf, gbm, symptom-count score).
14. **Subgroup, error and robustness analyses**: age and BMI groups on pooled train-OOF + test scores (AUC only when >= 15 positives and >= 15 negatives); error profiles including withheld follicle counts; input-noise, answer-flip and missing-answer scenarios on the test set.

## Metrics and why
ROC-AUC (threshold-free ranking; primary selection metric), PR-AUC (prevalence 33%), sensitivity/specificity/precision/F1 at the fixed threshold (screening trade-off), Brier and calibration (probabilities are shown to users). Accuracy is reported but not used for selection.

## Statistical methods
Percentile bootstrap CIs; Nadeau-Bengio corrected paired comparisons across overlapping CV folds (approximate, used descriptively, no hypothesis-test claims); Spearman correlations. No null-hypothesis claims across dependent folds.

## Seeds
Split 42; CV 42 (feature study, tuning, OOF); nested CV outer 43, inner 42; bootstrap 42; model `random_state=42`.
