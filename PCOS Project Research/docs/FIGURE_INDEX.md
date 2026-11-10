# Figure Index

All figures are in `results/figures/` as PNG (300 dpi) and PDF, generated from real data/experiment outputs by the script listed.

| File | Title | What it shows / caption | Script |
|---|---|---|---|
| fig01_class_distribution | Class distribution | 364 non-PCOS (67.3%) vs 177 PCOS-labelled (32.7%) records | `01` |
| fig02_missingness | Missing values after cleaning | Only marriage years, AMH and fast food have missing cells (1 each) | `01` |
| fig03_continuous_inputs | Primary continuous inputs by label | Age, height, weight histograms by label (density) | `01` |
| fig04_label_rate_by_binary | Label rate by binary input | Unadjusted proportion labelled PCOS for each yes/no input | `01` |
| fig05_feature_correlation | Feature dependence | Spearman correlations among candidate inputs (weight-BMI, waist-hip highly correlated) | `01` |
| fig06_roc_pr_test | ROC and PR curves, held-out test | Four model families on the 109 held-out records; final model emphasised | `03` |
| fig07_confusion_matrix_test | Confusion matrix, held-out test | Final model at threshold 0.241: TP 31, FP 21, TN 52, FN 5 | `03` |
| fig08_calibration | Calibration | Reliability (5 quantile bins) on training OOF and test, with calibration slope and Brier score | `03` |
| fig09_cv_model_comparison | CV model comparison | Mean repeated-CV AUC on the training split (selection stage, not final performance) | `03` |
| fig10_label_circularity | Label circularity | Label vs crude Rotterdam-style proxy criteria; non-invasive OOF score vs withheld follicle count | `05` |
| fig11_logreg_odds_ratios | Logistic odds ratios | Odds ratios with 95% bootstrap intervals (association, not causation) | `06` |
| fig12_importance_shap_vs_permutation | Global importance | Mean abs SHAP vs permutation AUC drop on test | `06` |
| fig13_local_explanations | Local explanations | Exact SHAP contributions for a true positive, false positive, false negative and true negative from the test split | `06` |
| fig14_subgroup_auc | Subgroup AUC | AUC with bootstrap intervals by age and BMI group (pooled OOF + test) | `06` |
| fig15_ablation_nested_cv | Symptom/feature ablation | Nested-CV AUC (all 541 records) for removed or isolated input groups | `07` (data from `04`) |
| fig16_lab_ultrasound_comparison | Laboratory/ultrasound comparison | Comparison-only feature sets; never inputs of the deployed system | `07` (data from `04`) |
