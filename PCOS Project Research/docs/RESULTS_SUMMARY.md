# Results Summary

All numbers below were produced by the scripts in `experiments/` and are stored in `results/`. Nothing here is a clinical performance claim: the target is the Kerala dataset's own PCOS label (label concordance). There is **no external validation**. Raw-output paths are given for every table. Intervals are 95% percentile bootstrap over the 109 test records unless stated.

## 1. Data and split
541 records, 177 labelled PCOS (32.7%); frozen stratified split 432 train (141 positive) / 109 test (36 positive). Source: `results/metrics/data_quality_summary.json`, `results/metrics/split_definition.json` (script `01`, `02`).

## 2. Input set decision (training split only; `results/metrics/feature_redundancy_*.csv`, `feature_set_decision.json`)
- VIF in {height, weight, BMI, waist, hip}: height 37, weight 201, BMI 165 (severe redundancy); in {height, weight, waist, hip}: 1.3, 2.2, 4.6, 4.4.
- Repeated-CV AUC (5x5, train split, fixed hyper-parameters), logistic regression / random forest: BMI set 0.875 / 0.879; height+weight 0.874 / 0.883; height+weight+BMI 0.874 / 0.878; BMI+waist+hip 0.874 / 0.884; height+weight+waist+hip 0.873 / 0.886.
- Paired corrected 95% CIs of differences versus the BMI set all include 0 (largest: +0.007, CI -0.010 to +0.024). The pre-declared rule therefore kept the **smallest set: age, derived BMI, cycle irregularity, 5 symptoms, fast food, exercise (10 model features, 11 user inputs)**. Waist and hip added no detectable information.

## 3. Model selection (train split, 5x5 repeated CV; **selection stage, not final performance**; `results/metrics/final_evaluation.json`, `cv_grid_results.csv`)
| Family | Best CV AUC (mean, fold SD) | Selected params |
|---|---|---|
| Logistic regression | 0.876 (0.039) | C=0.1 |
| Decision tree | 0.823 (0.036) | depth 4, leaf 10 |
| Random forest | 0.880 (0.031) | depth 5, leaf 5, sqrt features |
| Gradient boosting | 0.872 (0.040) | 200 stumps, lr 0.1 |

Best = random forest (0.8798). One-SE cutoff (Nadeau-Bengio corrected SE 0.017): 0.863. Simplest family above the cutoff = **logistic regression (final model)**. Threshold from training OOF (>= 85% sensitivity) = **0.241**.

## 4. Held-out test, evaluated once (n=109, 36 positive; `results/metrics/final_evaluation.json`, `results/predictions/test_predictions.csv`)
| Model | AUC [95% CI] | PR-AUC | Sens | Spec | Precision | F1 | Brier |
|---|---|---|---|---|---|---|---|
| **Logistic regression (final, thr 0.241)** | **0.895 [0.825, 0.957]** | 0.835 [0.719, 0.938] | 0.861 [0.743, 0.968] | 0.712 [0.608, 0.814] | 0.596 [0.463, 0.731] | 0.705 | 0.117 |
| Random forest | 0.888 | 0.828 | 0.806 | 0.795 | 0.659 | 0.725 | 0.118 |
| Gradient boosting | 0.888 | 0.828 | 0.806 | 0.740 | 0.604 | 0.690 | 0.112 |
| Decision tree | 0.856 | 0.750 | 0.722 | 0.767 | 0.605 | 0.658 | 0.141 |
| Rule R4 (>= 3 of 5 symptoms; selected on train) | 0.820* | 0.646 | 0.778 | 0.863 | 0.737 | 0.757 | n/a |
| Rule R3 (irregular OR hair growth) | 0.800* | 0.595 | 0.806 | 0.795 | 0.659 | 0.725 | n/a |
| Rule R1 (irregular cycle) | 0.730* | 0.558 | 0.556 | 0.904 | 0.741 | 0.635 | n/a |
| Prior-probability dummy | 0.500 | 0.330 | 1.0 | 0.0 | 0.330 | 0.497 | 0.221 |

*AUC of a binary prediction. Confusion matrix of the final model: TP 31, FP 21, TN 52, FN 5. Paired bootstrap of test AUC differences: logistic minus random forest +0.007 (CI -0.015, +0.032); minus gradient boosting +0.007 (-0.009, +0.025); minus symptom-count score (5 symptoms + cycle; test AUC 0.870) +0.025 (-0.003, +0.059) (`results/metrics/explainability_summary.json`, `label_circularity.json`). **No reliable difference among logistic regression, random forest and gradient boosting, and only a small, uncertain advantage over a simple symptom count.** The rule rows use a different operating point than the 85%-sensitivity threshold, so their sensitivity/specificity are not like-for-like.

## 5. Calibration (`final_evaluation.json`, `fig08`)
Calibration-in-the-large ~0 (test -0.014; train OOF +0.004). Calibration slope 1.52 on test and 1.44 on train OOF (>1: predicted probabilities are too moderate; the heavily regularised model under-spreads risk). 5-bin ECE 0.081 (test), 0.061 (OOF); Brier 0.117 (test). No recalibration was applied.

## 6. Ablation and separate comparison (nested CV on all 541 records, outer 5x3; `results/metrics/ablation_comparison_nested_cv.csv`; secondary, different estimator from section 4)
Logistic regression AUC (random forest in the CSV): full primary set 0.877 (0.881).
| Variant | LogReg AUC | Diff vs full [corrected 95% CI] |
|---|---|---|
| drop each single symptom (weight gain / hair growth / skin darkening / hair loss / pimples) | 0.869 / 0.863 / 0.867 / 0.876 / 0.874 | -0.007 / -0.014 / -0.010 / -0.000 / -0.002; all CIs include 0 |
| drop cycle irregularity | 0.862 | -0.014 [-0.036, +0.008] |
| drop all 5 symptoms | 0.806 | -0.071 [-0.126, -0.016] |
| drop cycle and all symptoms (age, BMI, lifestyle only) | 0.758 | -0.118 [-0.187, -0.050] |
| drop lifestyle (fast food, exercise) | 0.877 | +0.001 [-0.009, +0.011] |
| symptoms only (5) | 0.855 | -0.022 [-0.047, +0.003] |
| cycle + symptoms only | 0.870 | -0.007 [-0.024, +0.010] |
| cycle only | 0.691 | -0.185 |
| age + BMI only | 0.660 | -0.217 |
| sensitivity: + cycle length (meaning unverified) | 0.876 | -0.000 [-0.013, +0.013] |

**Comparison experiments (never deployable; separate inputs):** ultrasound only 0.892 (RF 0.889); laboratory only 0.638 (RF 0.702); primary + laboratory 0.870 (RF 0.878; no gain over primary); primary + ultrasound 0.950 (RF 0.951; +0.073, CI +0.021 to +0.125); primary + lab + ultrasound 0.945 (RF 0.954). In this file the label tracks ultrasound findings far more than laboratory values, and the non-invasive set loses roughly 0.07 AUC against the non-invasive + ultrasound set. Laboratory inputs were log1p-transformed and untuned beyond the small shared grids; weak laboratory results may partly reflect outliers and undocumented units, so they say little about laboratory tests in general.

## 7. Label circularity (`results/metrics/label_circularity.json`, `fig10`)
- Our crude Rotterdam-style proxy (irregular cycle; hair growth, acne or hair loss; max follicles >= 12; this is an approximation, not the dataset's rule): records meeting 0/1/2/3 criteria are labelled PCOS 4.0% / 16.7% / 67.5% / 100% (n=124/245/126/46). Meeting >= 2 criteria reproduces the label with accuracy 0.839, sensitivity 0.740, specificity 0.887.
- Non-invasive model score correlates with the withheld follicle count (Spearman 0.41 train OOF, 0.35 test), yet still separates labels within follicle strata (AUC 0.82, 0.90, 0.96, 0.85 for max follicles <=5, 6-8, 9-11, >=12; pooled OOF + test scores).
- Interpretation: the label is strongly consistent with symptom-and-ultrasound criteria, symptoms carry most non-invasive signal (grouped permutation: 5 symptoms -0.219 AUC, cycle -0.034, age+BMI -0.004, lifestyle -0.015), and high concordance cannot be taken as independent predictive value.

## 8. Explainability (`results/metrics/explain_*.csv`, `explainability_summary.json`)
- Odds ratios (train, 1000-bootstrap 95% interval): skin darkening 2.40 [1.83, 3.16], hair growth 2.38 [1.82, 3.12], weight gain 2.16 [1.66, 2.88], cycle irregular 2.13 [1.59, 2.80], fast food 1.70 [1.31, 2.25], pimples 1.63 [1.25, 2.09], age 0.94 per year [0.90, 0.97]. Not distinguishable from 1: BMI 1.02 [0.97, 1.07], hair loss 1.10 [0.86, 1.40], exercise 1.19 [0.91, 1.55].
- Importance is spread across about seven features; the top two carry 32% of mean |SHAP|. Spearman agreement between permutation and SHAP importance: 0.98 (train), 0.73 (test, noisier). Which single symptom ranks first is unstable (top-1 in bootstrap: skin darkening 38%, hair growth 34%, weight gain 15%, cycle 12%) because the symptoms are correlated (pairwise phi 0.30-0.37) - group-level conclusions are more reliable than per-feature ranks.
- BMI contributes little once symptoms are included (grouped age+BMI drop 0.004), although its SHAP share grows in the BMI >= 30 group (0.15 vs 0.05).
- Fast food is more influential than expected (OR 1.7; alone AUC about 0.70 in Phase 1): possible artefact or confounder; not interpreted as a cause.

## 9. Subgroups, errors and robustness (`subgroup_evaluation.csv`, `error_analysis.csv`, `robustness_test.csv`)
- Subgroup AUC (pooled train OOF + test scores): age <=24 0.966 (n=50), 25-34 0.850 (332), >=35 0.887 (159); BMI <25 0.848 (314), 25-29.9 0.917 (184), >=30 0.888 (43). Intervals overlap widely (e.g. BMI >=30: 0.756-0.971). Specificity at the fixed threshold varies (age <=24: 0.54; >=35: 0.79; BMI <25: 0.79; BMI >=30: 0.53), i.e. false-alarm burden is uneven. No other demographic subgroup data exist.
- Errors (pooled): of 27 false negatives, 74% have >= 12 follicles on ultrasound but few symptoms (mean 0.6 in the test split, 0.95 in train OOF); of 110 false positives only 2.7% have >= 12 follicles (mean 2.4 symptoms). So false positives are mostly symptomatic records without polycystic-morphology evidence in this dataset.
- Robustness on test: BMI noise +-1 -> AUC 0.895 -> 0.895 (0.5% decisions flip); age noise +-2 y -> 0.894 (2.4%); 10% of yes/no answers flipped -> 0.864 (13% flip); all yes/no answers missing (mode-imputed) -> 0.619.

## 10. Reproducibility notes
Reruns of step 03 gave identical logistic-regression test probabilities; random-forest probabilities differ by up to 3e-16 (thread summation order). `tests/test_model_consistency.py` recomputes the reported test metrics from the saved predictions to 1e-9.
