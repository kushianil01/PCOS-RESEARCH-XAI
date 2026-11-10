# Limitations and Ethics

**This is an academic research prototype. It is not a diagnostic, screening-approval or treatment tool and has no clinical validation.**

## What the system does and does not measure
- The target is the Kerala dataset's existing `PCOS (Y/N)` label, whose construction is undocumented. Results are **label concordance**, not diagnostic accuracy.
- A model score is "how much do these non-invasive answers resemble records labelled PCOS in this cohort", not "probability that this person has PCOS".

## Dataset limitations
- One regional cohort (Kerala, India), ages 20-48, 541 records, 177 positive. Prevalence (32.7%) reflects a clinic population, not the community, so predictive values do not transfer.
- No documentation of label criteria, symptom definitions, assessor, hospital, date or ethics approval; license asserted by the uploader only.
- Cycle regularity coding is inferred; cycle length is ambiguous and unused.
- Entry errors exist; they were flagged, not repaired. Several symptom items may be clinician-assessed rather than self-reported, so the deployed self-report setting may differ from the training setting (distribution shift, unquantified).

## Label circularity (central caveat)
Hair growth, acne, hair loss, cycle irregularity and skin darkening overlap with PCOS diagnostic criteria and clinical signs. A crude Rotterdam-style proxy built from the same dataset reproduces the label with 83.9% accuracy, and all 46 records meeting all three proxy criteria are labelled PCOS (`results/metrics/label_circularity.json`). Symptom-only models are close to the full non-invasive model (`results/metrics/ablation_comparison_nested_cv.csv`). High agreement therefore partly reflects the labelling process; it does not show that the symptoms independently predict a future or hidden diagnosis.

## Evaluation limitations
- The held-out test set has 109 records (36 positive); confidence intervals are wide (AUC 95% CI 0.825-0.957, sensitivity 0.743-0.968).
- Repeated-CV and nested-CV folds overlap; corrected variance is approximate and no formal significance claims are made.
- Hyper-parameters, feature set and threshold used training data only; there is still one split, one seed.
- **No external validation** was done or is possible with open data found. Do not describe results as externally validated or generalisable.
- Subgroups (age, BMI) have small counts; intervals are wide and no fairness claim is made. Ethnicity, income and other protected attributes are absent, so fairness across them cannot be assessed.

## Explanation limitations
- Odds ratios, SHAP and permutation importance describe associations in the fitted model, not causes.
- Correlated inputs (skin darkening, weight gain, hair growth co-occur) share credit unstably; individual coefficient ranks are less reliable than the group-level pattern.
- BMI shows almost no independent contribution once symptoms are included, which could reflect dataset structure rather than biology.
- Fast-food consumption is unexpectedly influential; it may be a dataset artefact or confounder and should not be read as a risk factor.

## Risks of misinterpretation and responsible-use boundaries
- A "below threshold" result must not reassure anyone; a "flag" must not alarm. The threshold targets about 85% sensitivity and therefore produces many false positives (about 29% of non-labelled test records were flagged).
- Not for self-diagnosis, triage, treatment decisions, insurance, employment or any decision about a person.
- Allowed use: research, teaching, discussion of explainable decision-support design.
- Commercial use is prohibited by the data license (CC BY-NC-SA 4.0); share-alike applies to derived data.

## Privacy
The app asks only for age, height, weight and eight yes/no items, stores nothing, writes no logs of inputs and requests no identifiers. If the app is ever hosted, platform request logs must be reviewed separately.
