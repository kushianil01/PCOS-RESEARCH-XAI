# Phase 1 – Dataset Discovery, Audit and Recommendation (awaiting approval)

Access date: 2026-10-17. All numbers below come from `src/audit_candidates.py` and `src/feasibility_probe.py`
(raw outputs in `results/audit/`). Anything not run is labelled *unverified*.

## Constraint applied
The PCOS final system must use **non-invasive inputs only** (demographics, cycle pattern, symptoms, anthropometrics,
lifestyle). Blood, hormone and ultrasound variables are for comparison only, never required inputs.

## Candidates

| # | Candidate | Source / access | License | Rows | Label | Verdict |
|---|-----------|-----------------|---------|------|-------|---------|
| 1 | **Kerala PCOS (Kottarathil)** | Kaggle `prasoonkottarathil/polycystic-ovary-syndrome-pcos`; downloaded, hashes in `data/documentation/MANIFEST_raw_sha256.txt` | **CC BY-NC-SA 4.0** (Kaggle API metadata): non-commercial, attribution, share-alike | 541 x 45 (+ 541-row infertility CSV) | `PCOS (Y/N)`, 177 pos / 364 neg (32.7%) | **Only usable tabular PCOS set found** |
| 2 | Figshare "PCOS DATA.xlsx" (Elfadil & Elmugadam, doi 10.6084/m9.figshare.17103995) | Figshare API, downloaded | CC BY 4.0 | 300 x 8 | **No target column** (Age, FH, MC, LH, FSH, AMH, BMI, TT) | Rejected: unlabeled; 4 of 8 variables are hormone assays |
| 3 | Figshare/Zenodo PCOS ultrasound image sets (e.g. 12,680 images) | figshare 27682557, zenodo 14592001 | CC BY 4.0 | images | image labels | Rejected: not tabular, outside the non-invasive scope (not downloaded) |
| 4 | NHANES Reproductive Health questionnaire | CDC codebooks RHQ_I (2015-16), RHQ_J (2017-18), RHQ_L (2021-23) | US government public data | large | **No PCOS item in the three codebooks I checked** | Rejected for PCOS |
| 5 | Sills et al. 2001 (657 self-reports), healthdata.gov | metadata only | CC BY (article) | 657 | n/a | Rejected: no raw data deposit found |
| 6 | Zenodo 14575651 catalogue of 36 PCOS datasets | downloaded | CC BY 4.0 | catalogue | n/a | Discovery aid only. Of 36 entries, 4 numeric ones are Kerala derivatives (541 rows); one 768-row CC0 entry (D6) I did not identify |
| 7 | UCI Cervical Cancer Risk Factors (non-PCOS fallback) | UCI id 383, downloaded | not re-verified | 858 x 36 | Biopsy 6.4% positive | Fallback only: 4 targets, 23 duplicate rows, >10% missing in many features |
| 8 | UCI CDC Diabetes Health Indicators (non-PCOS fallback) | UCI id 891 page read; zip URL returned 404, **not downloaded** | page does not state one | 253,680 | binary, self-reported (BRFSS 2014) | Fallback only: page has conflicting feature counts and target definition |

## Kerala dataset audit (verified unless marked)
- Provenance: the Kaggle description says "collected from 10 different hospitals across Kerala, India". There is no
  peer-reviewed data descriptor, no ethics statement, no label definition and no collection dates. Some papers say one
  hospital (unresolved).
- No duplicate rows or IDs. Missing: 1 (marriage years), 1 (fast food), 1 non-numeric AMH (`a`), 1 malformed beta-HCG
  (`1.99.`), one fully empty column.
- Implausible values (likely entry errors): pulse 13 (2 rows <40), BP 12/8, cycle length 0 (1 row), FSH 5052, LH 2018,
  Vit D3 6014, 4 rows where BMI differs from weight/height by >0.5.
- Cycle(R/I) is coded 2/4 (plus one "5") with no documented meaning (2 = regular, 4 = irregular is my inference).
- **Label construction is undocumented.** Max follicle count >=10 matches the label in 85.4% of rows, so the label looks
  clinically derived from ultrasound plus symptoms (Rotterdam-like). This is a hypothesis, not documented fact.
- **Circularity risk:** hair growth, acne, cycle irregularity, weight gain and skin darkening are themselves components
  of PCOS diagnostic criteria. A model on them reproduces clinician labels; it does not show independent predictive value.
- No row-order/ID leakage found (AUC of file number = 0.538). Infertility CSV IDs are offset by 10,000 (10001-10541), so
  there is no direct overlap with the main file; whether they are the same patients is *unverified*.
- Collinearity in the non-invasive set: weight-BMI r=0.90, waist-hip r=0.87.
- Population: single-region Indian cohort, ages 20-48, no ethnicity or socioeconomic variables, many married/pregnant
  women (likely clinic attendees, so selection bias). No group or time variable, so stratified random splits are the only option.

## Can a non-invasive set support the question? (exploratory probe)
Untuned defaults, 5x5 repeated stratified CV on all 541 rows. No test set was used; these are screening numbers, not results.

| Feature set | LogReg AUC | RandForest AUC |
|---|---|---|
| Non-invasive full (19 features) | 0.877 | 0.887 |
| Symptoms + cycle + lifestyle (9) | 0.881 | 0.869 |
| Anthropometric only (7) | 0.655 | 0.589 |
| Hormone/lab only (10) | 0.645 | 0.711 |
| Ultrasound only (5) | 0.895 | 0.872 |
| All 41 features | 0.941 | 0.961 |

Reading: signal in the non-invasive set comes almost entirely from symptoms and cycle pattern, not body measures. Lab
hormones alone add little in this file. The fold-to-fold SD is about 0.03, so differences among the top non-invasive sets
are not meaningful. Strong performance is partly expected from the label circularity noted above.

## Recommendation
**Primary dataset: Kerala PCOS (candidate 1)**, with a non-invasive feature set as the *only* inputs of the deployed
system. Labs and ultrasound appear only in comparison experiments (what is lost by omitting them), never as app inputs.
External validation is not possible: no second labelled open PCOS dataset exists (candidate 2 has no labels), and I will not claim it.

**Refined research question:** *Using a single documented Kerala PCOS cohort, how well can a reproducible, explainable
model built only on non-invasive, self-reportable inputs reproduce the dataset's PCOS label, how much does it lose against
laboratory/ultrasound-based models, and how stable, calibrated and shortcut-prone are its predictions and explanations?*
The system is framed as label-concordance **screening support**, not diagnosis.

## Decisions needed from you
1. Approve the Kerala dataset and the refined question (alternative: switch domain to a fallback; I advise against it).
2. License: CC BY-NC-SA 4.0 permits academic non-commercial use with attribution; derived data and artifacts must be share-alike.
3. Input set: I plan to **exclude** blood group, Hb, pulse/RR/BP (clinic-measured, error-prone), pregnancy status, marriage
   years and abortions (reproductive-history proxies unrelated to symptom screening, possible selection artefacts). I would keep
   age, height, weight (BMI derived), waist, hip, cycle regularity/length, 5 symptoms, fast food and exercise. Keep pregnancy/marriage instead?
