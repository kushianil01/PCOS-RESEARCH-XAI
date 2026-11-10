"""Step 4: symptom/feature ablation (primary study) and SEPARATE laboratory/ultrasound comparison experiments.

Design: nested cross-validation on ALL 541 records (outer 5-fold x 3 repeats, inner 5-fold grid search, scoring ROC-AUC).
This is a secondary, lower-variance estimate than the 109-record held-out test; it does not replace the held-out result.
Paired differences use the Nadeau-Bengio corrected variance (approximate: outer folds overlap).
Lab and ultrasound sets are comparison-only; they are never inputs of the deployed system.
"""
import json, numpy as np, pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from pcos import config as C, data as D, modeling as M

df = D.load_processed()
y = df[C.TARGET]; F = C.PRIMARY_FEATURES
SMALL_GRIDS = {"logreg": M.GRIDS["logreg"],
               "rf": {"clf__max_depth": [3, 5], "clf__min_samples_leaf": [3, 5]}}
outer = RepeatedStratifiedKFold(n_splits=C.CV_FOLDS, n_repeats=C.NESTED_REPEATS, random_state=C.SEED + 1)
k = C.CV_FOLDS * C.NESTED_REPEATS


def nested(features, model):
    aucs, aps = [], []
    for tr, va in outer.split(df, y):
        gs = GridSearchCV(M.build_pipeline(model, features), SMALL_GRIDS[model], scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=C.SEED), n_jobs=-1, refit=True)
        gs.fit(df.iloc[tr][features], y.iloc[tr])
        p = gs.predict_proba(df.iloc[va][features])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], p)); aps.append(average_precision_score(y.iloc[va], p))
    return np.array(aucs), np.array(aps)


def drop(*cols): return [f for f in F if f not in cols]

ABLATION = {
    "primary_full": F,
    **{f"drop_{s}": drop(s) for s in C.SYMPTOMS},
    "drop_all_5_symptoms": drop(*C.SYMPTOMS),
    "drop_cycle_irregular": drop("cycle_irregular"),
    "drop_cycle_and_symptoms": drop("cycle_irregular", *C.SYMPTOMS),
    "drop_lifestyle": drop(*C.LIFESTYLE),
    "only_cycle_and_symptoms": ["cycle_irregular"] + C.SYMPTOMS,
    "only_symptoms": C.SYMPTOMS,
    "only_cycle": ["cycle_irregular"],
    "only_age_bmi": ["age_years", "bmi_derived"],
}
COMPARISON = {   # separate, clearly labelled comparison experiments (invasive / lab / imaging inputs)
    "cmp_ultrasound_only": C.ULTRASOUND,
    "cmp_lab_only": C.LAB,
    "cmp_primary_plus_lab": F + C.LAB,
    "cmp_primary_plus_ultrasound": F + C.ULTRASOUND,
    "cmp_primary_plus_lab_plus_ultrasound": F + C.LAB + C.ULTRASOUND,
}
SENSITIVITY = {"sens_primary_plus_cycle_length_(meaning_unverified)": F + ["cycle_length_days"]}

rows, folds = [], {}
for group, sets in [("ablation", ABLATION), ("comparison_NON_deployable", COMPARISON), ("sensitivity", SENSITIVITY)]:
    for sname, feats in sets.items():
        for model in ["logreg", "rf"]:
            a, ap = nested(feats, model); folds[(sname, model)] = a
            rows.append(dict(group=group, feature_set=sname, model=model, n_features=len(feats), auc_mean=a.mean(), auc_sd=a.std(ddof=1),
                             pr_auc_mean=ap.mean()))
            print(f"{group:26s}{sname:52s}{model:7s} AUC={a.mean():.3f} (sd {a.std(ddof=1):.3f})", flush=True)
res = pd.DataFrame(rows)

# paired difference to the full primary set
diffs = []
for (sname, model), a in folds.items():
    if sname == "primary_full": continue
    d = a - folds[("primary_full", model)]
    se = np.sqrt(d.var(ddof=1) * (1 / k + 1 / (C.CV_FOLDS - 1))); t = stats.t.ppf(.975, k - 1)
    diffs.append(dict(feature_set=sname, model=model, auc_diff_vs_primary=d.mean(), ci_low=d.mean() - t * se, ci_high=d.mean() + t * se))
diffs = pd.DataFrame(diffs)
res = res.merge(diffs, on=["feature_set", "model"], how="left").round(4)
res.to_csv(C.METRICS / "ablation_comparison_nested_cv.csv", index=False)
print(res[["group", "feature_set", "model", "auc_mean", "auc_diff_vs_primary", "ci_low", "ci_high"]].to_string(index=False))
