"""Step 2: create the frozen train/test split, then study feature redundancy on the TRAINING split only.

Pre-declared decision rule (written before running): start from the smallest candidate set and move to a larger set only if
(a) mean repeated-CV AUC improves by >= 0.01 for BOTH logistic regression and random forest AND
(b) the Nadeau-Bengio corrected 95% CI of the paired fold-wise AUC difference excludes 0 for both models.
Models use fixed (untuned) hyper-parameters so the comparison isolates the feature set.
"""
import json, numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pcos import config as C, data as D, split as S, modeling as M

df = D.load_processed()
sp = S.save_split(df)
train, test, _ = S.load_split(df)
print(f"split: train={sp['n_train']} (pos {sp['train_pos']}), test={sp['n_test']} (pos {sp['test_pos']})")
y = train[C.TARGET]

rest = ["cycle_irregular"] + C.SYMPTOMS + C.LIFESTYLE
SETS = {
    "BMI": ["age_years", "bmi_derived"] + rest,
    "HeightWeight": ["age_years", "height_cm", "weight_kg"] + rest,
    "Height+Weight+BMI": ["age_years", "height_cm", "weight_kg", "bmi_derived"] + rest,
    "BMI+Waist": ["age_years", "bmi_derived", "waist_inch"] + rest,
    "BMI+Waist+Hip": ["age_years", "bmi_derived", "waist_inch", "hip_inch"] + rest,
    "HeightWeight+Waist+Hip": ["age_years", "height_cm", "weight_kg", "waist_inch", "hip_inch"] + rest,
}
cv = M.cv_splitter()
fold_tab, summary = {}, []
for sname, feats in SETS.items():
    for mname in ["logreg", "rf"]:
        pipe = M.build_pipeline(mname, feats, {"clf__C": 1.0} if mname == "logreg" else {"clf__min_samples_leaf": 3})
        fs = M.fold_scores(pipe, train, y, feats, cv)
        fold_tab[(sname, mname)] = fs["auc"].values
        summary.append(dict(feature_set=sname, model=mname, n_features=len(feats), auc_mean=fs.auc.mean(), auc_sd=fs.auc.std(),
                            pr_auc_mean=fs.pr_auc.mean(), brier_mean=fs.brier.mean()))
res = pd.DataFrame(summary).round(4)

# paired differences vs the smallest set (BMI), Nadeau-Bengio corrected
k = C.CV_FOLDS * C.CV_REPEATS
n_tr = len(train) * (1 - 1 / C.CV_FOLDS); n_va = len(train) / C.CV_FOLDS
rows = []
for sname in SETS:
    if sname == "BMI": continue
    for mname in ["logreg", "rf"]:
        d = fold_tab[(sname, mname)] - fold_tab[("BMI", mname)]
        var = d.var(ddof=1) * (1 / k + n_va / n_tr)
        se = np.sqrt(var); t = stats.t.ppf(0.975, k - 1)
        rows.append(dict(feature_set=sname, model=mname, mean_auc_diff_vs_BMI=d.mean(), ci_low=d.mean() - t * se, ci_high=d.mean() + t * se))
diff = pd.DataFrame(rows).round(4)

# decision
def improves(sname):
    r = diff[diff.feature_set == sname]
    return bool(((r.mean_auc_diff_vs_BMI >= 0.01) & (r.ci_low > 0)).all())
chosen, why = "BMI", "no larger set met the pre-declared improvement rule"
cands = [s for s in SETS if s != "BMI" and improves(s)]
if cands:
    chosen = max(cands, key=lambda s: res[(res.feature_set == s)].auc_mean.mean()); why = f"{chosen} met the improvement rule"

# collinearity diagnostics (training split)
X = train[SETS["Height+Weight+BMI"][:4] + ["waist_inch", "hip_inch"]].dropna()
vif_hwb = {c: float(variance_inflation_factor(X.assign(const=1.0).values, i)) for i, c in enumerate(X.columns)}
X2 = train[["age_years", "height_cm", "weight_kg", "waist_inch", "hip_inch"]].dropna()
vif_hw_wh = {c: float(variance_inflation_factor(X2.assign(const=1.0).values, i)) for i, c in enumerate(X2.columns)}
pairs = {f"{a}~{b}": float(train[[a, b]].corr(method="spearman").iloc[0, 1])
         for a, b in [("height_cm", "weight_kg"), ("weight_kg", "bmi_derived"), ("bmi_derived", "waist_inch"), ("waist_inch", "hip_inch"), ("weight_kg", "waist_inch")]}

out = dict(decision=chosen, reason=why, features=SETS[chosen], candidate_sets=SETS,
           input_fields=["age_years", "height_cm", "weight_kg"] + rest, vif_height_weight_bmi_waist_hip=vif_hwb,
           vif_height_weight_waist_hip=vif_hw_wh, spearman_pairs=pairs,
           rule=__doc__.strip().split("\n")[2:7])
C.METRICS.mkdir(parents=True, exist_ok=True)
(C.METRICS / "feature_set_decision.json").write_text(json.dumps(out, indent=2))
res.to_csv(C.METRICS / "feature_redundancy_cv.csv", index=False); diff.to_csv(C.METRICS / "feature_redundancy_paired_diffs.csv", index=False)
print(res.to_string(index=False)); print(diff.to_string(index=False)); print(json.dumps({k: out[k] for k in ["decision", "reason", "features", "vif_height_weight_bmi_waist_hip", "vif_height_weight_waist_hip", "spearman_pairs"]}, indent=1))
