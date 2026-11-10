"""Step 6: explainability (coefficients, permutation importance, SHAP), stability, subgroup, error and robustness analyses
for the FINAL saved model. Explanations describe the fitted model's association with the dataset label; they are not causal.
"""
import json, numpy as np, pandas as pd, joblib, shap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from pcos import config as C, data as D, split as S, modeling as M

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
df = D.load_processed(); train, test, sp = S.load_split(df)
F = C.PRIMARY_FEATURES; ytr, yte = train[C.TARGET], test[C.TARGET]
meta = json.loads((C.MODELS / "model_metadata.json").read_text()); fin = json.loads((C.METRICS / "final_evaluation.json").read_text())
assert D.sha256(C.MODELS / meta["model_file"]) == meta["model_sha256"]
model = joblib.load(C.MODELS / meta["model_file"]); thr = meta["threshold"]
assert meta["model_family"] == "logreg", "explanation code below assumes a linear final model"
rng = np.random.default_rng(C.SEED)
out = {}

# ---------------------------------------------------------------- 1. coefficients + bootstrap stability
prep, clf = model.named_steps["prep"], model.named_steps["clf"]
Xtr_t = prep.transform(train[F]); names = list(Xtr_t.columns)
scale = dict(zip(prep.named_transformers_["cont"].feature_names_in_, prep.named_transformers_["cont"].named_steps["standardscaler"].scale_))
coef = pd.Series(clf.coef_[0], index=names)
per_unit = pd.Series({n: coef[n] / scale.get(n, 1.0) for n in names})          # log-odds per natural unit
B = 1000; boots = np.zeros((B, len(names)))
for b in range(B):
    idx = rng.integers(0, len(train), len(train))
    m = clone(model).fit(train.iloc[idx][F], ytr.iloc[idx])
    c = m.named_steps["clf"].coef_[0]; sc = {k: v for k, v in zip(m.named_steps["prep"].named_transformers_["cont"].feature_names_in_,
                                                              m.named_steps["prep"].named_transformers_["cont"].named_steps["standardscaler"].scale_)}
    boots[b] = [c[i] / sc.get(n, 1.0) for i, n in enumerate(m.named_steps["prep"].get_feature_names_out())]
bn = list(model.named_steps["prep"].get_feature_names_out()); boots = pd.DataFrame(boots, columns=bn)[names]
std_eff = boots * pd.Series({n: scale.get(n, 1.0) if n in scale else 1.0 for n in names})   # effect per 1 SD (cont) / per Yes (binary)
rank_freq_top1 = (std_eff.abs().rank(axis=1, ascending=False) == 1).mean()
coef_tab = pd.DataFrame({"unit": ["per year" if n == "age_years" else "per kg/m2" if n == "bmi_derived" else "Yes vs No" for n in names],
                         "log_odds": per_unit, "odds_ratio": np.exp(per_unit), "or_ci_low": np.exp(boots.quantile(.025)), "or_ci_high": np.exp(boots.quantile(.975)),
                         "sign_stability": (np.sign(boots) == np.sign(per_unit)).mean(), "top1_rank_freq": rank_freq_top1,
                         "std_effect": coef}).round(4)
coef_tab.to_csv(C.METRICS / "explain_logreg_coefficients.csv"); print(coef_tab.to_string())

# ---------------------------------------------------------------- 2. permutation importance (AUC drop)
pi_test = permutation_importance(model, test[F], yte, scoring="roc_auc", n_repeats=200, random_state=C.SEED)
# CV-based permutation importance on the training split (models refit per fold; permuted on validation fold only)
cv_imp = []
for tr, va in M.cv_splitter(n_repeats=2).split(train, ytr):
    m = clone(model).fit(train.iloc[tr][F], ytr.iloc[tr])
    r = permutation_importance(m, train.iloc[va][F], ytr.iloc[va], scoring="roc_auc", n_repeats=10, random_state=C.SEED)
    cv_imp.append(r.importances_mean)
cv_imp = np.array(cv_imp)
imp = pd.DataFrame({"perm_auc_drop_test": pi_test.importances_mean, "perm_test_sd": pi_test.importances_std,
                    "perm_auc_drop_cv_train": cv_imp.mean(0), "perm_cv_sd": cv_imp.std(0)}, index=F)

# grouped permutation (correlated / related features): symptoms, anthropometry(BMI)+age, lifestyle
def grouped(groups, X, y, n=200):
    res = {}
    base = roc_auc_score(y, model.predict_proba(X[F])[:, 1])
    for g, cols in groups.items():
        drops = []
        for _ in range(n):
            Xp = X[F].copy(); perm = rng.permutation(len(X)); Xp[cols] = Xp[cols].values[perm]
            drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        res[g] = dict(mean=float(np.mean(drops)), sd=float(np.std(drops)))
    return res
groups = {"all_5_symptoms": C.SYMPTOMS, "cycle_irregular": ["cycle_irregular"], "age+BMI": ["age_years", "bmi_derived"], "lifestyle": C.LIFESTYLE,
          "skin_darkening+weight_gain+hair_growth": ["skin_darkening", "weight_gain", "hair_growth"]}
out["grouped_permutation_auc_drop_test"] = grouped(groups, test, yte)

# ---------------------------------------------------------------- 3. SHAP (exact linear SHAP, interventional, log-odds)
expl = shap.LinearExplainer(clf, shap.maskers.Independent(Xtr_t, max_samples=len(Xtr_t)))
sv_te = pd.DataFrame(expl.shap_values(prep.transform(test[F])), columns=names)
sv_tr = pd.DataFrame(expl.shap_values(Xtr_t), columns=names)
imp["mean_abs_shap_test"] = sv_te.abs().mean(); imp["mean_abs_shap_train"] = sv_tr.abs().mean()
imp["share_of_total_shap_test"] = imp.mean_abs_shap_test / imp.mean_abs_shap_test.sum()
imp = imp.round(4).sort_values("mean_abs_shap_test", ascending=False); imp.to_csv(C.METRICS / "explain_importance_table.csv"); print(imp.to_string())
out["top2_share_of_mean_abs_shap_test"] = float(imp.share_of_total_shap_test.iloc[:2].sum())
rk = {"perm_test_vs_shap_test": spearmanr(imp.perm_auc_drop_test, imp.mean_abs_shap_test),
      "perm_cv_vs_shap_train": spearmanr(imp.perm_auc_drop_cv_train, imp.mean_abs_shap_train),
      "shap_train_vs_shap_test": spearmanr(imp.mean_abs_shap_train, imp.mean_abs_shap_test)}
out["rank_agreement_spearman"] = {k: [round(float(v.statistic), 3), round(float(v.pvalue), 4)] for k, v in rk.items()}
phi = df[C.BINARY].corr()  # phi coefficients among binary inputs
pairs = [(a, b, round(float(phi.loc[a, b]), 2)) for i, a in enumerate(C.BINARY) for b in C.BINARY[i + 1:] if abs(phi.loc[a, b]) >= 0.3]
out["binary_input_pairs_abs_phi_ge_0.3"] = pairs

# ---------------------------------------------------------------- 4. simple vs complex (paired bootstrap on TEST AUC)
tp = pd.read_csv(C.PREDS / "test_predictions.csv")
def paired_auc_diff(a, b, n=C.N_BOOT):
    r = np.random.default_rng(C.SEED); d = []
    for _ in range(n):
        i = r.integers(0, len(yte), len(yte))
        if yte.values[i].min() == yte.values[i].max(): continue
        d.append(roc_auc_score(yte.values[i], a[i]) - roc_auc_score(yte.values[i], b[i]))
    return dict(mean=float(np.mean(d)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
cnt = test[C.SYMPTOMS + ["cycle_irregular"]].sum(axis=1).values
out["paired_bootstrap_test_auc_diff"] = {"logreg_minus_rf": paired_auc_diff(tp.p_logreg.values, tp.p_rf.values),
                                         "logreg_minus_gbm": paired_auc_diff(tp.p_logreg.values, tp.p_gbm.values),
                                         "logreg_minus_symptom_count_score": paired_auc_diff(tp.p_logreg.values, cnt.astype(float))}

# ---------------------------------------------------------------- 5. subgroup evaluation (pooled train-OOF scores + test scores)
oof = pd.read_csv(C.PREDS / "train_oof_predictions.csv")
pool = pd.concat([train.merge(oof[["row_id", "oof_logreg"]].rename(columns={"oof_logreg": "score"}), on="row_id").assign(src="train_oof"),
                  test.merge(tp[["row_id", "final_prob"]].rename(columns={"final_prob": "score"}), on="row_id").assign(src="test")])
pool["age_group"] = pd.cut(pool.age_years, [0, 24, 34, 100], labels=["<=24", "25-34", ">=35"])
pool["bmi_group"] = pd.cut(pool.bmi_derived, [0, 24.99, 29.99, 100], labels=["<25", "25-29.9", ">=30"])
sub = []
for gcol in ["age_group", "bmi_group", "src"]:
    for lvl, g in pool.groupby(gcol, observed=True):
        n, pos = len(g), int(g.pcos.sum()); row = dict(grouping=gcol, level=str(lvl), n=n, n_pos=pos, label_rate=round(pos / n, 3), mean_score=round(g.score.mean(), 3))
        if pos >= 15 and n - pos >= 15:
            bs = [roc_auc_score(g.pcos.values[i], g.score.values[i]) for i in (rng.integers(0, n, n) for _ in range(500)) if 0 < g.pcos.values[i].sum() < n]
            row.update(auc=round(roc_auc_score(g.pcos, g.score), 3), auc_ci_low=round(np.percentile(bs, 2.5), 3), auc_ci_high=round(np.percentile(bs, 97.5), 3))
            mm = M.binary_metrics(g.pcos, g.score, thr); row.update(sensitivity=round(mm["sensitivity"], 3), specificity=round(mm["specificity"], 3))
        else:
            row["note"] = "too few positives or negatives for AUC (n_pos or n_neg < 15)"
        sub.append(row)
sub = pd.DataFrame(sub); sub.to_csv(C.METRICS / "subgroup_evaluation.csv", index=False); print(sub.to_string(index=False))
bm = pool.reset_index(drop=True); sv_pool = pd.DataFrame(expl.shap_values(prep.transform(bm[F])), columns=names)
out["mean_abs_shap_by_bmi_group"] = sv_pool.abs().groupby(bm.bmi_group.values, observed=True).mean().round(3).to_dict("index")
out["mean_abs_shap_by_age_group"] = sv_pool.abs().groupby(bm.age_group.values, observed=True).mean().round(3).to_dict("index")

# ---------------------------------------------------------------- 6. error analysis (test + train OOF)
errs = []
for src, g in pool.groupby("src"):
    g = g.assign(pred=(g.score >= thr).astype(int)); g["err"] = np.select([(g.pred == 1) & (g.pcos == 0), (g.pred == 0) & (g.pcos == 1)], ["FP", "FN"], "correct")
    g["n_sym"] = g[C.SYMPTOMS].sum(axis=1); g["fmax"] = g[["follicle_l", "follicle_r"]].max(axis=1)
    for e, h in g.groupby("err"):
        errs.append(dict(source=src, outcome=e, n=len(h), mean_age=h.age_years.mean(), mean_bmi=h.bmi_derived.mean(), pct_irregular=h.cycle_irregular.mean(),
                         mean_n_symptoms=h.n_sym.mean(), mean_max_follicles_WITHHELD=h.fmax.mean(), mean_score=h.score.mean()))
errs = pd.DataFrame(errs).round(2); errs.to_csv(C.METRICS / "error_analysis.csv", index=False); print(errs.to_string(index=False))
allg = pool.assign(pred=(pool.score >= thr).astype(int))
fn = allg[(allg.pred == 0) & (allg.pcos == 1)]; fp = allg[(allg.pred == 1) & (allg.pcos == 0)]
out["error_follicle_profile"] = dict(
    fn_n=len(fn), fn_share_with_max_follicles_ge12=float((fn[["follicle_l", "follicle_r"]].max(axis=1) >= 12).mean()),
    fp_n=len(fp), fp_share_with_max_follicles_ge12=float((fp[["follicle_l", "follicle_r"]].max(axis=1) >= 12).mean()),
    note="pooled train-OOF + test; follicle counts were never model inputs")

# ---------------------------------------------------------------- 7. robustness on TEST (input noise / answer flips)
base_p = model.predict_proba(test[F])[:, 1]; rob = []
for scenario in ["baseline", "BMI +-1.0 noise", "age +-2 yr noise", "10% of binary answers flipped", "all binary answers missing (mode-imputed)"]:
    aucs, flips = [], []
    for rep in range(200 if scenario != "baseline" else 1):
        X = test[F].copy()
        if scenario.startswith("BMI"): X["bmi_derived"] += rng.normal(0, 1.0, len(X))
        if scenario.startswith("age"): X["age_years"] += rng.normal(0, 2.0, len(X))
        if scenario.startswith("10%"):
            for c in C.BINARY:
                if c in F: X[c] = np.where(rng.random(len(X)) < .10, 1 - X[c], X[c])
        if scenario.startswith("all binary"):
            for c in C.BINARY:
                if c in F: X[c] = np.nan
        p = model.predict_proba(X)[:, 1]; aucs.append(roc_auc_score(yte, p)); flips.append(np.mean((p >= thr) != (base_p >= thr)))
    rob.append(dict(scenario=scenario, auc_mean=np.mean(aucs), auc_sd=np.std(aucs), decision_flip_rate=np.mean(flips)))
rob = pd.DataFrame(rob).round(4); rob.to_csv(C.METRICS / "robustness_test.csv", index=False); print(rob.to_string(index=False))
(C.METRICS / "explainability_summary.json").write_text(json.dumps(out, indent=2, default=str))

# ---------------------------------------------------------------- figures
ct = coef_tab.sort_values("odds_ratio")
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.errorbar(ct.odds_ratio, range(len(ct)), xerr=[ct.odds_ratio - ct.or_ci_low, ct.or_ci_high - ct.odds_ratio], fmt="o", color="#2a6f97", capsize=3)
ax.axvline(1, color="k", ls=":", lw=1); ax.set_xscale("log"); ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"{i} ({u})" for i, u in zip(ct.index, ct.unit)], fontsize=8)
ax.set(xlabel="Odds ratio (log scale), 95% bootstrap interval", title="Logistic regression: association with the dataset's PCOS label"); fig.tight_layout()
fig.savefig(C.FIGS / "fig11_logreg_odds_ratios.png", dpi=300); fig.savefig(C.FIGS / "fig11_logreg_odds_ratios.pdf"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 4)); o = imp.iloc[::-1]
ax[0].barh(o.index, o.mean_abs_shap_test, color="#2a6f97"); ax[0].set(xlabel="Mean |SHAP| (log-odds), test", title="SHAP global importance")
ax[1].barh(o.index, o.perm_auc_drop_test, xerr=o.perm_test_sd, color="#b8782d"); ax[1].set(xlabel="AUC drop when permuted (test)", title="Permutation importance"); ax[1].set_yticklabels([])
fig.tight_layout(); fig.savefig(C.FIGS / "fig12_importance_shap_vs_permutation.png", dpi=300); fig.savefig(C.FIGS / "fig12_importance_shap_vs_permutation.pdf"); plt.close(fig)

# local explanations for four archetypal held-out cases
te2 = test.assign(p=tp.set_index("row_id").loc[test.row_id].final_prob.values); te2["pred"] = (te2.p >= thr).astype(int)
picks = {"True positive": te2[(te2.pred == 1) & (te2.pcos == 1)].sort_values("p").iloc[-1], "False positive": te2[(te2.pred == 1) & (te2.pcos == 0)].sort_values("p").iloc[-1],
         "False negative": te2[(te2.pred == 0) & (te2.pcos == 1)].sort_values("p").iloc[0], "True negative": te2[(te2.pred == 0) & (te2.pcos == 0)].sort_values("p").iloc[0]}
fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
for a, (lab, r) in zip(axs, picks.items()):
    s = pd.Series(sv_te.iloc[test.index[test.row_id == r.row_id][0]].values, index=names).sort_values()
    a.barh(s.index, s.values, color=np.where(s.values > 0, "#c4553b", "#2a6f97")); a.axvline(0, color="k", lw=.8)
    a.set_title(f"{lab}\nP={r.p:.2f} (thr {thr:.2f}); row {int(r.row_id)}", fontsize=9); a.set_xlabel("SHAP (log-odds)")
    if a is not axs[0]: a.set_yticklabels([])
fig.tight_layout(); fig.savefig(C.FIGS / "fig13_local_explanations.png", dpi=300); fig.savefig(C.FIGS / "fig13_local_explanations.pdf"); plt.close(fig)

s2 = sub[sub.auc.notna()]
fig, ax = plt.subplots(figsize=(6, 3.4)); yy = range(len(s2))
ax.errorbar(s2.auc, yy, xerr=[s2.auc - s2.auc_ci_low, s2.auc_ci_high - s2.auc], fmt="o", color="#3f7d4e", capsize=3)
ax.set_yticks(list(yy)); ax.set_yticklabels([f"{g}: {l} (n={n}, pos={p})" for g, l, n, p in zip(s2.grouping, s2.level, s2.n, s2.n_pos)], fontsize=8)
ax.set(xlabel="AUC (95% bootstrap interval)", xlim=(.5, 1), title="Subgroup AUC (pooled train-OOF + test scores)"); fig.tight_layout()
fig.savefig(C.FIGS / "fig14_subgroup_auc.png", dpi=300); fig.savefig(C.FIGS / "fig14_subgroup_auc.pdf"); plt.close(fig)
print("done")
