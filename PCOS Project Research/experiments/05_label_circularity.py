"""Step 5: label-circularity investigation.

The dataset has no documented label rule. Here we ask, using only the variables present, (a) how well a crude, transparent
Rotterdam-style proxy reproduces the label, (b) whether the non-invasive model's score tracks the ultrasound findings it
never saw, and (c) how label agreement behaves within strata of the withheld follicle count.
All proxies below are OUR approximations (no Ferriman-Gallwey score, no ovarian volume, no hormone thresholds); they are not
the dataset's labelling procedure.
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from pcos import config as C, data as D, split as S, modeling as M

df = D.load_processed(); train, test, sp = S.load_split(df)
oof = pd.read_csv(C.PREDS / "train_oof_predictions.csv"); tp = pd.read_csv(C.PREDS / "test_predictions.csv")
fin = json.loads((C.METRICS / "final_evaluation.json").read_text()); fam = fin["final_model"]
out = {}

# (a) Rotterdam-style proxy on ALL records (descriptive; no fitting)
d = df.copy()
d["crit_oligo"] = d.cycle_irregular
d["crit_androgen"] = ((d.hair_growth + d.pimples + d.hair_loss) >= 1).astype(int)       # crude: hirsutism OR acne OR hair loss
d["crit_pcom"] = ((d[["follicle_l", "follicle_r"]].max(axis=1)) >= 12).astype(int)       # >=12 follicles in either ovary
d["n_crit"] = d[["crit_oligo", "crit_androgen", "crit_pcom"]].sum(axis=1)
tab = pd.crosstab(d.n_crit, d.pcos); tab["pct_labelled_pcos"] = (tab[1] / tab.sum(axis=1)).round(3)
proxy_pred = (d.n_crit >= 2).astype(int)
m = M.binary_metrics(d.pcos, proxy_pred, .5)
out["rotterdam_proxy_n_criteria_vs_label"] = tab.reset_index().to_dict("records")
out["rotterdam_proxy_ge2_vs_label"] = {k: m[k] for k in ["accuracy", "sensitivity", "specificity", "precision", "tp", "fp", "tn", "fn"]}
for name, cols in {"oligo+androgen (non-invasive proxy criteria only)": ["crit_oligo", "crit_androgen"], "pcom_only (ultrasound)": ["crit_pcom"],
                   "oligo only": ["crit_oligo"], "any 2 of 3": ["n_crit"]}.items():
    s = d[cols].sum(axis=1); out.setdefault("proxy_auc_vs_label", {})[name] = round(float(roc_auc_score(d.pcos, s)), 3)
print(tab.to_string()); print(out["rotterdam_proxy_ge2_vs_label"], out["proxy_auc_vs_label"])

# (b) does the non-invasive model's out-of-fold score track the withheld follicle count?
tr = train.merge(oof[["row_id", f"oof_{fam}"]], on="row_id"); tr["fmax"] = tr[["follicle_l", "follicle_r"]].max(axis=1)
te = test.merge(tp[["row_id", "final_prob"]], on="row_id"); te["fmax"] = te[["follicle_l", "follicle_r"]].max(axis=1)
rho_tr = spearmanr(tr[f"oof_{fam}"], tr.fmax); rho_te = spearmanr(te.final_prob, te.fmax)
out["score_vs_follicle_spearman"] = dict(train_oof=[round(rho_tr.statistic, 3), float(rho_tr.pvalue)], test=[round(rho_te.statistic, 3), float(rho_te.pvalue)])

# (c) AUC of the non-invasive score WITHIN follicle-count strata (does it still separate labels when ultrasound is held constant?)
allp = pd.concat([tr.rename(columns={f"oof_{fam}": "score"})[["score", "fmax", "pcos"]], te.rename(columns={"final_prob": "score"})[["score", "fmax", "pcos"]]])
strata = []
for lab, lo, hi in [("<=5", -1, 5), ("6-8", 6, 8), ("9-11", 9, 11), (">=12", 12, 99)]:
    s = allp[(allp.fmax >= lo) & (allp.fmax <= hi)]
    ok = s.pcos.nunique() == 2 and s.pcos.value_counts().min() >= 5
    strata.append(dict(max_follicles=lab, n=len(s), n_label_pos=int(s.pcos.sum()), label_rate=round(s.pcos.mean(), 3),
                       auc_nonInvasive_score=round(float(roc_auc_score(s.pcos, s.score)), 3) if ok else None))
out["auc_within_follicle_strata (train OOF + test scores pooled)"] = strata
print(pd.DataFrame(strata).to_string(index=False))

# (d) how much does the model add over the simple symptom-count rule?  (AUC of symptom count vs model, held-out test)
cnt = test[C.SYMPTOMS + ["cycle_irregular"]].sum(axis=1)
out["test_auc_symptom_count_score"] = round(float(roc_auc_score(test.pcos, cnt)), 3)
out["test_auc_final_model"] = round(float(roc_auc_score(test.pcos, te.set_index('row_id').loc[test.row_id].final_prob)), 3)
(C.METRICS / "label_circularity.json").write_text(json.dumps(out, indent=2, default=str))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
tab[[0, 1]].plot.bar(stacked=True, ax=ax[0], color=["#7b8fa6", "#c4553b"], rot=0); ax[0].set(xlabel="Rotterdam-style proxy criteria met (of 3)", ylabel="Records", title="Label vs crude Rotterdam proxy (all 541)")
ax[0].legend(["Non-PCOS", "PCOS"], frameon=False)
ax[1].scatter(tr.fmax + np.random.default_rng(0).uniform(-.25, .25, len(tr)), tr[f"oof_{fam}"], c=np.where(tr.pcos == 1, "#c4553b", "#7b8fa6"), s=10, alpha=.6)
ax[1].set(xlabel="Max follicle count (WITHHELD from model)", ylabel="Non-invasive OOF probability", title=f"Score vs withheld ultrasound (rho={rho_tr.statistic:.2f})")
fig.tight_layout(); fig.savefig(C.FIGS / "fig10_label_circularity.png", dpi=300); fig.savefig(C.FIGS / "fig10_label_circularity.pdf"); plt.close(fig)
