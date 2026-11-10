"""Step 3: baselines, tuning (repeated CV on TRAIN only), final-model selection, one-shot held-out TEST evaluation,
artefact export. The test split is touched exactly once, after every modelling decision is fixed.

Pre-declared selection rule: among tuned model families, choose the SIMPLEST (order logreg < tree < rf < gbm) whose mean CV
AUC is within one Nadeau-Bengio-corrected standard error of the best family. Threshold: highest threshold that reaches
>= 85% sensitivity on training out-of-fold predictions (screening-oriented; chosen without test data).
"""
import json, hashlib, platform, datetime, sys
import numpy as np, pandas as pd, joblib, sklearn, scipy
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import roc_curve, precision_recall_curve
from pcos import config as C, data as D, split as S, modeling as M

df = D.load_processed()
train, test, sp = S.load_split(df)
F = C.PRIMARY_FEATURES
dec = json.loads((C.METRICS / "feature_set_decision.json").read_text())
assert dec["features"] == F, "config.PRIMARY_FEATURES differs from the feature-set decision"
ytr, yte = train[C.TARGET], test[C.TARGET]
print(f"features ({len(F)}): {F}\ntrain n={len(train)} pos={ytr.sum()} | test n={len(test)} pos={yte.sum()}")

# ---------------------------------------------------------------- baselines (no test access for any choice)
rule_train = []
for r in M.RULES:
    pr = M.rule_predict(r, train)
    m = M.binary_metrics(ytr, pr, 0.5); rule_train.append(dict(rule=r, **{k: m[k] for k in ["balanced_accuracy", "sensitivity", "specificity", "precision"]}))
rule_train = pd.DataFrame(rule_train)
best_rule = rule_train.sort_values("balanced_accuracy", ascending=False).iloc[0]["rule"]
print("rule baselines (TRAIN):\n", rule_train.round(3).to_string(index=False), "\n-> selected rule:", best_rule)

# ---------------------------------------------------------------- tuning on train
k = C.CV_FOLDS * C.CV_REPEATS
nb_factor = np.sqrt(1 / k + (1 / C.CV_FOLDS) / (1 - 1 / C.CV_FOLDS))
fam, grid_rows = {}, []
for name in ["logreg", "tree", "rf", "gbm"]:
    gs = M.tune(name, train, ytr, F)
    cvr = pd.DataFrame(gs.cv_results_)
    best = cvr.loc[cvr["rank_test_score"].idxmin()]
    # per-fold SD of the best configuration
    sd = float(np.std([best[f"split{i}_test_score"] for i in range(k)], ddof=1))
    fam[name] = dict(params=best["params"], auc_mean=float(best["mean_test_score"]), auc_sd=sd, se_nb=sd * nb_factor)
    cvr["family"] = name; grid_rows.append(cvr[["family", "params", "mean_test_score", "std_test_score", "rank_test_score"]])
    print(f"{name:7s} best CV AUC={fam[name]['auc_mean']:.4f} sd={sd:.4f} params={fam[name]['params']}")
pd.concat(grid_rows).astype({"params": str}).to_csv(C.METRICS / "cv_grid_results.csv", index=False)

best_fam = max(fam, key=lambda n: fam[n]["auc_mean"])
thr_auc = fam[best_fam]["auc_mean"] - fam[best_fam]["se_nb"]
final_name = next(n for n in M.COMPLEXITY_ORDER if fam[n]["auc_mean"] >= thr_auc)
print(f"best family={best_fam}; 1-SE(NB) cutoff AUC={thr_auc:.4f}; FINAL (simplest within cutoff)={final_name}")

# ---------------------------------------------------------------- fit on train, OOF predictions, thresholds
fitted, oof, thr = {}, {}, {}
for name in fam:
    pipe = M.build_pipeline(name, F, fam[name]["params"])
    oof[name] = M.oof_predictions(pipe, train, ytr, F)
    thr[name] = M.threshold_for_sensitivity(ytr, oof[name])
    fitted[name] = clone(pipe).fit(train[F], ytr)
dummy = M.build_pipeline("dummy", F).fit(train[F], ytr)
oof_dummy = M.oof_predictions(M.build_pipeline("dummy", F), train, ytr, F, n_repeats=1)

# ---------------------------------------------------------------- one-shot TEST evaluation
test_pred = pd.DataFrame({"row_id": test.row_id, "y_true": yte})
results = {}
for name in fam:
    p = fitted[name].predict_proba(test[F])[:, 1]; test_pred[f"p_{name}"] = p
    m = M.binary_metrics(yte, p, thr[name]); m["ci95"] = M.bootstrap_ci(yte, p, thr[name]); m["cv_auc_train"] = fam[name]["auc_mean"]
    results[name] = m
pd_ = dummy.predict_proba(test[F])[:, 1]; test_pred["p_dummy"] = pd_
results["dummy"] = {**M.binary_metrics(yte, pd_, float(ytr.mean())), "note": "prior-probability classifier; threshold = prevalence"}
for r in M.RULES:
    pr = M.rule_predict(r, test); test_pred[f"rule_{r}"] = pr
    results[r] = {**M.binary_metrics(yte, pr, 0.5), "selected_on_train": r == best_rule}
final_p = test_pred[f"p_{final_name}"].values
test_pred["final_prob"] = final_p; test_pred["final_pred"] = (final_p >= thr[final_name]).astype(int)
test_pred.to_csv(C.PREDS / "test_predictions.csv", index=False)
oof_df = pd.DataFrame({"row_id": train.row_id, "y_true": ytr, **{f"oof_{n}": oof[n] for n in oof}, "oof_dummy": oof_dummy})
oof_df.to_csv(C.PREDS / "train_oof_predictions.csv", index=False)

cal_test = M.calibration_stats(yte, final_p); cal_oof = M.calibration_stats(ytr, oof[final_name])
print("\nTEST (held-out), final model:", {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in results[final_name].items() if k_ != "ci95"})
print("95% CI:", {k_: [round(x, 3) for x in v] for k_, v in results[final_name]["ci95"].items()})

# ---------------------------------------------------------------- export model + metadata
C.MODELS.mkdir(exist_ok=True)
model_path = C.MODELS / "pcos_primary_model.joblib"
joblib.dump(fitted[final_name], model_path)
rng = {c: dict(min=float(train[c].min()), p01=float(train[c].quantile(.01)), p99=float(train[c].quantile(.99)), max=float(train[c].max()))
       for c in ["age_years", "height_cm", "weight_kg", "bmi_derived"]}
oof_final = oof[final_name]; bins = pd.qcut(oof_final, 5, duplicates="drop")
oof_bins = pd.DataFrame({"y": ytr.values, "p": oof_final, "b": bins}).groupby("b", observed=True).agg(
    p_low=("p", "min"), p_high=("p", "max"), n=("y", "size"), observed_rate=("y", "mean")).reset_index(drop=True).round(4).to_dict("records")
meta = dict(
    model_family=final_name, model_params={k_: (None if v is None else v) for k_, v in fam[final_name]["params"].items()},
    model_file=model_path.name, model_sha256=D.sha256(model_path), model_features=F, input_fields=C.INPUT_FIELDS,
    threshold=thr[final_name], threshold_rule=f"highest threshold with >= {C.TARGET_SENSITIVITY:.0%} sensitivity on training out-of-fold predictions",
    training_ranges=rng, oof_probability_bins=oof_bins, split_sha256=hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest(),
    processed_data_sha256=D.sha256(C.PROCESSED), seed=C.SEED,
    versions=dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__, scipy=scipy.__version__),
    created=datetime.datetime.now().isoformat(timespec="seconds"),
    cycle_irregular_coding="inferred (2=regular, 4/5=irregular); original codebook not available",
    # mean of the transformed training features: baseline for exact linear SHAP contributions at inference (logreg only)
    transformed_train_means=(fitted[final_name].named_steps["prep"].transform(train[F]).mean().to_dict() if final_name == "logreg" else None),
)
(C.MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2))

final = dict(final_model=final_name, selection_rule=__doc__.strip().split("\n\n")[1].replace("\n", " "), features=F, threshold=thr[final_name],
             cv_summary_train=fam, nb_se_factor=nb_factor, one_se_cutoff=thr_auc,
             test_metrics=results, calibration_test=cal_test, calibration_train_oof=cal_oof,
             rule_baselines_train=rule_train.round(4).to_dict("records"), selected_rule=best_rule,
             split=dict(n_train=sp["n_train"], n_test=sp["n_test"], train_pos=sp["train_pos"], test_pos=sp["test_pos"]),
             model_sha256=meta["model_sha256"])
(C.METRICS / "final_evaluation.json").write_text(json.dumps(final, indent=2, default=str))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
cols = {"logreg": "#2a6f97", "tree": "#8a8a8a", "rf": "#3f7d4e", "gbm": "#b8782d"}
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
for n in fam:
    fpr, tpr, _ = roc_curve(yte, test_pred[f"p_{n}"]); ax[0].plot(fpr, tpr, color=cols[n], lw=2.4 if n == final_name else 1.2, label=f"{n} (AUC {results[n]['auc']:.3f})")
    pr, rc, _ = precision_recall_curve(yte, test_pred[f"p_{n}"]); ax[1].plot(rc, pr, color=cols[n], lw=2.4 if n == final_name else 1.2, label=f"{n} (AP {results[n]['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[1].axhline(yte.mean(), color="k", ls=":", lw=1)
ax[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="ROC, held-out test (n=109)"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall, held-out test")
ax[0].legend(frameon=False, fontsize=8, loc="lower right"); ax[1].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(C.FIGS / "fig06_roc_pr_test.png", dpi=300); fig.savefig(C.FIGS / "fig06_roc_pr_test.pdf"); plt.close(fig)

m = results[final_name]; cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
fig, ax = plt.subplots(figsize=(3.8, 3.4)); ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["Pred non-PCOS", "Pred PCOS"]); ax.set_yticklabels(["Label non-PCOS", "Label PCOS"])
ax.set_title(f"{final_name}: held-out test, thr={thr[final_name]:.2f}", fontsize=9); fig.tight_layout()
fig.savefig(C.FIGS / "fig07_confusion_matrix_test.png", dpi=300); fig.savefig(C.FIGS / "fig07_confusion_matrix_test.pdf"); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.8))
for a, (lab, cal) in zip(ax, [("Training OOF (CV)", cal_oof), ("Held-out test", cal_test)]):
    b = cal["bins"]; a.plot([0, 1], [0, 1], "k:", lw=1); a.plot(b["pred"], b["obs"], "o-", color=cols[final_name])
    for x_, y_, n_ in zip(b["pred"], b["obs"], b["n"]): a.annotate(f"n={n_}", (x_, y_), textcoords="offset points", xytext=(4, -10), fontsize=7)
    a.set(xlabel="Mean predicted probability", ylabel="Observed PCOS-label rate", title=f"Calibration: {lab}\nslope={cal['calibration_slope']:.2f}, Brier={cal['brier']:.3f}", xlim=(0, 1), ylim=(0, 1))
fig.tight_layout(); fig.savefig(C.FIGS / "fig08_calibration.png", dpi=300); fig.savefig(C.FIGS / "fig08_calibration.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.4))
names = list(fam); ax.barh(names, [fam[n]["auc_mean"] for n in names], xerr=[fam[n]["auc_sd"] for n in names], color=[cols[n] for n in names], alpha=.85)
ax.axvline(0.5, color="k", ls=":", lw=1); ax.set(xlim=(0.5, 1), xlabel="Mean CV AUC on training split (5x5 repeated; bars = fold SD)", title="Model-family comparison (CV, selection stage)")
fig.tight_layout(); fig.savefig(C.FIGS / "fig09_cv_model_comparison.png", dpi=300); fig.savefig(C.FIGS / "fig09_cv_model_comparison.pdf"); plt.close(fig)
print("done")
