"""Step 7: figures for the nested-CV ablation and the (non-deployable) laboratory/ultrasound comparison."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pcos import config as C

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
r = pd.read_csv(C.METRICS / "ablation_comparison_nested_cv.csv")
for group, fname, title, color in [("ablation", "fig15_ablation_nested_cv", "Symptom / feature ablation (non-invasive inputs)", "#2a6f97"),
                                   ("comparison_NON_deployable", "fig16_lab_ultrasound_comparison", "Comparison only: laboratory / ultrasound inputs (never app inputs)", "#b8782d")]:
    g = r[r.group == group] if group != "comparison_NON_deployable" else pd.concat([r[r.group == group], r[r.feature_set == "primary_full"]])
    p = g.pivot(index="feature_set", columns="model", values="auc_mean"); sd = g.pivot(index="feature_set", columns="model", values="auc_sd")
    p = p.sort_values("logreg"); sd = sd.loc[p.index]
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(p) + 1.4)); y = np.arange(len(p)); h = .38
    ax.barh(y - h / 2, p.logreg, h, xerr=sd.logreg, color=color, label="Logistic regression")
    ax.barh(y + h / 2, p.rf, h, xerr=sd.rf, color="#8a8a8a", label="Random forest")
    ax.set_yticks(y); ax.set_yticklabels(p.index); ax.axvline(0.5, color="k", ls=":", lw=1); ax.set_xlim(0.5, 1.0)
    ax.set(xlabel="Nested-CV AUC on all 541 records (outer 5x3; bars = fold SD)", title=title); ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(C.FIGS / f"{fname}.png", dpi=300); fig.savefig(C.FIGS / f"{fname}.pdf"); plt.close(fig)
print("done")
