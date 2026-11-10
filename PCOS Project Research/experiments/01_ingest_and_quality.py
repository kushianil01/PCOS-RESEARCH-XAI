"""Step 1: verify raw data, validate schema, clean conservatively, write processed data + data-quality artefacts."""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pcos import config as C, data as D

for p in (C.METRICS, C.FIGS, C.LOGS, C.PROCESSED.parent):
    p.mkdir(parents=True, exist_ok=True)

zip_hash = D.sha256(C.RAW_ZIP)
assert zip_hash == D.EXPECTED_ZIP_SHA256, f"raw zip hash mismatch: {zip_hash}"
raw = D.load_raw()
problems = D.validate_raw(raw)
assert not problems, problems

df, clog = D.clean(raw)
flags = D.quality_flags(df)
df.to_csv(C.PROCESSED, index=False)
clog.to_csv(C.METRICS / "cleaning_log.csv", index=False)
flags.to_csv(C.METRICS / "quality_flags.csv", index=False)

y = df[C.TARGET]
prim = C.PRIMARY_FEATURES
summary = {
    "raw_zip_sha256": zip_hash, "n_rows": len(df), "n_columns_processed": df.shape[1],
    "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()), "prevalence": round(float(y.mean()), 4),
    "duplicate_rows_excluding_id": int(df.drop(columns="row_id").duplicated().sum()),
    "missing_cells_nonempty_columns": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
    "n_cleaning_actions": int(len(clog)), "n_quality_flags": int(len(flags)),
    "flags_by_column": flags.groupby("column").size().to_dict() if len(flags) else {},
    "cycle_code_raw_counts": df["cycle_code_raw"].value_counts().to_dict(),
    "primary_feature_summary": df[prim].describe().T[["min", "50%", "max"]].round(2).to_dict("index"),
    "prevalence_by_primary_binary": {c: df.groupby(c)[C.TARGET].mean().round(3).to_dict() for c in C.BINARY if c in df},
}
(C.METRICS / "data_quality_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps({k: v for k, v in summary.items() if k != "primary_feature_summary"}, indent=2, default=str))

# ---- figures --------------------------------------------------------------------------------------------------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(4, 3.2))
ax.bar(["Non-PCOS", "PCOS"], [(1 - y).sum(), y.sum()], color=["#7b8fa6", "#c4553b"])
for i, v in enumerate([(1 - y).sum(), y.sum()]): ax.text(i, v + 5, f"{int(v)} ({v/len(y):.1%})", ha="center")
ax.set(ylabel="Records", title="Class distribution (Kerala cohort, n=541)")
fig.tight_layout(); fig.savefig(C.FIGS / "fig01_class_distribution.png", dpi=300); fig.savefig(C.FIGS / "fig01_class_distribution.pdf"); plt.close(fig)

miss = df.isna().sum(); miss = miss[miss > 0].sort_values()
fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(miss.index, miss.values, color="#7b8fa6"); ax.set(xlabel="Missing values (n)", title="Missing values after cleaning (columns with any)")
fig.tight_layout(); fig.savefig(C.FIGS / "fig02_missingness.png", dpi=300); fig.savefig(C.FIGS / "fig02_missingness.pdf"); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, col, lab in zip(axes, ["age_years", "height_cm", "weight_kg"], ["Age (years)", "Height (cm)", "Weight (kg)"]):
    for v, c, n in [(0, "#7b8fa6", "Non-PCOS"), (1, "#c4553b", "PCOS")]:
        ax.hist(df.loc[y == v, col], bins=18, alpha=.65, color=c, label=n, density=True)
    ax.set(xlabel=lab, ylabel="Density")
axes[0].legend(frameon=False); fig.suptitle("Primary continuous inputs by label", y=1.02)
fig.tight_layout(); fig.savefig(C.FIGS / "fig03_continuous_inputs.png", dpi=300, bbox_inches="tight"); fig.savefig(C.FIGS / "fig03_continuous_inputs.pdf", bbox_inches="tight"); plt.close(fig)

bin_cols = [c for c in C.BINARY if c in df]
rate = pd.DataFrame({c: df.groupby(c)[C.TARGET].agg(["mean", "size"]).T.stack() for c in bin_cols})
fig, ax = plt.subplots(figsize=(7, 3.4))
x = np.arange(len(bin_cols)); w = .38
r0 = [df.loc[df[c] == 0, C.TARGET].mean() for c in bin_cols]; r1 = [df.loc[df[c] == 1, C.TARGET].mean() for c in bin_cols]
ax.bar(x - w/2, r0, w, label="No / regular", color="#7b8fa6"); ax.bar(x + w/2, r1, w, label="Yes / irregular", color="#c4553b")
ax.set_xticks(x); ax.set_xticklabels(bin_cols, rotation=35, ha="right"); ax.set(ylabel="Proportion labelled PCOS", title="Label rate by binary input (unadjusted association)")
ax.legend(frameon=False); fig.tight_layout(); fig.savefig(C.FIGS / "fig04_label_rate_by_binary.png", dpi=300); fig.savefig(C.FIGS / "fig04_label_rate_by_binary.pdf"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
corr = df[C.CONTINUOUS_BASE + ["bmi_derived", "waist_inch", "hip_inch", "whr"] + bin_cols].corr(method="spearman")
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=7); ax.set_yticklabels(corr.columns, fontsize=7)
fig.colorbar(im, label="Spearman rho"); ax.set_title("Feature dependence (candidate non-invasive inputs)")
fig.tight_layout(); fig.savefig(C.FIGS / "fig05_feature_correlation.png", dpi=300); fig.savefig(C.FIGS / "fig05_feature_correlation.pdf"); plt.close(fig)
print("OK: processed data and quality artefacts written")
