"""Research-prototype interface. Every number shown is loaded from files produced by experiments/*.py; the prediction page uses
the saved, hash-checked pipeline via pcos.predict (no model is trained here). Nothing entered is stored or logged."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, streamlit as st
from pcos import config as C
from pcos import predict as P

st.set_page_config(page_title="PCOS Label-Concordance Screening Support (research prototype)", layout="wide")
M, FIG = C.METRICS, C.FIGS


@st.cache_data
def jload(name): return json.loads((M / name).read_text())
@st.cache_data
def cload(name, **kw): return pd.read_csv(M / name, **kw)
def fig(name, caption):
    st.image(str(FIG / name), caption=caption, use_container_width=True)


BANNER = ("RESEARCH PROTOTYPE - not a diagnostic or treatment tool and not clinically validated. It estimates how well a person's "
          "non-invasive answers resemble records labelled PCOS in one Kerala dataset.")
st.sidebar.title("PCOS screening-support prototype")
page = st.sidebar.radio("Section", ["Overview", "Dataset & data quality", "Model evaluation", "Prediction", "Explainability", "Responsible use"])
st.sidebar.warning(BANNER)

try:
    ev, meta = jload("final_evaluation.json"), json.loads((C.MODELS / "model_metadata.json").read_text())
except FileNotFoundError:
    st.error("Result files not found. Run the experiments first (see REPRODUCIBILITY.md)."); st.stop()

# ------------------------------------------------------------------------------------------------------------------
if page == "Overview":
    st.title("Explainable, non-invasive PCOS label-concordance screening support")
    st.caption(BANNER)
    st.markdown("""
**Problem.** Public PCOS machine-learning work often depends on blood hormones and ultrasound. This prototype asks how far
*non-invasive, self-reportable* information (age, BMI, cycle regularity, five symptoms, two lifestyle items) can reproduce the PCOS
label in a single documented dataset, and how transparent, calibrated and robust such a model is.

**Intended users.** Researchers and students studying explainable decision support. It is **not** for patients to self-diagnose
and **not** for clinicians to make decisions.

**Data-to-decision workflow.**
1. Raw Kerala workbook (hash-verified) -> schema validation -> conservative cleaning (nothing deleted or altered to help results).
2. Frozen stratified train/test split. Feature redundancy and model selection use the training split only.
3. Logistic regression chosen by a pre-declared one-standard-error rule; threshold fixed from training out-of-fold predictions.
4. One-shot held-out test evaluation; ablation, calibration, subgroup, error and robustness analyses.
5. This application loads the *same* saved pipeline and threshold, validates inputs, and shows an exact per-feature explanation.
""")
    c = st.columns(4); t = ev["test_metrics"][ev["final_model"]]
    c[0].metric("Records", f"{ev['split']['n_train'] + ev['split']['n_test']}")
    c[1].metric("Model", ev["final_model"].replace("logreg", "Logistic regression"))
    c[2].metric("Held-out test AUC", f"{t['auc']:.3f}", help="n=109 test records; see Model evaluation for intervals")
    c[3].metric("Inputs", f"{len(C.INPUT_FIELDS)} non-invasive")
    st.info("The target is the dataset's existing PCOS label, whose construction is undocumented. Results describe label concordance, not diagnostic accuracy.")

# ------------------------------------------------------------------------------------------------------------------
elif page == "Dataset & data quality":
    q = jload("data_quality_summary.json")
    st.title("Dataset and data quality")
    st.markdown("**Source:** Kottarathil (2020), *Polycystic ovary syndrome (PCOS)*, Kaggle, described as collected from 10 hospitals in Kerala, India. "
                "**License:** CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). See DATASET_CARD.md.")
    c = st.columns(4)
    c[0].metric("Rows", q["n_rows"]); c[1].metric("PCOS-labelled", f"{q['n_positive']} ({q['prevalence']:.1%})")
    c[2].metric("Duplicate rows", q["duplicate_rows_excluding_id"]); c[3].metric("Flagged implausible values", q["n_quality_flags"])
    st.subheader("Findings")
    st.markdown(f"""
- Missing values after cleaning (non-empty columns): `{q['missing_cells_nonempty_columns']}`; {q['n_cleaning_actions']} cleaning actions are logged in `results/metrics/cleaning_log.csv`.
- Flagged (not altered) values by column: `{q['flags_by_column']}`.
- Cycle regularity is coded 2/4/5 in the file (`{q['cycle_code_raw_counts']}`); its meaning is **inferred** (2 = regular, 4/5 = irregular) from secondary reports, not an original codebook.
- Several variables (cycle length, symptom definitions) are undocumented; cycle length (median 5 days) is excluded because its meaning is unclear.
- Single regional cohort with no external validation set.""")
    a, b = st.columns(2)
    with a: fig("fig01_class_distribution.png", "Class distribution")
    with b: fig("fig02_missingness.png", "Missing values after cleaning")
    a, b = st.columns(2)
    with a: fig("fig04_label_rate_by_binary.png", "Unadjusted label rate by binary input")
    with b: fig("fig05_feature_correlation.png", "Dependence among candidate inputs")
    fig("fig03_continuous_inputs.png", "Continuous inputs by label")
    st.subheader("Cleaning log"); st.dataframe(cload("cleaning_log.csv"), use_container_width=True)
    st.subheader("Flagged values"); st.dataframe(cload("quality_flags.csv"), use_container_width=True)

# ------------------------------------------------------------------------------------------------------------------
elif page == "Model evaluation":
    st.title("Model evaluation")
    fn = ev["final_model"]; t = ev["test_metrics"][fn]; ci = t["ci95"]
    st.subheader("A. Held-out test (n=109, 36 labelled PCOS) - final model, evaluated once")
    cols = st.columns(5)
    for col, (k, lab) in zip(cols, [("auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("brier", "Brier")]):
        col.metric(lab, f"{t[k]:.3f}", help=f"95% bootstrap CI {ci[k][0]:.3f}-{ci[k][1]:.3f}"); col.caption(f"95% CI {ci[k][0]:.3f}-{ci[k][1]:.3f}")
    st.caption(f"Decision threshold {ev['threshold']:.3f} was fixed from training out-of-fold predictions (>=85% sensitivity). CIs reflect test-sample variability only.")
    rows = []
    for name, m in ev["test_metrics"].items():
        rows.append(dict(model=name, auc=m.get("auc"), pr_auc=m.get("pr_auc"), sensitivity=m["sensitivity"], specificity=m["specificity"], precision=m["precision"], f1=m["f1"],
                         brier=m.get("brier"), note="selected by CV" if name == fn else ("rule: selected on train" if m.get("selected_on_train") else "")))
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True)
    st.caption("Rows other than the final model are shown for comparison only; none was used for selection. Rules and the dummy classifier are baselines.")
    a, b = st.columns(2)
    with a: fig("fig06_roc_pr_test.png", "ROC and precision-recall curves, held-out test")
    with b:
        fig("fig07_confusion_matrix_test.png", "Confusion matrix, held-out test")
    st.subheader("B. Cross-validation on the training split (selection stage; NOT final performance)")
    cv = pd.DataFrame(ev["cv_summary_train"]).T[["auc_mean", "auc_sd"]].round(3); st.dataframe(cv, use_container_width=True)
    fig("fig09_cv_model_comparison.png", "5x5 repeated stratified CV on the training split")
    st.subheader("C. Calibration"); fig("fig08_calibration.png", "Calibration: training OOF and held-out test")
    st.subheader("D. Ablation and separate comparison experiments (nested CV on all 541 records)")
    ab = cload("ablation_comparison_nested_cv.csv"); st.dataframe(ab[ab.model == "logreg"].drop(columns="model").round(3), use_container_width=True)
    st.caption("Groups tagged 'comparison_NON_deployable' use laboratory/ultrasound inputs and are never inputs of this application.")
    st.subheader("E. Label circularity"); lc = jload("label_circularity.json"); st.json(lc["rotterdam_proxy_ge2_vs_label"])
    fig("fig10_label_circularity.png", "Label vs a crude Rotterdam-style proxy; score vs withheld follicle count")

# ------------------------------------------------------------------------------------------------------------------
elif page == "Prediction":
    st.title("Prediction (research prototype)")
    st.warning(BANNER)
    st.caption("All fields are required. Nothing is stored. No names, dates of birth, locations or other identifiers are requested.")
    with st.form("pred"):
        c = st.columns(3)
        age = c[0].number_input(P.FIELD_LABELS["age_years"], value=None, step=1.0, placeholder="e.g. 28")
        height = c[1].number_input(P.FIELD_LABELS["height_cm"], value=None, step=0.5, placeholder="e.g. 158")
        weight = c[2].number_input(P.FIELD_LABELS["weight_kg"], value=None, step=0.5, placeholder="e.g. 60")
        yn = {}
        cc = st.columns(3)
        for i, f in enumerate(C.INPUT_FIELDS[3:]):
            opts = ["Irregular", "Regular"] if f == "cycle_irregular" else ["Yes", "No"]
            yn[f] = cc[i % 3].selectbox(P.FIELD_LABELS[f], opts, index=None, placeholder="Select...")
        go = st.form_submit_button("Estimate")
    if go:
        raw = {"age_years": age, "height_cm": height, "weight_kg": weight}
        for f, v in yn.items():
            raw[f] = None if v is None else int(v in ("Yes", "Irregular"))
        try:
            res = P.predict(raw)
        except P.ValidationError as e:
            st.error("Please correct the following:")
            for k, m in e.errors.items(): st.write(f"- **{P.FIELD_LABELS.get(k, k)}**: {m}")
        else:
            for w in res["warnings"]: st.warning(w)
            p = res["probability"]
            c = st.columns(3)
            c[0].metric("Model score (probability of PCOS-label concordance)", f"{p:.0%}")
            c[1].metric("Screening threshold", f"{res['threshold']:.0%}")
            c[2].metric("Flag", "Above threshold" if res["label_concordant_flag"] else "Below threshold")
            st.markdown("**How to read this.** The score estimates how much these answers resemble records the dataset labels PCOS. "
                        "It is not a diagnosis. The threshold was chosen to catch about 85% of labelled cases in training data, which also produces many false alarms.")
            b = res["calibration_context"]
            if b:
                st.info(f"Calibration context: among {b['n']} training records (out-of-fold) with scores between {b['p_low']:.0%} and {b['p_high']:.0%}, "
                        f"{b['observed_rate']:.0%} carried the PCOS label. This is one small, single-region cohort.")
            st.subheader("Why this score? (exact additive contributions, log-odds)")
            ctr = pd.Series(res["contributions"]).sort_values()
            f_, ax = plt.subplots(figsize=(7, 3.6))
            ax.barh([P.FEATURE_LABELS.get(i, i) for i in ctr.index], ctr.values, color=np.where(ctr.values > 0, "#c4553b", "#2a6f97")); ax.axvline(0, color="k", lw=.8)
            ax.set_xlabel("Contribution relative to the average training record (log-odds)"); ax.spines[["top", "right"]].set_visible(False); st.pyplot(f_)
            st.caption("Red raises and blue lowers the score relative to the average training record. These are associations learned from one dataset, not causes, "
                       "and correlated inputs (e.g. several symptoms) share credit arbitrarily.")
            st.caption(f"Model: {res['model_family']} | sha256 {res['model_sha256'][:12]}... (matches the evaluated model)")

# ------------------------------------------------------------------------------------------------------------------
elif page == "Explainability":
    ex = jload("explainability_summary.json")
    st.title("Explainability")
    st.markdown("""
**Global** explanations describe the whole fitted model; **local** explanations describe one prediction. Methods: logistic-regression odds ratios with
bootstrap intervals, permutation importance (AUC drop), and exact linear SHAP. None of them shows causation, and none can separate correlated inputs reliably.""")
    st.subheader("Odds ratios"); st.dataframe(cload("explain_logreg_coefficients.csv", index_col=0).round(3), use_container_width=True)
    fig("fig11_logreg_odds_ratios.png", "Odds ratios (training split, bootstrap 95% interval)")
    st.subheader("Global importance"); st.dataframe(cload("explain_importance_table.csv", index_col=0), use_container_width=True)
    fig("fig12_importance_shap_vs_permutation.png", "SHAP vs permutation importance")
    st.markdown(f"Top-2 features carry {ex['top2_share_of_mean_abs_shap_test']:.0%} of total mean |SHAP|; rank agreement (Spearman) between methods: `{ex['rank_agreement_spearman']}`.")
    st.subheader("Local examples (held-out test)"); fig("fig13_local_explanations.png", "SHAP for a true positive, false positive, false negative and true negative")
    st.subheader("Subgroups (pooled out-of-fold + test scores)"); st.dataframe(cload("subgroup_evaluation.csv"), use_container_width=True)
    fig("fig14_subgroup_auc.png", "Subgroup AUC")
    st.subheader("Robustness"); st.dataframe(cload("robustness_test.csv"), use_container_width=True)
    st.subheader("Error analysis"); st.dataframe(cload("error_analysis.csv"), use_container_width=True)

# ------------------------------------------------------------------------------------------------------------------
else:
    st.title("Responsible use and limitations")
    st.error(BANNER)
    st.markdown(open(C.ROOT / "docs/LIMITATIONS_AND_ETHICS.md", encoding="utf-8").read() if (C.ROOT / "docs/LIMITATIONS_AND_ETHICS.md").exists()
                else "See LIMITATIONS_AND_ETHICS.md.")
