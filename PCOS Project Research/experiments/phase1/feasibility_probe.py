"""Exploratory feasibility probe (Phase 1 only). Repeated stratified CV on the full
Kerala file with UNTUNED default models, to judge whether a non-invasive feature set
carries signal. NOT final results; no held-out test set is touched or created here."""
import sys, io, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
R = sys.argv[1]
df = pd.read_excel(f"{R}/kerala_kaggle/extracted/PCOS_data_without_infertility.xlsx", sheet_name="Full_new")
df.columns = [c.strip() for c in df.columns]
for c in df.columns: 
    if df[c].dtype==object and c!="Unnamed: 44": df[c]=pd.to_numeric(df[c].astype(str).str.rstrip("."), errors="coerce")
y = df["PCOS (Y/N)"]
print("Row-order / ID leakage check: AUC of Patient File No. =", round(roc_auc_score(y, df["Patient File No."]),3))
print("Label rate by ID decile:", y.groupby(pd.qcut(df["Patient File No."],10)).mean().round(2).tolist())
print("Implausible values: pulse<40:", (df["Pulse rate(bpm)"]<40).sum(), "| SBP<70:", (df["BP _Systolic (mmHg)"]<70).sum(),
      "| DBP<40:", (df["BP _Diastolic (mmHg)"]<40).sum(), "| cycle length==0:", (df["Cycle length(days)"]==0).sum(),
      "| FSH>200:", (df["FSH(mIU/mL)"]>200).sum(), "| LH>200:", (df["LH(mIU/mL)"]>200).sum(), "| VitD3>500:", (df["Vit D3 (ng/mL)"]>500).sum())
print("Cycle(R/I) coding vs label:\n", pd.crosstab(df["Cycle(R/I)"], y))
print("Cycle length vs label:\n", pd.crosstab(df["Cycle length(days)"], y).T.to_string())
sets = {
 "NI_symptoms_cycle_lifestyle": ["Cycle(R/I)","Cycle length(days)","Weight gain(Y/N)","hair growth(Y/N)","Skin darkening (Y/N)","Hair loss(Y/N)","Pimples(Y/N)","Fast food (Y/N)","Reg.Exercise(Y/N)"],
 "NI_full_noninvasive": ["Age (yrs)","Weight (Kg)","Height(Cm)","BMI","Waist(inch)","Hip(inch)","Waist:Hip Ratio","Cycle(R/I)","Cycle length(days)","Marraige Status (Yrs)","Pregnant(Y/N)","No. of aborptions","Weight gain(Y/N)","hair growth(Y/N)","Skin darkening (Y/N)","Hair loss(Y/N)","Pimples(Y/N)","Fast food (Y/N)","Reg.Exercise(Y/N)"],
 "NI_anthropometric_only": ["Age (yrs)","Weight (Kg)","Height(Cm)","BMI","Waist(inch)","Hip(inch)","Waist:Hip Ratio"],
 "Lab_hormones_only": ["FSH(mIU/mL)","LH(mIU/mL)","FSH/LH","TSH (mIU/L)","AMH(ng/mL)","PRL(ng/mL)","PRG(ng/mL)","Vit D3 (ng/mL)","RBS(mg/dl)","Hb(g/dl)"],
 "Ultrasound_only": ["Follicle No. (L)","Follicle No. (R)","Avg. F size (L) (mm)","Avg. F size (R) (mm)","Endometrium (mm)"],
 "All_features": [c for c in df.columns if c not in ["PCOS (Y/N)","Sl. No","Patient File No.","Unnamed: 44"]],
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
models = {"LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)),
          "RandForest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))}
rows=[]
for sn, cols in sets.items():
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    for mn, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=["roc_auc","average_precision","balanced_accuracy"])
        rows.append(dict(feature_set=sn, n_feat=len(cols), model=mn, auc=r["test_roc_auc"].mean(), auc_sd=r["test_roc_auc"].std(),
                         pr_auc=r["test_average_precision"].mean(), bal_acc=r["test_balanced_accuracy"].mean()))
out = pd.DataFrame(rows).round(3); print("\n=== 5x5 repeated stratified CV, untuned defaults (prevalence=%.3f)" % y.mean()); print(out.to_string(index=False))
out.to_csv("results/audit/feasibility_probe_cv.csv", index=False)
# collinearity within non-invasive set
ni = df[sets["NI_full_noninvasive"]].apply(pd.to_numeric, errors="coerce").corr().abs()
pairs = [(a,b,round(ni.loc[a,b],2)) for i,a in enumerate(ni.columns) for b in ni.columns[i+1:] if ni.loc[a,b]>0.7]
print("\nNon-invasive |r|>0.7 pairs:", pairs)
# Infertility csv
inf = pd.read_csv(f"{R}/kerala_kaggle/extracted/PCOS_infertility.csv"); inf.columns=[c.strip() for c in inf.columns]
print("\ninfertility csv: id range", inf["Patient File No."].min(), inf["Patient File No."].max(), "| label rate", inf["PCOS (Y/N)"].mean().round(3), "| shape", inf.shape)
print("overlap of IDs with main:", inf["Patient File No."].isin(df["Patient File No."]).sum())
