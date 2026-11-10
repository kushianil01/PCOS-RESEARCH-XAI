"""Exploratory audit of candidate datasets (read-only; no tuning, no test set)."""
import sys, json, pandas as pd, numpy as np
pd.set_option("display.width", 250, "display.max_columns", 60, "display.max_colwidth", 200)
R = sys.argv[1]
print("=== Kerala: Instructions sheet")
x = pd.ExcelFile(f"{R}/kerala_kaggle/extracted/PCOS_data_without_infertility.xlsx")
print(x.parse("Instructions", header=None).to_string())
df = x.parse("Full_new"); df.columns = [c.strip() for c in df.columns]
print("\n=== shape", df.shape, "| dup rows (excl id cols):", df.drop(columns=["Sl. No","Patient File No."]).duplicated().sum(),
      "| dup patient ids:", df["Patient File No."].duplicated().sum())
print("class balance:", df["PCOS (Y/N)"].value_counts().to_dict())
print("\n=== missing / non-numeric")
print(df.isna().sum()[lambda s: s>0].to_string())
for c in ["II    beta-HCG(mIU/mL)","AMH(ng/mL)"]:
    c2=[k for k in df.columns if k.split("(")[0].strip()==c.split("(")[0].strip()][0]
    bad = pd.to_numeric(df[c2], errors="coerce").isna() & df[c2].notna()
    print(c2, "non-numeric entries:", df.loc[bad, c2].tolist())
print("\n=== describe (selected, check units/range)")
print(df.describe().T[["min","25%","50%","75%","max"]].to_string())
print("\nBlood group codes:", df["Blood Group"].value_counts().to_dict())
print("Cycle(R/I):", df["Cycle(R/I)"].value_counts().to_dict(), "| Cycle length:", df["Cycle length(days)"].value_counts().sort_index().to_dict())
# BMI consistency
bmi = df["Weight (Kg)"]/(df["Height(Cm)"]/100)**2
print("\nBMI recompute max abs diff:", (bmi-df["BMI"]).abs().max(), "| rows diff>0.5:", ((bmi-df["BMI"]).abs()>0.5).sum())
whr = df["Waist(inch)"]/df["Hip(inch)"]
print("WHR recompute rows diff>0.02:", ((whr-df["Waist:Hip Ratio"]).abs()>0.02).sum())
# label construction: follicle rule
fm = np.maximum(df["Follicle No. (L)"], df["Follicle No. (R)"])
for t in [10,12,15]:
    rule=(fm>=t).astype(int); print(f"rule max-follicle>={t}: agreement with label {np.mean(rule==df['PCOS (Y/N)']):.3f}")
print(pd.crosstab(pd.cut(fm,[-1,5,8,11,14,20,40]), df["PCOS (Y/N)"]))
# univariate AUC
from sklearn.metrics import roc_auc_score
y=df["PCOS (Y/N)"]; num=df.drop(columns=["PCOS (Y/N)","Sl. No","Patient File No.","Unnamed: 44"]).apply(pd.to_numeric, errors="coerce")
rows=[]
for c in num: 
    m=num[c].notna(); a=roc_auc_score(y[m], num.loc[m,c]); rows.append((c, round(max(a,1-a),3), int((~m).sum())))
print("\n=== univariate AUC (orientation-free), sorted"); print(pd.DataFrame(rows,columns=["feature","auc","n_missing"]).sort_values("auc",ascending=False).to_string(index=False))
# infertility file consistency
inf=pd.read_csv(f"{R}/kerala_kaggle/extracted/PCOS_infertility.csv"); inf.columns=[c.strip() for c in inf.columns]
m=df.merge(inf,on="Patient File No.",suffixes=("","_inf")); print("\ninfertility csv merge rows:",len(m),"| label agrees:",(m["PCOS (Y/N)"]==m["PCOS (Y/N)_inf"]).mean())
# Figshare
print("\n=== Figshare Elfadil")
f=pd.read_excel(f"{R}/figshare_elfadil/PCOS_DATA.xlsx"); print(f.shape); print(f.head(5).to_string()); print(f.isna().sum().to_dict()); print(f.describe(include="all").T.head(20).to_string())
# Zenodo catalog
print("\n=== Zenodo catalog")
for n in ["infoofdatasets","infooftools"]:
    z=pd.read_excel(f"{R}/zenodo_pcos_catalog/{n}.xlsx", sheet_name=None)
    for s,t in z.items(): print(n,s,t.shape); print(t.head(3).to_string()[:600].encode("ascii","replace").decode())
# Cervical
print("\n=== UCI cervical"); c=pd.read_csv(f"{R}/uci_cervical/extracted/risk_factors_cervical_cancer.csv", na_values="?"); print(c.shape); print(c.isna().mean().round(2)[lambda s:s>0.05].to_dict()); print(c[["Biopsy","Hinselmann","Schiller","Citology"]].mean().round(3).to_dict(), "dups:", c.duplicated().sum())
