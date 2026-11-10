import sys, pandas as pd
d = sys.argv[1]
x = pd.ExcelFile(f"{d}/PCOS_data_without_infertility.xlsx")
print(x.sheet_names)
for s in x.sheet_names:
    df = x.parse(s); print(s, df.shape)
df = x.parse("Full_new")
print(df.dtypes.to_string())
print(df.head(3).T.to_string())
inf = pd.read_csv(f"{d}/PCOS_infertility.csv"); print(inf.shape, list(inf.columns))
