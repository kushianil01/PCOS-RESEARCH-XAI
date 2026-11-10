"""Ingestion, schema validation and conservative cleaning of the Kerala PCOS workbook.

Principle: never delete rows, never overwrite values to improve results. Every coercion is written to a cleaning log;
implausible values are flagged, not altered.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from . import config as C

EXPECTED_ROWS = 541
EXPECTED_ZIP_SHA256 = "380ce7ff504cb8aa89349a251c64841a09aadbfe410627df94e6a62e67a23891"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(C.RAW_XLSX, sheet_name=C.RAW_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_raw(df: pd.DataFrame) -> list[str]:
    """Return a list of schema problems (empty list = OK)."""
    problems = []
    expected = set(C.RAW_TO_CANON) | set(C.DROP_RAW)
    missing = sorted(expected - set(df.columns))
    extra = sorted(set(df.columns) - expected)
    if missing: problems.append(f"missing columns: {missing}")
    if extra: problems.append(f"unexpected columns: {extra}")
    if len(df) != EXPECTED_ROWS: problems.append(f"expected {EXPECTED_ROWS} rows, found {len(df)}")
    if "PCOS (Y/N)" in df and not set(df["PCOS (Y/N)"].dropna().unique()) <= {0, 1}:
        problems.append("target not binary 0/1")
    return problems


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clean dataframe, cleaning log)."""
    log: list[dict] = []
    df = raw.drop(columns=[c for c in C.DROP_RAW if c in raw]).rename(columns=C.RAW_TO_CANON).copy()

    # 1. numeric coercion of object columns (typos such as '1.99.' or 'a')
    for col in df.columns:
        if df[col].dtype == object:
            for idx, val in df[col].items():
                if isinstance(val, str):
                    s = val.strip()
                    try:
                        new = float(s)
                    except ValueError:
                        try:
                            new = float(s.rstrip("."))
                            action = "stripped trailing '.'"
                        except ValueError:
                            new, action = np.nan, "unparseable -> NaN"
                    else:
                        action = "string -> float"
                    log.append(dict(row_id=int(df.loc[idx, "row_id"]), column=col, raw_value=val, new_value=new, action=action))
                    df.loc[idx, col] = new
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. cycle regularity recoding (INFERRED coding; see docs/DATASET_CARD.md): 2 -> regular, 4/5 -> irregular
    df["cycle_irregular"] = df["cycle_code_raw"].map({2: 0, 4: 1, 5: 1})
    n_five = int((df["cycle_code_raw"] == 5).sum())
    log.append(dict(row_id=-1, column="cycle_irregular", raw_value="2/4/5", new_value="0/1/1",
                    action=f"inferred coding (2=regular, 4 and 5=irregular); {n_five} row(s) with code 5"))

    # 3. derived BMI from height/weight (reported BMI kept as bmi_reported)
    df["bmi_derived"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
    for _, r in df[(df["bmi_derived"] - df["bmi_reported"]).abs() > 0.5].iterrows():
        log.append(dict(row_id=int(r.row_id), column="bmi_reported", raw_value=r.bmi_reported, new_value=r.bmi_derived,
                        action="FLAG only: reported BMI differs from weight/height by >0.5; derived BMI stored separately"))

    # 4. Y/N columns must be 0/1
    for col in C.SYMPTOMS + C.LIFESTYLE + ["pregnant", "pcos"]:
        bad = ~df[col].isin([0, 1]) & df[col].notna()
        if bad.any():
            raise ValueError(f"{col} has non-binary values: {df.loc[bad, col].unique()}")
    return df, pd.DataFrame(log)


# plausibility flags: (column, test, description). Flag only; values are never changed.
_FLAGS = [
    ("pulse_bpm", lambda s: s < 40, "pulse < 40 bpm (excluded from model by design)"),
    ("bp_sys", lambda s: s < 70, "systolic BP < 70 (likely entry error; excluded by design)"),
    ("bp_dia", lambda s: s < 40, "diastolic BP < 40 (likely entry error; excluded by design)"),
    ("cycle_length_days", lambda s: s < 1, "cycle length 0 days"),
    ("fsh", lambda s: s > 200, "FSH > 200 mIU/mL (implausible/unit error?)"),
    ("lh", lambda s: s > 200, "LH > 200 mIU/mL (implausible/unit error?)"),
    ("vit_d3", lambda s: s > 500, "Vit D3 > 500 ng/mL (implausible)"),
    ("bmi_derived", lambda s: (s < 14) | (s > 50), "derived BMI outside 14-50"),
    ("weight_kg", lambda s: (s < 35) | (s > 150), "weight outside 35-150 kg"),
    ("height_cm", lambda s: (s < 135) | (s > 190), "height outside 135-190 cm"),
]


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, test, desc in _FLAGS:
        for _, r in df[test(df[col])].iterrows():
            rows.append(dict(row_id=int(r.row_id), column=col, value=r[col], flag=desc))
    return pd.DataFrame(rows)


def load_processed() -> pd.DataFrame:
    return pd.read_csv(C.PROCESSED)
