import numpy as np, pandas as pd, pytest
from pcos import config as C, data as D


def test_raw_archive_hash_matches_recorded_original():
    assert D.sha256(C.RAW_ZIP) == D.EXPECTED_ZIP_SHA256


def test_raw_schema_valid():
    assert D.validate_raw(D.load_raw()) == []


def test_schema_validation_detects_problems():
    raw = D.load_raw()
    assert D.validate_raw(raw.drop(columns=["Age (yrs)"]))
    assert D.validate_raw(raw.head(10))
    bad = raw.copy(); bad.loc[0, "PCOS (Y/N)"] = 3
    assert D.validate_raw(bad)


def test_cleaning_never_drops_rows_and_logs_coercions():
    raw = D.load_raw(); df, log = D.clean(raw)
    assert len(df) == len(raw) == 541
    assert df["bhcg_2"].dtype.kind == "f" and df["amh"].dtype.kind == "f"
    assert ((log.raw_value == "a") & (log.action.str.contains("NaN"))).any()
    assert ((log.raw_value == "1.99.") & (log.new_value == 1.99)).any()
    assert set(df.cycle_irregular.unique()) == {0, 1}
    assert (df.cycle_irregular == (raw["Cycle(R/I)"] != 2)).all()     # 2 -> 0, 4/5 -> 1


def test_cleaning_does_not_alter_target_or_primary_values():
    raw = D.load_raw(); df, _ = D.clean(raw)
    assert (df.pcos.values == raw["PCOS (Y/N)"].values).all()
    assert (df.age_years.values == raw["Age (yrs)"].values).all()
    assert np.allclose(df.weight_kg, raw["Weight (Kg)"])


def test_processed_file_matches_fresh_cleaning():
    df, _ = D.clean(D.load_raw()); p = D.load_processed()
    pd.testing.assert_frame_equal(df.reset_index(drop=True), p.reset_index(drop=True), check_dtype=False, atol=1e-9)


def test_missing_values_in_primary_features_are_few():
    p = D.load_processed()
    assert p[C.PRIMARY_FEATURES].isna().sum().sum() <= 2    # 1 fast_food value; imputed inside the pipeline (train-fitted)
