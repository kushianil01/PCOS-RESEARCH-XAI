import math
import numpy as np
import pytest
from pcos import config as C, predict as P

VALID = dict(age_years=28, height_cm=158.0, weight_kg=60.0, cycle_irregular=1, weight_gain=0, hair_growth=1,
             skin_darkening=0, hair_loss=0, pimples=1, fast_food=0, regular_exercise=0)


def test_valid_input_passes_and_derives_bmi():
    clean, warnings = P.validate_input(VALID)
    assert clean["bmi_derived"] == pytest.approx(60 / 1.58 ** 2)
    assert isinstance(warnings, list)


@pytest.mark.parametrize("field", C.INPUT_FIELDS)
def test_each_missing_field_is_rejected_not_zero_filled(field):
    bad = {k: v for k, v in VALID.items() if k != field}
    with pytest.raises(P.ValidationError) as e:
        P.validate_input(bad)
    assert field in e.value.errors and "missing" in e.value.errors[field]


@pytest.mark.parametrize("field", C.INPUT_FIELDS)
def test_none_and_nan_are_missing(field):
    for v in (None, float("nan")):
        with pytest.raises(P.ValidationError) as e:
            P.validate_input({**VALID, field: v})
        assert field in e.value.errors


@pytest.mark.parametrize("field,value", [("age_years", "28"), ("height_cm", "tall"), ("weight_kg", [60]), ("age_years", True),
                                         ("cycle_irregular", "yes"), ("pimples", 2), ("hair_growth", -1), ("fast_food", 0.5), ("hair_loss", "1")])
def test_invalid_types_and_values_rejected(field, value):
    with pytest.raises(P.ValidationError) as e:
        P.validate_input({**VALID, field: value})
    assert field in e.value.errors


@pytest.mark.parametrize("field,value", [("age_years", 5), ("age_years", 120), ("height_cm", 50), ("height_cm", 300), ("weight_kg", 0),
                                         ("weight_kg", 1000), ("age_years", -3), ("weight_kg", float("inf"))])
def test_out_of_range_rejected(field, value):
    with pytest.raises(P.ValidationError) as e:
        P.validate_input({**VALID, field: value})
    assert field in e.value.errors


def test_zero_is_a_legitimate_binary_answer():
    zeros = {**VALID, **{f: 0 for f in C.BINARY}}
    clean, _ = P.validate_input(zeros)
    assert all(clean[f] == 0 for f in C.BINARY)
    assert P.predict(zeros)["probability"] >= 0     # predicts; zero is not treated as missing


def test_bool_accepted_for_binary_fields_and_numpy_numbers_for_continuous():
    clean, _ = P.validate_input({**VALID, "pimples": True, "age_years": np.int64(30), "weight_kg": np.float64(55.5)})
    assert clean["pimples"] == 1 and clean["age_years"] == 30.0


def test_unknown_field_rejected_so_invasive_inputs_cannot_be_supplied():
    with pytest.raises(P.ValidationError) as e:
        P.validate_input({**VALID, "lh": 5.0, "amh": 3.1})
    assert "lh" in e.value.errors and "amh" in e.value.errors


def test_multiple_errors_reported_together():
    with pytest.raises(P.ValidationError) as e:
        P.validate_input({**VALID, "age_years": "x", "height_cm": None, "pimples": 7})
    assert set(e.value.errors) == {"age_years", "height_cm", "pimples"}


def test_implausible_bmi_rejected():
    with pytest.raises(P.ValidationError):
        P.validate_input({**VALID, "height_cm": 210, "weight_kg": 25})


def test_out_of_training_range_gives_warning_not_error():
    _, w = P.validate_input({**VALID, "age_years": 60})
    assert any("extrapolation" in x for x in w)


def test_predict_rejects_invalid_input_before_touching_model():
    with pytest.raises(P.ValidationError):
        P.predict({**VALID, "weight_kg": None})
