"""Inference path: strict input validation -> the EXACT saved pipeline -> probability, decision at the saved threshold,
exact per-feature contributions. Nothing is trained here and nothing is silently imputed: every input field is required.
"""
from __future__ import annotations
import json, math
from functools import lru_cache
import joblib, numpy as np, pandas as pd
from . import config as C
from .data import sha256

FIELD_LABELS = {
    "age_years": "Age (years)", "height_cm": "Height (cm)", "weight_kg": "Weight (kg)",
    "cycle_irregular": "Menstrual cycle irregular (dataset coding: Cycle R/I)", "weight_gain": "Recent weight gain",
    "hair_growth": "Excess hair growth", "skin_darkening": "Skin darkening", "hair_loss": "Hair loss", "pimples": "Pimples / acne",
    "fast_food": "Regular fast-food consumption", "regular_exercise": "Regular exercise",
}
FEATURE_LABELS = {**FIELD_LABELS, "bmi_derived": "BMI (derived from height and weight, kg/m2)"}
BMI_LIMITS = (10.0, 70.0)


class ValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


@lru_cache(maxsize=1)
def load_bundle():
    meta = json.loads((C.MODELS / "model_metadata.json").read_text())
    path = C.MODELS / meta["model_file"]
    if sha256(path) != meta["model_sha256"]:
        raise RuntimeError("Saved model file does not match the hash recorded at evaluation time; refusing to load.")
    return joblib.load(path), meta


def _is_number(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def validate_input(raw: dict) -> tuple[dict, list[str]]:
    """Return (clean input dict, warnings). Raise ValidationError listing every problem found."""
    errors, warnings, clean = {}, [], {}
    unknown = sorted(set(raw) - set(C.INPUT_FIELDS))
    for u in unknown:
        errors[u] = "unknown field (the system accepts only the documented non-invasive inputs)"
    for f in C.INPUT_FIELDS:
        if f not in raw or raw[f] is None or (isinstance(raw[f], float) and math.isnan(raw[f])):
            errors[f] = "required value is missing"; continue
        v = raw[f]
        if f in C.INPUT_RANGES:
            if not _is_number(v):
                errors[f] = f"must be a number, got {type(v).__name__}"; continue
            if not math.isfinite(float(v)):
                errors[f] = "must be finite"; continue
            lo, hi = C.INPUT_RANGES[f]
            if not lo <= v <= hi:
                errors[f] = f"{v} is outside the allowed range {lo}-{hi}"; continue
            clean[f] = float(v)
        else:  # binary field: 0/1/False/True only (0 is a legitimate answer, not a missing value)
            if isinstance(v, (bool, np.bool_)) or (_is_number(v) and v in (0, 1)):
                clean[f] = int(v)
            else:
                errors[f] = f"must be 0 (No) or 1 (Yes), got {v!r}"
    if errors:
        raise ValidationError(errors)
    bmi = clean["weight_kg"] / (clean["height_cm"] / 100.0) ** 2
    if not BMI_LIMITS[0] <= bmi <= BMI_LIMITS[1]:
        raise ValidationError({"weight_kg": f"height and weight imply an implausible BMI ({bmi:.1f}); check both values"})
    clean["bmi_derived"] = bmi
    _, meta = load_bundle()
    for f, v in [("age_years", clean["age_years"]), ("bmi_derived", bmi), ("height_cm", clean["height_cm"]), ("weight_kg", clean["weight_kg"])]:
        r = meta["training_ranges"][f]
        if not r["p01"] <= v <= r["p99"]:
            warnings.append(f"{FEATURE_LABELS[f]} = {v:.1f} lies outside the central 98% of the training data "
                            f"({r['p01']:.1f}-{r['p99']:.1f}); the prediction is an extrapolation and less reliable.")
    return clean, warnings


def build_frame(clean: dict) -> pd.DataFrame:
    _, meta = load_bundle()
    return pd.DataFrame([{f: clean[f] for f in meta["model_features"]}])


def predict(raw: dict) -> dict:
    clean, warnings = validate_input(raw)
    model, meta = load_bundle()
    X = build_frame(clean)
    p = float(model.predict_proba(X)[:, 1][0])
    contrib = contributions(model, meta, X)
    bin_ = next((b for b in meta["oof_probability_bins"] if b["p_low"] <= p <= b["p_high"]), None)
    return dict(probability=p, label_concordant_flag=bool(p >= meta["threshold"]), threshold=meta["threshold"], contributions=contrib,
                warnings=warnings, calibration_context=bin_, model_sha256=meta["model_sha256"], model_family=meta["model_family"],
                inputs=clean)


def contributions(model, meta, X: pd.DataFrame) -> dict[str, float]:
    """Exact linear SHAP values in log-odds: coef_j * (z_j - mean_train(z_j)). Sum + baseline = model log-odds."""
    if meta["model_family"] != "logreg" or not meta.get("transformed_train_means"):
        return {}
    z = model.named_steps["prep"].transform(X).iloc[0]
    coef = pd.Series(model.named_steps["clf"].coef_[0], index=z.index)
    base = pd.Series(meta["transformed_train_means"])[z.index]
    return {k: float(v) for k, v in (coef * (z - base)).items()}


def baseline_log_odds(model, meta) -> float:
    z0 = pd.Series(meta["transformed_train_means"])
    c = model.named_steps["clf"]
    return float(c.intercept_[0] + (pd.Series(c.coef_[0], index=model.named_steps["prep"].get_feature_names_out()) * z0[model.named_steps["prep"].get_feature_names_out()]).sum())
