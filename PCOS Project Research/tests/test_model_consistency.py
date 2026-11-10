"""Evaluation <-> deployment consistency: the app's inference path must reproduce the saved evaluation exactly."""
import json
import numpy as np, pandas as pd, pytest, shap
from sklearn.metrics import roc_auc_score, average_precision_score
from pcos import config as C, data as D, predict as P, split as S, modeling as M

TOL = 1e-9


@pytest.fixture(scope="module")
def bundle(): return P.load_bundle()
@pytest.fixture(scope="module")
def frames():
    df = D.load_processed(); tr, te, _ = S.load_split(df); return df, tr, te
@pytest.fixture(scope="module")
def evalj(): return json.loads((C.METRICS / "final_evaluation.json").read_text())


def test_model_loads_and_hash_matches(bundle):
    model, meta = bundle
    assert D.sha256(C.MODELS / meta["model_file"]) == meta["model_sha256"]
    assert hasattr(model, "predict_proba")


def test_feature_lists_are_consistent_everywhere(bundle):
    _, meta = bundle
    decision = json.loads((C.METRICS / "feature_set_decision.json").read_text())
    assert meta["model_features"] == C.PRIMARY_FEATURES == decision["features"]
    assert meta["input_fields"] == C.INPUT_FIELDS
    assert not set(meta["model_features"]) & (set(C.LAB) | set(C.ULTRASOUND) | set(C.EXCLUDED_BY_DESIGN))   # non-invasive only


def test_processed_data_and_split_unchanged_since_training(bundle):
    import hashlib
    _, meta = bundle
    assert D.sha256(C.PROCESSED) == meta["processed_data_sha256"]
    assert hashlib.sha256(C.SPLIT_FILE.read_bytes()).hexdigest() == meta["split_sha256"]


def test_split_is_disjoint_and_stratified(frames):
    df, tr, te = frames
    assert not set(tr.row_id) & set(te.row_id) and len(tr) + len(te) == len(df)
    assert abs(tr.pcos.mean() - te.pcos.mean()) < 0.02


def test_inference_path_reproduces_saved_test_predictions(bundle, frames):
    model, meta = bundle; _, _, te = frames
    # Rows with a missing input cannot go through the app (it refuses to impute); the training pipeline imputes them with
    # train-fitted values, so those rows are excluded from this app-path comparison only.
    te = te.dropna(subset=C.INPUT_FIELDS)
    saved = pd.read_csv(C.PREDS / "test_predictions.csv").set_index("row_id").loc[te.row_id]
    got = [P.predict({f: r[f] for f in C.INPUT_FIELDS})["probability"] for _, r in te.iterrows()]
    # inputs are the raw fields; BMI is re-derived by the app from height/weight and must match the stored derived BMI
    assert len(got) >= 100 and np.allclose(got, saved.final_prob.values, atol=1e-9)
    assert np.allclose(te.bmi_derived, te.weight_kg / (te.height_cm / 100) ** 2)


def test_decisions_match_saved_threshold(bundle, frames):
    _, meta = bundle; _, _, te = frames
    saved = pd.read_csv(C.PREDS / "test_predictions.csv")
    assert ((saved.final_prob >= meta["threshold"]).astype(int) == saved.final_pred).all()


def test_reported_metrics_recompute_from_saved_predictions(evalj):
    tp = pd.read_csv(C.PREDS / "test_predictions.csv"); fam = evalj["final_model"]
    m = evalj["test_metrics"][fam]
    assert roc_auc_score(tp.y_true, tp.final_prob) == pytest.approx(m["auc"], abs=TOL)
    assert average_precision_score(tp.y_true, tp.final_prob) == pytest.approx(m["pr_auc"], abs=TOL)
    r = M.binary_metrics(tp.y_true, tp.final_prob, evalj["threshold"])
    for k in ["sensitivity", "specificity", "precision", "f1", "accuracy", "tp", "fp", "tn", "fn", "brier"]:
        assert r[k] == pytest.approx(m[k], abs=TOL), k


def test_test_set_labels_in_predictions_match_data(frames):
    _, _, te = frames
    tp = pd.read_csv(C.PREDS / "test_predictions.csv").set_index("row_id").loc[te.row_id]
    assert (tp.y_true.values == te.pcos.values).all()


def test_contributions_are_exact_and_match_shap(bundle, frames):
    model, meta = bundle; _, tr, te = frames
    assert meta["model_family"] == "logreg"
    X = te[C.PRIMARY_FEATURES].head(10)
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    ref = shap.LinearExplainer(clf, shap.maskers.Independent(prep.transform(tr[C.PRIMARY_FEATURES]), max_samples=len(tr)))
    sv = ref.shap_values(prep.transform(X))
    for i in range(len(X)):
        c = P.contributions(model, meta, X.iloc[[i]])
        assert np.allclose([c[n] for n in prep.get_feature_names_out()], sv[i], atol=1e-9)
        logit = np.log(model.predict_proba(X.iloc[[i]])[0, 1] / model.predict_proba(X.iloc[[i]])[0, 0])
        assert sum(c.values()) + P.baseline_log_odds(model, meta) == pytest.approx(logit, abs=1e-9)


def test_feature_order_of_input_dict_is_irrelevant(bundle):
    base = dict(age_years=30, height_cm=160.0, weight_kg=55.0, cycle_irregular=0, weight_gain=1, hair_growth=0, skin_darkening=1,
                hair_loss=0, pimples=0, fast_food=1, regular_exercise=1)
    a = P.predict(base)["probability"]; b = P.predict(dict(reversed(list(base.items()))))["probability"]
    assert a == b


def test_prediction_output_contract():
    r = P.predict(dict(age_years=30, height_cm=160.0, weight_kg=55.0, cycle_irregular=1, weight_gain=1, hair_growth=1, skin_darkening=1,
                       hair_loss=0, pimples=1, fast_food=1, regular_exercise=0))
    assert 0 <= r["probability"] <= 1 and isinstance(r["label_concordant_flag"], bool)
    assert set(r["contributions"]) == set(C.PRIMARY_FEATURES) and r["model_sha256"]
