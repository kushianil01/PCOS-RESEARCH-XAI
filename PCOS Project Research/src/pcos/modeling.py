"""Pipelines, hyper-parameter grids, CV helpers and metrics. All learned preprocessing lives INSIDE the pipeline so it is
fitted on training folds only."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from . import config as C

LOG_FEATURES = set(C.LAB)  # right-skewed laboratory values are log1p-transformed (comparison experiments only)


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    binary = [f for f in features if f in C.BINARY]
    cont = [f for f in features if f not in C.BINARY]
    logged = [f for f in cont if f in LOG_FEATURES]
    plain = [f for f in cont if f not in LOG_FEATURES]
    tf = []
    if plain:
        tf.append(("cont", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), plain))
    if logged:
        tf.append(("lab", make_pipeline(SimpleImputer(strategy="median"),
                                        FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one"),
                                        StandardScaler()), logged))
    if binary:
        tf.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    return ColumnTransformer(tf, verbose_feature_names_out=False).set_output(transform="pandas")


def _estimator(name: str, seed: int = C.SEED):
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logreg": LogisticRegression(max_iter=5000, random_state=seed),
        "tree": DecisionTreeClassifier(random_state=seed),
        "rf": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "gbm": GradientBoostingClassifier(random_state=seed),
    }[name]


GRIDS = {
    "dummy": {},
    "logreg": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
    "tree": {"clf__max_depth": [2, 3, 4, 5], "clf__min_samples_leaf": [5, 10, 20]},
    "rf": {"clf__max_depth": [3, 5, None], "clf__min_samples_leaf": [1, 3, 5], "clf__max_features": ["sqrt", 0.5]},
    "gbm": {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.03, 0.1], "clf__max_depth": [1, 2, 3]},
}
COMPLEXITY_ORDER = ["logreg", "tree", "rf", "gbm"]   # used by the pre-declared one-standard-error rule


def build_pipeline(name: str, features: list[str], params: dict | None = None) -> Pipeline:
    pipe = Pipeline([("prep", make_preprocessor(features)), ("clf", _estimator(name))])
    if params:
        pipe.set_params(**params)
    return pipe


def cv_splitter(n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS, seed=C.SEED):
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)


def tune(name: str, X: pd.DataFrame, y: pd.Series, features: list[str], n_repeats=C.CV_REPEATS, n_jobs=-1) -> GridSearchCV:
    pipe = build_pipeline(name, features)
    gs = GridSearchCV(pipe, GRIDS[name] or {"clf__strategy": ["prior"]}, scoring="roc_auc",
                      cv=cv_splitter(n_repeats=n_repeats), n_jobs=n_jobs, refit=False, return_train_score=False)
    return gs.fit(X[features], y)


def fold_scores(pipe: Pipeline, X, y, features, cv) -> pd.DataFrame:
    """Per-fold AUC / PR-AUC / Brier for a fixed pipeline (preprocessing refitted per fold)."""
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y)):
        m = clone(pipe).fit(X.iloc[tr][features], y.iloc[tr])
        p = m.predict_proba(X.iloc[va][features])[:, 1]
        rows.append(dict(fold=k, auc=roc_auc_score(y.iloc[va], p), pr_auc=average_precision_score(y.iloc[va], p),
                         brier=brier_score_loss(y.iloc[va], p)))
    return pd.DataFrame(rows)


def oof_predictions(pipe: Pipeline, X, y, features, n_splits=C.CV_FOLDS, n_repeats=C.CV_REPEATS, seed=C.SEED) -> np.ndarray:
    """Out-of-fold probabilities averaged over repeats (every row predicted by models that never saw it)."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    s, c = np.zeros(len(y)), np.zeros(len(y))
    for tr, va in cv.split(X, y):
        m = clone(pipe).fit(X.iloc[tr][features], y.iloc[tr])
        s[va] += m.predict_proba(X.iloc[va][features])[:, 1]; c[va] += 1
    return s / c


# ------------------------------------------------------------------------------------------------------ metrics
def binary_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y); p = np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    prec = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) and not np.isnan(prec + sens) else np.nan
    return dict(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), brier=brier_score_loss(y, p),
                accuracy=(tp + tn) / len(y), balanced_accuracy=(sens + spec) / 2, sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn), n=len(y), threshold=float(thr))


def bootstrap_ci(y, p, thr: float, n_boot=C.N_BOOT, seed=C.SEED) -> dict:
    """Percentile bootstrap CIs (resampling test records; ignores training-set variability)."""
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    keys = ["auc", "pr_auc", "brier", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1"]
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        m = binary_metrics(y[idx], p[idx], thr)
        for k in keys: vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def threshold_for_sensitivity(y, p, target=C.TARGET_SENSITIVITY) -> float:
    """Highest threshold whose sensitivity on (training OOF) data is >= target."""
    y = np.asarray(y); p = np.asarray(p)
    for t in np.unique(p)[::-1]:
        if ((p >= t) & (y == 1)).sum() / (y == 1).sum() >= target:
            return float(t)
    return float(p.min())


def calibration_stats(y, p) -> dict:
    """Calibration intercept/slope (logistic recalibration on logit(p)), ECE (5 quantile bins) and Brier."""
    import statsmodels.api as sm
    y = np.asarray(y); p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p))
    slope = sm.Logit(y, sm.add_constant(z)).fit(disp=0).params
    # intercept with slope fixed at 1 (calibration-in-the-large)
    cil = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=z).fit().params[0]
    bins = pd.qcut(p, 5, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "b": bins}).groupby("b", observed=True).agg(obs=("y", "mean"), pred=("p", "mean"), n=("y", "size"))
    ece = float((g.n * (g.obs - g.pred).abs()).sum() / g.n.sum())
    return dict(calibration_slope=float(slope[1]), calibration_intercept_in_large=float(cil), ece_5bin=ece,
                brier=float(brier_score_loss(y, p)), bins=g.reset_index(drop=True).to_dict("list"))


# ------------------------------------------------------------------------------------------------ rule baselines
def rule_predict(rule: str, X: pd.DataFrame) -> np.ndarray:
    """Transparent, pre-declared rules (no fitted parameters)."""
    irr = X["cycle_irregular"].fillna(0).values
    androgen = (X[["hair_growth", "pimples", "hair_loss"]].fillna(0).sum(axis=1) >= 1).values
    n_sym = X[C.SYMPTOMS].fillna(0).sum(axis=1).values
    return {
        "R1_irregular_cycle": irr == 1,
        "R2_irregular_and_androgen_sign": (irr == 1) & androgen,
        "R3_irregular_or_hair_growth": (irr == 1) | (X["hair_growth"].fillna(0).values == 1),
        "R4_three_or_more_symptoms": n_sym >= 3,
    }[rule].astype(int)


RULES = ["R1_irregular_cycle", "R2_irregular_and_androgen_sign", "R3_irregular_or_hair_growth", "R4_three_or_more_symptoms"]
