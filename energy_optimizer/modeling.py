"""Train and load demand-forecast (regression) and anomaly (classification) models."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from energy_optimizer.config import (
    CLF_MODEL_PATH,
    MODELS_DIR,
    RANDOM_STATE,
    REG_MODEL_PATH,
    ROOT,
)
from energy_optimizer.preprocessing import (
    REG_FEATURE_COLS,
    clf_features,
    prepare_ai4i,
    prepare_ccpp,
)

# Legacy notebook artifacts at project root
LEGACY_REG = ROOT / "best_rf_model.pkl"
LEGACY_CLF = ROOT / "best_rf_classifier.pkl"


def train_regressor(ccpp: pd.DataFrame | None = None) -> RandomForestRegressor:
    data = prepare_ccpp(ccpp)
    x = data[REG_FEATURE_COLS]
    y = data["EnergyOutput"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
    }
    return model, metrics


def _build_classifier_pipeline() -> Pipeline:
    x_cols = clf_features(prepare_ai4i().head(1))
    cat_cols = ["Type"]
    num_cols = [c for c in x_cols.columns if c not in cat_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("type", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_classifier(ai4i: pd.DataFrame | None = None) -> Pipeline:
    data = prepare_ai4i(ai4i)
    x = clf_features(data)
    y = data["Machine failure"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    model = _build_classifier_pipeline()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "report": classification_report(y_test, preds, zero_division=0),
    }
    return model, metrics


def save_models(reg, clf, models_dir: Path = MODELS_DIR) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, REG_MODEL_PATH)
    joblib.dump(clf, CLF_MODEL_PATH)


def _resolve_model_path(primary: Path, legacy: Path) -> Path:
    if primary.exists():
        return primary
    if legacy.exists():
        return legacy
    raise FileNotFoundError(
        f"Model not found at {primary} or {legacy}. Run: python train.py"
    )


def load_regressor() -> RandomForestRegressor:
    path = _resolve_model_path(REG_MODEL_PATH, LEGACY_REG)
    return joblib.load(path)


def load_classifier() -> Pipeline:
    if CLF_MODEL_PATH.exists():
        return joblib.load(CLF_MODEL_PATH)
    if LEGACY_CLF.exists():
        raise FileNotFoundError(
            "Legacy classifier is incompatible. Run: python train.py"
        )
    raise FileNotFoundError(f"Classifier not found. Run: python train.py")


def predict_demand(ccpp_row: pd.DataFrame, model=None) -> pd.Series:
    model = model or load_regressor()
    features = ccpp_row[REG_FEATURE_COLS]
    return pd.Series(model.predict(features), index=ccpp_row.index)


def predict_failure_risk(ai4i_df: pd.DataFrame, model=None) -> pd.Series:
    model = model or load_classifier()
    x = clf_features(ai4i_df)
    if not hasattr(model, "predict_proba"):
        return pd.Series(0.0, index=ai4i_df.index)
    proba = model.predict_proba(x)
    if proba.shape[1] == 2:
        return pd.Series(proba[:, 1], index=ai4i_df.index)
    return pd.Series(model.predict(x), index=ai4i_df.index)
