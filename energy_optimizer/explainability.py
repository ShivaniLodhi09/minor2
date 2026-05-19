"""SHAP-based explainability for regression and classification models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from energy_optimizer.modeling import load_classifier, load_regressor
from energy_optimizer.preprocessing import REG_FEATURE_COLS, clf_features, prepare_ai4i, prepare_ccpp

SHAP_SAMPLE = 200


def _tree_shap(model, x_matrix: np.ndarray):
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(x_matrix)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    return explainer, values


def explain_demand(sample_size: int = SHAP_SAMPLE) -> dict:
    ccpp = prepare_ccpp()
    model = load_regressor()
    sample = ccpp[REG_FEATURE_COLS].sample(
        n=min(sample_size, len(ccpp)), random_state=42
    )
    explainer, shap_values = _tree_shap(model, sample.values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": REG_FEATURE_COLS, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    return {
        "importance": importance,
        "sample": sample,
        "shap_values": shap_values,
        "expected_value": float(np.mean(explainer.expected_value)),
    }


def explain_anomalies(sample_size: int = SHAP_SAMPLE) -> dict:
    ai4i = prepare_ai4i()
    pipeline = load_classifier()
    x = clf_features(ai4i)
    sample = x.sample(n=min(sample_size, len(x)), random_state=42)
    x_transformed = pipeline.named_steps["preprocess"].transform(sample)
    model = pipeline.named_steps["clf"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    explainer, shap_values = _tree_shap(model, x_transformed)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": list(feature_names), "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=False)
    return {
        "importance": importance,
        "sample": sample,
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
    }


def top_reasons(importance: pd.DataFrame, n: int = 3) -> list[str]:
    lines = []
    for _, row in importance.head(n).iterrows():
        lines.append(
            f"**{row['feature']}** drives outcomes (mean |SHAP| = {row['mean_abs_shap']:.3f})"
        )
    return lines
