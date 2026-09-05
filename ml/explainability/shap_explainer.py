"""
SHAP explainability (Feature 3).

Works against the sklearn Pipeline persisted by train_denial_model.py
(ColumnTransformer -> CalibratedClassifierCV). We explain the *base
estimator* inside the calibration wrapper with a generic SHAP explainer
on the transformed feature space, then map contributions back to
human-readable feature names for the API/UI.

IMPORTANT: these are associational feature attributions from a fitted
model, not causal effects. Every consumer of this module must keep the
"association, not causation" disclaimer attached to the output.
"""
import numpy as np
import pandas as pd
import shap


def _get_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"feature_{i}" for i in range(preprocessor.transform.__self__.n_features_in_)]


def explain_prediction(pipeline, X_row: pd.DataFrame, top_k: int = 6) -> dict:
    preprocessor = pipeline.named_steps["prep"]
    calibrated = pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(X_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = _get_feature_names(preprocessor)

    # Use the first calibrated fold's base estimator for a tree/linear explainer.
    base_estimator = calibrated.calibrated_classifiers_[0].estimator

    try:
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(X_transformed)
        base_value = explainer.expected_value

        # Normalize across shap/sklearn version quirks: TreeExplainer on a
        # binary-classification RandomForest/XGBoost model can return either
        # a list of two (n_samples, n_features) arrays (one per class), or a
        # single (n_samples, n_features, n_classes) ndarray. Reduce both to
        # a plain (n_samples, n_features) array for the positive class.
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[1]) if len(shap_values) > 1 else np.array(shap_values[0])
        else:
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, -1]

        if isinstance(base_value, (list, np.ndarray)):
            bv = np.atleast_1d(base_value)
            base_value = float(bv[-1])
        else:
            base_value = float(base_value)
    except Exception:
        # Non-tree model (e.g. Logistic Regression) -> linear/kernel fallback.
        explainer = shap.LinearExplainer(base_estimator, X_transformed, feature_names=feature_names)
        shap_values = np.array(explainer.shap_values(X_transformed))
        base_value = float(np.atleast_1d(explainer.expected_value)[0])

    row_values = np.asarray(shap_values)[0]
    row_values = np.ravel(row_values)[: len(feature_names)]
    contributions = list(zip(feature_names, row_values.tolist()))
    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)

    positive = [{"feature": f, "contribution": round(v, 4)} for f, v in contributions if v > 0][:top_k]
    negative = [{"feature": f, "contribution": round(v, 4)} for f, v in contributions if v < 0][:top_k]

    return {
        "base_value": round(float(base_value), 4),
        "top_positive_factors": positive,
        "top_negative_factors": negative,
        "disclaimer": "Model explanation reflects statistical association learned from training data, not a causal claim about this specific claim.",
    }
