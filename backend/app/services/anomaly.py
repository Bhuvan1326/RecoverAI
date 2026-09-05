"""
AnomalyDetectionService (Phase 2, Feature A4). Wraps the trained Isolation
Forest, normalizes its raw decision-function output into an application-
level 0-100 score + severity tier, and identifies which input features most
plausibly drove a low (anomalous) score -- via each feature's per-claim
z-score against the training distribution, which is a reasonable proxy for
"contribution" for an unsupervised model where SHAP-style attribution isn't
directly applicable.

Terminology note (enforced throughout the codebase and UI): this is
"Anomaly Detection", never "Fraud Detection" -- fraud is a legal
determination this model has no basis to make.
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.features.build_anomaly_features import ANOMALY_FEATURES, load_anomaly_frame  # noqa: E402

from app.models.domain import ModelVersion
from sqlalchemy import select


class AnomalyModelNotTrainedError(Exception):
    pass


@lru_cache(maxsize=2)
def _load_artifact(path: str):
    return joblib.load(path)


def get_anomaly_model_version(db: Session) -> ModelVersion:
    mv = db.execute(
        select(ModelVersion).where(ModelVersion.model_type == "anomaly_detection", ModelVersion.is_champion.is_(True))
    ).scalar_one_or_none()
    if mv is None:
        raise AnomalyModelNotTrainedError(
            "No anomaly-detection model registered. Run: python -m ml.training.train_anomaly_model"
        )
    return mv


def severity_from_score(score_0_100: float) -> str:
    if score_0_100 >= 80:
        return "HIGH"
    if score_0_100 >= 55:
        return "MEDIUM"
    if score_0_100 >= 30:
        return "LOW"
    return "NORMAL"


def score_all_claims(db: Session) -> tuple[ModelVersion, list[dict]]:
    """
    Fully vectorized batch-scoring path for analytics/aggregate views
    (Feature A6). Loads the feature frame and trained artifact ONCE, runs
    the scaler/model transform across the WHOLE frame in a single call
    (not per-claim), and only loops in Python for cheap per-row dict
    formatting -- O(n), not O(n^2). The naive "call score_claim() per
    claim" approach reloads the full feature frame from the DB on every
    call; even a first-pass fix that reused a preloaded frame was still
    O(n^2) because it re-filtered the whole dataframe (`df[df.claim_id ==
    x]`) on every iteration. This measurably matters: at ~1,500 claims the
    naive version timed out over HTTP; this version returns in well under
    a second.
    """
    mv = get_anomaly_model_version(db)
    df = load_anomaly_frame(db)
    if df.empty:
        return mv, []

    artifact = _load_artifact(mv.artifact_path)
    model, scaler, features = artifact["model"], artifact["scaler"], artifact["features"]

    X = df[features].fillna(0).values
    X_scaled = scaler.transform(X)

    raw_scores = model.decision_function(X_scaled)  # (n,) higher = more normal
    is_outlier = model.predict(X_scaled) == -1  # (n,) bool
    anomaly_scores = np.round(100 / (1 + np.exp(6 * raw_scores)), 1)

    feature_means = df[features].mean()
    feature_stds = df[features].std().replace(0, 1)
    z_frame = ((df[features] - feature_means) / feature_stds).abs()

    version_label = f"{mv.model_name}-{mv.version_tag}"
    results = []
    for i, claim_id in enumerate(df["claim_id"]):
        score = float(anomaly_scores[i])
        z_row = z_frame.iloc[i].sort_values(ascending=False)
        contributing = [
            {"feature": f, "z_score": round(float(z), 2)} for f, z in z_row.head(4).items() if z > 1.0
        ]
        results.append(
            {
                "claim_id": claim_id,
                "anomaly_score": score,
                "is_anomaly": bool(is_outlier[i] or score >= 55),
                "severity": severity_from_score(score),
                "contributing_features": contributing,
                "model_version": version_label,
                "disclaimer": "This is a statistical anomaly signal, not a fraud determination.",
            }
        )
    return mv, results


def score_claim(db: Session, claim_id: str) -> dict:
    mv = get_anomaly_model_version(db)
    df = load_anomaly_frame(db)
    return _score_row(mv, df, claim_id)


def _score_row(mv: ModelVersion, df, claim_id: str, _preloaded_artifact=None) -> dict:
    if df.empty or claim_id not in set(df["claim_id"]):
        return {
            "claim_id": claim_id,
            "anomaly_score": 0,
            "is_anomaly": False,
            "severity": "NORMAL",
            "contributing_features": [],
            "model_version": f"{mv.model_name}-{mv.version_tag}",
            "note": "Claim not found in anomaly feature set (e.g. no claim lines yet).",
        }

    artifact = _preloaded_artifact or _load_artifact(mv.artifact_path)
    model, scaler, features = artifact["model"], artifact["scaler"], artifact["features"]

    row = df[df["claim_id"] == claim_id].iloc[0]
    X = row[features].fillna(0).values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    raw_score = float(model.decision_function(X_scaled)[0])  # higher = more normal, roughly [-0.5, 0.5]
    is_outlier = model.predict(X_scaled)[0] == -1

    # Normalize decision_function output (unbounded-ish, centered near 0) to
    # a 0-100 "how anomalous" score via a bounded logistic-style squash so
    # very extreme raw scores don't blow past the 0-100 range.
    anomaly_score = round(100 / (1 + np.exp(6 * raw_score)), 1)  # raw_score negative (outlier) -> higher score

    # Per-feature contribution proxy: z-score of this claim's feature value
    # against the full dataset distribution loaded above.
    feature_means = df[features].mean()
    feature_stds = df[features].std().replace(0, 1)
    z_scores = ((row[features] - feature_means) / feature_stds).abs().sort_values(ascending=False)
    contributing = [
        {"feature": f, "z_score": round(float(z), 2)}
        for f, z in z_scores.head(4).items()
        if z > 1.0  # only report features that are meaningfully unusual
    ]

    return {
        "claim_id": claim_id,
        "anomaly_score": anomaly_score,
        "is_anomaly": bool(is_outlier or anomaly_score >= 55),
        "severity": severity_from_score(anomaly_score),
        "contributing_features": contributing,
        "model_version": f"{mv.model_name}-{mv.version_tag}",
        "disclaimer": "This is a statistical anomaly signal, not a fraud determination.",
    }
