"""
Automated model-drift computation (Phase 2, Feature D).

Reference vs. current split: reference is the SAME chronological training
slice the champion model was actually trained on (first ~80% of claims by
submission_date -- matching ml/features/build_features.py's split logic);
current is the most recent slice of claims (configurable size). This is a
deliberate, documented engineering choice -- not "the only correct" way to
define drift windows -- because this project doesn't yet have a separate
persisted "reference distribution snapshot" from training time; recomputing
the same chronological split from live data is a reasonable, honest
approximation and is cheap enough to run on every request.

Metrics:
- Data drift (numeric): Population Stability Index (PSI) using reference-
  derived decile bins, plus the Kolmogorov-Smirnov two-sample statistic.
- Data drift (categorical): PSI over category frequency distributions.
- Prediction drift: PSI/absolute-difference on the champion denial-risk
  model's predicted-probability distribution, average risk score, and
  high-risk percentage, reference vs. current.
- Missing-value drift: null-rate per feature, reference vs. current.

Thresholds are configurable via Settings (DRIFT_PSI_WARNING /
DRIFT_PSI_CRITICAL) and are documented engineering defaults (PSI > 0.1 /
> 0.25 are commonly cited rule-of-thumb thresholds in the MLOps literature),
not a claim of universal correctness for this specific domain.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent / "backend"
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.config import get_settings
from app.models.domain import DriftMetric, ModelVersion
from ml.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix

EPS = 1e-6


def _status_for_psi(psi: float, settings) -> tuple[str, bool]:
    if psi >= settings.DRIFT_PSI_CRITICAL:
        return "CRITICAL", True
    if psi >= settings.DRIFT_PSI_WARNING:
        return "WARNING", True
    return "NORMAL", False


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if len(reference) < 10 or len(current) < 10:
        return 0.0

    try:
        bin_edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    except Exception:
        return 0.0
    if len(bin_edges) < 3:
        return 0.0  # not enough distinct values to bin meaningfully

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current, bins=bin_edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1) + EPS
    cur_pct = cur_counts / max(cur_counts.sum(), 1) + EPS

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    if not categories:
        return 0.0

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    ref_total = max(len(reference), 1)
    cur_total = max(len(current), 1)

    psi = 0.0
    for cat in categories:
        ref_pct = ref_counts.get(cat, 0) / ref_total + EPS
        cur_pct = cur_counts.get(cat, 0) / cur_total + EPS
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return round(float(psi), 6)


def split_reference_current(df: pd.DataFrame, current_window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same chronological-split philosophy as training: reference = first
    ~80%, current = the most recent `current_window` claims (or the last
    20%, whichever is smaller) -- so "current" always means "the newest
    data the model hasn't been trained on yet."""
    df = df.sort_values("submission_date").reset_index(drop=True)
    split_point = int(len(df) * 0.8)
    reference = df.iloc[:split_point]
    current = df.iloc[split_point:].tail(current_window) if current_window else df.iloc[split_point:]
    if current.empty:
        current = df.tail(min(current_window, len(df)))
    return reference, current


def compute_data_drift(db: Session, current_window: int = 500) -> list[dict]:
    """Feature D1/D3: numeric PSI+KS, categorical PSI, and missing-value
    rate drift for every feature the denial-risk model uses."""
    settings = get_settings()
    df = build_feature_matrix(db)
    if df.empty or len(df) < 20:
        return []

    reference, current = split_reference_current(df, current_window)
    results = []

    for feature in NUMERIC_FEATURES:
        psi = _psi_numeric(reference[feature], current[feature])
        status, detected = _status_for_psi(psi, settings)
        ks_stat, ks_p = ks_2samp(reference[feature].dropna(), current[feature].dropna()) if len(current) >= 2 else (0.0, 1.0)
        results.append(
            {
                "metric_type": "data_drift",
                "metric_name": "PSI",
                "feature_name": feature,
                "baseline_value": round(float(reference[feature].mean()), 6),
                "current_value": round(float(current[feature].mean()), 6),
                "drift_score": psi,
                "status": status,
                "is_drift_detected": detected,
                "extra": {"ks_statistic": round(float(ks_stat), 6), "ks_pvalue": round(float(ks_p), 6)},
            }
        )

        missing_ref = float(reference[feature].isna().mean())
        missing_cur = float(current[feature].isna().mean())
        missing_drift = abs(missing_cur - missing_ref)
        results.append(
            {
                "metric_type": "missing_value",
                "metric_name": "missing_rate",
                "feature_name": feature,
                "baseline_value": round(missing_ref, 6),
                "current_value": round(missing_cur, 6),
                "drift_score": round(missing_drift, 6),
                "status": "WARNING" if missing_drift >= settings.DRIFT_PSI_WARNING else "NORMAL",
                "is_drift_detected": missing_drift >= settings.DRIFT_PSI_WARNING,
                "extra": {},
            }
        )

    for feature in CATEGORICAL_FEATURES:
        psi = _psi_categorical(reference[feature], current[feature])
        status, detected = _status_for_psi(psi, settings)
        results.append(
            {
                "metric_type": "data_drift",
                "metric_name": "PSI",
                "feature_name": feature,
                "baseline_value": 0.0,  # categorical -- no single "mean"; distributions are in `extra`
                "current_value": 0.0,
                "drift_score": psi,
                "status": status,
                "is_drift_detected": detected,
                "extra": {
                    "reference_distribution": reference[feature].value_counts(normalize=True).round(4).to_dict(),
                    "current_distribution": current[feature].value_counts(normalize=True).round(4).to_dict(),
                },
            }
        )

    return results


def compute_prediction_drift(db: Session, current_window: int = 500) -> list[dict]:
    """Feature D2: drift in the champion denial-risk model's OWN predicted
    probabilities, reference vs. current -- catches "the world changed
    under the model" even when individual input features look stable."""
    from app.services import ml_inference

    settings = get_settings()
    df = build_feature_matrix(db)
    if df.empty or len(df) < 20:
        return []

    try:
        mv = ml_inference.get_champion_model_version(db)
    except ml_inference.ModelNotTrainedError:
        return []

    pipeline = ml_inference._load_artifact(mv.artifact_path)
    reference, current = split_reference_current(df, current_window)

    from ml.features.build_features import ALL_FEATURES

    ref_probs = pipeline.predict_proba(reference[ALL_FEATURES])[:, 1]
    cur_probs = pipeline.predict_proba(current[ALL_FEATURES])[:, 1]

    psi = _psi_numeric(pd.Series(ref_probs), pd.Series(cur_probs))
    status, detected = _status_for_psi(psi, settings)

    ref_high_risk = float((ref_probs >= 0.5).mean())
    cur_high_risk = float((cur_probs >= 0.5).mean())

    return [
        {
            "metric_type": "prediction_drift",
            "metric_name": "PSI",
            "feature_name": "denial_probability",
            "baseline_value": round(float(ref_probs.mean()), 6),
            "current_value": round(float(cur_probs.mean()), 6),
            "drift_score": psi,
            "status": status,
            "is_drift_detected": detected,
            "extra": {
                "reference_high_risk_pct": round(ref_high_risk, 4),
                "current_high_risk_pct": round(cur_high_risk, 4),
                "model_version": f"{mv.model_name}-{mv.version_tag}",
            },
        }
    ]


def persist_drift_metrics(db: Session, results: list[dict], model_version_id: str | None = None) -> list[DriftMetric]:
    reference_period_label = "training split (first ~80% of claims chronologically)"
    current_period_label = "most recent claims"

    rows = []
    for r in results:
        row = DriftMetric(
            model_version_id=model_version_id,
            metric_type=r["metric_type"],
            metric_name=r["metric_name"],
            feature_name=r.get("feature_name"),
            baseline_value=r["baseline_value"],
            current_value=r["current_value"],
            drift_score=r["drift_score"],
            status=r["status"],
            is_drift_detected=r["is_drift_detected"],
            reference_period=reference_period_label,
            current_period=current_period_label,
        )
        db.add(row)
        rows.append(row)
    db.commit()
    return rows


def run_full_drift_check(db: Session, current_window: int = 500) -> dict:
    """Entry point used by both the API's on-demand trigger and the Celery
    scheduled task (Feature D6/E5) -- identical code path either way."""
    from app.services import ml_inference

    data_drift = compute_data_drift(db, current_window)
    prediction_drift = compute_prediction_drift(db, current_window)

    model_version_id = None
    try:
        model_version_id = ml_inference.get_champion_model_version(db).id
    except ml_inference.ModelNotTrainedError:
        pass

    all_results = data_drift + prediction_drift
    persisted = persist_drift_metrics(db, all_results, model_version_id)

    return {
        "n_metrics_computed": len(all_results),
        "n_warning": sum(1 for r in all_results if r["status"] == "WARNING"),
        "n_critical": sum(1 for r in all_results if r["status"] == "CRITICAL"),
        "metric_ids": [row.id for row in persisted],
    }


# --------------------------------------------------------------------------
# Thin class-based facades (spec Section D5 names these explicitly). These
# wrap the module-level functions above rather than duplicating logic --
# kept as classes because a Celery task or a future scheduler config reads
# more naturally calling `DriftDetectionService(db).run()` than a bare
# function, and it gives a clear extension point if these need per-instance
# state (a cached reference snapshot, etc.) later.
# --------------------------------------------------------------------------
class DataQualityMonitoringService:
    def __init__(self, db: Session):
        self.db = db

    def compute_missing_value_drift(self, current_window: int = 500) -> list[dict]:
        return [r for r in compute_data_drift(self.db, current_window) if r["metric_type"] == "missing_value"]


class DriftDetectionService:
    """Data drift (numeric PSI/KS + categorical PSI)."""

    def __init__(self, db: Session):
        self.db = db

    def run(self, current_window: int = 500) -> list[dict]:
        return compute_data_drift(self.db, current_window)


class PredictionDriftService:
    def __init__(self, db: Session):
        self.db = db

    def run(self, current_window: int = 500) -> list[dict]:
        return compute_prediction_drift(self.db, current_window)
