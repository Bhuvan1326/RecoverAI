#!/usr/bin/env python3
"""
Train the Isolation Forest anomaly-detection model (Phase 2, Feature A).

Usage:
    python -m ml.training.train_anomaly_model
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.core.config import get_settings
from app.core import database as database_module
from app.models.domain import ModelVersion
from sqlalchemy import update

from ml.versioning import get_git_commit, new_version_tag
from ml.features.build_anomaly_features import ANOMALY_FEATURES, load_anomaly_frame

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    db = database_module.SessionLocal()
    try:
        df = load_anomaly_frame(db)
        if len(df) < 20:
            print("Not enough claims to train an anomaly model. Run scripts/generate_data.py first.")
            return

        X = df[ANOMALY_FEATURES].fillna(0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = IsolationForest(
            n_estimators=200,
            contamination=settings.ANOMALY_CONTAMINATION,
            random_state=settings.ANOMALY_RANDOM_STATE,
        )
        model.fit(X_scaled)

        raw_scores = model.decision_function(X_scaled)  # higher = more normal
        predictions = model.predict(X_scaled)  # -1 = anomaly, 1 = normal

        n_anomalies = int((predictions == -1).sum())
        metrics = {
            "n_claims_trained": int(len(df)),
            "n_anomalies_flagged": n_anomalies,
            "anomaly_rate": round(n_anomalies / len(df), 4),
            "contamination_param": settings.ANOMALY_CONTAMINATION,
            "score_min": round(float(raw_scores.min()), 4),
            "score_max": round(float(raw_scores.max()), 4),
            "score_mean": round(float(raw_scores.mean()), 4),
        }
        print(json.dumps(metrics, indent=2))

        artifact_path = ARTIFACT_DIR / "anomaly_isolation_forest.joblib"
        joblib.dump({"model": model, "scaler": scaler, "features": ANOMALY_FEATURES}, artifact_path)

        mv = ModelVersion(
            model_name="isolation_forest",
            model_type="anomaly_detection",
            version_tag=new_version_tag("isolation_forest"),
            artifact_path=str(artifact_path),
            metrics=metrics,
            feature_names=ANOMALY_FEATURES,
            params={
                "contamination": settings.ANOMALY_CONTAMINATION,
                "random_state": settings.ANOMALY_RANDOM_STATE,
                "n_estimators": 200,
                "git_commit": get_git_commit(),
            },
            calibration_status="none",
            is_champion=False,
        )
        db.add(mv)
        db.flush()

        # Champion promotion (Section 9/10): unsupervised, so there's no
        # held-out classification metric to gate promotion on the way the
        # other three model types do -- always promote the freshly trained
        # model, but (the actual bug fix) un-champion prior versions rather
        # than DELETING them, so training history is preserved for audit
        # and comparison instead of being silently destroyed on every
        # retrain.
        db.execute(
            update(ModelVersion).where(ModelVersion.model_type == "anomaly_detection").values(is_champion=False)
        )
        mv.is_champion = True
        db.commit()
        print(f"\nAnomaly model trained and registered (model_version_id={mv.id}, version_tag={mv.version_tag}).")
    finally:
        db.close()


if __name__ == "__main__":
    main()
