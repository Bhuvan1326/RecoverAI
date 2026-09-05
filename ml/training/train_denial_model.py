#!/usr/bin/env python3
"""
Train denial-risk models: Logistic Regression (interpretable baseline),
Random Forest, XGBoost, and CatBoost (challengers), with isotonic
calibration and a proper CHRONOLOGICAL train/test split (never random) so
no future claims leak into training. Persists artifacts + registers
ModelVersion/ModelMetric rows so the API and /model-monitoring page can
serve real trained models.

Usage:
    python -m ml.training.train_denial_model
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.core import database as database_module
from app.models.domain import ModelMetric, ModelVersion, SyntheticDataRun
from ml.features.build_features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix
from ml.versioning import get_git_commit, new_run_timestamp, new_version_tag, promote_if_better

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def evaluate(y_true, y_prob) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(set(y_true)) > 1 else 0.5,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n_test": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def train_one(name: str, base_estimator, df: pd.DataFrame, train_idx, test_idx) -> tuple[Pipeline, dict]:
    X = df[ALL_FEATURES]
    y = df["is_denied"].astype(int)

    preprocessor = make_preprocessor()
    calibrated = CalibratedClassifierCV(base_estimator, method="isotonic", cv=3)
    pipeline = Pipeline([("prep", preprocessor), ("clf", calibrated)])

    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = evaluate(y.iloc[test_idx].values, y_prob)
    print(f"[{name}] {json.dumps(metrics, indent=2)}")
    return pipeline, metrics


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    db = database_module.SessionLocal()
    try:
        df = build_feature_matrix(db)
        if len(df) < 50:
            print("Not enough data to train. Run scripts/generate_data.py first.")
            return
        df = df.sort_values("submission_date").reset_index(drop=True)

        split_point = int(len(df) * 0.8)
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, len(df))
        print(f"Chronological split: {len(train_idx)} train / {len(test_idx)} test claims")

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
            "xgboost": XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            ),
            # Challenger (Section 8 of the design doc): CatBoost's gradient
            # boosting often edges out plain XGBoost on payer/procedure-heavy
            # categorical data. Run through the SAME preprocessor+calibration
            # pipeline as everything else here for a fair, apples-to-apples
            # champion/challenger comparison rather than letting it use its
            # native (un-one-hot-encoded) categorical handling.
            "catboost": CatBoostClassifier(
                iterations=200,
                depth=4,
                learning_rate=0.08,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
        }

        results = {}
        run_timestamp = new_run_timestamp()
        git_commit = get_git_commit()
        latest_run = db.query(SyntheticDataRun).order_by(SyntheticDataRun.created_at.desc()).first()
        dataset_version = str(latest_run.id) if latest_run else None

        for name, est in models.items():
            pipeline, metrics = train_one(name, est, df, train_idx, test_idx)
            artifact_path = ARTIFACT_DIR / f"denial_risk_{name}.joblib"
            joblib.dump(pipeline, artifact_path)
            results[name] = {"metrics": metrics, "artifact_path": str(artifact_path)}

            mv = ModelVersion(
                model_name=name,
                model_type="denial_risk",
                version_tag=new_version_tag(name, run_timestamp),
                artifact_path=str(artifact_path),
                metrics=metrics,
                feature_names=ALL_FEATURES,
                params={"git_commit": git_commit},
                dataset_version=dataset_version,
                calibration_status="isotonic",
                is_champion=False,
            )
            db.add(mv)
            db.flush()
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    db.add(ModelMetric(model_version_id=mv.id, metric_name=metric_name, metric_value=metric_value))
            results[name]["model_version_id"] = mv.id

        db.commit()

        # Champion promotion gate (Section 10): only promotes if the best
        # new candidate actually beats the CURRENT champion (from any prior
        # run), never unconditionally promotes this run's best.
        champion_name = promote_if_better(db, "denial_risk", results, primary_metric="pr_auc")

        if champion_name:
            print(f"\nChampion promoted: {champion_name} (PR-AUC={results[champion_name]['metrics']['pr_auc']:.4f})")
        else:
            print("\nNo promotion -- existing champion retained (or no valid candidate).")

        summary_path = ARTIFACT_DIR / "denial_risk_training_summary.json"
        summary_path.write_text(json.dumps({"promoted_champion": champion_name, "results": {k: v["metrics"] for k, v in results.items()}}, indent=2))
        print(f"Summary written to {summary_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
