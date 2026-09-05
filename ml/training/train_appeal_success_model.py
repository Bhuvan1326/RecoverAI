#!/usr/bin/env python3
"""
Train the appeal-success model (Phase 2, Feature C). Same disciplined
pattern as ml/training/train_denial_model.py: chronological (never random)
train/test split by appeal_date, isotonic calibration (this probability
feeds directly into $ calculations via Expected Recovery Value, so
calibration quality matters more here than almost anywhere else in the
system), champion selection by PR-AUC, full metrics persisted to
ModelVersion/ModelMetric.

Usage:
    python -m ml.training.train_appeal_success_model
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier
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
from app.models.domain import ModelVersion, ModelMetric, SyntheticDataRun
from ml.versioning import get_git_commit, new_run_timestamp, new_version_tag, promote_if_better
from ml.features.build_appeal_features import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    FORBIDDEN_POST_RESOLUTION_FIELDS,
    NUMERIC_FEATURES,
    build_feature_matrix,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"

MIN_TRAINING_ROWS = 80  # below this, a held-out chronological split is too noisy to trust


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


def train_one(name: str, base_estimator, df: pd.DataFrame, train_idx, test_idx):
    X = df[ALL_FEATURES]
    y = df["appeal_success"].astype(int)

    preprocessor = make_preprocessor()
    # Calibration is not optional here (C4): this probability is multiplied
    # directly into a dollar figure downstream (Expected Recovery Value).
    calibrated = CalibratedClassifierCV(base_estimator, method="isotonic", cv=3)
    pipeline = Pipeline([("prep", preprocessor), ("clf", calibrated)])

    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_prob = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
    metrics = evaluate(y.iloc[test_idx].values, y_prob)
    print(f"[{name}] {json.dumps(metrics, indent=2)}")
    return pipeline, metrics


def main():
    assert FORBIDDEN_POST_RESOLUTION_FIELDS.isdisjoint(ALL_FEATURES), (
        "Leakage guard tripped: a post-resolution field is present in ALL_FEATURES."
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    db = database_module.SessionLocal()
    try:
        df = build_feature_matrix(db)
        if len(df) < MIN_TRAINING_ROWS:
            print(
                f"Only {len(df)} resolved appeals available (need >= {MIN_TRAINING_ROWS}). "
                "Generate more synthetic data first: python scripts/generate_data.py --claims 5000"
            )
            return

        df = df.sort_values("appeal_date").reset_index(drop=True)
        split_point = int(len(df) * 0.8)
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, len(df))
        print(f"Chronological split: {len(train_idx)} train / {len(test_idx)} test resolved appeals")

        latest_run = db.query(SyntheticDataRun).order_by(SyntheticDataRun.created_at.desc()).first()
        dataset_version = latest_run.id if latest_run else "unknown"

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
            "xgboost": XGBClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
                eval_metric="logloss", random_state=42,
            ),
            "catboost": CatBoostClassifier(
                iterations=150,
                depth=3,
                learning_rate=0.08,
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
        }

        results = {}
        run_timestamp = new_run_timestamp()
        git_commit = get_git_commit()
        for name, est in models.items():
            pipeline, metrics = train_one(name, est, df, train_idx, test_idx)
            artifact_path = ARTIFACT_DIR / f"appeal_success_{name}.joblib"
            joblib.dump(pipeline, artifact_path)
            results[name] = {"metrics": metrics, "artifact_path": str(artifact_path)}

            mv = ModelVersion(
                model_name=name,
                model_type="appeal_success",
                version_tag=new_version_tag(name, run_timestamp),
                artifact_path=str(artifact_path),
                metrics=metrics,
                feature_names=ALL_FEATURES,
                params={"git_commit": git_commit},
                is_champion=False,
                dataset_version=str(dataset_version),
                feature_schema_version="1.0",
                calibration_status="isotonic",
            )
            db.add(mv)
            db.flush()
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    db.add(ModelMetric(model_version_id=mv.id, metric_name=metric_name, metric_value=metric_value))
            results[name]["model_version_id"] = mv.id

        db.commit()

        # Champion promotion gate (Section 10): only promotes if the new
        # best candidate (by PR-AUC -- never raw accuracy on an
        # imbalanced/small appeal-outcome dataset) actually beats the
        # CURRENT champion from any prior run.
        champion_name = promote_if_better(db, "appeal_success", results, primary_metric="pr_auc")

        if champion_name:
            print(f"\nChampion promoted: {champion_name} (PR-AUC={results[champion_name]['metrics']['pr_auc']:.4f})")
        else:
            print("\nNo promotion -- existing champion retained (or no valid candidate).")

        summary_path = ARTIFACT_DIR / "appeal_success_training_summary.json"
        summary_path.write_text(
            json.dumps({"promoted_champion": champion_name, "results": {k: v["metrics"] for k, v in results.items()}}, indent=2)
        )
        print(f"Summary written to {summary_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
