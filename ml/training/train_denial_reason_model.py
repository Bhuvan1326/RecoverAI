#!/usr/bin/env python3
"""
Train the multiclass denial-reason model (Phase 2, Feature B). Reuses the
SAME pre-submission, leakage-guarded feature set as the denial-risk model
(ml/features/build_features.py) -- if a feature isn't safe to use for
predicting *whether* a claim is denied, it isn't safe for predicting *why*
either, since both are computed at the same point in time.

Trained on ALL denied claims (whatever their reason), but at inference time
(app/services/denial_reason.py) it is only ever consulted for the residual
that the deterministic rules (missing auth, failed eligibility, timely
filing, severely incomplete docs) don't already resolve -- rules always
take precedence per Section B1.

Label note: denial_reason_code itself comes from scripts/generate_data.py's
documented synthetic-generation rules (see that file's docstring). For a
meaningful chunk of rows the reason is itself deterministically derived
from claim fields there too -- which is realistic (this mirrors how the
service layer's rule-first design works) but means honest macro-F1 on the
genuinely ambiguous residual classes (CODING_MISMATCH, MEDICAL_NECESSITY,
DUPLICATE_CLAIM, OTHER) will be modest, since those are close to randomly
assigned in the generator by construction. Reported as-is, not inflated.

Usage:
    python -m ml.training.train_denial_reason_model
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "backend"))

from app.core import database as database_module
from app.models.domain import ModelVersion, ModelMetric, SyntheticDataRun
from ml.features.build_features import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_feature_matrix
from ml.versioning import get_git_commit, new_run_timestamp, new_version_tag, promote_if_better

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "models" / "artifacts"
MIN_TRAINING_ROWS = 100


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def evaluate(y_true, y_pred, class_names) -> dict:
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_test": int(len(y_true)),
        "per_class": {
            class_names[i]: {"precision": float(precision[i]), "recall": float(recall[i]), "support": int(support[i])}
            for i in range(len(class_names))
        },
        "class_distribution": {class_names[i]: int(support[i]) for i in range(len(class_names))},
    }


def train_one(name: str, estimator, df: pd.DataFrame, train_idx, test_idx, label_encoder):
    X = df[ALL_FEATURES]
    y = label_encoder.transform(df["denial_reason_code"])

    preprocessor = make_preprocessor()
    pipeline = Pipeline([("prep", preprocessor), ("clf", estimator)])
    pipeline.fit(X.iloc[train_idx], y[train_idx])
    y_pred = pipeline.predict(X.iloc[test_idx])
    y_pred = np.asarray(y_pred).ravel()  # CatBoost's multiclass predict() returns shape (n, 1), not (n,)

    metrics = evaluate(y[test_idx], y_pred, list(label_encoder.classes_))
    cm = confusion_matrix(y[test_idx], y_pred, labels=range(len(label_encoder.classes_))).tolist()
    metrics["confusion_matrix"] = cm
    print(f"[{name}] macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f} n_test={metrics['n_test']}")
    return pipeline, metrics


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    db = database_module.SessionLocal()
    try:
        df = build_feature_matrix(db)
        if df.empty:
            print("No claims found. Run scripts/generate_data.py first.")
            return
        df = df[(df["is_denied"] == 1) & df["denial_reason_code"].notna()].copy()
        if len(df) < MIN_TRAINING_ROWS:
            print(f"Only {len(df)} denied claims available (need >= {MIN_TRAINING_ROWS}). Generate more data first.")
            return

        df = df.sort_values("submission_date").reset_index(drop=True)
        label_encoder = LabelEncoder()
        label_encoder.fit(df["denial_reason_code"])
        print(f"Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

        split_point = int(len(df) * 0.8)
        train_idx = np.arange(0, split_point)
        test_idx = np.arange(split_point, len(df))
        print(f"Chronological split: {len(train_idx)} train / {len(test_idx)} test denied claims")

        latest_run = db.query(SyntheticDataRun).order_by(SyntheticDataRun.created_at.desc()).first()
        dataset_version = str(latest_run.id) if latest_run else "unknown"

        models = {
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
            "xgboost": XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
                num_class=len(label_encoder.classes_), eval_metric="mlogloss", random_state=42,
            ),
            "catboost": CatBoostClassifier(
                iterations=150,
                depth=4,
                learning_rate=0.1,
                loss_function="MultiClass",
                classes_count=len(label_encoder.classes_),
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
            ),
        }

        results = {}
        run_timestamp = new_run_timestamp()
        git_commit = get_git_commit()
        for name, est in models.items():
            pipeline, metrics = train_one(name, est, df, train_idx, test_idx, label_encoder)
            artifact_path = ARTIFACT_DIR / f"denial_reason_{name}.joblib"
            joblib.dump({"pipeline": pipeline, "label_encoder": label_encoder, "features": ALL_FEATURES}, artifact_path)
            results[name] = {"metrics": metrics, "artifact_path": str(artifact_path)}

            mv = ModelVersion(
                model_name=name,
                model_type="denial_reason",
                version_tag=new_version_tag(name, run_timestamp),
                artifact_path=str(artifact_path),
                metrics={k: v for k, v in metrics.items() if k != "confusion_matrix"},  # keep ModelMetric rows numeric-only
                feature_names=ALL_FEATURES,
                params={"git_commit": git_commit},
                is_champion=False,
                dataset_version=dataset_version,
                feature_schema_version="1.0",
                calibration_status="none",  # multiclass predict_proba not calibrated in this pass -- documented limitation
            )
            db.add(mv)
            db.flush()
            db.add(ModelMetric(model_version_id=mv.id, metric_name="macro_f1", metric_value=metrics["macro_f1"]))
            db.add(ModelMetric(model_version_id=mv.id, metric_name="weighted_f1", metric_value=metrics["weighted_f1"]))
            results[name]["model_version_id"] = mv.id

        db.commit()

        # Champion promotion gate (Section 10): only promotes if the new
        # best candidate (by macro F1 -- protects minority denial-reason
        # classes, never raw accuracy) actually beats the CURRENT champion
        # from any prior run.
        champion_name = promote_if_better(db, "denial_reason", results, primary_metric="macro_f1")

        if champion_name:
            print(f"\nChampion promoted: {champion_name} (macro_f1={results[champion_name]['metrics']['macro_f1']:.4f})")
        else:
            print("\nNo promotion -- existing champion retained (or no valid candidate).")

        summary_path = ARTIFACT_DIR / "denial_reason_training_summary.json"
        summary_path.write_text(
            json.dumps({"promoted_champion": champion_name, "classes": list(label_encoder.classes_), "results": {k: v["metrics"] for k, v in results.items()}}, indent=2)
        )
        print(f"Summary written to {summary_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
