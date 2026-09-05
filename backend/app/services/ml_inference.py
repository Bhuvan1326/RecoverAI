"""
Loads the current champion denial-risk model and serves predictions +
SHAP explanations against live claim data. Model artifacts are trained
offline via `python -m ml.training.train_denial_model` (Section 37) --
this service never retrains on request/startup (Section 37 constraint).
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from ml.explainability.shap_explainer import explain_prediction  # noqa: E402
from ml.features.build_features import ALL_FEATURES, compute_asof_denial_rate  # noqa: E402

from app.models.domain import Claim, ClaimLine, ModelVersion, Payer, Provider


class ModelNotTrainedError(Exception):
    pass


@lru_cache(maxsize=4)
def _load_artifact(path: str):
    return joblib.load(path)


def get_champion_model_version(db: Session, model_type: str = "denial_risk") -> ModelVersion:
    mv = db.execute(
        select(ModelVersion).where(ModelVersion.model_type == model_type, ModelVersion.is_champion.is_(True))
    ).scalar_one_or_none()
    if mv is None:
        raise ModelNotTrainedError(
            f"No champion model registered for '{model_type}'. Run: python -m ml.training.train_denial_model"
        )
    return mv


def _claim_to_feature_row(db: Session, claim: Claim) -> pd.DataFrame:
    payer = db.get(Payer, claim.payer_id)
    provider = db.get(Provider, claim.provider_id)
    lines = db.execute(select(ClaimLine).where(ClaimLine.claim_id == claim.id)).scalars().all()

    submission_date = claim.submission_date or claim.service_date
    days_to_submission = (submission_date - claim.service_date).days if claim.submission_date else 0

    # Real as-of-`submission_date` historical rates (Section 6 fix) --
    # queried from actual prior claims for this payer/provider, not a
    # hardcoded constant. See
    # ml/features/build_features.py:compute_asof_denial_rate for the
    # leakage-safe "claims strictly before this cutoff" query and the
    # documented cold-start fallback it uses when too little history exists.
    payer_rate = compute_asof_denial_rate(db, "payer_id", claim.payer_id, submission_date)
    provider_rate = compute_asof_denial_rate(db, "provider_id", claim.provider_id, submission_date)

    row = {
        "claim_type": claim.claim_type,
        "place_of_service": claim.place_of_service,
        "payer_type": payer.payer_type if payer else "unknown",
        "provider_specialty": provider.specialty if provider else "unknown",
        "procedure_code": lines[0].procedure_code if lines else "unknown",
        "claim_amount": float(claim.claim_amount),
        "documentation_completeness": float(claim.documentation_completeness),
        "days_to_submission": days_to_submission,
        "modifier_count": sum(1 for l in lines if l.modifiers),
        "procedure_count": len({l.procedure_code for l in lines}),
        "line_count": len(lines),
        "payer_historical_denial_rate": payer_rate,
        "provider_historical_denial_rate": provider_rate,
        "auth_missing": int(claim.authorization_status == "MISSING"),
        "eligibility_fail": int(claim.eligibility_status == "FAIL"),
    }
    return pd.DataFrame([row])[ALL_FEATURES]


def risk_category(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.5:
        return "HIGH"
    if prob >= 0.25:
        return "MEDIUM"
    return "LOW"


def score_claim(db: Session, claim: Claim) -> dict:
    mv = get_champion_model_version(db)
    pipeline = _load_artifact(mv.artifact_path)
    X = _claim_to_feature_row(db, claim)

    prob = float(pipeline.predict_proba(X)[0, 1])
    return {
        "claim_id": claim.id,
        "denial_probability": round(prob, 4),
        "risk_category": risk_category(prob),
        "model_version": f"{mv.model_name}-{mv.version_tag}",
        "model_version_id": mv.id,
    }


def explain_claim(db: Session, claim: Claim) -> dict:
    mv = get_champion_model_version(db)
    pipeline = _load_artifact(mv.artifact_path)
    X = _claim_to_feature_row(db, claim)
    prob = float(pipeline.predict_proba(X)[0, 1])
    explanation = explain_prediction(pipeline, X)
    explanation["denial_probability"] = round(prob, 4)
    explanation["model_version"] = f"{mv.model_name}-{mv.version_tag}"
    return explanation
