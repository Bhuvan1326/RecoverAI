"""
Hybrid denial-reason prediction (Feature 2 / Phase 2 Feature B).

Deterministic rules fire first for cases that are *known facts*, not
predictions (missing auth, failed eligibility, timely-filing breach --
these aren't things a model should "predict", they're already true).
Only the genuinely ambiguous residual falls through to the trained
multiclass model (ml/training/train_denial_reason_model.py) when one is
registered, or a simple heuristic otherwise. Rules always take precedence
per Section B1 -- this is enforced by construction (the ML/heuristic path
is only ever reached after _rule_based_reason returns None).
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib

BACKEND_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Claim, ModelVersion


class DenialReasonModelNotTrainedError(Exception):
    pass


@lru_cache(maxsize=4)
def _load_artifact(path: str):
    return joblib.load(path)


def get_champion_model_version(db: Session) -> ModelVersion:
    mv = db.execute(
        select(ModelVersion).where(ModelVersion.model_type == "denial_reason", ModelVersion.is_champion.is_(True))
    ).scalar_one_or_none()
    if mv is None:
        raise DenialReasonModelNotTrainedError(
            "No champion denial-reason model registered. Run: python -m ml.training.train_denial_reason_model"
        )
    return mv


def _rule_based_reason(claim: Claim) -> str | None:
    if claim.authorization_status == "MISSING":
        return "MISSING_AUTHORIZATION"
    if claim.eligibility_status == "FAIL":
        return "ELIGIBILITY_ISSUE"
    if claim.timely_filing_deadline and claim.submission_date and claim.submission_date > claim.timely_filing_deadline:
        return "TIMELY_FILING"
    if float(claim.documentation_completeness) < 60:
        return "MISSING_DOCUMENTATION"
    return None


def _ml_residual_prediction(db: Session, claim: Claim) -> dict:
    """Only called once rules have already failed to resolve a reason."""
    from app.services.ml_inference import _claim_to_feature_row

    mv = get_champion_model_version(db)
    artifact = _load_artifact(mv.artifact_path)
    pipeline, label_encoder, features = artifact["pipeline"], artifact["label_encoder"], artifact["features"]

    X = _claim_to_feature_row(db, claim)[features]
    probabilities = pipeline.predict_proba(X)[0]
    ranked_idx = probabilities.argsort()[::-1]
    classes = label_encoder.classes_

    top_idx = ranked_idx[0]
    alternatives = [
        {"reason": classes[i], "confidence": round(float(probabilities[i]), 4)} for i in ranked_idx[1:4]
    ]

    return {
        "predicted_reason": classes[top_idx],
        "confidence": round(float(probabilities[top_idx]), 4),
        "source": "ml",
        "alternatives": alternatives,
        "model_version": f"{mv.model_name}-{mv.version_tag}",
    }


def _heuristic_residual_prediction(claim: Claim) -> dict:
    """Fallback used only when no denial-reason model is trained yet."""
    candidates = {
        "CODING_MISMATCH": 0.15 + (0.3 if float(claim.documentation_completeness) < 85 else 0),
        "MEDICAL_NECESSITY": 0.15,
        "DUPLICATE_CLAIM": 0.05,
        "OTHER": 0.10,
    }
    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    top_reason, top_score = ranked[0]
    total = sum(candidates.values()) or 1
    return {
        "predicted_reason": top_reason,
        "confidence": round(top_score / total, 4),
        "source": "heuristic",
        "alternatives": [{"reason": r, "confidence": round(s / total, 4)} for r, s in ranked[1:]],
        "model_version": None,
    }


def predict_denial_reason(db: Session | None, claim: Claim) -> dict:
    """
    Primary entry point. `db` is optional ONLY to preserve the ability to
    call this in contexts with no live model (rule-only path always works
    without a session) -- pass a real Session whenever one is available so
    the ML residual path can run.
    """
    rule_reason = _rule_based_reason(claim)
    if rule_reason:
        return {
            "predicted_reason": rule_reason,
            "confidence": 1.0,
            "source": "rule",
            "alternatives": [],
            "model_version": None,
        }

    if db is not None:
        try:
            return _ml_residual_prediction(db, claim)
        except DenialReasonModelNotTrainedError:
            pass

    return _heuristic_residual_prediction(claim)
