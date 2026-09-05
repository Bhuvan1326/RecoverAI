"""
AppealSuccessService (Phase 2, Feature C). Serves the trained, calibrated
appeal-success model when one is registered; falls back to the documented
heuristic (recovery.predict_appeal_success_heuristic) when it isn't --
e.g. before the first `python -m ml.training.train_appeal_success_model`
run, or if too few resolved appeals exist yet to train reliably. The
fallback is never silent: every response's `source` field says which path
was used, and recovery-value/priority-queue consumers don't need to care
which one produced the number.
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

from ml.features.build_appeal_features import ALL_FEATURES, compute_asof_appeal_rate  # noqa: E402

from app.models.domain import Claim, ClaimLine, DenialEvent, ModelVersion, Payer, Provider
from app.services.recovery import predict_appeal_success_heuristic


class AppealModelNotTrainedError(Exception):
    pass


@lru_cache(maxsize=4)
def _load_artifact(path: str):
    return joblib.load(path)


def get_champion_model_version(db: Session) -> ModelVersion:
    mv = db.execute(
        select(ModelVersion).where(ModelVersion.model_type == "appeal_success", ModelVersion.is_champion.is_(True))
    ).scalar_one_or_none()
    if mv is None:
        raise AppealModelNotTrainedError(
            "No champion appeal-success model registered. Run: python -m ml.training.train_appeal_success_model"
        )
    return mv


def _claim_denial_to_feature_row(db: Session, claim: Claim, denial: DenialEvent) -> pd.DataFrame:
    payer = db.get(Payer, claim.payer_id)
    provider = db.get(Provider, claim.provider_id)
    lines = db.execute(select(ClaimLine).where(ClaimLine.claim_id == claim.id)).scalars().all()

    timely_filing_breach = bool(
        claim.timely_filing_deadline and claim.submission_date and claim.submission_date > claim.timely_filing_deadline
    )

    # Real as-of-`denial.denial_date` historical appeal rates (Section 7
    # fix) -- queried from actual resolved prior appeals for this payer
    # and denial reason, not a hardcoded constant. See
    # ml/features/build_appeal_features.py:compute_asof_appeal_rate.
    payer_rate = compute_asof_appeal_rate(db, "payer_id", claim.payer_id, denial.denial_date)
    reason_rate = compute_asof_appeal_rate(db, "denial_reason", denial.denial_reason_code, denial.denial_date)

    row = {
        "denial_reason": denial.denial_reason_code,
        "payer_type": payer.payer_type if payer else "unknown",
        "provider_specialty": provider.specialty if provider else "unknown",
        "procedure_code": lines[0].procedure_code if lines else "unknown",
        "claim_amount": float(claim.claim_amount),
        "documentation_completeness": float(claim.documentation_completeness),
        "authorization_missing": int(claim.authorization_status == "MISSING"),
        "eligibility_fail": int(claim.eligibility_status == "FAIL"),
        "timely_filing_breach": int(timely_filing_breach),
        # This is a genuinely documented cold-start assumption, not a
        # leakage shortcut: at scoring time the appeal hasn't been filed
        # yet, so "days until filed" is inherently unknown. 0 ("file
        # immediately") is used because callers (recovery queue, what-if)
        # only need this for relative ranking across claims scored the
        # same way, not an absolute prediction of real filing behavior.
        "appeal_timing_days": 0,
        "historical_payer_appeal_rate": payer_rate,
        "historical_reason_appeal_rate": reason_rate,
    }
    return pd.DataFrame([row])[ALL_FEATURES]


def predict_appeal_success(db: Session, claim: Claim, denial: DenialEvent | None) -> dict:
    """
    Primary entry point used everywhere in the app (recovery engine,
    recovery queue, appeals API, agent tools). Tries the trained champion
    model first; falls back to the heuristic baseline if no model is
    trained yet or the claim has no denial on record.
    """
    if denial is None:
        result = predict_appeal_success_heuristic(claim, denial)
        result["model_version"] = None
        result["calibrated"] = False
        return result

    try:
        mv = get_champion_model_version(db)
    except AppealModelNotTrainedError:
        result = predict_appeal_success_heuristic(claim, denial)
        result["model_version"] = None
        result["calibrated"] = False
        return result

    pipeline = _load_artifact(mv.artifact_path)
    X = _claim_denial_to_feature_row(db, claim, denial)
    probability = float(pipeline.predict_proba(X)[0, 1])

    return {
        "appeal_success_probability": round(probability, 4),
        "source": "ml",
        "model_version": f"{mv.model_name}-{mv.version_tag}",
        "calibrated": mv.calibration_status == "isotonic",
        "disclaimer": "Trained on synthetic appeal-outcome labels (see ml/features/build_appeal_features.py); not validated against real-world appeal outcomes.",
    }


def risk_category(probability: float) -> str:
    if probability >= 0.65:
        return "HIGH_RECOVERY_POTENTIAL"
    if probability >= 0.35:
        return "MODERATE_RECOVERY_POTENTIAL"
    return "LOW_RECOVERY_POTENTIAL"


def predict_appeal_success_batch(db: Session, claim_denial_pairs: list[tuple[Claim, DenialEvent]]) -> dict[str, dict]:
    """
    Batch variant for dashboard/analytics aggregation over many denied
    claims at once. Fetches the champion model ONCE (not per claim -- see
    the anomaly-detection O(n^2) bug fixed earlier in this project for why
    that matters) and predicts in a single vectorized call. Falls back to
    the per-claim heuristic entry for any claim where the champion model
    isn't trained. Returns {claim_id: result_dict}.
    """
    results: dict[str, dict] = {}
    if not claim_denial_pairs:
        return results

    try:
        mv = get_champion_model_version(db)
    except AppealModelNotTrainedError:
        for claim, denial in claim_denial_pairs:
            r = predict_appeal_success_heuristic(claim, denial)
            r["model_version"] = None
            r["calibrated"] = False
            results[claim.id] = r
        return results

    pipeline = _load_artifact(mv.artifact_path)
    rows = [_claim_denial_to_feature_row(db, claim, denial) for claim, denial in claim_denial_pairs]
    X = pd.concat(rows, ignore_index=True)
    probabilities = pipeline.predict_proba(X)[:, 1]

    for (claim, _denial), probability in zip(claim_denial_pairs, probabilities):
        results[claim.id] = {
            "appeal_success_probability": round(float(probability), 4),
            "source": "ml",
            "model_version": f"{mv.model_name}-{mv.version_tag}",
            "calibrated": mv.calibration_status == "isotonic",
        }
    return results
