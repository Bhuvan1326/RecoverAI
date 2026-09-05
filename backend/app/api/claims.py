import copy

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.domain import Claim, ClaimLine, DenialEvent, User, UserRole
from app.schemas.schemas import ClaimCreate, ClaimOut, DenialScoreOut
from app.services import ml_inference
from app.services import anomaly as anomaly_service
from app.services import appeal_success as appeal_success_service
from app.services.audit import record_event
from app.services.denial_reason import predict_denial_reason
from app.services.validator import validate_claim

router = APIRouter(prefix="/claims", tags=["claims"])


@router.post("", response_model=ClaimOut)
def create_claim(
    payload: ClaimCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN, UserRole.BILLER)),
):
    existing = db.execute(select(Claim).where(Claim.claim_number == payload.claim_number)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail="claim_number already exists")

    claim = Claim(
        claim_number=payload.claim_number,
        provider_id=payload.provider_id,
        payer_id=payload.payer_id,
        patient_ref=payload.patient_ref,
        claim_amount=payload.claim_amount,
        claim_type=payload.claim_type,
        place_of_service=payload.place_of_service,
        eligibility_status=payload.eligibility_status,
        authorization_status=payload.authorization_status,
        documentation_completeness=payload.documentation_completeness,
        service_date=payload.service_date,
        submission_date=payload.submission_date,
        is_synthetic=False,
        data_source="api_manual_entry",
    )
    for line in payload.lines:
        claim.lines.append(ClaimLine(**line.model_dump()))

    db.add(claim)
    db.commit()
    db.refresh(claim)
    record_event(db, actor_type="user", actor_id=user.id, event_type="claim.created", claim_id=claim.id, payload={"claim_number": claim.claim_number})
    return claim


@router.get("", response_model=list[ClaimOut])
def list_claims(
    status: str | None = None,
    payer_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    q = select(Claim)
    if status:
        q = q.where(Claim.status == status)
    if payer_id:
        q = q.where(Claim.payer_id == payer_id)
    q = q.order_by(Claim.created_at.desc()).offset(offset).limit(min(limit, 200))
    return db.execute(q).scalars().all()


def _get_claim_or_404(db: Session, claim_id: str) -> Claim:
    claim = db.get(Claim, claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim


@router.get("/{claim_id}", response_model=ClaimOut)
def get_claim(claim_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    return _get_claim_or_404(db, claim_id)


@router.post("/{claim_id}/score", response_model=DenialScoreOut)
def score_claim(claim_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    claim = _get_claim_or_404(db, claim_id)
    try:
        result = ml_inference.score_claim(db, claim)
    except ml_inference.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record_event(db, actor_type="agent", actor_id="denial_risk_model", event_type="claim.scored", claim_id=claim.id, payload=result)
    return result


@router.get("/{claim_id}/explanation")
def get_explanation(claim_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    claim = _get_claim_or_404(db, claim_id)
    try:
        explanation = ml_inference.explain_claim(db, claim)
    except ml_inference.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    denial_reason = predict_denial_reason(db, claim)
    explanation["denial_reason"] = denial_reason
    record_event(db, actor_type="agent", actor_id="shap_explainer", event_type="claim.explained", claim_id=claim.id, payload={"model_version": explanation.get("model_version")})
    return explanation


@router.post("/{claim_id}/validate")
def validate(claim_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    claim = _get_claim_or_404(db, claim_id)
    result = validate_claim(claim)
    record_event(db, actor_type="agent", actor_id="claim_validator", event_type="claim.validated", claim_id=claim.id, payload={"readiness_score": result["readiness_score"]})
    return result


@router.post("/{claim_id}/anomaly-score")
def score_anomaly(claim_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    claim = _get_claim_or_404(db, claim_id)
    try:
        result = anomaly_service.score_claim(db, claim.id)
    except anomaly_service.AnomalyModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record_event(db, actor_type="agent", actor_id="anomaly_detector", event_type="claim.anomaly_scored", claim_id=claim.id, payload=result)
    return result


@router.get("/{claim_id}/anomaly")
def get_anomaly(claim_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    claim = _get_claim_or_404(db, claim_id)
    try:
        return anomaly_service.score_claim(db, claim.id)
    except anomaly_service.AnomalyModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/{claim_id}/appeal-success-score")
def score_appeal_success(claim_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """
    Feature C5. Scores appeal-success probability for a claim using the
    trained champion model (heuristic fallback if none is trained yet --
    see app/services/appeal_success.py). This is the same function the
    recovery engine and recovery queue call internally; exposed directly
    here so the frontend can show/refresh it independent of the full
    recovery-value calculation.
    """
    claim = _get_claim_or_404(db, claim_id)
    denial = db.execute(select(DenialEvent).where(DenialEvent.claim_id == claim_id)).scalar_one_or_none()
    result = appeal_success_service.predict_appeal_success(db, claim, denial)
    result["claim_id"] = claim_id
    result["risk_category"] = appeal_success_service.risk_category(result["appeal_success_probability"])

    record_event(
        db, actor_type="agent", actor_id="appeal_success_model", event_type="claim.appeal_success_scored",
        claim_id=claim_id, payload={"probability": result["appeal_success_probability"], "source": result["source"]},
    )
    return result


@router.get("/{claim_id}/appeal-success")
def get_appeal_success(claim_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    claim = _get_claim_or_404(db, claim_id)
    denial = db.execute(select(DenialEvent).where(DenialEvent.claim_id == claim_id)).scalar_one_or_none()
    result = appeal_success_service.predict_appeal_success(db, claim, denial)
    result["claim_id"] = claim_id
    result["risk_category"] = appeal_success_service.risk_category(result["appeal_success_probability"])
    return result


@router.post("/{claim_id}/simulate")
def simulate(
    claim_id: str,
    overrides: dict,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    What-If Simulator (Feature 14). Applies `overrides` to an in-memory
    COPY of the claim (never persisted) and re-scores it. Returns
    before/after risk so the effect of a proposed fix is visible without
    ever mutating real claim data via this endpoint.
    """
    claim = _get_claim_or_404(db, claim_id)
    try:
        original = ml_inference.score_claim(db, claim)
    except ml_inference.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    simulated_claim = copy.deepcopy(claim)
    allowed_fields = {"authorization_status", "eligibility_status", "documentation_completeness", "claim_amount"}
    applied = {}
    for key, value in overrides.items():
        if key in allowed_fields:
            setattr(simulated_claim, key, value)
            applied[key] = value

    try:
        simulated = ml_inference.score_claim(db, simulated_claim)
    except ml_inference.ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))

    record_event(
        db, actor_type="user", actor_id=user.id, event_type="claim.what_if_simulated", claim_id=claim.id,
        payload={"applied_overrides": applied, "original_risk": original["denial_probability"], "simulated_risk": simulated["denial_probability"]},
    )

    return {
        "claim_id": claim.id,
        "applied_overrides": applied,
        "original_risk": original["denial_probability"],
        "simulated_risk": simulated["denial_probability"],
        "risk_difference": round(original["denial_probability"] - simulated["denial_probability"], 4),
        "disclaimer": "This reflects the model's association between these features and denial risk, not a proven causal effect.",
    }
