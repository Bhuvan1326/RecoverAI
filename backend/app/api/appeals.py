from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.orchestrator import draft_appeal
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.domain import Claim, DenialEvent, User, WorkflowAction
from app.services.appeal_success import predict_appeal_success, risk_category
from app.services.audit import record_event
from app.services.recovery import expected_recovery_value

router = APIRouter(prefix="/appeals", tags=["appeals"])


@router.post("/predict")
def predict(claim_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    claim = db.get(Claim, claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")
    denial = db.execute(select(DenialEvent).where(DenialEvent.claim_id == claim_id)).scalar_one_or_none()
    appeal = predict_appeal_success(db, claim, denial)
    recovery = expected_recovery_value(float(claim.claim_amount), appeal["appeal_success_probability"])
    priority = "HIGH" if recovery["expected_recovery_value"] > 5000 else "MEDIUM" if recovery["expected_recovery_value"] > 0 else "LOW"
    return {**appeal, "expected_recovery": recovery["expected_recovery_value"], "priority": priority}


@router.post("/draft")
def draft(claim_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    result = draft_appeal(db, claim_id, created_by=user.id)
    return result


@router.get("/{appeal_id}")
def get_appeal(appeal_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    action = db.get(WorkflowAction, appeal_id)
    if not action:
        raise HTTPException(status_code=404, detail="Appeal draft not found")
    return {
        "id": action.id,
        "claim_id": action.claim_id,
        "status": action.status.value,
        "draft_text": action.payload.get("draft_text"),
        "citations": action.payload.get("citations", []),
        "missing_evidence": action.payload.get("missing_evidence", []),
    }
