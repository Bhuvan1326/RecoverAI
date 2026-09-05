from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.domain import Claim, ClaimStatus, DenialEvent, User
from app.services.appeal_success import predict_appeal_success
from app.services.recovery import (
    expected_recovery_value,
    priority_score,
    recommend_next_best_action,
    urgency_factor,
)
from app.services.strategy_simulator import simulate_strategies

router = APIRouter(prefix="/recovery-queue", tags=["recovery"])


def _build_queue_entry(db: Session, claim: Claim, denial: DenialEvent) -> dict:
    appeal = predict_appeal_success(db, claim, denial)
    recovery = expected_recovery_value(float(claim.claim_amount), appeal["appeal_success_probability"])
    days_since_denial = (datetime.now(timezone.utc) - denial.denial_date.replace(tzinfo=timezone.utc)).days
    urgency = urgency_factor(days_since_denial)
    recoverability = appeal["appeal_success_probability"]
    effort = 30.0
    p_score = priority_score(recovery["expected_recovery_value"], urgency, recoverability, effort)

    tier = "CRITICAL" if p_score > 200 else "HIGH" if p_score > 80 else "MEDIUM" if p_score > 20 else "LOW"

    nba = recommend_next_best_action(claim, denial.denial_reason_code, recovery["expected_recovery_value"], appeal["appeal_success_probability"])

    return {
        "claim_id": claim.id,
        "claim_number": claim.claim_number,
        "claim_amount": float(claim.claim_amount),
        "denial_reason": denial.denial_reason_code,
        "payer_id": claim.payer_id,
        "days_since_denial": days_since_denial,
        "appeal_success_probability": appeal["appeal_success_probability"],
        "expected_recovery": recovery["expected_recovery_value"],
        "priority_score": p_score,
        "priority_tier": tier,
        "recommended_action": nba["recommended_action"],
    }


@router.get("")
def get_recovery_queue(
    tier: str | None = None,
    payer_id: str | None = None,
    sort_by: str = "priority_score",
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    q = select(Claim, DenialEvent).join(DenialEvent, DenialEvent.claim_id == Claim.id).where(Claim.status == ClaimStatus.DENIED)
    if payer_id:
        q = q.where(Claim.payer_id == payer_id)

    rows = db.execute(q.limit(500)).all()
    entries = [_build_queue_entry(db, claim, denial) for claim, denial in rows]

    if tier:
        entries = [e for e in entries if e["priority_tier"] == tier]

    reverse = sort_by in {"priority_score", "expected_recovery", "claim_amount"}
    entries.sort(key=lambda e: e.get(sort_by, 0), reverse=reverse)

    return {
        "total": len(entries),
        "items": entries[offset : offset + min(limit, 200)],
    }


@router.get("/{claim_id}")
def get_recovery_queue_item(claim_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    claim = db.get(Claim, claim_id)
    denial = db.execute(select(DenialEvent).where(DenialEvent.claim_id == claim_id)).scalar_one_or_none()
    if not claim or not denial:
        return {"error": "Claim not found or not denied"}
    return _build_queue_entry(db, claim, denial)


@router.post("/simulate-strategy")
def simulate_recovery_strategy(
    staff_hours: float = 40.0,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Recovery Strategy Simulator (design doc Feature 13). Compares
    highest-claim-amount-first, highest-appeal-probability-first, and
    highest-expected-recovery-first (what the live queue does by default)
    against a real staff-capacity budget, using the actual current set of
    denied claims and the same effort estimates the Next-Best-Action engine
    produces. POST (not GET) since staff_hours is a scenario input the
    caller controls, not a resource identifier -- registered as its own
    literal path so it can never collide with GET /{claim_id}.
    """
    if staff_hours <= 0:
        raise HTTPException(status_code=400, detail="staff_hours must be positive")
    if staff_hours > 10000:
        raise HTTPException(status_code=400, detail="staff_hours is implausibly large")

    return simulate_strategies(db, staff_hours)
