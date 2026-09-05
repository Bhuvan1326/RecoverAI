from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import require_roles
from app.models.domain import AuditLog, User, UserRole
from app.services.audit import verify_chain

router = APIRouter(prefix="/audit-logs", tags=["audit"])


@router.get("")
def list_audit_logs(
    claim_id: str | None = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db),
    _user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    q = select(AuditLog)
    if claim_id:
        q = q.where(AuditLog.claim_id == claim_id)
    q = q.order_by(AuditLog.created_at.desc()).offset(offset).limit(min(limit, 500))
    rows = db.execute(q).scalars().all()
    return [
        {
            "id": r.id,
            "actor_type": r.actor_type,
            "actor_id": r.actor_id,
            "event_type": r.event_type,
            "claim_id": r.claim_id,
            "payload": r.payload,
            "hash": r.hash,
            "created_at": r.created_at,
        }
        for r in rows
    ]


@router.get("/verify")
def verify(db: Session = Depends(get_db), _user: User = Depends(require_roles(UserRole.ADMIN))):
    ok, broken_at = verify_chain(db)
    return {"chain_valid": ok, "broken_at_id": broken_at}
