"""
Workflow-action approval endpoints. This is the ONLY place in the entire
application where a WorkflowAction transitions out of PENDING_APPROVAL --
it requires REVIEWER or ADMIN role, server-side-enforced, and is never
callable from agent code (see guardrails/engine.py BLOCKED_AI_ACTIONS,
which includes approve_workflow_action/reject_workflow_action).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import require_roles
from app.models.domain import HumanReview, User, UserRole, WorkflowAction, WorkflowActionStatus
from app.schemas.schemas import WorkflowActionApprovalIn, WorkflowActionOut
from app.services.audit import record_event

router = APIRouter(prefix="/workflow-actions", tags=["workflow"])


@router.get("/{action_id}", response_model=WorkflowActionOut)
def get_action(action_id: str, db: Session = Depends(get_db), _user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER, UserRole.BILLER, UserRole.ANALYST))):
    action = db.get(WorkflowAction, action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Workflow action not found")
    return action


@router.post("/{action_id}/approve", response_model=WorkflowActionOut)
def approve(
    action_id: str,
    payload: WorkflowActionApprovalIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    action = db.get(WorkflowAction, action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Workflow action not found")
    if action.status != WorkflowActionStatus.PENDING_APPROVAL:
        raise HTTPException(status_code=400, detail=f"Action is not pending approval (status={action.status.value})")

    action.status = WorkflowActionStatus.APPROVED
    db.add(HumanReview(workflow_action_id=action.id, reviewer_id=user.id, decision="APPROVED", notes=payload.notes))
    db.commit()
    db.refresh(action)

    record_event(db, actor_type="user", actor_id=user.id, event_type="workflow_action.approved", claim_id=action.claim_id, payload={"action_id": action.id, "notes": payload.notes})
    return action


@router.post("/{action_id}/reject", response_model=WorkflowActionOut)
def reject(
    action_id: str,
    payload: WorkflowActionApprovalIn,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    action = db.get(WorkflowAction, action_id)
    if not action:
        raise HTTPException(status_code=404, detail="Workflow action not found")
    if action.status != WorkflowActionStatus.PENDING_APPROVAL:
        raise HTTPException(status_code=400, detail=f"Action is not pending approval (status={action.status.value})")

    action.status = WorkflowActionStatus.REJECTED
    db.add(HumanReview(workflow_action_id=action.id, reviewer_id=user.id, decision="REJECTED", notes=payload.notes))
    db.commit()
    db.refresh(action)

    record_event(db, actor_type="user", actor_id=user.id, event_type="workflow_action.rejected", claim_id=action.claim_id, payload={"action_id": action.id, "notes": payload.notes})
    return action
