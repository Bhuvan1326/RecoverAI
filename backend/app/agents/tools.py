"""
Agent tools (Feature 18). Every tool is decorated with @guarded(action_name)
so it passes through the guardrail engine (app/guardrails/engine.py) on
every call -- this is enforced in code, independent of what the LLM/agent
"decides" to do, satisfying Section 26's "guardrails in code, not prompts
only" requirement.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.guardrails.engine import guarded
from app.models.domain import Claim, DenialEvent, WorkflowAction, WorkflowActionStatus, WorkflowActionType
from app.rag.retrieval import retrieve
from app.services import ml_inference
from app.services.recovery import expected_recovery_value, recommend_next_best_action
from app.services.appeal_success import predict_appeal_success
from app.services.validator import validate_claim


@guarded("get_claim")
def get_claim(db: Session, claim_id: str) -> dict:
    claim = db.get(Claim, claim_id)
    if not claim:
        return {}
    return {
        "id": claim.id,
        "claim_number": claim.claim_number,
        "claim_amount": float(claim.claim_amount),
        "status": claim.status.value,
        "authorization_status": claim.authorization_status,
        "eligibility_status": claim.eligibility_status,
        "documentation_completeness": float(claim.documentation_completeness),
    }


@guarded("get_denial")
def get_denial(db: Session, claim_id: str) -> dict | None:
    denial = db.execute(select(DenialEvent).where(DenialEvent.claim_id == claim_id)).scalar_one_or_none()
    if not denial:
        return None
    return {"denial_reason_code": denial.denial_reason_code, "denial_date": str(denial.denial_date), "id": denial.id}


@guarded("predict_denial")
def tool_predict_denial(db: Session, claim: Claim) -> dict:
    return ml_inference.score_claim(db, claim)


@guarded("get_shap_explanation")
def tool_get_shap_explanation(db: Session, claim: Claim) -> dict:
    return ml_inference.explain_claim(db, claim)


@guarded("calculate_anomaly")
def tool_calculate_anomaly(db: Session, claim_id: str) -> dict:
    from app.services import anomaly as anomaly_service

    try:
        return anomaly_service.score_claim(db, claim_id)
    except anomaly_service.AnomalyModelNotTrainedError:
        return {"claim_id": claim_id, "anomaly_score": None, "note": "Anomaly model not trained yet."}


@guarded("search_documents")
def tool_search_documents(db: Session, query: str, payer_id: str | None = None, top_k: int = 5) -> list[dict]:
    return retrieve(db, query, top_k=top_k, payer_id=payer_id)


@guarded("calculate_recovery")
def tool_calculate_recovery(db: Session, claim: Claim, denial: DenialEvent | None) -> dict:
    appeal = predict_appeal_success(db, claim, denial)
    return expected_recovery_value(float(claim.claim_amount), appeal["appeal_success_probability"])


@guarded("validate_claim")
def tool_validate_claim(claim: Claim) -> dict:
    return validate_claim(claim)


@guarded("predict_appeal_success")
def tool_predict_appeal_success(db: Session, claim: Claim, denial: DenialEvent | None) -> dict:
    return predict_appeal_success(db, claim, denial)


@guarded("recommend_action")
def tool_recommend_action(claim: Claim, denial_reason: str | None, expected_recovery: float, appeal_prob: float) -> dict:
    return recommend_next_best_action(claim, denial_reason, expected_recovery, appeal_prob)


@guarded("create_workflow_action")
def tool_create_workflow_action(db: Session, claim_id: str, action_type: WorkflowActionType, payload: dict) -> WorkflowAction:
    """
    Creates a PENDING_APPROVAL row ONLY. This tool never sets status to
    APPROVED/EXECUTED -- that transition can only happen via the human
    approval endpoint (app/api/workflow.py), which itself is gated by
    require_roles(REVIEWER, ADMIN) and is NOT reachable from agent code.
    """
    action = WorkflowAction(
        claim_id=claim_id,
        action_type=action_type,
        recommended_by="agent",
        payload=payload,
        status=WorkflowActionStatus.PENDING_APPROVAL,
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action
