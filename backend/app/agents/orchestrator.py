"""
Recovery Agent Orchestrator (Feature 18).

Implemented as an explicit, auditable tool-calling PIPELINE rather than a
free-form LLM agent loop -- every step is a real guarded tool call, in a
fixed, inspectable order, with every intermediate result written to the
audit log. This is a deliberate engineering choice: it gives 100% of the
guardrail/audit value of an "agent" with none of the non-determinism risk
of letting an LLM freely choose tool order for a financial workflow. The
LLM (via rag/llm.py) is used ONLY for the final natural-language appeal
drafting step, constrained to retrieved facts.

Flow (mirrors Section 25 / Section 45 of the spec):
  investigate -> retrieve policy -> analyze risk -> evidence check
  -> calculate recovery -> recommend action -> draft appeal (if evidence OK)
  -> create workflow_action (PENDING_APPROVAL) -> await human decision
"""
from sqlalchemy.orm import Session

from app.agents import tools
from app.models.domain import Claim, DenialEvent, WorkflowActionType
from app.rag.llm import get_llm_provider
from app.services.audit import record_event
from app.services.denial_reason import predict_denial_reason
from app.services.evidence_checker import check_evidence_completeness


def investigate_and_recommend(db: Session, claim_id: str) -> dict:
    from sqlalchemy import select

    claim = db.get(Claim, claim_id)
    if not claim:
        return {"error": "claim not found"}
    denial = db.execute(select(DenialEvent).where(DenialEvent.claim_id == claim_id)).scalar_one_or_none()

    record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.investigation_started", claim_id=claim_id)

    risk = tools.tool_predict_denial(db, claim)
    record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.risk_analyzed", claim_id=claim_id, payload=risk)

    reason_prediction = predict_denial_reason(db, claim)
    denial_reason = denial.denial_reason_code if denial else reason_prediction["predicted_reason"]

    retrieved = tools.tool_search_documents(db, query=f"{denial_reason} appeal policy", payer_id=claim.payer_id, top_k=5)
    record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.policy_retrieved", claim_id=claim_id, payload={"n_chunks": len(retrieved)})

    evidence = check_evidence_completeness(claim, denial, retrieved)
    record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.evidence_checked", claim_id=claim_id, payload=evidence)

    recovery = tools.tool_calculate_recovery(db, claim, denial)
    appeal_pred = tools.tool_predict_appeal_success(db, claim, denial)
    anomaly = tools.tool_calculate_anomaly(db, claim_id)
    record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.anomaly_checked", claim_id=claim_id, payload={"anomaly_score": anomaly.get("anomaly_score"), "severity": anomaly.get("severity")})
    nba = tools.tool_recommend_action(claim, denial_reason, recovery["expected_recovery_value"], appeal_pred["appeal_success_probability"])
    record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.action_recommended", claim_id=claim_id, payload=nba)

    return {
        "claim_id": claim_id,
        "risk": risk,
        "denial_reason": denial_reason,
        "retrieved_policy_chunks": retrieved,
        "evidence_completeness": evidence,
        "expected_recovery": recovery,
        "appeal_success_prediction": appeal_pred,
        "anomaly": anomaly,
        "next_best_action": nba,
    }


def draft_appeal(db: Session, claim_id: str, created_by: str | None = None) -> dict:
    """
    Runs the full investigation, then -- ONLY if the evidence-completeness
    gate passes -- generates a citation-grounded draft and creates a
    PENDING_APPROVAL workflow_action. Never submits anything; the agent's
    authority ends at proposing a draft for human review.
    """

    investigation = investigate_and_recommend(db, claim_id)
    if "error" in investigation:
        return investigation

    claim = db.get(Claim, claim_id)
    evidence = investigation["evidence_completeness"]

    if not evidence["generation_allowed"]:
        record_event(db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.draft_blocked_insufficient_evidence", claim_id=claim_id, payload=evidence)
        return {
            "blocked": True,
            "reason": "Evidence completeness below threshold; appeal draft not generated.",
            "evidence_completeness": evidence,
        }

    claim_facts = {
        "claim_number": claim.claim_number,
        "claim_amount": float(claim.claim_amount),
        "service_date": str(claim.service_date),
        "documentation_completeness": float(claim.documentation_completeness),
        "authorization_status": claim.authorization_status,
    }
    llm = get_llm_provider()
    draft = llm.draft_appeal(
        claim_facts=claim_facts,
        denial_reason=investigation["denial_reason"],
        retrieved_chunks=investigation["retrieved_policy_chunks"],
        missing_evidence=evidence["missing_evidence"],
    )

    action = tools.tool_create_workflow_action(
        db,
        claim_id=claim_id,
        action_type=WorkflowActionType.DRAFT_APPEAL,
        payload={
            "draft_text": draft["draft_text"],
            "citations": draft["citations"],
            "missing_evidence": draft["missing_evidence"],
            "investigation": investigation,
        },
    )
    record_event(
        db, actor_type="agent", actor_id="recovery_orchestrator", event_type="agent.appeal_drafted",
        claim_id=claim_id, payload={"workflow_action_id": action.id, "n_citations": len(draft["citations"])},
    )

    return {
        "blocked": False,
        "workflow_action_id": action.id,
        "draft_text": draft["draft_text"],
        "citations": draft["citations"],
        "status": action.status.value,
        "investigation": investigation,
    }
