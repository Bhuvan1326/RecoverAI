"""
Evidence completeness checker (Feature 17). Runs BEFORE appeal-draft
generation is permitted; blocks generation when required evidence is
missing rather than letting the LLM paper over a gap.
"""
from app.models.domain import Claim, DenialEvent

REQUIRED_EVIDENCE_ITEMS = [
    "claim_record",
    "denial_reason",
    "procedure_information",
    "authorization_status",
    "supporting_documentation",
    "payer_policy_retrieved",
]

READINESS_THRESHOLD = 60  # below this, appeal-draft generation is blocked


def check_evidence_completeness(claim: Claim, denial: DenialEvent | None, retrieved_chunks: list[dict]) -> dict:
    present = {
        "claim_record": True,
        "denial_reason": denial is not None,
        "procedure_information": bool(claim.lines),
        "authorization_status": claim.authorization_status is not None,
        "supporting_documentation": float(claim.documentation_completeness) >= 70,
        "payer_policy_retrieved": len(retrieved_chunks) > 0,
    }
    missing = [k for k, v in present.items() if not v]
    readiness_pct = round(100 * sum(present.values()) / len(present))

    return {
        "readiness_percent": readiness_pct,
        "items": present,
        "missing_evidence": missing,
        "generation_allowed": readiness_pct >= READINESS_THRESHOLD and "denial_reason" not in missing,
    }
