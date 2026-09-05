"""
Expected Recovery Value engine, appeal-success heuristic, and Next-Best-Action
scoring engine (Features 4, 6, 15). All formulas are explicit and every
component of the calculation is returned so the UI can show its work
(Section 11 requirement: "do not hardcode the output").
"""
from app.models.domain import Claim, DenialEvent

# Appealability priors by denial reason -- same documented assumption used
# by the synthetic data generator (scripts/generate_data.py), kept in sync
# here so the heuristic is internally consistent with how labels were made.
APPEALABILITY_PRIOR = {
    "MISSING_AUTHORIZATION": 0.75,
    "ELIGIBILITY_ISSUE": 0.55,
    "CODING_MISMATCH": 0.65,
    "DUPLICATE_CLAIM": 0.10,
    "MISSING_DOCUMENTATION": 0.70,
    "TIMELY_FILING": 0.15,
    "MEDICAL_NECESSITY": 0.45,
    "OTHER": 0.30,
}

DEFAULT_PROCESSING_COST = 250.0
DEFAULT_PAYMENT_RECOVERY_PROBABILITY = 0.90  # P(payer actually pays once an appeal is won)


def predict_appeal_success_heuristic(claim: Claim, denial: DenialEvent | None) -> dict:
    """
    Heuristic appeal-success baseline (Feature 15). Used as the fallback
    when no trained model is registered yet -- see
    app/services/appeal_success.py:predict_appeal_success, which is the
    function the rest of the app calls. Explicitly labeled as a heuristic,
    NOT a validated supervised model.
    """
    reason = denial.denial_reason_code if denial else "OTHER"
    prior = APPEALABILITY_PRIOR.get(reason, 0.30)
    doc_factor = float(claim.documentation_completeness) / 100.0
    probability = max(0.02, min(0.97, prior * (0.5 + 0.5 * doc_factor)))
    return {
        "appeal_success_probability": round(probability, 4),
        "source": "heuristic_baseline",
        "basis": {"denial_reason": reason, "appealability_prior": prior, "documentation_completeness": doc_factor},
        "disclaimer": "Heuristic estimate based on documented assumptions, not a validated ML model.",
    }


def expected_recovery_value(
    claim_amount: float,
    appeal_success_probability: float,
    payment_recovery_probability: float = DEFAULT_PAYMENT_RECOVERY_PROBABILITY,
    processing_cost: float = DEFAULT_PROCESSING_COST,
) -> dict:
    value = claim_amount * appeal_success_probability * payment_recovery_probability - processing_cost
    return {
        "claim_amount": round(claim_amount, 2),
        "appeal_success_probability": round(appeal_success_probability, 4),
        "payment_recovery_probability": round(payment_recovery_probability, 4),
        "processing_cost": round(processing_cost, 2),
        "expected_recovery_value": round(value, 2),
    }


def priority_score(expected_recovery: float, urgency_factor: float, recoverability_factor: float, estimated_effort_minutes: float) -> float:
    effort = max(estimated_effort_minutes, 1)
    return round((expected_recovery * urgency_factor * recoverability_factor) / effort, 4)


def urgency_factor(days_since_denial: int, days_to_appeal_deadline: int = 60) -> float:
    remaining = max(days_to_appeal_deadline - days_since_denial, 1)
    return round(1.0 + (days_to_appeal_deadline - remaining) / days_to_appeal_deadline, 4)


NEXT_BEST_ACTION_RULES: list[tuple[str, str, str]] = [
    # (condition_name, action, human_reason) -- evaluated in order, first match wins.
    ("auth_missing", "OBTAIN_AUTHORIZATION", "Claim is missing prior authorization; obtaining it is the highest-leverage fix."),
    ("eligibility_fail", "VERIFY_ELIGIBILITY", "Eligibility could not be verified; confirm coverage before further action."),
    ("low_documentation", "REQUEST_DOCUMENTATION", "Documentation completeness is below threshold; request supporting records."),
    ("coding_mismatch", "CORRECT_CODING", "Coding pattern suggests a procedure/diagnosis mismatch; review and correct coding."),
    ("timely_filing_risk", "ESCALATE", "Claim is near or past the timely filing deadline; escalate for urgent handling."),
    ("low_expected_recovery", "STOP_RECOVERY", "Expected recovery value is negative or negligible; further work is not cost-justified."),
]


def recommend_next_best_action(
    claim: Claim,
    denial_reason: str | None,
    expected_recovery: float,
    appeal_success_probability: float,
) -> dict:
    if claim.authorization_status == "MISSING":
        action, reason = "OBTAIN_AUTHORIZATION", "Claim is missing prior authorization; obtaining it is the highest-leverage fix."
    elif claim.eligibility_status == "FAIL":
        action, reason = "VERIFY_ELIGIBILITY", "Eligibility could not be verified; confirm coverage before further action."
    elif float(claim.documentation_completeness) < 70:
        action, reason = "REQUEST_DOCUMENTATION", "Documentation completeness is below threshold; request supporting records."
    elif denial_reason == "CODING_MISMATCH":
        action, reason = "CORRECT_CODING", "Coding pattern suggests a procedure/diagnosis mismatch; review and correct coding."
    elif expected_recovery <= 0:
        action, reason = "STOP_RECOVERY", "Expected recovery value is negative or negligible; further work is not cost-justified."
    elif appeal_success_probability >= 0.5:
        action, reason = "APPEAL", "Appeal success probability and expected recovery both support pursuing an appeal."
    else:
        action, reason = "CONTACT_PAYER", "Ambiguous denial; contact payer for clarification before committing further effort."

    effort_minutes = {
        "OBTAIN_AUTHORIZATION": 20,
        "VERIFY_ELIGIBILITY": 10,
        "REQUEST_DOCUMENTATION": 15,
        "CORRECT_CODING": 25,
        "APPEAL": 45,
        "CONTACT_PAYER": 20,
        "ESCALATE": 30,
        "STOP_RECOVERY": 5,
        "RESUBMIT": 15,
    }.get(action, 20)

    return {
        "recommended_action": action,
        "reason": reason,
        "expected_recovery": round(expected_recovery, 2),
        "success_probability": round(appeal_success_probability, 4),
        "estimated_effort_minutes": effort_minutes,
        "evidence": {
            "authorization_status": claim.authorization_status,
            "eligibility_status": claim.eligibility_status,
            "documentation_completeness": float(claim.documentation_completeness),
            "denial_reason": denial_reason,
        },
    }
