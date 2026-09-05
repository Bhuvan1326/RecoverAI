"""
Recovery Strategy Simulator (design doc Feature 13). Compares real
prioritization strategies over the actual set of currently denied claims,
under a configurable staff-capacity constraint, and reports what each
strategy would actually capture -- not a canned demo number.

Strategies compared:
  - "highest_claim_amount": work the biggest-dollar claims first, ignoring
    likelihood of success entirely.
  - "highest_appeal_probability": work the claims most likely to succeed
    first, ignoring dollar amount.
  - "highest_expected_recovery": work claims ranked by expected recovery
    value (amount x probability) first -- what the live Recovery Queue
    actually does by default.

For each strategy, claims are processed in that order until the staff-hour
budget (converted to minutes, using the SAME per-action effort estimates
`recommend_next_best_action` already produces -- no separate/fake effort
model) is exhausted. Reports claims worked, staff-hours used, and
expected-recovery captured, so strategies can be compared on a like-for-like
capacity basis. Every result is labeled SIMULATED since actual appeal
outcomes are unknown until an appeal is actually resolved.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.domain import Claim, ClaimStatus, DenialEvent
from app.services.appeal_success import predict_appeal_success_batch
from app.services.recovery import expected_recovery_value, recommend_next_best_action

STRATEGIES = ["highest_claim_amount", "highest_appeal_probability", "highest_expected_recovery"]


def _load_denied_claims_with_context(db: Session) -> list[dict]:
    rows = db.execute(
        select(Claim, DenialEvent).join(DenialEvent, DenialEvent.claim_id == Claim.id).where(Claim.status == ClaimStatus.DENIED)
    ).all()
    if not rows:
        return []

    claims = [c for c, _ in rows]
    denials = [d for _, d in rows]
    appeal_predictions = predict_appeal_success_batch(db, list(zip(claims, denials)))

    context = []
    for claim, denial in zip(claims, denials):
        appeal = appeal_predictions[claim.id]
        recovery = expected_recovery_value(float(claim.claim_amount), appeal["appeal_success_probability"])
        nba = recommend_next_best_action(
            claim, denial.denial_reason_code, recovery["expected_recovery_value"], appeal["appeal_success_probability"]
        )
        context.append(
            {
                "claim_id": claim.id,
                "claim_number": claim.claim_number,
                "claim_amount": float(claim.claim_amount),
                "appeal_success_probability": appeal["appeal_success_probability"],
                "expected_recovery_value": recovery["expected_recovery_value"],
                "estimated_effort_minutes": nba["estimated_effort_minutes"],
            }
        )
    return context


def _order_for_strategy(claims: list[dict], strategy: str) -> list[dict]:
    if strategy == "highest_claim_amount":
        return sorted(claims, key=lambda c: c["claim_amount"], reverse=True)
    if strategy == "highest_appeal_probability":
        return sorted(claims, key=lambda c: c["appeal_success_probability"], reverse=True)
    if strategy == "highest_expected_recovery":
        return sorted(claims, key=lambda c: c["expected_recovery_value"], reverse=True)
    raise ValueError(f"Unknown strategy: {strategy}")


def _run_strategy(claims: list[dict], strategy: str, staff_minutes_budget: float) -> dict:
    ordered = _order_for_strategy(claims, strategy)

    minutes_used = 0.0
    claims_worked = 0
    recovery_captured = 0.0
    worked_claim_ids = []

    for claim in ordered:
        effort = max(claim["estimated_effort_minutes"], 1)
        if minutes_used + effort > staff_minutes_budget:
            break
        minutes_used += effort
        claims_worked += 1
        recovery_captured += claim["expected_recovery_value"] if claim["expected_recovery_value"] > 0 else 0.0
        worked_claim_ids.append(claim["claim_id"])

    total_expected_recovery = sum(c["expected_recovery_value"] for c in claims if c["expected_recovery_value"] > 0)
    recovery_yield = round(recovery_captured / total_expected_recovery, 4) if total_expected_recovery > 0 else 0.0

    return {
        "strategy": strategy,
        "claims_processed": claims_worked,
        "claims_available": len(claims),
        "staff_hours_used": round(minutes_used / 60, 2),
        "staff_hours_budget": round(staff_minutes_budget / 60, 2),
        "expected_recovery_captured": round(recovery_captured, 2),
        "total_expected_recovery_if_unlimited_capacity": round(total_expected_recovery, 2),
        "recovery_yield": recovery_yield,
        "worked_claim_ids": worked_claim_ids[:20],  # cap payload size; full ordering is reproducible from the strategy
    }


def simulate_strategies(db: Session, staff_hours: float) -> dict:
    """
    Runs all three strategies against the SAME snapshot of currently denied
    claims (fetched once so the comparison is apples-to-apples, not
    affected by claims changing status mid-simulation) and the same staff
    capacity budget.
    """
    claims = _load_denied_claims_with_context(db)
    if not claims:
        return {"label": "SIMULATED", "claims_available": 0, "strategies": [], "best_strategy": None}

    staff_minutes_budget = staff_hours * 60
    results = [_run_strategy(claims, s, staff_minutes_budget) for s in STRATEGIES]
    best = max(results, key=lambda r: r["expected_recovery_captured"])["strategy"]

    return {
        "label": "SIMULATED — expected values, not guaranteed outcomes",
        "claims_available": len(claims),
        "staff_hours_budget": staff_hours,
        "strategies": results,
        "best_strategy": best,
    }
