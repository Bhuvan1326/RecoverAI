"""Pre-submission claim validator (Feature 8). Pure deterministic rules -- no ML."""
from app.models.domain import Claim


def validate_claim(claim: Claim) -> dict:
    checks: list[dict] = []
    errors: list[str] = []
    warnings: list[str] = []

    def check(name: str, passed: bool, level: str, message: str, pass_message: str | None = None):
        status = "PASS" if passed else level
        display_message = (pass_message or f"{name.replace('_', ' ').title()} OK.") if passed else message
        checks.append({"name": name, "status": status, "message": display_message})
        if not passed:
            (errors if level == "ERROR" else warnings).append(message)

    check(
        "eligibility",
        claim.eligibility_status == "VERIFIED",
        "ERROR",
        "Eligibility not verified for this claim.",
    )
    check(
        "authorization",
        claim.authorization_status == "PRESENT",
        "WARNING",
        "Authorization is missing; verify whether this service requires prior auth.",
    )
    check(
        "documentation",
        float(claim.documentation_completeness) >= 80,
        "WARNING",
        f"Documentation completeness is {claim.documentation_completeness}% (recommended >= 80%).",
    )
    check(
        "coding",
        bool(claim.lines) and all(l.procedure_code and l.diagnosis_code for l in claim.lines),
        "ERROR",
        "One or more claim lines is missing a procedure or diagnosis code.",
    )
    if claim.timely_filing_deadline and claim.submission_date:
        check(
            "timely_filing",
            claim.submission_date <= claim.timely_filing_deadline,
            "ERROR",
            "Submission date is after the timely filing deadline.",
        )
    else:
        checks.append({"name": "timely_filing", "status": "PASS", "message": "No filing deadline risk detected."})

    check(
        "required_fields",
        bool(claim.claim_number and claim.provider_id and claim.payer_id and claim.claim_amount),
        "ERROR",
        "One or more required claim fields is missing.",
    )

    total = len(checks)
    passed = sum(1 for c in checks if c["status"] == "PASS")
    # Errors weigh more heavily than warnings in the readiness score.
    penalty = len(errors) * 25 + len(warnings) * 10
    readiness_score = max(0, min(100, round(100 - penalty)))

    recommended_corrections = [c["message"] for c in checks if c["status"] != "PASS"]

    return {
        "claim_id": claim.id,
        "readiness_score": readiness_score,
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "recommended_corrections": recommended_corrections,
        "passed_checks": passed,
        "total_checks": total,
    }
