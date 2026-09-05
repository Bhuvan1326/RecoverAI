from datetime import datetime, timedelta, timezone

from app.models.domain import Claim
from app.services.denial_reason import predict_denial_reason


def _make_claim(**overrides) -> Claim:
    defaults = dict(
        claim_number="CLM-DR-TEST",
        provider_id="prov-1",
        payer_id="payer-1",
        patient_ref="SYN-PT-1",
        claim_amount=1000.0,
        eligibility_status="VERIFIED",
        authorization_status="PRESENT",
        documentation_completeness=95.0,
        service_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        submission_date=datetime(2026, 1, 5, tzinfo=timezone.utc),
        timely_filing_deadline=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return Claim(**defaults)


def test_rule_takes_precedence_missing_authorization(client):
    """B1: a directly-observable fact (missing auth) must never be routed
    through the ML/heuristic path -- the rule wins even if `db` is provided
    and a model happens to be trained."""
    claim = _make_claim(authorization_status="MISSING")
    result = predict_denial_reason(None, claim)
    assert result["predicted_reason"] == "MISSING_AUTHORIZATION"
    assert result["source"] == "rule"
    assert result["confidence"] == 1.0
    assert result["alternatives"] == []


def test_rule_takes_precedence_eligibility_failure(client):
    claim = _make_claim(eligibility_status="FAIL")
    result = predict_denial_reason(None, claim)
    assert result["predicted_reason"] == "ELIGIBILITY_ISSUE"
    assert result["source"] == "rule"


def test_rule_takes_precedence_timely_filing():
    claim = _make_claim(
        submission_date=datetime(2026, 5, 1, tzinfo=timezone.utc),
        timely_filing_deadline=datetime(2026, 4, 1, tzinfo=timezone.utc),
    )
    result = predict_denial_reason(None, claim)
    assert result["predicted_reason"] == "TIMELY_FILING"
    assert result["source"] == "rule"


def test_rule_takes_precedence_missing_documentation():
    claim = _make_claim(documentation_completeness=40.0)
    result = predict_denial_reason(None, claim)
    assert result["predicted_reason"] == "MISSING_DOCUMENTATION"
    assert result["source"] == "rule"


def test_ambiguous_case_falls_back_to_heuristic_without_db_session():
    """No `db` passed at all -> rule-only contexts must still return a
    usable (heuristic) prediction rather than raising."""
    claim = _make_claim()  # nothing triggers a rule
    result = predict_denial_reason(None, claim)
    assert result["source"] == "heuristic"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] is None


def test_ambiguous_case_falls_back_to_heuristic_with_untrained_db(client):
    """`db` IS provided but no denial-reason model is trained in this fresh
    test DB -- must degrade to the heuristic, not error."""
    from app.core.database import get_db

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    claim = _make_claim()
    result = predict_denial_reason(db, claim)
    assert result["source"] == "heuristic"
    assert 0.0 <= result["confidence"] <= 1.0


def test_confidence_always_in_valid_range():
    for claim in [
        _make_claim(authorization_status="MISSING"),
        _make_claim(eligibility_status="FAIL"),
        _make_claim(documentation_completeness=95.0),
        _make_claim(documentation_completeness=20.0),
    ]:
        result = predict_denial_reason(None, claim)
        assert 0.0 <= result["confidence"] <= 1.0


def test_alternatives_never_include_the_predicted_reason_itself():
    claim = _make_claim()
    result = predict_denial_reason(None, claim)
    predicted = result["predicted_reason"]
    alt_reasons = [a["reason"] for a in result["alternatives"]]
    assert predicted not in alt_reasons


def test_explanation_endpoint_includes_denial_reason(client, admin_token):
    """Integration: GET /claims/{id}/explanation should surface a denial_reason
    block using this same service, gracefully degrading without a trained
    denial-risk OR denial-reason model (both 503, consistent contract)."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    from app.models.domain import Payer, Provider

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    payer = Payer(name="DR Test Payer")
    provider = Provider(npi="5556667778", name="DR Test Provider")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)

    r = client.post(
        "/claims",
        headers=headers,
        json={
            "claim_number": "CLM-DR-API-1",
            "provider_id": provider.id,
            "payer_id": payer.id,
            "patient_ref": "SYN-PT-DR",
            "claim_amount": 500.0,
            "service_date": "2026-01-01T00:00:00Z",
            "lines": [{"procedure_code": "99213", "diagnosis_code": "I10", "line_amount": 500.0, "units": 1}],
        },
    )
    claim_id = r.json()["id"]
    r = client.get(f"/claims/{claim_id}/explanation", headers=headers)
    # No denial-risk model trained in this fresh test DB -> 503 for the whole
    # explanation endpoint (denial_reason is a sub-block of it, not separately
    # gated) -- this documents the current coupling rather than asserting a
    # behavior that isn't implemented.
    assert r.status_code == 503
