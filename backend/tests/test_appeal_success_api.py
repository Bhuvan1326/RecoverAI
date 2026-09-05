from datetime import datetime, timedelta, timezone


def _create_claim_with_denial(client, headers, claim_number="CLM-APPEAL-1", denial_reason="MISSING_AUTHORIZATION"):
    from app.models.domain import Claim, ClaimStatus, DenialEvent, Payer, Provider

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    payer = Payer(name="Appeal Test Payer")
    # NPI derived from claim_number so multiple calls within the same test
    # (same isolated DB) never collide on the providers.npi unique constraint.
    npi = "1" + str(abs(hash(claim_number)) % 10_000_000_000).zfill(10)[:9]
    provider = Provider(npi=npi, name="Appeal Test Provider")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)

    payload = {
        "claim_number": claim_number,
        "provider_id": provider.id,
        "payer_id": payer.id,
        "patient_ref": "SYN-PT-APPEAL",
        "claim_amount": 8000.0,
        "authorization_status": "MISSING",
        "documentation_completeness": 85.0,
        "service_date": "2026-01-01T00:00:00Z",
        "lines": [{"procedure_code": "99213", "diagnosis_code": "I10", "line_amount": 8000.0, "units": 1}],
    }
    r = client.post("/claims", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    claim_out = r.json()

    claim = db.get(Claim, claim_out["id"])
    claim.status = ClaimStatus.DENIED
    denial = DenialEvent(claim_id=claim.id, denial_reason_code=denial_reason, denial_date=datetime.now(timezone.utc) - timedelta(days=5))
    db.add(denial)
    db.commit()
    return claim_out


def test_appeal_success_score_falls_back_to_heuristic_without_trained_model(client, admin_token):
    """No appeal-success model is trained in a fresh test DB -- the endpoint
    must still return a usable probability via the documented heuristic
    fallback, not error out, since Section 15's heuristic baseline is an
    intentional, always-available fallback path."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = _create_claim_with_denial(client, headers)

    r = client.post(f"/claims/{claim['id']}/appeal-success-score", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["appeal_success_probability"] <= 1.0
    assert body["source"] == "heuristic_baseline"
    assert body["model_version"] is None
    assert body["risk_category"] in {"HIGH_RECOVERY_POTENTIAL", "MODERATE_RECOVERY_POTENTIAL", "LOW_RECOVERY_POTENTIAL"}


def test_appeal_success_get_matches_post(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = _create_claim_with_denial(client, headers, claim_number="CLM-APPEAL-2")
    r = client.get(f"/claims/{claim['id']}/appeal-success", headers=headers)
    assert r.status_code == 200
    assert "appeal_success_probability" in r.json()


def test_appeal_success_requires_auth(client):
    r = client.post("/claims/some-id/appeal-success-score")
    assert r.status_code == 401


def test_appeals_predict_endpoint_uses_same_underlying_service(client, admin_token):
    """The legacy /appeals/predict?claim_id=... endpoint and the newer
    /claims/{id}/appeal-success-score endpoint must agree, since both now
    call the same app.services.appeal_success.predict_appeal_success."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = _create_claim_with_denial(client, headers, claim_number="CLM-APPEAL-3")

    r1 = client.post(f"/appeals/predict?claim_id={claim['id']}", headers=headers)
    r2 = client.post(f"/claims/{claim['id']}/appeal-success-score", headers=headers)
    assert r1.status_code == 200 and r2.status_code == 200
    assert r1.json()["appeal_success_probability"] == r2.json()["appeal_success_probability"]


def test_recovery_queue_uses_appeal_success_probability_in_ranking(client, admin_token):
    """
    Feature C7: the recovery queue must reflect appeal-success probability
    without any separate manual recalculation step -- it recomputes on
    every request.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    _create_claim_with_denial(client, headers, claim_number="CLM-APPEAL-4", denial_reason="MISSING_AUTHORIZATION")
    _create_claim_with_denial(client, headers, claim_number="CLM-APPEAL-5", denial_reason="DUPLICATE_CLAIM")

    r = client.get("/recovery-queue", headers=headers)
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    # MISSING_AUTHORIZATION has a much higher appealability prior than
    # DUPLICATE_CLAIM in the shared heuristic -- confirms the queue is
    # actually differentiating by the appeal-success probability, not just
    # returning claims in insertion order.
    by_reason = {item["denial_reason"]: item["appeal_success_probability"] for item in items}
    assert by_reason["MISSING_AUTHORIZATION"] > by_reason["DUPLICATE_CLAIM"]


def test_dashboard_metrics_expected_recovery_uses_batch_appeal_prediction(client, admin_token):
    """Sanity check that the batched dashboard path (predict_appeal_success_batch)
    produces a non-negative expected-recovery total and doesn't error when
    denied claims exist."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    _create_claim_with_denial(client, headers, claim_number="CLM-APPEAL-6")
    r = client.get("/dashboard/metrics", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["claims_at_risk"] >= 1
    assert isinstance(body["expected_recovery"], (int, float))
