


def _create_payer_provider(client, token):
    # No dedicated payer/provider creation endpoints in the MVP API surface
    # (they're seeded via the synthetic generator); insert directly via the
    # ORM session for this unit test instead.
    from app.core.database import SessionLocal  # noqa
    return None


def create_test_claim(client, headers, **overrides):
    from app.models.domain import Payer, Provider

    # Use the same overridden session factory the app is using in this test run.
    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    payer = Payer(name="Test Payer")
    provider = Provider(npi="1234567890", name="Test Provider")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)

    payload = {
        "claim_number": overrides.get("claim_number", "CLM-UNIT-1"),
        "provider_id": provider.id,
        "payer_id": payer.id,
        "patient_ref": "SYN-PT-1",
        "claim_amount": overrides.get("claim_amount", 1500.0),
        "eligibility_status": overrides.get("eligibility_status", "VERIFIED"),
        "authorization_status": overrides.get("authorization_status", "PRESENT"),
        "documentation_completeness": overrides.get("documentation_completeness", 95.0),
        "service_date": "2026-01-01T00:00:00Z",
        "lines": [{"procedure_code": "99213", "diagnosis_code": "I10", "line_amount": 1500.0, "units": 1}],
    }
    r = client.post("/claims", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_create_and_get_claim(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = create_test_claim(client, headers)
    r = client.get(f"/claims/{claim['id']}", headers=headers)
    assert r.status_code == 200
    assert r.json()["claim_number"] == "CLM-UNIT-1"


def test_duplicate_claim_number_rejected(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    create_test_claim(client, headers, claim_number="CLM-DUP")
    r = client.post(
        "/claims",
        headers=headers,
        json={
            "claim_number": "CLM-DUP",
            "provider_id": "x",
            "payer_id": "y",
            "patient_ref": "SYN-2",
            "claim_amount": 1.0,
            "service_date": "2026-01-01T00:00:00Z",
        },
    )
    assert r.status_code == 400


def test_validator_flags_missing_authorization(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = create_test_claim(client, headers, authorization_status="MISSING", claim_number="CLM-NOAUTH")
    r = client.post(f"/claims/{claim['id']}/validate", headers=headers)
    assert r.status_code == 200
    body = r.json()
    auth_check = next(c for c in body["checks"] if c["name"] == "authorization")
    assert auth_check["status"] == "WARNING"
    assert body["readiness_score"] < 100


def test_validator_passes_clean_claim(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = create_test_claim(client, headers, claim_number="CLM-CLEAN")
    r = client.post(f"/claims/{claim['id']}/validate", headers=headers)
    body = r.json()
    assert body["readiness_score"] == 100
    assert body["errors"] == []


def test_score_endpoint_returns_503_when_no_champion_model(client, admin_token):
    """No model has been trained/registered in this fresh test DB -- the API
    must fail with a clear 503, never with a silently fabricated score."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = create_test_claim(client, headers, claim_number="CLM-NOMODEL")
    r = client.post(f"/claims/{claim['id']}/score", headers=headers)
    assert r.status_code == 503
    assert "No champion model" in r.json()["detail"]
