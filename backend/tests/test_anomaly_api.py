def _create_payer_provider_claim(client, headers, claim_number="CLM-ANOM-1"):
    from app.models.domain import Payer, Provider

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    payer = Payer(name="Anomaly Test Payer")
    provider = Provider(npi="9998887776", name="Anomaly Test Provider")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)

    payload = {
        "claim_number": claim_number,
        "provider_id": provider.id,
        "payer_id": payer.id,
        "patient_ref": "SYN-PT-ANOM",
        "claim_amount": 5000.0,
        "service_date": "2026-01-01T00:00:00Z",
        "lines": [{"procedure_code": "99213", "diagnosis_code": "I10", "line_amount": 5000.0, "units": 1}],
    }
    r = client.post("/claims", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_anomaly_score_endpoint_returns_503_without_trained_model(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = _create_payer_provider_claim(client, headers)
    r = client.post(f"/claims/{claim['id']}/anomaly-score", headers=headers)
    assert r.status_code == 503
    assert "anomaly" in r.json()["detail"].lower()


def test_anomaly_get_endpoint_returns_503_without_trained_model(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = _create_payer_provider_claim(client, headers, claim_number="CLM-ANOM-2")
    r = client.get(f"/claims/{claim['id']}/anomaly", headers=headers)
    assert r.status_code == 503


def test_anomaly_score_endpoint_requires_auth(client):
    r = client.post("/claims/some-id/anomaly-score")
    assert r.status_code == 401


def test_anomaly_score_endpoint_404_for_missing_claim(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/claims/nonexistent-claim-id/anomaly-score", headers=headers)
    assert r.status_code == 404


def test_analytics_anomalies_degrades_gracefully_without_model(client, admin_token):
    """
    Unlike the per-claim endpoints (which 503 -- a single claim genuinely
    can't be scored), the aggregate analytics view is designed to degrade
    gracefully: it returns 200 with an explanatory `error` field and an
    empty result set, so the dashboard doesn't hard-fail just because the
    anomaly model hasn't been trained yet.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/analytics/anomalies", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["items"] == []
    assert "error" in body
