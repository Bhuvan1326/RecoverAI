def test_drift_compute_requires_reviewer_or_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.post("/model-monitoring/drift/compute", headers=headers)
    assert r.status_code == 403


def test_drift_compute_runs_without_error_on_empty_db(client, admin_token):
    """No claims/model trained in a fresh test DB -- must degrade gracefully
    (zero metrics computed), not error."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/model-monitoring/drift/compute", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["n_metrics_computed"] == 0


def test_drift_get_endpoint_requires_auth(client):
    r = client.get("/model-monitoring/drift")
    assert r.status_code == 401


def test_drift_get_endpoint_returns_list(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/model-monitoring/drift", headers=headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_drift_for_nonexistent_model_version_returns_404(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/model-monitoring/drift/nonexistent-id", headers=headers)
    assert r.status_code == 404
