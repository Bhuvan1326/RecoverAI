def test_recovery_ranking_get_requires_auth(client):
    r = client.get("/model-monitoring/recovery-ranking")
    assert r.status_code == 401


def test_recovery_ranking_get_reports_a_shape_either_way(client, admin_token):
    """
    Whether or not an evaluation has been run before (this depends on
    whatever's on disk in ml/models/artifacts/ at test time, which this
    test deliberately doesn't control), the endpoint must always return a
    `computed` boolean and never error.
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/model-monitoring/recovery-ranking", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "computed" in body
    if not body["computed"]:
        assert "message" in body
    else:
        assert "strategies" in body


def test_recovery_ranking_compute_requires_reviewer_or_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.post("/model-monitoring/recovery-ranking/compute", headers=headers)
    assert r.status_code == 403


def test_recovery_ranking_compute_skips_gracefully_on_empty_db(client, admin_token):
    """No resolved appeals in a fresh test DB -- must return a clear
    'skipped' result, not error."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/model-monitoring/recovery-ranking/compute", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["skipped"] is True
    assert "reason" in body
