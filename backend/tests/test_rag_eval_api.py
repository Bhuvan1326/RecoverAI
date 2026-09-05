def test_rag_eval_get_requires_auth(client):
    r = client.get("/model-monitoring/rag-eval")
    assert r.status_code == 401


def test_rag_eval_get_reports_a_shape_either_way(client, admin_token):
    """Whether or not an eval was run before (depends on whatever's on disk
    at test time), the endpoint must return a `computed` boolean and never error."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/model-monitoring/rag-eval", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "computed" in body
    if not body["computed"]:
        assert "message" in body
    else:
        assert "retrieval_recall" in body
        assert "citation_correctness" in body


def test_rag_eval_compute_requires_reviewer_or_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.post("/model-monitoring/rag-eval/compute", headers=headers)
    assert r.status_code == 403


def test_rag_eval_compute_skips_gracefully_on_empty_db(client, admin_token):
    """No ingested documents and no denied claims in a fresh test DB --
    both sub-evaluations must report skipped=True, not error."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/model-monitoring/rag-eval/compute", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval_recall"]["skipped"] is True
    assert body["citation_correctness"]["skipped"] is True
