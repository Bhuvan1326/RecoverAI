"""
Tests run with CELERY_TASK_ALWAYS_EAGER=true (set in conftest.py), so
task.delay(...) executes synchronously in-process -- no live Redis/worker
required, while still exercising the REAL task bodies in app/workers/tasks.py
(not a mock). This verifies task logic correctness; it does not verify
actual Redis broker connectivity, which needs `docker compose up` (see
README's Docker verification section).
"""
from datetime import datetime, timezone


def _create_claim(client, headers, claim_number="CLM-JOB-1"):
    from app.models.domain import Payer, Provider

    override_gen_fn = next(iter(client.app.dependency_overrides.values()))
    db = next(override_gen_fn())
    payer = Payer(name="Job Test Payer")
    npi = "2" + str(abs(hash(claim_number)) % 10_000_000_000).zfill(10)[:9]
    provider = Provider(npi=npi, name="Job Test Provider")
    db.add(payer)
    db.add(provider)
    db.commit()
    db.refresh(payer)
    db.refresh(provider)

    payload = {
        "claim_number": claim_number,
        "provider_id": provider.id,
        "payer_id": payer.id,
        "patient_ref": "SYN-PT-JOB",
        "claim_amount": 2500.0,
        "service_date": "2026-01-01T00:00:00Z",
        "lines": [{"procedure_code": "99213", "diagnosis_code": "I10", "line_amount": 2500.0, "units": 1}],
    }
    r = client.post("/claims", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    return r.json()


def test_celery_app_is_eager_in_tests():
    from app.workers.celery_app import celery_app

    assert celery_app.conf.task_always_eager is True


def test_batch_score_endpoint_requires_auth(client):
    r = client.post("/claims/batch-score", json=["some-id"])
    assert r.status_code == 401


def test_batch_score_rejects_empty_list(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/claims/batch-score", headers=headers, json=[])
    assert r.status_code == 400


def test_batch_score_analyst_allowed_reviewer_role_matrix(client, admin_token, analyst_token):
    """BILLER/ANALYST/ADMIN may enqueue batch scoring (read/analyze-class
    work); this isn't a consequential/approval action so REVIEWER-only
    restriction doesn't apply here."""
    headers = {"Authorization": f"Bearer {analyst_token}"}
    claim = _create_claim(client, {"Authorization": f"Bearer {admin_token}"})
    r = client.post("/claims/batch-score", headers=headers, json=[claim["id"]])
    assert r.status_code == 200


def test_batch_score_job_completes_synchronously_under_eager_mode(client, admin_token):
    """
    Under CELERY_TASK_ALWAYS_EAGER, by the time /claims/batch-score returns,
    the task has already run to completion -- job status should already be
    SUCCESS (no trained denial model in this fresh test DB, so the task
    itself records a per-claim error rather than failing outright, and the
    JOB is still SUCCESS since the batch operation as a whole completed).
    """
    headers = {"Authorization": f"Bearer {admin_token}"}
    claim = _create_claim(client, headers)

    r = client.post("/claims/batch-score", headers=headers, json=[claim["id"]])
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "SUCCESS"

    r = client.get(f"/jobs/{job_id}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "SUCCESS"
    assert body["task_name"] == "score_claim_batch"
    assert body["started_at"] is not None
    assert body["finished_at"] is not None
    # No trained model -> the claim shows up in `errors`, not `results`.
    assert body["result"]["scored"] == 0
    assert len(body["result"]["errors"]) == 1


def test_job_status_404_for_unknown_job(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.get("/jobs/nonexistent-job-id", headers=headers)
    assert r.status_code == 404


def test_job_status_requires_auth(client):
    r = client.get("/jobs/some-id")
    assert r.status_code == 401


def test_drift_compute_async_job_persists_metrics(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/model-monitoring/drift/compute-async", headers=headers)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert r.json()["status"] == "SUCCESS"

    r = client.get(f"/jobs/{job_id}", headers=headers)
    body = r.json()
    assert body["result"]["n_metrics_computed"] == 0  # empty DB in this test


def test_drift_compute_async_requires_reviewer_or_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.post("/model-monitoring/drift/compute-async", headers=headers)
    assert r.status_code == 403


def test_ingest_documents_async_requires_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.post("/documents/ingest-async", headers=headers)
    assert r.status_code == 403


def test_ingest_documents_async_job_ingests_real_documents(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/documents/ingest-async", headers=headers)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    r = client.get(f"/jobs/{job_id}", headers=headers)
    body = r.json()
    assert body["status"] == "SUCCESS"
    assert body["result"]["documents_processed"] == 5
    assert body["result"]["chunks_created"] >= 5


def test_train_model_async_rejects_invalid_model_type(client, admin_token):
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/models/not_a_real_model/train-async", headers=headers)
    assert r.status_code == 400


def test_train_model_async_requires_admin(client, analyst_token):
    headers = {"Authorization": f"Bearer {analyst_token}"}
    r = client.post("/models/denial_risk/train-async", headers=headers)
    assert r.status_code == 403


def test_train_model_async_job_runs_and_reports_no_data_gracefully(client, admin_token):
    """No synthetic data in this fresh test DB -- the training script itself
    prints/returns early rather than crashing; the job should still report
    SUCCESS (the task completed; "not enough data" is a valid training
    outcome, not a task failure)."""
    headers = {"Authorization": f"Bearer {admin_token}"}
    r = client.post("/models/anomaly_detection/train-async", headers=headers)
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    r = client.get(f"/jobs/{job_id}", headers=headers)
    body = r.json()
    assert body["status"] == "SUCCESS"
    assert body["result"]["model_type"] == "anomaly_detection"
