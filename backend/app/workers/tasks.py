"""
Celery background tasks (Phase 2, Feature E). Each task:
  1. opens its OWN DB session (workers are separate processes/threads --
     can't reuse a FastAPI request-scoped session),
  2. updates its BackgroundJob row's status as it progresses,
  3. does REAL work by calling the same service/ml functions the
     synchronous API endpoints use -- no separate "fake async" logic path.

Per Section E5/E1: simple single-claim operations (score one claim, explain
one claim) stay synchronous in the API -- these tasks are specifically for
work that's expensive across MANY claims/documents, or scheduled (drift).
"""
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone

BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core import database as database_module
from app.models.domain import BackgroundJob
from app.workers.celery_app import celery_app


def _start_job(db, job_id: str):
    job = db.get(BackgroundJob, job_id)
    if job:
        job.status = "RUNNING"
        job.started_at = datetime.now(timezone.utc)
        db.commit()
    return job


def _finish_job_success(db, job_id: str, result: dict):
    job = db.get(BackgroundJob, job_id)
    if job:
        job.status = "SUCCESS"
        job.result = result
        job.finished_at = datetime.now(timezone.utc)
        db.commit()


def _finish_job_failure(db, job_id: str, error: str):
    job = db.get(BackgroundJob, job_id)
    if job:
        job.status = "FAILURE"
        job.error = error
        job.finished_at = datetime.now(timezone.utc)
        db.commit()


def _run_job(job_id: str, work_fn):
    """Shared boilerplate: open a session, mark RUNNING, run `work_fn(db)`,
    mark SUCCESS/FAILURE with the real exception on failure (never swallowed)."""
    db = database_module.SessionLocal()
    try:
        _start_job(db, job_id)
        result = work_fn(db)
        _finish_job_success(db, job_id, result)
        return result
    except Exception as e:
        _finish_job_failure(db, job_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")
        raise
    finally:
        db.close()


def _create_and_run_job(task_name: str, params: dict, work_fn):
    """
    Same contract as _run_job, but for tasks with no pre-existing
    BackgroundJob row -- i.e. tasks triggered by celery-beat on a schedule
    rather than an API call. Creates its own BackgroundJob row up front so
    scheduled runs show up in GET /jobs/{id} / the job history exactly like
    an API-enqueued one, giving real production observability into "did the
    nightly drift check run, and what happened" without needing a separate
    monitoring mechanism.
    """
    db = database_module.SessionLocal()
    try:
        job = BackgroundJob(task_name=task_name, status="QUEUED", params=params)
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    return _run_job(job_id, work_fn)


@celery_app.task(name="app.workers.tasks.score_claim_batch", bind=True, max_retries=2, default_retry_delay=5)
def score_claim_batch(self, job_id: str, claim_ids: list[str]):
    """Real batch denial-risk scoring across many claims (Section E5.1).
    Uses the same ml_inference.score_claim() the single-claim endpoint uses --
    no separate/fake batch logic path."""
    from app.models.domain import Claim
    from app.services import ml_inference

    def work(db):
        results = []
        errors = []
        for claim_id in claim_ids:
            claim = db.get(Claim, claim_id)
            if not claim:
                errors.append({"claim_id": claim_id, "error": "not found"})
                continue
            try:
                results.append(ml_inference.score_claim(db, claim))
            except ml_inference.ModelNotTrainedError as e:
                errors.append({"claim_id": claim_id, "error": str(e)})
        return {"scored": len(results), "errors": errors, "results": results}

    try:
        return _run_job(job_id, work)
    except Exception as exc:
        raise self.retry(exc=exc)


@celery_app.task(name="app.workers.tasks.generate_shap_explanations", bind=True)
def generate_shap_explanations(self, job_id: str, claim_ids: list[str]):
    """Batch SHAP generation (Section E5.5) -- SHAP is the slowest per-claim
    operation in the app, so batching it is the clearest case for async."""
    from app.models.domain import Claim
    from app.services import ml_inference

    def work(db):
        explanations = {}
        errors = []
        for claim_id in claim_ids:
            claim = db.get(Claim, claim_id)
            if not claim:
                errors.append({"claim_id": claim_id, "error": "not found"})
                continue
            try:
                explanations[claim_id] = ml_inference.explain_claim(db, claim)
            except ml_inference.ModelNotTrainedError as e:
                errors.append({"claim_id": claim_id, "error": str(e)})
        return {"explained": len(explanations), "errors": errors}

    return _run_job(job_id, work)


@celery_app.task(name="app.workers.tasks.ingest_documents", bind=True)
def ingest_documents_task(self, job_id: str, file_path: str | None = None):
    """RAG embedding generation for a document batch (Section E5.2) --
    wraps the same ingestion logic scripts/ingest_documents.py uses."""
    import json
    from app.models.domain import Document, DocumentChunk
    from app.rag.chunking import chunk_text
    from app.rag.embeddings import get_embedding_provider

    default_file = REPO_ROOT / "data" / "synthetic" / "sample_payer_policies.json"
    target_file = Path(file_path) if file_path else default_file

    def work(db):
        provider = get_embedding_provider()
        docs = json.loads(target_file.read_text())
        total_chunks = 0
        for d in docs:
            existing = db.query(Document).filter(Document.title == d["title"]).first()
            if existing:
                continue
            document = Document(title=d["title"], source_type=d.get("source_type", "payer_policy"), version=d.get("version", "1.0"))
            db.add(document)
            db.flush()
            for i, chunk in enumerate(chunk_text(d["text"])):
                embedding = provider.embed(chunk)
                db.add(DocumentChunk(document_id=document.id, chunk_index=i, chunk_text=chunk, embedding=embedding))
                total_chunks += 1
        db.commit()
        return {"documents_processed": len(docs), "chunks_created": total_chunks}

    return _run_job(job_id, work)


@celery_app.task(name="app.workers.tasks.calculate_drift_metrics", bind=True)
def calculate_drift_metrics(self, job_id: str | None = None, current_window: int = 500):
    """Scheduled (celery-beat, daily -- see celery_app.py's beat_schedule)
    or on-demand drift computation (Section D6/E5.3) -- identical code path
    as the synchronous /model-monitoring/drift/compute API endpoint."""
    from ml.monitoring.drift import run_full_drift_check

    def work(db):
        return run_full_drift_check(db, current_window=current_window)

    if job_id:
        return _run_job(job_id, work)
    # celery-beat invocation: no pre-existing job_id from an API call, so
    # create and track our own BackgroundJob row for observability into
    # scheduled runs (see _create_and_run_job's docstring).
    return _create_and_run_job("calculate_drift_metrics", {"current_window": current_window, "trigger": "beat"}, work)


def _make_train_work_fn(model_type: str):
    """Shared dispatch used by both the on-demand train-async endpoint task
    and the scheduled weekly retrain task -- one place that maps
    model_type -> training script, so they can never drift apart."""

    def work(db):
        if model_type == "denial_risk":
            from ml.training.train_denial_model import main as train_main
        elif model_type == "denial_reason":
            from ml.training.train_denial_reason_model import main as train_main
        elif model_type == "appeal_success":
            from ml.training.train_appeal_success_model import main as train_main
        elif model_type == "anomaly_detection":
            from ml.training.train_anomaly_model import main as train_main
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        train_main()
        return {"model_type": model_type, "status": "training script completed"}

    return work


@celery_app.task(name="app.workers.tasks.scheduled_retrain_all_models", bind=True)
def scheduled_retrain_all_models(self):
    """
    Weekly production retraining job (celery-beat -- see celery_app.py's
    beat_schedule). Retrains all 4 model types in sequence, one
    BackgroundJob row per model type so a failure in one model's training
    (e.g. not enough new resolved appeals yet) doesn't hide whether the
    others succeeded. This is the scheduled counterpart to the on-demand
    POST /models/{type}/train-async endpoint -- same underlying training
    scripts (via _make_train_work_fn), no separate "fake retrain" logic path.
    """
    model_types = ["denial_risk", "denial_reason", "appeal_success", "anomaly_detection"]
    results = {}
    for model_type in model_types:
        try:
            results[model_type] = _create_and_run_job(
                "scheduled_retrain_all_models", {"model_type": model_type, "trigger": "beat"}, _make_train_work_fn(model_type)
            )
        except Exception as e:
            # One model's training failing (e.g. too little new data) must
            # not abort the rest -- each is independently job-tracked above,
            # so the failure is already visible in that model's own
            # BackgroundJob row; record it here too and keep going.
            results[model_type] = {"error": str(e)}

    return results


@celery_app.task(name="app.workers.tasks.train_model", bind=True)
def train_model_task(self, job_id: str, model_type: str):
    """
    Wraps the CLI training scripts (Section E1) so training can be kicked
    off via the API without blocking the request -- training a denial-risk
    or appeal-success model over thousands of claims is exactly the kind
    of expensive operation Section E1 says shouldn't run synchronously
    inside a request handler.
    """
    return _run_job(job_id, _make_train_work_fn(model_type))
