"""
Background jobs API (Phase 2, Feature E4). Enqueuing endpoints return
{"job_id", "status": "QUEUED"} immediately; GET /jobs/{job_id} reports
QUEUED/RUNNING/SUCCESS/FAILURE by reading the BackgroundJob row the task
itself updates (app/workers/tasks.py) -- not by querying Celery's result
backend directly, so this works identically whether Celery runs eagerly
(tests, CELERY_TASK_ALWAYS_EAGER=true) or via a real worker+Redis.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.domain import BackgroundJob, Claim, User, UserRole
from app.workers.tasks import (
    calculate_drift_metrics,
    generate_shap_explanations,
    ingest_documents_task,
    score_claim_batch,
    train_model_task,
)

router = APIRouter(tags=["jobs"])


def _enqueue(db: Session, task_name: str, params: dict, celery_task, task_args: tuple) -> dict:
    job = BackgroundJob(task_name=task_name, status="QUEUED", params=params)
    db.add(job)
    db.commit()
    db.refresh(job)

    # .delay() enqueues via the Redis broker normally; under
    # CELERY_TASK_ALWAYS_EAGER (tests, or local dev without a worker
    # running) this executes synchronously in-process and the job's status
    # will already be SUCCESS/FAILURE by the time this call returns.
    celery_task.delay(job.id, *task_args)

    db.refresh(job)
    return {"job_id": job.id, "status": job.status}


@router.post("/claims/batch-score")
def batch_score_claims(
    claim_ids: list[str],
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN, UserRole.BILLER, UserRole.ANALYST)),
):
    if not claim_ids:
        raise HTTPException(status_code=400, detail="claim_ids must be non-empty")
    if len(claim_ids) > 5000:
        raise HTTPException(status_code=400, detail="claim_ids exceeds max batch size (5000)")
    return _enqueue(db, "score_claim_batch", {"n_claims": len(claim_ids)}, score_claim_batch, (claim_ids,))


@router.post("/claims/batch-explain")
def batch_explain_claims(
    claim_ids: list[str],
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN, UserRole.BILLER, UserRole.ANALYST)),
):
    if not claim_ids:
        raise HTTPException(status_code=400, detail="claim_ids must be non-empty")
    return _enqueue(db, "generate_shap_explanations", {"n_claims": len(claim_ids)}, generate_shap_explanations, (claim_ids,))


@router.post("/documents/ingest-async")
def ingest_documents_async(
    file_path: str | None = None,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN)),
):
    return _enqueue(db, "ingest_documents", {"file_path": file_path}, ingest_documents_task, (file_path,))


@router.post("/model-monitoring/drift/compute-async")
def compute_drift_async(
    current_window: int = 500,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    return _enqueue(db, "calculate_drift_metrics", {"current_window": current_window}, calculate_drift_metrics, (current_window,))


@router.post("/models/{model_type}/train-async")
def train_model_async(
    model_type: str,
    db: Session = Depends(get_db),
    user: User = Depends(require_roles(UserRole.ADMIN)),
):
    valid_types = {"denial_risk", "denial_reason", "appeal_success", "anomaly_detection"}
    if model_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"model_type must be one of {sorted(valid_types)}")
    return _enqueue(db, "train_model", {"model_type": model_type}, train_model_task, (model_type,))


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    job = db.get(BackgroundJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "task_name": job.task_name,
        "status": job.status,
        "params": job.params,
        "result": job.result,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }
