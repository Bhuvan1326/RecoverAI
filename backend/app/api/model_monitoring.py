import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user, require_roles
from app.models.domain import DriftMetric, ModelMetric, ModelVersion, User, UserRole

router = APIRouter(tags=["model-monitoring"])


@router.get("/model-versions")
def list_model_versions(model_type: str | None = None, db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    q = select(ModelVersion)
    if model_type:
        q = q.where(ModelVersion.model_type == model_type)
    versions = db.execute(q.order_by(ModelVersion.trained_at.desc())).scalars().all()

    def ranking_metrics_for(version_id: str) -> dict:
        # Ranking-eval metrics (Precision@K/NDCG@K/recovery-captured@K) are
        # persisted as ModelMetric rows rather than baked into
        # ModelVersion.metrics at training time (they're computed by a
        # separate evaluation step, potentially re-run later against the
        # same trained artifact) -- surfaced here so they're visible
        # anywhere a caller lists model versions, not only via the
        # dedicated /model-monitoring/recovery-ranking endpoint.
        rows = db.execute(
            select(ModelMetric).where(ModelMetric.model_version_id == version_id, ModelMetric.metric_name.like("ranking_%"))
        ).scalars().all()
        return {r.metric_name: float(r.metric_value) for r in rows}

    return [
        {
            "id": v.id,
            "model_name": v.model_name,
            "model_type": v.model_type,
            "version_tag": v.version_tag,
            "metrics": v.metrics,
            "ranking_metrics": ranking_metrics_for(v.id) or None,
            "is_champion": v.is_champion,
            "trained_at": v.trained_at,
        }
        for v in versions
    ]


@router.get("/model-monitoring")
def model_monitoring(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    versions = db.execute(select(ModelVersion)).scalars().all()
    champions = [v for v in versions if v.is_champion]
    drift = db.execute(select(DriftMetric).order_by(DriftMetric.recorded_at.desc()).limit(20)).scalars().all()

    comparison = sorted(
        [{"model_name": v.model_name, "model_type": v.model_type, "metrics": v.metrics, "is_champion": v.is_champion} for v in versions],
        key=lambda r: r["metrics"].get("pr_auc", 0),
        reverse=True,
    )

    return {
        "champions": [{"model_name": c.model_name, "model_type": c.model_type, "metrics": c.metrics} for c in champions],
        "champion_vs_challenger": comparison,
        "drift_metrics": [
            {
                "metric_name": d.metric_name,
                "feature_name": d.feature_name,
                "baseline_value": float(d.baseline_value),
                "current_value": float(d.current_value),
                "drift_score": float(d.drift_score),
                "is_drift_detected": d.is_drift_detected,
                "recorded_at": d.recorded_at,
            }
            for d in drift
        ],
    }


@router.post("/model-versions/{version_id}/promote")
def promote_champion(version_id: str, db: Session = Depends(get_db), _user: User = Depends(require_roles(UserRole.ADMIN))):
    version = db.get(ModelVersion, version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    peers = db.execute(select(ModelVersion).where(ModelVersion.model_type == version.model_type)).scalars().all()
    for p in peers:
        p.is_champion = p.id == version_id
    db.commit()
    return {"promoted": version_id, "model_type": version.model_type}


@router.get("/model-monitoring/drift")
def get_drift(
    metric_type: str | None = None,
    status: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Feature D7. Returns the most recently computed, PERSISTED drift metrics
    (see /model-monitoring/drift/compute to trigger a fresh computation).
    Read-only and cheap -- safe for a dashboard to poll.
    """
    q = select(DriftMetric)
    if metric_type:
        q = q.where(DriftMetric.metric_type == metric_type)
    if status:
        q = q.where(DriftMetric.status == status)
    rows = db.execute(q.order_by(DriftMetric.recorded_at.desc()).limit(min(limit, 500))).scalars().all()
    return [
        {
            "id": d.id,
            "model_version_id": d.model_version_id,
            "metric_type": d.metric_type,
            "metric_name": d.metric_name,
            "feature": d.feature_name,
            "metric": d.metric_name,
            "value": float(d.drift_score),
            "baseline_value": float(d.baseline_value),
            "current_value": float(d.current_value),
            "status": d.status,
            "is_drift_detected": d.is_drift_detected,
            "reference_period": d.reference_period,
            "current_period": d.current_period,
            "recorded_at": d.recorded_at,
        }
        for d in rows
    ]


@router.get("/model-monitoring/drift/{model_version_id}")
def get_drift_for_model_version(
    model_version_id: str, db: Session = Depends(get_db), _user: User = Depends(get_current_user)
):
    version = db.get(ModelVersion, model_version_id)
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    rows = (
        db.execute(
            select(DriftMetric)
            .where(DriftMetric.model_version_id == model_version_id)
            .order_by(DriftMetric.recorded_at.desc())
        )
        .scalars()
        .all()
    )
    return {
        "model_version": f"{version.model_name}-{version.version_tag}",
        "metrics": [
            {
                "metric_type": d.metric_type,
                "feature": d.feature_name,
                "metric": d.metric_name,
                "value": float(d.drift_score),
                "status": d.status,
                "is_drift_detected": d.is_drift_detected,
                "recorded_at": d.recorded_at,
            }
            for d in rows
        ],
    }


@router.post("/model-monitoring/drift/compute")
def compute_drift_now(
    current_window: int = 500,
    db: Session = Depends(get_db),
    _user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    """
    Synchronous on-demand trigger for real drift computation (Feature D1-D5).
    In Docker/production this same logic also runs on a schedule via the
    Celery `calculate_drift_metrics` task (Feature D6/E5) -- this endpoint
    lets an admin/reviewer trigger it immediately without waiting for the
    schedule, and is what CI/tests exercise directly (no live Celery/Redis
    required for the underlying computation to be real and correct).
    """
    from ml.monitoring.drift import run_full_drift_check

    result = run_full_drift_check(db, current_window=current_window)
    return result


@router.get("/model-monitoring/recovery-ranking")
def get_recovery_ranking_eval(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    """
    Feature (Section 11): returns the most recently persisted recovery-
    ranking evaluation summary, if one has been computed (see
    /model-monitoring/recovery-ranking/compute to trigger a fresh one, or
    `python -m ml.evaluation.eval_recovery_ranking`). Reads straight from
    the JSON artifact the eval script writes -- kept simple/read-only here
    since, unlike drift, this evaluation isn't (yet) run on a schedule.
    """
    summary_path = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "models" / "artifacts" / "recovery_ranking_eval_summary.json"
    if not summary_path.exists():
        return {"computed": False, "message": "No recovery-ranking evaluation has been run yet."}
    return {"computed": True, **json.loads(summary_path.read_text())}


@router.post("/model-monitoring/recovery-ranking/compute")
def compute_recovery_ranking_eval(
    db: Session = Depends(get_db),
    _user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    """
    Synchronous on-demand trigger for the recovery-ranking evaluation
    (Precision@K / NDCG@K / expected-recovery-captured@K, comparing the
    live priority-queue's model-driven ranking against naive baselines on
    real held-out appeal outcomes). Same underlying function
    (ml/evaluation/eval_recovery_ranking.py:run_ranking_evaluation) the CLI
    script uses -- persists to ModelMetric and returns the full comparison.
    """
    from ml.evaluation.eval_recovery_ranking import run_ranking_evaluation

    return run_ranking_evaluation(db)


@router.get("/model-monitoring/rag-eval")
def get_rag_eval(db: Session = Depends(get_db), _user: User = Depends(get_current_user)):
    """
    Returns the most recently persisted RAG evaluation (retrieval
    recall@K + citation correctness -- referential integrity and excerpt
    fidelity, checked against the real ingested corpus and real generated
    appeal drafts). See /model-monitoring/rag-eval/compute to trigger a
    fresh one, or `python -m ml.evaluation.eval_rag`.
    """
    summary_path = Path(__file__).resolve().parent.parent.parent.parent / "ml" / "models" / "artifacts" / "rag_eval_summary.json"
    if not summary_path.exists():
        return {"computed": False, "message": "No RAG evaluation has been run yet."}
    return {"computed": True, **json.loads(summary_path.read_text())}


@router.post("/model-monitoring/rag-eval/compute")
def compute_rag_eval(
    db: Session = Depends(get_db),
    _user: User = Depends(require_roles(UserRole.ADMIN, UserRole.REVIEWER)),
):
    """
    Synchronous on-demand trigger for the RAG evaluation. Runs the REAL
    agentic draft_appeal() pipeline against actual denied claims (requires
    a trained denial-risk model, since that's a real step in the
    investigation the agent runs before drafting) and the real retrieval
    function against the real ingested corpus -- same underlying function
    (ml/evaluation/eval_rag.py:run_rag_evaluation) the CLI script uses.
    """
    from ml.evaluation.eval_rag import run_rag_evaluation

    return run_rag_evaluation(db)
