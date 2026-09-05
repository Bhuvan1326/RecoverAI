"""
Celery application (Phase 2, Feature E). Broker and result backend are both
Redis, configured entirely from Settings/environment variables (E2) --
never hardcoded.

CELERY_TASK_ALWAYS_EAGER (from Settings, forced true in tests via
backend/tests/conftest.py) makes tasks execute synchronously in-process
with no broker required -- this is what lets the test suite exercise real
task logic without a live Redis/worker, per the project's existing pattern
of "real computation, gracefully degrading infra dependency" used
throughout (mock LLM/embedding providers, SQLite-vs-Postgres, etc.).

Includes a production celery-beat schedule (see bottom of this file): a
daily drift check and a weekly full model retrain, both using real
crontab() entries at configurable times, not a placeholder raw-second
interval.
"""
from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "recoverai",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Ensures @celery_app.task-decorated functions in tasks.py are registered
# on import of this module (needed for both the worker process and eager
# in-process execution during tests/API calls).
celery_app.autodiscover_tasks(["app.workers"], related_name="tasks")

# Production periodic schedule (Feature D6 / E, celery-beat). Real crontab()
# entries -- not a raw interval -- so these run at a fixed, predictable time
# of day/week rather than "N seconds after the beat process happened to
# start". Times are configurable via Settings (see .env.example) so a
# deployment can move these off peak-traffic windows without a code change.
# Only takes effect if a celery-beat process is actually run (see
# docker-compose.yml's celery-beat service); the app works fully without it
# -- both jobs can always be triggered on demand via their API endpoints
# (POST /model-monitoring/drift/compute(-async), POST /models/{type}/train-async).
celery_app.conf.beat_schedule = {
    "daily-drift-check": {
        "task": "app.workers.tasks.calculate_drift_metrics",
        "schedule": crontab(hour=settings.DRIFT_CHECK_HOUR_UTC, minute=settings.DRIFT_CHECK_MINUTE_UTC),
        "args": (None, settings.DRIFT_CHECK_WINDOW),  # job_id=None -> task creates+tracks its own BackgroundJob
    },
    "weekly-model-retrain": {
        "task": "app.workers.tasks.scheduled_retrain_all_models",
        "schedule": crontab(
            hour=settings.MODEL_RETRAIN_HOUR_UTC,
            minute=settings.MODEL_RETRAIN_MINUTE_UTC,
            day_of_week=settings.MODEL_RETRAIN_DAY_OF_WEEK,
        ),
    },
}
